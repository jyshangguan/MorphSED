import ezgal
import pyprofit
from scipy.integrate import trapz
from astropy.convolution import convolve_fft
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d
import sys
sys.path.append('/Users/liruancun/Works/GitHub/MorphSED/morphsed/')
import sed_interp as SEDs

'''
"allbands":
'acs_f625w', '4star_m_j2', 'wfc3_f555w', 'wfc3_f139m', 'acs_f475w', 'ukidss_h', 'ndwfs_r', '4star_m_j3', 'acs_f435w',
    'ch1', 'ndwfs_i', '4star_j', 'galex_nuv', 'wfc3_f606w', '4star_m_hlong', 'wfc3_f125w', 'newfirm_j', 'newfirm_ks',
    'sloan_z', 'wfc3_f140w', 'sloan_g', 'sloan_i', 'wfc3_f814w', 'sloan_u', 'wfc3_f775w', 'sloan_r', 'r', 'wise_ch1',
    '4star_m_hshort', 'i', '4star_ks', 'wfpc2_f450w', 'README', 'h', 'wfc3_f275w', '4star_h', 'ch3', 'ukidss_k',
    'wfc3_f218w', 'ch4', 'ukidss_y', '4star_m_j1', 'wfc3_f110w', 'ukidss_j', 'ch2', 'wfc3_f153m', 'acs_f606w', 'ndwfs_bw',
    'wfpc2_f814w', 'galex_fuv', 'wfc3_f225w', 'wfpc2_f675w', 'ks', 'acs_f555w', 'wfc3_f625w', 'wfc3_f127m', 'wfc3_f475w',
    'wfpc2_f555w', 'wfc3_f438w', 'wfc3_f105w', 'newfirm_h', 'wfc3_f160w', 'j', 'v', 'acs_f775w', 'wfpc2_f606w', 'wise_ch2',
    'acs_f814w', 'wfc3_f850lp', 'b', 'wise_ch3', 'wise_ch4', 'acs_f850lp'
'''

class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

def IFU_to_img(IFU,wave,band,step=0.5):
    '''
    Transform a IFU data cube in to a 2D image

    IFU:   the input 3D array with z as the wavelength dimension
    wave:  1D array shows the sampled wavelength
    band:  choose one from "all bands"
    step:  float, wavelength accuracy to integrate flux
    filterpath = where the ezgal installed
    '''
    filterpath = '/Users/liruancun/Softwares/anaconda3/lib/python3.7/site-packages/ezgal/data/filters/'
    resp = Table.read(filterpath + band,format='ascii')
    filter_x=resp['col1']
    filter_y=resp['col2']
    tminx = np.max([np.min(filter_x),np.min(wave)])
    tmaxx = np.min([np.max(filter_x),np.max(wave)])
    interX = np.arange(tminx,tmaxx,step)
    f2=interp1d(filter_x,filter_y,bounds_error=False,fill_value=0.)
    ax=trapz(f2(interX),x=interX)
    nz,ny,nx = IFU.shape
    image = np.zeros((ny,nx))
    for loopy in range(ny):
        for loopx in range(nx):
            f1=interp1d(wave,IFU[:,loopy,loopx],bounds_error=False,fill_value=0.)
            tof=lambda x : f1(x)*f2(x)
            image[loopy][loopx] = trapz(tof(interX),x=interX)
    image /= ax
    return image

def Cal_map(r, type, paradic):
    for case in switch(type):
        if case('linear'):
            return paradic['b'] + paradic['k']*r
            break
        if case('exp'):
            return (paradic['in']-paradic['out'])*np.exp(-r/paradic['k'])+paradic['out']
            break
        if case('const'):
            return np.ones_like(r,dtype=float)*paradic['value']
            break
        if case():
            raise ValueError("Unidentified method for calculate age or Z map")

def Cal_gradient_map(r_grid, r_map, method, type, paradic, flux_band, f2, interX, ax, **kwargs):
    if type =='const':
        return np.ones_like(r_map,dtype=float)
    else:
        fratio_list = []
        targe_para = Cal_map(r_grid,type,paradic)
        for loopr in range(r_grid):
            if method =='age':
                centerSED = SEDs.get_host_SED(interX, 0., kwargs['f_cont_zero'], targe_para[loopr], kwargs['Z_zero'],kwargs['Av_zero'],1.)
            elif method =='f_cont':
                centerSED = SEDs.get_host_SED(interX, 0., targe_para[loopr], kwargs['age_zero'], kwargs['Z_zero'],kwargs['Av_zero'],1.)
            elif method =='Z':
                centerSED = SEDs.get_host_SED(interX, 0., kwargs['f_cont_zero'], kwargs['age_zero'], targe_para[loopr],kwargs['Av_zero'],1.)
            elif method =='Av':
                centerSED = SEDs.get_host_SED(interX, 0., kwargs['f_cont_zero'], kwargs['age_zero'], kwargs['Z_zero'],targe_para[loopr],1.)
            else:
                raise ValueError("Unidentified method for calculate gradient map")
            flux = trapz(centerSED*f2(interX),x=interX)/ax
            fratio_list.append(flux/flux_band)
        fratio_list = np.array(fratio_list)
        targe_map = Cal_map(r_map,type,paradic)
        gradient_map = np.interp(targe_map,np.array(targe_para),fratio_age)
        return gradient_map

class Galaxy(object):
    '''
    the galaxy object
    with physical subcomponents and parameters
    '''
    def __init__(self, mass=1e9, z=0., ebv_G=0.):
        '''
        galaxy object is initialed from a given mass
        '''
        self.mass = mass
        self.Nsub=0
        self.subCs = {}
        self.ageparams={}
        self.Zparams={}
        self.f_cont={}
        self.Avparams={}
        self.maglist = []
        self.imshape = None
        self.mass_map = {}
        self.r_map = {}
        self.redshift = z
        self.ebv_G = ebv_G
    def reset_mass(self,mass):
        '''
        reset the mass of a galaxy object
        '''
        self.mass = mass
    def add_subC(self,Pro_names,params,ageparam,Zparam,f_cont,Avparam):
        '''
        To add a subcomponent for a galaxy object

        Pro_names: the name of the profiles
        e.g. = "sersic" "coresersic" "brokenexp" "moffat" "ferrer" "king" "pointsource"

        params: a dictionary of the parameters for this subcomponent
        e.g. for sersic: {'xcen': 50.0, 'ycen': 50.0, 'frac': 0.704, 're': 10.0,
        'nser': 3.0, 'ang': -32.70422048691768, 'axrat': 1.0, 'box': 0.0, 'convolve': False}

        ageparam: a dictionary of the age dsitribution parameters for this subcomponent
        e.g. {'type': 'linear', 'paradic': {'k': -0.05, 'b': 9.0}}
             {'type': 'const', 'paradic': {'value': 5.0}}

        Zparam: a dictionary of the matallicity dsitribution parameters for this subcomponent
        e.g. {'type': 'linear', 'paradic': {'k': 0.0, 'b': 0.02}}

        f_cont: a dictionary of the fraction of starformation dsitribution parameters for this subcomponent
        between [0.,1.], fraction of starformation
        '''
        params['mag']=params.pop("frac")
        params['mag'] = 10. - 2.5*np.log10(params['mag'])
        if Pro_names in self.subCs.keys():
            self.subCs[Pro_names].append(params)
            self.ageparams[Pro_names].append(ageparam)
            self.Zparams[Pro_names].append(Zparam)
            self.Avparams[Pro_names].append(Avparam)
            self.f_cont[Pro_names].append(f_cont)
        else:
            self.subCs.update({Pro_names : [params]})
            self.ageparams.update({Pro_names : [ageparam]})
            self.Zparams.update({Pro_names : [Zparam]})
            self.Avparams.update({Pro_names : [Avparam]})
            self.f_cont.update({Pro_names : [f_cont]})
            self.mass_map.update({Pro_names : []})
            self.r_map.update({Pro_names : []})
        #print (self.mass_map)
        self.maglist.append(params['mag'])
        #print (self.maglist)

    def generate_mass_map(self,shape,convolve_func):
        '''
        gemerate the mass distribution map for a galaxy object
        shape: return 2D image shape
        convolve_func: a 2D kernel if convolution is needed
        -----
        Caution, in future, the r_map calculation can be used as elliptical_func
        -----
        '''
        mags = np.array(self.maglist)
        magzero = 2.5*np.log10(self.mass/np.sum(np.power(10,mags/(-2.5))))
        profit_model = {'width':  shape[1],
                'height': shape[0],
                'magzero': magzero,
                'psf': convolve_func,
                'profiles': self.subCs
               }
        image, _ = pyprofit.make_model(profit_model)
        ny,nx=shape
        self.shape = shape
        image = np.zeros(shape,dtype=float)
        xaxis = np.arange(nx)
        yaxis = np.arange(ny)
        xmesh, ymesh = np.meshgrid(xaxis, yaxis)
        for key in self.subCs:
            self.mass_map[key]=[]
            for loop in range(len(self.subCs[key])):
                params = self.subCs[key][loop]
                profit_model = {'width':  nx,
                    'height': ny,
                    'magzero': magzero,
                    'psf': convolve_func,
                    'profiles': {key:[params]}
                   }
                mass_map, _ = pyprofit.make_model(profit_model)
                mass_map = np.array(mass_map)
                mass_map = np.array(mass_map.tolist())
                self.mass_map[key].append(mass_map)
                image += mass_map
                r = np.sqrt( (xmesh+0.5 - self.subCs[key][loop]['xcen'])**2. + (ymesh+0.5 - self.subCs[key][loop]['ycen'])**2.)
                self.r_map[key].append(r)
                #self.ageparams[key][loop].update({'age_map' : Cal_map(r,self.ageparams[key][loop]['type'],self.ageparams[key][loop]['paradic'])})
                #self.Zparams[key][loop].update({'Z_map' : Cal_map(r,self.Zparams[key][loop]['type'],self.Zparams[key][loop]['paradic'])})
        #print (self.Zparams)
        return image

    def generate_SED_IFU(self,shape,convolve_func,wavelength,resolution=10.):
        '''
        gemerate the SED IFU for a galaxy object
        shape: return 2D spatial shape
        convolve_func: a 2D kernel if convolution is needed
        wavelength: 1D array, the wavelength sample
        resolution: logr grid to sample SED
        '''
        ny = shape[0]
        nx = shape[1]
        mags = np.array(self.maglist)
        magzero = 2.5*np.log10(self.mass/np.sum(np.power(10,mags/(-2.5))))
        tot_IFU = np.zeros((len(wavelength),ny,nx))
        for key in self.subCs:
            for loop in range(len(self.subCs[key])):
                params = self.subCs[key][loop]
                profit_model = {'width':  nx,
                    'height': ny,
                    'magzero': magzero,
                    'psf': convolve_func,
                    'profiles': {key:[params]}
                   }
                mass_map, _ = pyprofit.make_model(profit_model)
                sub_IFU = np.zeros((len(wavelength),ny,nx))
                xaxis = np.arange(nx)
                yaxis = np.arange(ny)
                xmesh, ymesh = np.meshgrid(xaxis, yaxis)
                r = np.sqrt( (xmesh+0.5 - self.subCs[key][loop]['xcen'])**2. + (ymesh+0.5 - self.subCs[key][loop]['ycen'])**2.)
                age_map = Cal_map(r,self.ageparams[key][loop]['type'],self.ageparams[key][loop]['paradic'])
                Z_map = Cal_map(r,self.Zparams[key][loop]['type'],self.Zparams[key][loop]['paradic'])
                for loopy in range(ny):
                    for loopx in range(nx):
                        sub_IFU[:,loopy,loopx] = SEDs.get_host_SED(wavelength, Z_map[loopy][loopx], age_map[loopy][loopx], np.log10(mass_map[loopy][loopx]))
                tot_IFU += sub_IFU
        return tot_IFU

    def generate_image(self,band,convolve_func,inte_step=10):
        filterpath = '/Users/liruancun/Softwares/anaconda3/lib/python3.7/site-packages/ezgal/data/filters/'
        resp = Table.read(filterpath + band,format='ascii')
        ny = self.shape[0]
        nx = self.shape[1]
        filter_x=resp['col1']
        filter_y=resp['col2']
        tminx = np.min(filter_x)
        tmaxx = np.max(filter_x)
        interX = np.linspace(tminx,tmaxx,np.max([100,len(filter_x)]))
        f2=interp1d(filter_x,filter_y,bounds_error=False,fill_value=0.)
        ax=trapz(f2(interX),x=interX)
        r_grid = np.linspace(0.,0.5*np.sqrt(nx**2+ny**2),inte_step)
        totalflux = np.zeros(self.shape,dtype=float)
        #print (r_grid)
        interX_intrin = interX/(1.+self.redshift)
        for key in self.subCs:
            for loop in range(len(self.subCs[key])):
                age_zero = Cal_map(0.,self.ageparams[key][loop]['type'],self.ageparams[key][loop]['paradic'])
                Z_zero = Cal_map(0.,self.Zparams[key][loop]['type'],self.Zparams[key][loop]['paradic'])
                f_cont_zero = Cal_map(0.,self.f_cont[key][loop]['type'],self.f_cont[key][loop]['paradic'])
                Av_zero = Cal_map(0.,self.Avparams[key][loop]['type'],self.Avparams[key][loop]['paradic'])
                centerSED = SEDs.get_host_SED(interX_intrin, 0., f_cont_zero, age_zero, Z_zero, Av_zero, 1.)
                flux_intrin = trapz(centerSED*f2(interX_intrin),x=interX_intrin)/ax
                x, sed_obs = SEDs.sed_to_obse(interX_intrin,centerSED,self.redshift,self.ebv_G)
                flux_band = trapz(sed_obs*f2(interX),x=interX)/ax
                age_gradient = Cal_gradient_map(r_grid, self.r_map[key][loop], 'age', self.ageparams[key][loop]['type'],self.ageparams[key][loop]['paradic']
                                , flux_intrin, f2, interX_intrin, ax, age_zero=age_zero, Z_zero=Z_zero, f_cont_zero=f_cont_zero, Av_zero=Av_zero)
                Z_gradient = Cal_gradient_map(r_grid, self.r_map[key][loop], 'Z', self.Zparams[key][loop]['type'],self.Zparams[key][loop]['paradic']
                                , flux_intrin, f2, interX_intrin, ax, age_zero=age_zero, Z_zero=Z_zero, f_cont_zero=f_cont_zero, Av_zero=Av_zero)
                f_cont_gradient = Cal_gradient_map(r_grid, self.r_map[key][loop], 'f_cont', self.f_cont[key][loop]['type'],self.f_cont[key][loop]['paradic']
                                , flux_intrin, f2, interX_intrin, ax, age_zero=age_zero, Z_zero=Z_zero, f_cont_zero=f_cont_zero, Av_zero=Av_zero)
                Av_gradient = Cal_gradient_map(r_grid, self.r_map[key][loop], 'Av', self.Avparams[key][loop]['type'],self.Avparams[key][loop]['paradic']
                                , flux_intrin, f2, interX_intrin, ax, age_zero=age_zero, Z_zero=Z_zero, f_cont_zero=f_cont_zero, Av_zero=Av_zero)
                #print (age_zero,Z_zero,f_cont_zero,Av_zero,Av_gradient)
                totalflux += flux_band*self.mass_map[key][loop]*age_gradient*Z_gradient*f_cont_gradient*Av_gradient
                #print (self.mass_map[key][loop])
                #print (np.sum(self.mass_map[key][loop]))
                #print (totalflux)
        return convolve_fft(totalflux,convolve_func)


class AGN(object):
    '''
    the AGN object
    with physical subcomponents and parameters
    '''
    def __init__(self,logM_BH=8.,logLedd=-1.,astar=0., Av=0., z=0., ebv_G=0.):
        '''
        galaxy object is initialed from a given mass
        '''
        self.logM_BH = logM_BH
        self.logLedd=logLedd
        self.astar = astar
        self.Av = Av
        self.redshift = z
        self.ebv_G = ebv_G

    def generate_image(self, shape,band, convolve_func, psfparams, psftype='psf'):
        '''
        Parameters:
        shape: (y,x) of the output image

        band: band of the output image

        convolve_func: 2D array, the shape of empirical PSF

        {psftype: [psfparams]}: a dict, the point spread function
        eg.  {'psf': [{'xcen':50, 'ycen':50}]}     stands for a point sources which have same shape as the empirical PSF
             {'moffat': [{'xcen':50, 'ycen':50, 'fwhm':3., 'con':'5.'}]}
        '''
        filterpath = '/Users/liruancun/Softwares/anaconda3/lib/python3.7/site-packages/ezgal/data/filters/'
        resp = Table.read(filterpath + band,format='ascii')
        ny = shape[0]
        nx = shape[1]
        filter_x=resp['col1']
        filter_y=resp['col2']
        tminx = np.min(filter_x)
        tmaxx = np.max(filter_x)
        interX = np.linspace(tminx,tmaxx,100)
        f2=interp1d(filter_x,filter_y,bounds_error=False,fill_value=0.)
        ax=trapz(f2(interX),x=interX)
        waveintrin = interX/(1.+self.redshift)
        agnsed_rest = SEDs.get_AGN_SED(waveintrin,self.logM_BH,self.logLedd,self.astar,self.Av,1.)
        x,agnsed = SEDs.sed_to_obse(waveintrin,agnsed_rest,self.redshift,self.ebv_G)
        flux_band = trapz(agnsed*f2(interX),x=interX)/ax
        magzero = 18.
        mag = -2.5*np.log10(flux_band)+magzero
        #print (mag)
        psfparams.update({'mag':mag})
        profit_model = {'width':  nx,
            'height': ny,
            'magzero': magzero,
            'psf': convolve_func,
            'profiles': {psftype:[psfparams]}
           }
        agn_map, _ = pyprofit.make_model(profit_model)
        return np.array(agn_map)
