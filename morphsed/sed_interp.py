import numpy as np
from scipy.interpolate import RegularGridInterpolator
import extinction
from astropy.convolution import Gaussian1DKernel,convolve_fft
import os
#from .utils import *

#__all__ = ['get_host_SED']

c=2.9979246e5
C_G=6.67259e-20  #kg km3 s-2
C_year=31556926
C_day=86400.
C_ms=1.9891*1e30   # kg
L_edd=1.25e38
tl2=2*np.sqrt(2*np.log(2))      # fwhm = tl2 * sigma

NeV={
    'name' : 'NeV',
    'wave' : 3346.79,
    'type' : 'N',
}

NeVI={
    'name' : 'NeVI',
    'wave' : 3426.85,
    'type' : 'N',
}

OII={
    'name' : 'OII',
    'wave' : 3729.875,
    'type' : 'N',
}

NeIII={
    'name' : 'NeIII',
    'wave' : 3869.81,
    'type' : 'N',
}

Hg={
    'name' : 'Hg',
    'wave' : 4341.68,
    'type' : 'N',
}

Hb={
    'name' : 'Hb',
    'wave' : 4862.68,
    'type' : 'Nabs',
}

OIII_4959={
    'name' : 'OIII_4959',
    'wave' : 4960.295,
    'type' : 'N',
}

OIII_5007={
    'name' : 'OIII_5007',
    'wave' : 5008.24,
    'type' : 'N',
}

HeI={
    'name' : 'HeI',
    'wave' : 5877.3,
    'type' : 'N',
}

NaD={
    'name' : 'NaD',
    'wave' : 5892.9,
    'type' : 'abs',
}

OI_6302={
    'name' : 'OI_6302',
    'wave' :  6302.05,
    'type' : 'N',
}

NII_6549={
    'name' : 'NII_6549',
    'wave' : 6549.86,
    'type' : 'N',
}

Ha={
    'name' : 'Ha',
    'wave' : 6564.61,
    'type' : 'Nabs',
}

NII_6583={
    'name' : 'NII_6583',
    'wave' : 6585.27,
    'type' : 'N',
}

SII_6716={
    'name' : 'SII_6716',
    'wave' : 6718.29,
    'type' : 'N',
}

SII_6731={
    'name' : 'SII_6731',
    'wave' : 6732.67,
    'type' : 'N',
}

ALLLINES = [NeV,NeVI,OII,NeIII,Hg,Hb,Ha,OIII_4959,OIII_5007,HeI,NaD,OI_6302,NII_6549,NII_6583,SII_6716,SII_6731]

if "MorphSED_DATA_PATH" not in os.environ:
    raise Exception('You should set environment varialbe `MorphSED_DATA_PATH` in your .bashrc (or rc file for other shells)')
else:
    DATA_PATH=os.environ.get('MorphSED_DATA_PATH')

sed_data = np.load('{0}/templates/thin_disk_lowr.npz'.format(DATA_PATH))
points = (sed_data['spin'], sed_data['logMdot'], sed_data['logM'], sed_data['wave'])
intp_thindisk = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

sed_data = np.load('{0}/templates/host_conti.npz'.format(DATA_PATH))
points = (sed_data['Z'], sed_data['age'], sed_data['wave'])
intp_host_cont = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

sed_data = np.load('{0}/templates/host_inst.npz'.format(DATA_PATH))
points = (sed_data['Z'], sed_data['age'], sed_data['wave'])
intp_host_inst = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

sed_data = np.load('{0}/templates/host_conti_hires.npz'.format(DATA_PATH))
points = (sed_data['Z'], sed_data['age'], sed_data['wave'])
intp_host_hres_cont = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

sed_data = np.load('{0}/templates/host_inst_hires.npz'.format(DATA_PATH))
points = (sed_data['Z'], sed_data['age'], sed_data['wave'])
intp_host_hres_inst = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)
wave_temp = np.loadtxt('{0}/templates/wave_hires.txt'.format(DATA_PATH))

def gaussian(x, amp, cen, wid):
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

def gaussian3D(x, amp, cen, wid):
    ny,nx = amp.shape
    tlen = len(x)
    IFU = np.zeros((tlen,ny,nx))
    for loop in range(tlen):
        IFU[loop,:,:] = (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x[loop]-cen)**2 / (2*wid**2))
    return IFU

def conv_spec(x, wave, flux, kernel, dcenlog, referwave):
    '''
    convolve the give spectra ['wave', 'flux'] with any given 'kernel'
    return the flux respect to x
    '''
    osfactor=2
    logw = np.log(wave)
    wavegrid = np.linspace(np.min(logw),np.max(logw),osfactor*len(wave))
    gridsparse = np.abs(wavegrid[1]-wavegrid[0])
    newkernelsize = int((len(kernel)-1)/(gridsparse*referwave))
    newkernel = np.interp(gridsparse*referwave*np.arange(newkernelsize),np.arange(len(kernel)),kernel)
    #newkernel*=10*np.sqrt(gridsparse*referwave)
    f_new = np.interp(np.exp(wavegrid), wave, flux)
    f_conv = convolve_fft(f_new,kernel=newkernel)
    return np.interp(x, np.exp(wavegrid+dcenlog), f_conv)

def get_host_SED_3D(x, logM, f_cont, age, Z, sigma, Av, C_unit):
    # intp_burst_cont normalized by SFR = 10 Ms/yr
    # intp_burst_inst normalized by M_ast = 10^7 Ms
    # x: wavelength [91, 1.6e6] Angs
    # M: overall stellar mass;
    # f_cont: float, [0, 1.], fraction of the continuum starformation component
    # age: the age of the starburst component [1e-5, 13.69] Gyr
    # Z : metalicity, [0.001,0.04], 0.02 for solar
    # sigma  : velocity dispersion km/s
    # Av : V Attenuation
    # Return: restframe L_lambda in ergs/s
    M=10**logM
    age_yr = age*1e9
    M_cont = M*f_cont
    SFR = M_cont*1e-10
    M_inst = M-M_cont
    sed1 = (10**intp_host_hres_inst((Z, age_yr, wave_temp)))*M_inst*1e-7
    if f_cont == 0. :
        sed2 = np.zeros_like(sed1,dtype=float)
    else:
        sed2 = (10**intp_host_hres_cont((Z, 1e10, wave_temp)))*SFR*1e-1
    wavemodel = wave_temp.copy()
    fluxmodel = C_unit*(sed1+sed2)
    std = sigma/c
    referwave = 5000.
    kersize = np.max([int(std*referwave*5),15])
    kernel = Gaussian1DKernel(referwave*std,x_size=kersize)
    kernel.normalize()
    mask = (wavemodel > (1.-5*std)*x.min())&(wavemodel < (1.+5*std)*x.max())
    wave = wavemodel[mask]
    flux = fluxmodel[mask]
    if mask[0] == 1:
        startx = np.argmax(x > wavemodel[0]*(1.+5*std) )
    else:
        startx = 0
    if mask[-1] == 1:
        stopx = np.argmax(x > wavemodel[-1]*(1.-5*std))-1
    else:
        stopx = -1
    if (stopx == -1)&(startx==0):
        rest_spec = conv_spec(x,wave,flux,kernel.array,0.,referwave)
    else:
        mask = np.zeros_like(x,dtype=bool)
        rest_spec = np.zeros_like(x,dtype=float)
        mask[startx:stopx] = 1
        rest_spec[mask] = conv_spec(x[mask],wave,flux,kernel.array,0.,referwave)
        sed1 = (10**intp_host_inst((Z, age_yr, x[~mask])))*M_inst*1e-7
        sed2 = (10**intp_host_cont((Z, 1e10, x[~mask])))*SFR*1e-1
        rest_spec[~mask]=C_unit*(sed1+sed2)
    cm=extinction.ccm89(x,Av,3.1)/2.5
    return rest_spec/(10**cm)

def get_host_SED(x, logM, f_cont, age, Z, Av, C_unit):
    # intp_burst_cont normalized by SFR = 10 Ms/yr
    # intp_burst_inst normalized by M_ast = 10^7 Ms
    # x: wavelength [91, 1.6e6] Angs
    # M: overall stellar mass;
    # f_cont: float, [0, 1.], fraction of the continuum starformation component
    # age: the age of the starburst component [1e-5, 13.69] Gyr
    # Z : metalicity, [0.001,0.04], 0.02 for solar
    # Av : V Attenuation
    # Return: restframe L_lambda in ergs/s
    M=10**logM
    age_yr = age*1e9
    M_cont = M*f_cont
    SFR = M_cont*1e-10
    M_inst = M-M_cont
    sed1 = (10**intp_host_inst((Z, age_yr, x)))*M_inst*1e-7
    if f_cont == 0. :
        sed2 = np.zeros_like(sed1,dtype=float)
    else:
        sed2 = (10**intp_host_cont((Z, 1e10, x)))*SFR*1e-1
    fluxmodel = C_unit*(sed1+sed2)
    cm=extinction.ccm89(x,Av,3.1)/2.5
    return fluxmodel/(10**cm)

def get_AGN_SED(x, logM,logMdot,spin,Av,C_unit):
    # derive a SED of standard thin disk
    # x:       wavelength, in range[125,24700]
    # logM:    log10(M_BH), in range[5,10]
    # logMdot: log10(dotM), in range[-4,2]
    # spin:    BH spin, in range[0,0.99]
    # Return: restframe L_lambda in ergs/s
    fluxmodel=C_unit*intp_thindisk((spin, logMdot, logM, x))
    if Av == 0.:
        return fluxmodel
    else:
        cm=extinction.ccm89(x,Av,3.1)/2.5
        return fluxmodel/(10**cm)

def sed_to_obse(x,y,z,ebv):
    xnew = x.copy()
    ynew = y.copy()
    av=3.1*ebv
    xnew*=(1+z)
    ynew/=(1+z)**3
    cm=extinction.ccm89(xnew,av,3.1)
    ynew/=np.power(10,cm/(2.5))
    return xnew,ynew

def sed_to_rest(x,y,z,ebv):
    xnew = x.copy()
    ynew = y.copy()
    av=3.1*ebv
    cm=extinction.ccm89(xnew,av,3.1)
    ynew*=np.power(10,cm/(2.5))
    xnew/=(1+z)
    ynew*=(1+z)**3
    return xnew,ynew
