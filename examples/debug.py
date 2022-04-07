import sys,csv
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import itertools
import lmfit,corner
from astropy.convolution import Gaussian2DKernel
from astropy.visualization import MinMaxInterval,SqrtStretch,AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import make_lupton_rgb
sys.path.append('/Users/liruancun/Works/GitHub/')
from MorphSED.morphsed import Galaxy, AGN
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM,z_at_value
from astropy.stats import sigma_clipped_stats
from astropy.visualization.mpl_normalize import simple_norm
targname = '2'
Band = ['g','r','i','z','y']
phys_to_image ={
    'g'  : 2.12e18,  #4810
    'r'  : 3.50e18,  #6170
    'i'  : 5.20e18,  #7520
    'z'  : 6.89e18,  #8660
    'y'  : 8.50e18,  #9620
    #counts_rate/flux
}
phys_to_counts_rate = np.ones(5,dtype=float)
filepath = '/Users/liruancun/Works/GitHub/MorphSED/examples/data/test/'
images = []
psfs = []
for band in Band:
    hdu = fits.open(filepath + '{0}_{1}.fits'.format(targname,band))
    header = hdu[0].header
    hdu = fits.open(filepath + '{0}cut_{1}.fits'.format(targname,band))
    images.append(np.array(hdu[0].data)/header['EXPTIME'])
    hdu = fits.open(filepath + '{0}_{1}_psf.fits'.format(targname,band))
    psfs.append(np.array(hdu[0].data))
    #print (np.sum(images[0]))
    #break
z = 0.301712
ebv = 0.0341
cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.308)
d=cosmo.luminosity_distance(z)
dc=d.to(u.cm)
dis=dc.value
C_unit=1./(4*np.pi*dis**2)
for loop in range(5):
    phys_to_counts_rate[loop] = phys_to_image[Band[loop]]*C_unit
ny,nx = images[0].shape
par_total = lmfit.Parameters()
par_total.add('logM', value=10., min=8., max=13.)
sesicparam = ['x', 'y', 'Re', 'n', 'ang', 'axrat']
par_total.add('sersic1_x', value=46, min=41., max=51)
par_total.add('sersic1_y', value=46, min=41., max=51)
par_total.add('sersic1_Re', value=4., min=0.5, max=15.)
par_total.add('sersic1_n', value=3., min=1., max=6.)
par_total.add('sersic1_ang', value=-20., min=-90., max=90.)
par_total.add('sersic1_axrat', value=0.8, min=0.2, max=1.)
par_total.add('sersic1_f_cont', value=0.5, min=0., max=1.)
par_total.add('sersic1_age', value=5., min=0.1, max=13.)
par_total.add('sersic1_Z', value=0.02, min=0.001, max=0.04,vary=False)
par_total.add('sersic1_Av', value=0.7, min =0.3, max=5.1)
par_total.add('agn_x', value=46, min=41., max=51)
par_total.add('agn_y', value=46, min=41., max=51)
par_total.add('agn_logM', value=7.85, min=5., max=10.,vary=False)
par_total.add('agn_logLedd', value=-1, min=-4, max=2.)
par_total.add('agn_spin', value=0., min=0., max=0.99,vary=False)
par_total.add('agn_Av', value=0., min =0., max=3.1,vary=False)
par_total.add('sky_g', value=0., min =-1., max=1.)
par_total.add('sky_r', value=0., min =-1., max=1.)
par_total.add('sky_i', value=0., min =-1., max=1.)
par_total.add('sky_z', value=0., min =-1., max=1.)
par_total.add('sky_y', value=0., min =-1., max=1.)
def residualcon(parc):
    Mygalaxy = Galaxy(mass = 10**parc['logM'].value, z=z, ebv_G=ebv)
    Myagn = AGN(logM_BH=parc['agn_logM'].value,logLedd= parc['agn_logLedd'].value,
                astar=parc['agn_spin'].value,Av =parc['agn_Av'].value, z=z, ebv_G=ebv)
    strucure_para = {'xcen': parc['sersic1_x'].value, 'ycen': parc['sersic1_y'].value,
                     'frac': 100., 're': parc['sersic1_Re'].value, 'nser': parc['sersic1_n'].value,
                     'ang': parc['sersic1_ang'].value, 'axrat': parc['sersic1_axrat'].value, 'box': 0.0, 'convolve': False}
    age = {'type': "const", 'paradic':{'value': parc['sersic1_age'].value}}
    Z = {'type': "const", 'paradic':{'value':  parc['sersic1_Z'].value}}
    f_cont = {'type': "const", 'paradic':{'value': parc['sersic1_f_cont'].value}}
    Av = {'type': "const", 'paradic':{'value': parc['sersic1_Av'].value}}
    Mygalaxy.add_subC('sersic',strucure_para,age,Z,f_cont,Av)
    totalmass = Mygalaxy.generate_mass_map((ny,nx),np.array(psfs[0]))
    agnlocaltion = {'xcen': parc['agn_x'].value, 'ycen': parc['agn_y'].value,}
    residual_image=[]
    for loop in range(5):
        band = Band[loop]
        image = Mygalaxy.generate_image('panstarrs_{0}'.format(band),psfs[loop])
        image += Myagn.generate_image([ny,nx],'panstarrs_{0}'.format(band),psfs[loop],agnlocaltion)
        image *= phys_to_counts_rate[loop]
        image += np.ones_like(image)*parc['sky_{0}'.format(band)].value
        residual_image.append(images[loop]-image)
    residu_flat = residual_image[0].ravel()
    for loop in range(4):
        residu_flat=np.append(residu_flat,residual_image[loop+1].ravel())
    del (Mygalaxy)
    del (Myagn)
    return residu_flat
def multi_model(parc):
    Mygalaxy = Galaxy(mass = 10**parc['logM'].value, z=z, ebv_G=ebv)
    Myagn = AGN(logM_BH=parc['agn_logM'].value,logLedd= parc['agn_logLedd'].value,
                astar=parc['agn_spin'].value,Av =parc['agn_Av'].value, z=z, ebv_G=ebv)
    strucure_para = {'xcen': parc['sersic1_x'].value, 'ycen': parc['sersic1_y'].value,
                     'frac': 100., 're': parc['sersic1_Re'].value, 'nser': parc['sersic1_n'].value,
                     'ang': parc['sersic1_ang'].value, 'axrat': parc['sersic1_axrat'].value, 'box': 0.0, 'convolve': False}
    age = {'type': "const", 'paradic':{'value': parc['sersic1_age'].value}}
    Z = {'type': "const", 'paradic':{'value':  parc['sersic1_Z'].value}}
    f_cont = {'type': "const", 'paradic':{'value': parc['sersic1_f_cont'].value}}
    Av = {'type': "const", 'paradic':{'value': parc['sersic1_Av'].value}}
    Mygalaxy.add_subC('sersic',strucure_para,age,Z,f_cont,Av)
    totalmass = Mygalaxy.generate_mass_map((ny,nx),np.array(psfs[0]))
    agnlocaltion = {'xcen': parc['agn_x'].value, 'ycen': parc['agn_y'].value,}
    model_images=[]
    residual_images=[]
    for loop in range(5):
        band = Band[loop]
        image = Mygalaxy.generate_image('panstarrs_{0}'.format(band),psfs[loop])
        #print (image.shape)
        #print (np.sum(image))
        AGNflux = Myagn.generate_image([ny,nx],'panstarrs_{0}'.format(band),psfs[loop],agnlocaltion)
        image += AGNflux
        #print (np.max(AGNflux))
        #print (np.sum(AGNflux))
        image *= phys_to_counts_rate[loop]
        image += np.ones_like(image)*parc['sky_{0}'.format(band)].value
        model_images.append(image)
        residual_images.append(images[loop]-image)
    del (Mygalaxy)
    del (Myagn)
    return model_images,residual_images


fitresult = lmfit.minimize(residualcon,par_total,nan_policy='omit' #)
            ,method='emcee',allrandom=False,nwalkers=48,steps=5000,burn=2500,workers=16)
            #,method='ultranested',, nlive=160, maxiters=100, dlogz=0.02, workers=16)
            #,method='emcee',allrandom=False,nan_policy='omit',nwalkers=192,steps=2000,burn=1500,workers=48)
            #,method='nested',nan_policy='omit',sample_method='slice',dynamical=False, nlive=200, maxiters=100000, dlogz=0.02, workers=40)
    #            ,method='nested',nan_policy='omit',sample_method='rstagger',bound='multi',dynamical=True, nlive=150, maxiters=100000, dlogz=0.2, workers=4,
    #            dynesty_kwargs={'nlive_batch': 300,})
fitresult.flatchain.to_csv(filepath+"/chain_{0}.csv".format(targname),index=False)
par_total = fitresult.params
bestpar = par_total.valuesdict()
model_images,residual_images = multi_model(par_total)
data_all = [images,model_images,residual_images]
w = csv.writer(open(filepath+"/bestpar.csv", "w"))
for key, val in bestpar.items():
    w.writerow([key, val])
w = csv.writer(open(filepath+"/fake.csv", "w"))
for loop in range(5):
    band=Band[loop]
    hdu0 = fits.PrimaryHDU(model_images[loop].astype('float32'))
    hdulist = fits.HDUList([hdu0])
    hdulist.writeto(filepath + '{0}_{1}_model.fits'.format(targname,band),overwrite=True)
fig=plt.figure(figsize=(15,15))
emcee_corner = corner.corner(fitresult.flatchain, labels=fitresult.var_names,truths=list(fitresult.params.valuesdict().values()),
                quantiles=[0.16, 0.5, 0.84],show_titles=True, title_kwargs={"fontsize": 16})
plt.savefig(filepath+'{0}_corner.png'.format(targname),dpi=200)
plt.close()
shape=[ny,nx]
nrows = 5
ncols = 3
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 25),squeeze=True)
ax = ax.ravel()
fignumber=15
for i in range(nrows):
    sky_mean, sky_median, sky_std = sigma_clipped_stats(images[i], sigma=3.0, maxiters=5)
    norm = simple_norm([0.5*sky_std, 3*np.max(data_all[1][i])], 'log', percent=100)
    for j in range(ncols):
        ax[3*i+j].imshow(data_all[j][i], cmap='Greys', origin='lower', norm=norm,
                           interpolation='nearest')
    ax[3*i+0].text(3,80, "{0}-band".format(Band[i]), size = 25, color = 'k', weight = "light" )
ax[0].text(3,3, "data", size = 25, color = 'k', weight = "light" )
ax[1].text(3,3, "model", size = 25, color = 'k', weight = "light" )
ax[2].text(3,3, "residual", size = 25, color = 'k', weight = "light" )
plt.savefig(filepath+'{0}_multifit_mcmc.png'.format(targname),dpi=200.,bbox_inches='tight')
plt.close()
