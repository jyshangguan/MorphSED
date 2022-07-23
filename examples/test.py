import sys
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import itertools
import lmfit
from astropy.convolution import Gaussian2DKernel
from astropy.visualization import MinMaxInterval,SqrtStretch,AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import make_lupton_rgb
sys.path.append('/Users/liruancun/Works/GitHub/')
from MorphSED.morphsed import Galaxy, AGN, Galaxy3D
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM,z_at_value
from astropy.stats import sigma_clipped_stats
from astropy.visualization.mpl_normalize import simple_norm
import time

Mygalaxy = Galaxy3D(mass = 1e11,z=0.03,ebv_G=0.07)
names  = ['%s_%s' % (profile, prop) for prop,profile in itertools.product(('xcen','ycen','mag','re','nser','ang','axrat','box'), ('sersic1','sersic2'))]
model0 = np.array((50, 50, 50, 50, 30., 70., 10, 35, 3.0, 1.0, 180/np.pi-90, 180/np.pi-90, 1.,    0.5, 0,     0))
tofit  = np.array((True,  False, True,  False, True,  True,  True, True, True, False, True,  True,  True, True,  False,  False))
params = np.array(model0[tofit])
allparams = model0.copy()
fields = ['xcen','ycen','frac','re','nser','ang','axrat','box']
s1params = [x for i,x in enumerate(allparams) if i%2 == 0]
s2params = [x for i,x in enumerate(allparams) if i%2 != 0]
fields.append('convolve')
s1params.append(False)
s2params.append(False)
sparams = [{name: val for name, val in zip(fields, params)} for params in (s1params, s2params)]
#print (sparams[0])
#print (sparams[1])

age = {'type': "linear", 'paradic':{'k':-0.05, 'b':9.}}
Z = {'type': "linear", 'paradic':{'k':0., 'b':0.02}}
f_cont = {'type': "const", 'paradic':{'value': 0.05}}
Av = {'type': "const", 'paradic':{'value': 1.5}}
sigma = {'type': "const", 'paradic':{'value': 150}}
Mygalaxy.add_subC('sersic',sparams[0],age,Z,f_cont,Av,sigma)
age = {'type': "linear", 'paradic':{'k':0., 'b':0.1}}
Z = {'type': "linear", 'paradic':{'k':0., 'b':0.05}}
f_cont = {'type': "const", 'paradic':{'value': 0.35}}
Av = {'type': "const", 'paradic':{'value': 1.0}}
sigma = {'type': "const", 'paradic':{'value': 80}}
Mygalaxy.add_subC('sersic',sparams[1],age,Z,f_cont,Av,sigma)
psfFWHM=1.44
psf = Gaussian2DKernel(psfFWHM, x_size=15, y_size=15)
psf.normalize()
totalmass = Mygalaxy.generate_mass_map((100,100),np.array(psf))
r_curve = {
    'function_name':'arctan',
    'params':{'v0':0.,'vc':220.,'r0':0.,'rt':20.,},
}
geometry = {
            'xcen' : 50.,
            'ycen' : 50., # in pix
            'i'  :  60.,      # in degree
            'PA' :  57.3, # in degree
            'rotation_curve': r_curve, # {'function_name','params'}
        }
v_map = Mygalaxy.geometry_3D(geometry)
wavelength = np.linspace(3500,7000,2000)
start_t = time.time()
IFU = Mygalaxy.generate_SED_IFU(wavelength,resolution=10)
stop_t = time.time()
print ("elapse time {0:.2f} s".format(stop_t-start_t))