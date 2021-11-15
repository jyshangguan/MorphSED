import sys
import numpy as np
import itertools
from astropy.convolution import Gaussian2DKernel
sys.path.append('/Users/liruancun/Works/GitHub/')
from MorphSED.morphsed import Galaxy
Mygalaxy = Galaxy(mass = 1e10)
#just usual dictionary
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
Mygalaxy.add_subC('sersic',sparams[0],age,Z)
age = {'type': "linear", 'paradic':{'k':0., 'b':1.}}
Z = {'type': "linear", 'paradic':{'k':0., 'b':0.05}}
Mygalaxy.add_subC('sersic',sparams[1],age,Z)
psfFWHM=1.44
psf = Gaussian2DKernel(psfFWHM, x_size=15, y_size=15)
psf.normalize()
totalmass = Mygalaxy.generate_mass_map((100,100),np.array(psf))
sky_mean=np.mean(totalmass)
sky_median=np.median(totalmass)
sky_std = np.std(totalmass)
image_fuv = Mygalaxy.generate_image('galex_nuv',psf,inte_step=10)
