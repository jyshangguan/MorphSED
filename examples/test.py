import sys
import numpy as np
import itertools
from astropy.convolution import Gaussian2DKernel
sys.path.append('/Users/liruancun/Works/GitHub/')
from MorphSED.morphsed import Galaxy,AGN
Mygalaxy = Galaxy(mass = 1e10)
#just usual dictionary
Myagn = AGN(logM_BH=7.5,logLedd=-0.5,astar=0.)
psfFWHM=1.44
kernel = Gaussian2DKernel(psfFWHM, x_size=15, y_size=15)
kernel.normalize()
agnlocaltion = {'xcen':50.01, 'ycen':50.01,'fwhm':1.44, 'con':5.}
kernel = np.array(kernel)
daa= Myagn.generate_image([100,100],'sloan_u',kernel,agnlocaltion, psftype='moffat')
print (np.sum(daa))
