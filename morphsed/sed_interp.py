import numpy as np
from scipy.interpolate import RegularGridInterpolator
#from .utils import *

__all__ = ['sed_bc03']

package_path='/Users/liruancun/Works/GitHub/MorphSED/morphsed'
sed_data = np.load('{0}/templates/sed_bc03_chab.npz'.format(package_path))
points = (sed_data['mt'], sed_data['age'], sed_data['wave'])
intp_bc03 = RegularGridInterpolator(points, sed_data['sed'])

def sed_bc03(wave, z, age, logMs=0):
    '''
    Calculate the interpolated BC03 ssp model.

    Parameters
    ----------
    wave : 1D array
        Wavelength, units: angstrom.
    z : float
        Metallicity, range 0.008 to 0.05.
    age : float
        Stellar age, range 0.05 to 13 Gyr.

    Returns
    -------
    flux : 1D array
        SED at given wavelength, units: erg/s/cm^2/Hz.
    '''
    flux = intp_bc03((z, age, wave)) * 10**logMs
    return flux
