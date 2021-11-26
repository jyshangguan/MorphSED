import numpy as np
from scipy.interpolate import RegularGridInterpolator
#from .utils import *

__all__ = ['sed_bc03']

package_path='/Users/liruancun/Works/GitHub/MorphSED/morphsed'
sed_data = np.load('{0}/templates/sed_bc03_chab.npz'.format(package_path))
points = (sed_data['mt'], sed_data['age'], sed_data['wave'])
intp_bc03 = RegularGridInterpolator(points, sed_data['sed'])

sed_data = np.load('{0}/templates/thin_disk_lowr.npz'.format(package_path))
points = (sed_data['spin'], sed_data['logMdot'], sed_data['logM'], sed_data['wave'])
intp_thindisk = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

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
    flux = intp_bc03((z, age, wave)) * 10**logMs/wave
    return flux

def get_AGN_SED(x, logM,logMdot,spin,C_unit):
    # derive a SED of standard thin disk
    # x:       wavelength, in range[125,24700]
    # logM:    log10(M_BH), in range[5,10]
    # logMdot: log10(dotM), in range[-4,2]
    # spin:    BH spin, in range[0,0.99]
    # Return: restframe L_lambda in ergs/s
    flux=intp_thindisk((spin, logMdot, logM, x))
    return C_unit*flux
