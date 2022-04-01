import numpy as np
from scipy.interpolate import RegularGridInterpolator
import extinction
#from .utils import *

__all__ = ['get_host_SED']

package_path='/Users/liruancun/Works/GitHub/MorphSED/morphsed'


sed_data = np.load('{0}/templates/thin_disk_lowr.npz'.format(package_path))
points = (sed_data['spin'], sed_data['logMdot'], sed_data['logM'], sed_data['wave'])
intp_thindisk = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

sed_data = np.load('{0}/templates/host_conti.npz'.format(package_path))
points = (sed_data['Z'], sed_data['age'], sed_data['wave'])
intp_host_cont = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

sed_data = np.load('{0}/templates/host_inst.npz'.format(package_path))
points = (sed_data['Z'], sed_data['age'], sed_data['wave'])
intp_host_inst = RegularGridInterpolator(points, sed_data['sed'],bounds_error=False,fill_value=0.)

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
