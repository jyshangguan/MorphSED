import numpy as np
from astropy.modeling import models, fitting

def elliptical_func(xmesh,posi,a,ellipticity,PA,ymesh=None):
    x=xmesh-posi[0]
    b=a*(1.-ellipticity)
    theta=PA*np.pi/180.
    A=(a**2)*(np.sin(theta))**2+(b**2)*(np.cos(theta))**2
    B=2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
    C=(a**2)*(np.cos(theta))**2+(b**2)*(np.sin(theta))**2
    if ymesh is not None:
        y=ymesh-posi[1]
        return A*x**2+B*x*y+C*y**2-(a**2)*(b**2)
    else:
        coeff = [C, B*x, A*x**2-(a**2)*(b**2)]
        return np.roots(coeff)+posi[1]

def Maskellipse(mask,posi,a,ellipticity,PA,antimask=False):
    #PA from x axis anti-clockwise, degree
    masktemp=mask.copy()
    ny,nx=mask.shape
    xaxis = np.arange(nx)
    yaxis = np.arange(ny)
    xmesh, ymesh = np.meshgrid(xaxis, yaxis)
    if not antimask:
        masktemp[elliptical_func(xmesh+0.5,posi,a,ellipticity,PA,ymesh+0.5)<0]=1.
    else:
        masktemp[elliptical_func(xmesh+0.5,posi,a,ellipticity,PA,ymesh+0.5)<0]=0.
    return masktemp.astype(mask.dtype)

def polynomialfit(data,mask,order=3):
    i, j = np.mgrid[:data.shape[0], :data.shape[1]]
    i=data.shape[0]-i
    imask=i[~mask.astype(bool)]
    jmask=j[~mask.astype(bool)]
    datamask=data[~mask.astype(bool)]
    p_init = models.Polynomial2D(degree=order)
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = fit_p(p_init, imask, jmask, datamask)
    background=p(i,j)
    datap=data-background
    result={
        'bkgfunc':p,
        'bkg':background,
        'residual':datap
    }
    return result
