import numpy as np
import matplotlib.pyplot as plt
#from astropy.io import fits
#from photutils.isophote import (EllipseGeometry, Ellipse, Isophote, IsophoteList)
#from photutils.isophote.sample import EllipseSample, CentralEllipseSample
#from photutils.isophote.fitter import CentralEllipseFitter
#from astropy import units as u
from astropy.visualization import (MinMaxInterval, LinearStretch, SqrtStretch,
                                   LogStretch, AsinhStretch, ImageNormalize)

__all__ = ['plot_image']


def plot_image(image, pixel_scales=None, stretch='asinh', units='arcsec',
               vmin=None, vmax=None, a=None, ax=None, plain=False, **kwargs):
    '''
    Plot an image.

    Parameters
    ----------
    image : 2D array
        The image to be ploted.
    pixel_scales (optional) : tuple
        Pixel scales along the first and second directions, units: arcsec.
    stretch : string (default: 'asinh')
        Choice of stretch: asinh, linear, sqrt, log.
    units : string (default: 'arcsec')
        Units of pixel scale.
    vmin (optional) : float
        Minimal value of imshow.
    vmax (optional) : float
        Maximal value of imshow.
    a (optional) : float
        Scale factor of some stretch function.
    ax (optional) : matplotlib.Axis
        Axis to plot the image.
    plain : bool (default: False)
        If False, tune the image.
    **kwargs : Additional parameters goes into plt.imshow()

    Returns
    -------
    ax : matplotlib.Axis
        Axis to plot the image.
    '''
    if pixel_scales is None:
        extent = None
        units = 'pixel'
    else:
        nrow, ncol = image.shape
        x_len = ncol * pixel_scales[0].to(units).value
        y_len = nrow * pixel_scales[1].to(units).value
        extent = (-x_len/2, x_len/2, -y_len/2, y_len/2)
    stretchDict = {
        'linear': LinearStretch,
        'sqrt': SqrtStretch,
        'log': LogStretch,
        'asinh': AsinhStretch,
    }
    if a is None:
        stretch_use = stretchDict[stretch]()
    else:
        stretch_use = stretchDict[stretch](a=a)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch_use)

    if ax is None:
        plt.figure(figsize=(7, 7))
        ax = plt.gca()

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'Greys'
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'
    if 'norm' not in kwargs:
        kwargs['norm'] = norm
    if 'extent' not in kwargs:
        kwargs['extent'] = extent
    ax.imshow(image, **kwargs)

    if plain is False:
        ax.minorticks_on()
        ax.set_aspect('equal', adjustable='box')
    return ax
