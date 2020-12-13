import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs.utils import proj_plane_pixel_scales
from .plot import plot_image
from .instrument_info import get_zp

__all__ = ['image', 'image_atlas']

class image(object):
    '''
    A single image object.

    Functions
    ---------
    * Read from fits file use CCDData.
    * get_size : Get the image size.
    * plot : Plot the image.
    * sigma_clipped_stats : Calculate the basic statistics of the image.
    * set_data : Load from numpy array.
    * set_mask : Set image mask.
    * set_pixel_scales : Set the pixel scales along two axes.
    * set_zero_point : Set magnitude zero point.
    '''
    def __init__(self, filename=None, hdu=0, unit=None, zero_point=None,
                 pixel_scales=None, mask=None, verbose=True):
        '''
        Parameters
        ----------
        filename (optional) : string
            FITS file name of the image.
        hdu : int (default: 0)
            The number of extension to load from the FITS file.
        unit (optional) : string
            Unit of the image flux for CCDData.
        zero_point (optional) : float
            Magnitude zero point.
        pixel_scales (optional) : tuple
            Pixel scales along the first and second directions, units: arcsec.
        mask (optional) : 2D bool array
            The image mask.
        verbose : bool (default: True)
            Print out auxiliary data.
        '''
        if filename is None:
            self.data = None
        else:
            self.data = CCDData.read(filename, hdu=hdu, unit=unit, mask=mask)
            if self.data.wcs and (pixel_scales is None):
                pixel_scales = proj_plane_pixel_scales(self.data.wcs) * u.degree.to('arcsec')

        self.zero_point = zero_point
        if pixel_scales is None:
            self.pixel_scales = None
        else:
            self.pixel_scales = (pixel_scales[0]*u.arcsec, pixel_scales[1]*u.arcsec)

    def get_size(self, units='pixel'):
        '''
        Get the size of the image.

        Parameters
        ----------
        units : string
            Units of the size (pixel or angular units).

        Returns
        -------
        x, y : float
            Size along X and Y axes.
        '''
        nrow, ncol = self.data.shape
        if units == 'pixel':
            x = ncol
            y = nrow
        else:
            x = ncol * self.pixel_scales[0].to(units).value
            y = nrow * self.pixel_scales[1].to(units).value
        return (x, y)

    def sigma_clipped_stats(self, **kwargs):
        '''
        Run astropy.stats.sigma_clipped_stats to get the basic statistics of
        the image.

        Parameters
        ----------
        All of the parameters go to astropy.stats.sigma_clipped_stats().

        Returns
        -------
        mean, median, stddev : float
            The mean, median, and standard deviation of the sigma-clipped data.
        '''
        return sigma_clipped_stats(self.data.data, mask=self.data.mask, **kwargs)

    def plot(self, stretch='asinh', units='arcsec', vmin=None, vmax=None,
             a=None, ax=None, plain=False, **kwargs):
        '''
        Plot an image.

        Parameters
        ----------
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
        assert self.data is not None, 'Set data first!'
        ax = plot_image(self.data, self.pixel_scales, stretch=stretch,
                        units=units, vmin=vmin, vmax=vmax, a=a, ax=ax,
                        plain=plain, **kwargs)
        if plain is False:
            ax.set_xlabel(r'$\Delta X$ ({0})'.format(units), fontsize=24)
            ax.set_ylabel(r'$\Delta Y$ ({0})'.format(units), fontsize=24)
        return ax

    def set_data(self, data, unit):
        '''
        Parameters
        ----------
        data : 2D array
            Image data.
        unit : string
            Unit for CCDData.
        '''
        self.data = CCDData(data, unit=unit)

    def set_mask(self, mask):
        '''
        Set mask for the extension.

        Parameters
        ----------
        mask : 2D array
            The mask.
        '''
        assert self.data.shape == mask.shape, 'Mask shape incorrect!'
        self.data.mask = mask

    def set_pixel_scales(self, pixel_scales):
        '''
        Parameters
        ----------
        pixel_scales (optional) : tuple
            Pixel scales along the first and second directions, units: arcsec.
        '''
        self.pixel_scales = (pixel_scales[0]*u.arcsec, pixel_scales[1]*u.arcsec)

    def set_zero_point(self, zp):
        '''
        Set magnitude zero point.
        '''
        self.zero_point = zp


class image_atlas(object):
    '''
    Many images.
    '''
    def __init__(self, image_list=None, zp_list=None, band_list=None):
        '''
        '''
        if image_list is None:
            self.image_list = []
        else:
            self.image_list = image_list

        if band_list is None:
            self.band_list = []
        else:
            self.band_list = band_list

        if (zp_list is None) and (band_list is not None):
            zp_list = []
            for b in band_list:
                zp_list.append(get_zp(b))

            for loop, img in enumerate(self.image_list):
                img.set_zero_point(zp_list[loop])

    def __getitem__(self, filter_name):
        '''
        Get the image data using the filter name.
        '''
        idx = self.band_list.index(filter_name)
        return self.image_list[idx]
