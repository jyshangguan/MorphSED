import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs.utils import proj_plane_pixel_scales
from .plot import plot_image
from .instrument_info import get_zp
from .utils import get_wcs_rotation
from .math import Maskellipse,polynomialfit,cross_match
from photutils.segmentation import deblend_sources
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_threshold
from photutils import detect_sources
from photutils import source_properties
from astropy.table import Table, Column, join, join_skycoord
from astropy.wcs import WCS

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
                 pixel_scales=None, wcs_rotation=None, mask=None, verbose=True):
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
        wcs_rotation (optional) : float
            WCS rotation, east of north, units: radian.
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

        if self.data.wcs and (wcs_rotation is None):
            self.wcs_rotation = get_wcs_rotation(self.data.wcs)
        elif wcs_rotation is not None:
            self.wcs_rotation = wcs_rotation * u.radian
        else:
            self.wcs_rotation = None
        self.sources_catalog = None
        self.sources_skycord = None
        self.ss_data = None

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

    def get_data_info(self):
        '''
        Data information to generate model image.

        Returns
        -------
        d : dict
            shape : (ny, nx)
                Image array shape.
            pixel_scale : (pixelscale_x, pixelscale_y), default units: arcsec
                Pixel scales.
            wcs_rotation : angle, default units: radian
                WCS rotation, east of north.
        '''
        d = dict(shape=self.data.shape,
                 pixel_scale=self.pixel_scale,
                 wcs_rotation=self.wcs_rotation)
        return d

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

    def plot_direction(self, ax, xy=(0, 0), len_E=None, len_N=None, color='k', fontsize=20,
                       linewidth=2, frac_len=0.1, units='arcsec', backextend=0.05):
        '''
        Plot the direction arrow. Only applied to plots using WCS.

        Parameters
        ----------
        ax : Axis
            Axis to plot the direction.
        xy : (x, y)
            Coordinate of the origin of the arrows.
        length : float
            Length of the arrows, units: pixel.
        units: string (default: arcsec)
            Units of xy.
        '''
        xlim = ax.get_xlim()
        len_total = np.abs(xlim[1] - xlim[0])
        pixelscale = self.pixel_scales[0].to('degree').value
        if len_E is None:
            len_E = len_total * frac_len / pixelscale
        if len_N is None:
            len_N = len_total * frac_len / pixelscale

        wcs = self.data.wcs
        header = wcs.to_header()
        d_ra = len_E * pixelscale
        d_dec = len_N * pixelscale
        ra = [header['CRVAL1'], header['CRVAL1']+d_ra, header['CRVAL1']]
        dec = [header['CRVAL2'], header['CRVAL2'], header['CRVAL2']+d_dec]
        ra_pix, dec_pix = wcs.all_world2pix(ra, dec, 1)
        d_arrow1 = [ra_pix[1]-ra_pix[0], dec_pix[1]-dec_pix[0]]
        d_arrow2 = [ra_pix[2]-ra_pix[0], dec_pix[2]-dec_pix[0]]
        l_arrow1 = np.sqrt(d_arrow1[0]**2 + d_arrow1[1]**2)
        l_arrow2 = np.sqrt(d_arrow2[0]**2 + d_arrow2[1]**2)
        d_arrow1 = np.array(d_arrow1) / l_arrow1 * len_E * pixelscale
        d_arrow2 = np.array(d_arrow2) / l_arrow2 * len_N * pixelscale

        def sign_2_align(sign):
            '''
            Determine the alignment of the text.
            '''
            if sign[0] < 0:
                ha = 'right'
            else:
                ha = 'left'
            if sign[1] < 0:
                va = 'top'
            else:
                va = 'bottom'
            return ha, va
        ha1, va1 = sign_2_align(np.sign(d_arrow1))
        ha2, va2 = sign_2_align(np.sign(d_arrow2))

        xy_e = (xy[0] - d_arrow1[0] * backextend, xy[1] - d_arrow1[1] * backextend)
        ax.annotate('E', xy=xy_e, xycoords='data', fontsize=fontsize,
                    xytext=(d_arrow1[0]+xy[0], d_arrow1[1]+xy[1]), color=color,
                    arrowprops=dict(color=color, arrowstyle="<-", lw=linewidth),
                    ha=ha1, va=va1)
        xy_n = (xy[0] - d_arrow2[0] * backextend, xy[1] - d_arrow2[1] * backextend)
        ax.annotate('N', xy=xy_n, xycoords='data', fontsize=fontsize,
                    xytext=(d_arrow2[0]+xy[0], d_arrow2[1]+xy[1]), color=color,
                    arrowprops=dict(color=color, arrowstyle="<-", lw=linewidth),
                    ha=ha2, va=va2)

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


    def source_detection_individual(self, psfFWHM, nsigma=3.0, sc_key=''):
        '''
        Parameters
        ----------
        psfFWHM : float
            FWHM of the imaging point spread function
        nsigma : float
            source detection threshold
        '''
        data = np.array(self.data.copy())
        psfFWHMpix = psfFWHM / self.pixel_scales[0].value
        thresholder = detect_threshold(data, nsigma=nsigma)
        sigma = psfFWHMpix * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
        kernel.normalize()
        segm = detect_sources(data, thresholder, npixels=5, filter_kernel=kernel)
        props = source_properties(data, segm)
        tab = Table(props.to_table())
        self.sources_catalog = tab
        srcPstradec = self.data.wcs.all_pix2world(tab['xcentroid'], tab['ycentroid'],1)
        sc = SkyCoord(srcPstradec[0], srcPstradec[1], unit='deg')
        sctab = Table([sc,np.arange(len(sc))],names=['sc','sloop_{0}'.format(sc_key)])
        self.sources_skycord = sctab


    def make_mask(self,sources=None,magnification=3.):
        '''
        make mask for the extension.

        Parameters
        ----------
        sources : a to-be masked source table (can generate from photutils source detection)
                  if None, will use its own source catalog
        magnification : expand factor to generate mask
        '''
        mask=np.zeros_like(self.data, dtype=bool)
        mask[np.isnan(self.data)] = True
        mask[np.isinf(self.data)] = True
        if sources is None:
            sources = self.sources_catalog
        for loop in range(len(sources)):
            position = (sources['xcentroid'][loop],sources['ycentroid'][loop])
            a = sources['semimajor_axis_sigma'][loop]
            b = sources['semiminor_axis_sigma'][loop]
            theta = sources['orientation'][loop]*180./np.pi
            mask=Maskellipse(mask,position,magnification*a,(1-b/a),theta)
        self.data.mask = mask
        if self.ss_data is not None:
            self.ss_data.mask = mask


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
        if self.ss_data is not Nont:
            self.ss_data.mask = mask

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

    def sky_subtraction(self, order=3 ):
        '''
        Do polynomial-fitting sky subtraction
        Parameters
        ----------
        order (optional) : int
            order of the polynomial
        '''
        data = np.array(self.data.copy())
        maskplus = self.data.mask.copy()
        backR=polynomialfit(data,maskplus.astype(bool),order=order)
        background=backR['bkg']
        self.ss_data = CCDData(data-background, unit=self.data.unit)
        self.ss_data.mask = maskplus

class image_atlas(object):
    '''
    Many images.
    '''
    def __init__(self, image_list=None, zp_list=None, band_list=None, psfFWHM_list=None):
        '''
        Parameters
        ----------
        image_list (optional) : List
            List of `image`.
        zp_list (optional) : List
            List of magnitude zeropoint.
        band_list (optional) : List
            List of band name.  Check `instrument_info` for band names.
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

        if psfFWHM_list is None:
            self.psfFWHM_list = []
        else:
            self.psfFWHM_list = psfFWHM_list

        self.__length = len(image_list)
        self.common_catalog = None

    def __getitem__(self, key):
        '''
        Get the image data using the filter name or number index.
        '''
        if type(key) is str:
            idx = self.band_list.index(key)
        elif type(key) is int:
            idx = key
        return self.image_list[idx]

    def __len__(self):
        '''
        Get the length of the data list.
        '''
        return self.__length

    def source_detection(self,nsigma=3.0):
        '''
        Do multi-band source detection
        Parameters
        ----------
        nsigma : float, or a array with same size as image_atlas
            source detection threshold
        '''
        if type(nsigma) == float:
            nsigma = nsigma * np.ones(self.__length,dtype=float)
        for loop in range(self.__length):
            self.image_list[loop].source_detection_individual(self.psfFWHM_list[loop],nsigma=nsigma[loop],sc_key=loop+1)


    def make_common_catalog(self,CM_separation=2.5,magnification=3.0,applylist=None):
        '''
        Do multi-band source detection
        Parameters
        ----------
        CM_separation : float
        angular separation used to do sky coordinates crossmatching, unit in deg
        magnification : float, or a array with same size as image_atlas
                        magnification for generating mask foe each image
        applylist : [list of index]
        None for all images
        '''
        if type(magnification) == float:
            magnification = magnification * np.ones(self.__length,dtype=float)
        if applylist is None:
            applylist = np.arange(self.__length)
        cats = []
        for loop in applylist:
            cats.append(self.image_list[loop].sources_skycord)
        comc = cross_match(cats,angular_sep = 2.5)
        lencc = len(comc)
        master_a = np.zeros(lencc, dtype = float)
        master_b = np.zeros(lencc, dtype = float)
        for loop in range(len(comc)):
            a = []
            b = []
            for loop2 in applylist:
                a.append(self.image_list[loop2].sources_catalog['semimajor_axis_sigma'][comc['sloop_{0}'.format(loop2+1)][loop]]
                *magnification[loop2]*self.image_list[loop2].pixel_scales[0].value)
                b.append(self.image_list[loop2].sources_catalog['semiminor_axis_sigma'][comc['sloop_{0}'.format(loop2+1)][loop]]
                *magnification[loop2]*self.image_list[loop2].pixel_scales[0].value)
            master_a[loop] = np.max(np.array(a))
            master_b[loop] = np.max(np.array(b))
        comc.add_column(Column(master_a, name = 'master_a'))
        comc.add_column(Column(master_b, name = 'master_b'))
        self.common_catalog = comc


    def master_mask(self, magnification=3.0, applylist=None):
        '''
        Do multi-band source masking
        Parameters
        ----------
        magnification : float, or a array with same size as image_atlas
                        magnification for generating mask foe each image
        applylist : [list of index]
        None for all images
        '''
        if type(magnification) == float:
            magnification = magnification * np.ones(self.__length,dtype=float)
        if applylist is None:
            applylist = np.arange(self.__length)
        comc = self.common_catalog.copy()
        for loop2 in applylist:
            for loop in range(len(comc)):
                self.image_list[loop2].sources_catalog['semimajor_axis_sigma'][comc['sloop_{0}'.format(loop2+1)][loop]] = comc['master_a'][loop]/(magnification[loop2]*self.image_list[loop2].pixel_scales[0].value)
                self.image_list[loop2].sources_catalog['semiminor_axis_sigma'][comc['sloop_{0}'.format(loop2+1)][loop]] = comc['master_b'][loop]/(magnification[loop2]*self.image_list[loop2].pixel_scales[0].value)
            indexes = np.delete(np.arange(len(self.image_list[loop2].sources_catalog)), comc['sloop_{0}'.format(loop2+1)])
            self.image_list[loop2].sources_catalog.remove_rows(indexes)
        for loop2 in range(self.__length):
            self.image_list[loop2].make_mask(magnification=magnification[loop2])
