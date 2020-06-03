import numpy as np
from photutils.centroids import fit_2dgaussian
from astropy.modeling.functional_models import Ellipse2D
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astropy.constants import si
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


def gaussian_2d(x, amp, mean, std, theta):
    # two dimensional elliptical gaussian PDF with 2D offset and rotation
    # x, mean, std and offset should all be arrays of the form [x, y]
    # not actually useful because photutils does this for me, but interesting
    
    # extend all of the other arrays to the coordinate meshgrid
    extarr = np.ones(np.shape(x)[:2])
    ampext = amp*extarr
    
    extarr = np.stack((extarr, extarr), axis=2)
    meanext = extarr*mean
    stdext = extarr*std
    rotext = np.stack(((x[:,:,0]*np.cos(theta) - x[:,:,1]*np.sin(theta)),
                      x[:,:,0]*np.sin(theta) + x[:,:,1]*np.cos(theta)), axis=2)

    
    # two dimensional gaussian function
    gauss = amp*np.exp(-np.sum(((mean - rotext)/std)**2, axis=2)/2)
    
    return gauss



def get_FWHMa(data, cutout):
    ''' Use photutils' fit_2dgaussian function to determine the FWHM of an elliptical signal. Data should be the whole
        that is to be fit, and cutout is [xmin, xmax] in native pixels around the signal. returns an array
        of [x FHWM, y FHWM]
    '''
    gfit = fit_2dgaussian(data[cutout[0]:cutout[1], cutout[0]:cutout[1]])
    # get the Gaussian FWHM from the fitted model: FWHM = sigma * sqrt(8 ln(2))
    stds = np.array([gfit.x_stddev.value, gfit.y_stddev.value])
    FWHMa = stds*np.sqrt(8*np.log(2))

    return FWHMa




def plot_gauss_fit(data, scicoords, cutout):
    '''plots the gaussian fitted to the signal using photutils' fit_2dgaussian function, as well as residuals and the 
       original signal. data should be the entire frame, and cutout is [xmin, xmax] in native pixels around the signal
    '''

    xmin, xmax = cutout[0], cutout[1]


    scifit = fit_2dgaussian(data[xmin:xmax, xmin:xmax])

    # fitted gaussian
    scieval = scifit.evaluate(scicoords[:,:,0], scicoords[:,:,1], scifit.constant.value, scifit.amplitude.value, 
                              scifit.x_mean.value + xmin, scifit.y_mean.value + xmin, scifit.x_stddev.value, 
                              scifit.y_stddev.value, scifit.theta.value)

    # colormap extrema
    vmin, vmax = np.min(data), np.max(data)

    # residuals
    resids = data - scieval


    # plot
    fig,axs = plt.subplots(1, 3, figsize=(15,6))

    # original
    im = axs[0].pcolormesh(data[xmin:xmax, xmin:xmax], cmap=plt.cm.inferno, vmin=vmin, vmax=vmax)
    axs[0].set_aspect(aspect=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.07, ax=axs[0])

    # fitted gaussian
    im = axs[1].pcolormesh(scieval[xmin:xmax, xmin:xmax], cmap=plt.cm.inferno, vmin=vmin, vmax=vmax)
    axs[1].set_aspect(aspect=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.07, ax=axs[1])

    # residuals
    resids = data - scieval
    im = axs[2].pcolormesh(resids[xmin:xmax, xmin:xmax], cmap=plt.cm.gist_rainbow)
    axs[2].set_aspect(aspect=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.07, ax=axs[2])
    return


def plot_spectral_profile(file, nFWHMa):
    ''' Function to plot a 1D spectral profile of a given region about the center of a pointing. a, b
        are the beam semimajor and semiminor axes, respectively, and are in arcminutes. 
    '''
    
    # get the data from the file
    hdu = fits.open(file)[0]
    data = hdu.data
    # remove the stokes axis
    data = data[0,:,:,:]
    # map NaNs to zero
    data[np.where(np.isnan(data))] = 0.
    
    # get the semimajor and semiminor axis values from the header
    a = (hdu.header['BMAJ']*u.deg).to(u.arcsec)
    b = (hdu.header['BMIN']*u.deg).to(u.arcsec)
    
    # define an elliptical mask with axes equal to nfwhma times the semimajor/semiminor axes
                  
    
    cpix = data.shape[1] // 2
    
    


def line_mask(data, cutout, use_stds=False):
    ''' Function to, when passed a frame with a spectral line signal, determine a 4 FWHM elliptical region about
        the center of the source. returns a 2D array with the ellipse pixels=1 and all other pixels=0. Data
        should be a single frame and cutout should be a region of the form [xmin,xmax] around the center of 
        the desired signal. if use_stds is true, 4 sigma will be used instead of 4 FWHMa
    '''
    
    xmin, xmax = cutout[0], cutout[1]
    
    # to get theta and the mean values, fit
    datafit = fit_2dgaussian(data[xmin:xmax, xmin:xmax])
   
    if use_stds == True:
        # use the standard deviation values to calculate the limits of the ellipse
        FWHMa = (datafit.x_stddev.value, datafit.y_stddev.value)
    else:
        # fit the cutout data to a 2D gaussian ellipse to determine the FWHMa. call get_FWHMa to do this
        FWHMa = get_FWHMa(data, cutout)
    
    
    # ellipse2d object with axes 4*fwhma
    ell = Ellipse2D(1, datafit.x_mean.value + xmin, datafit.y_mean.value + xmin, 4*FWHMa[0], 4*FWHMa[1], 
                    datafit.theta.value)
    
    # x and y coords for evaluation
    datx = np.arange(np.shape(data)[0])
    x, y = np.meshgrid(datx, datx)
    return ell(x, y), FWHMa


def gaussian(x, amp, mu, sig, C):
    '''simple 1d gaussian function with included offset'''
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + C



def int_gaussian(amp, mu, sig, C):
    '''integral of a gaussian function assuming the bounds are essentially infinite.
       Ignores the constant offset because it's assumed to be part of the continuum, but it's included
       in the input parameters to make it easy to just unpack opt
       (This equation is just taken from wikipedia)'''
    return np.sqrt(2*np.pi)*amp*np.abs(sig)


def get_velocity_int_flux(moment_file, cube_file, cutout, p0, use_stds=True):
    ''' Function to, when passed an image cube and the Moment-0 map calculated from the cube, calculate and 
        return the CO intensity in two different ways: firstly, by finding the point of maximum flux in the 
        moment 0 map and returning that value as Jy/km/s, and secondly by defining a 4-sigma region about the
        center of a 2d elliptical gaussian fit to the signal, integrating over the region in each channel to 
        determine the spectral profile of the signal, and fitting the profile to a gaussian function. The 
        area under this second Gaussian is another measure of the maximum intensity.
        
        INPUTS: moment_file: file pointer to the moment 0 plot
                cube_file: file pointer to the image cube
                cutout: (xmin, xmax) region about the center of the signal in the moment file
                p0 = [amplitude, mean, standard deviation, offset]: best-guess parameters for spectral gaussian
                                                                    fit. will automatically adjust mean to be the
                                                                    center of the velocity axis
        OUTPUTS: mommax: value of the most intense pixel in the moment 0 plot
                 specmax: area under a gaussian fit to the spectral profile of the 4sigma region about the
                          signal in Jy
                 mean: The central frequency value of the signal under study  
    '''
    
    # get data from the moment 0 plot
    hdul = fits.open(moment_file)
    sci = hdul['PRIMARY'].data
    momwcs = wcs.WCS(hdul[0].header)
    sci = sci[0,0,:,:].astype('float64')
    
    # pass the nans in the sci array to zeros
    sci[np.where(np.isnan(sci))] = 0.
    
    # beam area in arcseconds^2
    abeam = (np.pi*hdul[0].header['BMAJ']*hdul[0].header['BMIN'] * u.deg**2).to(u.arcsec**2).value
    
    mommax = np.nanmax(sci)*abeam # value of the most intense pixel in Jy*km/s
    
    # prepare coordinates to get the fitted mask
    scix = np.arange(np.shape(sci)[0])
    scicoords = np.stack(np.meshgrid(scix, scix), axis=2).astype('float64')
    
    # elliptical 4sigma mask about the mean of the moment signal
    window, FWHMa = line_mask(sci, cutout, use_stds)
    
    # sigma used to determine the mask
    sigx, sigy = FWHMa
    
    # get data from the image cube 
    hdul = fits.open(cube_file)
    cube = hdul['PRIMARY'].data
    cubewcs = wcs.WCS(hdul[0].header)
    cubecelwcs = cubewcs.sub(['celestial'])
    # first axis is empty so discard it
    cube = cube[0,:,:,:]
    # pass NaNs to zeros
    cube[np.where(np.isnan(cube))] = 0.0
    
    # sometimes the moment 0 map and the image cube have different spatial dimensions - this adjusts the 
    # mask to agree with the cube
    dimdiff = int((np.shape(sci)[0] - cube.shape[1])/2)
    if dimdiff != 0:
        window = window[dimdiff:-dimdiff, dimdiff:-dimdiff]
    
    # **** FIX: IF THEY HAVE DIFFERENT SPATIAL DIMENSIONS THE FWHMA WILL BE DIFFERENT BETWEEN THE WINDOW AND 
    # THE IMAGE CUBE
    
    # apply mask to cube
    masked_cube = cube*window
    
    # get spectral profile
    total_i = np.sum(masked_cube, axis=(1,2))*u.Jy
    
    # change the bandwidth hertz units in the header to km/s
    # ref: https://keflavich-astropy.readthedocs.io/en/latest/units/equivalencies.html#a-slightly-more-complica
    # ted-example-spectral-doppler-equivalencies

    restfreq = (hdul['PRIMARY'].header['RESTFRQ']*u.Hz).to(u.GHz)  # rest frequency of 12 CO 2-1 in GHz

    # *** THIS DOESN'T WORK IN RELATIVE UNITS - need to get the vel value of each channel and subtract them 
    # from each other

    freq_to_vel = u.doppler_radio(restfreq)
    nchans = masked_cube.shape[0]
    chan_idx = np.stack((np.zeros(nchans), np.zeros(nchans), np.arange(nchans), np.zeros(nchans)), axis=1)
    chan_freqs = cubewcs.array_index_to_world_values(chan_idx)[:,2]

    chan_vels = (chan_freqs*u.Hz).to(u.km / u.s, equivalencies=freq_to_vel)
    meanvel = np.median(chan_vels)
    p0[1] = meanvel.value

    # bandwidth
    deltanu = -np.diff(chan_vels)[0]

    # multiply by clean beam area (in arcseconds squared ***)
    # if the image cube beam is different from the moment one, that's accounted for here automatically because
    # the header is called individually
    abeam = (np.pi*hdul[0].header['BMAJ']*hdul[0].header['BMIN'] * u.deg**2).to(u.arcsec**2).value
    
    # also need to find the area of the window region and divide by that area, because we're integrating over it
    # currently assuming the ellipse has no rotation, so the sigma values are just in x and y in native pixels
    centerpix = cube.shape[2]//2 # middle pixel coordinate
    skycoordx1, skycoordx2 = cubecelwcs.pixel_to_world([[centerpix, centerpix], [centerpix + sigx, centerpix]], 
                                              [centerpix, centerpix])
    skycoordy1, skycoordy2 = cubecelwcs.pixel_to_world([[centerpix, centerpix], [centerpix, centerpix + sigy]],
                                                   [centerpix, centerpix])
    
    # semi-major and -minor axes in arcseconds
    rega = skycoordx2.separation(skycoordx1).to(u.arcsec)[0]
    regb = skycoordy2.separation(skycoordy1).to(u.arcsec)[1]
    
    # area of the 4sigma window region in arcseconds squared
    areg = np.pi * rega * regb * 16

    # (Jy/beam)*beam = Jy integrated over the window region divided by the region's area
    total_i = total_i * abeam / areg.value
    
    # fit to a gaussian
    opt, cov = curve_fit(gaussian, chan_vels.value, total_i.value, p0=p0)
    
    # find the center value of the signal in Hz
    mean = (opt[1]*u.km/u.s).to(u.Hz, equivalencies=freq_to_vel)
    
    # plot the intensity data to make sure the fit worked
    x = np.linspace(np.min(chan_vels.value), np.max(chan_vels.value), num=500)
    plt.plot(chan_vels, total_i)
    plt.ylabel('Integrated Intensity (mJy)')
    plt.xlabel('Velocity (km/s, LSRK)')
    plt.plot(x, gaussian(x, *opt))
    
    # find area under gaussian fit - integration means -> Jy*km/s
    specmax = int_gaussian(*opt)
    
    return mommax*u.Jy*u.km/u.s, specmax*u.Jy*u.km/u.s, mean



def co_line_luminosity(vel_int_flux, nu_obs, z, cosmo = FlatLambdaCDM(H0=75, Om0=0.3)):
    '''equation to find the line luminosity of CO. Taken from Solomon and Vanden Bout (2005). 
       INPUTS: vel_int_flux: velocity integrated flux (Jy.km/s)
               nu_obs: The center observed frequency of the CO line (GHz)
               cosmo: an astropy.cosmology object describing the desired cosmology (needed to calculate the
                      luminosity distance). Default is definition in Webb (2015): flat with H0=75 km/s.Mpc and
                      Omega_dark matter = 0.7
               z: the redshift of the source
        RETURNS the CO line luminosity in solar luminosities
    '''
    DL = cosmo.luminosity_distance(z) # luminosity distance in Mpc
    return (3.25e7 * vel_int_flux * np.power(nu_obs, -2) * DL**2 * np.power(1+z, -3)).value



def M_gas(L_line, r_21=0.85, alpha_CO=1):
    '''equation to find total gas mass from a CO line luminosity. taken from Noble et al. (2017). Defaults
       are from the ALMA proposal
       Returns total gas mass in solar masses
    '''
    return alpha_CO * (L_line / r_21)



def get_M_gas(moment_file, cube_file, z, cutout=(95, 120), p0=[0.00004, 131700, 100, 0], 
              cosmo=FlatLambdaCDM(H0=75, Om0=0.3), r_21=0.85, alpha_CO=1, use_momflux=False):
    ''' Wrapper function encompassing all of the others - passing an image to this function should return the total gas mass
        in solar masses corresponding to the CO detection in the image.
        INPUTS: moment_file: file containing the moment 0 plot of the line detection
                cube_file: image cube
                z: the redshift of the target galaxy
                cutout: (xmin,xmax): the region (in native image pixels) of the moment0 map the ellipse fitter should look
                                     for signal
                p0: [amplitude, mean, standard deviation, y offset]: the best guess parameters for the gaussian fit to the 
                                                                     spectral profile
                cosmo: FlatLambdaCDM object describing the desired cosmology (used to calculate the luminosity distance)
                use_momflux: if True, the calculation will use the peak intensity value from the moment 0 map, instead of the 
                             value from fitting to the spectral profile of the detection
        RETURNS: Total gas mass of the source in the moment 0 plot in solar masses
    
    '''
    
    # first, find the total velocity integrated flux of the co signal
    momflux, specflux, centfreq = get_velocity_int_flux(moment_file, cube_file, cutout, p0)
    
    # use the flux found by getting the area under a gaussian fit to the spectral profile of the detection
    if use_momflux == True:
        totflux = momflux
    else:
        totflux = specflux
    
    # get the center frequency of the detection from the gaussian fit (change to GHz)
    centfreq = centfreq.to(u.GHz)
    
    # line luminosity in solar luminosities
    Lco = co_line_luminosity(totflux, centfreq, z, cosmo)
    
    # gas mass in solar masses
    Mgas = M_gas(Lco, r_21, alpha_CO)
    
    return Mgas


def plot_spectral_profile(file, nFWHMa, sigma=None):
    ''' Function to plot a 1D spectral profile of a given region about the center of a pointing. a, b
        are the beam semimajor and semiminor axes, respectively, and are in arcminutes. 
    '''
    
    # get the data from the file
    hdu = fits.open(file)[0]
    data = hdu.data
    # remove the stokes axis
    data = data[0,:,:,:]
    # map NaNs to zero
    data[np.where(np.isnan(data))] = 0.

    # get the semimajor and semiminor axis values from the header
    a = (hdu.header['BMAJ']*u.deg)
    b = (hdu.header['BMIN']*u.deg)

    # get wcs
    datwcs = wcs.WCS(hdu.header).sub(['celestial'])
    alldatwcs = wcs.WCS(hdu.header)

    # define skycoord objects for the extent of the axes to change into pix
    center = SkyCoord(hdu.header['CRVAL1']*u.deg, hdu.header['CRVAL2']*u.deg, frame='fk5')
    ext = SkyCoord(hdu.header['CRVAL1']*u.deg - nFWHMa*a, hdu.header['CRVAL2']*u.deg + nFWHMa*b, frame='fk5')

    # change the wcs into native pixels
    centerpix = datwcs.world_to_pixel(center)
    extpix = datwcs.world_to_pixel(ext)

    # center values are the center of the ellipse, the lengths are the differences between center and ext
    arad, brad = int(extpix[0] - centerpix[0]), int(extpix[1] - centerpix[1])

    # define the Ellipse2D object IGNORING ANY ANGLE IN THE BEAM FOR NOW***
    window = Ellipse2D(1, int(centerpix[0]), int(centerpix[1]), arad, brad, 0)

    # to get an actual array, need to pass it a meshgrid of x and y coordinates
    datx = np.arange(np.shape(data)[1])
    x, y = np.meshgrid(datx, datx)

    # apply the mask
    data = data * window(x,y)
    
    # sum over the spatial axes
    total_i = np.sum(data, axis=(1,2))*u.Jy
    
    # divide by the area of the window in arcsecs to normalize
    abeam = (np.pi * a * b).to(u.arcsec**2)
    total_i = total_i/abeam
    
    # change the bandwidth hertz units in the header to km/s
    # ref: https://keflavich-astropy.readthedocs.io/en/latest/units/equivalencies.html#a-slightly-more-complica
    # ted-example-spectral-doppler-equivalencies

    restfreq = (hdu.header['RESTFRQ']*u.Hz).to(u.GHz)  # rest frequency of 12 CO 2-1 in GHz

    # *** THIS DOESN'T WORK IN RELATIVE UNITS - need to get the vel value of each channel and subtract them 
    # from each other

    freq_to_vel = u.doppler_radio(restfreq)
    nchans = data.shape[0]
    chan_idx = np.stack((np.zeros(nchans), np.zeros(nchans), np.arange(nchans), np.zeros(nchans)), axis=1)
    chan_freqs = alldatwcs.array_index_to_world_values(chan_idx)[:,2]

    chan_vels = (chan_freqs*u.Hz).to(u.km / u.s, equivalencies=freq_to_vel)
    
    
    # add smoothing as an option
    if sigma is not None:
        total_i_sm = gaussian_filter1d(total_i, sigma=sigma, truncate=5)
    
    # plot
    plt.plot(chan_vels, total_i, zorder=10, lw=1, label='Raw')
    if sigma is not None:
        plt.plot(chan_vels, total_i_sm, zorder=11, lw=3, label='Smoothed (std = {})'.format(sigma))
        plt.legend()
    plt.ylabel('Integrated Intensity (Jy)')
    plt.xlabel('Velocity (km/s, LSRK)')
    plt.axhline(0, color='k')
   
    
    return    