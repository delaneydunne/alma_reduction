from astropy.io import fits
from astropy import wcs
import astropy.units as u
from photutils import detect_sources, source_properties
#ref: https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceProperties.html#photutils.segmentation.SourceProperties
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
from matplotlib.colors import ListedColormap
from astropy.visualization import astropy_mpl_style
from astropy.convolution import Tophat2DKernel
from astropy.modeling.functional_models import Ellipse2D
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from math import floor
plt.style.use(astropy_mpl_style)

def find_optical_centriods(file, nstds, npix):
    ''' Search in an image for sources above a certain threshold, and return an array of their 
        centroid coordinates in RA and DEC (as skycoords)
        INPUTS: file: file pointer to the image
                nstds: the number of standard deviations above the image mean at which the threshold
                       should be set
                npix: the minimum number of clumped pixels above the threshold that define a source
        OUTPUTS: skycoords: list of source centroids in wcs units
                 sources: photutils segmentation image containing the sources
    '''
    
    # load in data from fits
    hdu = fits.open(file)[0]
    sci = hdu.data
    sciwcs = wcs.WCS(hdu.header)
    
    # make a threshold image
    tval = np.nanmean(sci) + nstds * np.nanstd(sci)
    threshold = np.ones(np.shape(sci)) * tval
    
    # get a segmentation image containing the sources
    sources = detect_sources(sci, threshold, npix, connectivity = 4)
    cat = source_properties(sci, sources)
    
    centcoords = []
    
    # read the centroids into an array
    for source in cat:
        centcoords.append([source.xcentroid.value, source.ycentroid.value])
        
    centcoords = np.array(centcoords)
    
    # use wcs to transfer the optical pixels into co pixels
    skycoords = wcs.utils.pixel_to_skycoord(centcoords[:,0], centcoords[:,1], sciwcs)
    
    # return the array of centroids (as skycoords) and the segmentation image
    return skycoords, sources, sciwcs
    


def find_blobs(file, optfile, pbfile, nstds, npix, ndist, noptstds = 0.25):
    
    # find where the sources are in the optical file
    opthdul = fits.open(optfile)
    optsci = opthdul['PRIMARY'].data
    optwcs = wcs.WCS(opthdul[0].header)
    
    # read in co data
    hdul = fits.open(file)
    sci = hdul['PRIMARY'].data
    momwcs = wcs.WCS(hdul[0].header)        
    # get rid of the fourth dimension, which is empty
    sci = sci[0,:,:,:]
    
    # read in the primary beam response
    pbhdul = fits.open(pbfile)
    pb = pbhdul['PRIMARY'].data
    # remove the empty stokes axis
    pb = pb[0,:,:,:]
    
        
    # the primary beam correction DIVIDES the sky brightness distribution by the antenna response, because
    # the original visibilities are convolved with the antenna response, so their Fourier transform will 
    # be multiplied by it. To search for sources without having to deal with this uneven noise distribution,
    # multiply the primary beam back in
    
    # sometimes the pb file has a different number of channels from the science one: if this is the case
    # just treat the first channel as the pb response for all frequencies
    if pb.shape[0] != sci.shape[0]:
        sci = sci*pb[0, :, :]
        
    else:
        sci = pb*sci

    sourcelist = []
    bloblist = []
    zzzpix = []
    
    fsize = sci.shape[1]
    nchannels = sci.shape[0]
    print(nchannels)
    
    # find sources in the optical image and plot the segmentation image to be sure it worked
    skycoords, optsources = find_optical_centroids(optfile, noptstds, npix)
    plt.pcolormesh(optsources)
    
    momcoords = skycoords.to_pixel(momwcs)
    momcoordsx, momcoordsy = momcoords[0], momcoords[1]
    momcoords = np.stack((momcoordsx, momcoordsy), axis=1)
    cocoords = momcoords[np.where(np.logical_and(np.logical_and(momcoordsx > 1, momcoordsx < fsize - 2),
                                                np.logical_and(momcoordsy > 1, momcoordsy < fsize - 2)))]

    
    # read properties of the sources into the empty array poss
            
    poss = []
    perims = []
    
    # first pass the nans to zeros to make the data useable
    sci[np.where(np.isnan(sci))] = 0.

    
    # loop through the channels of the cube. for each, make a threshold image using the mean and the standard
    # deviation (start with 2 sigma above mean) and then use photutils' detectsources to find large clumps
    for channel in range(sci.shape[0]):
        
        fm = sci[channel,:,:]
        
        # make a threshold image using a tophat kernel for now because the outer pixels are weighted more 
        # heavily in the images:

        # make the kernel 
        #  smallkern = Tophat2DKernel(50).array
        # kern = np.zeros(np.shape(fm))
    
        # val = round((fsize - smallkern.shape[0])/2 + 0.5)
        # kern[(val - 1):-val, (val - 1):-val] = smallkern

        # kern[np.where(kern != 0.)] = 1.
        # outidx = np.where(kern == 0.)

        mean = np.mean(fm)
        std = np.std(fm)
        threshold = np.ones(fm.shape) * (mean + (std * nstds))

        #innerpix = fm*kern
        #instd = np.std(innerpix)

        #outerpix = fm[outidx]
        #outstd = np.std(outerpix)

        #threshold = kern * (mean + (instd * nstds))
        #threshold[outidx] = mean + (outstd * nstds)
        
        sources = detect_sources(fm, threshold, npix, connectivity=4) 
        if sources == None:
            continue
        
        cat = source_properties(fm, sources)
        
        sourcelist.append(sources)
        bloblist.append(cat)


        for n in range(len(cat)):

            coords = cat[n].coords
            totEl = np.sum(fm[coords]) #sum over each pixel in the frame belonging to the source
                                          # to find the total number of electrons making up the source

            size = len(coords[0])


            #fitted ellipse properties of the source
            a = cat[n].semimajor_axis_sigma.value * r
            b = cat[n].semiminor_axis_sigma.value * r
            theta = cat[n].orientation.value


            #centroid of the source in native (detector) pixels
            x = cat[n].xcentroid.value
            y = cat[n].ycentroid.value

            if np.isnan(x):
                x,y = np.mean(coords, axis=1)

            #ellipticity (1-a/b where a and b are the lengths of the semimajor and semiminor axes)
            #ref: https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceProperties.
            #       html#photutils.segmentation.SourceProperties.ellipticity
            ell = cat[n].ellipticity.value


            #read all values into the 'poss' array
            poss.append([channel, x, y, a, b, theta, totEl, size, -1])
            perims.append(cat[n].bbox.extent) #so surrounding pixels can be returned to be plotted


            #raise a warning if a source is returning no centroid
            if np.isnan(cat[n].xcentroid.value):
                print(file, channel)

            #END for n in cat
            
        # END for channel in file
        
        
    # mark sources that appear in the same spot (+/- ndist pix) as one of the co coordinates

    poss = np.array(poss)

    for n in range(len(poss)):
        x = poss[n,1]
        y = poss[n,2]
        
        for m in range(len(cocoords)):
            if cocoords[m,0] < (x+ndist) and cocoords[m,0] > (x-ndist) and cocoords[m,1] < (y+ndist) and cocoords[m,1] > (y-ndist):
                poss[n,8] = m
            

#         optmatchidx = np.where(np.logical_and(np.logical_and(cocoords[:,0] < (x+ndist), cocoords[:,0] > (x-ndist)),
#                               np.logical_and(cocoords[:,1] < (y+ndist), cocoords[:,1] > (x-ndist))))
#         optmatchidx = np.array(optmatchidx)

#         if optmatchidx[0].size > 0:
#             poss[n,8] = optmatchidx[0,0]




#     #mark sources that appear in the same spot (+/-ndist pix) across several frames with a 1 in the [n,10]th column
    
#     poss = np.array(poss)
    
#     for n in range(len(poss)):
    
#         x = poss[n,1]
#         y = poss[n,2]
        
#         if poss[n,7] > 25:
#             nd = ndist + 10
#         else:
#             nd = ndist

#         for m in np.arange(n+1, len(poss)):

#             if poss[m,1] < (x+nd) and poss[m,1] > (x-nd) and poss[m,2] < (y+nd) and poss[m,2] > (y-nd):
#                 poss[m,8] = n
#                 poss[n,8] = m
#                 continue
                
#             else:
#                 break
                
        

    return sourcelist, bloblist, perims, poss, sci, cocoords


def blobVisualize(data, poss, incoords, fms):
    ''' Marks the sources found using find_blobs on a series of subplots showing the input image cube channels
        data should be the actual sci data from the cube, poss the poss array returned by find_blobs, incoords the 
        coordinates to the optical sources also marked by find_blobs, and frames the desired frames to plot (should be 
        no bigger than ~20 frames because otherwise the image won't load.
    '''
    
    num = int(len(fms)/2 + 0.5)
    
    fig, axs = plt.subplots(num, 2, figsize=(11, num*7))
    
    vmin = np.min(data[fms,:,:])
    vmax = np.max(data[fms,:,:])
    
    for idx, n in enumerate(fms):
        
        i, j = divmod(idx, 2)
            
        fmdata = data[n,:,:]
        
        im = axs[i, j].pcolormesh(fmdata, cmap=plt.cm.gist_rainbow, vmin=vmin, vmax=vmax)
        
        axs[i, j].set_xlabel("Frame: {}".format(n + 1))
        
        fmidx = np.where(poss[:,0] == n)[0]
        
        for sidx in fmidx:
                  
            sourcex = poss[sidx, 1]
            sourcey = poss[sidx, 2]
            
            rep = poss[sidx, 8]
            
            if rep == -1:
                ec = 'g'
            else:
                ec = 'white'
        
        
            axs[i, j].scatter(sourcex, sourcey, marker="o", facecolors='none', edgecolor=ec, s=40)
        
        axs[i, j].scatter(incoords[:,0], incoords[:,1], marker="o", facecolors='none', edgecolor='k', s=100)
        
        if j == 1:
            # put colorbars on the rightmost plots only to avoid crowding (all have the same vmin and vmax)
            cbar = fig.colorbar(im, ax = axs[i, j], fraction=0.046, pad=0.07)
        
        
    return



def find_consec(arr):
    ''' Given an array of channels, return only the values of consecutive channels '''
    consec_arr = np.concatenate((arr[np.where(np.diff(arr) == 1)],
                                       arr[np.add(np.where(np.diff(arr) == 1), 1)[0]]))

    consec_arr.sort()

    return np.unique(consec_arr)


def image_stack(files, nchans, rwindow, stackcenter=None, profonly=False, imonly=False, sigma=None):

    chanrange = nchans // 2

    # define a circular window about the center of the file to integrate over to get the spectral profile
    # currently, the program is cutting a big 160x160 square region out of the file and then clipping out
    # a circular window with radius rwindow. nothing's implemented to deal with cases where this goes out of frame
    window = Ellipse2D(1, 80, 80, rwindow, rwindow, 0)
    # to get an actual array, need to pass it a meshgrid of x and y coordinates
    datx = np.arange(160)
    x, y = np.meshgrid(datx, datx)

    mcube =[]
    mprof = []

    # read each relevant frame of the cubes into a single master array to average down
    for i, file in enumerate(files):
        # load in the data and WCS system from the fits file
        hdu = fits.open(file)[0]
        data = hdu.data
        datawcs = wcs.WCS(hdu.header).sub(['celestial'])
        # remove the empty stokes axis and keep only the channels and pixels around the center
        data = data[0, :, :, :]

        imchans = data.shape[0]
        impix = data.shape[1]

        centchan = imchans // 2
        minc, maxc = centchan - chanrange, centchan + chanrange

        if not stackcenter:
            centpix = impix // 2

            # if the image is small enough that this number +/- 80 will extend past the edges of the frame,
            # pad the edges with -99 (so they'll show up clearly in the final stacked image)
            if centpix < 80:
                diff = 80 - centpix
                npad = diff + 5
                data = np.pad(data, ((0, 0), (npad, npad), (npad, npad)), 'constant', constant_values=-99)
                centpix += npad

            minp, maxp = centpix - 80, centpix + 80

            data = data[minc:maxc, minp:maxp, minp:maxp]

        else:
            centx, centy = stackcenter[0], stackcenter[1]

            # here, pad selectively - if the x axis is going to over extend, fix it, and then check the y axis
            # individually
            if centx < 80:
                diff = 80 - centx
                npad = diff + 5
                data = np.pad(data, ((0, 0), (npad, 0), (0, 0)), 'constant', constant_values=-99)
                centx = centx + npad

            elif centx > (impix - 80):
                diff = centx - (impix - 80)
                npad = diff + 5
                data = np.pad(data, ((0, 0), (0, npad), (0, 0)), 'constant', constant_values=-99)

            if centy < 80:
                diff = 80 - centy
                npad = diff + 5
                data = np.pad(data, ((0, 0), (0, 0), (npad, 0)), 'constant', constant_values=-99)
                centy = centy + npad

            elif centy > (impix - 80):
                diff = centy - (impix - 80)
                npad = diff + 5
                data = np.pad(data, ((0, 0), (0, 0), (0, npad)), 'constant', constant_values=-99)

            minx, maxx = centx - 80, centx + 80
            miny, maxy = centy - 80, centy + 80

            centpix = centx

            data = data[minc:maxc, minx:maxx, miny:maxy]

        # map NaNs to zero
        data[np.where(np.isnan(data))] = 0.

        # multiply the data by window to clip out only the desired area around the center
        data = data * window(x, y)
        
        # change rwindow into arcseconds according to the specific wcs of each image to divide
        # by the area of the window when making the spectral profile
        skycoordx1, skycoordx2 = datawcs.pixel_to_world([[centpix, centpix], 
                                                         [centpix + rwindow, centpix + rwindow]], 
                                                        [centpix, centpix])
        # semi-major and -minor axes in arcseconds
        rega = skycoordx2.separation(skycoordx1).to(u.arcsec)[0]
        regb = skycoordx2.separation(skycoordx1).to(u.arcsec)[1]

        # area of the 4sigma window region in arcseconds squared
        areg = np.pi * rega * regb
        
        # find beam area and multiply by the beam?
        abeam = (np.pi*hdu.header['BMAJ']*hdu.header['BMIN'] * u.deg**2).to(u.arcsec**2).value
        
        # for each cube, get an individual spectral profile of the data in the window by summing
        # over both spatial axes
        prof = np.sum(data, axis=(1, 2)) / areg

        mprof.append(prof)
        
        # append onto mcube
        if i == 0:
            mcube = data
        else: 
            mcube = np.concatenate((mcube, data), axis=0)
            
    
    ''' for the first subplot: single large average over all relevant channels (centered around the center of each cube) '''
    
    # average down the channel axis
    avg = np.mean(mcube, axis=0)
    
    ''' second subplot: spectral profile of pix in window - average down the list of files '''
    mprof = np.array(mprof)
    
    avgprof = np.mean(mprof, axis=0)
    
    # add smoothing as an option
    if sigma is not None:
        avgprofsm = gaussian_filter1d(avgprof, sigma=sigma, truncate=5)
    
    # also have to divide by area over which you're integrating
    
    if imonly == True:
        ''' plot first only'''
        fig,axs = plt.subplots(1, figsize=(8, 6))

        xmin, xmax = 80 - rwindow, 80 + rwindow

        im = axs.pcolormesh(avg[xmin:xmax, xmin:xmax])
        axs.set_aspect(aspect=1)
        cbar = plt.colorbar(im, ax=axs)
        cbar.ax.set_ylabel('Intensity (Jy/Beam)')
        
        
    elif profonly == True:
        
        fig,axs = plt.subplots(1, figsize=(9,6))

        # plot the intensity data to make sure the fit worked
        axs.bar(np.arange(-chanrange, chanrange), avgprof, width=1)
        
        if sigma is not None:
            axs.plot(np.arange(-chanrange, chanrange), avgprofsm, lw=3, label='Smoothed (std = {})'.format(sigma),
                     color='orange', zorder=10)
            axs.legend()
        axs.axhline(0, color='k')
        axs.set_ylabel('Intensity (Jy/Beam)')
        axs.set_xlabel('Distance from Central Channel')
        
    else:

        ''' plot first'''
        fig,axs = plt.subplots(2, figsize=(8, 6))

        xmin, xmax = 80 - rwindow, 80 + rwindow

        im = axs[0].pcolormesh(avg[xmin:xmax, xmin:xmax])
        axs[0].set_aspect(aspect=1)
        cbar = plt.colorbar(im, ax=axs[0])
        cbar.ax.set_ylabel('Intensity (Jy/Beam)')



        ''' plot second'''
        axs[1].bar(np.arange(-chanrange, chanrange), avgprof, width=1)
        if sigma is not None:
            axs[1].plot(np.arange(-chanrange, chanrange), avgprofsm, lw=3, label='Smoothed (std = {})'.format(sigma),
                        color='orange', zorder=10)
            axs[1].legend()
        axs[1].axhline(0, color='k')
        axs[1].set_ylabel('Intensity (Jy/Beam)')
        axs[1].set_xlabel('Distance from central channel')

    return avg, avgprof


def get_source_CRTF(cubefile, optfile, nstds=0.5, npix=5, r = 4., optext=(1, 200), v=False):
    # first figure out the coordinates covered by the CO pointing
    cubehdul = fits.open(cubefile)[0]
    cubewcs = wcs.WCS(cubehdul.header).sub(['celestial'])
    cubehdr = cubehdul.header
    RAdown = (cubehdr['CRVAL1'] - cubehdr['CRPIX1'] * cubehdr['CDELT1']) * u.deg
    RAup = (cubehdr['CRVAL1'] + cubehdr['CRPIX1'] * cubehdr['CDELT1']) * u.deg
    DECdown = (cubehdr['CRVAL2'] - cubehdr['CRPIX2'] * cubehdr['CDELT2']) * u.deg
    DECup = (cubehdr['CRVAL2'] + cubehdr['CRPIX2'] * cubehdr['CDELT2']) * u.deg

    # open the optical file
    opthdul = fits.open(optfile)[0]
    optwcs = wcs.WCS(opthdul.header)
    optdata = opthdul.data

    # use the photutils functions to find sources anywhere in the optical file
    # sources will be called a source if they contain at least npix pixels touching that have values above the
    # threshold level
    threshold = np.nanstd(optdata) * nstds + np.nanmean(optdata)

    # source is a photutils segmentation image, cat is a list of photutils source objects that have attributes
    # corresponding to their various properties
    sources = detect_sources(optdata, threshold, npix, connectivity=4)
    cat = source_properties(optdata, sources, wcs=optwcs)

    # define elliptical apertures covering the sources with the correct angle and eccentricity
    # r is a size parameter that's adjustable
    apertures = []
    for obj in cat:
        position = np.transpose((obj.xcentroid.value, obj.ycentroid.value))
        a = obj.semimajor_axis_sigma.value * r
        b = obj.semiminor_axis_sigma.value * r
        theta = obj.orientation.to(u.rad).value
        apertures.append(EllipticalAperture(position, a, b, theta=theta))

    # plot the optical image with sources overlaid to make sure they line up and there aren't too many/too few detected
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=cubewcs)
    ax.pcolormesh(optdata, vmin=optext[0], vmax=optext[1], cmap='gray')

    for aperture in apertures:
        aperture.plot(axes=ax, color='orange', lw=1.5, zorder=20)

    # open a file to store the region information
    fname = 'optregions.crtf'
    f = open(fname, 'w')
    f.write('#CRTF \n')

    pixtodeg = opthdul.header['CD1_1'] / cubehdul.header['CDELT1']

    factor = 4

    for obj in cat:
        ra = obj.sky_centroid.icrs.ra.deg
        dec = obj.sky_centroid.icrs.dec.deg
        if ra < RAup.value or ra > RAdown.value:
            continue
        elif dec < DECdown.value or dec > DECup.value:
            continue

        if v:
            print("passed: ra {} dec {}".format(ra, dec))

        xval, yval = skycoord_to_pixel(SkyCoord(ra * u.deg, dec * u.deg), wcs=cubewcs)

        if v:
            print(xval, yval)

        a = obj.semimajor_axis_sigma.value * factor
        b = obj.semiminor_axis_sigma.value * factor
        f.write('ellipse[[' + str(xval) + 'pix, ' + str(yval) + 'pix], [' + str(a) + 'pix, ' + str(b) + 'pix], ' +
                str(obj.orientation.value) + 'deg], coord=ICRS \n')

    f.close()

    return


def opt_image_stack(files, optfiles, nchans, rwindow, profonly=False, imonly=False, sigma=None,
                    noptstds=0.25):
    ''' Function to find all the sources in an optical image that fall into the FoV of a pointing, and
        to stack a circular window with radius rwindow around each of their centroids to see if theres
        faint flux at any of the locations.

        INPUTS: files: image cube files
                optfiles: optical files (MUST BE IN THE SAME ORDER AS THE FILES)
    '''

    nfiles = len(files)
    chanrange = nchans // 2

    # define a circular window about the center of the file to integrate over to get the spectral profile
    window = Ellipse2D(1, 80, 80, rwindow, rwindow, 0)
    # to get an actual array, need to pass it a meshgrid of x and y coordinates
    datx = np.arange(160)
    x, y = np.meshgrid(datx, datx)

    # load all image data into a single master image cube, placing channels the same distance from the center
    # next to each other
    mcube = []
    mprof = []

    for i, file in enumerate(files):
        hdu = fits.open(file)[0]
        data = hdu.data
        datawcs = wcs.WCS(hdu.header).sub(['celestial'])
        # remove the empty stokes axis and keep only the channels and pixels around the center
        data = data[0, :, :, :]
        # map nans to zero
        data[np.where(np.isnan(data))] = 0.

        # find the source centroids of the related optical file (in skycoords)
        optcoords, sources, optwcs = ss.find_optical_centroids(optfiles[i], noptstds, 10)  # ****

        # change the centroid values to native image cube pixels
        optcoords = optcoords.to_pixel(datawcs)

        # the channel indices apply for all sourcse in the file
        centchan = data.shape[0] // 2
        minc, maxc = centchan - chanrange, centchan + chanrange

        data = data[minc:maxc, :, :]

        # slap the whole data array into a larger array of zeros, so if the sources are
        # too close to the edge they'll just pick up extra zeros, which won't affect the statistics
        extdata = np.zeros(data.shape + np.array([0, 160, 160]))
        extdata[:, 80:-80, 80:-80] = data

        # SOME SORT OF VERIFICATION SYSTEM?
        # loop through the optical sources in the file and cut out a window around each
        for j in range(len(optcoords)):

            centx = optcoords[j, 0] + 80
            centy = optcoords[j, 1] + 80

            minx, maxx = centx, centx + 160
            miny, maxy = centy, centy + 160

            sdata = extdata[:, minx:maxx, miny:maxy]

            # multiply data by window to get only the desired area around the center
            sdata = sdata * window(x, y)

            # change rwindow into arcseconds according to the specific wcs of each image to divide
            # by the area of the window when making the spectral profile
            # here need the actual pixel values of the center so subtract 80 again
            centx -= 80
            centy -= 80
            skycoordx1, skycoordx2 = datawcs.pixel_to_world([[centx, centy],
                                                             [centx + rwindow, centy + rwindow]],
                                                            [centx, centy])
            # semi-major and -minor axes in arcseconds
            rega = skycoordx2.separation(skycoordx1).to(u.arcsec)[0]
            regb = skycoordx2.separation(skycoordx1).to(u.arcsec)[1]

            # area of the window region in arcseconds squared
            areg = np.pi * rega * regb

            #         # find beam area and multiply by the beam?
            #         abeam = (np.pi*hdu.header['BMAJ']*hdu.header['BMIN'] * u.deg**2).to(u.arcsec**2).value

            # for each optical source in the cube, get an individual spectral profile of the data in the window
            # by summing over both spatial axes
            prof = np.sum(sdata, axis=(1, 2)) / areg
            mprof.append(prof)

            # append onto mcube
            if i == 0 and j == 1:
                mcube = sdata
            else:
                mcube = np.concatenate((mcube, sdata), axis=0)

            # END for coord in optcoords
            # END for file in files

    ''' for the first subplot: single large average over all relevant channels (centered around the center of each cube) '''

    # average down the channel axis
    avg = np.mean(mcube, axis=0)

    ''' second subplot: spectral profile of pix in window - average down the list of files '''
    mprof = np.array(mprof)

    avgprof = np.mean(mprof, axis=0)

    # add smoothing as an option
    if sigma is not None:
        avgprofsm = gaussian_filter1d(avgprof, sigma=sigma, truncate=5)

    # also have to divide by area over which you're integrating

    if imonly == True:
        ''' plot first only'''
        fig, axs = plt.subplots(1, figsize=(8, 6))

        xmin, xmax = 80 - rwindow, 80 + rwindow

        im = axs.pcolormesh(avg[xmin:xmax, xmin:xmax])
        axs.set_aspect(aspect=1)
        cbar = plt.colorbar(im, ax=axs)
        cbar.ax.set_ylabel('Intensity (Jy/Beam)')


    elif profonly == True:

        fig, axs = plt.subplots(1, figsize=(9, 6))

        axs.plot(np.arange(-chanrange, chanrange), avgprof, lw=1, label='Raw')

        if sigma is not None:
            axs.plot(np.arange(-chanrange, chanrange), avgprofsm, lw=3, label='Smoothed (std = {})'.format(sigma))
            axs.legend()
        axs.axhline(0, color='k')
        axs.set_ylabel('Intensity (Jy/Beam)')
        axs.set_xlabel('Distance from Central Channel')

    else:

        ''' plot first'''
        fig, axs = plt.subplots(2, figsize=(8, 6))

        xmin, xmax = 80 - rwindow, 80 + rwindow

        im = axs[0].pcolormesh(avg[xmin:xmax, xmin:xmax])
        axs[0].set_aspect(aspect=1)
        cbar = plt.colorbar(im, ax=axs[0])
        cbar.ax.set_ylabel('Intensity (Jy/Beam)')

        ''' plot second'''
        axs[1].plot(np.arange(-chanrange, chanrange), avgprof, lw=1, label='Raw')
        if sigma is not None:
            axs[1].plot(np.arange(-chanrange, chanrange), avgprofsm, lw=3, label='Smoothed (std = {})'.format(sigma))
            axs[1].legend()
        axs[1].axhline(0, color='k')
        axs[1].set_ylabel('Intensity (Jy/Beam)')
        axs[1].set_xlabel('Distance from central channel')

    return avg, avgprof

