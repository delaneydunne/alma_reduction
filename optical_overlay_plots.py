import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Ellipse



def plot_contour_overlay(files, optmin, optmax, xlims, ylims, freqlabel=False, whole_image=False, use_Hz=False):
    ''' function to plot the contours of a radio spectral line integrated flux over an optical image.
        files should be structured ('optical file path', 'moment 0 file path') to fits files. Optmin and optmax are
        the vmin, vmax of the colourmap for the optical image, and lims are the (min,max) limits of the
        coordinates that should be plotted, in native pixels to the radio contours.
    '''

    # open the images
    opthdu = fits.open(files[0])[0]
    momhdu = fits.open(files[1])[0]

    # WCS info for both images
    optwcs = wcs.WCS(opthdu.header)
    momwcs = wcs.WCS(momhdu.header)

    momwcs = momwcs.sub(['celestial'])

    # define contour levels starting at 2 sigma away from the most intense flux and increasing in 1sigma intervals
    momdata = momhdu.data[0,0,:,:]
    mommax = np.nanmax(momdata)
    momstd = np.nanstd(momdata)
    levels = np.ones(8)*mommax
    stds = np.arange(10, 2, step=-1)*momstd
    levels = (levels - stds)

    # plot the contours first
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection=momwcs)
    ax.contour(momdata, colors='red', levels=levels)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # WCS transform to line the optical image up
    opt_transform = ax.get_transform(optwcs)
    
    # plot the optical
    im = ax.imshow(opthdu.data, cmap=plt.cm.gray, transform=opt_transform, vmin=optmin, vmax=optmax)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.07)
    
    if freqlabel == True:
        # add a legend showing the central wavelength of the moment contours
        custom_lines = [Line2D([0], [0], color='red', lw=2)]

        # get velocity of the moment frequency
        nuobs = momhdu.header['cfreq']
        if use_Hz == False:
            # coordinate transformation from hz to km/s
            freq_to_vel = u.doppler_radio(momhdu.header['RESTFRQ']*u.Hz)
            vel = (nuobs*u.Hz).to(u.km / u.s, equivalencies=freq_to_vel)

            ax.legend(custom_lines, ['{:.1f}'.format(vel)])

        else:
            vel = (nuobs*u.Hz).to(u.GHz)
            ax.legend(custom_lines, ['{:.3f}'.format(vel)])
        
        
    # get the semimajor and semiminor axis values from the header
    a = (momhdu.header['BMAJ']*u.deg)
    b = (momhdu.header['BMIN']*u.deg)


    # define skycoord objects for the extent of the axes to change into pix
    center = SkyCoord(momhdu.header['CRVAL1']*u.deg, momhdu.header['CRVAL2']*u.deg, frame='fk5')
    ext = SkyCoord(momhdu.header['CRVAL1']*u.deg - a, momhdu.header['CRVAL2']*u.deg + b, frame='fk5')

    # change the wcs into native pixels
    centerpix = momwcs.world_to_pixel(center)
    extpix = momwcs.world_to_pixel(ext)

    # center values are the center of the ellipse, the lengths are the differences between center and ext
    arad, brad = int(extpix[0] - centerpix[0]), int(extpix[1] - centerpix[1])  
    
    

    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    
    if whole_image == True:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        beambkg = Rectangle((xlim[0], ylim[0]), width=arad*1.5, height=brad*1.5, facecolor='w',
                            edgecolor='k', zorder=10, alpha=0.9)
        beam = Ellipse((xlim[0] + arad*0.75, ylim[0] + brad*0.75), 
                       width=arad, height=brad, facecolor='w', edgecolor='k', zorder=11)
        
    else:
        ax.set_xlim([*xlims])
        ax.set_ylim([*ylims])
        
        beambkg = Rectangle((xlims[0], ylims[0]), width=arad*1.5, height=brad*1.5, facecolor='w', 
                            edgecolor='k',zorder=10, alpha=0.9)
        beam = Ellipse((xlims[0] + arad*0.75, ylims[0] + brad*0.75), 
                       width=arad, height=brad, facecolor='w', edgecolor='k', zorder=11)
        
    ax.add_patch(beambkg)
    ax.add_patch(beam)
        

    return




def plot_velocity_overlay(files, optmin, optmax, xlims, ylims):
    ''' function to plot the filled contours of a radio velocity field over an optical image.
        files should be structured ('optical file path', 'moment 1 file path') to fits files. Optmin and optmax are
        the vmin, vmax of the colourmap for the optical image.
    '''

    opthdu = fits.open(files[0])[0]
    momhdu = fits.open(files[1])[0]

    # read in WCS information for both images
    optwcs = wcs.WCS(opthdu.header)
    momwcs = wcs.WCS(momhdu.header)

    momwcs = momwcs.sub(['celestial'])

    # define contour levels for the velocity field. Contours start at 2 sigma and increase by 1sigma intervals
    momdata = momhdu.data[0,0,:,:]
    mommax = np.nanmax(momdata)
    momstd = np.nanstd(momdata)
    levels = np.ones(8)*mommax
    stds = np.arange(10, 2, step=-1)*momstd
    levels = (levels - stds)

    # plot the velocity field
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection=momwcs)
    cont = ax.contourf(momdata, cmap=plt.cm.Spectral)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # WCS transform to line the images up
    opt_transform = ax.get_transform(optwcs)
    
    # plot the optical
    im = ax.imshow(opthdu.data, cmap=plt.cm.gray, transform=opt_transform, vmin=optmin, vmax=optmax)

    # colorbar will correspond to the velocity field, in units of km/s
    cbar = plt.colorbar(cont, fraction=0.046, pad=0.07)
    
    cbar.ax.set_ylabel("Velocity (km/s)")
    
    # get the semimajor and semiminor axis values from the header
    a = (momhdu.header['BMAJ']*u.deg)
    b = (momhdu.header['BMIN']*u.deg)


    # define skycoord objects for the extent of the axes to change into pix
    center = SkyCoord(momhdu.header['CRVAL1']*u.deg, momhdu.header['CRVAL2']*u.deg, frame='fk5')
    ext = SkyCoord(momhdu.header['CRVAL1']*u.deg - a, momhdu.header['CRVAL2']*u.deg + b, frame='fk5')

    # change the wcs into native pixels
    centerpix = momwcs.world_to_pixel(center)
    extpix = momwcs.world_to_pixel(ext)

    # center values are the center of the ellipse, the lengths are the differences between center and ext
    arad, brad = int(extpix[0] - centerpix[0]), int(extpix[1] - centerpix[1])  
    
    

    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    
    
    ax.set_xlim([*xlims])
    ax.set_ylim([*ylims])

    beambkg = Rectangle((xlims[0], ylims[0]), width=arad*1.5, height=brad*1.5, facecolor='w', 
                        edgecolor='k',zorder=10, alpha=0.9)
    beam = Ellipse((xlims[0] + arad*0.75, ylims[0] + brad*0.75), 
                   width=arad, height=brad, facecolor='w', edgecolor='k', zorder=11)

    ax.add_patch(beambkg)
    ax.add_patch(beam)

    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    ax.set_xlim([*xlims])
    ax.set_ylim([*ylims])
    return



'''
SPECIAL 2-LINE MOMENT PLOT FOR XMM-8
opthdu = fits.open(xmm5_cubefiles[1])[0]
momhdu1 = fits.open(xmm5_cubefiles[5])[0]
momhdu2 = fits.open(xmm5_cubefiles[6])[0]

optwcs = wcs.WCS(opthdu.header)
momwcs = wcs.WCS(momhdu1.header)

momwcs = momwcs.sub(['celestial'])

%matplotlib notebook

momdata1 = momhdu1.data[0,0,:,:]
mommax1 = np.nanmax(momdata1)
momstd1 = np.nanstd(momdata1)
levels1 = np.ones(8)*mommax1
stds1 = np.arange(10, 2, step=-1)*momstd1
levels1 = (levels1 - stds1)

momdata2 = momhdu2.data[0,0,:,:]
mommax2 = np.nanmax(momdata2)
momstd2 = np.nanstd(momdata2)
levels2 = np.ones(8)*mommax2
stds2 = np.arange(10, 2, step=-1)*momstd2
levels2 = (levels2 - stds2)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection=momwcs)
cs1 = ax.contour(momdata1, colors='red', levels=levels1)
cs2 = ax.contour(momdata2, colors='lime', levels=levels2)

custom_lines = [Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='lime', lw=2)]

ax.legend(custom_lines, ['52574.1 km/s', '52208.3 km/s'])

xlim = ax.get_xlim()
ylim = ax.get_ylim()

opt_transform = ax.get_transform(optwcs)

im = ax.imshow(opthdu.data, cmap=plt.cm.gray, transform=opt_transform, vmin=1500, vmax=1700)

cbar = plt.colorbar(im, fraction=0.046, pad=0.07)

# get the semimajor and semiminor axis values from the header
a = (momhdu1.header['BMAJ']*u.deg)
b = (momhdu1.header['BMIN']*u.deg)


# define skycoord objects for the extent of the axes to change into pix
center = SkyCoord(momhdu1.header['CRVAL1']*u.deg, momhdu1.header['CRVAL2']*u.deg, frame='fk5')
ext = SkyCoord(momhdu1.header['CRVAL1']*u.deg - a, momhdu1.header['CRVAL2']*u.deg + b, frame='fk5')

# change the wcs into native pixels
centerpix = momwcs.world_to_pixel(center)
extpix = momwcs.world_to_pixel(ext)

# center values are the center of the ellipse, the lengths are the differences between center and ext
arad, brad = int(extpix[0] - centerpix[0]), int(extpix[1] - centerpix[1])  



ax.set_xlabel('RA')
ax.set_ylabel('DEC')


beambkg = Rectangle((83, 83), width=arad*1.5, height=brad*1.5, facecolor='w', 
                    edgecolor='k',zorder=10, alpha=0.9)
beam = Ellipse((83 + arad*0.75, 83 + brad*0.75), 
               width=arad, height=brad, facecolor='w', edgecolor='k', zorder=11)

ax.add_patch(beambkg)
ax.add_patch(beam)


ax.set_xlim([83, 133])
ax.set_ylim([83, 133])
'''


