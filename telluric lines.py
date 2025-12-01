from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from astropy.convolution import Gaussian1DKernel, convolve
from numpy.fft import fft, ifft, fftshift

# =====================================================
# 1) LOAD TELESCOPE DATA
# =====================================================
file_path_data = "ADP.2020-06-26T11:14:16.300.fits"
hdul_data = fits.open(file_path_data)

for hdu in hdul_data:
    if isinstance(hdu, fits.BinTableHDU) and all(col in hdu.data.names for col in ['WAVE', 'FLUX']):
        data = hdu.data

        # original data
        wave = data['WAVE'][0]
        flux = data['FLUX'][0]


        # === select 6000–8000 Å ===
        mask = (wave >= 6000) & (wave <= 6800)
        wave_filtered = wave[mask]
        flux_filtered = flux[mask]

        #normalization
        #z = np.polyfit(wave_filtered, flux_filtered, 1)
        #p = np.poly1d(z)
        #cnt = p(wave_filtered)
        #flux_normalized = flux_filtered / cnt

        #flux_normalized /= np.nanmax(flux_normalized)  # forces the peak to exactly 1.0
        #flux_filtered = flux_normalized

        # === NEW NORMALIZATION FOR TELESCOPE ===

        # 1) fit a smooth continuum with a higher-order polynomial
        z = np.polyfit(wave_filtered, flux_filtered, 3)  # 3rd-degree is much more stable
        p = np.poly1d(z)
        cnt = p(wave_filtered)

        # 2) divide flux by continuum (this makes continuum ~1 everywhere)
        flux_normalized = flux_filtered / cnt

        # 3) DO NOT divide by maximum → this distorts continuum
        # flux_normalized /= np.nanmax(flux_normalized)  # remove this line completely

        # 4) optional: force median continuum to exactly 1
        cont_mask = flux_normalized > 0.9  # pixels near continuum
        scale = np.median(flux_normalized[cont_mask])
        flux_normalized /= scale

        # final result
        flux_filtered = flux_normalized

        break

hdul_data.close()

# =====================================================
# 2) LOAD SKYCALC TRANSMISSION
# =====================================================
file_path_sky = "skytable (2).fits"
hdul_sky = fits.open(file_path_sky)

for hdu in hdul_sky:
    if isinstance(hdu, fits.BinTableHDU) and all(col in hdu.data.names for col in ['lam', 'trans']):
        sky = hdu.data

        # SkyCalc lam is in nm → convert to Å
        wave_sky = sky['lam'] * 10.0
        trans_sky = sky['trans']

        mask_sky = (wave_sky >= 6000) & (wave_sky <= 6800)
        wave_sky_filt = wave_sky[mask_sky]
        trans_sky_filt = trans_sky[mask_sky]

        # normalization
        z = np.polyfit(wave_sky_filt, trans_sky_filt, 1)
        p = np.poly1d(z)
        cnt = p(wave_sky_filt)
        trans_sky_filt = trans_sky_filt / cnt

        break

hdul_sky.close()




# =====================================================
# 3) Res broad
# =====================================================

def res_broad(wave, flux, to_res, from_res=None, method='direct'):

    # ---------------------------------------
    # determine Gaussian sigma in pixel size
    # ---------------------------------------

    # determine distance between each two adjacent wavelengths - (len = Nd - 1)
    wave_bin = wave[1:] - wave[:-1]

    # define the edge of each bin as half the wavelength distance to the bin next to it
    edges = wave[:-1] + 0.5 * (wave_bin)

    # define the edges for the first and last measure which where out of the previous calculations
    first_edge = wave[0] - 0.5*wave_bin[0]
    last_edge = wave[-1] + 0.5*wave_bin[-1]

    # Build the final edges array by combining all edges
    edges = np.array([first_edge] + edges.tolist() + [last_edge])

    # Bin width - this is width of each pixel - (len = Nd + 1)
    pixel_width = edges[1:] - edges[:-1]


    # ----------------------------------------------------------
    # estimate broadening fwhm for each wavelength - (len = Nd)
    # ----------------------------------------------------------
    fwhm = np.sqrt((wave/to_res)**2 - (wave/from_res)**2)
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))

    # Convert from wavelength units to pixel
    fwhm_pixel = fwhm / pixel_width
    sigma_pixel = sigma / pixel_width

    # Round pixels width to upper value
    nbins = np.ceil(fwhm_pixel)

    # R_grid = (wave[1:-1] + wave[0:-2]) / (wave[1:-1] - wave[0:-2]) / 2.
    # fwhm = np.sqrt( (np.median(R_grid)/to_res)**2 - (np.median(R_grid)/from_res)**2)

    # ----------------------------------------------
    # convolve each bin window with Gaussian kernel
    # ----------------------------------------------
    if method == 'direct':

        # move window along wavelength axis [len(nbins) = len(wave)]
        nwaveobs = len(wave)
        convolved_flux = np.zeros(len(wave))
        for loopi in np.arange(len(nbins)):

            # extract each section
            current_nbins = 2 * nbins[loopi] # Each side if bin
            current_center = wave[loopi]  # as the center of bin
            current_sigma = sigma[loopi]     # kernel sigma in A

            # Find lower and uper index for the gaussian
            lower_pos = int(max(0, loopi - current_nbins))
            upper_pos = int(min(nwaveobs, loopi + current_nbins + 1))

            # Select only the flux values for the segment
            flux_segment = flux[lower_pos:upper_pos+1]
            waveobs_segment = wave[lower_pos:upper_pos+1]

            # Build the gaussian kernel corresponding to the instrumental spread function SF
            gaussian = np.exp(- ((waveobs_segment - current_center)**2) / (2*current_sigma**2)) /         np.sqrt(2*np.pi*current_sigma**2)
            # Whether to normalize the kernel to have a sum of one
            gaussian = gaussian / np.sum(gaussian)

            # convolve
            only_positive_fluxes = flux_segment > 0
            weighted_flux = flux_segment[only_positive_fluxes] * gaussian[only_positive_fluxes]
            current_convolved_flux = weighted_flux.sum()
            convolved_flux[loopi] = current_convolved_flux



    # ----------------------------------------------
    # convolve each bin window with Gaussian kernel
    # ----------------------------------------------
    if method == 'median':

        kernel = Gaussian1DKernel(stddev=np.median(sigma_pixel))
        convolved_flux = convolve(flux, kernel, normalize_kernel=True, boundary='extend')

        # convolved_flux = scipy.signal.fftconvolve(flux, kernel, 'same')
        # # remove NaN values returned by fftconvolve
        # idxnan = np.argwhere(np.isnan(convolved_flux))
        # if len(idxnan) > 0:
        #     remove_indices = []
        #     for loopnan in range(len(idxnan)): remove_indices.append(idxnan[loopnan][0])
        #     # remove from list
        #     waveobs = [i for j, i in enumerate(waveobs) if j not in remove_indices]
        #     convolved_flux = [i for j, i in enumerate(convolved_flux) if j not in remove_indices]


    return wave, convolved_flux

#Resolutions (tel from header and skycalc from skycalc)
res_tel = 45990
res_skycalc =320000
#broadening on skycalc data
wave_sky_broadened, trans_sky_broadened = res_broad(wave_sky_filt, trans_sky_filt, res_tel, res_skycalc )

# =====================================================
# 4) PLOT BOTH TOGETHER
# =====================================================

plt.figure(figsize=(12, 6))

# Telescope flux
plt.plot(wave_filtered, flux_filtered, label='Telescope Flux', alpha=0.7)

# SkyCalc broadened transmission
plt.plot(wave_sky_broadened, trans_sky_broadened,
         label='SkyCalc Transmission',
         alpha=0.7)

# Final settings
plt.xlabel("Wavelength (Å)")
plt.ylabel("Normalized Flux / Transmission")
plt.title("Telescope Flux vs SkyCalc Transmission (6000–6800 Å)")
plt.legend()
plt.grid(True)

plt.xlim(6000, 6800)    # exact same axis range
plt.ylim(0, 1.2)        # adjust so both fit (your telescope goes up to ~1.1)
plt.show()



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# define the same grid interpolation
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def same_grid(wave1, flux1, wave2, flux2):
    # find the min and max intervals
    minv = min(wave1)
    maxv = max(wave1)
    if min(wave2) > minv: minv = min(wave2)
    if max(wave2) < maxv: maxv = max(wave2)
    # make a grid
    grid = np.arange(minv+0.02, maxv-0.02, 0.02)
    f = interpolate.interp1d(np.array(wave1), np.array(flux1), kind='cubic')
    flux1 = f(grid)
    f = interpolate.interp1d(np.array(wave2), np.array(flux2), kind='cubic')
    flux2 = f(grid)
    return grid, flux1, flux2


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#cross correlation
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def cross_correlat(x1, y1, x2, y2, yshift=False):
    # -----------------------------------------------------------
    #  apply cross correlation on data
    #  the x1, y1 are reference spectrum and the x2, y2 would move
    #  to the same grid similar to the reference.
    #  Example:
    #         deltaX = cross_correlat(x1, y1, x2, y2)
    #         x2 = x2 + deltaX
    # -----------------------------------------------------------
    Lm = np.diff(x1)[0]
    grid, y2, y1 = same_grid(x2, y2, x1, y1)


    if yshift == False:
        assert len(y2) == len(y1)
        f1 = fft(y2)
        f2 = fft(np.flipud(y1))
        real_part = np.real(ifft(f1 * f2))
        cc = fftshift(real_part)
        assert len(cc) == len(y2)
        zero_index = int(len(y2) / 2) - 1
        lshift = zero_index - np.argmax(cc)
        delta_shift = (lshift*Lm)

    # if yshift == True:

    return delta_shift

# wavelength range of the telluric lines
mask_telluric = (wave_filtered >= 6298) & (wave_filtered <= 6300)
wave_tel_region = wave_filtered[mask_telluric]
flux_tel_region = flux_filtered[mask_telluric]

mask_sky = (wave_sky_broadened >= 6298) & (wave_sky_broadened <= 6300)
wave_sky_region = wave_sky_broadened[mask_sky]
flux_sky_region = trans_sky_broadened[mask_sky]

delta_shift = cross_correlat(wave_sky_region, flux_sky_region, wave_tel_region, flux_tel_region)
print("Wavelength shift in Å for 6298–6300 region:", delta_shift)
