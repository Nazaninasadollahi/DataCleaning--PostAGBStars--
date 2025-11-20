from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
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
        z = np.polyfit(wave_filtered, flux_filtered, 1)
        p = np.poly1d(z)
        cnt = p(wave_filtered)
        flux_normalized = flux_filtered / cnt

        flux_normalized /= np.nanmax(flux_normalized)  # forces the peak to exactly 1.0
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
# 4) INTERPOLATE SKYCALC TO YOUR WAVELENGTH GRID
# =====================================================
from scipy.interpolate import interp1d
trans_interp = interp1d(wave_sky_broadened, trans_sky_broadened, kind='nearest',
                        bounds_error=False, fill_value='extrapolate')
trans_on_tel = trans_interp(wave_filtered)

coeffs = np.polyfit(trans_on_tel, flux_normalized, 1)
a, b = coeffs
trans_scaled = trans_on_tel * a + b



# =====================================================
# 5) PLOT BOTH TOGETHER
# =====================================================
plt.figure(figsize=(13, 6))

# spectrum
plt.plot(wave_filtered, flux_filtered, color='steelblue', lw=1.2,
         label='Telescope spectrum (normalized)')

# transmission
plt.plot(wave_filtered, trans_scaled, color='green', lw=1.8, alpha=0.9,
         label='SkyCalc atmospheric transmission (scaled)')

plt.title("Telescope Spectrum + SkyCalc Transmission (6000–6800 Å)\n")
plt.xlabel("Wavelength (Å)")
plt.ylabel("Normalized flux / Transmission")
plt.xlim(6000, 6800)
plt.ylim(0.3, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

