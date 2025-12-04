from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from astropy.convolution import Gaussian1DKernel, convolve
from numpy.fft import fft, ifft, fftshift
from pathlib import Path

# -------------------------
# 1) LOAD TELESCOPE DATA
# -------------------------
file_path_data = "ADP.2020-06-26T11:14:16.300.fits"
with fits.open(file_path_data) as hdul_data:
    data = None
    for hdu in hdul_data:
        if isinstance(hdu, fits.BinTableHDU) and all(col in hdu.data.names for col in ['WAVE', 'FLUX']):
            data = hdu.data
            wave = np.array(data['WAVE'][0], dtype=float)
            flux = np.array(data['FLUX'][0], dtype=float)
            break
    if data is None:
        raise RuntimeError("Couldn't find a BinTableHDU with WAVE and FLUX in the telescope FITS file.")

mask = (wave >= 6000) & (wave <= 6800)
wave_filtered = wave[mask]
flux_filtered = flux[mask]


#Continuum normalization

z = np.polyfit(wave_filtered, flux_filtered, 3)  # 3rd-degree is much more stable
p = np.poly1d(z)
cnt = p(wave_filtered)
flux_normalized = flux_filtered / cnt

cont_mask = flux_normalized > 0.9
scale = np.median(flux_normalized[cont_mask])
flux_filtered = flux_normalized / scale

# -------------------------
# 2) LOAD SKYCALC TRANSMISSION
# -------------------------
file_path_sky = "skytable (2).fits"
with fits.open(file_path_sky) as hdul_sky:
    sky = None
    for hdu in hdul_sky:
        if isinstance(hdu, fits.BinTableHDU) and all(col in hdu.data.names for col in ['lam', 'trans']):
            sky = hdu.data
            wave_sky = np.array(sky['lam'], dtype=float) * 10.0
            trans_sky = np.array(sky['trans'], dtype=float)
            break
    if sky is None:
        raise RuntimeError("Couldn't find lam/trans in sky FITS file.")

mask_sky = (wave_sky >= 6000) & (wave_sky <= 6800)
wave_sky_filt = wave_sky[mask_sky]
trans_sky_filt = trans_sky[mask_sky]

# normalization
z = np.polyfit(wave_sky_filt, trans_sky_filt, 1)
p = np.poly1d(z)
cnt = p(wave_sky_filt)
trans_sky_filt = trans_sky_filt / cnt

# -------------------------
# 3) RESOLUTION BROADENING
# -------------------------
def res_broad(wave, flux, to_res, from_res=None, method='direct'):


    # determine Gaussian sigma in pixel size

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


    # estimate broadening fwhm for each wavelength - (len = Nd)

    fwhm = np.sqrt((wave/to_res)**2 - (wave/from_res)**2)
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))

    # Convert from wavelength units to pixel
    fwhm_pixel = fwhm / pixel_width
    sigma_pixel = sigma / pixel_width

    # Round pixels width to upper value
    nbins = np.ceil(fwhm_pixel)

    # R_grid = (wave[1:-1] + wave[0:-2]) / (wave[1:-1] - wave[0:-2]) / 2.
    # fwhm = np.sqrt( (np.median(R_grid)/to_res)**2 - (np.median(R_grid)/from_res)**2)


    # convolve each bin window with Gaussian kernel

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




    # convolve each bin window with Gaussian kernel

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

res_tel = 45990
res_skycalc = 320000
wave_sky_broadened, trans_sky_broadened = res_broad(
    wave_sky_filt, trans_sky_filt, res_tel, from_res=res_skycalc
)

# -------------------------
# 4) PLOT BEFORE ANY SHIFT (Tel + SkyCalc)
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(wave_filtered, flux_filtered,
         label="Telescope (original)", alpha=0.8)
plt.plot(wave_sky_broadened, trans_sky_broadened,
         label="SkyCalc (broadened)", alpha=0.8)

plt.xlabel("Wavelength (Å)")
plt.ylabel("Normalized Flux / Transmission")
plt.title("BEFORE SHIFT: Telescope vs SkyCalc (6000–6800 Å)")
plt.grid(True)
plt.legend()
plt.xlim(6000, 6800)
plt.ylim(0, 1.2)
plt.show()



# -------------------------
# 5) CROSS CORRELATION
# -------------------------
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


# =========================================================
# 6) INTERACTIVE SHIFT INSPECTION: 100 Å WINDOWS
# =========================================================
approved_shifts = []

for w_start in np.arange(6000, 6800, 100):
    w_end = w_start + 100

    print("\n" + "="*60)
    print(f"Processing window: {w_start:.0f} – {w_end:.0f} Å")

    # --- telescope window ---
    mask_tel = (wave_filtered >= w_start) & (wave_filtered <= w_end)
    wave_tel_region = wave_filtered[mask_tel]
    flux_tel_region = flux_filtered[mask_tel]

    # --- sky window ---
    mask_sky = (wave_sky_broadened >= w_start) & (wave_sky_broadened <= w_end)
    wave_sky_region = wave_sky_broadened[mask_sky]
    flux_sky_region = trans_sky_broadened[mask_sky]

    if len(wave_tel_region) < 10 or len(wave_sky_region) < 10:
        print("⚠ Skipping: not enough data in this window.")
        continue

    # --- automatic cross-correlation shift ---
    try:
        delta_shift = cross_correlat(
            wave_sky_region, flux_sky_region,
            wave_tel_region, flux_tel_region
        )
    except Exception as e:
        print("❌ Cross-correlation failed:", e)
        continue

    print(f"✅ Measured shift = {delta_shift:.6f} Å")

    # =====================================================
    # ✅ PLOT TELESCOPE + SKYCALC TOGETHER (BEFORE SHIFT)
    # =====================================================
    plt.figure(figsize=(9, 4))
    plt.plot(wave_tel_region, flux_tel_region,
             label="Telescope (original)", alpha=0.8)
    plt.plot(wave_sky_region, flux_sky_region,
             label="SkyCalc (broadened)", alpha=0.8)

    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Normalized Flux / Transmission")
    plt.title(f"BEFORE Shift: {w_start:.0f}–{w_end:.0f} Å")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- user decision ---
    user_choice = input("Type 0 to accept & go next | Type 1 to enter manual shift: ").strip()

    if user_choice == "1":
        try:
            delta_shift_used = float(input("Enter your manual delta shift (Å): "))
            print(f"✏ Using MANUAL shift = {delta_shift_used:.6f} Å")
        except:
            print("❌ Invalid input. Using automatic shift.")
            delta_shift_used = delta_shift
    else:
        print("✅ Using automatic shift.")
        delta_shift_used = delta_shift

    # --- apply selected shift to telescope only ---
    wave_tel_shifted = wave_tel_region + delta_shift_used

    # =====================================================
    # ✅ PLOT TELESCOPE (SHIFTED) + SKYCALC TOGETHER (AFTER)
    # =====================================================
    plt.figure(figsize=(9, 4))
    plt.plot(wave_tel_shifted, flux_tel_region,
             label="Telescope (shifted)", alpha=0.9)
    plt.plot(wave_sky_region, flux_sky_region,
             label="SkyCalc (broadened)", alpha=0.8)

    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Normalized Flux / Transmission")
    plt.title(
        f"AFTER Shift: {w_start:.0f}–{w_end:.0f} Å\n"
        f"Applied Shift = {delta_shift_used:.6f} Å"
    )
    plt.grid(True)
    plt.legend()
    plt.show()

    # ✅ store approved shift
    approved_shifts.append((w_start, w_end, delta_shift_used))

    input("➡ Press ENTER to continue to the next 100 Å window...")

# -------------------------
# 7) APPLY SHIFTS & SAVE FITS
# -------------------------
hdul_original = fits.open(file_path_data)
for idx, hdu in enumerate(hdul_original):
    if isinstance(hdu, fits.BinTableHDU) and all(col in hdu.data.names for col in ['WAVE', 'FLUX']):

        wave_new = np.array(wave, copy=True)
        flux_new = np.array(flux, copy=True)

        for w_start, w_end, dshift in approved_shifts:
            mask_full = (wave_new >= w_start) & (wave_new <= w_end)
            wave_new[mask_full] += dshift

        #Saving new fits file process
        colnames = [col.name for col in hdu.columns]
        wave_unit = hdu.columns['WAVE'].unit if 'WAVE' in colnames else None
        flux_unit = hdu.columns['FLUX'].unit if 'FLUX' in colnames else None

        col_wave = fits.Column(name='WAVE', array=[wave_new], format=f'{len(wave_new)}D', unit=wave_unit)
        col_flux = fits.Column(name='FLUX', array=[flux_new], format=f'{len(flux_new)}D', unit=flux_unit)

        new_table = fits.BinTableHDU.from_columns([col_wave, col_flux])
        new_table.header.extend(hdu.header, update=True)
        new_table.header['Tell_Cal'] = ("YES", "Telluric calibration applied")

        hdul_original[idx] = new_table
        break

output_file = f"{Path(file_path_data).stem}_shifted.fits"
hdul_original.writeto(output_file, overwrite=True)
hdul_original.close()

print("✅ SAVED:", output_file)


# -------------------------
# 8) Plot final results from new fits file
# -------------------------
file_path_shifted = "ADP.2020-06-26T11:14:16.300_shifted.fits"
with fits.open(file_path_shifted) as hdul_data:
    data = None
    for hdu in hdul_data:
        if isinstance(hdu, fits.BinTableHDU) and all(col in hdu.data.names for col in ['WAVE', 'FLUX']):
            data = hdu.data
            wave_shifted = np.array(data['WAVE'][0], dtype=float)
            flux_shifted = np.array(data['FLUX'][0], dtype=float)
            break
    if data is None:
        raise RuntimeError("Couldn't find a BinTableHDU with WAVE and FLUX in the shifted FITS file.")

mask = (wave_shifted >= 6000) & (wave_shifted <= 6800)
wave_shifted_filtered = wave_shifted[mask]
flux_shifted_filtered = flux_shifted[mask].astype(float)


z = np.polyfit(wave_shifted_filtered, flux_shifted_filtered, 3)
p = np.poly1d(z)
cnt = p(wave_shifted_filtered)
flux_shifted_normalized = flux_shifted_filtered / cnt


cont_mask = flux_shifted_normalized > 0.9
scale = np.median(flux_shifted_normalized[cont_mask])
flux_shifted_filtered = flux_shifted_normalized / scale

plt.figure(figsize=(10, 5))
plt.plot(wave_shifted_filtered, flux_shifted_filtered,
         label="Telescope (original)", alpha=0.8)
plt.plot(wave_sky_broadened, trans_sky_broadened,
         label="SkyCalc (broadened)", alpha=0.8)

plt.xlabel("Wavelength (Å)")
plt.ylabel("Normalized Flux / Transmission")
plt.title("After SHIFT: Telescope vs SkyCalc (6000–6800 Å)")
plt.grid(True)
plt.legend()
plt.xlim(6000, 6800)
plt.ylim(0, 1.2)
plt.show()
