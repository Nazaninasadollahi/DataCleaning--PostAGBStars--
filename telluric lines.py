from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

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
        mask = (wave >= 6000) & (wave <= 8000)
        wave_filtered = wave[mask]
        flux_filtered = flux[mask]

        break

hdul_data.close()

# =====================================================
# 2) LOAD SKYCALC TRANSMISSION
# =====================================================
file_path_sky = "skytable (1).fits"
hdul_sky = fits.open(file_path_sky)

for hdu in hdul_sky:
    if isinstance(hdu, fits.BinTableHDU) and all(col in hdu.data.names for col in ['lam', 'trans']):
        sky = hdu.data

        # SkyCalc lam is in nm → convert to Å
        wave_sky = sky['lam'] * 10.0
        trans_sky = sky['trans']

        break

hdul_sky.close()

# =====================================================
# 3) INTERPOLATE SKYCALC TO YOUR WAVELENGTH GRID
# =====================================================
trans_interp = np.interp(wave_filtered, wave_sky, trans_sky)

# =====================================================
# 4) PLOT BOTH TOGETHER
# =====================================================
plt.figure(figsize=(12, 6))

# Telescope flux
plt.plot(wave_filtered, flux_filtered, label='Telescope Flux (converted)', alpha=0.7)

# Transmission on second axis
ax2 = plt.gca().twinx()
ax2.plot(wave_filtered, trans_interp, color="green", label='SkyCalc Transmission', alpha=0.8)

plt.title("Telescope Spectrum + SkyCalc Transmission (6000–8000 Å)")
plt.xlabel("Wavelength (Å)")
plt.grid(True)

plt.legend(loc='upper left')
ax2.set_ylabel("Transmission (0–1)")

plt.show()
