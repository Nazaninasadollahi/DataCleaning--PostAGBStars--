from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# === FILE'S NAME ===
file_path = "telescope_shifted.fits"

# === OPEN FITS FILE ===
hdul = fits.open(file_path)

# === PRINT BASIC INFO ===
print("\n📁 FITS file information:")
print("=" * 70)
hdul.info()

# === PRINT NUMBER OF HDUs ===
num_hdus = len(hdul)
print(f"\n📄 This FITS file contains {num_hdus} HDUs (Header/Data Units).")

# === LOOP THROUGH ALL HDUs ===
for i, hdu in enumerate(hdul):
    print("\n" + "=" * 70)
    print(f"🔹 HDU {i} ({type(hdu).__name__})")
    print("=" * 70)

    # === HEADER ===
    header = hdu.header
    print(f"Contains {len(header)} header cards.\n")
    print(header)

    # Check reference-frame related keywords
    print("\n🔍 Reference frame keywords:")
    for key in ['SPECSYS', 'SSYSOBS', 'SSYSSRC', 'VELREF',
                'OBSGEO-L', 'OBSGEO-B', 'OBSGEO-H',
                'RADECSYS', 'WCSNAME', 'ESO TEL GEOLAT', 'ESO TEL GEOLON', 'ESO TEL GEOELEV', 'DATE-OBS',
                'RA', 'DEC', 'EQUINOX', 'APERTURE', 'HIERARCH ESO TEL FOCU SCALE', 'SPEC_RES']:
        value = header.get(key)
        if value is not None:
            try:
                comment = header.comments[key]
            except KeyError:
                comment = ""
            print(f"{key:<10} = {value:<15} ({comment})")
        else:
            print(f"{key:<10} = [Not found]")

    # === DATA ===
    print("\n📊 Data content:")
    if hdu.data is not None:
        data = hdu.data
        print(f"Data type: {type(data)}")
        print(f"Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

        # === UNITS ===
        print("\n🔧 Units:")
        if isinstance(hdu, fits.BinTableHDU):
            print("Data type: Table")
            print(f"Columns: {data.names}")
            print(f"Number of rows: {len(data)}")
            print("\nColumn units:")
            for j, colname in enumerate(data.names, 1):
                unit_key = f"TUNIT{j}"
                unit = header.get(unit_key, '[No unit specified]')
                print(f"Column {j}: {colname:<15} Unit: {unit}")
            print("\nrow")
            for row in data[:1]:
                print(row)

            # === VISUALIZATION (for BinTableHDU with WAVE and FLUX) ===
            if all(col in data.names for col in ['WAVE', 'FLUX']):
                wave = data['WAVE'][0]  # Single row, so [0]
                flux = data['FLUX'][0]

                # === UNIT CONVERSION: from erg/cm^2/s/Å to photons/s/m^2/µm/arcsec^2 ===
                h = 6.62607015e-34  # Planck constant [J s]
                c = 299792458.0  # speed of light [m/s]
                hc = h * c  # J·m

                aperture_deg = header.get('APERTURE', None)
                aperture_arcsec = aperture_deg * 3600
                area_arcsec2 = np.pi * (aperture_arcsec / 2) ** 2

                flux_photon_m2_s_um = (wave * (10 ** (-25)) * flux) / (area_arcsec2 * hc)

                # === Filter wavelength range: 6000–8000 Å ===
                mask = (wave >= 6000) & (wave <= 8000)
                wave_filtered = wave[mask]
                flux_filtered = flux_photon_m2_s_um[mask]

                # === PLOT FILTERED RANGE ===
                plt.figure(figsize=(10, 6))
                plt.plot(wave_filtered, flux_filtered, label='Converted Flux (ph/s/m²/µm/arcsec²)', color='red')
                plt.xlabel('Wavelength [Å]')
                plt.ylabel('Flux')
                plt.title(f'Spectrum from HDU {i} (6000–8000 Å)')
                plt.legend()
                plt.grid(True)
                plt.show()

# === CLOSE FITS FILE ===
hdul.close()
print("\n📂 FITS file closed.")
