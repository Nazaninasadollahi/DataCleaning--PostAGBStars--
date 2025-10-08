from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# === FILE'S NAME ===
file_path = "ADP.2020-06-26T11:14:15.747.fits"

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
                'RADECSYS', 'WCSNAME', 'ESO TEL GEOLAT', 'ESO TEL GEOLON', 'ESO TEL GEOELEV', 'DATE-OBS', 'RA', 'DEC']:
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

            # === VISUALIZATION (for BinTableHDU with WAVE and FLUX_REDUCED) ===
            if all(col in data.names for col in ['WAVE', 'FLUX_REDUCED']):
                wave = data['WAVE'][0]  # Single row, so [0]
                flux_reduced = data['FLUX_REDUCED'][0]

                plt.figure(figsize=(10, 6))
                plt.plot(wave, flux_reduced, label='Reduced Flux', color='blue')
                plt.xlabel(f'Wavelength ({header.get("TUNIT1", "Å")})')
                plt.ylabel(f'Flux ({header.get("TUNIT2", "ADU")})')
                plt.title(f'Spectrum from HDU {i}')
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                print("Required columns (WAVE, FLUX_REDUCED) not found in this HDU.")
        else:
            print("Data type: Not a table (skipping data processing).")
            unit = header.get('BUNIT', '[No unit specified]')
            print(f"Unit (if any): {unit}")
            print(f"Raw data: {data}")
    else:
        print("No data found in this HDU.")
        unit = header.get('BUNIT', '[No unit specified]')
        print(f"Unit (if any): {unit}")

# === CLOSE FITS FILE ===
hdul.close()
print("\n📂 FITS file closed.")