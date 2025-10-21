from astropy.io import fits
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, GCRS
from astropy.time import Time
import numpy as np

# ================================
# Step 1: Load the FITS file
# ================================
fits_file = 'ADP.2020-06-26T11:14:15.747.fits'
with fits.open(fits_file) as hdul:
    data = hdul[1].data  # Binary table
    header = hdul[0].header.copy()
    table_header = hdul[1].header.copy()

# ================================
# Step 2: Observatory location
# ================================
obs_lon = header.get('ESO TEL GEOLON', -70.404) * u.deg
obs_lat = header.get('ESO TEL GEOLAT', -24.627) * u.deg
obs_alt = header.get('ESO TEL GEOELEV', 2635.0) * u.m
observatory = EarthLocation(lat=obs_lat, lon=obs_lon, height=obs_alt)

# Observation time
date_obs = header['DATE-OBS']
obs_time = Time(date_obs, format='isot', scale='utc')

# ================================
# Step 3: Target coordinates
# ================================
try:
    ra = data['RA'][0] * u.deg
    dec = data['DEC'][0] * u.deg
    from_table = True
except KeyError:
    ra = header['RA'] * u.deg
    dec = header['DEC'] * u.deg
    from_table = False

coords = SkyCoord(ra=ra, dec=dec, frame='fk5', equinox='J2000.0',
                  obstime=obs_time, location=observatory)
coords_geocentric = coords.transform_to(GCRS(obstime=obs_time))

ra_geo = coords_geocentric.ra.to(u.deg)
dec_geo = coords_geocentric.dec.to(u.deg)

print(f"Original RA,DEC = ({ra:.6f}, {dec:.6f})")
print(f"Geocentric RA,DEC = ({ra_geo:.6f}, {dec_geo:.6f})")

# ================================
# Step 4: Update RA/DEC in header or table
# ================================
new_data = data.copy()

if from_table and 'RA' in data.names and 'DEC' in data.names:
    new_data['RA'] = np.full_like(data['RA'], ra_geo.value)
    new_data['DEC'] = np.full_like(data['DEC'], dec_geo.value)
    print("✅ Updated RA/DEC columns in data table.")
else:
    header['RA'] = ra_geo.value
    header['DEC'] = dec_geo.value
    table_header['RA'] = ra_geo.value
    table_header['DEC'] = dec_geo.value
    print("✅ Updated RA/DEC in headers (no table columns found).")

# --- Remove RA/DEC comments from headers ---
for hdu_header in [header, table_header]:
    for key in ['RA', 'DEC']:
        if key in hdu_header:
            hdu_header.set(key, hdu_header[key], comment='deg')

# ================================
# Step 5: Apply TOPOCENT → GEOCENT wavelength correction
# ================================
specsys = header.get('SPECSYS', table_header.get('SPECSYS', ''))
if specsys.upper() == 'TOPOCENT':
    print("Transforming spectral frame: TOPOCENT → GEOCENT")

    # Observatory velocity in GCRS
    obs_gcrs = observatory.get_gcrs(obstime=obs_time)
    velocity = obs_gcrs.velocity.d_xyz  # m/s components

    # Line-of-sight unit vector
    target_vec = coords_geocentric.cartesian.xyz
    target_unit = target_vec / np.linalg.norm(target_vec)

    # Radial velocity correction (dot product)
    velocity_correction = np.sum(velocity * target_unit)
    print(f"Radial velocity correction = {velocity_correction.to(u.m/u.s):.3f}")

    # Apply to all wavelengths in WAVE column
    if 'WAVE' in data.names:
        wavelength = data['WAVE'] * u.Angstrom
        c = 299792458 * u.m / u.s
        wavelength_geocentric = wavelength * (1 - velocity_correction / c)

        delta_lambda = (wavelength_geocentric - wavelength).to(u.m)
        mean_shift = np.mean(delta_lambda)
        print(f"Mean wavelength shift = {mean_shift:.3e}")

        new_data['WAVE'] = wavelength_geocentric.to(u.Angstrom).value
    else:
        print("⚠️ No 'WAVE' column found — skipping wavelength correction.")

    # Update headers
    for hdu_header in [header, table_header]:
        hdu_header['SPECSYS'] = 'GEOCENT'
        hdu_header['RADECSYS'] = 'GCRS'
        hdu_header['COMMENT'] = 'Converted from TOPOCENT to GEOCENT (RA/DEC + spectrum).'
else:
    print("No TOPOCENT system detected — no wavelength correction applied.")

# ================================
# Step 6: Save output file
# ================================
output_file = 'output_geocentric_file.fits'
primary_hdu = fits.PrimaryHDU(header=header)
table_hdu = fits.BinTableHDU(data=new_data, header=table_header)
hdul_out = fits.HDUList([primary_hdu, table_hdu])
hdul_out.writeto(output_file, overwrite=True)

print(f"✅ Geocentric FITS file saved as {output_file}")
