from astropy.io import fits
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, FK5, GCRS, ITRS
from astropy.time import Time
import numpy as np

# Step 1: Load the FITS file
fits_file = 'ADP.2020-06-26T11:14:15.747.fits'
try:
    with fits.open(fits_file) as hdul:
        data = hdul[1].data  # Binary table in HDU 1
        header = hdul[0].header.copy()  # Copy primary header
        table_header = hdul[1].header.copy()  # Copy table header
except Exception as e:
    raise Exception(f"Error loading FITS file: {e}")

# Step 2: Extract observatory coordinates and observation time
try:
    obs_lon = header.get('ESO TEL GEOLON', 0.0) * u.deg  # Longitude in degrees
    obs_lat = header.get('ESO TEL GEOLAT', 0.0) * u.deg  # Latitude in degrees
    obs_alt = header.get('ESO TEL GEOELEV', 0.0) * u.m   # Altitude in meters
except KeyError:
    print("Observatory coordinates not found in header. Using defaults (replace with actual values).")
    obs_lon = -70.404 * u.deg  # Example: Paranal longitude
    obs_lat = -24.627 * u.deg  # Example: Paranal latitude
    obs_alt = 2635.0 * u.m     # Example: Paranal altitude

# Define the observatory location
observatory = EarthLocation(lat=obs_lat, lon=obs_lon, height=obs_alt)

# Extract observation time from DATE-OBS
try:
    date_obs = header['DATE-OBS']  # e.g., '2008-08-06T23:04:13.446'
    obs_time = Time(date_obs, format='isot', scale='utc')
except KeyError:
    raise KeyError("DATE-OBS not found in header. Please provide observation time.")

# Step 3: Extract astrometric data (RA/Dec)
try:
    ra = data['RA'] * u.deg  # Assume RA in degrees
    dec = data['DEC'] * u.deg
except KeyError:
    try:
        ra = header['RA'] * u.deg
        dec = header['DEC'] * u.deg
        print("RA/DEC not found in data table. Using header values.")
    except KeyError:
        raise KeyError("RA/DEC not found in data or header.")
print(ra, dec)

# Step 4: Transform to geocentric frame (GCRS)
coords = SkyCoord(ra=ra, dec=dec, frame='fk5', equinox='J2000.0', obstime=obs_time, location=observatory)
coords_geocentric = coords.transform_to(GCRS(obstime=obs_time))

# Extract transformed RA/Dec
ra_geocentric = coords_geocentric.ra
dec_geocentric = coords_geocentric.dec
print(ra_geocentric, dec_geocentric)

# Step 5: Update the FITS data
new_data = data.copy()  # Copy to preserve table structure
if 'RA' in data.dtype.names and 'DEC' in data.dtype.names:
    new_data['RA'] = np.array(ra_geocentric.value, dtype=data['RA'].dtype)
    new_data['DEC'] = np.array(dec_geocentric.value, dtype=data['DEC'].dtype)
else:
    # Update RA/DEC in both primary header (HDU 0) and table header (HDU 1)
    header['RA'] = ra_geocentric.value
    header['DEC'] = dec_geocentric.value
    table_header['RA'] = ra_geocentric.value
    table_header['DEC'] = dec_geocentric.value
    print("No RA/DEC columns in data. Updated headers for HDU 0 and HDU 1.")

# Step 6: Update comments in headers to keep only '(deg)'
for key in ['RA', 'DEC']:
    if key in header:
        header.set(key, header[key], '(deg)')
    if key in table_header:
        table_header.set(key, table_header[key], '(deg)')

# Step 7: Transform spectral reference frame to geocentric if TOPOCENT
if header.get('SPECSYS') == 'TOPOCENT' or table_header.get('SPECSYS') == 'TOPOCENT':
    print("Transforming spectral reference frame from topocentric to geocentric.")
    # Calculate observatory's velocity relative to Earth's center
    # Get observatory's position in ITRS frame
    obs_itrs = observatory.get_itrs(obstime=obs_time)
    # Compute velocity by taking time derivative (approximate)
    dt = 1 * u.s  # Small time step for velocity calculation
    obs_itrs_t2 = observatory.get_itrs(obstime=obs_time + dt)
    velocity = (obs_itrs_t2.cartesian - obs_itrs.cartesian) / dt
    # Project velocity along line of sight to target
    target_direction = coords_icrs = coords.transform_to('icrs').cartesian
    velocity_correction = velocity.dot(target_direction)  # Radial velocity component
    c = 299792458 * u.m/u.s  # Speed of light
    # Update spectral axis (e.g., WAVE) in both headers, using Angstrom units
    for hdu_header in [header, table_header]:
        if 'WAVE' in hdu_header:
            wavelength = hdu_header['WAVE'] * u.Angstrom  # Use Angstrom as specified
            wavelength_geocentric = wavelength * (1 - velocity_correction/c)
            hdu_header['WAVE'] = wavelength_geocentric.to(u.Angstrom).value
            hdu_header['SPECSYS'] = 'GEOCENT'
            hdu_header['COMMENT'] = 'Spectral axis transformed from topocentric to geocentric'

# Step 8: Update other header metadata (both HDU 0 and HDU 1)
header['SPECSYS'] = 'GEOCENT'
header['ESO TEL GEOLAT'] = 0.0
header['ESO TEL GEOLON'] = 0.0
header['ESO TEL GEOELEV'] = 0.0
header['RADECSYS'] = 'ICRS'
header['COMMENT'] = 'Coordinates transformed from FK5 (J2000.0) to geocentric (GCRS) frame'
header['HISTORY'] = f'Original DATE-OBS: {date_obs}'
header['HISTORY'] = f'Original RA: {ra.value}, DEC: {dec.value}'

table_header['SPECSYS'] = 'GEOCENT'
table_header['ESO TEL GEOLAT'] = 0.0
table_header['ESO TEL GEOLON'] = 0.0
table_header['ESO TEL GEOELEV'] = 0.0
table_header['RADECSYS'] = 'ICRS'
table_header['COMMENT'] = 'Coordinates transformed from FK5 (J2000.0) to geocentric (GCRS) frame'
table_header['HISTORY'] = f'Original DATE-OBS: {date_obs}'
table_header['HISTORY'] = f'Original RA: {ra.value}, DEC: {dec.value}'

# Step 9: Save the new FITS file
output_file = 'output_geocentric_file.fits'
try:
    primary_hdu = fits.PrimaryHDU(header=header)
    table_hdu = fits.BinTableHDU(data=new_data, header=table_header)
    new_hdul = fits.HDUList([primary_hdu, table_hdu])
    new_hdul.writeto(output_file, overwrite=True)
    print(f"Geocentric FITS file saved as {output_file}")
except Exception as e:
    raise Exception(f"Error writing FITS file: {e}")
