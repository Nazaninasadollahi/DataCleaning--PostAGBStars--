from astropy.io import fits
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, FK5, GCRS
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

# Step 4: Transform to geocentric frame (GCRS)
# Use FK5 frame with EQUINOX = 2000.0
coords = SkyCoord(ra=ra, dec=dec, frame='fk5', equinox='J2000.0', obstime=obs_time, location=observatory)
coords_geocentric = coords.transform_to(GCRS(obstime=obs_time))

# Extract transformed RA/Dec
ra_geocentric = coords_geocentric.ra
dec_geocentric = coords_geocentric.dec

# Step 5: Update the FITS data
new_data = data.copy()  # Copy to preserve table structure
if 'RA' in data.dtype.names and 'DEC' in data.dtype.names:
    new_data['RA'] = np.array(ra_geocentric.value, dtype=data['RA'].dtype)
    new_data['DEC'] = np.array(dec_geocentric.value, dtype=data['DEC'].dtype)
else:
    header['RA'] = ra_geocentric.value
    header['DEC'] = dec_geocentric.value
    print("No RA/DEC columns in data. Updated header instead.")

# Step 6: Update the FITS header
header['FRAME'] = 'GEOCENT'  # Indicate geocentric frame
header['ESO TEL GEOLAT'] = 0.0  # Set to 0 for geocentric frame
header['ESO TEL GEOLON'] = 0.0
header['ESO TEL GEOELEV'] = 0.0
header['RADECSYS'] = 'ICRS'  # GCRS aligns with ICRS
header['COMMENT'] = 'Coordinates transformed from FK5 (J2000.0) to geocentric (GCRS) frame'
header['HISTORY'] = f'Original DATE-OBS: {date_obs}'

# Step 7: Save the new FITS file
output_file = 'output_geocentric_file.fits'
try:
    primary_hdu = fits.PrimaryHDU(header=header)
    table_hdu = fits.BinTableHDU(data=new_data, header=table_header)
    new_hdul = fits.HDUList([primary_hdu, table_hdu])
    new_hdul.writeto(output_file, overwrite=True)
    print(f"Geocentric FITS file saved as {output_file}")
except Exception as e:
    raise Exception(f"Error writing FITS file: {e}")
