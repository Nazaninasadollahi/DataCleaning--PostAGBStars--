from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1) LOAD RED DATA
# -------------------------
file_path_red = "ADP.2020-06-26T11:14:16.300.fits"

with fits.open(file_path_red) as hdul:
    wave_red = None
    flux_red = None
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and {'WAVE', 'FLUX'} <= set(hdu.columns.names):
            wave_red = np.array(hdu.data['WAVE'][0], dtype=float)
            flux_red = np.array(hdu.data['FLUX'][0], dtype=float)
            print(f"Found RED spectrum in HDU {hdu.name} (ext {hdul.index_of(hdu)})")
            break
    if wave_red is None:
        raise RuntimeError("Could not find WAVE/FLUX columns in red file")

# -------------------------
# 2) LOAD BLUE DATA
# -------------------------
file_path_blue = "ADP.2020-06-26T11:14:16.096.fits"

with fits.open(file_path_blue) as hdul:
    wave_blue = None
    flux_blue = None
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and {'WAVE', 'FLUX'} <= set(hdu.columns.names):
            wave_blue = np.array(hdu.data['WAVE'][0], dtype=float)
            flux_blue = np.array(hdu.data['FLUX'][0], dtype=float)
            print(f"Found BLUE spectrum in HDU {hdu.name} (ext {hdul.index_of(hdu)})")
            break
    if wave_blue is None:
        raise RuntimeError("Could not find WAVE/FLUX columns in blue file")


# -------------------------
# 3) HELIOCENTRIC CORRECTION – SEPARATE FOR RED AND BLUE
# -------------------------
def get_obs_params(filepath):
    """Return all parameters needed for heliocentric correction from a given FITS file"""
    with fits.open(filepath) as hdul:
        lat = lon = elev = None
        date_obs = ra_deg = dec_deg = cor_frame = None
        for hdu in hdul:
            hdr = hdu.header
            if 'ESO TEL GEOLAT' in hdr and lat is None:
                lat = hdr['ESO TEL GEOLAT']
            if 'ESO TEL GEOLON' in hdr and lon is None:
                lon = hdr['ESO TEL GEOLON']
            if 'ESO TEL GEOELEV' in hdr and elev is None:
                elev = hdr['ESO TEL GEOELEV']
            if 'DATE-OBS' in hdr and date_obs is None:
                date_obs = hdr['DATE-OBS']
            if 'RA' in hdr and ra_deg is None:
                ra_deg = hdr['RA']
            if 'DEC' in hdr and dec_deg is None:
                dec_deg = hdr['DEC']
            if 'RADECSYS' in hdr and cor_frame is None:
                cor_frame = hdr['RADECSYS']
    return lat, lon, elev, date_obs, ra_deg, dec_deg, cor_frame

# --- Red ---
lat_r, lon_r, elev_r, date_r, ra_r, dec_r, frame_r = get_obs_params(file_path_red)
target_r   = SkyCoord(ra=ra_r*u.deg, dec=dec_r*u.deg, frame=frame_r.lower())
time_r     = Time(date_r, scale='utc')
location_r = EarthLocation(lat=lat_r*u.deg, lon=lon_r*u.deg, height=elev_r*u.m)
# the velocity of the observer (Earth) projected toward the star
v_heli_red = target_r.radial_velocity_correction('heliocentric', obstime=time_r, location=location_r).to(u.km/u.s).value

# --- Blue---
lat_b, lon_b, elev_b, date_b, ra_b, dec_b, frame_b = get_obs_params(file_path_blue)
target_b   = SkyCoord(ra=ra_b*u.deg, dec=dec_b*u.deg, frame=frame_b.lower())
time_b     = Time(date_b, scale='utc')
location_b = EarthLocation(lat=lat_b*u.deg, lon=lon_b*u.deg, height=elev_b*u.m)
v_heli_blue = target_b.radial_velocity_correction('heliocentric', obstime=time_b, location=location_b).to(u.km/u.s).value

# Apply the two (slightly different) corrections
c = 299792.458  # km/s
wave_red_helio  = wave_red  * (1.0 + v_heli_red  / c)
wave_blue_helio = wave_blue * (1.0 + v_heli_blue / c)

print(f"\nHeliocentric correction:")
print(f"  RED  : Δv_heli = {v_heli_red :+8.3f} km/s")
print(f"  BLUE: Δv_heli = {v_heli_blue:+8.3f} km/s")
print(f"  Difference   : {v_heli_red - v_heli_blue:+.3f} km/s")

# =====================================================
# 5) CONVERT WAVELENGTH → VELOCITY (around l0)
# =====================================================
l0 = 3302.369

# Classical non-relativistic Doppler formula
vel_red  = c * (wave_red_helio) / l0   # km/s
vel_blue = c * (wave_blue_helio) / l0   # km/s

print(f"\nVelocity arrays created relative to λ₀ = {l0} Å")
print(f"   Red : {vel_red[0]:+.1f} ... {vel_red[-1]:+.1f} km/s")
print(f"   Blue : {vel_blue[0]:+.1f} ... {vel_blue[-1]:+.1f} km/s")


# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(12,5))

# Plot RED
plt.plot(vel_red, flux_red, 'r-', lw=1, label='RED')
# Plot BLUE
plt.plot(vel_blue, flux_blue, 'b-', lw=1, alpha=0.7, label='BLUE')
plt.xlabel(f"Velocity relative to λ = {l0} Å  (km/s)")
plt.ylabel("Flux")
plt.title(f"Heliocentric-corrected spectrum – centered on λ₀ = {l0} Å")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()