Data Cleaning

For the BD+33 2642 star, Red part (6000–7000 Å)

Step 1: Read the FITS files.

Step 2: Check the reference frame from Step 1 → If it is GEOCENT/TOPOCENT, it is ok.

Step 3: Use SKYCALC for simulation → Fill the first section on the SkyCalc website, including name, RA, and Dec (you can use the SIMBAD website to find other names). Click on “Transfer information to SkyCalc model.” Leave the other sections unchanged. In the Wavelength Grid section, set your wavelength range. Choose air for the simulation. Select Linear binning with a value of 0.002. You can see the simulation resolution here (λ/Δλ). Click Submit to get the output. Download the FITS file and use the Plotting SKYCALC results code. The "TRANS" column shows the telluric lines.

Step 4: Compare the telescope data with the telluric lines from SKYCALC to perform telluric calibration. Use the “telluric lines” code.

Step 5: Shift the telescope data for calibration and save the new FITS files.

**You can do step 4 and 5 using an interactive code (Red part.py)**
