import numpy as np
import astropy.io.fits as pyfits
import os

x_crop = 3000
y_crop = 1500
def crop_fits_center(fits_path, output_path, x_size=x_crop, y_size = y_crop):
    # Load data and header
    data, header = pyfits.getdata(fits_path, header=True)
    
    # Get the original image shape
    ny, nx = data.shape  # Assumes 2D image
    
    # Define the cropping box
    x_center, y_center = nx // 2, ny // 2
    x_min, x_max = x_center - x_size // 2, x_center + (x_size+500) // 2
    y_min, y_max = y_center - y_size // 2, y_center + y_size // 2
    
    # Crop the data
    cropped_data = data[y_min:y_max, x_min:x_max]
    
    # Update WCS info in the header
    header['CRPIX1'] -= x_min
    header['CRPIX2'] -= y_min
    
    # Save the cropped FITS file
    pyfits.writeto(output_path, cropped_data, header, overwrite=True)
    print(f"Cropped image saved to {output_path}")

# Define file paths
im_folder = '/Users/amartinez/Desktop/PhD/images/'
fits_files = ['gc3.6.fits', 'gc4.5.fits', 'gc8.0.fits']

# Crop each file
for fits_file in fits_files:
    input_path = os.path.join(im_folder, fits_file)
    output_path = os.path.join(im_folder, f"crop{x_crop}-{y_crop}_{fits_file}")
    crop_fits_center(input_path, output_path)
