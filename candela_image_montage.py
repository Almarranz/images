#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:03:46 2024

@author: amartinez
"""

# import sep

# import os
# import numpy as np
# import sys
# import glob
# from subprocess import call
# import astropy.units as u
# from astropy.utils.data import download_file
# from astropy.io import fits  # We use fits to open the actual data file
import matplotlib.pyplot as plt
# from astropy import units as u
# from astropy.coordinates import SkyCoord
# from astropy.wcs import WCS
# from matplotlib.colors import LogNorm
# from matplotlib.tri import Triangulation
# import regions
# from matplotlib.patches import Rectangle
# from matplotlib.patches import Circle
# import IPython
# import tkinter as tk
# from tkinter import simpledialog
# from astropy.stats import sigma_clip
# from numpy import mean
# from time import time
# from matplotlib.widgets import Slider
# import astroalign as aa
# from scipy.spatial import distance
# import re
# import time
# import random
# from astropy.stats import sigma_clip
# from matplotlib.ticker import FormatStrFormatter
# from regions import Regions
# from astropy.utils.data import get_pkg_data_filename
# import pandas as pd
# from astromy_ds9 import ds9_norm
# import multicolorfits as mcf
# from matplotlib.patches import Rectangle
# from matplotlib.patches import FancyArrowPatch
# %%plotting pa    metres
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 10})
rcParams.update({'figure.figsize':(30,30)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})

# %%
# Enable automatic plotting mode
# IPython.get_ipython().run_line_magic('matplotlib', 'auto')
# IPython.get_ipython().run_line_magic('matplotlib', 'inline')










# %%
import numpy as np
import astropy.io.fits as pyfits
import multicolorfits as mcf
import matplotlib.pyplot as plt
from astropy.visualization import LogStretch, ImageNormalize
import sys
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
# mcf.mcf_gui()  #This command runs the interactive viewer
# sys.exit()

# Folder where your FITS images are located



Arches = SkyCoord(ra = 266.4604, dec = -28.8244, unit = 'degree').galactic
Quint = SkyCoord(ra =  266.5579, dec = -28.83, unit = 'degree').galactic
NSC = SkyCoord(ra =  266.418, dec = -29.0078, unit = 'degree').galactic

im_folder = '/Users/amartinez/Desktop/PhD/images/'
csx =3000
csy =1500
candela = Table.read('/Users/amartinez/Desktop/for_people/for_Rainer/kmos/candela.txt',
                     format='ascii',
                     names=('RA', 'DE', '2Jmag', '3Hmag', '4Ksmag', '5ra', '6Dec', '7x_c', '8y_c',
                            '9mua', '10dmua', '11mud', '12dmud', '13time', '14n1', '15n2', '16ID',
                            'mul', 'mub', '19dmul', '20dmub', '21m139'))
cor_gal = SkyCoord(ra=candela['RA'], dec=candela['DE'], unit='degree').galactic
mapa = WCS(fits.getheader(im_folder + f'crop{csx}-{csy}_gc4.5.fits', ext=0)).celestial
can_pix = mapa.wcs_world2pix(cor_gal.l,cor_gal.b,1)
can_pix_x = can_pix[0]
can_pix_y = can_pix[1]
arch_pix = mapa.wcs_world2pix(Arches.l,Arches.b,1)
quint_pix = mapa.wcs_world2pix(Quint.l,Quint.b,1)
nsc_pix = mapa.wcs_world2pix(NSC.l,NSC.b,1)

factor = 8
candela_mul = candela['mul']*factor  # Proper motion in x-axis
candela_mub = candela['mub']*factor 


# Load the data
# gc36_dat, gc36_hdr = pyfits.getdata(im_folder + f'crop{csx}_gc3.6.fits', header=True)
# gc45_dat, gc45_hdr = pyfits.getdata(im_folder +  f'crop{csx}_gc4.5.fits', header=True)
# gc80_dat, gc80_hdr = pyfits.getdata(im_folder +  f'crop{csx}_gc8.0.fits', header=True)

gc36_dat, gc36_hdr = pyfits.getdata(im_folder + f'crop{csx}-{csy}_gc3.6.fits', header=True)
gc45_dat, gc45_hdr = pyfits.getdata(im_folder +  f'crop{csx}-{csy}_gc4.5.fits', header=True)
gc80_dat, gc80_hdr = pyfits.getdata(im_folder +  f'crop{csx}-{csy}_gc8.0.fits', header=True)

# Convert the single images to RGB format (using mcf functions)
gc36_greyRGB = mcf.greyRGBize_image(gc36_dat, rescalefn='sqrt')
gc45_greyRGB = mcf.greyRGBize_image(gc45_dat, rescalefn='log')
gc80_greyRGB = mcf.greyRGBize_image(gc80_dat, rescalefn='log')
red_factor = 0.7
# Colorize the greyscale images
gc36_color = mcf.colorize_image(gc36_greyRGB, '#09b6ff', colorintype='hex')
gc45_color = mcf.colorize_image(gc45_greyRGB, '#05ff8d', colorintype='hex')
gc80_color = mcf.colorize_image(gc80_greyRGB*red_factor, '#ff2c05', colorintype='hex')

# Combine the colorized images into one master RGB image
combined_image = mcf.combine_multicolor([gc36_color, gc45_color, gc80_color], gamma=5, inverse=False)

# Define the normalization (using similar parameters as before)
norm = ImageNormalize(vmin=2.35045e-05, vmax=8545.85, stretch=LogStretch(), clip=True)
# norm = ImageNormalize(vmin=1781, vmax=25000, stretch=LogStretch(), clip=True)

# Define the output path
output_image_path = im_folder + 'rgb_gc_candela.jpg'

# Now, manually create a plot so we can control the ticks:
fig, ax = plt.subplots(figsize=(12,6),subplot_kw={'projection': mapa})
# fig, ax = plt.subplots(figsize=(12,7))
ax.imshow(combined_image, norm=norm, origin='lower', cmap='inferno')
wx = 100
wy = 80
# Define the region for the zoomed inset (in data coordinates)
# x1, y1 = min(can_pix[0])-30,min(can_pix[1])-30 
x1, y1 = min(can_pix[0])-30,min(can_pix[1])-30 
x2, y2 = x1 + wx, y1 +wy
ax_z = ax.inset_axes(
    [0.65, 0.65, 0.3, 0.3],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
ax_z.imshow(combined_image, norm=norm, origin='lower', cmap='inferno')

colores = ['#ff7f0e','#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f']
sizes = np.array([4000, 3500, 3000, 2500, 2000, 1500, 1000])/12
types = ['O4','O6','O7','B0','B0','B1','B1?']
for i in range(len(can_pix[0])):
    print(colores[i])
    # ax.scatter(can_pix[0][i],can_pix[1][i], marker = '*',color = colores[i], s = sizes[i], edgecolor = 'k',lw =2, label = '%s'%(types[i]),zorder =3)
    ax_z.scatter(can_pix[0][i],can_pix[1][i], marker = '.',color = colores[i], s = sizes[i],lw =0.3, label = '%s'%(types[i]),zorder =7-i,edgecolor = 'k', alpha = 0.6 +i*0.057)
# ax.scatter(can_pix[0],can_pix[1], marker = '*',color = colores, s = sizes, edgecolor = 'k',lw =2)
    
    # ax_z.quiver(can_pix_x, can_pix_y, candela_mul, candela_mub, 
    #           angles='xy', scale_units='xy', scale=1, color= ['#ff7f0e','#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f'], alpha = 0.3, zorder = 3,
    #           edgecolor = 'k', lw = 0.3, width = 0.02,)

    ax_z.quiver(can_pix_x[i], can_pix_y[i], candela_mul[i], candela_mub[i], 
              angles='xy', scale_units='xy', scale=1, color = colores[i],  alpha = 0.6 +i*0.057, zorder =7-i,
              edgecolor = 'k', lw = 0.3, width = 0.02,)



ind = ax.indicate_inset_zoom(ax_z, edgecolor="k",lw = 2)

ax.set_xlabel('')  # Remove label text if desired
ax.set_ylabel('') 

# Remove ticks and tick labels from both axes
# ax.tick_params(bottom=False, top=False, left=False, right=False,
#                 labelbottom=False, labeltop=False, labelleft=False, labelright=False)
ax.set_xlabel('GLON')
ax.set_ylabel('GLAT')
ax_z.tick_params(bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
ax_z.text(x1+10,y1+5,'Candela 1', fontsize= 15, color = 'white')
lh = ax_z.legend(fontsize = 7)
# lh.set_alpha = 1
for leg_o in lh.legend_handles:
    leg_o.set_alpha([1])
    


arc_loc = [500,1200]
qui_loc = [300,300]
nsc_loc = [2000,300]
# ax.text(arch_pix[0],arch_pix[1],'A')
ax.text(arc_loc[0],arc_loc[1],'Arches Cluster', color ='white',fontsize = 10)
ax.text(qui_loc[0],qui_loc[1],'Quintuplet Cluster', color ='white',fontsize = 10)
ax.text(nsc_loc[0],nsc_loc[1],'Nuclear Star Cluster', color ='white',fontsize = 10)
off_l = 20
ax.plot([arc_loc[0]+ off_l +150, arch_pix[0]-off_l], [arc_loc[1]-off_l, arch_pix[1]+off_l], color='white', linewidth=0.5, zorder=2)
ax.plot([qui_loc[0]+ off_l +200, quint_pix[0]-off_l-10], [qui_loc[1]+off_l+30, quint_pix[1]-off_l], color='white', linewidth=0.5, zorder=2)
ax.plot([nsc_loc[0]+ off_l +150, nsc_pix[0]+off_l+10], [nsc_loc[1]+off_l+20, nsc_pix[1]-off_l*1.5], color='white', linewidth=0.5, zorder=2)


off = 120
ax.text(1400 +off, 170,'250 light years', color = 'white', fontsize = 8)
# ax.plot([600,2400],[150,150],color='white', linewidth=0.5, zorder=2)

ax.annotate(
    "", 
    xy=(2400 + off, 150), xytext=(600+off, 150),
    arrowprops=dict(arrowstyle='<|-|>', color='white', linewidth=0.5,mutation_scale=10)
)


for spine in ax.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(2)

plt.show()
# # Save the figure with tight bounding box and no padding
plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.5, dpi = 1000, edgecolor = 'white')
plt.close(fig)

print(f"Multicolor FITS mosaic created and saved at: {output_image_path}")








