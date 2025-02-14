#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:03:46 2024

@author: amartinez
"""

import sep

import os
import numpy as np
import sys
import glob
from subprocess import call
import astropy.units as u
from astropy.utils.data import download_file
from astropy.io import fits  # We use fits to open the actual data file
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation
import regions
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import IPython
import tkinter as tk
from tkinter import simpledialog
from astropy.stats import sigma_clip
from numpy import mean
from time import time
from matplotlib.widgets import Slider
import astroalign as aa
from scipy.spatial import distance
import re
import time
import random
from astropy.stats import sigma_clip
from matplotlib.ticker import FormatStrFormatter
from regions import Regions
from astropy.utils.data import get_pkg_data_filename
import pandas as pd
from astromy_ds9 import ds9_norm

from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch
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
rcParams.update({'font.size': 20})
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
IPython.get_ipython().run_line_magic('matplotlib', 'inline')




im_folder = '/Users/amartinez/Desktop/PhD/images/'
# gc4.5.fits'

gns = fits.getdata(im_folder + 'ban_cluster_on_gns1.fits', ext=0)

# im_data = fits.getdata(gns, ext=1)
im_data = gns
im_data = np.squeeze(im_data) 

mapa = WCS(fits.getheader(im_folder + 'ban_cluster_on_gns1.fits', ext=0)).celestial


fig, ax = plt.subplots(1,1,subplot_kw={'projection': mapa}, figsize = (12,12))
# fig, (ax,ax1) = plt.subplots(1,2, figsize = (12,12))

norm = ds9_norm(vmin=-38.0813, vmax=797.506, bias=0.619388, contrast=6.40798, stretch='log')
# norm = ds9_norm(vmin=2.35045e-05, vmax=8545.85, bias=0.7131, contrast=1.77049, stretch='log')

ax.imshow(im_data, norm = norm, origin='lower', cmap='binary_r', label ='KMOS')
# ax1.imshow(im_data, norm = norm, origin='lower', cmap='Greys', label ='KMOS')
ax.set_xlim(1000,2800)
ax.set_ylim(500,)

lon1 = ax.coords[0]
lat1 = ax.coords[1]
lon1.set_ticks(spacing=20. * u.arcsec)
lat1.set_ticks(spacing=20. * u.arcsec)
ax.tick_params(axis = 'y',which = 'both',labelright = False, labelleft = True)


from astropy.table import Table
# %%
# Whole cluster
# lib_pruebas = '/Users/amartinez/Desktop/PhD/Libralato_data/pruebas/'

# # Ra_cl, Dec_cl, mura_cl,mudec_cl, H_cl, Ks_cl,ms_id= np.loadtxt(lib_pruebas + 'clus_14996_16_55.txt', unpack=True)
# clus= Table.read(lib_pruebas + 'clus_14996_16_55.txt', format = 'ascii',
#                       names = ('Ra',	'Dec',	'mura',	'mudec',	'H_cl',	'Ks_cl',	'ms_id',))


# clus_co = SkyCoord(ra = clus['Ra'], dec = clus['Dec'], unit = 'degree',pm_ra_cosdec = clus['mura']*u.mas/u.yr, pm_dec = clus['mudec']*u.mas/u.yr).galactic

# clus_pix = mapa.wcs_world2pix(clus['Ra'],clus['Dec'],1)
# factor = 30
# clus_mul = clus_co.pm_l_cosb.value*factor
# clus_mub = clus_co.pm_b.value*factor
# ax.scatter(clus_pix[0],clus_pix[1], marker = '*', s = 400, color = 'cyan',edgecolor='k',label = 'Co-moving group')
# # ax.scatter(clus_pix[0],clus_pix[1], marker = '*', s = 400, color = 'cyan',edgecolor='k')
# ax.quiver(clus_pix[0], clus_pix[1], clus_mul, clus_mub, 
#           angles='xy', scale_units='xy', scale=1, color='cyan', alpha=1, zorder = 1,
#           edgecolor = 'k', lw = 2)

# ax.legend(loc = 1,fontsize = 30)

# ax.set_xlabel('RA')
# ax.set_ylabel('DEC')
# %%
# CANDELA
candela = Table.read('/Users/amartinez/Desktop/for_people/for_Rainer/kmos/' + 'candela.txt',
            format = 'ascii',
            names = ('RA',	'DE',	'2Jmag',	'3Hmag',	'4Ksmag',	'5ra',	'6Dec',	'7x_c',	'8y_c',	'9mua ',	'10dmua',	'11mud',	'12dmud',	'13time',	'14n1',	'15n2',	'16ID',	'mul',	'mub',	'19dmul',	'20dmub',	'21m139',))

# %
can_pix = mapa.wcs_world2pix(candela['RA'],candela['DE'],1)
# np.savetxt('/Users/amartinez/Desktop/for_people/for_Rainer/kmos/' + 'candela.txt', gns_young_all[bri][:,0:22])
factor = 80
candela_mul = candela['mul']*factor  # Proper motion in x-axis
candela_mub = candela['mub']*factor  # Proper motion in y-axis
can_pix_x = can_pix[0]  # Starting x-coordinates of the vectors
can_pix_y = can_pix[1]  # Starting y-coordinates of the vectors
# 
# Create the quiver plot


colores = ['#ff7f0e','#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f']
sizes = np.array([4000, 3500, 3000, 2500, 2000, 1500, 1000])/2
types = ['O4','O6','O7','B0','B0','B1','B1?']

for i in range(len(can_pix[0])):
    print(colores[i])
    # ax.scatter(can_pix[0][i],can_pix[1][i], marker = '*',color = colores[i], s = sizes[i], edgecolor = 'k',lw =2, label = '%s'%(types[i]),zorder =3)
    ax.scatter(can_pix[0][i],can_pix[1][i], marker = '.',color = colores[i], s = sizes[i],lw =1, label = '%s'%(types[i]),zorder =3,edgecolor = 'k')
# ax.scatter(can_pix[0],can_pix[1], marker = '*',color = colores, s = sizes, edgecolor = 'k',lw =2)
    
    ax.quiver(can_pix_x, can_pix_y, candela_mul, candela_mub, 
              angles='xy', scale_units='xy', scale=1, color= ['#ff7f0e','#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f'], alpha=1, zorder = 1,
              edgecolor = 'k', lw = 1)

# ax.quiver(can_pix_x, can_pix_y, candela_mul, candela_mub, 
#           angles='xy', scale_units='xy', scale=1, color='lime', alpha=1, zorder = 1,
#           edgecolor = 'k', lw = 2)

ax.set_xlabel('RA')
ax.set_ylabel('DEC')
ax.legend(loc = 1, fontsize = 30)
# %%
# plt.savefig('/Users/amartinez/Desktop/for_people/for_Rainer/kmos/' + 'candela_b.png', 
#             bbox_inches = 'tight')

# coord = SkyCoord(ra = )

# %%

im_folder = '/Users/amartinez/Desktop/PhD/images/'
# gc4.5.fits'

gns = fits.getdata(im_folder + 'gc4.5.fits', ext=0)

# im_data = fits.getdata(gns, ext=1)
im_data = gns
im_data = np.squeeze(im_data) 


mapa = WCS(fits.getheader(im_folder + 'gc4.5.fits', ext=0)).celestial



fig, ax = plt.subplots(1,1,subplot_kw={'projection': mapa}, figsize = (12,12))
# fig, (ax,ax1) = plt.subplots(1,2, figsize = (12,12))

norm = ds9_norm(vmin=2.35045e-05, vmax=8545.85, bias=0.7131, contrast=1.77049, stretch='log')

ax.imshow(im_data, norm = norm, origin='lower', cmap='binary_r', label ='KMOS')

ax.set_xlim(1000,8000)
ax.set_ylim(1000,6000)



ax.set_xlabel('Galactic longitude')
ax.set_ylabel('Galactic latitude')

# CANDELA
candela = Table.read('/Users/amartinez/Desktop/for_people/for_Rainer/kmos/' + 'candela.txt',
            format = 'ascii',
            names = ('RA',	'DE',	'2Jmag',	'3Hmag',	'4Ksmag',	'5ra',	'6Dec',	'7x_c',	'8y_c',	'9mua ',	'10dmua',	'11mud',	'12dmud',	'13time',	'14n1',	'15n2',	'16ID',	'mul',	'mub',	'19dmul',	'20dmub',	'21m139',))

# %
cor_gal = SkyCoord(ra =candela['RA'], dec = candela['DE'], unit = 'degree' ).galactic
# can_pix = mapa.wcs_world2pix(candela['RA'],candela['DE'],1)
can_pix = mapa.wcs_world2pix(cor_gal.l,cor_gal.b,1)

# for i in range(len(can_pix[0])):
#     print(colores[i])
#     # ax.scatter(can_pix[0][i],can_pix[1][i], marker = '*',color = colores[i], s = sizes[i], edgecolor = 'k',lw =2, label = '%s'%(types[i]),zorder =3)
#     ax.scatter(can_pix[0][i],can_pix[1][i], marker = '.',color = colores[i], s = sizes[i],lw =1, label = '%s'%(types[i]),zorder =3,edgecolor = 'k')

# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Load image data
im_folder = '/Users/amartinez/Desktop/PhD/images/'
gns = fits.getdata(im_folder + 'gc4.5.fits', ext=0)
im_data = np.squeeze(gns)

# Load candela data
candela = Table.read('/Users/amartinez/Desktop/for_people/for_Rainer/kmos/candela.txt',
                     format='ascii',
                     names=('RA', 'DE', '2Jmag', '3Hmag', '4Ksmag', '5ra', '6Dec', '7x_c', '8y_c',
                            '9mua', '10dmua', '11mud', '12dmud', '13time', '14n1', '15n2', '16ID',
                            'mul', 'mub', '19dmul', '20dmub', '21m139'))

# Convert RA and Dec to Galactic coordinates
cor_gal = SkyCoord(ra=candela['RA'], dec=candela['DE'], unit='degree').galactic
can_pix = mapa.wcs_world2pix(cor_gal.l,cor_gal.b,1)
# Load WCS information
mapa = WCS(fits.getheader(im_folder + 'gc4.5.fits', ext=0)).celestial

# Create main plot
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': mapa}, figsize=(12, 12))

# Display the image
norm = ds9_norm(vmin=2.35045e-05, vmax=8545.85, bias=0.7131, contrast=1.77049, stretch='log')
ax.imshow(im_data, norm=norm, origin='lower', cmap='binary_r', label='KMOS')

# Set limits for the main plot
ax.set_xlim(1000, 8000)
ax.set_ylim(1000, 6000)

# Set labels
ax.set_xlabel('Galactic longitude')
ax.set_ylabel('Galactic latitude')

wx = 100
wy = 80
# Define the region for the zoomed inset (in data coordinates)
x1, y1 = min(can_pix[0])-30,min(can_pix[1])-30 
x2, y2 = x1 + wx, y1 +wy
# Create an inset axis
# xmin = 3600
# xmax = xmin +300
# ymin = 550
# ymax = ymin +300
# x1, x2, y1, y2 = xmin,xmax,ymin,ymax  # subregion of the original image
ax_z = ax.inset_axes(
    [0.6, 0.6, 0.5, 0.5],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
ax_z.imshow(im_data,  norm=norm, origin='lower', cmap='binary_r')
# ax_z.imshow(im_data,  norm=norm, origin='lower', cmap='inferno')

# ax_z.scatter(can_pix[0],can_pix[1])

factor = 8
candela_mul = candela['mul']*factor  # Proper motion in x-axis
candela_mub = candela['mub']*factor 
can_pix_x = can_pix[0]  # Starting x-coordinates of the vectors
can_pix_y = can_pix[1]  # Starting y-coordinates of the vectors


colores = ['#ff7f0e','#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f']
sizes = np.array([4000, 3500, 3000, 2500, 2000, 1500, 1000])/2
types = ['O4','O6','O7','B0','B0','B1','B1?']
for i in range(len(can_pix[0])):
    print(colores[i])
    # ax.scatter(can_pix[0][i],can_pix[1][i], marker = '*',color = colores[i], s = sizes[i], edgecolor = 'k',lw =2, label = '%s'%(types[i]),zorder =3)
    ax_z.scatter(can_pix[0][i],can_pix[1][i], marker = '.',color = colores[i], s = sizes[i],lw =1, label = '%s'%(types[i]),zorder =1,edgecolor = 'k')
# ax.scatter(can_pix[0],can_pix[1], marker = '*',color = colores, s = sizes, edgecolor = 'k',lw =2)
    
    ax_z.quiver(can_pix_x, can_pix_y, candela_mul, candela_mub, 
              angles='xy', scale_units='xy', scale=1, color= ['#ff7f0e','#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f'], alpha=1, zorder = 3,
              edgecolor = 'k', lw = 1)
ind = ax.indicate_inset_zoom(ax_z, edgecolor="lime",lw = 2)

for spine in ax_z.spines.values():
    spine.set_edgecolor('lime')
    spine.set_linewidth(2)  # Optional: set the width of the frame

ax_z.legend()





