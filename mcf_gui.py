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
# IPython.get_ipython().run_line_magic('matplotlib', 'inline')










# %%
import numpy as np
import astropy.io.fits as pyfits
import multicolorfits as mcf
import matplotlib.pyplot as plt
from astropy.visualization import LogStretch, ImageNormalize
import sys

mcf.mcf_gui()  #This command runs the interactive viewer
sys.exit()

