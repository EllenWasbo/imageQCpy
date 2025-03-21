# -*- coding: utf-8 -*-
"""
Collection of methods for phantom-specific calculations xray.

@author: ellen
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import (find_peaks, fftconvolve, medfilt2d)
from skimage.transform import hough_line, hough_line_peaks, resize
from skimage import draw

# imageQC block start
import imageQC.scripts.mini_methods_calculate as mmcalc
# imageQC block end


def find_tor18f(image):
    roi_array = None
    # summed or averaged x-profile and y-profile, cropped central or full phantom?
    # if cropped - only MTF, else both MTF and low contrast
    # roi_array[0] = framed MTF central part, rotation
    # roi_array[1] = list of center MTF parts
    # roi_array[2] = list of center low contrast parts
    return roi_array