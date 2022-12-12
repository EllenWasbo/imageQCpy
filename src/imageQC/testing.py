# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 10:20:56 2022

@author: ellen
"""
'''
import pydicom
filename = r'C:\Users\ellen\CloudStation\ImageQCpy\DemoBilder\ConstancyImages\MR\Sandnes
\MR_SANDNESMR___AS_HEADTT_AS_HEADTT_28401_1.dcm'
ds = pydicom.dcmread(filename)
private_data = ds[0x2005, 0x1132]
pdf_data = private_data[0][0x2005, 0x1144]
print(type(pdf_data.value))
#class bytes
#somehow use convert to pixel_array in pydicom to convert this to image?
'''
