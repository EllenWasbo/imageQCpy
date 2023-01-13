#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constants accessible for several modules within imageQC.

@author: Ellen Wasbø
"""

import os
# import configparser

# imageQC block start
import imageQC.config.config_classes as cfc
import imageQC.config.iQCconstants_functions as iQCconstants_functions
# imageQC block end


USERNAME = os.getlogin()

upper_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                __file__))))

# cfg_path = os.path.join(upper_path, 'setup.cfg')
VERSION = '3.0.4alpha'
'''if os.path.exists(cfg_path):
    config = configparser.ConfigParser()
    try:
        config.read(cfg_path)
        VERSION = config['metadata']['version']
    except:
        print("Failed reading version from setup.cfg")
        '''

APPDATA = os.path.join(os.environ['APPDATA'], 'imageQC')
TEMPDIR = r'C:\Windows\Temp\imageQC'  # alternative to APPDATA if needed

# os.environ variable keys to save global settings in session
ENV_USER_PREFS_PATH = 'IMAGEQC_USER_PREFS_PATH'
LOG_FILENAME = 'automation.log'
ENV_CONFIG_FOLDER = 'IMAGEQC_CONFIG_FOLDER'
ENV_ICON_PATH = 'IMAGEQC_ICON_PATH'

USER_PREFS_FNAME = 'user_preferences.yaml'

QUICKTEST_OPTIONS = {
    'CT': ['DCM', 'ROI', 'Hom', 'Noi', 'Sli', 'MTF', 'CTn',
           'HUw', 'Dim', 'Rin', 'NPS'],
    'Xray': ['DCM', 'ROI', 'Hom', 'Noi', 'MTF', 'NPS', 'STP', 'Var'],
    'NM': ['DCM', 'ROI', 'Uni', 'SNI', 'MTF', 'Spe', 'Bar'],
    'SPECT': ['DCM', 'ROI', 'MTF', 'Con'],
    'PET': ['DCM', 'ROI', 'Hom', 'Cro'],
    'MR': ['DCM', 'ROI', 'SNR', 'PIU', 'Gho', 'Geo', 'Sli']}
"""dict: with lists defining modalities and their corresponding
list of tests with QuickTest as option."""

ALTERNATIVES = {
    'CT': {
        'Sli': ['Wire ramp Catphan',
                'Beaded ramp Catphan (helical)',
                'Vertical beaded ramps GE phantom'],
        'MTF': ['bead', 'wire', 'circular edge'],
        },
    'Xray': {
        'Hom': ['Avg and stdev for each ROI',
                'Avg for each ROI + difference from avg of all',
                'Avg for each ROI + % difference from avg of all']
        }
    }
"""dict: with lists defining the alternative methods/table displays
 if more than one option leading to different columns in table."""

CALCULATION_OPTIONS = ['=', 'min', 'max', 'mean', 'stdev', 'max abs']
#  options for QuickTestOutput settings - type of calculations

roi_headers = ['Average', 'Stdev']

HEADERS = {
    'CT': {
        'ROI': {'alt0': roi_headers},
        'Hom': {
            'alt0': ['HU at12', 'HU at15', 'HU at18', 'HU at21', 'HU center',
                     'diff at12', 'diff at15', 'diff at18', 'diff at21'],
            'altSup': ['Stdev at12', 'Stdev at15', 'Stdev at18', 'Stdev at21',
                       'Stdev Center']
            },
        'Noi': {
            'alt0': ['CT number (HU)', 'Noise=Stdev (HU)',
                     'Diff avg noise(%)', 'Avg noise (HU)']
            },
        'CTn': {},
        'Sli': {
            'alt0': ['Nominal', 'H1', 'H2', 'V1', 'V2', 'Avg',
                     'Diff nominal (%)'],
            'alt1': ['Nominal', 'H1', 'H2', 'V1', 'V2',
                     'inner V1', 'inner V2'],
            'alt2': ['Nominal', 'V1', 'V2']
            },
        'MTF': {
            'alt0': ['MTFx 50%', 'MTFx 10%', 'MTFx 2%',
                     'MTFy 50%', 'MTFy 10%', 'MTFy 2%'],
            'alt1': ['MTF 50%', 'MTF 10%', 'MTF 2%'],
            'alt2': ['MTF 50%', 'MTF 10%', 'MTF 2%']
            },
        'HUw': {'alt0': ['CT number (HU)', 'Noise=Stdev']},
        'Dim': {
            'alt0': ['Upper', 'Lower', 'Left', 'Right', 'Diagonal 1', 'Diagonal 2']
            },
        'Rin': {'alt0': ['Min diff from trend (HU)', 'Max diff from trend (HU)']}
        },
    'Xray': {
        'ROI': {'alt0': roi_headers},
        'Hom': {
            'alt0': ['Center', 'ROI 1', 'ROI 2', 'ROI 3', 'ROI 4',
                     'Std Center', 'Std 1', 'Std 2', 'Std 3', 'Std 4'],
            'alt1': ['Center', 'ROI1 UL', 'ROI2 LL', 'ROI3 UR', 'ROI4 LR',
                     'C - avg', 'ROI1 - avg', 'ROI2 - avg', 'ROI3 - avg',
                     'ROI4 - avg'],
            'alt2': ['Center', 'ROI1 UL', 'ROI2 LL', 'ROI3 UR', 'ROI4 LR',
                     'C - avg %', 'ROI1 - avg %', 'ROI2 - avg %',
                     'ROI3 - avg %', 'ROI4 - avg %']
            },
        'Noi': {'alt0': ['Avg pixel value', 'Noise=Stdev']},
        'MTF': {
            'alt0': ['MTF @ 0.5/mm', 'MTF @ 1.0/mm', 'MTF @ 1.5/mm',
                     'MTF @ 2.0/mm', 'MTF @ 2.5/mm', 'Freq @ MTF 0.5']
            },
        'STP': {'alt0': ['Dose', 'Q', 'Mean pix', 'Stdev pix']}
        },
    'NM': {
        'ROI': {'alt0': roi_headers},
        'Uni': {'alt0': ['IU_UFOV %', 'DU_UFOV %', 'IU_CFOV %', 'DU_CFOV %']},
        'SNI': {
            'alt0': ['SNI max', 'SNI L1', 'SNI L2', 'SNI S1', 'SNI S2',
                     'SNI S3', 'SNI S4', 'SNI S5', 'SNI S6'],
            },
        'Bar': {
            'alt0': ['MTF @ F1', 'MTF @ F2', 'MTF @ F3', 'MTF @ F4',
                     'FWHM1', 'FWHM2', 'FWHM3', 'FWHM4']
            }
        },
    'SPECT': {
        'ROI': {'alt0': roi_headers},
        'MTF': {},
        'Con': {}
        },
    'PET': {
        'ROI': {'alt0': roi_headers},
        'Hom': {
            'alt0': ['Center', 'at12', 'at15', 'at18', 'at21',
                     'dMean% C', 'dMean% at12', 'dMean% at15',
                     'dMean% at18', 'dMean% at21']
            }
        },
    'MR': {
        'ROI': {'alt0': roi_headers},
        'SNR': {'alt0': ['S img 1', 'S img 2', 'S mean', 'stdev diff', 'SNR']},
        'PIU': {'alt0': ['min', 'max', 'PIU']},
        'Gho': {'alt0': ['Center', 'top', 'bottom', 'left', 'right', 'PSG']},
        'Geo': {
            'alt0': ['width_0', 'width_90', 'width_45', 'width_135',
                     'GD_0', 'GD_90', 'GD_45', 'GD_135']
            },
        'Sli': {'alt0': ['Nominal (mm)', 'Measured (mm)', 'Diff (mm)', 'Diff (%)']}
        }
    }

# all modalities should be present with at least {}
# altAll if same supplement table for all
# else same alt0..n as for HEADERS ([] if none)
HEADERS_SUP = {
    'CT': {
        'Hom': {
            'altAll': ['Stdev at12', 'Stdev at15', 'Stdev at18', 'Stdev at21',
                       'Stdev Center']
            },
        'CTn': {'altAll': ['R-squared', 'fitted intercept', 'fitted slope']},
        'MTF': {
            'alt0': ['A1_x', 'sigma1_x', 'A2_x', 'sigma2_x',
                     'A1_y', 'sigma1_y', 'A2_y', 'sigma2_y'],
            'alt1': ['A1', 'sigma1', 'A2', 'sigma2'],
            'alt2': ['A1', 'sigma1', 'A2', 'sigma2']
            },
        'Dim': {
            'alt0': ['Upper', 'Lower', 'Left', 'Right', 'Diagonal 1', 'Diagonal 2']
            },
        },
    'Xray': {},
    'NM': {
        'Uni': {
            'altAll': ['FitX (mm from center)', 'FitY (mm from center)',
                       'Fit distance (mm)']
            },
        'SNI': {
            'altAll': ['FitX (mm from center)', 'FitY (mm from center)',
                       'Fit distance (mm)']
            },
        },
    'SPECT': {},
    'PET':  {},
    'MR': {
        'PIU': {
            'altAll': ['x min (pix from lower left)', 'y min',
                       'x max', 'y max']
            },
        }
    }

VENDOR_FILE_OPTIONS = {
    'CT': ['Siemens CT Constancy/Daily Reports (.pdf)'],
    'Xray': ['GE QAP (.txt)'],
    'NM': ['Siemens exported energy spectrum'],
    'SPECT': [],
    'PET': ['Siemens PET-CT DailyQC Reports (.pdf)',
            'Siemens PET-MR DailyQC Reports (.xml)'],
    'MR': ['Philips MR PIQT / SPT report (.pdf)',
           'Philips MR ACR report (.pdf)']
    }
"""dict: with lists defining modalities and their corresponding
list of vendor file types to be read."""

tag_infos_default = iQCconstants_functions.read_tag_infos_from_yaml()
CONFIG_FNAMES = {
    'paramsets': {
        'saved_as': 'modality_dict',
        'default': iQCconstants_functions.empty_template_dict(
            QUICKTEST_OPTIONS, dummy=cfc.ParamSet())
        },
    'tag_infos': {
        'saved_as': 'object_list',
        'default': tag_infos_default
        },
    'tag_patterns_special': {
        'saved_as': 'modality_dict',
        'default': iQCconstants_functions.set_tag_patterns_special_default(
            QUICKTEST_OPTIONS, tag_infos_default)
        },
    'tag_patterns_format': {
        'saved_as': 'modality_dict',
        'default': iQCconstants_functions.empty_template_dict(
            QUICKTEST_OPTIONS, dummy=cfc.TagPatternFormat())
        },
    'tag_patterns_sort': {
        'saved_as': 'modality_dict',
        'default': iQCconstants_functions.empty_template_dict(
            QUICKTEST_OPTIONS, dummy=cfc.TagPatternSort())
        },
    'rename_patterns': {
        'saved_as': 'modality_dict',
        'default': iQCconstants_functions.empty_template_dict(
            QUICKTEST_OPTIONS, dummy=cfc.RenamePattern())
        },
    'quicktest_templates': {
        'saved_as': 'modality_dict',
        'default': iQCconstants_functions.empty_template_dict(
            QUICKTEST_OPTIONS, dummy=cfc.QuickTestTemplate())
        },
    'auto_common': {
        'saved_as': 'object',
        'default': iQCconstants_functions.set_auto_common_default()
        },
    'auto_templates': {
        'saved_as': 'modality_dict',
        'default': iQCconstants_functions.empty_template_dict(
            QUICKTEST_OPTIONS, dummy=cfc.AutoTemplate()),
        },
    'auto_vendor_templates': {
        'saved_as': 'modality_dict',
        'default': iQCconstants_functions.empty_template_dict(
            QUICKTEST_OPTIONS, dummy=cfc.AutoVendorTemplate()),
        },
    'active_users': {
        'saved_as': 'dict',
        'default': {}
        },
    'last_modified': {
        'saved_as': 'object',
        'default': cfc.LastModified()
        }
    }
