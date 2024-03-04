#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constants accessible for several modules within imageQC.

@author: Ellen Wasbø
"""
import sys
import os

# imageQC block start
import imageQC.config.config_classes as cfc
from imageQC.config import iQCconstants_functions
# imageQC block end


USERNAME = os.getlogin()

# version string used to caluclate increasing number for comparison
# convention: A.B.C-bD where A,B,C,D is numbers < 100 and always increasing
VERSION = '3.0.8-b10'

if sys.platform.startswith("win"):
    APPDATA = os.path.join(os.environ['APPDATA'], 'imageQC')
    TEMPDIR = r'C:\Windows\Temp\imageQC'  # alternative to APPDATA if needed
else:  # assume Linux for now
    APPDATA = os.path.expanduser('~/.config/imageQC')
    TEMPDIR = r'/etc/opt/imageQC'

# os.environ variable keys to save global settings in session
ENV_USER_PREFS_PATH = 'IMAGEQC_USER_PREFS_PATH'
LOG_FILENAME = 'automation.log'
ENV_CONFIG_FOLDER = 'IMAGEQC_CONFIG_FOLDER'
ENV_ICON_PATH = 'IMAGEQC_ICON_PATH'

USER_PREFS_FNAME = 'user_preferences.yaml'

# dict: with lists defining modalities and their corresponding
#  list of tests with QuickTest as option."""
QUICKTEST_OPTIONS = {
    'CT': ['DCM', 'ROI', 'Num', 'Hom', 'Noi', 'Sli', 'MTF', 'TTF', 'CTn',
           'HUw', 'Rin', 'Dim', 'NPS'],
    'Xray': ['DCM', 'ROI', 'Num', 'Hom', 'Noi', 'MTF', 'NPS', 'STP', 'Var'],
    'Mammo': ['DCM', 'ROI', 'Num', 'SDN', 'Hom', 'RLR', 'Gho', 'MTF', 'NPS'],
    'NM': ['DCM', 'ROI', 'Num', 'Uni', 'SNI', 'MTF', 'Spe', 'Bar'],
    'SPECT': ['DCM', 'ROI', 'Num', 'MTF', 'Rin'],
    'PET': ['DCM', 'ROI', 'Num', 'Hom', 'Cro', 'Rec'],
    'MR': ['DCM', 'ROI', 'Num', 'SNR', 'PIU', 'Gho', 'Geo', 'Sli', 'MTF']}

COLORS = ['r', 'b', 'g', 'y', 'c', 'm', 'skyblue', 'orange']

HALFLIFE = {'F18': 109.77}
ALTERNATIVES_ROI = ['One ROI',
                    'ROIs from table, same shape',
                    'ROIs from table, rectangle defined per ROI']
# dict: with lists defining the alternative methods/table displays
#  if more than one option leading to different columns in table."""
ALTERNATIVES = {
    'CT': {
        'ROI': ALTERNATIVES_ROI,
        'Sli': ['Wire ramp Catphan',
                'Beaded ramp Catphan (helical)',
                'Vertical beaded ramps GE phantom'],
        'MTF': ['bead', 'wire', 'circular edge'],
        },
    'Xray': {
        'ROI': ALTERNATIVES_ROI,
        'Hom': ['Avg and stdev for each ROI',
                'Avg for each ROI + difference from avg of all',
                'Avg for each ROI + % difference from avg of all']
        },
    'Mammo': {
       'ROI': ALTERNATIVES_ROI,
        },
    'NM': {
        'ROI': ALTERNATIVES_ROI,
        'SNI': ['ROIs 2 large, 6 small', 'ROI grid, size by full ratio',
                'ROI grid, size by number of pixels',
                'ROIs matched Siemens gamma camera'],
        'MTF': ['Point', 'One line source', 'Two perpendicular line sources', 'Edge']
        },
    'SPECT': {
        'ROI': ALTERNATIVES_ROI,
        'MTF': ['Point source', 'Line source']
        },
    'PET': {
        'ROI': ALTERNATIVES_ROI,
        'Rec': ['Recovery coefficients, average',
                'Recovery coefficients, max',
                'Recovery coefficients, peak',
                'Bq/ml from images, average',
                'Bq/ml from images, max',
                'Bq/ml from images, peak'],
        },
    'MR': {
        'ROI': ALTERNATIVES_ROI,
        'SNR': [
            'Noise from subtraction of two images (NEMA method 1)',
            'Noise from background ROIs per image (NEMA method 4)'],
        'Sli': ['Ramp', 'Wedge']
        }
    }

CALCULATION_OPTIONS = ['=', 'min', 'max', 'mean', 'stdev', 'max abs', 'width (max-min)']
#  options for QuickTestOutput settings - type of calculations

roi_headers = ['Average', 'Stdev']
roi_headers_sup = ['Min', 'Max']

HEADERS = {
    'CT': {
        'ROI': {'alt0': roi_headers},
        'Num': {},
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
            'alt1': ['MTFx 50%', 'MTFx 10%', 'MTFx 2%',
                     'MTFy 50%', 'MTFy 10%', 'MTFy 2%'],
            'alt2': ['MTF 50%', 'MTF 10%', 'MTF 2%']
            },
        'TTF': {
            'alt0': ['Material', 'MTF 50%', 'MTF 10%', 'MTF 2%'],
            },
        'HUw': {'alt0': ['CT number (HU)', 'Noise=Stdev']},
        'Dim': {
            'alt0': ['Upper', 'Lower', 'Left', 'Right', 'Diagonal 1', 'Diagonal 2']
            },
        'Rin': {'alt0': ['Min diff from trend (HU)', 'Max diff from trend (HU)']},
        'NPS': {'alt0': ['Median frequency (1/mm)', 'Average AUC unnormalized',
                         'Average variance', 'ROIs avg HU', 'ROIs stdev HU (noise)']}
        },
    'Xray': {
        'ROI': {'alt0': roi_headers},
        'Num': {},
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
        'NPS': {'alt0': ['Average variance', 'Large area signal',
                         'Large area stdev (noise)', 'AUC horiz/AUC vert']},
        'STP': {'alt0': ['Dose', 'Q', 'Mean pix', 'Stdev pix']},
        'Var': {'alt0': ['Min variance', 'Max variance', 'Median variance']},
        },
    'Mammo': {
        'ROI': {'alt0': roi_headers},
        'Num': {},
        'SDN': {
            'alt0': ['Avg signal', 'Std signal',
                     'Avg background', 'Std background', 'SDNR']
            },
        'Hom': {
            'alt0': ['Avg', 'Avg SNR', 'n ROIs',
                     'Deviating avgs', 'Deviating SNRs', 'Deviating ROIs',
                     '% dev ROIs', 'Deviating pixels', '% dev pixels']
            },
        'RLR': {
            'alt0': ['Average', 'Stdev']
            },
        'Gho': {
            'alt0': ['ROI_1_avg', 'ROI_2_avg', 'ROI_3_avg', 'Ghost factor']
            },
        'MTF': {
            'alt0': ['MTF @ 1/mm', 'MTF @ 2/mm', 'MTF @ 3/mm',
                     'MTF @ 4/mm', 'MTF @ 5/mm', 'Freq @ MTF 0.5']
            },
        'NPS': {'alt0': ['Average variance', 'Large area signal',
                         'Large area stdev (noise)', 'AUC horiz/AUC vert']},
         },
    'NM': {
        'ROI': {'alt0': roi_headers},
        'Num': {},
        'Uni': {'alt0': ['IU_UFOV %', 'DU_UFOV %', 'IU_CFOV %', 'DU_CFOV %']},
        'SNI': {
            'alt0': ['SNI max', 'SNI L1', 'SNI L2', 'SNI S1', 'SNI S2',
                     'SNI S3', 'SNI S4', 'SNI S5', 'SNI S6'],
            'alt1': ['SNI max', 'SNI avg', 'SNI median'],
            'alt2': ['SNI max', 'SNI avg', 'SNI median'],
            'alt3': ['SNI max', 'SNI avg', 'SNI median'],
            },
        'Bar': {
            'alt0': ['MTF @ F1', 'MTF @ F2', 'MTF @ F3', 'MTF @ F4',
                     'FWHM1', 'FWHM2', 'FWHM3', 'FWHM4']
            },
        'Spe': {
            'alt0': ['Min diff from average (%)', 'Max diff from average (%)']
            },
        'MTF': {
            'alt0': ['FWHM x', 'FWTM x', 'FWHM y', 'FWTM y'],
            'alt1': ['FWHM', 'FWTM'],
            'alt2': ['FWHM x', 'FWTM x', 'FWHM y', 'FWTM y'],
            'alt3': ['FWHM', 'FWTM'],
            },
        },
    'SPECT': {
        'ROI': {'alt0': roi_headers},
        'Num': {},
        'MTF': {
            'alt0': ['FWHM x', 'FWTM x', 'FWHM y', 'FWTM y'],
            'alt1': ['FWHM x', 'FWTM x', 'FWHM y', 'FWTM y']
            },
        'Rin': {'alt0': ['Min diff from trend (HU)', 'Max diff from trend (HU)']},
        },
    'PET': {
        'ROI': {'alt0': roi_headers},
        'Num': {},
        'Hom': {
            'alt0': ['Center', 'at12', 'at15', 'at18', 'at21',
                     'dMean% C', 'dMean% at12', 'dMean% at15',
                     'dMean% at18', 'dMean% at21']
            },
        'Cro': {
            'alt0': [
                'Injected (MBq)', 'Inj. time', 'Scan time',
                'Activity at scan (MBq)', 'Bq/ml',
                'Bq/ml from images', 'SUV',
                'New calibration factor']
            },
        'Rec': {
            'alt0': [f'Avg {i+1}' for i in range(6)] + ['background'],
            'alt1': [f'Max {i+1}' for i in range(6)] + ['background'],
            'alt2': [f'Peak {i+1}' for i in range(6)] + ['background'],
            'alt3': [f'Avg {i+1}' for i in range(6)] + ['background'],
            'alt4': [f'Max {i+1}' for i in range(6)] + ['background'],
            'alt5': [f'Peak {i+1}' for i in range(6)] + ['background']
            }
        },
    'MR': {
        'ROI': {'alt0': roi_headers},
        'Num': {},
        'SNR': {
            'alt0': ['S img 1', 'S img 2', 'S mean', 'stdev diff', 'SNR'],
            'alt1': ['Central ROI mean', 'SD background ROIs',
                     'Estimated image noise', 'SNR'],
            },
        'PIU': {'alt0': ['min', 'max', 'PIU']},
        'Gho': {'alt0': ['Center', 'top', 'bottom', 'left', 'right', 'PSG']},
        'Geo': {
            'alt0': ['width_0', 'width_90', 'width_45', 'width_135',
                     'GD_0', 'GD_90', 'GD_45', 'GD_135']
            },
        'Sli': {'alt0': ['Nominal (mm)', 'Measured (mm)', 'Diff (mm)', 'Diff (%)']},
        'MTF': {'alt0': ['MTF 50%', 'MTF 10%', 'MTF 2%']}
        }
    }

# all modalities should be present with at least {}
# altAll if same supplement table for all
# else same alt0..n as for HEADERS ([] if none)
HEADERS_SUP = {
    'CT': {
        'ROI': {'alt0': roi_headers_sup},
        'Hom': {
            'altAll': ['Stdev at12', 'Stdev at15', 'Stdev at18', 'Stdev at21',
                       'Stdev Center']
            },
        'CTn': {'altAll': ['R-squared', 'fitted intercept', 'fitted slope']},
        'TTF': {
            'alt0': ['Material', 'A1', 'sigma1', 'A2', 'sigma2'],
            },
        'MTF': {
            'alt0': ['A1_x', 'sigma1_x', 'A2_x', 'sigma2_x',
                     'A1_y', 'sigma1_y', 'A2_y', 'sigma2_y'],
            'alt1': ['A1_x', 'sigma1_x', 'A2_x', 'sigma2_x',
                     'A1_y', 'sigma1_y', 'A2_y', 'sigma2_y'],
            'alt2': ['A1', 'sigma1', 'A2', 'sigma2']
            },
        'Dim': {
            'alt0': ['Upper', 'Lower', 'Left', 'Right', 'Diagonal 1', 'Diagonal 2']
            },
        },
    'Xray': {
        'ROI': {'alt0': roi_headers_sup},
        'MTF': {'alt0': ['A1', 'sigma1', 'A2', 'sigma2']}
        },
    'Mammo': {
       'ROI': {'alt0': roi_headers_sup},
       'RLR': {
           'alt0': ['Min', 'Max']
           },
       'Hom': {
           'alt0': ['Min pixel', 'Max pixel', 'Min Avg', 'Max Avg',
                    'Min SNR', 'Max SNR', 'n ROIs x', 'n ROIs y',
                    'n masked ROIs', 'n masked pixels']
           },
       'MTF': {'alt0': ['A1', 'sigma1', 'A2', 'sigma2']}
       },
    'NM': {
        'ROI': {'alt0': roi_headers_sup},
        'Uni': {
            'altAll': ['FitX (mm from center)', 'FitY (mm from center)',
                       'Fit distance (mm)']
            },
        'SNI': {
            'altAll': ['FitX (mm from center)', 'FitY (mm from center)',
                       'Fit distance (mm)']
            },
        'MTF': {
            'alt0': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y'],
            'alt1': ['A1', 'sigma1'],
            'alt2': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y'],
            'alt3': ['A1', 'sigma1'],
            },
        },
    'SPECT': {
        'ROI': {'alt0': roi_headers_sup},
        'MTF': {
            'alt0': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y'],
            'alt1': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y']
            },
        },
    'PET':  {
        'ROI': {'alt0': roi_headers_sup},
        'Rec': {
            'alt0': ['Scan start (HHMMSS)', 'Spheres at scan start (Bq/mL)',
                     'Background at scan start (Bq/mL)'],
            },
        },
    'MR': {
        'ROI': {'alt0': roi_headers_sup},
        'SNR': {'alt0': [], 'alt1': ['Number of background pixels']},
        'PIU': {
            'altAll': ['x min (pix from lower left)', 'y min',
                       'x max', 'y max']
            },
        'Sli': {'altAll': ['FWHM upper (mm)', 'FWHM lower (mm)']},
        'MTF': {'alt0': ['A1', 'sigma1']}
        }
    }

# should end with '(.suffix)' to get file_suffix
VENDOR_FILE_OPTIONS = {
    'CT': ['Siemens CT Constancy/Daily Reports (.pdf)'],
    'Xray': ['GE QAP (.txt)'],
    'Mammo': ['GE Mammo QAP (txt)'],
    'NM': ['Siemens exported energy spectrum (.txt)'],
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
        'saved_as': 'object_list',
        'default': []
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
    'digit_templates': {
        'saved_as': 'modality_dict',
        'default': iQCconstants_functions.empty_template_dict(
            QUICKTEST_OPTIONS, dummy=cfc.DigitTemplate()),
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
    'dash_settings': {
        'saved_as': 'object',
        'default': cfc.DashSettings(),
        },
    'limits_and_plot_templates': {
        'saved_as': 'modality_dict',
        'default': iQCconstants_functions.empty_template_dict(
            QUICKTEST_OPTIONS, dummy=cfc.LimitsAndPlotTemplate()),
        },
    'active_users': {
        'saved_as': 'dict',
        'default': {},
        },
    'last_modified': {
        'saved_as': 'object',
        'default': cfc.LastModified(),
        }
    }
