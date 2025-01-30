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
# A when major changes, B when new exe release (to come),
#   C new python release (or small fix to exe)
VERSION = '3.1.6'

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
    'Xray': ['DCM', 'ROI', 'Num', 'Hom', 'Noi', 'MTF', 'NPS', 'STP', 'Var',
             'Foc', 'Def'],
    'Mammo': ['DCM', 'ROI', 'Num', 'SDN', 'Hom', 'Var', 'RLR', 'Gho', 'MTF',
              'NPS', 'CDM'],
    'NM': ['DCM', 'ROI', 'Num', 'Uni', 'SNI', 'MTF', 'Spe', 'Bar'],
    'SPECT': ['DCM', 'ROI', 'Num', 'MTF', 'Rin'],
    'PET': ['DCM', 'ROI', 'Num', 'Hom', 'Cro', 'Rec', 'MTF'],
    'MR': ['DCM', 'ROI', 'Num', 'SNR', 'PIU', 'Gho', 'Geo', 'Sli', 'MTF'],
    'SR': ['DCM']}

COLORS = ['r', 'b', 'g', 'y', 'c', 'm', 'skyblue', 'orange']

HALFLIFE = {'F18': 109.77}
ALTERNATIVES_ROI = ['One ROI',
                    'ROIs from table, same shape',
                    'ROIs from table, rectangle defined per ROI']
# dict:
#   display text for alternative methods - linked to HEADERS(_SUB)
#   if not 1-to-1 alternative text vs headers
#        - specify in settings_reusables like for NM-SNI
#   NB - to add verification output-settings vs paramset settings: add to
#       ui_main_test_tabs.py / param_changed_from_gui (verify_output)
#       config_func.py / get_test_alternative
ALTERNATIVES = {
    'CT': {
        'ROI': ALTERNATIVES_ROI,
        'Sli': ['Wire ramps Catphan (axial)',
                'Beaded ramps Catphan (helical)',
                'Vertical beaded ramps GE phantom',
                'Wire ramp Siemens',
                'Horizontal wire ramps GE QA phantom'],
        'MTF': ['bead', 'wire', 'circular edge',
                'z-resolution, wire', 'z-resolution, edge'],
        },
    'Xray': {
        'ROI': ALTERNATIVES_ROI,
        'Hom': ['Central + quadrants ROI, avg and stdev for each ROI',
                'Central + quadrants ROI, avg + difference from overall average',
                'Central + quadrants ROI, avg + % difference from overall average',
                'Flat field test from Mammo',
                'Flat field test AAPM TG150'],
        },
    'Mammo': {
       'ROI': ALTERNATIVES_ROI,
        },
    'NM': {
        'ROI': ALTERNATIVES_ROI,
        'SNI': [
            '6 small ROIs',
            'ROI grid, size by full ratio',
            'ROI grid, size by number of pixels',
            'ROIs matched Siemens gamma camera'],
        'MTF': ['Point', 'One line source', 'Two perpendicular line sources', 'Edge']
        },
    'SPECT': {
        'ROI': ALTERNATIVES_ROI,
        'MTF': ['Point source', 'Line source', 'Line source, sliding window',
                'z-resolution, line source(s)', 'z-resolution, edge']
        },
    'PET': {
        'ROI': ALTERNATIVES_ROI,
        'Rec': ['Recovery coefficients, average',
                'Recovery coefficients, max',
                'Recovery coefficients, peak',
                'Bq/ml from images, average',
                'Bq/ml from images, max',
                'Bq/ml from images, peak'],
        'MTF': ['Point source', 'Line source', 'Line source, sliding window',
                'z-resolution, line source(s)', 'z-resolution, edge']
        },
    'MR': {
        'ROI': ALTERNATIVES_ROI,
        'SNR': [
            'Noise from subtraction of two images (NEMA method 1)',
            'Noise from background ROIs per image (NEMA method 4)'],
        'Sli': ['Ramp', 'Wedge']
        },
    'SR': {}
    }

CALCULATION_OPTIONS = ['=', 'min', 'max', 'mean', 'stdev', 'max abs', 'width (max-min)']
#  options for QuickTestOutput settings - type of calculations

roi_headers = ['Average', 'Stdev']
roi_headers_sup = ['Min', 'Max']

# Headers of result table with optional alternatives.
# Use altAll if all alternatives use same headers
# Use alt0..N if each alterantive have their own headers
# use {} if headers depend on other dynamic parameters (like for Num, CTn)
# ROI also special dynamic case
# Where headers depend on dynamic parameters - change also
#   settings_reusables.py / QuickTestOutputSubDialog / update_data (on update_columns)
#   ui_main_test_tabs.py / param_changed_from_gui (verify_output)
HEADERS = {
    'CT': {
        'ROI': {'alt0': roi_headers},
        'Num': {},
        'Hom': {
            'altAll': ['HU at12', 'HU at15', 'HU at18', 'HU at21', 'HU center',
                       'diff at12', 'diff at15', 'diff at18', 'diff at21']
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
            'alt2': ['Nominal', 'V1', 'V2'],
            'alt3': ['Nominal', 'Measured'],
            'alt4': ['Nominal', 'H1', 'H2']
            },
        'MTF': {
            'alt0': ['MTFx 50%', 'MTFx 10%', 'MTFx 2%',
                     'MTFy 50%', 'MTFy 10%', 'MTFy 2%'],
            'alt1': ['MTFx 50%', 'MTFx 10%', 'MTFx 2%',
                     'MTFy 50%', 'MTFy 10%', 'MTFy 2%'],
            'alt2': ['MTF 50%', 'MTF 10%', 'MTF 2%'],
            'alt3': ['MTFz 50% line1', 'MTFz 10% line1', 'MTFz 2% line1',
                     'MTFz 50% line2', 'MTFz 10% line2', 'MTFz 2% line2'],
            'alt4': ['MTFz 50%', 'MTFz 10%', 'MTFz 2%'],
            },
        'TTF': {
            'alt0': ['Material', 'MTF 50%', 'MTF 10%', 'MTF 2%', 'Contrast'],
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
                     'ROI3 - avg %', 'ROI4 - avg %'],
            'alt3': ['Avg', 'Avg SNR', 'n ROIs',
                     'Deviating avgs', 'Deviating SNRs', 'Deviating ROIs',
                     '% dev ROIs', 'Deviating pixels', '% dev pixels'],
            'alt4': ['AvgPix', 'AvgNoi', 'AvgSNR', 'MinSNR', 'nAnomalousPix',
                     'Max nAnomPrROI',
                     'L Unif', 'G Unif', 'L NoiUnif', 'G NoiUnif', 'L SNRUnif',
                     'G SNRUnif', 'relSDrow', 'relSDcol']
            },
        'Noi': {'alt0': ['Avg pixel value', 'Noise=Stdev']},
        'MTF': {
            'alt0': ['MTF @ 0.5/mm', 'MTF @ 1.0/mm', 'MTF @ 1.5/mm',
                     'MTF @ 2.0/mm', 'MTF @ 2.5/mm', 'Freq @ MTF 0.5']
            },
        'NPS': {'alt0': ['Average variance', 'Large area signal',
                         'Large area stdev (noise)', 'AUC horiz/AUC vert']},
        'STP': {'alt0': ['Dose', 'Q', 'Mean pix', 'Stdev pix']},
        'Var': {'alt0': ['Max var 1', 'Median var 1', 'Max/median var 1',
                         'Max var 2', 'Median var 2', 'Max/median var 2']},
        'Foc': {'alt0': ['Star diameter (mm)', 'Magnification',
                         'Blur diameter x (mm)', 'Blur diameter y (mm)',
                         'FS x (mm)', 'FS y (mm)']},
        'Def': {'alt0': ['# equal to avg of 8 neighbours',
                         '# equal to avg of 4 neighbours']},
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
        'Var': {'alt0': ['Max var 1', 'Median var 1', 'Max/median var 1',
                         'Max var 2', 'Median var 2', 'Max/median var 2']},
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
        'CDM': {'alt0': ['Diameter', 'Threshold thickness auto',
                         'Threshold thickness predicted', 'Fit to predicted']}
         },
    'NM': {
        'ROI': {'alt0': roi_headers},
        'Num': {},
        'Uni': {'alt0': ['IU_UFOV %', 'DU_UFOV %', 'IU_CFOV %', 'DU_CFOV %']},
        'SNI': {  # NM differently than others - if changed be aware sni_alt parameter calculate_qc, ui_main_test_tabs and 'sni' in settings_reusables
            'alt0': ['SNI max', 'SNI L1', 'SNI L2', 'SNI S1', 'SNI S2',
                     'SNI S3', 'SNI S4', 'SNI S5', 'SNI S6'],
            'alt1': ['SNI L1 low', 'SNI L2 low', 'SNI S low max', 'SNI S low avg',
                     'SNI L1 high', 'SNI L2 high', 'SNI S high max', 'SNI S high avg'],
            'alt2': ['SNI L1', 'SNI L2', 'SNI S max', 'SNI S avg', 'SNI S median'],
            'alt3': ['SNI L1 low', 'SNI L2 low', 'SNI S low max', 'SNI S low avg',
                     'SNI L1 high', 'SNI L2 high', 'SNI S high max', 'SNI S high avg'],
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
            'alt1': ['FWHM x', 'FWTM x', 'FWHM y', 'FWTM y'],
            'alt2': ['FWHM x', 'FWTM x', 'FWHM y', 'FWTM y'],
            'alt3': ['FWHM z line1', 'FWTM z line1', 'FWHM z line2', 'FWTM z line2'],
            'alt4': ['FWHM z', 'FWTM z']
            },
        'Rin': {'alt0': ['Min diff from trend', 'Max diff from trend']},
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
            },
        'MTF': {
            'alt0': ['FWHM x', 'FWTM x', 'FWHM y', 'FWTM y'],
            'alt1': ['FWHM x', 'FWTM x', 'FWHM y', 'FWTM y'],
            'alt2': ['FWHM x', 'FWTM x', 'FWHM y', 'FWTM y'],
            'alt3': ['FWHM z line1', 'FWTM z line1', 'FWHM z line2', 'FWTM z line2'],
            'alt4': ['FWHM z', 'FWTM z']
            },
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
        'Sli': {'altAll': ['Nominal (mm)', 'Measured (mm)', 'Diff (mm)', 'Diff (%)']},
        'MTF': {'alt0': ['MTF 50%', 'MTF 10%', 'MTF 2%']}
        },
    'SR': {}
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
                     'A1_y', 'sigma1_y', 'A2_y', 'sigma2_y', 'sigma_prefilt'],
            'alt1': ['A1_x', 'sigma1_x', 'A2_x', 'sigma2_x',
                     'A1_y', 'sigma1_y', 'A2_y', 'sigma2_y', 'sigma_prefilt'],
            'alt2': ['A1', 'sigma1', 'A2', 'sigma2', 'sigma_prefilt'],
            'alt3': ['A1 line1', 'sigma1 line1', 'A2 line1', 'sigma2 line1',
                     'A1 line2', 'sigma1 line2', 'A2 line2', 'sigma2 line2',
                     'sigma_prefilt'],
            'alt4': ['A1', 'sigma1', 'A2', 'sigma2', 'sigma_prefilt'],
            },
        'Dim': {
            'alt0': ['Upper', 'Lower', 'Left', 'Right', 'Diagonal 1', 'Diagonal 2']
            },
        'NPS': {
            'alt0': ['Median frequency (1/mm)', 'Average AUC unnormalized',
                     'Average variance', 'ROIs avg HU', 'ROIs stdev HU (noise)']
            },
        },
    'Xray': {
        'ROI': {'alt0': roi_headers_sup},
        'Hom': {
            'alt0': [], 'alt1': [], 'alt2': [],
            'alt3': ['Min pixel', 'Max pixel', 'Min Avg', 'Max Avg',
                 'Min SNR', 'Max SNR', 'n ROIs x', 'n ROIs y',
                 'n masked ROIs', 'n masked pixels'],
            'alt4': []
            },
        'MTF': {'alt0': ['A1', 'sigma1', 'A2', 'sigma2', 'sigma_prefilt']},
        'Def': {'alt0': [
            'max def frac 8', 'n max def frac 8',
            'max def frac 4', 'n max def frac 4']},
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
       'MTF': {'alt0': ['A1', 'sigma1', 'A2', 'sigma2', 'sigma_prefilt']}
       },
    'NM': {
        'ROI': {'alt0': roi_headers_sup},
        'Uni': {
            'altAll': ['FitX (mm from center)', 'FitY (mm from center)',
                       'Fit distance (mm)',
                       'Scaled pixel size (mm)',
                       'Center pixel count (after scaling)']
            },
        'SNI': {
            'altAll': ['FitX (mm from center)', 'FitY (mm from center)',
                       'Fit distance (mm)', 'ROI max Large', 'ROI max Small']
            },
        'MTF': {
            'alt0': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y', 'sigma_prefilt'],
            'alt1': ['A1', 'sigma1', 'sigma_prefilt'],
            'alt2': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y', 'sigma_prefilt'],
            'alt3': ['A1', 'sigma1', 'sigma_prefilt'],
            },
        },
    'SPECT': {
        'ROI': {'alt0': roi_headers_sup},
        'MTF': {
            'alt0': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y', 'sigma_prefilt'],
            'alt1': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y', 'sigma_prefilt'],
            'alt2': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y', 'sigma_prefilt',
                     'x offset (mm)', 'y offset (mm)'],
            'alt3': ['A line1', 'sigma line1', 'A line2', 'sigma line2', 'sigma_prefilt'],
            'alt4': ['A', 'sigma', 'sigma_prefilt']
            },
        },
    'PET':  {
        'ROI': {'alt0': roi_headers_sup},
        'Rec': {
            'alt0': ['Scan start (HHMMSS)', 'Spheres at scan start (Bq/mL)',
                     'Background at scan start (Bq/mL)'],
            },
        'MTF': {
            'alt0': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y', 'sigma_prefilt'],
            'alt1': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y', 'sigma_prefilt'],
            'alt2': ['A1_x', 'sigma1_x', 'A1_y', 'sigma1_y', 'sigma_prefilt',
                     'x offset (mm)', 'y offset (mm)'],
            'alt3': ['A line1', 'sigma line1', 'A line2', 'sigma line2',
                     'sigma_prefilt'],
            'alt4': ['A', 'sigma', 'sigma_prefilt']
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
        'MTF': {'alt0': ['A1', 'sigma1', 'sigma_prefilt']}
        },
    'SR': {}
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
           'Philips MR ACR report (.pdf)'],
    'SR': []
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
    'report_templates': {
        'saved_as': 'modality_dict',
        'default': iQCconstants_functions.empty_template_dict(
            QUICKTEST_OPTIONS, dummy=cfc.ReportTemplate()),
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
