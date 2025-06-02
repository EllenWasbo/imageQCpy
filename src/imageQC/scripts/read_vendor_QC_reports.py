#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read different QC reports from different vendors.

@author: EllenWasbo
"""
import re
from fnmatch import fnmatch
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pdfplumber

# imageQC block start
from imageQC.scripts.mini_methods_calculate import (
    get_width_center_at_threshold, get_curve_values)
# imageQC block end

try:  # only needed with pyinstaller, importerror if run from spyder
    from charset_normalizer import md__mypyc
except ImportError:
    pass

# TODO - fnmatch is case insensitive on Windows. Adding option for .lower for handeling
#   other os


def read_vendor_template(template, filepath):
    """Read vendor files based on AutoVendorTemplate information.

    Parameters
    ----------
    template : AutoVendorTemplate
        from config.config_classes
    filepath : str
        filepath to read
    """
    if template.file_type == 'Siemens CT Constancy/Daily Reports (.pdf)':
        txt = get_pdf_txt(filepath)
        results = read_Siemens_CT_QC(txt)
    elif template.file_type == 'Planmeca CBCT report (.html)':
        results = read_Planmeca_html(filepath)
    elif template.file_type == 'Siemens exported energy spectrum (.txt)':
        results = read_e_spectrum_Siemens_gamma_camera(filepath)
    elif template.file_type == 'Siemens PET-CT DailyQC Reports (.pdf)':
        txt = get_pdf_txt(filepath)
        results = read_Siemens_PET_dailyQC(txt)
    elif template.file_type == 'Siemens PET-MR DailyQC Reports (.xml)':
        root = get_xml_tree(filepath)
        results = read_Siemens_PET_dailyQC_xml(root)
    elif template.file_type == 'Philips MR ACR report (.pdf)':
        txt = get_pdf_txt(filepath)
        results = read_Philips_MR_ACR_report(txt)
    elif template.file_type == 'GE QAP (.txt)':
        results = read_GE_QAP(filepath)
    elif template.file_type == 'GE Mammo QAP (txt)':
        results = read_GE_Mammo_QAP(filepath)
    else:
        results = None

    return results


def read_pdf_dummy(txt):
    """Framework to read specific structured pdf files. Copy this and edit.

    Parameters
    ----------
    txt : list of str
        text from pdf

    Returns
    -------
    data : dict
        'status': bool
        'errmsg': list of str
        'values': list of str,
        'headers': list of str}
    """
    values = []
    headers = []
    errmsg = []
    status = False

    '''Explained:
        txt is a list where each element is one line of text from the pdf
        Start with some code to identify that the content is as expected.
        Fx for PET daily QC:
            if txt[0] == 'System Quality Report': ....
        To search for specific strings indicating start of text of interest:
            short_txt = [x[0:9] for x in txt] (to search start of lines only)
            search_txt = 'Scan Date' (replace Scan Date with your text)
            if search_txt in short_txt:
                rowno = short_txt.index(search_txt)
                split_txt = txt[rowno].split(' ')
                some_value = None
                some_value = split_txt[1]
                or
                try:
                    some_value = float(split_txt[1])
                except (ValueError, IndexError):
                    pass ( orr errmsg.append('...could not be found...'))

        ... more values

        values = [some_value, ...]
        headers = ['description of some value', ...]

        status = True
        '''

    '''
    To use this:
        config/iQCconstants.py
            Add option in VENDOR_FILE_OPTIONS
            option text should end with (.pdf)
        scripts/read_vendor_QC_reports.py
            in function open_vendor_files:
                add option text (same as above) to list implemented_types
        function read_vendor_template (first in the current file)
            add
            elif template.file_type == option text as above:
                txt = get_pdf_txt(filepath)
                results = read_xxxxthenewfunctionxxx(txt)
        test...
    '''

    data = {'status': status, 'errmsg': errmsg,
            'values': values, 'headers': headers}
    return data


def read_e_spectrum_Siemens_gamma_camera(filepath):
    """Read energy spectrum from .txt from Analyser Siemens gamma camera."""
    lines = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
    lines = lines[1:]  # skip first line
    curve_energy = []
    curve_counts = []
    status = False
    errmsg = None
    headers = None
    values = None
    details = None
    for line in lines:
        line = line.replace(',', '.')
        vals = line.split()
        if len(vals) not in [2, 3]:
            errmsg = 'File not in expected format: Tabular separated, 2 or 3 columns'
            break
        else:
            if len(vals) == 3:
                vals = vals[1:]
            curve_energy.append(float(vals[0]))
            curve_counts.append(float(vals[1]))

    if len(curve_counts) > 1:
        status = True
        headers = ['Max (counts)', 'keV at max', 'FWHM', 'Energy resolution (%)']
        max_counts = max(curve_counts)
        idx_max = curve_counts.index(max_counts)
        width, center = get_width_center_at_threshold(curve_counts, 0.5*max_counts)
        kev_fwhm_start_stop = get_curve_values(
                curve_energy, np.arange(len(curve_energy)),
                [center-width/2, center+width/2]
                )
        denergy = curve_energy[idx_max] - curve_energy[idx_max-1]
        fwhm = width * denergy
        kev_at_max = curve_energy[idx_max]
        values = [max_counts, kev_at_max, fwhm, 100.*fwhm/kev_at_max]
        details = {'curve_counts': np.array(curve_counts),
                   'curve_energy': np.array(curve_energy),
                   'keV_fwhm_start_stop': kev_fwhm_start_stop, 'keV_max': kev_at_max}

    data = {'status': status, 'errmsg': errmsg,
            'values': values, 'headers': headers,
            'details': details
            }
    return data


def read_Siemens_PET_dailyQC(txt):
    """Read Siemens PET dailyQC report from list of str.

    Parameters
    ----------
    txt : list of str
        text from pdf

    Returns
    -------
    data : dict
        'status': bool
        'errmsg': list of str
        'values': list of str,
        'headers': list of str}
    """
    values = []
    headers = []
    errmsg = []
    status = False

    if txt[0] == 'System Quality Report':
        short_txt = [x[0:9] for x in txt]
        # date
        date = ''
        search_txt = 'Scan Date'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            date_str = txt[rowno].split(',')
            date_md = date_str[0].split(' ')
            if len(date_md) >= 4:
                day = f'{int(date_md[3]):02}'
                month_dt = datetime.datetime.strptime(date_md[2], '%B')
                month = f'{month_dt.month:02}'
                year = date_str[1].strip()
                date = f'{day}.{month}.{year}'

        # scanner name
        ics_name = ''
        search_txt = 'ICS Name '
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            ics_name = split_txt[-1]

        # type of test
        partial = ''
        search_txt = 'Partial s'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            if split_txt[-1] == 'true':
                partial = 'X'
        full = ''
        search_txt = 'Full setu'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            if split_txt[-1] == 'true':
                full = 'X'
        timealign = ''
        search_txt = 'Time Alig'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            if split_txt[-1] == 'true':
                timealign = 'X'

        # calibration factor
        calib_factor = None
        search_txt = 'Calibrati'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            try:
                calib_factor = float(split_txt[-1])
            except ValueError:
                pass

        # Block Noise 3 [crystal] 0 [crystal] 0 Blocks
        n_blocks_noise = None
        search_txt = 'Block Noi'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno-1].split(' ')
            try:
                n_blocks_noise = int(split_txt[0])
            except ValueError:
                pass

        # Block Efficiency 120 [%] 80 [%] 0 Blocks
        n_blocks_efficiency = None
        search_txt = 'Block Eff'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno-1].split(' ')
            try:
                n_blocks_efficiency = int(split_txt[0])
            except ValueError:
                pass

        # Randoms 115 [%] 85 [%] 103.8 [%] Passed
        measured_randoms = None
        search_txt = 'Randoms'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno-1].split(' ')
            split_txt = [x for x in split_txt if x != '']
            try:
                measured_randoms = float(split_txt[-3])
            except ValueError:
                pass

        # Scanner Efficiency
        scanner_efficiency = None
        search_txt = 'Scanner E'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            if len(split_txt) == 3:  # version VG80B or similar
                split_txt = txt[rowno-1].split(' ')
                scanner_efficiency_txt = split_txt[-2]
            else:  # assume VG60A or similiar
                split_txt = [x for x in split_txt if x != '']
                scanner_efficiency_txt = split_txt[-3]
            try:
                scanner_efficiency = float(scanner_efficiency_txt)
            except ValueError:
                pass

        # Scatter Ratio 35.2 [%] 28.8 [%] 30.7 [%] Passed
        scatter_ratio = None
        search_txt = 'Scatter R'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            split_txt = [x for x in split_txt if x != '']
            try:
                scatter_ratio = float(split_txt[-3])
            except ValueError:
                pass

        # ECF
        ecf = None
        search_txt = 'Scanner e'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            split_txt = [x for x in split_txt if x != '']
            try:
                ecf = float(split_txt[-1])
            except ValueError:  # seen when split in 2, not 3 lines
                split_txt = txt[rowno + 1].split(' ')
                split_txt = [x for x in split_txt if x != '']
                try:
                    ecf = float(split_txt[-2])
                except ValueError:
                    pass

        # Image Plane Efficiency
        n_img_efficiency = None
        search_txt = 'Image Pla'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            split_txt = [x for x in split_txt if x != '']
            try:
                n_img_efficiency = int(split_txt[2])
            except ValueError:
                pass

        # Block Timing
        '''
        n_blocks_timing_offset = '-1'
        n_blocks_timing_width = '-1'
        search_txt = 'Block Ave'
        if search_txt in short_txt:  # VG80B or similar
            search_txt = 'Timing Re'
            if search_txt in short_txt:
                rowno = short_txt.index(search_txt)
                try:
                    split_txt = txt[rowno-1].split(' ')
                    split_txt = [x for x in split_txt if x != '']
                    n_blocks_timing_offset = split_txt[0]
                except IndexError:
                    pass
            search_txt = 'Timing Wi'
            if search_txt in short_txt:
                rowno = short_txt.index(search_txt)
                try:
                    split_txt = txt[rowno-2].split(' ')
                    split_txt = [x for x in split_txt if x != '']
                    n_blocks_timing_width = split_txt[2]
                except IndexError:
                    pass
        else:
            search_txt = 'Offset'
            if search_txt in short_txt:
                rowno = short_txt.index(search_txt)
                try:
                    split_txt = txt[rowno-2].split(' ')
                    split_txt = [x for x in split_txt if x != '']
                    n_blocks_timing_offset = split_txt[2]
                except IndexError:
                    pass
            search_txt = 'Width'
            if search_txt in short_txt:
                rowno = short_txt.index(search_txt)
                try:
                    split_txt = txt[rowno-2].split(' ')
                    split_txt = [x for x in split_txt if x != '']
                    n_blocks_timing_width = split_txt[2]
                except IndexError:
                    pass
        '''

        # Phantom position
        phantom_pos_x = None
        phantom_pos_y = None
        search_txt = 'Axis Valu'  # older versions
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno+1].split(' ')
            try:
                phantom_pos_x = float(split_txt[-1])
                split_txt = txt[rowno+2].split(' ')
                phantom_pos_y = float(split_txt[-1])
            except ValueError:
                pass
        else:
            search_txt = 'Phantom P'
            if search_txt in short_txt:
                indexes = [
                    index for index in range(len(short_txt))
                    if short_txt[index] == search_txt]
                if len(indexes) == 4:
                    split_txt = txt[indexes[1]+1].split(' ')
                    split_txt = [x for x in split_txt if x != '']
                    try:
                        phantom_pos_x = float(split_txt[4])
                        split_txt = txt[indexes[2]+1].split(' ')
                        split_txt = [x for x in split_txt if x != '']
                        phantom_pos_y = float(split_txt[4])
                    except ValueError:
                        pass

        values = [date, ics_name, partial, full, timealign,
                  calib_factor, measured_randoms,
                  scanner_efficiency, scatter_ratio,
                  ecf,
                  n_blocks_noise, n_blocks_efficiency,
                  n_img_efficiency,
                  phantom_pos_x, phantom_pos_y]
        headers = ['Date', 'ICSname', 'Partial', 'Full', 'TimeAlignment',
                   'Calibration Factor', 'Measured randoms %',
                   'Scanner Efficiency [cps/Bq/cc]', 'Scatter Ratio %',
                   'ECF [Bq*s/ECAT counts]',
                   'Blocks out of range noise',
                   'Blocks out of range efficiency',
                   'Image planes out of range efficiency',
                   'Phantom Pos x', 'Phantom Pos y'
                   ]

        status = True

    data = {'status': status, 'errmsg': errmsg,
            'values': values, 'headers': headers}
    return data


def read_Siemens_PET_dailyQC_xml(root):
    """Read Siemens PET dailyQC report from xml root.

    Parameters
    ----------
    root: root element
        from xml.etree.ElementTree.getroot()

    Returns
    -------
    data : dict
        'status': bool
        'errmsg': list of str
        'values': list of str,
        'headers': list of str}

    """
    values = []
    headers = []
    errmsg = []
    status = False

    if root.tag == 'SystemQualityReport':

        def_err_text = '?'
        def_err_msg = (
            'One or more parameters could not be found as expected in file.')
        details = root.find('fDetailedSystemQualityReport').find('aItem')
        if details is not None:
            try:
                date = root.find('bScandate').find('bddmmyy').text
            except AttributeError:
                date = def_err_text
                errmsg = def_err_msg
            try:
                calib_factor = float(root.find('cPhantomParameters').find(
                    'eCalibrationFactor').text)
            except AttributeError:
                calib_factor = def_err_text
                errmsg = def_err_msg
            try:
                measured_randoms = float(details.find('cMeasureRandoms').find(
                    'cBlkValue').find('aValue').text)
            except AttributeError:
                measured_randoms = def_err_text
                errmsg = def_err_msg
            try:
                scanner_efficiency = float(details.find('dScannerEfficiency').find(
                    'cBlkValue').find('aValue').text)
            except AttributeError:
                scanner_efficiency = def_err_text
                errmsg = def_err_msg
            try:
                scatter_ratio = float(details.find('eScatterRatio').find(
                    'cBlkValue').find('aValue').text)
            except AttributeError:
                scatter_ratio = def_err_text
                errmsg = def_err_msg
            try:
                ecf = float(details.find('fECF').find('cBlkValue').find(
                    'aValue').text)
            except AttributeError:
                ecf = def_err_text
                errmsg = def_err_msg
            try:
                n_blocks_noise = int(details.find('aBlockNoise').find(
                    'cBlkValue').find('aValue').text)
            except AttributeError:
                n_blocks_noise = def_err_text
                errmsg = def_err_msg
            try:
                n_blocks_efficiency = int(details.find('bBlockEfficiency').find(
                    'cBlkValue').find('aValue').text)
            except AttributeError:
                n_blocks_efficiency = def_err_text
                errmsg = def_err_msg
            try:
                n_img_efficiency = int(details.find('gPlaneEff').find(
                    'cBlkValue').find('aValue').text)
            except AttributeError:
                n_img_efficiency = def_err_text
                errmsg = def_err_msg

            values = [date, calib_factor, measured_randoms,
                      scanner_efficiency, scatter_ratio,
                      ecf,
                      n_blocks_noise, n_blocks_efficiency,
                      n_img_efficiency]
            headers = ['Date', 'Calibration Factor', 'Measured randoms %',
                       'Scanner Efficiency [cps/Bq/cc]', 'Scatter Ratio %',
                       'ECF [Bq*s/ECAT counts]',
                       'Blocks out of range noise',
                       'Blocks out of range efficiency',
                       'Image planes out of range efficiency'
                       ]

            status = True

        else:
            errmsg = (
                'Content of file not as expected for Siemens PET dailyQC xml.'
                'Missing tag fDetailedSystemQualityReport with aItem.')
    else:
        errmsg = ('Content  not as expected for Siemens PET dailyQC xml.'
                  'Missing root tag SystemQualityReport.')

    data = {'status': status, 'errmsg': errmsg,
            'values': values, 'headers': headers}
    return data


def get_Siemens_CT_QC_type_language(txt, file_type='standard'):
    """Get type (daily or constancy) and language of the report.

    Parameters
    ----------
    txt : list of str
        File content as list of strings.

    Returns
    -------
    dict
        language_dict: dict
            search_strings according to the found language in file
        report_type: str
            'daily' or 'constancy'
    """
    # TODO search strings in config folder as yaml file editable by gui or txt editor
    search_strings = {
        'English': {
            'daily': 'Quality Daily',
            'constancy': 'Quality*Constancy',
            'page1': 'Page: 1',
            'tester_name': 'Tester Name',
            'customer': 'Customer',
            'zipcode': 'Zip Code',
            'street': 'Street',
            'product_name': 'Product Name',
            'serial_number': 'Serial Number',
            'tube_assembly': 'Tube Asse',
            'description': 'Description',
            'value': 'Value',
            'test_result': 'Test Result',
            'slice': 'Slice',
            'slicewidth': 'Slice width',
            'homogeneity': 'Homogeneity',
            'noise': 'Noise',
            'tolerance': 'Tolerance',
            'n_images': 'Number of images',
            'test_typical_head': 'Test*Typical head',
            'test_typical_body': 'Test*Typical body',
            'test_sharpest_mode': 'Test*Sharpest mode',
            'results': 'Results',
            'result': 'Result'
            },
        'Norsk': {
            'daily': 'Kvalitet daglig',
            'constancy': 'Kvalitets*konstan?',
            'page1': 'Side: 1',
            'tester_name': 'Kontroll?rnavn',
            'customer': 'Kunde',
            'zipcode': 'Postnummer',
            'street': 'Gate',
            'product_name': 'Produktnavn',
            'serial_number': 'Serienummer',
            'tube_assembly': 'R?renhet',
            'description': 'Beskrivelse',
            'value': 'Verdi',
            'test_result': 'Test Resultat',
            'slice': 'Snitt',
            'slicewidth': 'Snittbredde',
            'homogeneity': 'Homogenitet',
            'noise': 'St?y',
            'tolerance': 'Toleranse',
            'n_images': 'Antall bilder',
            'test_typical_head': 'Test*Typisk hode',
            'test_typical_body': 'Test*Typisk kropp',
            'test_sharpest_mode': 'Test*Skarpeste modus',
            'results': 'Resultater',
            'result': 'Resultat'
            }
        }

    report_type = ''
    language = ''
    max_line = 20
    if len(txt) > max_line:
        txt = txt[0:max_line]

    for langu in search_strings:
        constancy_string = search_strings[langu]['constancy']
        for txt_line in txt:
            if fnmatch(txt_line, constancy_string):
                report_type = 'constancy'
                language = langu
                break
        if report_type == '':
            daily_string = search_strings[langu]['daily']
            for txt_line in txt:
                if fnmatch(txt_line, daily_string):
                    report_type = 'daily'
                    language = langu
                    break

        if report_type != '':
            break

    if report_type == '':
        if 'Intevo' in txt[1]:
            report_type = 'daily'
            language = 'English'

    if language == '':
        return_strings = {}
    else:
        return_strings = search_strings[language]

    return {
        'language_dict': return_strings,
        'report_type': report_type
        }


def read_Siemens_CT_QC_Intevo(txt, type_str, search_strings, info, errmsg):
    """Read Siemens Intevo CT daily or constancy report from pdf content.

    Parameters
    ----------
    txt : list of str
        pdf content
    type_str : str
        'daily'' or ''constancy'
    search_strings : dict
    info : list of str
        [date, tester_name, product_name, serial_number, serial_tube_A]
    errmsg : list of str

    Returns
    -------
    data : dict
        'status': bool
        'errmsg': list of str
        'values': list of str,
        'headers': list of str}
    """
    values = info
    status = True

    # slice thickness -------------------------------------------------------
    slice_head_min = ''
    slice_head_max = ''
    slice_body_min = ''
    slice_body_max = ''

    match_str = '*' + type_str + '*' + search_strings['slice'] + '*'

    rownos = [
        index for index in range(len(txt))
        if fnmatch(txt[index], match_str)]
    if len(rownos) > 0:  # constancy == 2, daily == 1
        row_test = rownos[-1]
    else:
        row_test = -1

    if row_test > -1:

        match_str = search_strings['n_images'] + '*'
        test_rows = [
            index for index in range(row_test, len(txt))
            if fnmatch(txt[index], match_str)]
        if len(test_rows) >= 2:
            for row in test_rows[0:2]:
                slice_res = []
                for i in range(row + 1, len(txt)):
                    if fnmatch(
                            txt[i], search_strings['slicewidth']+'*'):
                        break  # next test
                    else:
                        if fnmatch(txt[i], search_strings['slice']+'*'):
                            split_txt = txt[i].split()
                            if len(split_txt) > 2:
                                slice_res.append(float(split_txt[1]))

                if slice_head_min == '':
                    slice_head_min = min(slice_res)
                    slice_head_max = max(slice_res)
                else:
                    slice_body_min = min(slice_res)
                    slice_body_max = max(slice_res)

    # homogeneity / water HUwater -------------------------------------------
    '''6 tests
    extract test 3 (head full coll, 130 kV)
    + test 6 (body full coll, 130 kV)'''
    HUwater_head_min = ''
    HUwater_head_max = ''
    diff_head_max = ''
    HUwater_body_min = ''
    HUwater_body_max = ''
    diff_body_max = ''

    match_str = '*'+type_str+'*'+search_strings['homogeneity']+'*'

    rownos = [
        index for index in range(len(txt))
        if fnmatch(txt[index], match_str)]
    if len(rownos) > 0:  # constancy == 2, daily == 1
        row_test = rownos[-1]
    else:
        row_test = -1

    if row_test > -1:

        match_str = search_strings['n_images'] + '*'
        test_rows = [
            index for index in range(row_test, len(txt))
            if fnmatch(txt[index], match_str)]
        if len(test_rows) >= 6:
            actual_test_rows = [test_rows[i] for i in (2, 5)]
            # full coll 130 kV only
            for row in actual_test_rows:

                res_HU_val = []
                res_HU_diff = []

                read_HU = True
                read_diff = False

                for i in range(row + 1, len(txt)):
                    if i in test_rows:
                        break  # next test
                    elif fnmatch(txt[i], search_strings['slicewidth']+'*'):
                        break  # next test
                    else:
                        if len(res_HU_val) > 0:
                            if fnmatch(
                                    txt[i], '*'+search_strings['result']+'*'):
                                read_HU = False
                                read_diff = True

                        if fnmatch(txt[i], search_strings['slice'] + '*'):
                            split_txt = txt[i].split()
                            if read_HU:
                                if len(split_txt) >= 3:
                                    res_HU_val.append(float(split_txt[1]))
                            if read_diff:
                                if len(split_txt) == 11:  # daily
                                    res_HU_diff.extend(
                                        [split_txt[3], split_txt[5],
                                         split_txt[7], split_txt[9]])
                                elif len(split_txt) == 16:  # constancy
                                    res_HU_diff.extend(
                                        [split_txt[4], split_txt[7],
                                         split_txt[10], split_txt[13]])

                res_HU_diff = [float(val) for val in res_HU_diff]

                if HUwater_head_min == '':
                    HUwater_head_min = min(res_HU_val)
                    HUwater_head_max = max(res_HU_val)
                    if len(res_HU_diff) > 0:
                        diff_head_max = max(map(abs, res_HU_diff))
                else:
                    HUwater_body_min = min(res_HU_val)
                    HUwater_body_max = max(res_HU_val)
                    if len(res_HU_diff) > 0:
                        diff_body_max = max(map(abs, res_HU_diff))

    # noise -------------------------------------------------------
    # 7 tests: extract test 6 (body full coll, 130 kV) + test 7 (head full coll, 130 kV)
    noise_head_max = ''
    noise_body_max = ''

    match_str = '*'+type_str+'*'+search_strings['noise']+'*'

    rownos = [
        index for index in range(len(txt))
        if fnmatch(txt[index], match_str)]
    if len(rownos) > 0:  # constancy == 2, daily == 1
        row_test = rownos[-1]
    else:
        row_test = -1

    if row_test > -1:
        match_str = search_strings['n_images'] + '*'
        test_rows = [
            index for index in range(row_test, len(txt))
            if fnmatch(txt[index], match_str)]
        if len(test_rows) >= 7:
            actual_test_rows = [test_rows[i] for i in (6, 5)]
            # full coll 130 kV only
            for row in actual_test_rows:
                noise_res = []
                for i in range(row + 1, len(txt)):
                    if i in test_rows:
                        break  # next test
                    elif fnmatch(txt[i], search_strings['slicewidth'] + '*'):
                        break  # next test
                    else:
                        if fnmatch(txt[i], search_strings['slice'] + '*'):
                            split_txt = txt[i].split()
                            if len(split_txt) >= 3:
                                noise_res.append(float(split_txt[1]))

                if len(noise_res) == 0:
                    # Quality_Constancy_Noise files - no result per slice
                    for i in range(row + 1, len(txt)):
                        if i in test_rows:
                            break  # next test
                        elif fnmatch(txt[i], search_strings['slicewidth'] + '*'):
                            break  # next test
                        else:
                            split_txt = txt[i].split()
                            if len(split_txt) == 5:
                                if split_txt[-1] == 'kV':
                                    noise_res.append(float(split_txt[1]))
                                    break
                try:
                    if noise_head_max == '':
                        noise_head_max = max(noise_res)
                    else:
                        noise_body_max = max(noise_res)
                except ValueError:
                    pass

    # MTF -------------------------------------------------------
    # 5 tests: extract test 1-3 (130 kV body B41, head H31, U90)'''
    MTF50_body_smooth = ''
    MTF10_body_smooth = ''
    MTF50_head_smooth = ''
    MTF10_head_smooth = ''
    MTF50_UHR = ''
    MTF10_UHR = ''

    match_str = '*' + type_str + '*MTF*'
    rownos = [
        index for index in range(len(txt))
        if fnmatch(txt[index], match_str)]
    if len(rownos) > 0:  # constancy == 2, daily == 1
        row_test = rownos[-1]
    else:
        row_test = -1

    if row_test > -1:

        match_str = search_strings['n_images'] + '*'
        test_rows = [
            index for index in range(row_test, len(txt))
            if fnmatch(txt[index], match_str)]
        if len(test_rows) >= 5:
            actual_test_rows = [test_rows[i] for i in (0, 1, 2)]  # test 1-3
            for row in actual_test_rows:

                mtf50_res = []
                mtf10_res = []

                for i in range(row + 1, len(txt)):
                    if i in test_rows:
                        break  # next test
                    elif fnmatch(txt[i], search_strings['slicewidth'] + '*'):
                        break  # next test or stop
                    else:
                        if fnmatch(txt[i],
                                   search_strings['slice'] + '*'):
                            split_txt = txt[i].split()
                            if len(split_txt) == 7:
                                mtf50_res.append(float(split_txt[1]))
                                mtf10_res.append(float(split_txt[3]))
                            elif len(split_txt) == 10:
                                mtf50_res.append(float(split_txt[1]))
                                mtf10_res.append(float(split_txt[4]))

                try:
                    if row == actual_test_rows[0]:
                        MTF50_body_smooth = sum(mtf50_res)/len(mtf50_res)
                        MTF10_body_smooth = sum(mtf10_res)/len(mtf10_res)
                    elif row == actual_test_rows[1]:
                        MTF50_head_smooth = sum(mtf50_res)/len(mtf50_res)
                        MTF10_head_smooth = sum(mtf10_res)/len(mtf10_res)
                    else:
                        MTF50_UHR = sum(mtf50_res)/len(mtf50_res)
                        MTF10_UHR = sum(mtf10_res)/len(mtf10_res)
                except ZeroDivisionError:
                    pass

    headers = ['Date', 'Tester name', 'Product Name', 'Serial Number',
               'Tube ID',
               'HUwater head min', 'HUwater head max',
               'HUwater body min', 'HUwater body max',
               'Diff head max(abs)', 'Diff body max(abs)',
               'Noise head max', 'Noise body max',
               'Slice head min', 'Slice head max',
               'Slice body min', 'Slice body max',
               'MTF50 B smooth', 'MTF10 B smooth',
               'MTF50 H smooth', 'MTF10 H smooth',
               'MTF50 UHR', 'MTF10 UHR']

    values.extend([
        HUwater_head_min, HUwater_head_max,
        HUwater_body_min, HUwater_body_max,
        diff_head_max, diff_body_max,
        noise_head_max, noise_body_max,
        slice_head_min, slice_head_max,
        slice_body_min, slice_body_max,
        MTF50_body_smooth, MTF10_body_smooth,
        MTF50_head_smooth, MTF10_head_smooth,
        MTF50_UHR, MTF10_UHR
        ])

    data = {'status': status, 'errmsg': errmsg,
            'values': values, 'headers': headers}
    return data


def read_Siemens_CT_QC(txt):
    """Read Siemens CT daily or constancy report from pdf content.

    Parameters
    ----------
    txt : list of str
        pdf content

    Returns
    -------
    data : dict
        'status': bool
        'errmsg': list of str
        'values': list of str,
        'headers': list of str}
    """
    data = {'status': False, 'errmsg': 'Reader for Symbia not implemented',
            'values': [], 'headers': []}

    def get_slice_thickness_SiemensCT(txt, match_str,
                                      rownos_description, search_strings):
        """Read slice thickness parameters.

        Parameters
        ----------
        txt : list of str
            pdf content
        match_str : str
            string to find position of test in txt
        rownos_description : list of int
            all positions in txt where description (new test starts)
        search_strings : dict
            search strings for given language

        Returns
        -------
        list
            slice results
        """
        slice_head_min = ''
        slice_head_max = ''
        slice_body_min = ''
        slice_body_max = ''
        slice_dblA_min = ''
        slice_dblA_max = ''
        slice_dblB_min = ''
        slice_dblB_max = ''

        rownos = [
            index for index in range(len(txt))
            if fnmatch(txt[index], match_str)]
        if len(rownos) == 2:
            try:
                next_test = list(filter(
                    lambda i: i > rownos[-1], rownos_description))[1]
            except IndexError:
                next_test = -1

            for test_str in [
                    search_strings['test_typical_head'],
                    search_strings['test_typical_body']]:

                max_test = next_test if next_test > -1 else len(txt)
                test_rows = [row for row
                             in range(rownos[-1], max_test)
                             if fnmatch(txt[row], test_str + '*')]

                if len(test_rows) > 0:
                    for test_row in test_rows:  # body twice if dual tube
                        started = False
                        slice_res = []
                        slice_res_B = []

                        for i in range(test_row + 1, len(txt)):
                            if i in test_rows:
                                break  # next test
                            else:
                                if started is False:
                                    if fnmatch(txt[i],
                                               search_strings['slicewidth'] + '*'):
                                        started = True
                                else:
                                    if i in rownos_description:
                                        break  # next test
                                    else:
                                        if fnmatch(txt[i],
                                                   search_strings['slice'] + '*'):
                                            if fnmatch(txt[i],
                                                       search_strings[
                                                           'slicewidth'] + '*'):
                                                break  # next test
                                            else:
                                                split_txt = txt[i].split()
                                                if len(split_txt) >= 3:
                                                    slice_res.append(
                                                        float(split_txt[2]))
                                                if len(split_txt) == 6:
                                                    # dual tube
                                                    slice_res_B.append(
                                                        float(split_txt[4]))
                                                elif len(split_txt) == 8:
                                                    # dual tube
                                                    slice_res_B.append(
                                                        float(split_txt[5]))

                        if test_str == search_strings['test_typical_head']:
                            slice_head_min = min(slice_res)
                            slice_head_max = max(slice_res)
                        else:
                            if len(slice_res_B) == 0:
                                slice_body_min = min(slice_res)
                                slice_body_max = max(slice_res)
                            else:
                                slice_dblA_min = min(slice_res)
                                slice_dblA_max = max(slice_res)
                                slice_dblB_min = min(slice_res_B)
                                slice_dblB_max = max(slice_res_B)

        return [slice_head_min, slice_head_max, slice_body_min, slice_body_max,
                slice_dblA_min, slice_dblA_max, slice_dblB_min, slice_dblB_max]

    def get_homog_water_SiemensCT(txt, match_str,
                                  rownos_description, search_strings):
        """Read homogeneity and HU i water parameters from Siemens CT report.

        Parameters
        ----------
        txt : list of str
            pdf content
        match_str : str
            string to find position of test in txt
        rownos_description : list of int
            all positions in txt where description (new test starts)
        search_strings : dict
            search strings for given language

        Returns
        -------
        list
            homog water results
        """
        HUwater_head_min = ''
        HUwater_head_max = ''
        diff_head_max = ''
        HUwater_body_min = ''
        HUwater_body_max = ''
        diff_body_max = ''
        HUwater_dblA_min = ''
        HUwater_dblA_max = ''
        HUwater_dblB_min = ''
        HUwater_dblB_max = ''
        diff_dblA_max = ''
        diff_dblB_max = ''

        rownos = [
            index for index in range(len(txt))
            if fnmatch(txt[index], match_str)]
        if len(rownos) == 2:
            try:
                next_test = list(filter(
                    lambda i: i > rownos[-1], rownos_description))[1]
            except IndexError:
                next_test = -1

            for test_str in [
                    search_strings['test_typical_head'],
                    search_strings['test_typical_body']]:

                max_test = next_test if next_test > -1 else len(txt)
                test_rows = [row for row
                             in range(rownos[-1], max_test)
                             if fnmatch(txt[row], test_str + '*')]

                if len(test_rows) > 0:
                    for test_row in test_rows:  # body twice if dual tube
                        res_HU_val = []
                        res_HU_diff = []
                        res_HU_val_B = []
                        res_HU_diff_B = []
                        read_HU = False
                        read_diff = False
                        read_diff_B = False

                        for i in range(test_row + 1, len(txt)):
                            if i in test_rows:
                                break  # next test
                            else:
                                if not any(
                                        [read_HU, read_diff, read_diff_B]):
                                    if fnmatch(txt[i],
                                               search_strings['slicewidth'] + '*'):
                                        read_HU = True
                                else:
                                    if read_diff and len(res_HU_diff) > 0:
                                        if fnmatch(txt[i],
                                                   '*'+search_strings[
                                                       'results']+'*'):
                                            read_diff = False
                                            if len(test_rows) > 1:
                                                if test_row == test_rows[-1]:
                                                    read_diff_B = True
                                            else:
                                                break

                                    if len(res_HU_val) > 0 and read_diff_B is False:
                                        if fnmatch(txt[i],
                                                   '*'+search_strings[
                                                       'results']+'*'):
                                            read_HU = False
                                            read_diff = True

                                    if fnmatch(txt[i],
                                               search_strings['slice']+'*'):
                                        split_txt = txt[i].split()
                                        if read_HU:
                                            if len(split_txt) >= 3:
                                                res_HU_val.append(
                                                    float(split_txt[2]))
                                            if len(split_txt) == 6:
                                                # dual tube
                                                res_HU_val_B.append(
                                                    float(split_txt[4]))
                                            elif len(split_txt) == 8:
                                                # dual tube
                                                res_HU_val_B.append(
                                                    float(split_txt[5]))

                                        if read_diff:
                                            if len(split_txt) == 12:
                                                res_HU_diff.extend(
                                                    [split_txt[4], split_txt[6],
                                                     split_txt[8], split_txt[10]])
                                            elif len(split_txt) == 17:
                                                res_HU_diff.extend(
                                                    [split_txt[5], split_txt[8],
                                                     split_txt[11], split_txt[14]])
                                        if read_diff_B:
                                            if len(split_txt) == 12:
                                                res_HU_diff_B.extend(
                                                    [split_txt[4], split_txt[6],
                                                     split_txt[8], split_txt[10]])
                                            elif len(split_txt) == 17:
                                                res_HU_diff_B.extend(
                                                    [split_txt[5], split_txt[8],
                                                     split_txt[11], split_txt[14]])
                                            if len(res_HU_diff_B) == len(res_HU_diff):
                                                break

                        res_HU_diff = [
                            float(val) for val in res_HU_diff]
                        res_HU_diff_B = [
                            float(val) for val in res_HU_diff_B]

                        if test_str == search_strings['test_typical_head']:
                            HUwater_head_min = min(res_HU_val)
                            HUwater_head_max = max(res_HU_val)
                            diff_head_max = max(map(abs, res_HU_diff))
                        else:
                            if len(res_HU_val_B) == 0:
                                HUwater_body_min = min(res_HU_val)
                                HUwater_body_max = max(res_HU_val)
                                diff_body_max = max(map(abs, res_HU_diff))
                            else:
                                HUwater_dblA_min = min(res_HU_val)
                                HUwater_dblA_max = max(res_HU_val)
                                HUwater_dblB_min = min(res_HU_val_B)
                                HUwater_dblB_max = max(res_HU_val_B)
                                diff_dblA_max = max(map(abs, res_HU_diff))
                                diff_dblB_max = max(map(abs, res_HU_diff_B))

        return [HUwater_head_min, HUwater_head_max,
                HUwater_body_min, HUwater_body_max,
                diff_head_max, diff_body_max,
                HUwater_dblA_min, HUwater_dblA_max,
                HUwater_dblB_min, HUwater_dblB_max,
                diff_dblA_max, diff_dblB_max]

    def get_noise_SiemensCT(txt, match_str,
                            rownos_description, search_strings):
        """Read noise parameters from Siemens CT report.

        Parameters
        ----------
        txt : list of str
            pdf content
        match_str : str
            string to find position of test in txt
        rownos_description : list of int
            all positions in txt where description (new test starts)
        search_strings : dict
            search strings for given language

        Returns
        -------
        list
            noise results
        """
        noise_head_max = ''
        noise_body_max = ''
        noise_dblA_max = ''
        noise_dblB_max = ''

        rownos = [
            index for index in range(len(txt))
            if fnmatch(txt[index], match_str)]
        if len(rownos) == 2:
            try:
                next_test = list(filter(
                    lambda i: i > rownos[-1], rownos_description))[1]
            except IndexError:
                next_test = -1

            for test_str in [
                    search_strings['test_typical_head'],
                    search_strings['test_typical_body']]:

                max_test = next_test if next_test > -1 else len(txt)
                test_rows = [row for row
                             in range(rownos[-1], max_test)
                             if fnmatch(txt[row], test_str + '*')]

                if len(test_rows) > 0:
                    for test_row in test_rows:  # body twice if dual tube
                        started = False
                        noise_res = []
                        noise_res_B = []

                        for i in range(test_row, len(txt)):
                            if started is False:
                                if fnmatch(txt[i],
                                           search_strings['slicewidth']+'*'):
                                    started = True
                            else:
                                if fnmatch(txt[i], search_strings['slice']+'*'):

                                    if fnmatch(txt[i],
                                               search_strings['slicewidth']+'*'):
                                        break  # next test
                                    else:
                                        split_txt = txt[i].split()
                                        if len(split_txt) >= 3:
                                            noise_res.append(float(split_txt[2]))
                                        if len(split_txt) == 6:
                                            # dual tube
                                            noise_res_B.append(float(split_txt[4]))
                                        elif len(split_txt) == 8:
                                            # dual tube
                                            noise_res_B.append(float(split_txt[5]))

                        if test_str == search_strings['test_typical_head']:
                            noise_head_max = max(noise_res)
                        else:
                            if len(noise_res_B) == 0:
                                noise_body_max = max(noise_res)
                            else:
                                noise_dblA_max = max(noise_res)
                                noise_dblB_max = max(noise_res_B)

        return [noise_head_max, noise_body_max,
                noise_dblA_max, noise_dblB_max]

    def get_MTF_SiemensCT(txt, match_str,
                          rownos_description, search_strings):
        """Read MTF parameters from Siemens CT report.

        Parameters
        ----------
        txt : list of str
            pdf content
        match_str : str
            string to find position of test in txt
        rownos_description : list of int
            all positions in txt where description (new test starts)
        search_strings : dict
            search strings for given language

        Returns
        -------
        list
            MTF results
        """
        MTF50_body_smooth = None
        MTF10_body_smooth = None
        MTF50_head_smooth = None
        MTF10_head_smooth = None
        MTF50_head_sharp = None
        MTF10_head_sharp = None
        MTF50_UHR = None
        MTF10_UHR = None
        MTF50_dblA_smooth = None
        MTF10_dblA_smooth = None
        MTF50_dblB_smooth = None
        MTF10_dblB_smooth = None

        rownos = [
            index for index in range(len(txt))
            if fnmatch(txt[index], match_str)]
        if len(rownos) == 2:
            try:
                next_test = list(filter(
                    lambda i: i > rownos[-1], rownos_description))[1]
            except IndexError:
                next_test = -1

            for test_str in [
                    search_strings['test_typical_head'],
                    search_strings['test_typical_body'],
                    search_strings['test_sharpest_mode'],
                    ]:

                max_test = next_test if next_test > -1 else len(txt)
                test_rows = [row for row
                             in range(rownos[-1], max_test)
                             if fnmatch(txt[row], test_str + '*')]

                if len(test_rows) > 0:
                    for test_row in test_rows:
                        # body twice if dual tube
                        # + twice sharpest (without/with UHR)
                        started = False
                        mtf50_res = []
                        mtf10_res = []

                        for i in range(test_row, len(txt)):
                            if started is False:
                                if fnmatch(
                                        txt[i],
                                        search_strings['slicewidth'] + '*'):
                                    started = True
                            else:
                                if fnmatch(
                                        txt[i],
                                        search_strings['slicewidth'] + '*'):
                                    break  # next test has started
                                elif fnmatch(
                                        txt[i],
                                        search_strings['slice'] + '*'):
                                    split_txt = txt[i].split()
                                    if len(split_txt) == 6:
                                        mtf50_res.append(float(split_txt[2]))
                                        mtf10_res.append(float(split_txt[4]))
                                    elif len(split_txt) == 8:  # old style
                                        mtf50_res.append(float(split_txt[2]))
                                        mtf10_res.append(float(split_txt[5]))

                        if test_str == search_strings['test_typical_head']:
                            MTF50_head_smooth = sum(mtf50_res)/len(mtf50_res)
                            MTF10_head_smooth = sum(mtf10_res)/len(mtf10_res)
                        elif test_str == search_strings[
                                'test_typical_body']:
                            if MTF50_body_smooth is None:
                                MTF50_body_smooth = sum(mtf50_res)/len(mtf50_res)
                                MTF10_body_smooth = sum(mtf10_res)/len(mtf10_res)
                            else:
                                # dual tube results combined
                                # first half is tube A
                                v = int(len(mtf50_res)/2)
                                MTF50_dblA_smooth = sum(mtf50_res[0:v])/v
                                MTF50_dblB_smooth = sum(mtf50_res[v:])/v
                                MTF10_dblA_smooth = sum(mtf10_res[0:v])/v
                                MTF10_dblB_smooth = sum(mtf10_res[v:])/v
                        else:
                            # without and with UHR, without first
                            if MTF50_head_sharp is None:
                                MTF50_head_sharp = sum(mtf50_res)/len(mtf50_res)
                                MTF10_head_sharp = sum(mtf10_res)/len(mtf10_res)
                            else:
                                MTF50_UHR = sum(mtf50_res)/len(mtf50_res)
                                MTF10_UHR = sum(mtf10_res)/len(mtf10_res)

        return [MTF50_body_smooth, MTF10_body_smooth,
                MTF50_head_smooth, MTF10_head_smooth,
                MTF50_head_sharp, MTF10_head_sharp,
                MTF50_UHR, MTF10_UHR,
                MTF50_dblA_smooth, MTF10_dblA_smooth,
                MTF50_dblB_smooth, MTF10_dblB_smooth]

    ###################################################################
    file_type = 'standard'
    for txt_line in txt[0:2]:
        if fnmatch(txt_line, '*Symbia T*'):
            file_type = 'Symbia'
        else:
            if fnmatch(txt_line, '*Symbia Intevo*'):
                file_type = 'Intevo'

    if file_type == 'Symbia':
        # data = read_Siemens_CT_QC_Symbia(txt)
        data['errmsg'] = 'Reader for Symbia not implemented'
    else:
        values = []
        errmsg = []
        status = False
        headers = []

        res = get_Siemens_CT_QC_type_language(txt)
        search_strings = res['language_dict']

        proceed = True
        if res['report_type'] == '' or len(search_strings) == 0:
            errmsg = errmsg.append(
                'Unexpected content of expected Siemens CT QC report.')
            proceed = False

        if proceed:
            type_str = search_strings[res['report_type']]
            match_str = search_strings['description']
            rownos_description = [
                    index for index in range(len(txt))
                    if fnmatch(txt[index], match_str)]

            max_short_txt = 9
            short_txt = [x[0:max_short_txt] for x in txt]
            def_err_text = '?'
            def_err_msg = (
                'One or more parameters not found as expected in file.')

            # get date
            rowno = -1
            date = def_err_text
            for lineno, txt_line in enumerate(txt):
                if fnmatch(txt_line, '*' + search_strings['page1'] + '*'):
                    rowno = lineno
                    break
            if rowno > -1:
                split_txt = txt[rowno].split(' ')
                split_date = split_txt[0].split('-')
                if len(split_date) == 3:
                    date = f'{split_date[2]}.{split_date[1]}.{split_date[0]}'
                else:
                    errmsg = def_err_msg

            # get serial numbers tube A/B
            if fnmatch(txt[0], '*' + search_strings['serial_number'] + '*'):
                split_txt = txt[rowno + 1].split(' ')
                serial_number = split_txt[-1]
            else:
                serial_number = def_err_text
                errmsg = def_err_msg

            match_str = search_strings['tube_assembly']+'*'
            serial_tube_A = '-'
            serial_tube_B = '-'
            if len(txt) < 51:
                proceed = False
                data = {'status': False,
                        'errmsg': 'Content of file is too short',
                        'values': [], 'headers': []}
            else:
                rownos = [
                    index for index in range(50)
                    if fnmatch(txt[index], match_str)]
                if len(rownos) > 0:
                    split_txt = txt[rownos[0]].split(' ')
                    if len(split_txt) >= 4:
                        serial_tube_A = split_txt[-2]
                    else:
                        serial_tube_A = def_err_text
                        errmsg = def_err_msg
                    if len(rownos) > 1:
                        split_txt = txt[rownos[1]].split(' ')
                        if len(split_txt) >= 4:
                            serial_tube_B = split_txt[-2]
                        else:
                            serial_tube_B = def_err_text
                            errmsg = def_err_msg
    
                # get tester_name
                tester_name = ''
                match_str = search_strings['tester_name']+'*'
                rownos = [
                    index for index in range(50)
                    if fnmatch(txt[index], match_str)]
                if len(rownos) > 0:
                    split_txt = txt[rownos[0]].split(search_strings['customer'])
                    split_txt = split_txt[0].split()
                    tester_name = ' '.join(
                        split_txt[len(search_strings['tester_name'].split()):])
    
                # get product_name
                product_name = ''
                search_txt = search_strings['product_name'][0:max_short_txt]
                if search_txt in short_txt:
                    rowno = short_txt.index(search_txt)
                    split_txt = txt[rowno].split(search_strings['customer'])
                    if len(split_txt) == 1:
                        split_txt = txt[rowno].split(search_strings['zipcode'])
                        if len(split_txt) == 1:
                            split_txt = txt[rowno].split(search_strings['street'])
                    split_txt = split_txt[0].split()
                    product_name = ' '.join(
                        split_txt[len(search_strings['product_name'].split()):])
    
                if file_type == 'Intevo':
                    proceed = False
                    info = [date, tester_name, product_name,
                            serial_number, serial_tube_A]
                    data = read_Siemens_CT_QC_Intevo(
                        txt, type_str, search_strings, info, errmsg)

        if proceed:
            status = True

            match_str = '*' + type_str + '*' + search_strings['slice'] + '*'
            slice_res = get_slice_thickness_SiemensCT(
                txt, match_str, rownos_description, search_strings)

            match_str = '*'+type_str+'*'+search_strings['homogeneity']+'*'
            homog_water_res = get_homog_water_SiemensCT(
                txt, match_str, rownos_description, search_strings)

            match_str = '*' + type_str + '*' + search_strings['noise'] + '*'
            noise_res = get_noise_SiemensCT(
                txt, match_str, rownos_description, search_strings)

            match_str = '*' + type_str + '*MTF*'
            MTF_res = get_MTF_SiemensCT(
                txt, match_str, rownos_description, search_strings)

            headers = ['Date', 'Tester name', 'Product Name', 'Serial Number',
                       'Serial Tube A', 'Serial Tube B',
                       'HUwater head min', 'HUwater head max',
                       'HUwater body min', 'HUwater body max',
                       'Diff head max(abs)', 'Diff body max(abs)',
                       'Noise head max', 'Noise body max',
                       'Slice head min', 'Slice head max',
                       'Slice body min', 'Slice body max',
                       'MTF50 body smooth', 'MTF10 body smooth',
                       'MTF50 head smooth', 'MTF10 head smooth',
                       'MTF50 head sharp', 'MTF10 head sharp',
                       'MTF50 UHR', 'MTF10 UHR',
                       'HUwater dblA min', 'HUwater dblA max',
                       'HUwater dblB min', 'HUwater dblB max',
                       'Diff dblA max(abs)', 'Diff dblB max(abs)',
                       'Noise dblA max', 'Noise dblB max',
                       'Slice dblA min', 'Slice dblA max',
                       'Slice dblB min', 'Slice dblB max',
                       'MTF50 dblA smooth', 'MTF10 dblA smooth',
                       'MTF50 dblB smooth', 'MTF10 dblB smooth']

            values = [date, tester_name, product_name, serial_number,
                      serial_tube_A, serial_tube_B]
            values.extend(homog_water_res[0:6])
            values.extend(noise_res[0:2])
            values.extend(slice_res[0:4])
            values.extend(MTF_res[0:8])
            values.extend(homog_water_res[6:])
            values.extend(noise_res[2:])
            values.extend(slice_res[4:])
            values.extend(MTF_res[8:])

            data = {'status': status, 'errmsg': errmsg,
                    'values': values, 'headers': headers}

    if txt[0] == 'System Quality Report':
        # TODO - what is this, when to use it? date is changed, but not data...
        short_txt = [x[0:9] for x in txt]
        # date
        date = ''
        search_txt = 'Scan Date'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            date_str = txt[rowno].split(',')
            date_md = date_str[0].split(' ')
            if len(date_md) >= 4:
                day = f'{int(date_md[3]):02}'
                month_dt = datetime.datetime.strptime(date_md[2], '%B')
                month = f'{month_dt.month:02}'
                year = date_str[1].strip()
                date = f'{day}.{month}.{year}'

        status = True

    return data


def read_Philips_MR_ACR_report(txt):
    """Read Philips MR ACR report from list of str.

    Parameters
    ----------
    txt : list of str
        text from pdf

    Returns
    -------
    data : dict
        'status': bool
        'errmsg': list of str
        'values': list of str,
        'headers': list of str}
    """
    values = []
    headers = []
    errmsg = []
    status = False

    if 'ACR R' in txt[0]:

        # ensure page break do not hamper results
        to_remove = []
        for i, line in enumerate(txt):
            if line[0:6] == 'file:/' or 'ACR Report Page' in line:
                to_remove.insert(0, i)
        for i in to_remove:
            txt.pop(i)

        txt = [x.strip() for x in txt]
        short_txt = [x[0:9] for x in txt]

        # date
        date = ''
        search_txt = 'Acquisiti'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            date_str = txt[rowno].split(',')
            if len(date_str) == 3:  # expected
                date_md = date_str[1].strip().split(' ')
                if len(date_md) == 2:
                    day = date_md[1]
                    month_dt = datetime.datetime.strptime(date_md[0], '%B')
                    month = f'{month_dt.month:02}'
                    year = date_str[2].strip()
                    date = f'{day}.{month}'
                year = date_str[2].strip().split(' ')[0]
                date = date + f'.{year}'

        # test type
        test_type = ''
        search_txt = 'Test Type'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(':')
            test_type = split_txt[-1].strip()

        # table position
        table_position = None
        search_txt = 'Table Pos'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt) + 1  # (second lind)
            split_txt = txt[rowno].split(' ')
            try:
                table_position = float(split_txt[-1])
            except ValueError:
                pass

        # center frequency
        center_frequency = None
        search_txt = 'Center fr'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            try:
                center_frequency = int(split_txt[-1])
            except ValueError:
                pass

        # Transmit Gain
        transmit_gain = None
        search_txt = 'Transmit '
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            try:
                transmit_gain = float(split_txt[-1])
            except ValueError:
                pass

        # Geometric Accuracy
        geometric_vals = [None] * 4
        search_txt = 'Geometric Measurement Table:'
        if search_txt in txt:
            rowno = txt.index(search_txt)
            # slice 5, min max
            rows = np.arange(5, 9) + rowno
            vals = []
            for row in rows:
                split_txt = txt[row].split(' ')
                try:
                    val = float(split_txt[-1])
                    vals.append(val)
                except ValueError:
                    pass
            if len(vals) == 4:
                geometric_vals = vals

        # High /low contrast resolution
        hc_vals = [None] * 4
        lc_vals = [None] * 4
        search_txt = 'Low-Contrast Detectability:'
        if search_txt in txt:
            rowno = txt.index(search_txt)
            # High contrast
            rows = np.arange(-2, 0) + rowno
            vals = []
            for row in rows:
                split_txt = txt[row].split(')')
                if len(split_txt) == 4:
                    split_sub = split_txt[-2].strip().split(' ')
                    vals.append(split_sub[0])  # 1.0mm
                    vals.append(split_txt[-1].strip())  # 0.9mm
                elif len(split_txt) == 3:
                    split_sub = split_txt[-1].strip().split(' ')
                    vals.append(split_sub[-2])  # 1.0mm
                    vals.append(split_sub[-1])  # 0.9mm
            if len(vals) == 4:
                hc_vals = vals

            # Low contrast
            rows = np.arange(3, 7) + rowno
            vals = []
            for row in rows:
                split_txt = txt[row].split(' ')
                if len(split_txt) == 6:
                    try:
                        val = int(split_txt[-1])
                        vals.append(val)
                    except ValueError:
                        pass
            if len(vals) == 4:
                lc_vals = vals

        values = (
            [date, test_type, table_position, center_frequency, transmit_gain]
            + geometric_vals + hc_vals + lc_vals)
        headers = ['Date', 'Test type', 'Table position (mm)',
                   'Center frequency (Hz)', 'Transmit Gain',
                   'Geometric top bottom', 'Geometric left right',
                   'Geometric Diag-UL', 'Geometric Diag-UR',
                   'HC Hor 1mm', 'HC Hor 0.9mm',
                   'HC Ver 1mm', 'HC Ver 0.9mm',
                   'LC Slice8 resolved', 'LC Slice9 resolved',
                   'LC Slice10 resolved', 'LC Slice11 resolved'
                   ]

        status = True

    data = {'status': status, 'errmsg': errmsg,
            'values': values, 'headers': headers}
    return data


def read_GE_QAP(filepath):
    """Read GE QAP report from txt-file.

    Parameters
    ----------
    filepath : str
        .txt file

    Returns
    -------
    data : dict
        'status': bool
        'errmsg': list of str
        'values': list of str,
        'headers': list of str}
    """
    values = []
    headers = []
    errmsg = []
    status = False

    filepath = Path(filepath)
    if len(filepath.name) > 8:
        datestr = filepath.name[:8]
        date = f'{datestr[6:8]}.{datestr[4:6]}.{datestr[:4]}'
    else:
        date = ''

    txt = []
    with open(filepath, 'r') as file:
        txt = file.readlines()

    short_txt = [x[0:9] for x in txt]

    if len(txt) > 6:
        if 'IMAGE QUALITY' in txt[0]:
            detector_serial = txt[1].split()[-1]

            overall_result = ''
            search_txt = 'OVERALL R'
            if search_txt in short_txt:
                rowno = short_txt.index(search_txt)
                overall_result = txt[rowno].split()[-1]

            bad_pixels = ''
            search_txt = 'No. Of Ba'
            if search_txt in short_txt:
                rowno = short_txt.index(search_txt)
                try:
                    bad_pixels = int(float(txt[rowno].split()[-4]))
                except (IndexError, ValueError):
                    pass

            global_non_unif = ''
            search_txt = 'Global Br'
            if search_txt in short_txt:
                rowno = short_txt.index(search_txt)
                try:
                    global_non_unif = float(txt[rowno].split()[-4])
                except (IndexError, ValueError):
                    pass

            local_non_unif = ''
            search_txt = 'Local Bri'
            if search_txt in short_txt:
                rowno = short_txt.index(search_txt)
                try:
                    local_non_unif = float(txt[rowno].split()[-4])
                except (IndexError, ValueError):
                    pass

            SNR_non_unif = ''
            search_txt = 'SNR Non U'
            if search_txt in short_txt:
                rowno = short_txt.index(search_txt)
                try:
                    SNR_non_unif = float(txt[rowno].split()[-4])
                except (IndexError, ValueError):
                    pass

            MTF_vals = []
            MTF_headers = []
            search_txt = 'Spatial M'
            if search_txt in short_txt:
                rowno = short_txt.index(search_txt)
                try:
                    MTF_vals = [float(txt[rowno+i].split()[-4]) for i in range(5)]
                    MTF_headers = [f'MTF at {txt[rowno+i].split()[3]} lp/mm' for i
                                   in range(5)]
                except (IndexError, ValueError):
                    pass

            values = [date, detector_serial, overall_result,
                      bad_pixels, global_non_unif, local_non_unif, SNR_non_unif]
            headers = ['Date', 'Detector serial', 'Overall result',
                       'Bad pixels', 'Global brightness non uniformity',
                       'Local brightness non uniformity', 'SNR non uniformity']
            if len(MTF_vals) > 0 and len(MTF_headers) > 0:
                values.extend(MTF_vals)
                headers.extend(MTF_headers)
            status = True

    data = {'status': status, 'errmsg': errmsg,
            'values': values, 'headers': headers}

    return data


def read_GE_Mammo_date(filepath):
    """Read GE Mammo QAP date from txt-file path string.

    Parameters
    ----------
    filepath : Path
        .txt file

    Returns
    -------
    dd, mm, yyyy : tuple of str
    """
    ddmmyyyy = ('', '', '')
    splitname = filepath.name.split('_')
    if len(splitname) >= 3:
        datestr = splitname[-2]
        if len(datestr) == 8:
            dd = datestr.split('.')
            if len(dd) == 3:
                ddmmyyyy = (dd[0], dd[1], f'20{dd[2]}')
                # assume GE or this software outdated in year 2100 :)
    return ddmmyyyy


def read_GE_Mammo_QAP(filepath):
    """Read GE Mammo QAP report from txt-file.

    Parameters
    ----------
    filepath : str
        .txt file

    Returns
    -------
    data : dict
        'status': bool
        'errmsg': list of str
        'values': list of str,
        'headers': list of str}
    """
    values = []
    headers = []
    limits = []
    errmsg = []
    testname = ''
    date = ''
    status = False

    filepath = Path(filepath)
    splitname = filepath.name.split('_BasicResults')
    if len(splitname) >= 2:
        testname = splitname[0]
        dd, mm, yyyy = read_GE_Mammo_date(filepath)
        date = f'{dd}.{mm}.{yyyy}'

    txt = []
    with open(filepath, 'r') as file:
        txt = file.readlines()

    if len(txt) > 4:
        overall_result = 'PASSED'
        for line in txt[4:]:
            line_split = line.split()
            if len(line_split) >= 5:
                header_this = '?'
                value_this = None
                lower_limit_this = None
                upper_limit_this = None
                shift = len(line_split) - 5
                if shift >= 0:
                    if shift > 0:
                        header_this = ' '.join(line_split[0:shift+1])
                    else:
                        header_this = line_split[0]
                    value_this = line_split[shift + 1]
                    lower_limit_this = line_split[shift + 2]
                    upper_limit_this = line_split[shift + 3]
                try:
                    value_this = float(value_this)
                except (ValueError, TypeError):
                    pass
                try:
                    lower_limit_this = float(lower_limit_this)
                except (ValueError, TypeError):
                    lower_limit_this = None
                try:
                    upper_limit_this = float(upper_limit_this)
                except (ValueError, TypeError):
                    upper_limit_this = None

                if line_split[-1] != 'PASSED':
                    overall_result = line_split[-1]
                headers.append(header_this)
                values.append(value_this)
                limits.append([lower_limit_this, upper_limit_this])

        values = [date, testname, overall_result] + values
        headers = ['Date', 'Testname', 'Overall_result'] + headers
        limits = [[None, None]] * 3 + limits
        status = True

    data = {'status': status, 'errmsg': errmsg,
            'values': values, 'headers': headers,
            'limits': limits}

    return data


def read_Planmeca_html(filepath):
    """Read Planmeca report from Viso QA.

    Parameters
    ----------
    root: root element
        from xml.etree.ElementTree.getroot()

    Returns
    -------
    data : dict
        'status': bool
        'errmsg': list of str
        'values': list of str,
        'headers': list of str}

    """
    txt = []
    with open(filepath, 'r') as file:
        txt = file.readlines()

    def get_td_value(td_line):
        """Extract value from <td ...>value</td>."""
        split_string = re.split('<|>', td_line)
        split_string = list(filter(None, split_string))
        res = None
        if len(split_string) == 5:
            res = split_string[2]
            try:
                res = float(res)
            except TypeError:
                pass
        return res

    values = []
    headers = []
    errmsg = []
    status = False

    if '<HTML>' not in txt[0]:
        errmsg = ['Expected file to start with <HTML>. File reading aborted.']
    else:
        # date
        filepath = Path(filepath)
        date = ''
        try:
            datestr = filepath.name[7:7+9]
            date = f'{datestr[6:8]}.{datestr[4:6]}.{datestr[:4]}'
        except IndexError:
            pass

        rownos = [
            index for index in range(len(txt))
            if fnmatch(txt[index], '<H4>Mode </H4>\n')]
        if len(rownos) == 3:
            for rowno in rownos:
                for addrow in [11, 19, 27, 35, 43, 51, 59, 67, 75]:
                    values.append(get_td_value(txt[rowno + addrow]))
        values = [date] + values
        headers_one = ['MTF 50', 'Air Density', 'Air Uniformity', 
                       'Aluminum Density', 'Aluminum SNR', 'Aluminum Uniformity',
                       'Acryl Density', 'Acryl SNR', 'Acryl Uniformity']
        headers = (
            ['Date'] 
            + ['1 ' + x for x in headers_one]
            + ['2 ' + x for x in headers_one]
            + ['3 ' + x for x in headers_one])
        status = True

    data = {'status': status, 'errmsg': errmsg,
            'values': values, 'headers': headers}
    return data


def get_pdf_txt(path):
    """Extract text from pdf file using pdfplumber.

    Return
    ------
    pdf_txt : list of str
    """
    pdf_txt = []
    with pdfplumber.open(path) as file:
        for page in file.pages:
            pdf_txt.extend(page.extract_text().splitlines())

    return pdf_txt


def get_xml_tree(path):
    """Read xml tree.

    Return
    ------
    root : object
    """
    tree = ET.parse(path)
    root = tree.getroot()

    return root
