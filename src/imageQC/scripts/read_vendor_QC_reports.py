#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read different QC reports from different vendors.

@author: EllenWasbo
"""

from fnmatch import fnmatch
import datetime
import xml.etree.ElementTree as ET

import pdfplumber
from charset_normalizer import md__mypyc

#TODO - fnmatch is case insensitive on Windows. Adding option for .lower for handeling other os


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
        'values_txt': list of str,
        'headers': list of str}
    """
    result_txt = []
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
        calib_factor = '-1'
        search_txt = 'Calibrati'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            calib_factor = split_txt[-1]

        # Block Noise 3 [crystal] 0 [crystal] 0 Blocks
        n_blocks_noise = '-1'
        search_txt = 'Block Noi'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno-1].split(' ')
            n_blocks_noise = split_txt[0]

        # Block Efficiency 120 [%] 80 [%] 0 Blocks
        n_blocks_efficiency = '-1'
        search_txt = 'Block Eff'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno-1].split(' ')
            n_blocks_efficiency = split_txt[0]

        # Randoms 115 [%] 85 [%] 103.8 [%] Passed
        measured_randoms = '-1'
        search_txt = 'Randoms'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno-1].split(' ')
            split_txt = [x for x in split_txt if x != '']
            measured_randoms = split_txt[-3]

        # Scanner Efficiency
        scanner_efficiency = '-1'
        search_txt = 'Scanner E'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            if len(split_txt) == 3:  # version VG80B or similar
                split_txt = txt[rowno-1].split(' ')
                scanner_efficiency = split_txt[-2]
            else:  # assume VG60A or similiar
                split_txt = [x for x in split_txt if x != '']
                scanner_efficiency = split_txt[-3]

        # Scatter Ratio 35.2 [%] 28.8 [%] 30.7 [%] Passed
        scatter_ratio = '-1'
        search_txt = 'Scatter R'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            split_txt = [x for x in split_txt if x != '']
            scatter_ratio = split_txt[-3]

        # ECF
        ECF = '-1'
        search_txt = 'Scanner e'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            split_txt = [x for x in split_txt if x != '']
            ECF = split_txt[-1]

        # Image Plane Efficiency
        n_img_efficiency = '-1'
        search_txt = 'Image Pla'
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno].split(' ')
            split_txt = [x for x in split_txt if x != '']
            n_img_efficiency = split_txt[2]

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
        phantom_pos_x = '-1'
        phantom_pos_y = '-1'
        search_txt = 'Axis Valu'  # older versions
        if search_txt in short_txt:
            rowno = short_txt.index(search_txt)
            split_txt = txt[rowno+1].split(' ')
            phantom_pos_x = split_txt[-1]
            split_txt = txt[rowno+2].split(' ')
            phantom_pos_y = split_txt[-1]
        else:
            search_txt = 'Phantom P'
            if search_txt in short_txt:
                indexes = [
                    index for index in range(len(short_txt))
                    if short_txt[index] == search_txt]
                if len(indexes) == 4:
                    split_txt = txt[indexes[1]+1].split(' ')
                    split_txt = [x for x in split_txt if x != '']
                    phantom_pos_x = split_txt[4]
                    split_txt = txt[indexes[2]+1].split(' ')
                    split_txt = [x for x in split_txt if x != '']
                    phantom_pos_y = split_txt[4]

        result_txt = [date, ics_name, partial, full, timealign,
                      calib_factor, measured_randoms,
                      scanner_efficiency, scatter_ratio,
                      ECF,
                      n_blocks_noise, n_blocks_efficiency,
                      n_img_efficiency,
                      phantom_pos_x, phantom_pos_y]
        headers = ['Date', 'ICSname', 'Partial', 'Full', 'TimeAlignment',
                   'Calibration Factor', 'Measured randoms %',
                   'Scanner Efficiency [cps/Bq/cc]', ' Scatter Ratio %',
                   'ECF [Bq*s/ECAT counts]',
                   'Blocks out of range noise',
                   'Blocks out of range efficiency',
                   'Image planes out of range efficiency',
                   'Phantom Pos x', 'Phantom Pos y'
                   ]

        status = True

    data = {'status': status, 'errmsg': errmsg,
            'values_txt': result_txt, 'headers': headers}
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
        'values_txt': list of str,
        'headers': list of str}

    """
    result_txt = []
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
                calib_factor = root.find('cPhantomParameters').find(
                    'eCalibrationFactor').text
            except AttributeError:
                calib_factor = def_err_text
                errmsg = def_err_msg
            try:
                measured_randoms = details.find('cMeasureRandoms').find(
                    'cBlkValue').find('aValue').text
            except AttributeError:
                measured_randoms = def_err_text
                errmsg = def_err_msg
            try:
                scanner_efficiency = details.find('dScannerEfficiency').find(
                    'cBlkValue').find('aValue').text
            except AttributeError:
                scanner_efficiency = def_err_text
                errmsg = def_err_msg
            try:
                scatter_ratio = details.find('eScatterRatio').find(
                    'cBlkValue').find('aValue').text
            except AttributeError:
                scatter_ratio = def_err_text
                errmsg = def_err_msg
            try:
                ECF = details.find('fECF').find('cBlkValue').find(
                    'aValue').text
            except AttributeError:
                ECF = def_err_text
                errmsg = def_err_msg
            try:
                n_blocks_noise = details.find('aBlockNoise').find(
                    'cBlkValue').find('aValue').text
            except AttributeError:
                n_blocks_noise = def_err_text
                errmsg = def_err_msg
            try:
                n_blocks_efficiency = details.find('bBlockEfficiency').find(
                    'cBlkValue').find('aValue').text
            except AttributeError:
                n_blocks_efficiency = def_err_text
                errmsg = def_err_msg
            try:
                n_img_efficiency = details.find('gPlaneEff').find(
                    'cBlkValue').find('aValue').text
            except AttributeError:
                n_img_efficiency = def_err_text
                errmsg = def_err_msg

            result_txt = [date, calib_factor, measured_randoms,
                          scanner_efficiency, scatter_ratio,
                          ECF,
                          n_blocks_noise, n_blocks_efficiency,
                          n_img_efficiency]
            headers = ['Date', 'Calibration Factor', 'Measured randoms %',
                       'Scanner Efficiency [cps/Bq/cc]', ' Scatter Ratio %',
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
            'values_txt': result_txt, 'headers': headers}
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
    #TODO put search strings in config folder as yaml file editable by gui or txt editor
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
            'result': 'Result',
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
            'result': 'Resultat',
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
        '''if file_type == 'Symbia':  # inconsistent language
            for txt_line in txt:
                if fnmatch(txt_line,
                                   search_strings[langu]['tester_name']):
                    language = langu
                    break'''

        if report_type != '':
            break

    if language == '':
        return_strings = {}
    else:
        return_strings = search_strings[language]

    return {
        'language_dict': return_strings,
        'report_type': report_type
        }

    '''
# Probably not relevant at the time of this program to finish.
#End of life model.
def read_Siemens_CT_QC_Symbia(txt):
    result_txt = []
    errmsg = []
    status = False
    headers = []

    res = get_Siemens_CT_QC_type_language(txt)

    proceed = True
    if res['report_type'] == '' or len(res['language_dict']) == 0:
        errmsg = errmsg.append(
            'Unexpected content of expected Siemens CT QC report.')
        proceed = False

    if proceed:

        #TODO .........

        headers = ['Date', 'Tester name', 'Product Name', 'Serial Number',
                   'Tube ID',
                   'HUwater 110kV min', 'HUwater 110kV max',
                   'HUwater 130kV min', 'HUwater 130kV max',
                   'Diff 110kV max(abs)', 'Diff 130kV max(abs)',
                   'Noise 80kV', 'Noise 110kV', 'Noise 130kV',
                   'Slice 1mm', 'Slice 1.5mm', 'Slice 2.5mm',
                   'Slice 4mm', 'Slice 5mm',
                   'MTF50 B31s', 'MTF10 B31s',
                   'MTF50 H41s', 'MTF10 H41s',
                   'MTF50 U90s', 'MTF10 U90s']

        result_txt = [date, tester_name, product_name, serial_number,
                      serial_tube,
                      HUwater_110kV_min, HUwater_110kV_max,
                      HUwater_130kV_min, HUwater_130kV_max,
                      diff_110kV_max, diff_130kV_max,
                      noise_80kV_max, noise_110kV_max, noise_130kV_max,
                      slice_1, slice_1_5, slice_2_5, slice_4, slice_5,
                      MTF50_B31s, MTF10_B31s,
                      MTF50_H41s, MTF10_H41s,
                      MTF50_U90s, MTF10_U90s
                      ]

    data = {'status': status, 'errmsg': errmsg,
            'values_txt': result_txt, 'headers': headers}
    return data
'''


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
        'values_txt': list of str,
        'headers': list of str}
    """
    result_txt = info
    status = True

    # slice thickness
    slice_head_min = ''
    slice_head_max = ''
    slice_body_min = ''
    slice_body_max = ''

    match_str = '*' + type_str + '*' + search_strings['slice'] + '*'
    row_test = -1
    for i in range(len(txt)):
        if fnmatch(txt[i], match_str):
            row_test = i
            break

    if row_test > -1:

        match_str = search_strings['n_images'] + '*'
        test_rows = [
            index for index in range(row_test, len(txt))
            if fnmatch(txt[index], match_str)]
        if len(test_rows) == 2:
            for row in test_rows:
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
                    slice_head_min = f'{min(slice_res)}'
                    slice_head_max = f'{max(slice_res)}'
                else:
                    slice_body_min = f'{min(slice_res)}'
                    slice_body_max = f'{max(slice_res)}'

    # homogeneity / water HUwater
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
    row_test = -1
    for i in range(len(txt)):
        if fnmatch(txt[i], match_str):
            row_test = i
            break

    if row_test > -1:

        match_str = search_strings['n_images'] + '*'
        test_rows = [
            index for index in range(row_test, len(txt))
            if fnmatch(txt[index], match_str)]
        if len(test_rows) == 6:
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
                                if len(split_txt) == 16:
                                    res_HU_diff.extend(
                                        [split_txt[4], split_txt[7],
                                         split_txt[10], split_txt[13]])

                res_HU_diff = [float(val) for val in res_HU_diff]

                if HUwater_head_min == '':
                    HUwater_head_min = f'{min(res_HU_val)}'
                    HUwater_head_max = f'{max(res_HU_val)}'
                    diff_head_max = f'{max(map(abs, res_HU_diff))}'
                else:
                    HUwater_body_min = f'{min(res_HU_val)}'
                    HUwater_body_max = f'{max(res_HU_val)}'
                    diff_body_max = f'{max(map(abs, res_HU_diff))}'

    # noise
    '''7 tests
    extract test 6 (body full coll, 130 kV)
    + test 7 (head full coll, 130 kV)'''
    noise_head_max = ''
    noise_body_max = ''

    match_str = '*'+type_str+'*'+search_strings['noise']+'*'
    row_test = -1
    for i in range(len(txt)):
        if fnmatch(txt[i], match_str):
            row_test = i
            break

    if row_test > -1:
        match_str = search_strings['n_images'] + '*'
        test_rows = [
            index for index in range(row_test, len(txt))
            if fnmatch(txt[index], match_str)]
        if len(test_rows) == 7:
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

                if noise_head_max == '':
                    noise_head_max = f'{max(noise_res)}'
                else:
                    noise_body_max = f'{max(noise_res)}'

    # MTF
    '''5 tests
    extract test 1-3 (130 kV body B41, head H31, U90)'''
    MTF50_body_smooth = ''
    MTF10_body_smooth = ''
    MTF50_head_smooth = ''
    MTF10_head_smooth = ''
    MTF50_UHR = ''
    MTF10_UHR = ''

    match_str = '*' + type_str + '*MTF*'
    row_test = -1
    for i in range(len(txt)):
        if fnmatch(txt[i], match_str):
            row_test = i
            break

    if row_test > -1:

        match_str = search_strings['n_images'] + '*'
        test_rows = [
            index for index in range(row_test, len(txt))
            if fnmatch(txt[index], match_str)]
        if len(test_rows) == 5:
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
                            if len(split_txt) == 10:
                                mtf50_res.append(float(split_txt[1]))
                                mtf10_res.append(float(split_txt[4]))

                if row == actual_test_rows[0]:
                    MTF50_body_smooth = f'{sum(mtf50_res)/len(mtf50_res):.2f}'
                    MTF10_body_smooth = f'{sum(mtf10_res)/len(mtf10_res):.2f}'
                elif row == actual_test_rows[1]:
                    MTF50_head_smooth = f'{sum(mtf50_res)/len(mtf50_res):.2f}'
                    MTF10_head_smooth = f'{sum(mtf10_res)/len(mtf10_res):.2f}'
                else:
                    MTF50_UHR = f'{sum(mtf50_res)/len(mtf50_res):.2f}'
                    MTF10_UHR = f'{sum(mtf10_res)/len(mtf10_res):.2f}'

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

    result_txt.extend([
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
            'values_txt': result_txt, 'headers': headers}
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
        'values_txt': list of str,
        'headers': list of str}
    """
    data = {'status': False, 'errmsg': 'Reader for Symbia not implemented',
            'values_txt': [], 'headers': []}

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
        result_txt = []
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
            for r, txt_line in enumerate(txt):
                if fnmatch(txt_line, '*' + search_strings['page1'] + '*'):
                    rowno = r
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

            result_txt = [date, tester_name, product_name, serial_number,
                          serial_tube_A, serial_tube_B]
            result_txt.extend(homog_water_res[0:6])
            result_txt.extend(noise_res[0:2])
            result_txt.extend(slice_res[0:4])
            result_txt.extend(MTF_res[0:8])
            result_txt.extend(homog_water_res[6:])
            result_txt.extend(noise_res[2:])
            result_txt.extend(slice_res[4:])
            result_txt.extend(MTF_res[8:])

            data = {'status': status, 'errmsg': errmsg,
                    'values_txt': result_txt, 'headers': headers}

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
                        slice_head_min = f'{min(slice_res)}'
                        slice_head_max = f'{max(slice_res)}'
                    else:
                        if len(slice_res_B) == 0:
                            slice_body_min = f'{min(slice_res)}'
                            slice_body_max = f'{max(slice_res)}'
                        else:
                            slice_dblA_min = f'{min(slice_res)}'
                            slice_dblA_max = f'{max(slice_res)}'
                            slice_dblB_min = f'{min(slice_res_B)}'
                            slice_dblB_max = f'{max(slice_res_B)}'

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
                        HUwater_head_min = f'{min(res_HU_val)}'
                        HUwater_head_max = f'{max(res_HU_val)}'
                        diff_head_max = f'{max(map(abs, res_HU_diff))}'
                    else:
                        if len(res_HU_val_B) == 0:
                            HUwater_body_min = f'{min(res_HU_val)}'
                            HUwater_body_max = f'{max(res_HU_val)}'
                            diff_body_max = (
                                f'{max(map(abs, res_HU_diff))}')
                        else:
                            HUwater_dblA_min = f'{min(res_HU_val)}'
                            HUwater_dblA_max = f'{max(res_HU_val)}'
                            HUwater_dblB_min = f'{min(res_HU_val_B)}'
                            HUwater_dblB_max = f'{max(res_HU_val_B)}'
                            diff_dblA_max = (
                                f'{max(map(abs, res_HU_diff))}')
                            diff_dblB_max = (
                                f'{max(map(abs, res_HU_diff_B))}')

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
                        noise_head_max = f'{max(noise_res)}'
                    else:
                        if len(noise_res_B) == 0:
                            noise_body_max = f'{max(noise_res)}'
                        else:
                            noise_dblA_max = f'{max(noise_res)}'
                            noise_dblB_max = f'{max(noise_res_B)}'

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
    MTF50_body_smooth = ''
    MTF10_body_smooth = ''
    MTF50_head_smooth = ''
    MTF10_head_smooth = ''
    MTF50_head_sharp = ''
    MTF10_head_sharp = ''
    MTF50_UHR = ''
    MTF10_UHR = ''
    MTF50_dblA_smooth = ''
    MTF10_dblA_smooth = ''
    MTF50_dblB_smooth = ''
    MTF10_dblB_smooth = ''

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
                            elif fnmatch(txt[i],
                                         search_strings['slice'] + '*'):
                                split_txt = txt[i].split()
                                if len(split_txt) == 6:
                                    mtf50_res.append(float(split_txt[2]))
                                    mtf10_res.append(float(split_txt[4]))
                                elif len(split_txt) == 8:  # old style
                                    mtf50_res.append(float(split_txt[2]))
                                    mtf10_res.append(float(split_txt[5]))

                    if test_str == search_strings['test_typical_head']:
                        MTF50_head_smooth = (
                            f'{sum(mtf50_res)/len(mtf50_res):.2f}')
                        MTF10_head_smooth = (
                            f'{sum(mtf10_res)/len(mtf10_res):.2f}')
                    elif test_str == search_strings[
                            'test_typical_body']:
                        if len(MTF50_body_smooth) == 0:
                            MTF50_body_smooth = (
                                f'{sum(mtf50_res)/len(mtf50_res):.2f}')
                            MTF10_body_smooth = (
                                f'{sum(mtf10_res)/len(mtf10_res):.2f}')
                        else:
                            # dual tube results combined
                            # first half is tube A
                            v = int(len(mtf50_res)/2)
                            MTF50_dblA_smooth = f'{sum(mtf50_res[0:v])/v:.2f}'
                            MTF50_dblB_smooth = f'{sum(mtf50_res[v:])/v:.2f}'
                            MTF10_dblA_smooth = f'{sum(mtf10_res[0:v])/v:.2f}'
                            MTF10_dblB_smooth = f'{sum(mtf10_res[v:])/v:.2f}'
                    else:
                        # without and with UHR, without first
                        if len(MTF50_head_sharp) == 0:
                            MTF50_head_sharp = (
                                f'{sum(mtf50_res)/len(mtf50_res):.2f}')
                            MTF10_head_sharp = (
                                f'{sum(mtf10_res)/len(mtf10_res):.2f}')
                        else:
                            MTF50_UHR = (
                                f'{sum(mtf50_res)/len(mtf50_res):.2f}')
                            MTF10_UHR = (
                                f'{sum(mtf10_res)/len(mtf10_res):.2f}')

    return [MTF50_body_smooth, MTF10_body_smooth,
            MTF50_head_smooth, MTF10_head_smooth,
            MTF50_head_sharp, MTF10_head_sharp,
            MTF50_UHR, MTF10_UHR,
            MTF50_dblA_smooth, MTF10_dblA_smooth,
            MTF50_dblB_smooth, MTF10_dblB_smooth]


def get_pdf_txt(path):
    """Extract text from pdf file using pdfplumber.
    
    Return
    ------
    pdf_txt : list of str
    """
    pdf_txt = []
    with pdfplumber.open(path) as f:
        for page in f.pages:
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

#path = r"C:\Users\ellen\CloudStation\ImageQCpy\30jun2022_edit\tests\test_inputs\vendor_QC_reports\SiemensCT\Siemens_CT_constancy_VB20A_Force.pdf"
#res = read_Siemens_CT_QC(get_pdf_txt(path))
#res = read_Siemens_PET_dailyQC_xml(get_xml_tree(path))