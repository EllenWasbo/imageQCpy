#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculation processes for the different tests.

@author: Ellen Wasbø
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
import scipy as sp
import skimage

from PyQt5.QtWidgets import qApp

# imageQC block start
import imageQC.scripts.dcm as dcm
from imageQC.scripts.calculate_roi import get_rois
import imageQC.scripts.mini_methods_format as mmf
import imageQC.scripts.mini_methods as mm
import imageQC.scripts.mini_methods_calculate as mmcalc
from imageQC.config.iQCconstants import HEADERS, HEADERS_SUP, QUICKTEST_OPTIONS
import imageQC.config.config_classes as cfc
# imageQC block end


@dataclass
class Results:
    """Class holding results."""

    errmsg: list = field(default_factory=list)
    headers: list = field(default_factory=list)
    headers_sup: list = field(default_factory=list)
    values: list = field(default_factory=list)
    values_sup: list = field(default_factory=list)
    values_info: str = ''
    values_sup_info: str = ''
    details_dict: dict = field(default_factory=dict)
    alternative: int = 0
    pr_image: bool = True


def extract_values(values, columns=[], calculation='='):
    """Extract values as defined in QuickTestOutputSub.

    Parameters
    ----------
    values : list
        list of values (or if per_group nested list of values)
    columns : list of int
        as defined in QuickTestOutputSub. The default is [] (= all)
    calculation : str, optional
        as defined in CALCULATION_OPTIONS (iQCconstants). The default is = (as is).

    Returns
    -------
    new_values : list
        if calculated singel value in list
        else list or nested list as is for the columns
    """
    new_values = []
    if columns == []:
        columns = list(range(len(values)))
    if len(values) > 0:
        selected_values = []
        allvals = []
        if isinstance(values[0], list):  # per group
            for row in values:
                row_vals = [row[col] for col in columns]
                selected_values.append(row_vals)
                allvals.extend(row_vals)
        else:
            vals = [values[col] for col in columns]
            selected_values = vals
            allvals = vals

        for i, val in enumerate(allvals):
            try:
                val = float(val)
                allvals[i] = val
            except ValueError:
                pass

        is_string = [isinstance(val, str) for val in allvals]

        if calculation == '=' or any(is_string):
            new_values = selected_values
        else:
            allvals = np.array(allvals)
            if calculation == 'min':
                new_values = [float(np.min(allvals))]
            elif calculation == 'max':
                new_values = [float(np.max(allvals))]
            elif calculation == 'mean':
                new_values = [float(np.mean(allvals))]
            elif calculation == 'stdev':
                new_values = [float(np.std(allvals))]
            elif calculation == 'max abs':
                new_values = [float(np.max(np.abs(allvals)))]

    return new_values


def quicktest_output(input_main):
    """Extract results to row or column of strings according to output_temp.

    Parameters
    ----------
    input_main : object
        of class MainWindow or InputMain containing specific attributes

    Returns
    -------
    string_list : list of str
        row or colum of strings to append to file or copy to clipboard
    header_list : list of str
    """
    string_list = []
    header_list = []

    if input_main.results != {}:
        n_imgs = len(input_main.imgs)
        # get set names of images or generate default names (indexed names)
        image_names = [f'img{i}' for i in range(n_imgs)]
        if len(input_main.current_quicktest.image_names) > 0:
            set_names = input_main.current_quicktest.image_names
            if any(set_names):
                for i in range(n_imgs):
                    if set_names[i] != '':
                        image_names[i] = set_names[i]
        uniq_group_ids_all = mm.get_uniq_ordered(input_main.current_group_indicators)
        group_names = []
        for i in range(n_imgs):
            group_idx = uniq_group_ids_all.index(input_main.current_group_indicators[i])
            group_names.append(f'group{group_idx}')
        if len(input_main.current_quicktest.group_names) > 0:
            for uniq_id in uniq_group_ids_all:
                idxs_this_group = mm.get_all_matches(
                    input_main.current_group_indicators, uniq_id)
                set_names = [input_main.current_quicktest.group_names[idx]
                             for idx in idxs_this_group]
                if any(set_names):
                    name = next(s for s in set_names if s)
                    for idx in idxs_this_group:
                        group_names[idx] = name

        dm = input_main.current_paramset.output.decimal_mark

        # first all output_sub defined, then defaults where no output defined
        output_all_actual = {}
        for test, sublist in input_main.current_paramset.output.tests.items():
            if test in input_main.results:
                if input_main.results[test] is not None:
                    output_all_actual[test] = input_main.current_paramset.output.tests[
                        test]
        for test in input_main.results:  # add defaults if not defined
            if input_main.results[test] is not None:
                if test not in output_all_actual:
                    output_all_actual[test] = [cfc.QuickTestOutputSub(columns=[])]

        for test in output_all_actual:
            output_subs = output_all_actual[test]

            res_pr_image = input_main.results[test]['pr_image']

            # for each sub-output for current test
            for sub in output_subs:
                values = None
                headers = None
                if sub.alternative > 9:  # supplement table to output
                    values = input_main.results[test]['values_sup']
                    headers = input_main.results[test]['headers_sup']
                else:
                    values = input_main.results[test]['values']
                    headers = input_main.results[test]['headers']
                if values is not None:
                    suffixes = []  # _imgno/name or _groupno/name
                    if sub.per_group:
                        # for each group where len(values[i]) > 0
                        actual_values = []
                        actual_group_names = []
                        actual_image_names = []
                        for rowno, row in enumerate(values):
                            try:
                                if test in input_main.current_quicktest.tests[rowno]:
                                    if row is None:
                                        row = [None] * len(headers)
                                    elif all([
                                            test == 'MTF',
                                            input_main.current_modality == 'CT']):
                                        if not input_main.current_paramset.mtf_cy_pr_mm:
                                            row = list(10 * np.array(row))
                                    actual_values.append(row)
                                    actual_image_names.append(image_names[rowno])
                                    actual_group_names.append(group_names[rowno])
                            except IndexError:
                                pass  # if more images than rows in quicktests values

                        uniq_group_ids = mm.get_uniq_ordered(actual_group_names)
                        for g, group_id in enumerate(uniq_group_ids):
                            values_this = []
                            group_names_this = []
                            image_names_this = []
                            proceed = True
                            while proceed:
                                if group_id in actual_group_names:
                                    idx = actual_group_names.index(group_id)
                                    values_this.append(actual_values[idx])
                                    group_names_this.append(actual_group_names[idx])
                                    image_names_this.append(actual_image_names[idx])
                                    del actual_values[idx]
                                    del actual_group_names[idx]
                                    del actual_image_names[idx]
                                else:
                                    proceed = False
                            out_values = extract_values(
                                values_this,
                                columns=sub.columns,
                                calculation=sub.calculation
                                )
                            string_list.extend(
                                mmf.val_2_str(
                                    out_values, decimal_mark=dm, format_same=False)
                                )
                            if len(out_values) > 1:
                                suffixes.append(image_names_this)
                            else:
                                suffixes.append(group_names_this[0])

                    else:  # each image individually (or 3d res)
                        for r, row in enumerate(values):
                            if row is None:
                                row = [None] * len(headers)
                            if any(row):
                                out_values = extract_values(
                                    row,
                                    columns=sub.columns,
                                    calculation=sub.calculation
                                    )
                                if test == 'DCM':
                                    all_format_strings = (
                                        input_main.current_paramset.dcm_tagpattern.list_format)
                                    if len(sub.columns) == 0:
                                        act_format_strings = all_format_strings
                                    else:
                                        act_format_strings = [
                                            form for idx, form
                                            in enumerate(all_format_strings)
                                            if idx in sub.columns]
                                    string_list.extend(
                                        mmf.val_2_str(
                                            out_values, decimal_mark=dm,
                                            format_same=False,
                                            format_strings=act_format_strings)
                                        )
                                else:
                                    string_list.extend(
                                        mmf.val_2_str(out_values, decimal_mark=dm,
                                                      format_same=False)
                                        )
                                if res_pr_image:
                                    suffixes.append(image_names[r])

                    # output label or table header + image_name or group_name
                    if len(out_values) > 1:  # as is or group/calculation failed
                        if sub.columns == []:
                            headers_this = headers
                        else:
                            headers_this = [
                                header for c, header in enumerate(headers)
                                if c in sub.columns]
                    else:
                        headers_this = [sub.label]
                    all_headers_this = []
                    if suffixes != []:
                        for suffix in suffixes:
                            for header in headers_this:
                                all_headers_this.append(header + '_' + suffix)
                    else:
                        all_headers_this = headers_this
                    header_list.extend(all_headers_this)

    return (string_list, header_list)


def convert_taglists_to_numbers(taglists):
    """Try convert to number.

    Parameters
    ----------
    taglists : list of lists
        extracted dicom parameters

    Returns
    -------
    taglists : list of lists
        DESCRIPTION.

    """
    for r, row in enumerate(taglists):
        for c, val in enumerate(row):
            new_val = val
            try:
                new_val = float(val)
            except ValueError:
                try:
                    new_val = int(val)
                except ValueError:
                    pass
            taglists[r][c] = new_val

    return taglists


def calculate_qc(input_main,
                 wid_auto=None, auto_template_label='', auto_template_session=''):
    """Calculate tests according to current info in main.

    Parameters
    ----------
    input_main : object
        of class MainWindow or MainAuto containing specific attributes
        see try block below
        alter results of input_main
        if data pr image results[test]['pr_image'] = True
        results[test]['details_dict'] is always list of dict [{}] (or None)
            either pr image (results[test]['details_dict'][imgno])
            could also be list of list if different results pr image (MTF point)
            or pr different results when pr_image=False
                e.g. MTF point [{x_data}, {y_data}]
    wid_auto : OpenAutomationDialog, optional
        if not None - draw image on OpenAutomationDialog canvas when test finished
    auto_template_label : str, optional
        AutoTemplate.label if automation - to display if wid_auto
    auto_template_session : str
        String representing session number if more than one image set found
    """
    current_test_before = input_main.current_test
    delta_xya = [0, 0, 0.0]  # only for GUI version
    modality = input_main.current_modality
    errmsgs = []
    if 'MainWindow' in str(type(input_main)):
        input_main.start_wait_cursor()
        input_main.status_bar.showMessage('Calculating...')
        delta_xya = [
            input_main.gui.delta_x,
            input_main.gui.delta_y,
            input_main.gui.delta_a]
    elif wid_auto is not None:
        wid_auto.status_label.showMessage(f'{auto_template_label}: Preparing...')

    paramset = input_main.current_paramset
    if wid_auto is not None:
        wid_auto.wid_image_display.canvas.main.current_paramset = paramset
    img_infos = input_main.imgs
    tag_infos = input_main.tag_infos
    quicktest = input_main.current_quicktest

    proceed = True
    if len(img_infos) == 0:
        proceed = False

    flattened_marked = []
    if proceed:
        # load marked images (if not only DCM test)
        # run 2d tests while loading next image
        # if any 3d tests - keep loaded images to build 3d arrays
        n_img = len(img_infos)
        marked = quicktest.tests
        n_analyse = min([n_img, len(marked)])

        # get list of images to read (either as tags or image)
        flattened_marked = [elem for sublist in marked for elem in sublist]
        if len(flattened_marked) > 0:
            read_tags = [False] * n_img
            extra_tag_pattern = None
            read_image = [False] * n_img
            NM_count = [False] * n_img
            for i in range(n_analyse):
                if len(marked[i]) > 0:
                    if 'DCM' in marked[i]:
                        read_tags[i] = True
                    if marked[i].count('DCM') != len(marked[i]):
                        read_image[i] = True  # not only DCM
                    if input_main.current_modality == 'NM' and 'DCM' in marked[i]:
                        if 'CountsAccumulated' in paramset.dcm_tagpattern.list_tags:
                            read_image[i] = True
                            NM_count[i] = True

            # remove old results
            existing_tests = [*input_main.results]
            for test in existing_tests:
                if test in flattened_marked:
                    input_main.results.pop(test, None)

            # loading file and calculate all 2d tests
            # if any 3d tests - keep loaded images
            # calculate 3d tests
            matrix = [None for i in range(n_img)]
            tag_lists = [[] for i in range(n_img)]
            extra_tag_list = None
            marked_3d = []
            input_main.current_group_indicators = ['' for i in range(n_img)]
            for i in range(n_analyse):
                marked_3d.append([])
                if modality == 'CT':
                    if 'MTF' in marked[i]:
                        if paramset.mtf_type > 0:
                            marked_3d[i].append('MTF')
                            extra_tag_pattern = cfc.TagPatternFormat(
                                list_tags=['ConvolutionKernel'])
                            extra_tag_list = []
                            read_tags[i] = True
                elif 'Uni' in marked[i]:
                    if paramset.uni_sum_first:
                        marked_3d[i].append('Uni')
                elif 'SNI' in marked[i]:
                    if paramset.sni_sum_first:
                        marked_3d[i].append('SNI')
                elif modality == 'SPECT':
                    if 'MTF' in marked[i]:
                        if paramset.mtf_type > 0:
                            marked_3d[i].append('MTF')
                elif modality == 'PET':
                    if 'Cro' in marked[i]:
                        marked_3d[i].append('Cro')
                        extra_tag_pattern = cfc.TagPatternFormat(
                            list_tags=['AcquisitionTime', 'RadionuclideTotalDose',
                                       'RadiopharmaceuticalStartTime', 'Units'])
                        extra_tag_list = []
                        read_tags[i] = True
                elif modality == 'MR':
                    if 'SNR' in marked[i]:
                        marked_3d[i].append('SNR')

            # list of shape + pix for testing if new roi need to be calculated
            xypix = []
            for i, img_info in enumerate(img_infos):
                shape_pix_list = list(img_info.shape) + list(img_info.pix)
                xypix.append(shape_pix_list)

            prev_image_xypix = {}
            prev_roi = {}
            for i in range(n_analyse):
                if 'MainWindow' in str(type(input_main)):
                    input_main.status_bar.showMessage(
                        f'Reading image data {i} of {n_img}')
                elif wid_auto is not None:
                    wid_auto.status_label.showMessage(
                        f'{auto_template_label}: Reading image data {i} of {n_img}')
                # read image or tags as needed
                group_pattern = cfc.TagPatternFormat(list_tags=paramset.output.group_by)
                image = None
                if read_tags[i]:
                    if read_image[i]:
                        tag_patterns = [paramset.dcm_tagpattern, group_pattern]
                        if extra_tag_pattern is not None:
                            tag_patterns.append(extra_tag_pattern)
                        image, tags = dcm.get_img(
                            img_infos[i].filepath,
                            frame_number=img_infos[i].frame_number,
                            tag_patterns=tag_patterns,
                            tag_infos=tag_infos, NM_count=NM_count[i],
                            get_window_level=any(auto_template_label)
                            )
                        if isinstance(tags[0], dict):
                            tag_lists[i] = tags[0]['dummy']
                            if extra_tag_pattern is not None:
                                extra_tag_list.append(tags[2]['dummy'])
                        else:
                            tag_lists[i] = tags[0]
                            if extra_tag_pattern is not None:
                                extra_tag_list.append(tags[2])
                        if extra_tag_pattern is not None:
                            if len(extra_tag_list) == 2:
                                if extra_tag_list[0] != extra_tag_list[1]:
                                    errmsgs.append(
                                        f'Warning image {i}: Differences found in '
                                        'DICOM header '
                                        'that are supposed to be equal for this test:'
                                        f'{extra_tag_pattern.list_tags}')
                                extra_tag_list.pop()  # always chech against first
                    else:
                        tags = dcm.get_tags(
                            img_infos[i].filepath,
                            frame_number=img_infos[i].frame_number,
                            tag_patterns=[paramset.dcm_tagpattern, group_pattern],
                            tag_infos=tag_infos
                            )
                        if isinstance(tags[0], dict):
                            tag_lists[i] = tags[0]['dummy']
                        else:
                            tag_lists[i] = tags[0]
                    input_main.current_group_indicators[i] = '_'.join(tags[1])
                else:
                    if read_image[i]:
                        image, tags = dcm.get_img(
                            img_infos[i].filepath,
                            frame_number=img_infos[i].frame_number,
                            tag_patterns=[group_pattern],
                            tag_infos=tag_infos,
                            get_window_level=any(auto_template_label)
                            )
                        input_main.current_group_indicators[i] = '_'.join(tags[0])
                if wid_auto is not None and image is not None:
                    wid_auto.wid_image_display.canvas.main.active_img = image
                    wid_auto.wid_image_display.canvas.img_draw(
                        auto=True, window_level=tags[-1])

                for test in marked[i]:
                    input_main.current_test = test
                    if test not in ['DCM', '']:
                        if test in marked_3d[i]:
                            if matrix[i] is None:  # only once pr image
                                matrix[i] = image  # save image for later
                        else:
                            calculate_new_roi = True
                            try:
                                if xypix[i] == prev_image_xypix[test]:
                                    calculate_new_roi = False
                            except KeyError:
                                pass

                            prev_image_xypix[test] = xypix[i]

                            if calculate_new_roi:
                                prev_roi[test], errmsg = get_rois(
                                    image, i, input_main)
                                if errmsg is not None:
                                    errmsgs.append(f'{test} get ROI image {i}:')
                                    errmsgs.append(errmsg)
                            if wid_auto is not None:
                                wid_auto.wid_image_display.canvas.main.current_test = (
                                    test)
                                wid_auto.wid_image_display.canvas.main.current_roi = (
                                    prev_roi[test])
                                wid_auto.wid_image_display.canvas.roi_draw()
                                wid_auto.status_label.showMessage((
                                    f'{auto_template_label}: Calculating '
                                    f'img {i+1}/{n_img} ({auto_template_session})'
                                    ))
                                qApp.processEvents()

                            result = calculate_2d(
                                image, prev_roi[test], img_infos[i],
                                modality, paramset, test, delta_xya)
                            if result is not None:
                                if result.errmsg is not None:
                                    intro = f'{test} image {i}:'
                                    if isinstance(result.errmsg, list):
                                        if any(result.errmsg):
                                            errmsgs.append(intro)
                                            errmsgs.extend(result.errmsg)
                                    elif isinstance(result.errmsg, str):
                                        if result.errmsg != '':
                                            errmsgs.append(intro)
                                            errmsgs.append(result.errmsg)

                            if test not in [*input_main.results]:
                                # initiate results
                                input_main.results[test] = {
                                    'headers': result.headers,
                                    'values': [[] for i in range(n_img)],
                                    'alternative': result.alternative,
                                    'headers_sup': result.headers_sup,
                                    'values_sup': [[] for i in range(n_img)],
                                    'details_dict': [{} for i in range(n_img)],
                                    'pr_image': True,
                                    'values_info': result.values_info,
                                    'values_sup_info': result.values_sup_info
                                    }
                            input_main.results[
                                test]['values'][i] = result.values
                            input_main.results[
                                test]['values_sup'][i] = result.values_sup
                            input_main.results[
                                test]['details_dict'][i] = result.details_dict

            # post processing - where values depend on all images
            if modality == 'CT':
                if 'Noi' in flattened_marked:
                    try:
                        noise = [
                            row[1] for row in input_main.results['Noi']['values']]
                        avg_noise = sum(noise)/len(noise)
                        for row in input_main.results['Noi']['values']:
                            # diff from avg (%)
                            row[2] = 100.0 * (row[1] - avg_noise) / avg_noise
                            row[3] = avg_noise
                        # removed if closed images in ui_main update results
                    except IndexError:
                        # average and diff from avg ignored if not all tested
                        for row in input_main.results['Noi']['values']:
                            if row:
                                row[2] = 'NA'
                                row[3] = 'NA'
                if 'Dim' in flattened_marked:
                    if 'MainWindow' in str(type(input_main)):
                        input_main.update_roi()

            if any(marked_3d):
                results_dict = calculate_3d(
                    matrix, marked_3d, input_main, extra_tag_list)
                for test in results_dict:
                    if results_dict[test].errmsg is not None:
                        if isinstance(results_dict[test].errmsg, list):
                            if any(results_dict[test].errmsg):
                                errmsgs.append(f'{test}:')
                                errmsgs.extend(results_dict[test].errmsg)
                        elif isinstance(results_dict[test].errmsg, str):
                            errmsgs.append(results_dict[test].errmsg)

                    input_main.results[test] = asdict(results_dict[test])

            # For test DCM
            if any(read_tags) and 'DCM' in flattened_marked:
                ignore_cols = []
                for idx, val in enumerate(paramset.dcm_tagpattern.list_format):
                    if len(val) > 2:
                        if val[2] == '0':
                            ignore_cols.append(idx)
                tag_lists = mmf.convert_lists_to_numbers(
                    tag_lists, ignore_columns=ignore_cols)
                input_main.results['DCM'] = {
                    'headers': paramset.dcm_tagpattern.list_tags,
                    'values': tag_lists,
                    'values_info': '',
                    'alternative': 0,
                    'pr_image': True
                    }
    msgs = []
    if len(errmsgs) > 0:
        # flatten errmsgs and remove None and '' from errmsgs
        for suberr in errmsgs:
            if suberr is not None:
                if isinstance(suberr, list):
                    msgs.extend([err for err in suberr if err not in [None, '']])
                else:
                    msgs.append(suberr)  # str
    if len(msgs) > 0:
        input_main.display_errmsg(msgs)
    else:
        if 'MainWindow' in str(type(input_main)):
            if input_main.automation_active is False:
                input_main.status_bar.showMessage('Finished', 1000)
            elif wid_auto is not None:
                wid_auto.status_label.showMessage(f'{auto_template_label}: Finished')

    if 'MainWindow' in str(type(input_main)):
        if input_main.automation_active is False:
            if 'Cro' in input_main.results:  # refresh output to widgets
                old_calib = input_main.tab_pet.cro_calibration_factor.value()
                try:
                    corr_calib = input_main.results['Cro']['values'][0][-1]
                    input_main.results['Cro']['values'][0][-1] = corr_calib * old_calib
                except (IndexError, KeyError):
                    pass
            if current_test_before in flattened_marked or len(flattened_marked) == 0:
                set_current_test = current_test_before
            else:
                set_current_test = flattened_marked[0]
            idx_set_test = QUICKTEST_OPTIONS[modality].index(set_current_test)
            widget = input_main.stack_test_tabs.currentWidget()
            widget.setCurrentIndex(idx_set_test)
            input_main.current_test = set_current_test
            input_main.refresh_results_display()
            input_main.stop_wait_cursor()


def calculate_2d(image2d, roi_array, image_info, modality,
                 paramset, test_code, delta_xya):
    """Calculate tests based on 2d-image.

    Parameters
    ----------
    image2d : ndarray(dtype=float, ndim=2)
        image to be analysed. Set to None if only headers of interest.
    roi_array : depending on test
        usually ndarray type bool
        might be list of ndarray if more than one ROI
        other than ndarray type bool e.g. line coordinates for slicethickness
    image_info : DcmInfo
        as defined in scripts/dcm.py
    modality : str
        current modality selected (ignore actual image modality)
    paramset : ParamSetXX
        current_paramset
    test_code : str
        code as defined in iQCconstants.py
    delta_xya : list
        center and angle offset from gui

    Returns
    -------
    result : Results
    """

    def ROI():
        values = []
        if image2d is not None:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            avg_val = np.mean(arr)
            std_val = np.std(arr)
            values = [avg_val, std_val]
        headers = HEADERS[modality][test_code]['alt0']
        res = Results(headers=headers, values=values)
        return res

    def Noi():
        if image2d is not None:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            avg_val = np.mean(arr)
            std_val = np.std(arr)
        if modality == 'CT':
            headers = HEADERS[modality][test_code]['alt0']
            if image2d is not None:
                values = [avg_val, std_val, 0, 0]
        elif modality == 'Xray':
            headers = HEADERS[modality][test_code]['alt0']
            if image2d is not None:
                values = [avg_val, std_val]

        res = Results(headers=headers, values=values)
        return res

    def Hom():
        headers_sup = []
        values_sup = []
        avgs = []
        stds = []
        alt = 0
        if image2d is not None:
            for i in range(np.shape(roi_array)[0]):
                arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[i]))
                avgs.append(np.mean(arr))
                stds.append(np.std(arr))

        if modality == 'CT':
            headers = HEADERS[modality][test_code]['alt0']
            headers_sup = HEADERS_SUP[modality][test_code]['altAll']
            if image2d is not None:
                values = [avgs[1], avgs[2], avgs[3], avgs[4], avgs[0],
                          avgs[1] - avgs[0], avgs[2] - avgs[0],
                          avgs[3] - avgs[0], avgs[4] - avgs[0]]
                values_sup = [stds[1], stds[2], stds[3], stds[4], stds[0]]

        elif modality == 'Xray':
            alt = paramset.hom_tab_alt
            headers = HEADERS[modality][test_code]['alt'+str(alt)]
            if image2d is not None:
                if alt == 0:
                    values = avgs + stds
                else:
                    avg_all = np.sum(avgs) / len(avgs)
                    diffs = [(avg - avg_all) for avg in avgs]
                    if alt == 1:
                        values = avgs + diffs
                    elif alt == 2:
                        diffs_percent = [100. * (diff / avg_all) for diff in diffs]
                        values = avgs + diffs_percent

        elif modality == 'PET':
            headers = HEADERS[modality][test_code]['alt0']
            if image2d is not None:
                avg = sum(avgs) / len(avgs)
                diffs = [100.*(avgs[i] - avg)/avg for i in range(5)]
                values = avgs + diffs

        res = Results(headers=headers, values=values,
                      headers_sup=headers_sup, values_sup=values_sup,
                      alternative=alt)
        return res

    def CTn():
        headers = paramset.ctn_table.materials
        headers_sup = HEADERS_SUP[modality][test_code]['altAll']
        if image2d is not None:
            values = []
            for i in range(len(paramset.ctn_table.materials)):
                arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[i]))
                values.append(np.mean(arr))
            res = sp.stats.linregress(
                values, paramset.ctn_table.relative_mass_density)
            values_sup = [res.rvalue**2, res.intercept, res.slope]
            res = Results(headers=headers, values=values,
                          headers_sup=headers_sup, values_sup=values_sup)
        else:
            res = Results(headers=headers, headers_sup=headers_sup)
        return res

    def HUw():
        headers = HEADERS[modality][test_code]['alt0']
        values = []
        if image2d is not None:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            avg_val = np.mean(arr)
            std_val = np.std(arr)
            values = [avg_val, std_val]
        res = Results(headers=headers, values=values)
        return res

    def Sli():
        if modality == 'CT':
            alt = paramset.sli_type
            headers = HEADERS[modality][test_code]['alt' + str(alt)]
            if image2d is not None:
                lines = roi_array
                values, details_dict, errmsg = calculate_slicethickness(
                    image2d, image_info, paramset, lines, delta_xya)
                if alt == 0:
                    try:
                        values.append(np.mean(values[1:]))
                        values.append(100. * (values[-1] - values[0]) / values[0])
                    except TypeError:
                        values.append(None)
                        values.append(None)
                res = Results(
                    headers=headers, values=values,
                    details_dict=details_dict,
                    alternative=alt, errmsg=errmsg)
            else:
                res = Results(headers=headers, alternative=alt)
        elif modality == 'MR':
            headers = HEADERS[modality][test_code]['alt0']
            # ['Nominal (mm)', 'Slice thickness (mm)', 'Diff (mm)', 'Diff (%)']
            headers_sup = HEADERS_SUP[modality][test_code]['altAll']
            # FWHM1, FWHM2
            if image2d is not None:
                lines = roi_array
                values, details_dict, errmsg = calculate_slicethickness(
                    image2d, image_info, paramset, lines, delta_xya, modality='MR')
                values_sup = values[1:3]
                try:
                    fwhm = values[1:3]
                    slice_thickness = 0.2 * (fwhm[0] * fwhm[1])/(fwhm[0] + fwhm[1])
                    values[1] = slice_thickness
                    values[2] = values[1] - values[0]
                    values.append(100. * values[2] / values[0])
                except (TypeError, IndexError):
                    values = [values[0], None, None, None]
                res = Results(
                    headers=headers, values=values,
                    headers_sup=headers_sup, values_sup=values_sup,
                    details_dict=details_dict,
                    alternative=0, errmsg=errmsg)
            else:
                res = Results(headers=headers)
        else:
            res = Results()

        return res

    def MTF():
        errmsg = []
        if modality in ['CT', 'SPECT']:
            # only bead/point method 2d (alt0)
            #TODO round edge option 2d
            alt = paramset.mtf_type
            headers = HEADERS[modality][test_code]['alt' + str(alt)]
            headers_sup = HEADERS_SUP[modality][test_code]['alt' + str(alt)]
            if image2d is None:
                res = Results(headers=headers, headers_sup=headers_sup)
            else:
                rows = np.max(roi_array[0], axis=1)
                cols = np.max(roi_array[0], axis=0)
                sub = image2d[rows][:, cols]
                if roi_array[1].any():
                    arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[1]))
                    background = np.mean(arr)
                else:
                    background = 0
                    errmsg = ['Warning: width of background too narrow.'
                              ' Background not corrected.']
                details, errmsg_calc = calculate_MTF_point(
                    sub - background, image_info, paramset)
                errmsg.extend(errmsg_calc)
                if len(details) > 1:
                    details[0]['matrix'] = sub
                    prefix = 'g' if paramset.mtf_gaussian else 'd'
                    values = (
                        details[0][prefix + 'MTF_details']['values']
                        + details[1][prefix + 'MTF_details']['values']
                        )
                    values_sup = (
                        list(details[0]['gMTF_details']['LSF_fit_params'])
                        + list(details[1]['gMTF_details']['LSF_fit_params'])
                        )
                    res = Results(
                        headers=headers, values=values,
                        headers_sup=headers_sup, values_sup=values_sup,
                        details_dict=details, alternative=alt, errmsg=errmsg)
                else:
                    res = Results(
                        headers=headers, headers_sup=headers_sup, errmsg=errmsg)

        elif modality in ['Xray', 'MR']:
            headers = HEADERS[modality][test_code]['alt0']
            headers_sup = HEADERS_SUP[modality][test_code]['alt0']
            if image2d is None:
                res = Results(headers=headers, headers_sup=headers_sup)
            else:
                if roi_array is None:
                    res = Results(
                        headers=headers, headers_sup=headers_sup,
                        errmsg='Failed finding ROI')
                else:
                    sub = []
                    if isinstance(roi_array, list):  # auto center
                        for roi in roi_array:
                            rows = np.max(roi, axis=1)
                            cols = np.max(roi, axis=0)
                            sub.append(image2d[rows][:, cols])
                        if len(sub) == 2 or len(sub) == 5:
                            sub.pop()
                    else:
                        rows = np.max(roi_array, axis=1)
                        cols = np.max(roi_array, axis=0)
                        sub.append(image2d[rows][:, cols])

                    details, errmsg = calculate_MTF_2d_line_edge(
                        sub, image_info.pix[0], paramset, mode='edge')
                    prefix = 'g' if paramset.mtf_gaussian else 'd'
                    try:
                        values = details[prefix + 'MTF_details']['values']
                        values_sup = details['gMTF_details']['LSF_fit_params']
                        res = Results(
                            headers=headers, values=values,
                            headers_sup=headers_sup, values_sup=values_sup,
                            details_dict=details, errmsg=errmsg)
                    except (KeyError, TypeError):
                        res = Results(
                            headers=headers, headers_sup=headers_sup,
                            errmsg=errmsg)

        elif modality == 'NM':
            alt = paramset.mtf_type
            headers = HEADERS[modality][test_code]['alt' + str(alt)]
            headers_sup = HEADERS_SUP[modality][test_code]['alt' + str(alt)]
            if image2d is None:
                res = Results(headers=headers, headers_sup=headers_sup)
            else:
                sub = []
                if isinstance(roi_array, list):  # auto center
                    for roi in roi_array:
                        rows = np.max(roi, axis=1)
                        cols = np.max(roi, axis=0)
                        sub.append(image2d[rows][:, cols])
                else:
                    rows = np.max(roi_array, axis=1)
                    cols = np.max(roi_array, axis=0)
                    sub.append(image2d[rows][:, cols])

                if paramset.mtf_type == 0:
                    details, errmsg = calculate_MTF_point(
                        sub[0], image_info, paramset)
                    details[0]['matrix'] = sub[0]
                elif paramset.mtf_type == 1:
                    details, errmsg = calculate_MTF_2d_line_edge(
                        sub, image_info.pix[0], paramset, mode='line')
                else:  # paramset.mtf_type == 2
                    details, errmsg = calculate_MTF_2d_line_edge(
                        sub, image_info.pix[0], paramset, mode='line', pr_roi=True)
                prefix = 'g' if paramset.mtf_gaussian else 'd'
                try:
                    if isinstance(details, list):  # paramset.mtf_type 0 or 2
                        values = []
                        values_sup = []
                        for s_idx in [0, 1]:
                            values.extend(
                                details[s_idx][prefix + 'MTF_details']['values'])
                            values_sup.extend(
                                list(details[s_idx]['gMTF_details']['LSF_fit_params']))
                    else:
                        values = details[prefix + 'MTF_details']['values']
                        values_sup = list(details['gMTF_details']['LSF_fit_params'])
                    res = Results(
                        headers=headers, values=values,
                        headers_sup=headers_sup, values_sup=values_sup,
                        details_dict=details, errmsg=errmsg)
                except KeyError:
                    res = Results(
                        headers=headers, headers_sup=headers_sup,
                        errmsg=errmsg)
                except IndexError:
                    res = Results(
                        headers=headers, headers_sup=headers_sup,
                        errmsg='Failed calculating MTF')
                    # TODO better handle this, seen when NM point input, calc line
        return res

    def Rin():
        headers = HEADERS[modality][test_code]['alt0']
        if image2d is None:
            res = Results(headers=headers)
        else:
            details_dict = {}
            if paramset.rin_sigma_image > 0:
                image_filt = sp.ndimage.gaussian_filter(
                    image2d, sigma=paramset.rin_sigma_image / image_info.pix[0])
            else:
                image_filt = image2d
            arr = np.ma.masked_array(image_filt, mask=np.invert(roi_array[1]))
            arr2 = np.ma.masked_array(arr, mask=roi_array[0])
            details_dict['processed_image'] = arr2
            start_dist = max(4*image_info.pix[0], paramset.rin_range_start)
            radial_profile_x, radial_profile = mmcalc.get_radial_profile(
                image_filt, pix=image_info.pix[0], start_dist=start_dist,
                stop_dist=paramset.rin_range_stop, step_size=image_info.pix[0])

            details_dict['radial_profile_x'] = radial_profile_x
            details_dict['radial_profile'] = radial_profile
            if paramset.rin_sigma_profile > 0:
                radial_profile = sp.ndimage.gaussian_filter(
                    radial_profile, sigma=paramset.rin_sigma_profile)
                details_dict['radial_profile_smoothed'] = radial_profile

            if paramset.rin_subtract_trend:
                res = sp.stats.linregress(radial_profile_x, radial_profile)
                yfit = radial_profile_x * res.slope + res.intercept
                details_dict['radial_profile_trend'] = yfit
                radial_profile = np.subtract(radial_profile, yfit)
            else:  # subtract mean
                mean_profile = np.mean(radial_profile)
                details_dict['mean_profile'] = mean_profile
                radial_profile = radial_profile - mean_profile

            values = [np.min(radial_profile), np.max(radial_profile)]
            res = Results(headers=headers, values=values, details_dict=details_dict)

        return res

    def Dim():
        headers = HEADERS[modality][test_code]['alt0']
        headers_sup = HEADERS_SUP[modality][test_code]['alt0']
        if image2d is None:
            res = Results(headers=headers, headers_sup=headers_sup)
        else:
            errmsg = ''
            details_dict = {}
            dx, dy = roi_array[-1]
            centers_x = []
            centers_y = []
            for i in range(4):
                rows = np.max(roi_array[i], axis=1)
                cols = np.max(roi_array[i], axis=0)
                sub = image2d[rows][:, cols]
                prof_y = np.sum(sub, axis=1)
                prof_x = np.sum(sub, axis=0)
                width_x, center_x = mmcalc.get_width_center_at_threshold(
                    prof_x, 0.5 * (max(prof_x) + min(prof_x)))
                width_y, center_y = mmcalc.get_width_center_at_threshold(
                    prof_y, 0.5 * (max(prof_y) + min(prof_y)))
                centers_x.append(center_x - prof_x.size / 2 + dx[i])
                centers_y.append(center_y - prof_y.size / 2 + dy[i])
            if all(centers_x) and all(centers_y):
                pix = image_info.pix[0]
                diffs = []
                dds = []
                dist = 50.
                diag_dist = np.sqrt(2*50**2)
                for i in range(4):
                    diff = pix * np.sqrt(
                        (centers_x[i] - centers_x[-i-1])**2
                        + (centers_y[i] - centers_y[-i-1])**2
                        )
                    diffs.append(diff)
                    dds.append(diff - dist)
                d1 = pix * np.sqrt(
                    (centers_x[2] - centers_x[0])**2
                    + (centers_y[2] - centers_y[0])**2
                    )
                d2 = pix * np.sqrt(
                    (centers_x[3] - centers_x[1])**2
                    + (centers_y[3] - centers_y[1])**2
                    )
                values = [diffs[1], diffs[3], diffs[0], diffs[2], d1, d2]
                values_sup = [dds[1], dds[3], dds[0], dds[2],
                              d1 - diag_dist, d2 - diag_dist]
                details_dict = {'centers_x': centers_x, 'centers_y': centers_y}
            else:
                errmsg = 'Could not find center of all 4 rods.'

            values_info = 'Distance [mm] between rods'
            values_sup_info = (
                'Difference [mm] from expected distance (50 mm) between rods.')

            res = Results(
                headers=headers, values=values, values_info=values_info,
                headers_sup=headers_sup, values_sup=values_sup,
                values_sup_info=values_sup_info,
                details_dict=details_dict, errmsg=errmsg)

        return res

    def NPS():
        headers = HEADERS[modality][test_code]['alt0']
        if image2d is None:
            res = Results(headers=headers)
        else:
            values, details_dict = calculate_NPS(
                image2d, roi_array, image_info, paramset,
                modality=modality)
            res = Results(headers=headers, values=values,
                          details_dict=details_dict)
        return res

    def STP():
        values = []
        if image2d is not None:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            avg_val = np.mean(arr)
            std_val = np.std(arr)
            values = [None, None, avg_val, std_val]
        headers = HEADERS[modality][test_code]['alt0']
        res = Results(headers=headers, values=values)
        return res

    def Var():
        headers = HEADERS[modality][test_code]['alt0']
        if image2d is None:
            res = Results(headers=headers)
        else:
            # code adapted from:
            # https://www.imageeprocessing.com/2015/10/edge-detection-using-local-variance.html
            roi_size_in_pix = round(paramset.var_roi_size / image_info.pix[0])
            kernel = np.full((roi_size_in_pix, roi_size_in_pix),
                             1./(roi_size_in_pix**2))
            mu = sp.signal.fftconvolve(image2d, kernel, mode='same')
            ii = sp.signal.fftconvolve(image2d ** 2, kernel, mode='same')
            variance_image = ii - mu**2
            rows = np.max(roi_array[0], axis=1)
            cols = np.max(roi_array[0], axis=0)
            sub = variance_image[rows][:, cols]
            values = [np.min(sub), np.max(sub), np.median(sub)]
            details_dict = {'variance_image': sub}
            res = Results(headers=headers, values=values,
                          details_dict=details_dict)
        return res

    def Uni():
        headers = HEADERS[modality][test_code]['alt0']
        headers_sup = HEADERS_SUP[modality][test_code]['altAll']
        if image2d is None:
            res = Results(headers=headers, headers_sup=headers_sup)
        else:
            details_dict = {}
            errmsg = None
            values_sup = [None] * 3
            if paramset.uni_correct:
                lock_z = paramset.uni_radius if paramset.uni_lock_radius else None
                res, errmsg = get_corrections_point_source(
                    image2d, image_info, roi_array[0],
                    fit_x=paramset.uni_correct_pos_x,
                    fit_y=paramset.uni_correct_pos_y,
                    lock_z=lock_z
                    )
                image_input = res['corrected_image']
                values_sup = [res['dx'], res['dy'], res['distance']]
                details_dict = res
            else:
                image_input = image2d
            res = calculate_NM_uniformity(
                image_input, roi_array, image_info.pix[0])
            details_dict['matrix'] = res['matrix']
            details_dict['du_matrix'] = res['du_matrix']
            values = res['values']

            res = Results(
                headers=headers, values=values,
                headers_sup=headers_sup, values_sup=values_sup,
                details_dict=details_dict, errmsg=[errmsg])

        return res

    def SNI():
        headers = HEADERS[modality][test_code]['alt0']
        headers_sup = HEADERS_SUP[modality][test_code]['altAll']
        if image2d is None:
            res = Results(headers=headers, headers_sup=headers_sup)
        else:
            values, values_sup, details_dict, errmsg = calculate_NM_SNI(
                image2d, roi_array, image_info, paramset)

            res = Results(
                headers=headers, values=values,
                headers_sup=headers_sup, values_sup=values_sup,
                details_dict=details_dict, errmsg=errmsg)

        return res

    def Spe():
        headers = HEADERS[modality][test_code]['alt0']
        if image2d is None:
            res = Results(headers=headers)
        else:
            rows = np.max(roi_array, axis=1)
            cols = np.max(roi_array, axis=0)
            sub = image2d[rows][:, cols]
            profile = np.mean(sub, axis=1)
            if paramset.spe_filter_w > 0:
                profile = sp.ndimage.gaussian_filter(
                    profile, sigma=paramset.spe_filter_w)
            profile_pos = np.arange(profile.size) * image_info.pix[0]
            mean_profile = np.mean(profile)
            diff_profile = 100. * (profile - mean_profile) / mean_profile
            details_dict = {'profile': profile, 'profile_pos': profile_pos,
                            'mean_profile': mean_profile, 'diff_profile': diff_profile}
            values = [np.min(diff_profile), np.max(diff_profile)]

            res = Results(
                headers=headers, values=values,
                details_dict=details_dict)

        return res

    def Bar():
        headers = HEADERS[modality][test_code]['alt0']
        # ['MTF @ F1', 'MTF @ F2', 'MTF @ F3', 'MTF @ F4',
        #    'FWHM1', 'FWHM2', 'FWHM3', 'FWHM4']
        if image2d is None:
            res = Results(headers=headers)
        else:
            if len(roi_array) == 4:
                values = []
                for roi_this in roi_array:
                    arr = np.ma.masked_array(image2d, mask=np.invert(roi_this))
                    avg_val = np.mean(arr)
                    var_val = np.var(arr)
                    values.append(np.sqrt(2*(var_val - avg_val)) / avg_val)  # MTF

                const = 4./np.pi * np.sqrt(np.log(2))
                bar_widths = np.array([paramset.bar_width_1, paramset.bar_width_2,
                                       paramset.bar_width_3, paramset.bar_width_4])
                fwhms = const * np.multiply(
                    bar_widths,
                    np.sqrt(np.log(1/np.array(values)))
                    )
                values.extend(list(fwhms))
                res = Results(headers=headers, values=values)
            else:
                res = Results(headers=headers)

        return res

    def PIU():
        headers = HEADERS[modality][test_code]['alt0']
        # ['min', 'max', 'PIU'],
        headers_sup = HEADERS_SUP[modality][test_code]['altAll']
        # ['x min (pix from upper left)', 'y min', 'x max', 'y max']
        if image2d is None:
            res = Results(headers=headers, headers_sup=headers_sup)
        else:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            min_val = np.min(arr)
            max_val = np.max(arr)
            piu = 100.*(1-(max_val-min_val)/(max_val+min_val))
            values = [min_val, max_val, piu]

            min_idx, max_idx = mmcalc.get_min_max_pos_2d(image2d, roi_array)
            values_sup = [min_idx[1], min_idx[0],
                          max_idx[1], max_idx[0]]

            res = Results(
                headers=headers, values=values,
                headers_sup=headers_sup, values_sup=values_sup)

        return res

    def Gho():
        headers = HEADERS[modality][test_code]['alt0']
        # ['Center', 'top', 'bottom', 'left', 'right', 'PSG']
        if image2d is None:
            res = Results(headers=headers)
        else:
            avgs = []
            for i in range(np.shape(roi_array)[0]):
                arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[i]))
                avgs.append(np.mean(arr))
            PSG = abs(100.*0.5*(
                (avgs[3]+avgs[4]) - (avgs[1]+avgs[2])
                )/avgs[0])
            values = avgs
            values.append(PSG)
            res = Results(headers=headers, values=values)

        return res

    def Geo():
        headers = HEADERS[modality][test_code]['alt0']
        # ['width_0', 'width_90', 'width_45', 'width_135',
        #         'GD_0', 'GD_90', 'GD_45', 'GD_135']
        if image2d is None:
            res = Results(headers=headers)
        else:
            mask_outer = round(paramset.geo_mask_outer / image_info.pix[0])
            actual_size = paramset.geo_actual_size
            rotate = [0, 45]
            widths = []
            for r in rotate:
                if r > 0:
                    minval = np.min(image2d)
                    inside = np.full(image_info.shape, False)
                    inside[mask_outer:-mask_outer, mask_outer:-mask_outer] = True
                    image2d[inside == False] = minval
                    im = sp.ndimage.rotate(image2d, r, reshape=False, cval=minval)
                else:
                    im = image2d
                # get maximum profiles x and y
                if mask_outer == 0:
                    prof_y = np.max(im, axis=1)
                    prof_x = np.max(im, axis=0)
                else:
                    prof_y = np.max(
                        im[mask_outer:-mask_outer, mask_outer:-mask_outer], axis=1)
                    prof_x = np.max(
                        im[mask_outer:-mask_outer, mask_outer:-mask_outer], axis=0)
                # get width at halfmax and center for profiles
                width_x, center_x = mmcalc.get_width_center_at_threshold(
                    prof_x, 0.5 * (
                        max(prof_x[prof_x.size//4:-prof_x.size//4])+min(prof_x)),
                    force_above=True)
                width_y, center_y = mmcalc.get_width_center_at_threshold(
                    prof_y, 0.5 * (
                        max(prof_y[prof_y.size//4:-prof_y.size//4])+min(prof_y)),
                    force_above=True)
                if width_x is not None and width_y is not None:
                    widths.append(width_x * image_info.pix[0])
                    widths.append(width_y * image_info.pix[0])
                else:
                    break

            if len(widths) == 4:
                values = widths
                GDs = 100./actual_size * (np.array(widths) - actual_size)
                values.extend(list(GDs))
                res = Results(headers=headers, values=values)
            else:
                res = Results(headers=headers, errmsg='Failed finding object size')

        return res

    try:
        result = locals()[test_code]()
    except KeyError:
        result = None

    return result


def calculate_3d(matrix, marked_3d, input_main, extra_taglists):
    """Calculate tests in 3d mode i.e. tests depending on more than one image.

    Parameters
    ----------
    matrix : np.arrau
        list length = all images, [] if not marked for 3d else 2d image
    marked_3d : list of list of str
        for each image which 3d tests to perform
    input_main : object
        of class MainWindow or InputMain containing specific attributes
        see try block below
        alter results of input_main
    extra_taglists : None or list of strings
        used for test PET Cross calibration

    Returns
    -------
    results : Results
        NB if pr_image is False one row values should be [values]
        and one details_dict should be [details_dict]
    """
    img_infos = input_main.imgs
    modality = input_main.current_modality
    paramset = input_main.current_paramset

    results = {}
    errmsg = []
    flat_marked = [item for sublist in marked_3d for item in sublist]
    all_tests = list(set(flat_marked))

    def MTF(images_to_test):
        if modality in ['CT', 'SPECT']:
            headers = HEADERS[modality][test_code][f'alt{paramset.mtf_type}']
            headers_sup = HEADERS_SUP[modality][test_code][f'alt{paramset.mtf_type}']
            if len(images_to_test) == 0:
                res = Results(headers=headers, headers_sup=headers_sup)
            else:
                details_dict = {}
                sum_image = matrix[images_to_test[0]]
                if len(images_to_test) > 1:
                    for imgNo in images_to_test[1:]:
                        sum_image = np.add(sum_image, matrix[imgNo])
                roi_array, errmsg = get_rois(
                    sum_image, images_to_test[0], input_main)
                if paramset.mtf_type == 1:
                    roi_array_inner = roi_array[0]
                else:
                    roi_array_inner = roi_array
                rows = np.max(roi_array_inner, axis=1)
                cols = np.max(roi_array_inner, axis=0)
                sub = []
                for sli in matrix:
                    if sli is not None:
                        this_sub = sli[rows][:, cols]
                        if paramset.mtf_type == 1:  # background subtract
                            arr = np.ma.masked_array(
                                sli, mask=np.invert(roi_array[1]))
                            this_sub = this_sub - np.mean(arr)
                        sub.append(this_sub)
                    else:
                        sub.append(None)

                if paramset.mtf_type == 1:  # wire
                    if len(images_to_test) > 2:
                        details_dict, errmsg = calculate_MTF_3d_line(
                            sub, roi_array_inner[rows][:, cols], images_to_test,
                            img_infos, paramset)
                    else:
                        errmsg = 'At least 3 images required for MTF wire test in 3d'
                elif paramset.mtf_type == 2:  # circular disc
                    details_dict, errmsg = calculate_MTF_circular_edge(
                        sub, roi_array[rows][:, cols],
                        img_infos[images_to_test[0]].pix[0], paramset)
                    details_dict = [details_dict]

                values = None
                values_sup = None
                prefix = 'g' if paramset.mtf_gaussian else 'd'
                values = details_dict[0][prefix + 'MTF_details']['values']
                values_sup = list(details_dict[0][
                    'gMTF_details']['LSF_fit_params'])
                try:  # x-dir, y-dir
                    values.extend(
                        details_dict[1][prefix + 'MTF_details']['values'])
                    values_sup.extend(list(
                        details_dict[1]['gMTF_details']['LSF_fit_params']))
                except (IndexError, KeyError):
                    pass
                res = Results(headers=headers, values=[values],
                              headers_sup=headers_sup, values_sup=[values_sup],
                              details_dict=details_dict,
                              alternative=paramset.mtf_type, pr_image=False,
                              errmsg=errmsg)
        else:
            res = None

        return res

    def Cro(images_to_test):
        """PET cross calibration."""
        headers = HEADERS[modality][test_code]['alt0']
        if len(images_to_test) == 0:
            res = Results(headers=headers)
        else:
            # ['AcquisitionTime', 'RadionuclideTotalDose',
            # RadiopharmaceuticalStartTime', 'Units']
            if extra_taglists[0][-1] != 'BQML':
                res = Results(
                    headers=headers,
                    errmsg=f'BQML expected as Unit. Found {extra_taglists[0][0]}')
            else:
                errmsgs = []
                fmt = '%H%M%S'
                acq_time = datetime.strptime(
                    extra_taglists[0][0].split('.')[0], fmt)
                inj_time = datetime.strptime(
                    extra_taglists[0][2].split('.')[0], fmt)
                activity_inj = int(extra_taglists[0][1]) * 1/1000000  # Bq to MBq
                time_diff = acq_time - inj_time
                activity_at_scan = activity_inj * np.exp(
                    -np.log(2)*time_diff.seconds/60/109.77)
                actual_concentr = 1000000 * activity_at_scan / paramset.cro_volume

                roi_array, errmsg = get_rois(
                    matrix[images_to_test[0]], images_to_test[0], input_main)
                if errmsg is not None:
                    errmsgs.append(errmsg)
                avgs = []
                for image in matrix:
                    if image is not None:
                        arr = np.ma.masked_array(image, mask=np.invert(roi_array))
                        avgs.append(np.mean(arr))
                zpos = [float(img_info.zpos) for img_info in input_main.imgs]
                zpos = np.array(zpos)[images_to_test]

                sort_idx = np.argsort(zpos)
                zpos = zpos[sort_idx]
                avgs = np.array(avgs)[sort_idx]

                details_dict = {'roi_averages': avgs, 'zpos': zpos}

                if paramset.cro_auto_select_slices:
                    if avgs[0] < max(avgs)/2 and avgs[-1] < max(avgs)/2:
                        width, center = mmcalc.get_width_center_at_threshold(
                            avgs, max(avgs)/2, force_above=True)
                        if center is None or width is None:
                            errmsgs.append(
                                'Auto select slice failed. Could not find FWHM.')
                        else:
                            center_slice = round(center)
                            width = width * paramset.cro_percent_slices/100
                            start_slice = center_slice - round(width/2)
                            stop_slice = center_slice + round(width/2)
                            avgs = avgs[start_slice:stop_slice + 1]
                            zpos = zpos[start_slice:stop_slice + 1]
                    else:
                        errmsgs.append('Outer slices have average above half max. '
                                       'Auto select slice ignored.')
                details_dict['used_roi_averages'] = avgs
                details_dict['used_zpos'] = zpos

                if np.min(avgs) < 0.9*np.max(avgs):
                    errmsgs.append(
                        'Warning: min of rois < 90% of max of rois. '
                        'Consider excluding some slices.')

                read_concentr = np.mean(avgs)
                suv = read_concentr/actual_concentr
                new_calib = 1/suv

                values = [
                    activity_inj,
                    inj_time.strftime('%H:%M:%S'), acq_time.strftime('%H:%M:%S'),
                    activity_at_scan, actual_concentr, read_concentr, suv, new_calib]

                res = Results(headers=headers, values=[values],
                              details_dict=details_dict, pr_image=False,
                              errmsg=errmsgs)
        return res

    def SNR(images_to_test):
        # use two and two images
        values_first_imgs = []
        idxs_first_imgs = []
        headers = HEADERS[modality][test_code]['alt0']
        if len(images_to_test) <= 1:
            res = Results(
                headers=headers,
                errmsg='At least to images required. SNR based on subtraction image.')
        else:
            n_pairs = len(images_to_test)//2
            for i in range(n_pairs):
                idx1 = images_to_test[i*2]
                idx2 = images_to_test[i*2+1]
                image1 = matrix[idx1]
                image2 = matrix[idx2]
                image_subtract = None
                if img_infos[idx1].shape == img_infos[idx2].shape:
                    if img_infos[idx1].pix == img_infos[idx2].pix:
                        roi_array, errmsg = get_rois(
                            image1, images_to_test[i*2], input_main)
                        arr = np.ma.masked_array(
                            image1, mask=np.invert(roi_array))
                        avg1 = np.mean(arr)
                        arr = np.ma.masked_array(
                            image2, mask=np.invert(roi_array))
                        avg2 = np.mean(arr)
                        avg = 0.5*(avg1+avg2)

                        image_subtract = image1-image2
                        arr = np.ma.masked_array(
                            image_subtract, mask=np.invert(roi_array))
                        stdev = np.std(arr)
                        SNR = (avg * np.sqrt(2))/stdev
                        values_first_imgs.append([avg1, avg2, avg, stdev, SNR])
                        idxs_first_imgs.append(idx1)

            values = [[] for i in range(len(img_infos))]
            if len(idxs_first_imgs) > 0:
                for res_i, img_i in enumerate(idxs_first_imgs):
                    values[img_i] = values_first_imgs[res_i]

            res = Results(values=values, headers=headers, errmsg=errmsg, pr_image=True)
        return res

    def sum_matrix(images_to_test):
        sum_matrix = matrix[images_to_test[0]]
        for i in images_to_test[1:]:
            sum_matrix = np.add(sum_matrix, matrix[i])
        return sum_matrix

    def Uni(images_to_test):
        headers = HEADERS[modality][test_code]['alt0']
        errmsgs = []
        if len(images_to_test) == 0:
            res = Results(headers=headers)
        else:
            image_input = sum_matrix(images_to_test)  # always if 3d option
            if paramset.uni_correct:
                errmsgs.append('Point source correction ignored when image sum used.')
            roi_array, errmsg = get_rois(
                image_input, images_to_test[0], input_main)
            res = calculate_NM_uniformity(
                image_input, roi_array, img_infos[0].pix[0])
            details_dict = {
                'sum_image': image_input,
                'matrix': res['matrix'],
                'du_matrix': res['du_matrix']
                }
            values = res['values']

            res = Results(
                headers=headers, values=[values],
                details_dict=[details_dict],
                errmsg=errmsgs, pr_image=False)

        return res

    def SNI(images_to_test):
        headers = HEADERS[modality][test_code]['alt0']
        errmsgs = []
        if len(images_to_test) == 0:
            res = Results(headers=headers)
        else:
            image_input = sum_matrix(images_to_test)  # always if 3d option
            if paramset.sni_correct:
                errmsgs.append('Point source correction ignored when image sum used.')
            roi_array, errmsg = get_rois(
                image_input, images_to_test[0], input_main)
            details_dict = {'sum_image': image_input}
            values, _, details_dict2, _ = calculate_NM_SNI(
                image_input, roi_array, img_infos[0], paramset)
            details_dict.update(details_dict2)

            res = Results(headers=headers, values=[values],
                          details_dict=[details_dict], errmsg=errmsgs, pr_image=False)

        return res

    for test_code in all_tests:
        images_to_test = []
        for i, tests in enumerate(marked_3d):
            if test_code in tests:
                images_to_test.append(i)

        try:
            input_main.current_test = test_code
            result = locals()[test_code](images_to_test)
        except KeyError:
            result = None

        if result is not None:
            results[test_code] = result

    return results


def calculate_slicethickness(
        image, img_info, paramset, lines, delta_xya, modality='CT'):
    """Calculate slice thickness for CT and MR.

    Parameters
    ----------
    image : numpy.ndarray
        2d image
    img_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetCT or ParamsetMR
        settings for the test as defined in config_classes.py
    lines : dict
        h_lines, v_lines as defined in
        calculate_roi.py, get_slicethickness_start_stop
    delta_xya : list
        center and angle offset from gui

    Returns
    -------
    values: list of float
        according to defined headers of selected test type
    details_dict: dict
        dictionary with image profiles that form the basis of the calculations
    errmsgs : list of str
    """
    details_dict = {'profiles': [], 'envelope_profiles': [],
                    'background': [], 'peak': [], 'halfpeak': [],
                    'start_x': [], 'end_x': []}
    errmsgs = []

    sli_signal_low_density = False
    if modality == 'CT':
        if paramset.sli_type == 0:
            details_dict['labels'] = ['upper H1', 'lower H2', 'left V1', 'right V2']
        elif paramset.sli_type == 1:
            details_dict['labels'] = [
                'upper H1', 'lower H2', 'left V1', 'right V2', 'inner V1', 'inner V2']
        elif paramset.sli_type == 2:
            details_dict['labels'] = ['left', 'right']
            sli_signal_low_density = True
    else:
        details_dict['labels'] = ['upper', 'lower']

    pno = 0
    for line in lines['h_lines']:
        profile, errmsg = get_profile_sli(image, paramset, line)
        details_dict['profiles'].append(profile)
        if errmsg is not None:
            errmsgs.append(f'{details_dict["labels"][pno]}: {errmsg}')
        pno += 1
    for line in lines['v_lines']:
        profile, errmsg = get_profile_sli(image, paramset, line, direction='v')
        details_dict['profiles'].append(profile)
        if errmsg is not None:
            errmsgs.append(f'{details_dict["labels"][pno]}: {errmsg}')
        pno += 1
    details_dict['dx'] = img_info.pix[0] * np.cos(np.deg2rad(delta_xya[2]))

    values = [img_info.slice_thickness]
    n_background = round(paramset.sli_background_width / img_info.pix[0])
    for pno, profile in enumerate(details_dict['profiles']):
        slice_thickness = None
        background_start = np.mean(profile[0:n_background])
        background_end = np.mean(profile[-n_background:])
        background = 0.5 * (background_start + background_end)
        details_dict['background'].append(background)
        profile = profile - background
        if sli_signal_low_density:
            peak_value = np.min(profile)
            profile = -1. * profile
            background = -1. * background
        else:
            peak_value = np.max(profile)
        details_dict['peak'].append(peak_value + background)
        halfmax = 0.5 * peak_value
        details_dict['halfpeak'].append(halfmax + background)

        if modality == 'CT':
            if paramset.sli_type == 0:  # wire ramp Catphan
                width, center = mmcalc.get_width_center_at_threshold(
                    profile, halfmax, force_above=True)
                if width is not None:
                    slice_thickness = 0.42 * width * img_info.pix[0]  # 0.42*FWHM
                    if delta_xya[2] != 0:
                        slice_thickness = slice_thickness / np.cos(
                            np.deg2rad(delta_xya[2]))
                    details_dict['start_x'].append(
                        (center - 0.5 * width) * img_info.pix[0])
                    details_dict['end_x'].append(
                        (center + 0.5 * width) * img_info.pix[0])
                else:
                    details_dict['start_x'].append(0)
                    details_dict['end_x'].append(0)
            else:  # beaded ramp, find envelope curve
                # find upper envelope curve
                local_max = (np.diff(np.sign(np.diff(profile))) < 0).nonzero()[0] + 1
                new_x, envelope_y = mmcalc.resample(profile[local_max], local_max, 1,
                                                    n_steps=len(profile))
                width, center = mmcalc.get_width_center_at_threshold(
                    envelope_y, halfmax)
                details_dict['envelope_profiles'].append(envelope_y + background)
                if width is not None:
                    xy_increment = 2  # mm spacing between beads in xy-direction
                    z_increment = 1  # mm spacing between beads in z-direction
                    if paramset.sli_type == 1 and pno > 3:
                        z_increment = 0.25  # inner beaded ramps
                    slice_thickness = (
                        (z_increment/xy_increment) * width * img_info.pix[0])
                    if delta_xya[2] != 0:
                        slice_thickness = (
                            slice_thickness / np.cos(np.deg2rad(delta_xya[2])))
                    details_dict['start_x'].append(
                        (center - 0.5 * width) * img_info.pix[0])
                    details_dict['end_x'].append(
                        (center + 0.5 * width) * img_info.pix[0])
                else:
                    details_dict['start_x'].append(0)
                    details_dict['end_x'].append(0)

            if slice_thickness is None:
                errmsgs.append(
                    f'{details_dict["labels"][pno]}: failed finding slicethickness')

            values.append(slice_thickness)

        else:  # MR
            width, center = mmcalc.get_width_center_at_threshold(
                profile, halfmax, force_above=True)
            if width is not None:
                fwhm_this = width * img_info.pix[0]
                if delta_xya[2] != 0:
                    fwhm_this = fwhm_this / np.cos(np.deg2rad(delta_xya[2]))
                values.append(fwhm_this)
                details_dict['start_x'].append(
                    (center - 0.5 * width) * img_info.pix[0])
                details_dict['end_x'].append(
                    (center + 0.5 * width) * img_info.pix[0])
            else:
                details_dict['start_x'].append(0)
                details_dict['end_x'].append(0)
                errmsgs.append(
                    f'{details_dict["labels"][pno]}: failed finding slicethickness')

    return (values, details_dict, errmsgs)


def get_profile_sli(image, paramset, line, direction='h'):
    """Extract (averaged) profile from image based on xy coordinates.

    Parameters
    ----------
    image : np.2darray.
    paramset : ParamSetXX
        depending on modality - defined in config_classes
    line : tuple of float
        row0, col0, row1, col1
    direction : str, optional
        define direction for averaging if specified in paramset. The default is 'h'.

    Returns
    -------
    profile : np.1darray
    errmsg : str
    """
    profile = None
    errmsg = None
    n_search = round(paramset.sli_search_width)
    n_avg = paramset.sli_average_width
    if hasattr(paramset, 'sli_type'):
        sli_signal_low_density = True if paramset.sli_type == 2 else False
    else:
        sli_signal_low_density = False

    r0, c0, r1, c1 = line
    if n_search > 0:
        profile_sums = []
        profiles = []
        for i in range(-n_search, n_search + 1):
            if direction == 'h':
                rr, cc = skimage.draw.line(r0+i, c0, r1+i, c1)
            else:
                rr, cc = skimage.draw.line(r0, c0+i, r1, c1+i)
            profiles.append(image[rr, cc])
            profile_sums.append(np.sum(image[rr, cc]))
        if sli_signal_low_density:
            max_sum = np.min(profile_sums)
        else:
            max_sum = np.max(profile_sums)
        max_i = np.where(profile_sums == max_sum)
        max_i = max_i[0][0]

        profile = profiles[max_i]
        if n_avg > 0:
            if max_i-n_avg > 0 and max_i+n_avg+1 < len(profile_sums):
                profile = np.mean(profiles[max_i-n_avg:max_i+n_avg+1],
                                  axis=0)
            else:
                errmsg = 'Profile too close to border to be averaged.'
    else:
        rr, cc = skimage.draw.line(r0, c0, r1, c1)
        profile = image[rr, cc]

    return (profile, errmsg)


def calculate_MTF_point(matrix, img_info, paramset):
    """Calculate MTF from point source.

    Based on J Appl Clin Med Phys, Vol 14. No4, 2013, pp216..

    Parameters
    ----------
    matrix : numpy.ndarray
        part of image limited to the point source, background-corrected
    img_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetXX
        depending on modality

    Returns
    -------
    details : list of dict
        [results in x direction, y direction]
        center: float
            center in roi_array current direction
        matrix: np.array 2d
            image within roi
        LSF_x: np.array
            x-axis  - positions for LSF
        LSF: np.array
        dMTF_details: dict
            as returned by mmcalc.get_MTF_discrete
            + values [MTF 50%, 10%, 2%]
        gMTF_details: dict
            as returned by mmcalc.get_MTF_gauss
            + values [MTF 50%, 10%, 2%]
    errmsgs : list of str
    """
    details = []
    errmsgs = []
    for ax in [0, 1]:
        details_dict = {}
        profile = np.sum(matrix, axis=ax)
        width, center = mmcalc.get_width_center_at_threshold(
            profile, np.max(profile)/2)
        if center is None:
            errmsgs.append('Could not find center of point in ROI.')
        else:
            pos = (np.arange(len(profile)) - center) * img_info.pix[0]

            # modality specific settings
            fade_lsf_fwhm = None
            try:
                cut_lsf = paramset.mtf_cut_lsf
                cut_lsf_fwhm = paramset.mtf_cut_lsf_w
                try:
                    fade_lsf_fwhm = paramset.mtf_cut_lsf_w_fade
                except AttributeError:
                    pass
            except AttributeError:
                cut_lsf = False
                cut_lsf_fwhm = None

            cw = 0
            cwf = 0
            if cut_lsf:
                profile, cw, cwf = mmcalc.cut_and_fade_LSF(
                    profile, center=center, fwhm=width,
                    cut_width=cut_lsf_fwhm, fade_width=fade_lsf_fwhm)

            details_dict['center'] = center
            details_dict['LSF_x'] = pos
            details_dict['LSF'] = profile

            # Discrete MTF
            res = mmcalc.get_MTF_discrete(profile, dx=img_info.pix[0])
            res['cut_width'] = cw * img_info.pix[0]
            res['cut_width_fade'] = cwf * img_info.pix[0]
            if isinstance(paramset, cfc.ParamSetCT):
                res['values'] = mmcalc.get_curve_values(
                    res['MTF_freq'], res['MTF'], [0.5, 0.1, 0.02])
            else:  # NM or SPECT
                fwtm, _ = mmcalc.get_width_center_at_threshold(
                    profile, np.max(profile)/10)
                if width is not None:
                    width = width*img_info.pix[0]
                if fwtm is not None:
                    fwtm = fwtm*img_info.pix[0]
                res['values'] = [width, fwtm]
            details_dict['dMTF_details'] = res

            # Gaussian MTF
            if isinstance(paramset, cfc.ParamSetCT):
                gaussfit = 'double'
            else:  # NM or SPECT
                gaussfit = 'single'
            res, err = mmcalc.get_MTF_gauss(
                profile, dx=img_info.pix[0], gaussfit=gaussfit)
            if err is not None:
                errmsgs.append(err)
            else:
                if isinstance(paramset, cfc.ParamSetCT):
                    res['values'] = mmcalc.get_curve_values(
                        res['MTF_freq'], res['MTF'], [0.5, 0.1, 0.02])
                else:  # NM or SPECT
                    profile = res['LSF_fit']
                    fwhm, _ = mmcalc.get_width_center_at_threshold(
                        profile, np.max(profile)/2)
                    fwtm, _ = mmcalc.get_width_center_at_threshold(
                        profile, np.max(profile)/10)
                    if fwhm is not None:
                        fwhm = fwhm*img_info.pix[0]
                    if fwtm is not None:
                        fwtm = fwtm*img_info.pix[0]
                    res['values'] = [fwhm, fwtm]
                details_dict['gMTF_details'] = res

                details.append(details_dict)

    return (details, errmsgs)


def calculate_MTF_3d_line(matrix, roi, images_to_test, image_infos, paramset):
    """Calculate MTF from line ~normal to slices.

    Parameters
    ----------
    matrix : numpy.2darray or list of numpy.array
        list of 2d part of slice limited to disc (or None if ignored slice)
    roi : numpy.2darray of bool
        2d part of larger roi
    images_to_test : list of int
        image numbers to test
    image_infos : list of DcmInfo
        DcmInfo as defined in scripts/dcm.py
    paramset : ParamSetXX
        depending on modality

    Returns
    -------
    details_dict: list of dict
        x_dir, y_dir, common_details
    errmsg : list of str
    """
    details_dict = []
    common_details_dict = {}
    errmsg = []
    matrix = [sli for sli in matrix if sli is not None]
    zpos = np.array([image_infos[sli].zpos for sli in images_to_test])
    max_values = np.array([np.max(sli) for sli in matrix])
    common_details_dict['max_roi_marked_images'] = max_values
    common_details_dict['zpos_marked_images'] = zpos

    ignore_slices = []
    try:  # ignore slices with max outside tolerance
        tolerance = paramset.mtf_line_tolerance
        sort_idxs = np.argsort(max_values)
        max_3highest = np.mean(max_values[sort_idxs[-4:]])
        diff = 100/max_3highest * (np.array(max_values) - max_3highest)
        idxs = np.where(np.abs(diff) > tolerance)
        ignore_slices = list(idxs[0])
    except AttributeError:
        pass

    proceed = True
    zpos_used = None
    if len(ignore_slices) > 0:
        if len(matrix) - len(ignore_slices) < 3:
            errmsg.append(
                'Need at least 3 valid images for this test. Found less.')
            proceed = False
        else:
            zpos_copy = np.copy(zpos)
            zpos = np.delete(zpos, ignore_slices)
            max_values_copy = np.copy(max_values)
            ignore_slices.reverse()
            for i in ignore_slices:
                del matrix[i]
                max_values_copy[i] = np.NaN
                zpos_copy[i] = np.NaN
            common_details_dict['max_roi_used'] = max_values_copy
            common_details_dict['zpos_used'] = zpos_copy
            zpos_used = zpos_copy

    if proceed:
        pix = image_infos[images_to_test[0]].pix[0]

        for i in [0, 1]:
            axis = 2 if i == 0 else 1
            matrix_xz = np.sum(matrix, axis=axis)

            details_dict_this, errmsg_this = calculate_MTF_2d_line_edge(
                matrix_xz, pix, paramset, mode='line',
                vertical_positions_mm=zpos_used)
            details_dict.append(details_dict_this)
            errmsg.append(errmsg_this)

    details_dict.append(common_details_dict)

    return (details_dict, errmsg)


def calculate_MTF_2d_line_edge(matrix, pix, paramset, mode='edge',
                               pr_roi=False, vertical_positions_mm=None):
    """Calculate MTF from straight line.

    Parameters
    ----------
    matrix : numpy.2darray or list of numpy.2darray
        image within roi(s) - result will be combined from all rois
    pix : float
        pixelsize in mm
    paramset : ParamSetXX
        depending on modality
    mode : str, optional
        'edge' or 'line'. Default is 'edge'
    pr_roi : bool, Optional
        MTF pr roi in matrix or average. Default is False.
    vertical_positions_mm : list of float, optional
        if vertical pix size != pix and possiby irregular. Default is None
        list of y pix positions in mm

    Returns
    -------
    details_dict: dict
    errmsg: list
    """
    edge = True if mode == 'edge' else False
    details_dict = []  # {} if only one
    details_dicts_edge = []  # one for each roi (edge=line)
    ESF_all = []  # if edge
    LSF_all = []
    LSF_no_filt_all = []
    ESF_x = None  # if edge
    LSF_x = None
    errmsg = []
    step_size = 0.1 * pix
    if not isinstance(matrix, list):
        matrix = [matrix]

    def ensure_vertical(sub):
        """Rotate matrix if edge/line not in y direction."""
        if edge:
            x1 = np.mean(sub[:, :2])
            x2 = np.mean(sub[:, -2:])
            diff_x = abs(x1 - x2)
            y1 = np.mean(sub[:2, :])
            y2 = np.mean(sub[-2:, :])
            diff_y = abs(y1 - y2)
            if diff_x < diff_y:
                sub = np.rot90(sub)
                halfmax = 0.5 * (y1 + y2)
            else:
                halfmax = 0.5 * (x1 + x2)
        else:  # line
            prof_y = np.sum(sub, axis=1)
            prof_x = np.sum(sub, axis=0)
            if np.max(prof_y) > np.max(prof_x):
                sub = np.rot90(sub)
            halfmax = 0.5 * (np.max(sub) + np.min(sub))
        return (sub, halfmax)

    def get_edge_pos(sub, halfmax, txt_roi=''):
        """Find position of edge/line for each row in matrix."""
        edge_pos = []
        x = np.arange(sub.shape[1])
        if edge:
            smoothed = sp.ndimage.gaussian_filter(sub, sigma=3)
            for i in range(smoothed.shape[0]):
                res = mmcalc.get_curve_values(x, smoothed[i, :], [halfmax])
                edge_pos.append(res[0])
        else:
            for i in range(sub.shape[0]):
                _, center = mmcalc.get_width_center_at_threshold(sub[i, :], halfmax)
                edge_pos.append(center)
        proceed = True
        ys = np.arange(sub.shape[0])

        if None in edge_pos:
            n_found = len(edge_pos) - edge_pos.count(None)
            txt_mode = 'Edge' if edge else 'Line'
            if n_found < 0.5 * len(edge_pos):
                proceed = False
                errmsg.append(
                    f'{txt_mode} position found for < 50% of ROI {txt_roi}. '
                    'Test failed.')
            else:
                idx_None = mm.get_all_matches(edge_pos, None)
                edge_pos = [e for i, e in enumerate(edge_pos) if i not in idx_None]
                ys = [y for i, y in enumerate(list(ys)) if i not in idx_None]
            if errmsg == []:
                errmsg.append(
                    '{txt_mode} position not found for full ROI. Parts of ROI ignored.')
        return (edge_pos, x, ys, proceed)

    for m in range(len(matrix)):
        sub, halfmax = ensure_vertical(matrix[m])
        txt_roi = f'{m}' if len(matrix) > 1 else ''
        edge_pos, x, ys, proceed = get_edge_pos(sub, halfmax, txt_roi=txt_roi)

        if proceed:
            # linear fit of edge positions
            vertical_positions = None
            if vertical_positions_mm is not None:
                if isinstance(vertical_positions_mm, list):
                    if len(vertical_positions_mm) == sub.shape[0]:
                        y_pos_mm = [vertical_positions_mm[y] for y in ys]
                        ys = 1./pix * np.array(y_pos_mm)  # unit = pix
                        vertical_positions = ys
            res = sp.stats.linregress(ys, edge_pos)  # x = ay + b
            slope = 1./res.slope  # to y = (1/a)x + (-b/a) to avoid error when steep
            intercept = - res.intercept / res.slope
            x_fit = np.array([min(edge_pos), max(edge_pos)])
            y_fit = slope * x_fit + intercept
            angle = np.abs((180/np.pi) * np.arctan(
                (x_fit[1]-x_fit[0])/(y_fit[1]-y_fit[0])
                ))

            # sort pixels by position normal to edge
            dist_map = mmcalc.get_distance_map_edge(
                sub.shape, slope=slope, intercept=intercept,
                vertical_positions=vertical_positions)

            dist_map_flat = dist_map.flatten()
            values_flat = sub.flatten()
            sort_idxs = np.argsort(dist_map_flat)
            dists = dist_map_flat[sort_idxs]
            sorted_values = values_flat[sort_idxs]
            dists = pix * dists

            details_dicts_edge.append({
                'edge_pos': edge_pos, 'edge_row': ys,
                'edge_fit_x': x_fit, 'edge_fit_y': y_fit,
                'edge_r2': res.rvalue**2, 'angle': angle,
                'sorted_pixels_x': dists,
                'sorted_pixels': sorted_values,
                'edge_or_line': mode
                })

            new_x, new_y = mmcalc.resample_by_binning(
                input_y=sorted_values, input_x=dists,
                step=step_size,
                first_step=-sub.shape[1]/2 * pix,
                n_steps=10*sub.shape[1])

            if edge:
                if ESF_x is None:
                    ESF_x = new_x
                ESF_all.append(new_y)
            else:
                if LSF_x is None:
                    LSF_x = new_x
                LSF_all.append(new_y)

    sigma_f = 0
    if ESF_all:
        sigma_f = 3.  # if sigma_f=5 , FWHM ~9 newpix = ~ 1 original pix
        for ESF in ESF_all:
            LSF, LSF_no_filt, _ = mmcalc.ESF_to_LSF(ESF, prefilter_sigma=sigma_f)
            LSF = LSF/np.max(LSF)
            LSF_no_filt = LSF_no_filt/np.max(LSF_no_filt)
            LSF_all.append(LSF)
            LSF_no_filt_all.append(LSF_no_filt)

    if LSF_all:
        if len(LSF_all) > 1 and pr_roi is False:
            LSF = [np.mean(np.array(LSF_all), axis=0)]
            if LSF_no_filt_all:
                LSF_no_filt = [np.mean(np.array(LSF_no_filt_all), axis=0)]
        else:
            LSF = LSF_all
            if LSF_no_filt_all:
                LSF_no_filt = LSF_no_filt_all
        if not LSF_no_filt_all:
            LSF_no_filt = LSF

        cw = 0
        try:
            cut_lsf = paramset.mtf_cut_lsf
            cut_lsf_fwhm = paramset.mtf_cut_lsf_w
        except AttributeError:
            cut_lsf = False
            cut_lsf_fwhm = None

        if isinstance(paramset, cfc.ParamSetXray):
            gaussfit_type = 'double_both_positive'
            lp_vals = [0.5, 1, 1.5, 2, 2.5]
            mtf_vals = [0.5]
        elif isinstance(paramset, cfc.ParamSetCT):  # wire 3d
            gaussfit_type = 'double'
            lp_vals = None
            mtf_vals = [0.5, 0.1, 0.02]
        else:  # MR, NM, SPECT 3d linesource
            gaussfit_type = 'single'
            lp_vals = None
            mtf_vals = [0.5, 0.1, 0.02]

        for i in range(len(LSF)):
            width, center = mmcalc.get_width_center_at_threshold(
                LSF[i], np.max(LSF[i])/2)
            if width is not None:
                # Calculate gaussian and discrete MTF
                if cut_lsf:
                    LSF_no_filt_this, cw, _ = mmcalc.cut_and_fade_LSF(
                        LSF_no_filt[i], center=center, fwhm=width,
                        cut_width=cut_lsf_fwhm)
                else:
                    LSF_no_filt_this = LSF_no_filt[i]
                dMTF_details = mmcalc.get_MTF_discrete(LSF_no_filt_this, dx=step_size)
                dMTF_details['cut_width'] = cw * step_size

                LSF_x = step_size * (np.arange(LSF[i].size) - center)
                gMTF_details, err = mmcalc.get_MTF_gauss(
                    LSF[i], dx=step_size, prefilter_sigma=sigma_f*step_size,
                    gaussfit=gaussfit_type)

                if err is not None:
                    errmsg.append(err)
                else:
                    if (
                            isinstance(paramset, cfc.ParamSetNM)
                            or isinstance(paramset, cfc.ParamSetSPECT)):
                        fwhm, _ = mmcalc.get_width_center_at_threshold(
                            LSF[i], np.max(LSF[i])/2)
                        fwtm, _ = mmcalc.get_width_center_at_threshold(
                            LSF[i], np.max(LSF[i])/10)
                        dMTF_details['values'] = [step_size*fwhm, step_size*fwtm]
                        fwhm, _ = mmcalc.get_width_center_at_threshold(
                            gMTF_details['LSF_fit'], np.max(gMTF_details['LSF_fit'])/2)
                        fwtm, _ = mmcalc.get_width_center_at_threshold(
                            gMTF_details['LSF_fit'], np.max(gMTF_details['LSF_fit'])/10)
                        gMTF_details['values'] = [step_size*fwhm, step_size*fwtm]
                    else:
                        if lp_vals is not None:
                            gMTF_details['values'] = mmcalc.get_curve_values(
                                    gMTF_details['MTF'], gMTF_details['MTF_freq'],
                                    lp_vals)
                            dMTF_details['values'] = mmcalc.get_curve_values(
                                    dMTF_details['MTF'], dMTF_details['MTF_freq'],
                                    lp_vals,
                                    force_first_below=True)
                        gvals = mmcalc.get_curve_values(
                                gMTF_details['MTF_freq'], gMTF_details['MTF'], mtf_vals)
                        dvals = mmcalc.get_curve_values(
                                dMTF_details['MTF_freq'], dMTF_details['MTF'], mtf_vals,
                                force_first_below=True)
                        if lp_vals is not None:
                            gMTF_details['values'].extend(gvals)
                            dMTF_details['values'].extend(dvals)
                        else:
                            gMTF_details['values'] = gvals
                            dMTF_details['values'] = dvals

                details_dict.append({
                    'LSF_x': LSF_x, 'LSF': LSF_no_filt_this,
                    'ESF_x': ESF_x, 'ESF': ESF_all,
                    'sigma_prefilter': sigma_f*step_size,
                    'dMTF_details': dMTF_details, 'gMTF_details': gMTF_details,
                    'edge_details': details_dicts_edge})
            else:
                errmsg = f'Could not find {mode}.'

    if len(details_dict) == 1:
        details_dict = details_dict[0]

    return (details_dict, errmsg)


def calculate_MTF_circular_edge(matrix, roi, pix, paramset):
    """Calculate MTF from circular edge.

    Based on Richard et al: Towards task-based assessment of CT performance,
    Med Phys 39(7) 2012

    Parameters
    ----------
    matrix : numpy.2darray or list of numpy.array
        list of 2d part of slice limited to disc (or None if ignored slice)
    roi : numpy.2darray of bool
        2d part of larger roi
    pix : float
        pixelsize in mm
    paramset : ParamSetXX
        depending on modality

    Returns
    -------
    details_dict
    """
    details_dict = {}
    errmsg = []

    try:
        if matrix.ndim == 2:
            matrix = [matrix]
    except AttributeError:
        pass

    # find center of disc with high precision
    center_xy = []
    errtxt = ''
    for slino, sli in enumerate(matrix):
        if sli is not None:
            center_this = mmcalc.center_xy_of_disc(
                sli, roi=roi, mode='max_or_min', sigma=1)
            if None not in center_this:
                center_xy.append(center_this)
            else:
                if errtxt == '':
                    errtxt = str(slino)
                else:
                    errtxt = ', '.join([errtxt, str(slino)])
    if errtxt != '':
        errmsg = f'Could not find center of object for image {errtxt}'
    if len(center_xy) > 0:
        center_x = [vals[0] for vals in center_xy]
        center_y = [vals[1] for vals in center_xy]
        center_xy = [np.mean(np.array(center_x)), np.mean(np.array(center_y))]
        # sort pixel values from center
        dist_map = mmcalc.get_distance_map_point(
            matrix[0].shape,
            center_dx=center_xy[0] - 0.5 * matrix[0].shape[1],
            center_dy=center_xy[1] - 0.5 * matrix[0].shape[0])
        dists_flat = dist_map.flatten()
        sort_idxs = np.argsort(dists_flat)
        dists = dists_flat[sort_idxs]

        dists = pix * dists
        values_all = []
        nsum = 0
        total_imgs = None
        for sli in matrix:
            if sli is not None:
                if total_imgs is None:
                    total_imgs = sli
                else:
                    total_imgs = total_imgs + sli
                values_all.append(sli.flatten()[sort_idxs])
                nsum += 1
        img_avg = 1/nsum * total_imgs
        img_avg_flat = img_avg.flatten()
        values_avg = img_avg_flat[sort_idxs]

        # ignore dists > radius
        radius = 0.5 * matrix[0].shape[0] * pix
        dists_cut = dists[dists < radius]
        values = values_avg[dists < radius]

        step_size = .1 * pix
        new_x, new_y = mmcalc.resample_by_binning(
            input_y=values, input_x=dists_cut, step=step_size)
        sigma_f = 5.
        LSF, LSF_no_filt, ESF_filtered = mmcalc.ESF_to_LSF(
            new_y, prefilter_sigma=sigma_f)

        width, center = mmcalc.get_width_center_at_threshold(LSF, np.max(LSF)/2)
        if width is not None:
            # Discrete MTF
            # modality specific settings
            fade_lsf_fwhm = 0
            cw = 0
            cwf = 0
            try:
                cut_lsf = paramset.mtf_cut_lsf
                cut_lsf_fwhm = paramset.mtf_cut_lsf_w
                try:
                    fade_lsf_fwhm = paramset.mtf_cut_lsf_w_fade
                except AttributeError:
                    pass
            except AttributeError:
                cut_lsf = False
                cut_lsf_fwhm = None
            if cut_lsf:
                LSF_no_filt, cw, cwf = mmcalc.cut_and_fade_LSF(
                    LSF_no_filt, center=center, fwhm=width,
                    cut_width=cut_lsf_fwhm, fade_width=fade_lsf_fwhm)
            dMTF_details = mmcalc.get_MTF_discrete(LSF_no_filt, dx=step_size)
            dMTF_details['cut_width'] = cw * step_size
            dMTF_details['cut_width_fade'] = cwf * step_size

            LSF_x = step_size * (np.arange(LSF.size) - center)

            gMTF_details, err = mmcalc.get_MTF_gauss(
                LSF, dx=step_size, prefilter_sigma=sigma_f*step_size, gaussfit='double')

            if err is not None:
                errmsg.append(err)
            else:
                gMTF_details['values'] = mmcalc.get_curve_values(
                        gMTF_details['MTF_freq'], gMTF_details['MTF'], [0.5, 0.1, 0.02])
            dMTF_details['values'] = mmcalc.get_curve_values(
                    dMTF_details['MTF_freq'], dMTF_details['MTF'], [0.5, 0.1, 0.02],
                    force_first_below=True)

            details_dict = {
                'matrix': matrix, 'center_xy': center_xy,
                'LSF_x': LSF_x, 'LSF': LSF_no_filt,
                'sigma_prefilter': sigma_f*step_size,
                'sorted_pixels_x': dists, 'sorted_pixels': values_all,
                'interpolated_x': new_x, 'interpolated': new_y,
                'presmoothed': ESF_filtered,
                'dMTF_details': dMTF_details, 'gMTF_details': gMTF_details}
        else:
            errmsg = 'Could not find circular edge.'

    return (details_dict, errmsg)


def calculate_NPS(image2d, roi_array, img_info, paramset, modality='CT'):
    """Calculate NPS for each roi in array.

    Parameters
    ----------
    image2d : numpy.ndarray
        input image full resolution
    roi_array : list of numpy.ndarray
    img_info :  DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetCT or ParamsetXray
    modality: str
        'CT' or 'Xray'

    Returns
    -------
    values
    dict
        NPS-array: 2d NPS
        if CT:
        freq: frequency -values power spectrum
        radial_profile: power spectrum
        median_freq: median_frequency of power spectrum
        median_val: value at median frequency
        if Xray:
        freq: frequency values x -dir
        u_profile: power spectrum x-dir
        v_profile: power spectrum y-dir
    """
    def smooth_profile(profile):
        kernel_size = round(paramset.nps_smooth_width / paramset.nps_sampling_frequency)
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(profile, kernel, mode='same')

    values = []
    details_dict = {}
    NPS_total = np.zeros((paramset.nps_roi_size, paramset.nps_roi_size))
    unit = 1/(img_info.pix[0]*paramset.nps_roi_size)

    if modality == 'CT':
        stdev_img = 0
        avg_img = 0
        for roi in roi_array:
            rows = np.max(roi, axis=1)
            cols = np.max(roi, axis=0)
            sub = image2d[rows][:, cols]
            stdev_img += np.std(sub)
            avg_img += np.mean(sub)
            subtract_sub = mmcalc.polyfit_2d(sub, max_order=2)
            sub = sub - subtract_sub
            NPS_this = mmcalc.get_2d_NPS(sub, img_info.pix[0])
            NPS_total = NPS_total + NPS_this
        stdev_img = stdev_img / len(roi_array)
        large_area_signal = avg_img / len(roi_array)
        NPS = (1 / len(roi_array)) * NPS_total

        freq, radial_profile = mmcalc.get_radial_profile(
            NPS, pix=unit, step_size=paramset.nps_sampling_frequency)
        if paramset.nps_smooth_width > 0:
            radial_profile = smooth_profile(radial_profile)
        AUC = np.sum(radial_profile) * paramset.nps_sampling_frequency
        median_frequency, median_val = mmcalc.find_median_spectrum(
            freq, radial_profile)
        values = [median_frequency, AUC, np.sum(NPS)*unit ** 2,
                  large_area_signal, stdev_img]
        details_dict['median_freq'] = median_frequency
        details_dict['median_val'] = median_val

    elif modality == 'Xray':
        large_area_roi = np.array(roi_array)
        # replace large area with trend-subtracted values
        roi = np.sum(large_area_roi, axis=0, dtype=bool)
        rows = np.max(roi, axis=1)
        cols = np.max(roi, axis=0)
        sub = image2d[rows][:, cols]
        large_area_signal = np.mean(sub)
        large_area_stdev = np.std(sub)
        subtract_sub = mmcalc.polyfit_2d(sub, max_order=2)
        sub = sub - subtract_sub
        details_dict['trend_corrected_sub_matrix'] = sub
        roi_flat = roi.flatten()
        img_flat = image2d.flatten()
        img_flat[roi_flat == True] = sub.flatten()
        image2d = np.reshape(img_flat, roi.shape)

        # calculate NPS for each roi, average
        for roi in roi_array:
            rows = np.max(roi, axis=1)
            cols = np.max(roi, axis=0)
            sub = image2d[rows][:, cols]
            NPS_this = mmcalc.get_2d_NPS(sub, img_info.pix[0])
            NPS_total = NPS_total + NPS_this
        NPS = (1 / len(roi_array)) * NPS_total

        # set central axis of NPS array to nan
        line = NPS.shape[0] // 2
        NPS[line] = np.nan
        NPS[:, line] = np.nan

        freq_uv, u_profile, v_profile = mmcalc.get_NPSuv_profile(
            NPS,  nlines=7, exclude_axis=True, pix=img_info.pix[0],
            step_size=paramset.nps_sampling_frequency)
        freq, radial_profile = mmcalc.get_radial_profile(
            NPS, pix=unit, step_size=paramset.nps_sampling_frequency,
            start_dist=3*unit)
        AUC = np.sum(radial_profile) * paramset.nps_sampling_frequency
        if paramset.nps_smooth_width > 0:
            u_profile = smooth_profile(u_profile)
            v_profile = smooth_profile(v_profile)
        AUC_u = np.sum(u_profile) * (freq_uv[1] - freq_uv[0])
        AUC_v = np.sum(v_profile) * (freq_uv[1] - freq_uv[0])
        values = [np.nansum(NPS)*unit ** 2, large_area_signal, large_area_stdev,
                  AUC_u/AUC_v]

        details_dict['freq_uv'] = freq_uv
        details_dict['u_profile'] = u_profile
        details_dict['v_profile'] = v_profile
        details_dict['u_profile_AUC'] = AUC_u
        details_dict['v_profile_AUC'] = AUC_v

        # set central axis of NPS array to zero (for display)
        line = NPS.shape[0] // 2
        NPS[line] = 0
        NPS[:, line] = 0

    details_dict['NPS_array'] = NPS
    details_dict['freq'] = freq
    details_dict['radial_profile'] = radial_profile
    details_dict['radial_profile_AUC'] = AUC
    details_dict['large_area_signal'] = large_area_signal

    return (values, details_dict)


def get_corrections_point_source(
        image2d, img_info, roi_array,
        fit_x=False, fit_y=False, lock_z=None, estimate_noise=False):
    """Estimate xyz of point source and generate correction matrix.

    Parameters
    ----------
    image2d : numpy.ndarray
        input image full resolution
    img_info :  DcmInfo
        as defined in scripts/dcm.py
    roi_array : numpy.ndarray
        roi_array for largest ROI used for the fit (ufov)
    fit_x : bool, optional
        Fit in x direction. The default is False.
    fit_y : bool, optional
        Fit in y direction. The default is False.
    lock_z : float, optional
        Use locked distance to source when fitting. The default is None.
    estimate_noise : bool
        if True estimate poisson noise for fitted image

    Returns
    -------
    dict
        corrected_image : numpy.2darray
        correction_matrix : numpy.2darray
            subtract this matrix to obtain corrected image2d
        fit_matrix : numpy.2darray
            fitted image
        estimated_noise_image : numpy.2darray
            fitted_image with estimated poisson noise
        dx : corrected x position (mm)
        dy : corrected y position (mm)
        distance : corrected distance (mm)
    str
        errormessage
    """
    corrected_image = image2d
    correction_matrix = np.ones(image2d.shape)
    fit_matrix = None
    distance = 0.
    errmsg = None

    # get subarray
    rows = np.max(roi_array, axis=1)
    cols = np.max(roi_array, axis=0)
    image_in_ufov = image2d[rows][:, cols]
    ufov_denoised = sp.ndimage.gaussian_filter(image_in_ufov, sigma=1.)

    # fit center position
    fit = [fit_x, fit_y]
    offset = [0., 0.]
    for a in range(2):
        if fit[a]:
            sum_vector = np.sum(ufov_denoised, axis=a)
            popt = mmcalc.gauss_4param_fit(
                np.arange(len(sum_vector)), sum_vector)
            if popt is not None:
                C, A, x0, sigma = popt
                offset[a] = x0 - 0.5*len(sum_vector)

    dx, dy = offset

    # calculate distances from center in plane
    shape_ufov_y, shape_ufov_x = ufov_denoised.shape
    dists_inplane = mmcalc.get_distance_map_point(
        image2d.shape, center_dx=dx, center_dy=dy)
    dists_inplane = img_info.pix[0] * dists_inplane

    dists_ufov = dists_inplane[rows][:, cols]
    dists_ufov_flat = dists_ufov.flatten()
    values_flat = ufov_denoised.flatten()
    sort_idxs = np.argsort(dists_ufov_flat)
    values = values_flat[sort_idxs]
    dists = dists_ufov_flat[sort_idxs]

    if lock_z is None:
        try:
            nm_radius = img_info.nm_radius
        except AttributeError:
            nm_radius = 400
        lock_radius = False
    else:
        nm_radius = lock_z
        lock_radius = True

    if nm_radius is None:
        errmsg = (
            'Failed fitting matrix to point source. nm_radius is None. Report error.')
    else:
        popt = mmcalc.point_source_func_fit(
            dists, values,
            center_value=np.max(ufov_denoised),
            avg_radius=nm_radius, lock_radius=lock_radius)
        if popt is not None:
            C, distance = popt
            fit = mmcalc.point_source_func(dists_inplane.flatten(), C, distance)

            # estimate noise
            fit_matrix = fit.reshape(image2d.shape)
            rng = np.random.default_rng()
            estimated_noise_image = rng.poisson(fit_matrix)

            # correct input image
            corr_sub = fit - fit[0]
            corr_sub = (1./np.max(fit)) * fit  # matrix to subtract
            corr_sub = 1./corr_sub
            correction_matrix = corr_sub.reshape(image2d.shape)
            corrected_image = image2d * correction_matrix
        else:
            errmsg = 'Failed fitting matrix to point source.'

    return ({'corrected_image': corrected_image,
             'correction_matrix': correction_matrix,
             'fit_matrix': fit_matrix,
             'estimated_noise_image': estimated_noise_image,
             'dx': dx, 'dy': dy, 'distance': distance}, errmsg)


def calculate_NM_uniformity(image2d, roi_array, pix):
    """Calculate uniformity parameters.

    Parameters
    ----------
    image2d : numpy.ndarray
    roi_array : list of numpy.ndarray
        2d mask for ufov [0] and cfov [1]
    pix : float
        pixel size of image2d

    Returns
    -------
    matrix : numpy.ndarray
        downscaled and smoothed image used for calculation of uniformity values
    du_matrix : numpy.ndarray
        calculated du_values maximum of x/y direction
    values : list of float
        ['IU_UFOV %', 'DU_UFOV %', 'IU_CFOV %', 'DU_CFOV %']
    """
    def get_differential_uniformity(image):
        sz_y, sz_x = image.shape
        du_cols = np.zeros(image.shape)
        for x in range(sz_x):
            for y in range(2, sz_y - 2):
                sub = image[y-2:y+2, x]
                max_val = np.max(sub)
                min_val = np.min(sub)
                du_cols[y][x] = 100. * (max_val - min_val) / (max_val + min_val)
        du_rows = np.zeros(image.shape)
        for y in range(sz_y):
            for x in range(2, sz_x - 2):
                sub = image[y, x-2:x+2]
                max_val = np.max(sub)
                min_val = np.min(sub)
                du_rows[y][x] = 100. * (max_val - min_val) / (max_val + min_val)
        du_matrix = np.maximum(du_cols, du_rows)
        return {'du_matrix': du_matrix, 'du': np.max(du_matrix)}

    # resize to 6.4+/-30% mm pixels
    pix_range = [6.4*0.7, 6.4*1.3]
    scale_factors = np.array([2, 4, 8, 16, 32])
    possible_pix = pix*scale_factors
    selected_pix = np.where(possible_pix >= pix_range[0])
    scale_factor = scale_factors[selected_pix[0][0]]
    image64 = skimage.measure.block_reduce(
        image2d, (scale_factor, scale_factor), np.mean)  # scale down to 6.4mm/pix
    roi64 = []
    for i in range(2):
        reduced_roi = skimage.measure.block_reduce(
            roi_array[i], (scale_factor, scale_factor), np.mean)
        roi64.append(np.where(reduced_roi > 0.5, True, False))

    # smooth (after subarr to avoid edge effects)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    kernel = kernel / np.sum(kernel)
    smooth64 = sp.signal.convolve2d(image64, kernel, boundary='symm', mode='same')

    rows = np.max(roi64[0], axis=1)
    cols = np.max(roi64[0], axis=0)
    smooth64ufov = smooth64[rows][:, cols]
    rows = np.max(roi64[1], axis=1)
    cols = np.max(roi64[1], axis=0)
    smooth64cfov = smooth64[rows][:, cols]
    # differential uniformity
    du_ufov_dict = get_differential_uniformity(smooth64ufov)
    du_cfov_dict = get_differential_uniformity(smooth64cfov)
    du_ufov = du_ufov_dict['du']
    du_cfov = du_cfov_dict['du']

    # integral uniformity
    max_val = np.max(smooth64ufov)
    min_val = np.min(smooth64ufov)
    iu_ufov = 100. * (max_val - min_val) / (max_val + min_val)
    max_val = np.max(smooth64cfov)
    min_val = np.min(smooth64cfov)
    iu_cfov = 100. * (max_val - min_val) / (max_val + min_val)

    return {'matrix': smooth64, 'du_matrix': du_ufov_dict['du_matrix'],
            'values': [iu_ufov, du_ufov, iu_cfov, du_cfov]}


def get_eye_filter(roi_size, pix, c):
    """Get eye filter V(r)=r^1.3*exp(-cr^2).

    Parameters
    ----------
    roi_size : int
        size of roi or image in pix
    pix : float
        mm pr pix in image
    c : float
        adjusted to have V(r) max 4 cy/degree
            (Nelson et al uses c=28 = display size 65mm, viewing distance 1.5m)

    Returns
    -------
    eye_filter : dict
        filter 2d : np.ndarray quadratic with size roi_height
        curve: dict of r and V
    """
    def eye_filter_func(r, c):
        return r**1.3 * np.exp(-c*r**2)

    freq = np.fft.fftfreq(roi_size, d=pix)
    freq = freq[:freq.size//2]  # from center
    V = eye_filter_func(freq, c)
    eye_filter_1d = {'r': freq, 'V': 1/np.max(V) * V}

    unit = freq[1] - freq[0]
    dists = mmcalc.get_distance_map_point((roi_size, roi_size))
    eye_filter_2d = eye_filter_func(unit*dists, c)
    eye_filter_2d = 1/np.max(eye_filter_2d) * eye_filter_2d

    return {'filter_2d': eye_filter_2d, 'curve': eye_filter_1d, 'unit': unit}


def calculate_SNI_ROI(image2d, roi_array_this, eye_filter=None, unit=1.,
                      pix=1., fit_dict=None):
    """Calculate SNI for one ROI.

    Parameters
    ----------
    image2d : numpy.2darray
        if fit_matrix is not None, this is a flattened matrix
        (corrected for point source curvature)
    roi_array : numpy.2darray
        2d mask for the current ROI
    eye_filter : numpy.2darray
    unit : float
        unit of NPS and eye_filter
    pix : float
    fit_dict : dict
        dictionary from get_corrections_point_source (if corrections else None)

    Returns
    -------
    values : list of float
        ['SNI max', 'SNI L1', 'SNI L2', 'SNI S1', .. 'SNI S6']
    details_dict : dict
        NPS : numpy.2darray NPS in ROI
        quantum_noise : constant or numpy.2darray (NPS of estimated noise)
        freq : numpy.1darray - frequencies of radial NPS curve
        rNPS : radial NPS curve
        rNPS_filt
        rNPS_struct
        rNPS_struct_filt
    errmsgs : list of str
    """
    rows = np.max(roi_array_this, axis=1)
    cols = np.max(roi_array_this, axis=0)
    subarray = image2d[rows][:, cols]
    NPS = mmcalc.get_2d_NPS(subarray, pix)
    # set central axis of NPS array to 0
    #  to avoid issues with this high and not important value
    line = NPS.shape[0] // 2
    #NPS[line] = 0
    #NPS[:, line] = 0
    NPS[line, line] = 0
    rNPS_quantum_noise = None
    if fit_dict is None:  # uncorrected image
        quantum_noise = np.mean(image2d[rows][:, cols]) * pix**2
        """ explained how quantum noise is found above
        Mean count=variance=pixNPS^2*Total(NPS) where pixNPS=1./(ROIsz*pix)
        Total(NPS)=NPSvalue*ROIsz^2
        NPSvalue = Meancount/(pixNPS^2*ROIsz^2)=MeanCount*pix^2
        """
        NPS_struct = NPS - quantum_noise
    else:
        # point source - varying quantum noise
        sub_estimated_noise = fit_dict['estimated_noise_image'][rows][:, cols]
        quantum_noise = mmcalc.get_2d_NPS(sub_estimated_noise, pix)
        # set central axis of NPS array to 0
        #quantum_noise[line] = 0
        #quantum_noise[:, line] = 0
        quantum_noise[line, line] = 0
        _, rNPS_quantum_noise = mmcalc.get_radial_profile(
            quantum_noise, pix=unit, step_size=0.01)
        NPS_struct = np.subtract(NPS, quantum_noise)
    NPS_filt = NPS * eye_filter
    NPS_struct_filt = NPS_struct * eye_filter
    SNI = np.sum(NPS_struct_filt) / np.sum(NPS_filt)

    # radial NPS curves
    freq, rNPS = mmcalc.get_radial_profile(NPS, pix=unit, step_size=0.01)
    _, rNPS_filt = mmcalc.get_radial_profile(NPS_filt, pix=unit, step_size=0.01)
    _, rNPS_struct = mmcalc.get_radial_profile(NPS_struct, pix=unit, step_size=0.01)
    _, rNPS_struct_filt = mmcalc.get_radial_profile(
        NPS_struct_filt, pix=unit, step_size=0.01)

    details_dict_roi = {
        'NPS': NPS, 'quantum_noise': quantum_noise,
        'freq': freq, 'rNPS': rNPS, 'rNPS_filt': rNPS_filt,
        'rNPS_struct': rNPS_struct, 'rNPS_struct_filt': rNPS_struct_filt,
        'rNPS_quantum_noise': rNPS_quantum_noise
        }
    return (SNI, details_dict_roi)

def calculate_NM_SNI(image2d, roi_array, image_info, paramset):
    """Calculate uniformity parameters.

    Parameters
    ----------
    image2d : numpy.ndarray
    roi_array : list of numpy.ndarray
        list of 2d masks for all_rois, L1, L2, S1 ... S6
    image_info :  DcmInfo
        as defined in scripts/dcm.py
    paramset : cfc.ParamsetNM

    Returns
    -------
    values : list of float
        ['SNI max', 'SNI L1', 'SNI L2', 'SNI S1', .. 'SNI S6']
    details_dict : dict
    errmsgs : list of str
    """
    values_sup = [None] * 3
    details_dict = {}
    SNI_values = []
    errmsgs = []

    # point source correction
    fit_dict = None
    if paramset.sni_correct:
        lock_z = paramset.sni_radius if paramset.sni_lock_radius else None
        fit_dict, errmsg = get_corrections_point_source(
            image2d, image_info, roi_array[0],
            fit_x=paramset.sni_correct_pos_x,
            fit_y=paramset.sni_correct_pos_y,
            lock_z=lock_z
            )
        if errmsg is not None:
            errmsgs.append(errmsg)
        values_sup = [fit_dict['dx'], fit_dict['dy'], fit_dict['distance']]
        details_dict = fit_dict

    details_dict['pr_roi'] = []

    # large ROIs
    rows = np.max(roi_array[1], axis=1)
    eye_filter = get_eye_filter(
        np.count_nonzero(rows), image_info.pix[0], paramset.sni_eye_filter_c)
    details_dict['eye_filter_large'] = eye_filter['curve']
    for i in [1, 2]:
        SNI, details_dict_roi = calculate_SNI_ROI(
            image2d, roi_array[i],
            eye_filter=eye_filter['filter_2d'], unit=eye_filter['unit'],
            pix=image_info.pix[0], fit_dict=fit_dict)
        details_dict['pr_roi'].append(details_dict_roi)
        SNI_values.append(SNI)

    # small ROIs
    rows = np.max(roi_array[3], axis=1)
    eye_filter = get_eye_filter(
        np.count_nonzero(rows), image_info.pix[0], paramset.sni_eye_filter_c)
    details_dict['eye_filter_small'] = eye_filter['curve']
    for i in range(3, 9):
        SNI, details_dict_roi = calculate_SNI_ROI(
            image2d, roi_array[i],
            eye_filter['filter_2d'], unit=eye_filter['unit'],
            pix=image_info.pix[0], fit_dict=fit_dict)
        details_dict['pr_roi'].append(details_dict_roi)
        SNI_values.append(SNI)

    values = [np.max(SNI_values)] + SNI_values

    return (values, values_sup, details_dict, errmsgs)
