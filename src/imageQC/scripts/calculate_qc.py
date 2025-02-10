#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculation processes for the different tests.

@author: Ellen Wasbø
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy
import warnings

import numpy as np
import scipy as sp
from scipy.signal import find_peaks
import skimage
from PyQt5.QtWidgets import qApp

# imageQC block start
from imageQC.scripts import dcm
from imageQC.scripts.calculate_roi import (
    get_rois, get_roi_circle, get_roi_rectangle)
import imageQC.scripts.mtf_methods as mtf_methods
import imageQC.scripts.nm_methods as nm_methods
import imageQC.scripts.cdmam_methods as cdmam_methods
import imageQC.scripts.mini_methods_format as mmf
import imageQC.scripts.mini_methods as mm
import imageQC.scripts.mini_methods_calculate as mmcalc
from imageQC.config.iQCconstants import (
    HEADERS, HEADERS_SUP, QUICKTEST_OPTIONS, HALFLIFE)
import imageQC.config.config_classes as cfc
from imageQC.config.config_func import load_cdmam
from imageQC.scripts import digit_methods
from imageQC.scripts.artifact import apply_artifacts
# imageQC block end

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')


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
    pr_image_sup: bool = True


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
    if len(values) > 0:
        selected_values = []
        allvals = []
        if isinstance(values[0], list):
            if columns == []:
                columns = list(range(len(values[0])))
            for row in values:
                try:
                    row_vals = [row[col] for col in columns]
                except IndexError:
                    row_vals = [0] * len(columns)
                    print('Error extracting values. Output template not matching column numbers')
                selected_values.append(row_vals)
                allvals.extend(row_vals)
        else:
            if columns == []:
                columns = list(range(len(values)))
            try:
                vals = [values[col] for col in columns]
            except IndexError:
                vals = [0] * len(columns)
                print('Error extracting values. Output template not matching column numbers')
            selected_values = vals
            allvals = vals

        for i, val in enumerate(allvals):
            try:
                val = float(val)
                allvals[i] = val
            except (TypeError, ValueError):
                pass

        is_string = [isinstance(val, str) for val in allvals]

        if calculation == '=' or any(is_string):
            new_values = selected_values
        else:
            if all(allvals):
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
                elif calculation == 'width (max-min)':
                    new_values = [float(np.max(allvals)) - float(np.min(allvals))]
            else:
                is_zero = [val == 0 for val in allvals]
                if all(is_zero):
                    new_values = [0]
                else:
                    new_values = [None]

    return new_values


def mtf_multiply_10(row):
    """Multiply MTF values by 10 to cy/cm (cy/mm default), accept None in values."""
    new_row = []
    try:
        new_row = list(10 * np.array(row))
    except TypeError:
        for val in row:
            if val is not None:
                new_row.append(10 * val)
            else:
                new_row.append(None)
    return new_row


def format_result_table(input_main, test, values, headers):
    """Format numbers of result_table.

    Used by generate_report."""
    string_list = []
    paramset = input_main.current_paramset
    dm = input_main.current_paramset.output.decimal_mark
    for r, row in enumerate(values):
        if any(row):
            if all([test == 'MTF',
                    input_main.current_modality == 'CT']):
                if not paramset.mtf_cy_pr_mm:
                    row = mtf_multiply_10(row)
            out_values = extract_values(row)
            if test == 'DCM':
                string_list.append(
                    mmf.val_2_str(
                        out_values, decimal_mark=dm,
                        format_same=False,
                        format_strings=paramset.dcm_tagpattern.list_format)
                    )
            else:
                string_list.append(
                    mmf.val_2_str(out_values, decimal_mark=dm,
                                  format_same=False)
                    )

    return string_list


def get_image_names(input_main):
    """Get set names of images or generate default names (indexed names)."""
    image_names = [f'img{i}' for i in range(len(input_main.imgs))]
    if len(input_main.current_quicktest.image_names) > 0:
        set_names = input_main.current_quicktest.image_names
        if any(set_names):
            for i, set_name in enumerate(set_names):
                if set_name != '' and i < len(image_names):
                    image_names[i] = set_name
    return image_names


def get_group_names(input_main):
    """Get set group names for each image or default indexed groups."""
    uniq_group_ids_all = mm.get_uniq_ordered(
        input_main.current_group_indicators)
    group_names = []
    for i in range(len(input_main.imgs)):
        group_idx = uniq_group_ids_all.index(
            input_main.current_group_indicators[i])
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
    return group_names


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
        image_names = get_image_names(input_main)
        group_names = get_group_names(input_main)
        marked = input_main.current_quicktest.tests
        dm = input_main.current_paramset.output.decimal_mark
        paramset = input_main.current_paramset

        # first all output_sub defined, then defaults where no output defined
        output_all_actual = {}
        for test, sublist in paramset.output.tests.items():
            if test in input_main.results:
                if input_main.results[test] is not None:
                    output_all_actual[test] = paramset.output.tests[
                        test]
        for test in input_main.results:  # add defaults if not defined
            if input_main.results[test] is not None:
                if test not in output_all_actual or len(output_all_actual[test]) == 0:
                    output_all_actual[test] = [
                        cfc.QuickTestOutputSub(columns=[], alternative=-1)]

        for test in output_all_actual:
            output_subs = output_all_actual[test]
            res_pr_image = True

            # for each sub-output for current test
            for sub in output_subs:
                values = None
                headers = None
                if sub.alternative > 9:  # supplement table to output
                    values = input_main.results[test]['values_sup']
                    headers = input_main.results[test]['headers_sup']
                    res_pr_image = input_main.results[test]['pr_image_sup']
                else:
                    proceed = False
                    res_pr_image = input_main.results[test]['pr_image']
                    if sub.alternative == -1:  # not defined use all as default
                        proceed = True
                    else:
                        if input_main.results[test]['alternative'] == sub.alternative:
                            proceed = True
                    if proceed:
                        values = input_main.results[test]['values']
                        headers = input_main.results[test]['headers']
                out_values = []
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
                                        if not paramset.mtf_cy_pr_mm:
                                            row = mtf_multiply_10(row)
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
                            force_include = False
                            if row is None:
                                row = [None] * len(headers)
                                force_include = True
                            if len(values) == len(input_main.imgs):
                                # not 3d CHANGED from marked to input_main.imgs and
                                # added if len(row)=0
                                try:
                                    if test in marked[r]:
                                        force_include = True  # also if all None
                                        if len(row) == 0:
                                            row = [None] * len(headers)
                                except IndexError:
                                    pass
                            if any(row) or force_include:
                                if all([test == 'MTF',
                                        input_main.current_modality == 'CT']):
                                    if not paramset.mtf_cy_pr_mm:
                                        row = mtf_multiply_10(row)
                                out_values = extract_values(
                                    row,
                                    columns=sub.columns,
                                    calculation=sub.calculation
                                    )
                                if test == 'DCM':
                                    all_format_strings = (
                                        paramset.dcm_tagpattern.list_format)
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
                    if len(out_values) > 1:  # as is (=) or group/calculation failed
                        if sub.columns == []:
                            headers_this = headers
                        else:
                            headers_this = [
                                header for c, header in enumerate(headers)
                                if c in sub.columns]
                    else:
                        if sub.label != '':
                            headers_this = sub.label
                        else:
                            if len(sub.columns) == 1:
                                headers_this = headers[sub.columns[0]]
                            else:
                                if len(headers) > 0:
                                    headers_this = headers[0]
                                else:
                                    headers_this = None
                        if headers_this:
                            headers_this = [headers_this]
                        else:
                            headers_this = []

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


def calculate_qc(input_main, wid_auto=None,
                 auto_template_label='', auto_template_session=''):
    """Calculate tests according to current info in main.

    Parameters
    ----------
    input_main : object
        of class MainWindow or MainAuto containing specific attributes
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
    modality = input_main.current_modality
    errmsgs = []
    main_type = input_main.__class__.__name__
    try:
        input_main.progress_modal.setValue(1)
    except AttributeError:
        pass
    delta_xya = [
        input_main.gui.delta_x,
        input_main.gui.delta_y,
        input_main.gui.delta_a]
    overlay = input_main.gui.show_overlay

    paramset = input_main.current_paramset
    if wid_auto is not None:
        wid_auto.wid_image_display.canvas.main.current_paramset = paramset
    img_infos = input_main.imgs
    tag_infos = input_main.tag_infos
    quicktest = input_main.current_quicktest

    if len(quicktest.tests) > len(img_infos):
        if any(quicktest.tests[len(img_infos):]):
            errmsgs.append(
                f'QuickTest template {quicktest.label} specified for '
                f'{len(quicktest.tests)} images. Only {len(img_infos)} images loaded.')

    proceed = True
    if len(img_infos) == 0:
        proceed = False
    else:
        if not any(quicktest.tests):
            errmsgs.append(
                f'QuickTest template {quicktest.label} have no test specified.')
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
            extras = None  # placeholder for extra arguments to pass to calc_2d/3d
            if input_main.current_modality == 'NM' and 'SNI' in flattened_marked:
                if paramset.sni_ref_image != '':
                    extras = nm_methods.get_SNI_ref_image(paramset, tag_infos)
            if 'Num' in flattened_marked:
                digit_templates = input_main.digit_templates[
                    input_main.current_modality]
            else:
                digit_templates = []
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
            extra_tag_list_compare = None
            extra_tag_list_keep = False
            marked_3d = []
            input_main.current_group_indicators = ['' for i in range(n_img)]
            force_new_roi = []  # tests where a new roi should be set pr image
            for i in range(n_analyse):
                marked_3d.append([])
                if len(marked[i]) > 0:
                    if modality == 'CT':
                        if 'MTF' in marked[i]:
                            if paramset.mtf_type > 0:
                                marked_3d[i].append('MTF')
                                extra_tag_pattern = cfc.TagPatternFormat(
                                    list_tags=['ConvolutionKernel'])
                                extra_tag_list = []
                                read_tags[i] = True
                            else:
                                if paramset.mtf_auto_center:
                                    force_new_roi.append('MTF')
                        if 'TTF' in marked[i]:
                            marked_3d[i].append('TTF')
                            extra_tag_pattern = cfc.TagPatternFormat(
                                list_tags=['SeriesUID', 'ConvolutionKernel'])
                            extra_tag_list = []
                            read_tags[i] = True
                        if 'DPR' in marked[i]:
                            marked_3d[i].append('DPR')
                        if 'Sli' in marked[i]:
                            if paramset.sli_auto_center:
                                force_new_roi.append('Sli')
                        if 'CTn' in marked[i]:
                            if paramset.ctn_auto_center or paramset.ctn_search:
                                force_new_roi.append('CTn')
                    elif modality == 'Xray':
                        if 'Noi' in marked[i]:
                            force_new_roi.append('Noi')
                        if 'MTF' in marked[i]:
                            if paramset.mtf_auto_center:
                                force_new_roi.append('MTF')
                        if 'Foc' in marked[i]:
                            force_new_roi.append('Foc')
                        if 'Def' in marked[i]:
                            marked_3d[i].append('Def')
                    elif modality == 'Mammo':
                        if 'CDM' in marked[i]:
                            marked_3d[i].append('CDM')
                        if 'SDN' in marked[i]:
                            if paramset.sdn_auto_center:
                                force_new_roi.append('SDN')
                        if 'MTF' in marked[i]:
                            if paramset.mtf_auto_center:
                                force_new_roi.append('MTF')
                    elif modality == 'NM':
                        if 'Uni' in marked[i]:
                            if paramset.uni_sum_first:
                                marked_3d[i].append('Uni')
                            else:
                                force_new_roi.append('Uni')
                        if 'SNI' in marked[i]:
                            if paramset.sni_sum_first:
                                marked_3d[i].append('SNI')
                            else:
                                force_new_roi.append('SNI')
                        if 'MTF' in marked[i]:
                            if paramset.mtf_auto_center:
                                force_new_roi.append('MTF')
                        if 'Swe' in marked[i]:
                            marked_3d[i].append('Swe')
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
                            extra_tag_list_compare = [False, True, True, True]
                            read_tags[i] = True
                        if 'Rec' in marked[i]:
                            marked_3d[i].append('Rec')
                            extra_tag_pattern = cfc.TagPatternFormat(
                                list_tags=['AcquisitionTime', 'Units'])
                            extra_tag_list = []
                            extra_tag_list_keep = True
                            extra_tag_list_compare = [False, True]
                            read_tags[i] = True
                        if 'MTF' in marked[i]:
                            if paramset.mtf_type > 0:
                                marked_3d[i].append('MTF')
                    elif modality == 'MR':
                        if 'SNR' in marked[i]:
                            if paramset.snr_type == 0:
                                marked_3d[i].append('SNR')
                        # TODO? force_new_roi.append all with optimize center?

            # list of shape + pix for testing if new roi need to be calculated
            xypix = []
            for i, img_info in enumerate(img_infos):
                shape_pix_list = list(img_info.shape) + list(img_info.pix)
                xypix.append(shape_pix_list)

            prev_image_xypix = {}
            prev_roi = {}
            err_extra = False
            if wid_auto is not None:
                curr_progress_val = wid_auto.progress_modal.value()
                progress_increment = round(
                    wid_auto.progress_modal.sub_interval / n_analyse)
            else:
                curr_progress_val = 0
                if main_type == 'TaskBasedImageQualityDialog':
                    curr_progress_val = 100 * input_main.run_all_index

            cancelled = False
            for i in range(n_analyse):
                if main_type in ['MainWindow', 'TaskBasedImageQualityDialog']:
                    try:
                        input_main.progress_modal.setLabelText(
                            f'Reading image {i}/{n_img}')
                        input_main.progress_modal.setValue(
                             curr_progress_val + round(100 * i/n_analyse))
                    except AttributeError:
                        pass

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
                            get_window_level=any(auto_template_label),
                            overlay=overlay
                            )
                        try:
                            if len(input_main.imgs[i].artifacts) > 0:
                                image = apply_artifacts(
                                    image, input_main.imgs[i],
                                    input_main.artifacts,
                                    input_main.artifacts_3d, i)
                        except (TypeError, AttributeError, IndexError):
                            pass
                        if len(tags) > 0:
                            if isinstance(tags[0], dict):
                                tag_lists[i] = tags[0]['dummy']
                                if extra_tag_pattern is not None:
                                    extra_tag_list.append(tags[2]['dummy'])
                            else:
                                tag_lists[i] = tags[0]
                                if extra_tag_pattern is not None:
                                    extra_tag_list.append(tags[2])
                        if extra_tag_pattern is not None:
                            if len(extra_tag_list) > 1:
                                if extra_tag_list_compare is None:
                                    if extra_tag_list[0] != extra_tag_list[-1]:
                                        err_extra = True
                                else:
                                    for cno, comp in enumerate(extra_tag_list_compare):
                                        if comp:
                                            if (
                                                    extra_tag_list[0][cno]
                                                    != extra_tag_list[-1][cno]):
                                                err_extra = True
                                if extra_tag_list_keep is False:
                                    extra_tag_list.pop()  # always check against first
                    else:
                        tags = dcm.get_tags(
                            img_infos[i].filepath,
                            frame_number=img_infos[i].frame_number,
                            tag_patterns=[paramset.dcm_tagpattern, group_pattern],
                            tag_infos=tag_infos
                            )
                        if len(tags) > 0:
                            if isinstance(tags[0], dict):
                                tag_lists[i] = tags[0]['dummy']
                            else:
                                tag_lists[i] = tags[0]
                    if len(tags) > 0:
                        input_main.current_group_indicators[i] = '_'.join(tags[1])
                else:
                    if read_image[i]:
                        image, tags = dcm.get_img(
                            img_infos[i].filepath,
                            frame_number=img_infos[i].frame_number,
                            tag_patterns=[group_pattern],
                            tag_infos=tag_infos,
                            get_window_level=any(auto_template_label),
                            overlay=overlay
                            )
                        try:
                            if len(input_main.imgs[i].artifacts) > 0:
                                image = apply_artifacts(
                                    image, input_main.imgs[i],
                                    input_main.artifacts,
                                    input_main.artifacts_3d, i)
                        except (TypeError, AttributeError, IndexError):
                            pass
                        if len(tags) > 0:
                            input_main.current_group_indicators[i] = '_'.join(tags[0])
                if wid_auto is not None:
                    wid_auto.progress_modal.setLabelText(
                        f'{auto_template_label}: Calculating '
                        f'img {i+1}/{n_img} ({auto_template_session})'
                        )
                    wid_auto.progress_modal.setValue(
                        curr_progress_val + (i+1) * progress_increment)
                    if image is not None and wid_auto.chk_display_images.isChecked():
                        wid_auto.wid_image_display.canvas.main.active_img = image
                        wid_auto.wid_image_display.canvas.img_draw(
                            auto=True, window_level=tags[-1])
                    else:
                        wid_auto.wid_image_display.canvas.img_clear()

                for test in marked[i]:
                    input_main.current_test = test
                    if test not in ['DCM', '']:
                        if test in marked_3d[i]:
                            if matrix[i] is None:  # only once pr image
                                matrix[i] = image  # save image for later
                        else:
                            if test in force_new_roi:
                                calculate_new_roi = True
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
                                if wid_auto.chk_display_images.isChecked():
                                    canv_main = wid_auto.wid_image_display.canvas.main
                                    canv_main.current_test = test
                                    canv_main.current_roi = prev_roi[test]
                                    wid_auto.wid_image_display.canvas.roi_draw()
                                    qApp.processEvents()

                            result = calculate_2d(
                                image, prev_roi[test], img_infos[i],
                                modality, paramset, test, delta_xya,
                                digit_templates, extras)
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
                                    ''' NB if changing dict structure also update in:
                                            input_main.results['DCM'] (below)
                                            ui_main.update_results
                                    '''
                                    input_main.results[test] = {
                                        'headers': result.headers,
                                        'values': [[] for i in range(n_img)],
                                        'alternative': result.alternative,
                                        'headers_sup': result.headers_sup,
                                        'values_sup': [[] for i in range(n_img)],
                                        'details_dict': [{} for i in range(n_img)],
                                        'pr_image': True,
                                        'pr_image_sup': True,
                                        'values_info': result.values_info,
                                        'values_sup_info': result.values_sup_info
                                        }
                                input_main.results[
                                    test]['values'][i] = result.values
                                input_main.results[
                                    test]['values_sup'][i] = result.values_sup
                                input_main.results[
                                    test]['details_dict'][i] = result.details_dict
                try:
                    if input_main.progress_modal.wasCanceled():
                        cancelled = True
                        break
                except AttributeError:
                    pass

            if err_extra and cancelled is False:
                if extra_tag_list_compare is not None:
                    attr_list = []
                    for attr_no, attr in enumerate(extra_tag_pattern.list_tags):
                        if extra_tag_list_compare[attr_no]:
                            attr_list.append(attr)
                else:
                    attr_list = extra_tag_pattern.list_tags
                errmsgs.append(
                    f'Warning image: Differences found in DICOM header that are '
                    'supposed to be equal for this test:'
                    f'{attr_list}')

            # post processing - where values depend on all images
            if cancelled is False:
                if modality == 'CT':
                    if 'Noi' in flattened_marked:
                        try:
                            noise = [
                                row[1] for row
                                in input_main.results['Noi']['values']]
                            avg_noise = sum(noise)/len(noise)
                            for row in input_main.results['Noi']['values']:
                                # diff from avg (%)
                                row[2] = 100.0 * (row[1] - avg_noise) / avg_noise
                                row[3] = avg_noise
                            # removed if closed images in ui_main update results
                        except (KeyError, IndexError):
                            # average and diff from avg ignored if not all tested
                            for row in input_main.results['Noi']['values']:
                                if row:
                                    row[2] = 'NA'
                                    row[3] = 'NA'
                    if 'NPS' in flattened_marked:
                        input_main.results['NPS']['pr_image_sup'] = False
                        input_main.results['NPS']['values_sup_info'] = (
                            'Average from results pr image')
                        input_main.results['NPS']['headers_sup'] = copy.deepcopy(
                            HEADERS['CT']['NPS']['alt0'])
                        values = [row for row in input_main.results['NPS']['values']
                                  if len(row) > 0]
                        values = np.array(values)
                        values_avg = list(np.mean(values, axis=0))
                        input_main.results['NPS']['values_sup'] = [values_avg]
                    if 'Dim' in flattened_marked:
                        if 'MainWindow' in str(type(input_main)):
                            input_main.update_roi()

            if any(marked_3d) and cancelled is False:
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
            if any(read_tags) and 'DCM' in flattened_marked and cancelled is False:
                ignore_cols = []
                for idx, val in enumerate(paramset.dcm_tagpattern.list_format):
                    if len(val) > 2:
                        if val[2] == '0':
                            ignore_cols.append(idx)

                # ensure rows without DCM test (only extra tags) are not included
                for i, tags in enumerate(tag_lists):
                    try:
                        if 'DCM' not in marked[i]:
                            tag_lists[i] = []
                    except IndexError:
                        pass

                tag_lists = mmf.convert_lists_to_numbers(
                    tag_lists, ignore_columns=ignore_cols)
                input_main.results['DCM'] = {
                    'headers': paramset.dcm_tagpattern.list_tags,
                    'values': tag_lists,
                    'alternative': 0,
                    'headers_sup': [],
                    'values_sup': [[] for i in range(n_img)],
                    'details_dict': [{} for i in range(n_img)],
                    'pr_image': True,
                    'pr_image_sup': True,
                    'values_info': '',
                    'values_sup_info': ''
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

        if len(msgs) > 0 and input_main.automation_active:
            input_main.errmsgs = msgs

    if main_type == 'MainWindow':
        if input_main.automation_active is False:
            if len(msgs) > 0:
                input_main.display_errmsg(msgs)
            else:
                input_main.status_bar.showMessage('Finished', 1000)

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
            # refresh image display if result contain rois to display
            if input_main.current_modality == 'CT':
                if 'MTF' in input_main.results and paramset.mtf_type == 2:
                    input_main.refresh_img_display()
                if 'TTF' in input_main.results:
                    input_main.refresh_img_display()
            elif 'Foc' in input_main.results:
                input_main.refresh_img_display()
            elif 'Rec' in input_main.results:
                try:
                    input_main.wid_window_level.tb_wl.set_window_level(
                        'min_max', set_tools=True)
                    input_main.set_active_img(
                        input_main.results['Rec']['details_dict']['max_slice_idx'])
                except KeyError:
                    pass
            elif 'CDM' in input_main.results:
                try:
                    input_main.tab_mammo.cdm_cbox_diameter.clear()
                    input_main.tab_mammo.cdm_cbox_diameter.addItems(
                        [str(x) for x in
                         input_main.results['CDM']['details_dict'][-1]['diameters']])
                    input_main.tab_mammo.cdm_cbox_thickness.clear()
                    thickness = input_main.results[
                        'CDM']['details_dict'][-1]['thickness']
                    phantom = 40 if isinstance(thickness[0], list) else 34
                    if phantom == 40:
                        items = [str(idx) for idx in range(16)]
                        items[0] = items[0] + ' (thickest)'
                        items[-1] = items[-1] + ' (thinnest)'
                        row, col = 0, 0
                    else:
                        items = [str(x) for x in thickness]
                        row, col = 0, 6
                    input_main.tab_mammo.cdm_cbox_thickness.addItems(items)
                    input_main.tab_mammo.cdm_cbox_diameter.setCurrentIndex(col)
                    input_main.tab_mammo.cdm_cbox_thickness.setCurrentIndex(row)
                except:  # if cancelled
                    input_main.results['CDM'] = None
            try:
                input_main.progress_modal.setValue(input_main.progress_modal.maximum())
            except AttributeError:
                pass

    elif main_type == 'TaskBasedImageQualityDialog':
        if len(msgs) > 0:
            if input_main.errmsgs is None:
                input_main.errmsgs = msgs
            else:
                input_main.errmsgs.extend(msgs)
        if input_main.run_all_active is False:
            input_main.display_errmsg(msgs)
            if current_test_before in flattened_marked or len(flattened_marked) == 0:
                set_current_test = current_test_before
            else:
                set_current_test = flattened_marked[0]
            idx_set_test = input_main.tests.index(set_current_test)
            widget = input_main.tab_ct
            widget.setCurrentIndex(idx_set_test)
            input_main.current_test = set_current_test
            input_main.refresh_results_display()
            if 'TTF' in input_main.results:
                input_main.refresh_img_display()
            input_main.progress_modal.setValue(input_main.progress_modal.maximum())


def calculate_2d(image2d, roi_array, image_info, modality,
                 paramset, test_code, delta_xya, digit_templates, extras):
    """Calculate tests based on 2d-image.

    NB see code at bottom of this method with result = locals()[test_code]()

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
    digit_templates : list of DigitTemplate
        if Num to be tested else []
    extras : object
        holder for extra argumets. Default is None

    Returns
    -------
    result : Results
    """

    def Bar():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
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
                    if avg_val > 0 and var_val >= avg_val:  # avoid RuntimeWarning
                        values.append(np.sqrt(2*(var_val - avg_val)) / avg_val)  # MTF
                    else:
                        values.append(np.nan)

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

    def CTn():
        headers = paramset.ctn_table.labels
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['altAll'])
        if image2d is not None:
            values = []
            errmsg = []
            x_vals = []
            y_vals = []
            for i in range(len(paramset.ctn_table.labels)):
                arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[i]))
                values.append(np.mean(arr))
                try:
                    y_val = float(paramset.ctn_table.linearity_axis[i])
                    x_vals.append(np.mean(arr))
                    y_vals.append(y_val)
                except (ValueError, TypeError):
                    pass

            if len(x_vals) > 0:
                res = sp.stats.linregress(x_vals, y_vals)
                values_sup = [res.rvalue**2, res.intercept, res.slope]
            else:
                values_sup = [None, None, None]
                errmsg = (
                    f'Could not linear fit HU to {paramset.ctn_table.linearity_unit}. '
                    'Values not valid.'
                    )
            res = Results(headers=headers, values=values,
                          headers_sup=headers_sup, values_sup=values_sup,
                          errmsg=errmsg)
        else:
            res = Results(headers=headers, headers_sup=headers_sup)
        return res

    def Dim():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt0'])
        if image2d is None:
            res = Results(headers=headers, headers_sup=headers_sup)
        else:
            errmsg = ''
            values = []
            values_sup = []
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
                if center_x is not None:
                    centers_x.append(center_x - prof_x.size / 2 + dx[i])
                else:
                    centers_x.append(None)
                if center_y is not None:
                    centers_y.append(center_y - prof_y.size / 2 + dy[i])
                else:
                    centers_y.append(None)
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

    def Foc():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        # ['Star diameter (mm)', 'Magnification',
        # 'Blur diameter x (mm)', 'Blur diameter y (mm)',
        #'FS x (mm)', 'FS y (mm)']
        if image2d is None:
            res = Results(headers=headers)
        else:
            details_dict, values, errmsgs = calculate_focal_spot_size(
                image2d, roi_array, image_info, paramset)

            res = Results(
                headers=headers, values=values,
                details_dict=details_dict, errmsg=errmsgs)

        return res

    def Geo():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
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

    def Gho():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        if image2d is None:
            res = Results(headers=headers)
        else:
            avgs = []
            for i in range(np.shape(roi_array)[0]):
                arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[i]))
                avgs.append(np.mean(arr))
            values = avgs
            if modality == 'Mammo':
                ghost_factor = (avgs[2] - avgs[1]) / (avgs[0] - avgs[1])
                values.append(ghost_factor)
            elif modality == 'MR':
                PSG = abs(100.*0.5*(
                    (avgs[3]+avgs[4]) - (avgs[1]+avgs[2])
                    )/avgs[0])
                values.append(PSG)
            res = Results(headers=headers, values=values)

        return res

    def Hom():
        values = []
        headers_sup = []
        values_sup = []
        avgs = []
        stds = []
        alt = 0
        res = None
        if image2d is not None:
            proceed = True
            if modality == 'Mammo':
                proceed = False
            elif modality == 'Xray':
                if paramset.hom_tab_alt >= 3:
                    proceed = False
            if proceed:
                for i in range(np.shape(roi_array)[0]):
                    arr = np.ma.masked_array(
                        image2d, mask=np.invert(roi_array[i]))
                    avgs.append(np.mean(arr))
                    stds.append(np.std(arr))

        flatfield_mammo = False
        flatfield_aapm = False
        if modality == 'CT':
            headers = copy.deepcopy(HEADERS[modality][test_code]['altAll'])
            headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['altAll'])
            if image2d is not None:
                values = [avgs[1], avgs[2], avgs[3], avgs[4], avgs[0],
                          avgs[1] - avgs[0], avgs[2] - avgs[0],
                          avgs[3] - avgs[0], avgs[4] - avgs[0]]
                values_sup = [stds[1], stds[2], stds[3], stds[4], stds[0]]

        elif modality == 'Xray':
            alt = paramset.hom_tab_alt
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt'+str(alt)])
            headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt'+str(alt)])
            if image2d is not None:
                if alt == 0:
                    values = avgs + stds
                elif alt == 3:
                    flatfield_mammo = True
                elif alt == 4:
                    flatfield_aapm = True
                else:
                    avg_all = np.sum(avgs) / len(avgs)
                    diffs = [(avg - avg_all) for avg in avgs]
                    if alt == 1:
                        values = avgs + diffs
                    elif alt == 2:
                        diffs_percent = [100. * (diff / avg_all) for diff in diffs]
                        values = avgs + diffs_percent

        elif modality == 'Mammo':
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
            headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt0'])
            flatfield_mammo = True

        elif modality == 'PET':
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
            if image2d is not None:
                avg = sum(avgs) / len(avgs)
                diffs = [100.*(avgs[i] - avg)/avg for i in range(5)]
                values = avgs + diffs

        if flatfield_mammo:
            if image2d is not None:
                details = calculate_flatfield_mammo(
                    image2d, roi_array[-1], image_info, paramset)
                if details:
                    values = [
                        np.mean(details['averages']),
                        np.mean(details['snrs']),
                        details['n_rois'],
                        details['n_deviating_averages'],
                        details['n_deviating_snrs'],
                        details['n_deviating_rois'],
                        100 * details['n_deviating_rois'] / details['n_rois'],
                        details['n_deviating_pixels'],
                        100 * details['n_deviating_pixels'] / details['n_pixels'],
                        ]
                    masked_image = np.ma.masked_array(image2d, mask=details['roi_mask'])
                    values_sup = [
                        np.min(masked_image), np.max(masked_image),
                        np.min(details['averages']), np.max(details['averages']),
                        np.min(details['snrs']), np.max(details['snrs']),
                        details['deviating_rois'].shape[1],
                        details['deviating_rois'].shape[0],
                        details['n_masked_rois'], details['n_masked_pixels']
                        ]

                    res = Results(
                        headers=headers, values=values,
                        headers_sup=headers_sup, values_sup=values_sup,
                        details_dict=details, alternative=alt)
        elif flatfield_aapm:
            if image2d is not None:
                details = calculate_flatfield_aapm(
                    image2d, roi_array[-1], image_info, paramset)
                snrs = np.divide(details['averages'], details['stds'])
                if details:
                    values = [
                        np.mean(details['averages']),
                        np.mean(details['stds']),
                        np.mean(snrs),
                        np.min(snrs),
                        details['n_anomalous_pixels'],
                        np.max(details['n_anomalous_pixels_pr_roi']),
                        np.max(np.abs(details['local_uniformities'])),
                        details['global_uniformity'],
                        np.max(np.abs(details['local_noise_uniformities'])),
                        details['global_noise_uniformity'],
                        np.max(np.abs(details['local_snr_uniformities'])),
                        details['global_snr_uniformity'],
                        details['relSDrow'],
                        details['relSDcol']
                        ]

                    res = Results(
                        headers=headers, values=values,
                        details_dict=details, alternative=alt)

        if res is None:
            res = Results(headers=headers, values=values,
                          headers_sup=headers_sup, values_sup=values_sup,
                          alternative=alt)
        return res

    def HUw():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        values = []
        if image2d is not None:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            avg_val = np.mean(arr)
            std_val = np.std(arr)
            values = [avg_val, std_val]
        res = Results(headers=headers, values=values)
        return res

    def MTF():
        errmsg = []
        if modality in ['CT', 'SPECT', 'PET']:
            # only bead/point method 2d (alt0)
            alt = paramset.mtf_type
            try:
                headers = copy.deepcopy(
                    HEADERS[modality][test_code]['alt' + str(alt)])
            except KeyError:
                headers = copy.deepcopy(
                    HEADERS[modality][test_code]['altAll'])
            try:
                headers_sup = copy.deepcopy(
                    HEADERS_SUP[modality][test_code]['alt' + str(alt)])
            except KeyError:
                headers_sup = copy.deepcopy(
                    HEADERS_SUP[modality][test_code]['altAll'])
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
                    if paramset.mtf_background_width > 0:
                        errmsg = ['Warning: width of background too narrow.'
                                  ' Background not corrected.']
                details, errmsg_calc = mtf_methods.calculate_MTF_point(
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
                        + [details[0]['gMTF_details']['prefilter_sigma']]
                        )
                    res = Results(
                        headers=headers, values=values,
                        headers_sup=headers_sup, values_sup=values_sup,
                        details_dict=details, alternative=alt, errmsg=errmsg)
                else:
                    res = Results(
                        headers=headers, headers_sup=headers_sup, errmsg=errmsg)

        elif modality in ['Xray', 'Mammo', 'MR']:
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
            headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt0'])
            if image2d is None:
                res = Results(headers=headers, headers_sup=headers_sup)
            else:
                roi_mtf_exist = True
                if roi_array is None:
                    roi_mtf_exist = False
                else:
                    if isinstance(roi_array, list):
                        if roi_array[0] is None:
                            roi_mtf_exist = False

                if roi_mtf_exist is False:
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

                    details, errmsg = mtf_methods.calculate_MTF_2d_line_edge(
                        sub, image_info.pix[0], paramset, mode='edge')
                    prefix = 'g' if paramset.mtf_gaussian else 'd'
                    try:
                        values = details[prefix + 'MTF_details']['values']
                        values_sup = (
                            details['gMTF_details']['LSF_fit_params']
                            + [details['gMTF_details']['prefilter_sigma']])
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
            headers = copy.deepcopy(
                HEADERS[modality][test_code]['alt' + str(alt)])
            headers_sup = copy.deepcopy(
                HEADERS_SUP[modality][test_code]['alt' + str(alt)])
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
                    details, errmsg = mtf_methods.calculate_MTF_point(
                        sub[0], image_info, paramset)
                    details[0]['matrix'] = sub[0]
                elif paramset.mtf_type == 1:
                    details, errmsg = mtf_methods.calculate_MTF_2d_line_edge(
                        sub, image_info.pix[0], paramset, mode='line')
                elif paramset.mtf_type == 2:
                    details, errmsg = mtf_methods.calculate_MTF_2d_line_edge(
                        sub, image_info.pix[0], paramset, mode='line', pr_roi=True)
                elif paramset.mtf_type == 3:  # edge
                    details, errmsg = mtf_methods.calculate_MTF_2d_line_edge(
                        sub, image_info.pix[0], paramset, mode='edge')

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
                        values_sup.append(details[0]['gMTF_details']['prefilter_sigma'])
                    else:
                        values = details[prefix + 'MTF_details']['values']
                        values_sup = list(details['gMTF_details']['LSF_fit_params'])
                        values_sup.append(details['gMTF_details']['prefilter_sigma'])
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

    def Noi():
        values = []
        if image2d is not None:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            avg_val = np.mean(arr)
            std_val = np.std(arr)
        if modality == 'CT':
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
            if image2d is not None:
                values = [avg_val, std_val, 0, 0]
        elif modality == 'Xray':
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
            if image2d is not None:
                values = [avg_val, std_val]

        res = Results(headers=headers, values=values)
        return res

    def NPS():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        if image2d is None:
            res = Results(headers=headers)
        else:
            values, details_dict = calculate_NPS(
                image2d, roi_array, image_info, paramset,
                modality=modality)
            res = Results(headers=headers, values=values,
                          details_dict=details_dict)
        return res

    def Num():
        headers = paramset.num_table.labels
        if '' in headers:
            for i, lbl in enumerate(headers):
                if lbl == '':
                    headers[i] = f'ROI_{i}'
        values = []
        errmsgs = []
        if image2d is not None:
            if len(digit_templates) == 0:
                errmsgs.append(f'No digit templates defined for {modality}.')
            elif paramset.num_digit_label == '':
                errmsgs.append(
                    'No digit template defined in current parameter '
                    f'set {paramset.label}.')
            elif len(roi_array) > 0:
                labels = [temp.label for temp in digit_templates]
                if paramset.num_digit_label in labels:
                    temp_idx = labels.index(paramset.num_digit_label)
                else:
                    temp_idx = None
                    errmsgs.append(
                        f'Digit template {paramset.num_digit_label} not found.')
                if temp_idx is not None:
                    digit_template = digit_templates[temp_idx]
                    imgs_is_arr = [
                        isinstance(img, (np.ndarray, list)) for img
                        in digit_template.images[:-2]]
                    if not any(imgs_is_arr):
                        temp_idx = None
                        errmsgs.append(
                            'Digit template images not defined for any digit '
                            f'of Digit template {paramset.num_digit_label}'
                            f'[{modality}].')
                if temp_idx is not None:
                    for i, roi in enumerate(roi_array):
                        rows = np.max(roi, axis=1)
                        cols = np.max(roi, axis=0)
                        sub = image2d[rows][:, cols]
                        missing_roi = False
                        if np.min(sub.shape) > 2:
                            char_imgs, chop_idxs = (
                                digit_methods.extract_char_blocks(sub))
                        else:
                            char_imgs = []
                            missing_roi = True
                        digit = None
                        if len(char_imgs) == 0:
                            if missing_roi:
                                errmsgs.append(
                                    f'ROI too small or outside image for {headers[i]}.')
                            else:
                                errmsgs.append(
                                    f'Failed finding digits in {headers[i]}.')
                        else:
                            digit = digit_methods.compare_char_blocks_2_template(
                                char_imgs, digit_template)
                            if digit is None:
                                errmsgs.append(
                                    f'Found no match for the blocks of {headers[i]}.')
                        values.append(digit)

            else:
                errmsgs.append('No ROIs defined.')

        res = Results(headers=headers, values=values, errmsg=errmsgs)
        return res

    def PIU():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        # ['min', 'max', 'PIU'],
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['altAll'])
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

    def Rin():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
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

    def RLR():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt0'])
        if image2d is None:
            res = Results(headers=headers, headers_sup=headers_sup)
        else:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            values = [np.mean(arr), np.std(arr)]
            values_sup = [np.min(arr), np.max(arr)]

            res = Results(
                headers=headers, values=values,
                headers_sup=headers_sup, values_sup=values_sup)

        return res

    def ROI():
        values = []
        values_sup = []
        alt = paramset.roi_use_table
        errmsgs = []
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt0'])
        all_headers = headers + headers_sup
        if alt > 0:
            labels = paramset.roi_table.labels
            if '' in labels:
                for i, lbl in enumerate(labels):
                    if lbl == '':
                        labels[i] = f'ROI_{i}'
            headers = [
                f'{labels[i]}_{all_headers[paramset.roi_table_val]}'
                for i in range(len(labels))]
            headers_sup = [
                f'{labels[i]}_{all_headers[paramset.roi_table_val_sup]}'
                for i in range(len(labels))]
        if image2d is not None:
            if roi_array is None:
                errmsgs.append('Found no ROI defined.')
            else:
                if alt in [0, 1]:
                    for i, roi in enumerate(roi_array):
                        arr = np.ma.masked_array(image2d, mask=np.invert(roi))
                        vals = [np.mean(arr), np.std(arr), np.min(arr), np.max(arr)]
                        if alt == 0:
                            values = vals[:2]
                            values_sup = vals[2:]
                        else:
                            values.append(vals[paramset.roi_table_val])
                            values_sup.append(vals[paramset.roi_table_val_sup])
                else:  # zoomed area
                    for i, roi in enumerate(roi_array):
                        rows = np.max(roi, axis=1)
                        cols = np.max(roi, axis=0)
                        sub = image2d[rows][:, cols]
                        missing_roi = False
                        if np.min(sub.shape) > 2:
                            arr = np.ma.masked_array(image2d, mask=np.invert(roi))
                            vals = [np.mean(arr), np.std(arr), np.min(arr), np.max(arr)]
                            values.append(vals[paramset.roi_table_val])
                            values_sup.append(vals[paramset.roi_table_val_sup])
                        else:
                            missing_roi = True
                            values.append(None)
                            values_sup.append(None)

                        if missing_roi:
                            errmsgs.append(
                                f'ROI too small or outside image for {labels[i]}.')

        res = Results(
            headers=headers, values=values,
            headers_sup=headers_sup, values_sup=values_sup,
            alternative=alt, errmsg=errmsgs)
        return res

    def SDN():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        if image2d is None:
            res = Results(headers=headers)
        else:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[0]))
            signal_mean = np.mean(arr)
            signal_std = np.std(arr)
            bg_means = []
            bg_stds = []
            for i in range(1, 5):
                arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[i]))
                bg_means.append(np.mean(arr))
                bg_stds.append(np.std(arr))
            bg_mean = np.mean(bg_means)
            bg_std = np.mean(bg_stds)
            sdnr = np.abs(signal_mean - bg_mean) / np.sqrt(
                (bg_std ** 2 + signal_std ** 2) / 2)
            values = [signal_mean, signal_std, bg_mean, bg_std, sdnr]

            res = Results(headers=headers, values=values)

        return res

    def SNI():
        alt = paramset.sni_alt  # set in ui_main_test_tabs, param_changed_from_gui

        headers = copy.deepcopy(HEADERS[modality][test_code]['alt' + str(alt)])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['altAll'])
        if image2d is None:
            res = Results(headers=headers, headers_sup=headers_sup)
        else:
            reference_noise = None
            err = ''
            if extras is not None:
                if 'reference_image' in extras:
                    if extras['reference_image'][0].shape == image2d.shape:
                        if 'reference_estimated_noise' not in extras:
                            nm_methods.get_SNI_reference_noise(
                                extras, roi_array[0], paramset)
                        reference_noise = extras[
                            'reference_estimated_noise'][image_info.frame_number]
                    else:
                        err = (
                            'Reference image not same size as image to analyse. '
                            'Quantum noise estimated from signal.')

            values, values_sup, details_dict, errmsg = nm_methods.calculate_NM_SNI(
                image2d, roi_array, image_info, paramset, reference_noise)
            if err:
                errmsg.insert(0, err)

            res = Results(
                headers=headers, values=values,
                headers_sup=headers_sup, values_sup=values_sup,
                details_dict=details_dict, errmsg=errmsg)

        return res

    def SNR():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt1'])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt1'])
        if image2d is None:
            res = Results(headers=headers)
        else:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[0]))
            central_mean = np.mean(arr)
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[1]))
            background_stdev = np.std(arr)
            image_noise = background_stdev/0.66  # NEMA MS 1-2008 method 4
            snr = central_mean / image_noise
            values = [central_mean, background_stdev, image_noise, snr]
            values_sup = [np.count_nonzero(roi_array[1])]

            res = Results(headers=headers, values=values,
                          headers_sup=headers_sup, values_sup=values_sup)

        return res

    def Sli():
        alt = paramset.sli_type
        if modality == 'CT':
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt' + str(alt)])
            if image2d is not None:
                lines = roi_array
                values, details_dict, errmsg = calculate_slicethickness(
                    image2d, image_info, paramset, lines, delta_xya)
                if alt == 0:
                    try:
                        values.append(np.mean(values[1:]))
                        values.append(100. * (values[-1] - values[0]) / values[0])
                        if paramset.sli_ignore_direction:
                            for c in [3, 4]:
                                values.insert(c, None)
                    except TypeError:
                        values.append(None)
                        values.append(None)
                elif alt == 1 and paramset.sli_ignore_direction:
                    for c in [1, 2]:
                        values.insert(c, None)
                res = Results(
                    headers=headers, values=values,
                    details_dict=details_dict,
                    alternative=alt, errmsg=errmsg)
            else:
                res = Results(headers=headers, alternative=alt)
        elif modality == 'MR':
            headers = copy.deepcopy(HEADERS[modality][test_code]['altAll'])
            # ['Nominal (mm)', 'Slice thickness (mm)', 'Diff (mm)', 'Diff (%)']
            headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['altAll'])
            # FWHM1, FWHM2
            if image2d is not None:
                lines = roi_array
                values, details_dict, errmsg = calculate_slicethickness(
                    image2d, image_info, paramset, lines, delta_xya, modality='MR')
                values_sup = values[1:3]
                try:
                    fwhm = values[1:3]
                    harmonic_mean = 2 * (fwhm[0] * fwhm[1])/(fwhm[0] + fwhm[1])
                    slice_thickness = paramset.sli_tan_a * harmonic_mean
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

    def Spe():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
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

    def STP():
        values = []
        if image2d is not None:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            avg_val = np.mean(arr)
            std_val = np.std(arr)
            values = [None, None, avg_val, std_val]
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        res = Results(headers=headers, values=values)
        return res

    def Uni():  # NM uniformity
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['altAll'])
        if image2d is None:
            res = Results(headers=headers, headers_sup=headers_sup)
        else:
            details_dict = {}
            errmsg = None
            values_sup = [None] * 3
            if paramset.uni_correct:
                res, errmsg = nm_methods.get_corrections_point_source(
                    image2d, image_info, roi_array[0],
                    fit_x=paramset.uni_correct_pos_x,
                    fit_y=paramset.uni_correct_pos_y,
                    lock_z=paramset.uni_lock_radius, guess_z=paramset.uni_radius,
                    correction_type='multiply'
                    )
                image_input = res['corrected_image']
                values_sup = [res['dx'], res['dy'], res['distance']]
                details_dict = res
            else:
                image_input = image2d

            res = nm_methods.calculate_NM_uniformity(
                image_input, roi_array, image_info.pix[0], paramset.uni_scale_factor)
            details_dict['matrix_ufov'] = res['matrix_ufov']
            details_dict['du_matrix'] = res['du_matrix']
            details_dict['pix_size'] = res['pix_size']
            values = res['values']
            values_sup.append(res['pix_size'])
            values_sup.append(res['center_pixel_count'])
            errmsg = [errmsg]
            ''' A bit annoying message when automation and this is settled.
            if res['center_pixel_count'] < 10000:
                if errmsg:
                    errmsg.append(
                        f'Center pixel (after scaling) = {res["center_pixel_count"]} '
                        '< 10000 (minimum set by NEMA)')
            '''

            res = Results(
                headers=headers, values=values,
                headers_sup=headers_sup, values_sup=values_sup,
                details_dict=details_dict, errmsg=errmsg)

        return res

    def Var():
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        if image2d is None:
            res = Results(headers=headers)
        else:
            # code adapted from:
            # https://www.imageeprocessing.com/2015/10/edge-detection-using-local-variance.html

            if roi_array[2] is None:
                sub = image2d
            else:
                rows = np.max(roi_array[2], axis=1)
                cols = np.max(roi_array[2], axis=0)
                sub = image2d[rows][:, cols]

            roi_sizes_mm = [paramset.var_roi_size, paramset.var_roi_size2]
            values = []
            details_dict = {'variance_image': []}
            for roi_sz_mm in roi_sizes_mm:
                if roi_sz_mm == 0:
                    values.extend([None] * 3)
                    details_dict['variance_image'].append(None)
                else:
                    roi_sz_pix = round(roi_sz_mm / image_info.pix[0])
                    if roi_sz_pix < 3:
                        roi_sz_pix = 3

                    kernel = np.full((roi_sz_pix, roi_sz_pix),
                                     1./(roi_sz_pix**2))
                    mu = sp.signal.fftconvolve(sub, kernel, mode='valid')
                    ii = sp.signal.fftconvolve(sub ** 2, kernel, mode='valid')
                    variance_sub = ii - mu**2

                    # mask max?
                    masked = False
                    variance_sub_masked = None
                    if roi_array[-1] is not None:
                        # avoid rois where masked max involved
                        grow_max = sp.signal.fftconvolve(
                            1.*roi_array[-1], np.full(kernel.shape, 1), mode='same')
                        max_mask = np.zeros(roi_array[-1].shape, dtype=bool)
                        max_mask[grow_max > 1] = True

                        # valid part of full image, ensure same size as valid
                        valid_shape = variance_sub.shape
                        roi_mask_outer_valid = get_roi_rectangle(
                            image2d.shape,
                            roi_width=valid_shape[1], roi_height=valid_shape[0])
                        rows_ = np.max(roi_mask_outer_valid, axis=1)
                        cols_ = np.max(roi_mask_outer_valid, axis=0)
                        max_mask_valid_part = max_mask[rows_][:, cols_]

                        if np.any(max_mask_valid_part):
                            variance_sub_masked = np.ma.masked_array(
                                variance_sub, mask=max_mask_valid_part)
                            masked = True

                    if masked:
                        max_val = np.ma.max(variance_sub_masked)
                        med_val = np.ma.median(variance_sub_masked)
                        values.extend([max_val, med_val, max_val/med_val])
                        details_dict['variance_image'].append(variance_sub_masked)
                    else:
                        max_val = np.max(variance_sub)
                        med_val = np.median(variance_sub)
                        values.extend([max_val, med_val, max_val/med_val])
                        details_dict['variance_image'].append(variance_sub)

            res = Results(headers=headers, values=values,
                          details_dict=details_dict)
        return res

    try:
        result = locals()[test_code]()
    except (KeyError, IndexError):
        result = None

    return result


def calculate_3d(matrix, marked_3d, input_main, extra_taglists):
    """Calculate tests in 3d mode i.e. tests depending on more than one image.

    Parameters
    ----------
    matrix : list of np.2darray
        list length = all images, [] if not marked for 3d else 2d image
    marked_3d : list of list of str
        for each image which 3d tests to perform
    input_main : object
        of class MainWindow or InputMain containing specific attributes
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

    # NB see code at bottom of this method with
    #       result = locals()[test_code](images_to_test)

    def sum_matrix(images_to_test):
        sum_matrix = matrix[images_to_test[0]]
        for i in images_to_test[1:]:
            sum_matrix = np.add(sum_matrix, matrix[i])
        return sum_matrix

    def CDM(images_to_test):
        """Read CDMAM for multiple images."""
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        #headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['altAll'])

        if len(images_to_test) == 0:
            res = Results(headers=headers)
        else:
            cdmam_table_dict = {}
            errmsgs = []
            details_dicts = []
            prev_phantom = 0
            prev_pix = 0
            n_imgs = len(images_to_test)
            curr_progress_value = 0
            cancelled = False
            roi_dicts = [None for i in range(len(matrix))]
            for i, idx in enumerate(images_to_test):
                try:
                    input_main.progress_modal.setLabelText(
                        f'Finding cell positions image {i}/{n_imgs}')
                    curr_progress_value = round(30 * i/n_imgs)
                    input_main.progress_modal.setValue(
                         curr_progress_value)
                except AttributeError:
                    pass
                image = matrix[idx]
                pix = img_infos[idx].pix[0]  # TODO validate all same pixel size
                roi_dict = None
                if image is not None:
                    roi_dict, err_this = get_rois(image, idx, input_main)
                    if err_this:
                        errmsg.append(f'\tImage {idx}: {errmsg}')
                    elif roi_dict is None:
                        msg = 'Failed reading cell positions. Image ignored.'
                        errmsg.append(f'\tImage {idx}: {msg}')
                    if roi_dict is not None:
                        phantom = 34 if 'include_array' in roi_dict else 40
                        if prev_phantom == 0:
                            prev_phantom = phantom
                            from_yaml_dict = load_cdmam()
                            cdmam_table_dict = from_yaml_dict[f'CDMAM{phantom}']
                            prev_pix = pix
                        if phantom != prev_phantom:
                            errmsg.append(f'\tImage {idx}: Different phantom '
                                          'type than previous')
                            roi_dict = None
                        elif prev_pix != pix:
                            errmsg.append(f'\tImage {idx}: '
                                          'Different pixel sizes.')
                            roi_dict = None
                roi_dicts[idx] = roi_dict
                try:
                    if input_main.progress_modal.wasCanceled():
                        cancelled = True
                        break
                except AttributeError:
                    pass

            if cancelled is False:
                # set templates and kernels
                try:
                    input_main.progress_modal.setLabelText(
                        'Finding disc templates')
                except AttributeError:
                    pass
                pix_new = 0.05  # fixed pixelsize of 50 um for analysis
                pix = img_infos[0].pix[0]
                templates, wi, line_dist = cdmam_methods.get_templates(
                    matrix, roi_dicts, pix, pix_new,
                    paramset.cdm_rotate_k, cdmam_table_dict,
                    paramset.cdm_search_margin)
                kernels = cdmam_methods.get_kernels(cdmam_table_dict, pix_new)

                for i, idx in enumerate(images_to_test):
                    try:
                        input_main.progress_modal.setLabelText(
                            f'Finding discs of image {i}/{n_imgs}')
                        curr_progress_value = 35 + round(60 * i/n_imgs)
                        input_main.progress_modal.setValue(
                             curr_progress_value)
                    except AttributeError:
                        pass
                    if roi_dicts[i] is not None:
                        res = cdmam_methods.read_cdmam_image(
                            matrix[idx], img_infos[idx],
                            roi_dicts[i], cdmam_table_dict, paramset,
                            templates, kernels, wi, line_dist)
                        details_dicts.append(res)

                    try:
                        if input_main.progress_modal.wasCanceled():
                            cancelled = True
                            break
                    except AttributeError:
                        pass

            if cancelled is False:
                try:
                    input_main.progress_modal.setLabelText(
                        'Finishing calculations...')
                    input_main.progress_modal.setValue(95)
                except AttributeError:
                    pass
                if 'include_array' in roi_dict:
                    include_array = roi_dict['include_array']
                else:
                    include_array = None
                dict_updates = cdmam_methods.sum_detection_matrix(
                    details_dicts, include_array,
                    paramset.cdm_center_disc_option)

                cdmam_table_dict.update(dict_updates)
                if details_dicts and n_imgs > 3:
                    cdmam_methods.calculate_fitted_psychometric(
                        cdmam_table_dict, paramset)

                    res_table = cdmam_table_dict['psychometric_results']
                    values = np.array([
                            np.flip(res_table['thickness_predicts_fit_d']),
                            np.flip(res_table['thickness_founds']),
                            np.flip(res_table['thickness_predicts']),
                            np.flip(res_table['thickness_predicts_fit'])
                            ])
                    values = values.T
                else:
                    values = [[None]*4]
                details_dicts.append(cdmam_table_dict)

                res = Results(headers=headers, values=values,
                              details_dict=details_dicts, pr_image=False,
                              errmsg=errmsgs)
            else:
                res = Results(headers=headers, values=[[None]*4],
                              errmsg=errmsgs)
            try:
                input_main.progress_modal.setValue(98)
            except AttributeError:
                pass
        return res

    def Cro(images_to_test):
        """PET cross calibration."""
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        if len(images_to_test) == 0:
            res = Results(headers=headers)
        else:
            try:
                activity_dict = input_main.tab_pet.get_Cro_activities()
            except AttributeError:
                activity_dict = {}
            proceed = True
            activity_inj = None
            inj_time = None
            errmsgs = []
            fmt = '%H%M%S'
            if activity_dict:
                activity_inj = activity_dict['activity_Bq']
                inj_time = activity_dict['activity_time']
            else:
                # ['AcquisitionTime', 'RadionuclideTotalDose',
                # RadiopharmaceuticalStartTime', 'Units']
                if extra_taglists[0][-1] != 'BQML':
                    res = Results(
                        headers=headers,
                        errmsg=f'BQML expected as Unit. Found {extra_taglists[0][0]}')
                    proceed = False
                else:
                    inj_time = datetime.strptime(
                        extra_taglists[0][2].split('.')[0], fmt)
                    activity_inj = int(extra_taglists[0][1]) * 1/1000000  # Bq to MBq
            if proceed:
                acq_time = datetime.strptime(
                    extra_taglists[0][0].split('.')[0], fmt)
                try:
                    time_diff = acq_time - inj_time
                    time_diff_minutes = time_diff.seconds/60
                except TypeError:  # datetime.time from activity_dict
                    tdiff_h = acq_time.hour - inj_time.hour
                    tdiff_m = acq_time.minute - inj_time.minute
                    tdiff_s = acq_time.second - inj_time.second
                    time_diff_minutes = tdiff_h * 60 + tdiff_m + tdiff_s / 60
                activity_at_scan = activity_inj * np.exp(
                    -np.log(2)*time_diff_minutes/HALFLIFE['F18'])
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

    def Def(images_to_test):
        """Defective pixels."""
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt0'])
        if len(images_to_test) == 0:
            res = Results(headers=headers)
        else:
            errmsgs = []
            values = [None]
            details_dicts = []
            
            kernel = np.ones((3,3))
            kernel[1,1] = 0
            kernel = kernel / 8
            kernel_plus = np.zeros((3,3))
            kernel_plus[0, 1] = 1
            kernel_plus[-1, 1] = 1
            kernel_plus[1, 0] = 1
            kernel_plus[1, -1] = 1
            kernel_plus = kernel_plus / 4

            values = []
            n_diff_neighbours_is_zero = np.zeros(
                matrix[images_to_test[0]][1:-1,1:-1].shape)
            n_diff_nearest_is_zero = np.copy(n_diff_neighbours_is_zero)
            idxs_different_shape = []
            for i, sli in enumerate(matrix):
                if sli is None:
                    details_dict = None
                    values_sli = [None] * len(headers)
                else:
                    avg_all_neighbours = sp.signal.fftconvolve(
                        sli, kernel, mode='valid')
                    diff_all = avg_all_neighbours - sli[1:-1,1:-1]
                    diff_neighbours_is_zero = np.zeros(
                        diff_all.shape, dtype=bool)
                    diff_neighbours_is_zero[diff_all == 0] = True

                    avg_nearest_neighbours = sp.signal.fftconvolve(
                        sli, kernel_plus, mode='valid')
                    diff_nearest = avg_nearest_neighbours - sli[1:-1,1:-1]
                    diff_nearest_is_zero = np.zeros(
                        diff_nearest.shape, dtype=bool)
                    diff_nearest_is_zero[diff_nearest == 0] = True

                    pix_pr_cm = round(10./img_infos[i].pix[0])
                    sum_kernel = np.ones((pix_pr_cm, pix_pr_cm), dtype=int)
                    n_pr_roi_neighbours = sp.signal.fftconvolve(
                        diff_neighbours_is_zero, sum_kernel, mode='same')
                    n_pr_roi_nearest = sp.signal.fftconvolve(
                        diff_nearest_is_zero, sum_kernel, mode='same')
                    details_dict = {
                        'diff_neighbours_is_zero': diff_neighbours_is_zero,
                        'diff_nearest_is_zero': diff_nearest_is_zero,
                        'n_pr_roi_neighbours': n_pr_roi_neighbours,
                        'n_pr_roi_nearest': n_pr_roi_nearest
                        }
                    values_sli = [
                        np.count_nonzero(diff_neighbours_is_zero),
                        np.count_nonzero(diff_nearest_is_zero)]
                    if n_diff_neighbours_is_zero.shape != diff_neighbours_is_zero.shape:
                        idxs_different_shape.append(i)
                    else:
                        n_diff_neighbours_is_zero += 1.* diff_neighbours_is_zero
                        n_diff_nearest_is_zero += 1. * n_diff_nearest_is_zero
                values.append(values_sli)
                details_dicts.append(details_dict)

            n_images = len(images_to_test)
            if len(idxs_different_shape) > 0:
                errmsgs.append(f'Image number {idxs_different_shape} have a '
                               'different shape compared to first selected '
                               f'image ({images_to_test[0]}). Ignored when '
                               'calculating fraction of all selected.')
                n_images = n_images - len(idxs_different_shape)

            details_dicts.append({
                'frac_diff_neighbours_is_zero':
                    1/n_images * n_diff_neighbours_is_zero,
                'frac_diff_nearest_is_zero':
                    1/n_images * n_diff_nearest_is_zero
                })

            values_sup = []
            for _, arr in details_dicts[-1].items():
                values_sup.append(np.max(arr))
                idxs_max = np.where(arr == np.max(arr))
                values_sup.append(idxs_max[0].size)

            res = Results(headers=headers, values=values,
                          headers_sup=headers_sup, values_sup=[values_sup],
                          details_dict=details_dicts,
                          pr_image=True, pr_image_sup=False, errmsg=errmsgs)

        return res

    def DPR(images_to_test):
        """D-prime / detectability from TTF NPS results."""
        headers = []#copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        proceed = False
        ttf_details = None
        nps_details = None
        proceed = False
        if len(images_to_test) == 0:
            res = Results(headers=headers)
        else:
            if 'TTF' in input_main.results and 'NPS' in input_main.results:
                try:
                    ttf_details = input_main.results['TTF']['details_dict']
                    #breakpoint() not finished
                    nps_details = [dd for dd
                                   in input_main.results['NPS']['details_dict']
                                   if dd]
                    pix = img_infos[images_to_test[0]].pix[0]#TODO verify NPS and TTF images same pixelsize
                    power = None if paramset.dpr_designer is False else paramset.dpr_power
                    diameter_in_pix = paramset.dpr_size / pix
                    nps_sum = nps_details[0]['NPS_array']
                    for idx in range(1, len(nps_details)):
                        nps_sum = nps_sum + nps_details[idx]['NPS_array']
                    nps_2d_avg = nps_sum / len(nps_details)

                    w_task = mmcalc.get_w_task(
                        paramset.dpr_contrast, diameter_in_pix, power)
                    if paramset.dpr_gaussian_ttf:
                        ttf_2d = ttf_details#TODO....
                        # select ttf from material with closest contrast
                        # or option to override this if crappy data?
                        # alternative dlete crappy data from ttf dataset to override
                        # warning of crappy data possible?
                    proceed = True
                except KeyError:
                    pass
        if proceed:
            values = []
            details_dict_all = []
            errmsgs = []
            detail_dict_all = {}
            res = Results(headers=headers, values=values,
                          details_dict=details_dict_all,
                          pr_image=False, pr_image_sup=False,
                          errmsg=errmsgs)
        else:
            errmsgs = ['Missing TTF or NPS results (or both).']
            res = Results(headers=headers, errmsg=errmsgs)
        return res

    def MTF(images_to_test):
        res = None
        proceed = False
        if modality in ['CT', 'SPECT', 'PET']:
            alt = paramset.mtf_type
            try:
                headers = copy.deepcopy(
                    HEADERS[modality][test_code]['alt' + str(alt)])
            except KeyError:
                headers = copy.deepcopy(
                    HEADERS[modality][test_code]['altAll'])
            try:
                headers_sup = copy.deepcopy(
                    HEADERS_SUP[modality][test_code]['alt' + str(alt)])
            except KeyError:
                headers_sup = copy.deepcopy(
                    HEADERS_SUP[modality][test_code]['altAll'])
            if len(images_to_test) == 0:
                res = Results(headers=headers, headers_sup=headers_sup)
            else:
                proceed = True

        if proceed:
            details_dict = {}
            if alt == 4:
                roi_array, errmsg = get_rois(
                    matrix[images_to_test[0]], images_to_test[0], input_main)
            else:
                sum_image = matrix[images_to_test[0]]
                if len(images_to_test) > 1:
                    for imgNo in images_to_test[1:]:
                        sum_image = np.add(sum_image, matrix[imgNo])
                roi_array, errmsg = get_rois(
                    sum_image, images_to_test[0], input_main)
            if alt == 2 and modality == 'CT':
                roi_array_inner = roi_array
            elif alt == 4:
                roi_array_inner = roi_array
            else:
                roi_array_inner = roi_array[0]

            background_subtract = False
            if modality == 'CT' and paramset.mtf_type == 1:
                background_subtract = True
            elif modality in ['PET', 'SPECT'] and paramset.mtf_type < 3:
                background_subtract = True

            rows = np.max(roi_array_inner, axis=1)
            cols = np.max(roi_array_inner, axis=0)
            sub = []
            for sli in matrix:
                if sli is not None:
                    this_sub = sli[rows][:, cols]
                    if background_subtract:
                        arr = np.ma.masked_array(
                            sli, mask=np.invert(roi_array[1]))
                        this_sub = this_sub - np.mean(arr)
                    sub.append(this_sub)
                else:
                    sub.append(None)

            pr_image = False
            if paramset.mtf_type == 1:  # wire/line
                if len(images_to_test) > 2:
                    details_dict, errmsg = mtf_methods.calculate_MTF_3d_line(
                        sub, roi_array_inner[rows][:, cols], images_to_test,
                        img_infos, paramset)
                else:
                    errmsg = 'At least 3 images required for MTF wire/line in 3d'
            elif paramset.mtf_type == 2:
                if modality == 'CT': # circular disc
                    details_dict, errmsg = mtf_methods.calculate_MTF_circular_edge(
                        sub, roi_array[rows][:, cols],
                        img_infos[images_to_test[0]].pix[0],
                        paramset, images_to_test)
                    if details_dict['disc_radius_mm'] is not None:
                        row0 = np.where(rows)[0][0]
                        col0 = np.where(cols)[0][0]
                        dx_dy = (
                            col0 + details_dict['center_xy'][0] - sum_image.shape[1]//2,
                            row0 + details_dict['center_xy'][1] - sum_image.shape[0]//2
                            )
                        roi_disc = get_roi_circle(
                            sum_image.shape, dx_dy,
                            details_dict['disc_radius_mm'] / img_infos[
                                images_to_test[0]].pix[0])
                        details_dict['found_disc_roi'] = roi_disc
                    details_dict = [details_dict]
                else:  # SPECT/PET line source sliding window
                    pr_image = True
                    if len(images_to_test) > 2:
                        details_dict, errmsg = mtf_methods.calculate_MTF_3d_line(
                            sub, roi_array_inner[rows][:, cols], images_to_test,
                            img_infos, paramset)
                    else:
                        errmsg = 'At least 3 images required for MTF wire/line in 3d'
            elif paramset.mtf_type >= 3:  # z-resolution
                pr_image = False
                if len(images_to_test) > 4:
                    if paramset.mtf_type == 3:  # line
                        details_dict, errmsg = mtf_methods.calculate_MTF_3d_line(
                            sub, roi_array_inner[rows][:, cols],
                            images_to_test, img_infos, paramset)
                    elif paramset.mtf_type == 4:  # edge
                        details_dict, errmsg = mtf_methods.calculate_MTF_3d_z_edge(
                            sub, roi_array_inner[rows][:, cols],
                            images_to_test, img_infos, paramset)
                else:
                    errmsg = 'At least 5 images required for z-resolution'

            values = None
            values_sup = None
            prefix = 'g' if paramset.mtf_gaussian else 'd'
            if pr_image:
                offset_max = []
                pix = img_infos[images_to_test[0]].pix[0]
                for sli in matrix:
                    if sli is not None:
                        this_offset = mmcalc.get_offset_max_pos_2d(
                            sli, roi_array_inner, pix)
                        offset_max.append(this_offset)
                    else:
                        offset_max.append((None, None))
                # add to common details (last dict):
                details_dict[-1].update({'offset_max': offset_max})

                values = []
                values_sup = []
                for idx in images_to_test:
                    dd = details_dict[idx]
                    if dd:
                        try:
                            values.append(
                                dd[0][prefix + 'MTF_details']['values'])
                        except TypeError:
                            pass
                        try:
                            values_sup.append(
                                list(dd[0]['gMTF_details']['LSF_fit_params']))
                        except TypeError:
                            pass
                        try:  # x-dir, y-dir
                            values[-1].extend(
                                dd[1][prefix + 'MTF_details']['values'])
                            values_sup[-1].extend(list(
                                dd[1]['gMTF_details']['LSF_fit_params']))
                        except (IndexError, KeyError, AttributeError, TypeError):
                            pass
                        values_sup[-1].append(
                            dd[0]['gMTF_details']['prefilter_sigma'])
                        values_sup[-1].extend(list(offset_max[idx]))
                    else:
                        values.append([None] * len(headers))
                        sup_this = [None] * len(headers_sup)
                        sup_this[-2:] = list(offset_max[idx])
                        values_sup.append(sup_this)
            else:
                try:
                    values = details_dict[0][prefix + 'MTF_details']['values']
                except (TypeError, KeyError):
                    pass
                try:
                    values_sup = list(details_dict[0][
                        'gMTF_details']['LSF_fit_params'])
                except (TypeError, KeyError):
                    pass
                try:  # x-dir, y-dir or  2 lines
                    values.extend(
                        details_dict[1][prefix + 'MTF_details']['values'])
                    values_sup.extend(list(
                        details_dict[1]['gMTF_details']['LSF_fit_params']))
                except (IndexError, KeyError, AttributeError):
                    if paramset.mtf_type == 3 and values:
                        values.extend([None] * len(values))
                        values_sup.extend([None] * len(values_sup))
                values=[values]
                try:
                    values_sup.append(
                        details_dict[0]['gMTF_details']['prefilter_sigma'])
                except (IndexError, KeyError):
                    values_sup.append(None)
                values_sup=[values_sup]

            res = Results(headers=headers, values=values,
                          headers_sup=headers_sup, values_sup=values_sup,
                          details_dict=details_dict,
                          alternative=paramset.mtf_type,
                          pr_image=pr_image, pr_image_sup=pr_image,
                          errmsg=errmsg)

        return res

    def Rec(images_to_test):
        """PET Recovery Curve."""
        alt = paramset.rec_type
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt' + str(alt)])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt0'])
        errmsgs = []
        proceed = True
        if extra_taglists[0][-1] != 'BQML':
            errmsgs.append(f'BQML expected as Unit. Found {extra_taglists[0][0]}')
        try:
            activity_dict = input_main.tab_pet.get_Rec_activities()
            if not activity_dict:
                errmsgs.append(
                    'Missing input on activity, time or volume. '
                    'Recovery coefficients will not be calculated.')
        except AttributeError:
            activity_dict = {}
            errmsgs.append(
                'Automation not available for Recovery Curve, GUI input needed')
            proceed = False
        try:
            if input_main.test_mode:
                proceed = True  # force proceed without activity_dict
        except AttributeError:
            pass
        if len(images_to_test) < 3:
            if len(images_to_test) > 0:
                errmsgs.append('Too few images to analyse.')
            res = Results(headers=headers, headers_sup=headers_sup, errmsg=errmsgs)
        elif proceed is False:
            res = Results(headers=headers, headers_sup=headers_sup, errmsg=errmsgs)
        else:
            try:
                summed = sum_matrix(images_to_test)
                roi_array, errmsg = get_rois(
                    summed, images_to_test[0], input_main)
            except ValueError as err:
                errmsgs.append(
                    f'Failed summing images. {str(err)} Calculation aborted.')
                proceed = False

        if proceed:
            # find z-profile of first background roi
            avgs_background = []
            for image in matrix:
                if image is not None:
                    arr = np.ma.masked_array(image, mask=np.invert(roi_array[0]))
                    avgs_background.append(np.mean(arr))
            zpos = [float(img_info.zpos) for img_info in input_main.imgs]
            zpos = np.array(zpos)[images_to_test]
            sort_idx = np.argsort(zpos)
            zpos = zpos[sort_idx]
            avgs = np.array(avgs_background)[sort_idx]
            details_dict = {'roi_averages': avgs, 'zpos': zpos}

            # find slices to include based on z-profile of background
            avgs_bg = np.copy(avgs)
            zpos_bg = np.copy(zpos)
            if paramset.rec_auto_select_slices:
                if avgs[0] < max(avgs)/2 and avgs[-1] < max(avgs)/2:
                    width, center = mmcalc.get_width_center_at_threshold(
                        avgs, max(avgs)/2, force_above=True)
                    if center is None or width is None:
                        errmsgs.append(
                            'Auto select slice failed. Could not find FWHM. '
                            'All images used for background.')
                    else:
                        center_slice = round(center)
                        width = 0.01 * width * paramset.rec_percent_slices
                        start_slice = center_slice - round(width/2)
                        stop_slice = center_slice + round(width/2)
                        avgs_bg = avgs_bg[start_slice:stop_slice + 1]
                        zpos_bg = zpos_bg[start_slice:stop_slice + 1]
                else:
                    errmsgs.append('Outer slices have average above half max. '
                                   'Auto select slice ignored. '
                                   'All images used for background.')
            details_dict['used_roi_averages'] = avgs_bg
            details_dict['used_zpos'] = zpos_bg

            # find slices with spheres
            matrix_used = []
            for idx in sort_idx:
                matrix_used.append(matrix[idx])
            maxs = [np.max(image) for image in matrix_used]
            max_idx = np.where(maxs == np.max(maxs))
            zpos_max = zpos[max_idx[0]]
            diff_z = np.abs(zpos - zpos_max)
            idxs = np.where(diff_z < paramset.rec_sphere_diameters[-1])  # largest diam
            start_slice = np.min(idxs)
            stop_slice = np.max(idxs)
            maxs_sph = maxs[start_slice:stop_slice]
            zpos_sph = zpos[start_slice:stop_slice]
            details_dict['used_roi_maxs'] = maxs_sph
            details_dict['used_zpos_spheres'] = zpos_sph
            details_dict['max_slice_idx'] = np.where(zpos == zpos_max)[0][0]

            # get background from each roi
            if paramset.rec_background_full_phantom:
                start_slice_bg = 0
                stop_slice_bg = len(matrix_used) - 1
            else:
                start_slice_bg = start_slice
                stop_slice_bg = stop_slice
            background_values = []
            for i in range(start_slice_bg, stop_slice_bg + 1):
                if matrix_used[i] is not None:
                    for background_roi in roi_array[:-1]:
                        arr = np.ma.masked_array(
                            matrix_used[i], mask=np.invert(background_roi))
                        background_values.append(np.mean(arr))
            background_value = np.mean(background_values)
            details_dict['background_values'] = background_values

            # calculate sphere values
            res_dict, errmsg = calculate_recovery_curve(
                matrix_used[start_slice:stop_slice], input_main.imgs[0],
                roi_array[-1], zpos_sph, paramset, background_value)
            if errmsg is not None:
                errmsgs.append(errmsg)
            n_spheres = len(paramset.rec_sphere_diameters)
            if res_dict:
                details_dict.update(res_dict)
                values_sup = []
                rec_type = paramset.rec_type
                fmt = '%H%M%S'
                acq_times = [t[0] for t in extra_taglists]
                acq_times.sort()
                scan_start = acq_times[0].split('.')[0]
                if activity_dict:
                    acq_time = datetime.strptime(scan_start, fmt)
                    tdiff_h = acq_time.hour - activity_dict['sphere_time'].hour
                    tdiff_m = acq_time.minute - activity_dict['sphere_time'].minute
                    tdiff_s = acq_time.second - activity_dict['sphere_time'].second
                    time_diff = tdiff_h * 60 + tdiff_m + tdiff_s / 60
                    sph_act_at_scan = activity_dict['sphere_Bq_ml'] * np.exp(
                        -np.log(2)*time_diff/HALFLIFE['F18'])
                    tdiff_h = acq_time.hour - activity_dict['background_time'].hour
                    tdiff_m = acq_time.minute - activity_dict['background_time'].minute
                    tdiff_s = acq_time.second - activity_dict['background_time'].second
                    time_diff = tdiff_h * 60 + tdiff_m + tdiff_s / 60
                    bg_act_at_scan = activity_dict['background_Bq_ml'] * np.exp(
                        -np.log(2)*time_diff/HALFLIFE['F18'])
                    rc_values_all = []
                    for i in range(3):
                        img_values = np.array(details_dict['values'][i])
                        rc_values = img_values[:n_spheres] / sph_act_at_scan
                        rc_values = list(rc_values) + [img_values[-1] / bg_act_at_scan]
                        rc_values_all.append(rc_values)
                    values_sup = [scan_start, sph_act_at_scan, bg_act_at_scan]
                else:
                    rc_values = [None for i in range(n_spheres + 1)]
                    rc_values_all = [rc_values for i in range(3)]
                    rec_type = rec_type + 3  # image values, not RC values
                    headers = copy.deepcopy(
                        HEADERS[modality][test_code]['alt' + str(rec_type)])
                    try:
                        input_main.tab_pet.rec_type.setCurrentIndex(rec_type)
                    except AttributeError:
                        pass
                    values_sup = [scan_start, None, None]

                details_dict['values'] = rc_values_all + details_dict['values']
                values = details_dict['values'][rec_type]
                res = Results(headers=headers, values=[values],
                              headers_sup=headers_sup,
                              values_sup=[values_sup],
                              details_dict=details_dict,
                              pr_image=False, pr_image_sup=False,
                              errmsg=errmsgs)
            else:
                res = Results(headers=headers,
                              headers_sup=headers_sup,
                              errmsg=errmsgs)
        else:
            res = Results(headers=headers,
                          headers_sup=headers_sup,
                          errmsg=errmsgs)

        return res

    def SNI(images_to_test):
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
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
            values, _, details_dict2, _ = nm_methods.calculate_NM_SNI(
                image_input, roi_array, img_infos[0], paramset, None)
            details_dict.update(details_dict2)

            res = Results(headers=headers, values=[values],
                          details_dict=[details_dict], errmsg=errmsgs, pr_image=False)

        return res

    def SNR(images_to_test):
        # use two and two images
        values_first_imgs = []
        idxs_first_imgs = []
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
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

    def Swe(images_to_test):
        """NM sweep test."""
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt0'])
        if len(images_to_test) == 132:
            roi_array, errmsg = get_rois(matrix[0], 0, input_main)
            details_dicts, errmsgs = calculate_NM_sweep(
                matrix, img_infos, roi_array, paramset)
            try:
                values = (
                    details_dicts[0]['uni_values'] +
                    details_dicts[1]['uni_values'])
                values_sup = [
                    np.median(details_dicts[0]['fwhm_matrix']),
                    np.median(details_dicts[1]['fwhm_matrix']),
                    details_dicts[0]['fwhm_iu_ufov'],
                    details_dicts[0]['fwhm_iu_cfov'],
                    details_dicts[1]['fwhm_iu_ufov'],
                    details_dicts[1]['fwhm_iu_cfov'],
                    details_dicts[0]['diff_max_ufov'],
                    details_dicts[0]['diff_max_cfov'],
                    details_dicts[1]['diff_max_ufov'],
                    details_dicts[1]['diff_max_cfov'],
                    ]

                res = Results(headers=headers, values=[values],
                              headers_sup=headers_sup, values_sup=[values_sup],
                              details_dict=details_dicts,
                              pr_image=False, pr_image_sup=False,
                              errmsg=errmsgs)
            except KeyError:
                res = Results(headers=headers, errmsg=errmsgs)
        else:
            errmsg = 'Test failed: Expecting 132 images AutoQC Sweep verification'
            res = Results(headers=headers, errmsg=errmsg)

        return res

    def TTF(images_to_test):
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt0'])
        materials = paramset.ttf_table.labels
        if len(images_to_test) == 0:
            res = Results(headers=headers, headers_sup=headers_sup)
        elif len(materials) == 0:
            errmsgs = ['Missing ROIs for TTF. Materials not defined.']
            res = Results(headers=headers, headers_sup=headers_sup,
                          errmsg=errmsgs)
        else:
            details_dict_all = []  # list of dict
            values = []
            values_sup = []
            errmsgs = []
            prefix = 'g' if paramset.ttf_gaussian else 'd'

            sum_image = matrix[images_to_test[0]]
            if len(images_to_test) > 1:
                for imgNo in images_to_test[1:]:
                    sum_image = np.add(sum_image, matrix[imgNo])
            roi_array_all, errmsg = get_rois(
                sum_image, images_to_test[0], input_main)

            paramset_ttf_fix = copy.deepcopy(paramset)
            paramset_ttf_fix.mtf_cut_lsf = paramset_ttf_fix.ttf_cut_lsf
            paramset_ttf_fix.mtf_cut_lsf_w = paramset_ttf_fix.ttf_cut_lsf_w
            paramset_ttf_fix.mtf_cut_lsf_w_fade = paramset_ttf_fix.ttf_cut_lsf_w_fade
            contrasts = []
            for idx, roi_array in enumerate(roi_array_all):
                rows = np.max(roi_array, axis=1)
                cols = np.max(roi_array, axis=0)
                sub = []
                for sli in matrix:
                    if sli is not None:
                        this_sub = sli[rows][:, cols]
                        sub.append(this_sub)
                    else:
                        sub.append(None)

                details_dict, errmsg = mtf_methods.calculate_MTF_circular_edge(
                    sub, roi_array[rows][:, cols],
                    img_infos[images_to_test[0]].pix[0],
                    paramset_ttf_fix, images_to_test)
                if errmsg:
                    errmsgs.append(f'{materials[idx]}:')
                    errmsgs.extend(errmsg)
                if details_dict['disc_radius_mm'] is not None:
                    row0 = np.where(rows)[0][0]
                    col0 = np.where(cols)[0][0]
                    dx_dy = (
                        col0 + details_dict['center_xy'][0] - sum_image.shape[1]//2,
                        row0 + details_dict['center_xy'][1] - sum_image.shape[0]//2
                        )
                    roi_disc = get_roi_circle(
                        sum_image.shape, dx_dy,
                        details_dict['disc_radius_mm'] / img_infos[
                            images_to_test[0]].pix[0])
                    details_dict['found_disc_roi'] = roi_disc
                details_dict_all.append(details_dict)
                contrast =(
                    np.max(details_dict['interpolated'])
                    - np.min(details_dict['interpolated']))

                try:
                    values.append(
                        [materials[idx]]
                        + details_dict[prefix + 'MTF_details']['values']
                        + [contrast])
                except TypeError:
                    values.append([materials[idx]] + [None] * 3)
                try:
                    values_sup.append(
                        [materials[idx]]
                        + list(details_dict['gMTF_details']['LSF_fit_params'])
                        + [details_dict['gMTF_details']['prefilter_sigma']])
                except TypeError:
                    values_sup.append([materials[idx]] + [None] * 5)

            res = Results(headers=headers, values=values,
                          headers_sup=headers_sup, values_sup=values_sup,
                          details_dict=details_dict_all,
                          pr_image=False, pr_image_sup=False,
                          errmsg=errmsgs)

        return res

    def Uni(images_to_test):
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        errmsgs = []
        if len(images_to_test) == 0:
            res = Results(headers=headers)
        else:
            image_input = sum_matrix(images_to_test)  # always if 3d option
            if paramset.uni_correct:
                errmsgs.append('Point source correction ignored when image sum used.')
            roi_array, errmsg = get_rois(
                image_input, images_to_test[0], input_main)
            res = nm_methods.calculate_NM_uniformity(
                image_input, roi_array, img_infos[0].pix[0], paramset.uni_scale_factor)
            details_dict = {
                'sum_image': image_input,
                'du_matrix': res['du_matrix'],
                'matrix_ufov': res['matrix_ufov']
                }
            values = res['values']

            res = Results(
                headers=headers, values=[values],
                details_dict=[details_dict],
                errmsg=errmsgs, pr_image=False)

        return res

    for test_code in all_tests:
        images_to_test = []
        for i, tests in enumerate(marked_3d):
            if test_code in tests:
                if matrix[i] is not None:
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
    modality : str
        as used in imageQC

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
        elif paramset.sli_type == 3:
            details_dict['labels'] = ['line']
        elif paramset.sli_type == 4:
            details_dict['labels'] = ['upper', 'lower']
    else:  # MR
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
            profile = -1. * profile
            background = -1. * background
        peak_value = np.max(profile)
        halfmax = 0.5 * peak_value
        if sli_signal_low_density:
            details_dict['peak'].append(-(peak_value + background))
            details_dict['halfpeak'].append(-(halfmax + background))
        else:
            details_dict['peak'].append(peak_value + background)
            details_dict['halfpeak'].append(halfmax + background)

        if modality == 'CT':
            if paramset.sli_type in [0, 3, 4]:  # wire ramp Catphan or Siemens
                width, center = mmcalc.get_width_center_at_threshold(
                    profile, halfmax, force_above=True)
                if width is not None:
                    slice_thickness = paramset.sli_tan_a * width * img_info.pix[0]
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
                if local_max.size > 0:
                    new_x, envelope_y = mmcalc.resample(profile[local_max], local_max,
                                                        1, n_steps=len(profile))
                    width, center = mmcalc.get_width_center_at_threshold(
                        envelope_y, halfmax)
                    if sli_signal_low_density:
                        details_dict['envelope_profiles'].append(
                            -(envelope_y + background))
                    else:
                        details_dict['envelope_profiles'].append(
                            envelope_y + background)
                else:
                    width = None

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
    n_avg = paramset.sli_average_width
    if isinstance(paramset, cfc.ParamSetCT):
        sli_signal_low_density = True if paramset.sli_type == 2 else False
        n_search = round(paramset.sli_search_width)
    else:
        sli_signal_low_density = False
        n_search = n_avg

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
            p_start = np.max([max_i-n_avg, 0])
            p_stop = np.min([len(profiles), max_i+n_avg+1])
            profile = np.mean(profiles[p_start:p_stop], axis=0)

    else:
        rr, cc = skimage.draw.line(r0, c0, r1, c1)
        profile = image[rr, cc]

    if isinstance(paramset, cfc.ParamSetMR):
        if paramset.sli_type == 1:  # wedge
            profile = np.diff(profile)
        if paramset.sli_sigma > 0:
            profile = sp.ndimage.gaussian_filter(profile, sigma=paramset.sli_sigma)
    else:  # CT
        if paramset.sli_median_filter > 0:
            profile = sp.ndimage.median_filter(profile, size=paramset.sli_median_filter)

    return (profile, errmsg)


def calculate_NPS(image2d, roi_array, img_info, paramset, modality='CT'):
    """Calculate NPS for each roi in array.

    Parameters
    ----------
    image2d : numpy.ndarray
        input image full resolution
    roi_array : list of numpy.ndarray
    img_info :  DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetCT, ParamsetXray or ParamSetMammo
    modality: str
        'CT', 'Xray' or 'Mammo'

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
        if Xray/Mammo:
        freq: frequency values x -dir
        u_profile: power spectrum x-dir
        v_profile: power spectrum y-dir
    """
    try:
        nps_sampling_frequency = paramset.nps_sampling_frequency
    except AttributeError:  # task based
        nps_sampling_frequency = paramset.ttf_sampling_frequency

    def smooth_profile(profile):
        kernel_size = round(paramset.nps_smooth_width / nps_sampling_frequency)
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
            NPS, pix=unit, step_size=nps_sampling_frequency)
        if paramset.nps_smooth_width > 0:
            radial_profile = smooth_profile(radial_profile)
        AUC = np.sum(radial_profile) * nps_sampling_frequency
        median_frequency, median_val = mmcalc.find_median_spectrum(
            freq, radial_profile)
        values = [median_frequency, AUC, np.sum(NPS)*unit ** 2,
                  large_area_signal, stdev_img]
        details_dict['median_freq'] = median_frequency
        details_dict['median_val'] = median_val

    elif modality in ['Xray', 'Mammo']:
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
            step_size=nps_sampling_frequency)
        freq, radial_profile = mmcalc.get_radial_profile(
            NPS, pix=unit, step_size=nps_sampling_frequency,
            start_dist=3*unit)
        AUC = np.sum(radial_profile) * nps_sampling_frequency
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


def calculate_flatfield_aapm(image2d, mask_outer, image_info, paramset):
    """Calculate homogeneity Mammo according to EU guidelines.

    Parameters
    ----------
    image2d : numpy.ndarray
        input image
    mask_outer : numpy.ndarray
        mask of outer mm if paramset.hom_mask_outer_mm > 0
    image_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : cfc.ParamSetMammo

    Returns
    -------
    details : dict
    """
    roi_size_in_pix = round(paramset.hom_roi_size / image_info.pix[0])
    if mask_outer is not None:
        rows = np.max(mask_outer, axis=1)
        cols = np.max(mask_outer, axis=0)
        sub = image2d[rows][:, cols]
    else:
        sub = image2d

    # similar code to find variance image using fftconvolve
    kernel = np.full((roi_size_in_pix, roi_size_in_pix),
                     1./(roi_size_in_pix**2))
    mu = sp.signal.fftconvolve(sub, kernel, mode='valid')
    ii = sp.signal.fftconvolve(sub ** 2, kernel, mode='valid')
    noise_sub = np.sqrt(ii - mu**2)

    margin_valid = (np.array(sub.shape) - np.array(mu.shape)) // 2
    sub_valid_part = sub[margin_valid[0]:margin_valid[0] + mu.shape[0],
                         margin_valid[1]:margin_valid[1] + mu.shape[1]]
    diff_valid_part = np.abs(sub_valid_part - mu)  # abs. difference from mean
    anomalous_pixels = np.zeros(mu.shape, dtype=bool)
    anomalous_pixels = (
        diff_valid_part > paramset.hom_anomalous_factor * noise_sub)
    sum_kernel = np.ones((roi_size_in_pix, roi_size_in_pix), dtype=int)
    anomalous_pixels_pr_roi = sp.signal.fftconvolve(
        anomalous_pixels, sum_kernel, mode='same')

    uniformity_arrays = []
    global_uniformity = []

    neighbour = roi_size_in_pix // 2
    for input_array in [mu, noise_sub, np.divide(mu, noise_sub)]:
        uniformity_arrays.append(mmcalc.get_uniformity_map(
            input_array, neighbour_start=neighbour, neighbour_end=neighbour))
        global_uniformity.append(
            100. * (np.max(input_array) - np.min(input_array))
            / np.mean(input_array))

    # correlated noise TG150 4.3.9
    sd_col_row = []
    diff_profile_col_row = []
    avg_noise = np.mean(noise_sub)
    for i in range(2):
        avg_profile = np.mean(sub, axis=i)  # col then row
        std = np.sum((avg_profile - np.mean(avg_profile))**2)
        std = np.sqrt(std/(avg_profile.size - 1))
        sd_col_row.append(std / avg_noise)

        avg_profile = np.concatenate(
            ([avg_profile[0]], avg_profile, [avg_profile[-1]]))
        diff = (avg_profile 
                - 0.5* (np.roll(avg_profile, 1) + np.roll(avg_profile, -1)))
        diff_profile_col_row.append(diff[1:-1])

    details_dict = {
        'averages': mu,
        'stds': noise_sub,
        'anomalous_pixels': anomalous_pixels,
        'n_anomalous_pixels': np.count_nonzero(anomalous_pixels),
        'n_anomalous_pixels_pr_roi': anomalous_pixels_pr_roi,
        'local_uniformities': uniformity_arrays[0],
        'local_noise_uniformities': uniformity_arrays[1],
        'local_snr_uniformities': uniformity_arrays[2],
        'global_uniformity': global_uniformity[0],
        'global_noise_uniformity': global_uniformity[1],
        'global_snr_uniformity': global_uniformity[2],
        'relSDrow': sd_col_row[1], 'relSDcol': sd_col_row[0],
        'diff_neighbours_profile_col_row': diff_profile_col_row
        }

    return details_dict

def calculate_flatfield_mammo(image2d, mask_outer, image_info, paramset):
    """Calculate homogeneity Mammo according to EU guidelines.

    Parameters
    ----------
    image2d : numpy.ndarray
        input image
    mask_outer : numpy.ndarray
        mask of outer mm if paramset.hom_mask_outer_mm > 0
    image_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : cfc.ParamSetMammo

    Returns
    -------
    details : dict
    """
    details = {}

    # set result matrix
    roi_size_in_pix = round(paramset.hom_roi_size / image_info.pix[0])
    if roi_size_in_pix % 2 == 1:
        roi_size_in_pix += 1  # ensure even number roi size to move roi by half
    offset = 0  # left here if to be used later
    large_roi_shape = (np.array(image_info.shape) - 2 * offset)
    n_steps = np.ceil(large_roi_shape / (roi_size_in_pix / 2))  # shift < half ROI
    step_size = large_roi_shape / n_steps
    xstarts = np.arange(offset, image_info.shape[1] - (offset + roi_size_in_pix) + 1,
                        step=step_size[1])
    ystarts = np.arange(offset, image_info.shape[0] - (offset + roi_size_in_pix) + 1,
                        step=step_size[0])
    xstarts = [round(val) for val in list(xstarts)]
    ystarts = [round(val) for val in list(ystarts)]
    xstarts[-1] = image_info.shape[1] - 1 - (offset + roi_size_in_pix)
    ystarts[-1] = image_info.shape[0] - 1 - (offset + roi_size_in_pix)
    matrix_shape = (len(ystarts), len(xstarts))

    roi_mask = np.full(image2d.shape[0:2], False)
    if paramset.hom_mask_max:
        roi_mask[image2d == np.max(image2d)] = True
    if paramset.hom_mask_outer_mm > 0:
        roi_mask[mask_outer == False] = True
    masked_image = np.ma.masked_array(image2d, mask=roi_mask)

    avgs = np.zeros(matrix_shape)
    snrs = np.zeros(matrix_shape)
    variances = np.zeros(matrix_shape)
    masked_roi_matrix = np.full(matrix_shape, False)

    roi_var_in_pix = round(paramset.hom_roi_size_variance / image_info.pix[0])
    if roi_var_in_pix < 3:
        roi_var_in_pix = 3
    kernel = np.full((roi_var_in_pix, roi_var_in_pix),
                     1./(roi_var_in_pix**2))

    max_masked = 0
    if paramset.hom_ignore_roi_percent > 0:
        n_pix_roi = roi_size_in_pix ** 2
        max_masked = paramset.hom_ignore_roi_percent/100. * n_pix_roi
    for j, ystart in enumerate(ystarts):
        for i, xstart in enumerate(xstarts):
            sub = masked_image[ystart:ystart+roi_size_in_pix,
                               xstart:xstart+roi_size_in_pix]
            if np.ma.count_masked(sub) <= max_masked:
                avg_sub = np.ma.mean(sub)
                avgs[j, i] = avg_sub
                std_sub = np.ma.std(sub)
                if std_sub != 0 and roi_var_in_pix < roi_size_in_pix:
                    snrs[j, i] = avg_sub / std_sub
                    if np.ma.count_masked(sub) == 0:
                        if paramset.hom_variance:
                            mu = sp.signal.fftconvolve(
                                sub, kernel, mode='same')
                            ii = sp.signal.fftconvolve(
                                sub ** 2, kernel, mode='same')
                            variance_sub = ii - mu**2
                            variances[j, i] = np.mean(variance_sub)
                else:
                    variances[j, i] = np.inf
            else:
                masked_roi_matrix[j, i] = True

    if paramset.hom_variance is False:
        variances = None

    n_masked_rois = np.count_nonzero(masked_roi_matrix)
    n_rois = avgs.size - n_masked_rois
    n_masked_pixels = np.count_nonzero(roi_mask)
    n_pixels = image2d.size - n_masked_pixels

    masked_avgs = np.ma.masked_array(avgs, mask=masked_roi_matrix)
    masked_snrs = np.ma.masked_array(snrs, mask=masked_roi_matrix)
    overall_avg = np.ma.mean(masked_avgs)
    diff_avgs = 100 / overall_avg * (avgs - overall_avg)
    diff_snrs = 100 / np.ma.mean(masked_snrs) * (snrs - np.ma.mean(masked_snrs))
    deviating_rois = np.zeros(matrix_shape, dtype=bool)
    deviating_rois[np.logical_or(
        np.abs(diff_avgs) > paramset.hom_deviating_rois,
        np.abs(diff_snrs) > paramset.hom_deviating_rois)] = True
    deviating_rois[masked_roi_matrix == True] = False
    n_dev_rois = np.count_nonzero(deviating_rois)

    deviating_avgs = np.zeros(matrix_shape, dtype=bool)
    deviating_avgs[np.abs(diff_avgs) > paramset.hom_deviating_rois] = True
    deviating_avgs[masked_roi_matrix == True] = False
    deviating_snrs = np.zeros(matrix_shape, dtype=bool)
    deviating_snrs[np.abs(diff_snrs) > paramset.hom_deviating_rois] = True
    deviating_snrs[masked_roi_matrix == True] = False
    n_dev_avgs = np.count_nonzero(deviating_avgs)
    n_dev_snrs = np.count_nonzero(deviating_snrs)

    if variances is not None:
        variances = np.ma.masked_array(variances, mask=masked_roi_matrix)
    diff_avgs = np.ma.masked_array(diff_avgs, mask=masked_roi_matrix)
    diff_snrs = np.ma.masked_array(diff_snrs, mask=masked_roi_matrix)
    deviating_rois = np.ma.masked_array(deviating_rois, mask=masked_roi_matrix)

    diff_pixels = 100 / overall_avg * (image2d - overall_avg)
    deviating_pixels = np.zeros(image_info.shape, dtype=bool)
    deviating_pixels[np.abs(diff_pixels) > paramset.hom_deviating_pixels] = True
    deviating_pixels[roi_mask == True] = False
    n_deviating_pixels = np.count_nonzero(deviating_pixels)
    deviating_pixels = np.ma.masked_array(deviating_pixels, mask=roi_mask)
    diff_pixels = np.ma.masked_array(diff_pixels, mask=roi_mask)

    dev_pixel_clusters = None
    if n_deviating_pixels > 0:
        dev_pixel_clusters = np.zeros(matrix_shape, dtype=np.int32)
        for j, ystart in enumerate(ystarts):
            for i, xstart in enumerate(xstarts):
                sub = deviating_pixels[ystart:ystart+roi_size_in_pix,
                                       xstart:xstart+roi_size_in_pix]
                dev_pixel_clusters[j, i] = np.count_nonzero(sub)

    details = {'variances': variances, 'averages': masked_avgs, 'snrs': masked_snrs,
               'diff_averages': diff_avgs, 'diff_snrs': diff_snrs,
               'diff_pixels': diff_pixels, 'deviating_rois': deviating_rois,
               'n_deviating_averages': n_dev_avgs, 'n_deviating_snrs': n_dev_snrs,
               'n_deviating_rois': n_dev_rois,
               'n_rois': n_rois, 'n_masked_rois': n_masked_rois,
               'deviating_pixels': deviating_pixels,
               'n_deviating_pixels': n_deviating_pixels,
               'deviating_pixel_clusters': dev_pixel_clusters,
               'n_pixels': n_pixels, 'n_masked_pixels': n_masked_pixels,
               'roi_mask': roi_mask}

    return details


def calculate_focal_spot_size(image, roi_array, image_info, paramset):
    """Find star pattern and outer blur-diameter.

    Parameters
    ----------
    image : np.2darray.
    image_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetXray
        as defined in config/config_classes.py

    Returns
    -------
    details_dict : dict
    values : list of float or None
    errmsgs : list of str
    """
    values = None
    errmsgs = []
    details_dict = {}
    pix = image_info.pix[0]

    # find center from inner disc of low signal
    rows_inner = np.max(roi_array[1], axis=1)
    cols_inner = np.max(roi_array[1], axis=0)
    sub_inner = image[rows_inner][:, cols_inner]
    res = mmcalc.find_center_object(sub_inner)
    proceed = True
    off_x, off_y = 0, 0
    if res is None:
        errmsgs.append('Failed finding center of star pattern.')
        proceed = False
    else:
        center_x, center_y, width_x, width_y = res
        off_x = center_x - sub_inner.shape[1] // 2
        off_y = center_y - sub_inner.shape[0] // 2
        
        cx_img, cy_img, _, _ = mmcalc.find_center_object(1.*roi_array[1])
        offx_img = cx_img - image.shape[1] // 2
        offy_img = cy_img - image.shape[0] // 2
        
        details_dict['off_xy'] = [off_x + offx_img, off_y + offy_img]

    if proceed:
        # shift rois by off_center
        for i in [0, 2, 3]:
            roi_array[i] = np.roll(roi_array[i], round(off_x), axis=1)
            roi_array[i] = np.roll(roi_array[i], round(off_y), axis=0)
        rows = np.max(roi_array[0], axis=1)
        cols = np.max(roi_array[0], axis=0)
        sub_outer = image[rows][:, cols]
        for i in [0, 2, 3]:
            roi_array[i] = roi_array[i][rows][:, cols]

        # find outer dimensjon of star pattern
        dist_map = mmcalc.get_distance_map_point(sub_outer.shape)
        dists_flat = dist_map.flatten()
        sort_idxs = np.argsort(dists_flat)
        dists = pix * dists_flat[sort_idxs]
        values_flat = sub_outer.flatten()[sort_idxs]
        in_x = roi_array[2].flatten()[sort_idxs]
        dists_x = dists[in_x == True]
        vals_x = values_flat[in_x == True]
        radi_pattern_mm = paramset.foc_pattern_size / 2
        dists_x, vals_x = mmcalc.resample_by_binning(
            vals_x, dists_x, pix, first_step=radi_pattern_mm)  # assume M>1
        vals_x_sm = sp.ndimage.gaussian_filter(vals_x, sigma=2)
        diff = np.diff(vals_x_sm)
        try:
            lower = np.where(diff < 0.1*np.min(diff))  # first value lower than 10% of min
            before_lower = vals_x_sm[0:lower[0][0]]
            positives = np.where(before_lower > 0)
            first_drop_idx = positives[0][-1] + 1  # last positive
            radi_pattern_img_mm = dists_x[first_drop_idx]
            magnification = radi_pattern_img_mm / radi_pattern_mm
            details_dict['star_diameter_mm'] = radi_pattern_img_mm * 2
            details_dict['magnification'] = magnification
        except IndexError:
            errmsgs.append('Failed finding edge of star pattern.')
            proceed = False

        if proceed:
            # crop sub to star pattern
            in_pattern = np.zeros(sub_outer.shape, dtype=bool)
            in_pattern[dist_map * pix < radi_pattern_img_mm] = True
            rows = np.max(in_pattern, axis=1)
            cols = np.max(in_pattern, axis=0)
            sub_crop = sub_outer[rows][:, cols]
            dist_map = dist_map[rows][:, cols]
            for i in [0, 2, 3]:
                roi_array[i] = roi_array[i][rows][:, cols]

            # calculate variance map to identify minima
            kernel_sz_pix = round(2. / image_info.pix[0])  # 2mm kernel
            kernel = np.full((kernel_sz_pix, kernel_sz_pix),
                             1./(kernel_sz_pix**2))
            mu = sp.signal.fftconvolve(sub_crop, kernel, mode='same')
            ii = sp.signal.fftconvolve(sub_crop ** 2, kernel, mode='same')
            variance_crop = ii - mu**2
            perc = np.percentile(variance_crop, 50)
            variance_crop[variance_crop > perc] = perc

            # find radial profiles and convert variance minima to peaks
            dists_flat = dist_map.flatten()
            sort_idxs = np.argsort(dists_flat)
            dists = pix * dists_flat[sort_idxs]
            values_flat = variance_crop.flatten()[sort_idxs]
            blur_diameter_xy = []
            focal_size_xy = []
            profiles = []
            profiles_dists = []
            line_radians = np.deg2rad(paramset.foc_angle)
            roi_founds = []
            for idx in [2, 3]:
                in_x = roi_array[idx].flatten()[sort_idxs]
                dists_x = dists[in_x == True]
                vals_x = values_flat[in_x == True]
                dists_x, vals_x = mmcalc.resample_by_binning(vals_x, dists_x, pix)
                vals_x = perc - vals_x
                profiles.append(vals_x)
                profiles_dists.append(dists_x)
                peaks = find_peaks(vals_x)
                x_peak = dists_x[peaks[0][-1]]  # last peak
                blur_diameter_xy.append(x_peak * 2)
                fs = line_radians * (x_peak * 2) / (magnification - 1)
                focal_size_xy.append(fs)
                roi_found = roi_array[idx]
                roi_found[dist_map * pix > x_peak] = False
                roi_founds.append(roi_found)
            details_dict['blur_diameter_xy'] = blur_diameter_xy
            details_dict['focal_size_xy'] = focal_size_xy

            details_dict['variance_cropped'] = variance_crop
            details_dict['roi_found_x'] = roi_founds[0]
            details_dict['roi_found_y'] = roi_founds[1]

            details_dict['profiles'] = profiles
            details_dict['profiles_dists'] = profiles_dists

    try:
        values = [details_dict['star_diameter_mm'],
                  details_dict['magnification'],
                  details_dict['blur_diameter_xy'][0],
                  details_dict['blur_diameter_xy'][1],
                  details_dict['focal_size_xy'][0],
                  details_dict['focal_size_xy'][1]
                  ]
    except KeyError:
        values = [None] * 6

    return details_dict, values, errmsgs


def calculate_NM_sweep(matrix, img_infos, roi_array, paramset):
    """Calculate 2d map from Auto QC sweep test.

    Parameters
    ----------
    matrix : list of np.2darray
        132 images expected
    img_infos : list of DcmInfo
        DcmInfo as defined in scripts/dcm.py
    roi_array : list of numpy.ndarray
    paramset : ParamSetNM

    Returns
    -------
    details_dicts : list of dict
        one dict pr detector
        fwhm_matrix, diff_matrix
    errmsg : list of str
    """
    details_dicts = []
    errmsgs = None
    top, bottom = 55, 202  # ufov = 0.9
    cfov_ys, cfov_xs = ([10, -10], [5, -5])  # cfov 0.75 (pixels y, lines/images x)
    first_last_sum = [[1, 66], [67, 132]]  # images to sum
    first_last = [[4, 63], [70, 129]]  # image numbers where line source clear
    n_avg = 3  # TODO - in paramset

    def get_fwhm_center_pr_image(image):
        fwhm_col = []
        center_col = []
        for row in range(top, bottom):
            profile = np.sum(image[row-n_avg:row+n_avg], axis=0)
            fwhm, center = mmcalc.get_width_center_at_threshold(
                profile, np.max(profile)/2)
            fwhm_col.append(fwhm)
            center_col.append(center)
        diff_center_col = np.array(center_col) - np.mean(center_col)
        return np.array(fwhm_col), diff_center_col

    def get_intrinsic_uniformity(array):
        return (np.max(array) - np.min(array)) / np.mean(array)

    for det in range(2):
        first_sum, last_sum = first_last_sum[det]
        sum_matrix = np.zeros(matrix[0].shape)
        for i in range(first_sum, last_sum):
            sum_matrix = sum_matrix + matrix[i]
        uni_res = nm_methods.calculate_NM_uniformity(
            sum_matrix, roi_array, img_infos[0].pix[0], 0)

        first, last = first_last[det]
        fwhm_matrix = np.zeros((bottom-top, last-first))
        diff_matrix = np.zeros((bottom-top, last-first))
        for i in range(first, last):
            fwhm_arr, diff_center_arr = get_fwhm_center_pr_image(matrix[i])
            fwhm_matrix[:,i-first] = fwhm_arr
            diff_matrix[:,i-first] = diff_center_arr

        fwhm_matrix = img_infos[0].pix[0]*fwhm_matrix
        diff_matrix = img_infos[0].pix[0]*diff_matrix
        fwhm_iu_ufov = get_intrinsic_uniformity(fwhm_matrix)
        fwhm_iu_cfov = get_intrinsic_uniformity(
            fwhm_matrix[cfov_ys[0]:cfov_ys[1], cfov_xs[0]:cfov_xs[1]])
        diff_max_ufov = np.max(np.abs(diff_matrix))
        diff_max_cfov = np.max(np.abs(
            diff_matrix[cfov_ys[0]:cfov_ys[1], cfov_xs[0]:cfov_xs[1]]))

        # expand pixels pr image to same shape as ufov
        aspect = (
            uni_res['du_matrix'].shape[1] / uni_res['du_matrix'].shape[0])
        resize_x = round(aspect * fwhm_matrix.shape[0])  # resize when draw

        details_dict = {
            'fwhm_matrix': fwhm_matrix,
            'diff_matrix': diff_matrix,
            'fwhm_iu_ufov': fwhm_iu_ufov, 'fwhm_iu_cfov': fwhm_iu_cfov,
            'diff_max_ufov': diff_max_ufov, 'diff_max_cfov': diff_max_cfov,
            'resize_x': resize_x,
            'sum_matrix': sum_matrix,
            'ufov_matrix': uni_res['matrix_ufov'],
            'du_matrix': uni_res['du_matrix'],
            'uni_pix_size': uni_res['pix_size'],
            'uni_values': uni_res['values'],
            }
        details_dicts.append(details_dict)

    return (details_dicts, errmsgs)


def calculate_recovery_curve(matrix, img_info, center_roi, zpos, paramset, background):
    """Find spheres and calculculate recovery curve values.

    Parameters
    ----------
    matrix : list of np.2darray
        slices +/- largest sphere diameter from slice with max pixelvalue
    image_info :  DcmInfo
        as defined in scripts/dcm.py
    center_roi : np.2darray
        bool array with circular roi at found center
    zpos : np.array
        zpos for each slice in input matrix
    paramset : cfc.ParamsetPET
    background : float
        found pixel value for background activity

    Returns
    -------
    details_dict : dict
        DESCRIPTION.
        'values': [
            avg_values + [background],
            max_values + [background],
            peak_values + [background]
            ],
        'roi_spheres': list of foud rois representing the spheres,
        'roi_peaks': list of found rois representing the peak 1ccs
    errmsg : str or None
    """
    errmsg = None
    size_y, size_x = matrix[0].shape
    dist = paramset.rec_sphere_dist / img_info.pix[0]  # distance from center
    n_spheres = len(paramset.rec_sphere_diameters)

    # get center from center roi
    mask = np.where(center_roi, 0, 1)
    mask_pos = np.where(mask == 0)
    xpos = int(np.mean(mask_pos[1]))
    ypos = int(np.mean(mask_pos[0]))
    dx = xpos - size_x // 2
    dy = ypos - size_y // 2

    # sum centered image
    dists = [xpos, size_x - xpos, ypos, size_y - ypos]
    min_dists = np.min(dists) - 1
    summed_img = None
    for image in matrix:
        this_centered = image[ypos - min_dists:ypos + min_dists,
                              xpos - min_dists:xpos + min_dists]
        if summed_img is None:
            summed_img = this_centered
        else:
            summed_img = summed_img + this_centered

    # get position of spheres
    pol, (_, angs) = mmcalc.topolar(summed_img)
    prof = np.max(pol, axis=0)
    prof = prof - np.min(prof)
    peaks = find_peaks(prof, distance=prof.shape[0]/10)
    peaks_pos = peaks[0]
    roi_dx_dy = [(0, 0) for i in range(n_spheres)]
    if peaks_pos.shape[0] in [n_spheres, n_spheres + 1]:
        # +1 if one sphere split at 0
        peak_values = prof[peaks_pos]
        if peaks_pos.shape[0] == n_spheres + 1:
            if peak_values[0] > peak_values[-1]:
                peaks_pos = peaks_pos[0:n_spheres]
            else:
                peaks_pos = peaks_pos[1:]
        order_peaks = np.argsort(prof[peaks_pos])

        tan_angles = np.tan(angs[peaks_pos])
        for i, order in enumerate(order_peaks):
            pos_x = dist / np.sqrt(1 + tan_angles[i]**2)
            this_ang = angs[peaks_pos[i]]
            if this_ang > np.pi/2 and this_ang < 3*np.pi/2:
                pos_x = - pos_x
            pos_y = - pos_x * tan_angles[i]
            roi_dx_dy[order] = (pos_x + dx, pos_y + dy)

        # smooth matrix by 1cc spheres for peak-values
        radius_1cc = 10 * (3 / (4 * np.pi)) ** (1/3)
        pix_kernel = int(np.round(radius_1cc / img_info.pix[0]) * 2 + 1)
        dz = np.abs(zpos[1]-zpos[0])
        n_slice_kernel = int(np.ceil(radius_1cc / dz) * 2 + 1)
        peak_kernel = np.zeros((n_slice_kernel, pix_kernel, pix_kernel))
        dzs = dz * (np.arange(n_slice_kernel) - n_slice_kernel // 2)
        for i, zz in enumerate(dzs):
            radsq = radius_1cc ** 2 - zz ** 2
            if radsq > 0:
                radius_this = np.sqrt(radsq)
                if radius_this > 0:
                    peak_kernel[i] = get_roi_circle((pix_kernel, pix_kernel), (0, 0),
                                                    radius_this/img_info.pix[0])
        peak_kernel = 1./np.sum(peak_kernel) * peak_kernel
        matrix_smoothed_1cc = sp.signal.fftconvolve(matrix, peak_kernel, mode='same')
        peak_kernel_bool = np.full(peak_kernel.shape, False)
        peak_kernel_bool[peak_kernel > 0] = True

        # for each sphere - get spheric roi
        roi_radii = np.array(paramset.rec_sphere_diameters)  # search radius=Ø
        roi_radii[0] = roi_radii[1]  # smallest a bit extra margin
        zpos_center = zpos[len(zpos) // 2]
        zpos_diff = np.abs(zpos - zpos_center)
        roi_spheres = []
        roi_peaks = []
        max_values = []
        avg_values = []
        peak_values = []
        for roi_no, dx_dy in enumerate(roi_dx_dy):
            this_roi = []
            this_roi_peak = []
            values = []
            for i, image in enumerate(matrix):
                if zpos_diff[i] < roi_radii[roi_no]:
                    radius_this = np.sqrt(roi_radii[roi_no] ** 2 - zpos_diff[i] ** 2)
                    this_roi.append(get_roi_circle(
                        image.shape, dx_dy, radius_this / img_info.pix[0]))
                else:
                    this_roi.append(None)
            roi_spheres.append(this_roi)
            values = None
            max_all = []
            max_all_smoothed_1cc = []
            for i, image in enumerate(matrix):
                if this_roi[i] is not None:
                    mask = np.where(this_roi[i], 0, 1)
                    mask_pos = np.where(mask == 0)
                    if values is None:
                        values = image[mask_pos].flatten()
                    else:
                        values = np.append(values, image[mask_pos].flatten())
                    max_all.append(np.max(image[mask_pos]))
                    max_this_smoothed = np.max(matrix_smoothed_1cc[i][mask_pos])
                    max_all_smoothed_1cc.append(max_this_smoothed)
                else:
                    max_all.append(None)
                    max_all_smoothed_1cc.append(0)
            max_this = np.max(values)
            max_this_1cc = np.max(max_all_smoothed_1cc)
            max_values.append(max_this)
            if max_this is not None:
                threshold = paramset.rec_sphere_percent * 0.01 * (
                    max_this - background)
                threshold = threshold + background
                avg_values.append(np.mean(values[values > threshold]))

                # find peak - new
                peak_values.append(max_this_1cc)
                slice_where_max_1cc = np.where(
                    max_all_smoothed_1cc == max_this_1cc)
                peak_idx = slice_where_max_1cc[0][0]
                arr = np.ma.masked_array(
                    matrix_smoothed_1cc[peak_idx],
                    mask=np.invert(this_roi[peak_idx]))
                max_pos = np.where(arr == max_this_1cc)
                peak_xy = (max_pos[1][0], max_pos[0][0])
                this_roi_peak = [None for i in range(len(matrix))]

                for i in range(n_slice_kernel):
                    roi_this = np.full(image.shape, False)
                    xstart = peak_xy[0] - pix_kernel // 2
                    ystart = peak_xy[1] - pix_kernel // 2
                    roi_this[ystart:ystart + pix_kernel,
                             xstart:xstart + pix_kernel] = peak_kernel_bool[i]
                    try:
                        this_roi_peak[
                            peak_idx - n_slice_kernel // 2 + i] = roi_this
                    except IndexError:
                        pass

            else:
                avg_values.append(None)
                peak_values.append(None)

            roi_peaks.append(this_roi_peak)

        details_dict = {
            'values': [
                avg_values + [background],
                max_values + [background],
                peak_values + [background]
                ],
            'roi_spheres': roi_spheres,
            'roi_peaks': roi_peaks
            }
    else:
        details_dict = {}
        errmsg = 'Failed to find 6 spheres'

    return (details_dict, errmsg)
