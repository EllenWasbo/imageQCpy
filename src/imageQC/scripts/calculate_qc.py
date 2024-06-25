#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculation processes for the different tests.

@author: Ellen Wasbø
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import copy
import warnings

import numpy as np
import scipy as sp
from scipy.signal import find_peaks
import skimage
from PyQt5.QtWidgets import qApp

# imageQC block start
from imageQC.scripts import dcm
from imageQC.scripts.calculate_roi import (get_rois, get_roi_circle, get_roi_SNI)
import imageQC.scripts.mini_methods_format as mmf
import imageQC.scripts.mini_methods as mm
import imageQC.scripts.mini_methods_calculate as mmcalc
from imageQC.config.iQCconstants import (
    HEADERS, HEADERS_SUP, QUICKTEST_OPTIONS, HALFLIFE)
import imageQC.config.config_classes as cfc
from imageQC.config.config_func import get_config_folder
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

    if input_main.results != {}:
        n_imgs = len(input_main.imgs)
        # get set names of images or generate default names (indexed names)
        image_names = [f'img{i}' for i in range(n_imgs)]
        if len(input_main.current_quicktest.image_names) > 0:
            set_names = input_main.current_quicktest.image_names
            if any(set_names):
                for i, set_name in enumerate(set_names):
                    if set_name != '' and i < len(image_names):
                        image_names[i] = set_name
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

            res_pr_image = input_main.results[test]['pr_image']

            # for each sub-output for current test
            for sub in output_subs:
                values = None
                headers = None
                if sub.alternative > 9:  # supplement table to output
                    values = input_main.results[test]['values_sup']
                    headers = input_main.results[test]['headers_sup']
                else:
                    proceed = False
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
                    extras = get_SNI_ref_image(paramset, tag_infos)
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
                                list_tags=['SeriesUID'])
                            extra_tag_list = []
                            read_tags[i] = True
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
                    elif modality == 'Mammo':
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

            for i in range(n_analyse):
                if main_type in ['MainWindow', 'TaskBasedImageQualityDialog']:
                    try:
                        input_main.progress_modal.setLabelText(
                            f'Reading/calculating image {i}/{n_img}')
                        input_main.progress_modal.setValue(round(100 * i/n_img))
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
                            get_window_level=any(auto_template_label)
                            )
                        try:
                            if len(input_main.imgs[i].artifacts) > 0:
                                image = apply_artifacts(image, input_main.imgs[i],
                                                        input_main.artifacts)
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
                            get_window_level=any(auto_template_label)
                            )
                        try:
                            if len(input_main.imgs[i].artifacts) > 0:
                                image = apply_artifacts(image, input_main.imgs[i],
                                                        input_main.artifacts)
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
                                    'values_info': result.values_info,
                                    'values_sup_info': result.values_sup_info
                                    }
                            input_main.results[
                                test]['values'][i] = result.values
                            input_main.results[
                                test]['values_sup'][i] = result.values_sup
                            input_main.results[
                                test]['details_dict'][i] = result.details_dict
            if err_extra:
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
            if modality == 'CT':
                if 'Noi' in flattened_marked and 'Noi' in input_main.results:
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
            elif 'Rec' in input_main.results:
                input_main.wid_window_level.tb_wl.set_window_level(
                    'min_max', set_tools=True)
                input_main.set_active_img(
                    input_main.results['Rec']['details_dict']['max_slice_idx'])
            try:
                input_main.progress_modal.setValue(input_main.progress_modal.maximum())
            except AttributeError:
                pass

    elif main_type == 'TaskBasedImageQualityDialog':
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
            for i in range(np.shape(roi_array)[0]):
                arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[i]))
                avgs.append(np.mean(arr))
                stds.append(np.std(arr))

        if modality == 'CT':
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
            headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['altAll'])
            if image2d is not None:
                values = [avgs[1], avgs[2], avgs[3], avgs[4], avgs[0],
                          avgs[1] - avgs[0], avgs[2] - avgs[0],
                          avgs[3] - avgs[0], avgs[4] - avgs[0]]
                values_sup = [stds[1], stds[2], stds[3], stds[4], stds[0]]

        elif modality == 'Xray':
            alt = paramset.hom_tab_alt
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt'+str(alt)])
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

        elif modality == 'Mammo':
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
            headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt0'])
            if image2d is not None:
                details = calculate_flatfield_mammo(
                    image2d, roi_array[-2], roi_array[-1], image_info, paramset)
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
                        details_dict=details)

        elif modality == 'PET':
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
            if image2d is not None:
                avg = sum(avgs) / len(avgs)
                diffs = [100.*(avgs[i] - avg)/avg for i in range(5)]
                values = avgs + diffs

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
        if modality in ['CT', 'SPECT']:
            # only bead/point method 2d (alt0)
            alt = paramset.mtf_type
            headers = copy.deepcopy(
                HEADERS[modality][test_code]['alt' + str(alt)])
            headers_sup = copy.deepcopy(
                HEADERS_SUP[modality][test_code]['alt' + str(alt)])
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
                    details, errmsg = calculate_MTF_point(
                        sub[0], image_info, paramset)
                    details[0]['matrix'] = sub[0]
                elif paramset.mtf_type == 1:
                    details, errmsg = calculate_MTF_2d_line_edge(
                        sub, image_info.pix[0], paramset, mode='line')
                elif paramset.mtf_type == 2:
                    details, errmsg = calculate_MTF_2d_line_edge(
                        sub, image_info.pix[0], paramset, mode='line', pr_roi=True)
                else:  # type 3 edge
                    details, errmsg = calculate_MTF_2d_line_edge(
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
        alt = 0 if paramset.sni_type == 0 else 2
        if paramset.sni_channels:
            alt = alt + 1

        headers = copy.deepcopy(HEADERS[modality][test_code]['alt' + str(alt)])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['altAll'])
        if image2d is None:
            res = Results(headers=headers, headers_sup=headers_sup)
        else:
            reference_image = None
            if extras is not None:
                if 'reference_image' in extras:
                    reference_image = extras['reference_image'][image_info.frame_number]

            values, values_sup, details_dict, errmsg = calculate_NM_SNI(
                image2d, roi_array, image_info, paramset, reference_image)

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
        if modality == 'CT':
            alt = paramset.sli_type
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt' + str(alt)])
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
            headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
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
                res, errmsg = get_corrections_point_source(
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

            res = calculate_NM_uniformity(
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

    try:
        result = locals()[test_code]()
    except (KeyError, IndexError):
        result = None

    return result


def calculate_3d(matrix, marked_3d, input_main, extra_taglists):
    """Calculate tests in 3d mode i.e. tests depending on more than one image.

    Parameters
    ----------
    matrix : np.array
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

    def sum_matrix(images_to_test):
        sum_matrix = matrix[images_to_test[0]]
        for i in images_to_test[1:]:
            sum_matrix = np.add(sum_matrix, matrix[i])
        return sum_matrix

    def MTF(images_to_test):
        if modality in ['CT', 'SPECT']:
            headers = copy.deepcopy(
                HEADERS[modality][test_code][f'alt{paramset.mtf_type}'])
            headers_sup = copy.deepcopy(
                HEADERS_SUP[modality][test_code][f'alt{paramset.mtf_type}'])
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
                        img_infos[images_to_test[0]].pix[0], paramset, images_to_test)
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

                values = None
                values_sup = None
                prefix = 'g' if paramset.mtf_gaussian else 'd'
                try:
                    values = details_dict[0][prefix + 'MTF_details']['values']
                except TypeError:
                    pass
                try:
                    values_sup = list(details_dict[0][
                        'gMTF_details']['LSF_fit_params'])
                except TypeError:
                    pass
                try:  # x-dir, y-dir
                    values.extend(
                        details_dict[1][prefix + 'MTF_details']['values'])
                    values_sup.extend(list(
                        details_dict[1]['gMTF_details']['LSF_fit_params']))
                except (IndexError, KeyError, AttributeError):
                    pass
                res = Results(headers=headers, values=[values],
                              headers_sup=headers_sup, values_sup=[values_sup],
                              details_dict=details_dict,
                              alternative=paramset.mtf_type, pr_image=False,
                              errmsg=errmsg)
        else:
            res = None

        return res

    def TTF(images_to_test):
        headers = copy.deepcopy(HEADERS[modality][test_code]['alt0'])
        headers_sup = copy.deepcopy(HEADERS_SUP[modality][test_code]['alt0'])
        materials = paramset.ttf_table.labels
        if len(images_to_test) == 0 or len(materials) == 0:
            res = Results(headers=headers, headers_sup=headers_sup)
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

                details_dict, errmsg = calculate_MTF_circular_edge(
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

                try:
                    values.append(
                        [materials[idx]]
                        + details_dict[prefix + 'MTF_details']['values'])
                except TypeError:
                    values.append([materials[idx]] + [None] * 3)
                try:
                    values_sup.append(
                        [materials[idx]]
                        + list(details_dict['gMTF_details']['LSF_fit_params']))
                except TypeError:
                    values_sup.append([materials[idx]] + [None] * 4)

            res = Results(headers=headers, values=values,
                          headers_sup=headers_sup, values_sup=values_sup,
                          details_dict=details_dict_all,
                          pr_image=False,
                          errmsg=errmsgs)

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

    def Rec(images_to_test, test_mode=False):
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
            if input_main.test_mode is True:
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
            roi_array, errmsg = get_rois(
                sum_matrix(images_to_test), images_to_test[0], input_main)
            if errmsg is not None:
                errmsgs.append(errmsg)

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
                              details_dict=details_dict, pr_image=False,
                              errmsg=errmsgs)
            else:
                res = Results(headers=headers,
                              headers_sup=headers_sup,
                              errmsg=errmsgs)

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
            res = calculate_NM_uniformity(
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
            values, _, details_dict2, _ = calculate_NM_SNI(
                image_input, roi_array, img_infos[0], paramset, None)
            details_dict.update(details_dict2)

            res = Results(headers=headers, values=[values],
                          details_dict=[details_dict], errmsg=errmsgs, pr_image=False)

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
                max_values_copy[i] = np.nan
                zpos_copy[i] = np.nan
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
        elif isinstance(paramset, cfc.ParamSetMammo):
            gaussfit_type = 'double_both_positive'
            lp_vals = [1, 2, 3, 4, 5]
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
                    if isinstance(paramset, (cfc.ParamSetNM, cfc.ParamSetSPECT)):
                        fwhm, _ = mmcalc.get_width_center_at_threshold(
                            LSF[i], np.max(LSF[i])/2)
                        fwtm, _ = mmcalc.get_width_center_at_threshold(
                            LSF[i], np.max(LSF[i])/10)
                        if fwhm:
                            fwhm = step_size * fwhm
                        if fwtm:
                            fwtm = step_size * fwtm
                        dMTF_details['values'] = [fwhm, fwtm]
                        gMTF_details['values'] = [gMTF_details['FWHM'],
                                                  gMTF_details['FWTM']]
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


def calculate_MTF_circular_edge(matrix, roi, pix, paramset, images_to_test):
    """Calculate MTF from circular edge.

    Based on Richard et al: Towards task-based assessment of CT performance,
    Med Phys 39(7) 2012

    Parameters
    ----------
    matrix : numpy.2darray or list of numpy.array
        list of 2d part of slice limited to disc (or None if ignored slice)
    roi : numpy.2darray of bool
        2d part of larger roi limited to disc
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
            matrix[images_to_test[0]].shape,
            center_dx=center_xy[0] - 0.5 * matrix[images_to_test[0]].shape[1],
            center_dy=center_xy[1] - 0.5 * matrix[images_to_test[0]].shape[0])
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
        radius = 0.5 * matrix[images_to_test[0]].shape[0] * pix
        dists_cut = dists[dists < radius]
        values = values_avg[dists < radius]

        step_size = .1 * pix
        new_x, new_y = mmcalc.resample_by_binning(
            input_y=values, input_x=dists_cut, step=step_size)
        sigma_f = 5.
        LSF, LSF_no_filt, ESF_filtered = mmcalc.ESF_to_LSF(
            new_y, prefilter_sigma=sigma_f)

        width, center = mmcalc.get_width_center_at_threshold(LSF, np.max(LSF)/2)
        disc_radius_mm = None
        if width is not None:
            if center is not None:
                disc_radius_mm = center * step_size
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
                'matrix': matrix,
                'center_xy': center_xy, 'disc_radius_mm': disc_radius_mm,
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


def calculate_flatfield_mammo(image2d, mask_max, mask_outer, image_info, paramset):
    """Calculate homogeneity Mammo according to EU guidelines.

    Parameters
    ----------
    image2d : numpy.ndarray
        input image
    mask_max : numpy.ndarray
        mask of max values if paramset.hom_mask_max is set
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
                            mu = sp.signal.fftconvolve(sub, kernel, mode='same')
                            ii = sp.signal.fftconvolve(sub ** 2, kernel, mode='same')
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


def get_corrections_point_source(
        image2d, img_info, roi_array,
        fit_x=False, fit_y=False, lock_z=False, guess_z=0.1,
        correction_type='multiply', estimate_noise=False):
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
    lock_z : bool
        Set distance to source = guess_z when fitting. The default is False.
    guess_z : float
        If guess_z = 0.1 = DICOM radius or 400.
        If guess > 0.1 used as first approximation. Default is 0.1
    correction_type : str, optional
        'subtract' or 'multiply'. Default is 'multiply'
    estimate_noise : bool
        if True estimate poisson noise for fitted image

    Returns
    -------
    dict
        corrected_image : numpy.2darray
        correction_matrix : numpy.2darray
            add this matrix to obtain corrected image2d
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
    estimated_noise_image = None
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
    dists_inplane = mmcalc.get_distance_map_point(
        image2d.shape, center_dx=dx, center_dy=dy)
    dists_inplane = img_info.pix[0] * dists_inplane

    dists_ufov = dists_inplane[rows][:, cols]
    dists_ufov_flat = dists_ufov.flatten()
    values_flat = ufov_denoised.flatten()
    sort_idxs = np.argsort(dists_ufov_flat)
    values = values_flat[sort_idxs]
    dists = dists_ufov_flat[sort_idxs]

    if lock_z:
        nm_radius = guess_z
        lock_radius = True
    else:
        if guess_z <= 0.1:
            try:
                nm_radius = img_info.nm_radius
            except AttributeError:
                nm_radius = 400
        else:
            nm_radius = guess_z
        lock_radius = False

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
            if estimate_noise:
                rng = np.random.default_rng()
                estimated_noise_image = rng.poisson(fit_matrix)

            # correct input image
            if correction_type == 'multiply':
                normalized_fit_matrix = fit_matrix/np.max(fit)
                corrected_image = image2d / normalized_fit_matrix
            elif correction_type == 'subtract':
                correction_matrix = np.max(fit) - fit_matrix
                corrected_image = image2d + correction_matrix
        else:
            errmsg = 'Failed fitting matrix to point source.'

    return ({'corrected_image': corrected_image,
             'correction_matrix': correction_matrix,
             'fit_matrix': fit_matrix,
             'estimated_noise_image': estimated_noise_image,
             'dx': dx, 'dy': dy, 'distance': distance}, errmsg)


def calculate_NM_uniformity(image2d, roi_array, pix, scale_factor):
    """Calculate uniformity parameters.

    Parameters
    ----------
    image2d : numpy.ndarray
    roi_array : list of numpy.ndarray
        2d mask for ufov [0] and cfov [1]
    pix : float
        pixel size of image2d
    scale_factor : int
        scale factor to get 6.4 mm/pix. 0 = Auto, 1 = none, 2... = scale factor

    Returns
    -------
    matrix : numpy.ndarray
        downscaled and smoothed image used for calculation of uniformity values
    du_matrix : numpy.ndarray
        calculated du_values maximum of x/y direction
    values : list of float
        ['IU_UFOV %', 'DU_UFOV %', 'IU_CFOV %', 'DU_CFOV %']
    """
    def get_differential_uniformity(image, cfov):
        # assume image is part of image within ufov
        # cfov = True where inside cfov
        from numpy.lib.stride_tricks import sliding_window_view
        sz_y, sz_x = image.shape
        du_cols_ufov = np.zeros(image.shape)
        du_cols_cfov = np.zeros(image.shape)
        image_data_ufov = np.copy(image.data)
        image_data_ufov[image.mask == True] = np.nan
        image_data_cfov = np.copy(image.data)
        image_data_cfov[cfov == False] = np.nan
        for x in range(sz_x):
            view = sliding_window_view(image_data_ufov[:, x], 5)
            maxs = np.nanmax(view, axis=-1)
            mins = np.nanmin(view, axis=-1)
            du_cols_ufov[2:-2, x] = 100. * (maxs - mins) / (maxs + mins)
            cfov_data = image_data_cfov[:, x]
            if not np.isnan(cfov_data).all():
                view = sliding_window_view(cfov_data, 5)
                maxs = np.nanmax(view, axis=-1)
                mins = np.nanmin(view, axis=-1)
                du_cols_cfov[2:-2, x] = 100. * (maxs - mins) / (maxs + mins)

        du_rows_ufov = np.zeros(image.shape)
        du_rows_cfov = np.zeros(image.shape)
        for y in range(sz_y):
            view = sliding_window_view(image_data_ufov[y, :], 5)
            maxs = np.nanmax(view, axis=-1)
            mins = np.nanmin(view, axis=-1)
            du_rows_ufov[y, 2:-2] = 100. * (maxs - mins) / (maxs + mins)
            cfov_data = image_data_cfov[y, :]
            if not np.isnan(cfov_data).all():
                view = sliding_window_view(cfov_data, 5)
            maxs = np.nanmax(view, axis=-1)
            mins = np.nanmin(view, axis=-1)
            du_rows_cfov[y, 2:-2] = 100. * (maxs - mins) / (maxs + mins)

        du_matrix_ufov = np.maximum(du_cols_ufov, du_rows_ufov)
        du_matrix_ufov[image.mask == True] = np.nan
        du_matrix_cfov = np.maximum(du_cols_cfov, du_rows_cfov)
        du_matrix_cfov[cfov == False] = np.nan

        return {'du_matrix': du_matrix_ufov, 'du_ufov': np.nanmax(du_matrix_ufov),
                'du_cfov': np.nanmax(du_matrix_cfov)}

    # continue with image within ufov only
    rows = np.max(roi_array[0], axis=1)
    cols = np.max(roi_array[0], axis=0)
    image2d = image2d[rows][:, cols]
    cfov = roi_array[1][rows][:, cols]
    ufov_input = roi_array[0][rows][:, cols]

    # roi already handeled ignoring pixels with zero counts as nearest neighbour (NEMA)
    if scale_factor == 1:
        image = image2d
        block_size = 1
    else:
        # resize to 6.4+/-30% mm pixels according to NEMA
        # pix_range = [6.4*0.7, 6.4*1.3]
        block_size = scale_factor
        if scale_factor == 0:  # Auto scale
            scale_factors = [(np.floor(64/pix)), (np.ceil(6.4/pix))]
            pix_diff = np.abs(pix*np.array(scale_factors) - 6.4)
            selected_pix = np.where(pix_diff == np.min(pix_diff))
            block_size = int(scale_factors[selected_pix[0][0]])

        # skip pixels excess of size/block_size = NEMA at least 50% of pixel inside
        sz_y, sz_x = image2d.shape
        skip_y, skip_x = (sz_y % block_size, sz_x % block_size)
        n_blocks_y, n_blocks_x = (sz_y // block_size, sz_x // block_size)
        start_y, start_x = (skip_y // 2, skip_x // 2)
        end_x = start_x + n_blocks_x * block_size
        end_y = start_y + n_blocks_y * block_size
        cfov = cfov[start_y:end_y, start_x:end_x]
        ufov_input = ufov_input[start_y:end_y, start_x:end_x]
        image2d = image2d[start_y:end_y, start_x:end_x]
        image = skimage.measure.block_reduce(
            image2d, (block_size, block_size), np.sum)
        # scale down to ~6.4mm/pix

        # cfov, NEMA - at least 50% of the pixel should be inside UFOV to be included
        reduced_roi = skimage.measure.block_reduce(
            cfov, (block_size, block_size), np.mean)
        cfov = np.where(reduced_roi > 0.5, True, False)

        if False in ufov_input:
            reduced_roi = skimage.measure.block_reduce(
                ufov_input, (block_size, block_size), np.mean)
            ufov_input = np.where(reduced_roi > 0.5, True, False)

    # ufov, NEMA NU-1: ignore pixels < 75% of CFOV mean in outer rows/cols of UFOV
    arr_cfov = np.ma.masked_array(image, mask=np.invert(cfov))
    cfov_mean = np.mean(arr_cfov)  # TODO test against minimum 10000 (NEMA)
    ufov = np.where(image > 0.75*cfov_mean, True, False)
    if False in ufov:
        ufov[1:-1][:, 1:-1] = True  # ignore only outer rows/cols
    if False in ufov_input:
        ufov[ufov_input == False] = False

    sz_y, sz_x = image.shape
    center_pixel_count = image[sz_y//2][sz_x//2]

    # smooth masked array
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    kernel = kernel / np.sum(kernel)
    arr = np.ma.masked_array(image, mask=np.invert(ufov))
    smooth64 = mmcalc.masked_convolve2d(arr, kernel, boundary='symm', mode='same')

    # differential uniformity
    du_dict = get_differential_uniformity(smooth64, cfov)
    du_ufov = du_dict['du_ufov']
    du_cfov = du_dict['du_cfov']

    smooth64_data = smooth64.data

    # integral uniformity
    max_val = np.max(smooth64_data[ufov == True])
    min_val = np.min(smooth64_data[ufov == True])
    iu_ufov = 100. * (max_val - min_val) / (max_val + min_val)
    max_val = np.max(smooth64_data[cfov == True])
    min_val = np.min(smooth64_data[cfov == True])
    iu_cfov = 100. * (max_val - min_val) / (max_val + min_val)

    return {'matrix_ufov': smooth64_data,
            'du_matrix': du_dict['du_matrix'], 'pix_size': pix * block_size,
            'center_pixel_count': center_pixel_count,
            'values': [iu_ufov, du_ufov, iu_cfov, du_cfov]}


def get_eye_filter(roi_size, pix, paramset):
    """Get eye filter V(r)=r^1.3*exp(-cr^2).

    Parameters
    ----------
    roi_size : int
        size of roi or image in pix
    pix : float
        mm pr pix in image
    paramset : ParamSetNM

    Returns
    -------
    eye_filter : dict
        filter 2d : np.ndarray quadratic with size roi_height
        curve: dict of r and V
    """
    def tukey_filter_func(r, start_L_flat_ratio):
        """Calulates Tukey filter.

        Parameters
        ----------
        r : np.array
            frequency values
        start_L_tapered_ratio : tuple of floats
            start = minimum_frequency
            L = width (frequencies)
            flat ratio = 1- tapered_ratio

        Returns
        -------
        V : filter values
        """
        start, L, flat_ratio = start_L_flat_ratio
        t = L * (1-flat_ratio)
        V = np.ones(r.shape)
        V[r < start + t/2] = 0.5 * (1 + np.cos(2 * np.pi/t * (
            r[r < start + t/2] - start - t/2)))
        V[r > L + start - t/2] = 0.5 * (1 + np.cos(2 * np.pi/t * (
            r[r > L + start - t/2] - L - start + t/2)))
        V[r < start] = 0
        V[r > start + L] = 0
        return V

    def eye_filter_func(r, c):
        return r**1.3 * np.exp(-c*r**2)

    freq = np.fft.fftfreq(roi_size, d=pix)
    freq = freq[:freq.size//2]  # from center
    if paramset.sni_channels:
        V = tukey_filter_func(freq, paramset.sni_channels_table[0])
    else:
        V = eye_filter_func(freq, paramset.sni_eye_filter_c)
    eye_filter_1d = {'r': freq, 'V': 1/np.max(V) * V}
    if paramset.sni_channels:
        V2 = tukey_filter_func(freq, paramset.sni_channels_table[1])
        eye_filter_1d['V2'] = 1/np.max(V2) * V2

    unit = freq[1] - freq[0]
    dists = mmcalc.get_distance_map_point(
        (roi_size, roi_size))
    if paramset.sni_channels:
        eye_filter_2d = tukey_filter_func(unit*dists, paramset.sni_channels_table[0])
    else:
        eye_filter_2d = eye_filter_func(unit*dists, paramset.sni_eye_filter_c)
    eye_filter_2d = 1/np.max(eye_filter_2d) * eye_filter_2d
    if paramset.sni_channels:
        eye_filter_2d_2 = tukey_filter_func(
            unit*dists, paramset.sni_channels_table[1])
        eye_filter_2d = [eye_filter_2d, eye_filter_2d_2]

    return {'filter_2d': eye_filter_2d, 'curve': eye_filter_1d, 'unit': unit}


def get_SNI_ref_image(paramset, tag_infos):
    """Get noise from reference image.

    Parameters
    ----------
    paramset : cfc.ParamsetNM
    tag_infos : tag_infos to read reference image

    Returns
    -------
    dict
        'reference_image': list of nd.array
        empty list if image failed to read
        one image for each frame else
    """
    sni_ref_image = []
    filename = (
        Path(get_config_folder()) / 'SNI_ref_images' /
        (paramset.sni_ref_image + '.dcm'))
    ref_infos, _, _ = dcm.read_dcm_info([filename], GUI=False, tag_infos=tag_infos)
    for rno, ref_info in enumerate(ref_infos):
        image, _ = dcm.get_img(
            filename,
            frame_number=rno, tag_infos=tag_infos)
        if image is not None:
            sni_ref_image.append(image)
    return {'reference_image': sni_ref_image}


def calculate_SNI_ROI(image2d, roi_array_this, eye_filter=None, unit=1.,
                      pix=1., fit_dict=None, image_mean=0.,
                      sampling_frequency=0.01, sni_dim=0):
    """Calculate SNI for one ROI.

    Parameters
    ----------
    image2d : numpy.2darray
        if fit_matrix is not None, this is a flattened matrix
        (corrected for point source curvature)
    roi_array : numpy.2darray
        2d mask for the current ROI
    eye_filter : numpy.2darray or list of numpy.2darray
    unit : float
        unit of NPS and eye_filter
    pix : float
    fit_dict : dict
        dictionary from get_corrections_point_source (if corrections else None)
    image_mean : float
        average value within the large area
    sampling_frequency : float
        sampling frequency for radial profiles
    sni_dim : int
        options for sni calculations 0=2d NPS ratio, 1=radial profile ratio

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
    """
    rows = np.max(roi_array_this, axis=1)
    cols = np.max(roi_array_this, axis=0)
    subarray = image2d[rows][:, cols]
    line = subarray.shape[0] // 2  # position of 0 frequency
    rNPS_quantum_noise = None
    if fit_dict is None:  # uncorrected image
        NPS = mmcalc.get_2d_NPS(subarray, pix)
        quantum_noise = image_mean * pix**2
        """ explained how quantum noise is found above
        Mean count=variance=pixNPS^2*Total(NPS) where pixNPS=1./(ROIsz*pix)
        Total(NPS)=NPSvalue*ROIsz^2
        NPSvalue = Meancount/(pixNPS^2*ROIsz^2)=MeanCount*pix^2
        """
        NPS[line, line] = 0
        NPS_struct = NPS - quantum_noise
        NPS_struct[line, line] = 0
    else:
        if 'reference_image' in fit_dict:
            sub_estimated_noise = fit_dict['reference_image'][rows][:, cols]
        else:
            sub_estimated_noise = fit_dict['estimated_noise_image'][rows][:, cols]
        if 'correction_matrix' in fit_dict:
            # curve correct both subarray and quantum noise
            corr_matrix = fit_dict['correction_matrix'][rows][:, cols]
            subarray = subarray + corr_matrix
            sub_estimated_noise = sub_estimated_noise + corr_matrix
        # 2d NPS
        NPS = mmcalc.get_2d_NPS(subarray, pix)
        quantum_noise = mmcalc.get_2d_NPS(sub_estimated_noise, pix)
        # set lowest frequencies to 0 as these are extreme due to curvature
        quantum_noise[line, line] = 0
        NPS[line, line] = 0
        _, rNPS_quantum_noise = mmcalc.get_radial_profile(
            quantum_noise, pix=unit, step_size=sampling_frequency)
        NPS_struct = np.subtract(NPS, quantum_noise)

    rNPS_filt_2 = None
    rNPS_struct_filt_2 = None
    SNI_2 = None
    if isinstance(eye_filter, list):
        NPS_filt = NPS * eye_filter[0]
        NPS_struct_filt = NPS_struct * eye_filter[0]
        NPS_filt_2 = NPS * eye_filter[1]
        NPS_struct_filt_2 = NPS_struct * eye_filter[1]
    else:
        NPS_filt = NPS * eye_filter
        NPS_struct_filt = NPS_struct * eye_filter

    # radial NPS curves
    freq, rNPS = mmcalc.get_radial_profile(
        NPS, pix=unit, step_size=sampling_frequency, ignore_negative=True)
    _, rNPS_filt = mmcalc.get_radial_profile(
        NPS_filt, pix=unit, step_size=sampling_frequency, ignore_negative=True)
    _, rNPS_struct = mmcalc.get_radial_profile(
        NPS_struct, pix=unit, step_size=sampling_frequency, ignore_negative=True)
    _, rNPS_struct_filt = mmcalc.get_radial_profile(
        NPS_struct_filt, pix=unit, step_size=sampling_frequency, ignore_negative=True)
    if isinstance(eye_filter, list):
        _, rNPS_filt_2 = mmcalc.get_radial_profile(
            NPS_filt_2, pix=unit, step_size=sampling_frequency, ignore_negative=True)
        _, rNPS_struct_filt_2 = mmcalc.get_radial_profile(
            NPS_struct_filt_2, pix=unit, step_size=sampling_frequency,
            ignore_negative=True)

    if sni_dim == 0:
        ignore_negative = False  # ignore for testing - results show crappy data
        if ignore_negative:
            SNI = np.sum(NPS_struct_filt[NPS_struct_filt > 0]) / np.sum(NPS_filt)
        else:
            SNI = np.sum(NPS_struct_filt) / np.sum(NPS_filt)
        if isinstance(eye_filter, list):
            if ignore_negative:
                SNI_2 = np.sum(
                    NPS_struct_filt_2[NPS_struct_filt_2 > 0]) / np.sum(NPS_filt_2)
            else:
                SNI_2 = np.sum(NPS_struct_filt_2) / np.sum(NPS_filt_2)
    else:
        SNI = np.sum(rNPS_struct_filt) / np.sum(rNPS_filt)
        if isinstance(eye_filter, list):
            SNI_2 = np.sum(rNPS_struct_filt_2) / np.sum(rNPS_filt_2)

    details_dict_roi = {
        'NPS': NPS, 'quantum_noise': quantum_noise,
        'freq': freq, 'rNPS': rNPS, 'rNPS_filt': rNPS_filt, 'rNPS_filt_2': rNPS_filt_2,
        'rNPS_struct': rNPS_struct, 'rNPS_struct_filt': rNPS_struct_filt,
        'rNPS_struct_filt_2': rNPS_struct_filt_2,
        'rNPS_quantum_noise': rNPS_quantum_noise
        }
    return (SNI, SNI_2, details_dict_roi)


def calculate_NM_SNI(image2d, roi_array, image_info, paramset, reference_image):
    """Calculate Structured Noise Index.

    Parameters
    ----------
    image2d : numpy.ndarray
    roi_array : list of numpy.ndarray
        list of 2d masks for all_rois, full ROI, L1, L2, small ROIs
    image_info :  DcmInfo
        as defined in scripts/dcm.py
    paramset : cfc.ParamsetNM
    reference_image : numpy.ndarray or None

    Returns
    -------
    values : list of float
        according to defined headers in config/iQCconstants.py HEADERS
    details_dict : dict
    errmsgs : list of str
    """
    values_sup = [None] * 3
    details_dict = {}
    SNI_values = []
    SNI_values_2 = []
    errmsgs = []

    fit_dict = None
    if reference_image is not None:
        if reference_image.shape == image2d.shape:
            fit_dict = {}
            fit_dict['reference_image'] = reference_image
            reference_image = None
        else:
            errmsgs.append('Reference image not same size as image to analyse. '
                           'Quantum noise estimated from signal.')

    # point source correction
    if paramset.sni_correct:
        est_noise = False if reference_image is not None else True

        if reference_image and paramset.sni_ref_image_fit:
            fit_image = reference_image
        else:
            fit_image = image2d

        fit_dict, errmsg = get_corrections_point_source(
            fit_image, image_info, roi_array[0],
            fit_x=paramset.sni_correct_pos_x,
            fit_y=paramset.sni_correct_pos_y,
            lock_z=paramset.sni_lock_radius, guess_z=paramset.sni_radius,
            correction_type='subtract', estimate_noise=est_noise
            )
        if errmsg is not None:
            errmsgs.append(errmsg)
        values_sup = [fit_dict['dx'], fit_dict['dy'], fit_dict['distance']]

    block_size = paramset.sni_scale_factor
    if block_size > 1:
        # skip pixels excess of size/block_size
        sz_y, sz_x = image2d.shape
        skip_y, skip_x = (sz_y % block_size, sz_x % block_size)
        n_blocks_y, n_blocks_x = (sz_y // block_size, sz_x // block_size)
        start_y, start_x = (skip_y // 2, skip_x // 2)
        end_x = start_x + n_blocks_x * block_size
        end_y = start_y + n_blocks_y * block_size
        image2d = skimage.measure.block_reduce(
            image2d[start_y:end_y, start_x:end_x], (block_size, block_size), np.sum)
        '''
        for idx, arr in enumerate(roi_array):
            reduced_roi = skimage.measure.block_reduce(
                arr[start_y:end_y, start_x:end_x], (block_size, block_size), np.mean)
            roi_array[idx] = np.where(reduced_roi > 0.5, True, False)
        '''
        if fit_dict:
            if 'reference_image' in fit_dict:
                reference_image = skimage.measure.block_reduce(
                    reference_image[start_y:end_y, start_x:end_x],
                    (block_size, block_size), np.sum)
                fit_dict['reference_image'] = reference_image
        # recalculate rois (avoid block reduce on rois, might get non-quadratic)
        roi_array, errmsg = get_roi_SNI(image2d, image_info, paramset,
                                        block_size=block_size)

    if fit_dict is not None:
        details_dict = fit_dict

    arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[0]))
    image_mean = np.mean(arr)

    details_dict['pr_roi'] = []

    # large ROIs
    rows = np.max(roi_array[1], axis=1)
    eye_filter = get_eye_filter(
        np.count_nonzero(rows), image_info.pix[0]*block_size, paramset)
    details_dict['eye_filter_large'] = eye_filter['curve']
    for i in [1, 2]:
        SNI, SNI_2, details_dict_roi = calculate_SNI_ROI(
            image2d, roi_array[i],
            eye_filter=eye_filter['filter_2d'], unit=eye_filter['unit'],
            pix=image_info.pix[0]*block_size, fit_dict=fit_dict, image_mean=image_mean,
            sampling_frequency=paramset.sni_sampling_frequency,
            sni_dim=paramset.sni_ratio_dim)
        details_dict['pr_roi'].append(details_dict_roi)
        SNI_values.append(SNI)
        SNI_values_2.append(SNI_2)
    largest_L = '-'
    if SNI_values[0] > SNI_values[1]:
        largest_L = 'L1'
    elif SNI_values[0] < SNI_values[1]:
        largest_L = 'L2'
    if any(SNI_values_2):
        if SNI_values_2[0] > SNI_values_2[1]:
            largest_L = largest_L + ' / L1'
        elif SNI_values_2[0] < SNI_values_2[1]:
            largest_L = largest_L + ' / L2'
        else:
            largest_L + ' / -'
    values_sup.append(largest_L)

    if paramset.sni_type == 0:
        # small ROIs
        rows = np.max(roi_array[3], axis=1)
        eye_filter = get_eye_filter(
            np.count_nonzero(rows), image_info.pix[0]*block_size, paramset)
        details_dict['eye_filter_small'] = eye_filter['curve']
        for i in range(3, 9):
            SNI, SNI_2, details_dict_roi = calculate_SNI_ROI(
                image2d, roi_array[i],
                eye_filter['filter_2d'], unit=eye_filter['unit'],
                pix=image_info.pix[0]*block_size, fit_dict=fit_dict, image_mean=image_mean,
                sampling_frequency=paramset.sni_sampling_frequency,
                sni_dim=paramset.sni_ratio_dim)
            details_dict['pr_roi'].append(details_dict_roi)
            SNI_values.append(SNI)
            SNI_values_2.append(SNI_2)

        if paramset.sni_channels:
            values = (
                SNI_values[0:2]
                + [np.max(SNI_values[2:])] + [np.mean(SNI_values[2:])]
                + SNI_values_2[0:2]
                + [np.max(SNI_values_2[2:])] + [np.mean(SNI_values_2[2:])]
                )
        else:
            values = [np.max(SNI_values)] + SNI_values
        maxno = SNI_values[2:].index(max(SNI_values[2:]))
        if any(SNI_values_2):
            idx = SNI_values_2[2:].index(max(SNI_values_2[2:]))
            maxno_2 = f' / S{idx+1}'
        else:
            maxno_2 = ''
        values_sup.append(f'S{maxno+1}{maxno_2}')
        details_dict['SNI_values'] = SNI_values
        if paramset.sni_channels:
            details_dict['SNI_values_2'] = SNI_values_2
    else:
        idx_roi_row_0 = 3
        try:
            rows = np.max(roi_array[idx_roi_row_0][0], axis=1)
        except (np.AxisError, TypeError):  # if first is ignored Siemens
            rows = np.max(roi_array[idx_roi_row_0 + 2][3], axis=1)  # central
            # TODO better catch - what if this also (not likely) is None?
        eye_filter = get_eye_filter(
            np.count_nonzero(rows), image_info.pix[0]*block_size, paramset)
        details_dict['eye_filter_small'] = eye_filter['curve']

        if paramset.sni_type in [1, 2]:
            SNI_map = np.zeros(
                (len(roi_array)-idx_roi_row_0, len(roi_array[idx_roi_row_0])))
            SNI_map_2 = np.copy(SNI_map)
        else:  # 3 Siemens
            SNI_map = []
            SNI_map_2 = []
        rNPS_filt_sum = None
        rNPS_struct_filt_sum = None
        rNPS_filt_2_sum = None
        rNPS_struct_filt_2_sum = None
        small_names = []
        for rowno, row in enumerate(roi_array[idx_roi_row_0:]):
            for colno, roi in enumerate(row):
                small_names.append(f'r{rowno}_c{colno}')
                if roi is not None:
                    SNI, SNI_2, details_dict_roi = calculate_SNI_ROI(
                        image2d, roi,
                        eye_filter['filter_2d'], unit=eye_filter['unit'],
                        pix=image_info.pix[0]*block_size, fit_dict=fit_dict,
                        image_mean=image_mean,
                        sampling_frequency=paramset.sni_sampling_frequency,
                        sni_dim=paramset.sni_ratio_dim)
                    details_dict['pr_roi'].append(details_dict_roi)
                    if paramset.sni_type in [1, 2]:
                        SNI_map[rowno, colno] = SNI
                        SNI_map_2[rowno, colno] = SNI_2
                    else:  # 3 Siemens
                        SNI_map.append(SNI)
                        SNI_map_2.append(SNI_2)
                    SNI_values.append(SNI)
                    SNI_values_2.append(SNI_2)
                    if rNPS_filt_sum is None:
                        rNPS_filt_sum = details_dict_roi['rNPS_filt']
                        rNPS_struct_filt_sum = details_dict_roi['rNPS_struct_filt']
                    else:
                        rNPS_filt_sum = rNPS_filt_sum + details_dict_roi['rNPS_filt']
                        rNPS_struct_filt_sum = (
                            rNPS_struct_filt_sum + details_dict_roi['rNPS_struct_filt'])
                    if paramset.sni_channels:
                        if rNPS_filt_2_sum is None:
                            rNPS_filt_2_sum = details_dict_roi['rNPS_filt_2']
                            rNPS_struct_filt_2_sum = details_dict_roi['rNPS_struct_filt_2']
                        else:
                            rNPS_filt_2_sum = rNPS_filt_2_sum + details_dict_roi['rNPS_filt_2']
                            rNPS_struct_filt_2_sum = (
                                rNPS_struct_filt_2_sum + details_dict_roi['rNPS_struct_filt_2'])
                else:
                    details_dict['pr_roi'].append(None)
                    SNI_values.append(np.nan)
                    SNI_values_2.append(np.nan)
                    if paramset.sni_type == 3:  # always? if ignored
                        SNI_map.append(np.nan)
                        SNI_map_2.append(np.nan)
        values = []
        max_nos = []
        if paramset.sni_type == 3:  # avoid np.nan for ignored
            if paramset.sni_channels:
                arrs = [np.array(SNI_map), np.array(SNI_map_2)]
                vals = [SNI_values, SNI_values_2]
            else:
                arrs = [np.array(SNI_map)]
                vals = [SNI_values]
            for idx, arr in enumerate(arrs):
                max_no = np.where(arr == np.max(arr[arr > -1]))
                max_nos.append(max_no[0][0])
                values.extend([
                    vals[idx][0], vals[idx][1],
                    np.max(arr[arr > -1]),
                    np.mean(arr[arr > -1])
                    ])
                if paramset.sni_channels is False:
                    values.append(np.median(arr[arr > -1]))
        else:
            if paramset.sni_channels:
                vals = [SNI_values, SNI_values_2]
            else:
                vals = [SNI_values]
            for vals_this in vals:
                SNI_small = vals_this[2:]
                max_no = np.where(SNI_small == np.max(SNI_small))
                max_nos.append(max_no[0][0])
                values.extend(
                    SNI_values[0:2]
                    + [np.max(SNI_map), np.mean(SNI_map)])
                if paramset.sni_channels is False:
                    values.append(np.median(SNI_map))
        details_dict['roi_max_idx_small'] = max_nos[0]
        txt_small_max = small_names[max_nos[0]]
        if paramset.sni_channels:
            details_dict['roi_max_idx_small_2'] = max_nos[1]
            txt_small_max = txt_small_max + ' / ' + small_names[max_nos[1]]
        values_sup.append(txt_small_max)

        details_dict['avg_rNPS_filt_small'] = rNPS_filt_sum / len(SNI_values)
        details_dict['avg_rNPS_struct_filt_small'] = (
            rNPS_struct_filt_sum / len(SNI_values))
        details_dict['SNI_map'] = SNI_map
        details_dict['SNI_values'] = SNI_values
        if paramset.sni_channels:
            details_dict['avg_rNPS_filt_2_small'] = rNPS_filt_2_sum / len(SNI_values_2)
            details_dict['avg_rNPS_struct_filt_2_small'] = (
                rNPS_struct_filt_2_sum / len(SNI_values_2))
            details_dict['SNI_map_2'] = SNI_map_2
            details_dict['SNI_values_2'] = SNI_values_2

    return (values, values_sup, details_dict, errmsgs)


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
                    this_roi_peak[peak_idx - n_slice_kernel // 2 + i] = roi_this

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
