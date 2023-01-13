#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculation processes for the different tests.

@author: Ellen Wasbø
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import scipy as sp
import skimage

# imageQC block start
import imageQC.scripts.dcm as dcm
from imageQC.scripts.calculate_roi import get_rois
from imageQC.scripts.mini_methods import get_min_max_pos_2d
from imageQC.scripts.mini_methods_format import val_2_str
import imageQC.scripts.mini_methods as mm
import imageQC.scripts.mini_methods_calculate as mmcalc
from imageQC.config.iQCconstants import HEADERS, HEADERS_SUP
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
        columns = list(np.arange(len(values)))
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
    n_imgs = len(input_main.imgs)
    image_names = [f'img{i}' for i in range(n_imgs)]
    group_names = input_main.current_quicktest.group_names
    if input_main.results != {}:
        # image_names to headers?
        if input_main.current_paramset.output.include_header:
            set_names = input_main.current_quicktest.image_names
            if any(set_names):
                for i in range(n_imgs):
                    if set_names[i] != '':
                        image_names[i] = set_names[i]

        #TODO if output_temp.transpose_table:
        #TODO if output_temp.include_filename:

        include_header = True  # always included when automation if empty output file
        if input_main.automation_active is False:
            include_header = input_main.current_paramset.output.include_header

        for test in input_main.results:
            if test in input_main.current_paramset.output.tests:
                output_subs = input_main.current_paramset.output.tests[test]
            else:
                output_subs = [cfc.QuickTestOutputSub(columns=[])]  # default

            dm = input_main.current_paramset.output.decimal_mark

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
                        actual_group_ids = []
                        actual_values = []
                        actual_group_names = []
                        actual_image_names = []
                        for r, row in enumerate(values):
                            if any(row):
                                actual_values.append(row)
                                actual_group_ids.append(
                                    input_main.current_group_indicators[r])
                                actual_image_names.append(image_names[r])
                                actual_group_names.append(group_names[r])
                        uniq_group_ids = list(set(actual_group_ids))
                        for g, group_id in enumerate(uniq_group_ids):
                            values_this = []
                            group_names_this = []
                            image_names_this = []
                            proceed = True
                            while proceed:
                                if group_id in actual_group_ids:
                                    idx = actual_group_ids.index(group_id)
                                    values_this.append(actual_values[idx])
                                    group_names_this.append(actual_group_names[idx])
                                    image_names_this.append(actual_image_names[idx])
                                    del actual_values[idx]
                                    del actual_group_ids[idx]
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
                                val_2_str(out_values, decimal_mark=dm)
                                )
                            if len(out_values) > 1:
                                suffixes.append(image_names_this)
                            else:
                                if any(group_names_this):
                                    all_group_names = list(set(group_names_this))
                                    try:
                                        idx_empty = all_group_names.index('')
                                        del all_group_names[idx_empty]
                                    except ValueError:
                                        pass
                                    suffixes.append(all_group_names[0])
                                else:
                                    suffixes.append(f'group{g}')

                    else:  # each image individually
                        for r, row in enumerate(values):
                            if any(row):
                                out_values = extract_values(
                                    row,
                                    columns=sub.columns,
                                    calculation=sub.calculation
                                    )
                                string_list.extend(
                                    val_2_str(out_values, decimal_mark=dm)
                                    )
                                suffixes.append(image_names[r])

                    if include_header:
                        # output label or table header + image_name or group_name
                        if len(out_values) > 1:  # as is or group/calculation failed
                            if sub.columns == []:
                                headers_this = headers
                            else:
                                headers_this = []
                                for c, header in enumerate(headers):
                                    if c in sub.columns:
                                        headers_this.append(header)
                        else:
                            headers_this = [sub.label]
                        all_headers_this = []
                        for suffix in suffixes:
                            for header in headers_this:
                                all_headers_this.append(header + '_' + suffix)
                        header_list.extend(all_headers_this)
                    breakpoint()

    return (string_list, header_list)


def calculate_qc(input_main):
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
    """
    current_test_before = input_main.current_test
    delta_xya = [0, 0, 0.0]  # only for GUI version
    modality = input_main.current_modality
    pre_msg = ''
    errmsgs = []
    if 'MainWindow' in str(type(input_main)):
        if input_main.automation_active:
            pre_msg = input_main.statusBar.text()
        else:
            input_main.start_wait_cursor()
            input_main.statusBar.showMessage('Calculating...')
        delta_xya = [
            input_main.vGUI.delta_x,
            input_main.vGUI.delta_y,
            input_main.vGUI.delta_a]

    paramset = input_main.current_paramset
    img_infos = input_main.imgs
    tag_infos = input_main.tag_infos
    quicktest = input_main.current_quicktest

    proceed = True
    if len(img_infos) == 0:
        proceed = False

    if proceed:
        # load marked images (if not only DCM test)
        # run 2d tests while loading next image
        # if any 3d tests - keep loaded images to build 3d arrays
        n_img = len(img_infos)
        marked = quicktest.tests

        # get list of images to read (either as tags or image)
        flattened_marked = [elem for sublist in marked for elem in sublist]
        if len(flattened_marked) > 0:
            read_tags = [False] * n_img
            read_image = [False] * n_img
            for i in range(n_img):
                if len(marked[i]) > 0:
                    if 'DCM' in marked[i]:
                        read_tags[i] = True
                    if marked[i].count('DCM') != len(marked[i]):
                        read_image[i] = True  # not only DCM

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
            marked_3d = []
            input_main.current_group_indicators = [[''] for i in range(n_img)]
            for i in range(n_img):
                marked_3d.append([])
                if modality == 'CT':
                    if 'MTF' in marked[i]:
                        if paramset.mtf_type > 0:
                            marked_3d[i].append('MTF')
                elif 'Uni' in marked[i]:
                    if paramset.uni_sum_first:
                        marked_3d[i].append('Uni')
                elif 'SNI' in marked[i]:
                    if paramset.sni_sum_first:
                        marked_3d[i].append('SNI')
                elif modality == 'SPECT':
                    if 'MTF' in marked[i]:
                        if paramset.mtf_3d:
                            marked_3d[i].append('MTF')
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
            for i in range(n_img):
                if 'MainWindow' in str(type(input_main)):
                    input_main.statusBar.showMessage(
                        f'{pre_msg} Reading image data {i} of {n_img}')
                # read image or tags as needed
                group_pattern = cfc.TagPatternFormat(list_tags=paramset.output.group_by)
                if read_tags[i]:
                    if read_image[i]:
                        image, tags = dcm.get_img(
                            img_infos[i].filepath,
                            frame_number=img_infos[i].frame_number,
                            tag_patterns=[paramset.dcm_tagpattern, group_pattern],
                            tag_infos=tag_infos
                            )
                        tag_lists[i] = tags[0]
                        input_main.current_group_indicators[i] = '_'.join(tags[1])
                    else:
                        tags = dcm.get_tags(
                            img_infos[i].filepath,
                            frame_number=img_infos[i].frame_number,
                            tag_patterns=[paramset.dcm_tagpattern, group_pattern],
                            tag_infos=tag_infos
                            )
                        tag_lists[i] = tags[0]
                        input_main.current_group_indicators[i] = '_'.join(tags[1])
                else:
                    if read_image[i]:
                        image, tags = dcm.get_img(
                            img_infos[i].filepath,
                            frame_number=img_infos[i].frame_number,
                            tag_patterns=[group_pattern],
                            tag_infos=tag_infos
                            )
                        input_main.current_group_indicators[i] = '_'.join(tags[0])

                for test in marked[i]:
                    input_main.current_test = test
                    if test not in ['DCM', '']:
                        if test in marked_3d[i]:
                            if matrix[i] is None:  # only once
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
                                    errmsgs.append(f'{test} get ROI slice{i}:')
                                    errmsgs.append(errmsg)

                            result = calculate_2d(
                                image, prev_roi[test], img_infos[i],
                                modality, paramset, test, delta_xya)
                            if result is not None:
                                if result.errmsg is not None:
                                    if len(result.errmsg) > 0:
                                        errmsgs.append(f'{test} slice {i}:')
                                        if isinstance(result.errmsg, str):
                                            errmsgs.append(result.errmsg)
                                        else:
                                            errmsgs.extend(result.errmsg)
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
                    input_main.results['Noi']['pr_image'] = False
                    noise = [
                        row[1] for row in input_main.results['Noi']['values']]
                    avg_noise = sum(noise)/len(noise)
                    for row in input_main.results['Noi']['values']:
                        # diff from avg (%)
                        row[2] = 100.0 * (row[1] - avg_noise) / avg_noise
                        row[3] = avg_noise
                if 'Dim' in flattened_marked:
                    if 'MainWindow' in str(type(input_main)):
                        input_main.update_roi()

            if any(marked_3d):
                results_dict = calculate_3d(
                    matrix, marked_3d, input_main)
                for test in results_dict:
                    if results_dict[test].errmsg is not None:
                        if len(results_dict[test].errmsg) > 0:
                            errmsgs.append(f'{test} slice {i}:')
                        if isinstance(results_dict[test].errmsg, str):
                            errmsgs.append(results_dict[test].errmsg)
                        else:
                            errmsgs.extend(results_dict[test].errmsg)

                    input_main.results[test] = asdict(results_dict[test])

            # For test DCM
            if any(read_tags):
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
                input_main.statusBar.showMessage('Finished', 1000)

    if 'MainWindow' in str(type(input_main)):
        if input_main.automation_active is False:
            input_main.current_test = current_test_before
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
            if image2d is not None:
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
                values, details_dict, errmsg = calculate_slicethickness_CT(
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
            if image2d is not None:
                res = Results(headers=headers)
                #TODO
            else:
                res = Results(headers=headers)
        else:
            res = Results()

        return res

    def MTF():
        if modality == 'CT':
            # only bead method 2d (alt0)
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
                arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[1]))
                background = np.mean(arr)
                details = calculate_MTF_point(
                    sub - background, image_info, paramset)
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
                    details_dict=details, alternative=alt)

        elif modality == 'Xray':
            alt = paramset.mtf_type
            headers = HEADERS[modality][test_code]['alt0']
            if image2d is None:
                res = Results(headers=headers, headers_sup=headers_sup)
            else:
                sub = []
                if isinstance(roi_array, list):
                    for roi in roi_array:
                        rows = np.max(roi, axis=1)
                        cols = np.max(roi, axis=0)
                        sub.append(image2d[rows][:, cols])
                else:
                    rows = np.max(roi_array, axis=1)
                    cols = np.max(roi_array, axis=0)
                    sub.append(image2d[rows][:, cols])

                details, errmsg = calculate_MTF_edge(
                    sub, image_info.pix[0], paramset)
                prefix = 'g' if paramset.mtf_gaussian else 'd'
                values = details[prefix + 'MTF_details']['values']

                res = Results(
                    headers=headers, values=values,
                    details_dict=details, alternative=alt, errmsg=errmsg)

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

    def Uni():
        headers = HEADERS[modality][test_code]['alt0']
        headers_sup = HEADERS_SUP[modality][test_code]['altAll']
        if image2d is None:
            res = Results(headers=headers, headers_sup=headers_sup)
        else:
            details_dict = {}
            values_sup = [None] * 3
            if paramset.uni_correct:
                res = get_corrections_point_source(
                    image2d, image_info, roi_array[0],
                    fit_x=paramset.uni_correct_pos_x,
                    fit_y=paramset.uni_correct_pos_y,
                    lock_z=paramset.uni_correct_radius
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
                details_dict=details_dict)

        return res

    def SNI():
        headers = HEADERS[modality][test_code]['alt0']
        headers_sup = HEADERS_SUP[modality][test_code]['altAll']
        if image2d is None:
            res = Results(headers=headers, headers_sup=headers_sup)
        else:
            details_dict = {}
            values_sup = [None] * 3
            if paramset.sni_correct:
                res = get_corrections_point_source(
                    image2d, image_info, roi_array[0],
                    fit_x=paramset.sni_correct_pos_x,
                    fit_y=paramset.sni_correct_pos_y,
                    lock_z=paramset.sni_correct_radius
                    )
                image_input = res['corrected_image']
                values_sup = [res['dx'], res['dy'], res['distance']]
                details_dict = res
            else:
                image_input = image2d
            res = calculate_NM_SNI(
                image_input, roi_array, image_info.pix[0])
            values = res['values']

            res = Results(
                headers=headers, values=values,
                headers_sup=headers_sup, values_sup=values_sup,
                details_dict=details_dict)

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

            min_idx, max_idx = get_min_max_pos_2d(image2d, roi_array)
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

    try:
        result = locals()[test_code]()
    except KeyError:
        result = None

    return result


def calculate_3d(matrix, marked_3d, input_main):
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
        if modality == 'CT':
            #TODO verify same kernel all images to test
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
                rows = np.max(roi_array, axis=1)
                cols = np.max(roi_array, axis=0)
                sub = []
                for sli in matrix:
                    if sli is not None:
                        sub.append(sli[rows][:, cols])
                    else:
                        sub.append(None)

                if paramset.mtf_type == 1:  # wire
                    if len(images_to_test) > 2:
                        details_dict, errmsg = calculate_MTF_3d_line(
                            sub, roi_array[rows][:, cols], images_to_test,
                            img_infos, paramset)
                    else:
                        pass#TODO errormessage not possible with < 3 slices
                elif paramset.mtf_type == 2:  # circular disc
                    details_dict, errmsg = calculate_MTF_circular_edge(
                        sub, roi_array[rows][:, cols],
                        img_infos[images_to_test[0]].pix[0], paramset)

                prefix = 'g' if paramset.mtf_gaussian else 'd'
                values = details_dict[prefix + 'MTF_details']['values']
                values_sup = details_dict['gMTF_details']['LSF_fit_params']
                details_dict['matrix'] = sub

                res = Results(headers=headers, values=[values],
                              headers_sup=headers_sup, values_sup=[values_sup],
                              details_dict=[details_dict],
                              alternative=paramset.mtf_type, pr_image=False,
                              errmsg=errmsg)
        else:
            res = None

        return res

    def SNR(images_to_test):
        # use two and two images
        values_first_imgs = []
        idxs_first_imgs = []
        headers = HEADERS[modality][test_code]['alt0']
        if matrix.count(None) == len(matrix) or len(images_to_test) == 1:
            res = Results(headers=headers)
        else:
            n_pairs = len(images_to_test)/2
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
                        idxs_first_imgs.append([idx1])

            values = [[] for i in range(len(img_infos))]
            if len(idxs_first_imgs) > 0:
                for res_i, img_i in enumerate(len(idxs_first_imgs)):
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
        #headers_sup = HEADERS_SUP[modality][test_code]['altAll']
        if len(images_to_test) == 0:
            res = Results(headers=headers)
        else:
            image_input = sum_matrix(images_to_test)  # always if 3d option
            #TODO warning that corrections are ignored, makes no sence in 3d option
            roi_array, errmsg = get_rois(
                image_input, images_to_test[0], input_main)
            res = calculate_NM_uniformity(
                image_input, roi_array, img_infos[0].pix[0])
            values = res['values']

            res = Results(headers=headers, values=[values], errmsg=errmsg, pr_image=False)

        return res

    def SNI(images_to_test):
        headers = HEADERS[modality][test_code]['alt0']
        if len(images_to_test) == 0:
            res = Results(headers=headers)
        else:
            image_input = sum_matrix(images_to_test)  # always if 3d option
            #TODO warning that corrections are ignored, makes no sence in 3d option
            roi_array, errmsg = get_rois(
                image_input, images_to_test[0], input_main)
            res = calculate_NM_SNI(
                image_input, roi_array, img_infos.pix[0])
            details_dict = {
                'matrix': res['matrix'],
                'du_matrix': res['du_matrix']
                }
            values = res['values']

            res = Results(headers=headers, values=[values],
                          details_dict=[details_dict], errmsg=errmsg, pr_image=False)

        return res

    for test_code in all_tests:
        images_to_test = []
        for i, tests in enumerate(marked_3d):
            if test_code in tests:
                images_to_test.append(i)

        try:
            result = locals()[test_code](images_to_test)
        except KeyError:
            result = None

        if result is not None:
            results[test_code] = result

    return results


def get_distance_map_point(shape, center_dx=0., center_dy=0.):
    """Calculate distances from center point in image (optionally with offset).

    Parameters
    ----------
    shape : tuple
        shape of array do generate
    center_dx : float, optional
        offset from center in shape. The default is 0..
    center_dy : float, optional
        offset from center in shape. The default is 0..

    Returns
    -------
    distance_map : ndarray
        of shape equal to input shape
    """
    sz_y, sz_x = shape
    xs, ys = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
    center_pos_x = center_dx + 0.5 * sz_x
    center_pos_y = center_dy + 0.5 * sz_y
    distance_map = np.sqrt((xs-center_pos_x) ** 2 + (ys-center_pos_y) ** 2)

    return distance_map


def get_distance_map_edge(shape, slope=0., intercept=0.):
    """Calculate distances from edge defined by y = ax + b.

    distance from x0,y0 normal to line = (b + a*x0 - y0)/sqrt(1+a^2)
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_

    Parameters
    ----------
    shape : tuple
        shape of 2darray to generate map for
    slope : float, optional
        a in y = ax + b. The default is 0.
    intercept : float, optional
        b in y = ax + b. The default is 0.

    Returns
    -------
    distance_map : 2darray
        of shape equal to input shape
    """
    sz_y, sz_x = shape
    xs, ys = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
    distance_map = (ys - intercept - slope*xs) / np.sqrt(1 + slope**2)

    return distance_map


def calculate_slicethickness_CT(image, img_info, paramset, lines, delta_xya):
    """Calculate slice thickness for CT.

    Parameters
    ----------
    image : numpy.ndarray
        2d image
    img_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetCT
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
    if paramset.sli_type == 0:
        details_dict['labels'] = ['upper H1', 'lower H2', 'left V1', 'right V2']
    elif paramset.sli_type == 1:
        details_dict['labels'] = [
            'upper H1', 'lower H2', 'left V1', 'right V2', 'inner V1', 'inner V2']
    elif paramset.sli_type == 2:
        details_dict['labels'] = ['left', 'right']
        sli_signal_low_density = True

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

        if paramset.sli_type == 0:  # wire ramp Catphan
            width, center = mmcalc.get_width_center_at_threshold(
                profile, halfmax, force_above=True)
            if width is not None:
                slice_thickness = 0.42 * width * img_info.pix[0]  # 0.42*FWHM
                if delta_xya[2] != 0:
                    slice_thickness = slice_thickness / np.cos(
                        np.deg2rad(delta_xya[2]))
                details_dict['start_x'].append((center - 0.5 * width) * img_info.pix[0])
                details_dict['end_x'].append((center + 0.5 * width) * img_info.pix[0])
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
                slice_thickness = (z_increment/xy_increment) * width * img_info.pix[0]
                details_dict['start_x'].append((center - 0.5 * width) * img_info.pix[0])
                details_dict['end_x'].append((center + 0.5 * width) * img_info.pix[0])
            else:
                details_dict['start_x'].append(0)
                details_dict['end_x'].append(0)

        if slice_thickness is None:
            errmsgs.append(
                f'{details_dict["labels"][pno]}: failed finding slicethickness')
            if delta_xya[2] != 0:
                slice_thickness = slice_thickness / np.cos(np.deg2rad(delta_xya[2]))

        values.append(slice_thickness)

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
    sli_signal_low_density = True if paramset.sli_type == 2 else False

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
    """
    values = []
    details = []
    for ax in [1, 0]:
        details_dict = {}
        profile = np.sum(matrix, axis=ax)
        width, center = mmcalc.get_width_center_at_threshold(
            profile, np.max(profile)/2)
        if center is not None:
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
            res['values'] = mmcalc.get_curve_values(
                res['MTF_freq'], res['MTF'], [0.5, 0.1, 0.02])
            details_dict['dMTF_details'] = res

            # Gaussian MTF
            if isinstance(paramset, cfc.ParamSetCT):
                gaussfit = 'double'
            else:
                gaussfit = 'single'
            res = mmcalc.get_MTF_gauss(profile, dx=img_info.pix[0], gaussfit=gaussfit)
            res['values'] = mmcalc.get_curve_values(
                res['MTF_freq'], res['MTF'], [0.5, 0.1, 0.02])
            details_dict['gMTF_details'] = res

            details.append(details_dict)

    return details


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
    values : list of float
        [MTF 50%, 10%, 2%]
    values_sup : list of float
        [A1 mu1, sigma1, A2, mu2, sigma2]
    details_dict: dict
    """
    details_dict = {}

    # find center of line for each slice
    center_xy = []
    for sli in matrix:
        if sli is not None:
            center_xy.append(mmcalc.center_xy_of_disc(sli, roi=roi, mode='max'))
    center_x = np.array([vals[0] for vals in center_xy])
    center_y = np.array([vals[1] for vals in center_xy])
    zpos = np.array([image_infos[sli].zpos for sli in images_to_test])
    # linear fit x, y
    fit_xy = []
    for values in [center_x, center_y]:
        #try:
        res = sp.stats.linregress(values, zpos)
        fit_xy.append(zpos * res.slope + res.intercept)
    #TODO if valid so far... else proceed False
    proceed = True

    if proceed:
        pix = image_infos[images_to_test[0]].pix[0]

        # sort pixel values from center
        dist_map = get_distance_map_point(
            matrix[0].shape,
            center_dx=center_xy[0] - matrix[0].shape[1]/2,
            center_dy=center_xy[1] - matrix[0].shape[0]/2)
        dists_flat = dist_map.flatten()
        sort_idxs = np.argsort(dists_flat)
        dists = dists_flat[sort_idxs]
        dists = pix * dists
        values_all = []
        values_sum = np.zeros(dists.shape)
        nsum = 0
        for sli in matrix:
            if sli is not None:
                values_flat = sli.flatten()
                values_all.append(values_flat[sort_idxs])
                values_sum = np.add(values_sum, values_flat[sort_idxs])
                nsum += 1
        values_avg = 1/nsum * values_sum

        # ignore dists > radius
        radius = 0.5 * matrix[0].shape[0] * pix
        dists_cut = dists[dists < radius]
        values = values_avg[dists < radius]

        # reduce noise using small window width ~0.1*pix assumed to not affect edge
        diff_x = np.diff(dists_cut)
        sigma_n = 0.1 * pix // np.median(diff_x)  # TODO *.5?
        if sigma_n > 0:
            values = sp.ndimage.gaussian_filter(values, sigma=sigma_n)

        # rebin to 1/10th pix size
        step_size = .1 * pix
        new_x, new_y = mmcalc.resample(
            input_y=values, input_x=dists_cut, step=step_size)

        # smooth before gaussian fit, with known gaussian filter
        # (to be corrected for in gaussian mtf)
        sigma_f = 5.  # if sigma_f=5 , FWHM ~9 newpix = ~ 1 original pix
        values_f = sp.ndimage.gaussian_filter(new_y, sigma=sigma_f)

        # calculate LSF from ESF
        LSF = np.diff(values_f)  # for gaussian fit
        LSF_no_filt = np.diff(new_y)  # for discrete MTF
        if abs(min(LSF)) > abs(max(LSF)):  # ensure positive gauss
            LSF = -1. * LSF
            LSF_no_filt = -1. * LSF_no_filt
        width, center = mmcalc.get_width_center_at_threshold(LSF, np.max(LSF)/2)

        if center is not None:
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

            gMTF_details = mmcalc.get_MTF_gauss(
                LSF, dx=step_size, gaussfit='double')

            gMTF_details['values'] = mmcalc.get_curve_values(
                    gMTF_details['MTF_freq'], gMTF_details['MTF'], [0.5, 0.1, 0.02])
            dMTF_details['values'] = mmcalc.get_curve_values(
                    dMTF_details['MTF_freq'], dMTF_details['MTF'], [0.5, 0.1, 0.02],
                    force_first_below=True)

            details_dict = {
                'center_xy': center_xy,
                'LSF_x': LSF_x, 'LSF': LSF_no_filt,
                'sorted_pixels_x': dists, 'sorted_pixels': values_all,
                'interpolated_x': new_x, 'interpolated': new_y, 'presmoothed': values_f,
                'dMTF_details': dMTF_details, 'gMTF_details': gMTF_details}

    return details_dict


def calculate_MTF_edge(matrix, pix, paramset):
    """Calculate MTF from straight edge.

    Parameters
    ----------
    matrix : numpy.2darray or list of numpy.2darray
        image within roi(s)
    pix : float
        pixelsize in mm
    paramset : ParamSetXX
        depending on modality

    Returns
    -------
    details_dict: dict
    errmsg: list
    """
    details_dict = {}
    details_dicts_edge = []  # one for each roi (edge)
    ESF_all = []
    ESF_x = None
    errmsg = []
    step_size = .1 * pix

    if not isinstance(matrix, list):
        matrix = [matrix]

    for m in range(len(matrix)):
        # rotate matrix if edge not in y direction
        x1 = np.mean(matrix[m][:, :2])
        x2 = np.mean(matrix[m][:, -2:])
        diff_x = abs(x1 - x2)
        y1 = np.mean(matrix[m][:2, :])
        y2 = np.mean(matrix[m][-2:, :])
        diff_y = abs(y1 - y2)
        if diff_x < diff_y:
            matrix[m] = np.rot90(matrix[m])
            halfmax = 0.5 * (y1 + y2)
        else:
            halfmax = 0.5 * (x1 + x2)

        # find edge
        smoothed = sp.ndimage.gaussian_filter(matrix[m], sigma=3)
        edge_pos = []
        x = np.arange(smoothed.shape[1])
        for i in range(smoothed.shape[0]):
            res = mmcalc.get_curve_values(x, smoothed[i, :], [halfmax])
            edge_pos.append(res[0])
        proceed = True
        ys = np.arange(smoothed.shape[0])

        if None in edge_pos:
            n_found = len(edge_pos) - edge_pos.count(None)
            if n_found < 0.5 * len(edge_pos):
                proceed = False
                txt_m = f'{m}' if len(matrix) > 1 else ''
                errmsg.append(
                    f'Edge position found for < 50% of ROI {txt_m}. Test failed.')
            else:
                idx_None = mm.get_all_matches(edge_pos, None)
                edge_pos = [e for i, e in enumerate(edge_pos) if i not in idx_None]
                ys = [y for i, y in enumerate(list(ys)) if i not in idx_None]
            if errmsg == []:
                errmsg.append(
                    'Edge position not found for full ROI. Parts of ROI ignored.')

        if proceed:
            matrix_this = matrix[m]
            # linear fit of edge positions
            res = sp.stats.linregress(ys, edge_pos)  # x = ay + b
            slope = 1./res.slope  # to y = (1/a)x + (-b/a) to avoid error when steep
            intercept = - res.intercept / res.slope
            x_fit = np.array([min(edge_pos), max(edge_pos)])
            y_fit = slope * x_fit + intercept
            angle = np.abs((180/np.pi) * np.arctan(
                (x_fit[1]-x_fit[0])/(y_fit[1]-y_fit[0])
                ))  # TODO *pix(0)/pix(1) if pix not isotropic

            # sort pixels by position normal to edge
            dist_map = get_distance_map_edge(
                matrix_this.shape, slope=slope, intercept=intercept)
            dist_map_flat = dist_map.flatten()
            values_flat = matrix_this.flatten()
            sort_idxs = np.argsort(dist_map_flat)
            dists = dist_map_flat[sort_idxs]
            sorted_values = values_flat[sort_idxs]
            dists = pix * dists

            details_dicts_edge.append({
                'edge_pos': edge_pos, 'edge_row': ys,
                'edge_fit_x': x_fit, 'edge_fit_y': y_fit,
                'edge_r2': res.rvalue**2, 'angle': angle,
                'sorted_pixels_x': dists,
                'sorted_pixels': sorted_values
                })

            # reduce noise by narrow filter and resample to 1/10th pix size
            diff_x = np.diff(dists)
            if step_size // np.median(diff_x) > 10:  # this value is >> 100 for my test case....
                sorted_values = sp.ndimage.gaussian_filter(
                    sorted_values, sigma=3)  # TODO test with sparce data (NM)
            new_x, new_y = mmcalc.resample(
                input_y=sorted_values, input_x=dists,
                step=step_size,
                first_step=-matrix_this.shape[1]/2 * pix,
                n_steps=10*matrix_this.shape[1])
            if ESF_x is None:
                ESF_x = new_x
            ESF_all.append(new_y)

    if len(ESF_all) > 0:
        sigma_f = 5.  # if sigma_f=5 , FWHM ~9 newpix = ~ 1 original pix
        LSF_all = []
        LSF_no_filt_all = []
        for ESF in ESF_all:
            LSF, LSF_no_filt, _ = mmcalc.ESF_to_LSF(ESF, prefilter_sigma=sigma_f)
            LSF_all.append(LSF)
            LSF_no_filt_all.append(LSF_no_filt)
        if len(LSF_all) > 1:
            LSF = np.mean(np.array(LSF_all), axis=0)
            LSF_no_filt = np.mean(np.array(LSF_no_filt_all), axis=0)
        else:
            LSF = LSF_all[0]
            LSF_no_filt = LSF_no_filt_all[0]

        width, center = mmcalc.get_width_center_at_threshold(LSF, np.max(LSF)/2)

        if width is not None:
            # Calculate gaussian and discrete MTF
            cw = 0
            try:
                cut_lsf = paramset.mtf_cut_lsf
                cut_lsf_fwhm = paramset.mtf_cut_lsf_w
            except AttributeError:
                cut_lsf = False
                cut_lsf_fwhm = None
            if cut_lsf:
                LSF_no_filt, cw, _ = mmcalc.cut_and_fade_LSF(
                    LSF_no_filt, center=center, fwhm=width,
                    cut_width=cut_lsf_fwhm)
            dMTF_details = mmcalc.get_MTF_discrete(LSF_no_filt, dx=step_size)
            dMTF_details['cut_width'] = cw * step_size

            LSF_x = step_size * (np.arange(LSF.size) - center)

            gMTF_details = mmcalc.get_MTF_gauss(
                LSF, dx=step_size, prefilter_sigma=sigma_f*step_size, gaussfit='single')

            lp_vals = [0.5, 1, 1.5, 2, 2.5]
            gMTF_details['values'] = mmcalc.get_curve_values(
                    gMTF_details['MTF'], gMTF_details['MTF_freq'], lp_vals)
            gMTF_details['values'].extend(mmcalc.get_curve_values(
                    gMTF_details['MTF_freq'], gMTF_details['MTF'], [0.5]))
            dMTF_details['values'] = mmcalc.get_curve_values(
                    dMTF_details['MTF'], dMTF_details['MTF_freq'], lp_vals,
                    force_first_below=True)
            dMTF_details['values'].extend(mmcalc.get_curve_values(
                    dMTF_details['MTF_freq'], dMTF_details['MTF'], [0.5],
                    force_first_below=True))

            details_dict = {
                'LSF_x': LSF_x, 'LSF': LSF_no_filt, 'ESF': ESF_all,
                'sigma_prefilter': sigma_f*step_size,
                'dMTF_details': dMTF_details, 'gMTF_details': gMTF_details,
                'edge_details': details_dicts_edge}
        else:
            errmsg = 'Could not find edge.'

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
            center_this = mmcalc.center_xy_of_disc(sli, roi=roi)
            if None not in center_this:
                center_xy.append(center_this)
            else:
                errtxt = ', '.join(errtxt, str(slino))
    if errtxt != '':
        errmsg = f'Could not find center of object for slice {errtxt}'
    if len(center_xy) > 0:
        center_x = [vals[0] for vals in center_xy]
        center_y = [vals[1] for vals in center_xy]
        center_xy = [np.mean(np.array(center_x)), np.mean(np.array(center_y))]

        # sort pixel values from center
        dist_map = get_distance_map_point(
            matrix[0].shape,
            center_dx=center_xy[0] - matrix[0].shape[1]/2,
            center_dy=center_xy[1] - matrix[0].shape[0]/2)
        dists_flat = dist_map.flatten()
        sort_idxs = np.argsort(dists_flat)
        dists = dists_flat[sort_idxs]
        dists = pix * dists
        values_all = []
        values_sum = np.zeros(dists.shape)
        nsum = 0
        for sli in matrix:
            if sli is not None:
                values_flat = sli.flatten()
                values_all.append(values_flat[sort_idxs])
                values_sum = np.add(values_sum, values_flat[sort_idxs])
                nsum += 1
        values_avg = 1/nsum * values_sum

        # ignore dists > radius
        radius = 0.5 * matrix[0].shape[0] * pix
        dists_cut = dists[dists < radius]
        values = values_avg[dists < radius]

        # reduce noise using small window width ~0.1*pix assumed to not affect edge
        diff_x = np.diff(dists_cut)
        sigma_n = 0.1 * pix // np.median(diff_x)  # TODO *.5?
        if sigma_n > 0:
            values = sp.ndimage.gaussian_filter(values, sigma=sigma_n)
        # rebin to 1/10th pix size
        step_size = .1 * pix
        new_x, new_y = mmcalc.resample(
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

            gMTF_details = mmcalc.get_MTF_gauss(
                LSF, dx=step_size, prefilter_sigma=sigma_f*step_size, gaussfit='double')

            gMTF_details['values'] = mmcalc.get_curve_values(
                    gMTF_details['MTF_freq'], gMTF_details['MTF'], [0.5, 0.1, 0.02])
            dMTF_details['values'] = mmcalc.get_curve_values(
                    dMTF_details['MTF_freq'], dMTF_details['MTF'], [0.5, 0.1, 0.02],
                    force_first_below=True)

            details_dict = {
                'center_xy': center_xy,
                'LSF_x': LSF_x, 'LSF': LSF_no_filt,
                'sigma_prefilter': sigma_f*step_size,
                'sorted_pixels_x': dists, 'sorted_pixels': values_all,
                'interpolated_x': new_x, 'interpolated': new_y,
                'presmoothed': ESF_filtered,
                'dMTF_details': dMTF_details, 'gMTF_details': gMTF_details}
        else:
            errmsg = 'Could not find circular edge.'

    return (details_dict, errmsg)


def get_corrections_point_source(
        image2d, img_info, roi_array,
        fit_x=False, fit_y=False, lock_z=-1.):
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
        IF not -1., do not fit z, use locked value. The default is -1

    Returns
    -------
    dict
        corrected_image : numpy.ndarray
        correction_matrix : numpy.ndarray
            subtract this matrix to obtain corrected image2d
        dx : corrected x position (mm)
        dy : corrected y position (mm)
        distance : corrected distance (mm)
    """
    corrected_image = image2d
    correction_matrix = np.ones(image2d.shape)
    distance = 0.

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
            C, A, x0, sigma = mmcalc.gauss_4param_fit(
                np.arange(len(sum_vector)), sum_vector)
            offset[a] = x0 - 0.5*len(sum_vector)

    dx, dy = offset
    if lock_z == -1.:
        # calculate distances from center in plane
        shape_ufov_y, shape_ufov_x = ufov_denoised.shape
        '''
        sz_y, sz_x = image2d.shape
        xs, ys = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
        center_pos_x = dx + round(0.5*sz_x)
        center_pos_y = dy + round(0.5*sz_y)
        dists_inplane = np.sqrt((xs-center_pos_x) ** 2 + (ys-center_pos_y) ** 2)
        '''
        #TODO - precision offset from fit vs center in distance_map (up to 1 pixel wrong?)
        dists_inplane = get_distance_map_point(
            image2d.shape, center_dx=dx, center_dy=dy)
        dists_inplane = img_info.pix[0] * dists_inplane

        dists_ufov = dists_inplane[rows][:, cols]
        dists_ufov_flat = dists_ufov.flatten()
        values_flat = ufov_denoised.flatten()
        sort_idxs = np.argsort(dists_ufov_flat)
        values = values_flat[sort_idxs]
        dists = dists_ufov_flat[sort_idxs]

        # fit values
        nm_radius = img_info.nm_radius if img_info.nm_radius != -1. else 400.
        C, distance = mmcalc.point_source_func_fit(
            dists, values,
            center_value=np.max(ufov_denoised),
            avg_radius=nm_radius)
        fit = mmcalc.point_source_func(dists_inplane.flatten(), C, distance)
        corr_sub = fit - fit[0]
        corr_sub = (1./np.max(fit)) * fit  # matrix to subtract
        corr_sub = 1./corr_sub
        correction_matrix = corr_sub.reshape(image2d.shape)
        corrected_image = image2d * correction_matrix
    else:
        distance = lock_z

    return {'corrected_image': corrected_image,
            'correction_matrix': correction_matrix,
            'dx': dx, 'dy': dy, 'distance': distance}


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
                du_cols[y, x] = 100. * (max_val - min_val) / (max_val + min_val)
        du_rows = np.zeros(image.shape)
        for y in range(sz_y):
            for x in range(2, sz_x - 2):
                sub = image[y, x-2:x+2]
                max_val = np.max(sub)
                min_val = np.min(sub)
                du_rows[y, x] = 100. * (max_val - min_val) / (max_val + min_val)
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


def calculate_NM_SNI(image2d, roi_array, pix):
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

    def get_eye_filter(roi_size, pix, f, c, d):
        """Get eye filter V(r)=r^1.3*exp(-cr^2).

        Parameters
        ----------
        roi_size : int
            size of roi or image in pix
        pix : float
            pixelsize
        f : float
            TODO - describe
        c : float
            adjusted to have V(r) max @ some cy/degree
                (Nelson et al uses c=28 = viewing distance 1.5m)
        d : float
            displaysize (mm on screen) -- Nelson et al uses 65 mm

        Returns
        -------
        eye_filter : dict
            filter 2d : np.ndarray quadratic with size roi_height
            curve: dict of r and V
        """
        def eye_filter_func(r, f, c):
            return r**f * np.exp(-c*r**2)

        roi_mm = roi_size * pix
        r = (1/d) * np.arange(roi_mm/2)  #TODO correct? pix mm .... d r ?
        eye_filter_1d = {'r': r, 'V': eye_filter_func(r, f, c)}

        dists = get_distance_map_point((roi_size, roi_size))
        eye_filter_2d = eye_filter_func(dists, f, c)
        breakpoint()

        return {'filter_2d': eye_filter_2d, 'curve': eye_filter_1d}

    SNI_values = []

    subarray_large = []
    rows = np.max(roi_array[1], axis=1)
    cols = np.max(roi_array[1], axis=0)
    subarray_large.append(image2d[rows][:, cols])
    rows = np.max(roi_array[2], axis=1)
    cols = np.max(roi_array[2], axis=0)
    subarray_large.append(image2d[rows][:, cols])

    return SNI_values

    
'''
function get_rNPS, NPS_2d, ROIsz, pix, dists, sorting, smoothWidth, sampFreq
  
  NPSvals=NPS_2d(sorting)
  unity=1./(ROIsz*pix(0))
  width=smoothWidth/unity

  ;avg over unique dists
;  uu=uniq(dists)
;  nvals=N_ELEMENTS(uu)
;  NPSvalsU=FLTARR(nvals)
;  NPSvalsU(0)=MEAN(NPSvals[0:uu(0)])
;  FOR i=1, nvals-1 DO NPSvalsU(i)=MEAN(NPSvals[uu(i-1):uu(i)])
;  newdists=dists[UNIQ(dists, SORT(dists))];keep only uniq dists
  
  ;need to speed up code and avg over close to unique dists (1/10 of the width regarded as small enough) 
  tenthWidths=ROUND((10./width)*dists)
  u=uniq(tenthWidths)
  nvals=N_ELEMENTS(u)
  valsU=FLTARR(nvals)
  valsU(0)=MEAN(NPSvals[0:u(0)])
  FOR i=1, nvals-1 DO valsU(i)=MEAN(NPSvals[u(i-1):u(i)])
  nd=tenthWidths[UNIQ(tenthWidths, SORT(tenthWidths))];keep only uniq dists
  newdists=width/10.*nd

  ;smooth irregularly sampled data within given width
  NPSvalsU=smoothIrreg(newdists,valsU, width);smoothIrreg in a0_functionsMini.pro

  ;regular sampling
  sampRelUnity=sampFreq/unity
  newdistsReg=FINDGEN(ROUND(max(newdists)/sampRelUnity))*sampRelUnity
  NPSvalsInterp=INTERPOL(NPSvalsU, newdists, newdistsReg); linear interpolation
  nn=N_ELEMENTS(NPSValsInterp)

  dr=(FINDGEN(nn))*(sampRelUnity/(ROIsz*pix(0)))
  
  return, CREATE_STRUCT('dr',dr,'rNPS',NPSvalsInterp)
end


;Calculate Structured Noise Index (SNI) based on Nelson et al, J Nucl Med 2014;55:169-174
function calculateSNI, noiseImg, corrMat, SNIroi, pix, fcd, smoothWidth, sampFreq

  quantNoiseL=FLTARR(2)
;  IF N_ELEMENTS(corrMat) NE 0 THEN BEGIN
  quantNoiseL(0)=MEAN(noiseImg[firstX:firstX+largeDim-1,firstY:firstY+largeDim-1])
  quantNoiseL(1)=MEAN(noiseImg[lastX-largeDim+1:lastX,firstY:firstY+largeDim-1])
;  ENDIF

  ;small ROIs (6)
  subS=FLTARR(smallDim, smallDim, 6)
  mid=(lastX+firstX)/2
  firstM=mid-(smallDim/2-1)
  subS[*,*,0]=noiseImgCorr[firstX:firstX+smallDim-1,lastY-smallDim+1:lastY];upper lft
  subS[*,*,1]=noiseImgCorr[firstM:firstM+smallDim-1,lastY-smallDim+1:lastY];upper mid
  subS[*,*,2]=noiseImgCorr[lastX-smallDim+1:lastX,lastY-smallDim+1:lastY];upper rgt
  subS[*,*,3]=noiseImgCorr[firstX:firstX+smallDim-1,firstY:firstY+smallDim-1];lower lft
  subS[*,*,4]=noiseImgCorr[firstM:firstM+smallDim-1,firstY:firstY+smallDim-1];lower mid
  subS[*,*,5]=noiseImgCorr[lastX-smallDim+1:lastX,firstY:firstY+smallDim-1];lower rgt

  quantNoiseS=FLTARR(6)
  quantNoiseS(0)=MEAN(noiseImg[firstX:firstX+smallDim-1,lastY-smallDim+1:lastY])
  quantNoiseS(1)=MEAN(noiseImg[firstM:firstM+smallDim-1,lastY-smallDim+1:lastY])
  quantNoiseS(2)=MEAN(noiseImg[lastX-smallDim+1:lastX,lastY-smallDim+1:lastY])
  quantNoiseS(3)=MEAN(noiseImg[firstX:firstX+smallDim-1,firstY:firstY+smallDim-1])
  quantNoiseS(4)=MEAN(noiseImg[firstM:firstM+smallDim-1,firstY:firstY+smallDim-1])
  quantNoiseS(5)=MEAN(noiseImg[lastX-smallDim+1:lastX,firstY:firstY+smallDim-1])

  ;****2d fourier of each ROI***
  NPS_L=FLTARR(largeDim,largeDim,2)
  NPS_S=FLTARR(smallDim, smallDim, 6)
  NPS_L_filt=NPS_L
  NPS_S_filt=NPS_S
  rNPS_L=!Null;radial NPS large ROIs
  rNPS_S=!Null;radial NPS small ROIs
  ;filter with human visual response filter (also removing peak on very low ferquency - trendremoval not needed)
  large=generateHumVisFilter(largeDim,pix, fcd)
  small=generateHumVisFilter(smallDim,pix, fcd)
  humVisFilterLarge=large.filt2d
  humVisFilterSmall=small.filt2d
  humVisFiltCurve=CREATE_STRUCT('L',large.curve,'S',small.curve)
  
  SNIvalues=FLTARR(9);max,L1,L2,S1..6
  
  FOR i = 0, 1 DO BEGIN
    subM=subL[*,*,i]
    temp=FFT(subM-MEAN(subM),/CENTER)
    NPS_L[*,*,i]=largeDim^2*pix^2*(REAL_PART(temp)^2+IMAGINARY(temp)^2)
    ;remove quantum noise to be left with structural noise
    NPS_Lstruc=NPS_L[*,*,i]-quantNoiseL(i)*pix^2;Meancount=variance=pixNPS^2*Total(NPS) where pixNPS=1./(ROIsz*pix), Total(NPS)=NPSvalue*ROIsz^2, NPSvalue=Meancount/(pixNPS^2*ROIsz^2)=MeanCount*pix^2
    ;filter with human visual response filter
    NPS_L_filt[*,*,i]=NPS_L[*,*,i]*humVisFilterLarge
    NPS_Lstruc_filt=NPS_Lstruc*humVisFilterLarge
    SNIvalues(i+1)=TOTAL(NPS_Lstruc_filt)/TOTAL(NPS_L_filt[*,*,i])
    
  ENDFOR

  FOR i = 0, 5 DO BEGIN
    subM=subS[*,*,i]
    temp=FFT(subM-MEAN(subM),/CENTER)
    NPS_S[*,*,i]=smallDim^2*pix^2*(REAL_PART(temp)^2+IMAGINARY(temp)^2)
    ;remove quantum noise
    NPS_Sstruc=NPS_S[*,*,i]-quantNoiseS(i)*pix^2;Meancount=variance=pixNPS^2*Total(NPS) where pixNPS=1./(ROIsz*pix), Total(NPS)=NPSvalue*ROIsz^2, NPSvalue=Meancount/(pixNPS^2*ROIsz^2)=MeanCount*pix^2
    ;filter with human visual response filter
    NPS_S_filt[*,*,i]=NPS_S[*,*,i]*humVisFilterSmall
    NPS_Sstruc_filt=NPS_Sstruc*humVisFilterSmall
    SNIvalues(i+3)=TOTAL(NPS_Sstruc_filt)/TOTAL(NPS_S_filt[*,*,i])
  ENDFOR

  SNIvalues(0)=MAX(SNIvalues)

  NPS_filt=CREATE_STRUCT('L1',NPS_L_filt[*,*,0],'L2',NPS_L_filt[*,*,1],'S1',NPS_S_filt[*,*,0],'S2',NPS_S_filt[*,*,1],'S3',NPS_S_filt[*,*,2],'S4',NPS_S_filt[*,*,3],'S5',NPS_S_filt[*,*,4],'S6',NPS_S_filt[*,*,5])
  
  ds=get_dists_sorting(largeDim)
  rNPS_L1=get_rNPS(NPS_L[*,*,0], largeDim, pix, ds.dists, ds.sorting, smoothWidth, sampFreq)
  rNPS_L2=get_rNPS(NPS_L[*,*,1], largeDim, pix, ds.dists, ds.sorting, smoothWidth, sampFreq)
  ds=get_dists_sorting(smallDim)
  rNPS_S1=get_rNPS(NPS_S[*,*,0], smallDim, pix, ds.dists, ds.sorting, smoothWidth, sampFreq)
  rNPS_S2=get_rNPS(NPS_S[*,*,1], smallDim, pix, ds.dists, ds.sorting, smoothWidth, sampFreq)
  rNPS_S3=get_rNPS(NPS_S[*,*,2], smallDim, pix, ds.dists, ds.sorting, smoothWidth, sampFreq)
  rNPS_S4=get_rNPS(NPS_S[*,*,3], smallDim, pix, ds.dists, ds.sorting, smoothWidth, sampFreq)
  rNPS_S5=get_rNPS(NPS_S[*,*,4], smallDim, pix, ds.dists, ds.sorting, smoothWidth, sampFreq)
  rNPS_S6=get_rNPS(NPS_S[*,*,5], smallDim, pix, ds.dists, ds.sorting, smoothWidth, sampFreq)
  
  rNPS = CREATE_STRUCT('L1',rNPS_L1,'L2',rNPS_L2,'S1',rNPS_S1,'S2',rNPS_S2,'S3',rNPS_S3,'S4',rNPS_S4,'S5',rNPS_S5,'S6',rNPS_S6)
  
  SNIstruc=CREATE_STRUCT('SNIvalues',SNIvalues, 'NPS_filt', NPS_filt, 'rNPS', rNPS, 'estQuantNoiseL1',quantNoiseL(0)*pix^2,'humVisFiltCurve',humVisFiltCurve)
  IF N_ELEMENTS(corrMat) NE 0 THEN SNIstruc=CREATE_STRUCT(SNIstruc,'corrMatrix',noiseImgCorr)
  return, SNIstruc
end
'''