#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculation processes for the different tests.

@author: Ellen Wasbø
"""
import numpy as np
import scipy as sp
import skimage

import imageQC.scripts.dcm as dcm
from imageQC.scripts.calculate_roi import get_rois
from imageQC.scripts.mini_methods import get_min_max_pos_2d
from imageQC.scripts.mini_methods_format import val_2_str
from imageQC.scripts.mini_methods_calculate import (
    get_width_center_at_threshold,
    gauss_4param, gauss_4param_fit,
    gauss_fit, gauss, gauss_double_fit, gauss_double,
    point_source_func_fit, point_source_func
    )
from imageQC.config.iQCconstants import HEADERS
import imageQC.config.config_classes as cfc


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
        # headers
        if input_main.current_paramset.output.include_header:
            set_names = input_main.current_quicktest.image_names
            if any(set_names):
                for i in range(n_imgs):
                    if set_names[i] != '':
                        image_names[i] = set_names[i]

        #TODO if output_temp.transpose_table:
        #TODO if output_temp.include_filename:

        include_header = True
        if input_main.automation_active is False:
            include_header = input_main.current_paramset.output.include_header

        for test in input_main.results:
            if test in input_main.current_paramset.output.tests:
                output_subs = input_main.current_paramset.output.tests[test]
            else:
                output_subs = [cfc.QuickTestOutputSub(columns=[])]

            dm = input_main.current_paramset.output.decimal_mark
            for sub in output_subs:
                values = None
                headers = None
                if sub.alternative == -1:  # supplement table to output
                    values = input_main.results[test]['values_sup']
                    headers = input_main.results[test]['headers_sup']
                elif sub.alternative == input_main.results[test]['alternative']:
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

    return (string_list, header_list)


def calculate_qc(input_main):
    """Calculate tests according to current info in main.

    Parameters
    ----------
    input_main : object
        of class MainWindow or MainAuto containing specific attributes
        see try block below
        alter results of input_main
    """
    current_test_before = input_main.current_test
    delta_xya = [0, 0, 0.0]  # only for GUI version
    modality = input_main.current_modality
    pre_msg = ''
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
            matrix = [[] for i in range(n_img)]
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
                    if test != 'DCM':
                        if test in marked_3d[i] and matrix[i] == []:
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
                                prev_roi[test] = get_rois(
                                    image, i, input_main)

                            result = calculate_2d(
                                image, prev_roi[test], img_infos[i],
                                modality, paramset, test, delta_xya)
                            if test not in [*input_main.results]:
                                # initiate results
                                input_main.results[test] = {
                                    'headers': [],
                                    'values': [[] for i in range(n_img)],
                                    'alternative': 0,
                                    'headers_sup': [],
                                    'values_sup': [[] for i in range(n_img)],
                                    'details_dict': [[] for i in range(n_img)],
                                    'pr_image': True
                                    }
                            input_main.results[
                                test]['values'][i] = result['values']
                            input_main.results[
                                test]['values_sup'][i] = result['values_sup']
                            input_main.results[
                                test]['details_dict'][i] = result['details_dict']
                            if len(input_main.results[test]['headers']) == 0:
                                input_main.results[
                                    test]['headers'] = result['headers']
                            if len(input_main.results[test]['headers_sup']) == 0:
                                input_main.results[
                                    test]['headers_sup'] = result['headers_sup']

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

            if any(marked_3d):
                results = calculate_3d(
                    matrix, marked_3d, input_main)
                for test in results:
                    input_main.results[test] = results[test]
                    input_main.results[test]['pr_image'] = False

            # For test DCM
            if any(read_tags):
                input_main.results['DCM'] = {
                    'headers': paramset.dcm_tagpattern.list_tags,
                    'values': tag_lists,
                    'alternative': 0,
                    'pr_image': True
                    }

    if 'MainWindow' in str(type(input_main)):
        if input_main.automation_active is False:
            input_main.current_test = current_test_before
            input_main.refresh_results_display()
            input_main.statusBar.showMessage('Finished', 1000)
            input_main.stop_wait_cursor()


def calculate_2d(image2d, roi_array, image_dict, modality,
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
    image_dict : dict
        image info as dict - as defined in ui_main.py
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
    result : dict
        'headers': list[str] of headers of result
        'values': list of values
            [column1, column2, ...] where column1 is list of column values
    """
    result = None
    headers = []
    headers_sup = []
    values = []
    values_sup = []
    details_dict = {}
    alt = 0
    if test_code == 'ROI':
        if image2d is not None:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            avg_val = np.mean(arr)
            std_val = np.std(arr)
            values = [avg_val, std_val]
        headers = HEADERS[modality][test_code]['alt0']

    elif test_code == 'Noi':
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

    elif test_code == 'Hom':
        avgs = []
        stds = []
        if image2d is not None:
            for i in range(np.shape(roi_array)[0]):
                arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[i]))
                avgs.append(np.mean(arr))
                stds.append(np.std(arr))

        if modality == 'CT':
            headers = HEADERS[modality][test_code]['alt0']
            headers_sup = HEADERS[modality][test_code]['altSup']
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

    elif test_code == 'CTn':
        headers = paramset.ctn_table.materials
        headers_sup = HEADERS[modality][test_code]['altSup']
        if image2d is not None:
            values = []
            if image2d is not None:
                for i in range(len(paramset.ctn_table.materials)):
                    arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[i]))
                    values.append(np.mean(arr))
                res = sp.stats.linregress(
                    values, paramset.ctn_table.relative_mass_density)
                values_sup = [res.rvalue**2, res.intercept, res.slope]

    elif test_code == 'HUw':
        headers = HEADERS[modality][test_code]['alt0']
        if image2d is not None:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            avg_val = np.mean(arr)
            std_val = np.std(arr)
            values = [avg_val, std_val]

    elif test_code == 'Sli':
        if modality == 'CT':
            alt = paramset.sli_type
            headers = HEADERS[modality][test_code]['alt' + str(alt)]
            if image2d is not None:
                lines = roi_array
                values, details_dict, err_logg = calculate_slicethickness_CT(
                    image2d, image_dict, paramset, lines, delta_xya)
                if alt == 0:
                    values.append(np.mean(values[1:]))
                    values.append(100. * (values[-1] - values[0]) / values[0])
        elif modality == 'MR':
            headers = HEADERS[modality][test_code]['alt0']
            if image2d is not None:
                pass
                #TODO
        else:
            pass

    elif test_code == 'MTF':
        alt = paramset.mtf_type
        headers = HEADERS[modality][test_code]['alt' + str(alt)]
        if image2d is not None:
            if modality == 'CT':
                # only bead method 2d
                rows = np.max(roi_array[0], axis=1)
                cols = np.max(roi_array[0], axis=0)
                sub = image2d[rows][:, cols]
                arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[1]))
                background = np.mean(arr)
                values = calculate_MTF_point(sub, background, image_dict)

    elif test_code == 'Uni':
        headers = HEADERS[modality][test_code]['alt0']
        headers_sup = HEADERS[modality][test_code]['altSup']
        if image2d is not None:
            details_dict = {}
            values_sup = [None] * 3
            if paramset.uni_correct:
                res = get_corrections_point_source(
                    image2d, image_dict, roi_array[0],
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
                image_input, roi_array, image_dict.pix[0])
            details_dict['matrix'] = res['matrix']
            details_dict['du_matrix'] = res['du_matrix']
            values = res['values']

    elif test_code == 'SNI':
        headers = HEADERS[modality][test_code]['alt0']
        headers_sup = HEADERS[modality][test_code]['altSup']
        if image2d is not None:
            details_dict = {}
            values_sup = [None] * 3
            if paramset.sni_correct:
                res = get_corrections_point_source(
                    image2d, image_dict, roi_array[0],
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
                image_input, roi_array, image_dict.pix[0])
            #details_dict['matrix'] = res['matrix']
            #details_dict['du_matrix'] = res['du_matrix']
            values = res['values']

    elif test_code == 'PIU':
        headers = HEADERS[modality][test_code]['alt0']
        # ['min', 'max', 'PIU'],
        headers_sup = HEADERS[modality][test_code]['altSup']
        # ['x min (pix from upper left)', 'y min', 'x max', 'y max']
        if image2d is not None:
            arr = np.ma.masked_array(image2d, mask=np.invert(roi_array))
            min_val = np.min(arr)
            max_val = np.max(arr)
            piu = 100.*(1-(max_val-min_val)/(max_val+min_val))
            values = [min_val, max_val, piu]

            min_idx, max_idx = get_min_max_pos_2d(image2d, roi_array)
            values_sup = [min_idx[1], min_idx[0],
                          max_idx[1], max_idx[0]]

    elif test_code == 'Gho':
        headers = HEADERS[modality][test_code]['alt0']
        # ['Center', 'top', 'bottom', 'left', 'right', 'PSG']
        if image2d is not None:
            avgs = []
            for i in range(np.shape(roi_array)[0]):
                arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[i]))
                avgs.append(np.mean(arr))
            PSG = abs(100.*0.5*(
                (avgs[3]+avgs[4]) - (avgs[1]+avgs[2])
                )/avgs[0])
            values = avgs
            values.append(PSG)

    result = {'values': values, 'headers': headers, 'alternative': alt,
              'values_sup': values_sup, 'headers_sup': headers_sup,
              'details_dict': details_dict}

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
    results : dict
        key = test code
        value = as result of calculate_2d
    """
    img_infos = input_main.imgs
    modality = input_main.current_modality
    paramset = input_main.current_paramset

    results = {}
    flat_marked = [item for sublist in marked_3d for item in sublist]
    all_tests = list(set(flat_marked))

    for test_code in all_tests:
        images_to_test = []
        for i, tests in enumerate(marked_3d):
            if test_code in tests:
                images_to_test.append(i)

        values = []
        headers = []
        values_sup = []
        headers_sup = []
        alt = 0
        details_dict = [{}]
        if test_code == 'MTF':
            headers = HEADERS[modality][test_code]['alt2']
            if len(images_to_test) > 0:
                if paramset.mtf_type == 1:  #wire
                    pass
                elif paramset.mtf_type == 2:  # circular disk
                    pass

        elif test_code == 'SNR':
            # use two and two images
            values_first_imgs = []
            idxs_first_imgs = []
            headers = HEADERS[modality][test_code]['alt0']
            if (matrix.count([]) + matrix.count(None)) != len(matrix):
                if len(images_to_test) > 1:
                    n_pairs = len(images_to_test)/2
                    for i in range(n_pairs):
                        idx1 = images_to_test[i*2]
                        idx2 = images_to_test[i*2+1]
                        image1 = matrix[idx1]
                        image2 = matrix[idx2]
                        image_subtract = None
                        if img_infos[idx1].shape == img_infos[idx2].shape:
                            if img_infos[idx1].pix == img_infos[idx2].pix:
                                roi_array = get_rois(
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

        elif test_code in ['Uni', 'SNI']:
            # assume sum_first  - only then 3d
            sum_matrix = matrix[images_to_test[0]]
            for i in images_to_test[1:]:
                sum_matrix = np.add(sum_matrix, matrix[i])
            matrix = sum_matrix

            headers = HEADERS[modality][test_code]['alt0']
            headers_sup = HEADERS[modality][test_code]['altSup']
            if paramset.uni_correct and test_code == 'Uni':
                res = get_corrections_point_source(
                    matrix, img_infos[0], roi_array[0],
                    fit_x=paramset.uni_correct_pos_x,
                    fit_y=paramset.uni_correct_pos_y,
                    lock_z=paramset.uni_correct_radius
                    )
                image_input = res['corrected_image']
                values_sup = [res['dx'], res['dy'], res['distance']]
                details_dict = res
            elif paramset.sni_correct and test_code == 'SNI':
                res = get_corrections_point_source(
                    matrix, img_infos[0], roi_array[0],
                    fit_x=paramset.sni_correct_pos_x,
                    fit_y=paramset.sni_correct_pos_y,
                    lock_z=paramset.sni_correct_radius
                    )
                image_input = res['corrected_image']
                values_sup = [[res['dx'], res['dy'], res['distance']]]
                details_dict = [res]
            else:
                image_input = matrix

            if test_code == 'Uni':
                roi_array = get_rois(
                    sum_matrix, images_to_test[0], input_main)
                res = calculate_NM_uniformity(
                    image_input, roi_array, img_infos[0].pix[0])
                details_dict[0]['matrix'] = res['matrix']
                details_dict[0]['du_matrix'] = res['du_matrix']
                values = [res['values']]
            elif test_code == 'SNI':
                pass

        if any(values):
            results[test_code] = {
                'values': values, 'headers': headers, 'alternative': alt,
                'values_sup': values_sup, 'headers_sup': headers_sup,
                'details_dict': details_dict
                }

    return results


def get_distance_map(shape, center_dx=0., center_dy=0.):
    """Calculate distances from center in image (optionally with offset).

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
    """
    details_dict = {'profiles': [],
                    'background': [], 'peak': [], 'halfpeak': [],
                    'start_x': [], 'end_x': []}
    err_logg = []

    if paramset.sli_type == 0:
        details_dict['labels'] = ['lower H1', 'upper H2', 'right V1', 'left V2']
    elif paramset.sli_type == 1:
        details_dict['labels'] = [
            'lower H1', 'upper H2', 'right V1', 'left V2', 'inner V1', 'inner V2']
    elif paramset.sli_type == 2:
        details_dict['labels'] = ['right', 'left']

    for line in lines['h_lines']:
        profile, err_msg = get_profile_sli(image, paramset, line)
        details_dict['profiles'].append(profile)
        err_logg.append(err_msg)
    for line in lines['v_lines']:
        profile, err_msg = get_profile_sli(image, paramset, line, direction='v')
        details_dict['profiles'].append(profile)
        err_logg.append(err_msg)
    details_dict['dx'] = img_info.pix[0] * np.cos(np.deg2rad(delta_xya[2]))

    values = [img_info.slice_thickness]
    n_background = round(paramset.sli_background_width / img_info.pix[0])
    for profile in details_dict['profiles']:
        slice_thickness = -1.
        background_start = np.mean(profile[0:n_background])
        background_end = np.mean(profile[-n_background:])
        background = 0.5 * (background_start + background_end)
        details_dict['background'].append(background)
        profile_orig = profile
        background_orig = background
        profile = profile - background
        if paramset.sli_signal_low_density:
            peak_value = np.min(profile)
            profile = -1. * profile
            background = -1. * background
        else:
            peak_value = np.max(profile)
        details_dict['peak'].append(peak_value + background)
        halfmax = 0.5 * peak_value
        '''TODO delete?
        if paramset.sli_signal_low_density:
            halfmax_orig = 0.5 * (np.min(profile_orig) + background_orig)
        else:
            halfmax_orig = halfmax
        '''
        details_dict['halfpeak'].append(halfmax + background)

        if paramset.sli_type == 0:  # wire ramp Catphan
            width, center = get_width_center_at_threshold(profile, halfmax)
            slice_thickness = 0.42 * width * img_info.pix[0]  # 0.42*FWHM
            if delta_xya[2] != 0:
                slice_thickness = slice_thickness / np.cos(
                    np.deg2rad(delta_xya[2]))
            details_dict['start_x'].append((center - 0.5 * width) * img_info.pix[0])
            details_dict['end_x'].append((center + 0.5 * width) * img_info.pix[0])
        else:  # beaded ramp, find envelope curve
            # find upper envelope curve
            derived = profile - np.roll(profile, 1)
            subz = np.where(derived < 0) - 1
            dsubz = subz - np.roll(subz, 1)
            ss = [] * len(dsubz)
            ss[1] = 1
            for s in range(2, len(dsubz)):
                if dsubz[s] > 1:
                    ss[s] = 1
            idxes = np.where(ss == 1)
            idxmax = subz[idxes]
            #TODO find envelope pythonic
            '''
            envelope = INTERPOL(profile[idxmax] ,idxmax, INDGEN(N_ELEMENTS(vec)));interpolate to regular stepsize
            res=getWidthAtThreshold(vecInt, halfmax)
            zinc=1.
            IF ramptype EQ 1 AND l GE 4 THEN zinc=.25;0.25mm z spacing between beads
            resArr[l+1,i]=zinc/2.0*(res(0)*pix(0)/cos(daRad)) ; 2mm axial spacing
            ENDELSE
            structTemp=CREATE_STRUCT('background',backGrOrig,'nBackGr',nPixBackG,'vector',vecOrig,'halfMax',halfMaxOrig,'firstLast',[res(1)-res(0)/2.,res(1)+res(0)/2.],'peakVal',peakVal)
            IF l EQ sta THEN lineStruct=CREATE_STRUCT('L'+STRING(l, FORMAT='(i0)'),structTemp) ELSE lineStruct=CREATE_STRUCT(lineStruct,'L'+STRING(l, FORMAT='(i0)'),structTemp)
            '''

        values.append(slice_thickness)

    return (values, details_dict, err_logg)


def get_profile_sli(image, paramset, line, direction='h'):

    profile = []
    err_logg = []
    n_search = round(paramset.sli_search_width)
    n_avg = paramset.sli_average_width

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
        if paramset.sli_signal_low_density:
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
                err_logg = 'Not average.'  # TODO better msg
    else:
        rr, cc = skimage.draw.line(r0, c0, r1, c1)
        profile = image[rr, cc]

    return (profile, err_logg)


def calculate_MTF_point(sub_image, background_value, img_info, gaussfit='single',
                        cut_lsf=False, cut_lsf_fwhm=0., cut_lsf_fade=0.):
    """Calculate MTF from point source.

    Parameters
    ----------
    sub_image : numpy.ndarray
        part of image limited to the point source
    background_value : float
    img_info : DcmInfo
        as defined in scripts/dcm.py
    gaussfit : str ('single' or 'double')
        fit to single or double (sum of two) gaussian
    cut_lsf : bool, optional
        cut lsf tails. Default is False.
    cut_lsf_fwhm : float, optional
        cut lsf tails #fwhm outside halfmax. If zero ignored. Default is 0.
    cut_lsf_fade : float, optional
        fade out within #fwhm outside the lsf tails cut.
        If zero or cut is zero ignored. Default is 0.

    Returns
    -------
    values : list of float
        [MTFx 50%, 10%, 2%, MTFy 50%, 10%, 2%]
    values_sup : list of float
        [A1 mu1, sigma1, A2, mu2, sigma2]
    details_dict: dict
        centerpos_xy: list of float
    """
    values = []
    values_sup = []
    matrix = sub_image - background_value
    LSF_xy = []
    LSF_xy.append(np.sum(matrix, axis=1))
    LSF_xy.append(np.sum(matrix, axis=0))
    dd_xy = []
    centerpos_xy = []
    for profile in LSF_xy:
        width, center = get_width_center_at_threshold(profile, np.max(profile)/2)
        pos = (np.arange(len(profile)) - center) * img_info.pix[0]
        dd_xy.append(pos)
        centerpos_xy.append(center)
        if gaussfit == 'double':
            popt = gauss_double_fit(pos, profile, cut_width_fwhm=0, fwhm1=width)
        else:
            popt = gauss_fit(pos, profile, cut_width_fwhm=0, fwhm1=width)
    details_dict = {'centerpos_xy': centerpos_xy, 'dd_xy': dd_xy}

    # Gaussian MTF
    '''
    res2=getWidthAtThreshold(Y,min(Y)/2)
    IF res2(0) NE -1 THEN BEGIN
      FWHM2=res2(0)*pix
      sigma2=FWHM2/(2*SQRT(2*ALOG(2)))
      IF sigma2 LT sigma1 THEN sigma2=2.*sigma1
    ENDIF ELSE sigma2=2.*sigma1

    IF fitWidthFactor EQ 0 THEN ss1=0 ELSE ss1=ROUND(center/pix)-ROUND(FWHM1/pix)*fitWidthFactor
    IF fitWidthFactor EQ 0 THEN ss2=nn-1 ELSE ss2=ROUND(center/pix)+ROUND(FWHM1/pix)*fitWidthFactor
    IF ss1 LT 0 THEN ss1=0
    IF ss2 GT nn-1 THEN ss2=nn-1
    A = [max(Y[ss1:ss2])-min(Y[ss1:ss2]),1.5*min(Y[ss1:ss2]),sigma1, sigma2];first guess parameters for curvefit gaussFitAdd2

      resX=getGaussFit(ddx,smLSFx,pix(0),fitWidthFactor)
      resY=getGaussFit(ddy,smLSFy,pix(1),fitWidthFactor)
      fitLSFx=resX.yfit & fitLSFy=resY.yfit

      IF N_ELEMENTS(fitLSFx) NE 1 THEN gMTFx=getMTFgauss(resX.A, sigmaF*pix(0))
      IF N_ELEMENTS(fitLSFy) NE 1 THEN gMTFy=getMTFgauss(resY.A, sigmaF*pix(1))

      szPadded=factorPad*szM(0)

      IF cutLSF THEN BEGIN
        nn=N_ELEMENTS(LSFx)
        smdLSF=SMOOTH(LSFx,3)
        smdLSF=smdLSF/max(smdLSF)
        over05=WHERE(smdLSF GT 0.5, nover05)
        pp1=over05(0) & pp2=over05(nover05-1)
        ppFWHM=pp2-pp1
        first=ROUND(pp1-cutW*ppFWHM)
        last=ROUND(pp2+cutW*ppFWHM)
        IF first GT 0 THEN LSFx[0:first]=0
        IF last LT nn-1 THEN LSFx[last:nn-1]=0

        nn=N_ELEMENTS(LSFy)
        smdLSF=SMOOTH(LSFy,3)
        smdLSF=smdLSF/max(smdLSF)
        over05=WHERE(smdLSF GT 0.5, nover05)
        pp1=over05(0) & pp2=over05(nover05-1)
        ppFWHM=pp2-pp1
        first=ROUND(pp1-cutW*ppFWHM)
        last=ROUND(pp2+cutW*ppFWHM)
        IF first GT 0 THEN LSFy[0:first]=0
        IF last LT nn-1 THEN LSFy[last:nn-1]=0
      ENDIF

      MTFx=FFTvector(LSFx, factorPad)
      Nx=N_ELEMENTS(MTFx)
      fx=FINDGEN(Nx)*(1./(szPadded*pix(0)))

      MTFy=FFTvector(LSFy, factorPad)
      Ny=N_ELEMENTS(MTFy)
      fy=FINDGEN(Ny)*(1./(szPadded*pix(1)))
    '''
    return {'values': values, 'values_sup': values_sup}


def calculate_MTF_CT(image2d, image_info, paramset, roi_array, delta_xya):
    """Calculate MTF for CT.

    Parameters
    ----------
    image2d : numpy.ndarray
    img_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetCT
        settings for the test as defined in config_classes.py
    roi_array : numpy.ndarray
        dtype bool, True = inside
    delta_xya : list
        center and angle offset from gui

    Returns
    -------
    values : list of float
        according to defined headers of selected test type
    values_sup : list of float

    details_dict : dict
    """
    values = []
    details_dict = {}

    

    return (values, details_dict)


def get_corrections_point_source(
        image2d, img_dict, roi_array,
        fit_x=False, fit_y=False, lock_z=-1.):
    """Estimate xyz of point source and generate correction matrix.

    Parameters
    ----------
    image2d : numpy.ndarray
        input image full resolution
    img_dict :  DcmInfo
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
            C, A, x0, sigma = gauss_4param_fit(np.arange(len(sum_vector)), sum_vector)
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
        dists_inplane = get_distance_map(image2d.shape, center_dx=dx, center_dy=dy)
        dists_inplane = img_dict.pix[0] * dists_inplane

        dists_ufov = dists_inplane[rows][:, cols]
        dists_ufov_flat = dists_ufov.flatten()
        values_flat = ufov_denoised.flatten()
        sort_idxs = np.argsort(dists_ufov_flat)
        values = values_flat[sort_idxs]
        dists = dists_ufov_flat[sort_idxs]

        # fit values
        nm_radius = img_dict.nm_radius if img_dict.nm_radius != -1. else 400.
        C, distance = point_source_func_fit(
            dists, values,
            center_value=np.max(ufov_denoised),
            avg_radius=nm_radius)
        fit = point_source_func(dists_inplane.flatten(), C, distance)
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

        dists = get_distance_map((roi_size, roi_size))
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
    #['SNI max', 'SNI L1', 'SNI L2', 'SNI S1', 'SNI S2',
    #         'SNI S3', 'SNI S4', 'SNI S5', 'SNI S6'],
    
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