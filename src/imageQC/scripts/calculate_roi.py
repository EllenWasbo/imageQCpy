#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculation processes for the different tests.

@author: Ellen Wasbø
"""

import numpy as np
from scipy import ndimage

from imageQC.scripts.mini_methods_calculate import (
    get_width_center_at_threshold, rotate_point
    )


def get_rois(image, image_number, input_main):
    """Get ROI array depending on test, modality and input image.

    Parameters
    ----------
    image : np.ndarray
        input image (or empty if pixelvalues irrelevant for the ROI)
    image_number : int
        image number in input_main.imgs
    input_main : object
            of class MainWindow or MainAuto containing specific attributes
            see try block below
            will not alter input_main

    Returns
    -------
    roi_array : np.ndarray or dict
        dimension ndarray according to input image or one extra dimension
        True/False where True = inside ROI, False = outside ROI
        dict if slice thickness CT
    """
    roi_array = None

    test_code = input_main.current_test
    paramset = input_main.current_paramset
    image_dict = input_main.imgs[image_number]
    img_shape = np.shape(image)
    try:
        delta_xya = [
            input_main.vGUI.delta_x,
            input_main.vGUI.delta_y,
            input_main.vGUI.delta_a]
    except AttributeError:
        delta_xya = [0, 0, 0.0]

    if test_code == 'ROI':
        # roi_type 0=circular, 1=rectangular, 2=rectangular wi
        if paramset.roi_type == 0:
            roi_size_in_pix = paramset.roi_radius / image_dict.pix[0]
            roi_array = get_roi_circle(
                img_shape, (delta_xya[0], delta_xya[1]), roi_size_in_pix)
        else:
            w = paramset.roi_x / image_dict.pix[0]
            h = paramset.roi_y / image_dict.pix[1]
            roi_array = get_roi_rectangle(
                img_shape, roi_width=w, roi_height=h,
                offcenter_xy=delta_xya[0:2])
            if paramset.roi_type == 2:  # rotated ROI
                roi_array = ndimage.rotate(
                    roi_array, paramset.roi_a, reshape=False)

    elif test_code == 'Hom':
        roi_array = get_roi_hom(image_dict, paramset,
                                delta_xya=delta_xya,
                                modality=input_main.current_modality)

    elif test_code == 'Noi':
        if input_main.current_modality == 'CT':
            roi_size_in_pix = paramset.noi_roi_size / image_dict.pix[0]
            roi_array = get_roi_circle(
                img_shape, (delta_xya[0], delta_xya[1]), roi_size_in_pix)
        if input_main.current_modality == 'Xray':
            w = 0.01 * paramset.noi_percent * img_shape[1]
            h = 0.01 * paramset.noi_percent * img_shape[0]
            roi_array = get_roi_rectangle(
                img_shape, roi_width=w, roi_height=h, offcenter_xy=(0, 0))

    elif test_code == 'HUw':
        roi_size_in_pix = paramset.huw_roi_size / image_dict.pix[0]
        roi_array = get_roi_circle(
            img_shape, (delta_xya[0], delta_xya[1]), roi_size_in_pix)

    elif test_code == 'MTF':
        if input_main.current_modality == 'CT':
            off_center_xy = delta_xya[0:2]
            roi_size_in_pix = paramset.mtf_roi_size / image_dict.pix[0]
            if paramset.mtf_offset_mm:
                extra_xy = np.array(paramset.mtf_offset_xy) / image_dict.pix[0]
            else:
                extra_xy = np.array(paramset.mtf_offset_xy)
            off_center_xy = np.add(extra_xy, np.array(off_center_xy))
            if paramset.mtf_type == 0:  # bead
                if paramset.mtf_auto_center:
                    yxmax = get_max_pos_yx(image)
                    off_center_xy = np.array(yxmax) - 0.5 * np.array(image.shape)
                roi_array = [[], []]
                roi_array[0] = get_roi_rectangle(
                    img_shape,
                    roi_width=2*roi_size_in_pix + 1,
                    roi_height=2*roi_size_in_pix + 1,
                    offcenter_xy=off_center_xy)
                bg_width_in_pix = paramset.mtf_background_width / image_dict.pix[0]
                background_outer = get_roi_rectangle(
                    img_shape,
                    roi_width=2*(roi_size_in_pix + bg_width_in_pix) + 1,
                    roi_height=2*(roi_size_in_pix + bg_width_in_pix) + 1,
                    offcenter_xy=off_center_xy)
                background_outer[roi_array[0] == True] = False
                roi_array[1] = background_outer
            elif paramset.mtf_type == 1:  # wire
                if paramset.mtf_auto_center:
                    pass#find postion of max in image or summed image? (summed of marked for gui, sum of group for actual calc)
                roi_array = get_roi_rectangle(
                    img_shape,
                    roi_width=2*roi_size_in_pix + 1,
                    roi_height=2*roi_size_in_pix + 1,
                    offcenter_xy=delta_xya[0:2])
            else:  # circular edge
                if paramset.mtf_auto_center:
                    pass#find position of center of max plateu?
                roi_array = get_roi_circle(
                    img_shape,
                    (delta_xya[0], delta_xya[1]), roi_size_in_pix)
        elif input_main.current_modality == 'Xray':
            dx = delta_xya[0]  # center of ROI
            dy = delta_xya[1]
            if paramset.mtf_auto_center:
                pass
                #center of object, then 4 rois - left/right top/btm roi + mean?
                #or x closest to center
            else:
                if any(paramset.mtf_offset_xy):
                    if paramset.mtf_offset_mm:
                        dx += paramset.mtf_offset_xy[0] / image_dict.pix[0]
                        dy += paramset.mtf_offset_xy[1] / image_dict.pix[0]
                    else:
                        dx += paramset.mtf_offset_xy[0]
                        dy += paramset.mtf_offset_xy[1]
            roi_array = get_roi_rectangle(
                img_shape,
                roi_width=paramset.mtf_roi_size_x / image_dict.pix[0],
                roi_height=paramset.mtf_roi_size_y / image_dict.pix[0],
                offcenter_xy=[dx, dy]
                )

    elif test_code == 'CTn':
        roi_array = get_roi_ctn(image, image_dict, paramset, delta_xya=delta_xya)

    elif test_code == 'Sli':
        if input_main.current_modality == 'CT':
            roi_array = get_slicethickness_start_stop(
                image_dict, paramset, delta_xya)
        elif input_main.current_modality == 'MR':
            pass#TODO

    elif test_code == 'Uni':
        roi_array = get_ratio_NM(
            image, image_dict,
            ufov_ratio=paramset.uni_ufov_ratio,
            cfov_ratio=paramset.uni_cfov_ratio
            )

    elif test_code == 'SNI':
        roi_array = get_roi_SNI(image, image_dict, paramset)

    elif test_code in ['SNR', 'PIU', 'Gho']:
        roi_array = get_roi_circle_MR(image, image_dict,
                                      paramset, test_code,
                                      (delta_xya[0], delta_xya[1])
                                      )

    return roi_array


def get_max_pos_yx(image):
    """Find position (row, col) with maximum pixel value."""
    row_maxs = np.max(image, axis=0)
    row_max_id = np.argmax(row_maxs)
    col_maxs = np.max(image, axis=1)
    col_max_id = np.argmax(col_maxs)

    return (row_max_id, col_max_id)


def get_roi_rectangle(image_shape,
                      roi_width=0, roi_height=0, offcenter_xy=(0, 0)):
    """Generate circular roi given center position and radius.

    Parameters
    ----------
    image_shape : tuple of ints
        image (rows,columns)
    roi_width : int
        width of ROI in pix
    roi_height : int
        height of ROI in pix
    offcenter_xy : arraylike of floats
        center of roi relative to center of image

    Returns
    -------
    roi : ndarray
        2d array with type 'bool'
    """
    inside = np.full(image_shape, False)
    if roi_width > 0 and roi_height > 0:
        center_pos_xy = [round(offcenter_xy[0] + 0.5*image_shape[1]),
                         round(offcenter_xy[1] + 0.5*image_shape[0])]
        start_x = center_pos_xy[0] - round(0.5*roi_width)
        start_y = center_pos_xy[1] - round(0.5*roi_height)
        end_x = start_x + round(roi_width)
        end_y = start_y + round(roi_height)
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        if end_x > image_shape[1]:
            end_x == image_shape[1]
        if end_y > image_shape[0]:
            end_y = image_shape[0]

        inside[start_y:end_y, start_x:end_x] = True

    return inside


def get_roi_circle(image_shape, delta_xy, radius):
    """Generate circular roi given center position and radius.

    Parameters
    ----------
    image_shape : tuple of ints
        image (rows,columns)
    delta_xy : tuple
        image center offset in pix from mid_image (delta_x, delta_y)
    radius : float
        radius of roi circle in pix

    Returns
    -------
    roi : ndarray
        2d array with type 'bool'
    """
    xs, ys = np.meshgrid(
        np.arange(0, image_shape[1], dtype=int),
        np.arange(0, image_shape[0], dtype=int),
        sparse=True)
    center_pos = [delta_xy[0] + round(0.5*image_shape[1]),
                  delta_xy[1] + round(0.5*image_shape[0])]

    zs = np.sqrt((xs-center_pos[0]) ** 2 + (ys-center_pos[1]) ** 2)
    inside = zs <= radius

    return inside


def get_outer_ring(radius):
    """Generate circular ring mask to extract background values (outer ring).

    Parameters
    ----------
    radius : int
        radius of subarr

    Returns
    -------
    mask : ndarray
        2d array with type 'bool' shape = (radius * 2, radius * 2)
    """
    xs, ys = np.meshgrid(
        np.arange(0, radius * 2, dtype=int),
        np.arange(0, radius * 2, dtype=int),
        sparse=True)

    zs = np.sqrt((xs-radius) ** 2 + (ys-radius) ** 2)
    mask = (zs > radius) | (zs < radius-1)

    return mask


def get_roi_hom(image_dict,
                test_params, delta_xya=[0, 0, 0.0], modality='CT'):
    """Calculate roi array with center roi and periferral rois.

    Parameters
    ----------
    image_dict : dict
        as minimum defined in scripts/dcm.py - read_dcm_info
    test_params : ParamSetXX
        ParamSet for given modality as defined in config/config_classes.py
    delta_xya : list
        center and angle offset [center x, center y, rotation].
        Default is [0,-0, 0.0]
    modality : str
        current_modality

    Returns
    -------
    roi_all : list of np.array
    """
    off_centers = []  # [x,y] = center of roi relative to centerpos

    roi_size_in_pix = test_params.hom_roi_size / image_dict.pix[0]

    if modality in ['CT', 'PET']:
        roi_dist_in_pix = round(
            test_params.hom_roi_distance / image_dict.pix[0])
        # central + 4 rois at 12, 15, 18, 21 o'clock optionally rotated
        rotation = delta_xya[2]
        if image_dict.modality == 'CT':
            rotation = rotation + test_params.hom_roi_rotation
        rotation_radians = np.deg2rad(rotation)
        if rotation_radians != 0:
            delta_rot = [np.cos(rotation_radians), np.sin(rotation_radians)]
            delta_rotation_xy = [roi_dist_in_pix * x for x in delta_rot]
        else:
            delta_rotation_xy = [roi_dist_in_pix, 0]

        off_centers = [delta_xya[0:2] for i in range(5)]
        dd_0, dd_1 = delta_rotation_xy
        off_centers[1][0] += dd_1  # 12 o'clock
        off_centers[1][1] -= dd_0
        off_centers[2][0] += dd_0  # 15 o'clock
        off_centers[2][1] += dd_1
        off_centers[3][0] -= dd_1  # 18 o'clock
        off_centers[3][1] += dd_0
        off_centers[4][0] -= dd_0  # 21 o'clock
        off_centers[4][1] -= dd_1

        roi_array = []
        for i in range(5):
            this_roi = get_roi_circle(
                image_dict.shape, tuple(off_centers[i]), roi_size_in_pix)
            roi_array.append(this_roi)

    elif modality == 'Xray':
        # central + 1 roi in each quadrant
        # optionally at specific distance and rotated

        rotation = delta_xya[2]
        rotation = rotation + test_params.hom_roi_rotation
        rotation_radians = np.deg2rad(rotation - 45)

        if rotation != 0 or test_params.hom_roi_distance > 0:

            center = [round(0.5 * image_dict.shape[1]) + delta_xya[0],
                      round(0.5 * image_dict.shape[0]) + delta_xya[1]]
            min_size = min([center[0], center[1],
                            image_dict.shape[1] - center[0],
                            image_dict.shape[0] - center[1]])
            if test_params.hom_roi_distance == 0:
                distance_pix = min_size * 0.5
                # centered in rotated quadrant with min dist to border
                # as reference distance
            else:
                distance_pix = round(
                    min_size * 0.01 * test_params.hom_roi_distance)
                # hom_roi_distane = % of shortest distance from center

            if rotation_radians != 0:
                delta_rot = [np.cos(rotation_radians),
                             np.sin(rotation_radians)]
                delta_rotation_xy = [distance_pix * x for x in delta_rot]
            else:
                delta_rotation_xy = [distance_pix, 0]

            off_centers = [delta_xya[0:2] for i in range(5)]
            dd_0, dd_1 = delta_rotation_xy
            off_centers[1][0] += dd_1  # 12 o'clock - 45 deg
            off_centers[1][1] -= dd_0
            off_centers[2][0] += dd_0  # 15 o'clock - 45 deg
            off_centers[2][1] += dd_1
            off_centers[3][0] -= dd_1  # 18 o'clock - 45 deg
            off_centers[3][1] += dd_0
            off_centers[4][0] -= dd_0  # 21 o'clock - 45 eeg
            off_centers[4][1] -= dd_1

        else:  # in quadrants
            dd_0 = round(image_dict.shape[1]/4)
            dd_1 = round(image_dict.shape[0]/4)

            off_centers = [delta_xya[0:2] for i in range(5)]
            off_centers[1][0] -= dd_0  # upper left
            off_centers[1][1] -= dd_1
            off_centers[2][0] -= dd_0  # lower left
            off_centers[2][1] += dd_1
            off_centers[3][0] += dd_0  # upper right
            off_centers[3][1] -= dd_1
            off_centers[4][0] += dd_0  # lower right
            off_centers[4][1] += dd_1

        roi_array = []
        for i in range(5):
            this_roi = get_roi_circle(
                image_dict.shape, tuple(off_centers[i]), roi_size_in_pix)
            roi_array.append(this_roi)
    else:
        roi_array = None

    return roi_array


def get_roi_ctn(image, image_dict, test_params, delta_xya=[0, 0, 0.]):
    """Calculate roi array with center roi and periferral rois.

    Parameters
    ----------
    image : np.ndarray
        pixeldata
    image_dict : dict
        as minimum defined in scripts/dcm.py - read_dcm_info
    test_params : ParamSetCT
        ParamSetCT as defined in config/config_classes.py
    delta_xya : list
        center and angle offset [center x, center y, rotation].
        Default is [0,-0, 0.0]

    Returns
    -------
    roi_all : list of np.array
        one 2d array for each material
        x 2 = first for actual ROI, next for search ROI (larger)
    """
    roi_size_in_pix = round(test_params.ctn_roi_size / image_dict.pix[0])

    roi_array = None

    if roi_size_in_pix > 0:
        n_rois = len(test_params.ctn_table.pos_x)
        off_centers = []
        rot_a = np.deg2rad(delta_xya[2])
        for r in range(n_rois):
            x = test_params.ctn_table.pos_x[r] / image_dict.pix[0]
            y = test_params.ctn_table.pos_y[r] / image_dict.pix[0]
            x += delta_xya[0]
            y += delta_xya[1]
            if rot_a != 0:
                x, y = rotate_point(
                    (x, y),
                    (delta_xya[0], delta_xya[1]),
                    delta_xya[2]
                    )
            off_centers.append([x, y])

        roi_search_array = []
        if test_params.ctn_search:  # optimize off_centers
            search_size_in_pix = round(
                test_params.ctn_search_size / image_dict.pix[0])
            for r in range(n_rois):
                this_roi = get_roi_circle(
                    image_dict.shape, tuple(off_centers[r]),
                    search_size_in_pix)
                roi_search_array.append(this_roi)

            # adjust off_center by finding center of object within
            radius = search_size_in_pix
            cy = 0.5 * image_dict.shape[0]
            cx = 0.5 * image_dict.shape[1]
            outer_val_ring_mask = get_outer_ring(radius)
            for r in range(n_rois):
                y = round(off_centers[r][1] + cy)
                x = round(off_centers[r][0] + cx)
                subarr = image[y-radius:y+radius, x-radius:x+radius]
                roi_mask = roi_search_array[r][y-radius:y+radius, x-radius:x+radius]
                background_arr = np.ma.masked_array(subarr, mask=outer_val_ring_mask)
                subarr[roi_mask == False] = np.mean(background_arr)
                size_y, size_x = subarr.shape
                if size_y > 0 and size_x > 0:
                    prof_y = np.sum(subarr, axis=1)
                    prof_x = np.sum(subarr, axis=0)
                    # get width at halfmax and center for profiles
                    width_x, center_x = get_width_center_at_threshold(
                        prof_x, 0.5 * (max(prof_x) + min(prof_x)))
                    width_y, center_y = get_width_center_at_threshold(
                        prof_y, 0.5 * (max(prof_y) + min(prof_y)))
                    off_centers[r][1] += center_y - radius
                    off_centers[r][0] += center_x - radius

        roi_array = []
        for r in range(n_rois):
            this_roi = get_roi_circle(
                image_dict.shape, tuple(off_centers[r]), roi_size_in_pix)
            roi_array.append(this_roi)

        if len(roi_search_array) == n_rois:
            roi_array.extend(roi_search_array)

    return roi_array


def get_ratio_NM(image, image_dict, ufov_ratio=0.95, cfov_ratio=0.75):
    """Calculate rectangular roi array for given ratio of UFOV, CFOV.

    First find non-zero part of image (ignoring padded part of image).

    Parameters
    ----------
    image : np.ndarray
        pixeldata
    image_dict : dict
        as minimum defined in scripts/dcm.py - read_dcm_info
    test_params : ParamSetNM
        ParamSetNM as defined in config/config_classes.py

    Returns
    -------
    roi_array : list of np.array
        one 2d array for UFOV [0] and CFOV [1]
    """
    roi_array = []

    # image might be padded, avoid padding
    image_binary = np.zeros(image.shape)
    image_binary[image > 0] = 1
    prof_y = np.max(image_binary, axis=1)
    width_y = np.count_nonzero(prof_y)
    prof_x = np.max(image_binary, axis=0)
    width_x = np.count_nonzero(prof_x)

    roi_array.append(
        get_roi_rectangle(
            image.shape,
            roi_width=ufov_ratio * width_x,
            roi_height=ufov_ratio * width_y
            )
        )
    roi_array.append(
        get_roi_rectangle(
            image.shape,
            roi_width=cfov_ratio * width_x,
            roi_height=cfov_ratio * width_y
            )
        )

    return roi_array


def get_roi_SNI(image, image_dict, test_params):
    """Generate roi_array for NM SNI test.

    Parameters
    ----------
    image : nparr
    image_dict : dict
        as minimum defined in scripts/dcm.py - read_dcm_info
    test_params : ParamSetNM
        for given modality as defined in config/config_classes.py

    Returns
    -------
    roi : list of ndarray
        2d arrays with type 'bool'
        2 large ROIs (left, rigth), 6 smaller ROIs left to right, top to bottom
    """
    roi_full, not_used = get_ratio_NM(
        image, image_dict, ufov_ratio=test_params.sni_area_ratio)
    roi_array = [roi_full]

    rows = np.max(roi_full, axis=1)
    width_y = np.count_nonzero(rows)
    cols = np.max(roi_full, axis=0)
    width_x = np.count_nonzero(cols)
    large_dim = width_y
    small_dim = round(0.5*large_dim)

    idxs_col = np.where(cols == True)
    first_col = idxs_col[0][0]
    idxs_row = np.where(rows == True)
    first_row = idxs_row[0][0]

    # large ROIs (2)
    left_large = np.full(roi_full.shape, False)
    left_large[rows, first_col:first_col + large_dim] = True
    roi_array.append(left_large)
    roi_array.append(np.fliplr(left_large))
    # small ROIs (6)
    upper_left = np.full(roi_full.shape, False)
    upper_left[first_row:first_row + small_dim, first_col:first_col + small_dim] = True
    roi_array.append(upper_left)
    upper_mid = np.full(roi_full.shape, False)
    s2 = round(small_dim/2)
    sz_y, sz_x = image.shape
    w2 = round(sz_x/2)
    first_mid = w2-1-s2
    upper_mid[first_row:first_row + small_dim, first_mid:first_mid + small_dim] = True
    roi_array.append(upper_mid)
    roi_array.append(np.fliplr(upper_left))
    roi_array.append(np.flipud(upper_left))
    roi_array.append(np.flipud(upper_mid))
    roi_array.append(np.flipud(np.fliplr(upper_left)))

    return roi_array


def get_roi_circle_MR(image, image_dict, test_params, test_code, delta_xy):
    """Generate circular roi for MR tests.

    Circular ROI.
    Optionally optimized center based on maximum projection in x,y.
    Optionally cut top to avoid phantom structures on top (ACR phantom).

    Parameters
    ----------
    image : nparr
    image_dict : dict
        as minimum defined in scripts/dcm.py - read_dcm_info
    test_params : ParamSetMR
        for given modality as defined in config/config_classes.py
    test_code : str
    delta_xy : tuple
        image center offset in pix from mid_image (delta_x, delta_y)

    Returns
    -------
    roi : ndarray
        2d array with type 'bool'
    """
    roi_array = None

    # get maximum profiles x and y
    prof_y = np.max(image, axis=1)
    prof_x = np.max(image, axis=0)
    # get width at halfmax and center for profiles
    width_x, center_x = get_width_center_at_threshold(
        prof_x, 0.5 * (max(prof_x)+min(prof_x)))
    width_y, center_y = get_width_center_at_threshold(
        prof_y, 0.5 * (max(prof_y)+min(prof_y)))
    if width_x > -1 and width_y > -1:
        radius = -1
        width = 0.5 * (width_x + width_y)
        optimize_center = True
        cut_top = 0
        if test_code == 'SNR':
            radius = 0.01 * test_params.snr_roi_percent * 0.5 * width
            cut_top = test_params.snr_roi_cut_top
        elif test_code == 'PIU':
            radius = 0.01 * test_params.piu_roi_percent * 0.5 * width
            cut_top = test_params.piu_roi_cut_top
        elif test_code == 'Gho':
            radius = test_params.gho_roi_central / image_dict.pix[0]
            optimize_center = test_params.gho_optimize_center
            cut_top = test_params.gho_roi_cut_top
        if optimize_center:
            delta_xy = (
                center_x - 0.5*image_dict.shape[1],
                center_y - 0.5*image_dict.shape[0]
                )
        roi_array = get_roi_circle(image_dict.shape, delta_xy, radius)

        if cut_top > 0:
            cut_top = cut_top / image_dict.pix[1]
            y_top = 0.5*image_dict.shape[1] + delta_xy[1] - radius + cut_top
            y_start = 0
            y_stop = round(y_top)
            roi_array[y_start:y_stop, :] = False

        if test_code == 'Gho':  # add rectangular ROIs at borders
            # rois: 'Center', 'top', 'bottom', 'left', 'right'
            roi_array = [roi_array]
            roi_w = round(test_params.gho_roi_w / image_dict.pix[0])
            roi_h = round(test_params.gho_roi_h / image_dict.pix[0])
            roi_d = round(test_params.gho_roi_dist / image_dict.pix[0])
            w = [roi_w, roi_w, roi_h, roi_h]
            h = w[::-1]
            delta = roi_h/2 + roi_d
            offxy = [
                (0, -(image_dict.shape[1]/2 - delta)),
                (0, image_dict.shape[1]/2 - delta),
                (-(image_dict.shape[0]/2 - delta), 0),
                (image_dict.shape[0]/2 - delta, 0)
                ]

            for i in range(4):
                roi_array.append(
                    get_roi_rectangle(
                        image_dict.shape,
                        roi_width=w[i], roi_height=h[i],
                        offcenter_xy=(round(offxy[i][0]), round(offxy[i][1]))
                        )
                    )

    return roi_array


def get_slicethickness_start_stop(image_info, paramset, dxya):
    """Get start and stop coordinates for lines for slicethickness test CT.

    Parameters
    ----------
    image_info : dict
        as minimum defined in scripts/dcm.py - read_dcm_info
    paramset : ParamSetXX
        for given modality as defined in config/config_classes.py
    dxya : list of float
        image center offset in pix from mid_image [delta_x, delta_y, rot_deg]

    Returns
    -------
    dict
        h_lines: [[r0,c0,r1,c1],[r0,c0,r1,c1]]
        v_lines: as h_lines
        number of lines depend on paramset.sli_type
    """
    size_xhalf = 0.5 * image_info.shape[1]
    size_yhalf = 0.5 * image_info.shape[0]
    prof_half = 0.5 * (paramset.sli_ramp_length / image_info.pix[0])
    if paramset.sli_type != 1:
        dist = paramset.sli_ramp_distance / image_info.pix[0]
    else:
        dist = 45. / image_info.pix[0]  # beaded ramp Catphan outer ramps
    dist_b = 25. / image_info.pix[0]  # beaded ramp Catphan inner ramps

    # for each line: [y1, x1, y2, x2]
    h_lines = []
    v_lines = []
    rotation_radians = np.deg2rad(-dxya[2])
    cos_rot = np.cos(rotation_radians)
    sin_rot = np.sin(rotation_radians)

    center_x = size_xhalf + dxya[0]
    center_y = size_yhalf + dxya[1]

    # vertical line 1
    if dxya[2] == 0:
        x1 = center_x - dist
        x2 = x1
        y1 = center_y - prof_half
        y2 = center_y + prof_half
    else:
        x1 = center_x - dist*cos_rot - prof_half*sin_rot
        x2 = center_x - dist*cos_rot + prof_half*sin_rot
        y1 = center_y + dist*sin_rot - prof_half*cos_rot
        y2 = center_y + dist*sin_rot + prof_half*cos_rot
    v_lines.append([round(y1), round(x1), round(y2), round(x2)])

    # vertical line 2
    if dxya[2] == 0:
        x1 = center_x + dist
        x2 = x1
    else:
        x1 = center_x + dist*cos_rot - prof_half*sin_rot
        x2 = center_x + dist*cos_rot + prof_half*sin_rot
        y1 = center_y - dist*sin_rot - prof_half*cos_rot
        y2 = center_y - dist*sin_rot + prof_half*cos_rot
    v_lines.append([round(y1), round(x1), round(y2), round(x2)])

    if paramset.sli_type in [0, 1]:  # Catphan wire or beaded

        # first horizontal line coordinates
        if dxya[2] == 0:
            x1 = center_x - prof_half
            x2 = center_x + prof_half
            y1 = center_y + dist
            y2 = y1
        else:
            x1 = center_x + dist*sin_rot - prof_half*cos_rot
            x2 = center_x + dist*sin_rot + prof_half*cos_rot
            y1 = center_y + dist*cos_rot + prof_half*sin_rot
            y2 = center_y + dist*cos_rot - prof_half*sin_rot
        h_lines.append([round(y1), round(x1), round(y2), round(x2)])

        # second horizontal line coordinates
        if dxya[2] == 0:
            y1 = center_y - dist
            y2 = y1
        else:
            x1 = center_x - dist*sin_rot - prof_half*cos_rot
            x2 = center_x - dist*sin_rot + prof_half*cos_rot
            y1 = center_y - dist*cos_rot + prof_half*sin_rot
            y2 = center_y - dist*cos_rot - prof_half*sin_rot
        h_lines.append([round(y1), round(x1), round(y2), round(x2)])

        if paramset.sli_type == 1:  # Catphan beaded helical
            dist = dist_b
            # third horizontal line coordinates
            if dxya[2] == 0:
                x1 = center_x - prof_half
                x2 = center_x + prof_half
                y1 = center_y + dist
                y2 = y1
            else:
                x1 = center_x + dist*sin_rot - prof_half*cos_rot
                x2 = center_x + dist*sin_rot + prof_half*cos_rot
                y1 = center_y + dist*cos_rot + prof_half*sin_rot
                y2 = center_y + dist*cos_rot - prof_half*sin_rot
            h_lines.append([round(y1), round(x1), round(y2), round(x2)])

            # forth horizontal line coordinates
            if dxya[2] == 0:
                y1 = center_y - dist
                y2 = y1
            else:
                x1 = center_x - dist*sin_rot - prof_half*cos_rot
                x2 = center_x - dist*sin_rot + prof_half*cos_rot
                y1 = center_y - dist*cos_rot + prof_half*sin_rot
                y2 = center_y - dist*cos_rot - prof_half*sin_rot
            h_lines.append([round(y1), round(x1), round(y2), round(x2)])

    return {'h_lines': h_lines, 'v_lines': v_lines}
