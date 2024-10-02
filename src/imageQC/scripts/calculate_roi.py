#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculation processes for the different tests.

@author: Ellen Wasbø
"""
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import feature

# imageQC block start
import imageQC.scripts.mini_methods_calculate as mmcalc
import imageQC.scripts.dcm as dcm
# imageQC block end


def get_rois(image, image_number, input_main):
    """Get ROI array depending on test, modality and input image.

    INFO:
    The inner functions pr testcode should return either
    roi_array or
    (roi_array, errmsg)
    Ther roi_array should either be a numpy 2d.array or a list of numpy 2d.arrays.

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
    errmsg : str or list of str or None
    """
    roi_array = None
    errmsg = None

    test_code = input_main.current_test
    paramset = input_main.current_paramset
    image_info = input_main.imgs[image_number]
    img_shape = np.shape(image)
    try:
        delta_xya = [
            input_main.gui.delta_x,
            input_main.gui.delta_y,
            input_main.gui.delta_a]
    except AttributeError:
        delta_xya = [0, 0, 0.0]

    def Bar():  # Bar phantom NM
        return get_roi_NM_bar(image, image_info, paramset)

    def Cro():  # PET Crosscalibration
        roi_size_in_pix = paramset.cro_roi_size / image_info.pix[0]
        return get_roi_circle(img_shape, (delta_xya[0], delta_xya[1]), roi_size_in_pix)

    def CTn():  # CT number
        return get_roi_CTn_TTF('ctn', image, image_info, paramset, delta_xya=delta_xya)

    def Dim():  # CT Dimensions
        roi_this = []
        roi_size_in_pix = 10./image_info.pix[0]  # search radius 10mm
        dist = np.sqrt(2 * 25.**2)/image_info.pix[0]
        # search centers +/- 25mm from center

        rotation_radians = np.deg2rad(delta_xya[2] - 45)
        delta_rot = [np.cos(rotation_radians), np.sin(rotation_radians)]
        delta_rotation_xy = [dist * x for x in delta_rot]

        centers = [delta_xya[0:2] for i in range(4)]
        dd_0, dd_1 = delta_rotation_xy
        centers[0][0] += dd_1  # 12 o'clock - 45
        centers[0][1] -= dd_0
        centers[1][0] += dd_0  # 15 o'clock - 45
        centers[1][1] += dd_1
        centers[2][0] -= dd_1  # 18 o'clock - 45
        centers[2][1] += dd_0
        centers[3][0] -= dd_0  # 21 o'clock - 45
        centers[3][1] -= dd_1

        for center in centers:
            roi_this.append(get_roi_circle(
                img_shape, center, roi_size_in_pix))

        centers_x = [x[0] + img_shape[1] // 2 for x in centers]
        centers_y = [x[1] + img_shape[0] // 2 for x in centers]
        roi_this.append([centers_x, centers_y])

        return (roi_this, errmsg)

    def Geo():  # Geometry
        return get_roi_circle_MR(
            image, image_info, paramset, test_code, (delta_xya[0], delta_xya[1])
            )

    def Gho():  # Ghost
        roi_this = None
        if input_main.current_modality == 'Mammo':
            xs = [x for x in paramset.gho_table.pos_x]
            if paramset.gho_relative_to_right:
                xs = image_info.pix[0] * img_shape[1] / 2 - np.array(xs)
            else:
                xs = - image_info.pix[0] * img_shape[1] / 2 + np.array(xs)
            ys = [y + delta_xya[1]*image_info.pix[1] for y in paramset.gho_table.pos_y]
            xs = xs / image_info.pix[0]
            ys = np.array(ys) / image_info.pix[0]
            off_center_xys = [(xs[i], ys[i]) for i in range(len(xs))]
            w = paramset.gho_roi_size / image_info.pix[0]
            roi_this = [get_roi_rectangle(
                img_shape, roi_width=w, roi_height=w,
                offcenter_xy=off_xy) for off_xy in off_center_xys]
        elif input_main.current_modality == 'MR':
            roi_this, errmsg = get_roi_circle_MR(
                image, image_info, paramset, test_code, (delta_xya[0], delta_xya[1])
                )
        return (roi_this, errmsg)

    def Hom():  # Homogeneity
        flatfield = False
        if input_main.current_modality == 'Mammo':
            flatfield = True
        elif input_main.current_modality == 'Xray':
            if paramset.hom_tab_alt == 3:
                flatfield = True
        if flatfield:
            roi_array = get_roi_hom_flatfield(image_info, paramset)
            roi_mask = np.full(image_info.shape[0:2], False)
            if paramset.hom_mask_max:
                roi_mask[image == np.max(image)] = True
            roi_array.append(roi_mask)
            roi_mask_outer = np.full(image_info.shape[0:2], False)
            if paramset.hom_mask_outer_mm > 0:
                n_pix = round(paramset.hom_mask_outer_mm / image_info.pix[0])
                if n_pix > 0:
                    roi_mask_outer[n_pix:-n_pix, n_pix:-n_pix] = True
            roi_array.append(roi_mask_outer)
        else:
            roi_array = get_roi_hom(image_info, paramset, delta_xya=delta_xya,
                                    modality=input_main.current_modality)

        return roi_array

    def HUw():  # CT HU water
        roi_size_in_pix = paramset.huw_roi_size / image_info.pix[0]
        return get_roi_circle(
            img_shape, (delta_xya[0], delta_xya[1]), roi_size_in_pix)

    def MTF():
        roi_this = None
        errmsg = None
        off_center_xy = delta_xya[0:2]
        try:
            if paramset.mtf_offset_mm:
                extra_xy = np.array(paramset.mtf_offset_xy) / image_info.pix[0]
            else:
                extra_xy = np.array(paramset.mtf_offset_xy)
            off_center_xy = np.add(extra_xy, np.array(off_center_xy))
        except AttributeError:
            pass

        if input_main.current_modality in ['CT', 'SPECT', 'PET']:
            roi_size_in_pix = paramset.mtf_roi_size / image_info.pix[0]

            bg_rim = True
            if paramset.mtf_type == 0:  # bead / point source
                if paramset.mtf_auto_center:
                    filt_img = ndimage.gaussian_filter(image, sigma=5)
                    yxmax = get_max_pos_yx(filt_img)
                    off_center_xy = np.array(yxmax) - 0.5 * np.array(image.shape)
                roi_this = [[], []]
                roi_this[0] = get_roi_rectangle(
                    img_shape,
                    roi_width=2*roi_size_in_pix + 1,
                    roi_height=2*roi_size_in_pix + 1,
                    offcenter_xy=off_center_xy)
            else:  # circular edge / wire or cylinder source / line source
                if paramset.mtf_auto_center:
                    sigma = 5
                    if paramset.mtf_type == 2 and input_main.current_modality == 'CT':
                        summed_img = image
                    else:
                        if 'MainWindow' in str(type(input_main)):
                            if input_main.summed_img is None:  # avoid each time active slice changes
                                input_main.summed_img, _ = dcm.sum_marked_images(
                                    input_main.imgs,
                                    input_main.tree_file_list.get_marked_imgs_current_test(),
                                    tag_infos=input_main.tag_infos
                                    )
                            summed_img = input_main.summed_img
                        else:
                            summed_img = image
                        if paramset.mtf_type == 3:  # line z-res
                            sigma = 100
                        elif paramset.mtf_type == 4:  # cylinder z res
                            sigma = 100

                    filt_img = ndimage.gaussian_filter(summed_img, sigma=sigma)
                    yxmax = get_max_pos_yx(filt_img)
                    off_center_xy = np.array(yxmax) - 0.5 * np.array(summed_img.shape)

                if paramset.mtf_type == 2 and input_main.current_modality == 'CT':
                    # circular edge
                    roi_this = get_roi_circle(
                        img_shape, off_center_xy, roi_size_in_pix)
                    bg_rim = False
                elif paramset.mtf_type == 4:  # z res cylinder
                    roi_this = get_roi_rectangle(
                        img_shape,
                        roi_width=2*roi_size_in_pix + 1,
                        roi_height=2*roi_size_in_pix + 1,
                        offcenter_xy=off_center_xy)
                    bg_rim = False
                else:  # wire/line or z-res cylinder
                    roi_this = [[], []]
                    roi_this[0] = get_roi_rectangle(
                        img_shape,
                        roi_width=2*roi_size_in_pix + 1,
                        roi_height=2*roi_size_in_pix + 1,
                        offcenter_xy=off_center_xy)

            # outer background rim
            if bg_rim:  # bead/wire or point/line
                bg_width_in_pix = paramset.mtf_background_width // image_info.pix[0]
                background_outer = get_roi_rectangle(
                    img_shape,
                    roi_width=2*(roi_size_in_pix + bg_width_in_pix) + 1,
                    roi_height=2*(roi_size_in_pix + bg_width_in_pix) + 1,
                    offcenter_xy=off_center_xy)
                background_outer[roi_this[0] == True] = False
                roi_this[1] = background_outer

        elif input_main.current_modality in ['Xray', 'Mammo', 'MR']:
            dx = delta_xya[0]  # center of ROI
            dy = delta_xya[1]
            roi_sz_xy = [
                paramset.mtf_roi_size_x / image_info.pix[0],
                paramset.mtf_roi_size_y / image_info.pix[1]
                ]
            if paramset.mtf_auto_center:
                mask_outer_pix = round(
                    paramset.mtf_auto_center_mask_outer / image_info.pix[0])
                rect_dict = find_rectangle_object(image, mask_outer=mask_outer_pix)

                roi_this = []
                if rect_dict['centers_of_edges_xy'] is not None:
                    centers_of_edges_xy = rect_dict['centers_of_edges_xy']
                    cent = [image.shape[1] // 2, image.shape[0] // 2]
                    off_xy = [
                        [xy[0] - cent[0], xy[1] - cent[1]]
                        for xy in centers_of_edges_xy]
                    if paramset.mtf_auto_center_type == 1:
                        # find most central edge
                        distsq = [np.sum(np.power(xy, 2)) for xy in off_xy]
                        idx_min = np.argmin(distsq)
                        height_idx = idx_min % 2
                        width_idx = 1 if height_idx == 0 else 0
                        roi_this = [get_roi_rectangle(
                            img_shape,
                            roi_width=roi_sz_xy[width_idx],
                            roi_height=roi_sz_xy[height_idx],
                            offcenter_xy=off_xy[idx_min]
                            )]
                    else:
                        for i in range(len(off_xy)):
                            height_idx = i % 2
                            width_idx = 1 if height_idx == 0 else 0
                            roi_this.append(get_roi_rectangle(
                                img_shape,
                                roi_width=roi_sz_xy[width_idx],
                                roi_height=roi_sz_xy[height_idx],
                                offcenter_xy=off_xy[i]
                                )
                            )

                if mask_outer_pix > 0:
                    inside = np.full(img_shape, False)
                    inside[mask_outer_pix:-mask_outer_pix,
                           mask_outer_pix:-mask_outer_pix] = True
                    roi_this.append(inside)

            else:
                if any(paramset.mtf_offset_xy):
                    if paramset.mtf_offset_mm:
                        dx += roi_sz_xy[0]
                        dy += roi_sz_xy[1]
                    else:
                        dx += paramset.mtf_offset_xy[0]
                        dy += paramset.mtf_offset_xy[1]
                roi_this = get_roi_rectangle(
                    img_shape,
                    roi_width=roi_sz_xy[0], roi_height=roi_sz_xy[1],
                    offcenter_xy=[dx, dy]
                    )

        elif input_main.current_modality == 'NM':
            dx = delta_xya[0]  # center of ROI
            dy = delta_xya[1]
            roi_sz_xy = [
                paramset.mtf_roi_size_x / image_info.pix[0],
                paramset.mtf_roi_size_y / image_info.pix[1]
                ]
            if paramset.mtf_auto_center and paramset.mtf_type != 3:
                if paramset.mtf_type in [0, 1]:
                    filt_img = ndimage.gaussian_filter(image, sigma=5)
                    yxmax = get_max_pos_yx(filt_img)
                    offcenter_xy = np.array(yxmax) - 0.5 * np.array(image.shape)

                elif paramset.mtf_type == 2:  # perpendicular line sources
                    res = find_lines(image)
                    if len(res[0]) == 2 and len(res[1]) == 2:
                        roi_this = []
                        for i in [1, 0]:  # 1 first for line in xdir = y profile
                            height_idx = i % 2
                            width_idx = 1 if height_idx == 0 else 0
                            roi_this.append(get_roi_rectangle(
                                img_shape,
                                roi_width=roi_sz_xy[width_idx],
                                roi_height=roi_sz_xy[height_idx],
                                offcenter_xy=res[i]
                                )
                            )
                    else:
                        errmsg = 'Failed finding the two lines.'
            else:
                offcenter_xy = [dx, dy]

            if roi_this is None:
                roi_this = get_roi_rectangle(
                    img_shape,
                    roi_width=roi_sz_xy[0], roi_height=roi_sz_xy[1],
                    offcenter_xy=offcenter_xy
                    )

        return (roi_this, errmsg)

    def Noi():  # Noise
        roi_this = None
        if input_main.current_modality == 'CT':
            roi_size_in_pix = paramset.noi_roi_size / image_info.pix[0]
            roi_this = get_roi_circle(
                img_shape, (delta_xya[0], delta_xya[1]), roi_size_in_pix)
        if input_main.current_modality == 'Xray':
            w = 0.01 * paramset.noi_percent * img_shape[1]
            h = 0.01 * paramset.noi_percent * img_shape[0]
            roi_this = get_roi_rectangle(
                img_shape, roi_width=w, roi_height=h, offcenter_xy=(0, 0))

        return roi_this

    def NPS():
        if input_main.current_modality == 'CT':
            center_xy = (delta_xya[0] + img_shape[1] // 2,
                         delta_xya[1] + img_shape[0] // 2)
            dist_in_pix = paramset.nps_roi_distance / image_info.pix[0]
            first_xy = (center_xy[0], center_xy[1] - dist_in_pix)
            angle_dist = 360 / paramset.nps_n_sub
            roi_array = []
            for i in range(paramset.nps_n_sub):
                center_roi = mmcalc.rotate_point(first_xy, center_xy, i * angle_dist)
                offcenter_roi = (center_roi[0] - img_shape[1] // 2,
                                 center_roi[1] - img_shape[0] // 2)
                roi_array.append(get_roi_rectangle(
                    img_shape, roi_width=paramset.nps_roi_size,
                    roi_height=paramset.nps_roi_size, offcenter_xy=offcenter_roi))
        elif input_main.current_modality in ['Xray', 'Mammo']:
            roi_array = []
            pos = 0.5 * paramset.nps_roi_size * (
                np.arange(2*paramset.nps_n_sub-1) - (paramset.nps_n_sub-1))
            for i in pos:
                for j in pos:
                    roi_array.append(get_roi_rectangle(
                        img_shape, roi_width=paramset.nps_roi_size,
                        roi_height=paramset.nps_roi_size, offcenter_xy=(i, j)))

        else:
            roi_array = None
        return roi_array

    def Num():  # Number recognition
        roi_array = []
        if len(paramset.num_table.pos_x) > 0:
            for i in range(len(paramset.num_table.pos_x)):
                roi_array.append(get_roi_rectangle(
                    img_shape,
                    coords_x=paramset.num_table.pos_x[i],
                    coords_y=paramset.num_table.pos_y[i],
                    extra_offset=delta_xya[0:2]
                    ))

        return roi_array

    def PIU():  # MR
        return get_roi_circle_MR(
            image, image_info, paramset, test_code, (delta_xya[0], delta_xya[1])
            )

    def Rec():  # Recovery Curve
        if 'MainWindow' in str(type(input_main)):
            if input_main.summed_img is None:  # avoid each time active slice changes
                input_main.summed_img, _ = dcm.sum_marked_images(
                    input_main.imgs,
                    input_main.tree_file_list.get_marked_imgs_current_test(),
                    tag_infos=input_main.tag_infos
                    )
            summed_img = input_main.summed_img
        else:
            summed_img = image

        return get_roi_recovery_curve(summed_img, image_info, paramset)

    def Rin():  # Ring artifact
        roi_this = [None, None]
        rin_range_start = max(4*image_info.pix[0], paramset.rin_range_start)
        inner_roi_size = rin_range_start / image_info.pix[0]
        roi_this[0] = get_roi_circle(img_shape, (0, 0), inner_roi_size)
        outer_roi_size = paramset.rin_range_stop / image_info.pix[0]
        roi_this[1] = get_roi_circle(img_shape, (0, 0), outer_roi_size)

        return roi_this

    def RLR():  # ROI left/righy mammo
        roi_this = None
        if input_main.current_modality == 'Mammo':
            x_pos = paramset.rlr_x_mm / image_info.pix[0]
            if paramset.rlr_relative_to_right:
                x_pos = image_info.pix[0] * img_shape[1] / 2 - paramset.rlr_x_mm
            else:
                x_pos = - image_info.pix[0] * img_shape[1] / 2 + paramset.rlr_x_mm
            x_pos = x_pos / image_info.pix[0]
            w = paramset.rlr_roi_size / image_info.pix[0]
            roi_this = get_roi_rectangle(
                img_shape, roi_width=w, roi_height=w,
                offcenter_xy=(x_pos, 0))
        return roi_this

    def ROI():  # single og multiple ROI statistics
        roi_array = []

        if paramset.roi_use_table in [0, 1]:
            if paramset.roi_offset_mm:
                extra_xy = np.array(paramset.roi_offset_xy) / image_info.pix[0]
            else:
                extra_xy = np.array(paramset.roi_offset_xy)
            if paramset.roi_use_table == 0:
                off_center_xys = [tuple(np.add(extra_xy, np.array(delta_xya[0:2])))]
            else:  # 1
                xs = [x for x in paramset.roi_table.pos_x]
                xs = np.array(xs) / image_info.pix[0] + delta_xya[0]
                ys = [y for y in paramset.roi_table.pos_y]
                ys = np.array(ys) / image_info.pix[1] + delta_xya[1]
                off_center_xys = [(xs[i], ys[i]) for i in range(len(xs))]

            # roi_type 0=circular, 1=rectangular, 2=rectangular angled
            if paramset.roi_type == 0:
                roi_size_in_pix = paramset.roi_radius / image_info.pix[0]
                roi_array = [get_roi_circle(
                    img_shape, off_xy, roi_size_in_pix) for off_xy in off_center_xys]
            else:
                w = paramset.roi_x / image_info.pix[0]
                h = paramset.roi_y / image_info.pix[1]
                roi_array = [get_roi_rectangle(
                    img_shape, roi_width=w, roi_height=h,
                    offcenter_xy=off_xy) for off_xy in off_center_xys]
                if paramset.roi_type == 2 and paramset.roi_a != 0:  # rotated ROI
                    roi_array_rotated = []
                    for i, roi in enumerate(roi_array):
                        if any(off_center_xys[i]):
                            roi = mmcalc.rotate2d_offcenter(
                                roi.astype(float), -paramset.roi_a, off_center_xys[i])
                        else:
                            roi = ndimage.rotate(
                                roi.astype(float), -paramset.roi_a, reshape=False)
                        roi = np.round(roi)
                        roi = np.array(roi, dtype=bool)
                        roi_array_rotated.append(roi)
                    roi_array = roi_array_rotated

        else:  # zoomed table
            if len(paramset.roi_table.pos_x) > 0:
                for i in range(len(paramset.roi_table.pos_x)):
                    roi_array.append(get_roi_rectangle(
                        img_shape,
                        coords_x=paramset.roi_table.pos_x[i],
                        coords_y=paramset.roi_table.pos_y[i]
                        ))

        return roi_array

    def SDN():  # SDNR Mammo
        errmsg = None
        off_xy = None
        roi_sz = paramset.sdn_roi_size / image_info.pix[0]
        roi_dist = paramset.sdn_roi_dist / image_info.pix[0]
        mask_outer_pix = round(
            paramset.sdn_auto_center_mask_outer / image_info.pix[0])  # ignore outer
        if paramset.sdn_auto_center:
            res = mmcalc.find_center_object(image, mask_outer=mask_outer_pix,
                                            tolerances_width=[80, 120], sigma=5.)
            if res is None:
                errmsg = 'Failed finding center of object.'
            else:
                center_x, center_y, width_x, width_y = res

                off_xy = [
                    center_x - 0.5*image_info.shape[1],
                    center_y - 0.5*image_info.shape[0]
                    ]
        if off_xy is None:
            off_xy = delta_xya[0:2]

        roi_this = [get_roi_rectangle(
                img_shape, roi_width=roi_sz, roi_height=roi_sz,
                offcenter_xy=off_xy)]
        for delta_xy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            roi_this.append(
                get_roi_rectangle(
                    img_shape, roi_width=roi_sz, roi_height=roi_sz,
                    offcenter_xy=[
                        off_xy[0] + delta_xy[0]*roi_dist,
                        off_xy[1] + delta_xy[1]*roi_dist])
                )

        if mask_outer_pix > 0:
            inside = np.full(img_shape, False)
            inside[mask_outer_pix:-mask_outer_pix,
                   mask_outer_pix:-mask_outer_pix] = True
            roi_this.append(inside)

        return (roi_this, errmsg)

    def Sli():
        roi_this, errmsg = get_slicethickness_start_stop(
            image, image_info, paramset, delta_xya,
            modality=input_main.current_modality)

        return (roi_this, errmsg)

    def SNI():  # NM structured noise index
        return get_roi_SNI(image, image_info, paramset)

    def SNR():
        central_ROI, errmsg = get_roi_circle_MR(
            image, image_info, paramset, test_code, (delta_xya[0], delta_xya[1])
            )
        if paramset.snr_type == 1:  # singel image with background ROIs
            roi_sz = round(paramset.snr_background_size / image_info.pix[0])
            roi_dist = round(paramset.snr_background_dist / image_info.pix[0])
            bg_ROI = np.full(image_info.shape, False)
            bg_ROI[roi_dist:roi_dist+roi_sz, roi_dist:roi_dist+roi_sz] = True
            bg_ROI[
                image_info.shape[0]-roi_dist-roi_sz:image_info.shape[0]-roi_dist,
                image_info.shape[1]-roi_dist-roi_sz:image_info.shape[1]-roi_dist
                ] = True
            bg_ROI[
                roi_dist:roi_dist+roi_sz,
                image_info.shape[1]-roi_dist-roi_sz:image_info.shape[1]-roi_dist
                ] = True
            bg_ROI[
                image_info.shape[0]-roi_dist-roi_sz:image_info.shape[0]-roi_dist,
                roi_dist:roi_dist+roi_sz
                ] = True
            roi_array = [central_ROI, bg_ROI]
        else:
            roi_array = central_ROI
        return (roi_array, errmsg)

    def Spe():  # Speed WB NM
        roi_height_in_pix = 10. * paramset.spe_height / image_info.pix[0]
        return get_roi_rectangle(
            img_shape, roi_width=paramset.spe_avg, roi_height=roi_height_in_pix,
            offcenter_xy=delta_xya[0:2])

    def STP():
        roi_size_in_pix = paramset.stp_roi_size / image_info.pix[0]
        return get_roi_circle(img_shape, tuple(delta_xya[0:2]), roi_size_in_pix)

    def TTF():
        return get_roi_CTn_TTF('ttf', image, image_info, paramset, delta_xya=delta_xya)

    def Uni():  # Uniformity NM
        return get_ratio_NM(
            image, image_info,
            mask_corner=paramset.uni_mask_corner,
            ufov_ratio=paramset.uni_ufov_ratio,
            cfov_ratio=paramset.uni_cfov_ratio
            )

    def Var():  # Variance Xray
        roi_size_in_pix = paramset.var_roi_size / image_info.pix[0]
        roi_small = get_roi_rectangle(
            img_shape, roi_width=roi_size_in_pix, roi_height=roi_size_in_pix)
        w = 0.01 * paramset.var_percent * img_shape[1]
        h = 0.01 * paramset.var_percent * img_shape[0]
        roi_large = get_roi_rectangle(
            img_shape, roi_width=w, roi_height=h, offcenter_xy=(0, 0))
        return [roi_large, roi_small]

    if image is not None:
        try:
            res = locals()[test_code]()
            if isinstance(res, tuple):
                roi_array, errmsg = res
            else:
                roi_array = res
        except (KeyError, AttributeError, IndexError):
            pass

    return (roi_array, errmsg)


def get_max_pos_yx(image):
    """Find position (row, col) with maximum pixel value."""
    row_maxs = np.max(image, axis=0)
    row_max_id = np.argmax(row_maxs)
    col_maxs = np.max(image, axis=1)
    col_max_id = np.argmax(col_maxs)

    return (row_max_id, col_max_id)


def get_roi_rectangle(image_shape,
                      roi_width=0, roi_height=0, offcenter_xy=(0, 0),
                      coords_x=None, coords_y=None, extra_offset=(0, 0)):
    """Generate circular roi given center position and radius or coords_x/y.

    Parameters
    ----------
    image_shape : tuple of ints
        image (rows,columns)
    roi_width : int, optional
        width of ROI in pix
    roi_height : int, optional
        height of ROI in pix
    offcenter_xy : arraylike of floats, optional
        center of roi relative to center of image
    coords_x : tuple, optional
        if tuple start_x, end_x
    coords_y : tuple, optional
        if tuple start_y, end_y
    extra_offset : tuple, optional
        sshift to ROI for coords_x, coords_y. Default is (0,0)

    Returns
    -------
    roi : ndarray
        2d array with type 'bool'
    """
    inside = np.full(image_shape[0:2], False)
    start_y = None
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
    elif coords_x is not None and coords_y is not None:
        start_x = np.max([0, coords_x[0] + extra_offset[0]])
        start_y = np.max([0, coords_y[0] + extra_offset[1]])
        end_x = np.min([coords_x[1] + extra_offset[0], image_shape[1] - 1])
        end_y = np.min([coords_y[1] + extra_offset[1], image_shape[0] - 1])

    if start_y is not None:
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
    center_pos = [delta_xy[0] + image_shape[1] // 2,
                  delta_xy[1] + image_shape[0] // 2]

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


def get_roi_hom(image_info,
                paramset, delta_xya=[0, 0, 0.0], modality='CT'):
    """Calculate roi array with center roi and periferral rois.

    Parameters
    ----------
    image_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetXX
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
    roi_size_in_pix = paramset.hom_roi_size / image_info.pix[0]

    if modality in ['CT', 'PET']:
        roi_dist_in_pix = round(
            paramset.hom_roi_distance / image_info.pix[0])
        # central + 4 rois at 12, 15, 18, 21 o'clock optionally rotated
        rotation = delta_xya[2]
        if image_info.modality == 'CT':
            rotation = rotation + paramset.hom_roi_rotation
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
            roi_array.append(get_roi_circle(
                image_info.shape, tuple(off_centers[i]), roi_size_in_pix))

    elif modality == 'Xray':
        # central + 1 roi in each quadrant
        # optionally at specific distance and rotated

        rotation = delta_xya[2]
        rotation = rotation + paramset.hom_roi_rotation
        rotation_radians = np.deg2rad(rotation - 45)

        if rotation != 0 or paramset.hom_roi_distance > 0:

            center = [round(0.5 * image_info.shape[1]) + delta_xya[0],
                      round(0.5 * image_info.shape[0]) + delta_xya[1]]
            min_size = min([center[0], center[1],
                            image_info.shape[1] - center[0],
                            image_info.shape[0] - center[1]])
            if paramset.hom_roi_distance == 0:
                distance_pix = min_size * 0.5
                # centered in rotated quadrant with min dist to border
                # as reference distance
            else:
                distance_pix = round(
                    min_size * 0.01 * paramset.hom_roi_distance)
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
            dd_0 = round(image_info.shape[1]/4)
            dd_1 = round(image_info.shape[0]/4)

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
            roi_array.append(get_roi_circle(
                image_info.shape, tuple(off_centers[i]), roi_size_in_pix))
    else:
        roi_array = None

    return roi_array

def get_roi_hom_flatfield(image_info, paramset):
    """Calculate roi array when flatfield test Hom.

    Parameters
    ----------
    image_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetXX
        ParamSet for given modality as defined in config/config_classes.py

    Returns
    -------
    roi_all : list of np.array
    """
    roi_size_in_pix = paramset.hom_roi_size / image_info.pix[0]

    roi_small = get_roi_rectangle(
        image_info.shape, roi_width=roi_size_in_pix, roi_height=roi_size_in_pix)
    roi_var_in_pix = paramset.hom_roi_size_variance / image_info.pix[0]
    roi_variance = get_roi_rectangle(
        image_info.shape, roi_width=roi_var_in_pix, roi_height=roi_var_in_pix)
    roi_array = [roi_small, roi_variance]

    return roi_array

def get_roi_CTn_TTF(test, image, image_info, paramset, delta_xya=[0, 0, 0.]):
    """Calculate roi array with center roi and periferral rois.

    Parameters
    ----------
    test: str
        CTn or TTF
    image : np.ndarray
        pixeldata
    image_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetCT
        ParamSetCT as defined in config/config_classes.py
    delta_xya : list
        center and angle offset [center x, center y, rotation].
        Default is [0,-0, 0.0]

    Returns
    -------
    roi_all : list of np.array
        one 2d array for each material
        x 2 = first for actual ROI, next for search ROI (larger)
    errmsg
    """
    test = test.lower()
    roi_size_mm = getattr(paramset, f'{test}_roi_size', 0)
    roi_size_in_pix = round(roi_size_mm / image_info.pix[0])

    roi_array = None
    errmsg = None

    if roi_size_in_pix > 0:
        if test == 'ctn':
            if paramset.ctn_auto_center:
                res = mmcalc.optimize_center(image, 0)
                if res is not None:
                    center_x, center_y, _, _ = res
                    delta_xya[0] = center_x - 0.5*image_info.shape[1]
                    delta_xya[1] = center_y - 0.5*image_info.shape[0]
        filt_image = ndimage.gaussian_filter(image, sigma=5)
        pos_table = getattr(paramset, f'{test}_table')
        n_rois = len(pos_table.pos_x)
        off_centers = []
        rot_a = np.deg2rad(delta_xya[2])
        for r in range(n_rois):
            x = pos_table.pos_x[r] / image_info.pix[0]
            y = pos_table.pos_y[r] / image_info.pix[0]
            x += delta_xya[0]
            y += delta_xya[1]
            if rot_a != 0:
                x, y = mmcalc.rotate_point(
                    (x, y),
                    (delta_xya[0], delta_xya[1]),
                    delta_xya[2]
                    )
            off_centers.append([x, y])

        roi_search_array = []
        if test == 'ctn':
            if paramset.ctn_search:  # optimize off_centers
                search_size_in_pix = round(
                    paramset.ctn_search_size / image_info.pix[0])
                for r in range(n_rois):
                    roi_search_array.append(get_roi_circle(
                        image_info.shape, tuple(off_centers[r]),
                        search_size_in_pix))

                # adjust off_center by finding center of object within
                radius = search_size_in_pix
                cy = 0.5 * image_info.shape[0]
                cx = 0.5 * image_info.shape[1]
                outer_val_ring_mask = get_outer_ring(radius)
                n_err = 0
                for r in range(n_rois):
                    y = round(off_centers[r][1] + cy)
                    x = round(off_centers[r][0] + cx)
                    subarr = filt_image[y-radius:y+radius, x-radius:x+radius]
                    roi_mask = roi_search_array[r][y-radius:y+radius, x-radius:x+radius]
                    try:
                        background_arr = np.ma.masked_array(
                            subarr, mask=outer_val_ring_mask)
                        subarr[roi_mask == False] = np.mean(background_arr)
                    except np.ma.core.MaskError:
                        pass
                    size_y, size_x = subarr.shape
                    if size_y > 0 and size_x > 0:
                        prof_y = np.sum(subarr, axis=1)
                        prof_x = np.sum(subarr, axis=0)
                        # get width at halfmax and center for profiles
                        width_x, center_x = mmcalc.get_width_center_at_threshold(
                            prof_x, 0.5 * (max(prof_x) + min(prof_x)))
                        width_y, center_y = mmcalc.get_width_center_at_threshold(
                            prof_y, 0.5 * (max(prof_y) + min(prof_y)))
                        if center_y is not None:
                            off_centers[r][1] += center_y - radius
                        if center_x is not None:
                            off_centers[r][0] += center_x - radius
                        if center_y is None or center_x is None:
                            n_err += 1
                            txt = 'all' if n_err == n_rois else 'some'
                            errmsg = f'Failed finding center of object for {txt} ROIs.'

        roi_array = []
        for r in range(n_rois):
            roi_array.append(get_roi_circle(
                image_info.shape, tuple(off_centers[r]), roi_size_in_pix))

        if len(roi_search_array) == n_rois:
            roi_array.extend(roi_search_array)

    return (roi_array, errmsg)


def get_ratio_NM(image, image_info, mask_corner=0.0, ufov_ratio=0.95, cfov_ratio=0.75):
    """Calculate rectangular roi array for given ratio of UFOV, CFOV.

    First find non-zero part of image (ignoring padded part of image).
    Also ignore neighbour of zero counts
    (according to NEMA NU-1 about uniformity)

    Parameters
    ----------
    image : np.ndarray
        pixeldata
    image_info : DcmInfo
        as defined in scripts/dcm.py
    mask_corner : float
    ufov_ratio: float
    cfov_ratio: float

    Returns
    -------
    roi_array : list of np.array
        one 2d array for UFOV [0] and CFOV [1]
    errmsg
    """
    roi_array = []

    image_binary = np.zeros(image.shape)
    image_binary[image > 0.1] = 1  # 0.1 instead of 0 to ignore where very low values
    prof_y = np.max(image_binary, axis=1)
    width_y = np.count_nonzero(prof_y)
    prof_x = np.max(image_binary, axis=0)
    width_x = np.count_nonzero(prof_x)
    # avoid also neighbour of zero count pixels
    width_y = width_y - 2
    width_x = width_x - 2

    ufov = get_roi_rectangle(image.shape, roi_width=ufov_ratio * width_x,
                             roi_height=ufov_ratio * width_y)
    if mask_corner > 0:
        n_pix = round(mask_corner / image_info.pix[0])
        if n_pix > 0:
            rows = np.max(ufov, axis=1)
            cols = np.max(ufov, axis=0)
            sub = ufov[rows][:, cols]
            sz_y, sz_x = sub.shape
            corner = np.triu(np.array([1] * n_pix), 0)
            corner = np.array(corner, dtype=bool)
            sub[:n_pix, 0:n_pix] = np.flipud(corner)
            sub[:n_pix, sz_x-n_pix:] = corner.T
            sub[sz_y-n_pix:, :n_pix] = corner
            sub[sz_y-n_pix:, sz_x-n_pix:] = np.fliplr(corner)
            ufov[rows][:, cols] = sub
            rows = np.where(rows)
            cols = np.where(cols)
            ufov[rows[0][0]:rows[0][-1]+1, cols[0][0]:cols[0][-1]+1] = sub
    roi_array.append(ufov)
    roi_array.append(
        get_roi_rectangle(
            image.shape,
            roi_width=cfov_ratio * width_x,
            roi_height=cfov_ratio * width_y
            )
        )

    return roi_array


def generate_SNI_Siemens_image(SNI_values):
    """Generate image with SNI values displayed as circles corresponding to PMTs."""
    dist = 100
    image_shape = (dist * 6, dist * 8)
    dy = round(dist * np.sin(np.pi/3))  # hexagonal pattern
    radius = dist // 4
    dx7 = dist * (np.arange(7) - 3)
    dx6 = dist * (np.arange(6) - 2.5)
    roi_array = [
        [get_roi_circle(image_shape, (dx, -dy*2), radius) for dx in dx7],
        [get_roi_circle(image_shape, (dx, -dy), radius) for dx in dx6],
        [get_roi_circle(image_shape, (dx, 0), radius) for dx in dx7],
        [get_roi_circle(image_shape, (dx, dy), radius) for dx in dx6],
        [get_roi_circle(image_shape, (dx, dy*2), radius) for dx in dx7]
        ]
    image = np.zeros(image_shape)
    i = 0
    for roi_row in roi_array:
        for roi in roi_row:
            image[np.where(roi)] = SNI_values[i]
            i += 1

    return image


def get_roi_SNI(image, image_info, paramset, block_size=1):
    """Generate roi_array for NM SNI test.

    Parameters
    ----------
    image : nparr
    image_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetNM
        for given modality as defined in config/config_classes.py
    block_size : int
        used if merging pixels

    Returns
    -------
    roi : list of ndarray
        2d arrays with type 'bool'
        if sni_type == 0:
        2 large ROIs (left, rigth), 6 smaller ROIs left to right, top to bottom
        else
        list of all rois in grid
    errmsg
    """
    errmsg = None
    roi_full, not_used = get_ratio_NM(
        image, image_info, ufov_ratio=paramset.sni_area_ratio)
    roi_array = [roi_full]

    rows = np.max(roi_full, axis=1)
    width_y = np.count_nonzero(rows)
    cols = np.max(roi_full, axis=0)
    width_x = np.count_nonzero(cols)
    large_dim = min([width_x, width_y])
    small_dim = round(0.5*large_dim)

    idxs_col = np.where(cols == True)
    first_col = idxs_col[0][0]
    idxs_row = np.where(rows == True)
    first_row = idxs_row[0][0]

    # 2 large ROIs always
    if width_x > width_y:
        # large ROIs (2)
        left_large = np.full(roi_full.shape, False)
        left_large[rows, first_col:first_col + large_dim] = True
        roi_array.append(left_large)
        roi_array.append(np.fliplr(left_large))
    else:
        # large ROIs (2)
        first_large = np.full(roi_full.shape, False)
        first_large[first_row:first_row + large_dim, cols] = True
        roi_array.append(first_large)
        roi_array.append(np.flipud(first_large))

    if paramset.sni_type == 0:  # 6 small ROIs
        if width_x > width_y:
            upper_left = np.full(roi_full.shape, False)
            upper_left[
                first_row:first_row + small_dim, first_col:first_col + small_dim] = True
            roi_array.append(upper_left)
            upper_mid = np.full(roi_full.shape, False)
            s2 = round(small_dim/2)
            sz_y, sz_x = image.shape
            w2 = round(sz_x/2)
            first_mid = w2-1-s2
            upper_mid[
                first_row:first_row + small_dim, first_mid:first_mid + small_dim] = True
            roi_array.append(upper_mid)
            roi_array.append(np.fliplr(upper_left))
            roi_array.append(np.flipud(upper_left))
            roi_array.append(np.flipud(upper_mid))
            roi_array.append(np.flipud(np.fliplr(upper_left)))
        else:
            upper_left = np.full(roi_full.shape, False)
            upper_left[
                first_row:first_row + small_dim, first_col:first_col + small_dim] = True
            roi_array.append(upper_left)
            mid_left = np.full(roi_full.shape, False)
            s2 = round(small_dim/2)
            sz_y, sz_x = image.shape
            w2 = round(sz_y/2)
            first_mid = w2-1-s2
            mid_left[
                first_mid:first_mid + small_dim, first_col:first_col + small_dim] = True
            roi_array.append(mid_left)
            roi_array.append(np.flipud(upper_left))
            roi_array.append(np.fliplr(upper_left))
            roi_array.append(np.fliplr(mid_left))
            roi_array.append(np.flipud(np.fliplr(upper_left)))
    else:  # grid
        if paramset.sni_type == 1:
            roi_size = int(paramset.sni_roi_ratio * large_dim)
        else:
            roi_size = paramset.sni_roi_size // block_size
        if paramset.sni_type in [1, 2]:
            n_rois_x = width_x // (0.5 * roi_size) - 1
            n_rois_y = width_y // (0.5 * roi_size) - 1
            pos_x = width_x / (n_rois_x + 1) * np.arange(n_rois_x) + first_col
            pos_y = width_y / (n_rois_y + 1) * np.arange(n_rois_y) + first_row
            pos_x = [int(x) for x in pos_x]
            pos_y = [int(y) for y in pos_y]
            for j in pos_y:
                roi_row = []
                for i in pos_x:
                    roi_this = get_roi_rectangle(
                        image.shape,
                        coords_x=(i, i + roi_size),
                        coords_y=(j, j + roi_size))
                    roi_row.append(roi_this)
                roi_array.append(roi_row)
        else:  # sni_type 3 Siemens
            small_start_idx = 3
            dist = 76  # mm diameter PMTs = distance between centers
            dist_pix = round(dist / image_info.pix[0] / block_size)
            dy = round(dist_pix * np.sin(np.pi/3))  # hexagonal pattern
            radius = roi_size // 2
            dx7 = dist_pix * (np.arange(7) - 3)
            dx6 = np.round(dist_pix * (np.arange(6) - 2.5))
            roi_array.extend([
                [get_roi_circle(image.shape, (dx, -dy*2), radius) for dx in dx7],
                [get_roi_circle(image.shape, (dx, -dy), radius) for dx in dx6],
                [get_roi_circle(image.shape, (dx, 0), radius) for dx in dx7],
                [get_roi_circle(image.shape, (dx, dy), radius) for dx in dx6],
                [get_roi_circle(image.shape, (dx, dy*2), radius) for dx in dx7]
                ])

            dys = [-dy*2, -dy, 0, dy, dy*2]
            for rowno, roi_row in enumerate(roi_array[small_start_idx:]):
                dxs = dx7 if rowno % 2 == 0 else dx6
                for colno, roi in enumerate(roi_row):
                    roi_outside = np.copy(roi)
                    roi_outside[roi_full == True] = False
                    n_rows = np.sum(
                        roi_outside[:, int(dxs[colno] + image.shape[1] // 2)])
                    n_cols = np.sum(
                        roi_outside[int(dys[rowno] + image.shape[0] // 2)])
                    if n_rows + n_cols > 0:
                        if paramset.sni_roi_outside == 0:  # ignore
                            roi_array[rowno + small_start_idx][colno] = None
                        elif paramset.sni_roi_outside == 1:  # move
                            if n_cols < roi_size // 2 and n_rows < roi_size // 2:
                                xshift = n_cols
                                yshift = n_rows
                                if colno > len(roi_row) // 2:
                                    xshift = - xshift
                                if rowno < len(roi_array[small_start_idx:]) // 2:
                                    yshift = - yshift
                                roi_array[rowno + small_start_idx][colno] = get_roi_circle(
                                    image.shape,
                                    (dxs[colno] + xshift, dys[rowno] + yshift),
                                    radius)
                            else:  # ignore if > halv outside
                                roi_array[rowno + small_start_idx][colno] = None

    return (roi_array, errmsg)


def get_roi_NM_bar(image, image_info, paramset):
    """Generate circular rois for NM bar phantom tests.

    Circular ROI in each quadrant sorted by variance to get widest to narrowest bars.

    Parameters
    ----------
    image : nparr
    image_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetNM
        as defined in config/config_classes.py

    Returns
    -------
    roi_array : ndarray
        list of 4 2d arrays with type bool. Default is None
    errmsg : list of str. Default is []
    """
    roi_array = None

    vec_y = np.sum(image, axis=1)
    nozero = np.nonzero(vec_y)
    n_nozero = nozero[0].size
    dd = n_nozero // 4
    radius = paramset.bar_roi_size / image_info.pix[0]

    x_fac = [-1, 1, -1, 1]  # (left, right) * 2
    y_fac = [-1, -1, 1, 1]  # upper, upper, lower, lower
    variance = []
    roi_temp = []
    for i in range(4):
        center_xy = (x_fac[i]*dd, y_fac[i]*dd)
        roi_this = get_roi_circle(image_info.shape, center_xy, radius)
        arr = np.ma.masked_array(image, mask=np.invert(roi_this))
        variance.append(np.var(arr))
        roi_temp.append(roi_this)

    order_var = np.flip(np.argsort(np.array(variance)))
    roi_array = []
    for i in range(4):
        roi_array.append(roi_temp[order_var[i]])

    return roi_array


def get_roi_recovery_curve(summed_image, image_info, paramset):
    """Generate background and sphere rois based on image content.

    Parameters
    ----------
    summed_image : nparr
        sum of all selected images
    image_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetPET
        as defined in config/config_classes.py

    Returns
    -------
    roi_array : list of list of ndarray
        all background rois
        + last roi is found center marked by circular roi
    errmsg : str
        Default is None
    """
    errmsg = None

    # get mean of first background ROI
    roi_array = []
    radius = paramset.rec_roi_size / image_info.pix[0]
    assumed_bg_sum = 0.5 * np.max(summed_image)
    try:
        pos_x = paramset.rec_table.pos_x[0] / image_info.pix[0]
        pos_y = paramset.rec_table.pos_y[0] / image_info.pix[1]
        roi_first = get_roi_circle(
            summed_image.shape, (pos_x, pos_y), radius)
        arr = np.ma.masked_array(summed_image, mask=np.invert(roi_first))
        assumed_bg_sum = np.mean(arr)
    except (IndexError, AttributeError):  # if rec_table is None (maybe fixed)
        pass

    # search (lung insert in) center of image
    search_radius_mm = 50.
    roi_search_central_cylinder = get_roi_circle(
        summed_image.shape, (0, 0), search_radius_mm/image_info.pix[0])
    # allow for only a few images with the spheres
    high_vals = np.where(summed_image > assumed_bg_sum)
    topped_image = np.copy(summed_image)
    topped_image[high_vals] = assumed_bg_sum
    cx, cy = mmcalc.center_xy_of_disc(topped_image, roi=roi_search_central_cylinder)
    dx, dy = (0, 0)
    if cx is not None and cy is not None:
        dx, dy = (cx - summed_image.shape[1] // 2, cy - summed_image.shape[0] // 2)
    else:
        res = mmcalc.optimize_center(summed_image)  # center of object if no lung insert
        if res is not None:
            cx, cy, _, _ = res
            dx, dy = (cx - summed_image.shape[1] // 2, cy - summed_image.shape[0] // 2)

    # background ROIs with adjusted center
    try:
        for i in range(len(paramset.rec_table.pos_x)):
            pos_x = paramset.rec_table.pos_x[i] / image_info.pix[0]
            pos_y = paramset.rec_table.pos_y[i] / image_info.pix[1]
            roi_array.append(get_roi_circle(
                summed_image.shape, (dx + pos_x, dy + pos_y), radius))
    except AttributeError:  # if rec_table is None (maybe fixed)
        pass

    roi_array.append(get_roi_circle(
        summed_image.shape, (dx, dy), 5))

    return (roi_array, errmsg)


def get_roi_circle_MR(image, image_info, paramset, test_code, delta_xy):
    """Generate circular roi for MR tests.

    Circular ROI.
    Optionally optimized center based on maximum projection in x,y.
    Optionally cut top to avoid phantom structures on top (ACR phantom).

    Parameters
    ----------
    image : nparr
    image_info : DcmInfo
        as defined in scripts/dcm.py
    paramset : ParamSetMR
        for given modality as defined in config/config_classes.py
    test_code : str
    delta_xy : tuple
        image center offset in pix from mid_image (delta_x, delta_y)

    Returns
    -------
    roi : ndarray
        2d array with type bool
    errmsg : str or None
    """
    roi_array = None
    errmsg = None

    mask_outer = 0
    if test_code == 'Geo':
        mask_outer = round(paramset.geo_mask_outer / image_info.pix[0])

    res_center = mmcalc.optimize_center(image, mask_outer)
    if res_center is not None:
        center_x, center_y, width_x, width_y = res_center
    else:
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2
        width_x = 10
        width_y = 10

    radius = -1
    width = 0.5 * (width_x + width_y)
    optimize_center = True
    cut_top = 0
    if test_code == 'SNR':
        radius = 0.5 * width * np.sqrt(0.01 * paramset.snr_roi_percent)
        cut_top = paramset.snr_roi_cut_top
    elif test_code == 'PIU':
        radius = 0.5 * width * np.sqrt(0.01 * paramset.snr_roi_percent)
        cut_top = paramset.piu_roi_cut_top
    elif test_code == 'Gho':
        radius = paramset.gho_roi_central / image_info.pix[0]
        optimize_center = paramset.gho_optimize_center
        cut_top = paramset.gho_roi_cut_top
    elif test_code == 'Geo':
        radius = paramset.geo_actual_size / image_info.pix[0] / 2
    if optimize_center:
        delta_xy = (
            center_x - 0.5*image_info.shape[1],
            center_y - 0.5*image_info.shape[0]
            )
        if res_center is None:
            errmsg = ('Failed to find center and width of object in image.'
                      'Image center and widht 10 pix is used.')
    roi_array = get_roi_circle(image_info.shape, delta_xy, radius)
    if mask_outer > 0:
        inside = np.full(image_info.shape, False)
        inside[mask_outer:-mask_outer, mask_outer:-mask_outer] = True
        roi_array = [roi_array, inside]

    if cut_top > 0:
        cut_top = cut_top / image_info.pix[1]
        y_top = 0.5*image_info.shape[1] + delta_xy[1] - radius + cut_top
        y_start = 0
        y_stop = round(y_top)
        roi_array[y_start:y_stop, :] = False

    if test_code == 'Gho':  # add rectangular ROIs at borders
        # rois: 'Center', 'top', 'bottom', 'left', 'right'
        roi_array = [roi_array]
        roi_w = round(paramset.gho_roi_w / image_info.pix[0])
        roi_h = round(paramset.gho_roi_h / image_info.pix[0])
        roi_d = round(paramset.gho_roi_dist / image_info.pix[0])
        w = [roi_w, roi_w, roi_h, roi_h]
        h = w[::-1]
        delta = roi_h/2 + roi_d
        offxy = [
            (0, -(image_info.shape[1]/2 - delta)),
            (0, image_info.shape[1]/2 - delta),
            (-(image_info.shape[0]/2 - delta), 0),
            (image_info.shape[0]/2 - delta, 0)
            ]

        for i in range(4):
            roi_array.append(
                get_roi_rectangle(
                    image_info.shape,
                    roi_width=w[i], roi_height=h[i],
                    offcenter_xy=(round(offxy[i][0]), round(offxy[i][1]))
                    )
                )

    return (roi_array, errmsg)


def get_slicethickness_start_stop(image, image_info, paramset, dxya, modality='CT'):
    """Get start and stop coordinates for lines for slicethickness test CT.

    Parameters
    ----------
    image_info : DcmInfo
        as defined in scripts/dcm.py
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
    errmsg
    """
    errmsg = None
    size_xhalf = 0.5 * image_info.shape[1]
    size_yhalf = 0.5 * image_info.shape[0]
    prof_half = 0.5 * paramset.sli_ramp_length / image_info.pix[0]
    if modality == 'CT':
        if paramset.sli_auto_center:
            res = mmcalc.optimize_center(image, 0)
            if res is not None:
                center_x, center_y, _, _ = res
                dxya[0] = center_x - 0.5*image_info.shape[1]
                dxya[1] = center_y - 0.5*image_info.shape[0]
        dist_b = 25. / image_info.pix[0]  # beaded ramp Catphan inner ramps
        if paramset.sli_type == 1:
            dist = 45. / image_info.pix[0]  # beaded ramp Catphan outer ramps
        else:
            dist = paramset.sli_ramp_distance / image_info.pix[0]
    else:
        dist = [
            paramset.sli_dist_upper / image_info.pix[0],
            paramset.sli_dist_lower / image_info.pix[0]
            ]
        if paramset.sli_optimize_center:
            mask_outer = 10 / image_info.pix[0]
            res = mmcalc.optimize_center(image, mask_outer)
            if res is not None:
                center_x, center_y, _, _ = res
                dxya[0] = center_x - 0.5*image_info.shape[1]
                dxya[1] = center_y - 0.5*image_info.shape[0]

    # for each line: [y1, x1, y2, x2]
    h_lines = []
    v_lines = []
    rotation_radians = np.deg2rad(-dxya[2])
    cos_rot = np.cos(rotation_radians)
    sin_rot = np.sin(rotation_radians)

    center_x = size_xhalf + dxya[0]
    center_y = size_yhalf + dxya[1]

    # Horizontal profiles CT
    if modality == 'CT':
        if paramset.sli_type in [0, 1, 3, 4]:
            # all except only vertical (GE beaded)
            proceed = True
            if paramset.sli_type == 1 and paramset.sli_ignore_direction:
                proceed = False
            if proceed:
            # first horizontal line coordinates
                if dxya[2] == 0:
                    x1 = center_x - prof_half
                    x2 = center_x + prof_half
                    y1 = center_y - dist
                    y2 = y1
                else:
                    x1 = center_x - dist*sin_rot - prof_half*cos_rot
                    x2 = center_x - dist*sin_rot + prof_half*cos_rot
                    y1 = center_y - dist*cos_rot + prof_half*sin_rot
                    y2 = center_y - dist*cos_rot - prof_half*sin_rot
                h_lines.append([round(y1), round(x1),
                                round(y2), round(x2)])

                if paramset.sli_type in [0, 1, 4]:  # all with two horizontal
                    # second horizontal line coordinates
                    if dxya[2] == 0:
                        y1 = center_y + dist
                        y2 = y1
                    else:
                        x1 = center_x + dist*sin_rot - prof_half*cos_rot
                        x2 = center_x + dist*sin_rot + prof_half*cos_rot
                        y1 = center_y + dist*cos_rot + prof_half*sin_rot
                        y2 = center_y + dist*cos_rot - prof_half*sin_rot
                    h_lines.append([round(y1), round(x1),
                                    round(y2), round(x2)])

    else:
        for d in dist:
            if dxya[2] == 0:
                x1 = center_x - prof_half
                x2 = center_x + prof_half
                y1 = center_y - d
                y2 = y1
            else:
                x1 = center_x - d*sin_rot - prof_half*cos_rot
                x2 = center_x - d*sin_rot + prof_half*cos_rot
                y1 = center_y - d*cos_rot + prof_half*sin_rot
                y2 = center_y - d*cos_rot - prof_half*sin_rot
            h_lines.append([round(y1), round(x1), round(y2), round(x2)])

    if modality == 'CT':
        # Vertical profiles
        if paramset.sli_type == 1:  # Catphan beaded helical
            dists = [dist, dist_b]
        elif paramset.sli_type in [3, 4]:  # no vertical
            dists = []
        elif paramset.sli_type == 0 and paramset.sli_ignore_direction:
            dists = []
        else:
            dists = [dist]

        for dist in dists:
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

    return ({'h_lines': h_lines, 'v_lines': v_lines}, errmsg)


def find_rectangle_object(image, mask_outer=0):
    """Detect rectangle in image.

    Parameters
    ----------
    image : np.array
    mask_outer : int
        mask outer pixels and ignore signal there

    Returns
    -------
    dict:
        centers_of_edges_xy : list of list
            for each edge (top, right, btm, left) [x, y]
            longest/most central edge if not full rect in image
        corners_xy : list of list
            [toplft, toprgt, btmrgt, btmlft] [x, y]
            None if not 4 corners
    """
    centers_of_edges_xy = None
    corners_xy = None

    # find corners by thresholded matrix (min max found from inner quarter)
    x_h = image.shape[1] // 2
    x_q = x_h // 2
    y_h = image.shape[0] // 2
    y_q = y_h // 2
    inner_img = image[y_h - y_q:y_h + y_q, x_h - x_q:x_h + x_q]
    threshold = 0.5 * (np.min(inner_img) + np.max(inner_img))
    image_binary = np.zeros(image.shape)
    image_binary[image > threshold] = 1.
    inside = np.full(image.shape, False)
    if mask_outer > 0:
        inside[mask_outer:-mask_outer, mask_outer:-mask_outer] = True
        image_binary[inside == False] = 1.

    corn = feature.corner_peaks(
        feature.corner_fast(image_binary, 10),
        min_distance=10
        )

    if corn.shape[1] != 2 or corn.shape[0] != 4:  # try negative high signal
        if corn.shape[1] != 2:
            image_binary = np.logical_not(image_binary).astype(int)
        else:  # corn.shape[0] != 4
            image_binary = np.zeros(image.shape)
            image_binary[image < threshold] = 1.
            image_binary[inside == False] = 1.
        corn = feature.corner_peaks(
            feature.corner_fast(image_binary, 10),
            min_distance=10
            )

    if corn.shape[1] == 2:
        ys = np.array([c[0] for c in corn])
        xs = np.array([c[1] for c in corn])
        if corn.shape[0] == 4:
            # sort corners in toplft, toprgt, btmrgt, btmlft
            # first sort top to btm
            ys_sortidx = np.argsort(ys)
            ys_sort = ys[ys_sortidx]
            xs_sort = xs[ys_sortidx]
            # top lft to rgt
            if xs_sort[0] > xs_sort[1]:
                xs_sort = np.append(np.flip(xs_sort[:2]), xs_sort[2:])
                ys_sort = np.append(np.flip(ys_sort[:2]), ys_sort[2:])
            # btm rgt to lft
            if xs_sort[3] > xs_sort[2]:
                xs_sort = np.append(xs_sort[:2], np.flip(xs_sort[2:]))
                ys_sort = np.append(ys_sort[:2], np.flip(ys_sort[2:]))

            xs_diff = np.diff(np.append(xs_sort, xs_sort[0]))
            ys_diff = np.diff(np.append(ys_sort, ys_sort[0]))

            centers_of_edges_xy = [
                [xs_sort[i] + xs_diff[i] // 2, ys_sort[i] + ys_diff[i] // 2]
                for i in range(4)
                ]

            corners_xy = [[xs_sort[i], ys_sort[i]] for i in range(xs_sort.size)]
        else:
            # try fwhm method - assume square (for now)
            try:
                center_x, center_y, width_x, width_y = mmcalc.optimize_center(
                    image, mask_outer=0.05*image.shape[0])
                y0 = round(center_y)
                y1 = round(center_y + 0.1*width_y)
                center_y0_y1 = []
                for yval in [y0, y1]:
                    profile = image[yval]
                    _, center = mmcalc.get_width_center_at_threshold(
                        profile, 0.5*(np.max(profile) + np.min(profile)))
                    center_y0_y1.append(center)
                if None not in center_y0_y1:
                    tan_angle = (center_y0_y1[1] - center_y0_y1[0])/(y1-y0)
                    angle = np.arctan(tan_angle)
                    w_square = width_x * np.cos(angle)
                    centers_of_edges_xy = []
                    rot_angles = np.pi/2 * np.arange(4) - np.pi - angle
                    for i in rot_angles:
                        x, y = mmcalc.rotate_point(
                            [center_x + w_square//2, center_y],
                            [center_x, center_y], np.rad2deg(i))
                        centers_of_edges_xy.append([x, y])
            except TypeError:
                pass

    # TODO find edges also if full rectangle not imaged

    return {'centers_of_edges_xy': centers_of_edges_xy,
            'corners_xy': corners_xy}


def find_lines(image):
    """Detect 2 perpendicular lines in image and return center position of these.

    Parameters
    ----------
    image : np.array

    Returns
    -------
    centers_of_lines_xy : list of list
            for each line [center_x, center_y] relative to image center
    """
    prof_y = np.sum(image, axis=1)
    prof_x = np.sum(image, axis=0)

    _, center_ydir_x = mmcalc.get_width_center_at_threshold(
        prof_x, 0.9*(np.max(prof_x)))
    _, center_xdir_y = mmcalc.get_width_center_at_threshold(
        prof_y, 0.9*(np.max(prof_y)))
    _, center_xdir_x = mmcalc.get_width_center_at_threshold(
        prof_x, 0.1*(np.max(prof_x)))
    _, center_ydir_y = mmcalc.get_width_center_at_threshold(
        prof_y, 0.1*(np.max(prof_y)))

    xy_dir_xy = [center_xdir_x, center_xdir_y, center_ydir_x, center_ydir_y]
    if not all(xy_dir_xy):
        # try scipy.signal.find_peaks
        xy_dir_xy_peak = [None for i in range(4)]
        direction = 0
        for prof in [prof_x, prof_y]:
            peaks = find_peaks(prof, distance=10)
            peaks_pos = peaks[0]
            highest_peak_pos = None
            lowest_peak_pos = None
            if len(peaks_pos) >= 2:
                order_peaks = np.argsort(prof[peaks_pos])
                highest_peak_pos = peaks_pos[order_peaks[-1]]
                lowest_peak_pos = peaks_pos[order_peaks[-2]]
            if direction == 0:  # prof_x
                xy_dir_xy_peak[2] = highest_peak_pos  # xpos of ydir
                xy_dir_xy_peak[0] = lowest_peak_pos  # xpos of xdir
            else:  # prof_y
                xy_dir_xy_peak[1] = highest_peak_pos  # ypos of x dir
                xy_dir_xy_peak[3] = lowest_peak_pos  # ypos of y dir
            direction += 1

        # try combine (trust more center at threshold technique)
        xy_dir_xy_combine = []
        for i in range(4):
            if xy_dir_xy[i] is None:
                xy_dir_xy_combine.append(xy_dir_xy_peak[i])
            else:
                xy_dir_xy_combine.append(xy_dir_xy[i])
        xy_dir_xy = xy_dir_xy_combine

    if all(xy_dir_xy):
        center_line_ydir_xy = [
            xy_dir_xy[2] - image.shape[1]/2,
            xy_dir_xy[3] - image.shape[0]/2
            ]
        center_line_xdir_xy = [
            xy_dir_xy[0] - image.shape[1]/2,
            xy_dir_xy[1] - image.shape[0]/2
            ]
    else:
        center_line_xdir_xy = None
        center_line_ydir_xy = None

    return [center_line_xdir_xy, center_line_ydir_xy]
