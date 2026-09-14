# -*- coding: utf-8 -*-
"""
Collection of methods for phantom-specific calculations xray.

@author: ellen
"""
import numpy as np

from scipy import ndimage
from scipy.signal import find_peaks
from skimage import filters, draw, transform, measure, feature

# imageQC block start
from imageQC.scripts.calculate_roi import (
    get_roi_circle, get_roi_rectangle, find_rectangle_object_2,
    find_intercepts)
import imageQC.scripts.mini_methods_calculate as mmcalc
# imageQC block end


def calculate_phantom_xray(image, image_info, roi_array, paramset):

    if paramset.pha_type == 0:
        res = calculate_tor(image, image_info, roi_array, paramset)
    return res


def calculate_tor(image, image_info, roi_array, paramset):
    """Calculate low contrast and spatial resolution."""
    errmsgs = []
    values = []
    details_dict = {}
    lim_spat, lim_low_contrast = None, None

    if roi_array[0] is None:
        errmsgs = 'High contrast object within bar pattern not found.'
    else: # high contrast at bar pattern found
        pix = image_info.pix[0]
        sz_y, sz_x = image.shape
        invert = False  # indicator to invert image if highcontrast square higher values than background
        mask_coll = roi_array[2]

        cxys = roi_array[1]  # centers of high contrast square edges
        dx = cxys[2][0] - cxys[0][0]
        dy = cxys[2][1] - cxys[0][1]
        sz_square = round(np.sqrt(dx**2 + dy**2))
        cx, cy = np.mean(cxys, axis=0)
        cx, cy = round(cx), round(cy)
        min_ang = np.arctan(-dy/dx)

        # find which quadrant is where and flipping
        # profiles through high contrast square to find which quadrant is where
        ss25 = round(2.5 * sz_square)
        sub = image[cy - ss25:cy + ss25,cx - ss25:cx + ss25]
        var_map = mmcalc.get_variance_map(sub, round(2./pix), 'same')
        profiles_var = []
        profiles = []
        start = round(sz_square // 2)
        for i in range(2):
            xx = round(1.3*sz_square * np.cos(min_ang+i*np.pi/2))
            yy = - round(1.3*sz_square * np.sin(min_ang+i*np.pi/2))
            rr, cc = draw.line(ss25 - yy, ss25 - xx, ss25 + yy, ss25 + xx)
            profile = var_map[rr, cc]
            profiles_var.append(profile[-start:])
            profiles_var.append(profile[::-1][-start:])
            profile = sub[rr, cc]
            profiles.append(profile[-start:])
            profiles.append(profile[::-1][-start:])
        peak_pos = []
        max_diffs = []
        quadrant_bar_idxs = []
        for i, profile in enumerate(profiles_var):
            _, center = mmcalc.get_width_center_at_threshold(profile)
            peak_pos.append(center)
            max_diffs.append(np.max(profile) - np.min(profile))
            if center is None:
                quadrant_bar_idxs.append(i)
        if len(quadrant_bar_idxs) == 1:
            quadrant_bars = quadrant_bar_idxs[0]
        else:
            quadrant_bars = np.where(max_diffs == np.min(max_diffs))[0][0]
        peak_pos[quadrant_bars] = 1000
        quadrant_ref_square = np.where(peak_pos == np.min(peak_pos))[0][0]
        start_ref = ss25 - start + np.min(peak_pos)
        # invert?
        if sub[ss25,ss25] > profiles[quadrant_ref_square][-1]:
            invert = True
            image = - image
        idxs = [0, 2, 1, 3]  # order of profile directions
        quadrant_bars = idxs[quadrant_bars]
        quadrant_ref_square = idxs[quadrant_ref_square]
        #if quadrant_bars == np.roll([0,1,2,3], 1)[quadrant_ref_square]:
        #    flip = True

        # find background
        i = quadrant_ref_square
        xx = round(sz_square * 2.5 * np.cos(min_ang+i*np.pi/2))
        yy = - round(sz_square * 2.5 * np.sin(min_ang+i*np.pi/2))
        roi_ref_min = get_roi_circle(
            image.shape, (cx - sz_x//2 + xx, cy - sz_y//2 + yy),
            sz_square*0.3)
        arr = np.ma.masked_array(image, mask=np.invert(roi_ref_min))
        ref_min = np.ma.median(arr).item()
        var_min = np.std(arr) ** 2

        # find center of ref max square along profile
        rr, cc = draw.line(cy, cx, cy + yy, cx + xx)
        profile = image[rr, cc]
        start = round(1.2 * sz_square // 2)
        ref_max_part = profile[start:]
        _, center = mmcalc.get_width_center_at_threshold(ref_max_part)
        if center is not None:
            pos = round(center + start)
        else:
            pos = round(start_ref + sz_square / 2)
        cx_max, cy_max = cc[pos], rr[pos]
        roi_ref_square = get_roi_circle(
            image.shape, (cx_max - sz_x//2, cy_max - sz_y//2), sz_square*0.3)
        arr = np.ma.masked_array(image, mask=np.invert(roi_ref_square))
        ref_max = np.ma.median(arr).item()

        # find center of bar-pattern along 90 deg profile
        i = quadrant_bars
        xx = sz_square * np.cos(min_ang+i*np.pi/2)
        yy = - sz_square * np.sin(min_ang+i*np.pi/2)
        rr1, cc1 = draw.line(
            round(cy_max - 1.25*yy), round(cx_max - 1.25*xx),
            round(cy_max - 0.75*yy), round(cx_max - 0.75*xx))
        rr2, cc2 = draw.line(
            round(cy_max + 4.5*yy), round(cx_max + 4.5*xx),
            round(cy_max + 5.5*yy), round(cx_max + 5.5*xx))
        image_sobel = filters.sobel(image)
        profile1 = image_sobel[rr1, cc1]
        profile2 = image_sobel[rr2, cc2]
        peaks1 = find_peaks(
            profile1, distance=sz_square,
            height=0.5*(np.max(profile1)+np.min(profile1)))
        peaks2 = find_peaks(
            profile2, distance=sz_square,
            height=0.5*(np.max(profile2)+np.min(profile2)))
        proceed = True
        if peaks1[0].size == 0 or peaks2[0].size == 0:
            errmsgs.append('Failed locating borders of barpattern area.')
            proceed = False
        if proceed:
            side1_xy = cc1[peaks1[0][0]], rr1[peaks1[0][0]]
            side2_xy = cc2[peaks2[0][0]], rr2[peaks2[0][0]]
            dist = np.sqrt(
                (side1_xy[0]-side2_xy[0])**2 +
                (side1_xy[1]-side2_xy[1])**2)
            sz_square = dist / 5.7  # 57 mm bar pattern
            cx_bar = round(0.5*(side1_xy[0]+side2_xy[0]))
            cy_bar = round(0.5*(side1_xy[1]+side2_xy[1]))

            # finetune rotation of bars
            dd = round(sz_square)
            sub = image[cy_bar - dd:cy_bar + dd, cx_bar - dd:cx_bar + dd]
            theta = np.linspace(0.0, 180.0, 180, endpoint=False)
            sinogram = transform.radon(sub, theta=theta)
            profile_variances = np.var(sinogram, axis=0)
            dominant_angle_deg = theta[np.argmax(profile_variances)]

            # create template for bar pattern unit = cm = sz_square
            dx = [1.8, 0.6, -0.65]  # centers of 3 columns, sorted from high to low freq
            dy = [1.25, 0.3, -0.38, -0.9, -1.22, -1.47, -1.63]  # centers of 7 rows
            h = [0.6, 0.4, 0.3, 0.22, 0.20, 0.10, 0.05] # height of ROI each row
            idxs = []
            w = 0.8 * sz_square
            ss = round(sz_square * 4)
            temp_bars = np.zeros((ss*2 + 1, ss*2 + 1), dtype=np.uint8)
            idx = 1
            for i, y in enumerate(dy):
                for x in dx:
                    roi = get_roi_rectangle(
                        temp_bars.shape, roi_width=w,
                        roi_height=h[i] * dd,
                        offcenter_xy=(x * dd, y * dd))
                    temp_bars = temp_bars + roi * idx
                    idx += 1
            rot_temp_deg = dominant_angle_deg + 90# TODO check with different flipping/rotation for original
            roi = ndimage.rotate(temp_bars, rot_temp_deg,
                                 order=0, reshape=False)

            # get values from image
            sub = image[cy_bar - ss:cy_bar + ss + 1, cx_bar -ss:cx_bar + ss + 1]
            roi_full = np.zeros(image.shape)
            roi_full[cy_bar - ss:cy_bar + ss + 1, cx_bar -ss:cx_bar + ss + 1] = roi
            ctfs = []
            variances = []
            mins = []
            maxs = []
            amplitude = ref_max - ref_min
            mtf_factor = np.pi*np.sqrt(2)/(4*amplitude)
            for i in range(7*3):
                arr = np.ma.masked_array(sub, mask=roi != i+1)
                mina = np.min(arr)
                maxa = np.max(arr)
                mins.append(mina)
                maxs.append(maxa)
                ctfs.append((maxa - mina) / (maxa + mina))
                var_this = np.std(arr) ** 2
                variances.append(var_this)
            mtfs = mtf_factor * np.sqrt(np.array(variances) - var_min)

            roi_full[roi_ref_min == True] = True
            roi_full[roi_ref_square == True] = True

            details_dict = {
                'bar_pattern': {
                    'spatial_frequencies': [0.5, 0.56, 0.63,
                                            0.71, 0.8, 0.9,
                                            1., 1.12, 1.25,
                                            1.4, 1.6, 1.8,
                                            2., 2.24, 2.5,
                                            2.8, 3.15, 3.55,
                                            4., 4.5, 5],
                    'ctf_values': ctfs,
                    'variances': variances,
                    'amplitude': amplitude,
                    'variance_bg': var_min,
                    'mtf_values': mtfs,
                    'min_values':  mins,
                    'max_values': maxs,
                    'roi_ctfs': roi_full,
                    },
                'center_xy': (cx_bar, cy_bar),
                }

            mtf_50_10_2 = mmcalc.get_curve_values(
                details_dict['bar_pattern']['spatial_frequencies'],
                mtfs, [0.5, 0.1, 0.02])

            first_mtf_below_10perc = np.where(mtfs < 0.1)
            if np.min(mtfs) > 0.1:
                lim_spat = details_dict['bar_pattern']['spatial_frequencies'][-1]
            else:
                lim_spat = details_dict['bar_pattern']['spatial_frequencies'][
                    first_mtf_below_10perc[0][0]-1]

        # Contrast circles (if present)
        if mask_coll is not None:
            rows = np.max(mask_coll, axis=1)
            cols = np.max(mask_coll, axis=0)
            image_coll = image[rows][:, cols]
            sz_image_coll = image_coll.shape
            if np.min(sz_image_coll) < 2 * 7.*sz_square:
                proceed = False
        if proceed:
            # cnr circle positions in template
            angles = (np.arange(24) + 2) * (2*np.pi)/24  # 24 circles if all were there
            angles = angles + np.deg2rad(rot_temp_deg)
            # cnr pos, distance 60 mm relative to image center (shape//2)
            cx, cy = cx_bar, cy_bar
            dxys = [((6.*sz_square * np.sin(ang)),
                    (6.*sz_square * np.cos(ang)))
                   for i, ang in enumerate(angles)
                   if i not in [9, 10, 11, 21, 22, 23]]
            cnr_radii = np.array([0.25, .6, 0.8]) * sz_square # roi disc, inner/outer rim in mm

            details_dict.update({
                'contrast': {
                    'contrast_percents': [
                        16.7, 14.8, 12.8, 10.9, 8.8, 7.5, 6.7, 5.3, 4.5,
                        3.9, 3.2, 2.7, 2.2, 1.7, 1.5, 1.3, 1.1, 0.9],
                    'cnr_dxys': dxys,
                    'cnr_radii': cnr_radii,
                    }
                })

            ss = round(sz_square * 7)
            sub = image[cy_bar - ss:cy_bar + ss + 1, cx_bar -ss:cx_bar + ss + 1]
            # inner circles
            roi_circle_first = np.zeros(sub.shape, dtype=np.int8)
            means = []
            stds = []
            for i, dxy in enumerate(dxys):
                this_roi = get_roi_circle(sub.shape, dxy, cnr_radii[0])
                arr = np.ma.masked_array(sub, mask=np.invert(this_roi))
                means.append(np.mean(arr))
                stds.append(np.std(arr))
                if i == 0:
                    roi_circle_first = this_roi
            # background rims
            roi_bg_circles = np.zeros((len(dxys), ss*2+1,ss*2+1), dtype=bool)
            bg_means = []
            bg_stds = []
            for i, dxy in enumerate(dxys):
                outer = get_roi_circle(sub.shape, dxy, cnr_radii[2])
                inner = get_roi_circle(sub.shape, dxy, cnr_radii[1])
                this_roi = outer*np.invert(inner)
                arr = np.ma.masked_array(sub, mask=np.invert(this_roi))
                bg_means.append(np.mean(arr))
                bg_stds.append(np.std(arr))
                roi_bg_circles[i] = this_roi

            cnrs = []
            for mean, bg_mean, bg_std in zip(means, bg_means, bg_stds):
                cnrs.append(np.abs(mean - bg_mean)/bg_std)

            roi_cnrs = np.zeros(image.shape, dtype=bool)
            roi_cnrs[
                cy_bar - ss:cy_bar + ss + 1, cx_bar -ss:cx_bar + ss + 1] = (
                    roi_circle_first.astype(bool)
                    + roi_bg_circles[0].astype(bool))

            details_dict['contrast'].update({
                'cnr_values': cnrs,
                'disc_means': means,
                'disc_stds': stds,
                'background_means': bg_means,
                'background_stds': bg_stds,
                'roi_cnrs': roi_cnrs
                })
            first_disc_cnr_lower_than_1 = np.where(np.array(cnrs) < 1)[0][0]
            idx = first_disc_cnr_lower_than_1 - 1
            lim_low_contrast = details_dict[
                'contrast']['contrast_percents'][idx]
            details_dict['contrast'].update({
                'lim_idx': idx, 'lim_contrast': lim_low_contrast})

        values = [lim_spat, lim_low_contrast]
        values_sup = []

    return (details_dict, values, values_sup, errmsgs)
