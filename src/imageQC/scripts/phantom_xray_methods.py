# -*- coding: utf-8 -*-
"""
Collection of methods for phantom-specific calculations xray.

@author: ellen
"""
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve, find_peaks
from skimage.transform import hough_circle, hough_circle_peaks, resize
from skimage import feature, filters, draw


# imageQC block start
from imageQC.scripts.calculate_roi import (
    get_roi_circle, get_roi_rectangle, find_edges, find_intercepts)
import imageQC.scripts.mini_methods_calculate as mmcalc
# imageQC block end


def calculate_phantom_xray(image, image_info, roi_array, paramset):
    if paramset.pha_alt == 0:
        res = calculate_tor(image, image_info, roi_array, paramset)
    return res


def get_xy_from_angles_dist(ang_list, dist):
    """Calculate x and y from angle and distance to origin.

    Parameters
    ----------
    ang_list : arraylike of float
        angles in radians
    dist : float
        distance from center

    Returns
    -------
    cx : np.array of x positions relative to origin
    cy : same for y
    """
    tanang = np.tan(ang_list)
    cx = dist / np.sqrt(1 + tanang**2)
    neg_x = np.where(np.logical_and(
        ang_list > np.pi/2, ang_list < 3*np.pi/2))
    if neg_x.size > 0:
        cx[neg_x] = -cx[neg_x]
    cy = - cx * tanang
    pos_y = np.where(np.logical_and(
        ang_list > np.pi, ang_list < 2*np.pi))
    if pos_y.size > 0:
        breakpoint()
        cy[pos_y] = -cy[pos_y]
    return (cx, cy)


def find_cnr_from_sub(sub, radius_disc):
    """Locate disc and calculate avg/std of inner/outer signal to get CNR.

    Parameters
    ----------
    sub : np.array
        image cropped to disc + margin
    radius_disc : float
        expected radius of disc in sub

    Returns
    -------
    (cnr, inner_mean, inner_std, outer_mean, outer_std)
    """
    res = mmcalc.find_center_object(sub, sigma=3)
    if res:
        cx, cy, width_x, width_y = res
        if cx is not None and cy is not None:
            cx, cy = (cx - sub.shape[1] / 2, cy - sub.shape[0] / 2)
        else:
            cx, cy = (0, 0)
    else:
        cx, cy = (0, 0)

    dists = mmcalc.get_distance_map_point(
        sub.shape, center_dx=cx, center_dy=cy)
    inner = np.where(dists < radius_disc * 0.7)
    inner_mean = np.mean(sub[inner])
    inner_std = np.std(sub[inner])
    
    outer = np.where(np.logical_and(
        dists > radius_disc * 1.2, dists < radius_disc * 1.5))
    outer_mean = np.mean(sub[outer])
    outer_std = np.std(sub[outer])

    cnr = np.abs(inner_mean - outer_mean) / outer_std
    res = {
        'cnr': cnr, 'inner_mean': inner_mean, 'inner_std': inner_std,
        'outer_mean': outer_mean, 'outer_std': outer_std,
        'center_xy': np.array([cx, cy]) + sub.shape[0] // 2}
    return res


def calculate_tor(image, image_info, roi_array, paramset):
    # define linepairs and positions
    details_dict = {
        'spatial_frequencies': [0.5, 0.56, 0.63,
                                0.71, 0.8, 0.9,
                                1., 1.12, 1.25,
                                1.4, 1.6, 1.8,
                                2., 2.24, 2.5,
                                2.8, 3.15, 3.55,
                                4., 4.5, 5],
        'contrast_percents': [
            16.7, 14.8, 12.8, 10.9, 8.8, 7.5, 6.7, 5.3, 4.5,
            3.9, 3.2, 2.7, 2.2, 1.7, 1.5, 1.3, 1.1, 0.9]}

    errmsgs = None
    values = []
    pix = image_info.pix[0]

    # search for full phantom (circle)
    radius_large = 75./pix
    off_xy, radius = mmcalc.find_circle(
        image, radius_large,
        expected_radius_range=[0.9*radius_large, 1.1*radius_large],
        n_steps=9, downscale=5., edge_method='sobel', binary_threshold=0.1)

    edge_image = filters.sobel(image)

    # for bar pattern template:
    freq_0 = np.array([0.5, 0.71, 1., 1.4, 2., 2.8, 4.])
    widths_0 = 1./freq_0
    group_widths = 5.5*widths_0
    group_center_pos = np.cumsum(group_widths) - group_widths/2

    phantom_rot = 0  # rotation to square with highlight
    if radius:  # full phantom circle found
        # find contrast circle with highest contrast
        phantom_scale = radius / radius_large
        radius_small = 4./pix * phantom_scale  # contrast circles 8mm
        start, end = (phantom_scale / pix) * np.array([58, 62])
        # center of contrast circles 60mm
        roi_array = get_roi_circle(image.shape, off_xy, end)
        rows = np.max(roi_array, axis=1)
        cols = np.max(roi_array, axis=0)
        image_center = image[rows][:, cols]
        pol, (rads, angs) = mmcalc.topolar(image_center)
        rad1_idx = np.where(rads > start)[0][0]
        rad2_idx = np.where(rads > end)[0][0] - 1
        prof = np.mean(pol[rad1_idx:rad2_idx, :], axis=0)
        diff_prof = np.diff(prof)
        min_pos = np.where(diff_prof == np.min(diff_prof))
        max_pos = np.where(diff_prof == np.max(diff_prof))
        ang_min = angs[min_pos[0][0]]
        ang_max = angs[max_pos[0][0]]
        angles = None
        if np.abs(ang_max - ang_max) > 0.5:
            # split over zero i.e. rotation close to 0, not optimal
            # for now:
            errmsgs.append('Rotation close to zero. 45 degrees recommended.')
        else:
            zero_ang = np.mean([ang_max, ang_min])
            angles = np.pi/12 * np.arange(2, 11)
            # 15 degrees between circles

        if angles is not None:
            res_pr_disc = []
            roi_inners = []
            dist = 60 * (phantom_scale / pix)
            eval_angles = np.append(angles + zero_ang,
                                    np.flip(-angles + zero_ang))
            cx, cy = get_xy_from_angles_dist(eval_angles, dist)
            for i, ang in eval_angles:
                dx_dy = np.array([cx[i], cy[i]])
                roi = get_roi_circle(
                    image.shape, off_xy + dx_dy, radius_small*2)
                rows = np.max(roi, axis=1)
                cols = np.max(roi, axis=0)
                sub = image[rows][:, cols]
                res = find_cnr_from_sub(sub, radius_small)
                res['center_xy'] = res['center_xy'] + off_xy + dx_dy
                res_pr_disc.append(res)

        phantom_rot = zero_ang
    else:  # search for bar pattern at center of image
        # find center of rounded square
        ang = find_ang_object((0, 0), round(paramset.pha_roi_mm/pix))
        
        def find_ang_object(c_xy, radius_test_range):
            roi_array = get_roi_circle(image.shape, c_xy, radius_test_range[1])
            rows = np.max(roi_array, axis=1)
            cols = np.max(roi_array, axis=0)
            roi_inner = get_roi_circle(image.shape, c_xy, radius_test_range[0])
            edge_center = (edge_image*roi_array)[rows][:, cols]
            edge_center[roi_inner[rows][:, cols] == True] = 0
            pol, (rads, angs) = mmcalc.topolar(edge_center)
            first_row = []
            pol_rot = np.flipud(np.rot90(pol))
            for i in range(pol.shape[1]):
                above = np.where(pol_rot[i] > 0.5*np.max(pol_rot))
                if len(above[0]) > 0:
                    first = above[0][0]
                else:
                    first = pol.shape[0]
                first_row.append(first)
            ang_idx = np.where(first_row == np.min(first_row))
            ang = angs[round(np.mean(ang_idx))]
            return ang, np.min(first_row)
        
        max_ang, max_dist = find_ang_dist_object(
            off_xy, [0, round(paramset.pha_roi_mm/pix)])
        tanang = np.tan(ang)
        cx = max_dist / np.sqrt(1 + tanang**2)
        if max_ang > np.pi/2 and max_ang < 3*np.pi/2:
                cx = - cx
        cy = - cx * tanang

        # find rotation of rounded square
        dy, dx = 0.5 * np.array(edge_center.shape)
        dy, dx = int(dy + cy), int(dx + cx)
        mm = int(15./pix)
        edge_cut = edge_center[dy - mm:dy + mm, dx - mm:dx + mm]
        pol, (rads, angs) = mmcalc.topolar(edge_cut)
        # find edges closest to center of rounded square = normal to edges
        first = [np.where(prof > 0.5*np.max(prof))[0][0]
                 for prof in np.rot90(pol[0:mm])]
        peaks = find_peaks(first, distance=angs.size/5)  # angles where corners
        deg_from_cardinal = np.mean(np.rad2deg(angs[peaks[0]]) % 90) - 45

        # find direction to bar_pattern
        variance_center = mmcalc.get_variance_map(
            image_center, round(2./pix), 'same')
        dist_30mm = 30./pix
        variance_at_15mm = []
        profiles = []
        for i in range(4):
            x2, y2 = mmcalc.rotate_point(
                (dx + dist_30mm, dy), (dx, dy), deg_from_cardinal + 90*i)
            rr, cc = draw.line(dy, dx, int(np.round(y2)), int(np.round(x2)))
            try:
                profile = variance_center[rr, cc]
                profiles.append(profile)
                variance_at_15mm.append(
                    variance_center[rr[len(rr)//2], cc[len(cc)//2]])
            except IndexError:
                profiles.append(None)
                variance_at_15mm.append(0)
        max_idx = np.where(variance_at_15mm == np.max(variance_at_15mm))
        freq_0 = np.array([0.5, 0.71, 1., 1.4, 2., 2.8, 4.])
        widths_0 = 1./freq_0
        group_widths = 5.5*widths_0
        group_center_pos = np.cumsum(group_widths) - group_widths/2
        start = int(
            len(profiles[max_idx[0][0]])
            * 1.3 * 0.5 * group_widths[0] / 30)
        prof1 = profiles[max_idx[0][0] - 1][start:]
        prof2 = profiles[(max_idx[0][0] + 1) % 4][start:]
        peaks1 = find_peaks(prof1, distance=group_widths[0]/2/pix)
        peaks2 = find_peaks(prof2, distance=group_widths[0]/2/pix)
        if peaks1[0][0] < peaks2[0][0]:
            print('Test probably fails, try flipping left/right')
        breakpoint()

    return (details_dict, values, errmsgs)
