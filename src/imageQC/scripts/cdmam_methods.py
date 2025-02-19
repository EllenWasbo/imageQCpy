# -*- coding: utf-8 -*-
"""
Collection of methods for SNI and uniformity calculations
 for gamma camera in ImageQC.

@author: ellen
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import (find_peaks, fftconvolve, medfilt2d)
from skimage.transform import hough_line, hough_line_peaks, resize
from skimage import draw

# imageQC block start
import imageQC.scripts.mini_methods_calculate as mmcalc
# imageQC block end


def fix_cropped_sub(image, x, y, wi):
    """Restore to full sub size if sub partly cropped."""
    y1, y2 = y - wi, y + wi
    x1, x2 = x - wi, x + wi
    sub_sz = np.max([y2 - y1, x2 - x1])
    sub = np.zeros((sub_sz, sub_sz))
    if y1 < 0:
        sub[-y1:,:] = image[0:y2, x1:x2]
        sub[0:-y1,:] = np.median(sub)
    elif y2 >= image.shape[0]:
        cropped = image[y1:, x1:x2]
        sub[0:cropped.shape[0], :] = cropped
        sub[cropped.shape[1]:, :] = np.median(cropped)
    elif x1 < 0:
        sub[:,-x1:] = image[y1:y2, 0:x2]
        sub[:,0:-x1] = np.median(sub)
    elif x2 >= image.shape[1]:
        cropped = image[y1:y2, x1:]
        sub[:, 0:cropped.shape[1]] = cropped
        sub[:, cropped.shape[1]:] = np.median(cropped)

    return sub


def detrend_sub(sub, sigma_smooth, median_filter_width):
    """Median filter and flatten matrix, keeping most details. """
    smooth = ndimage.gaussian_filter(sub, sigma=sigma_smooth)
    if median_filter_width > 0:
        sub = medfilt2d(sub, median_filter_width)
    detrend_factors = smooth / np.median(sub)
    return sub / detrend_factors


def get_binary_sub(sub, percentile=25):
    """Get binary of matrix setting threshold to specified percentile value."""
    binary = np.zeros(sub.shape, dtype=bool)
    threshold = np.percentile(sub, percentile)
    binary[sub < threshold] = True
    return binary


def find_angles_sub(binary_sub, angle_range, margin, threshold_peaks):
    """Find lines in binary 2d array with Hough transform.

    Parameters
    ----------
    binary_sub : np.2darray of bool
    angle_range : list of float
        [min, max] angle (radians)
    margin : float

    Returns
    -------
    angles: list of angles found (radians)
    dists: list of distances (from origo) found
    """
    angle_resolution = np.deg2rad(0.5)
    n_angles = np.ceil(np.diff(angle_range)[0] / angle_resolution)
    n_angles = max(n_angles, 5)
    tested_angles = np.linspace(
        angle_range[0], angle_range[1],
        round(n_angles), endpoint=False)
    hspace, theta, d = hough_line(binary_sub, theta=tested_angles)
    threshold_peaks = threshold_peaks * np.max(hspace)
    _, angles, dists = hough_line_peaks(hspace, theta, d,
                                        threshold=threshold_peaks,
                                        min_distance=margin)
    return (angles, dists)


def get_min_pos_2d(image, roi_array):
    arr = np.copy(image)
    arr[roi_array == False] = np.max(arr)
    min_idx = np.where(arr == np.min(arr))

    return (min_idx[0][0], min_idx[1][0])

def get_lines(angles_and_dists, x_start_end, y_start):
    """Find line expressions from (angle, dist) pairs.

    Parameters
    ----------
    angles_and_dists : list of tuple
        tuple = (angle, dist). as output from find_angles_sub
    x_start_end : list of int
        sub_binary position in larger image
    y_start : int
        sub_binary position in larger image

    Returns
    -------
    lines : list of tuples
        tuple is ((x0, y0), slope a, intercept b) to define y=ax+b
    """
    lines = []

    for angle, dist in angles_and_dists:
        x0, y0 = dist * np.array([np.cos(angle), np.sin(angle)])
        x, y = (x0 + x_start_end[0], y0 + y_start)
        slope = np.tan(angle + np.pi / 2)
        intercept = y - slope * x
        line = [(x, y), slope, intercept]
        lines.append(line)
    return lines


def get_line_intersections(lines):
    """Find coordinates where input lines intersect.

    Parameters
    ----------
    lines: list of list of tuple
        list of output from get_lines()

    Returns
    -------
    intercept_xys_pr_row = lit of list of tuples
        tuple of (x, y) for the intersections, sorted by x
        grouped by row (similar y values), sorted by y
    dx : float
        median distance between intersections in x-direction
    """
    xys = []
    for line1 in lines[0]:
        for line2 in lines[1]:
            _, a1, b1 = line1
            _, a2, b2 = line2
            x = (b2 - b1) / (a1 - a2)
            y = a1 * x + b1
            xys.append((x, y))
    # sort by y
    xys = sorted(xys, key=lambda xy: xy[1])
    y_diffs = np.diff([xy[1] for xy in xys])
    halfmax = 0.5 * (np.max(y_diffs) - np.min(y_diffs))
    large_shifts = np.zeros(y_diffs.shape, dtype=bool)
    large_shifts[y_diffs > halfmax] = True
    idxs = list(np.where(large_shifts == True)[0] + 1)
    idxs.insert(0, 0)
    dx_all = []  # differenced within row of intercept points

    xys_pr_row = []
    for i in range(len(idxs)):
        try:
            xys_in_row = sorted(
                xys[idxs[i]:idxs[i+1]], key=lambda xy: xy[0])
            xys_pr_row.append(xys_in_row)
            xs_this = [xy[0] for xy in xys_in_row]
            dx_all.extend(list(np.diff(xs_this)))
        except IndexError:  # last single point
            xys_pr_row.append([xys[idxs[i]]])
    dx = np.median(dx_all)

    return (xys_pr_row, dx)


def find_phantom_part(image, margin, axis=0, pos_profile=None):
    """"Detect phantom in image along axis."""
    mid = image.shape[axis] // 2
    npix_mean = 50
    if axis == 0:  # x prof
        prof = np.mean(image[mid-npix_mean:mid+npix_mean, :], axis=axis)
    else:  # y prof
        prof = np.mean(image[:, mid-npix_mean:mid+npix_mean], axis=axis)
    in_phantom = np.where(np.logical_and(
        prof < np.median(prof)*1.1,
        prof > np.median(prof)*0.9))
    start, end = in_phantom[0][0], in_phantom[0][-1]
    start = start + margin
    end = end - margin
    sz_y, sz_x = image.shape
    if axis == 0:
        if pos_profile is None:
            pos_profile = sz_y // 2
        prof = np.mean(
            image[pos_profile - 10: pos_profile + 10, start:end], axis=0)
    else:
        if pos_profile is None:
            pos_profile = sz_x // 2
        prof = np.median(
            image[start:end, pos_profile - 10: pos_profile + 10], axis=1)
    smoothed = ndimage.gaussian_filter1d(prof, 7)
    detrended_profile = prof / (smoothed / np.median(prof))
    minval = np.min(detrended_profile)
    maxval = np.max(detrended_profile)
    medianval = np.median(detrended_profile)
    if medianval - minval > maxval - medianval:  # invert
        detrended_profile = - detrended_profile
    profile = detrended_profile - np.median(detrended_profile)

    return (start, end, profile)


def find_cdmam(image, image_info, paramset):
    """Find phantom in image, ignoring signal outside phantom."""
    errmsgs = []
    px_pr_mm = round(1./image_info.pix[0])
    margin = 5 * px_pr_mm
    margin_angle = paramset.cdm_tolerance_angle * 2 * np.pi / 360
    sz_y, sz_x = image.shape
    phantom = 0
    roi_dict = None
    invert = False  # indicator to invert image if grid/dots have higher values than background

    # taste inner part to find if and which CDMAM phantom
    xs = [sz_x//2 - sz_x//8, sz_x//2 + sz_x//8]
    ys = [sz_y//2 - sz_y//8, sz_y//2 + sz_y//8]
    center_part = image[ys[0]:ys[1], xs[0]:xs[1]]
    center_part = detrend_sub(center_part, 5*px_pr_mm, 3)
    if (np.median(center_part) - np.min(center_part)
            < np.max(center_part) - np.median(center_part)):
        image = - image
        center_part = - center_part
        invert = True
    center_binary = get_binary_sub(center_part)
    angles, dists = find_angles_sub(
        center_binary, [0 - margin_angle, np.pi / 4 + margin_angle],
        7*px_pr_mm, paramset.cdm_threshold_peaks)
    angles_dists_center = None
    if angles.shape[0] >= 2:
        ang_45 = np.where(np.logical_and(
            angles > np.pi / 4 - margin_angle,
            angles < np.pi / 4 + margin_angle))
        ang_0 = np.where(np.logical_and(
            angles > - margin_angle,
            angles < margin_angle))
        if ang_45[0].shape[0] > ang_0[0].shape[0]:
            phantom = 34
            angles_dists_center = [
                (angle, dists[idx]) for idx, angle in enumerate(angles)
                if idx in ang_45[0]]
        elif ang_0[0].shape[0] > 2:
            phantom = 40
            angles_dists_center = [
                (angle, dists[idx]) for idx, angle in enumerate(angles)
                if idx in ang_0[0]]

    if phantom == 0:
        errmsgs.append(
            'Failed to detect horizontal or diagonal lines.')
    elif phantom == 34:
        roi_dict = find_phantom_34(
            image, margin, margin_angle, angles_dists_center, paramset)
        roi_dict['invert'] = invert
    elif phantom == 40:
        roi_dict = find_phantom_40(
            image, px_pr_mm, margin_angle, angles_dists_center, paramset)
        roi_dict['invert'] = invert

    return (roi_dict, errmsgs)


def find_phantom_34(image, margin, margin_angle, angles_dists_center, paramset):
    xy_refs = [[], []]
    dx_intersections = []
    xs, ys = [[], []]
    sz_y, sz_x = image.shape
    errmsgs = []

    if paramset.cdm_rotate_k != 0:
        image = np.rot90(image, k=paramset.cdm_rotate_k)

    # search corner of phantom at top mid and btm mid part
    x_start, x_end, x_prof = find_phantom_part(image, margin)
    y_start, y_end, y_prof = find_phantom_part(image, 0, axis=1)
    width = x_end - x_start
    ys = [y_start, sz_y // 4, y_end - sz_y // 4, y_end]  # top
    xs = [x_start + width // 4, x_end - width // 4]  # mid
    subs = [
        image[ys[0]:ys[1], xs[0]:xs[1]],
        image[ys[2]:ys[3], xs[0]:xs[1]]]
    lines = [[], []]  # [(x0, y0), slope] for two groups (perpendicular)

    for sub_no, sub in enumerate(subs):
        # find lines
        sub = detrend_sub(sub, margin, 3)
        sub_binary = get_binary_sub(sub)
        y0 = ys[0] if sub_no == 0 else ys[2]
        lines = []
        for angle in [np.pi / 4, - np.pi / 4]:
            test_range = [angle - margin_angle, angle + margin_angle]
            angles, dists = find_angles_sub(
                sub_binary, test_range, margin,
                paramset.cdm_threshold_peaks)
            lines_this = get_lines([*zip(angles, dists)], xs, y0)
            lines.append(lines_this)
        intercept_xys_pr_row, dx = get_line_intersections(lines)

        if sub_no == 0:
            xy_refs[0] = intercept_xys_pr_row[0][0]
        elif sub_no == 1:
            xy_refs[1] = intercept_xys_pr_row[-1][0]
        dx_intersections.append(dx)

    ref_dx = xy_refs[1][0] - xy_refs[0][0]
    ref_dy = xy_refs[1][1] - xy_refs[0][1]
    diff_angle = np.arctan(ref_dx / ref_dy)

    angles_center = [ac[0] for ac in angles_dists_center]
    diff_angle_center = np.median(angles_center) - np.pi / 4
    if np.abs(diff_angle - diff_angle_center) > margin_angle / 2:
        errmsgs.append(['Verify found center of cells. Seems inaccurate.'])

    angle = np.median(angles_center)
    diagonal = np.median(dx_intersections)
    dist = np.cos(angle) * diagonal

    include_array = np.zeros((16, 16), dtype=bool)
    for row in range(16):
        for col in range(16):
            if row + col > 5:
                include_array[row, col] = True
                if row + col > 23:
                    include_array[row, col] = False
    include_array[0, -1] = False
    include_array[-1, 0] = False

    ny, nx = (16, 16)
    xs = np.zeros((16, 16), dtype=np.float64)
    ys = np.zeros((16, 16), dtype=np.float64)
    for row, y in enumerate(dist * np.arange(ny) + dist / 2):
        for col, x in enumerate(dist * np.arange(nx) + dist / 2):
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x) - (np.pi/4 + diff_angle)
            xs[row, col] = r * np.cos(phi)
            ys[row, col] = r * np.sin(phi)

    center_x = 0.5 * (xy_refs[0][0] + xy_refs[1][0])
    center_y = 0.5 * (xy_refs[0][1] + xy_refs[1][1])
    dx = center_x - xs[8, 8] + diagonal / 2
    dy = center_y - ys[8, 8]
    xs = xs + dx
    ys = ys + dy

    if paramset.cdm_rotate_k > 0:
        if paramset.cdm_rotate_k == 2:
            xs = image.shape[1] - xs
            ys = image.shape[0] - ys
        if paramset.cdm_rotate_k in [1, 3]:
            xxs = np.copy(xs)
            xs = np.copy(ys)
            ys = xxs
            if paramset.cdm_rotate_k == 3:
                #xs = image.shape[0] - xs
                ys = image.shape[1] - ys
                xs = np.flip(xs)
                ys = np.flip(ys)
            else:
                xs = image.shape[0] - xs
                xs = np.flip(xs)
                ys = np.flip(ys)

    roi_dict = {'include_array': include_array, 'phantom': 34,
                'xs': xs, 'ys':ys,
                'cell_width': diagonal, 'angle': angle}
    return roi_dict


def find_phantom_40(image, px_pr_mm, angle_margin, angles_center, paramset):
    dist = None
    errmsgs = []
    roi_estimate = paramset.cdm_roi_estimate
    estimate_x, estimate_y = roi_estimate, roi_estimate

    # finding vertical line positions (long axis)
    axis = 0 if paramset.cdm_rotate_k in [0, 2] else 1
    x_start_phantom, x_end_phantom, xprof = find_phantom_part(
        image, 5*px_pr_mm, axis=axis)
    if estimate_x is False:
        peaks = find_peaks(xprof, distance=7*px_pr_mm,
                           height=0.5*np.max(xprof))
        peaks_pos = peaks[0]

        if len(peaks_pos) == 17:
            x_start = peaks_pos[0] + x_start_phantom
            dist = (peaks_pos[-1] - peaks_pos[0]) / 16
        else:
            errmsgs.append(
                'Failed to find a grid 16 cells wide. x-positions '
                'centered on assumed phantom.')
            estimate_x = True

    x_off = 0.25 * (x_end_phantom - x_start_phantom) + x_start_phantom
    axis = 1 if paramset.cdm_rotate_k in [0, 2] else 0
    y_start_phantom, y_end_phantom, yprof = find_phantom_part(
        image, 5*px_pr_mm, axis=axis, pos_profile=round(x_off))

    if estimate_y is False:
        peaks = find_peaks(yprof, distance=3.5*px_pr_mm,
                           height=0.5*np.max(yprof))
        peaks_pos = peaks[0]
        dists = np.diff(peaks_pos)
        dist = np.median(dists)
        idx_sep = np.where(
            np.logical_and(dists < 1.05 * dist / 2, dists > 0.95 * dist / 2))

        if len(idx_sep[0]) == 3:
            y_start = peaks_pos[0] + y_start_phantom
        else:
            errmsgs.append(
                'Failed to find a grid 21 cells high. '
                'y-positions centered on assumed phantom.')
            estimate_y = True

    if dist is None:
        dist = 9 * px_pr_mm
    if estimate_x:
        mid_x = 0.5 * (x_end_phantom + x_start_phantom)
        x_start = mid_x - 8 * dist
    if estimate_y:
        mid_y = 0.5 * (y_end_phantom + y_start_phantom)
        y_start = mid_y - 22.5/2 * dist

    if dist and y_start:
        if paramset.cdm_rotate_k in [2, 3]:
            a, b, c = 3, 7, 11
        else:
            a, b, c = 10, 14, 18

        xs, ys = np.meshgrid(
            dist * (np.arange(16) + 0.5),
            dist * (np.arange(21) + 0.5))
        ys[a:, :] = ys[a:, :] + 0.5 * dist
        ys[b:, :] = ys[b:, :] + 0.5 * dist
        ys[c:, :] = ys[c:, :] + 0.5 * dist
        # center before rotate
        xs = xs - 8 * dist
        ys = ys - 22.5/2 * dist

        if paramset.cdm_rotate_k in [2, 3]:
            xs = np.rot90(xs, k=2)
            ys = np.rot90(ys, k=2)
        if paramset.cdm_rotate_k in [1, 3]:  # landscape
            xxs = np.copy(ys)
            ys = np.flip(xs)
            xs = xxs

        diff_angle = np.median(
            [angle for angle, dist in angles_center])

        for row in range(21):
            for col in range(16):
                x, y = xs[row, col], ys[row, col]
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x) + diff_angle
                xs[row, col] = r * np.cos(phi)
                ys[row, col] = r * np.sin(phi)

        if paramset.cdm_rotate_k in [0, 2]:  # portrait
            xs = xs + 16/2 * dist + x_start
            ys = ys + 22.5/2 * dist + y_start
        else:
            xs = xs + 22.5/2 * dist + y_start
            ys = ys + 16/2 * dist + x_start

        # finetune centering based on a central cell
        xx, yy = round(xs[8][8]), round(ys[8][8])
        wi = round(1.4 * dist / 2)
        sub = image[yy - wi:yy + wi, xx - wi:xx + wi]
        dx, dy = finetune_center_cell(sub, 40, None)
        xs = xs + dx
        ys = ys + dy

        # flip down to horizontal view as thickness list in yaml
        xs = np.rot90(xs, 1)
        ys = np.rot90(ys, 1)

        roi_dict = {'phantom': 40, 'xs': xs, 'ys':ys,
                    'cell_width': dist, 'angle': diff_angle}

    return roi_dict


def get_roi_circle(shape, delta_xy, radius):
    """Generate circular roi given center position and radius.

    Duplicate from calculate_rois, avoiding circle reference.
    """
    xs, ys = np.meshgrid(
        np.arange(0, shape[1], dtype=int),
        np.arange(0, shape[0], dtype=int),
        sparse=True)
    center_pos = [delta_xy[0] + shape[1] // 2,
                  delta_xy[1] + shape[0] // 2]

    zs = np.sqrt((xs-center_pos[0]) ** 2 + (ys-center_pos[1]) ** 2)
    inside = zs <= radius

    return inside


def find_offset_disc(sub, kernel, search_radius, xy_expected=(0, 0)):
    """Search offset of center and optionally corner disc compared to expected.

    Parameters
    ----------
    sub : np.ndarray
        sub resized and finetuned to center
    kernel : np.ndarray
        kernel for calculating averages
    search_radius : int
        roi_radius to search for disc
    xy_expected : tuple, optional
        x, y expected offset from center. The default is (0, 0) = center.

    Returns
    -------
    off_xy : tuple
    """
    cent = round(sub.shape[0]/2)
    avgs_sub = fftconvolve(sub, kernel, mode='same')
    roi = get_roi_circle(avgs_sub.shape, xy_expected, search_radius)
    #min_yx, _ = mmcalc.get_min_max_pos_2d(avgs_sub, roi)
    min_yx = get_min_pos_2d(avgs_sub, roi)
    off_xy = (min_yx[1] - cent, min_yx[0] - cent)

    return off_xy


def get_templates(matrix, roi_dicts, pix, pix_new,
                  rot90_k, tables, radius_search):
    """Find xy-offset of discs from sample cell with good visibility.

    Create templates where to search for minimum values.
    With more than three images fewer cells are used as samples.
    """
    cell_width = roi_dicts[0]['cell_width']
    phantom = roi_dicts[0]['phantom']

    if phantom == 34:
        line_dist = (cell_width * np.cos(np.pi / 4)**2) / 2
    else:
        line_dist = cell_width / 2
    line_dist = round((pix / pix_new) * line_dist)

    wi = round(cell_width / 2)
    wi_factor = 1
    if phantom == 34:
        width_sub = wi * 2 + 1
    else:  # v4.0
        wi_factor = 1.3
        wi = round(wi_factor * wi)  # add margin to cell
        width_sub = wi * 2 + 1
    width_sub_50um = round(width_sub * pix / pix_new)

    corner_indexes = np.array(tables['corner_index'])

    is_used = [image is not None for image in matrix]
    n_images = np.sum(is_used)

    pr_image = False if n_images > 1 else True

    corner_rows_cols = [[], [], [], []]  # for the 4 corners
    if phantom == 34:
        radius_kernel = 15
        diff_expect = width_sub_50um // 4
        off_xy_expect = [(0, -1), (-1, 0), (1, 0), (0, 1)]  # top,lft,rgt,btm
        # when seen with no rotation
        # exact position seems related to which corner, not rotation symmetric
        # estimate position from 2 first cells of rows with thickest discs
        for row in range(4):  # find indexes and sort into which corner
            # corner index = 0 means not used cell for v3.4
            first_cols = np.where(corner_indexes[row] > 0)[0][:2]
            for col in first_cols:
                corner_rows_cols[
                    corner_indexes[row][col]-1].append((row, col))
        #[[(0,6), (3,3)], [(0,7), (1,5), (2,4)], [(3,4)], [(1,6), (2,5)]]

        order = [0, 1, 2, 3]
        if rot90_k == 2:  # TODO other rotations
            order.reverse()
        elif rot90_k == 1:
            order = [1, 3, 0, 2]
        elif rot90_k == 3:
            order = [2, 0, 3, 1]

    else:  # v4.0
        radius_kernel = 7
        diff_expect = 47  # first guess estimated visually from samples
        off_xy_expect = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        # find corners = toplft, toprgt, btmrgt, btmlft seen with thickest as top row
        corner_rows_cols = [
            [(0,-5), (0,-4)], #0s
            [(0,-6), (0,-3)], #1s
            [(1,-7), (1,-4)], #2s
            [(1,-6), (0,-7)]] #3s
        # corner idxs first two rows:
        # [..., 2, 2, 3, 1, 0, 0, 1, 0, 0]
        # [..., 1, 3, 2, 3, 0, 2, 1, 2, 1]

        order = [1, 2, 3, 0]
        order = np.roll(order, rot90_k).tolist()

    off_xy_expect_rot = []
    for i in order:
        off_xy_expect_rot.append(off_xy_expect[i])
    off_xy_expect = off_xy_expect_rot

    if pr_image is False:
        # only first for each corner]
        corner_rows_cols = [[cells[0]] for cells in corner_rows_cols]
    off_xy_expect = diff_expect * np.array(off_xy_expect)
    off_xy_center = (0, 0)

    kernel = get_roi_circle(
        (radius_kernel, radius_kernel), (0, 0), radius_kernel)
    kernel = 1./np.sum(kernel) * kernel

    sum_center = np.zeros((width_sub_50um, width_sub_50um))
    sum_corners = [np.copy(sum_center) for i in range(4)]
    for idx, image in enumerate(matrix):
        if image is not None and roi_dicts[idx] is not None:
            for corner_idx, rows_cols in enumerate(corner_rows_cols):
                x = [round(roi_dicts[idx]['xs'][r, c]) for r, c in rows_cols]
                y = [round(roi_dicts[idx]['ys'][r, c]) for r, c in rows_cols]
                for i in range(len(x)):
                    sub = image[y[i] - wi:y[i] + wi, x[i] - wi:x[i] + wi]
                    sub = resize(sub, (width_sub_50um, width_sub_50um),
                                 anti_aliasing=True)
                    dx, dy = finetune_center_cell(sub, phantom, line_dist)
                    sub = np.roll(sub, (-dy, -dx), axis=(0, 1))
                    sum_center = sum_center + sub
                    sum_corners[corner_idx] = sum_corners[corner_idx] + sub

    off_xy_center = find_offset_disc(
        sum_center, kernel, 30, xy_expected=(0, 0))
    off_xy_corners = []
    for i, sum_corner in enumerate(sum_corners):
        off_xy_corner = find_offset_disc(
            sum_corner, kernel, 30, xy_expected=off_xy_expect[i])
        off_xy_corners.append(off_xy_corner)

    # create the templates with circle rois by given search radius
    templates = [get_roi_circle(
        (width_sub_50um, width_sub_50um), off_xy_center, radius_search)]
    for off_xy in off_xy_corners:
        templates.append(get_roi_circle(
            (width_sub_50um, width_sub_50um), off_xy, radius_search))

    if phantom == 34:
        # inverse diamond mask to avoid grid when averaging
        mask_grid = np.ones(templates[0].shape)
        w = round(0.9* templates[0].shape[0] / np.sqrt(2) / 2)
        half = templates[0].shape[0] // 2
        mask_grid[half - w:half + w, half - w:half + w] = 0
        mask_grid = ndimage.rotate(mask_grid, angle=45, cval=1, reshape=False)
        mask = np.zeros(mask_grid.shape, dtype=bool)
        mask[mask_grid > 0.5] = True
        templates.append(mask)

    # create mask to use for detrending
    radius_masks = 23  # 1+ mm cirle radius
    mask = 1 * get_roi_circle(
        (width_sub_50um, width_sub_50um), off_xy_center, radius_masks)
    for off_xy in off_xy_corners:
        mask = mask + 1 * get_roi_circle(
            (width_sub_50um, width_sub_50um), off_xy, radius_masks)
    if phantom == 34:
        mask = mask + 1 * templates[-1]
    for i in range(1, width_sub_50um, 2):  # reduce points for linear fit
        mask[i, :] = 1
        mask[:, i] = 1

    templates.append(mask.astype(bool))

    return templates, wi, line_dist


def get_kernels(tables, pix_new):
    radii = np.array(tables['diameters']) / (2 * pix_new)

    kernels = []
    for radius in radii:
        kernel_size = radius * 2 + 1
        kernel = get_roi_circle(
            (kernel_size, kernel_size), (0, 0), radius)
        kernels.append(1./np.sum(kernel) * kernel)

    return kernels


def finetune_center_cell(sub, phantom, line_dist):
    dx, dy = 0, 0
    sz = sub.shape[0]
    center = round(sz / 2)
    mids = []
    if phantom == 34:
        peak_expect = np.array([center - line_dist, center + line_dist])
        yx_yx = [(0, 0, sz - 1, sz - 1), (sz - 1, 0, 0, sz - 1)]
        for idx, (y1, x1, y2, x2) in enumerate(yx_yx):
            rr, cc = draw.line(y1, x1, y2, x2)
            profile = sub[rr, cc]
            prof = -1.*profile + np.median(profile)
            prof[:center - line_dist - 15] = 0
            prof[center + line_dist + 15:] = 0
            peaks_this = find_peaks(prof, height=0.3*np.max(prof))
            peak_pos = np.array([0, 0])
            for i in range(2):
                diff = np.abs(peaks_this[0] - peak_expect[i])
                idx_closest = np.where(diff == np.min(diff))
                peak_pos[i] = peaks_this[0][idx_closest[0][0]]
            mids.append(np.mean(peak_pos - peak_expect))
        dx = 0.5 / np.cos(np.pi / 4) * (mids[0] + mids[1])
        dy = 0.5 / np.sin(np.pi / 4) * (mids[0] - mids[1])
    else:  # 40
        qu = round(sz/8)
        profiles = [np.mean(sub[center-qu:center+qu, :], axis=0)]
        profiles.append(np.mean(sub[:, center-qu:center+qu], axis=1))
        for profile in profiles:
            prof = -1.*profile + np.median(profile)
            peaks_this = find_peaks(prof, height=0.5*np.max(prof))
            peak_pos = (peaks_this[0][0],peaks_this[0][-1])
            diff = round(np.mean(peak_pos)) - center
            mids.append(diff)
        dx = mids[0]
        dy = mids[1]

    return (round(dx), round(dy))


def read_cdmam_sub(sub, phantom, line_dist, disc_templates, kernel):#, fit2d):
    """Calculate results for sub in CDMAM."""
    if sub is None:
        res = None
    else:
        sub = sub - np.min(sub)
        sub = resize(sub, disc_templates[0].shape, anti_aliasing=True)
        dx, dy = finetune_center_cell(sub, phantom, line_dist)
        sub = np.roll(sub, (-dy, -dx), axis=(0, 1))
        ld, ks = 0, 0
        if phantom == 34:
            sub[disc_templates[-2] == True] = np.median(sub)
            fit = mmcalc.polyfit_2d(sub, max_order=1, mask=disc_templates[-1])
            sub = sub - fit
            avgs_sub = fftconvolve(sub, kernel, mode='same')
        elif phantom == 40:
            cent = round(sub.shape[0] // 2)
            ld = line_dist - 4  # -4 to avoid line signal
            sub_cropped = sub[cent-ld:cent+ld,cent-ld:cent+ld]
            mask = disc_templates[-1][cent-ld:cent+ld,cent-ld:cent+ld]
            fit = mmcalc.polyfit_2d(sub_cropped, max_order=1, mask=mask)
            sub_cropped = sub_cropped - fit
            avgs_sub = fftconvolve(sub_cropped, kernel, mode='valid')
            ks = round(kernel.shape[1] // 2)

        minvals = []
        minpos = []
        start, end = 0, 0
        if phantom == 40:
            start = cent - ld + ks
            end = start + avgs_sub.shape[0]
        for template in disc_templates[:5]:
            if phantom == 34:
                #min_yx, _ = mmcalc.get_min_max_pos_2d(avgs_sub, template)
                min_yx = get_min_pos_2d(avgs_sub, template)
                minpos.append(min_yx)
            elif phantom == 40:
                #min_yx, _ = mmcalc.get_min_max_pos_2d(
                min_yx = get_min_pos_2d(
                    avgs_sub, template[start:end,start:end])
                minpos.append(min_yx)
            minvals.append(avgs_sub[min_yx[0], min_yx[1]])

        corner_min = np.min(minvals[1:])
        idx_min = np.where(minvals[1:] == corner_min)[0]
        idxs_not_min = np.where(minvals[1:] > corner_min)[0]
        rest_min_vals = np.array(minvals)[idxs_not_min + 1]
        central_is_min = minvals[0] < np.min(rest_min_vals)

        if phantom == 34:
            idx_min += 1

        res = {
            'corner_index': idx_min,
            'central_disc_found': central_is_min,
            'min_positions': minpos,
            'processed_sub': avgs_sub,
            'start': start, 'end': end
            }

    return res


def read_cdmam_image(image, image_info, roi_dict,
                     cdmam_table_dict, paramset,
                     templates, kernels, wi, line_dist):
    """Read cdmam results for an image.

    Parameters
    ----------
    image : np.ndarray
    roi_dict : dict
        as generated in find_cdmam
    paramset : config_classes.ParamsetMammo

    Returns
    -------
    details_dict
    """
    include_array = None
    if 'include_array' in roi_dict:
        include_array = roi_dict['include_array']
    center_xs = roi_dict['xs']
    center_ys = roi_dict['ys']
    phantom = roi_dict['phantom']
    if roi_dict['invert']:
        image = -np.copy(image)

    sz_y, sz_x = center_xs.shape
    res_table = []
    nn = 0
    include = True
    for row in range(sz_y):
        res_table.append([])
        for col in range(sz_x):
            res_table[-1].append([])
            if phantom == 34:
                include = include_array[row, col]
            if include:
                x = round(center_xs[row, col])
                y = round(center_ys[row, col])
                sub = image[y - wi:y + wi, x - wi:x + wi]
                if sub.shape[0] != sub.shape[1]:
                    sub = fix_cropped_sub(image, x, y, wi)
                res = read_cdmam_sub(
                    sub, phantom, line_dist, templates, kernels[col])
                #paramset.cdm_fit)
                res_table[row][col] = res
                nn += 1
    '''Mutliprocessing (failed being faster)
    res_table = []
    include = True
    for row in range(sz_y):
        subs = []
        for col in range(sz_x):
            sub = None
            if phantom == 34:
                include = include_array[row, col]
            if include:
                x = round(center_xs[row, col])
                y = round(center_ys[row, col])
                sub = image[y - wi:y + wi, x - wi:x + wi]
                if sub.shape[0] != sub.shape[1]:
                    sub = fix_cropped_sub(image, x, y, wi)
            subs.append(sub)

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)
        futures_list = [
            pool.submit(
                read_cdmam_sub, subs[col], phantom, line_dist,
                               templates, kernels[col])
            for col in range(sz_x)]

        res = [f.result() for f in futures_list]
        res_table.append(res)
    '''

    found_correct_corner = np.zeros((sz_y, sz_x), dtype=bool)
    found_centers = np.zeros((sz_y, sz_x), dtype=bool)
    corners = np.array(
        cdmam_table_dict['corner_index'])

    # correctly found corner?
    for row in range(sz_y):
        for col in range(sz_x):
            if phantom == 34:
                include = include_array[row, col]
            if include:
                found_correct_corner[row, col] = (
                    res_table[row][col]['corner_index']
                    == corners[row, col])
                found_centers[row, col] = (
                    res_table[row][col]['central_disc_found'])

    details_dict = {
        'found_correct_corner': found_correct_corner,
        'include_array': include_array,
        'found_centers': found_centers,
        'corrected_neighbours': None,
        'res_table': res_table,
        'kernels': kernels, 'templates': templates,
        'roi_dict':roi_dict
        }

    if paramset.cdm_center_disc_option == 1:
        found_this = found_correct_corner * found_centers
    else:
        found_this = found_correct_corner
    if paramset.cdm_correct_neighbours:
        found_this = correct_neighbours(
            found_this, include_array=include_array)
        details_dict['corrected_neighbours'] = found_this

    details_dict['detection_matrix'] = 1.*found_this

    if paramset.cdm_center_disc_option == 2:
        if paramset.cdm_correct_neighbours:
            found_centers = correct_neighbours(
                found_centers, include_array=include_array)
            details_dict['corrected_neighbours_centers'] = found_centers
        details_dict['detection_matrix_centers'] = 1.*found_centers

    return details_dict


def correct_neighbours(founds, include_array=None):
    """Correct found according to nearest neighbours.

    If True and < 2 of 4 neighbours are True, corrected to False
    If False and > 2 of 4 neighbours are True, corrected to True
    If True and 0 of 2 neighbours are True, corrected to False
    If False and 2 of 2 neighbours are True, corrected to True

    Parameters
    ----------
    founds : np.2darray
        dtype bool. True if found
    include_array : np.2darray or None
        dtype bool. True if included

    Returns
    -------
    corrected : np.2darray
    """
    corrected = np.copy(founds)
    if include_array is None:
        include_array_ints = np.ones(founds.shape)
    else:
        include_array_ints = 1 * np.copy(include_array)
    founds_padded = np.pad(founds + include_array_ints, (1, 1), 'constant')
    neighbours = np.zeros(
        (4, founds_padded.shape[0], founds_padded.shape[1]), dtype=int)
    neighbours[0] = np.roll(founds_padded, 1, axis=0)
    neighbours[1] = np.roll(founds_padded, -1, axis=0)
    neighbours[2] = np.roll(founds_padded, 1, axis=1)
    neighbours[3] = np.roll(founds_padded, -1, axis=1)
    neighbours = neighbours[:,1:-1, 1:-1]
    n_neighbours = np.sum(neighbours.astype(bool), axis=0)
    true_neighbours = neighbours - 1
    true_neighbours[true_neighbours < 0] = 0
    true_neighbours = true_neighbours.astype(bool)
    n_trues = np.sum(true_neighbours, axis=0)

    if include_array is not None:
        n_neighbours[include_array == False] = 0
        n_trues[include_array == False] = 0

    corrected[np.logical_and(
        np.logical_and(founds == True, n_trues < 2),
        n_neighbours > 2)] = False
    corrected[np.logical_and(
        np.logical_and(founds == False, n_trues > 2),
        n_neighbours > 2)] = True
    corrected[np.logical_and(
        np.logical_and(founds == True, n_trues == 0),
        n_neighbours == 2)] = False
    corrected[np.logical_and(
        np.logical_and(founds == False, n_trues == 2),
        n_neighbours == 2)] = True

    return corrected


def sum_detection_matrix(details_dicts, include_array, cdm_center_disc_option):
    """Sum detection ratios from all images and optionally correct for center.

    Parameters
    ----------
    details_dicts : list of dicts
    include_array : np.ndarray
    cdm_center_disc_option : int

    Returns
    -------
    update_dict : dict
    """
    detection_matrix = details_dicts[0]['detection_matrix']
    detection_matrix_centers = None
    update_dict = {}
    if cdm_center_disc_option == 2:
        detection_matrix_centers = details_dicts[0]['detection_matrix_centers']
    for dd in details_dicts[1:]:
        detection_matrix = (
            detection_matrix + dd['detection_matrix'])
        if cdm_center_disc_option == 2:
            detection_matrix_centers = (
                detection_matrix_centers + dd['detection_matrix_centers'])
    n_imgs = len(details_dicts)
    detection_matrix = detection_matrix / n_imgs
    if cdm_center_disc_option == 2:
        detection_matrix_centers = detection_matrix_centers / n_imgs

        update_dict['detection_matrix_corners'] = np.copy(
            detection_matrix)
        update_dict['detection_matrix_centers'] = np.copy(
            detection_matrix_centers)
        detection_matrix[np.logical_and(
            detection_matrix < 0.25,
            include_array == True)] = 0.25
        detection_matrix_centers[np.logical_and(
            detection_matrix_centers < 0.25,
            include_array == True)] = 0.25
        detection_matrix = np.minimum(
            detection_matrix, detection_matrix_centers)
        update_dict['detection_matrix'] = detection_matrix
    update_dict['detection_matrix'] = detection_matrix

    return update_dict


def psychometric_curve(t, f, ct):
    """Calculate psychometric curve from t values and parameters."""
    mu = 0.19 # um-1 Young et al 2008 (Proc of SPIE, Vol 6913, 69131C-1)
    c = np.log(1 - np.exp(-mu * t))
    p = 0.25 + 0.75 / (1 + np.exp(-f * (c - ct)))
    return p


def psychometric_curve_fit(t, p, f_set=None):
    """Fit t, p to psychometric curve p(t)."""
    if f_set is None:
        f0 = 10
        f_min = 1
        f_max = 1000
    else:
        f_min = 0.99 * f_set
        f_max = 1.01 * f_set
        f0 = f_set
    try:
        popt, _ = curve_fit(
            psychometric_curve, t, p, p0=[f0, -5.],
            bounds=([f_min, -20], [f_max, -0.1])
            )
    except (ValueError, RuntimeError):
        popt = None
    return popt


def third_order_polynomial(x, a, b, c, d):
    return a + b/x + c/(x**2) + d/(x**3)


def third_order_polynomial_fit(x, p):
    try:
        popt, _ = curve_fit(
            third_order_polynomial, x, p, p0=[0.5]*4,
            bounds=([0]*4, [1.]*4), sigma=p
            )
    except (ValueError, RuntimeError):
        popt = None
    return popt


def calculate_fitted_psychometric(cdmam_table_dict, paramset):
    """Fit psychometric curves to values in detection matrix.

    Also, add results to cdmam_table_dict.

    Parameters
    ----------
    cdmam_table_dict : dict
        detection_matrix + diameter and thickness info for given phantom
    paramset: ParamsetMammo.
    """
    sigma = paramset.cdm_sigma
    diameters = cdmam_table_dict['diameters']
    thickness = np.array(cdmam_table_dict['thickness'])
    detection_matrix = np.copy(cdmam_table_dict['detection_matrix'])

    euref_thickness_contast = {
        # Table A6.1 in EUREF 4th edition page 162, nominal 28kV, Mo/Mo
        'thickness': [0.1, 0.5, 1.0, 1.5, 2.0],
        'contrast': [1.57, 7.6, 14.55, 20.92, 26.76],
        }

    include_array = None
    detection_matrix_corrected = detection_matrix
    out_diameter_idx_start_stop = []
    if isinstance(thickness[0], np.ndarray):
        #phantom = 40
        scale_r = {
            # inversly found from result table vendor software artinis
            'diameters': diameters[::-1],
            'rs': [1.41, 1.44, 1.47, 1.58, 1.61, 1.66, 1.69, 1.75, 1.81, 1.84, 1.88,
                   1.91, 1.95, 2., 2.05, 2.08, 2.1, 2.13, 2.14, 2.16, 2.19]
            }

        detection_matrix_corrected[detection_matrix < 0.25] = 0.25
        if sigma > 0:
            detection_matrix_corrected = ndimage.gaussian_filter(
                detection_matrix_corrected, sigma=sigma)

    else:
        #phantom = 34
        scale_r = {
            # table2 in NUKLEONIKA 2016;61(1):53-59
            'diameters': [.08, .1, .13, .16, .2, .25, .31, .4, .5, .63, .8, 1.],
            'rs': [1.4, 1.5, 1.6, 1.68, 1.75, 1.82, 1.88, 1.94, 1.98, 2.01, 2.06, 2.11]
            }
        out_diameter_idx_start_stop = [3, -1]  # 1.. 0.08
        corner_index_table = np.array(cdmam_table_dict['corner_index'])
        include_array = np.zeros(corner_index_table.shape, dtype=bool)
        include_array[corner_index_table > 0] = True
        thickness = np.broadcast_to(thickness, (16, 16)).T

        detection_matrix_corrected[
            np.logical_and(
                detection_matrix < 0.25, include_array == True)] = 0.25
        if sigma > 0:
            # avoid smoothing outside included part of array
            mask = 1.0 * include_array
            sm = ndimage.gaussian_filter(detection_matrix_corrected * mask,
                                         sigma=sigma)
            mask[mask == 0] = 0.0001  # to avoide divide by zero warning
            sm /= ndimage.gaussian_filter(mask, sigma=sigma)
            sm[include_array == False] = 0
            detection_matrix_corrected = sm
    cdmam_table_dict['detection_matrix_corrected'] = detection_matrix_corrected

    xs = []
    ys = []
    yfits = []
    thickness_founds = []
    thickness_predicts = []

    def fit(diameter_number, f_set=None):
        yvals = detection_matrix_corrected[:, diameter_number]
        xvals = thickness[:, diameter_number]
        r = mmcalc.get_curve_values(
            scale_r['rs'],
            scale_r['diameters'], [diameters[diameter_number]])

        if include_array is not None:
            xvals = xvals[include_array[:, diameter_number] == True]
            yvals = yvals[include_array[:, diameter_number] == True]

        popt = None
        yfit = None
        threshold_thickness = [None]
        thickness_predict = [0]
        if np.min(yvals) < 0.625 and np.max(yvals) > 0.625:
            popt = psychometric_curve_fit(xvals, yvals, f_set=f_set)
            if popt is not None:
                f, ct = popt
                yfit = psychometric_curve(xvals, f, ct)
                threshold_thickness = mmcalc.get_curve_values(
                    xvals, yfit, [0.625], extrapolate=True)
        if threshold_thickness[0] is None:
            if np.min(yvals) >= 0.625:  # all above
                threshold_thickness[0] = np.min(xvals)
            elif np.max(yvals) <= 0.625:  # all below
                threshold_thickness[0] = np.max(xvals)

        if r[0] is not None and threshold_thickness[0] is not None:
            # r[0] None if outside range of euref r-scale
            ##thickness_predict = r[0] * threshold_thickness[0]
            contrast_auto = mmcalc.get_curve_values(
                euref_thickness_contast['contrast'],
                euref_thickness_contast['thickness'],
                threshold_thickness, extrapolate=True)
            contrast_predict = r[0] * contrast_auto[0]
            thickness_predict = mmcalc.get_curve_values(
                euref_thickness_contast['thickness'],
                euref_thickness_contast['contrast'],
                [contrast_predict], extrapolate=True)

        return xvals, yvals, popt, yfit, threshold_thickness[0], thickness_predict[0]

    # estimate f from 3 curves
    f_set = None
    f_sets = []
    for i in [6, 8, 10]:
        _, _, popt, _, _, _ = fit(i)
        if popt is not None:
            f_sets.append(popt[0])
    if len(f_sets) > 0:
        f_set = np.mean(f_sets)

    if f_set is not None:
        for d in range(len(diameters)):
            xvals, yvals, _, yfit, thick_found, thick_predict = fit(
                d, f_set=f_set)
            xs.append(xvals)
            ys.append(yvals)
            yfits.append(yfit)
            thickness_founds.append(thick_found)
            thickness_predicts.append(thick_predict)
        thickness_predicts = np.array(thickness_predicts)
        idxs = np.where(thickness_predicts > 0)
        popt = third_order_polynomial_fit(
            np.array(diameters)[idxs], thickness_predicts[idxs])
        thickness_predicts[np.where(thickness_predicts == 0)] = None
        if out_diameter_idx_start_stop:
            thickness_founds = thickness_founds[
                out_diameter_idx_start_stop[0]:out_diameter_idx_start_stop[1]]
            thickness_predicts = thickness_predicts[
                out_diameter_idx_start_stop[0]:out_diameter_idx_start_stop[1]]
            diameters = np.array(diameters[
                out_diameter_idx_start_stop[0]:out_diameter_idx_start_stop[1]])
        else:
            diameters = np.array(diameters)
        thickness_predicts_fit = third_order_polynomial(
            diameters, popt[0], popt[1], popt[2], popt[3])

        cdmam_table_dict['psychometric_results'] = {
            'xs': xs, 'ys': ys, 'yfits': yfits,
            'thickness_founds': np.array(thickness_founds),
            'thickness_predicts': thickness_predicts,
            'thickness_predicts_fit_d': diameters,
            'thickness_predicts_fit': thickness_predicts_fit}

        cdmam_table_dict['EUREF_performance_limits'] = {  # table page 147 EUREF 4th edition
            'diameters': [2, 1, 0.5, 0.25, 0.1],
            'acceptable_thresholds_contrast': [1.05, 1.4, 2.35, 5.45, 23.],
            'achievable_thresholds_contrast': [.55, .85, 1.6, 3.8, 15.8],
            'acceptable_thresholds_thickness': [0.069, 0.091, 0.15, 0.352, 1.68],
            'achievable_thresholds_thickness': [0.038, 0.056, 0.103, 0.244, 1.1]
            }
