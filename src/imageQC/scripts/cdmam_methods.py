# -*- coding: utf-8 -*-
"""
Collection of methods for SNI and uniformity calculations
 for gamma camera in ImageQC.

@author: ellen
"""

import numpy as np
from scipy import ndimage
from scipy.signal import (find_peaks, fftconvolve, medfilt2d)
from skimage.transform import hough_line, hough_line_peaks, resize
from skimage import draw

# imageQC block start
import imageQC.scripts.mini_methods_calculate as mmcalc
# imageQC block end


def detrend_sub(sub, sigma_smooth, median_filter_width):
    """Median filter and flatten matrix, keeping most details. """
    smooth = ndimage.gaussian_filter(sub, sigma=sigma_smooth)
    sub = medfilt2d(sub, median_filter_width)
    detrend_factors = smooth / np.median(sub)
    return sub / detrend_factors


def get_binary_sub(sub, percentile=25):
    """Get binary of matrix setting threshold to specified percentile value."""
    sz_x_sub, sz_y_sub = sub.shape
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
    if n_angles < 5:
        n_angles = 5
    tested_angles = np.linspace(
        angle_range[0], angle_range[1],
        round(n_angles), endpoint=False)
    hspace, theta, d = hough_line(binary_sub, theta=tested_angles)
    threshold_peaks = threshold_peaks * np.max(hspace)
    _, angles, dists = hough_line_peaks(hspace, theta, d,
                                        threshold=threshold_peaks,
                                        min_distance=margin)
    return (angles, dists)


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


def find_phantom_part(image, margin, axis=0):
    """"Detect phantom in image along axis."""
    prof = np.mean(image, axis=axis)
    in_phantom = np.where(np.logical_and(
        prof < np.median(prof)*1.1,
        prof > np.median(prof)*0.9))
    start, end = in_phantom[0][0], in_phantom[0][-1]
    start = start + margin
    end = end - margin
    sz_y, sz_x = image.shape
    if axis == 0:
        prof = np.mean(
            image[sz_y // 2 - 10: sz_y // 2 + 10, start:end], axis=0)
    else:
        prof = np.median(
            image[start:end, sz_x // 2 - 10: sz_x // 2 + 10], axis=1)
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
    errmsgs = []
    px_pr_mm = round(1./image_info.pix[0])
    roi_estimate = paramset.cdm_roi_estimate
    margin = 5 * px_pr_mm
    margin_angle = paramset.cdm_tolerance_angle * 2 * np.pi / 360
    sz_y, sz_x = image.shape
    phantom = 0
    roi_array = None
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

        # if phantom > 0:
            #lines = get_lines(angles_dists_center, xs, ys[0])
    roi_array = None
    if phantom == 0:
        errmsgs.append(
            'Failed to detect horizontal or diagonal lines.')
    elif phantom == 34:
        roi_array = find_phantom_34(
            image, margin, margin_angle, angles_dists_center, paramset)
        roi_array.append(invert)
    elif phantom == 40:
        roi_array = find_phantom_40(
            image, px_pr_mm, margin_angle, angles_dists_center, paramset)
        roi_array.append(invert)

    return (roi_array, errmsgs)


def find_phantom_34(image, margin, margin_angle, angles_dists_center, paramset):
    xy_refs = [[], []]
    dx_intersections = []
    xs, ys = [[], []]
    sz_y, sz_x = image.shape
    errmsgs = []

    x_start, x_end, x_prof = find_phantom_part(image, margin)
    width = x_end - x_start
    ys = [margin, sz_y//4]  # top
    xs = [x_start + width // 4, x_end - width // 4]  # mid
    subs = [
        image[ys[0]:ys[1], xs[0]:xs[1]],
        image[-ys[1]:-ys[0], xs[0]:xs[1]]]
    lines = [[], []]  # [(x0, y0), slope] for two groups (perpendicular)

    for sub_no, sub in enumerate(subs):
        # find lines
        sub = detrend_sub(sub, margin, 3)
        sub_binary = get_binary_sub(sub)
        y_start = ys[0] if sub_no == 0 else sz_y - ys[1]
        lines = []
        for angle in [np.pi / 4, - np.pi / 4]:
            test_range = [angle - margin_angle, angle + margin_angle]
            angles, dists = find_angles_sub(
                sub_binary, test_range, margin,
                paramset.cdm_threshold_peaks)
            lines_this = get_lines([*zip(angles, dists)], xs, y_start)
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

    roi_array = [include_array, xs, ys, diagonal, angle]
    return roi_array


def find_phantom_40(image, px_pr_mm, angle_margin, angles_center, paramset):
    dist = None
    errmsgs = []
    roi_estimate = paramset.cdm_roi_estimate
    estimate_x, estimate_y = roi_estimate, roi_estimate
    # finding vertical line positions
    x_start, x_end, xprof = find_phantom_part(image, 5*px_pr_mm, axis=0)
    if estimate_x is False:
        #second_derived = np.diff(np.diff(xprof))
        peaks = find_peaks(xprof, distance=7*px_pr_mm,
                           height=0.5*np.max(xprof))
        peaks_pos = peaks[0]

        if len(peaks_pos) == 17:
            dx = x_start
            x_start, x_end = peaks_pos[0] + dx, peaks_pos[-1] + dx
            dist = (peaks_pos[-1] - peaks_pos[0]) / 16
        else:
            errmsgs.append(
                'Failed to find a grid 16 cells wide. x-positions '
                'centered on assumed phantom.')
            estimate_x = True

    y_start, y_end, yprof = find_phantom_part(image, 5*px_pr_mm, axis=1)
    if estimate_y is False:
        #second_derived = np.diff(np.diff(yprof))
        peaks = find_peaks(yprof, distance=3.5*px_pr_mm,
                           height=0.5*np.max(yprof))
        peaks_pos = peaks[0]
        if len(peaks_pos) == 25:
            dy = y_start
            y_start, y_end = peaks_pos[0] + dy, peaks_pos[-1] + dy
            if dist is None:
                dist = (peaks_pos[-1] - peaks_pos[0]) / 22.5
        else:  # look for the first separator (shorter dists)
            if dist is None:
                dist = 9 * px_pr_mm
            dists = np.diff(peaks_pos)
            idx_sep = np.where(np.logical_and(
                dists < 1.05 * dist / 2, dists > 0.95 * dist / 2))
            idx_sep = idx_sep[0]
            if len(idx_sep[0]) == 3:
                first_peak_sep = peaks_pos[idx_sep[0][0]]
                y_start = first_peak_sep - dist * 3
                y_end = y_start + dist * 22.5
            else:
                errmsgs.append(
                    'Failed to find a grid 21 cells high. '
                    'y-positions centered on assumed phantom.')
                estimate_y = True

    if dist is None:
        dist = 9 * px_pr_mm
    if estimate_x:
        mid_x = 0.5 * (x_end + x_start)
        x_start, x_end = mid_x - 8 * dist, mid_x + 8 * dist
    if estimate_y:
        mid_y = 0.5 * (y_end + y_start)
        y_start, y_end = mid_y - 22.5/2 * dist, mid_y + 22.5/2 * dist

    if dist and y_start:
        xs, ys = np.meshgrid(
            dist * (np.arange(16) + 0.5),
            dist * (np.arange(21) + 0.5))
        ys[3:, :] = ys[3:, :] + 0.5 * dist
        ys[7:, :] = ys[7:, :] + 0.5 * dist
        ys[11:, :] = ys[11:, :] + 0.5 * dist
        # center before rotate
        xs = xs - 8 * dist
        ys = ys - 22.5/2 * dist

        diff_angle = np.median(
            [angle for angle, dist in angles_center])
        for row in range(21):
            for col in range(16):
                x, y = xs[row, col], ys[row, col]
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x) + diff_angle
                xs[row, col] = r * np.cos(phi)
                ys[row, col] = r * np.sin(phi)

        xs = xs + 8 * dist + x_start
        ys = ys + 22.5/2 * dist + y_start

        roi_array = [None, xs, ys, dist, diff_angle]

    return roi_array


'''

def get_off_xy_periferal_disc(sub, phantom):
    inside = np.full(sub.shape, False)
    xy_c = sub.shape[0] // 2
    sz = sub.shape[0]
    if phantom == 34:
        # top disc
        inside[xy_c - sz * 3 // 8:xy_c - sz // 8,
               xy_c - sz // 8: xy_c + sz // 8] = True
        inside = np.round(rotate2d_offcenter(
            inside.astype(float), 45, (0, - sz * 1 // 4)))
        inside = np.array(inside, dtype=bool)
    else:
        # lower left disc, circular ROI with radius sz // 8
        nparange = np.arange(0, sz, dtype=int) - xy_c
        xs, ys = np.meshgrid(nparange, nparange, sparse=True)
        zs = np.sqrt((xs + 0.15 * sz)** 2 + (ys - 0.15 * sz) ** 2)
        inside = zs <= sz // 8
    xy_ref = center_xy_of_disc(sub, roi=inside, sigma=1.)
    breakpoint()

    return xy_ref
'''


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


def get_template_kernel(tables, phantom, pix, pix_new, cell_width,
                        angle, image, xy):
    radii = np.array(tables['diameters']) / (2 * pix_new)
    n_pix_extra_search = 4  # 200 um
    wi = round(cell_width / 2)
    if phantom == 34:
        width_sub = wi * 2 + 1
    else:
        width_sub = wi * 2 + 1 + 10
    factor = pix / pix_new
    width_sub_50um = round(width_sub * pix / pix_new)
    x, y = xy

    if phantom == 34:
        # find distance to corner disc based on sample in row 2, col 4
        sub = image[y - wi:y + wi, x - wi:x + wi]
        sub = resize(sub, (width_sub_50um, width_sub_50um), anti_aliasing=True)
        kernel = get_roi_circle((15, 15), (0, 0), 15)
        kernel = 1./np.sum(kernel) * kernel
        avgs_sub = fftconvolve(sub, kernel, mode='same')
        cent = round(width_sub_50um/2)
        prof = avgs_sub[cent, 30: cent - 30]
        prof2 = avgs_sub[cent, cent - 30: cent + 30]
        _, mid = mmcalc.get_width_center_at_threshold(prof)
        _, mid2 = mmcalc.get_width_center_at_threshold(prof2)
        if mid is not None and mid2 is not None:
            diff = round(mid2 + (cent-30) - (mid + 30))
        else:
            diff = round(2/3 * width_sub_50um / 2)
        off_xy_first = (0, -diff)
    else:
        off_center = (cell_width * factor) // 4
        off_xy_first = (-off_center, -off_center)

    templates = []
    kernels = []
    for radius in radii:
        template = []
        central = get_roi_circle((width_sub_50um, width_sub_50um), (0, 0),
                                 radius + n_pix_extra_search)
        template.append(central)
        corner = get_roi_circle((width_sub_50um, width_sub_50um), off_xy_first,
                                radius + n_pix_extra_search)
        if phantom == 40:
            template.extend([corner, np.fliplr(corner),
                             np.flipud(corner), np.flipud(np.fliplr(corner))])
        else:
            template.append(corner)
            for k in [1, 3, 2]:  # top, left, right, btm = idx 1,2,3,4
                template.append(np.rot90(corner, k))

        kernel_size = radius * 2 + 1
        kernel = get_roi_circle(
            (kernel_size, kernel_size), (0, 0), radius)
        kernels.append(1./np.sum(kernel) * kernel)
        templates.append(template)
    return (wi, templates, kernels)

def finetune_center_cell(sub, phantom, line_dist):
    dx, dy = 0, 0
    sz = sub.shape[0]
    center = round(sz / 2)
    peak_expect = np.array([center - line_dist, center + line_dist])
    if phantom == 34:
        mids = []
        yx_yx = [(0, 0, sz - 1, sz - 1), (sz - 1, 0, 0, sz - 1)]
        for idx, (y1, x1, y2, x2) in enumerate(yx_yx):
            rr, cc = draw.line(y1, x1, y2, x2)
            profile = sub[rr, cc]
            prof = -1.*profile + np.median(profile)
            peaks_this = find_peaks(prof, height=0.5*np.max(prof))
            peak_pos = np.array([0, 0])
            for i in range(2):
                diff = np.abs(peaks_this[0] - peak_expect[i])
                idx_closest = np.where(diff == np.min(diff))
                peak_pos[i] = peaks_this[0][idx_closest[0][0]]
            mids.append(np.mean(peak_pos - peak_expect))
        dx = 0.5 / np.cos(np.pi / 4) * (mids[0] + mids[1])
        dy = 0.5 / np.sin(np.pi / 4) * (mids[0] - mids[1])
            #parts = []
            #for expect_mid in peak_expect:
                #part = profile[expect_mid - 30 : expect_mid + 30]
                #_, mid = mmcalc.get_width_center_at_threshold(part)
                #if mid is None:
                #    mid = 0
                #mids.append(mid)
                #parts.append(part)
                #reverse = not reverse
        '''
        mid1 = np.mean(mids[:2]) - 10
        mid2 = np.mean(mids[2:]) - 10
        dx = 0.5 / np.cos(np.pi / 4) * (mid1 + mid2)
        dy = 0.5 / np.sin(np.pi / 4) * (mid1 - mid2)
        '''

    return (round(dx), round(dy))


def read_cdmam_sub(sub, phantom, line_dist, # test_angles, threshold_peaks,
                   disc_templates, kernel):
    """Calculate results for sub in CDMAM34."""
    sub = resize(sub, disc_templates[0].shape, anti_aliasing=True)
    dx, dy = finetune_center_cell(sub, phantom, line_dist)#, test_angles, threshold_peaks)
    avgs_sub = fftconvolve(sub, kernel, mode='same')
    minvals = []
    minpos = []
    for template in disc_templates:
        min_yx, _ = mmcalc.get_min_max_pos_2d(
            avgs_sub, np.roll(template, (dx, dy), axis=(1,0)))
        minvals.append(np.min(avgs_sub[min_yx[0], min_yx[1]]))
        minpos.append(min_yx)

    corner_min = np.min(minvals[1:])
    idx_min = np.where(minvals[1:] == corner_min)[0]
    idxs_not_min = np.where(minvals[1:] > corner_min)[0]
    rest_min_vals = np.array(minvals)[idxs_not_min + 1]
    central_is_min = minvals[0] < np.min(rest_min_vals)

    if phantom == 34:
        idx_min += 1

    return (idx_min, central_is_min, minpos, sub)
