# -*- coding: utf-8 -*-
"""
Collection of methods for SNI and uniformity calculations
 for gamma camera in ImageQC.

@author: ellen
"""
from time import time
import numpy as np
from scipy import ndimage
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
        sub[0:cropped.shape[1], :] = cropped
        sub[cropped.shape[1]:, :] = np.median(cropped)
    elif x1 < 0:
        sub[:,-x1:] = image[y1:y2, 0:x2]
        sub[:,0:-x1] = np.median(sub)
    elif x2 >= image.shape[1]:
        cropped = image[y1:y2, x1:]
        sub[:, 0:cropped.shape[0]] = cropped
        sub[:, cropped.shape[0]:] = np.median(cropped)

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

    # search corner of phantom at top mid and btm mid part
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

    roi_dict = {'include_array': include_array, 'phantom': 34,
                'xs': xs, 'ys':ys,
                'cell_width': diagonal, 'angle': angle}
    return roi_dict


def find_phantom_40(image, px_pr_mm, angle_margin, angles_center, paramset):
    dist = None
    errmsgs = []
    roi_estimate = paramset.cdm_roi_estimate
    estimate_x, estimate_y = roi_estimate, roi_estimate
    rot90_k = 0

    # finding vertical line positions
    x_start_phantom, x_end_phantom, xprof = find_phantom_part(
        image, 5*px_pr_mm, axis=0)
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

    y_start_phantom, y_end_phantom, yprof = find_phantom_part(
        image, 5*px_pr_mm, axis=1)

    if estimate_y is False:
        peaks = find_peaks(yprof, distance=3.5*px_pr_mm,
                           height=0.5*np.max(yprof))
        peaks_pos = peaks[0]
        dists = np.diff(peaks_pos)
        dist = np.median(dists)
        #if dist is None:
        #    dist = 9 * px_pr_mm
        idx_sep = np.where(np.logical_and(dists < 1.05 * dist / 2, dists > 0.95 * dist / 2))
        if len(idx_sep[0]) == 3:
            if idx_sep[0][-1] > idx_sep[0][0]:  # todo accept also 90/-90 rotation
                rot90_k = 2  # rotate 180deg
                y_start = peaks_pos[0] + dist  # correct for one cell
            else:
                y_start = peaks_pos[0]
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
        xs, ys = np.meshgrid(
            dist * (np.arange(16) + 0.5),
            dist * (np.arange(21) + 0.5))
        if rot90_k == 2:
            a, b, c = 3, 7, 11
        else:
            a, b, c = 11, 15, 18
        ys[a:, :] = ys[a:, :] + 0.5 * dist
        ys[b:, :] = ys[b:, :] + 0.5 * dist
        ys[c:, :] = ys[c:, :] + 0.5 * dist
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

        if rot90_k == 2:  # set order of xs an ys similar to rot90_k = 0
            xs = np.rot90(xs, rot90_k)
            ys = np.rot90(ys, rot90_k)

        xs = xs + 16/2 * dist + x_start
        ys = ys + 22.5/2 * dist + y_start

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
                    'cell_width': dist, 'angle': diff_angle,
                    'rot90_k': rot90_k}
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


def get_templates(image, xs, ys, pix, pix_new,
                  cell_width, line_dist_50um, phantom, rot90_k, tables):
    """Find xy-offset of corner disc from sample cell with good visibility.

    Create templates where to search for minimum values.
    """
    wi = round(cell_width / 2)
    wi_factor = 1
    kernel = None

    corner_indexes = np.array(tables['corner_index'])
    corner_xys = []
    if phantom == 34:
        width_sub = wi * 2 + 1
        kernel = get_roi_circle((15, 15), (0, 0), 15)
        kernel = 1./np.sum(kernel) * kernel
        # find corners = top, left, right, bottom
        # estimate position from 2 first cells of rows with thickest discs
        # exact position seems related to which corner, not rotation symmetric
        corner_rows_cols = [[], [], [], []]  # for top, left, rgt, btm
        for row in range(4):  # find indexes and sort into which corner
            first_cols = np.where(corner_indexes[row] > 0)[0][:2]
            for col in first_cols:
                corner_rows_cols[corner_indexes[row][col]-1].append((row, col))
        for rows_cols in corner_rows_cols:  # find xy positions of the cells
            sample_xs, sample_ys = [], []
            for row, col in rows_cols:
                sample_xs.append(round(xs[row, col]))
                sample_ys.append(round(ys[row, col]))
            corner_xys.append((sample_xs, sample_ys))
        if rot90_k == 2:
            corner_xys.revers()
    else:  # v4.0
        wi_factor = 1.3
        wi = round(wi_factor * wi)  # add margin to cell
        width_sub = wi * 2 + 1

        '''Phantom v4.0 week contrast to finetune, better to estimate positions
        # find corners = top left, top right, btm right, btm left in image
        # estimate position from 3 first cells of with thickest discs
        corner_rows_cols = [[], [], [], []]  # for topleft, topr, btmr, btmleft
        for row in range(2):  # find indexes and sort into which corner
            for col in range(3):
                corner_rows_cols[corner_indexes[row][col]].append((row, col))
        corner_xys_in_image = []
        for rows_cols in corner_rows_cols:  # find xy positions of the cells
            sample_xs, sample_ys = [], []
            for row, col in rows_cols:
                sample_xs.append(round(xs[row, col]))
                sample_ys.append(round(ys[row, col]))
            corner_xys_in_image.append((sample_xs, sample_ys))

        if rot90_k == 0:  # text upside down
            order = [3, 0, 1, 2]  # corner 0 = btmleft (3)
        elif rot90_k == 2:  # text correct
            order = [1, 2, 3, 0]  # corner 0 = topright (1)
        for i in order:
            corner_xys.append(corner_xys_in_image[i])
        '''

    width_sub_50um = round(width_sub * pix / pix_new)
    cent = round(width_sub_50um/2)
    off_xs_center, off_ys_center = [], []
    off_xy_corners = []
    off_xy_expect = []
    if phantom == 34:
        diff_expect = width_sub_50um // 4
        off_xy_expect = [(0, -1), (-1, 0), (1, 0), (0, 1)]  # top,lft,rgt,btm
        off_xy_expect = diff_expect * np.array(off_xy_expect)
    elif phantom == 40:
        diff_expect = 47  # estimated visually from samples
        off_xy_expect = [(-1, -1), (1, -1), (1, 1), (-1, 1)]  # topl,topr,btmr,btml
        off_xy_expect = diff_expect * np.array(off_xy_expect)

    for cc, (x, y) in enumerate(corner_xys):
        off_xs_corner, off_ys_corner = [], []
        for i in range(len(x)):
            sub = image[y[i] - wi:y[i] + wi, x[i] - wi:x[i] + wi]
            sub = resize(sub, (width_sub_50um, width_sub_50um),
                         anti_aliasing=True)
            dx, dy = finetune_center_cell(sub, phantom, line_dist_50um)
            sub = np.roll(sub, (-dy, -dx), axis=(0, 1))

            if phantom == 34:
                avgs_sub = fftconvolve(sub, kernel, mode='same')
                roi_center = get_roi_circle(avgs_sub.shape, (0, 0), 30)
                roi_corner = get_roi_circle(
                    avgs_sub.shape, off_xy_expect[cc], 30)
                min_yx_center, _ = mmcalc.get_min_max_pos_2d(
                    avgs_sub, roi_center)
                min_yx_corner, _ = mmcalc.get_min_max_pos_2d(
                    avgs_sub, roi_corner)
            '''
            else:
                # avoid grid-lines to affect smoothing
                sub_masked = np.full(sub.shape, np.median(sub))
                rim = round(1.2* width_sub_50um * (wi_factor - 1) / 2)
                sub_masked[rim:-rim, rim:-rim] = sub[rim:-rim, rim:-rim]

                smooth = ndimage.gaussian_filter(sub_masked, sigma=20)
                roi_center = get_roi_circle(smooth.shape, (0, 0), 35)
                roi_corner = get_roi_circle(smooth.shape, 
                                            off_xy_expect[cc], 35)
                min_yx_center, _ = mmcalc.get_min_max_pos_2d(
                    smooth, roi_center)
                min_yx_corner, _ = mmcalc.get_min_max_pos_2d(
                    smooth, roi_corner)
            '''
            off_xs_center.append(min_yx_center[1] - cent)
            off_ys_center.append(min_yx_center[0] - cent)
            off_xs_corner.append(min_yx_corner[1] - cent)
            off_ys_corner.append(min_yx_corner[0] - cent)

        off_xy_corners.append(
            (round(np.mean(off_xs_corner)), round(np.mean(off_ys_corner))))

    if len(off_xy_corners) == 0 and phantom == 40:
        if phantom == 40:
            if rot90_k == 2:  # text upside down
                order = [3, 0, 1, 2]  # corner 0 = btmleft (3)
            elif rot90_k == 0:  # text correct
                order = [1, 2, 3, 0]  # corner 0 = topright (1)
            for i in order:
                off_xy_corners.append(off_xy_expect[i])

    if len(off_xs_center) > 0:
        off_xy_center = (round(np.mean(off_xs_center)),
                         round(np.mean(off_ys_center)))
    else:
        off_xy_center = (0, 0)

    # create the templates
    radius_search = 3  # 150 um cirle radius
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

    return templates, wi


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
            peaks_this = find_peaks(prof, height=0.5*np.max(prof))
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


def read_cdmam_sub(sub, phantom, line_dist, disc_templates, kernel):
    """Calculate results for sub in CDMAM34."""
    sub = sub - np.min(sub)
    sub = resize(sub, disc_templates[0].shape, anti_aliasing=True)
    dx, dy = finetune_center_cell(sub, phantom, line_dist)
    sub = np.roll(sub, (-dy, -dx), axis=(0, 1))

    ld, ks = 0, 0
    if phantom == 34:
        sub[disc_templates[-1] == True] = np.median(sub)
        avgs_sub = fftconvolve(sub, kernel, mode='same')
    elif phantom == 40:
        cent = round(sub.shape[0] // 2)
        ld = line_dist
        sub_cropped = sub[cent-ld:cent+ld,cent-ld:cent+ld]
        avgs_sub = fftconvolve(sub_cropped, kernel, mode='valid')
        ks = round(kernel.shape[1] // 2)

    minvals = []
    minpos = []
    for template in disc_templates[:5]:
        if phantom == 34:
            min_yx, _ = mmcalc.get_min_max_pos_2d(
                avgs_sub, template)#np.roll(template, (dx, dy), axis=(1,0)))
            minpos.append(min_yx)
        elif phantom == 40:
            start = cent - ld + ks
            end = start + avgs_sub.shape[0]
            min_yx, _ = mmcalc.get_min_max_pos_2d(
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

    return {'corner_index': idx_min,
            'central_disc_found': central_is_min,
            'min_positions': minpos,
            'processed_sub': avgs_sub}