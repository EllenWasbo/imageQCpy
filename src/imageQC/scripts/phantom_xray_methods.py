# -*- coding: utf-8 -*-
"""
Collection of methods for phantom-specific calculations xray.

@author: ellen
"""
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve, find_peaks, medfilt2d
#from skimage.transform import (
#    hough_circle, hough_circle_peaks, resize, rescale,
#    warp_polar, warp, SimilarityTransform)
from skimage import feature, filters, draw, transform
from skimage.registration import phase_cross_correlation

import matplotlib.pyplot as plt


# imageQC block start
from imageQC.scripts.calculate_roi import (
    get_roi_circle, get_roi_rectangle, find_edges, find_intercepts,
    find_rectangle_object, find_rectangle_object_2)
import imageQC.scripts.mini_methods_calculate as mmcalc
# imageQC block end


def calculate_phantom_xray(image, image_info, roi_array, paramset):

    if paramset.pha_type == 0:
        res = calculate_tor(image, image_info, roi_array, paramset)
    return res


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
    hspace, theta, d = transform.hough_line(binary_sub, theta=tested_angles)
    threshold_peaks = threshold_peaks * np.max(hspace)
    _, angles, dists = transform.hough_line_peaks(hspace, theta, d,
                                        threshold=threshold_peaks,
                                        min_distance=margin)
    return (angles, dists)


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
        ang_list > np.pi/2, ang_list < 3*np.pi/2))[0]
    if neg_x.size > 0:
        cx[neg_x] = -cx[neg_x]
    cy = - cx * tanang
    pos_y = np.where(np.logical_and(
        ang_list > np.pi, ang_list < 2*np.pi))[0]
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
    #res = mmcalc.find_center_object(sub, sigma=3)
    #if res:
    #    cx, cy, width_x, width_y = res
    #    if cx is not None and cy is not None:
    #        cx, cy = (cx - sub.shape[1] / 2, cy - sub.shape[0] / 2)
    #    else:
    #        cx, cy = (0, 0)
    #else:
    cx, cy = (0, 0)

    dists = mmcalc.get_distance_map_point(
        sub.shape, center_dx=cx, center_dy=cy)
    inner = np.where(dists < radius_disc * 0.7)
    inner_mean = np.mean(sub[inner])
    inner_std = np.std(sub[inner])
    sub_roi_inner = np.zeros(sub.shape, dtype=bool)
    sub_roi_inner[inner] = True
    
    outer = np.where(np.logical_and(
        dists > radius_disc * 1.2, dists < radius_disc * 1.5))
    outer_mean = np.mean(sub[outer])
    outer_std = np.std(sub[outer])
    sub_roi_outer = np.zeros(sub.shape, dtype=bool)
    sub_roi_outer[outer] = True

    cnr = np.abs(inner_mean - outer_mean) / outer_std
    res = {
        'cnr': cnr, 'inner_mean': inner_mean, 'inner_std': inner_std,
        'outer_mean': outer_mean, 'outer_std': outer_std,
        'center_xy': np.array([cx, cy]) + sub.shape[0] // 2,
        'roi_sub_inner': sub_roi_inner, 'roi_sub_outer': sub_roi_outer}
    return res


def calculate_tor(image, image_info, roi_array, paramset):
    """Calculate low contrast and spatial resolution."""
    errmsgs = []
    values = []
    details_dict = {}

    if roi_array[0] is None:
        errmsgs = 'High contrast object within bar pattern not found.'
    else: # high contrast at bar pattern found
        pix = image_info.pix[0]
        pix_pr_mm = round(1./image_info.pix[0])
        sz_y, sz_x = image.shape
        invert = False  # indicator to invert image if highcontrast square higher values than background

        # use part with highcontrast square to find rotation of square and phantom
        cx = round(np.mean(roi_array[1][:-1,1]))
        cy = round(np.mean(roi_array[1][:-1,0]))
        xs = [cx - 10*pix_pr_mm, cx + 10*pix_pr_mm]
        ys = [cy - 10*pix_pr_mm, cy + 10*pix_pr_mm]
        sub = image[ys[0]:ys[1], xs[0]:xs[1]]
        sub_var = mmcalc.get_variance_map(sub, round(0.5/pix), 'valid')
        sub_var_polar, (rads, angs) = mmcalc.topolar(sub_var)
        edge_pos = []
        for row in np.transpose(sub_var_polar):
            width, center = mmcalc.get_width_center_at_threshold(
                row, force_above=True)
            edge_pos.append(center)
        ups = np.where(np.diff(edge_pos) > 0)[0]
        pos_min_dist = np.where(np.diff(ups) > 5)[0] + 1
        pos_min_dist = ups[pos_min_dist]
        if ups[0] > 0:
            pos_min_dist = np.insert(pos_min_dist, 0, ups[0])
        # min_dists = np.array(edge_pos)[pos_min_dist]
        min_angs = angs[pos_min_dist]  # expected 4 angles where each + 90

        # find which quadrant is where and flipping
        sz_square = np.sqrt( # dist between first two corners in roi_array[1]
            (roi_array[1][1,0] - roi_array[1][0,0])**2
            + (roi_array[1][1,1] - roi_array[1][0,1])**2)
        stds = []
        medians = []
        for i in range(4):
            xx = round(sz_square * np.cos(min_angs[0]+i*np.pi/2))
            yy = - round(sz_square * np.sin(min_angs[0]+i*np.pi/2))
            mask = np.roll(1.0*roi_array[0], (yy, xx), axis=(0, 1))
            xs = [cx + xx - 10*pix_pr_mm, cx + xx + 10*pix_pr_mm]
            ys = [cy + yy - 10*pix_pr_mm, cy + yy + 10*pix_pr_mm]
            sub = (image * mask)[ys[0]:ys[1], xs[0]:xs[1]]
            stds.append(np.std(sub[sub != 0]))
            medians.append(np.median(sub[sub != 0]))
        quadrant_bars = np.where(stds == np.max(stds))[0][0]
        possible_ref = [0, 1, 2, 3]
        possible_ref.pop(quadrant_bars)
        try:
            possible_ref.pop(quadrant_bars + 2)
        except IndexError:
            possible_ref.pop(quadrant_bars - 2)
        if medians[quadrant_bars] > 0.5 * (
                medians[possible_ref[0]] + medians[possible_ref[1]]):
            invert = True
            image = - image
        if medians[possible_ref[0]] > medians[possible_ref[1]]:
            quadrant_ref_square = possible_ref[0]
        else:
            quadrant_ref_square = possible_ref[1]
        #if quadrant_bars == np.roll([0,1,2,3], 1)[quadrant_ref_square]:
        #    flip = True
        ref_max = np.max(medians)

        # find background
        i = quadrant_ref_square
        xx = round(sz_square * 2.5 * np.cos(min_angs[0]+i*np.pi/2))
        yy = - round(sz_square * 2.5 * np.sin(min_angs[0]+i*np.pi/2))
        mask = np.roll(1.0*roi_array[0], (yy, xx), axis=(0, 1))
        xs = [cx + xx - 10*pix_pr_mm, cx + xx + 10*pix_pr_mm]
        ys = [cy + yy - 10*pix_pr_mm, cy + yy + 10*pix_pr_mm]
        sub = (image * mask)[ys[0]:ys[1], xs[0]:xs[1]]
        ref_min = np.median(sub[sub != 0])
        # find center of ref max square along profile
        rr, cc = draw.line(cy, cx, cy + yy, cx + xx)
        profile = image[rr, cc]
        start = round(1.2* sz_square // 2)
        ref_max_part = profile[start:]
        _, center = mmcalc.get_width_center_at_threshold(ref_max_part)
        pos = round(center + start)
        cx_max, cy_max = cc[pos], rr[pos]
        # find center of bar-pattern along 90 deg profile
        i = quadrant_bars
        xx = round(sz_square * 3 * np.cos(min_angs[0]+i*np.pi/2))
        yy = - round(sz_square * 3 * np.sin(min_angs[0]+i*np.pi/2))
        rr, cc = draw.line(cy_max - yy, cx_max - xx, cy_max + yy, cx_max + xx)
        profile = image[rr, cc]
        peaks = find_peaks(
            profile, distance=sz_square,
            height=0.5*(np.max(profile)+np.min(profile)))
        center_bar_pos = peaks[0][-1]
        cx_bar, cy_bar = cc[center_bar_pos], rr[center_bar_pos]

        # finetune rotation of bars (fineture rotation)
        dd = round(sz_square)
        sub = image[cy_bar - dd:cy_bar + dd, cx_bar - dd:cx_bar + dd]
        theta = np.linspace(0.0, 180.0, 180, endpoint=False)
        sinogram = transform.radon(sub, theta=theta)
        profile_variances = np.var(sinogram, axis=0)
        dominant_angle_deg = theta[np.argmax(profile_variances)]

        # create template for bar pattern unit = cm = sz_square
        dx = [1.8, 0.6, -0.7]  # centers of 3 columns, sorted from high to low freq
        dy = [1.3, 0.35, -0.35, -0.85, -1.22, -1.47, -1.63]  # centers of 7 rows
        h = [0.8, 0.5, 0.3, 0.22, 0.20, 0.10, 0.05] # height of ROI each row
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
        roi = ndimage.rotate(temp_bars, dominant_angle_deg + 90,
                             order=0, reshape=False) # TODO check with different flipping/rotation for original

        # get values from image
        sub = image[cy_bar - ss:cy_bar + ss + 1, cx_bar -ss:cx_bar + ss + 1]
        roi_full = np.zeros(image.shape)
        roi_full[cy_bar - ss:cy_bar + ss + 1, cx_bar -ss:cx_bar + ss + 1] = roi
        ctfs = []
        mins = []
        maxs = []
        for i in range(7*3):
            arr = np.ma.masked_array(sub, mask=roi != i+1)
            mina = np.min(arr)
            maxa = np.max(arr)
            mins.append(mina)
            maxs.append(maxa)
            ctfs.append((maxa - mina) / (maxa + mina))

        details_dict = {
            'spatial_frequencies': [0.5, 0.56, 0.63,
                                    0.71, 0.8, 0.9,
                                    1., 1.12, 1.25,
                                    1.4, 1.6, 1.8,
                                    2., 2.24, 2.5,
                                    2.8, 3.15, 3.55,
                                    4., 4.5, 5],
            'ctf_values': ctfs,
            'min_values':  mins,
            'max_values': maxs,
            'roi_ctfs': roi_full,
            }
        values = [None, None]

        '''
        # Contrast circles (if present)

        # cnr circle positions in template
        angles = (np.arange(24) + 2) * (2*np.pi)/24  # 24 circles if all were there
        angles = angles - np.deg2rad(angle)  # correct for rotation relative to template
        # cnr pos, distance 60 mm relative ti image center (shape//2)
        c_phantom_x, c_phantom_y = 0,0
        xys = [((60./pix * np.sin(ang)) + c_phantom_x,
                (60./pix * np.cos(ang)) + c_phantom_y)
               for i, ang in enumerate(angles)
               if i not in [9, 10, 11, 21, 22, 23]]
        cnr_radii = [25., 100., 130.]  # roi disc, inner/outer rim in mm

        details_dict.update({
            'contrast_percents': [
                16.7, 14.8, 12.8, 10.9, 8.8, 7.5, 6.7, 5.3, 4.5,
                3.9, 3.2, 2.7, 2.2, 1.7, 1.5, 1.3, 1.1, 0.9],
            'cnr_xys': xys,
            'cnr_radii': cnr_radii,
            })

        cnr_circles = [get_roi_circle(image.shape, xy, cnr_radii[0]/pix)
                       for xy in xys]
        cnr_rims = [np.logical_xor(
            get_roi_circle(image.shape, xy, cnr_radii[2]/pix),
            get_roi_circle(image.shape, xy, cnr_radii[1]/pix)) for xy in xys]
        # generate template for 0.1mm pix size and 150 mm matrix
        # resize to actual pixelsize and rotate/flip according to found squares

        # find contrast circle with highest contrast
        phantom_scale = radius / radius_large
        radius_small = 4./pix * phantom_scale  # contrast circles 8mm
        # center to contrast circles = 60mm, margin 2mm
        start, end = (phantom_scale / pix) * np.array([58, 62])
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

        # find cnr pr disc
        roi_inner = np.zeros(image.shape, dtype=bool)
        roi_outer = np.zeros(image.shape, dtype=bool)
        if angles is not None:
            res_pr_disc = []
            dist = 60 * (phantom_scale / pix)
            eval_angles = angles + zero_ang
            print(f'eval angels {eval_angles}')
            cx, cy = get_xy_from_angles_dist(eval_angles, dist)
            for i, ang in enumerate(eval_angles):
                dx_dy = np.array([cx[i], cy[i]])
                roi = get_roi_circle(
                    image.shape, off_xy + dx_dy, radius_small*2)
                rows = np.max(roi, axis=1)
                cols = np.max(roi, axis=0)
                sub = image[rows][:, cols]
                res = find_cnr_from_sub(sub, radius_small)
                res['center_xy'] = res['center_xy'] + off_xy + dx_dy
                res_pr_disc.append(res)

                x0, x1 = np.where(cols)[0][0], np.where(cols)[0][-1]+1
                y0, y1 = np.where(rows)[0][0], np.where(rows)[0][-1]+1
                roi_inner[y0:y1, x0:x1] += res['roi_sub_inner']
                roi_outer[y0:y1, x0:x1] += res['roi_sub_outer']
        '''

    return (details_dict, values, errmsgs)

def match_to_template(
        image, center_object_xy=(0,0), pix_image=0,
        template=None, center_object_template_xy=(0,0), pix_template=0,
        dist_from_object_mm=0):
    '''Find transformation matrix for template to match to image.

    Parameters
    ----------
    image : np.array (2d)
    center_object_xy : tuple
        center y, x of found object in image
    pix_image : float
        mm pr pixel for image
    template : np.array (2d)
        with similar image content as image
    center_object_template_xy:
        center ym x of found object in template
    pix_template : float
        mm pr pixel for template
    dist_from_object_mm : float
        maximum distance from center of found object to include in analysis

    Returns
    -------
    scale: float
        scaling factor for template to fit image
    rotation
        rotation of template around center to fit image
    '''
    def center_pad_and_crop(img, cx, cy, dist):
        """Center img in (cx, cy) and cut such that quadratic with size 2*dist."""
        cx, cy, dist = round(cx), round(cy), round(dist)
        img_res = np.zeros((dist*2 + 1, dist*2 + 1), dtype=float) + np.median(img)
        start_x, end_x = cx - dist, cx + dist
        st_x, st_y, et_x, et_y = 0, 0, dist*2, dist*2
        if start_x < 0:
            st_x = - start_x
            start_x = 0
        if end_x > img.shape[1]:
            end_x = img.shape[1]
            et_x = st_x + (end_x - start_x)
        start_y, end_y = cy - dist, cy + dist
        if start_y < 0:
            st_y = - start_y
            start_y = 0
        if end_y > img.shape[0]:
            end_y = img.shape[0]
            et_y = st_y + (end_y - start_y)

        img_res[st_y:et_y, st_x:et_x] = img[start_y:end_y, start_x:end_x]
        return img_res

    # center images on center of object and resize to similar sizes
    d_img_pix = dist_from_object_mm / pix_image
    c_image = center_pad_and_crop(
        image, center_object_xy[0], center_object_xy[1], d_img_pix)
    d_temp_pix = dist_from_object_mm / pix_template
    c_temp = center_pad_and_crop(
        template, center_object_template_xy[0],
        center_object_template_xy[1], d_temp_pix)
    scale = c_image.shape[0] / c_temp.shape[0]
    c_temp = transform.resize(
        c_temp, c_image.shape, anti_aliasing=True)

    # based on
    # https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html
    c_image = filters.difference_of_gaussians(c_image, 1, 5)
    c_temp = filters.difference_of_gaussians(c_temp, 1, 5)
    c_image = c_image * filters.window('hann', c_image.shape)
    c_temp = c_temp * filters.window('hann', c_temp.shape)
    f_image = np.abs(np.fft.fftshift(np.fft.fft2(c_image)))
    f_temp = np.abs(np.fft.fftshift(np.fft.fft2(c_temp)))

    # to polar to make rotation shift a translation shift in fourier space
    radius = f_image.shape[0] // 2 # limit radius of warp to be largest full circle
    w_image = transform.warp_polar(f_image, radius=radius)
    w_temp = transform.warp_polar( f_temp, radius=radius)

    shifts, error, phasediff = phase_cross_correlation(
        w_image, w_temp, upsample_factor=10)
    rotation = shifts[0]

    return (scale, rotation)

'''
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    ax = axes.ravel()
    ax[0].imshow(c_image)
    ax[1].imshow(c_temp)
    ax[2].imshow(w_image)
    ax[3].imshow(w_temp)
    f_image[f_image.shape[0]//2-7:f_image.shape[0]//2+7,
            f_image.shape[1]//2-7:f_image.shape[1]//2+7] = 0
    f_temp[f_temp.shape[0]//2-7:f_temp.shape[0]//2+7,
           f_temp.shape[1]//2-7:f_temp.shape[1]//2+7] = 0
    ax[4].imshow(f_image)
    ax[5].imshow(f_temp)
    plt.show()

    # Calculate rotation and scaling from translation params
    angle_deg = shifts[0] * 180. / w_image.shape[0]
    k_log = output_shape[1] / np.log(radius)
    estimated_scale = np.exp(shifts[1] / k_log)

    # continued with help of Gemini
    center_temp = np.array(template.shape) // 2
    center_img = np.array(image.shape) // 2
    best_score = -np.inf
    best_inv_transform = None
    # cannot distinguish between angle and angle + 180 degrees
    candidate_angles = [angle_deg, angle_deg + 180.0]

    for cand_angle in candidate_angles:
        rad = np.deg2rad(cand_angle)
        t_to_center = transform.AffineTransform(
            translation=-center_temp[::-1]) # shift template origin to center
        t_scale_rot = transform.AffineTransform(
            scale=(1.0/estimated_scale, 1.0/estimated_scale),
            rotation = rad)
        t_from_center = transform.AffineTransform(
            translation=center_img[::-1])  # shift to image center
        inv_transform = t_to_center + t_scale_rot + t_from_center
        temp_rot_scaled = transform.warp(
            template, inv_transform, output_shape=image.shape)

        # fix shifts back from log-polar to spatial
        spatial_shifts, error, _ = phase_cross_correlation(
            image_g, filters.difference_of_gaussians(
                temp_rot_scaled, 1, 5),
            upsample_factor=10)
        score = -error
        print(f'score {score} angle {cand_angle}')
        if score > best_score:
            
            best_score = score
            best_angle = cand_angle
            shift_y, shift_x = spatial_shifts
            t_final_from_center = transform.AffineTransform(
                translation=center_img[::-1] - np.array([shift_x, shift_y]))
            best_inv_transform = t_to_center + t_scale_rot + t_final_from_center
    temp_fitted = transform.warp(
        template, best_inv_transform, output_shape=image.shape)

    breakpoint()

    fig, axes = plt.subplots(1, 3, figsize=(10, 8))
    ax = axes.ravel()
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(template, cmap='gray')
    ax[2].imshow(temp_fitted, cmap='gray')
    plt.show()
    plt.pause(.1)
'''