#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of methods for MTF calculations in ImageQC.

@author: Ellen Wasbo
"""

import numpy as np
import scipy as sp

# imageQC block start
import imageQC.scripts.mini_methods as mm
import imageQC.scripts.mini_methods_calculate as mmcalc
import imageQC.config.config_classes as cfc
# imageQC block end


def ESF_to_LSF(ESF, prefilter_sigma):
    """Calculate LSF spread function from edge spread function.

    Parameters
    ----------
    ESF : np.1darray
    prefilter_sigma : float
        gaussian filter to be corrected when gaussian MTF is used

    Returns
    -------
    LSF : np.1darray
        line spread function after gaussian filter - for gaussian fit
    LSF_not_filtered : np.1darray
        line spread function without gaussian prefilter - for discrete MTF
    ESF_filtered : np.1darray
        ESF gaussian smoothed
    """
    values_f = mmcalc.gaussian_filter(ESF, sigma=prefilter_sigma)
    LSF = np.diff(values_f)
    LSF_not_filtered = np.diff(ESF)
    if np.abs(np.min(LSF)) > np.abs(np.max(LSF)):  # ensure positive gauss
        LSF = -1. * LSF
        LSF_not_filtered = -1. * LSF_not_filtered

    return (LSF, LSF_not_filtered, values_f)


def cut_and_fade_LSF(profile, center=0, fwhm=0, cut_width=0, fade_width=0):
    """Cut and fade profile fwhm from center to reduce noise in LSF.

    Parameters
    ----------
    profile : list of float
    center : float
        center pos in pix
    fwhm : float
        fwhm in pix
    cut_width : float
        cut profile cut_width*fwhm from halfmax. If zero, also fade width ignored.
    fade_width : float (or None)
        fade profile to background fade*fwhm inside cut_width

    Returns
    -------
    modified_profile : list of float
    cut_dist_pix : float
        distance from center where profile is cut (unit = pix)
    fade_dist_pix : float
        distance from center where profile starts fading (unit = pix)
    """
    modified_profile = profile
    n_fade = 0
    if cut_width > 0:
        cut_width += 0.5  # from halfmax = 0.5 fwhm
        dx = cut_width*fwhm
        first_x = max(round(center - dx), 0)
        last_x = min(round(center + dx), len(profile) - 1)
        if fade_width is None:
            fade_width = 0
        if fade_width > 0:
            fade_width = min(fade_width, cut_width)
            n_fade = round(fade_width*fwhm)
            first_fade_x = first_x + n_fade
            last_fade_x = last_x - n_fade
            nn = first_fade_x - first_x
            gradient = np.multiply(
                np.arange(nn)/nn, profile[first_x:first_fade_x])
            modified_profile[first_x:first_fade_x] = gradient
            nn = last_x - last_fade_x
            gradient = np.multiply(
                np.flip(np.arange(nn)/nn), profile[last_fade_x:last_x])
            modified_profile[last_fade_x:last_x] = gradient
        if first_x > 0:
            modified_profile[0:first_x] = 0
        if last_x < len(profile) - 2:
            modified_profile[last_x:] = 0

    return (modified_profile, dx, dx - n_fade)


def get_MTF_gauss(LSF, dx=1., prefilter_sigma=None, gaussfit='single'):
    """Fit LSF to gaussian and calculate gaussian MTF.

    Parameters
    ----------
    LSF : np.array
        LSF 1d
    dx : float, optional
        step_size, default is 1.
    prefilter_sigma : float, optional
        prefiltered with gaussian filter, sigma=prefilter_sigma. The default is None.
    gaussfit : str, optional
        single or double or double_allow_both_positive (sum of two gaussian).
        double assume by default 1 negative + 1 positive gauss.
        double_both_positive demand both positive.
        The default is 'single'.

    Returns
    -------
    dict
        with calculation details
    str
        error message
    """
    popt = None
    details = {}
    errmsg = None
    width, center = mmcalc.get_width_center_at_threshold(LSF, 0.5 * max(LSF))
    if center is None or width is None:
        errmsg = (
            'Failed finding center and/or width of LSF. '
            'Unexpected input data?')
    else:
        LSF_x = dx * (np.arange(LSF.size) - center)
        A2 = None
        sigma2 = None
        if gaussfit == 'double':
            popt = mmcalc.gauss_double_fit(LSF_x, LSF, fwhm1=width * dx)
            if popt is not None:
                if len(popt) == 4:
                    A1, sigma1, A2, sigma2 = popt
                    LSF_fit = mmcalc.gauss_double(LSF_x, *popt)
                else:
                    A1, sigma1 = popt
                    LSF_fit = mmcalc.gauss(LSF_x, *popt)
        elif gaussfit == 'double_both_positive':
            popt = mmcalc.gauss_double_fit(
                LSF_x, LSF, fwhm1=width * dx, A2_positive=True)
            if popt is not None:
                A1, sigma1, A2, sigma2 = popt
                LSF_fit = mmcalc.gauss_double(LSF_x, *popt)
        else:  # single
            popt = mmcalc.gauss_fit(LSF_x, LSF)
            if popt is not None:
                LSF_fit = mmcalc.gauss(LSF_x, *popt)
                A1, sigma1 = popt

        if popt is None:
            errmsg = 'Failed fitting LSF to gaussian.'
        else:
            n_steps = 200  # sample 20 steps from 0 to 1 stdv MTF curve (stdev = 1/sigma1)
            # TODO user configurable n_steps
            k_vals = np.arange(n_steps) * (10./n_steps) / sigma1
            MTF = mmcalc.gauss(k_vals, A1*sigma1, 1/sigma1)
            if A2 is not None and sigma2 is not None:
                F2 = mmcalc.gauss(k_vals, A2*sigma2, 1/sigma2)
                MTF = np.add(MTF, F2)
            MTF_filtered = None
            if prefilter_sigma is None:
                prefilter_sigma = 0
            if prefilter_sigma > 0:
                F_filter = mmcalc.gauss(k_vals, 1., 1/prefilter_sigma)
                MTF_filtered = 1/MTF[0] * MTF  # for display
                MTF = np.divide(MTF, F_filter)
            MTF = 1/MTF[0] * MTF
            k_vals = k_vals / (2*np.pi)

            fwhm = None
            fwtm = None
            LSF_corrected = None
            if gaussfit == 'single':
                if prefilter_sigma == 0:
                    fwhm = 2*np.sqrt(2*np.log(2))*sigma1
                    fwtm = 2*np.sqrt(2*np.log(10))*sigma1
                else:
                    # Gaussian fit of the corrected MTF to get fwhm/fwtm
                    k_vals_symmetric = np.concatenate((-k_vals[::-1], k_vals[1:]))
                    MTF_symmetric = np.concatenate((MTF[::-1], MTF[1:]))
                    poptMTF = mmcalc.gauss_fit(k_vals_symmetric, MTF_symmetric)
                    if poptMTF is not None:
                        _, sigmaMTF = poptMTF
                        LSF_corrected = mmcalc.gauss(
                            LSF_x, np.max(LSF_fit), 1/(2*np.pi*sigmaMTF))
                        LSF_sigma = 1./(2*np.pi*sigmaMTF)
                        fwhm = 2*np.sqrt(2*np.log(2))*LSF_sigma
                        fwtm = 2*np.sqrt(2*np.log(10))*LSF_sigma

            details = {
                'LSF_fit_x': LSF_x, 'LSF_fit': LSF_fit, 'LSF_prefit': LSF,
                'LSF_corrected': LSF_corrected,
                'LSF_fit_params': popt, 'prefilter_sigma': prefilter_sigma,
                'MTF_freq': k_vals, 'MTF': MTF, 'MTF_filtered': MTF_filtered,
                'FWHM': fwhm, 'FWTM': fwtm,
                }

    return (details, errmsg)


def get_MTF_discrete(LSF, dx=1, padding_factor=1):
    """Fourier transform of vector with zero-padding.

    Parameters
    ----------
    LSF : np.array of floats
        line spread function
    dx : float
        stepsize in LSF
    padding_factor: float, optional
        zero-pad each side by padding-factor * length of LSF. Default is 1.

    Returns
    -------
    freq : np.array of float
        frequency axis of MTF
    MTF : np.array of float
        modulation transfer function
    """
    nLSF = round(padding_factor * LSF.size)
    LSF = np.pad(LSF, pad_width=nLSF, mode='constant', constant_values=0)
    MTF = np.abs(np.fft.fft(LSF))
    freq = np.fft.fftfreq(LSF.size, d=dx)
    MTF = MTF[:MTF.size//2]
    freq = freq[:freq.size//2]
    MTF = MTF/MTF[0]

    return {'MTF_freq': freq, 'MTF': MTF}


def get_NEMA_spatial(profiles):
    """Calculate FWHM, FWTM from profiles as defined in NEMA PET 2024.

    Parameters
    ----------
    profiles : list of np.array or list
        [x, y (), z)] profiles through max of point
        assumed background subtracted (background = zero)

    Returns
    -------
    
    """
    def get_parabolic_fit_max(yvalues):
        """Calculate max from fit of three pixelvalues."""
        A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])
        b = np.array(yvalues)
        abc = np.linalg.solve(A, b)
        x_max = -abc[1]/(2*abc[0])
        y_max = abc[0] * (x_max**2) + abc[1] * x_max + abc[2]
        return y_max, abc

    modified_profiles = []
    fwhm_values = []
    fwtm_values = []
    for profile in profiles:
        # find max from parabolic fit of max and the two neighbours
        max_pos = np.where(profile == np.max(profile))[0][0]
        max_fit, abc = get_parabolic_fit_max(profile[max_pos-1:max_pos+2])
        # find interpolated fwhm/fwtm
        fwhm, center = mmcalc.get_width_center_at_threshold(
                profile, threshold=max_fit/2)
        fwtm, _ = mmcalc.get_width_center_at_threshold(
                profile, threshold=max_fit/10)
        fwhm_values.append(fwhm)
        fwtm_values.append(fwtm)
        # additional points to the profile -  parabolic fit (x10 res) and fwhm
        res10 = 0.1 * (np.arange(21) - 10)
        parabolic = [abc[0] * (x**2) + abc[1] * x + abc[2] for x in res10]
        res10 = list(res10 + max_pos)
        x_values = ([center - fwtm / 2, center - fwhm / 2]
                    + res10
                    + [center + fwhm / 2, center + fwtm / 2])
        y_values = [max_fit/10, max_fit/2] + parabolic + [max_fit/2, max_fit/10]
        orig_x_values = np.arange(profile.size)
        #x_values = np.append(x_values, orig_x_values[:max_pos-1])
        #x_values = np.append(x_values, orig_x_values[max_pos+2:])
        x_values = x_values - max_pos
        #order = np.argsort(x_values)
        #x_values = x_values[order]
        #y_values = np.append(y_values, profile[:max_pos-1])
        #y_values = np.append(y_values, profile[max_pos+2:])
        #y_values = y_values[order]
        modified_profiles.append([x_values, y_values])

    return modified_profiles, fwhm_values, fwtm_values


def calculate_MTF_point(matrix, img_info, paramset, vertical_pix_mm=None):
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
    vertical_pix_mm : float, optional
        if vertical pix size != pix
        (summed 3d where pix_vertical = slice increment)

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
            as returned by get_MTF_discrete
            + values [MTF 50%, 10%, 2%]
        gMTF_details: dict
            as returned by get_MTF_gauss
            + values [MTF 50%, 10%, 2%]
    errmsgs : list of str
    """
    details = []
    errmsgs = []
    for ax in [0, 1]:
        details_dict = {}
        profile = np.sum(matrix, axis=ax)
        width, center = mmcalc.get_width_center_at_threshold(
            profile, np.max(profile)/2)
        if center is None:
            errmsgs.append('Could not find center of point in ROI.')
        else:
            if ax == 1 and vertical_pix_mm is not None:
                pix = vertical_pix_mm
            else:
                pix = img_info.pix[0]
            pos = (np.arange(len(profile)) - center) * pix

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
                profile, cw, cwf = cut_and_fade_LSF(
                    profile, center=center, fwhm=width,
                    cut_width=cut_lsf_fwhm, fade_width=fade_lsf_fwhm)

            details_dict['center'] = center
            details_dict['LSF_x'] = pos
            details_dict['LSF'] = profile

            # Discrete MTF
            res = get_MTF_discrete(profile, dx=pix)
            res['cut_width'] = cw * pix
            res['cut_width_fade'] = cwf * pix
            if isinstance(paramset, cfc.ParamSetCT):
                res['values'] = mmcalc.get_curve_values(
                    res['MTF_freq'], res['MTF'], [0.5, 0.1, 0.02])
            else:  # NM, SPECT or PET
                fwtm, _ = mmcalc.get_width_center_at_threshold(
                    profile, np.max(profile)/10)
                if width is not None:
                    width = width * pix
                if fwtm is not None:
                    fwtm = fwtm * pix
                res['values'] = [width, fwtm]
            details_dict['dMTF_details'] = res

            # Gaussian MTF
            if isinstance(paramset, cfc.ParamSetCT):
                gaussfit = 'double'
            else:  # NM, SPECT or PET
                gaussfit = 'single'
            res, err = get_MTF_gauss(
                profile, dx=pix, gaussfit=gaussfit)
            if err is not None:
                errmsgs.append(err)
            else:
                if isinstance(paramset, cfc.ParamSetCT):
                    res['values'] = mmcalc.get_curve_values(
                        res['MTF_freq'], res['MTF'], [0.5, 0.1, 0.02])
                else:  # NM, SPECT or PET
                    profile = res['LSF_fit']
                    fwhm, _ = mmcalc.get_width_center_at_threshold(
                        profile, np.max(profile)/2)
                    fwtm, _ = mmcalc.get_width_center_at_threshold(
                        profile, np.max(profile)/10)
                    if fwhm is not None:
                        fwhm = fwhm * pix
                    if fwtm is not None:
                        fwtm = fwtm * pix
                    res['values'] = [fwhm, fwtm]
                details_dict['gMTF_details'] = res

                details.append(details_dict)

    return (details, errmsgs)


def calculate_MTF_3d_point(matrix, roi, images_to_test, image_infos, paramset):
    """Calculate MTF from point/bead ~normal to slices.

    Parameters
    ----------
    matrix : numpy.2darray or list of numpy.array
        list of 2d part of slice limited ROI
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
    details_dict: list of dict
        x_dir, y_dir, common_details
    errmsg : list of str
    """
    details_dict = []
    common_details_dict = {}
    errmsg = []
    matrix = [sli for sli in matrix if sli is not None]
    zpos = np.array([image_infos[sli].zpos for sli in images_to_test])
    max_values = np.array([np.max(sli) for sli in matrix])
    common_details_dict['max_roi_marked_images'] = max_values
    common_details_dict['zpos_marked_images'] = zpos

    max_z = np.where(max_values == np.max(max_values))
    idx_max = max_z[0][0]

    z_diffs = np.diff(zpos)
    z_dist = z_diffs[0]
    if np.min(z_diffs) < 0.95 * np.max(z_diffs):  # allow for precision issues
        errmsg.append('NB: z-increment differ for the slices. '
                      'Could cause erroneous results.')

    margin = (paramset.mtf_roi_size + paramset.mtf_background_width) // z_dist
    zpos_used = None
    if margin < 2:
        errmsg.append(
            'ROI radius + background widht used to select images. '
            'Set this value larger than 2 * slice thickness.')
    else:
        margin = round(margin)
        matrix_this = None
        try:
            matrix_this = matrix[idx_max - margin:idx_max + margin + 1]
            zpos_used = zpos[idx_max - margin:idx_max + margin + 1]
            common_details_dict['zpos_used'] = zpos_used
            common_details_dict['max_slice_idx'] = idx_max
        except (IndexError, ValueError):
            pass
        if matrix_this is not None:
            if paramset.mtf_type == 5:
                for i in [0, 1]:
                    axis = 1 if i == 0 else 2
                    matrix_xz = np.sum(matrix_this, axis=axis)

                    details_dict_this, errmsg_this = calculate_MTF_point(
                        matrix_xz, image_infos[0], paramset,
                        vertical_pix_mm=np.mean(z_diffs))
                    details_dict.extend(details_dict_this)
                    errmsg.append(errmsg_this)
                details_dict.pop(1)  # removing first Z results (x, Z, y, z)

            # add sentral profiles
            max_slice = matrix_this[margin]
            max_pos = np.where(max_slice == np.max(max_slice))
            dx, dy = max_pos[1][0], max_pos[0][0]
            x_prof = max_slice[dy,:]
            y_prof = max_slice[:, dx]
            z_prof = np.array([sub[dy,dx] for sub in matrix_this])
            common_details_dict['profile_xyz'] = [x_prof, y_prof, z_prof]
            dxy = (np.arange(x_prof.size) - x_prof.size//2) * image_infos[0].pix[0]
            dz = zpos_used - zpos[idx_max]
            common_details_dict['profile_xyz_dist'] = [dxy, dxy, dz]

            if paramset.mtf_type == 6:
                profs = common_details_dict['profile_xyz']
                profs_mod, fwhms, fwtms = get_NEMA_spatial(profs)
                values = []
                for i in range(3):
                    pix = image_infos[0].pix[0] if i < 2 else np.mean(z_diffs)
                    values.extend([fwhms[i] * pix, fwtms[i] * pix])
                    max_pos = np.where(profs[i] == np.max(profs[i]))[0][0]
                    details_dict_this = {'center': max_pos}
                    details_dict.append(details_dict_this)
                common_details_dict['NEMA_widths'] = values
                profs_mod[0][0] = np.array(profs_mod[0][0]) * image_infos[0].pix[0]
                profs_mod[1][0] = np.array(profs_mod[1][0]) * image_infos[0].pix[0]
                profs_mod[2][0] = np.array(profs_mod[2][0]) * np.abs(dz[1]-dz[0])
                common_details_dict['NEMA_modified_profiles'] = profs_mod

    details_dict.append(common_details_dict)

    return (details_dict, errmsg)


def calculate_MTF_3d_line(matrix, roi, images_to_test, image_infos, paramset):
    """Calculate MTF from line ~normal to slices.

    Parameters
    ----------
    matrix : numpy.2darray or list of numpy.array
        list of 2d part of slice limited ROI
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
    details_dict: list of dict
        x_dir, y_dir, common_details
    errmsg : list of str
    """
    details_dict = []
    common_details_dict = {}
    errmsg = []
    matrix = [sli for sli in matrix if sli is not None]
    zpos = np.array([image_infos[sli].zpos for sli in images_to_test])
    max_values = np.array([np.max(sli) for sli in matrix])
    common_details_dict['max_roi_marked_images'] = max_values
    common_details_dict['zpos_marked_images'] = zpos

    ignore_slices = []
    if paramset.mtf_type < 3:  # not performed for z-resolution
        try:  # ignore slices with max outside tolerance
            tolerance = paramset.mtf_line_tolerance
            sort_idxs = np.argsort(max_values)
            max_n_highest = np.mean(
                max_values[sort_idxs[-3:]])
            diff = 100/max_n_highest * (np.array(max_values) - max_n_highest)
            idxs = np.where(np.abs(diff) > tolerance)
            ignore_slices = list(idxs[0])
        except AttributeError:
            pass

    proceed = True
    zpos_used = None
    if len(ignore_slices) > 0:
        if len(matrix) - len(ignore_slices) < 3:
            errmsg.append(
                'Need at least 3 valid images for this test. Found less.')
            proceed = False
        else:
            zpos_copy = np.copy(zpos)
            zpos = np.delete(zpos, ignore_slices)
            max_values_copy = np.copy(max_values)
            ignore_slices.reverse()
            for i in ignore_slices:
                if paramset.mtf_type == 1:  # not sliding window
                    del matrix[i]
                max_values_copy[i] = np.nan
                zpos_copy[i] = np.nan
            common_details_dict['max_roi_used'] = max_values_copy
            common_details_dict['zpos_used'] = zpos_copy
            zpos_used = zpos_copy

    if proceed:
        pix = image_infos[images_to_test[0]].pix[0]

        if paramset.mtf_type == 1:
            for i in [0, 1]:
                axis = 2 if i == 0 else 1
                matrix_xz = np.sum(matrix, axis=axis)

                details_dict_this, errmsg_this = calculate_MTF_2d_line_edge(
                    matrix_xz, pix, paramset, mode='line',
                    vertical_positions_mm=zpos_used, rotate='no')
                details_dict.append(details_dict_this)
                errmsg.append(errmsg_this)
        elif paramset.mtf_type == 2:  # sliding window
            n_slices = paramset.mtf_sliding_window
            n_margin = n_slices // 2
            details_dict = [None] * n_margin
            for zz in range(n_margin, len(matrix) - n_margin):
                details_img = []
                matrix_this = matrix[zz - n_margin : zz + n_margin + 1]
                vert_pos = zpos_used[zz - n_margin : zz + n_margin + 1]
                if np.isnan(np.sum(vert_pos)):
                    details_dict.append(None)
                else:
                    for i in [0, 1]:
                        axis = 2 if i == 0 else 1
                        matrix_xz = np.sum(matrix_this, axis=axis)
                        details_dict_this, errmsg_this = calculate_MTF_2d_line_edge(
                            matrix_xz, pix, paramset, mode='line',
                            vertical_positions_mm=vert_pos, recalculate_halfmax=True,
                            rotate='no')
                        details_img.append(details_dict_this)
                        if errmsg_this:
                            errmsg.append(f'Center slice number {zz}, axis {i}: {errmsg_this}')
                    details_dict.append(details_img)
            details_dict.extend([None] * n_margin)
        elif paramset.mtf_type == 3:  # z-resolution line
            halfmax = 0.5 * (np.max(max_values) + np.min(max_values))
            above_halfmax = np.where(max_values > halfmax)
            groups = np.diff(above_halfmax)
            start_group2 = np.where(groups[0] > 1)
            n_groups = 0
            idxs = [None, None]  # first slice of groups
            idxs_end = [None, None]  # last slice of groups

            z_diffs = np.diff(zpos)
            z_dist = z_diffs[0]
            if np.min(z_diffs) < np.max(z_diffs):
                errmsg.append('NB: z-increment differ for the slices. Could cause erroneous results.')

            margin = paramset.mtf_roi_size // z_dist
            if margin < 2:
                errmsg.append(
                    'ROI radius used to select images. Set > 2 * slice thickness.')
            else:
                if len(start_group2[0]) == 1:
                    n_groups = 2
                    halfmax_end_group1 = above_halfmax[0][start_group2[0][0]]
                    mid_slice_1 = (
                        above_halfmax[0][0] + halfmax_end_group1) // 2
                    halfmax_start_group2 = above_halfmax[0][start_group2[0][0] + 1]
                    mid_slice_2 = (
                        halfmax_start_group2 + above_halfmax[0][-1]) // 2
                    idxs[0] = int(max([0, mid_slice_1 - margin]))
                    idxs_end[0] = int(min([mid_slice_1 + margin, halfmax_start_group2 - 1]))
                    idxs[1] = int(max([halfmax_end_group1 + 1, mid_slice_2 - margin]))
                    idxs_end[1] = int(min([mid_slice_2 + margin, len(matrix)]))
                elif len(start_group2[0]) == 0:
                    n_groups = 1
                    mid_slice = (
                        above_halfmax[0][-1] + above_halfmax[0][0]) // 2
                    idxs[0] = int(max([0, mid_slice - margin]))
                    idxs_end[0] = int(min([mid_slice + margin, len(matrix)]))

                empty = np.full(max_values.shape, np.nan)
                common_details_dict['max_roi_used'] = np.copy(empty)
                common_details_dict['zpos_used'] = np.copy(empty)
                if n_groups == 2:
                    common_details_dict['max_roi_used_2'] = np.copy(empty)
                    common_details_dict['zpos_used_2'] = np.copy(empty)

                for group in range(n_groups):
                    suffix = ''
                    if group == 1:
                        suffix = '_2'
                    common_details_dict[f'max_roi_used{suffix}'][
                        idxs[group] : idxs_end[group]] = max_values[
                            idxs[group] : idxs_end[group]]
                    common_details_dict[f'zpos_used{suffix}'][
                        idxs[group] : idxs_end[group]] = zpos[
                            idxs[group] : idxs_end[group]]
                    matrix_this = matrix[idxs[group] : idxs_end[group]]
                    matrix_xz = np.sum(matrix_this, axis=1)  # assume line in ~x dir 
                    # TODO allow also vertical line

                    # rotate matrix and use slice_thick as pix and xrange as vertical range
                    xpos = np.arange(matrix_xz.shape[1]) * pix
                    matrix_zx = np.rot90(matrix_xz)
                    details_dict_this, errmsg_this = calculate_MTF_2d_line_edge(
                        matrix_zx, z_dist, paramset, mode='line',
                        vertical_positions_mm=xpos, rotate='no')
                    details_dict.append(details_dict_this)
                    errmsg.append(errmsg_this)

    details_dict.append(common_details_dict)

    return (details_dict, errmsg)


def calculate_MTF_2d_line_edge(matrix, pix, paramset, mode='edge',
                               pr_roi=False, vertical_positions_mm=None,
                               rotate='auto',
                               recalculate_halfmax=False):
    """Calculate MTF from straight line.

    Parameters
    ----------
    matrix : numpy.2darray or list of numpy.2darray
        image within roi(s) - result will be combined from all rois
    pix : float
        pixelsize in mm
    paramset : ParamSetXX
        depending on modality
    mode : str, optional
        'edge' or 'line'. Default is 'edge'
    pr_roi : bool, Optional
        MTF pr roi in matrix or average. Default is False.
    vertical_positions_mm : list of float, optional
        if vertical pix size != pix and possiby irregular. Default is None
        list of y pix positions in mm
    rotate: str
        auto to auto-detect direction
        no/yes to avoid/force matrix rotiation (ensure vertical)
    recalculate_halfmax: bool
        Option to allow for recalculating halfmax when finding line position. Default is False.

    Returns
    -------
    details_dict: dict
    errmsg: list
    """
    edge = True if mode == 'edge' else False
    details_dict = []  # {} if only one
    details_dicts_edge = []  # one for each roi (edge=line)
    ESF_all = []  # if edge
    LSF_all = []
    LSF_no_filt_all = []
    ESF_x = None  # if edge
    LSF_x = None
    errmsg = []
    step_size = 0.1 * pix
    if not isinstance(matrix, list):
        matrix = [matrix]

    def ensure_vertical(sub):
        """Rotate matrix if edge/line not in y direction."""
        if edge:
            x1 = np.mean(sub[:, :2])
            x2 = np.mean(sub[:, -2:])
            diff_x = abs(x1 - x2)
            y1 = np.mean(sub[:2, :])
            y2 = np.mean(sub[-2:, :])
            diff_y = abs(y1 - y2)
            halfmax = 0.5 * (x1 + x2)
            if diff_x < diff_y:
                if rotate == 'auto':
                    sub = np.rot90(sub)
                    halfmax = 0.5 * (y1 + y2)
            if rotate == 'yes':
                sub = np.rot90(sub)
                halfmax = 0.5 * (y1 + y2)
        else:  # line
            prof_y = np.sum(sub, axis=1)
            prof_x = np.sum(sub, axis=0)
            range_y = np.max(prof_y) - np.min(prof_y)
            range_x = np.max(prof_x) - np.min(prof_x)
            if range_y > range_x:
                if rotate == 'auto':
                    sub = np.rot90(sub)
            if rotate == 'yes':
                sub = np.rot90(sub)
            halfmax = 0.5 * (np.max(sub) + np.min(sub))
        return (sub, halfmax)

    def get_edge_pos(sub, halfmax, txt_roi=''):
        """Find position of edge/line for each row in matrix."""
        edge_pos = []
        x = np.arange(sub.shape[1])
        if edge:
            smoothed = sp.ndimage.gaussian_filter(sub, sigma=3)
            for i in range(smoothed.shape[0]):
                res = mmcalc.get_curve_values(x, smoothed[i, :], [halfmax])
                edge_pos.append(res[0])
        else:
            for i in range(sub.shape[0]):
                if recalculate_halfmax:
                    halfmax_this = 0.5 * (np.max(sub[i, :]) + np.min(sub[i, :]))
                    _, center = mmcalc.get_width_center_at_threshold(sub[i, :], halfmax_this)
                else:
                    _, center = mmcalc.get_width_center_at_threshold(sub[i, :], halfmax)
                edge_pos.append(center)
        proceed = True
        ys = np.arange(sub.shape[0])

        if None in edge_pos:
            n_found = len(edge_pos) - edge_pos.count(None)
            txt_mode = 'Edge' if edge else 'Line'
            if n_found < 0.5 * len(edge_pos):
                proceed = False
                errmsg.append(
                    f'{txt_mode} position found for < 50% of ROI {txt_roi}. '
                    'Test failed.')
            else:
                idx_None = mm.get_all_matches(edge_pos, None)
                edge_pos = [e for i, e in enumerate(edge_pos) if i not in idx_None]
                ys = [y for i, y in enumerate(list(ys)) if i not in idx_None]
            if errmsg == []:
                errmsg.append(
                    f'{txt_mode} position not found for full ROI. Parts of ROI ignored.')
        return (edge_pos, x, ys, proceed)

    for m in range(len(matrix)):
        sub, halfmax = ensure_vertical(matrix[m])
        txt_roi = f'{m}' if len(matrix) > 1 else ''
        edge_pos, x, ys, proceed = get_edge_pos(sub, halfmax, txt_roi=txt_roi)

        if proceed:
            # linear fit of edge positions
            vertical_positions = None
            if vertical_positions_mm is not None:
                if isinstance(vertical_positions_mm, list):
                    if len(vertical_positions_mm) == sub.shape[0]:
                        y_pos_mm = [vertical_positions_mm[y] for y in ys]
                        ys = 1./pix * np.array(y_pos_mm)  # unit = pix
                        vertical_positions = ys
            res = sp.stats.linregress(ys, edge_pos)  # x = ay + b
            slope = 1./res.slope  # to y = (1/a)x + (-b/a) to avoid error when steep
            intercept = - res.intercept / res.slope
            x_fit = np.array([min(edge_pos), max(edge_pos)])
            y_fit = slope * x_fit + intercept
            angle = np.abs((180/np.pi) * np.arctan(
                (x_fit[1]-x_fit[0])/(y_fit[1]-y_fit[0])
                ))

            # sort pixels by position normal to edge
            dist_map = mmcalc.get_distance_map_edge(
                sub.shape, slope=slope, intercept=intercept,
                vertical_positions=vertical_positions)

            dist_map_flat = dist_map.flatten()
            values_flat = sub.flatten()
            sort_idxs = np.argsort(dist_map_flat)
            dists = dist_map_flat[sort_idxs]
            sorted_values = values_flat[sort_idxs]
            dists = pix * dists

            details_dicts_edge.append({
                'edge_pos': edge_pos, 'edge_row': ys,
                'edge_fit_x': x_fit, 'edge_fit_y': y_fit,
                'edge_r2': res.rvalue**2, 'angle': angle,
                'sorted_pixels_x': dists,
                'sorted_pixels': sorted_values,
                'edge_or_line': mode
                })

            new_x, new_y = mmcalc.resample_by_binning(
                input_y=sorted_values, input_x=dists,
                step=step_size,
                first_step=-sub.shape[1]/2 * pix,
                n_steps=10*sub.shape[1])

            if edge:
                if ESF_x is None:
                    ESF_x = new_x
                ESF_all.append(new_y)
            else:
                if LSF_x is None:
                    LSF_x = new_x
                LSF_all.append(new_y)

    sigma_f = 0
    if ESF_all:
        sigma_f = 3.  # if sigma_f=5 , FWHM ~9 newpix = ~ 1 original pix
        for ESF in ESF_all:
            LSF, LSF_no_filt, _ = ESF_to_LSF(ESF, prefilter_sigma=sigma_f)
            LSF = LSF/np.max(LSF)
            LSF_no_filt = LSF_no_filt/np.max(LSF_no_filt)
            LSF_all.append(LSF)
            LSF_no_filt_all.append(LSF_no_filt)

    if LSF_all:
        if len(LSF_all) > 1 and pr_roi is False:
            LSF = [np.mean(np.array(LSF_all), axis=0)]
            if LSF_no_filt_all:
                LSF_no_filt = [np.mean(np.array(LSF_no_filt_all), axis=0)]
        else:
            LSF = LSF_all
            if LSF_no_filt_all:
                LSF_no_filt = LSF_no_filt_all
        if not LSF_no_filt_all:
            LSF_no_filt = LSF

        cw = 0
        try:
            cut_lsf = paramset.mtf_cut_lsf
            cut_lsf_fwhm = paramset.mtf_cut_lsf_w
        except AttributeError:
            cut_lsf = False
            cut_lsf_fwhm = None

        if isinstance(paramset, cfc.ParamSetXray):
            gaussfit_type = 'double_both_positive'
            lp_vals = [0.5, 1, 1.5, 2, 2.5]
            mtf_vals = [0.5]
        elif isinstance(paramset, cfc.ParamSetMammo):
            gaussfit_type = 'double_both_positive'
            lp_vals = [1, 2, 3, 4, 5]
            mtf_vals = [0.5]
        elif isinstance(paramset, cfc.ParamSetCT):  # wire 3d
            gaussfit_type = 'double'
            lp_vals = None
            mtf_vals = [0.5, 0.1, 0.02]
        else:  # MR, NM, SPECT, PET 3d linesource
            gaussfit_type = 'single'
            lp_vals = None
            mtf_vals = [0.5, 0.1, 0.02]

        for i in range(len(LSF)):
            width, center = mmcalc.get_width_center_at_threshold(
                LSF[i], np.max(LSF[i])/2)
            if width is not None:
                # Calculate gaussian and discrete MTF
                if cut_lsf:
                    LSF_no_filt_this, cw, _ = cut_and_fade_LSF(
                        LSF_no_filt[i], center=center, fwhm=width,
                        cut_width=cut_lsf_fwhm)
                else:
                    LSF_no_filt_this = LSF_no_filt[i]
                dMTF_details = get_MTF_discrete(LSF_no_filt_this, dx=step_size)
                dMTF_details['cut_width'] = cw * step_size

                LSF_x = step_size * (np.arange(LSF[i].size) - center)
                gMTF_details, err = get_MTF_gauss(
                    LSF[i], dx=step_size, prefilter_sigma=sigma_f*step_size,
                    gaussfit=gaussfit_type)

                if err is not None:
                    errmsg.append(err)
                else:
                    if isinstance(
                            paramset, 
                            (cfc.ParamSetNM, cfc.ParamSetSPECT, cfc.ParamSetPET
                             )):
                        fwhm, _ = mmcalc.get_width_center_at_threshold(
                            LSF[i], np.max(LSF[i])/2)
                        fwtm, _ = mmcalc.get_width_center_at_threshold(
                            LSF[i], np.max(LSF[i])/10)
                        if fwhm:
                            fwhm = step_size * fwhm
                        if fwtm:
                            fwtm = step_size * fwtm
                        dMTF_details['values'] = [fwhm, fwtm]
                        gMTF_details['values'] = [gMTF_details['FWHM'],
                                                  gMTF_details['FWTM']]
                    else:
                        if lp_vals is not None:
                            gMTF_details['values'] = mmcalc.get_curve_values(
                                    gMTF_details['MTF'], gMTF_details['MTF_freq'],
                                    lp_vals)
                            dMTF_details['values'] = mmcalc.get_curve_values(
                                    dMTF_details['MTF'], dMTF_details['MTF_freq'],
                                    lp_vals,
                                    force_first_below=True)
                        gvals = mmcalc.get_curve_values(
                                gMTF_details['MTF_freq'], gMTF_details['MTF'], mtf_vals)
                        dvals = mmcalc.get_curve_values(
                                dMTF_details['MTF_freq'], dMTF_details['MTF'], mtf_vals,
                                force_first_below=True)
                        if lp_vals is not None:
                            gMTF_details['values'].extend(gvals)
                            dMTF_details['values'].extend(dvals)
                        else:
                            gMTF_details['values'] = gvals
                            dMTF_details['values'] = dvals

                details_dict.append({
                    'LSF_x': LSF_x, 'LSF': LSF_no_filt_this,
                    'ESF_x': ESF_x, 'ESF': ESF_all,
                    'sigma_prefilter': sigma_f*step_size,
                    'dMTF_details': dMTF_details, 'gMTF_details': gMTF_details,
                    'edge_details': details_dicts_edge})
            else:
                errmsg = f'Could not find {mode}.'

    if len(details_dict) == 1:
        details_dict = details_dict[0]

    return (details_dict, errmsg)


def calculate_MTF_circular_edge(matrix, roi, pix, paramset, images_to_test):
    """Calculate MTF from circular edge.

    Based on Richard et al: Towards task-based assessment of CT performance,
    Med Phys 39(7) 2012

    Parameters
    ----------
    matrix : numpy.2darray or list of numpy.array
        list of 2d part of slice limited to disc (or None if ignored slice)
    roi : numpy.2darray of bool
        2d part of larger roi limited to disc
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
            center_this = mmcalc.center_xy_of_disc(
                sli, roi=roi, mode='max_or_min', sigma=1)
            if None not in center_this:
                center_xy.append(center_this)
            else:
                if errtxt == '':
                    errtxt = str(slino)
                else:
                    errtxt = ', '.join([errtxt, str(slino)])
    if errtxt != '':
        errmsg.append(f'Could not find center of object for image {errtxt}')
    if len(center_xy) > 0:
        center_x = [vals[0] for vals in center_xy]
        center_y = [vals[1] for vals in center_xy]
        center_xy = [np.mean(np.array(center_x)), np.mean(np.array(center_y))]
        # sort pixel values from center
        dist_map = mmcalc.get_distance_map_point(
            matrix[images_to_test[0]].shape,
            center_dx=center_xy[0] - 0.5 * matrix[images_to_test[0]].shape[1],
            center_dy=center_xy[1] - 0.5 * matrix[images_to_test[0]].shape[0])
        dists_flat = dist_map.flatten()
        sort_idxs = np.argsort(dists_flat)
        dists = dists_flat[sort_idxs]

        dists = pix * dists
        values_all = []
        nsum = 0
        total_imgs = None
        for sli in matrix:
            if sli is not None:
                if total_imgs is None:
                    total_imgs = sli
                else:
                    total_imgs = total_imgs + sli
                values_all.append(sli.flatten()[sort_idxs])
                nsum += 1
        img_avg = 1/nsum * total_imgs
        img_avg_flat = img_avg.flatten()
        values_avg = img_avg_flat[sort_idxs]

        # ignore dists > radius
        radius = 0.5 * matrix[images_to_test[0]].shape[0] * pix
        dists_cut = dists[dists < radius]
        values = values_avg[dists < radius]

        step_size = .1 * pix
        new_x, new_y = mmcalc.resample_by_binning(
            input_y=values, input_x=dists_cut, step=step_size)
        sigma_f = 5.
        LSF, LSF_no_filt, ESF_filtered = ESF_to_LSF(
            new_y, prefilter_sigma=sigma_f)

        width, center = mmcalc.get_width_center_at_threshold(LSF, np.max(LSF)/2)
        disc_radius_mm = None
        if width is not None:
            if center is not None:
                disc_radius_mm = center * step_size
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
                LSF_no_filt, cw, cwf = cut_and_fade_LSF(
                    LSF_no_filt, center=center, fwhm=width,
                    cut_width=cut_lsf_fwhm, fade_width=fade_lsf_fwhm)
            dMTF_details = get_MTF_discrete(LSF_no_filt, dx=step_size)
            dMTF_details['cut_width'] = cw * step_size
            dMTF_details['cut_width_fade'] = cwf * step_size

            LSF_x = step_size * (np.arange(LSF.size) - center)

            gMTF_details, err = get_MTF_gauss(
                LSF, dx=step_size, prefilter_sigma=sigma_f*step_size, gaussfit='double')

            if err is not None:
                errmsg.append(err)
            else:
                gMTF_details['values'] = mmcalc.get_curve_values(
                        gMTF_details['MTF_freq'], gMTF_details['MTF'], [0.5, 0.1, 0.02])
            dMTF_details['values'] = mmcalc.get_curve_values(
                    dMTF_details['MTF_freq'], dMTF_details['MTF'], [0.5, 0.1, 0.02],
                    force_first_below=True)

            details_dict = {
                'matrix': matrix,
                'center_xy': center_xy, 'disc_radius_mm': disc_radius_mm,
                'LSF_x': LSF_x, 'LSF': LSF_no_filt,
                'sigma_prefilter': sigma_f*step_size,
                'sorted_pixels_x': dists, 'sorted_pixels': values_all,
                'interpolated_x': new_x, 'interpolated': new_y,
                'presmoothed': ESF_filtered,
                'dMTF_details': dMTF_details, 'gMTF_details': gMTF_details}
        else:
            errmsg.append('Could not find circular edge.')

    return (details_dict, errmsg)


def calculate_MTF_3d_z_edge(matrix, roi, images_to_test, image_infos, paramset):
    """Calculate MTF from edge ~parallel to slices.

    Parameters
    ----------
    matrix : numpy.2darray or list of numpy.array
        list of 2d part of slice limited by ROI
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
    details_dict: list of dict
        x_dir, y_dir, common_details
    errmsg : list of str
    """
    details_dict = []
    common_details_dict = {}
    errmsg = []
    matrix = [sli for sli in matrix if sli is not None]
    zpos = np.array([image_infos[sli].zpos for sli in images_to_test])
    avg_values = np.array([np.mean(sli) for sli in matrix])
    common_details_dict['max_roi_marked_images'] = avg_values
    common_details_dict['zpos_marked_images'] = zpos

    pix = image_infos[images_to_test[0]].pix[0]
    z_diffs = np.diff(zpos)
    z_dist = z_diffs[0]
    if np.min(z_diffs) < np.max(z_diffs):
        errmsg.append('NB: z-increment differ for the slices. Could cause erroneous results.')

    halfmax = 0.5 * (np.max(avg_values) + np.min(avg_values))
    above_halfmax = np.where(avg_values > halfmax)
    margin = paramset.mtf_roi_size // z_dist

    proceed = False
    if margin < 2:
        errmsg.append(
            'ROI radius used to select images. Set > 2 * slice thickness.')
    else:
        diff = np.diff(avg_values)
        if np.abs(np.min(diff)) > np.abs(np.max(diff)):  # high values first
            mid_slice = above_halfmax[0][-1]
        else:
            mid_slice = above_halfmax[0][0]
        idx_start = int(max([0, mid_slice - margin]))
        idx_end = int(min([mid_slice + margin, len(matrix)]))

        if idx_end - idx_start > 5:
            proceed = True
        else:
            errmsg.append(
                'Failed to find edge in z-direction with reasonable margin.')

    if proceed:
        empty = np.full(avg_values.shape, np.nan)
        common_details_dict['max_roi_used'] = np.copy(empty)
        common_details_dict['zpos_used'] = np.copy(empty)
        common_details_dict['max_roi_used'][
            idx_start : idx_end] = avg_values[idx_start : idx_end]
        zpos_used = zpos[idx_start : idx_end]
        common_details_dict['zpos_used'][
                idx_start : idx_end] = zpos_used
        matrix_this = matrix[idx_start : idx_end]

        sz_y, sz_x = matrix_this[0].shape
        edge_pos = np.full(matrix_this[0].shape, np.nan)

        sub = np.array(matrix_this)
        for i in range(sz_x):
            for j in range(sz_y):
                # TODO make polyfit_2d ignore nan #if roi[j, i]:
                smoothed = sp.ndimage.gaussian_filter1d(
                    sub[:, j, i], sigma=3)
                res = mmcalc.get_curve_values(
                    zpos_used, smoothed, [halfmax])
                edge_pos[j, i] = res[0]
        fit = mmcalc.polyfit_2d(edge_pos, max_order=1)
        abcd = mmcalc.get_plane_from_3_points(
            (fit[0, 0], 0, 0),
            (fit[-1, 0], pix*(sz_y-1), 0),
            (fit[0, -1], 0, pix*(sz_x-1))
            )
        coords = np.meshgrid(
            zpos_used, pix * np.arange(sz_y), pix * np.arange(sz_x),
            indexing='ij')
        # display fit plane Z = (d - a*X - b*Y) / c
        dist_matrix = mmcalc.get_distance_3d_plane(coords, abcd)

        dists_flat = dist_matrix.flatten()
        sort_idxs = np.argsort(dists_flat)
        dists = dists_flat[sort_idxs]
        values = sub.flatten()
        values = values[sort_idxs]

        step_size = .1 * z_dist
        new_x, new_y = mmcalc.resample_by_binning(
            input_y=values, input_x=dists, first_step=np.min(dists),
            step=step_size)
        sigma_f = 5.
        LSF, LSF_no_filt, ESF_filtered = ESF_to_LSF(
            new_y, prefilter_sigma=sigma_f)

        width, center = mmcalc.get_width_center_at_threshold(LSF, np.max(LSF)/2)
        if width is not None:
            # Discrete MTF
            fade_lsf_fwhm = 0
            cw = 0
            cwf = 0
            cut_lsf = paramset.mtf_cut_lsf
            cut_lsf_fwhm = paramset.mtf_cut_lsf_w
            fade_lsf_fwhm = paramset.mtf_cut_lsf_w_fade
            if cut_lsf:
                LSF_no_filt, cw, cwf = cut_and_fade_LSF(
                    LSF_no_filt, center=center, fwhm=width,
                    cut_width=cut_lsf_fwhm, fade_width=fade_lsf_fwhm)
            dMTF_details = get_MTF_discrete(LSF_no_filt, dx=step_size)
            dMTF_details['cut_width'] = cw * step_size
            dMTF_details['cut_width_fade'] = cwf * step_size

            LSF_x = step_size * (np.arange(LSF.size) - center)

            gMTF_details, err = get_MTF_gauss(
                LSF, dx=step_size, prefilter_sigma=sigma_f*step_size, gaussfit='double')

            if err is not None:
                errmsg.append(err)
            else:
                if isinstance(paramset, cfc.ParamSetCT):
                    gMTF_details['values'] = mmcalc.get_curve_values(
                        gMTF_details['MTF_freq'], gMTF_details['MTF'],
                        [0.5, 0.1, 0.02])
                else:  # SPECT or PET
                    fwtm, _ = mmcalc.get_width_center_at_threshold(
                        LSF, np.max(LSF)/10)
                    if width is not None:
                        width = width * step_size
                    if fwtm is not None:
                        fwtm = fwtm * step_size
                    gMTF_details['values'] = [width, fwtm]

            if isinstance(paramset, cfc.ParamSetCT):
                dMTF_details['values'] = mmcalc.get_curve_values(
                    dMTF_details['MTF_freq'], dMTF_details['MTF'],
                    [0.5, 0.1, 0.02],
                    force_first_below=True)
            else:  # SPECT or PET
                fwtm, _ = mmcalc.get_width_center_at_threshold(
                    LSF, np.max(LSF)/10)
                if width is not None:
                    width = width * step_size
                if fwtm is not None:
                    fwtm = fwtm * step_size
                dMTF_details['values'] = [width, fwtm]

            details_dict = [{
                'matrix': matrix,
                'LSF_x': LSF_x, 'LSF': LSF_no_filt,
                'sigma_prefilter': sigma_f*step_size,
                'sorted_pixels_x': dists, 'sorted_pixels': values,
                'interpolated_x': new_x, 'interpolated': new_y,
                'presmoothed': ESF_filtered,
                'dMTF_details': dMTF_details, 'gMTF_details': gMTF_details}]

    details_dict.append(common_details_dict)

    return (details_dict, errmsg)
