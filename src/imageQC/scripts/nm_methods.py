# -*- coding: utf-8 -*-
"""
Collection of methods for SNI and uniformity calculations
 for gamma camera in ImageQC.

@author: ellen
"""
from pathlib import Path
import copy

import numpy as np
import scipy as sp
import skimage

# imageQC block start
from imageQC.scripts import dcm
from imageQC.scripts.calculate_roi import get_roi_SNI
import imageQC.scripts.mini_methods_calculate as mmcalc
from imageQC.config.config_func import get_config_folder
# imageQC block end


def get_corrections_point_source(
        image2d, img_info, roi_array,
        fit_x=False, fit_y=False, lock_z=False, guess_z=0.1,
        correction_type='multiply', estimate_noise=0):
    """Estimate xyz of point source and generate correction matrix.

    Parameters
    ----------
    image2d : numpy.ndarray
        input image full resolution
    img_info :  DcmInfo
        as defined in scripts/dcm.py
    roi_array : numpy.ndarray
        roi_array for largest ROI used for the fit (ufov)
    fit_x : bool, optional
        Fit in x direction. The default is False.
    fit_y : bool, optional
        Fit in y direction. The default is False.
    lock_z : bool
        Set distance to source = guess_z when fitting. The default is False.
    guess_z : float
        If guess_z = 0.1 = DICOM radius or 400.
        If guess > 0.1 used as first approximation. Default is 0.1
    correction_type : str, optional
        'subtract' or 'multiply'. Default is 'multiply'
    estimate_noise : int
        if >0 estimate poisson noise for fitted image n times

    Returns
    -------
    dict
        corrected_image : numpy.2darray
        correction_matrix : numpy.2darray
            add this matrix to obtain corrected image2d
        fit_matrix : numpy.2darray
            fitted image
        estimated_noise_images : list of numpy.2darray
            (repeated) estimated poisson noise based on fitted image
        dx : corrected x position (mm)
        dy : corrected y position (mm)
        distance : corrected distance (mm)
    str
        errormessage
    """
    corrected_image = image2d
    correction_matrix = np.ones(image2d.shape)
    fit_matrix = None
    distance = 0.
    estimated_noise_images = None
    errmsg = None

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
            popt = mmcalc.gauss_4param_fit(
                np.arange(len(sum_vector)), sum_vector)
            if popt is not None:
                C, A, x0, sigma = popt
                offset[a] = x0 - 0.5*len(sum_vector)
    dx, dy = offset

    # calculate distances from center in plane
    dists_inplane = mmcalc.get_distance_map_point(
        image2d.shape, center_dx=dx, center_dy=dy)
    dists_inplane = img_info.pix[0] * dists_inplane

    dists_ufov = dists_inplane[rows][:, cols]
    dists_ufov_flat = dists_ufov.flatten()
    values_flat = ufov_denoised.flatten()
    sort_idxs = np.argsort(dists_ufov_flat)
    values = values_flat[sort_idxs]
    dists = dists_ufov_flat[sort_idxs]

    if lock_z:
        nm_radius = guess_z
        lock_radius = True
    else:
        if guess_z <= 0.1:
            try:
                nm_radius = img_info.nm_radius
            except AttributeError:
                nm_radius = 400
        else:
            nm_radius = guess_z
        lock_radius = False

    if nm_radius is None:
        errmsg = (
            'Failed fitting matrix to point source. nm_radius is None. Report error.')
    else:
        popt = mmcalc.point_source_func_fit(
            dists, values,
            center_value=np.max(ufov_denoised),
            avg_radius=nm_radius, lock_radius=lock_radius)
        if popt is not None:
            C, distance = popt
            fit = mmcalc.point_source_func(dists_inplane.flatten(), C, distance)

            # estimate noise
            fit_matrix = fit.reshape(image2d.shape)
            if estimate_noise > 0:
                rng = np.random.default_rng()
                estimated_noise_images = [
                    rng.poisson(fit_matrix) for i in range(estimate_noise)]

            # correct input image
            if correction_type == 'multiply':
                normalized_fit_matrix = fit_matrix/np.max(fit)
                corrected_image = image2d / normalized_fit_matrix
            elif correction_type == 'subtract':
                correction_matrix = np.max(fit) - fit_matrix
                corrected_image = image2d + correction_matrix
        else:
            errmsg = 'Failed fitting matrix to point source.'

    return ({'corrected_image': corrected_image,
             'correction_matrix': correction_matrix,
             'fit_matrix': fit_matrix,
             'estimated_noise_images': estimated_noise_images,
             'dx': dx, 'dy': dy, 'distance': distance}, errmsg)


def calculate_NM_uniformity(image2d, roi_array, pix, scale_factor):
    """Calculate uniformity parameters.

    Parameters
    ----------
    image2d : numpy.ndarray
    roi_array : list of numpy.ndarray
        2d mask for ufov [0] and cfov [1]
    pix : float
        pixel size of image2d
    scale_factor : int
        scale factor to get 6.4 mm/pix. 0 = Auto, 1 = none, 2... = scale factor

    Returns
    -------
    matrix : numpy.ndarray
        downscaled and smoothed image used for calculation of uniformity values
    du_matrix : numpy.ndarray
        calculated du_values maximum of x/y direction
    values : list of float
        ['IU_UFOV %', 'DU_UFOV %', 'IU_CFOV %', 'DU_CFOV %']
    """
    def get_differential_uniformity(image, cfov):
        # assume image is part of image within ufov
        # cfov = True where inside cfov
        from numpy.lib.stride_tricks import sliding_window_view
        sz_y, sz_x = image.shape
        du_cols_ufov = np.zeros(image.shape)
        du_cols_cfov = np.zeros(image.shape)
        image_data_ufov = np.copy(image.data)
        image_data_ufov[image.mask == True] = np.nan
        image_data_cfov = np.copy(image.data)
        image_data_cfov[cfov == False] = np.nan
        for x in range(sz_x):
            view = sliding_window_view(image_data_ufov[:, x], 5)
            maxs = np.nanmax(view, axis=-1)
            mins = np.nanmin(view, axis=-1)
            du_cols_ufov[2:-2, x] = 100. * (maxs - mins) / (maxs + mins)
            cfov_data = image_data_cfov[:, x]
            if not np.isnan(cfov_data).all():
                view = sliding_window_view(cfov_data, 5)
                maxs = np.nanmax(view, axis=-1)
                mins = np.nanmin(view, axis=-1)
                du_cols_cfov[2:-2, x] = 100. * (maxs - mins) / (maxs + mins)

        du_rows_ufov = np.zeros(image.shape)
        du_rows_cfov = np.zeros(image.shape)
        for y in range(sz_y):
            view = sliding_window_view(image_data_ufov[y, :], 5)
            maxs = np.nanmax(view, axis=-1)
            mins = np.nanmin(view, axis=-1)
            du_rows_ufov[y, 2:-2] = 100. * (maxs - mins) / (maxs + mins)
            cfov_data = image_data_cfov[y, :]
            if not np.isnan(cfov_data).all():
                view = sliding_window_view(cfov_data, 5)
            maxs = np.nanmax(view, axis=-1)
            mins = np.nanmin(view, axis=-1)
            du_rows_cfov[y, 2:-2] = 100. * (maxs - mins) / (maxs + mins)

        du_matrix_ufov = np.maximum(du_cols_ufov, du_rows_ufov)
        du_matrix_ufov[image.mask == True] = np.nan
        du_matrix_cfov = np.maximum(du_cols_cfov, du_rows_cfov)
        du_matrix_cfov[cfov == False] = np.nan

        return {'du_matrix': du_matrix_ufov, 'du_ufov': np.nanmax(du_matrix_ufov),
                'du_cfov': np.nanmax(du_matrix_cfov)}

    # continue with image within ufov only
    rows = np.max(roi_array[0], axis=1)
    cols = np.max(roi_array[0], axis=0)
    image2d = image2d[rows][:, cols]
    cfov = roi_array[1][rows][:, cols]
    ufov_input = roi_array[0][rows][:, cols]

    # roi already handeled ignoring pixels with zero counts as nearest neighbour (NEMA)
    if scale_factor == 1:
        image = image2d
        block_size = 1
    else:
        # resize to 6.4+/-30% mm pixels according to NEMA
        # pix_range = [6.4*0.7, 6.4*1.3]
        block_size = scale_factor
        if scale_factor == 0:  # Auto scale
            scale_factors = [(np.floor(64/pix)), (np.ceil(6.4/pix))]
            pix_diff = np.abs(pix*np.array(scale_factors) - 6.4)
            selected_pix = np.where(pix_diff == np.min(pix_diff))
            block_size = int(scale_factors[selected_pix[0][0]])

        # skip pixels excess of size/block_size = NEMA at least 50% of pixel inside
        sz_y, sz_x = image2d.shape
        skip_y, skip_x = (sz_y % block_size, sz_x % block_size)
        n_blocks_y, n_blocks_x = (sz_y // block_size, sz_x // block_size)
        start_y, start_x = (skip_y // 2, skip_x // 2)
        end_x = start_x + n_blocks_x * block_size
        end_y = start_y + n_blocks_y * block_size
        cfov = cfov[start_y:end_y, start_x:end_x]
        ufov_input = ufov_input[start_y:end_y, start_x:end_x]
        image2d = image2d[start_y:end_y, start_x:end_x]
        image = skimage.measure.block_reduce(
            image2d, (block_size, block_size), np.sum)
        # scale down to ~6.4mm/pix

        # cfov, NEMA - at least 50% of the pixel should be inside UFOV to be included
        reduced_roi = skimage.measure.block_reduce(
            cfov, (block_size, block_size), np.mean)
        cfov = np.where(reduced_roi > 0.5, True, False)

        if False in ufov_input:
            reduced_roi = skimage.measure.block_reduce(
                ufov_input, (block_size, block_size), np.mean)
            ufov_input = np.where(reduced_roi > 0.5, True, False)

    # ufov, NEMA NU-1: ignore pixels < 75% of CFOV mean in outer rows/cols of UFOV
    arr_cfov = np.ma.masked_array(image, mask=np.invert(cfov))
    cfov_mean = np.mean(arr_cfov)  # TODO test against minimum 10000 (NEMA)
    ufov = np.where(image > 0.75*cfov_mean, True, False)
    if False in ufov:
        ufov[1:-1][:, 1:-1] = True  # ignore only outer rows/cols
    if False in ufov_input:
        ufov[ufov_input == False] = False

    sz_y, sz_x = image.shape
    center_pixel_count = image[sz_y//2][sz_x//2]

    # smooth masked array
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    kernel = kernel / np.sum(kernel)
    arr = np.ma.masked_array(image, mask=np.invert(ufov))
    smooth64 = mmcalc.masked_convolve2d(arr, kernel, boundary='symm', mode='same')

    # differential uniformity
    du_dict = get_differential_uniformity(smooth64, cfov)
    du_ufov = du_dict['du_ufov']
    du_cfov = du_dict['du_cfov']

    smooth64_data = smooth64.data

    # integral uniformity
    max_val = np.max(smooth64_data[ufov == True])
    min_val = np.min(smooth64_data[ufov == True])
    iu_ufov = 100. * (max_val - min_val) / (max_val + min_val)
    max_val = np.max(smooth64_data[cfov == True])
    min_val = np.min(smooth64_data[cfov == True])
    iu_cfov = 100. * (max_val - min_val) / (max_val + min_val)

    return {'matrix_ufov': smooth64_data,
            'du_matrix': du_dict['du_matrix'], 'pix_size': pix * block_size,
            'center_pixel_count': center_pixel_count,
            'values': [iu_ufov, du_ufov, iu_cfov, du_cfov]}


def get_eye_filter(roi_size, pix, paramset):
    """Get eye filter V(r)=r^1.3*exp(-cr^2).

    And precalculations for radial profiles.

    Parameters
    ----------
    roi_size : int
        size of roi or image in pix
    pix : float
        mm pr pix in image
    paramset : ParamSetNM

    Returns
    -------
    eye_filter : dict
        filter 2d : np.ndarray quadratic with size roi_height
        curve: dict of r and V
    """
    def tukey_filter_func(r, start_L_flat_ratio):
        """Calulates Tukey filter.

        Parameters
        ----------
        r : np.array
            frequency values
        start_L_tapered_ratio : tuple of floats
            start = minimum_frequency
            L = width (frequencies)
            flat ratio = 1- tapered_ratio

        Returns
        -------
        V : filter values
        """
        start, L, flat_ratio = start_L_flat_ratio
        t = L * (1 - flat_ratio)
        V = np.ones(r.shape)
        V[r < start + t/2] = 0.5 * (1 + np.cos(2 * np.pi/t * (
            r[r < start + t/2] - start - t/2)))
        V[r > L + start - t/2] = 0.5 * (1 + np.cos(2 * np.pi/t * (
            r[r > L + start - t/2] - L - start + t/2)))
        V[r < start] = 0
        V[r > start + L] = 0
        return V

    def eye_filter_func(r, c):
        return r**1.3 * np.exp(-c*r**2)

    freq = np.fft.fftfreq(roi_size, d=pix)
    freq = freq[:freq.size//2]  # from center

    unit = freq[1] - freq[0]
    dists = mmcalc.get_distance_map_point(
        (roi_size, roi_size))
    if paramset.sni_channels:
        eye_filter_2d = tukey_filter_func(
            unit * dists, paramset.sni_channels_table[0])
    else:
        eye_filter_2d = eye_filter_func(unit*dists, paramset.sni_eye_filter_c)
    eye_filter_2d = 1/np.max(eye_filter_2d) * eye_filter_2d
    if paramset.sni_channels:
        eye_filter_2d_2 = tukey_filter_func(
            unit * dists, paramset.sni_channels_table[1])
        eye_filter_2d = [eye_filter_2d, eye_filter_2d_2]

    # precalculate distance map to avoid repeated argsort calls for each ROI
    dist_map = mmcalc.get_distance_map_point(
        (roi_size, roi_size), center_dx=-0.5, center_dy=-0.5)
    dists_flat = dist_map.flatten()
    sort_idxs = np.argsort(dists_flat)
    dists = unit * dists_flat[sort_idxs]
    nyq_freq = 1 / (2*pix)
    # keep only frequencies below nyquist
    idxs_above_nq = np.where(dists > nyq_freq)
    sort_idxs = sort_idxs[:idxs_above_nq[0][0]]
    dists = dists[:idxs_above_nq[0][0]]
    idxs_combine = np.round(dists / paramset.sni_sampling_frequency)
    idxs_unique = np.unique(idxs_combine)

    eye_filter_dict = {
        'filter_2d': eye_filter_2d, 'unit': unit,
        'sort_idxs': sort_idxs,
        'idxs_combine': idxs_combine, 'idxs_unique': idxs_unique}

    # add curve for plotting (radial profil of eye_filter_2d)
    if paramset.sni_channels:
        yvals, yvals_2 = get_radial_profile_speed(
            eye_filter_2d, eye_filter_dict)
    else:
        yvals = get_radial_profile_speed(
            [eye_filter_2d], eye_filter_dict)
        yvals = yvals[0]
        yvals_2 = None
    freq = paramset.sni_sampling_frequency * (
        np.arange(idxs_unique.size) + np.min(idxs_unique[0]))
    eye_filter_1d = {'r': freq, 'V': 1/np.max(yvals) * yvals}
    if yvals_2 is not None:
        eye_filter_1d.update({'V2': 1/np.max(yvals_2) * yvals_2})

    eye_filter_dict.update({'curve': eye_filter_1d})

    return eye_filter_dict


def get_radial_profile_speed(arrays, eye_filter_dict, ignore_negative=False):
    """Compressed speed version of mini_methods_calculate.get_radial_profile.

    Resample by binning

    Parameters
    ----------
    arrays : list of np.2darray
        list of same-shape images to calculate radial profile from
    eye_filter_dict: dict
        precalculated sorting and indexing of the arrays from get_eye_filter()
    ignore_negative : bool
        True = set negative values to zero. Default is False.

    Returns
    -------
    radial_profile : np.1darray
        profile values
    """
    sort_idxs = eye_filter_dict['sort_idxs']
    idxs_unique = eye_filter_dict['idxs_unique']
    idxs_combine = eye_filter_dict['idxs_combine']

    profiles = []
    for arr in arrays:
        if arr is not None:
            values = arr.flatten()[sort_idxs]
            profile = [
                np.mean(values[np.where(idxs_combine == i)])
                for i in idxs_unique]
            profile = np.array(profile)
            if ignore_negative and np.min(profile) < 0:
                profile[profile < 0] = 0
        else:
            profile = None
        profiles.append(profile)

    return profiles


def get_SNI_ref_image(paramset, tag_infos):
    """Get noise from reference image.

    Parameters
    ----------
    paramset : cfc.ParamsetNM
    tag_infos : tag_infos to read reference image

    Returns
    -------
    dict
        'reference_image': list of nd.array
        empty list if image failed to read
        one image for each frame else
    """
    sni_ref_image = []
    filename = (
        Path(get_config_folder()) / 'SNI_ref_images' /
        (paramset.sni_ref_image + '.dcm'))
    ref_infos, _, _ = dcm.read_dcm_info([filename], GUI=False, tag_infos=tag_infos)
    for rno, ref_info in enumerate(ref_infos):
        image, _ = dcm.get_img(
            filename,
            frame_number=rno, tag_infos=tag_infos)
        if image is not None:
            sni_ref_image.append(image)
    return {'reference_image': sni_ref_image,
            'reference_image_info': ref_infos}


def get_SNI_reference_noise(dict_extras, roi_array, paramset):
    """Calculate reference noise based on SNI reference image.

    Parameters
    ----------
    dict_extras : dict
        Dictionary holding info about reference image and corrections
    roi_array : np.ndarray
    paramset : cfc.ParamsetNM
    """
    if 'reference_image' in dict_extras:
        ref_noise = []
        if paramset.sni_correct:
            for i, image in enumerate(dict_extras['reference_image']):
                fit_dict, errmsg = get_corrections_point_source(
                    image, dict_extras['reference_image_info'][i], roi_array,
                    fit_x=paramset.sni_correct_pos_x,
                    fit_y=paramset.sni_correct_pos_y,
                    lock_z=paramset.sni_lock_radius, guess_z=paramset.sni_radius,
                    correction_type='subtract'
                    )
                ref_noise.append(fit_dict['corrected_image'])
        else:
            ref_noise = copy.deepcopy(dict_extras['reference_image'])

        block_size = paramset.sni_scale_factor
        if block_size > 1:
            # skip pixels excess of size/block_size
            sz_y, sz_x = dict_extras['reference_image'][0].shape
            skip_y, skip_x = (sz_y % block_size, sz_x % block_size)
            n_blocks_y, n_blocks_x = (sz_y // block_size, sz_x // block_size)
            start_y, start_x = (skip_y // 2, skip_x // 2)
            end_x = start_x + n_blocks_x * block_size
            end_y = start_y + n_blocks_y * block_size
            for i in range(len(ref_noise)):
                ref_noise[i] = skimage.measure.block_reduce(
                    ref_noise[i][start_y:end_y, start_x:end_x],
                    (block_size, block_size), np.sum)

        dict_extras.update({'reference_estimated_noise': ref_noise})


def calculate_SNI_ROI(image2d, roi_array_this, eye_filter_dict,
                      pix=1., fit_dict=None, image_mean=0.,
                      sampling_frequency=0.01, sni_dim=0):
    """Calculate SNI for one ROI.

    Parameters
    ----------
    image2d : numpy.2darray
        if fit_matrix is not None, this is a flattened matrix
        (corrected for point source curvature)
    roi_array : numpy.2darray
        2d mask for the current ROI
    eye_filter_dict : dict
        dict from method get_eye_filter
    pix : float
    fit_dict : dict
        dictionary from get_corrections_point_source (if corrections else None)
    image_mean : float
        average value within the large area
    sampling_frequency : float
        sampling frequency for radial profiles
    sni_dim : int
        options for sni calculations 0=2d NPS ratio, 1=radial profile ratio

    Returns
    -------
    values : list of float
        ['SNI max', 'SNI L1', 'SNI L2', 'SNI S1', .. 'SNI S6']
    details_dict : dict
        NPS : numpy.2darray NPS in ROI
        quantum_noise : constant or numpy.2darray (NPS of estimated noise)
        freq : numpy.1darray - frequencies of radial NPS curve
        rNPS : radial NPS curve
        rNPS_filt
        rNPS_struct
        rNPS_struct_filt
    """
    rows = np.max(roi_array_this, axis=1)
    cols = np.max(roi_array_this, axis=0)
    subarray = image2d[rows][:, cols]
    line = subarray.shape[0] // 2  # position of 0 frequency
    eye_filter = eye_filter_dict['filter_2d']
    array_list = [None]  # list of same size arrays to find radial_profiles
    if fit_dict is None:  # uncorrected image and without reference image
        NPS = mmcalc.get_2d_NPS(subarray, pix)
        quantum_noise = image_mean * pix**2
        """ explained how quantum noise is found above
        Mean count=variance=pixNPS^2*Total(NPS) where pixNPS=1./(ROIsz*pix)
        Total(NPS)=NPSvalue*ROIsz^2
        NPSvalue = Meancount/(pixNPS^2*ROIsz^2)=MeanCount*pix^2
        """
        NPS[line, line] = 0
        NPS_struct = NPS - quantum_noise
        NPS_struct[line, line] = 0
    else:
        if 'correction_matrix' in fit_dict:  # point source corrected
            # curve correct both subarray and quantum noise
            corr_matrix = fit_dict['correction_matrix'][rows][:, cols]
            subarray = subarray + corr_matrix
            if 'reference_estimated_noise' in fit_dict:
                sub_estimated_noise = [fit_dict[
                    'reference_estimated_noise'][rows][:, cols]]
            else:
                sub_estimated_noise = [
                    noise_img[rows][:, cols] + corr_matrix
                    for noise_img in fit_dict['estimated_noise_images']]
        else:  # reference image, not point source corrected
            sub_estimated_noise = [fit_dict[
                'reference_estimated_noise'][rows][:, cols]]
        # 2d NPS
        NPS = mmcalc.get_2d_NPS(subarray, pix)
        NPS[line, line] = 0  # ignore extreme values as zero frequency
        if len(sub_estimated_noise) == 1:
            quantum_noise = mmcalc.get_2d_NPS(sub_estimated_noise[0], pix)
        else:
            sum_quantum_noise = np.zeros(NPS.shape)
            for noise_sub in sub_estimated_noise:
                NPS_this = mmcalc.get_2d_NPS(noise_sub, pix)
                sum_quantum_noise = sum_quantum_noise + NPS_this
            quantum_noise = (1/len(sub_estimated_noise)) * sum_quantum_noise
        # ignore extreme values as zero frequency
        quantum_noise[line, line] = 0  # ignore extreme values as zero frequency
        array_list[0] = quantum_noise

        NPS_struct = np.subtract(NPS, quantum_noise)

    SNI_2 = None
    if isinstance(eye_filter, list):
        NPS_filt = NPS * eye_filter[0]
        NPS_struct_filt = NPS_struct * eye_filter[0]
        NPS_filt_2 = NPS * eye_filter[1]
        NPS_struct_filt_2 = NPS_struct * eye_filter[1]
    else:
        NPS_filt = NPS * eye_filter
        NPS_struct_filt = NPS_struct * eye_filter

    # radial NPS curves
    array_list.extend([NPS, NPS_filt, NPS_struct, NPS_struct_filt])
    if isinstance(eye_filter, list):
        array_list.extend([NPS_filt_2, NPS_struct_filt_2])
    else:
        array_list.extend([None, None])

    profiles = get_radial_profile_speed(
        array_list, eye_filter_dict, ignore_negative=False)
    (rNPS_quantum_noise, rNPS, rNPS_filt, rNPS_struct, rNPS_struct_filt,
     rNPS_filt_2, rNPS_struct_filt_2) = profiles
    freq = sampling_frequency * (
        np.arange(rNPS.size) + np.min(eye_filter_dict['idxs_unique'][0]))

    if sni_dim == 0:  # ratio struct from 2d integral NPS
        ignore_negative = False  # ignore True just for testing - results show crappy data
        if ignore_negative:
            SNI = np.sum(NPS_struct_filt[NPS_struct_filt > 0]) / np.sum(NPS_filt)
        else:
            SNI = np.sum(NPS_struct_filt) / np.sum(NPS_filt)
        if isinstance(eye_filter, list):
            if ignore_negative:
                SNI_2 = np.sum(
                    NPS_struct_filt_2[NPS_struct_filt_2 > 0]) / np.sum(NPS_filt_2)
            else:
                SNI_2 = np.sum(NPS_struct_filt_2) / np.sum(NPS_filt_2)
    else:  # ratio struct from 1d integral radial profiles
        SNI = np.sum(rNPS_struct_filt) / np.sum(rNPS_filt)
        if isinstance(eye_filter, list):
            SNI_2 = np.sum(rNPS_struct_filt_2) / np.sum(rNPS_filt_2)

    details_dict_roi = {
        'NPS': NPS, 'quantum_noise': quantum_noise,
        'freq': freq, 'rNPS': rNPS, 'rNPS_filt': rNPS_filt, 'rNPS_filt_2': rNPS_filt_2,
        'rNPS_struct': rNPS_struct, 'rNPS_struct_filt': rNPS_struct_filt,
        'rNPS_struct_filt_2': rNPS_struct_filt_2,
        'rNPS_quantum_noise': rNPS_quantum_noise
        }
    return (SNI, SNI_2, details_dict_roi)


def calculate_NM_SNI(image2d, roi_array, image_info, paramset, reference_noise):
    """Calculate Structured Noise Index.

    Parameters
    ----------
    image2d : numpy.ndarray
    roi_array : list of numpy.ndarray
        list of 2d masks for all_rois, full ROI, L1, L2, small ROIs
    image_info :  DcmInfo
        as defined in scripts/dcm.py
    paramset : cfc.ParamsetNM
    reference_noise : numpy.ndarray or None
        corrected and block_reduced reference image

    Returns
    -------
    values : list of float
        according to defined headers in config/iQCconstants.py HEADERS
    details_dict : dict
    errmsgs : list of str
    """
    values_sup = [None] * 3
    details_dict = {}
    SNI_values = []
    SNI_values_2 = []
    errmsgs = []

    fit_dict = None
    if reference_noise is not None:
        fit_dict = {}
        fit_dict['reference_estimated_noise'] = reference_noise

    # point source correction
    if paramset.sni_correct:
        est_noise = paramset.sni_n_sample_noise if reference_noise is None else 0

        #DELETE?
        #if reference_image is not None and paramset.sni_ref_image_fit:
        #    fit_image = reference_image
        #else:
        #    fit_image = image2d

        fit_dict_2, errmsg = get_corrections_point_source(
            image2d, image_info, roi_array[0],
            fit_x=paramset.sni_correct_pos_x,
            fit_y=paramset.sni_correct_pos_y,
            lock_z=paramset.sni_lock_radius, guess_z=paramset.sni_radius,
            correction_type='subtract', estimate_noise=est_noise
            )
        if fit_dict:
            fit_dict.update(fit_dict_2)
        else:
            fit_dict = fit_dict_2
        if errmsg is not None:
            errmsgs.append(errmsg)
        values_sup = [fit_dict['dx'], fit_dict['dy'], fit_dict['distance']]

    block_size = paramset.sni_scale_factor
    if block_size > 1:
        # skip pixels excess of size/block_size
        sz_y, sz_x = image2d.shape
        skip_y, skip_x = (sz_y % block_size, sz_x % block_size)
        n_blocks_y, n_blocks_x = (sz_y // block_size, sz_x // block_size)
        start_y, start_x = (skip_y // 2, skip_x // 2)
        end_x = start_x + n_blocks_x * block_size
        end_y = start_y + n_blocks_y * block_size
        image2d = skimage.measure.block_reduce(
            image2d[start_y:end_y, start_x:end_x], (block_size, block_size), np.sum)

        if fit_dict:
            key = 'correction_matrix'
            if key in fit_dict:
                if fit_dict[key] is not None:
                    temp = skimage.measure.block_reduce(
                        fit_dict[key][start_y:end_y, start_x:end_x],
                        (block_size, block_size), np.sum)
                    fit_dict[key] = temp
            key = 'estimated_noise_images'
            if key in fit_dict:
                if fit_dict[key] is not None:
                    for i, noise_img in enumerate(fit_dict[key]):
                        temp = skimage.measure.block_reduce(
                            noise_img[start_y:end_y, start_x:end_x],
                            (block_size, block_size), np.sum)
                        fit_dict[key][i] = temp

        # recalculate rois (avoid block reduce on rois, might get non-quadratic)
        roi_array, errmsg = get_roi_SNI(image2d, image_info, paramset,
                                        block_size=block_size)

    if fit_dict is not None:
        details_dict = fit_dict

    arr = np.ma.masked_array(image2d, mask=np.invert(roi_array[0]))
    image_mean = np.mean(arr)

    details_dict['pr_roi'] = []

    # large ROIs
    rows = np.max(roi_array[1], axis=1)
    eye_filter_dict = get_eye_filter(
        np.count_nonzero(rows), image_info.pix[0]*block_size, paramset)
    details_dict['eye_filter_large'] = eye_filter_dict['curve']
    for i in [1, 2]:
        SNI, SNI_2, details_dict_roi = calculate_SNI_ROI(
            image2d, roi_array[i], eye_filter_dict,
            pix=image_info.pix[0]*block_size, fit_dict=fit_dict, image_mean=image_mean,
            sampling_frequency=paramset.sni_sampling_frequency,
            sni_dim=paramset.sni_ratio_dim)
        details_dict['pr_roi'].append(details_dict_roi)
        SNI_values.append(SNI)
        SNI_values_2.append(SNI_2)

    largest_L = '-'
    if SNI_values[0] > SNI_values[1]:
        largest_L = 'L1'
    elif SNI_values[0] < SNI_values[1]:
        largest_L = 'L2'
    if any(SNI_values_2):
        if SNI_values_2[0] > SNI_values_2[1]:
            largest_L = largest_L + ' / L1'
        elif SNI_values_2[0] < SNI_values_2[1]:
            largest_L = largest_L + ' / L2'
        else:
            largest_L + ' / -'
    values_sup.append(largest_L)

    if paramset.sni_type == 0:
        # small ROIs
        rows = np.max(roi_array[3], axis=1)
        eye_filter_dict = get_eye_filter(
            np.count_nonzero(rows), image_info.pix[0]*block_size, paramset)
        details_dict['eye_filter_small'] = eye_filter_dict['curve']
        for i in range(3, 9):
            SNI, SNI_2, details_dict_roi = calculate_SNI_ROI(
                image2d, roi_array[i], eye_filter_dict,
                pix=image_info.pix[0]*block_size, fit_dict=fit_dict, image_mean=image_mean,
                sampling_frequency=paramset.sni_sampling_frequency,
                sni_dim=paramset.sni_ratio_dim)
            details_dict['pr_roi'].append(details_dict_roi)
            SNI_values.append(SNI)
            SNI_values_2.append(SNI_2)

        if paramset.sni_channels:
            values = (
                SNI_values[0:2]
                + [np.max(SNI_values[2:])] + [np.mean(SNI_values[2:])]
                + SNI_values_2[0:2]
                + [np.max(SNI_values_2[2:])] + [np.mean(SNI_values_2[2:])]
                )
        else:
            values = [np.max(SNI_values)] + SNI_values
        maxno = SNI_values[2:].index(max(SNI_values[2:]))
        if any(SNI_values_2):
            idx = SNI_values_2[2:].index(max(SNI_values_2[2:]))
            maxno_2 = f' / S{idx+1}'
        else:
            maxno_2 = ''
        values_sup.append(f'S{maxno+1}{maxno_2}')
        details_dict['SNI_values'] = SNI_values
        if paramset.sni_channels:
            details_dict['SNI_values_2'] = SNI_values_2
    else:
        idx_roi_row_0 = 3
        try:
            rows = np.max(roi_array[idx_roi_row_0][0], axis=1)
        except (np.AxisError, TypeError):  # if first is ignored Siemens
            rows = np.max(roi_array[idx_roi_row_0 + 2][3], axis=1)  # central
            # TODO better catch - what if this also (not likely) is None?
        eye_filter_dict = get_eye_filter(
            np.count_nonzero(rows), image_info.pix[0]*block_size, paramset)
        details_dict['eye_filter_small'] = eye_filter_dict['curve']

        if paramset.sni_type in [1, 2]:
            SNI_map = np.zeros(
                (len(roi_array)-idx_roi_row_0, len(roi_array[idx_roi_row_0])))
            SNI_map_2 = np.copy(SNI_map)
        else:  # 3 Siemens
            SNI_map = []
            SNI_map_2 = []
        rNPS_filt_sum = None
        rNPS_struct_filt_sum = None
        rNPS_filt_2_sum = None
        rNPS_struct_filt_2_sum = None
        small_names = []
        for rowno, row in enumerate(roi_array[idx_roi_row_0:]):
            for colno, roi in enumerate(row):
                small_names.append(f'r{rowno}_c{colno}')
                if roi is not None:
                    SNI, SNI_2, details_dict_roi = calculate_SNI_ROI(
                        image2d, roi, eye_filter_dict,
                        pix=image_info.pix[0]*block_size, fit_dict=fit_dict,
                        image_mean=image_mean,
                        sampling_frequency=paramset.sni_sampling_frequency,
                        sni_dim=paramset.sni_ratio_dim)
                    details_dict['pr_roi'].append(details_dict_roi)
                    if paramset.sni_type in [1, 2]:
                        SNI_map[rowno, colno] = SNI
                        SNI_map_2[rowno, colno] = SNI_2
                    else:  # 3 Siemens
                        SNI_map.append(SNI)
                        SNI_map_2.append(SNI_2)
                    SNI_values.append(SNI)
                    SNI_values_2.append(SNI_2)
                    if rNPS_filt_sum is None:
                        rNPS_filt_sum = details_dict_roi['rNPS_filt']
                        rNPS_struct_filt_sum = details_dict_roi['rNPS_struct_filt']
                    else:
                        rNPS_filt_sum = rNPS_filt_sum + details_dict_roi['rNPS_filt']
                        rNPS_struct_filt_sum = (
                            rNPS_struct_filt_sum + details_dict_roi['rNPS_struct_filt'])
                    if paramset.sni_channels:
                        if rNPS_filt_2_sum is None:
                            rNPS_filt_2_sum = details_dict_roi['rNPS_filt_2']
                            rNPS_struct_filt_2_sum = details_dict_roi['rNPS_struct_filt_2']
                        else:
                            rNPS_filt_2_sum = rNPS_filt_2_sum + details_dict_roi['rNPS_filt_2']
                            rNPS_struct_filt_2_sum = (
                                rNPS_struct_filt_2_sum + details_dict_roi['rNPS_struct_filt_2'])
                else:
                    details_dict['pr_roi'].append(None)
                    SNI_values.append(np.nan)
                    SNI_values_2.append(np.nan)
                    if paramset.sni_type == 3:  # always? if ignored
                        SNI_map.append(np.nan)
                        SNI_map_2.append(np.nan)
        values = []
        max_nos = []
        if paramset.sni_type == 3:  # avoid np.nan for ignored
            if paramset.sni_channels:
                arrs = [np.array(SNI_map), np.array(SNI_map_2)]
                vals = [SNI_values, SNI_values_2]
            else:
                arrs = [np.array(SNI_map)]
                vals = [SNI_values]
            for idx, arr in enumerate(arrs):
                max_no = np.where(arr == np.max(arr[arr > -1]))
                max_nos.append(max_no[0][0])
                values.extend([
                    vals[idx][0], vals[idx][1],
                    np.max(arr[arr > -1]),
                    np.mean(arr[arr > -1])
                    ])
                if paramset.sni_channels is False:
                    values.append(np.median(arr[arr > -1]))
        else:
            if paramset.sni_channels:
                vals = [SNI_values, SNI_values_2]
            else:
                vals = [SNI_values]
            for vals_this in vals:
                SNI_small = vals_this[2:]
                max_no = np.where(SNI_small == np.max(SNI_small))
                max_nos.append(max_no[0][0])
                values.extend(
                    vals_this[0:2]
                    + [np.max(SNI_small), np.mean(SNI_small)])
                if paramset.sni_channels is False:
                    values.append(np.median(SNI_small))
        details_dict['roi_max_idx_small'] = max_nos[0]
        txt_small_max = small_names[max_nos[0]]
        if paramset.sni_channels:
            details_dict['roi_max_idx_small_2'] = max_nos[1]
            txt_small_max = txt_small_max + ' / ' + small_names[max_nos[1]]
        values_sup.append(txt_small_max)

        details_dict['avg_rNPS_filt_small'] = rNPS_filt_sum / len(SNI_values)
        details_dict['avg_rNPS_struct_filt_small'] = (
            rNPS_struct_filt_sum / len(SNI_values))
        details_dict['SNI_map'] = SNI_map
        details_dict['SNI_values'] = SNI_values
        if paramset.sni_channels:
            details_dict['avg_rNPS_filt_2_small'] = rNPS_filt_2_sum / len(SNI_values_2)
            details_dict['avg_rNPS_struct_filt_2_small'] = (
                rNPS_struct_filt_2_sum / len(SNI_values_2))
            details_dict['SNI_map_2'] = SNI_map_2
            details_dict['SNI_values_2'] = SNI_values_2

    return (values, values_sup, details_dict, errmsgs)