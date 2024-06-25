#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code holding the data classes used for simulating artifacts.

@author: Ellen Wasbo
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage

# imageQC block start
from imageQC.scripts.calculate_roi import get_roi_circle, get_roi_rectangle
import imageQC.scripts.mini_methods_calculate as mmcalc
from imageQC.scripts.mini_methods import get_all_matches
# imageQC block end


@dataclass
class Artifact:
    """Class holding simulated artifact."""

    label:  str = ''  # unique to name and recognize same artifact all imgs
    form: str = ''  # circle, ring, rectangle
    x_offset: float = 0.  # center offset x, mm from image center
    y_offset: float = 0.  # center offset y, mm from image center
    size_1: float = 0.  # mm circle = radius, ring=outer radius, rect x-size
    size_2: float = 0.  # mm ring = inner radius, rect y-size
    rotation: float = 0.  # rotation in degrees if rectangle
    sigma: float = 0.  # sigma (mm) of guassian blur, distance if gamma camera point source
    method: str = ''  # 'adding', 'multiplying', 'adding poisson noise', 'adding gamma camera point source'
    value: float = 0.  # value to add or multiply with


def add_artifact(artifact_label, apply_idxs, main):
    """Add artifact label to image info.artifacts."""
    for idx in apply_idxs:
        if main.imgs[idx].artifacts is not None:
            if len(main.imgs[idx].artifacts) > 0:
                main.imgs[idx].artifacts.append(artifact_label)
            else:
                main.imgs[idx].artifacts = [artifact_label]
        else:
            main.imgs[idx].artifacts = [artifact_label]


def edit_artifact_label(old_label, new_label, main):
    """Update label or delete (if new label is '') applied artifacts."""
    for img in main.imgs:
        if img.artifacts is not None:
            if len(img.artifacts) > 0:
                if old_label in img.artifacts:
                    idxs = get_all_matches(img.artifacts, old_label)
                    delete_idxs = []
                    for idx in idxs:
                        if new_label == '':
                            delete_idxs.append(idx)
                        img.artifacts[idx] = new_label
                    if len(delete_idxs) > 0:
                        delete_idxs.sort(reverse=True)
                        for idx in delete_idxs:
                            img.artifacts.pop(idx)


def validate_new_artifact_label(main, artifact, edit=False):
    """Validate that a new label do not already exist or create new unique.

    Parameters
    ----------
    main : MainWindow
    artifact : Artifact
        with suggested .label. If label is '' autogenrated name.
    edit : bool
        True if artifact is edited, not new.

    Returns
    -------
    valid_label : str or None
        Return a valid label or None = suggest another name, never if suggested is ''.
    """
    suggested_label = artifact.label
    current_labels = [x.label for x in main.artifacts]
    if edit:
        split_name = suggested_label.split()
        if split_name[0] in ['circle', 'ring', 'rectangle']:
            # assume auto generated, generate new
            suggested_label = ''

    if suggested_label == '':
        suggested_label = (f'{artifact.form} x{artifact.x_offset:.0f} '
                           f'y{artifact.y_offset:.0f} s{artifact.size_1:.0f}')
        if artifact.size_2 > 0:
            suggested_label = f'{suggested_label}//{artifact.size_2:.0f}'
        if artifact.rotation != 0:
            suggested_label = f'{suggested_label} r{artifact.rotation:.0f}'
        if artifact.sigma != 0:
            if 'gamma camera' in artifact.method:
                suggested_label = f'{suggested_label} dist{artifact.sigma:.0f}'
            else:
                suggested_label = f'{suggested_label} sigma{artifact.sigma:.0f}'
        if 'noise' in artifact.method:
            suggested_label = suggested_label + 'add noise'
        elif 'gamma camera' in artifact.method:
            suggested_label = f'{suggested_label} point source max{artifact.value}'
        else:
            suggested_label = f'{suggested_label} {artifact.method[0]}{artifact.value}'
    if suggested_label in current_labels and edit is False:
        already_startwith = [
            label for label in current_labels if label.startswith(suggested_label)]
        suggested_label = f'{suggested_label}_{len(already_startwith):03}'

    return suggested_label


def apply_artifacts(image, image_info, artifacts):
    """Apply artifact to image number of main.imgs.

    Parameters
    ----------
    image : nparray
    image_info : DcmInfo
        with info on artifacts to apply
        as defined in scripts/dcm.py
    artifacts : list of Artifact
        all available artifacts from main.artifacts

    Returns
    -------
    image : nparray
        image with artifacts applied
    """
    if hasattr(image_info, 'artifacts'):
        labels_available = [x.label for x in artifacts]
        for artifact_label in image_info.artifacts:
            try:
                artifact_idx = labels_available.index(artifact_label)
                artifact = artifacts[artifact_idx]
            except ValueError:  # label not in list
                artifact = None

            if artifact:
                roi_array = None
                off_xy = tuple(
                    np.array(
                        [artifact.x_offset, artifact.y_offset]) / image_info.pix[0])

                # form options defined in ui_dialogs.py
                # self.form.addItems('circle', 'ring', 'rectangle')
                if artifact.form in ['circle', 'ring']:
                    roi_size_pix = artifact.size_1 / image_info.pix[0]
                    if roi_size_pix > 0:
                        roi_array = get_roi_circle(image.shape, off_xy, roi_size_pix)
                        if artifact.form == 'ring':
                            if artifact.size_2 > 0:
                                roi_size_2_pix = artifact.size_2 / image_info.pix[0]
                                inner_roi = get_roi_circle(
                                    image.shape, off_xy, roi_size_2_pix)
                                roi_array[inner_roi == True] = False
                elif artifact.form == 'rectangle':
                    w = artifact.size_1 / image_info.pix[0]
                    h = artifact.size_2 / image_info.pix[0]
                    if w > 0 and h > 0:
                        roi_array = get_roi_rectangle(
                            image.shape, roi_width=w, roi_height=h, offcenter_xy=off_xy)
                        if artifact.rotation != 0:
                            if any(off_xy):
                                roi_array = mmcalc.rotate2d_offcenter(
                                    roi_array.astype(float), -artifact.rotation, off_xy)
                            else:
                                roi_array = ndimage.rotate(
                                    roi_array.astype(float), -artifact.rotation,
                                    reshape=False)
                                roi_array = np.round(roi_array)

                if roi_array is not None:
                    artifact_array = roi_array.astype(float)
                    sigma = 0
                    if artifact.sigma > 0:
                        sigma = artifact.sigma / image_info.pix[0]

                    # method options defined in ui_dialogs.py
                    # self.method.addItems(['adding', 'multiplying'])
                    if artifact.method.startswith('add'):
                        if sigma > 0:
                            if 'gamma camera' not in artifact.method:
                                artifact_array = ndimage.gaussian_filter(
                                    artifact_array, sigma=sigma)
                            else:
                                dist_map_pix = mmcalc.get_distance_map_point(
                                    image.shape,
                                    center_dx=off_xy[0], center_dy=off_xy[1])
                                dist_map_mm = image_info.pix[0] * dist_map_pix
                                arr = mmcalc.point_source_func(
                                    dist_map_mm, 1., artifact.sigma)
                                artifact_array = artifact_array * arr/np.max(arr)
                        if 'noise' in artifact.method:
                            rng = np.random.default_rng()
                            vals_in = image[artifact_array == 1]
                            vals_in[vals_in <= 0] = 1
                            image[artifact_array == 1] = rng.poisson(vals_in)
                        else:
                            image = image + artifact_array*artifact.value
                    else:  # multiplying
                        if artifact.value != 0:
                            artifact_array[artifact_array == 1] = artifact.value
                            artifact_array[artifact_array == 0] = 1.
                        else:
                            artifact_array = np.invert(roi_array).astype(float)
                        if sigma > 0:
                            artifact_array = ndimage.gaussian_filter(
                                artifact_array, sigma=sigma)
                        image = image * artifact_array

    return image
