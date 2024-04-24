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

from PyQt5.QtWidgets import QMessageBox

# imageQC block start
from imageQC.scripts.calculate_roi import get_roi_circle, get_roi_rectangle
import imageQC.scripts.mini_methods_calculate as mmcalc
from imageQC.ui import messageboxes
# imageQC block end


@dataclass
class Artifact:
    """Class holding simulated artifact."""

    form: str = ''  # circle, ring, rectangle
    x_offset: float = 0.  # center offset x, mm from image center
    y_offset: float = 0.  # center offset y, mm from image center
    size_1: float = 0.  # mm circle = radius, ring=outer radius, rect x-size
    size_2: float = 0.  # mm ring = inner radius, rect y-size
    rotation: float = 0.  # rotation in degrees if rectangle
    sigma: float = 0.  # sigma (mm) of guassian blur
    method: str = ''  # add, multiply
    value: float = 0.  # value to add or multiply with


def add_artifact(artifact, main, overwrite=False):
    """Add artifact objects to image info."""
    if len(main.imgs) == 0:
        QMessageBox.information(main, 'No images loaded',
                                'No images loaded to add artifacts to.')
    else:
        apply_idx = -1  # default all
        if len(main.imgs) > 1:
            dlg = messageboxes.QuestionBox(
                parent=main, title='Apply to...',
                msg='Apply artifact to ...',
                yes_text='active image',
                no_text='all images')
            res = dlg.exec()
            if res:  # active image only
                apply_idx = main.gui.active_img_no
        if apply_idx == -1:
            apply_idxs = list(range(len(main.imgs)))
        else:
            apply_idxs = [apply_idx]
        for idx in apply_idxs:
            if main.imgs[idx].artifacts is not None:
                if len(main.imgs[idx].artifacts) > 0:
                    if overwrite:
                        main.imgs[idx].artifacts = [artifact]
                    else:
                        main.imgs[idx].artifacts.append(artifact)
                else:
                    main.imgs[idx].artifacts = [artifact]
            else:
                main.imgs[idx].artifacts = [artifact]
        main.update_active_img(
            main.tree_file_list.topLevelItem(main.gui.active_img_no))
        main.refresh_img_display()


def apply_artifacts(image, image_info):
    """Apply artifact to image number of main.imgs."""
    if hasattr(image_info, 'artifacts'):
        for artifact in image_info.artifacts:
            roi_array = None
            off_xy = tuple(
                np.array([artifact.x_offset, artifact.y_offset]) / image_info.pix[0])

            # form options defined in ui_dialogs.py
            # self.form.addItems('circular', 'ring', 'rectangle')
            if artifact.form in ['circular', 'ring']:
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
                if artifact.method == 'adding':
                    if sigma > 0:
                        artifact_array = ndimage.gaussian_filter(
                            artifact_array, sigma=sigma)
                    image = image + artifact_array*artifact.value
                else:  # multiplying
                    artifact_array[artifact_array == 1] = artifact.value
                    artifact_array[artifact_array == 0] = 1.
                    if sigma > 0:
                        artifact_array = ndimage.gaussian_filter(
                            artifact_array, sigma=sigma)
                    image = image * artifact_array

    return image
