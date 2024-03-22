#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InputMain used when no GUI or pytest to replace MainWindow.

@author: Ellen Wasbø
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

# imageQC block start
import imageQC.config.config_classes as cfc
# imageQC block end


@dataclass
class Gui():
    """Class to keep roi offset data for pytest and assure default values if not gui."""

    delta_x: int = 0
    delta_y: int = 0
    delta_a: float = 0.0
    show_axis: bool = False


@dataclass
class InputMain:
    """Dataclass with values as MainWindow when calculate_qc without main window."""

    test_mode: bool = False  # TODO delete?
    current_modality: str = 'CT'
    current_test: str = 'DCM'
    current_paramset: dict = field(default_factory=dict)
    # converted from dict to paramset of correct modality when used
    current_quicktest: cfc.QuickTestTemplate = field(
        default_factory=cfc.QuickTestTemplate)
    tag_infos: list = field(default_factory=list)
    digit_templates: dict = field(default_factory=dict)
    imgs: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    errmsgs: list = field(default_factory=list)
    current_group_indicators: list = field(default_factory=list)
    # string for each image if output set pr group with quicktest (paramset.output)
    automation_active: bool = True
    active_img: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    current_roi: list = field(default_factory=list)
    summed_img: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    gui: Gui = field(default_factory=Gui)
