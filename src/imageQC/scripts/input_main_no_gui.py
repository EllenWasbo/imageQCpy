#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InputMain used when no GUI or pytest to replace MainWindow.

@author: Ellen Wasbø
"""
from __future__ import annotations
from dataclasses import dataclass, field

# imageQC block start
import imageQC.config.config_classes as cfc
# imageQC block end


@dataclass
class InputMain:
    """Dataclass with values as MainWindow when without main window."""

    current_modality: str = 'CT'
    current_test: str = 'DCM'
    current_paramset: dict = field(default_factory=dict)
    # converted from dict to paramset of correct modality when used
    current_quicktest: cfc.QuickTestTemplate = field(
        default_factory=cfc.QuickTestTemplate)
    tag_infos: list = field(default_factory=list)
    imgs: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    current_group_indicators: list = field(default_factory=list)
    # string for each image if output set pr group with quicktest (paramset.output)
    automation_active: bool = True

    def display_errmsg(self, errmsg):
        """Display error messages on console."""
        if errmsg is not None:
            if isinstance(errmsg, str):
                print(errmsg)
            elif isinstance(errmsg, list):
                print('/n'.join(errmsg))
