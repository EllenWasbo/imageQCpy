# -*- coding: utf-8 -*-
"""
Tests on GUI.

@author: ewas
"""
import os

from imageQC.ui.ui_main import MainWindow
from imageQC.config.iQCconstants import (
    ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER, ENV_ICON_PATH)
from imageQC.config.config_func import get_icon_path
from imageQC.ui import rename_dicom
from imageQC.ui import task_based_image_quality
from imageQC.ui.settings import SettingsDialog
from imageQC.ui import automation_wizard
from imageQC.ui import open_multi
from imageQC.ui import open_automation

os.environ[ENV_USER_PREFS_PATH] = ''
os.environ[ENV_CONFIG_FOLDER] = ''
os.environ[ENV_ICON_PATH] = get_icon_path(False)


def test_start_open_multi(qtbot):
    main = MainWindow()
    dlg = open_multi.OpenMultiDialog(main)
    qtbot.addWidget(dlg)
    assert 1 == 1


def test_start_rename_dicom(qtbot):
    main = MainWindow()
    dlg = rename_dicom.RenameDicomDialog(main)
    qtbot.addWidget(dlg)
    assert 1 == 1


def test_start_settings(qtbot):
    main = MainWindow()
    dlg = SettingsDialog(main)
    qtbot.addWidget(dlg)
    assert 1 == 1
