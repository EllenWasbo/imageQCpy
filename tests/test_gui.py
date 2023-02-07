# -*- coding: utf-8 -*-
"""
Tests on GUI.

@author: ewas
"""
import os
from pathlib import Path

import pytest
from PyQt5 import QtCore

import imageQC.resources
from imageQC.ui.ui_main import MainWindow
from imageQC.config.iQCconstants import (
    ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER, ENV_ICON_PATH, QUICKTEST_OPTIONS)
from imageQC.config.config_func import get_icon_path


os.environ[ENV_USER_PREFS_PATH] = ''
os.environ[ENV_CONFIG_FOLDER] = ''
os.environ[ENV_ICON_PATH] = get_icon_path(False)

# to mouseclick:
#qtbot.mouseClick(main.tab_ct.btnRunHom, QtCore.Qt.LeftButton)

def test_run_test_CT_hom(qtbot):
    path_tests = Path(__file__).parent
    file_path = path_tests / 'test_inputs' / 'CT' / 'CTP486'
    files = [x for x in file_path.glob('**/*') if x.is_file()]
    main = MainWindow()
    qtbot.addWidget(main)
    main.open_files(file_list=files)
    assert len(main.imgs) == 3
    main.tab_ct.setCurrentIndex(2)
    main.tab_ct.run_current()
    assert len(main.results['Hom']['values']) == 3


def test_open_multiframe(qtbot):
    path_tests = Path(__file__).parent
    files = [
        path_tests / 'test_inputs' / 'MR' / 'ACR.dcm',
        path_tests / 'test_inputs' / 'MR' / 'MR_PIQT_oneframe.dcm',
        ]
    main = MainWindow()
    qtbot.addWidget(main)
    main.open_files(file_list=files)
    assert len(main.imgs) == 12
    main.tab_mr.run_current()
    assert len(main.results['DCM']['values']) == 12


def test_start_open_multi(qtbot):
    main = MainWindow()
    qtbot.addWidget(main)
    main.open_multi()
    assert 1 == 1


def test_start_rename_dicom(qtbot):
    main = MainWindow()
    qtbot.addWidget(main)
    main.run_rename_dicom()
    assert 1 == 1


def test_start_settings(qtbot):
    main = MainWindow()
    qtbot.addWidget(main)
    main.run_settings()
    assert 1 == 1


def test_change_modality(qtbot):
    main = MainWindow()
    qtbot.addWidget(main)
    for mod in QUICKTEST_OPTIONS:
        main.current_modality = mod
        main.update_mode()
    main.btn_read_vendor_file.setChecked(True)
    for mod in QUICKTEST_OPTIONS:
        main.current_modality = mod
        main.update_mode()
    assert 1 == 1
