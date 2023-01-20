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
import imageQC.ui.settings as settings
from imageQC.config.iQCconstants import (
    ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER, ENV_ICON_PATH)
from imageQC.config.config_func import get_icon_path


os.environ[ENV_USER_PREFS_PATH] = ''
os.environ[ENV_CONFIG_FOLDER] = ''
os.environ[ENV_ICON_PATH] = get_icon_path(False)


def test_run_test_CT_hom(qtbot):
    path_tests = Path(__file__).parent
    file_path = path_tests / 'test_inputs' / 'CT' / 'CTP486'
    files = [x for x in file_path.glob('**/*') if x.is_file()]
    main = MainWindow()
    qtbot.addWidget(main)
    main.open_files(file_list=files)
    assert len(main.imgs) == 3
    main.tabCT.setCurrentIndex(2)
    qtbot.mouseClick(main.tabCT.btnRunHom, QtCore.Qt.LeftButton)
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
