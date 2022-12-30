# -*- coding: utf-8 -*-
"""
Tests on GUI.

@author: ewas
"""
import os

from PyQt5 import QtCore

from imageQC.ui.ui_main import MainWindow


def test_open_files(qtbot):
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    dir_tests = os.path.dirname(__file__)
    p = os.path.join(dir_tests, 'test_inputs', 'DICOM', 'CTconstancy')
    files = [x for x in p.glob('**/*') if x.is_file()]
    window.open_files(file_list=files)
    window.wQuickTest.gbQT.setChecked(True)
    qtbot.keyClicks(window.wQuickTest.cbox_template, 'constancy')

    #qtbot.mouseClick(window.findButton, QtCore.Qt.LeftButton)

    assert len(window.imgs) == 13
