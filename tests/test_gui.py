# -*- coding: utf-8 -*-
"""
Tests on GUI.

@author: ewas
"""
import os
from pathlib import Path

from pandas import read_clipboard
#import pytest

#import imageQC.resources
from imageQC.ui.ui_main import MainWindow
from imageQC.config.iQCconstants import (
    ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER, ENV_ICON_PATH, QUICKTEST_OPTIONS,
    HEADERS)
from imageQC.config.config_func import get_icon_path
from imageQC.config import config_classes as cfc
from imageQC.scripts.calculate_qc import calculate_qc


os.environ[ENV_USER_PREFS_PATH] = ''
os.environ[ENV_CONFIG_FOLDER] = ''
os.environ[ENV_ICON_PATH] = get_icon_path(False)
path_tests = Path(__file__).parent

# to mouseclick:
#qtbot.mouseClick(main.tab_ct.btnRunHom, QtCore.Qt.LeftButton)

def test_run_test_CT_hom(qtbot):
    #path_tests = Path(__file__).parent
    file_path = path_tests / 'test_inputs' / 'CT' / 'CTP486'
    files = [x for x in file_path.glob('**/*') if x.is_file()]
    main = MainWindow()
    qtbot.addWidget(main)
    main.open_files(file_list=files)
    assert len(main.imgs) == 3
    main.tab_ct.setCurrentIndex(QUICKTEST_OPTIONS['CT'].index('Hom'))
    main.tab_ct.run_current()
    assert len(main.results['Hom']['values']) == 3


def test_MTF_plot_Xray(qtbot):
    mod = 'Xray'
    test_code = 'MTF'
    #path_tests = Path(__file__).parent
    file_path = path_tests / 'test_inputs' / mod / 'mtf.dcm'
    main = MainWindow()
    qtbot.addWidget(main)
    main.open_files(file_list=[file_path])
    main.tab_xray.setCurrentIndex(QUICKTEST_OPTIONS[mod].index(test_code))

    for auto_cent in [False, True]:
        main.current_paramset.mtf_auto_center = auto_cent
        main.tab_ct.run_current()
        main.tab_results.setCurrentIndex(1)
        plot_options = ['Edge position', 'Sorted pixel values', 'LSF', 'MTF']
        for opt in plot_options:
            main.tab_xray.mtf_plot.setCurrentText(opt)
        assert len(main.results[test_code]['values'][0]) == len(
            HEADERS[mod][test_code]['alt0'])


def test_MTF_plot_CT(qtbot):
    mod = 'CT'
    test_code = 'MTF'
    file_path = path_tests / 'test_inputs' / 'CT' / 'Wire_FOV50'
    files = [x for x in file_path.glob('**/*') if x.is_file()]
    main = MainWindow()
    qtbot.addWidget(main)
    main.open_files(file_list=files)
    main.tab_ct.setCurrentIndex(QUICKTEST_OPTIONS[mod].index(test_code))

    main.current_paramset.mtf_type = 0
    main.current_paramset.mtf_roi_size = 3.
    main.current_paramset.mtf_auto_center = True
    main.tab_ct.run_current()
    main.tab_results.setCurrentIndex(1)
    plot_options = ['Centered xy profiles', 'Sorted pixel values', 'LSF', 'MTF']
    for opt in plot_options:
        main.tab_ct.mtf_plot.setCurrentText(opt)
    assert len(main.results[test_code]['values'][0]) == len(
        HEADERS[mod][test_code]['alt0'])


def test_histogram_plot(qtbot):  # TODO delete or expand
    file_path = path_tests / 'test_inputs' / 'Xray' / 'hom.dcm'
    main = MainWindow()
    qtbot.addWidget(main)
    main.open_files(file_list=[file_path])

    assert 1 == 1


def test_quicktest_CT_13imgs(qtbot):
    file_path = path_tests / 'test_inputs' / 'CT' / 'Constancy13img'
    files = [x for x in file_path.glob('**/*') if x.is_file()]
    main = MainWindow()
    qtbot.addWidget(main)
    main.open_files(file_list=files)

    outp_temp = cfc.QuickTestOutputTemplate(
        include_header=True,
        transpose_table=True,
        decimal_mark=',',
        tests={
            'Hom': [cfc.QuickTestOutputSub(
                label='max_diff', alternative=0, columns=[5, 6, 7, 8],
                calculation='max abs', per_group=True)],
            'MTF': [cfc.QuickTestOutputSub(
                label='', alternative=0, columns=[0, 1],
                calculation='=', per_group=False)],
            'Noi': [cfc.QuickTestOutputSub(
                label='max_noise', alternative=0, columns=[1],
                calculation='max', per_group=True)]
            }
        )
    main.current_paramset = cfc.ParamSetCT(
        output=outp_temp,
        mtf_type=0, mtf_roi_size=3.0, mtf_auto_center=True)
    main.current_quicktest = cfc.QuickTestTemplate(
        tests=[[], [], ['MTF'], []])
    main.current_quicktest.tests.extend([['Hom', 'Noi']]*9)
    calculate_qc(main)
    main.wid_quicktest.extract_results(skip_questions=True)
    res = read_clipboard()
    expected_res = [
        ['MTFx 50%_img2', '1,116'],
        ['MTFx 10%_img2', '1,517'],
        ['max_diff_group0', '1,665'],
        ['max_diff_group1', '0,988'],
        ['max_noise_group0', '4,080'],
        ['max_noise_group1', '3,657']]
    assert expected_res == res.values.tolist()


def test_open_multiframe(qtbot):
    #path_tests = Path(__file__).parent
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
