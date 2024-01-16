# -*- coding: utf-8 -*-
"""
Tests on Rename DICOM.

@author: ewas
"""
import os
from pathlib import Path

from imageQC.ui.ui_main import MainWindow
from imageQC.ui import rename_dicom
from imageQC.config.iQCconstants import (
    ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER, ENV_ICON_PATH)
from imageQC.config.config_func import get_icon_path

os.environ[ENV_USER_PREFS_PATH] = ''
os.environ[ENV_CONFIG_FOLDER] = ''
os.environ[ENV_ICON_PATH] = get_icon_path(False)
path_tests = Path(__file__).parent


def test_rename_dicom_folders(qtbot):

    file_path = path_tests / 'test_inputs' / 'CT'

    main = MainWindow()
    dlg = rename_dicom.RenameDicomDialog(main)
    qtbot.addWidget(dlg)

    dlg.path.setText(file_path.absolute().as_posix())
    dlg.wid_rename_pattern.current_template.list_tags = [
        'AcquisitionDate', 'ProtocolName']
    dlg.wid_rename_pattern.current_template.list_format = ['', '']
    dlg.generate_names()

    # as in fill_table:
    orignames = [x.name for x in dlg.original_names if x != Path(dlg.path.text())]
    newnames = [x.name for x in dlg.new_names if x != Path(dlg.path.text())]

    assert orignames[0:3] == ['Constancy13img', 'CTP404_FOV150', 'CTP486']
    assert newnames[0:3] == ['20221016_Konstanstest', '20170522_AbdomenRoutine',
                             '20161020_Snittykkelse']

    dlg.rename(testmode=True)
    # NB - when new test images are added - these messages has to be verified and edited
    assert dlg.testmode_msgs == ['Renamed 5 subfolder(s) out of 5']
    assert len(dlg.testmode_errmsgs) == 0

    # multiple same names
    dlg.wid_rename_pattern.current_template.list_tags = [
        'Modality']
    dlg.wid_rename_pattern.current_template.list_format = ['', '']
    dlg.generate_names()
    orignames = [x.name for x in dlg.original_names if x != Path(dlg.path.text())]
    newnames = [x.name for x in dlg.new_names if x != Path(dlg.path.text())]

    assert orignames[0:3] == ['Constancy13img', 'CTP404_FOV150', 'CTP486']
    assert newnames[0:3] == ['CT_000', 'CT_001', 'CT_002']


def test_rename_dicom_files(qtbot):

    file_path = path_tests / 'test_inputs' / 'CT' / 'Constancy13img'

    main = MainWindow()
    dlg = rename_dicom.RenameDicomDialog(main)
    qtbot.addWidget(dlg)

    dlg.path.setText(file_path.absolute().as_posix())
    dlg.wid_rename_pattern.current_template.list_tags2 = [
        'SeriesNumber', 'SeriesDescription', 'SliceLocation']
    dlg.wid_rename_pattern.current_template.list_format2 = ['|:03|', '|:04|', '|:.1f|']
    dlg.generate_names()

    # as in fill_table:
    orignames = [x.name for x in dlg.original_names if x != Path(dlg.path.text())]
    newnames = [x.name for x in dlg.new_names if x != Path(dlg.path.text())]
    assert orignames[0:3] == ['001_001_Topogram__1.0__Tr20_001.dcm',
                              '002_002_wire_MTF__4.8__Hr68_001.dcm',
                              '002_002_wire_MTF__4.8__Hr68_002.dcm']
    assert newnames[0:3] == ['001_Topo_31.5.dcm',
                             '002_wire_103.7.dcm',
                             '002_wire_108.5.dcm']

    dlg.rename(testmode=True)
    # NB - when new test images are added - these messages has to be verified and edited
    assert dlg.testmode_msgs == ['Renamed 13 file(s) out of 13']
    assert len(dlg.testmode_errmsgs) == 0

    # multiple same names
    dlg.wid_rename_pattern.current_template.list_tags2 = [
        'SeriesDescription']
    dlg.wid_rename_pattern.current_template.list_format2 = ['|:04|',]
    dlg.generate_names()
    orignames = [x.name for x in dlg.original_names if x != Path(dlg.path.text())]
    newnames = [x.name for x in dlg.new_names if x != Path(dlg.path.text())]
    assert orignames[0:3] == ['001_001_Topogram__1.0__Tr20_001.dcm',
                              '002_002_wire_MTF__4.8__Hr68_001.dcm',
                              '002_002_wire_MTF__4.8__Hr68_002.dcm']
    assert newnames[0:3] == ['Topo.dcm', 'wire_000.dcm', 'wire_001.dcm']


def test_rename_dicom_mix(qtbot):  # folders and subfolders dicom and non-dicom

    file_path = path_tests / 'test_inputs'

    main = MainWindow()
    dlg = rename_dicom.RenameDicomDialog(main)
    qtbot.addWidget(dlg)

    dlg.path.setText(file_path.absolute().as_posix())
    dlg.wid_rename_pattern.current_template.list_tags = [
        'Modality', 'AcquisitionDate', 'ProtocolName']
    dlg.wid_rename_pattern.current_template.list_format = ['', '', '']
    dlg.wid_rename_pattern.current_template.list_tags2 = [
        'Modality', 'SeriesNumber', 'SeriesDescription']
    dlg.wid_rename_pattern.current_template.list_format2 = ['', '|:03|', '|:04|']
    dlg.generate_names()

    # as in fill_table + test that top-path included:
    orignames = [x.name for x in dlg.original_names]  # if x != Path(dlg.path.text())]
    newnames = [x.name for x in dlg.new_names]  # if x != Path(dlg.path.text())]
    assert len(orignames) == len(newnames)
    assert newnames[0] == 'test_inputs'  # first is selected folder - not renamed
    assert 'config_idl' not in orignames  # exclude non dicom folders
    assert 'z_non_dcm_dummy_rename_test2.png' not in orignames  # exclude non dicom files
    assert orignames[-3:] == ['dummy_rename_test1.dcm',
                              'dummy_rename_test2.dcm',
                              'dummy_rename_test3.dcm']
    assert newnames[-3:] == ['CT_003_Hode.dcm', 'CT_001_Topo.dcm', 'CT_002_wire.dcm']
    idx = orignames.index('CTP591_06mm')
    assert newnames[idx] == 'CT_20170223_QA_snitt_helical'

    dlg.rename(testmode=True)
    # NB - when new test images are added - these messages has to be verified and edited
    assert dlg.testmode_msgs[0] == 'Renamed 13 subfolder(s) out of 13'
    assert dlg.testmode_msgs[1] == 'Renamed 3 file(s) out of 3 in selected folder'
    assert dlg.testmode_msgs[2] == 'Renamed 84 file(s) out of 84'
    assert len(dlg.testmode_errmsgs) == 0
