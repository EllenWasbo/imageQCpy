# -*- coding: utf-8 -*-
"""
Tests on GUI.

@author: ewas
"""
import os
from pathlib import Path
import pandas as pd

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
from imageQC.ui import ui_dialogs
from imageQC.ui import settings_digits

os.environ[ENV_USER_PREFS_PATH] = ''
os.environ[ENV_CONFIG_FOLDER] = ''
os.environ[ENV_ICON_PATH] = get_icon_path(False)


def test_start_open_multi(qtbot):
    main = MainWindow()
    dlg = open_multi.OpenMultiDialog(main)
    qtbot.addWidget(dlg)
    assert 1 == 1

def test_start_open_auto(qtbot):
    main = MainWindow()
    dlg = open_automation.OpenAutomationDialog(main)
    qtbot.addWidget(dlg)
    dlg2 = ui_dialogs.ResetAutoTemplateDialog(dlg)
    qtbot.addWidget(dlg2)
    assert 1 == 1

def test_start_rename_dicom(qtbot):
    main = MainWindow()
    dlg = rename_dicom.RenameDicomDialog(main)
    qtbot.addWidget(dlg)
    assert 1 == 1

def test_start_settings(qtbot):
    main = MainWindow()
    dlg = SettingsDialog(main, initial_view='Digit templates')
    qtbot.addWidget(dlg)
    dlg_digits = settings_digits.DigitTemplateEditDialog()
    qtbot.addWidget(dlg_digits)
    dlg_digits.reject()
    assert 1 == 1

def test_start_automation_wizard(qtbot):
    main = MainWindow()
    dlg = automation_wizard.AutomationWizard(main)
    #dlg.open()
    qtbot.addWidget(dlg)
    assert 1==1

def test_start_task_based(qtbot):
    main = MainWindow()
    dlg = task_based_image_quality.TaskBasedImageQualityDialog(main)
    qtbot.addWidget(dlg)
    assert 1 == 1

def test_start_ui_dialogs(qtbot):
    main = MainWindow()
    dlg = ui_dialogs.AddArtifactsDialog(main)
    qtbot.addWidget(dlg)
    dlg = ui_dialogs.AboutDialog(main)
    qtbot.addWidget(dlg)
    dlg = ui_dialogs.StartUpDialog()
    qtbot.addWidget(dlg)
    dlg = ui_dialogs.OpenRawDialog(main, [''])
    qtbot.addWidget(dlg)
    dlg = ui_dialogs.PostProcessingDialog(main)
    qtbot.addWidget(dlg)
    dlg = ui_dialogs.SelectTextsDialog([''])
    qtbot.addWidget(dlg)
    dlg = ui_dialogs.EditAnnotationsDialog()
    qtbot.addWidget(dlg)
    dlg = ui_dialogs.WindowLevelEditDialog()
    qtbot.addWidget(dlg)
    dlg = ui_dialogs.CmapSelectDialog()
    qtbot.addWidget(dlg)
    dlg = ui_dialogs.QuickTestClipboardDialog()
    qtbot.addWidget(dlg)
    dlg = ui_dialogs.TextDisplay(main, 'some text')
    qtbot.addWidget(dlg)
    data = {'col1': [0, 1, 2], 'col2': [3, 4, 5]}
    df = pd.DataFrame(data)
    dlg = ui_dialogs.DataFrameDisplay(main, df)
    qtbot.addWidget(dlg)

    file_path = Path(__file__).parent / 'test_inputs' / 'CT' / 'CTP486'
    files = [x for x in file_path.glob('**/*') if x.is_file()]
    main.open_files(file_list=files)
    dlg = ui_dialogs.ProjectionPlotDialog(main)
    qtbot.addWidget(dlg)
    assert 1 == 1
