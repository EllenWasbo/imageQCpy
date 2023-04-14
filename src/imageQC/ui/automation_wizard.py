#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface for dialog Rename Dicom.

@author: Ellen Wasbo
"""
import os
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QCheckBox, QLineEdit
    )

# imageQC block start
from imageQC.config.iQCconstants import (
    ENV_ICON_PATH, QUICKTEST_OPTIONS, ENV_CONFIG_FOLDER
    )
from imageQC.config.config_func import (
    check_save_conflict, verify_config_folder,
    load_settings, save_settings
    )
import imageQC.ui.settings
from imageQC.config import config_classes as cfc
from imageQC.scripts import dcm
# imageQC block end


class WizardPage(QWizardPage):
    def __init__(self):
        super().__init__()


class DummyPage(WizardPage):
    def __init__(self):
        super().__init__()

        self.setTitle("Coming up...")
        self.setSubTitle('''Not implemented yet''')
        vLO = QVBoxLayout()
        self.setLayout(vLO)
        vLO.addWidget(
            QLabel('''The idea is to create automation templates from images and
                   settings currently in the main window.'''))


class StartupPage(WizardPage):
    def __init__(self, main):
        super().__init__()
        self.main = main

        self.setTitle('A walk-through to set automation templates')
        self.setSubTitle(
            '''Follow these steps to create and test an automation template <br>
            based on the currently loaded images and settings.''')
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        vlo.addWidget(QLabel('---- UNDER CONSTRUCTION - teaser only ----'))

        ok_image = QPixmap(f'{os.environ[ENV_ICON_PATH]}flag_ok.png')
        missing_image = QPixmap(f'{os.environ[ENV_ICON_PATH]}flag_missing.png')
        flo = QFormLayout()
        vlo.addLayout(flo)

        chk_imgs = QLabel()
        if len(self.main.imgs) > 0:
            chk_imgs.setPixmap(ok_image)
        else:
            chk_imgs.setPixmap(missing_image)
        flo.addRow(chk_imgs, QLabel('Load a representative set of images'))

        chk_sort = QLabel('-')
        if self.main.current_sort_pattern:
            chk_sort.setPixmap(ok_image)
        flo.addRow(chk_sort,
                   QLabel('Optionally sort the images based on DICOM header to ensure '
                          'same order each time'))

        current_qt = None
        if self.main.wid_quicktest.gb_quicktest.isChecked():
            self.main.wid_quicktest.get_current_template()
            current_qt = self.main.wid_quicktest.current_template
        chk_qt = QLabel()
        chk_qt.setPixmap(missing_image)
        if current_qt is not None:
            if any(current_qt.tests):
                chk_qt.setPixmap(ok_image)

        flo.addRow(chk_qt, QLabel(
            'Set which test to perform on which image (QuickTest pattern)'))

        chk_paramset = QLabel()
        chk_paramset.setPixmap(ok_image)
        flo.addRow(chk_paramset, QLabel(
            'Set how to perform the tests (Parameter set)'))
        flo.addRow(QLabel('-'),
                   QLabel(
                       'Optionally edit which results to output (test on next pages)'))


class SavePage(WizardPage):
    """Page to finish and save."""

    def __init__(self, main):
        super().__init__()
        self.main = main

        self.setTitle('Finish and save template')
        self.setSubTitle(
            '''Verify the parameters.''')
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        vlo.addWidget(QLabel('---- UNDER CONSTRUCTION - teaser only ----'))

        flo = QFormLayout()
        vlo.addLayout(flo)

        if len(self.main.imgs) > 0:
            statname = dcm.get_tags(
                self.main.imgs[0].filepath,
                tag_patterns=[cfc.TagPatternFormat(list_tags=['StationName'])],
                tag_infos=self.main.tag_infos)
            statname = statname[0][0]
            path_input = Path(self.main.imgs[0].filepath).parent
        else:
            statname = ''
            path_input = ''
        sort_pattern_txt = '-'
        if self.main.current_sort_pattern is not None:
            tags = self.main.current_sort_pattern.list_tags
            asc_txt = self.main.current_sort_pattern.list_sort
            tags_txt = []
            for i, tag in enumerate(tags):
                asc_txt = ('(ASC)' if self.main.current_sort_pattern.list_sort[i]
                           else '(DESC)')
                tags_txt.append(tags[i] + asc_txt)
            sort_pattern_txt = ', '.join(tags_txt)

        flo.addRow(QLabel('Modality:'), QLabel(self.main.current_modality))
        flo.addRow(QLabel('Station name:'), QLabel(statname))
        flo.addRow(QLabel('Input path:'), QLabel(str(path_input)))
        flo.addRow(QLabel('Output path:'), QLabel(''))
        flo.addRow(QLabel('Use parameter set:'),
                   QLabel(self.main.current_paramset.label))
        flo.addRow(QLabel('Use QuickTest template:'),
                   QLabel(self.main.wid_quicktest.cbox_template.currentText()))
        flo.addRow(QLabel('Sort images by:'), QLabel(sort_pattern_txt))

        chk_archive = QCheckBox()
        chk_archive.setChecked(True)
        flo.addRow(QLabel('Archive images when analysed:'), chk_archive)
        chk_active = QCheckBox()
        chk_active.setChecked(True)
        flo.addRow(QLabel('Set template active:'), chk_active)


class AutomationWizard(QWizard):
    """GUI setup for the Automation assistant window."""

    #ex https://codeloop.org/pyqt5-gui-creating-wizard-page-with-qwizard/

    def __init__(self, main):
        super().__init__()

        self.setWindowTitle('Automation wizard')
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOptions(
            QWizard.IndependentPages | QWizard.NoBackButtonOnStartPage)
        self.setPixmap(QWizard.WatermarkPixmap,
                       QPixmap(
                           f'{os.environ[ENV_ICON_PATH]}watermark_gears.png'))
        self.setPixmap(QWizard.LogoPixmap,
                       QPixmap(f'{os.environ[ENV_ICON_PATH]}iQC_icon128.png'))

        self.main = main

        proceed = True

        '''
        ok, path, self.auto_common = load_settings(fname='auto_common')
        ok, path, self.templates = load_settings(fname='auto_templates')
        ok, path, self.templates_vendor = load_settings(
            fname='auto_vendor_templates')
        self.lastload = time()
        '''

        self.addPage(StartupPage(self.main))
        self.addPage(SavePage(self.main))

        self.finished = False
        self.button(QWizard.FinishButton).clicked.connect(self.finish_clicked)

    def finish_clicked(self):
        self.finished = True
