#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface for dialog Rename Dicom.

@author: Ellen Wasbo
"""
import os
from time import time

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QWizard, QWizardPage, QMessageBox, QVBoxLayout, QHBoxLayout,
    QLabel
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
# imageQC block end


class WizardPage(QWizardPage):
    def __init__(self):
        super().__init__()


class DummyPage(WizardPage):
    def __init__(self):
        super().__init__()

        self.setTitle("Dummy title")
        self.setSubTitle('''Dummy subtitle''')
        vLO = QVBoxLayout()
        self.setLayout(vLO)
        vLO.addWidget(
            QLabel('''A label.'''))



class AutomationWizard(QWizard):
    """GUI setup for the Automation assistant window."""

    def __init__(self, main):
        super().__init__()

        self.setWindowTitle('Automation wizard')
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOptions(
            QWizard.IndependentPages|QWizard.NoBackButtonOnStartPage)
        self.setPixmap(QWizard.WatermarkPixmap,
                       QPixmap(
                           f'{os.environ[ENV_ICON_PATH]}watermark_gears.png'))
        self.setPixmap(QWizard.LogoPixmap,
                       QPixmap(f'{os.environ[ENV_ICON_PATH]}iQC_icon128.png'))

        self.main = main

        proceed = True

        ok, path, self.auto_common = load_settings(fname='auto_common')
        ok, path, self.templates = load_settings(fname='auto_templates')
        ok, path, self.templates_vendor = load_settings(
            fname='auto_vendor_templates')
        self.lastload = time()

        self.dummyPage = DummyPage()
        self.addPage(self.dummyPage)

        #images - from where? or vendor files - from where?
        #if images:
        #parmeterset? edited? - output?
        #quicktest template? edited? - sorting? output?

        self.finished = False
        self.button(QWizard.FinishButton).clicked.connect(self.finish_clicked)

    def finish_clicked(self):
        self.finished = True
