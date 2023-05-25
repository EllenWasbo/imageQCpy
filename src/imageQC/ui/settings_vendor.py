#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for configuration settings part automation.

@author: Ellen Wasbo
"""
import os
from time import time
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, qApp, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QToolBar, QLabel, QLineEdit, QPushButton, QAction, QSpinBox, QCheckBox,
    QListWidget, QComboBox, QMessageBox, QDialogButtonBox, QFileDialog
    )

# imageQC block start
from imageQC.config.iQCconstants import ENV_ICON_PATH, LOG_FILENAME, VENDOR_FILE_OPTIONS
from imageQC.config import config_func as cff
from imageQC.config import config_classes as cfc
from imageQC.ui.settings_reusables import (
    StackWidget, ToolBarImportIgnore, DicomCritWidget)
from imageQC.ui.tag_patterns import TagPatternWidget, TagPatternEditDialog
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.ui_dialogs import ImageQCDialog, TextDisplay
from imageQC.scripts.mini_methods import create_empty_file, create_empty_folder
from imageQC.scripts import read_vendor_QC_reports
from imageQC.scripts import dcm
# imageQC block end


class VendorInfoWidget(StackWidget):
    """Widget holding information about reading vendor files."""

    def __init__(self, dlg_settings):
        header = 'Reading vendor QC reports'
        subtxt = (
            '''
            Vendor specific QC quite often offer some kind of report file.<br>
            imageQC have algorithms to extract values from such reports and with some
            Python skills you may contribute to add more types of reports.
            '''
            )
        super().__init__(dlg_settings, header, subtxt)
        self.vlo.addStretch()


class SiemensCTLanguageWidget(StackWidget):
    """Widget holding language settings for Siemens CT QC reports."""

    def __init__(self, dlg_settings):
        header = 'Siemens CT QC language settigs'
        subtxt = (
            'Add or edit languages for the keywords used to find values in '
            'Siemens CT Constancy or Daily QC reports.'
            )
        super().__init__(dlg_settings, header, subtxt)
        self.fname = 'vendor_siemens_ct'
        hlo = QHBoxLayout()
        self.vlo.addLayout(hlo)

        if self.import_review_mode:
            wid_common.setEnabled(False)
        else:
            btn_save = QPushButton('Save language settings')
            btn_save.setIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'))
            btn_save.clicked.connect(self.save_settings)
            if self.save_blocked:
                btn_save.setEnabled(False)
            vlo_common.addWidget(btn_save)

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_from_yaml(self, initial_template_label=''):
        """Refresh settings from yaml file.

        Using self.templates as auto_common single template and
        self.current_template as TagPatternFormat to work smoothly
        with general code.
        """
        self.lastload = time()
        _, _, self.templates = cff.load_settings(fname=self.fname)
        self.update_data()
        self.flag_edit(False)

    def update_data(self):
        """Fill GUI with current data."""
        

    def save_siemens_ct_language_settings(self):
        """Get current settings and save to yaml file."""
        ...
        self.save()

    def mark_import(self, ignore=False):
        """If import review mode: Mark full template for import or ignore."""
        if ignore:
            self.marked = False
            self.marked_ignore = True
            self.import_review_mark_txt.setText('Ignore')
        else:
            self.marked = True
            self.marked_ignore = False
            self.import_review_mark_txt.setText('Import and overwrite current')


