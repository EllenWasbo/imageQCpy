#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for different dialogs of imageQC.

@author: Ellen Wasbo
"""

import os

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, qApp, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QMessageBox,
    QGroupBox, QButtonGroup, QDialogButtonBox, QSpinBox, QListWidget, QTextEdit,
    QLabel, QRadioButton, QCheckBox, QFileDialog
    )

# imageQC block start
from imageQC.config.iQCconstants import (
    APPDATA, TEMPDIR, ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER, ENV_ICON_PATH
    )
from imageQC.config.config_func import init_user_prefs
from imageQC.config.read_config_idl import ConfigIdl2Py
from imageQC.ui import messageboxes
from imageQC.ui import reusable_widgets as uir
import imageQC.resources
# imageQC block end


class ImageQCDialog(QDialog):
    """Dialog for reuse with imageQC icon and flags."""

    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

    def start_wait_cursor(self):
        """Block mouse events by wait cursor."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        qApp.processEvents()

    def stop_wait_cursor(self):
        """Return to normal mouse cursor after wait cursor."""
        QApplication.restoreOverrideCursor()


class StartUpDialog(ImageQCDialog):
    """Startup dialog if config file not found."""

    def __init__(self):
        super().__init__()
        self.setModal(True)
        self.setWindowTitle("Welcome to imageQC")

        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.addSpacing(20)
        logo = QLabel()
        im = QPixmap(':/icons/iQC_icon128.png')
        logo.setPixmap(im)
        hlo_top = QHBoxLayout()
        layout.addLayout(hlo_top)
        hlo_top.addStretch()
        hlo_top.addWidget(logo)
        hlo_top.addStretch()

        header_text = """<html><head/><body>
            <p><span style=\" font-size:14pt;\">Welcome to imageQC!</span></p>
            </body></html>"""
        hlo_header = QHBoxLayout()
        header = QLabel()
        header.setText(header_text)
        hlo_header.addStretch()
        hlo_header.addWidget(header)
        hlo_header.addStretch()
        layout.addLayout(hlo_header)

        info_text = f"""<html><head/><body>
            <p>imageQC offer options to configure settings for calculations,
            automation, visualizations and export options.<br>
            These settings can be saved at any desired location and can be
            shared between multiple users. <br>
            To let imageQC remember the path to the config folder, the path
            have to be saved locally.<br>
            As hospitals typically have different restrictions on how local
            settings can be saved these options are offered:</p>
            <ul>
            <li>Path saved on AppData <i>({APPDATA})</i></li>
            <li>Path saved on Temp <i>({TEMPDIR})</i></li>
            <li>Don't save the path, locate the path each time on startup</li>
            </ul>
            </body></html>"""
        label = QLabel()
        label.setText(info_text)
        layout.addWidget(label)
        layout.addSpacing(20)

        gb = QGroupBox('Options')
        lo = QVBoxLayout()
        gb.setLayout(lo)
        self.bGroup = QButtonGroup()

        btnTexts = [
            "Initiate user_preferences.yaml in AppData",
            "Initiate user_preferences.yaml in Temp",
            "Locate configuration folder for this session only",
            "No, I'll just have a look. Continue without config options."
            ]
        for i, t in enumerate(btnTexts):
            rb = QRadioButton(t)
            self.bGroup.addButton(rb, i)
            lo.addWidget(rb)

        self.bGroup.button(0).setChecked(True)

        self.chk_import_idl = QCheckBox(
            'Import config.dat file from IDL version'
            '(could also be done from the Settings manager later')

        layout.addWidget(gb)
        layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok,
            QtCore.Qt.Horizontal, self)
        buttons.accepted.connect(self.press_ok)
        buttons.rejected.connect(self.reject)
        layout.addSpacing(20)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_config_folder(self, ask_first=True):
        """Locate or initate config folder.

        Returns
        -------
        config_folder : str
            '' if failed or cancelled.
        """
        config_folder = ''
        locate = True
        if ask_first:
            quest = ('Locate or initiate shared configuration folder now?'
                     '(May alse be done later from the Settings manager)')
            res = messageboxes.QuestionBox(
                self, title='Shared config folder', msg=quest,
                yes_text='Yes, now', no_text='No, later')
            if res.exec() == 0:
                locate = False
        if locate:
            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.Directory)
            if dlg.exec():
                fname = dlg.selectedFiles()
                config_folder = os.path.normpath(fname[0])

        return config_folder

    def convert_idl(self):
        """Convert config.dat from IDL version to yaml files."""
        fname = QFileDialog.getOpenFileName(
            self, 'Convert config.dat from IDL version of imageQC',
            filter="dat file (*.dat)")
        if fname[0] != '':
            config_idl = ConfigIdl2Py(fname[0])
            if len(config_idl.errmsg) > 0:
                QMessageBox.warning(
                    self, 'Warnings', '\n'.join(config_idl.errmsg))
            # TODO ....

    def press_ok(self):
        """Verify selections when OK is pressed."""
        selection = self.bGroup.checkedId()
        convert_idl = self.chk_import_idl.isChecked()
        if selection == 3:
            self.reject()
        else:
            if selection in [0, 1]:
                config_folder = self.get_config_folder()
            else:
                config_folder = self.get_config_folder(ask_first=False)

            status = True
            if selection == 0:  # APPDATA
                status, user_prefs_path, errmsg = init_user_prefs(
                    path=APPDATA, config_folder=config_folder)
                if status:  # TEMPDIR
                    os.environ[ENV_USER_PREFS_PATH] = user_prefs_path
            elif selection == 1:
                status, user_prefs_path, errmsg = init_user_prefs(
                    path=TEMPDIR, config_folder=config_folder)
                if status:
                    os.environ[ENV_USER_PREFS_PATH] = user_prefs_path

            if status is False:
                QMessageBox.warning(self, 'Issue with permission', errmsg)

            os.environ[ENV_CONFIG_FOLDER] = config_folder

            if convert_idl:
                self.convert_idl()

            self.accept()

    def get_selection(self):
        """To get final selection from main window."""
        return (self.bGroup.checkedId(), self.chk_import_idl.isChecked())


class EditAnnotationsDialog(ImageQCDialog):
    """Dialog to set annotation settings."""

    def __init__(self, annotations=True, annotations_line_thick=0,
                 annotations_font_size=0):
        super().__init__()

        self.setWindowTitle('Set annotations')
        self.setMinimumHeight(300)
        self.setMinimumWidth(300)

        vlo = QVBoxLayout()
        self.setLayout(vlo)
        fLO = QFormLayout()
        vlo.addLayout(fLO)

        self.chk_annotations = QCheckBox('Show annotations')
        self.chk_annotations.setChecked(annotations)
        fLO.addRow(self.chk_annotations)

        self.spin_line = QSpinBox()
        self.spin_line.setRange(1, 10)
        self.spin_line.setValue(annotations_line_thick)
        fLO.addRow(QLabel('Line thickness'), self.spin_line)

        self.spin_font = QSpinBox()
        self.spin_font.setRange(5, 100)
        self.spin_font.setValue(annotations_font_size)
        fLO.addRow(QLabel('Font size'), self.spin_font)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

    def get_data(self):
        """Get settings.

        Returns
        -------
        bool
            annotations
        int
            line_thick
        int
            font_size
        """
        return (
            self.chk_annotations.isChecked(),
            self.spin_line.value(),
            self.spin_font.value()
            )


class QuickTestClipboardDialog(ImageQCDialog):
    """Dialog to set whether to include headers/transpose when QuickTest results."""

    def __init__(self, include_headers=True, transpose_table=False, ):
        super().__init__()

        self.setWindowTitle('QuickTest copy to clipboard settings')
        self.setMinimumHeight(200)
        self.setMinimumWidth(200)

        vlo = QVBoxLayout()
        self.setLayout(vlo)
        intro_text = (
            'With QuickTest it is possible to customize which values (from all'
            ' included tests) to output.<br>'
            'These settings can be found in output settings for the parameterset used.'
            '<br>'
            'The output will always include the acquisition date for the first image '
            'and all values are on one row or column.<br>'
            'The idea is that for repeated tests the output can be listed by date'
            'and loaded into text files or pasted to an Excel sheet.<br>'
            'Default is output as a row without headers (this is how it is used with '
            'automation). <br>'
            'Include headers to better see what the values represent when establishing '
            'your data e.g. include in your Excel sheet.'
            )
        vlo.addWidget(uir.LabelItalic(intro_text))
        vlo.addWidget(uir.HLine())

        hlo = QHBoxLayout()
        vlo.addLayout(hlo)
        hlo.addStretch()
        vlo_vals = QVBoxLayout()
        hlo.addLayout(vlo_vals)
        hlo.addStretch()

        self.chk_include_headers = QCheckBox('Include headers')
        self.chk_include_headers.setChecked(include_headers)
        vlo_vals.addWidget(self.chk_include_headers)

        self.chk_transpose_table = uir.BoolSelect(
            self, text_true='column', text_false='row (like if automation)')
        self.chk_transpose_table.setChecked(transpose_table)
        hlo_transpose = QHBoxLayout()
        hlo_transpose.addWidget(QLabel('Output values as'))
        hlo_transpose.addWidget(self.chk_transpose_table)
        vlo_vals.addLayout(hlo_transpose)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

    def get_data(self):
        """Get settings.

        Returns
        -------
        bool
            include_headers
        bool
            transpose_table
        """
        return (
            self.chk_include_headers.isChecked(),
            self.chk_transpose_table.isChecked()
            )


class ResetAutoTemplateDialog(ImageQCDialog):
    """Dialog to move directories/files in input_path/Archive to input_path."""

    def __init__(self, parent_widget, files=[], directories=[], template_name=''):
        super().__init__()
        self.setWindowTitle(f'Reset Automation template {template_name}')
        self.setMinimumHeight(300)
        self.setMinimumWidth(300)
        files_or_folders = 'files'
        if len(files) > 0:
            self.list_elements = [file.name for file in files]
        else:
            self.list_elements = [folder.name for folder in directories]
            files_or_folders = 'folders'

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        self.list_file_or_dirs = QListWidget()
        self.list_file_or_dirs.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_file_or_dirs.addItems(self.list_elements)
        self.list_file_or_dirs.setCurrentRow(len(self.list_elements) - 1)

        vlo.addWidget(QLabel(
            'Move files out of Archive to regard these files as incoming.'))
        vlo.addStretch()
        vlo.addWidget(QLabel(f'List of {files_or_folders} in Archive:'))
        vlo.addWidget(self.list_file_or_dirs)
        vlo.addStretch()
        hlo_buttons = QHBoxLayout()
        vlo.addLayout(hlo_buttons)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

    def get_idxs(self):
        """Return selected indexes in list."""
        idxs = []
        for sel in self.list_file_or_dirs.selectedIndexes():
            idxs.append(sel.row())
        return idxs


class TextDisplay(ImageQCDialog):
    """QDialog with QTextEdit.setPlainText to display text."""

    def __init__(self, parent_widget, text, title='',
                 read_only=True,
                 min_width=1000, min_height=1000):
        super().__init__()
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        txtEdit = QTextEdit('', self)
        txtEdit.setPlainText(text)
        txtEdit.setReadOnly(read_only)
        txtEdit.createStandardContextMenu()
        txtEdit.setMinimumWidth(min_width)
        txtEdit.setMinimumHeight(min_height)
        vlo.addWidget(txtEdit)
        buttons = QDialogButtonBox.Close
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

        self.setWindowTitle(title)
        self.setMinimumWidth(min_width)
        self.setMinimumHeight(min_height)
        #self.show()
