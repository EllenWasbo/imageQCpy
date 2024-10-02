#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for different dialogs of imageQC.

@author: Ellen Wasbo
"""

import os
import copy
from dataclasses import asdict
import numpy as np
from pathlib import Path
import pandas as pd

import yaml
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, qApp, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QButtonGroup, QDialogButtonBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QTextEdit, QPushButton, QLabel, QRadioButton, QCheckBox,
    QComboBox, QListWidget, QWidget, QToolBar, QAction,
    QTableWidget, QTableWidgetItem, QTabWidget, QMessageBox, QFileDialog
    )

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# imageQC block start
from imageQC.config.iQCconstants import (
    APPDATA, TEMPDIR, ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER, ENV_ICON_PATH
    )
from imageQC.config.config_func import init_user_prefs
import imageQC.config.config_classes as cfc
import imageQC.scripts.dcm as dcm
from imageQC.ui import messageboxes
from imageQC.ui import reusable_widgets as uir
from imageQC.scripts.artifact import (
    Artifact, add_artifact, edit_artifact_label, validate_new_artifact_label,
    update_artifact_3d, apply_artifacts)
from imageQC.scripts.read_vendor_QC_reports import read_GE_Mammo_date
import imageQC.resources
# imageQC block end


class ImageQCDialog(QDialog):
    """Dialog for reuse with imageQC icon and flags."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
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


class AboutDialog(ImageQCDialog):
    """Info about imageQC."""

    def __init__(self, version=''):
        super().__init__()
        self.setModal(True)
        self.setWindowTitle("imageQC")

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
            <p><span style=\" font-size:14pt;\">imageQC</span></p>
            </body></html>"""
        hlo_header = QHBoxLayout()
        header = QLabel()
        header.setText(header_text)
        hlo_header.addStretch()
        hlo_header.addWidget(header)
        hlo_header.addStretch()
        layout.addLayout(hlo_header)

        info_text = f"""<html><head/><body>
            <p>imageQC - a tool for the medical physicist working with
            DICOM images and information from medical imaging devices.<br><br>
            Author: Ellen Wasboe, Stavanger University Hospital<br>
            Current version: {version}
            </p>
            </body></html>"""
        label = QLabel()
        label.setText(info_text)
        layout.addWidget(label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok,
            QtCore.Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addSpacing(20)
        layout.addWidget(buttons)

        self.setLayout(layout)


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
            <li>Path saved in <i>({APPDATA})</i></li>
            <li>Path saved in <i>({TEMPDIR})</i></li>
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
            f"Initiate user_preferences.yaml in {APPDATA}",
            f"Initiate user_preferences.yaml in {TEMPDIR}",
            "Locate configuration folder for this session only",
            "No, I'll just have a look. Continue without config options."
            ]
        for i, t in enumerate(btnTexts):
            rb = QRadioButton(t)
            self.bGroup.addButton(rb, i)
            lo.addWidget(rb)

        self.bGroup.button(0).setChecked(True)

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
                     '(May also be done later from the Settings manager)')
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

    def press_ok(self):
        """Verify selections when OK is pressed."""
        selection = self.bGroup.checkedId()
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
                if status:
                    os.environ[ENV_USER_PREFS_PATH] = user_prefs_path
            elif selection == 1:  # TEMPDIR
                status, user_prefs_path, errmsg = init_user_prefs(
                    path=TEMPDIR, config_folder=config_folder)
                if status:
                    os.environ[ENV_USER_PREFS_PATH] = user_prefs_path

            if status is False:
                QMessageBox.warning(self, 'Issue with permission', errmsg)

            os.environ[ENV_CONFIG_FOLDER] = config_folder

            self.accept()


class SelectTextsDialog(ImageQCDialog):
    """Dialog to select texts."""

    def __init__(self, texts, title='Select texts', select_info='Select texts'):
        super().__init__()
        self.setWindowTitle(title)
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        vlo.addWidget(QLabel(select_info))
        self.list_widget = uir.ListWidgetCheckable(
            texts=texts, set_checked_ids=list(np.arange(len(texts))))
        vlo.addWidget(self.list_widget)
        self.btn_select_all = QPushButton('Deselect all')
        self.btn_select_all.clicked.connect(self.select_all)
        vlo.addWidget(self.btn_select_all)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        button_box = QDialogButtonBox(buttons)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        vlo.addWidget(button_box)

    def select_all(self):
        """Select or deselect all in list."""
        if self.btn_select_all.text() == 'Deselect all':
            set_state = Qt.Unchecked
            self.btn_select_all.setText('Select all')
        else:
            set_state = Qt.Checked
            self.btn_select_all.setText('Deselect all')

        for i in range(len(self.list_widget.texts)):
            item = self.list_widget.item(i)
            item.setCheckState(set_state)

    def get_checked_texts(self):
        """Get list of checked texts."""
        return self.list_widget.get_checked_texts()


class EditAnnotationsDialog(ImageQCDialog):
    """Dialog to set annotation settings."""

    def __init__(self, annotations=True, annotations_line_thick=0,
                 annotations_font_size=0, show_axis=False):
        super().__init__()

        self.setWindowTitle('Set display options')
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

        self.chk_show_axis = QCheckBox('Show axis')
        self.chk_show_axis.setChecked(show_axis)
        fLO.addRow(self.chk_show_axis)

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
        bool
            show_axis
        """
        return (
            self.chk_annotations.isChecked(),
            self.spin_line.value(),
            self.spin_font.value(),
            self.chk_show_axis.isChecked()
            )


class AddArtifactsDialog(ImageQCDialog):
    """Dialog to add simulated artifacts to images."""

    def __init__(self, main):
        self.forms_2d = ['circle', 'ring', 'rectangle']
        self.forms_3d = ['sphere', 'cylinder', 'rectangular prism']
        super().__init__()
        self.main = main
        self.edited = False  # True if unsaved changes

        self.setWindowTitle('Add simulated artifact')
        self.setMinimumHeight(300)
        self.setMinimumWidth(500)

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        self.tab = QTabWidget()
        wid_artifacts = QWidget()
        wid_applied_artifacts = QWidget()

        self.tab.addTab(wid_artifacts, 'Artifacts')
        self.tab.addTab(wid_applied_artifacts, 'Applied artifacts')
        vlo.addWidget(self.tab)

        # Artifacts
        vlo_a = QVBoxLayout()
        wid_artifacts.setLayout(vlo_a)
        fLO = QFormLayout()
        vlo_a.addLayout(fLO)
        self.label = QComboBox()
        self.label.setMinimumWidth(400)
        self.update_labels()
        self.label.currentIndexChanged.connect(self.label_changed)
        fLO.addRow(QLabel('Select artifact'), self.label)
        self.new_label = QLineEdit('')
        fLO.addRow(QLabel('Label new artifact'), self.new_label)
        self.form = QComboBox()
        self.form.addItems(self.forms_2d)
        self.form.addItems(self.forms_3d)
        self.form.setCurrentIndex(0)
        self.form.currentIndexChanged.connect(self.update_form)
        fLO.addRow(QLabel('Artifact form'), self.form)
        self.x_offset = QDoubleSpinBox(decimals=1)
        self.x_offset.setRange(-1000, 1000)
        self.x_offset.valueChanged.connect(self.value_edited)
        self.y_offset = QDoubleSpinBox(decimals=1)
        self.y_offset.setRange(-1000, 1000)
        self.y_offset.valueChanged.connect(self.value_edited)
        self.z_offset = QDoubleSpinBox(decimals=1)
        self.z_offset.setRange(-1000, 1000)
        self.z_offset.valueChanged.connect(self.value_edited)
        fLO.addRow(QLabel('Center offset x (mm)'), self.x_offset)
        fLO.addRow(QLabel('Center offset y (mm)'), self.y_offset)
        fLO.addRow(QLabel('Center offset z (mm)'), self.z_offset)
        self.size_1 = QDoubleSpinBox(decimals=1)
        self.size_1.setRange(0, 1000)
        self.size_1.valueChanged.connect(self.value_edited)
        self.size_1_txt = QLabel('')
        self.size_2 = QDoubleSpinBox(decimals=1)
        self.size_2.setRange(0, 1000)
        self.size_2.valueChanged.connect(self.value_edited)
        self.size_2_txt = QLabel('')
        self.size_3 = QDoubleSpinBox(decimals=1)
        self.size_3.setRange(0, 1000)
        self.size_3.valueChanged.connect(self.value_edited)
        self.size_3_txt = QLabel('')
        fLO.addRow(self.size_1_txt, self.size_1)
        fLO.addRow(self.size_2_txt, self.size_2)
        fLO.addRow(self.size_3_txt, self.size_3)
        self.rotation = QDoubleSpinBox(decimals=1)
        self.rotation.setRange(-359.9, 359.9)
        self.rotation.valueChanged.connect(self.value_edited)
        self.rotation_1 = QDoubleSpinBox(decimals=1)
        self.rotation_1.setRange(-359.9, 359.9)
        self.rotation_1.valueChanged.connect(self.value_edited)
        self.rotation_2 = QDoubleSpinBox(decimals=1)
        self.rotation_2.setRange(-359.9, 359.9)
        self.rotation_2.valueChanged.connect(self.value_edited)
        fLO.addRow(QLabel('Rotation (degrees)'), self.rotation)
        fLO.addRow(QLabel('Rotation (degrees) x'), self.rotation_1)
        fLO.addRow(QLabel('Rotation (degrees) y'), self.rotation_2)
        self.sigma = QDoubleSpinBox(decimals=2)
        self.sigma.setRange(0, 5500)
        self.sigma.valueChanged.connect(self.value_edited)
        self.sigma_label = QLabel('Gaussian blur, sigma (mm)')
        fLO.addRow(self.sigma_label, self.sigma)
        self.method = QComboBox()
        self.method.addItems(['adding', 'multiplying',
                              'adding poisson noise',
                              'adding gamma camera point source'])
        self.method.currentIndexChanged.connect(self.update_method)
        fLO.addRow(QLabel('Apply artifact value by'), self.method)
        self.value = QDoubleSpinBox(decimals=3)
        self.value.setRange(-1000000, 1000000)
        self.value.valueChanged.connect(self.value_edited)
        fLO.addRow(QLabel('Artifact value'), self.value)
        self.lbl_edited = QLabel('')

        # buttons related to list of artifacts
        toolbar_0 = QToolBar()
        hlo_tb0 = QHBoxLayout()
        vlo_a.addLayout(hlo_tb0)
        hlo_tb0.addStretch()
        hlo_tb0.addWidget(toolbar_0)

        self.act_edit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}ok_apply.png'),
            'Apply changes to selected artifact', self)
        self.act_edit.triggered.connect(self.edit)
        self.act_edit.setEnabled(False)
        act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add artifact to list of available artifacts', self)
        act_add.triggered.connect(self.add)
        act_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete artifact from list of available artifacts '
            '(and from images if applied)', self)
        act_delete.triggered.connect(self.delete)
        act_view_all = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'View all defined artifacts', self)
        act_view_all.triggered.connect(self.view_all_artifacts)
        toolbar_0.addActions([self.act_edit, act_add, act_delete, act_view_all])

        # Applied artifacts
        vlo_aa = QVBoxLayout()
        wid_applied_artifacts.setLayout(vlo_aa)
        self.cbox_imgs = QComboBox()
        self.cbox_imgs.addItems(self.get_img_names())
        self.cbox_imgs.currentIndexChanged.connect(self.selected_image_changed)
        self.list_artifacts = QListWidget()
        vlo_aa.addWidget(QLabel('Image'))
        vlo_aa.addWidget(self.cbox_imgs)
        vlo_aa.addWidget(QLabel('Applied artifacts'))
        hlo_aa = QHBoxLayout()
        vlo_aa.addLayout(hlo_aa)
        hlo_aa.addWidget(self.list_artifacts)
        toolbar = QToolBar()
        toolbar.setOrientation(Qt.Vertical)
        hlo_aa.addWidget(toolbar)

        act_clear = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}clear.png'),
            'Clear artifacts from selected image', self)
        act_clear.triggered.connect(self.image_delete_artifacts)
        act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add add artifact(s) to selected image', self)
        act_add.triggered.connect(self.image_add_artifacts)
        act_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete artifact from selected image', self)
        act_delete.triggered.connect(lambda: self.image_delete_artifacts(selected=True))
        toolbar.addActions([act_clear, act_add, act_delete])

        toolbar_all = QToolBar()
        toolbar_all.addWidget(QLabel('For all images: '))
        vlo_aa.addWidget(toolbar_all)
        act_clear_all = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}clear.png'),
            'Clear all artifacts from all images', self)
        act_clear_all.triggered.connect(
            lambda: self.image_delete_artifacts(delete_all=True))
        act_add_all = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add artifact(s) to all images', self)
        act_add_all.triggered.connect(lambda: self.image_add_artifacts(add_all=True))
        act_view_all_applied = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'View all artifacts applied to the images', self)
        act_view_all_applied.triggered.connect(self.view_all_applied_artifacts)
        toolbar_all.addActions([act_clear_all, act_add_all, act_view_all_applied])

        btn_empty = QPushButton('Start with empty image(s)')
        vlo_aa.addWidget(btn_empty)
        btn_empty.clicked.connect(self.start_with_empty)

        toolbar_btm = QToolBar()
        act_save_all = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Save artifacts, and optionally, how these are applied.', self)
        act_save_all.triggered.connect(self.save_all)

        act_import = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}import.png'),
            'Import artifacts from file.', self)
        act_import.triggered.connect(self.import_all)
        toolbar_btm.addActions([act_save_all, act_import])

        hlo_buttons_btm = QHBoxLayout()
        vlo.addLayout(hlo_buttons_btm)
        hlo_buttons_btm.addWidget(toolbar_btm)
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(self.reject)
        hlo_buttons_btm.addWidget(btn_close)

        self.update_form()

    def get_img_names(self):
        """Get names similar to the file list in main window."""
        names = []
        for idx, img in enumerate(self.main.imgs):
            if img.filepath == '':
                file_text = ' -- dummy -- '
            else:
                if self.main.cbox_file_list_display.currentIndex() == 0:
                    file_text = img.filepath
                else:
                    file_text = ' '.join(img.file_list_strings)
            names.append(f'{idx:03} {file_text}')
        return names

    def update_labels(self, set_text=''):
        """Update list of used labels."""
        used_labels = [x.label for x in self.main.artifacts]
        used_labels.insert(0, '')
        self.label.clear()
        self.label.addItems(used_labels)
        self.label.setCurrentText(set_text)

    def main_refresh(self, reset_results=True, update_image=True):
        """Reset results and/or update_image of main window."""
        if reset_results:
            self.main.results = {}
            self.main.refresh_results_display()
        if update_image:
            self.main.update_active_img(
                self.main.tree_file_list.topLevelItem(self.main.gui.active_img_no))
            self.main.refresh_img_display()

    def selected_image_changed(self):
        """Update when selected image change."""
        self.main.set_active_img(self.cbox_imgs.currentIndex())
        self.update_applied()

    def update_applied(self):
        """Update lists of applied artifacts."""
        sel_img = self.cbox_imgs.currentIndex()
        self.list_artifacts.clear()
        if self.main.imgs:
            if self.main.wid_window_level.tb_wl.get_window_level_mode() == 'dcm':
                self.main.wid_window_level.tb_wl.set_window_level(
                    'min_max', set_tools=True)
            if self.main.imgs[sel_img].artifacts:
                if len(self.main.imgs[sel_img].artifacts) > 0:
                    self.list_artifacts.addItems(self.main.imgs[sel_img].artifacts)
            self.main_refresh()

    def label_changed(self):
        """Update values when artifact label selected."""
        label = self.label.currentText()
        if label == '':
            artifact = Artifact()
            self.act_edit.setEnabled(False)
        else:
            artifact = self.get_artifact_by_label(label)
            self.act_edit.setEnabled(True)
        self.new_label.setText('')
        self.form.setCurrentText(artifact.form)
        self.x_offset.setValue(artifact.x_offset)
        self.y_offset.setValue(artifact.y_offset)
        self.z_offset.setValue(artifact.z_offset)
        self.size_1.setValue(artifact.size_1)
        self.size_2.setValue(artifact.size_2)
        self.size_3.setValue(artifact.size_3)
        self.rotation.setValue(artifact.rotation)
        self.rotation_1.setValue(artifact.rotation_1)
        self.rotation_2.setValue(artifact.rotation_2)
        self.sigma.setValue(artifact.sigma)
        self.method.setCurrentText(artifact.method)
        self.value.setValue(artifact.value)
        self.edited = False

    def get_artifact_by_label(self, label):
        """Return artifact by label."""
        available_labels = [x.label for x in self.main.artifacts]
        if label in available_labels:
            idx = available_labels.index(label)
            artifact = self.main.artifacts[idx]
        else:
            artifact = Artifact()
        return artifact

    def update_form(self):
        """Update ROI size descriptions when form changes."""
        self.value_edited()
        form = self.form.currentText()
        if form in ['circle', 'sphere', 'cylinder']:
            self.size_1_txt.setText('Radius (mm)')
            self.size_2_txt.setText('-')
            self.size_2.setEnabled(False)
            self.rotation.setEnabled(False)
        elif form == 'ring':
            self.size_1_txt.setText('Outer radius (mm)')
            self.size_2_txt.setText('Inner radius (mm)')
            self.size_2.setEnabled(True)
            self.rotation.setEnabled(False)
        else:
            self.size_1_txt.setText('Width (mm)')
            self.size_2_txt.setText('Height (mm)')
            self.size_2.setEnabled(True)
            self.rotation.setEnabled(True)
        if form in ['sphere', 'rectangular prism']:
            self.z_offset.setEnabled(True)
        else:
            self.z_offset.setEnabled(False)
        if form in ['rectangular prism', 'cylinder']:
            self.size_3_txt.setText('Depth (mm)')
            self.size_3.setEnabled(True)
            self.rotation_1.setEnabled(True)
            self.rotation_2.setEnabled(True)
        else:
            self.size_3_txt.setText('-')
            self.size_3.setEnabled(False)
            self.rotation_1.setEnabled(False)
            self.rotation_2.setEnabled(False)

    def update_method(self):
        """Update available parameters when method changes."""
        self.value_edited()
        method = self.method.currentText()
        if 'poisson' in method:
            self.value.setEnabled(False)
        else:
            self.value.setEnabled(True)
        if 'gamma camera' in method:
            self.sigma_label.setText('Source distance (mm)')
        else:
            self.sigma_label.setText('Gaussian blur, sigma (mm)')

    def get_artifact_object(self):
        """Get settings as artifact object."""
        obj = None
        errmsg = ''
        if self.size_2.value() == 0 and self.form.currentText() == 'rectangle':
            errmsg = 'ROI height cannot be zero.'
        elif self.value.value() == 0 and self.method.currentText() == 'adding':
            errmsg = 'Value of the artifact cannot be zero.'
        else:
            obj = Artifact(
                label=self.new_label.text(),
                form=self.form.currentText(),
                x_offset=self.x_offset.value(),
                y_offset=self.y_offset.value(),
                z_offset=self.z_offset.value(),
                size_1=self.size_1.value(),
                size_2=self.size_2.value(),
                size_3=self.size_3.value(),
                rotation=self.rotation.value(),
                rotation_1=self.rotation_1.value(),
                rotation_2=self.rotation_2.value(),
                sigma=self.sigma.value(),
                method=self.method.currentText(),
                value=self.value.value()
                )
            if obj.form in self.forms_3d:
                obj.type_3d = True
        if errmsg:
            QMessageBox.warning(self, 'Warning', errmsg)
        return obj

    def value_edited(self, edited=True):
        """Notify edited values."""
        if edited:
            self.edited = True
            if self.label.currentText() != '':
                self.lbl_edited.setText('* values changed')
        else:
            self.edited = False
            self.lbl_edited.setText('')

    def edit(self):
        """Edit selected artifact."""
        artifact = self.get_artifact_object()
        if artifact:
            old_label = self.label.currentText()
            if artifact.label == '':  # no new name given
                artifact.label = old_label  # keep old label?
            else:
                # already used by another?
                available_labels = [x.label for x in self.main.artifacts]
                if (
                        self.label.currentText() != artifact.label
                        and artifact.label in available_labels):
                    artifact.label = ''  # autogenerate, cannot use same as before
            new_label = validate_new_artifact_label(self.main, artifact, edit=True)
            if new_label is not None:
                if artifact.type_3d:
                    self.artifacts_3d = update_artifact_3d(
                        self.main.imgs, artifact, self.main.artifacts_3d,
                        new_label=new_label)
                artifact.label = new_label
                idx = self.label.currentIndex() - 1
                self.main.artifacts[idx] = artifact
                if old_label != new_label:
                    self.update_labels(set_text=new_label)
                    edit_artifact_label(old_label, new_label, self.main)
                self.update_applied()
                self.main_refresh()

    def add(self, artifact=None):
        """Add artifact to list of artifacts."""
        if artifact is None or artifact is False:
            artifact = self.get_artifact_object()
        if artifact:
            new_label = validate_new_artifact_label(self.main, artifact)
            if new_label is None:
                QMessageBox.warning(
                    self, 'Label already exist',
                    f'The artifact label {artifact.label} already exist. '
                    'Failed to add.')
            else:
                artifact.label = new_label
                self.main.artifacts.append(artifact)
                if artifact.type_3d:
                    self.artifacts_3d = update_artifact_3d(
                        self.main.imgs, artifact, self.main.artifacts_3d)
                self.update_labels(set_text=new_label)

    def delete(self):
        """Delete selected artifact from available artifacts."""
        idx = self.label.currentIndex() - 1
        label = self.label.currentText()
        if idx > -1:
            res = messageboxes.QuestionBox(
                self, title='Delete selected artifact',
                msg=f'Delete selected artifact {label}',
                yes_text='Yes', no_text='Cancel')
            if res.exec() == 1:
                self.main.artifacts.pop(idx)
                edit_artifact_label(label, '', self.main)
                self.update_labels()
                labels = [row[0] for row in self.main.artifacts_3d]
                if label in labels:
                    self.main.artifacts_3d.pop(idx)
                self.update_applied()
        else:
            QMessageBox.warning(
                self, 'No artifact selected',
                'No artifact selected to delete.')

    def image_delete_artifacts(self, selected=False, delete_all=False):
        """Delete all or selected applied artifacts from selected or all images."""
        if delete_all:
            for img in self.main.imgs:
                img.artifacts = None
        else:
            img_idx = self.cbox_imgs.currentIndex()
            if selected:  # only selected artifact
                sel = self.list_artifacts.selectedIndexes()
                if len(sel) > 0:
                    idx = sel[0].row()
                    self.main.imgs[img_idx].artifacts.pop(idx)
                else:
                    QMessageBox.warning(
                        self, 'No artifact selected',
                        'No artifact selected to delete.')
            else:  # clear artifacts from image
                self.main.imgs[img_idx].artifacts = None
        self.update_applied()

    def image_add_artifacts(self, add_all=False):
        """Apply artifacts to current image or all images."""
        if len(self.main.imgs) == 0:
            QMessageBox.information(self.main, 'No images loaded',
                                    'No images loaded. Can not apply artifacts.')
        else:
            labels = [x.label for x in self.main.artifacts]
            if len(labels) == 0:
                QMessageBox.information(
                    self.main, 'No artifacts available',
                    'No artifacts available. Create artifacts in tab "Artifacts".')
            else:
                dlg = SelectTextsDialog(labels, title='Select artifacts',
                                        select_info='Select artifacts')
                if dlg.exec():
                    selected_labels = dlg.get_checked_texts()
                    if len(selected_labels) > 0:
                        all_idxs = list(range(len(self.main.imgs)))
                        if add_all:
                            apply_idxs = np.copy(all_idxs)
                        else:
                            apply_idxs = [self.cbox_imgs.currentIndex()]
                        for label in selected_labels:
                            artifact_no = labels.index(label)
                            if self.main.artifacts[artifact_no].form in [
                                    'sphere', 'rectangle prism']:
                                add_artifact(label, all_idxs, self.main)
                            else:
                                add_artifact(label, apply_idxs, self.main)
                        self.update_applied()

    def start_with_empty(self):
        """Insert clear image as first artifact."""
        if len(self.main.imgs) == 0:
            QMessageBox.information(self.main, 'No images loaded',
                                    'No images loaded. Can not apply artifacts.')
        else:
            label = '***set to zero***'
            all_idxs = list(range(len(self.main.imgs)))
            res = messageboxes.QuestionBox(
                self, title='Start with empty image(s)',
                msg='Empty (set to zero) all images or just selected image',
                yes_text='All', no_text='Selected')
            if res.exec() == 1:
                apply_idxs = np.copy(all_idxs)
            else:
                apply_idxs = [self.cbox_imgs.currentIndex()]
            add_artifact(label, apply_idxs, self.main)
            self.update_applied()

    def view_all_artifacts(self):
        """View currently defined artifacts."""
        if len(self.main.artifacts) == 0:
            QMessageBox.information(self, 'Information', 'Found no artifacts.')
        else:
            dlg = DataFrameDisplay(self, pd.DataFrame(self.main.artifacts),
                              title='Currently defined artifacts',
                              min_width=1100, min_height=500)
            dlg.exec()

    def view_all_applied_artifacts(self):
        """View currently applied artifacts as text."""
        text = []
        for idx, img in enumerate(self.main.imgs):
            if img.artifacts is not None:
                text.append(f'Image {idx}:')
                for lbl in img.artifacts:
                    text.append('\t' + lbl)
        if len(text) == 0:
            QMessageBox.information(self, 'Information', 'Found no added artifacts.')
        else:
            dlg = TextDisplay(self, '\n'.join(text), title='Current applied artifacts',
                              min_width=1100, min_height=500)
            dlg.exec()

    def save_all(self):
        """Save artifacts and how these are applied to file."""
        path = ''

        def try_save(input_data):
            status = False
            try_again = False
            try:
                with open(path, 'w') as file:
                    if isinstance(input_data, list):
                        yaml.safe_dump_all(
                            input_data, file, default_flow_style=None, sort_keys=False)
                    else:
                        yaml.safe_dump(
                            input_data, file, default_flow_style=None, sort_keys=False)
                status = True
            except yaml.YAMLError:
                # try once more with eval(str(input_data))
                try_again = True
            except IOError as io_error:
                QMessageBox.warning(self, "Failed saving",
                                    f'Failed saving to {path} {io_error}')
            if try_again:
                try:
                    input_data = eval(str(input_data))
                    with open(path, 'w') as file:
                        if isinstance(input_data, list):
                            yaml.safe_dump_all(
                                input_data, file, default_flow_style=None,
                                sort_keys=False)
                        else:
                            yaml.safe_dump(
                                input_data, file, default_flow_style=None,
                                sort_keys=False)
                    status = True
                except yaml.YAMLError as yaml_error:
                    QMessageBox.warning(self, 'Failed saving',
                                        f'Failed saving to {path} {yaml_error}')
            return status

        if self.main.artifacts:
            fname = QFileDialog.getSaveFileName(
                self, 'Save artifacts',
                filter="YAML file (*.yaml)")
            proceed = False
            if fname[0] != '':
                path = fname[0]
                listofdict = [asdict(temp) for temp in self.main.artifacts]

                status = False
                try:
                    with open(path, 'w') as file:
                        yaml.safe_dump_all(
                            listofdict, file, default_flow_style=None, sort_keys=False)
                    status = True
                except yaml.YAMLError:
                    # try once more with eval(str(input_data))
                    print('yamlError')
                except IOError as io_error:
                    QMessageBox.warning(self, "Failed saving",
                                        f'Failed saving to {path} {io_error}')

                if status:
                    quest = ('Also save which artifact(s) are applied to the images?')
                    res = messageboxes.QuestionBox(
                        self, title='Save applied artifacts?', msg=quest,
                        yes_text='Yes', no_text='No')
                    if res.exec() == 1:
                        pp = Path(path)
                        path_applied = pp.parent / f'{pp.stem}_applied.yaml'
                        fname = QFileDialog.getSaveFileName(
                            self, 'Save applied artifacts', str(path_applied),
                            filter="YAML file (*.yaml)")
                        if fname[0] != '':
                            path = fname[0]
                            proceed = True
                if proceed:
                    with open(path, 'w') as file:
                        nestedlist = [info.artifacts for info in self.main.imgs]
                        yaml.safe_dump_all(nestedlist, file)
        else:
            QMessageBox.warning(self, 'Nothing to save',
                                'No artifact to save.')

    def import_all(self):
        """Import saved artifacts."""
        fname = QFileDialog.getOpenFileName(
            self, 'Open saved artifacts',
            filter="YAML file (*.yaml)")
        new_labels = []
        path_applied = ''
        if fname[0] != '':
            try:
                with open(fname[0], 'r') as file:
                    docs = yaml.safe_load_all(file)
                    for doc in docs:
                        artifact_this = Artifact(**doc)
                        new_label = validate_new_artifact_label(
                            self.main, artifact_this)
                        if new_label is None:
                            artifact_this.label = artifact_this.label + '_'
                        self.main.artifacts.append(artifact_this)
                        new_labels.append(artifact_this.label)
                if len(new_labels) == 0:
                    QMessageBox.warning(self, 'Failed loading',
                                        f'Failed loading any artifacts from {fname[0]}')
            except TypeError:
                if self.tab.currentIndex() == 0:
                    QMessageBox.warning(
                        self, 'Failed loading',
                        f'Failed loading any artifacts from {fname[0]}'
                        'Selected file not expected format.')
            except OSError as error:
                QMessageBox.warning(self, 'Failed loading',
                                    f'Failed loading {fname[0]}'
                                    f'{str(error)}')

        if self.main.artifacts and len(self.main.imgs) > 0:
            quest = 'Load saved pattern of artifact(s) applied to the images?'
            res = messageboxes.QuestionBox(
                self, title='Load applied artifacts?', msg=quest,
                yes_text='Yes', no_text='No')
            if res.exec() == 1:
                # any artifacts already applied?
                already_applied = []
                for idx, img in enumerate(self.main.imgs):
                    if img.artifacts is not None:
                        already_applied.extend(img.artifacts)
                overwrite = True
                if len(already_applied) > 0:
                    quest = 'Applied artifacts already exist. Overwrite or add.'
                    res = messageboxes.QuestionBox(
                        self, title='Overwrite or add applied artifacts?', msg=quest,
                        yes_text='Overwrite', no_text='Add')
                    if res.exec() == 0:
                        overwrite = False

                if fname[0] != '':
                    pp = Path(fname[0])
                    path_applied = pp.parent / f'{pp.stem}_applied.yaml'
                fname = QFileDialog.getOpenFileName(
                    self, 'Load saved pattern of artifacts',
                    str(path_applied), filter="YAML file (*.yaml)")
                if fname[0] != '':
                    warnings = []
                    listoflists = []
                    try:
                        with open(fname[0], 'r') as file:
                            docs = yaml.safe_load_all(file)
                            listoflists = [doc for doc in docs]
                    except OSError as error:
                        warnings.append(f'Failed reading file {fname[0]}. {str(error)}')
                    if len(listoflists) > 0:
                        for idx, img_info in enumerate(self.main.imgs):
                            try:
                                if overwrite:
                                    img_info.artifacts = listoflists[idx]
                                else:
                                    img_info.artifacts.append(listoflists[idx])
                            except IndexError:
                                warnings.append(
                                    'Saved pattern of artifacts with more images than '
                                    'currently loaded images. Pattern for missing '
                                    'images ignored.')
                                break
                    # any saved pattern loadable?
                    artifacts_available = [x.label for x in self.main.artifacts]
                    del_labels = []
                    for img_info in self.main.imgs:
                        artifacts_this = img_info.artifacts
                        if artifacts_this:
                            for artifact_label in artifacts_this:
                                if artifact_label not in artifacts_available:
                                    if artifact_label != '***set to zero***':
                                        del_labels.append(artifact_label)
                    del_labels = list(set(del_labels))
                    if del_labels:
                        warnings.append(
                            'Saved pattern of artifacts contain artifact labels not'
                            'currently loaded. These are ignored:\n'
                            f'{del_labels}')
                    for del_label in del_labels:
                        edit_artifact_label(del_label, '', self.main)

                    if len(warnings) > 0:
                        dlg = messageboxes.MessageBoxWithDetails(
                            self, title='Warnings',
                            msg='Found issues when loading pattern of artifacts',
                            info='See details',
                            icon=QMessageBox.Warning,
                            details=warnings)
                        dlg.exec()
        if len(new_labels) > 0:
            self.update_labels(set_text=new_labels[0])
        self.update_applied()


class WindowLevelEditDialog(ImageQCDialog):
    """Dialog to set window level by numbers."""

    def __init__(self, min_max=[0, 0], show_lock_wl=True, decimals=0,
                 positive_negative=False):
        """Initiate..

        Parameters
        ----------
        min_max : list of floats
            Current min and max for the window level.
        show_lock_wl : bool, optional
            Show the checkbox to set window level to locked (not update for each image).
            The default is True.
        decimals : int, optional
            Number of decimals to display. The default is 0.
        positive_negative : bool, optional
            Lock window level to be centered at zero. The default is False.
        """
        super().__init__()
        self.positive_negative = positive_negative
        self.setWindowTitle('Edit window level')
        self.setMinimumHeight(400)
        self.setMinimumWidth(300)

        vLO = QVBoxLayout()
        self.setLayout(vLO)
        fLO = QFormLayout()
        vLO.addLayout(fLO)

        self.spin_min = QDoubleSpinBox(decimals=decimals)
        self.spin_min.setRange(-1000000, 1000000)
        self.spin_min.setValue(min_max[0])
        if positive_negative:
            self.spin_min.setEnabled(False)
        else:
            self.spin_min.editingFinished.connect(
                lambda: self.recalculate_others(sender='min'))
        fLO.addRow(QLabel('Minimum'), self.spin_min)

        self.spin_max = QDoubleSpinBox(decimals=decimals)
        self.spin_max.setRange(-1000000, 1000000)
        self.spin_max.setValue(min_max[1])
        self.spin_max.editingFinished.connect(
            lambda: self.recalculate_others(sender='max'))
        fLO.addRow(QLabel('Maximum'), self.spin_max)

        self.spin_center = QDoubleSpinBox(decimals=decimals)
        self.spin_center.setRange(-1000000, 1000000)
        self.spin_center.setValue(0.5*(min_max[0] + min_max[1]))
        if positive_negative:
            self.spin_center.setEnabled(False)
        else:
            self.spin_center.editingFinished.connect(
                lambda: self.recalculate_others(sender='center'))
        fLO.addRow(QLabel('Center'), self.spin_center)

        self.spin_width = QDoubleSpinBox(decimals=decimals)
        self.spin_width.setRange(0, 2000000)
        self.spin_width.setValue(min_max[1] - min_max[0])
        self.spin_width.editingFinished.connect(
            lambda: self.recalculate_others(sender='width'))
        fLO.addRow(QLabel('Width'), self.spin_width)

        self.chk_lock = QCheckBox('')
        self.chk_lock.setChecked(True)
        if show_lock_wl:
            fLO.addRow(QLabel('Lock WL for all images'), self.chk_lock)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        vLO.addWidget(self.button_box)

    def accept(self):
        """Avoid close on enter if not ok button focus."""
        if self.button_box.button(QDialogButtonBox.Ok).hasFocus():
            if self.spin_width.value() == 0:
                QMessageBox.warning(
                    self, 'Warning',
                    'Window width should be larger than zero.')
            elif self.spin_min.value() >= self.spin_max.value():
                QMessageBox.warning(
                    self, 'Warning',
                    'Window max should be set larger than minimum.')
            else:
                super().accept()

    def recalculate_others(self, sender='min'):
        """Reset others based on input."""
        self.blockSignals(True)
        if self.positive_negative:
            if sender == 'max':
                self.spin_min.setValue(-self.spin_max.value())
            elif sender == 'width':
                self.spin_min.setValue(-self.spin_width.value()/2)
                self.spin_max.setValue(self.spin_width.value()/2)
        minval = self.spin_min.value()
        maxval = self.spin_max.value()
        width = self.spin_width.value()
        center = self.spin_center.value()
        if sender in ['min', 'max']:
            self.spin_center.setValue(0.5*(minval + maxval))
            self.spin_width.setValue(maxval-minval)
        else:  # sender in ['center', 'width']:
            self.spin_min.setValue(center - 0.5*width)
            self.spin_max.setValue(center + 0.5*width)
        self.blockSignals(False)

    def get_min_max_lock(self):
        """Get min max values an lock setting as tuple."""
        return (
            self.spin_min.value(),
            self.spin_max.value(),
            self.chk_lock.isChecked()
            )


class CmapSelectDialog(ImageQCDialog):
    """Dialog to select matplotlib colormap."""

    def __init__(self, current_cmap='gray'):
        super().__init__()
        self.setWindowTitle('Select colormap')
        self.setMinimumHeight(300)
        self.setMinimumWidth(300)
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        self.cmaps = ['gray', 'inferno', 'hot', 'rainbow', 'viridis', 'RdBu']
        self.list_cmaps = QComboBox()
        self.list_cmaps.addItems(self.cmaps)
        self.list_cmaps.setCurrentIndex(0)
        self.list_cmaps.currentIndexChanged.connect(self.update_preview)
        self.chk_reverse = QCheckBox('Reverse')
        self.chk_reverse.clicked.connect(self.update_preview)
        self.colorbar = uir.ColorBar()

        vlo.addWidget(QLabel('Select colormap:'))
        vlo.addWidget(self.list_cmaps)
        vlo.addWidget(self.chk_reverse)
        vlo.addWidget(self.colorbar)
        hlo_buttons = QHBoxLayout()
        vlo.addLayout(hlo_buttons)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

        self.update_preview()

    def update_preview(self):
        """Sort elements by name or date."""
        cmap = self.get_cmap()
        self.colorbar.colorbar_draw(cmap=cmap)

    def get_cmap(self):
        """Return selected indexes in list."""
        cmap = self.list_cmaps.currentText()
        if self.chk_reverse.isChecked():
            cmap = cmap + '_r'
        return cmap


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
            self, text_true='column',
            text_false='row (like if automation)',
            text_label='Output values as')
        self.chk_transpose_table.setChecked(transpose_table)
        vlo_vals.addWidget(self.chk_transpose_table)

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

    def __init__(self, parent_widget, files=[], directories=[], template_name='',
                 QAP_Mammo=False):
        super().__init__()
        self.setWindowTitle(f'Reset Automation template {template_name}')
        self.setMinimumHeight(300)
        self.setMinimumWidth(300)
        files_or_folders = 'files'
        self.sort_mtime = None
        if len(files) > 0:
            self.list_elements = [file.name for file in files]
            if QAP_Mammo:
                dates = []
                for file in files:
                    dd, mm, yyyy = read_GE_Mammo_date(file)
                    dates.append(f'{yyyy}{mm}{dd}')
                self.sort_mtime = np.argsort(dates)
            else:
                self.sort_mtime = np.argsort([file.stat().st_ctime for file in files])
        else:
            self.list_elements = [folder.name for folder in directories]
            files_or_folders = 'folders'

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        self.sort_by_name = False if QAP_Mammo else True
        self.txt_by_name_or_date = ['Sort by file creation date time', 'Sort by name']
        self.btn_by_name_or_date = QPushButton(
            self.txt_by_name_or_date[int(self.sort_by_name)])
        self.btn_by_name_or_date.clicked.connect(self.update_sort)
        self.list_file_or_dirs = QListWidget()
        self.list_file_or_dirs.setSelectionMode(QListWidget.ExtendedSelection)
        if self.sort_by_name:
            list_elements = copy.deepcopy(self.list_elements)
        else:
            list_elements = list(np.array(self.list_elements)[self.sort_mtime])
        self.list_file_or_dirs.addItems(list_elements)
        self.list_file_or_dirs.setCurrentRow(len(self.list_elements) - 1)

        vlo.addWidget(QLabel(
            'Select files to move out of Archive '))
        vlo.addWidget(QLabel(
            'to regard these files as incoming.'))
        vlo.addStretch()
        vlo.addWidget(QLabel(f'List of {files_or_folders} in Archive:'))
        vlo.addWidget(self.list_file_or_dirs)
        if len(files) > 0:
            vlo.addWidget(self.btn_by_name_or_date)
        vlo.addStretch()
        hlo_buttons = QHBoxLayout()
        vlo.addLayout(hlo_buttons)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

    def update_sort(self):
        """Sort elements by name or date."""
        self.list_file_or_dirs.clear()
        self.btn_by_name_or_date.setText(self.txt_by_name_or_date[
            int(self.sort_by_name)])
        self.sort_by_name = not self.sort_by_name
        list_elements = copy.deepcopy(self.list_elements)
        if self.sort_by_name is False:
            list_elements = list(np.array(list_elements)[self.sort_mtime])
        self.list_file_or_dirs.addItems(list_elements)
        self.list_file_or_dirs.setCurrentRow(len(list_elements) - 1)

    def get_idxs(self):
        """Return selected indexes in list."""
        idxs = []
        if self.sort_by_name:
            for sel in self.list_file_or_dirs.selectedIndexes():
                idxs.append(sel.row())
        else:
            orig_sort = np.arange(len(self.list_elements))
            mtime_sort = orig_sort[self.sort_mtime]
            for sel in self.list_file_or_dirs.selectedIndexes():
                idxs.append(mtime_sort[sel.row()])
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


class DataFrameDisplay(ImageQCDialog):
    """QDialog with QTextEdit.setPlainText to display text."""

    def __init__(self, parent_widget, dataframe, title='',
                 min_width=1000, min_height=1000):
        super().__init__()
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        table = QTableWidget(self)
        n_rows = len(dataframe)
        n_cols = len([*dataframe])
        table.setRowCount(n_rows)
        table.setColumnCount(n_cols)
        table.setHorizontalHeaderLabels([*dataframe])
        table.verticalHeader().setVisible(False)
        for c in range(n_cols):
            for r in range(n_rows):
                twi = QTableWidgetItem(str(dataframe.iat[r, c]))
                if c > 0:
                    twi.setTextAlignment(4)
                table.setItem(r, c, twi)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        vlo.addWidget(table)
        buttons = QDialogButtonBox.Close
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

        self.setWindowTitle(title)
        self.setMinimumWidth(min_width)
        self.setMinimumHeight(min_height)


class ProjectionPlotDialog(ImageQCDialog):
    """Dialog to generate MIP or similar with values plot above."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.cmap = main.wid_image_display.canvas.ax.get_images()[0].cmap.name
        self.setWindowTitle('Plot values on projection image')
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        self.canvas = ProjectionPlotCanvas(self)
        self.canvas.setMinimumHeight(round(0.6*self.main.gui.panel_height))
        self.canvas.setMinimumWidth(round(0.6*self.main.gui.panel_height))
        self.tb_canvas = uir.ImageNavigationToolbar(self.canvas, self,
                                                    remove_customize=True)

        self.projections = ['Average intensity projection  ',
                            'Maximum intensity projection  ']
        self.list_projections = QComboBox()
        self.list_projections.addItems(self.projections)
        self.list_projections.setCurrentIndex(0)
        self.list_projections.currentIndexChanged.connect(
            self.calculate_projection)
        self.directions = ['front', 'side']
        self.list_directions = QComboBox()
        self.list_directions.addItems(self.directions)
        self.list_directions.setCurrentIndex(0)
        self.list_directions.currentIndexChanged.connect(self.calculate_projection)

        self.headers = ['-- No results in table to plot --  ']
        if self.main.results:
            if self.main.current_test in self.main.results:
                if self.main.results[self.main.current_test]['pr_image']:
                    self.headers = self.main.results[self.main.current_test]['headers']
                else:
                    self.headers = ['-- Results pr image not available in table --']
        self.list_headers = QComboBox()
        self.list_headers.addItems(self.headers)
        self.list_headers.setCurrentIndex(0)
        self.list_headers.currentIndexChanged.connect(
            lambda: self.update_plot_values(update_figure=True))

        self.list_display_options = QComboBox()
        self.list_display_options.addItems([
            'projection + plot  ', 'projection only  ', 'plot only  '])
        self.list_display_options.setCurrentIndex(0)
        self.list_display_options.currentIndexChanged.connect(self.canvas.update_figure)
        self.list_layout = QComboBox()
        self.list_layout.addItems(['overlayed', 'horizontal'])
        self.list_layout.setCurrentIndex(0)
        self.list_layout.currentIndexChanged.connect(self.layout_changed)
        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(5, 100)
        self.spin_font_size.setValue(self.main.gui.annotations_font_size + 2)
        self.spin_font_size.valueChanged.connect(self.canvas.update_figure)
        self.chk_flip_ud = QCheckBox()
        self.chk_flip_ud.toggled.connect(self.canvas.update_figure)
        self.chk_flip_lr = QCheckBox()
        self.chk_flip_lr.toggled.connect(self.canvas.update_figure)
        self.spin_margin = QSpinBox()
        self.spin_margin.setRange(0, 40)
        self.spin_margin.setValue(20)
        self.spin_margin.valueChanged.connect(self.canvas.update_figure)
        self.spin_min = QDoubleSpinBox()
        self.spin_min.valueChanged.connect(self.canvas.update_figure)
        self.spin_max = QDoubleSpinBox()
        self.spin_max.valueChanged.connect(self.canvas.update_figure)

        hlo_selections = QHBoxLayout()
        vlo_image = QVBoxLayout()
        vlo.addLayout(hlo_selections)
        vlo.addLayout(vlo_image)
        flo = QFormLayout()
        vlo_first_col = QVBoxLayout()
        vlo_first_col.addLayout(flo)
        hlo_selections.addLayout(vlo_first_col)
        flo.addRow(QLabel('Select projection type:'), self.list_projections)
        flo.addRow(QLabel('Select projection direction:'), self.list_directions)
        flo.addRow(QLabel('Column to plot:'), self.list_headers)
        hlo_min_max = QHBoxLayout()
        vlo_first_col.addLayout(hlo_min_max)
        hlo_min_max.addWidget(QLabel('Plot min/max:'))
        hlo_min_max.addWidget(self.spin_min)
        hlo_min_max.addWidget(QLabel('/'))
        hlo_min_max.addWidget(self.spin_max)

        flo2 = QFormLayout()
        hlo_selections.addLayout(flo2)
        flo2.addRow(QLabel('Display options:'), self.list_display_options)
        flo2.addRow(QLabel('Layout:'), self.list_layout)
        flo2.addRow(QLabel('Font size:'), self.spin_font_size)
        flo2.addRow(QLabel('Left/right margin %:'), self.spin_margin)
        flo3 = QFormLayout()
        hlo_selections.addLayout(flo3)
        flo3.addRow(QLabel('Flip cran/caud:'), self.chk_flip_ud)
        flo3.addRow(QLabel('Flip left/right:'), self.chk_flip_lr)

        vlo_image.addWidget(self.tb_canvas)
        vlo_image.addWidget(self.canvas)

        hlo_buttons = QHBoxLayout()
        vlo.addLayout(hlo_buttons)
        buttons = QDialogButtonBox.Close
        buttonBox = QDialogButtonBox(buttons)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        hlo_buttons.addWidget(buttonBox)

        self.projection = None
        self.z_values = None
        self.z_label = ''
        self.plot_values = None
        self.projection_height = 1
        self.projection_width = 1

        QTimer.singleShot(300, lambda: self.calculate_projection())
        # allow time to show dialog before updating list

    def keyPressEvent(self, event):
        """Avoid close dialog on enter in widgets."""
        if event.key() == Qt.Key_Return:
            pass
        else:
            super().keyPressEvent(event)

    def reject(self):
        """Reset font size when closing."""
        plt.rcParams.update({'font.size': self.main.gui.annotations_font_size})
        super(ProjectionPlotDialog, self).reject()

    def layout_changed(self):
        """Switch layout."""
        self.blockSignals(True)
        if 'overlay' in self.list_layout.currentText():
            self.spin_margin.setValue(20)
        else:
            self.spin_margin.setValue(1)
        self.blockSignals(False)
        self.canvas.update_figure()

    def update_plot_values(self, update_figure=False):
        """Read selected result column from main results."""
        results = self.main.results
        test = self.main.current_test
        self.plot_values = None
        if test in results:
            try:
                if results[test]['pr_image']:
                    col = self.list_headers.currentIndex()
                    self.plot_values = [row[col] for row in results[test]['values']]
            except TypeError:
                pass

        if self.plot_values:
            minval = np.min(self.plot_values)
            maxval = np.max(self.plot_values)
            if maxval == minval:
                maxval = minval + 1
            diff = maxval - minval
            self.blockSignals(True)
            self.spin_min.setRange(minval - diff, maxval + diff)
            self.spin_max.setRange(minval - diff, maxval + diff)
            self.spin_min.setValue(minval - diff/10)
            self.spin_max.setValue(maxval + diff/10)
            if diff < 5:
                self.spin_min.setDecimals(2)
                self.spin_max.setDecimals(2)
            else:
                self.spin_min.setDecimals(0)
                self.spin_max.setDecimals(0)
            self.blockSignals(False)
        if update_figure:
            self.canvas.update_figure()

    def get_projection(self, img_infos, included_ids, tag_infos,
                       projection_type='max', direction='front',
                       progress_modal=None):
        """Extract projection from tomographic images.

        Parameters
        ----------
        img_infos :  list of DcmInfoGui
        included_ids : list of int
            image ids to include
        tag_infos : list of TagInfos
        projection_type : str, optional
            'max', 'min' or 'mean'. The default is 'max'.
        direction : str, optional
            'front' or 'side'. The default is 'front'.
        progress_modal : uir.ProgressModal

        Returns
        -------
        projection : np.2darray
            extracted projection from imgs
        errmsg : str
        """
        projection = None
        shape_first = None
        shape_failed = []
        errmsg = ''
        np_method = getattr(np, projection_type, None)
        axis = 0 if direction == 'front' else 1
        n_img = len(img_infos)
        patient_position = None
        tag_patterns = [cfc.TagPatternFormat(list_tags=['PatientPosition'])]
        for idx, img_info in enumerate(img_infos):
            if idx in included_ids:
                if progress_modal:
                    progress_modal.setValue(round(100*idx/n_img))
                    progress_modal.setLabelText(
                        f'Getting data from image {idx}/{n_img}')
                    if progress_modal.wasCanceled():
                        projection = None
                        progress_modal.setValue(100)
                        break
                image, tags_ = dcm.get_img(
                    img_info.filepath,
                    frame_number=img_info.frame_number, tag_infos=tag_infos,
                    tag_patterns=tag_patterns
                    )
                if len(img_info.artifacts) > 0:
                    image = apply_artifacts(
                        image, img_info,
                        self.main.artifacts, self.main.artifacts_3d, idx)
                if not patient_position:
                    try:
                        patient_position = tags_[0][0]
                    except (IndexError, KeyError):
                        pass
                profile = np_method(image, axis=axis)
                if projection is None:
                    shape_first = image.shape
                    projection = [profile]
                else:
                    if image.shape == shape_first:
                        projection.append(profile)
                    else:
                        shape_failed.append(idx)

        if len(shape_failed) > 0:
            errmsg = ('Could not generate projection due to different sizes. ' +
                      f'Image number {shape_failed} did not match the first marked image.')
            projection = None
        else:
            projection = np.array(projection)

        return (projection, patient_position, errmsg)

    def calculate_projection(self):
        """Calculate projection based on selections."""
        max_progress = 100  # %
        progress_modal = uir.ProgressModal(
            "Calculating...", "Cancel",
            0, max_progress, self, minimum_duration=0)
        proj = self.list_projections.currentText()
        projection_type = 'max'
        if 'Average' in proj:
            projection_type = 'mean'
        marked_this = self.main.get_marked_imgs_current_test()

        self.z_values = [i for i in range(len(self.main.imgs)) if i in marked_this]
        self.z_label = 'image number'
        self.projection, patient_position, errmsg = self.get_projection(
            self.main.imgs, marked_this, self.main.tag_infos,
            projection_type=projection_type,
            direction=self.list_directions.currentText(),
            progress_modal=progress_modal)
        progress_modal.setValue(max_progress)
        if patient_position:
            if patient_position[0:2] == 'HF':
                self.chk_flip_ud.setChecked(True)

        zpos_0 = self.main.imgs[marked_this[0]].zpos
        pix_0 = self.main.imgs[marked_this[0]].pix[0]
        sy, sx = self.projection.shape
        self.projection_width = sx
        self.projection_height = sy
        if zpos_0:
            zpos_1 = self.main.imgs[marked_this[1]].zpos
            slice_increment = abs(zpos_1 - zpos_0)
            self.projection_width = sx * pix_0
            self.projection_height = sy * slice_increment
        if errmsg:
            QMessageBox.information(self, 'Failed generating projection', errmsg)
            self.reject()
        else:
            self.canvas.update_figure()
        self.main.stop_wait_cursor()


class ProjectionPlotCanvas(FigureCanvasQTAgg):
    """Canvas for display of projection and plot."""

    def __init__(self, parent):
        self.fig = matplotlib.figure.Figure(figsize=(8, 2))
        self.parent = parent
        FigureCanvasQTAgg.__init__(self, self.fig)

    def update_figure(self):
        """Refresh histogram."""
        self.fig.clear()
        margin = self.parent.spin_margin.value() / 100
        self.fig.subplots_adjust(margin, .05, 1-margin, .95)
        ratio = self.parent.projection_height / self.parent.projection_width
        display = self.parent.list_display_options.currentText()
        if 'projection' in display and 'plot' in display:
            if self.parent.list_layout.currentText() == 'horizontal':
                gs = self.fig.add_gridspec(nrows=1, ncols=2)
                self.ax_img = self.fig.add_subplot(gs[0, 0])
                self.ax_plot = self.fig.add_subplot(gs[0, 1], sharey=self.ax_img)
            else:
                self.ax_img = self.fig.add_subplot(111)
                self.ax_plot = self.ax_img.twinx()
        elif 'projection' in display:
            self.ax_img = self.fig.add_subplot(111)
        elif 'plot' in display:
            self.ax_plot = self.fig.add_subplot(111)

        img_h, img_w = self.parent.projection.shape
        if 'projection' in display:
            image = self.parent.projection
            if self.parent.chk_flip_lr.isChecked():
                image = np.fliplr(image)
            if 'overlay' in self.parent.list_layout.currentText():
                image = image.transpose()
            self.ax_img.imshow(image, cmap=self.parent.cmap)
            x0, x1 = self.ax_img.get_xlim()
            y0, y1 = self.ax_img.get_ylim()
            if self.parent.list_layout.currentText() == 'horizontal':
                self.ax_img.set_aspect(float(ratio * abs((x1-x0)/(y1-y0))))
            else:
                self.ax_img.set_aspect(float(1./ratio * abs((x1-x0)/(y1-y0))))

        if 'plot' in display and self.parent.main.results:
            plt.rcParams.update({'font.size': self.parent.spin_font_size.value()})
            if self.parent.plot_values is None:
                self.parent.update_plot_values()

            if self.parent.plot_values:
                if 'overlay' in self.parent.list_layout.currentText():
                    self.ax_plot.plot(
                        self.parent.z_values, self.parent.plot_values, 'r')
                    self.ax_plot.set_ylabel(self.parent.list_headers.currentText())
                    self.ax_plot.set_ylim(
                        self.parent.spin_min.value(), self.parent.spin_max.value())
                else:
                    self.ax_plot.plot(
                        self.parent.plot_values, self.parent.z_values, 'r')
                    if 'projection' in display:
                        _, btm, _, height = self.ax_img.get_position().bounds
                        left, _, width, _ = self.ax_plot.get_position().bounds
                        self.ax_plot.set_position([left, btm, width, height])
                    self.ax_plot.set_xlabel(self.parent.list_headers.currentText())
                    self.ax_plot.set_xlim(
                        self.parent.spin_min.value(), self.parent.spin_max.value())
                if self.parent.chk_flip_ud.isChecked():
                    if 'overlay' in self.parent.list_layout.currentText():
                        self.ax_plot.invert_xaxis()
                    else:
                        self.ax_plot.invert_yaxis()
        else:
            if self.parent.chk_flip_ud.isChecked():
                if 'overlay' in self.parent.list_layout.currentText():
                    self.ax_img.invert_xaxis()
                else:
                    self.ax_img.invert_yaxis()
        if hasattr(self, 'ax_img'):
            self.ax_img.axis('off')
        self.draw()
