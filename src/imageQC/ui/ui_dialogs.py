#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for different dialogs of imageQC.

@author: Ellen Wasbo
"""

import os
import copy
import numpy as np

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, qApp, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QMessageBox,
    QGroupBox, QButtonGroup, QDialogButtonBox, QSpinBox, QDoubleSpinBox, QListWidget, QTextEdit,
    QPushButton, QLabel, QRadioButton, QCheckBox, QComboBox, QFileDialog
    )

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# imageQC block start
from imageQC.config.iQCconstants import (
    APPDATA, TEMPDIR, ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER, ENV_ICON_PATH
    )
from imageQC.config.config_func import init_user_prefs
from imageQC.ui import messageboxes
from imageQC.ui import reusable_widgets as uir
from imageQC.scripts.dcm import get_projection
from imageQC.scripts.read_vendor_QC_reports import read_GE_Mammo_date
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


class WindowLevelEditDialog(ImageQCDialog):
    """Dialog to set window level by numbers."""

    def __init__(self, min_max=[0, 0], show_lock_wl=True, decimals=0,
                 positive_negative=False):
        """

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
        """Aoid close on enter if not ok button focus."""
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
        self.list_projections.currentIndexChanged.connect(self.calculate_projection)
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
        self.projection, patient_position, errmsg = get_projection(
            self.main.imgs, marked_this, self.main.tag_infos,
            projection_type=projection_type,
            direction=self.list_directions.currentText(), progress_modal=progress_modal)
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
        self.fig = matplotlib.figure.Figure(figsize=(8, 2))  #, constrained_layout=True)
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
