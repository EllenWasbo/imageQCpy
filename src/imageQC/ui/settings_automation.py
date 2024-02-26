#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for configuration settings part automation.

@author: Ellen Wasbo
"""
import os
import copy
from time import time
import webbrowser
from pathlib import Path

from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QIcon, QBrush, QColor
from PyQt5.QtWidgets import (
    QApplication, qApp, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QToolBar, QLabel, QLineEdit, QPushButton, QAction, QSpinBox, QCheckBox,
    QListWidget, QComboBox, QDoubleSpinBox, QAbstractItemView, QTableWidget,
    QMessageBox, QDialogButtonBox, QInputDialog, QFileDialog, QColorDialog
    )

# imageQC block start
from imageQC.config.iQCconstants import (
    ENV_ICON_PATH, LOG_FILENAME, VENDOR_FILE_OPTIONS)
from imageQC.config import config_func as cff
from imageQC.config import config_classes as cfc
from imageQC.ui.settings_reusables import (
    StackWidget, ToolBarImportIgnore, DicomCritWidget)
from imageQC.ui.tag_patterns import TagPatternWidget, TagPatternEditDialog
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.ui_dialogs import ImageQCDialog, TextDisplay, SelectTextsDialog
from imageQC.ui import messageboxes
from imageQC.scripts.mini_methods import (
    create_empty_file, create_empty_folder, find_value_in_sublists,
    get_headers_first_values_in_path)
from imageQC.scripts.mini_methods_format import valid_template_name
from imageQC.scripts import read_vendor_QC_reports
from imageQC.scripts import dcm
from imageQC.scripts.read_vendor_QC_reports import read_vendor_template
from imageQC.dash_app import dash_app
# imageQC block end


class AutoInfoWidget(StackWidget):
    """Widget holding information about automation."""

    def __init__(self, dlg_settings):
        header = 'Automation'
        subtxt = (
            '''The main task for automation in imageQC is analysing constancy
            tests i.e. repeated tests with standardized output to follow trends.
            <br><br>
            <b>Import settings</b><br>
            ImageQC can be set to automatically sort (and rename) incoming images to
            the image pool into their respective folders defined in the Templates
            (DICOM).<br>
            Use the Import settings tab to set the details of this sorting process.
            <br><br>
            <b>Templates DICOM</b><br>
            These templates are meant for image based inputs.<br>
            Define settings for how to rename and sort incoming images into
             folders based on DICOM information.<br>
            Combine settings from parametersets and QuickTest templates to
            define how to analyse the images and where/how to output the
            results.
            <br><br>
            <b>Templates vendor reports</b><br>
            These templates are meant for vendor-report based inputs.<br>
            Define where to find the reports, type of report and output path.
            <br><br>
            <b>Dashboard settings</b><br>
            This is work in progress. Currently imageQC have no visualization tools
            for the trends (output of automation templates above).<br>
            Until that is in place, use f.x. Excel or PowerBI to visualize
            the trends.<br>
            '''
            )
        super().__init__(dlg_settings, header, subtxt)
        self.vlo.addStretch()


class AutoDeleteDialog(ImageQCDialog):
    """Dialog to set auto delete option."""

    def __init__(self, attribute, tag_infos, value='', search_path=''):
        """Initialize AutoDeleteDialog.

        Parameters
        ----------
        attribute : str
            DICOM attribute as in TagInfo
        tag_infos : list of TagInfo
        value : str
            Value of DICOM attribute (if edit). Default is ''
        search_path : str, optional
            path to sample file
        """
        super().__init__()
        self.setWindowTitle('Auto delete option')

        self.attribute = attribute
        self.tag_infos = tag_infos
        self.txt_value = QLineEdit(value)
        self.search_path = search_path

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        hlo = QHBoxLayout()
        vlo.addLayout(hlo)
        hlo.addWidget(QLabel(attribute))
        hlo.addWidget(self.txt_value)
        toolb = QToolBar()
        self.act_search = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}search.png'),
            'Get value from sample file', self)
        self.act_search.triggered.connect(self.search_value)
        toolb.addActions([self.act_search])
        hlo.addWidget(toolb)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

    def get_value(self):
        """Get value of attribute.

        Return
        ------
        value: str
        """
        return self.txt_value.text()

    def search_value(self):
        """Search for attribute value in sample file."""
        value = ''
        fname = QFileDialog.getOpenFileName(
                self, 'Get attribute value from sample file',
                self.search_path,
                filter="DICOM file (*.dcm *.IMA);;All files (*)")
        if len(fname[0]) > 0:
            tag_pattern_this = cfc.TagPatternFormat(list_tags=[self.attribute])
            tags = dcm.get_tags(
                fname[0], tag_patterns=[tag_pattern_this],
                tag_infos=self.tag_infos)
            value = tags[0][0]
        self.txt_value.setText(value)


class AutoCommonWidget(StackWidget):
    """Widget holding common settings for automation."""

    def __init__(self, dlg_settings):
        header = 'Import settings for automation'
        subtxt = (
            'Define general settings for the process of importing and sorting'
            ' incoming DICOM files from the image pool.<br>'
            'If no matching automation template is found, the import'
            ' process will simply rename the files according to the naming'
            ' template defined here.'
            )
        super().__init__(dlg_settings, header, subtxt)
        self.fname = 'auto_common'

        if self.import_review_mode:
            tb_marked = ToolBarImportIgnore(self, orientation=Qt.Horizontal)
            self.import_review_mark_txt = QLabel('Import and overwrite current')
            tb_marked.addWidget(self.import_review_mark_txt)
            hlo_import_tb = QHBoxLayout()
            hlo_import_tb.addStretch()
            hlo_import_tb.addWidget(tb_marked)
            hlo_import_tb.addStretch()
            self.vlo.addLayout(hlo_import_tb)

        wid_common = QWidget()
        self.vlo.addWidget(wid_common)
        vlo_common = QVBoxLayout()
        wid_common.setLayout(vlo_common)

        self.import_path = QLineEdit()
        self.import_path.setMinimumWidth(500)
        self.import_path.textChanged.connect(lambda: self.flag_edit(True))

        hlo_import_path = QHBoxLayout()
        hlo_import_path.addWidget(QLabel('Image pool path:'))
        hlo_import_path.addWidget(self.import_path)
        toolb = uir.ToolBarBrowse('Browse to find path')
        toolb.act_browse.triggered.connect(
            lambda: self.locate_folder(self.import_path))
        hlo_import_path.addWidget(toolb)
        vlo_common.addLayout(hlo_import_path)

        hlo = QHBoxLayout()
        vlo_left = QVBoxLayout()
        hlo.addLayout(vlo_left)
        vlo_common.addLayout(hlo)

        vlo_left.addWidget(uir.LabelHeader(
            'Auto delete incoming files with either', 4))
        self.list_auto_delete = QListWidget()
        self.list_auto_delete.setFixedWidth(400)
        hlo_auto_delete = QHBoxLayout()
        hlo_auto_delete.addWidget(self.list_auto_delete)

        self.tb_auto_delete = QToolBar()
        self.tb_auto_delete.setOrientation(Qt.Vertical)
        self.btn_push_crit_delete = QPushButton('<<')
        self.btn_push_crit_delete.setToolTip(
            'Add tag from DICOM tag list to Auto delete list with criterium.')
        self.btn_push_crit_delete.clicked.connect(self.push_auto_delete)
        self.tb_auto_delete.addWidget(self.btn_push_crit_delete)
        self.act_edit_auto_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit selected row', self)
        self.act_edit_auto_delete.triggered.connect(self.edit_auto_delete)
        self.act_pop_auto_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete selected row', self)
        self.act_pop_auto_delete.triggered.connect(self.pop_auto_delete)
        self.tb_auto_delete.addActions([
            self.act_edit_auto_delete, self.act_pop_auto_delete])
        hlo_auto_delete.addWidget(self.tb_auto_delete)
        vlo_left.addLayout(hlo_auto_delete)
        self.chk_auto_delete_empty_folders = QCheckBox(
            'Auto delete empty subfolders in image pool after import')
        self.chk_auto_delete_empty_folders.stateChanged.connect(
            lambda: self.flag_edit(True))
        vlo_left.addWidget(self.chk_auto_delete_empty_folders)
        vlo_left.addSpacing(20)

        vlo_left.addWidget(uir.LabelHeader(
            'Append or overwrite log file', 4))
        self.cbox_log = QComboBox()
        self.cbox_log.addItems(['overwrite', 'append'])
        self.cbox_log.currentIndexChanged.connect(lambda: self.flag_edit(True))
        hlo_log = QHBoxLayout()
        vlo_left.addLayout(hlo_log)
        hlo_log.addWidget(self.cbox_log)
        tb_log = QToolBar()
        act_info_log = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}info.png'),
            'More about the log file', tb_log)
        act_info_log.triggered.connect(self.info_log)
        act_view_log = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'View log file', tb_log)
        act_view_log.triggered.connect(self.view_log)
        tb_log.addActions([act_info_log, act_view_log])
        hlo_log.addWidget(tb_log)
        vlo_left.addSpacing(20)

        vlo_left.addWidget(uir.LabelHeader(
            'Ignore (leave unsorted) if old images', 4))
        self.chk_ignore_since = QCheckBox('Yes, ignore if more than ')
        self.chk_ignore_since.stateChanged.connect(lambda: self.flag_edit(True))
        self.ignore_since = QSpinBox()
        self.ignore_since.valueChanged.connect(lambda: self.flag_edit(True))
        self.ignore_since.setRange(1, 100)
        hlo_ignore_since = QHBoxLayout()
        vlo_left.addLayout(hlo_ignore_since)
        hlo_ignore_since.addWidget(self.chk_ignore_since)
        hlo_ignore_since.addWidget(self.ignore_since)
        hlo_ignore_since.addWidget(QLabel(' days old'))
        vlo_left.addSpacing(20)

        vlo_left.addWidget(uir.LabelHeader(
            'Default appearance if Automation run with GUI', 4))

        self.chk_display_images = QCheckBox(
            'Display images/rois while tests are run')
        self.chk_display_images.stateChanged.connect(
            lambda: self.flag_edit(True))
        vlo_left.addWidget(self.chk_display_images)

        hlo.addWidget(uir.VLine())
        vlo_right = QVBoxLayout()
        hlo.addLayout(vlo_right)
        vlo_right.addWidget(uir.LabelHeader('Import rename pattern', 4))
        self.wid_tag_pattern = TagPatternWidget(
            self, typestr='format', lock_on_general=True)
        vlo_right.addWidget(self.wid_tag_pattern)

        if self.import_review_mode:
            wid_common.setEnabled(False)
        else:
            btn_save = QPushButton('Save import settings')
            btn_save.setIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'))
            btn_save.clicked.connect(self.save_auto_common)
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
        _, _, self.tag_infos = cff.load_settings(fname='tag_infos')
        self.update_data()
        self.flag_edit(False)

    def update_data(self):
        """Fill GUI with current data."""
        self.current_template = self.templates.filename_pattern
        self.wid_tag_pattern.fill_list_tags('', avoid_special_tags=True)
        self.wid_tag_pattern.update_data()
        self.import_path.setText(self.templates.import_path)
        self.chk_auto_delete_empty_folders.setChecked(
            self.templates.auto_delete_empty_folders)
        self.cbox_log.setCurrentIndex(['w', 'a'].index(self.templates.log_mode))
        self.ignore_since.setValue(self.templates.ignore_since)
        self.chk_ignore_since.setChecked(self.templates.ignore_since > 0)
        self.chk_display_images.setChecked(self.templates.display_images)
        self.fill_auto_delete_list()

    def save_auto_common(self):
        """Get current settings and save to yaml file."""
        self.templates.import_path = self.import_path.text()
        if self.cbox_log.currentText() == 'overwrite':
            self.templates.log_mode = 'w'
        else:
            self.templates.log_mode = 'a'
        if self.chk_ignore_since.isChecked():
            self.templates.ignore_since = self.ignore_since.value()
        else:
            self.templates.ignore_since = 0
        self.templates.auto_delete_empty_folders = (
            self.chk_auto_delete_empty_folders.isChecked())
        self.templates.display_images = self.chk_display_images.isChecked()
        self.save()

    def fill_auto_delete_list(self):
        """Fill list of auto delete settings from current_template."""
        self.list_auto_delete.clear()
        if len(self.templates.auto_delete_criterion_attributenames) > 0:
            for i, attr in enumerate(
                    self.templates.auto_delete_criterion_attributenames):
                txt = f'{attr} = {self.templates.auto_delete_criterion_values[i]}'
                self.list_auto_delete.addItem(txt)

    def push_auto_delete(self):
        """Push currently selected DICOM tag to list of auto delete options."""
        sel_indexes = self.wid_tag_pattern.list_tags.selectedIndexes()
        if len(sel_indexes) > 0:
            rowno = sel_indexes[0].row()
            tag = self.wid_tag_pattern.list_tags.item(rowno).text()
            dlg = AutoDeleteDialog(
                tag, tag_infos=self.tag_infos,
                search_path=self.import_path.text())
            res = dlg.exec()
            if res:
                value = dlg.get_value()
                self.templates.auto_delete_criterion_attributenames.append(tag)
                self.templates.auto_delete_criterion_values.append(value)
                self.fill_auto_delete_list()
                self.flag_edit(True)
        else:
            QMessageBox.warning(
                self, 'Select a tag',
                'Select a tag from the DICOM tag list to push to the Auto delete list.')

    def edit_auto_delete(self):
        """Edit selected auto delete option."""
        sel_indexes = self.list_auto_delete.selectedIndexes()
        if len(sel_indexes) > 0:
            rowno = sel_indexes[0].row()
            dlg = AutoDeleteDialog(
                self.templates.auto_delete_criterion_attributenames[rowno],
                value=self.templates.auto_delete_criterion_values[rowno],
                tag_infos=self.tag_infos,
                search_path=self.import_path.text())
            res = dlg.exec()
            if res:
                value = dlg.get_value()
                self.templates.auto_delete_criterion_values[rowno] = value
                self.fill_auto_delete_list()
                self.flag_edit(True)
        else:
            QMessageBox.information(
                self, 'No row selected', 'Select a row to edit.')

    def pop_auto_delete(self):
        """Delete selected auto-delete option."""
        sel_indexes = self.list_auto_delete.selectedIndexes()
        if len(sel_indexes) > 0:
            idx = sel_indexes[0].row()
            self.templates.auto_delete_criterion_attributenames.pop(idx)
            self.templates.auto_delete_criterion_values.pop(idx)
            self.fill_auto_delete_list()
            self.flag_edit(True)
        else:
            QMessageBox.information(
                self, 'No row selected', 'Select a row to delete.')

    def info_log(self):
        """Show info about log."""
        _, path, _ = cff.load_user_prefs()
        text = [
            'A (local) log will be genereted during import from image pool and',
            'as automation templates are run. This log will be saved at the',
            'same location as the local user settings:',
            f'{path}',
            'The log may be rewritten each time import or automation is',
            'initiated or the log may append to the existing log.'
            ]
        dlg = TextDisplay(
            self, '\n'.join(text),
            title='About the automation log',
            min_width=500, min_height=300)
        res = dlg.exec()
        if res:
            pass  # just to wait for user to close message

    def view_log(self):
        """Display log file contents."""
        _, path, _ = cff.load_user_prefs()
        log_path = Path(path).parent / LOG_FILENAME
        path_log = str(log_path)
        if os.path.exists(path_log):
            os.startfile(path_log)

    def mark_import(self, ignore=False):
        """If import review mode: Mark AutoCommon for import or ignore."""
        if ignore:
            self.marked = False
            self.marked_ignore = True
            self.import_review_mark_txt.setText('Ignore')
        else:
            self.marked = True
            self.marked_ignore = False
            self.import_review_mark_txt.setText('Import and overwrite current')


class LimitsAndPlotRow(QWidget):
    """Widget for each parameter to set or unset."""

    def __init__(self, parent, text=''):
        super().__init__()
        self.parent = parent
        hlo = QHBoxLayout()
        self.gb_enable = QGroupBox()
        hlo_enable = QHBoxLayout()
        self.gb_enable.setLayout(hlo_enable)
        self.setLayout(hlo)
        self.checkbox = QCheckBox(text)
        self.checkbox.clicked.connect(self.state_edited)
        self.spin = QDoubleSpinBox()
        self.spin.editingFinished.connect(self.parent.flag_edit)
        self.ref_float = None

        hlo.addWidget(self.checkbox)
        hlo.addWidget(self.gb_enable)
        hlo_enable.addWidget(self.spin)

    def set_data(self, value, min_value=None, max_value=100,
                 decimals=1, ref_value=None):
        """Set data based on input. Validate ref_value if float.

        Parameters
        ----------
        value : float or None
            from defined group limits or ranges
        min_value : float, optional
            minimum value for the spinbox. The default is None.
            if None, minimum = -maximum
        max_value : float, optional
            maximum value for the spinbox. The default is 100.
        decimals : int, optional
            number of decimals to display. The default is 1.
        ref_value : float or None, optional
            sample value. The default is None.
        """
        self.blockSignals(True)
        self.checkbox.setChecked(value is not None)

        if value is None:
            self.gb_enable.setEnabled(False)
        else:
            self.gb_enable.setEnabled(True)
        self.spin.setDecimals(decimals)
        self.spin.setMaximum(max_value)
        if min_value is None:
            self.spin.setMinimum(-max_value)
        else:
            self.spin.setMinimum(min_value)
        if isinstance(value, (int, float)):
            self.spin.setValue(value)
        else:
            self.spin.setValue(0)
        self.ref_float = ref_value
        self.blockSignals(False)

    def state_edited(self):
        """Actions when checkbox state change by user."""
        if self.checkbox.isChecked():
            if self.ref_float is None:
                self.gb_enable.setEnabled(False)
            else:
                self.gb_enable.setEnabled(True)
        else:
            self.gb_enable.setEnabled(False)
        self.parent.flag_edit()

    def get_data(self):
        """Return value of spinbox.

        Returns
        -------
        value : float
        """
        value = None
        if self.checkbox.isChecked():
            value = self.spin.value()
        return value


class LimitsAndPlotContent(QWidget):
    """Widget for limits and plot settings used for both StackWidget and edit dialog."""

    def __init__(self, parent, sample_file_path='', initial_template_label='',
                 type_vendor=False):
        """Generate widget content to edit limits template.

        Parameters
        ----------
        parent : StackWidget or None
            parent holds the current_template
            if StackWidget register changes with parent.flag_edit(True)
        sample_file_path : str, optional
            Result file used to get column headers and sample values. The default is ''.
        initial_template_label : str, optional
            limits template label to watch or edit (start with). The default is ''.
        type_vendor : bool, optional
            Is the limits template used for vendor type automation?
            The default is False.
        """
        super().__init__()
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        self.initial_template_label = initial_template_label
        self.type_vendor = type_vendor
        self.parent = parent

        self.headers = []  # column headers flattened from template.groups
        self.group_numbers = []  # group number for each element in headers

        self.cbox_output_paths = QComboBox()
        self.cbox_output_paths.currentIndexChanged.connect(self.update_output_path)
        self.txt_sample_file_path = QLineEdit(sample_file_path)

        self.list_headers = QListWidget()
        self.list_headers.setFixedWidth(300)
        self.list_headers.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_headers.currentItemChanged.connect(self.item_selected)
        self.list_headers.itemClicked.connect(self.update_data_selected)

        self.lbl_sample_value = uir.LabelItalic('')
        self.txt_title = QLineEdit('')
        self.txt_title.editingFinished.connect(self.flag_edit)
        self.min_tolerance = LimitsAndPlotRow(self, text='Tolerance (min)')
        self.max_tolerance = LimitsAndPlotRow(self, text='Tolerance (max)')
        self.min_range = LimitsAndPlotRow(self, text='Y plot range (min)')
        self.max_range = LimitsAndPlotRow(self, text='Y plot range (max)')
        self.chk_hide = QCheckBox('Hide from plots in results dashboard')
        self.chk_hide.clicked.connect(self.flag_edit)

        btn_group_selected = QPushButton('Group selected')
        btn_group_selected.clicked.connect(self.group_selected)
        btn_ungroup_selected = QPushButton('Ungroup selected')
        btn_ungroup_selected.clicked.connect(self.ungroup_selected)

        if isinstance(self.parent, LimitsAndPlotWidget):
            hlo_outputs = QHBoxLayout()
            vlo.addLayout(hlo_outputs)
            hlo_outputs.addWidget(QLabel('Linked output paths: '))
            hlo_outputs.addWidget(self.cbox_output_paths)
            hlo_outputs.addStretch()

        hlo_sample = QHBoxLayout()
        vlo.addLayout(hlo_sample)
        self.txt_sample_file_path.setMinimumWidth(500)
        hlo_sample.addWidget(QLabel('Sample file:'))
        hlo_sample.addWidget(self.txt_sample_file_path)
        toolb = uir.ToolBarBrowse('Locate sample results file')
        toolb.act_browse.triggered.connect(self.locate_sample_file)
        hlo_sample.addWidget(toolb)

        hlo_content = QHBoxLayout()
        vlo.addLayout(hlo_content)
        vlo_list = QVBoxLayout()
        hlo_content.addLayout(vlo_list)
        vlo_list.addWidget(uir.LabelItalic('Column headers'))
        vlo_list.addWidget(self.list_headers)
        vlo_list.addWidget(btn_group_selected)
        vlo_list.addWidget(btn_ungroup_selected)

        toolb = QToolBar()
        toolb.setOrientation(Qt.Vertical)
        act_up = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
            'Move tag(s) up in pattern list', self)
        act_up.triggered.connect(lambda: self.move_group(direction='up'))
        act_down = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
            'Move tag(s) down in pattern list', self)
        act_down.triggered.connect(lambda: self.move_group(direction='down'))
        toolb.addSeparator()
        toolb.addActions([act_up, act_down])
        hlo_content.addWidget(toolb)

        gb_rows = QGroupBox('Settings for selected header/group')
        gb_rows.setFont(uir.FontItalic())
        vlo_rows = QVBoxLayout()
        gb_rows.setLayout(vlo_rows)
        vlo_gb = QVBoxLayout()
        hlo_content.addLayout(vlo_gb)
        vlo_gb.addWidget(gb_rows)
        vlo_rows.addWidget(self.lbl_sample_value)
        hlo_limits = QHBoxLayout()
        vlo_rows.addLayout(hlo_limits)
        hlo_limits.addWidget(self.min_tolerance)
        hlo_limits.addWidget(self.max_tolerance)
        hlo_ranges = QHBoxLayout()
        vlo_rows.addLayout(hlo_ranges)
        hlo_ranges.addWidget(self.min_range)
        hlo_ranges.addWidget(self.max_range)
        hlo_title = QHBoxLayout()
        vlo_rows.addLayout(hlo_title)
        hlo_title.addWidget(QLabel('Plot title'))
        hlo_title.addWidget(self.txt_title)
        vlo_rows.addWidget(self.chk_hide)
        vlo_gb.addStretch()
        hlo_content.addStretch()

    def flag_edit(self, flag=True):
        """Update current_template and send flag edited info to parent widget.

        Parameters
        ----------
        flag : bool, optional
            The default is True.
        """
        sels = self.list_headers.selectedIndexes()
        row = sels[0].row()
        group_idx = self.group_numbers[row]
        self.parent.current_template.groups_limits[group_idx] = [
            self.min_tolerance.get_data(), self.max_tolerance.get_data()]
        self.parent.current_template.groups_ranges[group_idx] = [
            self.min_range.get_data(), self.max_range.get_data()]
        self.parent.current_template.groups_hide[group_idx] = self.chk_hide.isChecked()
        self.parent.current_template.groups_title[group_idx] = self.txt_title.text()
        self.parent.flag_edit(flag)

    def get_current_modality(self):
        """Find current modality from self.parent."""
        if 'LimitsAndPlotWidget' in str(type(self.parent)):
            current_modality = self.parent.current_modality
        else:
            try:
                current_modality = self.parent.current_modality
            except AttributeError:
                pass
        return current_modality

    def locate_sample_file(self):
        """Locate sample output file from selecting auto_(vendor)_template."""
        type_vendor = self.type_vendor
        current_modality = self.get_current_modality()
        if 'LimitsAndPlotWidget' in str(type(self.parent)):
            res = messageboxes.QuestionBox(
                parent=self, title='Locate output file',
                msg='Locate output file from DICOM or vendor templates?',
                yes_text='DICOM based templates',
                no_text='Vendor report templates')
            if res.exec():
                type_vendor = False
            else:
                type_vendor = True

        if type_vendor:
            temps = self.parent.auto_vendor_templates[current_modality]
        else:
            temps = self.parent.auto_templates[current_modality]

        temp_outs = [
            (temp.label, temp.path_output)
            for temp in temps if temp.path_output != '']
        labels = [x[0] for x in temp_outs]
        label, ok = QInputDialog.getItem(
            self, "Select automation template",
            "Output file from automation template:   ", labels, 0, False)
        if ok:
            ok = True if label in labels else False

        if ok and label:
            paths = [x[1] for x in temp_outs]
            path = paths[labels.index(label)]
            self.txt_sample_file_path.setText(path)
            self.update_from_sample_file()

    def generate_empty_template(self):
        """Generate empty template based on available headers."""
        groups = [[header] for header in self.headers]
        self.parent.current_template = cfc.LimitsAndPlotTemplate(
            groups=groups, type_vendor=self.type_vendor)
        self.group_numbers = list(range(len(self.headers)))

    def set_template_from_label(self, label=''):
        """Set current template to already defined template with given label."""
        if label == '':
            self.generate_empty_template()
        else:
            current_modality = self.get_current_modality()
            labels = [temp.label for temp in self.parent.templates[current_modality]]
            if label in labels:
                idx = labels.index(label)
                self.parent.current_template = copy.deepcopy(
                    self.parent.templates[current_modality][idx])

    def update_header_order(self):
        """Update self.header/group_numbers/first_values after (un)grouping+.

        Return
        ------
        ignored_headers : list of tuple
            tuple = (group_index, header)
            if any header in original header list not in updated template
        """
        orig_headers = copy.deepcopy(self.headers)
        orig_first_values = copy.deepcopy(self.first_values)
        ignored = []
        self.headers = []
        self.group_numbers = []
        self.first_values = []
        for idx, group in enumerate(self.parent.current_template.groups):
            self.group_numbers.extend([idx] * len(group))
            for header in group:
                if header in orig_headers:
                    self.headers.append(header)
                    orig_idx = orig_headers.index(header)
                    self.first_values.append(orig_first_values[orig_idx])
                else:
                    ignored.append((idx, header))
        return ignored

    def validate_headers(self):
        """Validate that all headers in sample file can be found in template.

        Missing headers in sample file are ignored (deleted if accepted).
        """
        flatten_groups = [
            elem for sublist in self.parent.current_template.groups for elem in sublist]
        set_template_headers = set(flatten_groups)
        if hasattr(self.parent, 'output_headers'):
            flatten_output_headers = [
                elem for sublist in self.parent.output_headers for elem in sublist]
            set_sample_headers = set(flatten_output_headers)
        else:
            set_sample_headers = set(self.headers)
        missing_in_template = list(set_template_headers.difference(set_sample_headers))
        missing_in_sample = list(set_sample_headers.difference(set_template_headers))

        if len(set_sample_headers) > 0:
            if len(missing_in_template) > 0 and len(missing_in_sample) == 0:
                # Add to template or ignore?
                res = messageboxes.QuestionBox(
                    parent=self, title='Add or ignore missing headers?',
                    msg=('The LimitsAndPlot template are missing headers that are '
                         'found in the connected automation output file(s).'),
                    info='See details for the missing headers.',
                    details=missing_in_template,
                    yes_text='Add missing headers to LimitsAndPlot template',
                    no_text='Ignore missing headers')
                if res.exec():  # overwrite
                    for header in missing_in_template:
                        self.parent.current_template.add_group([header])
                    self.parent.flag_edit(True)
            elif len(missing_in_template) == 0 and len(missing_in_sample) > 0:
                res = messageboxes.MessageBoxWithDetails(
                    parent=self, title='Mismatching headers',
                    msg=('Some headers found in the LimitsAndPlot template cannot be '
                         'found in any of the connected automation output files. '
                         'Consider deleting these '
                         'headers from the LimitsAndPlot template.'),
                    info='See these headers in details.',
                    details=missing_in_sample)
                if res.exec():
                    pass
            elif len(missing_in_template) > 0 and len(missing_in_sample) > 0:
                proceed = True
                if hasattr(self.parent, 'output_headers'):
                    # same headers all files?
                    if len(self.parent.output_headers) > 1:
                        first_headers = self.parent.output_headers[0]
                        for headers in self.parent.output_headers:
                            if headers != first_headers:
                                proceed = False
                    if proceed is False:
                        details = ['Headers in output files']
                        for i, path in enumerate(self.parent.output_paths):
                            details.append(f'{path}:')
                            details.extend(self.parent.output_headers[i])
                        details.extend(['', 'Missing in template:'])
                        details.extend(missing_in_template)
                        details.extend(['', 'Missing in output file(s):'])
                        details.extend(missing_in_sample)
                        res = messageboxes.MessageBoxWithDetails(
                            parent=self, title='Mismatching headers',
                            msg=('Found mismatch between headers found in the Limits '
                                 'and plot template and headers found in at least one '
                                 'of the connected automation output files. The '
                                 'headers of connected output files differ. '
                                 'Consider connecting these to separate '
                                 'Limits and plot templates.'),
                            info='See mismatch headers in details.',
                            details=details)
                        if res.exec():
                            pass
                if proceed:
                    if hasattr(self.parent, 'output_headers'):
                        output_paths = self.parent.output_paths
                    else:
                        output_paths = [self.txt_sample_file_path.text()]
                    dlg = LimitsAndPlotFixHeadersDialog(
                        self,
                        limits_template=copy.deepcopy(self.parent.current_template),
                        output_paths=output_paths,
                        headers_in_output=self.headers,
                        headers_in_template=flatten_groups
                        )
                    res = dlg.exec()
                    if res:
                        self.parent.current_template = dlg.get_template()
                        self.headers = dlg.get_headers()
                        self.parent.flag_edit(True)
                    else:
                        QMessageBox.warning(
                            self, 'Mismatch persist',
                            'The mismatch still persist. To trigger the dailog to fix '
                            'this template again (the one you just canceled), select '
                            'another template and then select the current template '
                            'once again.')
        self.group_numbers = []
        for idx, group in enumerate(self.parent.current_template.groups):
            self.group_numbers.extend([idx] * len(group))

    def update_output_path(self):
        """Update textfield with sample path when another output path is selected."""
        try:
            self.txt_sample_file_path.setText(
                self.parent.output_paths[self.cbox_output_paths.currentIndex()])
        except IndexError:
            self.txt_sample_file_path.setText('')

    def update_from_sample_file(self, silent=False):
        """Find headers and sample data from sample file or startup if None."""
        self.headers = []
        self.first_values = None
        headers_as_groups = False
        if os.path.exists(self.txt_sample_file_path.text()):
            self.headers, self.first_values = get_headers_first_values_in_path(
                self.txt_sample_file_path.text())
            if len(self.headers) == 0:
                headers_as_groups = True
        elif self.txt_sample_file_path.text() == '':
            if self.parent.current_template is not None:
                headers_as_groups = True
        if headers_as_groups:
            self.headers = [
                elem for sublist in self.parent.current_template.groups
                for elem in sublist]

        if self.first_values is None:
            if silent is False:
                QMessageBox.information(
                    self, 'No output yet',
                    'To set the limits and plot settings there should be column headers'
                    ' of a sample file to extract headers from. '
                    'Output file is empty or are missing column headers.')
            self.first_values = ['' for i in range(len(self.headers))]
            self.group_numbers = []
            for idx, group in enumerate(self.parent.current_template.groups):
                self.group_numbers.extend([idx] * len(group))
        else:
            if all([self.initial_template_label == '',
                   self.parent.current_template is None]):
                # startup no input template
                self.generate_empty_template()
            elif all([self.initial_template_label != '',
                     self.parent.current_template is None]):
                # startup with input template
                self.set_template_from_label(label=self.initial_template_label)
                self.validate_headers()
            else:
                if len(self.parent.current_template.groups) == 0:  # empty template
                    self.generate_empty_template()
                else:
                    # changed sample file
                    self.validate_headers()
        self.update_data()

    def reset_data_display(self):
        """Set all data back to default."""
        self.min_tolerance.set_data(None)
        self.max_tolerance.set_data(None)
        self.min_range.set_data(None)
        self.max_range.set_data(None)
        self.blockSignals(True)
        self.chk_hide.setChecked(False)
        self.txt_title.setText('')
        self.blockSignals(False)

    def update_data(self, set_selected_idx=0):
        """Refresh list and trigger refresh of selected list item data."""
        self.list_headers.clear()
        if len(self.headers) > 0:
            self.list_headers.addItems(self.headers)
            self.list_headers.setCurrentRow(set_selected_idx)
        else:
            self.reset_data_display()

    def item_selected(self, new, old):
        """When item is selected in list - make sure actually selected."""
        if new is not None:
            if new.isSelected():
                self.update_data_selected()

    def update_data_selected(self):
        """Refresh data for selected column header and highlight all in same group."""
        sels = self.list_headers.selectedIndexes()
        if len(sels) > 0:
            sel_rows = [sel.row() for sel in sels]
            header_first_select = self.headers[sel_rows[0]]
            idx_in_sublists = find_value_in_sublists(
                self.parent.current_template.groups, header_first_select)
            if len(idx_in_sublists) > 0:
                group_idx = idx_in_sublists[0]
            else:
                group_idx = 0

            brush = QBrush(QColor(110, 148, 192))
            brush_default = QBrush(QColor(255, 255, 255, 0))
            for row, header in enumerate(self.headers):
                if header in self.parent.current_template.groups[group_idx]:
                    self.list_headers.item(row).setBackground(brush)
                else:
                    self.list_headers.item(row).setBackground(brush_default)

            # show current values
            try:
                sample_val = self.first_values[sel_rows[0]]
            except (IndexError, ValueError):
                sample_val = ''
            self.lbl_sample_value.setText(f'Sample value: {sample_val}')
            decimals = 0
            max_val = 100
            try:
                sample_val = sample_val.replace(',', '.')
                sample_split = sample_val.split('.')
                sample_val = float(sample_val)
                if len(sample_split) == 2:
                    decimals = len(sample_split[1])
                if abs(sample_val) > 100:
                    max_val = 10 * sample_val
                else:
                    max_val = 200
            except ValueError:
                sample_val = None

            limits = self.parent.current_template.groups_limits[group_idx]
            ranges = self.parent.current_template.groups_ranges[group_idx]
            self.min_tolerance.set_data(limits[0], max_value=max_val,
                                        decimals=decimals, ref_value=sample_val)
            self.max_tolerance.set_data(limits[1], max_value=max_val,
                                        decimals=decimals, ref_value=sample_val)
            self.min_range.set_data(ranges[0], max_value=max_val,
                                    decimals=decimals, ref_value=sample_val)
            self.max_range.set_data(ranges[1], max_value=max_val,
                                    decimals=decimals, ref_value=sample_val)
            self.blockSignals(True)
            self.chk_hide.setChecked(
                self.parent.current_template.groups_hide[group_idx])
            self.txt_title.setText(self.parent.current_template.groups_title[group_idx])
            self.blockSignals(False)
        else:
            self.reset_data_display()

    def group_selected(self):
        """Group selected headers."""
        sels = self.list_headers.selectedIndexes()
        sel_rows = [sel.row() for sel in sels]
        sel_headers = [self.headers[row] for row in sel_rows]
        self.parent.current_template.group_headers(sel_headers)
        _ = self.update_header_order()
        self.update_data(set_selected_idx=sel_rows[0])
        self.parent.flag_edit(True)

    def ungroup_selected(self):
        """Group selected headers."""
        sels = self.list_headers.selectedIndexes()
        sel_rows = [sel.row() for sel in sels]
        sel_headers = [self.headers[row] for row in sel_rows]
        self.parent.current_template.ungroup_headers(sel_headers)
        _ = self.update_header_order()
        self.update_data(set_selected_idx=sel_rows[0])
        self.parent.flag_edit(True)

    def move_group(self, direction='up'):
        """Move selected group up or down in list.

        Parameters
        ----------
        direction : str, optional
            'up' or 'down. The default is 'up'.
        """
        sels = self.list_headers.selectedIndexes()
        sel_row = sels[0].row()
        selected_header = self.headers[sel_row]
        self.group_numbers[sel_row]
        val = -1 if direction == 'up' else 1
        self.parent.current_template.move_group(
            old_group_number=self.group_numbers[sel_row],
            new_group_number=self.group_numbers[sel_row] + val)
        _ = self.update_header_order()
        self.update_data(set_selected_idx=self.headers.index(selected_header))
        self.parent.flag_edit(True)


class LimitsAndPlotEditDialog(ImageQCDialog):
    """Dialog for editing limits and plot settings."""

    def __init__(self, parent, templates, add=False):
        super().__init__()
        self.setWindowTitle('Edit limits and plot settings')
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        self.parent = parent
        self.edited = False
        self.current_modality = self.parent.current_modality
        self.current_template = None
        self.type_vendor = ('vendor' in self.parent.fname)
        self.auto_templates = self.parent.templates
        self.templates = templates
        self.lastload = time()
        if add:
            initial_template_label = ''
        else:
            initial_template_label = self.parent.cbox_limits_and_plot.currentText()
        self.wid_content = LimitsAndPlotContent(
            self,
            sample_file_path=self.parent.txt_output_path.text(),
            initial_template_label=initial_template_label,
            type_vendor=self.type_vendor)
        self.wid_content.update_from_sample_file(silent=True)
        vlo.addWidget(self.wid_content)

        hlo_dlg_btns = QHBoxLayout()
        vlo.addLayout(hlo_dlg_btns)
        hlo_dlg_btns.addStretch()
        btn_ok = QPushButton('Save and close')
        btn_ok.clicked.connect(self.accept)
        hlo_dlg_btns.addWidget(btn_ok)
        btn_cancel = QPushButton('Cancel')
        btn_cancel.clicked.connect(self.reject)
        hlo_dlg_btns.addWidget(btn_cancel)

    def flag_edit(self, flag=True):
        """Indicate whether template has changed."""
        self.edited = flag

    def save_current_template(self):
        """Save current limits_and_plot template."""
        if self.edited:
            fname = 'limits_and_plot_templates'
            proceed, errmsg = cff.check_save_conflict(fname, self.lastload)
            if errmsg != '':
                proceed = messageboxes.proceed_question(self, errmsg)
            if proceed:
                save_new = False
                if self.templates[self.current_modality][0].label == '':
                    save_new = True
                    save_index = 0
                else:
                    curr_label = self.current_template.label
                    if curr_label != '':
                        used_labels = [temp.limits_and_plot_label for temp
                                       in self.auto_templates[self.current_modality]]
                        if used_labels.count(curr_label) > 1:
                            res = messageboxes.QuestionBox(
                                parent=self, title='Overwrite or save new?',
                                msg=(
                                    'The template is used by other automation '
                                    'templates.'),
                                yes_text='Overwrite limits and plot template',
                                no_text='Save as new template')
                            if res.exec():  # overwrite
                                labels = [temp.label for temp
                                          in self.templates[self.current_modality]]
                                idx = labels.index(curr_label)
                                self.templates[self.current_modality][idx] = (
                                    self.current_template)
                            else:
                                save_new = True
                                save_index = -1
                    else:
                        save_new = True
                        save_index = -1
                if save_new:
                    text, proceed = QInputDialog.getText(
                        self, 'Set name ',
                        'Name the new limits and plot template                      ',
                        text=self.parent.current_template.label)
                    text = valid_template_name(text)
                    if proceed and text != '':
                        already = [temp.label for temp
                                   in self.templates[self.current_modality]]
                        if text in already:
                            QMessageBox.warning(
                                self, 'Name already in use',
                                ('This name is already in use. _RENAME added to '
                                 'template name. Please rename in Limits and plot '
                                 'widget.'))
                            text = text + '_RENAME'
                        self.current_template.label = text
                    if save_index != -1:
                        self.templates[self.current_modality][
                            save_index] = self.current_template
                    else:
                        self.templates[self.current_modality].append(
                            self.current_template)
                ok_save, path = cff.save_settings(self.templates, fname=fname)
        return self.current_template.label


class LimitsAndPlotFixHeadersDialog(ImageQCDialog):
    """Dialog for editing limits and plot settings."""

    def __init__(self, parent, limits_template=None, output_paths=None, auto_label='',
                 headers_in_output=None, headers_in_template=None):
        """Initiate LimitsAndPlotFixHeadersDialog.

        Parameters
        ----------
        parent : QWidget
        limits_template : LimitsAndPlotTemplate, optional
            The default is None.
        output_paths : list of str
            output paths to edit
        auto_label : str
        headers_in_output : list of str
        headers_in_template : list of str
        """
        super().__init__()
        self.setWindowTitle(
            'Handle mismatch between headers in output file and Limits and plot '
            'template')
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        self.parent = parent
        self.current_template = limits_template
        self.headers_in_output = headers_in_output
        self.output_paths = output_paths

        self.list_output_headers = QListWidget()
        self.list_output_headers.addItems(headers_in_output)
        self.list_template_headers = QListWidget()
        self.list_output_headers.setFixedWidth(400)
        self.list_template_headers.setFixedWidth(400)
        self.list_template_headers.setDragDropMode(QAbstractItemView.InternalMove)

        info_txt = [
            'Header of output file and Limits template need to match.', '',
            'Found headers with differing text strings or differing number of strings.',
            'Sort headers of the right list (drag/drop) such that the row numbers '
            'match the left list.',
            'This sorting will not affect the order of headers neither in the output '
            'path nor the template plot order.',
            'The sorting is just to find the matches.']
        vlo.addWidget(uir.LabelMultiline(txts=info_txt))
        hlo_lists = QHBoxLayout()
        vlo.addLayout(hlo_lists)
        vlo.addSpacing(20)

        vlo_list_output = QVBoxLayout()
        hlo_lists.addLayout(vlo_list_output)
        vlo_list_output.addWidget(uir.LabelHeader(
            f'Headers in output path of automation template {auto_label}', 3))
        vlo_list_output.addWidget(self.list_output_headers)

        vlo_list_temp = QVBoxLayout()
        hlo_lists.addLayout(vlo_list_temp)
        vlo_list_temp.addWidget(uir.LabelHeader(
            f'Headers in Limits and plot template {self.current_template.label}', 3))
        vlo_list_temp.addWidget(self.list_template_headers)

        self.fill_list_template_headers(headers_in_template)
        vlo.addSpacing(20)

        btn_replace_output_headers = QPushButton(
            'Replace headers in output file with those in Limits template')
        btn_replace_output_headers.clicked.connect(self.replace_output_headers)
        btn_replace_template_headers = QPushButton(
            'Replace headers in Limits template with those in the output file')
        btn_replace_template_headers.clicked.connect(self.replace_template_headers)

        hlo_dlg_btns = QHBoxLayout()
        vlo.addLayout(hlo_dlg_btns)
        hlo_dlg_btns.addStretch()
        btn_ok = QPushButton('Finished matching...')
        btn_ok.clicked.connect(self.fix)
        hlo_dlg_btns.addWidget(btn_ok)
        btn_cancel = QPushButton('Cancel')
        btn_cancel.clicked.connect(self.reject)
        hlo_dlg_btns.addWidget(btn_cancel)

    def fill_list_template_headers(self, headers):
        """Fill the lists of headers trying to find matches."""
        headers_order = [-1 for header in headers]
        no_match = []
        for i, header in enumerate(headers):
            if header in self.headers_in_output:
                headers_order[i] = self.headers_in_output.index(header)
            else:
                no_match.append(header)
        headers_ordered = ['' for header in headers]
        for i, header in enumerate(headers):
            if i in headers_order:
                headers_ordered[i] = header
        for header in headers:
            if header in no_match:
                idx = headers_ordered.index('')
                headers_ordered[idx] = header
        self.list_template_headers.addItems(headers_ordered)

    def fix(self):
        """Start fixing the mismatch based on the sorting."""
        headers_template = []
        for x in range(self.list_template_headers.count()):
            headers_template.append(self.list_template_headers.item(x).text())
        res = messageboxes.QuestionBox(
            parent=self, title='Correct the mismatch',
            msg='How to force the same headers of the output file and Limits template?',
            yes_text='Replace headers in output file',
            no_text='Replace headers in LimitsAndPlot template')
        if res.exec():
            self.replace_output_headers(headers_template)
        else:
            self.replace_template_headers(headers_template)
        self.accept()

    def replace_output_headers(self, headers_template):
        """Replace headers in output file."""
        for path in self.output_paths:
            lines = []
            with open(path, 'r') as file:
                lines = file.readlines()
            if len(lines) > 0:
                header_line = lines[0]
                for i, header in enumerate(self.headers_in_output):
                    if header in header_line:
                        lines[0] = lines[0].replace(header, headers_template[i])
                        self.headers_in_output[i] = headers_template[i]
                with open(path, 'w') as file:
                    for lin in lines:
                        file.write(lin)
        QMessageBox.information(
            self, 'Headers in output file(s) updated',
            'Finished replacing headers of output file(s).')

    def replace_template_headers(self, headers_template):
        """Replace headers in template."""
        for i, old_header in enumerate(headers_template):
            if old_header != self.headers_in_output[i]:
                group_idx = self.current_template.find_headers_group_index(old_header)
                idx = self.current_template.groups[group_idx].index(old_header)
                self.current_template.groups[group_idx][idx] = self.headers_in_output[i]

    def get_template(self):
        """Fetch current template."""
        return self.current_template

    def get_headers(self):
        """Fetch current headers."""
        return self.headers_in_output


class LimitsAndPlotWidget(StackWidget):
    """Widget holding limits and plot settings."""

    def __init__(self, dlg_settings):
        header = 'Limits and plot templates'
        subtxt = '''Configure tolerance limits and plot settings for the results from
        Automation templates. <br>
        After running automation templates linked to a LimitsAndPlot template the
        results will be compared to the tolerance limits set for the diffent
        columns.<br>
        Results from different columns can be grouped together to have the same
        settings. If outside tolerance, a warning will be displayed in the
        automation log. <br>
        You could also specify a text file to append these warnings. Use for example
        Sharepoint "Alert me" functionality to set up notification for specific users.
        <br>
        The limits and y-ranges and naming will be used if displaying the dashboard
        for the results.<br>
        '''
        super().__init__(dlg_settings, header, subtxt,
                         mod_temp=True, grouped=True)
        self.fname = 'limits_and_plot_templates'
        self.empty_template = cfc.LimitsAndPlotTemplate()
        self.current_template = None
        self.auto_labels = []  # auto_labels used in
        self.output_paths = []  # output paths for linked automation templates
        self.output_headers = []  # headers in output paths (list of list of str)

        if self.import_review_mode is False:
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_move_modality)
        self.wid_mod_temp.vlo.addWidget(
            QLabel('Selected template used in Automation template:'))
        self.list_used_in = QListWidget()
        self.wid_mod_temp.vlo.addWidget(self.list_used_in)

        vlo = QVBoxLayout()
        self.hlo.addLayout(vlo)

        self.wid_content = LimitsAndPlotContent(self)
        vlo.addWidget(self.wid_content)

        if self.import_review_mode:
            self.wid_content.setEnabled(False)

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def clear(self):
        """Replace the clear method of ModTempSelector."""
        label = self.current_template.label
        self.wid_content.headers = []
        self.wid_content.first_values = None
        self.wid_content.generate_empty_template()
        self.current_template.label = label
        self.wid_content.group_numbers = []
        self.wid_content.update_data()
        self.flag_edit(flag=True)

    def update_data(self):
        """Update GUI with selected template."""
        self.update_used_in()
        self.wid_content.cbox_output_paths.clear()
        self.wid_content.cbox_output_paths.addItems(
            [f'Auto template: {self.auto_labels[i]}, output file: {path}' for i, path
             in enumerate(self.output_paths)])
        self.wid_content.update_from_sample_file(silent=True)

    def update_used_in(self):
        """Update list of auto-templates where this template is used."""
        self.list_used_in.clear()
        self.auto_labels, self.output_paths = cff.get_auto_labels_output_used_in_lim(
                self.auto_templates, self.auto_vendor_templates, self.current_template,
                modality=self.current_modality)
        if len(self.auto_labels) > 0:
            self.list_used_in.addItems(self.auto_labels)
            self.output_headers = []
            for path in self.output_paths:
                headers, _ = get_headers_first_values_in_path(path)
                self.output_headers.append(headers)


class AutoTempWidgetBasic(StackWidget):
    """Common settings for AutoTemplates and AutoVendorTemplates."""

    def __init__(self, dlg_settings, header, subtxt):
        super().__init__(dlg_settings, header, subtxt,
                         mod_temp=True, grouped=True)

        self.txt_input_path = QLineEdit('')
        self.txt_input_path.textChanged.connect(self.flag_edit)
        self.txt_input_path.setMinimumWidth(500)

        self.txt_output_path = QLineEdit('')
        self.txt_output_path.textChanged.connect(self.flag_edit)
        self.txt_output_path.setMinimumWidth(500)

        self.txt_statname = QLineEdit('')
        self.txt_statname.textChanged.connect(self.flag_edit)
        self.txt_statname.setMinimumWidth(200)

        self.txt_warnings_path = QLineEdit('')
        self.txt_warnings_path.textChanged.connect(self.flag_edit)
        self.txt_warnings_path.setMinimumWidth(500)

        self.wid_mod_temp.vlo.addWidget(
            QLabel('Automation templates with same input path:'))
        self.list_same_input = QListWidget()
        self.list_same_input.setFixedHeight(80)
        self.wid_mod_temp.vlo.addWidget(self.list_same_input)
        self.wid_mod_temp.vlo.addWidget(
            QLabel('Automation templates with same output path:'))
        self.list_same_output = QListWidget()
        self.list_same_output.setFixedHeight(80)
        self.wid_mod_temp.vlo.addWidget(self.list_same_output)
        self.wid_mod_temp.vlo.addWidget(
            QLabel('Automation templates with same limits:'))
        self.list_same_limits = QListWidget()
        self.list_same_limits.setFixedHeight(80)
        self.wid_mod_temp.vlo.addWidget(self.list_same_limits)

        self.chk_archive = QCheckBox(
            'Archive files when analysed (Archive folder in input path).')
        self.chk_archive.stateChanged.connect(
            lambda: self.flag_edit(True))
        self.chk_deactivate = QCheckBox('Deactivate template')
        self.chk_deactivate.stateChanged.connect(
            lambda: self.flag_edit(True))

        self.wid_temp = QWidget(self)
        if self.import_review_mode:
            self.wid_temp.setEnabled(False)
        else:
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_move_modality)
        self.hlo.addWidget(self.wid_temp)
        self.vlo_temp = QVBoxLayout()
        self.wid_temp.setLayout(self.vlo_temp)

        self.gb_limits_and_plot = QGroupBox('Limits and plot settings')
        self.gb_limits_and_plot.setFont(uir.FontItalic())
        vlo_limits_and_plot = QVBoxLayout()
        self.gb_limits_and_plot.setLayout(vlo_limits_and_plot)

        vlo_limits_and_plot.addWidget(uir.LabelItalic(
            'Configure tolerance limits to trigger notifications and '
            'visualization settings for the results dashboard.'))
        hlo_limits_and_plot = QHBoxLayout()
        vlo_limits_and_plot.addLayout(hlo_limits_and_plot)
        hlo_limits_and_plot.addWidget(QLabel('Use LimitsAndPlot template:'))
        self.cbox_limits_and_plot = QComboBox()
        self.cbox_limits_and_plot.setMinimumWidth(300)
        self.cbox_limits_and_plot.currentIndexChanged.connect(
            lambda: self.flag_edit(True))
        hlo_limits_and_plot.addWidget(self.cbox_limits_and_plot)
        self.tb_edit_limits_and_plot = uir.ToolBarEdit(
            tooltip='Edit limits and plot settings', add_button=True)
        hlo_limits_and_plot.addWidget(self.tb_edit_limits_and_plot)
        hlo_limits_and_plot.addStretch()
        self.tb_edit_limits_and_plot.act_edit.triggered.connect(
            self.edit_limits_and_plot)
        self.tb_edit_limits_and_plot.act_add.triggered.connect(
            lambda: self.edit_limits_and_plot(add=True))
        vlo_limits_and_plot.addWidget(uir.HLine())

        hlo_input_path = QHBoxLayout()
        self.vlo_temp.addLayout(hlo_input_path)
        hlo_input_path.addWidget(QLabel('Input path '))
        hlo_input_path.addWidget(self.txt_input_path)
        toolb = uir.ToolBarBrowse('Browse to find path')
        toolb.act_browse.triggered.connect(
            lambda: self.locate_folder(self.txt_input_path))
        hlo_input_path.addWidget(toolb)

        self.hlo_output_path = QHBoxLayout()
        self.vlo_temp.addLayout(self.hlo_output_path)
        self.hlo_output_path.addWidget(QLabel('Output path '))
        self.hlo_output_path.addWidget(self.txt_output_path)
        toolb = uir.ToolBarBrowse('Browse to file')
        toolb.act_browse.triggered.connect(
            lambda: self.locate_file(
                self.txt_output_path, title='Locate output file',
                filter_str="Text file (*.txt)"))
        self.hlo_output_path.addWidget(toolb)
        act_output_view = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'Display output file', toolb)
        act_output_view.triggered.connect(
            lambda: self.view_file(filetype='output'))
        act_new_file = QAction(QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                               'Create an empty file', toolb)
        act_new_file.triggered.connect(
            lambda: self.locate_file(
                self.txt_output_path, title='Locate output file',
                filter_str="Text file (*.txt)", opensave=True))
        toolb.addActions([act_new_file, act_output_view])

        hlo_warnings_path = QHBoxLayout()
        vlo_limits_and_plot.addLayout(hlo_warnings_path)
        hlo_warnings_path.addWidget(QLabel('Warnings path '))
        hlo_warnings_path.addWidget(self.txt_warnings_path)
        toolb = uir.ToolBarBrowse('Browse to file')
        toolb.act_browse.triggered.connect(
            lambda: self.locate_file(
                self.txt_warnings_path, title='Locate warnings file',
                filter_str="Text file (*.txt)"))
        hlo_warnings_path.addWidget(toolb)
        act_warnings_view = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'Display warnings file', toolb)
        act_warnings_view.triggered.connect(
            lambda: self.view_file(filetype='warning'))
        act_new_warn_file = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Create an empty file', toolb)
        act_new_warn_file.triggered.connect(
            lambda: self.locate_file(
                self.txt_warnings_path, title='Locate warning file',
                filter_str="Text file (*.txt)", opensave=True))
        toolb.addActions([act_new_warn_file, act_warnings_view])
        vlo_limits_and_plot.addWidget(uir.LabelItalic(
            'Append warnings to file if limits are violated.'))
        vlo_limits_and_plot.addWidget(uir.LabelItalic(
            'Use i.e. "Alert me" functionality in Sharepoint to notify/email '
            'specific persons.'))

    def update_data(self):
        """Refresh GUI after selecting template."""
        self.txt_input_path.setText(self.current_template.path_input)
        self.txt_output_path.setText(self.current_template.path_output)
        self.txt_warnings_path.setText(self.current_template.path_warnings)
        self.txt_statname.setText(self.current_template.station_name)
        self.cbox_limits_and_plot.setCurrentText(
            self.current_template.limits_and_plot_label)
        self.chk_archive.setChecked(self.current_template.archive)
        self.chk_deactivate.setChecked(not self.current_template.active)
        self.cbox_limits_and_plot.setCurrentText(
            self.current_template.limits_and_plot_label)
        if self.import_review_mode is False:
            if (
                    self.cbox_limits_and_plot.currentText()
                    != self.current_template.limits_and_plot_label):
                QMessageBox.warning(
                    self, 'Warning',
                    ('Limits and plot template '
                     f'{self.current_template.limits_and_plot_label} set for this '
                     'template, but not defined in limits_and_plot_templates.yaml.'))
                self.cbox_limits_and_plot.setCurrentText('')

        # update used_in
        self.list_same_input.clear()
        self.list_same_output.clear()
        self.list_same_limits.clear()
        if self.current_template.label != '':
            if self.current_template.path_input != '':
                auto_labels = [
                    temp.label for temp in self.templates[self.current_modality]
                    if temp.path_input == self.current_template.path_input
                    ]
                if len(auto_labels) > 1:
                    auto_labels.remove(self.current_template.label)
                    self.list_same_input.addItems(auto_labels)

            if self.current_template.path_output != '':
                auto_labels = [
                    temp.label for temp in self.templates[self.current_modality]
                    if temp.path_output == self.current_template.path_output
                    ]
                if len(auto_labels) > 1:
                    auto_labels.remove(self.current_template.label)
                    self.list_same_output.addItems(auto_labels)

            if self.current_template.limits_and_plot_label != '':
                auto_labels = [
                    temp.label for temp in self.templates[self.current_modality]
                    if temp.limits_and_plot_label == self.current_template.limits_and_plot_label
                    ]
                if len(auto_labels) > 1:
                    auto_labels.remove(self.current_template.label)
                    self.list_same_limits.addItems(auto_labels)

    def fill_list_limits_and_plot(self):
        """Fill list of QuickTest templates."""
        self.cbox_limits_and_plot.clear()
        try:
            if self.current_modality in self.limits_and_plot_templates:
                labels = [obj.label for obj
                          in self.limits_and_plot_templates[self.current_modality]]
                labels.insert(0, '')
                self.cbox_limits_and_plot.addItems(labels)
        except AttributeError:  # if import review mode
            pass

    def get_current_template(self):
        """Get self.current_template where not dynamically set."""
        self.current_template.path_input = self.txt_input_path.text()
        if self.current_template.path_input != '':
            create_empty_folder(self.current_template.path_input,
                                self, proceed_info_txt='Input path do not exist.')
        self.current_template.path_output = self.txt_output_path.text()
        if self.current_template.path_output != '':
            create_empty_file(self.current_template.path_output,
                              self, proceed_info_txt='Output path do not exist.')
        self.current_template.path_warnings = self.txt_warnings_path.text()
        self.current_template.limits_and_plot_label = (
            self.cbox_limits_and_plot.currentText())
        self.current_template.station_name = self.txt_statname.text()
        self.current_template.active = not self.chk_deactivate.isChecked()
        self.current_template.archive = self.chk_archive.isChecked()

    def view_file(self, filetype='output'):
        """View output or warning file as txt."""
        if filetype == 'output':
            if os.path.exists(self.txt_output_path.text()):
                os.startfile(self.txt_output_path.text())
        elif filetype == 'warning':
            if os.path.exists(self.txt_warnings_path.text()):
                os.startfile(self.txt_warnings_path.text())

    def edit_limits_and_plot(self, add=False):
        """Open dialog to edit limits and plot settings."""
        if os.path.exists(self.txt_output_path.text()):
            dlg = LimitsAndPlotEditDialog(self, self.limits_and_plot_templates, add=add)
            res = dlg.exec()
            if res:
                if dlg.edited:
                    set_label = dlg.save_current_template()
                    self.current_template.limits_and_plot_label = set_label
                    all_items = [self.cbox_limits_and_plot.itemText(i) for i
                                 in range(self.cbox_limits_and_plot.count())]
                    if set_label not in all_items:
                        self.cbox_limits_and_plot.addItem(set_label)
                    self.cbox_limits_and_plot.setCurrentText(set_label)
                    self.flag_edit(True)
        else:
            QMessageBox.information(
                self, 'Missing output path',
                'The output file path is not defined or file not found. '
                'The Limits and plot template need to be defined based on '
                'headers found from an output file.')


class AutoTemplateWidget(AutoTempWidgetBasic):
    """Widget holding automation settings."""

    def __init__(self, dlg_settings):
        header = 'Automation templates DICOM'
        subtxt = '''The automation templates hold information on how to
         perform automated testing on DICOM images.<br>
        The automation template connect-parameter set and other templates
         to be able to use and reuse settings between similar test setups.'''
        super().__init__(dlg_settings, header, subtxt)
        self.fname = 'auto_templates'
        self.empty_template = cfc.AutoTemplate()
        self.sample_filepath = ''

        self.tree_crit = DicomCritWidget(self)
        self.cbox_paramset = QComboBox()
        self.cbox_paramset.currentIndexChanged.connect(lambda: self.flag_edit(True))
        self.cbox_quicktest = QComboBox()
        self.cbox_quicktest.currentIndexChanged.connect(lambda: self.flag_edit(True))
        self.list_sort_by = QListWidget()

        self.chk_import_only = QCheckBox(
            'Use template for import only (e.g. as supplement) - no analysis)')
        self.chk_import_only.clicked.connect(self.import_only_changed)

        hlo_temp = QHBoxLayout()
        self.vlo_temp.addLayout(hlo_temp)

        gb_import_settings = QGroupBox('Import criteria from image pool')
        gb_import_settings.setFont(uir.FontItalic())
        vlo_import = QVBoxLayout()
        gb_import_settings.setLayout(vlo_import)
        hlo_temp.addWidget(gb_import_settings)

        hlo_statname = QHBoxLayout()
        hlo_statname.addWidget(QLabel('Station name'))
        hlo_statname.addWidget(self.txt_statname)

        hlo_statname.addStretch()
        toolb = QToolBar()
        act_get_statname = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}import.png'),
            'Get station name and empty DICOM criteria values from sample file', toolb)
        act_get_statname.triggered.connect(self.set_sample_file)
        toolb.addActions([act_get_statname])
        hlo_statname.addWidget(toolb)
        hlo_statname.addStretch()
        vlo_import.addLayout(hlo_statname)
        vlo_import.addWidget(QLabel(
            'If station name is left empty, at least one additional DICOM criterion'))
        vlo_import.addWidget(QLabel('need to be set for the template to be valid.'))
        vlo_import.addSpacing(20)
        vlo_import.addWidget(uir.LabelItalic(
            'Additional DICOM criteria'))
        vlo_import.addWidget(self.tree_crit)

        vlo_right = QVBoxLayout()
        hlo_temp.addLayout(vlo_right)

        self.gb_analyse = QGroupBox('Image analysis settings')
        self.gb_analyse.setFont(uir.FontItalic())
        vlo_analyse = QVBoxLayout()
        self.gb_analyse.setLayout(vlo_analyse)
        vlo_right.addWidget(self.gb_analyse)

        vlo_analyse.addWidget(uir.LabelItalic(
            'Sort images for each date/studyUID by:'))
        hlo_sort_list = QHBoxLayout()
        vlo_analyse.addLayout(hlo_sort_list)
        hlo_sort_list.addWidget(self.list_sort_by)
        self.list_sort_by.setMinimumWidth(300)
        self.tb_edit_sort_by = uir.ToolBarEdit(tooltip='Edit sort list')
        hlo_sort_list.addWidget(self.tb_edit_sort_by)
        self.tb_edit_sort_by.act_edit.triggered.connect(self.edit_sort_by)

        flo_analyse = QFormLayout()
        vlo_analyse.addLayout(flo_analyse)
        flo_analyse.addRow(
            QLabel('Use parameter set: '),
            self.cbox_paramset)
        flo_analyse.addRow(
            QLabel('Use QuickTest template: '),
            self.cbox_quicktest)

        vlo_right.addWidget(self.gb_limits_and_plot)

        hlo_btm = QHBoxLayout()
        self.vlo_temp.addLayout(hlo_btm)
        vlo_chk = QVBoxLayout()
        hlo_btm.addLayout(vlo_chk)
        vlo_chk.addWidget(self.chk_archive)
        vlo_chk.addWidget(self.chk_import_only)
        vlo_chk.addWidget(self.chk_deactivate)

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_data(self):
        """Refresh GUI after selecting template."""
        super().update_data()
        self.tree_crit.update_data()
        self.cbox_paramset.setCurrentText(self.current_template.paramset_label)
        if self.cbox_paramset.currentText() != self.current_template.paramset_label:
            QMessageBox.warning(
                self, 'Warning',
                (f'Paramset {self.current_template.paramset_label} set for this '
                 'template, but not defined in paramsets.yaml.'))
            self.cbox_paramset.setCurrentText('')
        self.cbox_quicktest.setCurrentText(self.current_template.quicktemp_label)
        if self.cbox_quicktest.currentText() != self.current_template.quicktemp_label:
            QMessageBox.warning(
                self, 'Warning',
                (f'Paramset {self.current_template.quicktemp_label} set for this '
                 'template, but not defined in quicktest_templates.yaml.'))
            self.cbox_quicktest.setCurrentText('')
        self.fill_list_sort_by()
        self.chk_import_only.setChecked(self.current_template.import_only)

        self.sample_filepath = ''
        if self.import_review_mode is False:
            self.update_import_enabled()
        self.flag_edit(False)

    def get_current_template(self):
        """Get self.current_template where not dynamically set."""
        super().get_current_template()
        self.current_template.paramset_label = self.cbox_paramset.currentText()
        self.current_template.quicktemp_label = self.cbox_quicktest.currentText()
        self.current_template.import_only = self.chk_import_only.isChecked()

    def fill_lists(self):
        """Fill all lists on modality change."""
        self.fill_list_paramsets()
        self.fill_list_quicktest_templates()

    def fill_list_paramsets(self):
        """Find available paramsets and fill cbox."""
        self.cbox_paramset.clear()
        labels = [obj.label for obj in self.paramsets[self.current_modality]]
        labels.insert(0, '')
        self.cbox_paramset.addItems(labels)

    def fill_list_quicktest_templates(self):
        """Fill list of QuickTest templates."""
        self.cbox_quicktest.clear()
        if self.current_modality in self.quicktest_templates:
            labels = [obj.label for obj
                      in self.quicktest_templates[self.current_modality]]
            labels.insert(0, '')
            self.cbox_quicktest.addItems(labels)

    def fill_list_sort_by(self):
        """Fill list of sort tags."""
        self.list_sort_by.clear()
        list_tags = self.current_template.sort_pattern.list_tags
        if len(list_tags) > 0:
            list_sort = self.current_template.sort_pattern.list_sort
            for i, tag in enumerate(list_sort):
                asc_desc_txt = '(ASC)' if tag is True else '(DESC)'
                self.list_sort_by.addItem(' '.join([list_tags[i], asc_desc_txt]))

    def get_sample_file_data(self):
        """Update dicom criterions from sample file."""
        if self.sample_filepath != '':
            tag_pattern_this = cfc.TagPatternFormat(list_tags=['StationName'])
            for attr_name in self.current_template.dicom_crit_attributenames:
                tag_pattern_this.add_tag(attr_name)
            tags = dcm.get_tags(
                self.sample_filepath, tag_patterns=[tag_pattern_this],
                tag_infos=self.tag_infos)
            self.current_template.station_name = tags[0][0]
            self.txt_statname.setText(tags[0][0])
            if len(tags[0]) > 1:
                for i in range(1, len(tags[0])):
                    self.current_template.dicom_crit_values[i-1] = tags[0][i]
                self.tree_crit.update_data()

    def set_sample_file(self):
        """Set sample file."""
        if self.sample_filepath != '':
            def_path = self.sample_filepath
        else:
            def_path = self.txt_input_path.text()
            if def_path == '':
                def_path = self.auto_common.import_path
        fname = QFileDialog.getOpenFileName(
                self, 'Get station name (+ dicom criteria values) from sample file',
                def_path,
                filter="DICOM file (*.dcm *.IMA);;All files (*)")
        if len(fname[0]) > 0:
            self.sample_filepath = fname[0]
            self.get_sample_file_data()

    def edit_sort_by(self):
        """Edit list to sort images by."""
        dlg = TagPatternEditDialog(
            initial_pattern=self.current_template.sort_pattern,
            modality=self.current_modality,
            title='Sort images by DICOM header information',
            typestr='sort',
            accept_text='Use',
            reject_text='Cancel',
            save_blocked=self.save_blocked)
        res = dlg.exec()
        if res:
            sort_pattern = dlg.get_pattern()
            self.current_template.sort_pattern = sort_pattern
            self.flag_edit(True)
            self.fill_list_sort_by()

    def import_only_changed(self):
        """Click import only checkbox."""
        self.current_template.import_only = self.chk_import_only.isChecked()
        if self.current_template.import_only:
            self.get_current_template()
            params = self.current_template.paramset_label
            qt = self.current_template.quicktemp_label
            outp = self.current_template.path_output
            sort = '' if len(self.current_template.sort_pattern.list_tags) == 0 else '-'
            if params + qt + outp + sort != '':
                res = messageboxes.QuestionBox(
                    parent=self, title='Remove inactive settings?',
                    msg='Reset settings that only affect analysis or just deactivate',
                    yes_text='Yes, reset',
                    no_text='No, just deactivate')
                if res.exec():
                    self.current_template.paramset_label = ''
                    self.current_template.quicktemp_label = ''
                    self.current_template.path_output = ''
                    self.current_template.sort_pattern = copy.deepcopy(
                        self.empty_template.sort_pattern)
                    self.update_data()
        else:
            self.update_import_enabled()
        self.flag_edit(True)

    def update_import_enabled(self):
        """Update enabled guis whether import only or not."""
        self.gb_analyse.setEnabled(not self.current_template.import_only)
        self.gb_limits_and_plot.setEnabled(not self.current_template.import_only)
        idx = self.hlo_output_path.count() - 1
        while idx >= 0:
            this_widget = self.hlo_output_path.itemAt(idx).widget()
            this_widget.setEnabled(not self.current_template.import_only)
            idx -= 1


class AutoVendorTemplateWidget(AutoTempWidgetBasic):
    """Widget holding automation settings."""

    def __init__(self, dlg_settings):
        header = 'Automation templates vendor files'
        subtxt = '''The automation templates hold information on how to
         perform automated reading of vendor report files.<br>
         NB - decimal mark found from the default (first) parameterset of the
         same modality.'''
        super().__init__(dlg_settings, header, subtxt)
        self.fname = 'auto_vendor_templates'
        self.empty_template = cfc.AutoVendorTemplate()

        self.cbox_file_type = QComboBox()

        hlo_statname = QHBoxLayout()
        hlo_statname.addWidget(QLabel('Station ID'))
        hlo_statname.addWidget(self.txt_statname)

        toolb = QToolBar()
        act_get_statname = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}import.png'),
            'Get station name from sample file', toolb)
        act_get_statname.triggered.connect(self.get_station_name)
        toolb.addActions([act_get_statname])
        hlo_statname.addWidget(toolb)
        hlo_statname.addWidget(uir.LabelItalic(
            ('Only used for verification if station is set in vendor '
             + 'report file.')))
        hlo_statname.addStretch()
        self.vlo_temp.addLayout(hlo_statname)

        hlo_options = QHBoxLayout()
        hlo_options.addWidget(QLabel('Vendor file type: '))
        hlo_options.addWidget(self.cbox_file_type)
        hlo_options.addStretch()
        self.vlo_temp.addLayout(hlo_options)
        hlo_mammo_options = QHBoxLayout()
        self.vlo_temp.addLayout(hlo_mammo_options)
        self.info_Mammo_QAP = uir.LabelItalic(
            '''
            If GE Mammo QAP type:<br>
            Autogenerate templates based on files in your input folder.<br>
            This folder may contain files from different tests indicated
            by the input file name prefix.<br>
            You will also be prompted to auto generate Limits and Plots settings<br>
            based on the found lower/upper limits specified in the first found file.
            '''
            )
        self.info_Mammo_QAP.setVisible(False)
        self.btn_auto_generate_Mammo_QAP_templates = QPushButton(
            'Auto generate Mammo QAP templates...')
        self.btn_auto_generate_Mammo_QAP_templates.setVisible(False)
        self.btn_auto_generate_Mammo_QAP_templates.clicked.connect(
            self.auto_generate_Mammo_QAP_templates)
        hlo_options.addWidget(self.info_Mammo_QAP)
        hlo_options.addWidget(self.btn_auto_generate_Mammo_QAP_templates)

        self.vlo_temp.addStretch()
        hlo_lim_plot = QHBoxLayout()
        self.vlo_temp.addLayout(hlo_lim_plot)
        hlo_lim_plot.addWidget(self.gb_limits_and_plot)
        hlo_lim_plot.addStretch()
        self.vlo_temp.addStretch()
        self.vlo_temp.addWidget(self.chk_archive)
        self.vlo_temp.addWidget(self.chk_deactivate)
        self.vlo_temp.addStretch()

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_from_yaml(self, initial_template_label=''):
        """Refresh settings from yaml file."""
        super().update_from_yaml()
        self.update_file_types()

    def update_data(self):
        """Refresh GUI after selecting template."""
        super().update_data()
        if self.current_template.file_type != '':
            self.cbox_file_type.setCurrentText(
                self.current_template.file_type)
        self.info_Mammo_QAP.setVisible(
            self.current_modality == 'Mammo')
        self.btn_auto_generate_Mammo_QAP_templates.setVisible(
            self.current_modality == 'Mammo')
        self.flag_edit(False)

    def get_current_template(self):
        """Get self.current_template where not dynamically set."""
        super().get_current_template()
        file_type = self.cbox_file_type.currentText()
        self.current_template.file_type = file_type
        if file_type == 'GE Mammo QAP (txt)':
            self.current_template.file_suffix = ''
        else:
            try:
                self.current_template.file_suffix = file_type.split('(')[1][:-1]
            except IndexError:
                pass

    def get_station_name(self):
        """Get station name from sample file.

        Return
        ------
        statname : str
            station name if found/defined in vendor report file
        """
        statname = ''
        file_type = self.cbox_file_type.currentText()

        open_title = ''
        file_filter = ''
        old_status = self.status_label.text()
        if self.current_modality == 'CT':
            open_title = 'Open Siemens CT QC report file'
            file_filter = "PDF file (*.pdf)"
        elif self.current_modality == 'PET':
            if 'pdf' in file_type:
                open_title = 'Open Siemens CT QC report file'
                file_filter = "PDF file (*.pdf)"
        else:
            pass

        if open_title != '':
            res = {'status': False}
            fname = QFileDialog.getOpenFileName(
                    self, open_title, filter=file_filter)
            if len(fname[0]) > 0:
                self.status_label.setText('Please wait while reading file....')
                QApplication.setOverrideCursor(Qt.WaitCursor)
                qApp.processEvents()
                if self.current_modality == 'CT':
                    txt = read_vendor_QC_reports.get_pdf_txt(fname[0])
                    res = read_vendor_QC_reports.read_Siemens_CT_QC(txt)
                    if res['status']:
                        if len(res['values']) > 3:
                            statname = res['values'][3]
                elif self.current_modality == 'PET' and 'pdf' in file_type:
                    txt = read_vendor_QC_reports.get_pdf_txt(fname[0])
                    res = read_vendor_QC_reports.read_Siemens_PET_dailyQC(txt)
                    if res['status']:
                        if len(res['values']) > 1:
                            statname = res['values'][1]
                QApplication.restoreOverrideCursor()
                self.status_label.setText(old_status)

        if statname == '':
            QMessageBox.information(
                self, 'Station ID not found',
                'No station ID defined or found for the selected file type.')
        else:
            self.txt_statname.setText(statname)

    def update_file_types(self):
        """Update list of file_types on modality change."""
        self.cbox_file_type.clear()
        self.cbox_file_type.addItems(
            VENDOR_FILE_OPTIONS[self.current_modality])

    def auto_generate_Mammo_QAP_templates(self):
        """Generate templates reading GE Mammo QAP files based on files in input."""
        def pop_already(new_templates, existing_templates):
            labels_exist = [x.label for x in existing_templates]
            already_labels = [x.label for x in new_templates if x.label in labels_exist]
            if len(already_labels) > 0:
                tempname = new_templates[0].__class__.__name__
                res = messageboxes.QuestionBox(
                    parent=self, title='Labels already used',
                    msg=(f'Some {tempname} labels already exist. See details.'),
                    info='', details=['Already exists:'] + already_labels,
                    yes_text='Overwrite existing templates',
                    no_text='Add only new templates')
                if not res.exec():  # add only new
                    ids_to_delete = [i for i, temp in enumerate(new_templates)
                                     if temp.label in already_labels]
                    ids_to_delete.reverse()
                    for idx in ids_to_delete:
                        new_templates.pop(idx)
            return new_templates

        errmsg = ''
        test_names = []
        filenames = []
        if self.txt_input_path.text() == '':
            errmsg = 'Please specify an input path for locating the sample files.'
        else:
            filenames = [x.name for x in Path(self.txt_input_path.text()).glob('*')
                         if '_' in x.name]
            if len(filenames) > 0:
                test_names = [x.split('_BasicResults_')[0] for x in filenames]
                test_names = list(set(test_names))
                test_names.sort()
                dlg = SelectTextsDialog(
                    test_names, title='Found tests',
                    select_info='Select tests to generate templates for')
                if dlg.exec():
                    test_names = dlg.get_checked_texts()
                else:
                    test_names = []
            else:
                errmsg = ('Filenames expected to be <testname>_....date_time. '
                          'Found no filename as expected.')

        if errmsg:
            QMessageBox.information(
                self, 'Failed reading input files', errmsg)

        if len(test_names) > 0:
            station_name, proceed = QInputDialog.getText(
                self, 'Station name', 'Template names should start with:      ')
            output_folder = None
            if proceed:
                QMessageBox.information(
                    self, 'Locate output folder',
                    'You will now be asked to set the folder for the output files...')
                dlg = QFileDialog()
                dlg.setFileMode(QFileDialog.Directory)
                if dlg.exec():
                    fname = dlg.selectedFiles()
                    output_folder = fname[0]
            templates = []
            if output_folder:
                general_template = cfc.AutoVendorTemplate(
                    path_input=self.txt_input_path.text(),
                    file_type='GE Mammo QAP (txt)',
                    archive=True)
                for test_name in test_names:
                    path_output = os.path.join(output_folder, test_name + '.txt')
                    create_empty_file(
                        path_output, self, proceed_info_txt='', proceed=True)
                    if os.path.exists(path_output):
                        template_this = copy.deepcopy(general_template)
                        template_this.label = station_name + '_' + test_name
                        template_this.path_output = path_output
                        template_this.file_prefix = test_name + '_'
                        templates.append(template_this)

            if len(templates) > 0:
                templates = pop_already(templates, self.templates['Mammo'])

            # auto generate limits_and_plot_templates?
            if len(templates) > 0:
                res = messageboxes.QuestionBox(
                    parent=self, title='Generate Limits and Plot templates',
                    msg=('Read limits from files and set templates for plotting '
                         'using Dash-board?'),
                    yes_text='Yes, proceed',
                    no_text='No')
                templates_lim = []
                if res.exec():
                    for template in templates:
                        act_files = [x for x in filenames
                                     if x.startswith(template.file_prefix)]
                        res = read_vendor_template(
                            template=template,
                            filepath=os.path.join(template.path_input, act_files[0])
                            )
                        if res['status']:
                            template = cfc.LimitsAndPlotTemplate(
                                label=template.file_prefix,
                                type_vendor=True,
                                groups=[[header] for header in res['headers'][1:]],
                                groups_limits=res['limits'][1:],
                                groups_ranges=res['limits'][1:]
                                )
                            templates_lim.append(template)
                    # TODO specify path for warnings? litt forklaring...

                    templates_lim = pop_already(
                        templates_lim, self.limits_and_plot_templates['Mammo'])

                    # include link to limits_and_plot_template in auto_vendor_templates
                    labels_lim_new = [x.label for x in templates_lim]
                    if len(labels_lim_new) > 0:
                        for template in templates:
                            if template.file_prefix in labels_lim_new:
                                template.limits_and_plot_label = template.file_prefix

                # save all new
                if self.templates['Mammo'][0].label == '':
                    self.templates['Mammo'] = templates
                else:
                    self.templates['Mammo'].extend(templates)
                if len(templates_lim) > 0:
                    if self.limits_and_plot_templates['Mammo'][0].label == '':
                        self.limits_and_plot_templates['Mammo'] = templates_lim
                    else:
                        self.limits_and_plot_templates['Mammo'].extend(templates_lim)
                    save_more = True
                    more = [self.limits_and_plot_templates]
                    more_fnames = ['limits_and_plot_templates']
                else:
                    save_more = False
                    more = None
                    more_fnames = None
                self.save(save_more=save_more, more=more, more_fnames=more_fnames)


class DashWorker(QThread):
    """Refresh dash and display in webbrowser."""

    def __init__(self, dash_settings=None):
        super().__init__()
        self.dash_settings = dash_settings

    def run(self):
        """Run worker if possible."""
        dash_app.run_dash_app(dash_settings=self.dash_settings)


class DashSettingsWidget(StackWidget):
    """Widget holding settings for dashboard visualization."""

    def __init__(self, dlg_settings):
        header = 'Dashboard settings'
        subtxt = (
            'Define where to run the dash application to visualize the output'
            ' from automation templates.<br>'
            'Default host is on your local computer (127.x.x.x). <br>'
            'If the dashboard is always updated from the same computer and this is '
            ' accessable for other computers in the same network, set host to '
            '"0.0.0.0"<br>'
            'The dashboard can then be accessed from http:\\xxx.x.x.x:port where '
            'xxx.x.x.x is the IP address of the computer'
            ' updating the dashboard.<br>'
            )
        super().__init__(dlg_settings, header, subtxt)
        self.fname = 'dash_settings'

        if self.import_review_mode:
            tb_marked = ToolBarImportIgnore(self, orientation=Qt.Horizontal)
            self.import_review_mark_txt = QLabel('Import and overwrite current')
            tb_marked.addWidget(self.import_review_mark_txt)
            hlo_import_tb = QHBoxLayout()
            hlo_import_tb.addStretch()
            hlo_import_tb.addWidget(tb_marked)
            hlo_import_tb.addStretch()
            self.vlo.addLayout(hlo_import_tb)

        wid_settings = QWidget()
        self.vlo.addWidget(wid_settings)
        vlo_settings = QVBoxLayout()
        hlo_settings = QHBoxLayout()
        vlo_settings.addLayout(hlo_settings)
        wid_settings.setLayout(vlo_settings)
        vlo_right = QVBoxLayout()
        vlo_left = QVBoxLayout()
        hlo_settings.addLayout(vlo_left)
        hlo_settings.addLayout(vlo_right)

        self.host = QLineEdit()
        self.host.textChanged.connect(lambda: self.flag_edit(True))
        self.port = QSpinBox(minimum=0, maximum=9999)
        self.port.editingFinished.connect(lambda: self.flag_edit(True))
        self.url_logo = QLineEdit()
        self.url_logo.textChanged.connect(lambda: self.flag_edit(True))
        self.header = QLineEdit()
        self.header.textChanged.connect(lambda: self.flag_edit(True))
        self.days_since_limit = QSpinBox(minimum=0, maximum=9999)
        self.days_since_limit.editingFinished.connect(lambda: self.flag_edit(True))
        self.plot_height = QSpinBox(minimum=50, maximum=2000)
        self.plot_height.editingFinished.connect(lambda: self.flag_edit(True))
        self.txt_table_headers_0 = QLineEdit()
        self.txt_table_headers_0.editingFinished.connect(lambda: self.flag_edit(True))
        self.txt_table_headers_1 = QLineEdit()
        self.txt_table_headers_1.editingFinished.connect(lambda: self.flag_edit(True))
        self.txt_table_headers_2 = QLineEdit()
        self.txt_table_headers_2.editingFinished.connect(lambda: self.flag_edit(True))
        self.txt_table_headers_3 = QLineEdit()
        self.txt_table_headers_3.editingFinished.connect(lambda: self.flag_edit(True))

        hlo_form = QHBoxLayout()
        flo = QFormLayout()
        vlo_left.addLayout(hlo_form)
        hlo_form.addLayout(flo)
        flo.addRow(QLabel('Host:'), self.host)
        flo.addRow(QLabel('Port:'), self.port)
        flo.addRow(QLabel('Dashboard header:'), self.header)
        flo.addRow(QLabel('Dashboard header logo url:'), self.url_logo)
        flo.addRow(QLabel('Days limit'), self.days_since_limit)
        flo.addRow(QLabel(''), uir.LabelItalic(
            '       Red font in overview table if > days limit.'))
        flo.addRow(QLabel('Plot height (pixels)'), self.plot_height)
        hlo_form.addStretch()
        hlo_form.addStretch()
        hlo_table_headers = QHBoxLayout()
        vlo_left.addLayout(hlo_table_headers)
        hlo_table_headers.addWidget(QLabel('Overview table headers:'))
        hlo_table_headers.addSpacing(50)
        self.txt_table_headers_0.setFixedWidth(200)
        self.txt_table_headers_1.setFixedWidth(200)
        self.txt_table_headers_2.setFixedWidth(200)
        #TODO add when ready status: self.txt_table_headers_3.setFixedWidth(200)
        hlo_table_headers.addWidget(self.txt_table_headers_0)
        hlo_table_headers.addWidget(self.txt_table_headers_1)
        hlo_table_headers.addWidget(self.txt_table_headers_2)
        #TODO add when ready status: hlo_table_headers.addWidget(self.txt_table_headers_3)
        hlo_table_headers.addStretch()

        self.n_colors = 7
        self.colortable = QTableWidget(1, self.n_colors)
        self.colortable.setFixedHeight(50)
        self.colortable.setSelectionMode(QTableWidget.SingleSelection)
        self.colortable.horizontalHeader().setVisible(False)
        self.colortable.verticalHeader().setVisible(False)
        for i in range(self.n_colors):
            self.colortable.setColumnWidth(i, 120)
            self.colortable.setCellWidget(
                0, i, uir.PushColorCell(self, initial_color="#000000", row=0, col=i))
        hlo_colors = QHBoxLayout()
        vlo_left.addLayout(hlo_colors)
        hlo_colors.addWidget(QLabel('Plot colors (click to edit):'))
        hlo_colors.addWidget(self.colortable)
        hlo_colors.addStretch()
        vlo_left.addStretch()
        vlo_left.addWidget(uir.LabelItalic(
            'If changes do not affect the dashboard - '
            'try changing the port number to force an update or restart imageQC.'))
        vlo_left.addStretch()

        if self.import_review_mode:
            wid_settings.setEnabled(False)
        else:
            btn_dash = QPushButton('Start dash_app')
            btn_dash.setIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}globe.png'))
            btn_dash.clicked.connect(self.run_dash)
            vlo_settings.addWidget(btn_dash)
            btn_save = QPushButton('Save dashboard settings')
            btn_save.setIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'))
            btn_save.clicked.connect(self.save_dashboard_settings)
            if self.save_blocked:
                btn_save.setEnabled(False)
            vlo_settings.addWidget(btn_save)

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_from_yaml(self, initial_template_label=''):
        """Refresh settings from yaml file.

        Using self.templates as common single template and
        self.current_template as TagPatternFormat to work smoothly
        with general code.
        """
        self.lastload = time()
        _, _, self.templates = cff.load_settings(fname=self.fname)
        self.update_data()
        self.flag_edit(False)

    def update_data(self):
        """Fill GUI with current data."""
        self.host.setText(self.templates.host)
        self.port.setValue(self.templates.port)
        self.url_logo.setText(self.templates.url_logo)
        self.header.setText(self.templates.header)
        self.days_since_limit.setValue(self.templates.days_since_limit)
        self.plot_height.setValue(self.templates.plot_height)
        self.txt_table_headers_0.setText(self.templates.overview_table_headers[0])
        self.txt_table_headers_1.setText(self.templates.overview_table_headers[1])
        self.txt_table_headers_2.setText(self.templates.overview_table_headers[2])
        self.txt_table_headers_3.setText(self.templates.overview_table_headers[3])

        for i in range(self.n_colors):
            w = self.colortable.cellWidget(0, i)
            w.setStyleSheet(
                f'QPushButton{{background-color: {self.templates.colors[i]};}}')
            w.setText(self.templates.colors[i])

    def color_edit(self, row, col):
        """Edit color for active cell."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.templates.colors[col] = color.name()
            w = self.colortable.cellWidget(0, col)
            w.setStyleSheet(
                f'QPushButton{{background-color: {self.templates.colors[col]};}}')
            w.setText(self.templates.colors[col])
            self.flag_edit(True)

    def get_current_template(self):
        """Update self.templates with current values."""
        self.templates.host = self.host.text()
        self.templates.port = self.port.value()
        self.templates.url_logo = self.url_logo.text()
        self.templates.header = self.header.text()
        self.templates.days_since_limit = self.days_since_limit.value()
        self.templates.plot_height = self.plot_height.value()
        self.templates.overview_table_headers[0] = self.txt_table_headers_0.text()
        self.templates.overview_table_headers[1] = self.txt_table_headers_1.text()
        self.templates.overview_table_headers[2] = self.txt_table_headers_2.text()
        self.templates.overview_table_headers[3] = self.txt_table_headers_3.text()

    def save_dashboard_settings(self):
        """Get current settings and save to yaml file."""
        self.get_current_template()
        self.save()

    def mark_import(self, ignore=False):
        """If import review mode: Mark Dash Settings for import or ignore."""
        if ignore:
            self.marked = False
            self.marked_ignore = True
            self.import_review_mark_txt.setText('Ignore')
        else:
            self.marked = True
            self.marked_ignore = False
            self.import_review_mark_txt.setText('Import and overwrite current')

    def run_dash(self):
        """Show results in browser."""
        config_folder = cff.get_config_folder()
        filenames = [x.stem for x in Path(config_folder).glob('*')
                     if x.suffix == '.yaml']
        if 'auto_templates' in filenames or 'auto_vendor_templates' in filenames:
            self.get_current_template()
            self.dash_worker = DashWorker(dash_settings=self.templates)
            self.dash_worker.start()
            dlg = messageboxes.MessageBoxWithDetails(
                self, title='Dashboard in webbrowser',
                msg='Results will open in a webbrowser.',
                info=('If large datasets or slow file-access you might have to refresh '
                      'the webpage. Look for "Serving on http... in the command window '
                      'when finished (or issues).'),
                icon=QMessageBox.Information)
            dlg.exec()
            url = f'http://{self.templates.host}:{self.templates.port}'
            webbrowser.open(url=url, new=1)
            self.dash_worker.exit()
        else:
            QMessageBox.information(
                self, 'Missing automation templates',
                '''Found no automation templates to display results from.''')
