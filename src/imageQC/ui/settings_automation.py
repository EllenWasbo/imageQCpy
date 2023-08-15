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
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, qApp, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QToolBar, QLabel, QLineEdit, QPushButton, QAction, QSpinBox, QCheckBox,
    QListWidget, QTableWidget, QTableWidgetItem, QComboBox,
    QMessageBox, QDialogButtonBox, QFileDialog
    )

# imageQC block start
from imageQC.config.iQCconstants import (
    ENV_ICON_PATH, LOG_FILENAME, VENDOR_FILE_OPTIONS, QUICKTEST_OPTIONS)
from imageQC.config import config_func as cff
from imageQC.config import config_classes as cfc
from imageQC.ui.settings_reusables import (
    StackWidget, ToolBarImportIgnore, DicomCritWidget)
from imageQC.ui.tag_patterns import TagPatternWidget, TagPatternEditDialog
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.ui_dialogs import ImageQCDialog, TextDisplay
from imageQC.ui import messageboxes
from imageQC.scripts.mini_methods import create_empty_file, create_empty_folder
from imageQC.scripts import read_vendor_QC_reports
from imageQC.scripts import dcm
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
            <b>Persons to notify</b><br>
            This is work in progress too. Persons can be added to the automation
            templates to recieve email if values outside set limits. <br>
            This functionality is under construction.
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
                filter="DICOM file (*.dcm);;All files (*)")
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


class LimitEditDialog(ImageQCDialog):
    """Dialog for editing limits."""

    def __init__(self, headers=None, first_values=None, min_max=None):
        super().__init__()
        self.headers = headers
        self.min_max = min_max

        self.setWindowTitle('Edit limits')
        self.setMinimumWidth(800)
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        self.table = QTableWidget(self)
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)
        n_rows = 2 if first_values is None else 3
        self.table.setRowCount(n_rows)

        for col in range(len(self.headers)):
            col_vals = [None, None, None]
            try:
                col_vals[:2] = min_max[col]
            except IndexError:
                pass
            try:
                col_vals[2] = first_values[col]
            except (IndexError, TypeError):
                pass
            for r in range(n_rows):
                twi = QTableWidgetItem(str(col_vals[r]))
                twi.setTextAlignment(4)
                self.table.setItem(r, col, twi)

        labels = ['min', 'max', 'sample value']
        self.table.setVerticalHeaderLabels(labels[0:n_rows])
        self.table.resizeRowsToContents()
        vlo.addWidget(self.table)
        self.table.cellChanged.connect(self.edit_current_table)

        hlo_dlg_btns = QHBoxLayout()
        vlo.addLayout(hlo_dlg_btns)
        hlo_dlg_btns.addStretch()
        btn_ok = QPushButton('OK')
        btn_ok.clicked.connect(self.accept)
        hlo_dlg_btns.addWidget(btn_ok)
        btn_cancel = QPushButton('Cancel')
        btn_cancel.clicked.connect(self.reject)
        hlo_dlg_btns.addWidget(btn_cancel)

    def edit_current_table(self, row, col):
        """Verify input."""
        if row < 2:
            val = self.table.item(row, col).text()
            if '.' in val or ',' in val:
                try:
                    val = float(val.replace(',', '.'))
                except (ValueError, TypeError):
                    val = None
            else:
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    val = None
            if len(self.min_max) == 0:
                self.min_max = [[None, None] for i in range(len(self.headers))]

            try:
                self.min_max[col][row] = val
            except IndexError:
                pass #TODO handle mismatch saved limits vs headers from output file

            twi = QTableWidgetItem(str(val))
            twi.setTextAlignment(4)
            self.table.blockSignals(True)
            self.table.setItem(row, col, twi)
            self.table.blockSignals(False)

    def get_min_max(self):
        """Return set values when closing."""
        return self.min_max


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

        self.gb_limits = QGroupBox('Limits')
        self.gb_limits.setFont(uir.FontItalic())
        vlo_limits = QVBoxLayout()
        self.gb_limits.setLayout(vlo_limits)

        vlo_limits.addWidget(uir.LabelItalic(
            'Set limits (min/max) for visualization and to trigger notifications'))
        hlo_limits = QHBoxLayout()
        vlo_limits.addLayout(hlo_limits)
        self.lbl_limits = QLabel('No limit set')
        self.tb_edit_limits = uir.ToolBarEdit(tooltip='Edit limits')
        hlo_limits.addWidget(self.lbl_limits)
        hlo_limits.addWidget(self.tb_edit_limits)
        hlo_limits.addStretch()
        self.tb_edit_limits.act_edit.triggered.connect(self.edit_limits)
        hlo_persons = QHBoxLayout()
        vlo_limits.addLayout(hlo_persons)
        self.list_persons = QListWidget()
        self.list_persons.setFixedWidth(200)
        self.tb_edit_persons = uir.ToolBarEdit(
            edit_button=False, add_button=True, delete_button=True)
        self.tb_edit_persons.act_add.triggered.connect(self.add_persons)
        self.tb_edit_persons.act_delete.triggered.connect(self.delete_person)
        hlo_persons.addWidget(QLabel('Notify:'))
        hlo_persons.addWidget(self.list_persons)
        hlo_persons.addWidget(self.tb_edit_persons)
        hlo_persons.addStretch()
        hlo_persons.addWidget(uir.UnderConstruction(txt='Under construction...'))

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
            self.view_output_file)
        act_new_file = QAction(QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                               'Create an empty file', toolb)
        act_new_file.triggered.connect(
            lambda: self.locate_file(
                self.txt_output_path, title='Locate output file',
                filter_str="Text file (*.txt)", opensave=True))
        toolb.addActions([act_new_file, act_output_view])

    def update_data(self):
        """Refresh GUI after selecting template."""
        self.txt_input_path.setText(self.current_template.path_input)
        self.txt_output_path.setText(self.current_template.path_output)
        self.txt_statname.setText(self.current_template.station_name)
        self.chk_archive.setChecked(self.current_template.archive)
        self.chk_deactivate.setChecked(not self.current_template.active)
        if len(self.current_template.min_max) > 0:
            self.lbl_limits.setText('Edit to view limits')
        else:
            self.lbl_limits.setText('No limit set')
        self.list_persons.clear()
        if len(self.current_template.persons_to_notify) > 0:
            self.list_person.addItems(self.current_template.persons_to_notify)

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
        self.current_template.station_name = self.txt_statname.text()
        self.current_template.active = not self.chk_deactivate.isChecked()
        self.current_template.archive = self.chk_archive.isChecked()

    def view_output_file(self):
        """View output file as txt."""
        if os.path.exists(self.txt_output_path.text()):
            os.startfile(self.txt_output_path.text())

    def edit_limits(self):
        """Edit min/max limits."""
        # headers from output or qt results on sample images?
        headers = []
        first_values = None
        if os.path.exists(self.txt_output_path.text()):
            with open(self.txt_output_path.text()) as f:
                headers = f.readline().strip('\n').split('\t')
                first_values = f.readline().strip('\n').split('\t')
        if len(headers) < 2:
            QMessageBox.information(
                self, 'Not output yet',
                'To set the limits there should be column headers to extract from '
                'the output text file. Output file is empty or missing columns.')
        else:
            dlg = LimitEditDialog(
                headers=headers[1:],
                min_max=self.current_template.min_max,
                first_values=first_values[1:])
            res = dlg.exec()
            if res:
                self.current_template.min_max = dlg.get_min_max()
                self.flag_edit(True)
                self.lbl_limits.setText('Edit to view limits')

    def add_persons(self):
        """Add to persons list."""
        QMessageBox.information(
            self, 'Not implemented yet',
            'This functionality is not yet finished. To be continued....')
        #TODO - only possible to add those with email address defined? 

    def delete_person(self):
        """Delete from persons list."""
        QMessageBox.information(
            self, 'Not implemented yet',
            'This functionality is not yet finished. To be continued....')


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

        vlo_right.addWidget(self.gb_limits)

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
        self.flag_edit(False)
        self.update_import_enabled()

        # update used_in
        self.list_same_input.clear()
        self.list_same_output.clear()
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
                filter="DICOM file (*.dcm);;All files (*)")
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
            limits = '' if len(self.current_template.min_max) == 0 else '-'
            notify = '' if len(self.current_template.persons_to_notify) == 0 else '-'
            if params + qt + outp + sort + limits + notify != '':
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
                    self.current_template.min_max = []
                    self.current_template.persons_to_notify = []
                    self.update_data()
        else:
            self.update_import_enabled()
        self.flag_edit(True)

    def update_import_enabled(self):
        """Update enabled guis whether import only or not."""
        self.gb_analyse.setEnabled(not self.current_template.import_only)
        self.gb_limits.setEnabled(not self.current_template.import_only)
        idx = self.hlo_output_path.count() - 1
        while(idx >= 0):
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
        self.vlo_temp.addStretch()
        self.vlo_temp.addWidget(self.gb_limits)
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
        self.flag_edit(False)

    def get_current_template(self):
        """Get self.current_template where not dynamically set."""
        super().get_current_template()
        file_type = self.cbox_file_type.currentText()
        self.current_template.file_type = file_type
        self.current_template.file_suffix = file_type.split('(')[1][:-1]

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


class DashWorker(QThread):
    """Refresh dash and display in webbrowser."""

    def __init__(self, widget_parent=None):
        super().__init__()
        self.widget_parent = widget_parent

    def run(self):
        """Run worker if possible."""
        dash_app.run_dash_app(widget=self.widget_parent)


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
            'Content and customizable parameters not finished yet...'
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

        vlo_left.addWidget(uir.UnderConstruction(txt='Under construction...'))

        self.host = QLineEdit()
        self.host.textChanged.connect(lambda: self.flag_edit(True))
        self.port = QSpinBox(minimum=0, maximum=9999)
        self.port.valueChanged.connect(lambda: self.flag_edit(True))
        self.url_logo = QLineEdit()
        self.url_logo.textChanged.connect(lambda: self.flag_edit(True))
        self.header = QLineEdit()
        self.header.textChanged.connect(lambda: self.flag_edit(True))

        flo = QFormLayout()
        vlo_left.addLayout(flo)
        flo.addRow(QLabel('Host:'), self.host)
        flo.addRow(QLabel('Port:'), self.port)
        flo.addRow(QLabel('Header logo url:'), self.url_logo)
        flo.addRow(QLabel('Header:'), self.header)

        vlo_left.addStretch()

        if self.import_review_mode:
            wid_settings.setEnabled(False)
        else:
            btn_dash = QPushButton('Start dash_app')
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
        self.host.setText(self.templates.host)
        self.port.setValue(self.templates.port)
        self.url_logo.setText(self.templates.url_logo)
        self.header.setText(self.templates.header)

    def get_current_template(self):
        """Update self.templates with current values."""
        self.templates.host = self.host.text()
        self.templates.port = self.port.value()
        self.templates.url_logo = self.url_logo.text()
        self.templates.header = self.header.text()

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
        self.get_current_template()
        self.dash_worker = DashWorker(widget_parent=self)
        self.dash_worker.start()
        url = f'http://{self.templates.host}:{self.templates.port}'
        webbrowser.open(url=url, new=1)


class PersonsToNotifyWidget(StackWidget):
    """Widget holding settings for FollowUpPersons."""

    def __init__(self, dlg_settings=None):
        header = 'Persons to notify'
        subtxt = (
            'Set up email adresses as receipents for a warning when set limits '
            'for automation outputs are violated.'
            )
        super().__init__(dlg_settings, header, subtxt,
                         temp_alias='person',
                         mod_temp=True, grouped=False
                         )
        self.fname = 'persons_to_notify'

        self.empty_template = cfc.PersonToNotify()
        self.current_template = self.empty_template

        self.txt_email = QLineEdit('')
        self.txt_email.textChanged.connect(self.flag_edit)
        self.txt_email.setMinimumWidth(200)

        self.txt_note = QLineEdit('')
        self.txt_note.textChanged.connect(self.flag_edit)
        self.txt_note.setMinimumWidth(200)
        self.list_mod = uir.ListWidgetCheckable(texts=[*QUICKTEST_OPTIONS])
        self.chk_mute = QCheckBox()
        self.chk_mute.stateChanged.connect(
            lambda: self.flag_edit(True))

        self.wid_temp = QWidget(self)
        self.hlo.addWidget(self.wid_temp)
        self.vlo_temp = QVBoxLayout()
        self.wid_temp.setLayout(self.vlo_temp)

        self.vlo_temp.addWidget(uir.UnderConstruction(txt='Under construction...'))

        flo = QFormLayout()
        self.vlo_temp.addLayout(flo)
        flo.addRow(QLabel('E-mail:'), self.txt_email)
        flo.addRow(QLabel('Note/comment:'), self.txt_note)
        flo.addRow(QLabel('Send all notifications for modalities:'), self.list_mod)
        flo.addRow(QLabel('Stop sending emails to this person.'), self.chk_mute)
        self.vlo_temp.addStretch()

        if not self.import_review_mode:
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_move_modality)
        else:
            self.wid_temp.setEnabled(False)

        self.wid_mod_temp.vlo.addWidget(
            QLabel('Selected person added to Automation template(s):'))
        self.list_used_in = QListWidget()
        self.wid_mod_temp.vlo.addWidget(self.list_used_in)

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with the selected template."""
        self.txt_email.setText(self.current_template.email)
        self.txt_note.setText(self.current_template.note)
        self.list_mod.set_checked_texts(self.current_template.all_notifications_mods)
        self.chk_mute.setChecked(self.current_template.mute)
        self.flag_edit(False)

        self.update_used_in()

    def update_used_in(self):
        """Update list of auto-templates with link to this person."""
        self.list_used_in.clear()
        if self.current_template.label != '':
            try:
                auto_labels = [
                    temp.label for temp in self.auto_templates[self.current_modality]
                    if self.current_template.label in temp.persons_to_notify
                    ]
            except KeyError:
                auto_labels = []
            try:
                auto_labels_vendor = [
                    temp.label for temp in self.auto_vendor_templates[
                        self.current_modality]
                    if self.current_template.label in temp.persons_to_notify
                    ]
                auto_labels.extend(auto_labels_vendor)
            except KeyError:
                pass
            if len(auto_labels) > 0:
                self.list_used_in.addItems(auto_labels)

    def get_current_template(self):
        """Get self.current_template where not dynamically set."""
        self.current_template.email = self.txt_email.text()
        self.current_template.note = self.txt_note.text()
        self.current_template.mute = self.chk_mute.isChecked()
        self.current_template.all_notifications_mods = self.list_mod.get_checked_texts()
