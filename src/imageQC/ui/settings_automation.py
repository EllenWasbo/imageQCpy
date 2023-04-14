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
            Currently imageQC have no visualization tools for the trends.
            Until that is in place, use f.x. Excel or PowerBI to visualize
            the trends.
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
            ' incoming files from the image pool.<br>'
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
        self.chk_pause_between = QCheckBox(
            'Pause between each template (option to cancel)')
        self.chk_pause_between.stateChanged.connect(
            lambda: self.flag_edit(True))
        self.chk_display_images = QCheckBox(
            'Display images/rois while tests are run')
        self.chk_display_images.stateChanged.connect(
            lambda: self.flag_edit(True))
        vlo_left.addWidget(self.chk_pause_between)
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
            btn_save = QPushButton('Save general automation settings')
            btn_save.setIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'))
            btn_save.clicked.connect(self.save_auto_common)
            if self.save_blocked:
                btn_save.setEnabled(False)
            vlo_common.addWidget(btn_save)

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_from_yaml(self):
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
        self.chk_pause_between.setChecked(not self.templates.auto_continue)
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
        self.templates.auto_continue = not self.chk_pause_between.isChecked()
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
        rows2delete = []
        for idx in sel_indexes:
            rows2delete.insert(0, idx.row())
        if len(rows2delete) > 0:
            for i in enumerate(rows2delete):
                self.templates.auto_delete_criterion_attributenames.pop(i)
                self.templates.auto_delete_criterion_values.pop(i)
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
        if log_path.exists:
            os.startfile(str(log_path))

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


class AutoTempWidgetBasic(StackWidget):
    """Common settings for AutoTemplates and AutoVendorTemplates."""

    def __init__(self, dlg_settings, header, subtxt):
        super().__init__(dlg_settings, header, subtxt,
                         typestr='template',
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

        hlo_input_path = QHBoxLayout()
        self.vlo_temp.addLayout(hlo_input_path)
        hlo_input_path.addWidget(QLabel('Input path '))
        hlo_input_path.addWidget(self.txt_input_path)
        toolb = uir.ToolBarBrowse('Browse to find path')
        toolb.act_browse.triggered.connect(
            lambda: self.locate_folder(self.txt_input_path))
        hlo_input_path.addWidget(toolb)

        hlo_output_path = QHBoxLayout()
        self.vlo_temp.addLayout(hlo_output_path)
        hlo_output_path.addWidget(QLabel('Output path '))
        hlo_output_path.addWidget(self.txt_output_path)
        toolb = uir.ToolBarBrowse('Browse to file')
        toolb.act_browse.triggered.connect(
            lambda: self.locate_file(
                self.txt_output_path, title='Locate output file',
                filter_str="Text file (*.txt)"))
        hlo_output_path.addWidget(toolb)
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

        gb_analyse = QGroupBox('Image analysis settings')
        gb_analyse.setFont(uir.FontItalic())
        vlo_analyse = QVBoxLayout()
        gb_analyse.setLayout(vlo_analyse)
        hlo_temp.addWidget(gb_analyse)

        flo_analyse = QFormLayout()
        vlo_analyse.addLayout(flo_analyse)
        flo_analyse.addRow(
            QLabel('Use parameter set: '),
            self.cbox_paramset)
        flo_analyse.addRow(
            QLabel('Use QuickTest template: '),
            self.cbox_quicktest)
        vlo_analyse.addStretch()
        vlo_analyse.addWidget(uir.LabelItalic(
            'Sort images for each date/studyUID by:'))
        hlo_sort_list = QHBoxLayout()
        vlo_analyse.addLayout(hlo_sort_list)
        hlo_sort_list.addWidget(self.list_sort_by)
        self.list_sort_by.setMinimumWidth(300)
        self.tb_edit_sort_by = uir.ToolBarEdit(tooltip='Edit sort list')
        hlo_sort_list.addWidget(self.tb_edit_sort_by)
        self.tb_edit_sort_by.act_edit.triggered.connect(self.edit_sort_by)

        hlo_btm = QHBoxLayout()
        self.vlo_temp.addLayout(hlo_btm)
        vlo_chk = QVBoxLayout()
        hlo_btm.addLayout(vlo_chk)
        vlo_chk.addWidget(self.chk_archive)
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

        self.sample_filepath = ''
        self.flag_edit(False)

        # update used_in
        self.list_same_input.clear()
        self.list_same_output.clear()
        if self.current_template.label != '':
            auto_labels = [
                temp.label for temp in self.templates[self.current_modality]
                if temp.path_input == self.current_template.path_input
                ]
            if len(auto_labels) > 1:
                auto_labels.remove(self.current_template.label)
                self.list_same_input.addItems(auto_labels)
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


class AutoVendorTemplateWidget(AutoTempWidgetBasic):
    """Widget holding automation settings."""

    def __init__(self, dlg_settings):
        header = 'Automation templates vendor files'
        subtxt = '''The automation templates hold information on how to
         perform automated reading of vendor report files.<br>'''
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
        self.vlo_temp.addWidget(self.chk_archive)
        self.vlo_temp.addWidget(self.chk_deactivate)
        self.vlo_temp.addStretch()

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_from_yaml(self):
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
