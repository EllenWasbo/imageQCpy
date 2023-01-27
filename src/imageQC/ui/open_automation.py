#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface for dialog Rename Dicom.

@author: Ellen Wasbo
"""
import os
from time import time
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar,
    QLabel, QLineEdit, QPushButton, QAction, QSpinBox, QCheckBox, QListWidget,
    QFileDialog, QMessageBox
    )

# imageQC block start
import imageQC.config.config_func as cff
import imageQC.ui.reusables as uir
from imageQC.scripts.input_main_auto import InputMain
import imageQC.ui.ui_image_canvas as ui_image_canvas
from imageQC.config.iQCconstants import ENV_ICON_PATH, QUICKTEST_OPTIONS
import imageQC.ui.settings as settings
import imageQC.scripts.automation as automation
# imageQC block end


def reset_auto_template(auto_template=None, parent_widget=None):
    """Move all files in path_input/Archive to path_input.

    Parameters
    ----------
    auto_template : AutoTemplate
    """
    archive_path = Path(auto_template.path_input) / "Archive"
    if archive_path.exists():
        move_files = []
        move_dirs = []
        dirs = [x for x in archive_path.glob('*') if x.is_dir()]
        if len(dirs) > 0:
            dlg = uir.ResetAutoTemplateDialog(parent_widget, directories=dirs)
            res = dlg.exec()
            if res:
                idxs = dlg.get_idxs()
                move_dirs = [dirs[idx] for idx in idxs]
                if len(move_dirs) > 0:
                    for folder in move_dirs:
                        files = [x for x in folder.glob('*') if x.is_file()]
                        move_files.extend(files)
        else:
            files = [x for x in archive_path.glob('*') if x.is_file()]
            if len(files) > 0:
                dlg = uir.ResetAutoTemplateDialog(files=files)
                move_files = [files[idx] for idx in idxs]
            else:
                QMessageBox.information(
                    parent_widget, 'Found no files in Archive',
                    f'Found no files in Archive for template {auto_template.label}')

        if len(move_files) > 0:
            for file in move_files:
                file.rename(Path(auto_template.path_input) / file.name)
            if len(move_dirs) > 0:
                for folder in move_dirs:
                    if not any(folder.iterdir()):
                        folder.rmdir()
            QMessageBox.information(
                parent_widget, 'Finished moving files',
                f'{len(move_files)} files moved out of Archive.')
    else:
        QMessageBox.information(
            parent_widget, 'Found no Archive',
            f'Found no Archive to reset for the template {auto_template.label}')


class ImageWidget(QWidget):
    """Image widget."""

    def __init__(self):
        super().__init__()
        self.main = InputMain()
        self.canvas = ui_image_canvas.ImageCanvas(self, self.main)
        self.setFixedSize(700, 700)
        vLO = QVBoxLayout()
        self.setLayout(vLO)
        vLO.addWidget(self.canvas)


class OpenAutomationDialog(uir.ImageQCDialog):
    """GUI setup for the Automation dialog window."""

    def __init__(self, main):
        super().__init__()

        self.setWindowTitle('Initiate automated analysis')

        self.main = main

        ok, path, self.auto_common = cff.load_settings(fname='auto_common')
        ok, path, self.templates = cff.load_settings(fname='auto_templates')
        ok, path, self.templates_vendor = cff.load_settings(
            fname='auto_vendor_templates')
        ok, path, self.paramsets = cff.load_settings(fname='paramsets')
        ok, path, self.qt_templates = cff.load_settings(
            fname='quicktest_templates')
        ok, path, self.tag_infos = cff.load_settings(fname='tag_infos')
        self.lastload_auto_common = time()

        self.wImageDisplay = ImageWidget()

        self.txt_import_path = QLineEdit(self.auto_common.import_path)

        hLO = QHBoxLayout()
        self.setLayout(hLO)
        vLO = QVBoxLayout()
        hLO.addLayout(vLO)
        vLO.addWidget(uir.LabelHeader(
            'Import and sort images', 4))
        self.txt_import_path.setMinimumWidth(500)
        hLO_import_path = QHBoxLayout()
        hLO_import_path.addWidget(QLabel('Image pool path:'))
        hLO_import_path.addWidget(self.txt_import_path)
        tb = uir.ToolBarBrowse('Browse to edit path')
        tb.actBrowse.triggered.connect(
            lambda: self.locate_folder(self.txt_import_path))
        actSavePath = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Save path as new default import path', self)
        tb.addAction(actSavePath)
        actSavePath.triggered.connect(self.save_import_path)
        hLO_import_path.addWidget(tb)
        vLO.addLayout(hLO_import_path)

        hLO_last_import = QHBoxLayout()
        hLO_last_import.addWidget(QLabel('Last import date: '))
        self.last_import = QLabel(self.auto_common.last_import_date)
        vLO.addLayout(hLO_last_import)

        self.chk_ignore_since = QCheckBox('Leave unsorted if image more than ')
        self.chk_ignore_since.setChecked(self.auto_common.ignore_since > 0)
        self.spin_ignore_since = QSpinBox()
        self.spin_ignore_since.setRange(1, 100)
        self.spin_ignore_since.setValue(self.auto_common.ignore_since)
        hLO_ignore_since = QHBoxLayout()
        vLO.addLayout(hLO_ignore_since)
        hLO_ignore_since.addWidget(self.chk_ignore_since)
        hLO_ignore_since.addWidget(self.spin_ignore_since)
        hLO_ignore_since.addWidget(QLabel(' days old'))
        tb = QToolBar()
        act_save_ignore = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Save setting', tb)
        act_save_ignore.triggered.connect(self.save_ignore_since)
        tb.addActions([act_save_ignore])
        hLO_ignore_since.addWidget(tb)
        hLO_ignore_since.addStretch()

        hLO_import_sort = QHBoxLayout()
        btn_import_sort = QPushButton('Import and sort from image pool')
        btn_import_sort.clicked.connect(self.import_sort)
        hLO_import_sort.addStretch()
        hLO_import_sort.addWidget(btn_import_sort)
        vLO.addLayout(hLO_import_sort)

        vLO.addWidget(uir.HLine())

        vLO.addWidget(uir.LabelHeader(
            'Run automation templates', 4))

        self.list_templates = QListWidget()
        self.templates_mod = []
        self.templates_is_vendor = []
        self.templates_id = []
        self.list_templates.addItems(self.get_template_list())

        hLO_temps = QHBoxLayout()
        vLO.addLayout(hLO_temps)
        tb_temps = QToolBar()
        tb_temps.setOrientation(Qt.Vertical)
        hLO_temps.addWidget(tb_temps)
        actRefresh = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}refresh.png'),
            'Refresh list with count of incoming files', self)
        actRefresh.triggered.connect(self.refresh_list)
        actOpenResultFile = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'Open result file', self)
        actOpenResultFile.triggered.connect(self.open_result_file)
        actResetTemplate = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}reset.png'),
            ('Move files for selected template(s) out of "Archive" folder to '
             're-run automation'),
            self)
        actResetTemplate.triggered.connect(self.reset_auto_template)
        actSettings = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}gears.png'),
            'Edit templates', self)
        actSettings.triggered.connect(self.edit_template)
        tb_temps.addActions(
            [actRefresh, actOpenResultFile, actResetTemplate, actSettings])

        hLO_temps.addWidget(self.list_templates)
        vLO_buttons = QVBoxLayout()
        hLO_temps.addLayout(vLO_buttons)
        btn_run_all = QPushButton('Run all')
        btn_run_selected = QPushButton('Run selected')
        btn_run_selected_files = QPushButton('Run for selected files...')
        self.chk_pause_between = QCheckBox(
            'Pause between each template (option to cancel)')
        self.chk_pause_between.setChecked(not self.auto_common.auto_continue)
        self.chk_display_images = QCheckBox(
            'Display images/rois while tests are run')
        self.chk_display_images.setChecked(self.auto_common.display_images)
        vLO_buttons.addWidget(btn_run_all)
        vLO_buttons.addWidget(btn_run_selected)
        vLO_buttons.addWidget(btn_run_selected_files)
        vLO_buttons.addWidget(self.chk_pause_between)
        vLO_buttons.addWidget(self.chk_display_images)
        btn_run_all.clicked.connect(self.run_all)
        btn_run_selected.clicked.connect(self.run_selected)
        btn_run_selected_files.clicked.connect(self.run_selected_files)

        hLO.addWidget(self.wImageDisplay)

        self.statusBar = uir.StatusBar(self)
        vLO.addWidget(self.statusBar)

        self.refresh_list()

    def locate_folder(self, widget):
        """Locate folder and set widget.text() to path.

        Parameters
        ----------
        widget : QLineEdit
            reciever of the path text
        """
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec():
            fname = dlg.selectedFiles()
            widget.setText(os.path.normpath(fname[0]))

    def save_import_path(self):
        """Save new import path to AutoCommon."""
        path = self.txt_import_path.text()
        self.auto_common.import_path = path
        self.save_auto_common()

    def save_ignore_since(self):
        """Save ignore since setting to AutoCommon."""
        self.auto_common.ignore_since = self.spin_ignore_since.value()
        self.save_auto_common()

    def save_auto_common(self):
        """Save template."""
        proceed = cff.test_config_folder(self)
        if proceed:
            proceed, errmsg = cff.check_save_conflict(
                'auto_common', self.lastload_auto_common)
            if errmsg != '':
                proceed = uir.proceed_question(self, errmsg)
            if proceed:
                ok, path = cff.save_settings(
                    self.auto_common, fname='auto_common')
                if ok:
                    self.lastload_auto_common = time()
                else:
                    QMessageBox.warning(
                        self, 'Failed saving', f'Failed saving to {path}')

    def import_sort(self):
        """Import and sort images from image pool."""
        path = self.txt_import_path.text()
        if Path(path).exists():
            ignore_since = 0
            if self.chk_ignore_since.isChecked():
                ignore_since = self.spin_ignore_since.value()
            log_import = automation.import_incoming(
                self.auto_common, self.templates, self.tag_infos, parent_widget=self,
                ignore_since=ignore_since)
            self.stop_wait_cursor()
            self.statusBar.showMessage('Finished reading images', 2000)
            if len(log_import) > 0:
                dlg = uir.TextDisplay(
                    self, '\n'.join(log_import),
                    title='Information',
                    min_width=1000, min_height=300)
                res = dlg.exec()
                if res:
                    pass  # just to wait for user to close message
                
        else:
            QMessageBox.information(
                self, 'Warning', 'Import path not defined or do not exist.')

    def count_files(self, folder_path):
        """Count number of files in path.

        Parameters
        ----------
        path : str

        Returns
        -------
        n_files : int
        """
        n_files = 0
        try:
            for path in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, path)):
                    n_files += 1
        except FileNotFoundError:
            n_files = -1

        return n_files

    def get_template_list(self, count=False):
        """Get list of templates (modality - (vendor file) - label).

        Parameters
        ----------
        count : bool, optional
            Count and list number of incoming files. The default is False.

        Returns
        -------
        template_list : list of str
            list of template labels (+ modality and if vendor file template)
        """
        template_list = []
        modalities = [*QUICKTEST_OPTIONS]
        for mod in modalities:
            arr = []
            for temp in self.templates[mod]:
                if temp.label != '':
                    arr.append(mod + ' - ' + temp.label)
                    if count:
                        n_files = self.count_files(temp.path_input)
                        if n_files > 0:
                            arr[-1] = arr[-1] + '(' + str(n_files) + ')'
            if len(arr) > 0:
                self.templates_mod.extend([mod] * len(arr))
                self.templates_is_vendor.extend([False] * len(arr))
                self.templates_id.extend(
                    [i for i, temp in enumerate(self.templates[mod])])
                template_list.extend(arr)
            arr = []
            for temp in self.templates_vendor[mod]:
                if temp.label != '':
                    arr.append(mod + ' - (vendor file) ' + temp.label)
                    if count:
                        n_files = self.count_files(temp.path_input)
                        if n_files > 0:
                            arr[-1] = arr[-1] + '(' + str(n_files) + ')'
            if len(arr) > 0:
                self.templates_mod.extend([mod] * len(arr))
                self.templates_is_vendor.extend([True] * len(arr))
                self.templates_id.extend(
                    [i for i, temp in enumerate(self.templates_vendor[mod])])
                template_list.extend(arr)

        return template_list

    def refresh_list(self):
        """Refresh list of templates with count = True."""
        templist = self.get_template_list(count=True)
        self.list_templates.clear()
        if len(templist) > 0:
            self.list_templates.addItems(templist)

    def open_result_file(self):
        """Display result file."""
        sel = self.list_templates.selectedIndexes()
        if len(sel) > 0:
            tempno = sel[0].row()
            mod = self.templates_mod[tempno]
            if self.templates_is_vendor[tempno]:
                temp = self.templates_vendor[mod][self.templates_id[tempno]]
            else:
                temp = self.templates[mod][self.templates_id[tempno]]
            path = temp.path_output
            if os.path.exists(path):
                os.startfile(path)
            else:
                QMessageBox.warning(
                    self, 'File not found',
                    f'File not found {path}'
                    )

    def reset_auto_template(self):
        """Move files from Archive to input_path."""
        sel = self.list_templates.selectedIndexes()
        if len(sel) > 0:
            for s in sel:
                tempno = s.row()
                mod = self.templates_mod[tempno]
                id_this = self.templates_id[tempno]
                temp = self.templates[mod][id_this]
                reset_auto_template(auto_template=temp, parent_widget=self)

    def edit_template(self):
        """Run settings with current template selected."""
        view = 'Templates DICOM'
        sel = self.list_templates.selectedIndexes()
        if len(sel) > 0:
            tempno = sel[0].row()
            if self.templates_is_vendor[tempno]:
                view = 'Templates vendor files'
        dlg = settings.SettingsDialog(
            self.main, initial_view=view)
        dlg.exec()

    def run_all(self):
        """Run all templates."""
        tempno_list = [i for i in range(len(self.templates_mod))]
        self.run_templates(tempnos=tempno_list)

    def run_selected(self):
        """Run selected templates."""
        sels = self.list_templates.selectedIndexes()
        if len(sels) > 0:
            tempno_list = [sel.row() for sel in sels]
            self.run_templates(tempnos=tempno_list)

    def run_selected_files(self):
        """Run one selected template and define input files to use."""
        sels = self.list_templates.selectedIndexes()
        if len(sels) != 1:
            QMessageBox.warning(
                self, 'Select one template',
                'Select one template to run with manually selected files.'
                )
        else:
            tempno_list = [sels[0].row()]
            self.run_templates(tempnos=tempno_list, select_files=True)

    def run_templates(self, tempnos=[], select_files=False):
        """Run selected templates.

        Parameters
        ----------
        tempnos : list of int, optional
            The template numbers (in widget list) to run. The default is [].
        select_files : bool, optional
            True if offer manual input file selection. The default is False.
        """
        if len(tempnos) > 0:
            log = []
            self.automation_active = True
            #self.main.start_wait_cursor()

            for i, tempno in enumerate(tempnos):
                proceed = True
                if self.chk_pause_between.isChecked() and i > 0:
                    #self.main.stop_wait_cursor()
                    proceed = uir.proceed_question(self, 'Continue to next template?')
                    #self.main.start_wait_cursor()
                if proceed is False:
                    break
                mod = self.templates_mod[tempno]
                if self.templates_is_vendor[tempno]:
                    temp_this = self.templates_vendor[mod][self.templates_id[tempno]]
                else:
                    temp_this = self.templates[mod][self.templates_id[tempno]]

                pre_selected_files = []
                if select_files:
                    if self.templates_is_vendor[tempno]:
                        filt = f'File *.{temp_this.file_type}'
                    else:
                        filt = 'DICOM files (*.dcm);;All files (*)'

                    if temp_this.input_path == '':
                        fnames = QFileDialog.getOpenFileNames(
                            self, 'Open DICOM files', filter=filt)
                    else:
                        fnames = QFileDialog.getOpenFileNames(
                            self, 'Open DICOM files', temp_this.input_path, filter=filt)
                    pre_selected_files = fnames[0]

                if self.templates_is_vendor[tempno]:
                    pre_msg = self.main.statusBar.text()
                    self.main.statusBar.showMessage(
                        f'{pre_msg} Extracting data from vendor report'
                        )
                    msgs, not_written = automation.run_template_vendor(
                        temp_this, mod, pre_selected_files=pre_selected_files
                        )
                else:
                    msgs, not_written = automation.run_template(
                        temp_this, mod,
                        self.paramsets, self.qt_templates, self.tag_infos,
                        pre_selected_files=pre_selected_files,
                        parent_widget=self#, input_main_this=self.auto_main
                        )
                if len(msgs) > 0:
                    for msg in msgs:
                        log.append(msg)
                if len(not_written) > 0:
                    #TODO offer to clipboard
                    breakpoint()
                    pass

            self.automation_active = False
            self.main.refresh_results_display()
            self.main.statusBar.showMessage('Finished', 1000)
            #self.main.stop_wait_cursor()

            if len(log) > 0:
                if isinstance(log, list):
                    log = '\n'.join(log)
                QMessageBox.warning(
                    self, 'Automation log', log)
