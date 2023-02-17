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
from imageQC.config import config_func as cff
from imageQC.ui.ui_dialogs import ImageQCDialog, ResetAutoTemplateDialog, TextDisplay
from imageQC.ui import reusable_widgets as uir
from imageQC.ui import messageboxes
from imageQC.scripts.input_main_auto import InputMain
from imageQC.ui import ui_image_canvas
from imageQC.config.iQCconstants import ENV_ICON_PATH, QUICKTEST_OPTIONS
from imageQC.ui import settings
from imageQC.scripts import automation
from imageQC.scripts.dcm import sort_imgs
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
            dlg = ResetAutoTemplateDialog(parent_widget, directories=dirs)
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
                dlg = ResetAutoTemplateDialog(files=files)
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
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        vlo.addWidget(self.canvas)


class OpenAutomationDialog(ImageQCDialog):
    """GUI setup for the Automation dialog window."""

    def __init__(self, main):
        super().__init__()

        self.setWindowTitle('Initiate automated analysis')

        self.main = main

        _, _, self.auto_common = cff.load_settings(fname='auto_common')
        _, _, self.templates = cff.load_settings(fname='auto_templates')
        _, _, self.templates_vendor = cff.load_settings(
            fname='auto_vendor_templates')
        _, _, self.paramsets = cff.load_settings(fname='paramsets')
        self.quicktest_templates = self.main.quicktest_templates
        self.tag_infos = self.main.tag_infos
        self.lastload_auto_common = time()

        self.wid_image_display = ImageWidget()

        self.txt_import_path = QLineEdit(self.auto_common.import_path)

        hlo = QHBoxLayout()
        self.setLayout(hlo)
        vlo = QVBoxLayout()
        hlo.addLayout(vlo)
        vlo.addWidget(uir.LabelHeader(
            'Import and sort images', 4))
        self.txt_import_path.setMinimumWidth(500)
        hlo_import_path = QHBoxLayout()
        hlo_import_path.addWidget(QLabel('Image pool path:'))
        hlo_import_path.addWidget(self.txt_import_path)
        toolb = uir.ToolBarBrowse('Browse to edit path')
        toolb.act_browse.triggered.connect(
            lambda: self.locate_folder(self.txt_import_path))
        act_save_path = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Save path as new default import path', self)
        toolb.addAction(act_save_path)
        act_save_path.triggered.connect(self.save_import_path)
        hlo_import_path.addWidget(toolb)
        vlo.addLayout(hlo_import_path)

        hlo_last_import = QHBoxLayout()
        hlo_last_import.addWidget(QLabel('Last import date: '))
        self.last_import = QLabel(self.auto_common.last_import_date)
        vlo.addLayout(hlo_last_import)

        self.chk_ignore_since = QCheckBox('Leave unsorted if image more than ')
        self.chk_ignore_since.setChecked(self.auto_common.ignore_since > 0)
        self.spin_ignore_since = QSpinBox()
        self.spin_ignore_since.setRange(1, 100)
        self.spin_ignore_since.setValue(self.auto_common.ignore_since)
        hlo_ignore_since = QHBoxLayout()
        vlo.addLayout(hlo_ignore_since)
        hlo_ignore_since.addWidget(self.chk_ignore_since)
        hlo_ignore_since.addWidget(self.spin_ignore_since)
        hlo_ignore_since.addWidget(QLabel(' days old'))
        toolb = QToolBar()
        act_save_ignore = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Save setting', toolb)
        act_save_ignore.triggered.connect(self.save_ignore_since)
        toolb.addActions([act_save_ignore])
        hlo_ignore_since.addWidget(toolb)
        hlo_ignore_since.addStretch()

        hlo_import_sort = QHBoxLayout()
        btn_import_sort = QPushButton('Import and sort from image pool')
        btn_import_sort.clicked.connect(self.import_sort)
        hlo_import_sort.addStretch()
        hlo_import_sort.addWidget(btn_import_sort)
        vlo.addLayout(hlo_import_sort)

        vlo.addWidget(uir.HLine())

        vlo.addWidget(uir.LabelHeader(
            'Run automation templates', 4))

        self.list_templates = QListWidget()
        self.templates_mod = []
        self.templates_is_vendor = []
        self.templates_id = []
        self.list_templates.addItems(self.get_template_list())

        hlo_temps = QHBoxLayout()
        vlo.addLayout(hlo_temps)
        tb_temps = QToolBar()
        tb_temps.setOrientation(Qt.Vertical)
        hlo_temps.addWidget(tb_temps)
        act_refresh = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}refresh.png'),
            'Refresh list with count of incoming files', self)
        act_refresh.triggered.connect(self.refresh_list)
        act_open_result_file = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'Open result file', self)
        act_open_result_file.triggered.connect(self.open_result_file)
        act_reset_template = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}reset.png'),
            ('Move files for selected template(s) out of "Archive" folder to '
             're-run automation'),
            self)
        act_reset_template.triggered.connect(self.reset_auto_template)
        act_settings = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}gears.png'),
            'Edit templates', self)
        act_settings.triggered.connect(self.edit_template)
        tb_temps.addActions(
            [act_refresh, act_open_result_file, act_reset_template, act_settings])

        hlo_temps.addWidget(self.list_templates)
        vlo_buttons = QVBoxLayout()
        hlo_temps.addLayout(vlo_buttons)
        btn_run_all = QPushButton('Run all')
        btn_run_selected = QPushButton('Run selected')
        btn_run_selected_files = QPushButton('Run in main window for selected files...')
        self.chk_pause_between = QCheckBox(
            'Pause between each template (option to cancel)')
        self.chk_pause_between.setChecked(not self.auto_common.auto_continue)
        self.chk_display_images = QCheckBox(
            'Display images/rois while tests are run')
        self.chk_display_images.setChecked(self.auto_common.display_images)
        vlo_buttons.addWidget(btn_run_all)
        vlo_buttons.addWidget(btn_run_selected)
        vlo_buttons.addWidget(btn_run_selected_files)
        vlo_buttons.addWidget(self.chk_pause_between)
        vlo_buttons.addWidget(self.chk_display_images)
        btn_run_all.clicked.connect(self.run_all)
        btn_run_selected.clicked.connect(self.run_selected)
        btn_run_selected_files.clicked.connect(self.run_selected_files)

        hlo.addWidget(self.wid_image_display)

        self.status_bar = uir.StatusBar(self)
        vlo.addWidget(self.status_bar)

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
        proceed = cff.verify_config_folder(self)
        if proceed:
            proceed, errmsg = cff.check_save_conflict(
                'auto_common', self.lastload_auto_common)
            if errmsg != '':
                proceed = messageboxes.proceed_question(self, errmsg)
            if proceed:
                ok_save, path = cff.save_settings(
                    self.auto_common, fname='auto_common')
                if ok_save:
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
            self.status_bar.showMessage('Finished reading images', 2000)
            if len(log_import) > 0:
                dlg = TextDisplay(
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
        error_ex : errormessage
        """
        n_files = 0
        error_ex = None
        try:
            for path in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, path)):
                    n_files += 1
        except (FileNotFoundError, OSError) as ex:
            n_files = -1
            error_ex = f'{ex}'

        return (n_files, error_ex)

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
        errormsgs = []
        for mod in modalities:
            arr = []
            for temp in self.templates[mod]:
                if temp.label != '':
                    arr.append(mod + ' - ' + temp.label)
                    if count:
                        n_files, err = self.count_files(temp.path_input)
                        if n_files > 0:
                            arr[-1] = arr[-1] + '(' + str(n_files) + ')'
                        else:
                            if err is not None:
                                errormsgs.append(f'{err}')
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
                        if temp.path_input != '':
                            n_files, err = self.count_files(temp.path_input)
                            if n_files > 0:
                                arr[-1] = arr[-1] + '(' + str(n_files) + ')'
                            else:
                                if err is not None:
                                    errormsgs.append(f'{err}')
                        else:
                            arr[-1] = arr[-1] + '(input path unknown)'
            if len(arr) > 0:
                self.templates_mod.extend([mod] * len(arr))
                self.templates_is_vendor.extend([True] * len(arr))
                self.templates_id.extend(
                    [i for i, temp in enumerate(self.templates_vendor[mod])])
                template_list.extend(arr)

        if len(errormsgs) > 0:
            dlg = messageboxes.MessageBoxWithDetails(
                self, title='Issues when searching for files',
                msg='Something went wrong while searching for files. See details.',
                details=errormsgs, icon=QMessageBox.Warning)
            dlg.exec()

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
        """Move files from Archive to path_input."""
        sel = self.list_templates.selectedIndexes()
        if len(sel) > 0:
            for sno in sel:
                tempno = sno.row()
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
        tempno_list = list(range(len(self.templates_mod)))
        self.run_templates(tempnos=tempno_list)

    def run_selected(self):
        """Run selected templates."""
        sels = self.list_templates.selectedIndexes()
        if len(sels) > 0:
            tempno_list = [sel.row() for sel in sels]
            self.run_templates(tempnos=tempno_list)

    def run_selected_files(self):
        """Run one selected template in main window and define input files to use."""
        sels = self.list_templates.selectedIndexes()
        if len(sels) != 1:
            QMessageBox.warning(
                self, 'Select one template',
                'Select one template to run with manually selected files.'
                )
        else:
            tempno = sels[0].row()
            mod = self.templates_mod[tempno]
            if self.templates_is_vendor[tempno]:
                temp_this = self.templates_vendor[mod][self.templates_id[tempno]]
                filt = f'File *.{temp_this.file_type}'
                open_txt = 'Open vendor QC files'
            else:
                temp_this = self.templates[mod][self.templates_id[tempno]]
                filt = 'DICOM files (*.dcm);;All files (*)'
                open_txt = 'Open DICOM files'

            if temp_this.path_input == '':
                fnames = QFileDialog.getOpenFileNames(self, open_txt, filter=filt)
            else:
                fnames = QFileDialog.getOpenFileNames(
                    self, open_txt, temp_this.path_input, filter=filt)
            pre_selected_files = fnames[0]

            if self.templates_is_vendor[tempno]:
                self.main.clear_all_images()
                self.main.btn_read_vendor_file.setChecked(True)
                self.main.current_modality = mod
                self.main.update_mode()
                self.main.tab_vendor.run_template(template=temp_this, files=fnames)
            else:
                self.main.start_wait_cursor()
                self.main.chk_append.setChecked(False)
                self.main.open_files(file_list=pre_selected_files)
                self.main.imgs = sort_imgs(
                    self.main.imgs, temp_this.sort_pattern, self.tag_infos)
                self.main.tree_file_list.update_file_list()
                paramset_labels = [temp.label for temp in self.main.paramsets]
                quicktemp_labels = [
                    temp.label for temp in self.main.quicktest_templates[
                        self.main.current_modality]]
                proceed = True
                if temp_this.paramset_label in paramset_labels:
                    self.main.wid_paramset.cbox_template.setCurrentText(
                        temp_this.paramset_label)
                else:
                    proceed = False
                if temp_this.quicktemp_label in quicktemp_labels:
                    self.main.wid_quicktest.gb_quicktest.setChecked(True)
                    self.main.wid_quicktest.cbox_template.setCurrentText(
                        temp_this.quicktemp_label)
                else:
                    proceed = False
                if proceed:
                    self.main.wid_quicktest.run_quicktest()
                self.main.stop_wait_cursor()
            self.close()

    def run_templates(self, tempnos=[]):
        """Run selected templates.

        Parameters
        ----------
        tempnos : list of int, optional
            The template numbers (in widget list) to run. The default is [].
        """
        if len(tempnos) > 0:
            log = []
            self.automation_active = True
            self.main.start_wait_cursor()

            for i, tempno in enumerate(tempnos):
                proceed = True
                if self.chk_pause_between.isChecked() and i > 0:
                    self.main.stop_wait_cursor()
                    proceed = messageboxes.proceed_question(
                        self, 'Continue to next template?')
                    self.main.start_wait_cursor()
                if proceed is False:
                    break
                mod = self.templates_mod[tempno]
                if self.templates_is_vendor[tempno]:
                    temp_this = self.templates_vendor[mod][self.templates_id[tempno]]
                else:
                    temp_this = self.templates[mod][self.templates_id[tempno]]

                if self.templates_is_vendor[tempno]:
                    pre_msg = self.main.status_bar.text()
                    self.main.status_bar.showMessage(
                        f'{pre_msg} Extracting data from vendor report'
                        )
                    msgs, not_written = automation.run_template_vendor(temp_this, mod)
                else:
                    msgs, not_written = automation.run_template(
                        temp_this, mod,
                        self.paramsets, self.quicktest_templates, self.tag_infos,
                        parent_widget=self
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
            self.main.status_bar.showMessage('Finished', 1000)
            self.main.stop_wait_cursor()

            if len(log) > 0:
                if isinstance(log, list):
                    log = '\n'.join(log)
                QMessageBox.warning(
                    self, 'Automation log', log)
