#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface for dialog Rename Dicom.

@author: Ellen Wasbo
"""
import os
from time import time
import shutil
from datetime import datetime
from pathlib import Path
import logging

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QAbstractItemView,
    QLabel, QLineEdit, QAction, QSpinBox, QCheckBox, QListWidget,
    QFileDialog, QMessageBox
    )

# imageQC block start
from imageQC.config import config_func as cff
from imageQC.ui.ui_dialogs import ImageQCDialog, ResetAutoTemplateDialog, TextDisplay
from imageQC.ui import reusable_widgets as uir
from imageQC.ui import messageboxes
from imageQC.scripts.input_main import InputMain
from imageQC.ui import ui_image_canvas
from imageQC.config.iQCconstants import ENV_ICON_PATH, QUICKTEST_OPTIONS
from imageQC.ui import settings
from imageQC.scripts import automation
from imageQC.scripts.mini_methods import get_all_matches, find_files_prefix_suffix
from imageQC.scripts.dcm import sort_imgs
# imageQC block end

logger = logging.getLogger('imageQC')


def reset_auto_template(auto_template=None, parent_widget=None):
    """Move all files in path_input/Archive to path_input.

    Parameters
    ----------
    auto_template : AutoTemplate
    """
    archive_path = Path(auto_template.path_input) / "Archive"
    if os.path.exists(auto_template.path_input):
        move_files = []
        move_dirs = []
        dirs = [x for x in archive_path.glob('*') if x.is_dir()]
        if len(dirs) > 0:
            dlg = ResetAutoTemplateDialog(
                parent_widget, directories=dirs,
                template_name=auto_template.label,
                original_folder=str(archive_path),
                target_folder=auto_template.path_input)
            res = dlg.exec()
            if res:
                idxs = dlg.get_idxs()
                move_dirs = [dirs[idx] for idx in idxs]
                if len(move_dirs) > 0:
                    for folder in move_dirs:
                        files = [x for x in folder.glob('*') if x.is_file()]
                        move_files.extend(files)
        else:
            try:
                files, _ = find_files_prefix_suffix(
                    archive_path, auto_template.file_prefix, auto_template.file_suffix)
            except AttributeError:
                files = [x for x in archive_path.glob('*') if x.is_file()]

            if len(files) > 0:
                QAP_Mammo = False
                try:
                    if auto_template.file_type == 'GE Mammo QAP (txt)':
                        QAP_Mammo = True
                except AttributeError:
                    pass
                dlg = ResetAutoTemplateDialog(
                    parent_widget, files=files,
                    template_name=auto_template.label,
                    original_folder=str(archive_path),
                    target_folder=auto_template.path_input,
                    QAP_Mammo=QAP_Mammo)
                res = dlg.exec()
                if res:
                    idxs = dlg.get_idxs()
                    move_files = [files[idx] for idx in idxs]
            else:
                QMessageBox.information(
                    parent_widget, 'Found no files in Archive',
                    f'Found no files in Archive for template {auto_template.label}')

        if len(move_files) > 0:
            if parent_widget is not None:
                parent_widget.main.start_wait_cursor()
                parent_widget.status_label.showMessage(
                    f'Moving files for {auto_template.label}')
            errmsgs = []
            for file in move_files:
                try:
                    file.rename(Path(auto_template.path_input) / file.name)
                except FileExistsError as err:
                    errmsgs.append(str(err))
                except:
                    errmsgs.append(f'Failed moving {file.name}')
            if len(move_dirs) > 0:
                for folder in move_dirs:
                    if not any(folder.iterdir()):
                        folder.rmdir()
            if parent_widget is not None:
                parent_widget.main.stop_wait_cursor()
                parent_widget.status_label.clearMessage()
            if len(errmsgs) > 0:
                dlg = messageboxes.MessageBoxWithDetails(
                    parent_widget, title='Issues when moving files',
                    msg='Some files could not be moed. See details.',
                    details=errmsgs, icon=QMessageBox.Warning)
                dlg.exec()
            else:
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

        self.templates_mod = []  # list of modality for all templates in unfiltered list
        self.templates_is_vendor = []  # bool type vendor for all templ in unfilt list
        self.templates_id = []  # id within each template list
        self.templates_displayed_names_all = []  # name without count in unfilt list
        self.templates_displayed_ids = []  # curr displayed ids (ids of unfilt list)

        self.wid_image_display = ImageWidget()
        self.txt_import_path = QLineEdit('')

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
        hlo_last_import.addWidget(QLabel('Last import date/time: '))
        self.last_import = QLabel('')
        hlo_last_import.addWidget(self.last_import)
        hlo_last_import.addStretch()
        vlo.addLayout(hlo_last_import)

        self.chk_ignore_since = QCheckBox('Leave unsorted if image more than ')
        self.spin_ignore_since = QSpinBox()
        self.spin_ignore_since.setRange(1, 100)
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
        btn_width = 350
        btn_import_sort = uir.PushButtonWithIcon(
            'Import and sort from image pool',
            'import', align='left', width=btn_width)
        btn_import_sort.clicked.connect(self.import_sort)
        hlo_import_sort.addStretch()
        hlo_import_sort.addWidget(btn_import_sort)
        vlo.addLayout(hlo_import_sort)

        vlo.addWidget(uir.HLine())

        vlo.addWidget(uir.LabelHeader(
            'Run automation templates', 4))

        self.list_templates = QListWidget()
        self.list_templates.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_templates.currentItemChanged.connect(self.update_template_selected)

        hlo_temps = QHBoxLayout()
        vlo.addLayout(hlo_temps)
        tb_temps = QToolBar()
        tb_temps.setOrientation(Qt.Vertical)
        hlo_temps.addWidget(tb_temps)
        act_refresh = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}refresh.png'),
            'Refresh list', self)
        act_refresh.triggered.connect(self.refresh_list)
        act_open_result_file = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'Open result file', self)
        act_open_result_file.triggered.connect(self.open_result_file)
        act_open_input_path = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}open.png'),
            'Open input path in file explorer', self)
        act_open_input_path.triggered.connect(self.open_input_path)
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
            [act_open_result_file, act_open_input_path,
             act_reset_template, act_settings])

        vlo_list = QVBoxLayout()
        hlo_temps.addWidget(tb_temps)
        vlo_list.addWidget(self.list_templates)
        hlo_temps.addLayout(vlo_list)
        vlo_buttons = QVBoxLayout()
        hlo_temps.addLayout(vlo_buttons)

        all_modalities = [*QUICKTEST_OPTIONS][:-1]
        if len(self.main.gui.auto_filter_modalities) == 0:
            checked_modalities = list(range(len(all_modalities)))
        else:
            checked_modalities = [
                idx for idx, mod in enumerate(all_modalities)
                if mod in self.main.gui.auto_filter_modalities]
        self.list_modalities = uir.ListWidgetCheckable(
            texts=all_modalities, set_checked_ids=checked_modalities)
        hlo_mod = QHBoxLayout()
        tb_filter = QToolBar()
        tb_filter.setOrientation(Qt.Vertical)
        tb_filter.addAction(act_refresh)
        hlo_mod.addWidget(tb_filter)
        vlo_buttons.addWidget(uir.LabelItalic('Filter:'))
        vlo_buttons.addLayout(hlo_mod)
        hlo_mod.addSpacing(25)
        hlo_mod.addWidget(self.list_modalities)
        hlo_mod.addStretch()

        btn_run_all = uir.PushButtonWithIcon(
            'Run all', 'play', align='left', width=btn_width)
        btn_run_selected = uir.PushButtonWithIcon(
            'Run selected',
            'play', align='left', width=btn_width)
        btn_run_selected_files = uir.PushButtonWithIcon(
            'Run in main window for selected files',
            'play', align='left', width=btn_width)
        self.btn_import_qap = uir.PushButtonWithIcon(
            'Import GE QAP files', 'import',
            align='left', width=btn_width)
        self.btn_import_qap.setToolTip(
            'Import GE Mammo QAP txt results from user '
            'specified folder with all results. Import from '
            'set date to templates with same Input path.')
        self.btn_import_qap.clicked.connect(self.import_qap)
        self.btn_import_qap.setEnabled(False)
        self.chk_display_images = QCheckBox(
            'Display images/ROIs while tests are run')
        vlo_buttons.addWidget(btn_run_all)
        vlo_buttons.addWidget(btn_run_selected)
        vlo_buttons.addWidget(btn_run_selected_files)
        vlo_buttons.addWidget(self.btn_import_qap)
        vlo_buttons.addWidget(self.chk_display_images)
        btn_run_all.clicked.connect(self.run_templates)
        btn_run_selected.clicked.connect(self.run_selected)
        btn_run_selected_files.clicked.connect(self.run_selected_files)

        hlo.addWidget(self.wid_image_display)

        self.status_label = uir.StatusLabel(self)
        hlo_status = QHBoxLayout()
        hlo_status.addWidget(self.status_label)
        vlo.addLayout(hlo_status)
        self.progress_bar = uir.ProgressBar(self)
        vlo.addWidget(self.progress_bar)

        QTimer.singleShot(300, lambda: self.update_settings())
        # allow time to show dialog before updating list

    def update_settings(self):
        """Read settings from saved data at start and after settings edited."""
        self.main.start_wait_cursor()
        self.status_label.showMessage('Reading settings...')
        _, _, self.auto_common = cff.load_settings(fname='auto_common')
        self.txt_import_path.setText(self.auto_common.import_path)
        self.last_import.setText(self.auto_common.last_import_date)
        self.chk_ignore_since.setChecked(self.auto_common.ignore_since > 0)
        self.spin_ignore_since.setValue(self.auto_common.ignore_since)
        self.chk_display_images.setChecked(self.auto_common.display_images)

        _, _, self.templates = cff.load_settings(fname='auto_templates')
        _, _, self.templates_vendor = cff.load_settings(
            fname='auto_vendor_templates')
        _, _, self.limits_and_plot_templates = cff.load_settings(
            fname='limits_and_plot_templates')
        _, _, self.paramsets = cff.load_settings(fname='paramsets')
        self.quicktest_templates = self.main.quicktest_templates
        self.digit_templates = self.main.digit_templates
        self.tag_infos = self.main.tag_infos
        self.lastload_auto_common = time()
        self.main.stop_wait_cursor()
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
        if self.chk_ignore_since.isChecked():
            self.auto_common.ignore_since = self.spin_ignore_since.value()
        else:
            self.auto_common.ignore_since = -1
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
        if os.path.exists(path):
            ignore_since = -1
            if self.chk_ignore_since.isChecked():
                ignore_since = self.spin_ignore_since.value()
            import_status, log_import = automation.import_incoming(
                self.auto_common, self.templates, self.tag_infos, parent_widget=self,
                override_path=path, ignore_since=ignore_since)

            if import_status:
                fname = 'auto_common'
                proceed_save, errmsg = cff.check_save_conflict(
                    fname, self.lastload_auto_common)
                if proceed_save:
                    # save today as last import date to auto_common
                    self.auto_common.last_import_date = datetime.now().strftime(
                        "%d.%m.%Y %I:%M")
                    ok_save, path = cff.save_settings(
                        self.auto_common, fname=fname)
                    self.last_import.setText(self.auto_common.last_import_date)

            if len(log_import) > 0:
                self.status_label.showMessage('Finished reading images')
                log = '\n'.join(log_import)
                logger.info(log)
                dlg = TextDisplay(
                    self, log,
                    title='Information',
                    min_width=1000, min_height=300)
                res = dlg.exec()
                if res:
                    pass  # just to wait for user to close message
                self.refresh_list()
            self.status_label.clearMessage()
        else:
            QMessageBox.information(
                self, 'Warning', 'Import path not defined or do not exist.')

    def count_files(self, auto_template):
        """Count number of files in path.

        Parameters
        ----------
        auto_template : as defined in config_classes

        Returns
        -------
        n_files : int
        error_ex : errormessage
        """
        n_files = 0
        error_ex = None
        p_input = Path(auto_template.path_input)
        try:
            proceed = p_input.is_dir()
        except OSError as ex:
            proceed = False
            error_ex = f'{ex}'
        if proceed:
            try:
                files, error_ex = find_files_prefix_suffix(
                    p_input, auto_template.file_prefix, auto_template.file_suffix)
                n_files = len(files)
            except AttributeError:
                try:
                    files = [x for x in p_input.glob('*')
                             if x.is_file() and x.name != 'Thumbs.db']
                    n_files = len(files)
                except (FileNotFoundError, OSError) as ex:
                    n_files = -1
                    error_ex = f'{ex}'

        return (n_files, error_ex)

    def update_template_selected(self, current):
        """(De)activate import GE QAP."""
        set_enabled = False
        tempno = self.list_templates.currentIndex().row()
        if (
                self.templates_mod[tempno] == 'Mammo'
                and self.templates_is_vendor[tempno]):  # TODO assumes only one vendor type for now
            set_enabled = True
        self.btn_import_qap.setEnabled(set_enabled)

    def import_qap(self):
        """Select folder with QAP results (all history), select from date."""
        dlg = QFileDialog(self, 'Select folder with QAP results')
        dlg.setFileMode(QFileDialog.Directory)
        files = []
        folder = ''
        if dlg.exec():
            fname = dlg.selectedFiles()
            if fname[0] != '':
                mod, _, tempno = self.get_selected_templates_mod_id()
                auto_template = self.templates_vendor[mod[0]][tempno[0]]
                proceed = True
                if auto_template.path_input == '':
                    QMessageBox.warning(
                        self, 'No input path specified',
                        'The selected template has no input path defined. '
                        'Go to settings to fix this.'
                        )
                    proceed = False

                filenames_already = []
                if proceed:
                    # find files in given folder
                    # ignore if same filename already
                    # in input_path (including Archive)
                    self.main.start_wait_cursor()
                    p_input = Path(auto_template.path_input)
                    filenames_already = [
                        x.name for x in p_input.glob('**/*')
                        if x.is_file()]

                    folder = Path(fname[0])
                    files = [
                        x for x in folder.glob('**/*')
                        if (x.is_file()
                            and x.name not in filenames_already)
                        ]
                    self.main.stop_wait_cursor()

                if len(files) == 0:
                    txt_new = ''
                    if len(filenames_already) > 0:
                        txt_new = '(new)'
                    QMessageBox.warning(
                        self, 'No files found',
                        f'No {txt_new} valid files found in the '
                        'selected folder.')

        if len(files) > 0:
            dlg = ResetAutoTemplateDialog(
                self, files=files,
                template_name=auto_template.label,
                original_folder=str(folder),
                target_folder=auto_template.path_input,
                QAP_Mammo=True, import_Mammo=True)
            res = dlg.exec()
            if res:
                idxs = dlg.get_idxs()
                move_files = [files[idx] for idx in idxs]
                for file_orig in move_files:
                    file_new = (
                        Path(auto_template.path_input) / 
                        Path(file_orig).stem)
                    self.move_file(file_orig, file_new)

    def move_file(self, filepath_orig, filepath_new):
        """Rename and move input file."""
        status = False
        errmsg = ''
        if os.path.exists(filepath_new):
            errmsg = f'Failed to move\n{filepath_orig}\nto {filepath_new}\nExists already.'
        else:
            try:
                filepath_orig.rename(filepath_new)
                status = True
            except (PermissionError, OSError) as err:
                if 'WinError 17' in str(err):
                    shutil.move(filepath_orig, filepath_new)
                    status = True
                else:
                    errmsg = f'Failed to rename\n{filepath_orig}\nto {filepath_new}\n{err}'

        return (status, errmsg)

    def get_selected_templates_mod_id(self):
        """Get selected templates modalities and ids.

        Returns
        -------
        mods : list of str
            modalities for selected templates
        is_vendor : list of bool
            is_vendor? for selected templates
        ids : list of int
            id within template list for selected templates
        """
        sel = self.list_templates.selectedIndexes()
        mods = []
        is_vendor = []
        ids = []
        if len(sel) > 0:
            for idx in sel:
                tempno_displayed = idx.row()
                tempno = self.templates_displayed_ids[tempno_displayed]
                mods.append(self.templates_mod[tempno])
                is_vendor.append(self.templates_is_vendor[tempno])
                ids.append(self.templates_id[tempno])
        return (mods, is_vendor, ids)

    def get_template_list(self):
        """Get list of templates (modality - (vendor file) - label).

        Returns
        -------
        template_list : list of str
            list of template labels (+ modality and if vendor file template)
        """
        template_list = []
        modalities = [*QUICKTEST_OPTIONS]
        errormsgs = []
        self.templates_displayed_names_all = []
        self.templates_mod = []
        self.templates_is_vendor = []
        self.templates_id = []
        counter = 0
        n_temps = sum(
            [len(temps) for key, temps in self.templates.items()])
        n_temps_vendor = sum(
            [len(temps) for key, temps in self.templates_vendor.items()])
        n_temps_total = n_temps + n_temps_vendor
        self.progress_bar.setRange(0, n_temps_total)
        self.status_label.showMessage(
            'Refreshing list of templates and number of waiting files...')

        for mod in modalities:
            arr = []
            for temp in self.templates[mod]:
                if temp.label != '':
                    name = mod + ' - ' + temp.label
                    self.templates_displayed_names_all.append(name)
                    arr.append(name)
                    if temp.import_only:
                        if temp.active:
                            arr[-1] = f'-\t{arr[-1]}  (import_only)'
                        else:
                            arr[-1] = f'-\t{arr[-1]}  (import_only and deactivated)'
                    else:
                        if temp.active:
                            if temp.path_input != '':
                                n_files, err = self.count_files(temp)
                                if n_files > 0:
                                    arr[-1] = f'({n_files})\t{arr[-1]}'
                                else:
                                    arr[-1] = f'-\t{arr[-1]}'
                                    if err is not None:
                                        errormsgs.append(f'{err}')
                            else:
                                arr[-1] = f'-\t{arr[-1]} (input path unknown)'
                        else:
                            arr[-1] = f'-\t{arr[-1]} (deactivated)'

                    counter = counter + 1
                    self.progress_bar.setValue(counter)

            if len(arr) > 0:
                self.templates_mod.extend([mod] * len(arr))
                self.templates_is_vendor.extend([False] * len(arr))
                self.templates_id.extend(
                    [i for i, temp in enumerate(self.templates[mod])])
                template_list.extend(arr)
            arr = []
            for temp in self.templates_vendor[mod]:
                if temp.label != '':
                    name = mod + ' - (vendor file) ' + temp.label
                    self.templates_displayed_names_all.append(name)
                    arr.append(name)
                    if temp.active:
                        if temp.path_input != '':
                            n_files, err = self.count_files(temp)
                            if n_files > 0:
                                arr[-1] = f'({n_files})\t{arr[-1]}'
                            else:
                                arr[-1] = f'-\t{arr[-1]}'
                                if err is not None:
                                    errormsgs.append(f'{err}')
                        else:
                            arr[-1] = f'-\t{arr[-1]} (input path unknown)'
                    elif temp.active is False:
                        arr[-1] = f'-\t{arr[-1]} (deactivated)'
                    counter = counter + 1
                    self.progress_bar.setValue(counter)

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
        self.progress_bar.reset()
        self.status_label.clearMessage()

        return template_list

    def refresh_list(self, rows=None):
        """Refresh list of templates with count.

        Parameters
        ----------
        rows : list of int, optional
            Specify which templates to update count for. The default is None.
        """
        modalities = self.list_modalities.get_checked_texts()
        self.main.gui.auto_filter_modalities = modalities
        self.main.start_wait_cursor()
        if not isinstance(rows, list):
            templist = self.get_template_list()
            self.list_templates.clear()
            if len(templist) > 0:
                self.templates_displayed_ids = []
                for mod in modalities:
                    self.templates_displayed_ids.extend(
                        get_all_matches(self.templates_mod, mod))
                filt_templist = [
                    templist[i] for i in self.templates_displayed_ids]
                self.list_templates.addItems(filt_templist)

        else:
            ids = [self.templates_displayed_ids[row] for row in rows]
            for no, idx in enumerate(ids):
                mod = self.templates_mod[idx]
                if self.templates_is_vendor[idx]:
                    temp_this = self.templates_vendor[mod][self.templates_id[idx]]
                else:
                    temp_this = self.templates[mod][self.templates_id[idx]]

                if temp_this.active and temp_this.path_input != '':
                    n_files, err = self.count_files(temp_this)
                    if n_files > 0:
                        txt = f'({n_files})\t{self.templates_displayed_names_all[idx]}'
                    else:
                        txt = f'-\t{self.templates_displayed_names_all[idx]}'
                    item = self.list_templates.item(rows[no])
                    item.setText(txt)
        self.main.stop_wait_cursor()

    def open_result_file(self):
        """Display result file."""
        sel = self.list_templates.selectedIndexes()
        if len(sel) > 0:
            mods, is_vendors, ids = self.get_selected_templates_mod_id()
            if is_vendors[0]:
                temp = self.templates_vendor[mods[0]][ids[0]]
            else:
                temp = self.templates[mods[0]][ids[0]]

            path = Path(temp.path_output).resolve()
            if os.path.exists(path):
                os.startfile(path)
            else:
                QMessageBox.warning(
                    self, 'File not found',
                    f'File not found {path}'
                    )
            self.list_templates.setCurrentRow(sel[0].row())

    def open_input_path(self):
        """Open file explorer in input path."""
        sel = self.list_templates.selectedIndexes()
        if len(sel) > 0:
            mods, is_vendors, ids = self.get_selected_templates_mod_id()
            if is_vendors[0]:
                temp = self.templates_vendor[mods[0]][ids[0]]
            else:
                temp = self.templates[mods[0]][ids[0]]

            path = Path(temp.path_input).resolve()
            if os.path.exists(path):
                os.startfile(path)
            else:
                QMessageBox.warning(
                    self, 'Path not found',
                    f'Input path not found {path}'
                    )
            self.list_templates.setCurrentRow(sel[0].row())

    def reset_auto_template(self):
        """Move files from Archive to path_input."""
        sel = self.list_templates.selectedIndexes()
        if len(sel) > 0:
            mods, is_vendors, ids = self.get_selected_templates_mod_id()
            for i in range(len(ids)):
                if is_vendors[i]:
                    temp = self.templates_vendor[mods[i]][ids[i]]
                else:
                    temp = self.templates[mods[i]][ids[i]]
                reset_auto_template(auto_template=temp, parent_widget=self)
                self.refresh_list(rows=[sel[i].row()])

            self.list_templates.setCurrentRow(sel[0].row())

    def edit_template(self):
        """Run settings with current template selected."""
        view = 'Templates DICOM'
        mods, is_vendors, ids = self.get_selected_templates_mod_id()
        if len(is_vendors) > 0:
            if is_vendors[0]:
                view = 'Templates vendor files'
        try:
            if is_vendors[0]:
                temp_this = self.templates_vendor[mods[0]][ids[0]]
            else:
                temp_this = self.templates[mods[0]][ids[0]]
            dlg = settings.SettingsDialog(
                self.main, initial_view=view, initial_modality=mods[0],
                initial_template_label=temp_this.label
                )
        except IndexError:
            dlg = settings.SettingsDialog(self.main, initial_view=view)

        res = dlg.exec()
        if res == 0:  # when closing
            self.update_settings()

    def run_selected(self):
        """Run selected templates."""
        sels = self.list_templates.selectedIndexes()
        if len(sels) > 0:
            self.run_templates(selected_only=True)
            self.list_templates.setCurrentRow(sels[-1].row())

    def run_selected_files(self):
        """Run one selected template in main window and define input files to use."""
        sels = self.list_templates.selectedIndexes()
        if len(sels) != 1:
            QMessageBox.warning(
                self, 'Select one template',
                'Select one template to run with manually selected files.'
                )
        else:
            mods, is_vendors, ids = self.get_selected_templates_mod_id()
            proceed = True
            if is_vendors[0]:
                temp_this = self.templates_vendor[mods[0]][ids[0]]
                filt = f'File *.{temp_this.file_type}'
                open_txt = 'Open vendor QC files'
            else:
                temp_this = self.templates[mods[0]][ids[0]]
                if temp_this.import_only:
                    QMessageBox.warning(
                        self, 'Template import only',
                        'The selected template is marked as import only, no analysis.'
                        )
                    proceed = False
                else:
                    filt = 'DICOM files (*.dcm *.IMA);;All files (*)'
                    open_txt = 'Open DICOM files'
            if proceed:
                if temp_this.path_input == '':
                    fnames = QFileDialog.getOpenFileNames(
                        self, open_txt, filter=filt)
                else:
                    fnames = QFileDialog.getOpenFileNames(
                        self, open_txt, temp_this.path_input, filter=filt)
                pre_selected_files = fnames[0]

                if len(pre_selected_files) > 0:
                    self.main.clear_all_images()
                    if is_vendors[0]:
                        self.main.btn_read_vendor_file.setChecked(True)
                        self.main.current_modality = mods[0]
                        self.main.update_mode()
                        self.main.tab_vendor.run_template(
                            template=temp_this, files=fnames)
                        self.close()
                    else:
                        self.close()
                        self.main.start_wait_cursor()
                        self.main.chk_append.setChecked(False)
                        self.main.btn_read_vendor_file.setChecked(False)
                        self.main.wid_quicktest.gb_quicktest.setChecked(False)
                        self.main.wid_center.reset_delta()
                        self.main.open_files(file_list=pre_selected_files)
                        self.main.imgs, _ = sort_imgs(
                            self.main.imgs, temp_this.sort_pattern, self.tag_infos)
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
                        self.main.stop_wait_cursor()
                        if temp_this.quicktemp_label in quicktemp_labels:
                            self.main.gui.current_auto_template = temp_this.label
                            self.main.wid_quicktest.cbox_template.setCurrentText(
                                temp_this.quicktemp_label)
                            self.main.wid_quicktest.gb_quicktest.setChecked(True)
                        else:
                            proceed = False
                        if proceed:
                            self.main.tree_file_list.update_file_list()
                            self.main.wid_quicktest.run_quicktest()
                            if temp_this.path_output != '':
                                reply = QMessageBox.question(
                                    self, 'Open results file?',
                                    'Open template results file for easy access to '
                                    'editing if corrections are needed?',
                                    QMessageBox.Yes, QMessageBox.No)
                                if reply == QMessageBox.Yes:
                                    self.open_result_file()

    def run_templates(self, selected_only=False):
        """Run selected templates.

        Parameters
        ----------
        selected_only : bool, optional
            Limit to selected templates in list. The default is False.
            Filter will be used also if selected_only is False.
        """
        tempnos = []
        #TODO delete or recode below? mods, is_vendors, ids = self.get_selected_templates_mod_id()
        if selected_only:
            sel = self.list_templates.selectedIndexes()
            tempnos = [self.templates_displayed_ids[idx.row()] for idx in sel]
        else:
            tempnos = self.templates_displayed_ids
        if len(tempnos) > 0:
            log = []
            warnings_all = []
            self.automation_active = True
            max_progress = 100*len(tempnos)  # 0-100 within each temp
            self.progress_modal = uir.ProgressModal(
                "Running template.", "Cancel",
                0, max_progress, self, minimum_duration=0)

            for i, tempno in enumerate(tempnos):
                warnings = []
                mod = self.templates_mod[tempno]
                proceed = False
                if self.templates_is_vendor[tempno]:
                    temp_this = self.templates_vendor[mod][self.templates_id[tempno]]
                    proceed = temp_this.active
                else:
                    temp_this = self.templates[mod][self.templates_id[tempno]]
                    if temp_this.active and temp_this.import_only is False:
                        proceed = True

                if proceed:
                    if i == 0:
                        self.progress_modal.setLabelText("Preparing data...")
                        self.progress_modal.setValue(1)
                    else:
                        self.progress_modal.setValue(100 * i)
                    if self.templates_is_vendor[tempno]:
                        pre_msg = f'Template {mod}/{temp_this.label}:'
                        self.progress_modal.setLabelText(
                            f'{pre_msg} Extracting data from vendor report...')
                        msgs, warnings, not_written = automation.run_template_vendor(
                            temp_this, mod, self.limits_and_plot_templates,
                            decimal_mark=self.paramsets[mod][0].output.decimal_mark,
                            parent_widget=self)
                    else:
                        msgs, warnings, not_written = automation.run_template(
                            temp_this, mod,
                            self.paramsets,
                            self.quicktest_templates, self.digit_templates,
                            self.limits_and_plot_templates,
                            self.tag_infos, parent_widget=self
                            )
                    if len(msgs) > 0:
                        for msg in msgs:
                            log.append(msg)
                    if len(warnings) > 0:
                        paths_already = [msg_list[0] for msg_list in warnings_all]
                        if warnings[0] in paths_already:
                            idx = paths_already.index(warnings[0])
                            warnings_all[idx].extend(warnings[1:])
                        else:
                            warnings_all.append(warnings)
                    if len(not_written) > 0:
                        log.append('Results failed to be written to output file.')
                        # TODO offer to clipboard
                        pass
                if self.progress_modal.wasCanceled():
                    break
            self.status_label.clearMessage()
            self.progress_modal.setValue(max_progress)

            self.automation_active = False

            if len(warnings_all) > 0:
                for warnings in warnings_all:
                    proceed = os.path.exists(warnings[0])
                    if proceed:
                        with open(warnings[0], "a") as f:
                            f.write('\n'.join(warnings[1:]))
                    else:
                        log.append('Failed adding warnings for violated limits to '
                                   f'{warnings[0]}')

            if len(log) > 0:
                if isinstance(log, list):
                    log = '\n'.join(log)
                logger.info(log)
                dlg = TextDisplay(
                    self, log,
                    title='Information',
                    min_width=1000, min_height=300)
                res = dlg.exec()
                if res:
                    pass  # just to wait for user to close message

            if selected_only:
                sels = self.list_templates.selectedIndexes()
                rows = [sel.row() for sel in sels]
                self.refresh_list(rows=rows)
            else:
                self.refresh_list()
