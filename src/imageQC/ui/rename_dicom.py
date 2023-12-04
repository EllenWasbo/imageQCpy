#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface for dialog Rename Dicom.

@author: Ellen Wasbo
"""
import os
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QToolBar,
    QLabel, QLineEdit, QPushButton, QAction, QTreeWidget, QTreeWidgetItem, QFileDialog,
    QMessageBox
    )

# imageQC block start
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.ui import messageboxes
from imageQC.ui.settings_dicom_tags import RenamePatternWidget
from imageQC.config import config_classes as cfc
from imageQC.config.iQCconstants import ENV_ICON_PATH
from imageQC.scripts import dcm
from imageQC.scripts.mini_methods_format import (
    valid_path, generate_uniq_filepath)
from imageQC.scripts.mini_methods import get_all_matches
# imageQC block end


class RenameDicomDialog(ImageQCDialog):
    """GUI setup for the Rename Dicom dialog window."""

    def __init__(self, main):
        super().__init__()
        self.current_modality = main.current_modality

        self.orignal_names = []  # list for found files or folders (first step)
        self.new_names = []  # list for generated new names (first step)
        self.valid_dict = {}  # dict for folders (first) + files (second step)

        self.setWindowTitle('Rename DICOM')
        splitter = QSplitter(Qt.Vertical)
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        vlo.addWidget(splitter)

        subtxt = '''Rename DICOM files and folders by DICOM header data.
        Subfolders will be renamed by the first file found in the folder.<br>
        Note also option to split files into subfolders based on
        DICOM header or gather all DICOM files into one folder.<br>
        The split series will use seriesUID if subfolder template is not set.'''
        self.wid_rename_pattern = RenamePatternWidget(
            initial_modality=self.current_modality, header='', subtxt=subtxt)
        self.wid_rename_pattern.save_blocked = main.save_blocked
        splitter.addWidget(self.wid_rename_pattern)

        wid_btm = QWidget()
        splitter.addWidget(wid_btm)
        vlo2 = QVBoxLayout()
        wid_btm.setLayout(vlo2)
        hlo_browse = QHBoxLayout()
        vlo2.addLayout(hlo_browse)
        hlo_browse.addWidget(QLabel('Selected folder: '))
        self.path = QLineEdit()
        hlo_browse.addWidget(self.path)
        tb_browse = uir.ToolBarBrowse(
            'Browse for folder with DICOM files', clear=True)
        tb_browse.act_browse.triggered.connect(self.browse)
        tb_browse.act_clear.triggered.connect(self.clear_path)
        hlo_browse.addWidget(tb_browse)

        vlo2.addWidget(uir.LabelItalic(
            """If selected folder is left blank, open images
            in main window will be subject to renaming."""))

        tb_split_gather = QToolBar()
        hlo_browse.addWidget(tb_split_gather)
        act_split = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}split.png'),
            'Split/move files into folders defined by subfolder template or seriesUID',
            self)
        act_split.triggered.connect(self.split_series)
        act_gather = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}gather.png'),
            'Gather all DICOM files in subfolders to selected folder', self)
        act_gather.triggered.connect(self.gather_files)
        tb_split_gather.addActions([act_split, act_gather])

        hlo_table = QHBoxLayout()
        vlo2.addLayout(hlo_table)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(
            ['Original name', 'Suggested name'])
        self.table.setRootIsDecorated(False)
        self.table.setColumnWidth(0, 500)
        self.table.setColumnWidth(1, 500)
        hlo_table.addWidget(self.table)
        vlo_btns_table = QVBoxLayout()
        hlo_table.addLayout(vlo_btns_table)
        btn_test_10 = QPushButton('Test 10 first')
        btn_test_10.clicked.connect(self.test_10_first)
        vlo_btns_table.addWidget(btn_test_10)
        btn_generate_names = QPushButton('Prepare names')
        btn_generate_names.clicked.connect(self.generate_names)
        vlo_btns_table.addWidget(btn_generate_names)
        self.btn_rename = QPushButton('Rename')
        self.btn_rename.setDisabled(True)
        self.btn_rename.clicked.connect(self.rename)
        vlo_btns_table.addWidget(self.btn_rename)
        vlo_btns_table.addStretch()

        hlo_dlg_btns = QHBoxLayout()
        vlo.addLayout(hlo_dlg_btns)
        hlo_dlg_btns.addStretch()
        btn_close = QPushButton('Close window')
        btn_close.clicked.connect(self.accept)
        hlo_dlg_btns.addWidget(btn_close)

        self.wid_rename_pattern.update_from_yaml()

    def reset_names(self):
        """Reset variables/GUI if generated names no longer valid."""
        self.orignal_names = []
        self.new_names = []
        self.table.clear()
        self.btn_rename.setDisabled(True)

    def fill_table(self, limit=None):
        """Fill table with original and generated names."""
        self.table.clear()
        if len(self.original_names) > 0 and len(self.new_names) > 0:
            orignames = [x.name for x in self.original_names]
            newnames = [x.name for x in self.new_names]
            if len(orignames) > len(newnames):  # limited to a few
                orignames = orignames[:len(newnames)]

            for idx, fname in enumerate(orignames):
                row_strings = [fname, newnames[idx]]
                item = QTreeWidgetItem(row_strings)
                self.table.addTopLevelItem(item)

    def browse(self):
        """Locate folder."""
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec():
            fname = dlg.selectedFiles()
            self.path.setText(os.path.normpath(fname[0]))
            self.reset_names()

    def clear_path(self):
        """Reset path."""
        self.path.setText('')
        self.reset_names()

    def gather_files(self):
        """Move all dicom files in subfolders to this folder."""
        progress_modal = uir.ProgressModal(
            "Searching...", "Stop", 0, 100, self, hide_cancel=True)
        dcm_dict = dcm.find_all_valid_dcm_files(
            self.path.text(), parent_widget=self, grouped=False,
            progress_modal=progress_modal)
        progress_modal.reset()
        if len(dcm_dict['files']) > 0:
            # check if parent is the specified folder? Else no need to move
            proceed = True
            if (len(dcm_dict['folders']) == 1
                    and dcm_dict['folders'][0] == self.path.text()):
                proceed = False

            if proceed:
                proceed = messageboxes.proceed_question(
                    self,
                    f'Found {len(dcm_dict["files"])} valid DICOM files. '
                    'Proceed moving these files directly into the '
                    'specified folder?')

            if proceed:
                count_renamed = 0
                failed_paths = []
                n_files = len(dcm_dict['files'])
                progress_modal.setRange(0, n_files)
                for i, file in enumerate(dcm_dict['files']):
                    path = Path(file)
                    new_file = Path(self.path.text()) / path.name
                    new_file_str = generate_uniq_filepath(new_file.resolve())
                    msg = ''
                    if new_file_str != '':
                        try:
                            path.rename(new_file_str)
                            count_renamed += 1
                        except FileExistsError:
                            msg = (f'FileExistsError: Failed moving {file} to '
                                   f'{new_file_str}')
                        except FileNotFoundError:
                            msg = (f'FileNotFoundError: Failed moving {file} to '
                                   f'{new_file_str}')
                    else:
                        msg = f'Failed generating unique filename for {file}'
                    if msg:
                        failed_paths.append(msg)
                        print(msg)
                    progress_modal.setValue(i+1)
                    progress_modal.setLabelText(f'Renaming file {i+1}/{n_files}...')
                self.reset_names()
                if count_renamed != len(dcm_dict['files']):
                    dlg = messageboxes.MessageBoxWithDetails(
                        self, title='Issues',
                        msg=(f'Moved {count_renamed} of {len(dcm_dict["files"])}.'
                             'See details for paths failing.'),
                        details=failed_paths, icon=QMessageBox.Warning)
                    dlg.exec()

                proceed = messageboxes.proceed_question(
                    self, 'Remove empty folders?')

            if proceed:
                for root, dirs, _ in os.walk(
                        self.path.text(), topdown=False):
                    for name in dirs:
                        if len(os.listdir(os.path.join(root, name))) == 0:
                            os.rmdir(os.path.join(root, name))
                        else:
                            pass

    def split_series(self):
        """Split series into subfolders based on subfolder tag pattern."""
        errmsg = []
        series_uid = False
        if len(self.wid_rename_pattern.current_template.list_tags) == 0:
            series_uid = True
            rename_files = False
        else:
            tag_pattern = cfc.TagPatternFormat(
                list_tags=self.wid_rename_pattern.current_template.list_tags,
                list_format=self.wid_rename_pattern.current_template.list_format)
            tag_pattern2 = cfc.TagPatternFormat(
                list_tags=self.wid_rename_pattern.current_template.list_tags2,
                list_format=self.wid_rename_pattern.current_template.list_format2)
            rename_files = True if len(tag_pattern2.list_tags) > 0 else False

        proceed = True
        if os.access(self.path.text(), os.W_OK) is False:
            proceed = False
            errmsg = ['No writing permissing for given path.']

        if proceed:
            progress_modal = uir.ProgressModal(
                "Searching...", "Stop", 0, 100, self, hide_cancel=True)
            dcm_dict = dcm.find_all_valid_dcm_files(
                self.path.text(), parent_widget=self, progress_modal=progress_modal,
                grouped=False, search_subfolders=False)
            progress_modal.reset()
            if len(dcm_dict['files']) > 0:

                if series_uid:
                    info_txt = 'if they share series UID?'
                else:
                    info_txt = (
                        'if they share the same parameters defined in '
                        'the subfolder pattern?')
                proceed = messageboxes.proceed_question(
                    self,
                    (f'Found {len(dcm_dict["files"])} valid DICOM files. '
                     'Proceed grouping these files into subfolders '
                     f'{info_txt}')
                    )

                if proceed:
                    new_folders = []
                    new_filenames = []
                    dcm_files = dcm_dict['files']
                    max_val = len(dcm_files)
                    progress_modal = uir.ProgressModal(
                        "Building new subfolder names...", "Stop",
                        0, max_val, self)
                    if series_uid:
                        new_folders, dcm_files = self.sort_seriesUID(
                            dcm_files, progress_widget=progress_modal)
                    else:
                        for i, file in enumerate(dcm_files):
                            progress_modal.setValue(i)
                            pyd, _, _ = dcm.read_dcm(file.resolve())
                            name_parts = dcm.get_dcm_info_list(
                                pyd, tag_pattern, self.wid_rename_pattern.tag_infos,
                                prefix_separator='', suffix_separator='',
                                not_found_text='')
                            new_name = "_".join(name_parts)
                            new_name = valid_path(new_name, folder=True)
                            new_folders.append(file.parent / new_name)

                            if rename_files:
                                name_parts_file = dcm.get_dcm_info_list(
                                    pyd, tag_pattern2,
                                    self.wid_rename_pattern.tag_infos,
                                    prefix_separator='', suffix_separator='',
                                    not_found_text='')
                                new_name_file = "_".join(name_parts_file)
                                new_name_file = valid_path(new_name_file) + '.dcm'
                                new_filenames.append(new_folders[-1] / new_name_file)

                            if progress_modal.wasCanceled():
                                new_folders = []
                                break

                    progress_modal.reset()
                    if len(new_folders) > 0:
                        uniq_new_folders = list(set(new_folders))

                        proceed = messageboxes.proceed_question(
                            self,
                            ('Proceed sorting the files into '
                             f'{len(uniq_new_folders)} subfolders? '),
                            info_text=(
                                'Find suggested new folders in detailed text.'),
                            detailed_text='\n'.join(
                                [x.name for x in uniq_new_folders])
                            )
                        if proceed and series_uid is False:
                            for folder in uniq_new_folders:
                                idxs = get_all_matches(new_folders, folder)
                                files_in_folder = [new_filenames[i] for i in idxs]
                                uniq_names = self.get_uniq_filenames(files_in_folder)
                                for u, i in enumerate(idxs):
                                    new_filenames[i] = uniq_names[u]
                    else:
                        proceed = False

                    if proceed:
                        max_val = len(dcm_files)
                        progress_modal = uir.ProgressModal(
                            "Moving files into subfolders...", "Stop",
                            0, max_val, self)
                        for i, file in enumerate(dcm_files):
                            progress_modal.setValue(i)
                            new_folder = new_folders[i]
                            if new_folder.exists() is False:
                                try:
                                    os.mkdir(new_folder.resolve())
                                except (PermissionError, OSError) as e:
                                    errmsg.append(
                                        f'{new_folder.resolve()}: {e}')
                            if rename_files:
                                file_new_loc = new_filenames[i]
                            else:
                                file_new_loc = new_folder / file.name
                            try:
                                file.rename(file_new_loc)
                            except (PermissionError, OSError) as e:
                                errmsg.append(f'{file.resolve}: {e}')

                            if progress_modal.wasCanceled():
                                break
                        progress_modal.reset()

        if len(errmsg) > 0:
            dlg = messageboxes.MessageBoxWithDetails(
                self, title='Issues',
                msg='There were issues moving one or more files. See details.',
                details=errmsg, icon=QMessageBox.Warning)
            dlg.exec()

    def test_10_first(self):
        """Generate names for first 10 folders or files."""
        self.generate_names(limit=10)

    def generate_names(self, limit=0):
        """Generate names for folders or files."""
        progress_modal = uir.ProgressModal(
            "Searching for files...", "Stop", 0, 100, self, hide_cancel=True)
        self.valid_dict = dcm.find_all_valid_dcm_files(
            self.path.text(), parent_widget=self,
            progress_modal=progress_modal, grouped=True)
        progress_modal.reset()
        dcm_folders = self.valid_dict['folders']
        dcm_files = self.valid_dict['files']
        if len(dcm_folders) > 0:
            start_with_folders = True
            if len(dcm_folders) == 1 and dcm_folders[0] == Path(
                    self.path.text()):
                start_with_folders = False

            empty_tag_pattern = False
            if start_with_folders:
                self.original_names = dcm_folders
                self.new_names = []
                tag_pattern = cfc.TagPatternFormat(
                    list_tags=self.wid_rename_pattern.current_template.list_tags,
                    list_format=self.wid_rename_pattern.current_template.list_format)
                if len(tag_pattern.list_tags) > 0:
                    if limit > 0:
                        if len(dcm_folders) > limit:
                            dcm_folders = dcm_folders[:limit]
                    progress_modal.setRange(0, len(dcm_folders))
                    progress_modal.setLabelText('Generate new folder names...')
                    for i, folder in enumerate(dcm_folders):
                        progress_modal.setValue(i)
                        pyd, _, _ = dcm.read_dcm(dcm_files[i][0].resolve())
                        if pyd:
                            name_parts = dcm.get_dcm_info_list(
                                pyd, tag_pattern, self.wid_rename_pattern.tag_infos,
                                prefix_separator='', suffix_separator='',
                                not_found_text='')
                            new_name = "_".join(name_parts)
                            new_name = valid_path(new_name, folder=True)
                            self.new_names.append(folder.parent / new_name)
                        else:
                            self.new_names.append('')
                    idx_all_this = get_all_matches(self.new_names, '')
                    if len(idx_all_this):
                        for i in idx_all_this.reverse():
                            self.original_names.pop(i)
                            self.new_names.pop(i)
                    progress_modal.reset()
                else:
                    empty_tag_pattern = True

            else:  # no subfolders, only files directly in search path
                tag_pattern = cfc.TagPatternFormat(
                    list_tags=self.wid_rename_pattern.current_template.list_tags2,
                    list_format=self.wid_rename_pattern.current_template.list_format2)
                if len(tag_pattern.list_tags) > 0:
                    dcm_files = dcm_files[0]
                    self.original_names = dcm_files
                    self.new_names = []
                    if limit > 0:
                        if len(dcm_files) > limit:
                            dcm_files = dcm_files[:limit]
                    progress_modal.setRange(0, len(dcm_files))
                    progress_modal.setLabelText('Generate new file names...')
                    for i, file in enumerate(dcm_files):
                        progress_modal.setValue(i)
                        pyd, _, errmsg = dcm.read_dcm(file.resolve())
                        if pyd:
                            name_parts = dcm.get_dcm_info_list(
                                pyd, tag_pattern, self.wid_rename_pattern.tag_infos,
                                prefix_separator='', suffix_separator='',
                                not_found_text='')
                            new_name = "_".join(name_parts)
                            new_name = valid_path(new_name) + '.dcm'
                            self.new_names.append(file.parent / new_name)
                        else:
                            self.new_names.append('')
                            print(errmsg)
                    idx_all_this = get_all_matches(self.new_names, '')
                    if len(idx_all_this):
                        for i in idx_all_this.reverse():
                            self.original_names.pop(i)
                            self.new_names.pop(i)
                    progress_modal.reset()
                else:
                    empty_tag_pattern = True

            if empty_tag_pattern:
                QMessageBox.warning(
                    self, 'Empty tag pattern',
                    ('Tag pattern is empty. Tag pattern is '
                     'needed to generate names.'))

            if len(self.new_names) > 0:  # ensure unique names
                self.new_names = self.get_uniq_filenames(self.new_names)

            self.fill_table()
            if limit == 0 and len(self.new_names) > 0:
                self.btn_rename.setDisabled(False)

    def rename(self):
        """Rename folders or files."""
        errmsg = []
        confirm_msgs = []
        proceed = False
        if len(self.original_names) > 0:
            if len(self.original_names) == len(self.new_names):
                proceed = True

        type_name = 'folder(s)' if self.new_names[0].is_dir() else 'file(s)'
        n_first = 0
        if proceed:  # rename first step
            max_val = len(self.original_names)
            progress_modal = uir.ProgressModal(
                "Renaming ...", "Stop", 0, max_val, self)
            for i, path in enumerate(self.original_names):
                progress_modal.setValue(i)
                try:
                    path.rename(self.new_names[i])
                    n_first += 1
                except (PermissionError, OSError) as err:
                    errmsg.append(f'{path.resolve}: {err}')
                if progress_modal.wasCanceled():
                    break
            progress_modal.reset()
        if n_first > 0:
            confirm_msgs.append(f'Renamed {n_first} {type_name} out of '
                                f'{len(self.original_names)}')

        tag_pattern = cfc.TagPatternFormat(
            list_tags=self.wid_rename_pattern.current_template.list_tags2,
            list_format=self.wid_rename_pattern.current_template.list_format2)
        if (proceed and self.new_names[0].is_dir()
                and len(tag_pattern.list_tags) > 0):
            proceed = False
            if self.valid_dict['folders'] == self.original_names:
                proceed = messageboxes.proceed_question(
                    self,
                    'Proceed renaming files?')
        else:
            proceed = False

        if proceed:  # rename files in subfolders
            max_val = sum(len(li) for li in self.valid_dict['files'])
            progress_modal = uir.ProgressModal(
                "Renaming files in subfolders...", "Stop",
                0, max_val, self)

            counter = 0
            n_renamed = 0
            for folder_no, file_list in enumerate(self.valid_dict['files']):
                new_folder = self.new_names[folder_no]
                if new_folder.exists():
                    for file in file_list:
                        progress_modal.setValue(counter)
                        file_new_loc = new_folder / file.name
                        pyd, _, pyd_err = dcm.read_dcm(file_new_loc.resolve())
                        if pyd:
                            name_parts = dcm.get_dcm_info_list(
                                pyd, tag_pattern, self.wid_rename_pattern.tag_infos,
                                prefix_separator='', suffix_separator='',
                                not_found_text='')
                            new_name = "_".join(name_parts)
                            new_name = valid_path(new_name) + '.dcm'
                            try:
                                file_new_loc.rename(
                                    file_new_loc.parent / new_name)
                                n_renamed += 1
                            except (PermissionError, OSError) as err:
                                errmsg.append(f'{file.resolve}: {err}')
                        else:
                            errmsg.append(pyd_err)
                        if progress_modal.wasCanceled():
                            break
                        counter += 1
                if progress_modal.wasCanceled():
                    break
            progress_modal.reset()
            confirm_msgs.append(f'Renamed {n_renamed} files(s) out of {max_val}')

        if len(errmsg) > 0:
            dlg = messageboxes.MessageBoxWithDetails(
                self, title='Failed renaming',
                msg='Failed renaming one or more files. See details',
                details=errmsg, icon=QMessageBox.Warning)
            dlg.exec()
        if len(confirm_msgs) > 0:
            dlg = messageboxes.MessageBoxWithDetails(
                self, title='Finished renaming',
                msg='Finished renaming:',
                info='\n'.join(confirm_msgs), icon=QMessageBox.Information)
            dlg.exec()

    def sort_seriesUID(self, dcm_files, progress_widget=None):
        """Sort images based on series UID.

        Parameters
        ----------
        dcm_files : list of str
            dicom addresses to sort
        progress_widget : ProgressModal, Optional
            default is None

        Returns
        -------
        folder_names: list str
            series description + XXX if not uniq, for each file
        sorted_dcm_files: list of list of str
            all filepaths as nested list with one list for each series
        """
        uids = []
        series_descr = []
        series_descr_modified = []
        sorted_dcm_files = []

        tag_pattern = cfc.TagPatternFormat(
            list_tags=['SeriesInstanceUID', 'SeriesDescription'])

        proceed = True
        for i, file in enumerate(dcm_files):
            progress_widget.setValue(i)
            pyd, _, _ = dcm.read_dcm(file.resolve())
            info_list = dcm.get_dcm_info_list(
                pyd,
                tag_pattern, self.wid_rename_pattern.tag_infos,
                prefix_separator='', suffix_separator='',
                not_found_text='')
            uids.append(info_list[0])
            series_descr.append(info_list[1])

            if progress_widget.wasCanceled():
                proceed = False
                break

        if proceed:
            uniq_uids = list(set(uids))
            folder_names = []
            for uid in uniq_uids:
                idx_this = uids.index(uid)
                ser_descr_this = valid_path(series_descr[idx_this], folder=True)
                if ser_descr_this in folder_names:
                    n_already = folder_names.count(ser_descr_this)
                    ser_descr_this_mod = f'{ser_descr_this}_{n_already:03}'
                else:
                    ser_descr_this_mod = ser_descr_this
                series_descr_modified.append(ser_descr_this_mod)
                idx_all_this = get_all_matches(uids, uid)
                files_this = [dcm_files[i] for i in idx_all_this]
                sorted_dcm_files.extend(files_this)
                folder_names.extend(
                    [dcm_files[0].parent / ser_descr_this] * len(files_this))

        return (folder_names, sorted_dcm_files)

    def get_uniq_filenames(self, names):
        """Add suffix if some names are identical.

        Parameters
        ----------
        names : list of Path

        Returns
        -------
        names
        """
        uniq_names = list(set(names))
        for name in uniq_names:
            n_same = names.count(name)
            if n_same > 1:
                idxs = get_all_matches(names, name)
                base_name = str(name.parent / name.stem)
                for i, idx in enumerate(idxs):
                    new_name = f'{base_name}_({i:03}).dcm'
                    names[idx] = Path(new_name)
        return names
