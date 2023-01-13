#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface for dialog Rename Dicom.

@author: Ellen Wasbo
"""
import os
from pathlib import Path

import pydicom
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QDialog, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QToolBar,
    QLabel, QLineEdit, QPushButton, QAction,
    QTreeWidget, QTreeWidgetItem, QFileDialog,
    QMessageBox
    )

# imageQC block start
import imageQC.ui.reusables as uir
from imageQC.ui.settings import RenamePatternWidget
import imageQC.config.config_classes as cfc
from imageQC.config.iQCconstants import ENV_ICON_PATH
from imageQC.scripts.dcm import find_all_valid_dcm_files, get_dcm_info_list
from imageQC.scripts.mini_methods_format import (
    valid_path, generate_uniq_filepath)
# imageQC block end


class RenameDicomDialog(QDialog):
    """GUI setup for the Rename Dicom dialog window."""

    def __init__(self, open_filepaths=[], initial_modality='CT', tag_infos=None):
        super().__init__()

        self.start_files = open_filepaths
        self.current_modality = initial_modality

        self.orignal_names = []  # list for found files or folders (first step)
        self.new_names = []  # list for generated new names (first step)
        self.valid_dict = {}  # dict for folders (first) + files (second step)

        self.setWindowTitle('Rename DICOM')
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        splitter = QSplitter(Qt.Vertical)
        vLO = QVBoxLayout()
        self.setLayout(vLO)
        vLO.addWidget(splitter)

        subtxt = '''Rename DICOM files and folders by DICOM header data.
        Subfolders will be renamed by the first file found in the folder.<br>
        Note also option to split files into subfolders based on
        DICOM header or gather all DICOM files into one folder.<br>
        The split series will use seriesUID if subfolder template is not set.'''
        self.wTagPattern = RenamePatternWidget(
            initial_modality=self.current_modality, header='', subtxt=subtxt)
        splitter.addWidget(self.wTagPattern)

        wBtm = QWidget()
        splitter.addWidget(wBtm)
        vLO2 = QVBoxLayout()
        wBtm.setLayout(vLO2)
        hLObrowse = QHBoxLayout()
        vLO2.addLayout(hLObrowse)
        lbl = QLabel('Selected folder: ')
        hLObrowse.addWidget(lbl)
        self.path = QLineEdit()
        hLObrowse.addWidget(self.path)
        tbBrowse = uir.ToolBarBrowse(
            'Browse for folder with DICOM files', clear=True)
        tbBrowse.actBrowse.triggered.connect(self.browse)
        tbBrowse.actClear.triggered.connect(self.clear_path)
        hLObrowse.addWidget(tbBrowse)

        lbl = uir.LabelItalic(
            """If selected folder is left blank, open images
            in main window will be subject to renaming.""")
        vLO2.addWidget(lbl)

        tbSplitGather = QToolBar()
        hLObrowse.addWidget(tbSplitGather)
        actSplit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}split.png'),
            'Move files into folders according to subfolder template or seriesUID',
            self)
        actSplit.triggered.connect(self.split_series)
        '''Cannot use same icon twice with different meaning (open.png)
        actOpen = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}open.png'),
            'Open explorer for selected path to inspect content', self)
        actOpen.triggered.connect(self.open_path)'''
        actGather = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}gather.png'),
            'Gather all DICOM files in subfolders to selected folder', self)
        actGather.triggered.connect(self.gather_files)
        tbSplitGather.addActions([actSplit, actGather])

        hLOtable = QHBoxLayout()
        vLO2.addLayout(hLOtable)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(
            ['Original name', 'Suggested name'])
        self.table.setRootIsDecorated(False)
        self.table.setColumnWidth(0, 500)
        self.table.setColumnWidth(1, 500)
        hLOtable.addWidget(self.table)
        vLObtnsTable = QVBoxLayout()
        hLOtable.addLayout(vLObtnsTable)
        btnTest10 = QPushButton('Test 10 first')
        btnTest10.clicked.connect(self.test_10_first)
        vLObtnsTable.addWidget(btnTest10)
        btnGenerateNames = QPushButton('Prepare names')
        btnGenerateNames.clicked.connect(self.generate_names)
        vLObtnsTable.addWidget(btnGenerateNames)
        self.btnRename = QPushButton('Rename')
        self.btnRename.setDisabled(True)
        self.btnRename.clicked.connect(self.rename)
        vLObtnsTable.addWidget(self.btnRename)
        vLObtnsTable.addStretch()

        hLOdlgBtns = QHBoxLayout()
        vLO.addLayout(hLOdlgBtns)
        hLOdlgBtns.addStretch()
        btnClose = QPushButton('Close window')
        btnClose.clicked.connect(self.accept)
        hLOdlgBtns.addWidget(btnClose)

        self.statusBar = uir.StatusBar(self)

        self.wTagPattern.update_from_yaml()

    def reset_names(self):
        """Reset variables/GUI if generated names no longer valid."""
        self.orignal_names = []
        self.new_names = []
        self.table.clear()
        self.btnRename.setDisabled(True)

    def fill_table(self):
        """Fill table with original and generated names."""
        self.table.clear()
        if len(self.original_names) > 0 and len(self.new_names) > 0:
            orignames = [x.name for x in self.original_names]
            newnames = [x.name for x in self.new_names]

            for i, f in enumerate(orignames):
                row_strings = [f, newnames[i]]
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
        dcm_dict = find_all_valid_dcm_files(
            self.path.text(), parent_widget=self, grouped=False)
        if len(dcm_dict['files']) > 0:
            # check if parent is the specified folder? Else no need to move
            proceed = True
            if (len(dcm_dict['folders']) == 1
                    and dcm_dict['folders'][0] == self.path.text()):
                proceed = False

            if proceed:
                proceed = uir.proceed_question(
                    self,
                    f'Found {len(dcm_dict["files"])} valid DICOM files. '
                    'Proceed moving these files directly into the '
                    'specified folder?')

            if proceed:
                count_renamed = 0
                failed_paths = []
                for file in dcm_dict['files']:
                    p = Path(file)
                    new_file = Path(self.path.text()) / p.name
                    new_file_str = generate_uniq_filepath(new_file.resolve())
                    if new_file_str != '':
                        try:
                            p.rename(new_file_str)
                            count_renamed += 1
                        except FileExistsError as e:
                            failed_paths.append(file.resolve())
                            print(f'Faile renameing {file} to {new_file_str}/n{e}')
                    else:
                        failed_paths.append(file.resolve())
                self.reset_names()
                if count_renamed != len(dcm_dict['files']):
                    msg = QMessageBox.warning(
                        self, "Issues",
                        (f"Moved {count_renamed} of {len(dcm_dict['files'])}."
                         "See details for paths failing."))
                    msg.setDetailedText('\n'.join(failed_paths))
                    msg.exec()

                proceed = uir.proceed_question(
                    self,
                    'Remove empty folders?')

            if proceed:
                for root, dirs, files in os.walk(
                        self.path.text(), topdown=False):
                    for name in dirs:
                        if len(os.listdir(os.path.join(root, name))) == 0:
                            os.rmdir(os.path.join(root, name))
                        else:
                            pass

    def split_series(self):
        """Split series into subfolders based on subfolder tag pattern."""
        errmsg = []
        serUID = False
        if len(self.wTagPattern.current_template.list_tags) == 0:
            serUID = True
        else:
            tag_pattern = cfc.TagPatternFormat(
                list_tags=self.wTagPattern.current_template.list_tags,
                list_format=self.wTagPattern.current_template.list_format)

        proceed = True
        if os.access(self.path.text(), os.W_OK) is False:
            proceed = False
            errmsg = ['No writing permissing for given path.']

        if proceed:
            dcm_dict = find_all_valid_dcm_files(
                self.path.text(), parent_widget=self,
                grouped=False, search_subfolders=False)
            if len(dcm_dict['files']) > 0:

                if serUID:
                    info_txt = 'if they share series UID?'
                else:
                    info_txt = (
                        'if they share the same parameters defined in '
                        'the subfolder pattern?')
                proceed = uir.proceed_question(
                    self,
                    (f'Found {len(dcm_dict["files"])} valid DICOM files. '
                     'Proceed grouping these files into subfolders '
                     f'{info_txt}')
                    )

                if proceed:
                    new_folders = []
                    dcm_files = dcm_dict['files']
                    maxVal = len(dcm_files)
                    progress = uir.ProgressModal(
                        "Building new subfolder names...", "Stop",
                        0, maxVal, self)
                    if serUID:
                        new_folders, sorted_dcm_files = self.sort_seriesUID(
                            dcm_files, progress_widget=progress)
                    else:
                        for i, file in enumerate(dcm_files):
                            progress.setValue(i)
                            pd = pydicom.dcmread(
                                dcm_files[i].resolve(), stop_before_pixels=True)
                            name_parts = get_dcm_info_list(
                                pd, tag_pattern, self.wTagPattern.tag_infos,
                                prefix_separator='', suffix_separator='',
                                not_found_text='')
                            new_name = "_".join(name_parts)
                            new_name = valid_path(new_name, folder=True)
                            new_folders.append(file.parent / new_name)

                            if progress.wasCanceled():
                                new_folders = []
                                break

                    progress.setValue(maxVal)
                    if len(new_folders) > 0:
                        uniq_new_folders = list(set(new_folders))

                        proceed = uir.proceed_question(
                            self,
                            ('Proceed sorting the files into '
                             f'{len(uniq_new_folders)} subfolders? '),
                            info_text=(
                                'Find suggested new folders in detailed text.'),
                            detailed_text='\n'.join(
                                [x.name for x in uniq_new_folders])
                            )
                    else:
                        proceed = False

                    if proceed:
                        maxVal = len(dcm_files)
                        progress = uir.ProgressModal(
                            "Moving files into subfolders...", "Stop",
                            0, maxVal, self)
                        for i, file in enumerate(dcm_files):
                            progress.setValue(i)
                            new_folder = new_folders[i]
                            if new_folder.exists() is False:
                                try:
                                    os.mkdir(new_folder.resolve())
                                except (PermissionError, OSError) as e:
                                    errmsg.append(
                                        f'{new_folder.resolve()}: {e}')
                            file_new_loc = new_folder / file.name
                            try:
                                file.rename(file_new_loc)
                            except (PermissionError, OSError) as e:
                                errmsg.append(f'{file.resolve}: {e}')

                            if progress.wasCanceled():
                                break
                        progress.setValue(maxVal)

        if len(errmsg) > 0:
            msg = QMessageBox(
                QMessageBox.Warning,
                'Issues',
                'There were issues moving one or more files. See details.',
                parent=self)
            msg.setDetailedText('\n'.join(errmsg))
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()

    def test_10_first(self):
        """Generate names for first 10 folders or files."""
        self.generate_names(limit=10)

    def generate_names(self, limit=0):
        """Generate names for folders or files."""
        self.valid_dict = find_all_valid_dcm_files(
            self.path.text(), parent_widget=self, grouped=True)
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
                    list_tags=self.wTagPattern.current_template.list_tags,
                    list_format=self.wTagPattern.current_template.list_format)
                if len(tag_pattern.list_tags) > 0:
                    if limit > 0:
                        if len(dcm_folders) > limit:
                            dcm_folders = dcm_folders[:limit]
                    for i, folder in enumerate(dcm_folders):
                        pd = pydicom.dcmread(
                            dcm_files[i][0].resolve(), stop_before_pixels=True)
                        name_parts = get_dcm_info_list(
                            pd, tag_pattern, self.wTagPattern.tag_infos,
                            prefix_separator='', suffix_separator='',
                            not_found_text='')
                        new_name = "_".join(name_parts)
                        new_name = valid_path(new_name, folder=True)
                        self.new_names.append(folder.parent / new_name)
                else:
                    empty_tag_pattern = True

            else:  # no subfolders, only files directly in search path
                tag_pattern = cfc.TagPatternFormat(
                    list_tags=self.wTagPattern.current_template.list_tags2,
                    list_format=self.wTagPattern.current_template.list_format2)
                if len(tag_pattern.list_tags) > 0:
                    dcm_files = dcm_files[0]
                    self.original_names = dcm_files
                    self.new_names = []
                    if limit > 0:
                        if len(dcm_files) > limit:
                            dcm_files = dcm_files[:limit]
                    for file in dcm_files:
                        pd = pydicom.dcmread(
                            file.resolve(), stop_before_pixels=True)
                        name_parts = get_dcm_info_list(
                            pd, tag_pattern, self.wTagPattern.tag_infos,
                            prefix_separator='', suffix_separator='',
                            not_found_text='')
                        new_name = "_".join(name_parts)
                        new_name = valid_path(new_name) + '.dcm'
                        self.new_names.append(file.parent / new_name)
                else:
                    empty_tag_pattern = True

            if empty_tag_pattern:
                QMessageBox.warning(
                    self, 'Empty tag pattern',
                    ('Tag pattern is empty. Tag pattern is '
                     'needed to generate names.'))

            self.fill_table()
            if limit == 0 and len(self.new_names) > 0:
                self.btnRename.setDisabled(False)

    def rename(self):
        """Rename folders or files."""
        errmsg = []
        proceed = False
        if len(self.original_names) > 0:
            if len(self.original_names) == len(self.new_names):
                proceed = True

        if proceed:  # rename first step
            maxVal = len(self.original_names)
            progress = uir.ProgressModal(
                "Renaming ...", "Stop", 0, maxVal, self)
            for i, path in enumerate(self.original_names):
                progress.setValue(i)
                try:
                    path.rename(self.new_names[i])
                except (PermissionError, OSError) as e:
                    errmsg.append(f'{path.resolve}: {e}')
                if progress.wasCanceled():
                    break
            progress.setValue(maxVal)

        tag_pattern = cfc.TagPatternFormat(
            list_tags=self.wTagPattern.current_template.list_tags2,
            list_format=self.wTagPattern.current_template.list_format2)
        if (proceed and self.new_names[0].is_dir()
                and len(tag_pattern.list_tags) > 0):
            proceed = False
            if self.valid_dict['folders'] == self.original_names:
                proceed = uir.proceed_question(
                    self,
                    'Proceed renaming files?')
        else:
            proceed = False

        if proceed:  # rename files in subfolders
            maxVal = sum([len(li) for li in self.valid_dict['files']])
            progress = uir.ProgressModal(
                "Renaming files in subfolders...", "Stop",
                0, maxVal, self)

            counter = 0
            for folderNo, li in enumerate(self.valid_dict['files']):
                new_folder = self.new_names[folderNo]
                if new_folder.exists():
                    for file in li:
                        progress.setValue(counter)
                        file_new_loc = new_folder / file.name
                        try:
                            pd = pydicom.dcmread(
                                file_new_loc.resolve(),
                                stop_before_pixels=True)
                            name_parts = get_dcm_info_list(
                                pd, tag_pattern, self.wTagPattern.tag_infos,
                                prefix_separator='', suffix_separator='',
                                not_found_text='')
                            new_name = "_".join(name_parts)
                            new_name = valid_path(new_name) + '.dcm'
                            file_new_loc.rename(
                                file_new_loc.parent / new_name)
                        except pydicom.errors.InvalidDicomError:
                            pass  # should be validated already
                        except (PermissionError, OSError) as e:
                            errmsg.append(f'{path.resolve}: {e}')
                        if progress.wasCanceled():
                            break
                        counter += 1
                if progress.wasCanceled():
                    break
            progress.setValue(maxVal)

        if len(errmsg) > 0:
            msg = QMessageBox.warning(
                self, 'Failed renaming',
                'Failed renaming one or more files. Writing permission?')
            msg.setDetailedText('\n'.join(errmsg))
            msg.exec()

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
            list_tags=['SeriesInstanceUID', 'SeriesDescription'],
            list_format=['', ''])

        proceed = True
        for i, file in enumerate(dcm_files):
            progress_widget.setValue(i)
            pd = pydicom.dcmread(
                file.resolve(), stop_before_pixels=True)
            info_list = get_dcm_info_list(
                pd,
                tag_pattern, self.wTagPattern.tag_infos,
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
                ser_descr_this = series_descr[idx_this[0]]
                if ser_descr_this in folder_names:
                    n_already = folder_names.count(ser_descr_this)
                    series_descr_modified[idx_this] = (
                        f'{ser_descr_this}_{n_already:03}')
                else:
                    series_descr_modified[idx_this] = ser_descr_this
                folder_names.append(ser_descr_this)
                sorted_dcm_files.append(dcm_files[idx_this])

        return (folder_names, sorted_dcm_files)
