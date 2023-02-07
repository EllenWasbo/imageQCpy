#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface for open advanced option.

@author: Ellen Wasbo
"""
import os
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QButtonGroup,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QListWidget, QRadioButton, QAbstractItemView,
    QFileDialog, QDialogButtonBox,
    )

# imageQC block start
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.ui import reusable_widgets as uir
from imageQC.scripts.dcm import find_all_valid_dcm_files, read_dcm_info
# imageQC block end


class OpenMultiDialog(ImageQCDialog):
    """GUI for opening images using selection rules.

    Reads all images also in subfolder and sort these into series.
    Select which files to open by selecting in list or use selection patterns
    for consistency with repetitive tasks.
    """

    def __init__(self, main):
        super().__init__()
        self.setWindowTitle('Select images to open')
        self.main = main
        self.imgs = []
        self.open_imgs = []

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        hlo_browse = QHBoxLayout()
        vlo.addLayout(hlo_browse)
        lbl = QLabel('Selected folder: ')
        hlo_browse.addWidget(lbl)
        self.path = QLineEdit()
        hlo_browse.addWidget(self.path)
        toolb = uir.ToolBarBrowse('Browse to folder with DICOM images')
        toolb.act_browse.triggered.connect(self.browse)
        hlo_browse.addWidget(toolb)
        info_text = (
            'All DICOM files in the selected folder listed as series '
            'according to series number + series description.<br>'
            'Manually select and push images to open or use the selection '
            'rules.<br>'
            'Images will be displayed in the list as defined in Settings - '
            'Special tag patterns - File list display.'
        )
        vlo.addWidget(uir.LabelItalic(info_text))

        vlo.addWidget(uir.HLine())

        hlo_lists = QHBoxLayout()
        vlo.addLayout(hlo_lists)

        min_height = 200
        min_width = 400

        vlo_series = QVBoxLayout()
        hlo_lists.addLayout(vlo_series)
        lbl_series = uir.LabelHeader('Series', 4)
        lbl_series.setAlignment(Qt.AlignCenter)
        vlo_series.addWidget(lbl_series)
        self.list_series = QListWidget()
        self.list_series.setMinimumHeight(min_height)
        self.list_series.setMinimumWidth(min_width)
        self.list_series.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_series.currentRowChanged.connect(self.update_list_images)
        vlo_series.addWidget(self.list_series)
        vlo_series.addStretch()

        vlo_images = QVBoxLayout()
        hlo_lists.addLayout(vlo_images)
        lbl_images = uir.LabelHeader('Images', 4)
        lbl_images.setAlignment(Qt.AlignCenter)
        vlo_images.addWidget(lbl_images)
        self.list_images = QListWidget()
        self.list_images.setMinimumHeight(min_height)
        self.list_images.setSelectionMode(QAbstractItemView.ExtendedSelection)
        vlo_images.addWidget(self.list_images)

        gb_select_pattern = QGroupBox('Selection pattern')
        gb_select_pattern.setFont(uir.FontItalic())
        vlo_images.addWidget(gb_select_pattern)
        vlo_gb = QVBoxLayout()
        gb_select_pattern.setLayout(vlo_gb)
        self.bgrSelPattern = QButtonGroup()
        lbls = ['Select all', 'Select the ', 'Select the ']
        rb_select_pattern = []
        for i, lbl in enumerate(lbls):
            rb_select_pattern.append(QRadioButton(lbl))
            self.bgrSelPattern.addButton(rb_select_pattern[i], i)
        vlo_gb.addWidget(rb_select_pattern[0])
        hlo_sel1 = QHBoxLayout()
        hlo_sel1.addWidget(rb_select_pattern[1])
        self.spin_n_close = QSpinBox()
        self.spin_n_close.setMinimum(1)
        hlo_sel1.addWidget(self.spin_n_close)
        hlo_sel1.addWidget(QLabel('image(s) closest to zpos = '))
        self.spin_z_close = QDoubleSpinBox()
        self.spin_z_close.setDecimals(1)
        self.spin_z_close.setMinimum(-9999)
        self.spin_z_close.setMaximum(9999)
        hlo_sel1.addWidget(self.spin_z_close)
        vlo_gb.addLayout(hlo_sel1)

        hlo_sel2 = QHBoxLayout()
        hlo_sel2.addWidget(rb_select_pattern[2])
        self.spin_n_pos = QSpinBox()#TODO not in use?
        self.spin_n_pos.setMinimum(1)
        hlo_sel2.addWidget(self.spin_n_pos)
        self.bgr_select_pos = QButtonGroup()#TODO not in use?
        lbls = ['first', 'mid', 'last']
        rb_select_pos = []
        for i, lbl in enumerate(lbls):
            rb_select_pos.append(QRadioButton(lbl))
            self.bgr_select_pos.addButton(rb_select_pos[i], i)
            hlo_sel2.addWidget(rb_select_pos[i])
        hlo_sel2.addWidget(QLabel('image(s)'))
        vlo_gb.addLayout(hlo_sel2)

        btn_test_pattern = QPushButton(
            'Test pattern on current series')
        btn_test_pattern.clicked.connect(self.test_pattern)
        vlo_gb.addWidget(btn_test_pattern)
        btn_push_pattern = QPushButton(
            'Add images to list using pattern >>')
        btn_push_pattern.clicked.connect(self.push_pattern)
        vlo_gb.addWidget(btn_push_pattern)

        vlo_push = QVBoxLayout()
        hlo_lists.addLayout(vlo_push)
        btn_push = QPushButton('>>')
        btn_push.setToolTip('Add selected images to list of images to open')
        btn_push.clicked.connect(self.push_selected)
        vlo_push.addSpacing(100)
        vlo_push.addWidget(btn_push)
        vlo_push.addStretch()

        vlo_open = QVBoxLayout()
        hlo_lists.addLayout(vlo_open)
        lbl_open = uir.LabelHeader('Images to open', 4)
        lbl_open.setAlignment(Qt.AlignCenter)
        vlo_open.addWidget(lbl_open)
        self.list_open_images = QListWidget()
        self.list_open_images.setMinimumHeight(min_height)
        self.list_open_images.setMinimumWidth(min_width)
        self.list_open_images.setSelectionMode(QAbstractItemView.ExtendedSelection)
        vlo_open.addWidget(self.list_open_images)
        hlo_btns = QHBoxLayout()
        btn_remove_selected = QPushButton('Remove selected')
        btn_remove_selected.clicked.connect(self.remove_selected)
        hlo_btns.addWidget(btn_remove_selected)
        btn_clear_list = QPushButton('Clear list')
        btn_clear_list.clicked.connect(self.clear_list)
        hlo_btns.addWidget(btn_clear_list)
        vlo_open.addLayout(hlo_btns)
        vlo_open.addStretch()

        hlo_dlg_btns = QHBoxLayout()
        vlo.addLayout(hlo_dlg_btns)
        hlo_dlg_btns.addStretch()
        btns = QDialogButtonBox()
        btns.setOrientation(Qt.Horizontal)
        btns.setStandardButtons(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText('Open selected')
        btns.button(QDialogButtonBox.Ok).clicked.connect(self.accept)
        btns.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)
        hlo_dlg_btns.addWidget(btns)

    def browse(self):
        """Browse to set selected folder and start searching for images."""
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec():
            fname = dlg.selectedFiles()
            self.path.setText(os.path.normpath(fname[0]))
            self.find_all_dcm()

    def find_all_dcm(self):
        """Find all DICOM files and list these according to the pattern."""
        path = self.path.text()
        if path != '':
            self.imgs = []
            self.open_imgs = []
            dcm_dict = find_all_valid_dcm_files(
                path, parent_widget=self, grouped=False)
            if len(dcm_dict['files']) > 0:
                dcm_files = dcm_dict['files']
                img_infos, ignored_files = read_dcm_info(
                    dcm_files, tag_infos=self.main.tag_infos,
                    tag_patterns_special=self.main.tag_patterns_special)
                series_nmb_name = []
                self.imgs = []
                for img in img_infos:
                    ser = ' '.join(img.series_list_strings)
                    if ser in series_nmb_name:
                        serno = series_nmb_name.index(ser)
                    else:
                        serno = len(series_nmb_name)
                        series_nmb_name.append(ser)
                        self.imgs.append([])
                    self.imgs[serno].append(img)

                self.list_series.clear()
                self.list_series.addItems(series_nmb_name)
                self.list_series.setCurrentRow(0)
                self.update_list_images(0)

    def update_list_images(self, serno):
        """Fill list_images with all images in selected series."""
        if serno >= 0:
            img_strings = [
                ' '.join(img.file_list_strings) for img in self.imgs[serno]]
            self.list_images.clear()
            self.list_images.addItems(img_strings)

    def update_list_open_images(self):
        """Fill list_open_images with all images to be opened."""
        self.list_open_images.clear()
        if len(self.open_imgs) > 0:
            img_strings = [
                ' '.join(img.file_list_strings) for img in self.open_imgs]
            self.list_open_images.addItems(img_strings)

    def use_pattern(self, push=False):
        """Find which to mark. If push is False only visualize selected.

        Parameters
        ----------
        push : bool, optional
            push images to open_images. The default is False.
        """
        sel_series = self.list_series.selectedIndexes()
        btn_id = self.bgrSelPattern.checkedId()
        set_selected_ids = []
        if btn_id == 0:  # select all
            if push:
                for sel in sel_series:
                    self.open_imgs.extend(self.imgs[sel.row()])
            set_selected_ids = list(range(self.list_images.count()))
        elif btn_id == 1:  # select n images closest to zpos
            first = True
            for sel in sel_series:
                if first or push:
                    try:
                        zpos_arr = []
                        for img in self.imgs[sel.row()]:
                            zpos_arr.append(img.zpos)
                        if None not in zpos_arr:
                            n_img = self.spin_n_close.value()
                            zpos = self.spin_z_close.value()
                            diff = np.abs(np.asarray(zpos_arr) - zpos)
                            n_lowest = np.argsort(diff)[:n_img]
                            if push:
                                self.open_imgs.extend(
                                    self.imgs[sel.row()][n_lowest])
                            set_selected_ids = n_lowest
                    except AttributeError:
                        pass
                first = False
        else:
            pass
        for i in range(self.list_images.count()):
            if i in set_selected_ids:
                self.list_images.item(i).setSelected(True)
            else:
                self.list_images.item(i).setSelected(False)
        if push:
            self.update_list_open_images()

    def test_pattern(self):
        """Highlight images in list_images that fit the pattern."""
        self.use_pattern()

    def push_pattern(self):
        """Add images according to defined pattern to the list_open_images."""
        self.use_pattern(push=True)

    def push_selected(self):
        """Add selected images in list_images to list_open_images."""
        sels = self.list_images.selectedIndexes()
        ser = self.list_series.selectedIndexes()
        if len(ser) > 0:
            serno = ser[0].row()
            if len(sels) > 0:
                for sel in sels:
                    imgno = sel.row()
                    self.open_imgs.append(self.imgs[serno][imgno])
                self.update_list_open_images()

    def remove_selected(self):
        """Remove the selected images of list_open_images."""
        sels = self.list_images.selectedIndexes()
        if len(sels) > 0:
            img_nmbs = [sel.row() for sel in sels]
            img_nmbs.sort(reverse=True)
            for i in img_nmbs:
                self.open_imgs.pop(i)
            self.update_list_open_images()

    def clear_list(self):
        """Clear list_open_images."""
        self.open_imgs = []
        self.list_open_images.clear()
