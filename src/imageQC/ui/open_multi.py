#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface for open advanced option.

@author: Ellen Wasbo
"""
import os
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QButtonGroup,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QListWidget, QRadioButton, QAction, QAbstractItemView,
    QFileDialog, QDialogButtonBox,
)

from imageQC.config.iQCconstants import ENV_ICON_PATH
from imageQC.ui.reusables import (
    FontItalic, LabelItalic, LabelHeader, HLine, ToolBarBrowse)
from imageQC.scripts.dcm import find_all_valid_dcm_files, read_dcm_info


class OpenMultiDialog(QDialog):
    """GUI for opening images using selection rules.

    Reads all images also in subfolder and sort these into series.
    Select which files to open by selecting in list or use selection patterns
    for consistency with repetitive tasks.
    """

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.imgs = []
        self.open_imgs = []

        self.setWindowTitle('Select images to open')
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        vLO = QVBoxLayout()
        self.setLayout(vLO)

        hLObrowse = QHBoxLayout()
        vLO.addLayout(hLObrowse)
        lbl = QLabel('Selected folder: ')
        hLObrowse.addWidget(lbl)
        self.path = QLineEdit()
        hLObrowse.addWidget(self.path)
        tb = ToolBarBrowse('Browse to folder with DICOM images')
        tb.actBrowse.triggered.connect(self.browse)
        hLObrowse.addWidget(tb)
        info_text = (
            'All DICOM files in the selected folder listed as series '
            'according to series number + series description.<br>'
            'Manually select and push images to open or use the selection '
            'rules.<br>'
            'Images will be displayed in the list as defined in Settings - '
            'Special tag patterns - File list display.'
        )
        vLO.addWidget(LabelItalic(info_text))

        vLO.addWidget(HLine())

        hLOlists = QHBoxLayout()
        vLO.addLayout(hLOlists)

        minHeight = 200
        minWidth = 400

        vLOseries = QVBoxLayout()
        hLOlists.addLayout(vLOseries)
        lblSeries = LabelHeader('Series', 4)
        lblSeries.setAlignment(Qt.AlignCenter)
        vLOseries.addWidget(lblSeries)
        self.list_series = QListWidget()
        self.list_series.setMinimumHeight(minHeight)
        self.list_series.setMinimumWidth(minWidth)
        self.list_series.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_series.currentRowChanged.connect(self.update_list_images)
        vLOseries.addWidget(self.list_series)
        vLOseries.addStretch()

        vLOimages = QVBoxLayout()
        hLOlists.addLayout(vLOimages)
        lblimages = LabelHeader('Images', 4)
        lblimages.setAlignment(Qt.AlignCenter)
        vLOimages.addWidget(lblimages)
        self.list_images = QListWidget()
        self.list_images.setMinimumHeight(minHeight)
        self.list_images.setSelectionMode(QAbstractItemView.ExtendedSelection)
        vLOimages.addWidget(self.list_images)

        gbSelPattern = QGroupBox('Selection pattern')
        gbSelPattern.setFont(FontItalic())
        vLOimages.addWidget(gbSelPattern)
        vLOgb = QVBoxLayout()
        gbSelPattern.setLayout(vLOgb)
        self.bgrSelPattern = QButtonGroup()
        lbls = ['Select all', 'Select the ', 'Select the ']
        rbSelPattern = []
        for i, lbl in enumerate(lbls):
            rbSelPattern.append(QRadioButton(lbl))
            self.bgrSelPattern.addButton(rbSelPattern[i], i)
        vLOgb.addWidget(rbSelPattern[0])
        hLOsel1 = QHBoxLayout()
        hLOsel1.addWidget(rbSelPattern[1])
        self.spinNclose = QSpinBox()
        self.spinNclose.setMinimum(1)
        hLOsel1.addWidget(self.spinNclose)
        hLOsel1.addWidget(QLabel('image(s) closest to zpos = '))
        self.spinZclose = QDoubleSpinBox()
        self.spinZclose.setDecimals(1)
        self.spinZclose.setMinimum(-9999)
        self.spinZclose.setMaximum(9999)
        hLOsel1.addWidget(self.spinZclose)
        vLOgb.addLayout(hLOsel1)

        hLOsel2 = QHBoxLayout()
        hLOsel2.addWidget(rbSelPattern[2])
        self.spinNpos = QSpinBox()
        self.spinNpos.setMinimum(1)
        hLOsel2.addWidget(self.spinNpos)
        self.bgrSelPos = QButtonGroup()
        lbls = ['first', 'mid', 'last']
        rbSelPos = []
        for i, lbl in enumerate(lbls):
            rbSelPos.append(QRadioButton(lbl))
            self.bgrSelPos.addButton(rbSelPos[i], i)
            hLOsel2.addWidget(rbSelPos[i])
        hLOsel2.addWidget(QLabel('image(s)'))
        vLOgb.addLayout(hLOsel2)

        btnTestPattern = QPushButton(
            'Test pattern on current series')
        btnTestPattern.clicked.connect(self.test_pattern)
        vLOgb.addWidget(btnTestPattern)
        btnPushPattern = QPushButton(
            'Add images to list using pattern >>')
        btnPushPattern.clicked.connect(self.push_pattern)
        vLOgb.addWidget(btnPushPattern)

        vLOpush = QVBoxLayout()
        hLOlists.addLayout(vLOpush)
        btnPush = QPushButton('>>')
        btnPush.setToolTip('Add selected images to list of images to open')
        btnPush.clicked.connect(self.push_selected)
        vLOpush.addSpacing(100)
        vLOpush.addWidget(btnPush)
        vLOpush.addStretch()

        vLOopen = QVBoxLayout()
        hLOlists.addLayout(vLOopen)
        lblOpen = LabelHeader('Images to open', 4)
        lblOpen.setAlignment(Qt.AlignCenter)
        vLOopen.addWidget(lblOpen)
        self.list_open_images = QListWidget()
        self.list_open_images.setMinimumHeight(minHeight)
        self.list_open_images.setMinimumWidth(minWidth)
        self.list_open_images.setSelectionMode(QAbstractItemView.ExtendedSelection)
        vLOopen.addWidget(self.list_open_images)
        hLObtns = QHBoxLayout()
        btnRemSel = QPushButton('Remove selected')
        btnRemSel.clicked.connect(self.remove_selected)
        hLObtns.addWidget(btnRemSel)
        btnClearList = QPushButton('Clear list')
        btnClearList.clicked.connect(self.clear_list)
        hLObtns.addWidget(btnClearList)
        vLOopen.addLayout(hLObtns)
        vLOopen.addStretch()

        '''TODO - show quick_test together with list of images to open?
        vLOqt = QVBoxLayout()
        hLOlists.addLayout(vLOqt)
        lblQT = LabelHeader('QuickTest', 4)
        lblQT.setAlignment(Qt.AlignCenter)
        vLOqt.addWidget(lblQT)
        self.listQT = QListWidget()
        self.listQT.setMinimumHeight(minHeight)
        vLOqt.addWidget(self.listQT)
        vLOqt.addStretch()
        '''

        hLOdlgBtns = QHBoxLayout()
        vLO.addLayout(hLOdlgBtns)
        hLOdlgBtns.addStretch()
        btns = QDialogButtonBox()
        btns.setOrientation(Qt.Horizontal)
        btns.setStandardButtons(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText('Open selected')
        btns.button(QDialogButtonBox.Ok).clicked.connect(self.accept)
        btns.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)
        hLOdlgBtns.addWidget(btns)

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
                imgInfos, ignored_files = read_dcm_info(
                    dcm_files, tag_infos=self.main.tag_infos,
                    tag_patterns_special=self.main.tag_patterns_special)
                series_nmb_name = []
                self.imgs = []
                for img in imgInfos:
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
            set_selected_ids = [i for i in range(self.list_images.count())]
        elif btn_id == 1:  # select n images closest to zpos
            first = True
            for sel in sel_series:
                if first or push:
                    try:
                        zpos_arr = []
                        for img in self.imgs[sel.row()]:
                            zpos_arr.append(img.zpos)
                        if None not in zpos_arr:
                            n_img = self.spinNclose.value()
                            zpos = self.spinZclose.value()
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
