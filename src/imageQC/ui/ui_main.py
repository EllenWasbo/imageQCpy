#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for main window of imageQC.

@author: Ellen Wasbo
"""
import sys
import os
import copy
import numpy as np
from skimage import draw
from time import time
from dataclasses import dataclass
import pandas as pd

from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtWidgets import (
    QApplication, qApp, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QStatusBar, QSplitter, QGroupBox, QTabWidget,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QButtonGroup,
    QRadioButton, QComboBox, QSlider, QMenu, QAction, QToolBar, QToolButton,
    QMessageBox, QInputDialog, QTreeWidget, QTreeWidgetItem,
    QTableWidget, QTableWidgetItem, QAbstractItemView,
    QFileDialog, QScrollArea, QAbstractScrollArea,
    )
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)

from imageQC.ui.ui_dialogs import StartUpDialog, EditAnnotationsDialog
import imageQC.ui.ui_test_tabs
import imageQC.ui.rename_dicom
import imageQC.ui.automation_wizard
import imageQC.ui.settings
import imageQC.ui.open_multi
import imageQC.ui.open_automation
import imageQC.ui.reusables as uir
import imageQC.config.config_func as cff
from imageQC.config.iQCconstants import (
    QUICKTEST_OPTIONS, VERSION,
    ENV_ICON_PATH, ENV_CONFIG_FOLDER, ENV_USER_PREFS_PATH
    )
import imageQC.config.config_classes as cfc
import imageQC.scripts.dcm as dcm
from imageQC.scripts.calculate_roi import get_rois
from imageQC.scripts.mini_methods_format import val_2_str
from imageQC.scripts.mini_methods import get_min_max_pos_2d
from imageQC.scripts.calculate_qc import calculate_qc, quicktest_output


def get_rotated_crosshair(szx, szy, delta_xya):
    """Get xydata for rotated crosshair.

    Parameters
    ----------
    szx : int
        image size x direction
    szy : int
        image size y direction
    delta_xya : tuple
        dx, dy, dangle - center offset relative to center in image

    Returns
    -------
    x1, x2, y1, y2 : int
        xydata for 2Dlines
    """
    tan_a = np.tan(np.deg2rad(delta_xya[2]))
    dy1 = tan_a*(szx*0.5 + delta_xya[0])
    dy2 = tan_a*(szx*0.5 - delta_xya[0])
    dx1 = tan_a*(szy*0.5 - delta_xya[1])
    dx2 = tan_a*(szy*0.5 + delta_xya[1])

    x1 = szx*0.5+delta_xya[0]-dx1
    x2 = szx*0.5+delta_xya[0]+dx2
    y1 = szy*0.5+delta_xya[1]-dy1
    y2 = szy*0.5+delta_xya[1]+dy2

    return (x1, x2, y1, y2)


@dataclass
class GuiVariables():
    """Class to keep and send variables across GUI classes."""

    # related to loaded images, used in calculations
    # for positioning of center/rot.angle. Difference from imgSz/2 in pixels
    delta_x: int = 0
    delta_y: int = 0
    delta_a: float = 0.0
    delta_use: bool = True

    # related to loaded images, used in GUI
    active_img_no: int = -1
    # which img number in imgDicts is currently on display
    last_clicked_pos: tuple = (-1, -1)

    panel_width: int = 1400
    panel_height: int = 700
    annotations: bool = True
    annotations_line_thick: int = 3
    annotations_font_size: int = 14
    hidden_rgt_top: bool = False  # True if upper right panel collapsed


class MainWindow(QMainWindow):
    """Class main window of imageQC."""

    def __init__(self, scX=1400, scY=700):
        super().__init__()

        self.save_blocked = False
        if os.environ[ENV_USER_PREFS_PATH] == '':
            dlg = StartUpDialog()
            dlg.exec()
        if os.environ[ENV_USER_PREFS_PATH] == '':
            if os.environ[ENV_CONFIG_FOLDER] == '':
                self.save_blocked = True

        if os.environ[ENV_CONFIG_FOLDER] != '':
            cff.add_user_to_active_users()

        # minimum parameters as for scripts.automation.InputMain
        self.update_settings()  # sets self.tag_infos (and a lot more)
        self.current_modality = 'CT'
        self.current_test = QUICKTEST_OPTIONS['CT'][0]
        self.current_paramset = copy.deepcopy(
            self.paramsets[self.current_modality][0])
        self.current_quicktest = cfc.QuickTestTemplate()
        self.imgs = []
        self.results = {}
        self.current_group_indicators = []
        # string for each image if output set pr group with quicktest (paramset.output)
        self.automation_active = False

        # parameters specific to GUI version
        self.vGUI = GuiVariables()
        self.vGUI.panel_width = round(0.48*scX)
        self.vGUI.panel_height = round(0.86*scY)
        self.vGUI.annotations_line_thick = self.user_prefs.annotations_line_thick
        self.vGUI.annotations_font_size = self.user_prefs.annotations_font_size
        self.statusBar = StatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Starting up', 1000)
        self.active_img = None  # np.array pixeldata for active image
        self.summed_img = None  # sum of marked images if activated
        self.average_img = False  # true if summed_img should be averaged
        self.current_roi = None

        self.setWindowTitle('Image QC v ' + VERSION)
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setGeometry(self.vGUI.panel_width*0.02,
                         self.vGUI.panel_height*0.05,
                         self.vGUI.panel_width*2+30,
                         self.vGUI.panel_height+50)

        self.treeFileList = TreeFileList(self)
        self.create_menu_toolBar()
        self.wImageDisplay = ImageDisplayWidget(self)
        self.dicomHeaderWidget = DicomHeaderWidget(self)
        self.windowLevelWidget = WindowLevelWidget(self)
        self.centerWidget = CenterWidget(self)
        self.create_modality_selector()
        self.wQuickTest = SelectQuickTestWidget(self)
        self.wParamset = SelectParamsetWidget(self)
        self.create_result_tabs()
        self.create_test_tabs()

        # set main layout (left/right)
        bbox = QHBoxLayout()
        self.splitLftRgt = QSplitter(Qt.Horizontal)
        lftWidget = QWidget()
        rgtWidget = QWidget()
        lftBox = QVBoxLayout()
        rgtBox = QVBoxLayout()
        lftWidget.setLayout(lftBox)
        rgtWidget.setLayout(rgtBox)
        self.splitLftRgt.addWidget(lftWidget)
        self.splitLftRgt.addWidget(rgtWidget)
        bbox.addWidget(self.splitLftRgt)

        # Fill left box
        self.splitListRest = QSplitter(Qt.Vertical)
        self.splitImgHeader = QSplitter(Qt.Vertical)
        self.splitLeftImg = QSplitter(Qt.Horizontal)
        wWLcenter = QWidget()
        vLO_WLcenter = QVBoxLayout()
        vLO_WLcenter.addWidget(self.windowLevelWidget)
        vLO_WLcenter.addWidget(self.centerWidget)
        wWLcenter.setLayout(vLO_WLcenter)

        self.splitListRest.addWidget(self.treeFileList)
        self.splitListRest.addWidget(self.splitImgHeader)
        self.splitImgHeader.addWidget(self.splitLeftImg)
        self.splitImgHeader.addWidget(self.dicomHeaderWidget)
        self.splitLeftImg.addWidget(wWLcenter)
        self.splitLeftImg.addWidget(self.wImageDisplay)

        lftBox.addWidget(self.splitListRest)

        # Fill right box
        self.splitRgtTopRest = QSplitter(Qt.Vertical)
        rgtBoxTop = QWidget()
        vLOtop = QVBoxLayout()
        rgtBoxTop.setLayout(vLOtop)
        hLO_mod = QHBoxLayout()
        hLO_mod.addWidget(self.gb_modality)
        self.btn_read_vendor_file = QCheckBox('Read vendor file')
        self.btn_read_vendor_file.setChecked(False)
        self.btn_read_vendor_file.toggled.connect(self.update_mode)
        hLO_mod.addWidget(self.btn_read_vendor_file)
        vLOtop.addLayout(hLO_mod)
        vLOtop.addWidget(self.wQuickTest)
        vLOtop.addWidget(self.wParamset)
        self.splitRgtTopRest.addWidget(rgtBoxTop)

        self.splitRgtMidBtm = QSplitter(Qt.Vertical)
        self.splitRgtTopRest.addWidget(self.splitRgtMidBtm)
        self.splitRgtMidBtm.addWidget(self.sTestTabs)
        self.splitRgtMidBtm.addWidget(self.tabResults)
        rgtBox.addWidget(self.splitRgtTopRest)

        widFull = QWidget()
        widFull.setLayout(bbox)
        widFull.setFixedSize(2*self.vGUI.panel_width, self.vGUI.panel_height)

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widFull)

        self.setCentralWidget(scroll)
        self.reset_split_sizes()
        self.update_mode()

    def get_modality_index(self, modality_string):
        """Get index of given modality string.

        Parameters
        ----------
        modality_string : str
            modality as defined in imageQC (CT, Xray...)

        Returns
        -------
        int
            index of given modality
        """
        mods = [*QUICKTEST_OPTIONS]
        return mods.index(modality_string)

    def update_paramset(self):
        """Fill gui with params from selected paramset."""
        self.current_paramset = copy.deepcopy(
            self.paramsets[self.current_modality][
                self.wParamset.cbox_template.currentIndex()])

        widget = self.sTestTabs.currentWidget()
        widget.update_displayed_params()
        self.wResTable.tb_copy.parameters_output = self.current_paramset.output
        self.wResTable.tb_copy.update_checked()

    def open_files(self, file_list=None):
        """Open DICOM files and update GUI."""
        if file_list is False:
            fnames = QFileDialog.getOpenFileNames(
                self, 'Open DICOM files',
                filter="DICOM files (*.dcm);;All files (*)")
            file_list = fnames[0]
        if len(file_list) > 0:
            new_img_infos, ignored_files = dcm.read_dcm_info(
                file_list, tag_infos=self.tag_infos,
                tag_patterns_special=self.tag_patterns_special)
            if len(ignored_files) > 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(
                    f'{len(ignored_files)} files ignored missing DICOM image data')
                msg.setInformativeText(
                    'Try File > Read DICOM header. Ignored files listed in details.')
                msg.setWindowTitle('Some files ignored')
                msg.setDetailedText('\n'.join(ignored_files))
                msg.exec_()
            if len(new_img_infos) > 0:
                nImgBefore = len(self.imgs)
                if self.chkAppend.isChecked():
                    if self.chkAppend.isChecked() and nImgBefore > 0:
                        if self.wQuickTest.gbQT.isChecked() is False:
                            for d in new_img_infos:
                                d.marked = False
                    self.imgs = self.imgs + new_img_infos
                    self.update_results(n_added_imgs=len(new_img_infos))
                else:
                    self.imgs = new_img_infos
                    self.results = {}

                if self.wQuickTest.gbQT.isChecked():
                    self.wQuickTest.set_current_template_to_imgs()

                if self.chkAppend.isChecked() is False or nImgBefore == 0:
                    self.vGUI.active_img_no = 0
                    # update GUI according to first image
                    if self.current_modality != self.imgs[0].modality:
                        self.current_modality = self.imgs[0].modality
                        self.update_mode()
                    if self.windowLevelWidget.chkWLupdate.isChecked() is False:
                        self.windowLevelWidget.set_window_level('dcm')

                if self.summed_img is not None:
                    self.reset_summed_img()
                self.treeFileList.update_file_list()

    def open_multi(self):
        """Start open advanced dialog."""
        dlg = imageQC.ui.open_multi.OpenMultiDialog(self)
        res = dlg.exec()
        if res:
            if len(dlg.open_imgs) > 0:
                self.imgs.extend(dlg.open_imgs)

                if self.wQuickTest.gbQT.isChecked():
                    self.wQuickTest.set_current_template_to_imgs()

                if self.summed_img is not None:
                    self.reset_summed_img()

                self.treeFileList.update_file_list()

    def read_header(self):
        """View file as header."""
        fname = QFileDialog.getOpenFileName(
            self, 'Read DICOM header',
            filter="DICOM files (*.dcm);;All files (*)")
        filename = fname[0]
        if filename != '':
            dcm.dump_dicom(self, filename=filename)

    def open_auto(self):
        """Start open automation dialog."""
        dlg = imageQC.ui.open_automation.OpenAutomationDialog(self)
        dlg.exec()

    def flag_edit(self, flag=True):
        """Reset results if exist.

        Used by TestTable when checkmarks changes.
        """
        pass

    def set_active_img(self, imgno):
        """Set active image programmatically.

        Parameters
        ----------
        imgno : int
            image number to set active
        """
        self.treeFileList.setCurrentItem(
            self.treeFileList.topLevelItem(imgno))

    def update_active_img(self, current, prev):
        """Overwrite pixmap in memory with new active image, refresh GUI."""
        if len(self.imgs) > 0:
            self.vGUI.active_img_no = self.treeFileList.indexOfTopLevelItem(
                current)
            self.active_img, tags = dcm.get_img(
                self.imgs[self.vGUI.active_img_no].filepath,
                frame_number=self.imgs[self.vGUI.active_img_no].frame_number)
            if self.active_img is not None:
                amin = np.amin(self.active_img)
                amax = np.amax(self.active_img)
                self.windowLevelWidget.minWL.setRange(amin, amax)
                self.windowLevelWidget.maxWL.setRange(amin, amax)
                if len(np.shape(self.active_img)) == 2:
                    szActy, szActx = np.shape(self.active_img)
                else:
                    szActy, szActx, szActz = np.shape(self.active_img)
                self.centerWidget.valDeltaX.setRange(-szActx/2, szActx/2)
                self.centerWidget.valDeltaY.setRange(-szActy/2, szActy/2)
            self.dicomHeaderWidget.refresh_img_info(
                self.imgs[self.vGUI.active_img_no].info_list_general,
                self.imgs[self.vGUI.active_img_no].info_list_modality)

            self.refresh_img_display()
            if self.current_test in ['Sli', 'Uni']:
                self.wResPlot.plotcanvas.plot()
            if self.current_test in ['Uni']:
                self.wResImage.canvas.result_image_draw()

    def update_summed_img(self, recalculate_sum=True):
        """Overwrite pixmap in memory with new summed image, refresh GUI."""
        if len(self.imgs) > 0:
            if recalculate_sum:
                self.start_wait_cursor()
                self.statusBar.showMessage('Calculating sum of marked images...')
                testcode = ''
                if self.wQuickTest.gbQT.isChecked():
                    testcode = self.current_test
                print(f'testcode {testcode}')
                breakpoint()
                self.summed_img, errmsg = dcm.sum_marked_images(
                    self.imgs, testcode=testcode)
                self.stop_wait_cursor()
                self.statusBar.showMessage('Finished summing marked images', 2000)
            if self.summed_img is not None:
                if self.average_img:
                    if self.wQuickTest.gbQT.isChecked():
                        marked = [img_info.marked_quicktest for img_info in self.imgs]
                    else:
                        marked = [img_info.marked for img_info in self.imgs]
                    self.active_img = self.summed_img * (1./len(marked))
                else:
                    self.active_img = self.summed_img
                amin = np.amin(self.active_img)
                amax = np.amax(self.active_img)
                self.windowLevelWidget.minWL.setRange(amin, amax)
                self.windowLevelWidget.maxWL.setRange(amin, amax)
                self.refresh_img_display()
            else:
                if errmsg != '':
                    QMessageBox.information(
                        self, 'Failed summing images', errmsg)

    def reset_summed_img(self):
        """Turn off display of summed image and reset summed_img."""
        self.wImageDisplay.tool_sum.setChecked(False)
        self.wImageDisplay.tool_sum.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}sigma.png'))
        self.summed_img = None
        self.average_img = False
        self.active_img, tags = dcm.get_img(
            self.imgs[self.vGUI.active_img_no].filepath,
            frame_number=self.imgs[self.vGUI.active_img_no].frame_number)

    def mode_changed(self):
        """Modality selection changed by user input, initiate update gui."""
        if self.wParamset.lbl_edit.text() == '*':
            self.wParamset.ask_to_save_changes()
        self.current_modality = self.btnMode.checkedButton().text()
        self.update_mode()

    def update_mode(self):
        """Update GUI when modality has changed."""
        curr_mod_idx = self.get_modality_index(self.current_modality)
        if curr_mod_idx != self.btnMode.checkedId():
            self.btnMode.button(curr_mod_idx).setChecked(True)

        if self.btn_read_vendor_file.isChecked():
            self.sTestTabs.setCurrentIndex(len(QUICKTEST_OPTIONS))
            self.tabVendor.update_table()
        else:
            self.sTestTabs.setCurrentIndex(curr_mod_idx)

        # update list of available parametersets / QuickTests
        if len(self.paramsets[self.current_modality]) > 0:
            self.current_paramset = self.paramsets[self.current_modality][0]
        self.wParamset.fill_template_list()
        self.update_paramset()
        self.wQuickTest.fill_template_list()
        if self.wQuickTest.gbQT.isChecked():
            self.current_quicktest = cfc.QuickTestTemplate()
            self.refresh_img_display()

        reset_test_idx = False
        if self.current_test not in ['DCM', 'ROI']:
            reset_test_idx = True
        self.update_current_test(reset_index=reset_test_idx)

    def update_current_test(self, reset_index=False):
        """Update GUI when selected test change.

        Parameters
        ----------
        reset_index : bool
            Rest test index if mode change. Default is False
        """
        widget = self.sTestTabs.currentWidget()
        if widget is not None:
            if hasattr(widget, 'currentIndex'):
                if isinstance(reset_index, bool) and reset_index:
                    widget.setCurrentIndex(0)
                test_idx = widget.currentIndex()
                self.current_test = QUICKTEST_OPTIONS[
                    self.current_modality][test_idx]
                if self.active_img is not None:
                    self.update_roi()
                    self.refresh_results_display()

    def update_roi(self):
        """Recalculate ROI."""
        if self.active_img is not None:
            self.current_roi = get_rois(
                self.active_img,
                self.vGUI.active_img_no, self)
        else:
            self.current_roi = None
        self.wImageDisplay.canvas.roi_draw()

    def update_results(self, n_added_imgs=0, deleted_idxs=[], sort_idxs=[]):
        """Update self.results if added / deleted images.

        Parameters
        ----------
        n_added_imgs : int, optional
            number of added imgs. The default is 0.
        deleted_idxs : list of int, optional
            list of deleted image numbers. The default is [].
        sort_idxs : list of indx, optional
            new sort order. The default is [].
        """
        if n_added_imgs > 0:
            for test, res_dict in self.results.items():
                if res_dict['pr_image']:
                    empty_extend = [[] for i in range(n_added_imgs)]
                    res_dict['values'].extend(empty_extend)
                    if 'values_sup' in res_dict:
                        res_dict['values_sup'].extend(empty_extend)
                    if 'details_dict' in res_dict:
                        res_dict['details_dict'].extend(empty_extend)
                else:
                    del self.results[test]

        if len(deleted_idxs):
            deleted_idxs.sort(reverse=True)
            for test, res_dict in self.results.items():
                if res_dict['pr_image']:
                    for idx in deleted_idxs:
                        del res_dict['values'][idx]
                        if 'values_sup' in res_dict:
                            del res_dict['values_sup'][idx]
                        if 'details_dict' in res_dict:
                            del res_dict['details_dict'][idx]
                else:
                    del self.results[test]

        if len(sort_idxs):
            orig_res = copy.deepcopy(self.results)
            self.results = {}
            for test, res_dict in orig_res.items():
                if res_dict['pr_image']:
                    self.results[test] = {
                        'headers': res_dict['headers'],
                        'values': [],
                        'alternative': res_dict['alternative'],
                        'headers_sup': res_dict['headers_sup'],
                        'values_sup': [],
                        'details_dict': [],
                        'pr_image': True
                        }
                    for idx in sort_idxs:
                        self.results[test]['values'].append(orig_res['values'][idx])
                        self.results[test]['values_sup'].append(
                            orig_res['values_sup'][idx])
                        self.results[test]['details_dict'].append(
                            orig_res['details_dict'][idx])

        self.refresh_results_display()

    def refresh_results_display(self):
        """Update GUI for test results when results or selections change."""
        if self.vGUI.hidden_rgt_top is False:
            if self.current_test in self.results:
                self.hide_rgt_top()  # maximize results displays first time
        try:
            self.wResTable.result_table.fill_table(
                col_labels=self.results[self.current_test]['headers'],
                values_rows=self.results[self.current_test]['values'],
                linked_image_list=self.results[self.current_test]['pr_image'])
        except (KeyError, TypeError):
            self.wResTable.result_table.clear()

        try:
            self.wResTableSup.result_table.fill_table(
                col_labels=self.results[self.current_test]['headers_sup'],
                values_rows=self.results[self.current_test]['values_sup'],
                linked_image_list=self.results[self.current_test]['pr_image'])
        except (KeyError, TypeError):
            self.wResTableSup.result_table.clear()

        self.wResPlot.plotcanvas.plot()
        self.wResImage.canvas.result_image_draw()

    def refresh_img_display(self):
        """Refresh image related gui."""
        if self.active_img is not None:
            self.current_roi = get_rois(
                self.active_img,
                self.vGUI.active_img_no, self)
            self.wImageDisplay.canvas.img_draw()
            self.windowLevelWidget.WindowLevelHistoCanvas.plot(
                self.active_img)
            self.dicomHeaderWidget.refresh_img_info(
                self.imgs[self.vGUI.active_img_no].info_list_general,
                self.imgs[self.vGUI.active_img_no].info_list_modality)
        else:
            self.wImageDisplay.canvas.img_is_missing()

    def sort_imgs(self):
        """Resort images by dicom info."""
        sortTemplate = cfc.TagPatternSort()
        dlg = uir.TagPatternEditDialog(
            initial_pattern=sortTemplate,
            modality=self.current_modality,
            title='Sort images by DICOM header information',
            typestr='sort',
            accept_text='Sort',
            reject_text='Cancel',
            save_blocked=self.save_blocked)
        res = dlg.exec()  # returing TagPatternSort
        if res:
            sortTemplate = dlg.get_pattern()
            # use sortTemplate
            print(sortTemplate)
            # TODO NB coordinate with TestTabs how to return
            self.imgs = dcm.sort_imgs(self.imgs, sortTemplate, self.tag_infos)
            self.treeFileList.update_file_list()

    def move(self, to=''):
        """Resort images. Move selected image to...

        Parameters
        ----------
        to : str
            to where: top/down/up/bottom
        """
        if to == 'top':
            insertNo = 0
        elif to == 'down':
            insertNo = self.vGUI.active_img_no + 1
        elif to == 'up':
            insertNo = self.vGUI.active_img_no - 1
        elif to == 'bottom':
            insertNo = -1
        else:
            insertNo = -2
        if insertNo > -2:
            this = self.imgs.pop(self.vGUI.active_img_no)
            if insertNo == -1:
                self.imgs.append(this)
                self.vGUI.active_img_no = len(self.imgs) - 1
            else:
                self.imgs.insert(insertNo, this)
                self.vGUI.active_img_no = insertNo
            self.treeFileList.update_file_list()

    def reset_split_sizes(self):
        """Set and reset QSplitter sizes."""
        self.splitListRest.setSizes(
            [self.vGUI.panel_height*0.2, self.vGUI.panel_height*0.8])
        self.splitImgHeader.setSizes(
            [self.vGUI.panel_height*0.55, self.vGUI.panel_height*0.25])
        self.splitLeftImg.setSizes(
            [self.vGUI.panel_width*0.32, self.vGUI.panel_width*0.68])
        self.splitLftRgt.setSizes(
            [self.vGUI.panel_width*1.2, self.vGUI.panel_width*0.8])
        self.splitRgtTopRest.setSizes(
            [self.vGUI.panel_height*0.2, self.vGUI.panel_height*0.8])
        self.splitRgtMidBtm.setSizes(
            [self.vGUI.panel_height*0.4, self.vGUI.panel_height*0.4])

    def set_split_max_img(self):
        """Set QSplitter to maximized image."""
        self.splitListRest.setSizes(
            [self.vGUI.panel_height*0.1, self.vGUI.panel_height*0.9])
        self.splitImgHeader.setSizes(
            [self.vGUI.panel_height*0.9, self.vGUI.panel_height*0.])
        self.splitLeftImg.setSizes(
            [self.vGUI.panel_width*0., self.vGUI.panel_width*1.])

    def reset_split_max_img(self):
        """Set QSplitter to maximized image."""
        self.splitListRest.setSizes(
            [self.vGUI.panel_height*0.2, self.vGUI.panel_height*0.8])
        self.splitImgHeader.setSizes(
            [self.vGUI.panel_height*0.55, self.vGUI.panel_height*0.25])
        self.splitLeftImg.setSizes(
            [self.vGUI.panel_width*0.32, self.vGUI.panel_width*0.68])

    def set_maximize_results(self):
        """Set QSplitter to maximized results."""
        self.splitLftRgt.setSizes(
            [self.vGUI.panel_width*0.8, self.vGUI.panel_width*1.2])
        self.splitRgtTopRest.setSizes(
            [self.vGUI.panel_height*0., self.vGUI.panel_height*1.])
        self.splitRgtMidBtm.setSizes(
            [self.vGUI.panel_height*0., self.vGUI.panel_height*1.])

    def reset_maximize_results(self):
        """Set QSplitter to maximized results."""
        self.splitLftRgt.setSizes(
            [self.vGUI.panel_width*1.2, self.vGUI.panel_width*0.8])
        self.splitRgtTopRest.setSizes(
            [self.vGUI.panel_height*0.2, self.vGUI.panel_height*0.8])
        self.splitRgtMidBtm.setSizes(
            [self.vGUI.panel_height*0.4, self.vGUI.panel_height*0.4])

    def hide_rgt_top(self):
        """Hide QSplitter right top to better display results."""
        self.splitRgtTopRest.setSizes(
            [self.vGUI.panel_height*0., self.vGUI.panel_height*1.])
        self.splitRgtMidBtm.setSizes(
            [self.vGUI.panel_height*0.5, self.vGUI.panel_height*0.5])

    def run_auto_wizard(self):
        """Start the automation wizard."""
        if os.environ[ENV_CONFIG_FOLDER] == '':
            QMessageBox.information(
                self, 'Missing config folder',
                '''To be able to save settings you will need to specify a \n
                configuration folder for these settings.\n
                Start the wizard again when the configuration folder is set.
                ''')
            dlg = imageQC.ui.settings.SettingsDialog(
                self, initial_view='Config folder')
            dlg.exec()
        else:
            self.wizard = imageQC.ui.automation_wizard.AutomationWizard(self)
            self.wizard.open()

    def run_rename_dicom(self):
        """Start Rename Dicom dialog."""
        open_file_names = [imgDict.filepath for imgDict in self.imgs]
        dlg = imageQC.ui.rename_dicom.RenameDicomDialog(
            open_file_names, initial_modality=self.current_modality)
        res = dlg.exec()
        """
        open_files_are_renamed, new_names = dlg.new_names()
        selection = dlg.selection()
        if open_files_are_renamed:
            """

    def run_settings(self, initial_view=''):
        """Display settings dialog."""
        if initial_view == '':
            dlg = imageQC.ui.settings.SettingsDialog(self)
        else:
            dlg = imageQC.ui.settings.SettingsDialog(
                self, initial_view=initial_view)
        dlg.exec()
        self.update_settings()
        self.update_mode()

    def update_settings(self):
        """Refresh data from settings files affecting GUI in main window."""
        self.lastload = time()
        status, path, self.user_prefs = cff.load_user_prefs()
        status, path, self.paramsets = cff.load_settings(fname='paramsets')
        status, path, self.quicktests = cff.load_settings(
            fname='quicktest_templates')
        status, path, self.tag_infos = cff.load_settings(fname='tag_infos')
        status, path, self.tag_patterns_special = cff.load_settings(
            fname='tag_patterns_special')

        #TODO verify settings against
        #   version (available modalities and testcodes)
        #   slettede filer med f.eks. paramset som linket til i andre
        #   display paramset med endret tag_info attr.names

    def start_wait_cursor(self):
        """Block mouse events by wait cursor."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        qApp.processEvents()

    def stop_wait_cursor(self):
        """Return to normal mouse cursor after wait cursor."""
        QApplication.restoreOverrideCursor()

    def closeEvent(self, event):
        """Exit app by x in the corner."""
        cff.remove_user_from_active_users()
        event.accept()

    def exit_app(self):
        """Exit app by menu."""
        if self.wParamset.lbl_edit.text() == '*':
            self.wParamset.ask_to_save_changes()
        cff.remove_user_from_active_users()
        # save current settings to user prefs
        self.user_prefs.annotations_line_thick = self.vGUI.annotations_line_thick
        self.user_prefs.annotations_font_size = self.vGUI.annotations_font_size
        ok, path = cff.save_user_prefs(self.user_prefs, parentwidget=self)
        sys.exit()

    def clicked_resultsize(self, tool=None):
        """Maximize or reset results size."""
        if tool.isChecked():
            tool.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}layout_resetimg.png'))
            self.set_maximize_results()
        else:
            tool.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}layout_maximg.png'))
            self.reset_maximize_results()

    def create_menu_toolBar(self):
        """GUI of MenuBar and main ToolBar."""
        mb = self.menuBar()
        tb = self.addToolBar('first')

        self.chkAppend = QCheckBox('Append')
        self.cbox_file_list_display = QComboBox()
        self.cbox_file_list_display.addItems(
            ['File path', 'Format pattern'])
        self.cbox_file_list_display.setToolTip(
            'Format pattern = Special tag pattern as defined in settings')
        self.cbox_file_list_display.currentIndexChanged.connect(
            self.treeFileList.update_file_list)

        actOpen = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}open.png'),
            'Open DICOM image(s)', self)
        actOpen.setShortcut('Ctrl+O')
        actOpen.triggered.connect(self.open_files)

        actOpenAdv = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}openMulti.png'),
            'Open DICOM image(s) with advanced options', self)
        actOpenAdv.triggered.connect(self.open_multi)

        actReadHeader = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}tags.png'),
            'Read DICOM header', self)
        actReadHeader.triggered.connect(self.read_header)

        actOpenAuto = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}play.png'),
            'Run automation templates', self)
        actOpenAuto.triggered.connect(self.open_auto)

        actWizardAuto = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}playGears.png'),
            'Start wizard to config automation with current settings', self)
        actWizardAuto.triggered.connect(self.run_auto_wizard)

        actRenameDICOM = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}rename_dicom.png'),
            'Run Rename DICOM', self)
        actRenameDICOM.triggered.connect(self.run_rename_dicom)

        actToTop = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveTop.png'),
            'Move selected images to top', self)
        actToTop.triggered.connect(lambda: self.move(to='top'))

        actToUp = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
            'Move selected images up', self)
        actToUp.triggered.connect(lambda: self.move(to='up'))

        actToDown = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
            'Move selected images down', self)
        actToDown.triggered.connect(lambda: self.move(to='down'))

        actToBottom = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveBottom.png'),
            'Move selected images to bottom', self)
        actToBottom.triggered.connect(lambda: self.move(to='bottom'))

        actSort = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}sortAZ.png'),
            'Sort images from patterns based on DICOM header', self)
        actSort.triggered.connect(self.sort_imgs)

        actSettings = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}gears.png'),
            'Settings', self)
        actSettings.triggered.connect(self.run_settings)

        actResetSplit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}layout.png'),
            'Reset layout', self)
        actResetSplit.triggered.connect(self.reset_split_sizes)

        actQuit = QAction('&Quit', self)
        actQuit.setShortcut('Ctrl+Q')
        actQuit.triggered.connect(self.exit_app)

        # fill menus
        mFile = QMenu('&File', self)
        mFile.addActions([actOpen, actOpenAdv, actReadHeader,
                          actOpenAuto, actWizardAuto,
                          actRenameDICOM, actQuit])
        mb.addMenu(mFile)
        mSett = QMenu('&Settings', self)
        mSett.addAction(actSettings)
        mb.addMenu(mSett)
        mb.addMenu(QMenu('&Help', self))

        # fill toolbar
        tb.addActions([actOpen, actOpenAdv, actOpenAuto, actWizardAuto])
        tb.addSeparator()
        tb.addWidget(self.chkAppend)
        tb.addSeparator()
        tb.addActions([actToTop, actToUp, actToDown, actToBottom,
                       actSort])
        self.lblNloaded = QLabel('0    ')
        tb.addWidget(QLabel('Loaded images:'))
        tb.addWidget(self.lblNloaded)
        tb.addWidget(QLabel('             '))
        tb.addWidget(QLabel('List files as'))
        tb.addWidget(self.cbox_file_list_display)

        tb.addWidget(QLabel('             '))
        tb.addActions([actResetSplit, actRenameDICOM, actSettings])

    def create_modality_selector(self):
        """Groupbox with modality selection."""
        self.gb_modality = QGroupBox('Modality')
        self.gb_modality.setFont(uir.FontItalic())
        self.btnMode = QButtonGroup()
        self.gb_modality.setFixedWidth(self.vGUI.panel_width*0.75)
        lo = QHBoxLayout()

        for m, (key, val) in enumerate(QUICKTEST_OPTIONS.items()):
            rb = QRadioButton(key)
            self.btnMode.addButton(rb, m)
            lo.addWidget(rb)
            rb.clicked.connect(self.mode_changed)

        idx = self.get_modality_index(self.current_modality)
        self.btnMode.button(idx).setChecked(True)
        self.gb_modality.setLayout(lo)

    def create_test_tabs(self):
        """Initiate GUI for the stacked test tabs."""
        self.sTestTabs = QStackedWidget()
        self.tabCT = imageQC.ui.ui_test_tabs.TestTabCT(self)
        self.tabXray = imageQC.ui.ui_test_tabs.TestTabXray(self)
        self.tabNM = imageQC.ui.ui_test_tabs.TestTabNM(self)
        self.tabSPECT = imageQC.ui.ui_test_tabs.TestTabSPECT(self)
        self.tabPET = imageQC.ui.ui_test_tabs.TestTabPET(self)
        self.tabMR = imageQC.ui.ui_test_tabs.TestTabMR(self)
        self.tabVendor = imageQC.ui.ui_test_tabs.TestTabVendor(self)

        self.sTestTabs.addWidget(self.tabCT)
        self.sTestTabs.addWidget(self.tabXray)
        self.sTestTabs.addWidget(self.tabNM)
        self.sTestTabs.addWidget(self.tabSPECT)
        self.sTestTabs.addWidget(self.tabPET)
        self.sTestTabs.addWidget(self.tabMR)
        self.sTestTabs.addWidget(self.tabVendor)

    def create_result_tabs(self):
        """Initiate GUI for the stacked result tabs."""
        self.tabResults = QTabWidget()
        self.wResTable = ResultTableWidget(self)
        self.wResPlot = ResultPlotWidget(self, ResultPlotCanvas(self))
        self.wResImage = ResultImageWidget(self)
        self.wResTableSup = ResultTableWidget(self)

        self.tabResults.addTab(self.wResTable, 'Results table')
        self.tabResults.addTab(self.wResPlot, 'Results plot')
        self.tabResults.addTab(self.wResImage, 'Results image')
        self.tabResults.addTab(self.wResTableSup, 'Supplement table')


class TreeFileList(QTreeWidget):
    """QTreeWidget for list of images marked for testing."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent
        self.setColumnCount(3)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setHeaderLabels(['Image', 'Frame', 'Test'])
        self.setColumnWidth(0, 0.8*self.main.vGUI.panel_width)
        self.setColumnWidth(1, 90)
        self.currentItemChanged.connect(self.main.update_active_img)
        self.installEventFilter(self)
        self.setRootIsDecorated(False)

    def get_selected_imgs(self):
        """Get selected images in file list.

        Returns
        -------
        selrows : list of int
            selected rows
        """
        selrows = []
        for sel in self.selectedIndexes():
            if sel.column() > -1:
                selrows.append(sel.row())
        if len(selrows) > 0:
            selrows = list(set(selrows))  # remove duplicates
        return selrows

    def update_file_list(self):
        """Populate tree with filepath/pattern and test indicators."""
        self.clear()
        if len(self.main.imgs) == 0:
            QTreeWidgetItem(self, ['', '', ''])
        else:
            for fi in self.main.imgs:
                if self.main.wQuickTest.gbQT.isChecked():
                    test_string = '+'.join(fi.marked_quicktest)
                else:
                    test_string = 'x' if fi.marked else ''
                frameno = f'{fi.frame_number}' if fi.frame_number > -1 else ''

                if self.main.cbox_file_list_display.currentIndex() == 0:
                    file_text = fi.filepath
                else:
                    file_text = ' '.join(fi.file_list_strings)
                QTreeWidgetItem(self, [file_text, frameno, test_string])
            self.main.lblNloaded.setText(str(len(self.main.imgs)))
            self.setCurrentItem(self.topLevelItem(
                self.main.vGUI.active_img_no))

    def dragEnterEvent(self, event):
        """Handle drag enter event. Opening files by drag/drop."""
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Handle drag move event. Opening files by drag/drop."""
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Accept drop event with file paths."""
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            file_paths = []
            for url in event.mimeData().urls():
                file_paths.append(str(url.toLocalFile()))
            if len(file_paths) > 0:
                self.main.open_files(file_list=file_paths)
        else:
            event.ignore()

    def eventFilter(self, source, event):
        """Handle context menu events (rightclicks)."""
        if event.type() == QEvent.ContextMenu:
            ctxMenu = QMenu(self)
            actMark = QAction('Mark selected')
            actMark.triggered.connect(self.set_marking)
            actRemoveMarkSel = QAction('Remove mark from selected')
            actRemoveMarkSel.triggered.connect(
                lambda: self.set_marking(remove_mark=True))
            actRemoveAllMarks = QAction('Remove all marks')
            actRemoveAllMarks.triggered.connect(self.clear_marking)
            actMarkImgNo = QAction('Mark imgNo .. to ..')
            #actMarkImgNo.triggered.connect(self.mark_imgNo)
            actCloseSel = QAction('Close selected')
            actCloseSel.triggered.connect(self.close_selected)
            ctxMenu.addActions(
                    [actMark, actRemoveMarkSel, actRemoveAllMarks,
                     actMarkImgNo, actCloseSel])

            ctxMenu.exec(event.globalPos())

        return False

    def close_selected(self):
        """Select inverse of the currently selected images."""
        selrows = self.get_selected_imgs()
        if len(selrows) > 0:
            selrows.sort(reverse=True)
            for row in selrows:
                del self.main.imgs[row]
            if self.main.summed_img is not None:
                self.main.reset_summed_img()
            self.update_file_list()
            self.main.update_results(deleted_idxs=selrows)

    def clear_marking(self):
        """Remove all marks for testing from selected images."""
        for imgd in self.main.imgs:
            imgd.marked_quicktest = []
            imgd.marked = False
        if self.main.summed_img is not None:
            self.main.reset_summed_img()
        self.update_file_list()
        self.main.results = {}
        self.main.refresh_results_display()

    def set_marking(self, remove_mark=False, remove_all=False):
        """Set or remove mark for testing from selected images."""
        selrows = self.get_selected_imgs()
        for sel in selrows:
            if self.main.wQuickTest.gbQT.isChecked():
                tests_this = self.main.imgs[sel].marked_quicktest
                if remove_mark:
                    if self.main.current_test in tests_this:
                        self.main.imgs[sel].marked_quicktest.remove(
                            self.main.current_test)
                else:
                    if self.main.current_test not in tests_this:
                        self.main.imgs[sel].marked_quicktest.append(
                            self.main.current_test)
                self.main.wQuickTest.flag_edit(True)
            if remove_mark:
                self.main.imgs[sel].marked = False
            else:
                self.main.imgs[sel].marked = True

        if self.main.summed_img is not None:
            self.main.reset_summed_img()
        self.update_file_list()
        if remove_all:
            self.main.results = {}
        if remove_mark:
            self.main.update_results(deleted_idxs=selrows)
        else:
            self.main.results = {}
        self.main.refresh_results_display()


class DicomHeaderWidget(QWidget):
    """Holder for the Dicom header widget."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent
        hLO = QHBoxLayout()
        self.setLayout(hLO)
        vLO = QVBoxLayout()
        header = QLabel('DICOM header')
        header.setFont(uir.FontItalic())
        vLO.addWidget(header)
        QLabel()
        hLO.addLayout(vLO)
        tb1 = QToolBar()
        vLO.addWidget(tb1)
        tb2 = QToolBar()
        vLO.addWidget(tb2)
        vLO.addStretch()

        actDCMdump = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}tags.png'),
            "View DICOM dump", self)
        actDCMdump.triggered.connect(self.dump_dicom)
        actDCMclipboard = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}copy2clipboard.png'),
            "Send specified DICOM header information as table to clipboard",
            self)
        #actDCMclipboard.triggered.connect(lambda: self.table_DCM('clipboard'))
        actDCMexport = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}fileCSV.png'),
            "Save specified DICOM header information as .csv", self)
        #actDCMexport.triggered.connect(lambda: self.table_DCM('csv'))
        tb1.addActions([actDCMdump, actDCMclipboard])
        tb2.addAction(actDCMexport)

        self.listInfoGeneral = QTreeWidget(columnCount=1)
        self.listInfoGeneral.setHeaderLabels(['General attributes'])
        self.listInfoGeneral.setRootIsDecorated(False)
        hLO.addWidget(self.listInfoGeneral)
        self.listInfoModality = QTreeWidget(columnCount=1)
        self.listInfoModality.setHeaderLabels(['Modality specific attributes'])
        self.listInfoModality.setRootIsDecorated(False)
        hLO.addWidget(self.listInfoModality)

    def dump_dicom(self):
        """Dump dicom elements for active file to text."""
        proceed = True
        if self.main.vGUI.active_img_no < 0:
            QMessageBox.information(self, 'Missing input',
                                    'No file selected.')
            proceed = False
        if proceed:
            fi = self.main.imgs[self.main.vGUI.active_img_no].filepath
            dcm.dump_dicom(self, filename=fi)

    def refresh_img_info(self, info_list_general, info_list_modality):
        """Refresh dicom header information for selected image."""
        self.listInfoGeneral.clear()
        if len(info_list_general) > 0:
            for attr in info_list_general:
                QTreeWidgetItem(self.listInfoGeneral, [attr])
        self.listInfoModality.clear()
        if len(info_list_modality) > 0:
            for attr in info_list_modality:
                QTreeWidgetItem(self.listInfoModality, [attr])


class GenericImageWidget(QWidget):
    """General image widget."""

    def __init__(self, parent, canvas):
        super().__init__()
        self.parent = parent
        self.canvas = canvas
        self.canvas.mpl_connect('motion_notify_event', self.image_on_move)
        self.canvas.mpl_connect('button_press_event', self.image_on_click)
        self.canvas.mpl_connect('button_release_event', self.image_on_release)
        self.tool_profile = QToolButton()
        self.tool_profile.setToolTip(
            'Toggle to plot image profile when click/drag line in image')
        self.tool_profile.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}profile.png'))
        self.tool_profile.clicked.connect(self.clicked_profile)
        self.tool_profile.setCheckable(True)

        self.mouse_pressed = False

    def image_on_move(self, event):
        """Actions on mouse move event."""
        if self.mouse_pressed and self.tool_profile.isChecked():
            if event.inaxes and len(event.inaxes.get_images()) > 0:
                if self.canvas.last_clicked_pos != (-1, -1):
                    plotstatus = self.canvas.profile_draw(
                        round(event.xdata), round(event.ydata))

    def image_on_release(self, event, pix=None):
        """Actions when image canvas release mouse button."""
        if event.inaxes and len(event.inaxes.get_images()) > 0:
            if self.tool_profile.isChecked():
                if self.canvas.last_clicked_pos != (-1, -1):
                    plotstatus = self.canvas.profile_draw(
                        round(event.xdata), round(event.ydata))
                    self.mouse_pressed = False
                    if plotstatus:
                        self.plot_profile(round(event.xdata), round(event.ydata))

    def image_on_click(self, event):
        """Actions when image canvas is clicked."""
        if event.inaxes and len(event.inaxes.get_images()) > 0:
            if self.tool_profile.isChecked():
                self.canvas.profile_remove()
                self.canvas.draw()
                self.canvas.last_clicked_pos = (
                    round(event.xdata), round(event.ydata))
                self.mouse_pressed = True

    def clicked_profile(self):
        """Refresh image when deactivated profile."""
        if self.tool_profile.isChecked() is False:
            self.canvas.profile_remove()
            self.canvas.draw()

    def plot_profile(self, x2, y2, pix=None):
        """Pop up dialog window with plot of profile."""
        x1 = self.canvas.last_clicked_pos[0]
        y1 = self.canvas.last_clicked_pos[1]
        rr, cc = draw.line(y1, x1, y2, x2)
        profile = self.canvas.current_image[rr, cc]
        len_pix = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        len_pr_pix = len_pix / len(profile)
        if pix is not None:
            len_pr_pix = len_pr_pix * pix
            xtitle = 'pos (mm)'
        else:
            xtitle = 'pos (pix)'
        xvals = np.arange(len(profile)) * len_pr_pix
        dlg = uir.PlotDialog(self.main, title='Image profile')
        dlg.plotcanvas.plot(xvals=[xvals], yvals=[profile],
                            xtitle=xtitle, ytitle='Pixel value',
                            title='', labels=['pixel_values'])
        dlg.exec()


class GenericImageCanvas(FigureCanvasQTAgg):
    """Canvas for display of image."""

    def __init__(self, parent, main):
        self.main = main
        self.parent = parent
        self.fig = matplotlib.figure.Figure(dpi=150)
        self.fig.subplots_adjust(0., 0., 1., 1.)
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent = parent
        self.ax = self.fig.add_subplot(111)
        self.last_clicked_pos = (-1, -1)
        self.profile_length = 20  # assume click drag > length in pix = draw profile
        self.current_image = None

        # default display
        self.img = self.ax.imshow(np.zeros((2, 2)))
        self.ax.cla()
        self.ax.axis('off')

    def profile_draw(self, x2, y2, pix=None):
        """Draw line for profile.

        Parameters
        ----------
        x2 : float
            endpoint x coordinate
        y2 : float
            endpoint y coordinate
        pix : float, optional
            pixelsize. The default is None.

        Returns
        -------
        plotstatus : bool
            True if plot was possible
        """
        self.profile_remove()
        plotstatus = False
        if self.last_clicked_pos != (-1, -1):
            x1 = self.last_clicked_pos[0]
            y1 = self.last_clicked_pos[1]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > self.profile_length:
                self.ax.add_artist(matplotlib.lines.Line2D(
                    [x1, x2], [y1, y2],
                    color='red', linewidth=self.main.vGUI.annotations_line_thick,
                    gid='profile'))
                self.draw()
                plotstatus = True
        return plotstatus

    def profile_remove(self):
        """Clear profile line."""
        if hasattr(self.ax, 'lines'):
            for line in self.ax.lines:
                if line.get_gid() == 'profile':
                    line.remove()
                    break

    def draw(self):
        """Avoid super().draw when figure collapsed by sliders."""
        try:
            super().draw()
        except ValueError:
            pass

class GenericImageToolbarPosVal(QToolBar):
    """Toolbar for showing cursor position and value."""

    def __init__(self, canvas, window):
        super().__init__()

        self.xypos = QLabel('')
        self.xypos.setMinimumWidth(500)
        self.addWidget(self.xypos)
        try:
            self.delta_x = window.vGUI.delta_x
            self.delta_y = window.vGUI.delta_y
        except AttributeError:
            self.delta_x = 0
            self.delta_y = 0

        canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        """When mouse cursor is moving in the canvas."""
        if event.inaxes and len(event.inaxes.get_images()) > 0:
            szImg = event.inaxes.get_images()[0].get_array().shape
            xpos = event.xdata - 0.5 * szImg[0] + self.delta_x
            ypos = event.ydata - 0.5 * szImg[1] + self.delta_y
            xyval = event.inaxes.get_images()[0].get_cursor_data(event)
            try:
                self.xypos.setText(
                    f'xy = ({xpos:.0f}, {ypos:.0f}), val = {xyval:.1f}')
            except TypeError:
                self.xypos.setText('')
        else:
            self.xypos.setText('')


class ImageDisplayWidget(GenericImageWidget):
    """Image display widget."""

    def __init__(self, parent):
        super().__init__(parent, ImageCanvas(self, parent))
        self.main = parent

        tbimg = ImageNavigationToolbar(self.canvas, self.main)
        tbimg2 = GenericImageToolbarPosVal(self.canvas, self.main)
        hlo = QHBoxLayout()
        vlo_tb = QVBoxLayout()
        hlo.addLayout(vlo_tb)

        self.tool_sum = QToolButton()
        self.tool_sum.setToolTip(
            'Toggle to display sum of marked images, press again to display average.')
        self.tool_sum.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}sigma.png'))
        self.tool_sum.clicked.connect(self.clicked_sum)
        self.tool_sum.setCheckable(True)
        actEditAnnotations = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit annotations', self)
        actEditAnnotations.triggered.connect(self.edit_annotations)
        self.tool_imgsize = QToolButton()
        self.tool_imgsize.setToolTip('Maximize image')
        self.tool_imgsize.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}layout_maximg.png'))
        self.tool_imgsize.clicked.connect(self.clicked_imgsize)
        self.tool_imgsize.setCheckable(True)
        tbimg.addWidget(self.tool_profile)
        tbimg.addWidget(self.tool_sum)
        tbimg.addAction(actEditAnnotations)
        tbimg.addWidget(self.tool_imgsize)

        vLOimg = QVBoxLayout()
        vLOimg.addWidget(tbimg)
        vLOimg.addWidget(tbimg2)
        vLOimg.addWidget(self.canvas)

        self.setLayout(vLOimg)

        self.mouse_pressed = False

    def image_on_click(self, event):
        """Actions when image canvas is clicked."""
        super().image_on_click(event)
        if event.inaxes and len(event.inaxes.get_images()) > 0:
            self.main.vGUI.last_clicked_pos = (
                    round(event.xdata), round(event.ydata))
            if event.dblclick:
                self.main.centerWidget.set_center_to_clickpos()

    def edit_annotations(self):
        """Pop up dialog to edit annotations settings."""
        dlg = EditAnnotationsDialog(
            annotations=self.main.vGUI.annotations,
            annotations_line_thick=self.main.vGUI.annotations_line_thick,
            annotations_font_size=self.main.vGUI.annotations_font_size)
        res = dlg.exec()
        if res:
            ann, line_thick, font_size = dlg.get_data()
            self.main.vGUI.annotations = ann
            self.main.vGUI.annotations_line_thick = line_thick
            self.main.vGUI.annotations_font_size = font_size
            if self.main.vGUI.active_img_no > -1:
                self.canvas.img_draw()

    def clicked_imgsize(self):
        """Maximize or reset image size."""
        if self.tool_imgsize.isChecked():
            self.tool_imgsize.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}layout_resetimg.png'))
            self.main.set_split_max_img()
        else:
            self.tool_imgsize.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}layout_maximg.png'))
            self.main.reset_split_max_img()

    def clicked_sum(self):
        """Activate or deactive display sum of marked images."""
        if self.main.summed_img is not None and self.main.average_img is False:
            self.main.average_img = True
            self.tool_sum.setChecked(True)
            self.tool_sum.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}xmean.png'))
        if self.tool_sum.isChecked():
            if self.main.average_img:
                self.main.update_summed_img(recalculate_sum=False)
            else:
                self.main.update_summed_img()
        else:
            self.main.reset_summed_img()
            self.canvas.img_draw()

    def plot_profile(self, x2, y2, pix=None):
        """Pop up dialog window with plot of profile."""
        pix = self.main.imgs[self.main.vGUI.active_img_no].pix[0]
        super().plot_profile(x2, y2, pix=pix)


class ImageCanvas(GenericImageCanvas):
    """Canvas for drawing the active DICOM image."""

    def __init__(self, parent, main):
        super().__init__(parent, main)

    def img_is_missing(self):
        """Show message when pixel_data is missing."""
        self.ax.cla()
        self.img = self.ax.imshow(np.zeros((100, 100)))
        self.ax.axis('off')
        at = matplotlib.offsetbox.AnchoredText(
            'Pixel data not found',
            prop=dict(size=14, color='gray'),
            frameon=False, loc='center')
        self.ax.add_artist(at)
        self.draw()

    def img_draw(self):
        """Refresh image."""
        self.ax.cla()

        nparr = self.main.active_img
        WLmin, WLmax = self.main.windowLevelWidget.get_min_max(
            self.main.active_img)
        annotate = self.main.vGUI.annotations

        if len(np.shape(nparr)) == 2:
            self.img = self.ax.imshow(
                nparr, cmap='gray', vmin=WLmin, vmax=WLmax)
        elif len(np.shape(nparr)) == 3:
            # rgb to grayscale NTSC formula
            nparr = (0.299 * nparr[:, :, 0]
                     + 0.587 * nparr[:, :, 1]
                     + 0.114 * nparr[:, :, 2])
            self.img = self.ax.imshow(nparr, cmap='gray')
            annotate = False
        self.ax.axis('off')
        if annotate:
            # central crosshair
            szy, szx = np.shape(nparr)
            if self.main.vGUI.delta_a == 0:
                self.ax.axhline(
                    y=szy*0.5 + self.main.vGUI.delta_y,
                    color='red', linewidth=1., linestyle='--')
                self.ax.axvline(
                    x=szx*0.5 + self.main.vGUI.delta_x,
                    color='red', linewidth=1., linestyle='--')
            else:
                x1, x2, y1, y2 = get_rotated_crosshair(
                    szx, szy,
                    (self.main.vGUI.delta_x,
                     self.main.vGUI.delta_y,
                     self.main.vGUI.delta_a)
                    )
                # NB keep these two lines as first and second in ax.lines
                self.ax.add_artist(matplotlib.lines.Line2D(
                    [0, szx], [y1, y2],
                    color='red', linewidth=1., linestyle='--',
                    gid='axis1'))
                self.ax.add_artist(matplotlib.lines.Line2D(
                    [x1, x2], [szy, 0],
                    color='red', linewidth=1., linestyle='--',
                    gid='axis2'))
            # DICOM annotations
            if self.parent.tool_sum.isChecked():
                annot_text = (
                    ['Average image', ''] if self.main.average_img
                    else ['Summed image', '']
                    )
            else:
                annot_text = self.main.imgs[
                    self.main.vGUI.active_img_no].annotation_list
            at = matplotlib.offsetbox.AnchoredText(
                '\n'.join(annot_text),
                prop=dict(size=self.main.vGUI.annotations_font_size, color='red'),
                frameon=False, loc='upper left')
            self.ax.add_artist(at)
            self.roi_draw()
        else:
            self.draw()
        self.current_image = nparr

    '''
    def profile_draw(self, x2, y2):
        """Draw line for profile."""
        self.profile_remove()

        plotstatus = False

        if self.main.vGUI.last_clicked_pos != (-1, -1):
            x1 = self.main.vGUI.last_clicked_pos[0]
            y1 = self.main.vGUI.last_clicked_pos[1]
            if np.sqrt((x2-x1)**2 + (y2-y1)**2) > self.profile_length:
                self.ax.add_artist(matplotlib.lines.Line2D(
                    [x1, x2], [y1, y2],
                    color='red', linewidth=self.main.vGUI.annotations_line_thick,
                    gid='profile'))
                self.draw()
                plotstatus = True

        return plotstatus

    def profile_remove(self):
        """Clear profile line."""
        if hasattr(self.ax, 'lines'):
            for line in self.ax.lines:
                if line.get_gid() == 'profile':
                    line.remove()
                    break
    '''

    def roi_draw(self):
        """Update ROI countours on image."""
        if hasattr(self, 'contours'):
            for contour in self.contours:
                for coll in contour.collections:
                    try:
                        coll.remove()
                    except ValueError:
                        pass
        if hasattr(self, 'scatters'):
            for s in self.scatters:
                try:
                    s.remove()
                except ValueError:
                    pass
        if hasattr(self.ax, 'lines'):
            for l_idx, line in enumerate(self.ax.lines):
                if l_idx > 1:
                    try:
                        line.remove()
                    except ValueError:
                        pass

        self.ax.texts.clear()

        if self.main.current_roi is not None:

            linewidth = self.main.vGUI.annotations_line_thick
            fontsize = self.main.vGUI.annotations_font_size

            if self.main.current_test in ['Hom', 'Gho']:
                colors = ['red', 'blue', 'green', 'yellow', 'cyan']
                self.contours = []
                for i in range(5):
                    mask = np.where(self.main.current_roi[i], 0, 1)
                    contour = self.ax.contour(
                        mask, levels=[0.9],
                        colors=colors[i], alpha=0.5, linewidths=linewidth)
                    self.contours.append(contour)

            elif self.main.current_test == 'CTn':
                self.contours = []
                ctn_table = self.main.current_paramset.ctn_table
                for i in range(len(ctn_table.materials)):
                    mask = np.where(self.main.current_roi[i], 0, 1)
                    contour = self.ax.contour(
                        mask, levels=[0.9],
                        colors='red', alpha=0.5, linewidths=linewidth)
                    self.contours.append(contour)
                    mask_pos = np.where(mask == 0)
                    xpos = np.mean(mask_pos[1])
                    ypos = np.mean(mask_pos[0])
                    if np.isfinite(xpos) and np.isfinite(ypos):
                        self.ax.text(xpos, ypos, ctn_table.materials[i],
                                     fontsize=fontsize, color='red')

                if len(self.main.current_roi) == 2 * len(ctn_table.materials):
                    # draw search rois
                    nroi = len(ctn_table.materials)
                    for i in range(nroi, 2 * nroi):
                        mask = np.where(self.main.current_roi[i], 0, 1)
                        contour = self.ax.contour(
                            mask, levels=[0.9],
                            colors='blue', alpha=0.5, linewidths=linewidth)
                        self.contours.append(contour)

            elif self.main.current_test == 'Sli':
                h_colors = ['k', 'g', 'pink', 'c']
                v_colors = ['b', 'r']
                search_margin = self.main.current_paramset.sli_search_width
                background_length = self.main.current_paramset.sli_background_width
                pix = self.main.imgs[self.main.vGUI.active_img_no].pix
                background_length = background_length / pix[0]
                for l_idx, line in enumerate(self.main.current_roi['h_lines']):
                    y1, x1, y2, x2 = line
                    self.ax.add_artist(matplotlib.lines.Line2D(
                        [x1, x2], [y1, y2],
                        color=h_colors[l_idx], linewidth=linewidth,
                        linestyle='dotted',
                        ))
                    self.ax.add_artist(matplotlib.lines.Line2D(
                        [x1, x2], [y1 - search_margin, y2 - search_margin],
                        color=h_colors[l_idx], linewidth=1.
                        ))
                    self.ax.add_artist(matplotlib.lines.Line2D(
                        [x1, x2], [y1 + search_margin, y2 + search_margin],
                        color=h_colors[l_idx], linewidth=1.
                        ))
                    self.ax.add_artist(matplotlib.lines.Line2D(
                        [x1 + background_length, x1 + background_length],
                        [y1 - search_margin, y1 + search_margin],
                        color=h_colors[l_idx], linewidth=1.
                        ))
                    self.ax.add_artist(matplotlib.lines.Line2D(
                        [x2 - background_length, x2 - background_length],
                        [y2 - search_margin, y2 + search_margin],
                        color=h_colors[l_idx], linewidth=1.
                        ))
                for l_idx, line in enumerate(self.main.current_roi['v_lines']):
                    y1, x1, y2, x2 = line
                    self.ax.add_artist(matplotlib.lines.Line2D(
                        [x1, x2], [y1, y2],
                        color=v_colors[l_idx], linewidth=linewidth,
                        linestyle='dotted',
                        ))
                    self.ax.add_artist(matplotlib.lines.Line2D(
                        [x1 - search_margin, x2 - search_margin], [y1, y2],
                        color=v_colors[l_idx], linewidth=1.
                        ))
                    self.ax.add_artist(matplotlib.lines.Line2D(
                        [x1 + search_margin, x2 + search_margin], [y1, y2],
                        color=v_colors[l_idx], linewidth=1.
                        ))
                    self.ax.add_artist(matplotlib.lines.Line2D(
                        [x1 - search_margin, x1 + search_margin],
                        [y1 + background_length, y1 + background_length],
                        color=v_colors[l_idx], linewidth=1.
                        ))
                    self.ax.add_artist(matplotlib.lines.Line2D(
                        [x2 - search_margin, x2 + search_margin],
                        [y2 - background_length, y2 - background_length],
                        color=v_colors[l_idx], linewidth=1.
                        ))

            elif self.main.current_test == 'MTF':
                if isinstance(self.main.current_roi, list):
                    this_roi = self.main.current_roi[1]  # bead show background rim
                else:
                    this_roi = self.main.current_roi

                mask = np.where(this_roi, 0, 1)
                self.contours = []
                contour = self.ax.contour(
                    mask, levels=[0.9],
                    colors='red', alpha=0.5, linewidths=linewidth)
                self.contours.append(contour)

            elif self.main.current_test == 'Uni':
                colors = ['red', 'blue']
                self.contours = []
                for i in range(2):
                    mask = np.where(self.main.current_roi[i], 0, 1)
                    contour = self.ax.contour(
                        mask, levels=[0.9],
                        colors=colors[i], alpha=0.5, linewidths=linewidth)
                    self.contours.append(contour)

            elif self.main.current_test == 'SNI':
                colors = ['red', 'blue']
                self.contours = []
                # 2 large
                for i in range(2):
                    mask = np.where(self.main.current_roi[i+1], 0, 1)
                    contour = self.ax.contour(
                        mask, levels=[0.9],
                        colors=colors[i], alpha=0.5, linewidths=linewidth)
                    self.contours.append(contour)
                # first 2 small with contour, else only label S1..S6
                colors = ['green', 'cyan']
                for i in range(6):
                    mask = np.where(self.main.current_roi[i+3], 0, 1)
                    color = 'yellow'
                    if i < 2:
                        contour = self.ax.contour(
                            mask, levels=[0.9],
                            colors=colors[i], alpha=0.5, linewidths=linewidth)
                        self.contours.append(contour)
                        color = colors[i]
                    mask_pos = np.where(mask == 0)
                    xpos = np.mean(mask_pos[1])
                    ypos = np.mean(mask_pos[0])
                    if np.isfinite(xpos) and np.isfinite(ypos):
                        self.ax.text(xpos-fontsize, ypos+fontsize, f'S{i}',
                                     fontsize=fontsize, color=color)

            elif self.main.current_test == 'PIU':
                mask = np.where(self.main.current_roi, 0, 1)
                self.contours = []
                contour = self.ax.contour(
                    mask, levels=[0.9],
                    colors='red', alpha=0.5, linewidths=linewidth)
                self.contours.append(contour)
                # display min, max pos
                self.scatters = []
                min_idx, max_idx = get_min_max_pos_2d(
                    self.main.active_img, self.main.current_roi)
                scatter = self.ax.scatter(min_idx[1], min_idx[0], s=40,
                                          c='blue', marker='D')
                self.scatters.append(scatter)
                self.ax.text(min_idx[1], min_idx[0]+10,
                             'min', fontsize=fontsize, color='blue')
                scatter = self.ax.scatter(max_idx[1], max_idx[0], s=40,
                                          c='fuchsia', marker='D')
                self.scatters.append(scatter)
                self.ax.text(max_idx[1], max_idx[0]+10,
                             'max', fontsize=fontsize, color='fuchsia')

            else:
                mask = np.where(self.main.current_roi, 0, 1)
                self.contours = []
                contour = self.ax.contour(
                    mask, levels=[0.9],
                    colors='red', alpha=0.5, linewidths=linewidth)
                self.contours.append(contour)

        self.draw()


class ImageNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib navigation toolbar with some modifications."""

    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        for x in self.actions():
            if x.text() == 'Subplots':
                self.removeAction(x)

    def set_message(self, s):
        """Hide cursor position and value text."""
        pass


class ImageExtraToolbar(QToolBar):
    """Extra toolbar for showing more cursor position and value."""

    def __init__(self, canvas, window):
        super().__init__()

        self.xypos = QLabel('')
        self.xypos.setMinimumWidth(500)
        self.addWidget(self.xypos)
        try:
            self.delta_x = window.vGUI.delta_x
            self.delta_y = window.vGUI.delta_y
        except AttributeError:
            self.delta_x = 0
            self.delta_y = 0

        canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        """When mouse cursor is moving in the canvas."""
        if event.inaxes and len(event.inaxes.get_images()) > 0:
            szImg = event.inaxes.get_images()[0].get_array().shape
            xpos = event.xdata - 0.5 * szImg[0] + self.delta_x
            ypos = event.ydata - 0.5 * szImg[1] + self.delta_y
            xyval = event.inaxes.get_images()[0].get_cursor_data(event)
            try:
                self.xypos.setText(
                    f'xy = ({xpos:.0f}, {ypos:.0f}), val = {xyval:.1f}')
            except TypeError:
                self.xypos.setText('')
        else:
            self.xypos.setText('')


class CenterWidget(QGroupBox):
    """Widget with groupbox holding center/rotation display."""

    def __init__(self, parent):
        super().__init__('Center / rotation')
        self.main = parent
        self.setFont(uir.FontItalic())

        self.valDeltaX = QSpinBox()
        self.valDeltaY = QSpinBox()
        self.valDeltaA = QDoubleSpinBox()
        self.chkDeltaUse = QCheckBox('Use offset')

        self.valDeltaX.setRange(-256, 256)
        self.valDeltaX.valueChanged.connect(self.update_delta)
        self.valDeltaY.setRange(-256, 256)
        self.valDeltaY.valueChanged.connect(self.update_delta)
        self.valDeltaA = QDoubleSpinBox()
        self.valDeltaA.setRange(-180., 180.)
        self.valDeltaA.setDecimals(1)
        self.valDeltaA.valueChanged.connect(self.update_delta)
        self.chkDeltaUse.setChecked(True)

        self.valDeltaX.setFixedSize(110, 48)
        self.valDeltaX.setAlignment(Qt.AlignCenter)
        urlLeft = f'{os.environ[ENV_ICON_PATH]}arrowLeft.png'
        urlRight = f'{os.environ[ENV_ICON_PATH]}arrowRight.png'
        css = f"""QSpinBox {{
                margin-left: 0px;
                border: 1px solid gray;
                border-radius: 1px;
                max-width: 20px;
                }}
            QSpinBox::up-button  {{
                subcontrol-origin: margin;
                subcontrol-position: center right;
                image: url({urlRight});
                width: 20px;
                right: 1px;
                }}
            QSpinBox::down-button  {{
                subcontrol-origin: margin;
                subcontrol-position: center left;
                image: url({urlLeft});
                width: 20px;
                left: 1px;
                }}"""
        self.valDeltaX.setStyleSheet(css)

        self.valDeltaY.setFixedSize(64, 96)
        self.valDeltaY.setAlignment(Qt.AlignCenter)
        urlUp = f'{os.environ[ENV_ICON_PATH]}arrowUp.png'
        urlDown = f'{os.environ[ENV_ICON_PATH]}arrowDown.png'
        css = f"""QSpinBox {{
                border: 1px solid gray;
                border-radius: 3px;
                max-height: 30px;
                }}
            QSpinBox::up-button  {{
                subcontrol-origin: margin;
                subcontrol-position: center top;
                width: 30px;
                height: 30px;
                right: 3px;
                bottom: 2px;
                image: url({urlUp});
                }}
            QSpinBox::down-button  {{
                subcontrol-origin: ;margin
                subcontrol-position: center bottom;
                width: 30px;
                height: 30px;
                right: 3px;
                bottom: 2px;
                image: url({urlDown});
                }}"""
        self.valDeltaY.setStyleSheet(css)

        tb = QToolBar()
        # tb.setOrientation(Qt.Vertical)
        actSearch = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}search.png'),
            'Search geometric center of mass by threshold', self)
        actSearch.triggered.connect(self.set_center_threshold)
        actSelect = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}selectArrow.png'),
            'Set center to last mouse click position', self)
        actSelect.triggered.connect(self.set_center_to_clickpos)
        actReset = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}resetZero.png'),
            'Reset center and rotation', self)
        actReset.triggered.connect(self.reset_delta)
        tb.addActions([actReset, actSelect, actSearch])

        LOdelta = QHBoxLayout()
        LOdelta.addWidget(QLabel('dx'))
        LOdelta.addWidget(self.valDeltaX)
        LOdelta.addSpacing(20)
        LOdelta.addWidget(QLabel('dy'))
        LOdelta.addWidget(self.valDeltaY)
        LOdelta2 = QHBoxLayout()
        LOdelta2.addWidget(QLabel('da'))
        LOdelta2.addWidget(self.valDeltaA)
        LOdelta2.addWidget(self.chkDeltaUse)

        boxCenterV = QVBoxLayout()
        boxCenterV.addLayout(LOdelta)
        boxCenterV.addLayout(LOdelta2)
        boxCenterV.addWidget(tb)
        self.setLayout(boxCenterV)

    def set_center_to_clickpos(self):
        """Set delta xy to last clicked position."""
        if self.main.active_img is not None:
            szActy, szActx = np.shape(self.main.active_img)
            self.valDeltaX.setValue(
                self.main.vGUI.last_clicked_pos[0] - 0.5 * szActx)
            self.valDeltaY.setValue(
                self.main.vGUI.last_clicked_pos[1] - 0.5 * szActy)
            self.update_delta(validate_pos=False)

    def set_center_threshold(self):
        """Set center position on geometric center of mass + thresholding."""
        if self.main.active_img is not None:
            num, ok = QInputDialog.getInt(self,
                                          "Search center based on threshold",
                                          "Set threshold", value=0)
            if ok:
                masked_img = np.where(self.main.active_img > num, 1, 0)
                if np.amax(masked_img) > 0:

                    center = [np.average(indices)
                              for indices in np.where(
                                      self.main.active_img > num)]
                    szAct = np.shape(self.main.active_img)
                    self.valDeltaX.setValue(center[1] - szAct[1]*0.5)
                    self.valDeltaY.setValue(center[0] - szAct[0]*0.5)
                    self.update_delta()
        else:
            QMessageBox.information(self, 'Information',
                                    'No image loaded to threshold.')

    def update_delta(self, validate_pos=True):
        """Update delta x,y,a - make sure valid values."""
        self.main.vGUI.delta_x = self.valDeltaX.value()
        self.main.vGUI.delta_y = self.valDeltaY.value()
        self.main.vGUI.delta_a = self.valDeltaA.value()

        if self.main.vGUI.annotations and self.main.active_img is not None:
            szy, szx = np.shape(self.main.active_img)
            if self.valDeltaA.value() == 0:
                self.main.wImageDisplay.canvas.ax.lines[0].set_ydata(
                    y=szy * 0.5 + self.valDeltaY.value())
                self.main.wImageDisplay.canvas.ax.lines[1].set_xdata(
                    x=szx * 0.5 + self.valDeltaX.value())
            else:
                x1, x2, y1, y2 = get_rotated_crosshair(
                    szx, szy,
                    (self.main.vGUI.delta_x, self.main.vGUI.delta_y,
                     self.main.vGUI.delta_a))
                self.main.wImageDisplay.canvas.ax.lines[0].set_ydata(
                    [y1, y2])
                self.main.wImageDisplay.canvas.ax.lines[1].set_xdata(
                    [x1, x2])

            self.main.wImageDisplay.canvas.draw()
            self.main.update_roi()

    def reset_delta(self):
        """Reset center displacement and rotation."""
        self.valDeltaX.setValue(0)
        self.valDeltaY.setValue(0)
        self.valDeltaA.setValue(0)
        self.update_delta(validate_pos=False)


class WindowLevelWidget(QGroupBox):
    """Widget with groupbox holding WindowLevel display."""

    def __init__(self, parent):
        super().__init__('Window Level')
        self.main = parent
        self.setFont(uir.FontItalic())

        self.minWL = QSlider(Qt.Horizontal)
        self.maxWL = QSlider(Qt.Horizontal)
        self.minWLlbl = QLabel('-200')
        self.maxWLlbl = QLabel('200')
        self.WindowLevelHistoCanvas = WindowLevelHistoCanvas()

        self.maxWL.setRange(-200, 200)
        self.maxWL.setValue(200)
        self.minWL.setRange(-200, 200)
        self.minWL.setValue(-200)
        self.minWL.sliderReleased.connect(self.correct_window_level_sliders)
        self.maxWL.sliderReleased.connect(self.correct_window_level_sliders)
        self.lbl_center = QLabel('0')
        self.lbl_width = QLabel('400')

        boxWL = QVBoxLayout()

        tbWL = QToolBar()
        self.toolMinMaxWL = QToolButton()
        self.toolMinMaxWL.setToolTip("Set WL to [min,max] of active image")
        self.toolMinMaxWL.setIcon(
            QIcon(f'{os.environ[ENV_ICON_PATH]}minmax.png'))
        self.toolMinMaxWL.clicked.connect(
            lambda: self.clicked_window_level('min_max'))
        self.toolRangeWL = QToolButton()
        self.toolRangeWL.setToolTip(
            "Set WL to [mean-std,mean+std] of active image")
        self.toolRangeWL.setIcon(
            QIcon(f'{os.environ[ENV_ICON_PATH]}range.png'))
        self.toolRangeWL.clicked.connect(
            lambda: self.clicked_window_level('mean_stdev'))
        self.toolDCM_WL = QToolButton()
        self.toolDCM_WL.setToolTip(
            "Set WL as defined in the DICOM header of active image")
        self.toolDCM_WL.setIcon(
            QIcon(f'{os.environ[ENV_ICON_PATH]}fileDCM.png'))
        self.toolDCM_WL.clicked.connect(
            lambda: self.clicked_window_level('dcm'))
        tbWL.addWidget(self.toolMinMaxWL)
        tbWL.addWidget(self.toolRangeWL)
        tbWL.addWidget(self.toolDCM_WL)
        self.chkWLupdate = QCheckBox('Lock WL')
        self.chkWLupdate.toggled.connect(self.update_window_level_mode)
        tbWL.addWidget(self.chkWLupdate)

        hboxSlider = QHBoxLayout()
        boxMin = QVBoxLayout()
        boxMin.addSpacing(20)
        boxMin.addWidget(self.minWLlbl)
        boxMin.addStretch()
        hboxSlider.addLayout(boxMin)
        vboxSlider = QVBoxLayout()
        vboxSlider.addWidget(self.minWL)
        vboxSlider.addWidget(self.maxWL)
        vboxSlider.addWidget(self.WindowLevelHistoCanvas)
        hboxSlider.addLayout(vboxSlider)
        boxMax = QVBoxLayout()
        boxMax.addSpacing(20)
        boxMax.addWidget(self.maxWLlbl)
        boxMax.addStretch()
        hboxSlider.addLayout(boxMax)
        boxWL.addWidget(tbWL)
        boxWL.addLayout(hboxSlider)
        hbox_cw = QHBoxLayout()
        hbox_cw.addStretch()
        hbox_cw.addWidget(QLabel('C: '))
        hbox_cw.addWidget(self.lbl_center)
        hbox_cw.addSpacing(20)
        hbox_cw.addWidget(QLabel('W: '))
        hbox_cw.addWidget(self.lbl_width)
        hbox_cw.addStretch()
        boxWL.addLayout(hbox_cw)

        self.setLayout(boxWL)

        self.update_window_level_mode()

    def update_window_level_mode(self):
        """Set and unset lock on window level when selecting a new image."""
        if self.chkWLupdate.isChecked():
            self.toolMinMaxWL.setCheckable(False)
            self.toolRangeWL.setCheckable(False)
            self.toolDCM_WL.setCheckable(False)
        else:
            self.toolMinMaxWL.setCheckable(True)
            self.toolRangeWL.setCheckable(True)
            self.toolDCM_WL.setCheckable(True)
            self.toolDCM_WL.setChecked(True)
            self.set_window_level('dcm')

    def get_min_max(self, img):
        """Get lower and upper window level based on image.

        Parameters
        ----------
        img : nparray
            active image

        Returns
        -------
        minWL : int
            lower window level
        maxWL : TYPE
            upper window level
        """
        if self.chkWLupdate.isChecked() is False:
            if self.toolMinMaxWL.isChecked():
                self.set_window_level('min_max')
            elif self.toolRangeWL.isChecked():
                self.set_window_level('mean_stdev')
            else:
                self.set_window_level('dcm')

        return (self.minWL.value(), self.maxWL.value())

    def clicked_window_level(self, arg):
        """When one of the window level toolbuttons is toggled."""
        if self.chkWLupdate.isChecked() is False:
            # unCheck others, check selected
            if arg == 'min_max':
                self.toolMinMaxWL.setChecked(True)
                self.toolRangeWL.setChecked(False)
                self.toolDCM_WL.setChecked(False)
            elif arg == 'mean_stdev':
                self.toolMinMaxWL.setChecked(False)
                self.toolRangeWL.setChecked(True)
                self.toolDCM_WL.setChecked(False)
            elif arg == 'dcm':
                self.toolMinMaxWL.setChecked(False)
                self.toolRangeWL.setChecked(False)
                self.toolDCM_WL.setChecked(True)

        self.set_window_level(arg)

    def set_window_level(self, arg):
        """Set window level based on active image."""
        if self.main.active_img is not None:
            minval = 0
            maxval = 0
            if arg == 'dcm':
                imgno = self.main.vGUI.active_img_no
                if self.main.imgs[imgno].window_width > 0:
                    minval = self.main.imgs[imgno].window_center - \
                        0.5*self.main.imgs[imgno].window_width
                    maxval = self.main.imgs[imgno].window_center + \
                        0.5*self.main.imgs[imgno].window_width
            elif arg == 'min_max':
                minval = np.amin(self.main.active_img)
                maxval = np.amax(self.main.active_img)
            elif arg == 'mean_stdev':
                meanval = np.mean(self.main.active_img)
                stdval = np.std(self.main.active_img)
                minval = meanval-stdval
                maxval = meanval+stdval

            if maxval == minval:
                minval = np.amin(self.main.active_img)
                maxval = np.amax(self.main.active_img)

            self.update_window_level(minval, maxval)
            self.main.wImageDisplay.canvas.img.set_clim(vmin=minval, vmax=maxval)
            self.main.wImageDisplay.canvas.draw()

    def update_window_level(self, minval, maxval):
        """Update GUI for window level sliders."""
        self.minWL.setValue(minval)
        self.maxWL.setValue(maxval)
        self.minWLlbl.setText(f'{minval:.0f}')
        self.maxWLlbl.setText(f'{maxval:.0f}')
        self.lbl_center.setText(f'{0.5*(minval+maxval):.0f}')
        self.lbl_width.setText(f'{(maxval-minval):.0f}')

    def correct_window_level_sliders(self):
        """Make sure minWL < maxWL after user input."""
        if self.maxWL.value() < self.minWL.value():
            maxval = self.minWL.value()
            self.update_window_level(self.maxWL.value(), maxval)
        else:
            self.update_window_level(self.minWL.value(), self.maxWL.value())

        if self.main.active_img is not None:
            self.main.wImageDisplay.canvas.img.set_clim(
                vmin=self.minWL.value(), vmax=self.maxWL.value())
            self.main.wImageDisplay.canvas.draw()


class WindowLevelHistoCanvas(FigureCanvasQTAgg):
    """Canvas for display of histogram for the active image."""

    def __init__(self):
        self.fig = matplotlib.figure.Figure(figsize=(2, 1))
        self.fig.subplots_adjust(0., 0., 1., 1.)
        FigureCanvasQTAgg.__init__(self, self.fig)

    def plot(self, nparr, WLmin=-200, WLmax=200):
        """Refresh histogram."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        amin, amax = (np.amin(nparr), np.amax(nparr))
        try:
            hist, bins = np.histogram(nparr, bins=np.arange(
                amin, amax, (amax - amin)/100.))
            self.ax.plot(bins[:-1], hist)
            self.ax.axis('off')

            self.draw()
        except ValueError:
            pass


class SelectTemplateWidget(QWidget):
    """General widget for inheritance to QuickTest and Paramset widgets."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent
        self.edited = False
        self.lbl_edit = QLabel('')
        self.cbox_template = QComboBox()

    def flag_edit(self, flag=True):
        """Indicate some change."""
        if flag:
            self.edited = True
            self.lbl_edit.setText('*')
        else:
            self.edited = False
            self.lbl_edit.setText('')

    def fill_template_list(self, set_label=''):
        """Fill list of templates for current modality."""
        self.cbox_template.blockSignals(True)
        self.cbox_template.clear()
        labels = [qt.label for qt in self.modality_dict[self.main.current_modality]]
        if len(labels) > 0:
            if self.fname == 'quicktest_templates':
                labels.insert(0, '')
            self.cbox_template.addItems(labels)
            if set_label in labels:
                set_index = labels.index(set_label)
            else:
                set_index = 0
            self.cbox_template.setCurrentIndex(set_index)
        self.cbox_template.blockSignals(False)

    def add_current_template(self):
        """Add current template."""
        if self.fname == 'quicktest_templates':
            self.get_current_template()
        elif self.fname == 'paramsets':
            self.current_template = self.main.current_paramset
        status = uir.add_to_modality_dict(
            self.modality_dict,
            self.main.current_modality,
            self.current_template,
            parent_widget=self.main
            )
        if status:
            self.save(new_added=True)

    def save_current_template(self):
        """Overwrite selected QuickTest Template if any, else add new."""
        if self.modality_dict[self.main.current_modality][0].label == '':
            self.add_current_template()
        else:
            template_id = self.cbox_template.currentIndex()
            if self.fname == 'quicktest_templates':
                self.get_current_template()
                template_id -= 1  # first is empty
            elif self.fname == 'paramsets':
                self.current_template = self.main.current_paramset
            self.modality_dict[
                self.main.current_modality][template_id] = copy.deepcopy(
                    self.current_template)
            self.save()

    def save(self, new_added=False):
        """Save to file."""
        proceed = cff.test_config_folder(self)
        if proceed:
            proceed, errmsg = cff.check_save_conflict(self.fname, self.lastload)
            if errmsg != '':
                proceed = uir.proceed_question(self, errmsg)
            if proceed:
                ok, path = cff.save_settings(
                    self.modality_dict, fname=self.fname)
                if ok:
                    self.lbl_edit.setText('')
                    self.lastload = time()
                    self.flag_edit(False)

                    if new_added:
                        self.fill_template_list(set_label=self.current_template.label)

                else:
                    QMessageBox.warning(
                        self, 'Failed saving', f'Failed saving to {path}')

    def ask_to_save_changes(self):
        """Ask user if changes to current parameter set should be saved."""
        reply = QMessageBox.question(
            self, 'Unsaved changes',
            f'Save changes to {self.fname}?',
            QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.save_current_template()
        else:
            self.flag_edit(False)


class SelectQuickTestWidget(SelectTemplateWidget):
    """Widget for selecting and saving QuickTest templates."""

    def __init__(self, parent):
        super().__init__(parent)

        self.fname = 'quicktest_templates'
        self.modality_dict = self.main.quicktests
        self.current_template = self.main.current_quicktest

        self.gbQT = QGroupBox('QuickTest')

        hLO = QHBoxLayout()
        self.setLayout(hLO)

        self.gbQT.setCheckable(True)
        self.gbQT.setChecked(False)
        self.gbQT.toggled.connect(self.update_current_template)
        self.gbQT.setFont(uir.FontItalic())
        self.cbox_template.setMinimumWidth(150)
        self.cbox_template.currentIndexChanged.connect(self.update_current_template)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('QuickTest template'))
        hbox.addWidget(self.cbox_template)
        hbox.addWidget(self.lbl_edit)

        actAddQT = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add current test pattern as QuickTest', self)
        actAddQT.triggered.connect(self.add_current_template)

        actSaveQT = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Overwrite current test pattern as QuickTest', self)
        actSaveQT.triggered.connect(self.save_current_template)
        if self.main.save_blocked:
            actSaveQT.setEnabled(False)

        actSettingsQT = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}gears.png'),
            'Edit/manage QuickTest templates', self)
        actSettingsQT.triggered.connect(
            lambda: self.main.run_settings(initial_view='QuickTest templates'))

        actExecQT = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}play.png'),
            'Run QuickTest', self)
        actExecQT.triggered.connect(self.run_quicktest)

        actClipQT = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}copy.png'),
            'Copy QuickTest results to clipboard', self)
        actClipQT.triggered.connect(self.extract_results)

        tb = QToolBar()
        tb.addActions([actAddQT, actSaveQT, actSettingsQT])
        tb.addSeparator()
        tb.addActions([actExecQT, actClipQT])

        hbox.addWidget(tb)
        hbox.addStretch()
        self.gbQT.setLayout(hbox)
        hLO.addWidget(self.gbQT)

        self.lastload = time()

        self.fill_template_list()

    def update_current_template(self):
        """Set current_template according to selected label."""
        template_id = self.cbox_template.currentIndex()
        if template_id == 0:
            self.current_template = cfc.QuickTestTemplate()
        else:
            self.current_template = copy.deepcopy(
                self.modality_dict[self.main.current_modality][template_id - 1])
        self.set_current_template_to_imgs()
        self.main.treeFileList.update_file_list()

    def set_current_template_to_imgs(self):
        """Set image-dict values according to current template."""
        for i, d in enumerate(self.main.imgs):
            try:
                d.marked_quicktest = self.current_template.tests[i]
                d.quicktest_image_name = \
                    self.current_template.image_names[i]
                d.quicktest_group_name = \
                    self.current_template.group_names[i]
            except IndexError:
                d.marked_quicktest = []
                d.quicktest_image_name = ''
                d.quicktest_group_name = ''
                self.flag_edit(True)

    def get_current_template(self):
        """Fill current_template with values for imgs."""
        lbl = self.current_template.label
        self.current_template = cfc.QuickTestTemplate(label=lbl)
        for d in self.main.imgs:
            self.current_template.add_index(
                test_list=d.marked_quicktest,
                image_name=d.quicktest_image_name,
                group_name=d.quicktest_group_name
                )

    def run_quicktest(self):
        """Run quicktest with current settings."""
        self.get_current_template()
        self.main.current_quicktest = self.current_template
        calculate_qc(self.main)

    def extract_results(self):
        """Extract result values according to paramset.output to clipboard."""
        value_list, header_list = quicktest_output(self.main)
        date_str = self.main.imgs[0].acq_date  # yyyymmdd to dd.mm.yyyy
        date_str_out = date_str[6:8] + '.' + date_str[4:6] + '.' + date_str[0:4]
        value_list = [date_str_out] + value_list
        if self.main.current_paramset.output.include_header:
            header_list = ['Date'] + header_list
            df = pd.DataFrame([value_list], columns=header_list)
            df.to_clipboard(index=False, excel=True, sep=None)
        else:
            df = pd.DataFrame([value_list])
            df.to_clipboard(index=False, excel=True, sep=None)
        self.main.statusBar.showMessage('Values in clipboard', 2000)


class SelectParamsetWidget(SelectTemplateWidget):
    """Widget for selecting and saving parameter sets."""

    def __init__(self, parent):
        super().__init__(parent)
        self.main = parent
        self.fname = 'paramsets'
        self.modality_dict = self.main.paramsets
        self.current_template = self.main.current_paramset

        hLO = QHBoxLayout()
        self.setLayout(hLO)

        hLO.addWidget(QLabel('Parameter set:'))
        self.cbox_template.setMinimumWidth(150)
        self.cbox_template.currentIndexChanged.connect(
            self.main.update_paramset)
        hLO.addWidget(self.cbox_template)
        hLO.addWidget(self.lbl_edit)

        actAddParam = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Save current parameters as new parameter set', self)
        actAddParam.triggered.connect(self.add_current_template)

        actSaveParam = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Overwrite current parameter set', self)
        actSaveParam.triggered.connect(self.save_current_template)
        if self.main.save_blocked:
            actSaveParam.setEnabled(False)

        actSettingsParam = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}gears.png'),
            'Manage parameter sets', self)
        actSettingsParam.triggered.connect(
            lambda: self.main.run_settings(initial_view='Parameter sets'))

        tb = QToolBar()
        tb.addActions([actAddParam, actSaveParam, actSettingsParam])
        hLO.addWidget(tb)

        hLO.addStretch()

        self.lastload = time()
        self.fill_template_list()


class ResultTableWidget(QWidget):
    """Results table widget."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent

        hlo = QHBoxLayout()
        self.setLayout(hlo)
        vlo_tb = QVBoxLayout()
        hlo.addLayout(vlo_tb)

        actCopy = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}copy.png'),
            'Copy table to clipboard', self)
        actCopy.triggered.connect(self.copy_table)
        tb = QToolBar()
        tb.setOrientation(Qt.Vertical)
        tb.addActions([actCopy])
        vlo_tb.addWidget(tb)
        self.tb_copy = uir.ToolBarTableExport(
            self, parameters_output=self.main.current_paramset.output)
        vlo_tb.addWidget(self.tb_copy)
        vlo_tb.addStretch()

        self.result_table = ResultTable(self.main)
        hlo.addWidget(self.result_table)

    def copy_table(self):
        """Copy contents of table to clipboard."""
        decimal_mark = '.'
        if self.tb_copy.tool_decimal.isChecked():
            decimal_mark = ','
        values = [
            val_2_str(
                col,
                decimal_mark=decimal_mark)
            for col in self.result_table.values]

        if self.tb_copy.tool_header.isChecked():  # insert headers
            if self.result_table.row_labels[0] == '':
                for i in range(len(values)):
                    values[i].insert(0, self.result_table.col_labels[i])
            else:
                # row headers true headers
                values.insert(0, self.result_table.row_labels)

        if self.tb_copy.tool_transpose.isChecked() is False:
            values = np.array(values).T.tolist()

        df = pd.DataFrame(values)
        df.to_clipboard(index=False, excel=True, header=None)
        self.main.statusBar.showMessage('Values in clipboard', 2000)


class ResultTable(QTableWidget):
    """Results table.

    Parameters
    ----------
    parent : MainWindow
        for link to active image
    """

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.linked_image_list = True
        self.cellClicked.connect(self.cell_selected)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.values = [[]]  # always as columns, converted if input is rows
        self.row_labels = []
        self.col_labels = []

    def cell_selected(self):
        """Set new active image when current cell changed."""
        if self.linked_image_list:
            self.main.set_active_img(self.currentRow())

    def fill_table(self, row_labels=[], col_labels=[],
                   values_cols=[[]], values_rows=[[]],
                   linked_image_list=True):
        """Populate table.

        Parameters
        ----------
        row_labels : list(str)
            if empty list, none will show
        col_labels : list(str)
            if empty list, numbers shown
        values_cols : list(list(str/float/int))
            one list for each column of values
        values_rows : list(list(str/float/int))
            one list for each row of values
        linked_image_list : bool
            selected table row also change the selection in image list.
            Default is True
        """
        self.clear()
        if values_rows == [[]]:
            n_cols = len(values_cols)
            n_rows = len(row_labels)
        else:
            n_cols = len(col_labels)
            n_rows = len(values_rows)
        self.setColumnCount(n_cols)
        self.setRowCount(n_rows)

        self.row_labels = (
            row_labels if len(row_labels) > 0 else [''] * n_rows)
        self.col_labels = (
            col_labels if len(col_labels) > 0 else [
                str(i) for i in range(n_cols)])

        self.linked_image_list = linked_image_list
        self.setHorizontalHeaderLabels(self.col_labels)
        if len(row_labels) > 0:
            self.setVerticalHeaderLabels(self.row_labels)
            self.verticalHeader().setVisible(True)
        else:
            self.verticalHeader().setVisible(False)

        if values_cols == [[]]:
            if values_rows != [[]]:
                # convert rows to columns for better formatting (precision)
                for r in range(n_rows):
                    if len(values_rows[r]) == 0:
                        values_rows[r] = [None] * n_cols
                values_cols = []
                for c in range(n_cols):
                    values_cols.append([row[c] for row in values_rows])
        if values_cols != [[]]:
            for c in range(len(values_cols)):
                this_col = val_2_str(values_cols[c])
                for r in range(len(this_col)):
                    twi = QTableWidgetItem(this_col[r])
                    twi.setTextAlignment(4)
                    self.setItem(r, c, twi)

        self.values = values_cols
        self.resizeColumnsToContents()
        self.resizeRowsToContents()


class ToolMaximizeResults(QToolButton):
    """Toolbutton with to maximize results panel."""

    def __init__(self, main):
        super().__init__()
        self.main = main

        self.setToolTip('Maximize')
        self.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}layout_maximg.png'))
        self.clicked.connect(
            lambda: self.main.clicked_resultsize(tool=self))
        self.setCheckable(True)


class ResultPlotWidget(uir.PlotWidget):
    """Widget for display of results as plot."""

    def __init__(self, main, plotcanvas):
        super().__init__(main, plotcanvas)

        tb = QToolBar()
        self.tool_resultsize = ToolMaximizeResults(main)
        tb.addWidget(self.tool_resultsize)
        tb.setOrientation(Qt.Vertical)
        self.hlo.addWidget(tb)


class ResultPlotCanvas(uir.PlotCanvas):
    """Canvas for display of results as plot."""

    def __init__(self, main):
        super().__init__(main)

    def plot(self):
        """Refresh plot."""
        self.ax.cla()
        title = ''
        self.xtitle = 'x'
        ytitle = 'y'
        default_range_x = [None, None]
        default_range_y = [None, None]
        legend_location = 'upper right'
        xvals = []
        yvals = []
        self.curves = []

        zpos_all = [img.zpos for img in self.main.imgs]
        img_nos = []

        if self.main.current_test in [*self.main.results]:
            if self.main.current_test == 'ROI':
                for i, row in enumerate(self.main.results['ROI']['values']):
                    if len(row) > 0:
                        xvals.append(i)
                        yvals.append(row[0])
                curve = {'label': 'Average',
                         'xvals': xvals,
                         'yvals': yvals,
                         'style': '-b'}
                self.curves.append(curve)
                self.xtitle = 'Image index'
                ytitle = 'Average pixel value'
                if self.main.current_modality in ['Xray', 'NM']:
                    self.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
                    self.ax.set_xticks(xvals)

            elif self.main.current_test == 'HUw':
                for i, row in enumerate(self.main.results['HUw']['values']):
                    if len(row) > 0:
                        img_nos.append(i)
                        xvals.append(zpos_all[i])
                        yvals.append(row[0])
                curve = {'label': 'Average HU',
                         'xvals': xvals,
                         'yvals': yvals,
                         'style': '-r'}
                self.curves.append(curve)
                self.xtitle = 'zpos (mm)'
                if None in xvals:
                    xvals = img_nos
                    self.xtitle = 'Image number'
                ytitle = 'Average HU'
                tolmax = {'label': 'tolerance max',
                          'xvals': [min(xvals), max(xvals)],
                          'yvals': [4, 4],
                          'style': '--k'}
                tolmin = tolmax.copy()
                tolmin['yvals'] = [-4, -4]
                tolmin['label'] = 'tolerance min'
                self.curves.append(tolmin)
                self.curves.append(tolmax)
                default_range_y = [-6, 6]

            elif self.main.current_test == 'Hom':

                if self.main.current_modality == 'CT':
                    title = 'Difference (HU) from center'
                    yvals12 = []
                    yvals15 = []
                    yvals18 = []
                    yvals21 = []
                    for i, row in enumerate(self.main.results['Hom']['values']):
                        if len(row) > 0:
                            img_nos.append(i)
                            xvals.append(zpos_all[i])
                            yvals12.append(row[5])
                            yvals15.append(row[6])
                            yvals18.append(row[7])
                            yvals21.append(row[8])
                    self.curves.append(
                        {'label': 'at 12', 'xvals': xvals,
                         'yvals': yvals12, 'style': '-b'})
                    self.curves.append(
                        {'label': 'at 15', 'xvals': xvals,
                         'yvals': yvals15, 'style': '-g'})
                    self.curves.append(
                        {'label': 'at 18', 'xvals': xvals,
                         'yvals': yvals18, 'style': '-y'})
                    self.curves.append(
                        {'label': 'at 21', 'xvals': xvals,
                         'yvals': yvals21, 'style': '-c'})
                    self.xtitle = 'zpos (mm)'
                    if None in xvals:
                        xvals = img_nos
                        self.xtitle = 'Image number'
                    ytitle = 'Difference (HU)'
                    tolmax = {'label': 'tolerance max',
                              'xvals': [min(xvals), max(xvals)],
                              'yvals': [4, 4],
                              'style': '--k'}
                    tolmin = tolmax.copy()
                    tolmin['label'] = 'tolerance min'
                    tolmin['yvals'] = [-4, -4]
                    self.curves.append(tolmin)
                    self.curves.append(tolmax)
                    default_range_y = [-6, 6]
                elif self.main.current_modality == 'PET':
                    title = '% difference from mean of all means'
                    yvalsC = []
                    yvals12 = []
                    yvals15 = []
                    yvals18 = []
                    yvals21 = []
                    for i, row in enumerate(self.main.results['Hom']['values']):
                        if len(row) > 0:
                            img_nos.append(i)
                            xvals.append(zpos_all[i])
                            yvalsC.append(row[5])
                            yvals12.append(row[6])
                            yvals15.append(row[7])
                            yvals18.append(row[8])
                            yvals21.append(row[9])
                    self.curves.append(
                        {'label': 'Center', 'xvals': xvals,
                         'yvals': yvalsC, 'style': '-r'})
                    self.curves.append(
                        {'label': 'at 12', 'xvals': xvals,
                         'yvals': yvals12, 'style': '-b'})
                    self.curves.append(
                        {'label': 'at 15', 'xvals': xvals,
                         'yvals': yvals15, 'style': '-g'})
                    self.curves.append(
                        {'label': 'at 18', 'xvals': xvals,
                         'yvals': yvals18, 'style': '-y'})
                    self.curves.append(
                        {'label': 'at 21', 'xvals': xvals,
                         'yvals': yvals21, 'style': '-c'})
                    self.xtitle = 'zpos (mm)'
                    if None in xvals:
                        xvals = img_nos
                        self.xtitle = 'Image number'
                    ytitle = '% difference'
                    tolmax = {'label': 'tolerance max',
                              'xvals': [min(xvals), max(xvals)],
                              'yvals': [5, 5],
                              'style': '--k'}
                    tolmin = tolmax.copy()
                    tolmin['label'] = 'tolerance min'
                    tolmin['yvals'] = [-5, -5]
                    self.curves.append(tolmin)
                    self.curves.append(tolmax)
                    default_range_y = [-7, 7]

            elif self.main.current_test == 'CTn':
                title = 'CT linearity'
                ytitle = 'Relative mass density'
                yvals = self.main.current_paramset.ctn_table.relative_mass_density
                imgno = self.main.vGUI.active_img_no
                xvals = self.main.results['CTn']['values'][imgno]
                self.curves.append(
                    {'label': 'HU mean', 'xvals': xvals,
                     'yvals': yvals, 'style': '-bo'})
                fit_r2 = self.main.results['CTn']['values_sup'][imgno][0]
                fit_b = self.main.results['CTn']['values_sup'][imgno][1]
                fit_a = self.main.results['CTn']['values_sup'][imgno][2]
                yvals = fit_a * np.array(xvals) + fit_b
                self.curves.append(
                    {'label': 'fitted', 'xvals': xvals,
                     'yvals': yvals, 'style': 'b:'}
                    )
                at = matplotlib.offsetbox.AnchoredText(
                    f'$R^2$ = {fit_r2:.4f}', loc='lower right')
                self.ax.add_artist(at)
                self.xtitle = 'HU value'

            elif self.main.current_test == 'Sli':

                if self.main.current_modality == 'CT':
                    title = 'Profiles for slice thickness calculations'
                    imgno = self.main.vGUI.active_img_no
                    details_dict = self.main.results['Sli']['details_dict'][imgno]
                    n_pix = len(details_dict['profiles'][0])
                    xvals = [details_dict['dx'] * i for i in range(n_pix)]
                    if self.main.current_paramset.sli_type == 0:
                        styles = ['k', 'g', 'b', 'r']
                    elif self.main.current_paramset.sli_type == 1:
                        styles = ['k', 'g', 'b', 'r', 'pink', 'c']
                    elif self.main.current_paramset.sli_type == 2:
                        styles = ['b', 'r']
                    for l_idx, profile in enumerate(details_dict['profiles']):
                        self.curves.append({'label': details_dict['labels'][l_idx],
                                            'xvals': xvals,
                                            'yvals': profile,
                                            'style': styles[l_idx]})
                        self.curves.append({
                            'label': '_nolegend_',
                            'xvals': [min(xvals), max(xvals)],
                            'yvals': [details_dict['background'][l_idx]] * 2,
                            'style': ':'+styles[l_idx]})
                        self.curves.append({
                            'label': '_nolegend_',
                            'xvals': [min(xvals), max(xvals)],
                            'yvals': [details_dict['peak'][l_idx]] * 2,
                            'style': ':'+styles[l_idx]})
                        self.curves.append({
                            'label': '_nolegend_',
                            'xvals': [details_dict['start_x'][l_idx],
                                      details_dict['end_x'][l_idx]],
                            'yvals': [details_dict['halfpeak'][l_idx]] * 2,
                            'style': '--'+styles[l_idx]})
                    self.xtitle = 'pos (mm)'
                    ytitle = 'HU'

            elif self.main.current_test == 'Uni':
                plot_idx = self.main.tabNM.uni_plot.currentIndex()
                if plot_idx == 0:
                    title = 'Uniformity result for all images'
                    yvals_iu_ufov = []
                    yvals_du_ufov = []
                    yvals_iu_cfov = []
                    yvals_du_cfov = []
                    xvals = []
                    for i, row in enumerate(self.main.results['Uni']['values']):
                        if len(row) > 0:
                            xvals.append(i)
                            yvals_iu_ufov.append(row[0])
                            yvals_du_ufov.append(row[1])
                            yvals_iu_cfov.append(row[2])
                            yvals_du_cfov.append(row[3])
                    if len(xvals) > 1:
                        self.xtitle = 'Image number'

                    self.curves.append(
                        {'label': 'IU UFOV', 'xvals': xvals,
                         'yvals': yvals_iu_ufov, 'style': '-bo'})
                    self.curves.append(
                        {'label': 'DU UFOV', 'xvals': xvals,
                         'yvals': yvals_du_ufov, 'style': '-ro'})
                    self.curves.append(
                        {'label': 'IU CFOV', 'xvals': xvals,
                         'yvals': yvals_iu_cfov, 'style': ':bo'})
                    self.curves.append(
                        {'label': 'DU CFOV', 'xvals': xvals,
                         'yvals': yvals_du_cfov, 'style': ':ro'})
                    self.xtitle = 'Image number'
                    ytitle = 'Uniformity %'
                    default_range_y = [0, 7]
                elif plot_idx == 1:
                    title = 'Curvature correction check'
                    imgno = self.main.vGUI.active_img_no
                    details_dict = self.main.results['Uni']['details_dict'][imgno]
                    if 'correction_matrix' in details_dict:
                        # averaging central 10% rows/cols
                        temp_img = self.main.active_img
                        corrected_img = details_dict['corrected_image']
                        sz_y, sz_x = corrected_img.shape
                        nx = round(0.05 * sz_x)
                        ny = round(0.05 * sz_y)
                        xhalf = round(sz_x/2)
                        yhalf = round(sz_y/2)
                        prof_y = np.mean(temp_img[:, xhalf-nx:xhalf+nx], axis=1)
                        prof_x = np.mean(temp_img[yhalf-ny:yhalf+ny, :], axis=0)
                        corr_prof_y = np.mean(
                            corrected_img[:, xhalf-nx:xhalf+nx], axis=1)
                        corr_prof_x = np.mean(
                            corrected_img[yhalf-ny:yhalf+ny, :], axis=0)
                        self.curves.append({'label': 'Central 10% rows corrected',
                                            'xvals': np.arange(len(corr_prof_x)),
                                            'yvals': corr_prof_x,
                                            'style': 'r'})
                        self.curves.append({'label': 'Central 10% rows original',
                                            'xvals': np.arange(len(prof_x)),
                                            'yvals': prof_x,
                                            'style': ':r'})
                        self.curves.append({'label': 'Central 10% columns corrected',
                                            'xvals': np.arange(len(corr_prof_y)),
                                            'yvals': corr_prof_y,
                                            'style': 'b'})
                        self.curves.append({'label': 'Central 10% columns original',
                                            'xvals': np.arange(len(prof_y)),
                                            'yvals': prof_y,
                                            'style': ':b'})
                        self.xtitle = 'pixel number'
                        ytitle = 'Average pixel value'
                        legend_location = 'lower center'
                    else:
                        at = matplotlib.offsetbox.AnchoredText(
                            'No curvature correction applied',
                            prop=dict(size=self.main.vGUI.annotations_font_size,
                                      color='red'),
                            frameon=False, loc='upper left')
                        self.ax.add_artist(at)

        if len(self.curves) > 0:
            x_only_int = True
            for curve in self.curves:
                self.ax.plot(curve['xvals'], curve['yvals'],
                             curve['style'], label=curve['label'])
                if x_only_int:
                    xx = list(curve['xvals'])
                    if not isinstance(xx[0], int):
                        x_only_int = False
            if x_only_int:
                self.ax.xaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(integer=True))
            if len(self.curves) > 1:
                self.ax.legend(loc=legend_location)
            if len(title) > 0:
                self.ax.set_title(title)
                self.fig.subplots_adjust(0.15, 0.25, 0.95, 0.85)
            else:
                self.fig.subplots_adjust(0.15, 0.2, 0.95, .95)
            self.ax.set_xlabel(self.xtitle)
            self.ax.set_ylabel(ytitle)
            if None not in default_range_x:
                self.ax.set_xlim(default_range_x)
            if None not in default_range_y:
                self.ax.set_ylim(default_range_y)

        self.draw()


class ResultImageWidget(GenericImageWidget):
    """Results image widget."""

    def __init__(self, main):
        canvas = ResultImageCanvas(self, main)
        super().__init__(main, canvas)
        self.main = main
        tb = ResultImageNavigationToolbar(self.canvas, self)
        hlo = QHBoxLayout()

        tbm = QToolBar()
        self.tool_resultsize = ToolMaximizeResults(self.main)
        tbm.addWidget(self.tool_resultsize)
        tbm.setOrientation(Qt.Vertical)
        tbm.addWidget(self.tool_profile)

        tb_top = QToolBar()
        tb_top.addWidget(GenericImageToolbarPosVal(self.canvas, self))

        hlo.addWidget(tb)
        vlo_mid = QVBoxLayout()
        vlo_mid.addWidget(tb_top)
        vlo_mid.addWidget(self.canvas)
        hlo.addLayout(vlo_mid)
        hlo.addWidget(tbm)
        self.setLayout(hlo)


class ResultImageCanvas(GenericImageCanvas):
    """Canvas for display of results as image."""

    def __init__(self, parent, main):
        super().__init__(parent, main)

    def result_image_draw(self):
        """Refresh result image."""
        self.ax.cla()
        nparr = None
        imgno = self.main.vGUI.active_img_no
        cmap = 'gray'
        min_val = None
        max_val = None
        title = ''
        details_dict = {}

        if self.main.current_test in [*self.main.results]:
            if self.main.current_test == 'Uni':
                if self.main.current_paramset.uni_sum_first:
                    try:
                        details_dict = self.main.results['Uni']['details_dict'][0]
                    except KeyError:
                        pass
                else:
                    try:
                        details_dict = self.main.results['Uni']['details_dict'][imgno]
                    except KeyError:
                        pass
                cmap = 'viridis'
                type_img = self.main.tabNM.uni_result_image.currentIndex()
                if type_img == 0:
                    title = 'Differential uniformity map in UFOV (max in x/y direction)'
                    if 'du_matrix' in details_dict:
                        nparr = details_dict['du_matrix']
                elif type_img == 1:
                    title = 'Processed image minimum 6.4 mm pr pix'
                    if 'matrix' in details_dict:
                        nparr = details_dict['matrix']
                elif type_img == 2:
                    title = 'Curvature corrected image'
                    if 'corrected_image' in details_dict:
                        nparr = details_dict['corrected_image']
                

        self.current_image = nparr
        if nparr is not None:
            if min_val is None:
                min_val = np.min(nparr)
            if max_val is None:
                max_val = np.max(nparr)
            self.img = self.ax.imshow(
                nparr, cmap=cmap, vmin=min_val, vmax=max_val)
            self.ax.set_title(title)
        else:
            self.img = self.ax.imshow(np.zeros((100, 100)))
            at = matplotlib.offsetbox.AnchoredText(
                'No result to display',
                prop=dict(size=14, color='gray'),
                frameon=False, loc='center')
            self.ax.add_artist(at)
        self.ax.axis('off')
        self.draw()


class ResultImageNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib navigation toolbar with some modifications."""

    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        for x in self.actions():
            if x.text() in ['Back', 'Forward', 'Pan', 'Subplots']:
                self.removeAction(x)
        self.setOrientation(Qt.Vertical)

    def set_message(self, s):
        """Hide cursor position and value text."""
        pass


class StatusBar(QStatusBar):
    """Tweeks to QStatusBar."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet("QStatusBar{padding-left: 8px;}")
        self.default_color = self.palette().window().color().name()
        self.message = QLabel('')
        self.message.setAlignment(Qt.AlignCenter)
        self.addWidget(self.message, 1)
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.clearMessage)

    def showMessage(self, txt, timeout=0):
        """Set background color when message is shown."""
        self.setStyleSheet("QStatusBar{background:#6e94c0;}")
        self.message.setText(txt)
        if timeout > 0:
            self.timer.start(timeout)
        else:
            self.timer.start()
        qApp.processEvents()

    def clearMessage(self):
        """Reset background and clear message."""
        self.setStyleSheet(
            "QStatusBar{background:" + self.default_color + ";}")
        self.message.setText('')
        qApp.processEvents()
