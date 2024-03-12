#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for main window of imageQC.

@author: Ellen Wasbo
"""
import sys
import os
import copy
from dataclasses import dataclass
from time import time, ctime
from pathlib import Path
import numpy as np
import pandas as pd
import webbrowser

from PyQt5.QtGui import QIcon, QScreen
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, qApp, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QSplitter, QGroupBox, QTabWidget,
    QLabel, QCheckBox, QButtonGroup, QRadioButton, QComboBox, QMenu, QAction,
    QMessageBox, QFileDialog, QScrollArea
    )
import matplotlib.pyplot as plt

# imageQC block start
from imageQC.ui import ui_main_methods
from imageQC.ui import ui_main_left_side
from imageQC.ui.ui_main_image_widgets import ImageDisplayWidget
from imageQC.ui import ui_main_quicktest_paramset_select
from imageQC.ui import ui_main_test_tabs
from imageQC.ui.ui_main_test_tabs_vendor import ParamsTabVendor
from imageQC.ui import ui_main_result_tabs
from imageQC.ui import rename_dicom
from imageQC.ui import task_based_image_quality
from imageQC.ui.settings import SettingsDialog
from imageQC.ui import automation_wizard
from imageQC.ui import open_multi
from imageQC.ui import open_automation
from imageQC.ui.ui_dialogs import TextDisplay, AboutDialog
from imageQC.ui.tag_patterns import TagPatternEditDialog
from imageQC.ui import reusable_widgets as uir
from imageQC.ui import messageboxes
from imageQC.ui.settings_automation import DashWorker
from imageQC.config import config_func as cff
from imageQC.config.iQCconstants import (
    QUICKTEST_OPTIONS, VERSION,
    ENV_ICON_PATH, ENV_CONFIG_FOLDER, ENV_USER_PREFS_PATH
    )
from imageQC.config import config_classes as cfc
from imageQC.scripts import dcm
from imageQC.scripts.calculate_roi import get_rois
from imageQC.scripts.mini_methods import get_modality_index
# imageQC block end


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
    last_clicked_pos: tuple = (-1, -1)  # x,y
    current_auto_template: str = ''
    # if open auto, run files main, reset on mode change

    panel_width: int = 1400
    panel_height: int = 700
    char_width: int = 7
    annotations: bool = True
    annotations_line_thick: int = 3
    annotations_font_size: int = 14


class MainWindow(QMainWindow):
    """Class main window of imageQC."""

    screenChanged = pyqtSignal(QScreen, QScreen)

    def __init__(self, scX=1400, scY=700, char_width=7, developer_mode=False,
                 warnings=[]):
        super().__init__()
        self.developer_mode = developer_mode  # option to hide some options if True

        self.save_blocked = False
        if os.environ[ENV_USER_PREFS_PATH] == '':
            if os.environ[ENV_CONFIG_FOLDER] == '':
                self.save_blocked = True

        if os.environ[ENV_CONFIG_FOLDER] != '':
            cff.add_user_to_active_users()

        # minimum parameters as for scripts.automation.InputMain
        self.current_modality = 'CT'
        self.current_test = QUICKTEST_OPTIONS['CT'][0]
        self.update_settings()  # sets self.tag_infos (and a lot more)
        plt.rcParams.update({'font.size': self.user_prefs.font_size})
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0
        plt.set_loglevel('WARNING')

        self.current_paramset = copy.deepcopy(self.paramsets[0])
        self.current_quicktest = cfc.QuickTestTemplate()
        self.current_sort_pattern = None
        self.clear_all_images()
        self.current_group_indicators = []
        # string for each image if output set pr group with quicktest (paramset.output)
        self.automation_active = False
        # parameters specific to GUI version
        self.gui = GuiVariables()
        self.gui.panel_width = round(0.48*scX)
        self.gui.panel_height = round(0.86*scY)
        self.gui.char_width = char_width
        self.gui.annotations_line_thick = self.user_prefs.annotations_line_thick
        self.gui.annotations_font_size = self.user_prefs.annotations_font_size

        self.progress_modal = None
        self.status_bar = StatusBar(self)
        self.setStatusBar(self.status_bar)

        self.setWindowTitle('Image QC v ' + VERSION)
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setGeometry(
            round(self.gui.panel_width*0.02),
            round(self.gui.panel_height*0.05),
            round(self.gui.panel_width*2+30),
            round(self.gui.panel_height+50)
            )

        self.tree_file_list = ui_main_left_side.TreeFileList(self)
        self.create_menu_toolBar()
        self.wid_image_display = ImageDisplayWidget(self)
        self.wid_dcm_header = ui_main_left_side.DicomHeaderWidget(self)
        self.wid_window_level = uir.WindowLevelWidget(self)
        self.wid_center = ui_main_left_side.CenterWidget(self)
        self.create_modality_selector()
        self.wid_quicktest = (
            ui_main_quicktest_paramset_select.SelectQuickTestWidget(self))
        self.wid_paramset = ui_main_quicktest_paramset_select.SelectParamsetWidget(self)
        self.create_result_tabs()
        self.create_test_tab_CT()
        # set main layout (left/right)
        bbox = QHBoxLayout()
        self.split_lft_rgt = QSplitter(Qt.Horizontal)
        wid_lft = QWidget()
        wid_rgt = QWidget()
        lo_lft = QVBoxLayout()
        lo_rgt = QVBoxLayout()
        wid_lft.setLayout(lo_lft)
        wid_rgt.setLayout(lo_rgt)
        self.split_lft_rgt.addWidget(wid_lft)
        self.split_lft_rgt.addWidget(wid_rgt)
        bbox.addWidget(self.split_lft_rgt)

        # Fill left box
        self.split_list_rest = QSplitter(Qt.Vertical)
        self.split_img_header = QSplitter(Qt.Vertical)
        self.split_lft_img = QSplitter(Qt.Horizontal)
        wid_win_lev_center = QWidget()
        vlo_win_lev_center = QVBoxLayout()
        vlo_win_lev_center.addWidget(self.wid_window_level)
        vlo_win_lev_center.addWidget(self.wid_center)
        wid_win_lev_center.setLayout(vlo_win_lev_center)

        self.split_list_rest.addWidget(self.tree_file_list)
        self.split_list_rest.addWidget(self.split_img_header)
        self.split_img_header.addWidget(self.split_lft_img)
        self.split_img_header.addWidget(self.wid_dcm_header)
        self.split_lft_img.addWidget(wid_win_lev_center)
        self.split_lft_img.addWidget(self.wid_image_display)

        lo_lft.addWidget(self.split_list_rest)

        # Fill right box
        self.split_rgt_top_rest = QSplitter(Qt.Vertical)
        wid_rgt_top = QWidget()
        vlo_top = QVBoxLayout()
        wid_rgt_top.setLayout(vlo_top)
        hlo_mod = QHBoxLayout()
        hlo_mod.addWidget(self.gb_modality)
        self.btn_read_vendor_file = QCheckBox('Read vendor file')
        self.btn_read_vendor_file.setChecked(False)
        self.btn_read_vendor_file.toggled.connect(self.update_mode)
        hlo_mod.addWidget(self.btn_read_vendor_file)
        vlo_top.addLayout(hlo_mod)
        vlo_top.addWidget(self.wid_quicktest)
        vlo_top.addWidget(self.wid_paramset)
        self.split_rgt_top_rest.addWidget(wid_rgt_top)

        self.split_rgt_mid_btm = QSplitter(Qt.Vertical)
        self.split_rgt_top_rest.addWidget(self.split_rgt_mid_btm)
        self.split_rgt_mid_btm.addWidget(self.stack_test_tabs)
        self.split_rgt_mid_btm.addWidget(self.tab_results)
        lo_rgt.addWidget(self.split_rgt_top_rest)

        self.wid_full = QWidget()
        self.wid_full.setLayout(bbox)
        self.wid_full.setFixedSize(2*self.gui.panel_width, self.gui.panel_height)

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.wid_full)

        self.setCentralWidget(scroll)
        self.reset_split_sizes()

        QTimer.singleShot(1000, lambda: self.create_test_tabs_rest())
        # allow time to show dialog before creating test stacks not to be visualized yet
        self.update_mode()
        if len(warnings) > 0:
            QTimer.singleShot(300, lambda: self.show_warnings(warnings))

    def show_warnings(self, warnings=[]):
        """Show startup warnings when screen initialised."""
        dlg = messageboxes.MessageBoxWithDetails(
            self, title='Warnings',
            msg='Found issues during startup',
            info='See details',
            icon=QMessageBox.Warning,
            details=warnings)
        dlg.exec()

    def clear_all_images(self):
        """Set empty values at startup and when all images closed."""
        self.imgs = []
        self.results = {}
        self.active_img = None  # np.array pixeldata for active image
        self.summed_img = None  # sum of marked images if activated
        self.average_img = False  # true if summed_img should be averaged
        self.current_roi = None
        try:  # if gui all set
            self.wid_image_display.tool_sum.setChecked(False)
            self.wid_image_display.tool_sum.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}sigma.png'))
            self.tree_file_list.update_file_list()
            self.wid_image_display.canvas.ax.cla()
            self.wid_image_display.canvas.ax.axis('off')
            self.wid_image_display.canvas.draw()
            self.refresh_results_display()
        except AttributeError:
            pass

    def update_paramset(self):
        """Fill gui with params from selected paramset."""
        self.current_paramset = copy.deepcopy(
            self.paramsets[self.wid_paramset.cbox_template.currentIndex()])
        if self.results:
            self.reset_results()
        widget = self.stack_test_tabs.currentWidget()
        widget.update_displayed_params()
        self.wid_res_tbl.tb_copy.parameters_output = self.current_paramset.output
        self.wid_res_tbl.tb_copy.update_checked()
        self.update_roi()
        self.wid_paramset.flag_edit(False)

    def open_files(self, file_list=None):
        """Open DICOM files and update GUI."""
        if file_list is False:
            fnames = QFileDialog.getOpenFileNames(
                self, 'Open DICOM files',
                filter="DICOM files (*.dcm *.IMA);;All files (*)")
            file_list = fnames[0]
        if len(file_list) > 0:
            max_progress = 100  # %
            progress_modal = uir.ProgressModal(
                "Calculating...", "Cancel",
                0, max_progress, self, minimum_duration=0)
            new_img_infos, ignored_files, warnings = dcm.read_dcm_info(
                file_list, tag_infos=self.tag_infos,
                tag_patterns_special=self.tag_patterns_special,
                statusbar=self.status_bar, progress_modal=progress_modal)
            progress_modal.setValue(max_progress)
            if ignored_files:
                dlg = messageboxes.MessageBoxWithDetails(
                    self, title='Some files ignored',
                    msg=f'{len(ignored_files)} files ignored missing DICOM image data',
                    info='Try File->Read DICOM header. Ignored files in details.',
                    details=ignored_files, icon=QMessageBox.Information)
                dlg.exec()
            if len(warnings) > 0:
                dlg = messageboxes.MessageBoxWithDetails(
                    self, title='Some files opened with warnings',
                    msg='See details for warning messages',
                    details=warnings, icon=QMessageBox.Warning)
                dlg.exec()
            if len(new_img_infos) > 0:
                self.update_on_new_images(new_img_infos)

    def open_multi(self):
        """Start open advanced dialog."""
        dlg = open_multi.OpenMultiDialog(self)
        res = dlg.exec()
        if res:
            if len(dlg.wid.open_imgs) > 0:
                self.update_on_new_images(dlg.wid.open_imgs)

    def add_dummy(self, n_dummies=1):
        """Add place-holder / dummy image when missing images for QuickTest temp."""
        if not n_dummies:
            n_dummies = 1
        dummy_list = [dcm.DcmInfoGui(modality=self.current_modality)] * n_dummies
        self.update_on_new_images(dummy_list, append=True)

    def update_on_new_images(self, new_img_info_list, append=False):
        """Update GUI on new images."""
        n_img_before = len(self.imgs)
        if self.chk_append.isChecked():
            append = True
        if append:
            self.imgs = self.imgs + new_img_info_list
            self.update_results(n_added_imgs=len(new_img_info_list))
        else:
            self.imgs = new_img_info_list
            self.results = {}

        if append is False or n_img_before == 0:
            self.gui.active_img_no = 0
            # update GUI according to first image
            if self.current_modality != self.imgs[0].modality:
                self.current_modality = self.imgs[0].modality
                self.update_mode()
            if self.wid_window_level.tb_wl.chk_wl_update.isChecked() is False:
                self.wid_window_level.tb_wl.set_window_level('dcm', set_tools=True)

        if self.wid_quicktest.gb_quicktest.isChecked():
            self.wid_quicktest.set_current_template_to_imgs()

        if self.summed_img is not None:
            self.reset_summed_img()
        self.tree_file_list.update_file_list()
        self.current_sort_pattern = None

    def read_header(self):
        """View file as header."""
        fname = QFileDialog.getOpenFileName(
            self, 'Read DICOM header',
            filter="DICOM files (*.dcm *.IMA);;All files (*)")
        filename = fname[0]
        if filename != '':
            dcm.dump_dicom(self, filename=filename)

    def open_auto(self):
        """Start open automation dialog."""
        dlg = open_automation.OpenAutomationDialog(self)
        dlg.exec()

    def set_active_img(self, imgno):
        """Set active image programmatically.

        Parameters
        ----------
        imgno : int
            image number to set active
        """
        self.tree_file_list.setCurrentItem(
            self.tree_file_list.topLevelItem(imgno))

    def update_active_img(self, current):
        """Overwrite pixmap in memory with new active image, refresh GUI."""
        if len(self.imgs) > 0:
            self.tree_file_list.blockSignals(True)
            if current is not None:
                self.gui.active_img_no = self.tree_file_list.indexOfTopLevelItem(
                    current)
            else:
                self.gui.active_img_no = -1 if len(self.imgs) == 0 else 0
            read_img = True
            if self.wid_image_display.tool_sum.isChecked():
                marked_idxs = self.get_marked_imgs_current_test()
                if self.gui.active_img_no in marked_idxs:
                    self.active_img = self.summed_img
                    read_img = False
            if read_img:
                self.active_img, _ = dcm.get_img(
                    self.imgs[self.gui.active_img_no].filepath,
                    frame_number=self.imgs[self.gui.active_img_no].frame_number,
                    tag_infos=self.tag_infos)
            if self.active_img is not None:
                amin = round(np.amin(self.active_img))
                amax = round(np.amax(self.active_img))
                self.wid_window_level.min_wl.setRange(amin, amax)
                self.wid_window_level.max_wl.setRange(amin, amax)
                if len(np.shape(self.active_img)) == 2:
                    sz_acty, sz_actx = np.shape(self.active_img)
                else:
                    sz_acty, sz_actx, sz_actz = np.shape(self.active_img)
                self.wid_center.val_delta_x.setRange(-sz_actx//2, sz_actx//2)
                self.wid_center.val_delta_y.setRange(-sz_acty//2, sz_acty//2)
            self.wid_dcm_header.refresh_img_info(
                self.imgs[self.gui.active_img_no].info_list_general,
                self.imgs[self.gui.active_img_no].info_list_modality)

            self.refresh_img_display()
            self.refresh_results_display(update_table=False)
            self.refresh_selected_table_row()
            self.tree_file_list.blockSignals(False)

    def update_summed_img(self, recalculate_sum=True):
        """Overwrite pixmap in memory with new summed image, refresh GUI."""
        if len(self.imgs) > 0:
            if recalculate_sum:
                self.start_wait_cursor()
                self.status_bar.showMessage('Calculating sum of marked images...')

                self.summed_img, errmsg = dcm.sum_marked_images(
                    self.imgs, self.get_marked_imgs_current_test(),
                    tag_infos=self.tag_infos)
                self.stop_wait_cursor()
                self.status_bar.showMessage('Finished summing marked images', 2000)
            if self.summed_img is not None:
                if self.average_img:
                    if self.wid_quicktest.gb_quicktest.isChecked():
                        marked = [img_info.marked_quicktest for img_info in self.imgs]
                    else:
                        marked = [img_info.marked for img_info in self.imgs]
                    self.active_img = self.summed_img * (1./len(marked))
                else:
                    self.active_img = self.summed_img
                amin = np.amin(self.active_img)
                amax = np.amax(self.active_img)
                self.wid_window_level.min_wl.setRange(amin, amax)
                self.wid_window_level.max_wl.setRange(amin, amax)
                self.refresh_img_display()
            else:
                if errmsg != '':
                    QMessageBox.information(self, 'Failed summing images', errmsg)

    def reset_summed_img(self):
        """Turn off display of summed image and reset summed_img."""
        self.wid_image_display.tool_sum.setChecked(False)
        self.wid_image_display.tool_sum.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}sigma.png'))
        self.summed_img = None
        self.average_img = False
        try:
            self.active_img, tags = dcm.get_img(
                self.imgs[self.gui.active_img_no].filepath,
                frame_number=self.imgs[self.gui.active_img_no].frame_number,
                tag_infos=self.tag_infos)
        except IndexError:
            pass

    def set_active_image_min_max(self, minval, maxval):
        """Update window level."""
        if self.active_img is not None:
            self.wid_image_display.canvas.img.set_clim(vmin=minval, vmax=maxval)
            self.wid_image_display.canvas.draw()

    def mode_changed(self):
        """Modality selection changed by user input, initiate update gui."""
        if self.wid_paramset.lbl_edit.text() == '*':
            self.wid_paramset.ask_to_save_changes()
        self.current_modality = self.btns_mode.checkedButton().text()
        self.update_mode()

    def update_mode(self):
        """Update GUI when modality has changed."""
        self.wid_quicktest.gb_quicktest.setChecked(False)
        curr_mod_idx = get_modality_index(self.current_modality)
        if curr_mod_idx != self.btns_mode.checkedId():
            self.btns_mode.button(curr_mod_idx).setChecked(True)

        if self.btn_read_vendor_file.isChecked():
            self.current_test = 'vendor'
            self.stack_test_tabs.setCurrentIndex(len(QUICKTEST_OPTIONS))  # last stack
            self.tab_vendor.update_table()
        else:
            self.stack_test_tabs.setCurrentIndex(curr_mod_idx)

            # update list of available parametersets / QuickTests
            paramsets_fname = f'paramsets_{self.current_modality}'
            _, _, self.paramsets = cff.load_settings(fname=paramsets_fname)
            self.current_paramset = self.paramsets[0]
            self.wid_paramset.fname = paramsets_fname
            self.wid_paramset.modality_dict[self.current_modality] = self.paramsets
            self.wid_paramset.fill_template_list()

            reset_test_idx = False
            if self.current_test not in ['DCM', 'ROI']:
                reset_test_idx = True
            self.update_current_test(reset_index=reset_test_idx, refresh_display=False)
            self.update_paramset()
            self.wid_quicktest.fill_template_list()
            if self.wid_quicktest.gb_quicktest.isChecked():
                self.current_quicktest = cfc.QuickTestTemplate()
                self.refresh_img_display()
            self.wid_paramset.flag_edit(False)
        self.gui.current_auto_template = ''
        self.wid_center.reset_delta()
        self.reset_results()

    def update_current_test(self, reset_index=False, refresh_display=True):
        """Update GUI when selected test change.

        Parameters
        ----------
        reset_index : bool
            Reset test index if mode change. Default is False
        """
        self.start_wait_cursor()
        widget = self.stack_test_tabs.currentWidget()
        if widget is not None:
            if hasattr(widget, 'currentIndex'):
                if isinstance(reset_index, bool) and reset_index:
                    widget.setCurrentIndex(0)
                test_idx = widget.currentIndex()
                self.current_test = QUICKTEST_OPTIONS[
                    self.current_modality][test_idx]
                if self.active_img is not None and refresh_display:
                    self.update_roi()
                    self.refresh_results_display()
                self.wid_image_display.tool_rectangle.setChecked(
                    self.current_test == 'Num')
        self.stop_wait_cursor()

    def get_marked_imgs_current_test(self):
        """Link here due to shared functionality with task_based."""
        return self.tree_file_list.get_marked_imgs_current_test()

    def update_roi(self, clear_results_test=False):
        ui_main_methods.update_roi(self, clear_results_test=clear_results_test)
        """Recalculate ROI."""
        '''
        errmsg = None
        if self.active_img is not None:
            self.start_wait_cursor()
            self.status_bar.showMessage('Updating ROI...')
            self.current_roi, errmsg = get_rois(
                self.active_img,
                self.gui.active_img_no, self)
            self.status_bar.clearMessage()
            self.stop_wait_cursor()
        else:
            self.current_roi = None
        self.wid_image_display.canvas.roi_draw()
        self.display_errmsg(errmsg)
        if clear_results_test:
            if self.current_test in [*self.results]:
                self.results[self.current_test] = None
                self.refresh_results_display()
        '''

    def reset_results(self):
        """Clear results and update display."""
        self.results = {}
        self.refresh_results_display()

    def update_results(self, n_added_imgs=0, deleted_idxs=None, sort_idxs=None):
        """Update self.results if added / deleted images.

        Parameters
        ----------
        n_added_imgs : int, optional
            number of added imgs. The default is 0.
        deleted_idxs : list of int, optional
            list of deleted image numbers. The default is None.
        sort_idxs : list of indx, optional
            new sort order. The default is None.
        """
        if n_added_imgs > 0:
            del_keys = []
            for test, res_dict in self.results.items():
                if res_dict is not None:
                    if res_dict['pr_image']:
                        empty_extend = [[] for i in range(n_added_imgs)]
                        res_dict['values'].extend(empty_extend)
                        if 'values_sup' in res_dict:
                            res_dict['values_sup'].extend(empty_extend)
                        if 'details_dict' in res_dict:
                            res_dict['details_dict'].extend(empty_extend)
                    else:
                        del_keys.append(test)
                else:
                    del_keys.append(test)
            if del_keys:
                for key in del_keys:
                    del self.results[key]
            # if results and none marked - set all old imgs to marked
            if self.wid_quicktest.gb_quicktest.isChecked() is False:
                if self.results:
                    marked_img_ids = [
                        i for i, im in enumerate(self.imgs) if im.marked]
                    if len(marked_img_ids) == 0:
                        for i, im in enumerate(self.imgs):
                            if i < len(self.imgs) - n_added_imgs:
                                im.marked = True
                            else:
                                break

        if deleted_idxs:
            deleted_idxs.sort(reverse=True)
            del_keys = []
            for test, res_dict in self.results.items():
                if res_dict is not None:
                    if res_dict['pr_image']:
                        for idx in deleted_idxs:
                            del res_dict['values'][idx]
                            if 'values_sup' in res_dict:
                                del res_dict['values_sup'][idx]
                            if 'details_dict' in res_dict:
                                del res_dict['details_dict'][idx]
                        if self.current_modality == 'CT' and test == 'Noi':
                            for row in range(len(res_dict['values'])):
                                res_dict['values'][row][2:3] = ['NA', 'NA']
                    else:
                        del_keys.append(test)
                else:
                    del_keys.append(test)
            if del_keys:
                for key in del_keys:
                    del self.results[key]

        if sort_idxs is not None:
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
                        'pr_image': True,
                        'values_info': res_dict['values_info'],
                        'values_sup_info': res_dict['values_sup_info'],
                        }
                    for idx in sort_idxs:
                        if 'values' in orig_res[test]:
                            self.results[test]['values'].append(
                                orig_res[test]['values'][idx])
                        if 'values_sup' in orig_res[test]:
                            self.results[test]['values_sup'].append(
                                orig_res[test]['values_sup'][idx])
                        if 'details_dict' in orig_res[test]:
                            self.results[test]['details_dict'].append(
                                orig_res[test]['details_dict'][idx])

        self.refresh_results_display()

    def refresh_results_display(self, update_table=True):
        ui_main_methods.refresh_results_display(self, update_table=update_table)

    def refresh_img_display(self):
        ui_main_methods.refresh_img_display(self)

    def refresh_selected_table_row(self):
        """Set selected results table row to the same as image selected file."""
        if self.current_test in self.results:
            if self.results[self.current_test] is not None:
                if self.results[self.current_test]['pr_image']:
                    wid = self.tab_results.currentWidget()
                    if isinstance(wid, ui_main_result_tabs.ResultTableWidget):
                        marked_imgs = self.get_marked_imgs_current_test()
                        if self.gui.active_img_no in marked_imgs:
                            idx = marked_imgs.index(self.gui.active_img_no)
                            self.wid_res_tbl.result_table.blockSignals(True)
                            self.wid_res_tbl_sup.result_table.blockSignals(True)
                            self.wid_res_tbl.result_table.selectRow(idx)
                            self.wid_res_tbl_sup.result_table.selectRow(idx)
                            self.wid_res_tbl.result_table.blockSignals(False)
                            self.wid_res_tbl_sup.result_table.blockSignals(False)

    def sort_imgs(self):
        """Resort images by dicom info."""
        sort_template = cfc.TagPatternSort()
        dlg = TagPatternEditDialog(
            initial_pattern=sort_template,
            modality=self.current_modality,
            title='Sort images by DICOM header information',
            typestr='sort',
            accept_text='Sort',
            reject_text='Cancel',
            save_blocked=self.save_blocked)
        res = dlg.exec()  # returning TagPatternSort
        if res:
            sort_template = dlg.get_pattern()
            self.start_wait_cursor()
            self.imgs, order = dcm.sort_imgs(self.imgs, sort_template, self.tag_infos)
            self.stop_wait_cursor()
            self.tree_file_list.update_file_list()
            self.current_sort_pattern = sort_template
            if self.results:
                self.update_results(sort_idxs=order)
                self.refresh_selected_table_row()

    def reset_split_sizes(self):
        """Set and reset QSplitter sizes."""
        self.split_list_rest.setSizes(
            [round(self.gui.panel_height*0.2), round(self.gui.panel_height*0.8)])
        self.split_img_header.setSizes(
            [round(self.gui.panel_height*0.55), round(self.gui.panel_height*0.25)])
        self.split_lft_img.setSizes(
            [round(self.gui.panel_width*0.32), round(self.gui.panel_width*0.68)])
        self.split_lft_rgt.setSizes(
            [round(self.gui.panel_width*1.2), round(self.gui.panel_width*0.8)])
        self.split_rgt_top_rest.setSizes(
            [round(self.gui.panel_height*0.2), round(self.gui.panel_height*0.8)])
        self.split_rgt_mid_btm.setSizes(
            [round(self.gui.panel_height*0.4), round(self.gui.panel_height*0.4)])

    def set_split_max_img(self):
        """Set QSplitter to maximized image."""
        self.split_list_rest.setSizes(
            [round(self.gui.panel_height*0.1), round(self.gui.panel_height*0.9)])
        self.split_img_header.setSizes(
            [round(self.gui.panel_height*0.9), round(self.gui.panel_height*0.)])
        self.split_lft_img.setSizes(
            [0, self.gui.panel_width])

    def reset_split_max_img(self):
        """Set QSplitter to maximized image."""
        self.split_list_rest.setSizes(
            [round(self.gui.panel_height*0.2), round(self.gui.panel_height*0.8)])
        self.split_img_header.setSizes(
            [round(self.gui.panel_height*0.55), round(self.gui.panel_height*0.25)])
        self.split_lft_img.setSizes(
            [round(self.gui.panel_width*0.32), round(self.gui.panel_width*0.68)])

    def set_maximize_results(self):
        """Set QSplitter to maximized results."""
        self.split_lft_rgt.setSizes(
            [round(self.gui.panel_width*0.8), round(self.gui.panel_width*1.2)])
        self.split_rgt_top_rest.setSizes(
            [0, self.gui.panel_height])
        self.split_rgt_mid_btm.setSizes(
            [0, self.gui.panel_height])
        self.wid_res_image.hide_window_level()

    def reset_maximize_results(self):
        """Set QSplitter to maximized results."""
        self.split_lft_rgt.setSizes(
            [round(self.gui.panel_width*1.2), round(self.gui.panel_width*0.8)])
        self.split_rgt_top_rest.setSizes(
            [round(self.gui.panel_height*0.2), round(self.gui.panel_height*0.8)])
        self.split_rgt_mid_btm.setSizes(
            [round(self.gui.panel_height*0.4), round(self.gui.panel_height*0.4)])
        self.wid_res_image.reset_split_sizes()

    def resize_main(self, new_resolution=False):
        """Reset geometry of MainWindow."""
        if new_resolution:
            screens = QApplication.instance().screens()
            screen_geometry = screens[0].geometry()
            if len(screens) > 1:  # find which screen is current
                screen_xs = [
                    [screen.geometry().x(),
                     screen.geometry().x() + screen.geometry().width()]
                    for screen in screens]
                for screen_id, xs in enumerate(screen_xs):
                    if self.pos().x() > xs[0] and self.pos().x() < xs[1]:
                        screen_geometry = screens[screen_id].geometry()
                        break
            self.gui.panel_width = round(0.48*screen_geometry.width())
            self.gui.panel_height = round(0.86*screen_geometry.height())
            self.wid_full.setFixedSize(
                2*self.gui.panel_width, self.gui.panel_height)
        self.resize(
            round(self.gui.panel_width*2+30),
            round(self.gui.panel_height+50)
            )
        if new_resolution:
            self.reset_split_sizes()
        self.show()

    def moveEvent(self, event):
        """If window moved on screen or to other monitor."""
        try:
            old_screen = QApplication.screenAt(event.oldPos())
            new_screen = QApplication.screenAt(event.pos())

            if not old_screen == new_screen:
                if not old_screen.geometry() == new_screen.geometry():
                    sz_screen = new_screen.geometry()
                    self.gui.panel_width = round(0.48*sz_screen.width())
                    self.gui.panel_height = round(0.86*sz_screen.height())
                    self.resize_main()
                    self.reset_split_sizes()
        except AttributeError:
            pass
        return super().moveEvent(event)

    def run_auto_wizard(self):
        """Start the automation wizard."""
        if os.environ[ENV_CONFIG_FOLDER] == '':
            QMessageBox.information(
                self, 'Missing config folder',
                '''To be able to save settings you will need to specify a \n
                configuration folder for these settings.\n
                Start the wizard again when the configuration folder is set.
                ''')
        else:
            self.wizard = automation_wizard.AutomationWizard(self)
            self.wizard.open()

    def run_dash(self):
        """Show automation results in browser."""
        config_folder = cff.get_config_folder()
        filenames = [x.stem for x in Path(config_folder).glob('*')
                     if x.suffix == '.yaml']
        if 'auto_templates' in filenames or 'auto_vendor_templates' in filenames:
            _, _, dash_settings = cff.load_settings(fname='dash_settings')
            self.dash_worker = DashWorker(dash_settings=dash_settings)
            self.dash_worker.start()
            dlg = messageboxes.MessageBoxWithDetails(
                self, title='Dashboard in webbrowser',
                msg='Results will open in a webbrowser.',
                info=('If large datasets or slow file-access you might have to refresh '
                      'the webpage. Look for "Serving on http... in the command window '
                      'when finished (or issues).'),
                icon=QMessageBox.Information)
            dlg.exec()
            url = f'http://{dash_settings.host}:{dash_settings.port}'
            webbrowser.open(url=url, new=1)
            self.dash_worker.exit()
        else:
            QMessageBox.information(
                self, 'Missing automation templates',
                '''Found no automation templates to display results from.''')

    def run_rename_dicom(self):
        """Start Rename Dicom dialog."""
        dlg = rename_dicom.RenameDicomDialog(self)
        dlg.exec()

    def run_task_based_auto(self):
        """Start Task Based image quality analysis dialog."""
        dlg = task_based_image_quality.TaskBasedImageQualityDialog(self)
        dlg.exec()

    def run_settings(self, initial_view='', initial_template_label='',
                     paramset_output=False):
        """Display settings dialog."""
        proceed = True
        if self.wid_paramset.edited or self.wid_quicktest.edited:
            txt = 'parameter set' if self.wid_paramset.edited else ''
            if self.wid_quicktest.edited:
                if txt:
                    txt = txt + ' and QuickTest template'
                else:
                    txt = 'QuickTest template'
            msg = f'There are unsaved changes to {txt}. Ignore and continue?'
            reply = QMessageBox.question(
                self, 'Save changes first?', msg,
                QMessageBox.Yes, QMessageBox.No)
            if reply == QMessageBox.No:
                proceed = False
        if proceed:
            if initial_view == '':
                dlg = SettingsDialog(self)
            else:
                dlg = SettingsDialog(
                    self, initial_view=initial_view,
                    initial_modality=self.current_modality,
                    paramset_output=paramset_output,
                    initial_template_label=initial_template_label)
            dlg.exec()
            self.update_settings(after_edit_settings=True)

    def update_settings(self, after_edit_settings=False):
        """Refresh data from settings files affecting GUI in main window."""
        if after_edit_settings is False:
            print('Reading configuration settings...')
        self.lastload = time()
        _, _, self.user_prefs = cff.load_user_prefs()
        if self.user_prefs.dark_mode:
            plt.style.use('dark_background')
        _, _, self.paramsets = cff.load_settings(
            fname=f'paramsets_{self.current_modality}')
        _, _, self.quicktest_templates = cff.load_settings(
            fname='quicktest_templates')
        _, _, self.tag_infos = cff.load_settings(fname='tag_infos')
        _, _, self.tag_patterns_special = cff.load_settings(
            fname='tag_patterns_special')
        _, _, self.digit_templates = cff.load_settings(
            fname='digit_templates')

        try:  # avoid error before gui ready
            prev_label = self.wid_quicktest.current_template.label
            self.wid_quicktest.modality_dict = self.quicktest_templates
            self.wid_quicktest.fill_template_list(set_label=prev_label)
            prev_label = self.current_paramset.label
            self.wid_paramset.modality_dict = {
                f'{self.current_modality}': self.paramsets}
            self.wid_paramset.fill_template_list(set_label=prev_label)
        except AttributeError:
            pass

        if after_edit_settings:
            self.tab_nm.sni_correct.update_reference_images()
            self.update_paramset()  # update Num digit templates list
        else:
            print('imageQC is ready')

    def version_control(self):
        """Compare version number of tag_infos with current saved."""
        cff.version_control(self)

    def display_clipboard(self, title='Clipboard content'):
        """Display clipboard content e.g. when testing QuickTest output."""
        dataf = pd.read_clipboard()
        nrows, ncols = dataf.shape
        txt = ''
        if nrows == 0:
            row_list = [*dataf.to_dict()]
            txt = '\t'.join(row_list)
        else:
            row_list = [[*dataf.to_dict()]]
            for row in range(nrows):
                row_list.append([str(dataf.iat[row, col]) for col in range(ncols)])
            lines_list = ['\t'.join(row) for row in row_list]
            txt = '\n'.join(lines_list)
        dlg = TextDisplay(
            self, txt, title=title,
            min_width=1000, min_height=300)
        res = dlg.exec()
        if res:
            pass  # just to wait for user to close message

    def display_errmsg(self, errmsg):
        """Display error in statusbar or as popup if long."""
        if errmsg is not None:
            if isinstance(errmsg, str):
                self.status_bar.showMessage(errmsg, 2000, warning=True)
            elif isinstance(errmsg, list):
                self.status_bar.showMessage(
                    'Finished with issues', 2000, warning=True,
                    add_warnings=errmsg)

    def display_previous_warnings(self):
        """Display saved statusbar warnings."""
        if len(self.status_bar.saved_warnings) > 0:
            dlg = TextDisplay(
                self, '\n'.join(self.status_bar.saved_warnings),
                title='Previous warnings',
                min_width=1000, min_height=300)
            dlg.exec()

            reply = QMessageBox.question(
                self, 'Reset warnings?',
                'Reset saved warnings?',
                QMessageBox.Yes, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.status_bar.saved_warnings = []
                self.act_warning.setEnabled(False)
        else:
            msg = 'No previous warnings to show'
            QMessageBox.information(self, 'No data', msg)
            self.act_warning.setEnabled(False)

    def start_wait_cursor(self):
        """Block mouse events by wait cursor."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        qApp.processEvents()

    def stop_wait_cursor(self):
        """Return to normal mouse cursor after wait cursor."""
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        qApp.processEvents()

    def finish_cleanup(self):
        """Cleanup/save before exit."""
        proceed_to_close = True
        if self.wid_paramset.lbl_edit.text() == '*' and self.developer_mode == False:
            proceed_to_close = messageboxes.proceed_question(
                self,
                f'Unsaved changes to {self.wid_paramset.fname}. Exit without saving?')
        if proceed_to_close:
            try:
                cff.remove_user_from_active_users()
                # save current settings to user prefs
                self.user_prefs.annotations_line_thick = self.gui.annotations_line_thick
                self.user_prefs.annotations_font_size = self.gui.annotations_font_size
                ok, path = cff.save_user_prefs(self.user_prefs, parentwidget=self)
            except:  # on pytest
                pass
        return proceed_to_close

    def closeEvent(self, event):
        """Exit app by x in the corner."""
        proceed = self.finish_cleanup()
        if proceed:
            event.accept()
        else:
            event.ignore()

    def exit_app(self):
        """Exit app by menu."""
        proceed = self.finish_cleanup()
        if proceed:
            sys.exit()

    def about(self):
        """Show about info."""
        dlg = AboutDialog(version=VERSION)
        dlg.exec()

    def wiki(self):
        """Open wiki url."""
        url = 'https://github.com/EllenWasbo/imageQCpy/wiki'
        webbrowser.open(url=url, new=1)

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
        menu_bar = self.menuBar()
        tool_bar = self.addToolBar('first')

        self.chk_append = QCheckBox('Append')
        self.cbox_file_list_display = QComboBox()
        self.cbox_file_list_display.addItems(
            ['File path', 'Format pattern'])
        self.cbox_file_list_display.setCurrentIndex(1)
        self.cbox_file_list_display.setToolTip(
            '''Display file path or header info in file list.
            Format pattern = Special tag pattern as defined in settings.''')
        self.cbox_file_list_display.currentIndexChanged.connect(
            self.tree_file_list.update_file_list)

        act_open = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}open.png'),
            'Open DICOM image(s)', self)
        act_open.setShortcut('Ctrl+O')
        act_open.triggered.connect(self.open_files)
        act_open_adv = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}openMulti.png'),
            'Open DICOM image(s) with advanced options', self)
        act_open_adv.triggered.connect(self.open_multi)
        act_read_header = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}tags.png'),
            'Read DICOM header', self)
        act_read_header.triggered.connect(self.read_header)
        act_open_auto = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}play.png'),
            'Run automation templates', self)
        act_open_auto.triggered.connect(self.open_auto)
        act_wizard_auto = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}playGears.png'),
            'Start wizard to config automation with current settings', self)
        act_wizard_auto.triggered.connect(self.run_auto_wizard)
        act_dash = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}globe.png'),
            'Refreash and show dashboard with results from automation templates', self)
        act_dash.triggered.connect(self.run_dash)
        act_rename_dcm = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}rename_dicom.png'),
            'Run Rename DICOM', self)
        act_rename_dcm.triggered.connect(self.run_rename_dicom)
        act_to_top = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveTop.png'),
            'Move selected images to top', self)
        act_to_top.triggered.connect(
            lambda: self.tree_file_list.move_file(direction='top'))
        act_to_up = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
            'Move selected images up', self)
        act_to_up.triggered.connect(
            lambda: self.tree_file_list.move_file(direction='up'))
        act_to_down = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
            'Move selected images down', self)
        act_to_down.triggered.connect(
            lambda: self.tree_file_list.move_file(direction='down'))
        act_to_bottom = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveBottom.png'),
            'Move selected images to bottom', self)
        act_to_bottom.triggered.connect(
            lambda: self.tree_file_list.move_file(direction='bottom'))
        act_to_pos = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveTo.png'),
            'Move selected images specific position', self)
        act_to_pos.triggered.connect(
            lambda: self.tree_file_list.move_file(direction='to'))
        act_add_dummy = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add_dummy.png'),
            'Add dummy image(s) if missing images for QuickTest templates', self)
        act_add_dummy.triggered.connect(self.add_dummy)
        act_sort = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}sortAZ.png'),
            'Sort images from patterns based on DICOM header', self)
        act_sort.triggered.connect(self.sort_imgs)
        act_settings = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}gears.png'),
            'Settings', self)
        act_settings.triggered.connect(self.run_settings)
        act_resize_main = QAction('Refresh layout after resolution change', self)
        act_resize_main.triggered.connect(lambda: self.resize_main(new_resolution=True))
        act_reset_split = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}layout.png'),
            'Reset layout', self)
        act_reset_split.triggered.connect(self.reset_split_sizes)
        self.act_warning = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}warning.png'),
            'Show warnings', self)
        self.act_warning.triggered.connect(self.display_previous_warnings)
        self.act_warning.setEnabled(False)

        act_close = QAction('Close selected images', self)
        act_close.setShortcut('Ctrl+W')
        act_close.triggered.connect(self.tree_file_list.close_selected)
        act_close_all = QAction('Close all images', self)
        act_close_all.setShortcut('Ctrl+Shift+W')
        act_close_all.triggered.connect(self.clear_all_images)
        act_quit = QAction('&Quit', self)
        act_quit.setShortcut('Ctrl+Q')
        act_quit.triggered.connect(self.exit_app)
        act_about = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}info.png'),
            'About imageQC...', self)
        act_about.triggered.connect(self.about)
        act_wiki = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}globe.png'),
            'Wiki ...', self)
        act_wiki.triggered.connect(self.wiki)
        act_task_based_auto = QAction('CT task based image quality analysis...', self)
        act_task_based_auto.triggered.connect(self.run_task_based_auto)

        # fill menus
        menu_file = QMenu('&File', self)
        menu_file.addActions([
            act_open, act_open_adv, act_read_header, act_open_auto, act_wizard_auto,
            act_rename_dcm, #act_task_based_auto,
            act_close, act_close_all, act_quit])
        menu_bar.addMenu(menu_file)
        menu_settings = QMenu('&Settings', self)
        menu_settings.addAction(act_settings)
        menu_bar.addMenu(menu_settings)
        menu_layout = QMenu('&Layout', self)
        menu_layout.addActions([act_resize_main, act_reset_split])
        menu_bar.addMenu(menu_layout)
        menu_help = QMenu('&Help', self)
        menu_help.addActions([act_about, act_wiki])
        menu_bar.addMenu(menu_help)

        # fill toolbar
        tool_bar.addActions([
            act_open, act_open_adv, act_open_auto, act_wizard_auto, act_dash])
        tool_bar.addSeparator()
        tool_bar.addWidget(self.chk_append)
        tool_bar.addSeparator()
        tool_bar.addActions([
            act_to_top, act_to_up, act_to_down, act_to_bottom, act_to_pos, act_sort,
            act_add_dummy])
        self.lbl_n_loaded = QLabel('0    ')
        tool_bar.addWidget(QLabel('Loaded images:'))
        tool_bar.addWidget(self.lbl_n_loaded)
        tool_bar.addWidget(QLabel('             '))
        tool_bar.addWidget(QLabel('List files as'))
        tool_bar.addWidget(self.cbox_file_list_display)

        tool_bar.addWidget(QLabel('             '))
        tool_bar.addActions([act_reset_split, act_rename_dcm, act_settings])
        tool_bar.addWidget(QLabel('             '))
        tool_bar.addAction(self.act_warning)

    def create_modality_selector(self):
        """Groupbox with modality selection."""
        self.gb_modality = QGroupBox('Modality')
        self.gb_modality.setFont(uir.FontItalic())
        self.btns_mode = QButtonGroup()
        self.gb_modality.setFixedWidth(round(self.gui.panel_width*0.75))
        hlo = QHBoxLayout()

        for mod, (key, _) in enumerate(QUICKTEST_OPTIONS.items()):
            rbtn = QRadioButton(key)
            self.btns_mode.addButton(rbtn, mod)
            hlo.addWidget(rbtn)
            rbtn.clicked.connect(self.mode_changed)

        idx = get_modality_index(self.current_modality)
        self.btns_mode.button(idx).setChecked(True)
        self.gb_modality.setLayout(hlo)

    def create_test_tab_CT(self):
        """Initiate GUI for the stacked test tab CT."""
        self.stack_test_tabs = QStackedWidget()
        self.tab_ct = ui_main_test_tabs.ParamsTabCT(self)
        self.stack_test_tabs.addWidget(self.tab_ct)

    def create_test_tabs_rest(self):
        """Initiate GUI for the stacked test tabs - not CT."""
        self.start_wait_cursor()
        self.tab_xray = ui_main_test_tabs.ParamsTabXray(self)
        self.tab_mammo = ui_main_test_tabs.ParamsTabMammo(self)
        self.tab_nm = ui_main_test_tabs.ParamsTabNM(self)
        self.tab_spect = ui_main_test_tabs.ParamsTabSPECT(self)
        self.tab_pet = ui_main_test_tabs.ParamsTabPET(self)
        self.tab_mr = ui_main_test_tabs.ParamsTabMR(self)
        self.tab_vendor = ParamsTabVendor(self)
        self.stack_test_tabs.addWidget(self.tab_xray)
        self.stack_test_tabs.addWidget(self.tab_mammo)
        self.stack_test_tabs.addWidget(self.tab_nm)
        self.stack_test_tabs.addWidget(self.tab_spect)
        self.stack_test_tabs.addWidget(self.tab_pet)
        self.stack_test_tabs.addWidget(self.tab_mr)
        self.stack_test_tabs.addWidget(self.tab_vendor)
        self.stop_wait_cursor()

    def create_result_tabs(self):
        """Initiate GUI for the stacked result tabs."""
        self.tab_results = QTabWidget()
        self.tab_results.currentChanged.connect(
            lambda: self.refresh_results_display(update_table=True))
        self.wid_res_tbl = ui_main_result_tabs.ResultTableWidget(self)
        self.wid_res_plot = ui_main_result_tabs.ResultPlotWidget(
            self, ui_main_result_tabs.ResultPlotCanvas(self))
        self.wid_res_image = ui_main_result_tabs.ResultImageWidget(self)
        self.wid_res_tbl_sup = ui_main_result_tabs.ResultTableWidget(self)

        self.tab_results.addTab(self.wid_res_tbl, 'Results table')
        self.tab_results.addTab(self.wid_res_plot, 'Results plot')
        self.tab_results.addTab(self.wid_res_image, 'Results image')
        self.tab_results.addTab(self.wid_res_tbl_sup, 'Supplement table')


class StatusBar(uir.StatusBar):
    """Tweeks to uir.StatusBar."""

    def __init__(self, parent):
        super().__init__(parent)
        self.saved_warnings = []

    def showMessage(self, txt, timeout=0, warning=False, add_warnings=None):
        """Set background color when message is shown."""
        if warning:
            txt_ = f'{ctime()}: {txt}'
            self.saved_warnings.append(txt_)
            if add_warnings:
                self.saved_warnings.extend(add_warnings)
            self.main.act_warning.setEnabled(True)

        super().showMessage(txt, timeout=timeout, warning=warning)
