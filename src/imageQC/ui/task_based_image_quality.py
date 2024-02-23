# -*- coding: utf-8 -*-
"""
Run TTF and NPS, d-prime on dataset.

@author: Ellen Wasbo
"""
import os
import copy
from time import time
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, qApp,
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QToolBar, QAbstractItemView,
    QTabWidget, QStackedWidget, QDoubleSpinBox,
    QLabel, QLineEdit, QPushButton, QAction, QSpinBox, QCheckBox, QListWidget,
    QComboBox, QFileDialog, QMessageBox
    )

# imageQC block start
from imageQC.ui import ui_main_methods
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.ui.ui_main_image_widgets import ImageDisplayWidget
from imageQC.ui import open_multi
from imageQC.ui import reusable_widgets as uir
from imageQC.config import config_classes as cfc
from imageQC.config import config_func as cff
from imageQC.scripts import dcm
from imageQC.ui import messageboxes
from imageQC.scripts import input_main
from imageQC.ui import ui_main_quicktest_paramset_select
from imageQC.ui.ui_main_test_tabs import ParamsTabCT, ParamsWidget
from imageQC.ui import ui_main_result_tabs
from imageQC.ui import ui_image_canvas
from imageQC.scripts.calculate_roi import get_rois
from imageQC.config.iQCconstants import ENV_ICON_PATH
from imageQC.scripts.mini_methods import get_uniq_ordered
# imageQC block end


class ReadFilesDialog(ImageQCDialog):
    """GUI for opening images similar to OpenMultiDialog in open_multi.py."""

    def __init__(self, main):
        super().__init__()
        self.setWindowTitle('Read images')
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        # TODO option to save/load default
        temp = cfc.TagPatternFormat(list_tags=['SeriesUID'], list_format=[''])
        self.wid = open_multi.OpenMultiWidget(
            main, input_template=temp, lock_on_modality='CT')
        vlo.addWidget(self.wid)


class TaskBasedImageQualityDialog(ImageQCDialog):
    """GUI setup for the dialog window."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setWindowTitle('CT task based image quality automated analysis')

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        infotxt = (
            'This tool will facilitate effective calculation of MTF, NPS and '
            'detectability index for CT images acquired and reconstructed under '
            'different conditions.'
            )

        vlo.addWidget(uir.UnderConstruction())
        vlo.addWidget(uir.LabelItalic(infotxt))
        vlo.addWidget(uir.HLine())
        vlo_right = QVBoxLayout()
        vlo_left = QVBoxLayout()
        hlo = QHBoxLayout()
        vlo.addLayout(hlo)
        hlo.addLayout(vlo_left)
        hlo.addLayout(vlo_right)

        # widget need parameters from input_main.py/InputMain and MainWindow
        self.current_modality = 'CT'
        self.current_test = 'DCM'
        self.tests = ['DCM', 'TTF', 'NPS', 'DPR']
        self.lastload = time()
        self.fname = 'paramsets_CT_task_based'
        _, _, self.paramsets = cff.load_settings(fname=self.fname)
        self.current_paramset = copy.deepcopy(self.paramsets[0])
        self.current_quicktest = cfc.QuickTestTemplate()
        self.imgs = []
        self.results = {}
        self.errmsgs = []
        self.current_group_indicators = []
        # string for each image if output set pr group with quicktest (paramset.output)
        self.automation_active = True
        self.active_img = None
        self.current_roi = None
        self.summed_img = None
        self.gui = copy.deepcopy(main.gui)
        self.gui.active_img_no = -1
        self.imgs_series_numbers = []
        self.current_series = ''
        self.series_strings = []
        self.user_prefs = main.user_prefs
        self.tag_infos = main.tag_infos
        self.tag_patterns_special = main.tag_patterns_special
        self.save_blocked = main.save_blocked
        self.wid_image_display = ImageDisplayWidget(self, toolbar_right=False)
        self.wid_paramset = ui_main_quicktest_paramset_select.SelectParamsetWidget(
            self, fname=self.fname)
        self.create_result_tabs()

        min_height = 200
        min_width = 400
        self.list_series = QListWidget()
        self.list_series.setMinimumHeight(min_height)
        self.list_series.setMinimumWidth(min_width)
        self.list_series.currentRowChanged.connect(self.update_list_images)
        self.list_images = QListWidget()
        self.list_images.setMinimumHeight(min_height)
        self.list_images.setMinimumWidth(min_width)
        self.list_images.currentRowChanged.connect(self.update_active_img)

        self.btn_locate_images = QPushButton('Open images...')
        self.btn_locate_images.clicked.connect(self.open_multi)
        self.lbl_n_loaded = QLabel('0 / 0')

        hlo_load = QHBoxLayout()
        vlo_left.addLayout(hlo_load)
        hlo_load.addWidget(self.btn_locate_images)
        hlo_load.addWidget(QLabel('Loaded series / images: '))
        hlo_load.addWidget(self.lbl_n_loaded)

        # How to locate the images - which is connected, which is MTF/NPS
        # identify MTF and NPS images based on
        # - Assume Mercury phantom (both in same series)
        # - z value range
        # - MTF / NPS in seriesdescription
        # - manually select series
        # - image content

        # - autoselect which MTF images to include/exclude based on variation in CT number for material with highest density
        # if delta_z between images larger than increment - regard as separate (e.g. Mercury)
        #Settings + Update button / list of series used for MTF / list of series used for NPS / show images for selected series
        hlo_lists = QHBoxLayout()
        vlo_left.addLayout(hlo_lists)

        vlo_series = QVBoxLayout()
        hlo_lists.addLayout(vlo_series)
        vlo_series.addWidget(uir.LabelHeader('Groups', 4))
        vlo_series.addWidget(self.list_series)

        vlo_images = QVBoxLayout()
        hlo_lists.addLayout(vlo_images)
        vlo_images.addWidget(uir.LabelHeader('Images in selected group', 4))
        vlo_images.addWidget(self.list_images)

        vlo_left.addWidget(self.wid_image_display)

        # How to analyse
        hlo_analyse = QHBoxLayout()
        vlo_right.addLayout(hlo_analyse)
        self.stack_test_tabs = QStackedWidget()
        self.tab_ct = ParamsTabCT(self, task_based=True)
        self.stack_test_tabs.addWidget(self.tab_ct)
        self.tab_ct.create_tab_dpr()
        vlo_right.addWidget(self.stack_test_tabs)

        # Results / output
        vlo_right.addWidget(self.tab_results)
        self.update_paramset()

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

    def update_paramset(self):
        """Fill gui with params from selected paramset."""
        self.current_paramset = copy.deepcopy(self.paramsets[0])
        if self.results:
            self.reset_results()
        self.tab_ct.update_displayed_params()
        self.wid_res_tbl.tb_copy.parameters_output = self.current_paramset.output
        self.wid_res_tbl.tb_copy.update_checked()
        self.update_roi()
        #self.wid_paramset.flag_edit(False)

    def open_multi(self):
        """Start open advanced dialog."""
        dlg = open_multi.OpenMultiDialog(self)
        res = dlg.exec()
        if res:
            if len(dlg.wid.open_imgs) > 0:
                self.imgs = dlg.wid.open_imgs
                series_strings = [' '.join(img.series_list_strings)
                                  for img in self.imgs]
                self.series_strings = get_uniq_ordered(series_strings)
                self.current_series = series_strings[0]
                self.imgs_series_numbers = []
                for imgno, img in enumerate(self.imgs):
                    ser = ' '.join(img.series_list_strings)
                    serno = self.series_strings.index(ser)
                    self.imgs_series_numbers.append(serno)
                self.lbl_n_loaded.setText(
                    f'{len(self.series_strings)} / {len(self.imgs)}')
                self.blockSignals(True)
                self.list_series.clear()
                self.list_series.addItems(self.series_strings)
                self.list_series.setCurrentRow(0)
                self.blockSignals(False)
                self.update_list_images(serno=0)
        self.stop_wait_cursor()

    def update_list_images(self, serno=-1):
        """Fill list_images with all images in selected groups."""
        if serno >= 0:
            self.blockSignals(True)
            img_strings = [
                ' '.join(img.file_list_strings) for i, img in enumerate(self.imgs)
                if self.imgs_series_numbers[i] == serno]
            self.list_images.clear()
            self.list_images.addItems(img_strings)
            self.list_images.setCurrentRow(0)
            self.blockSignals(False)
            self.update_active_img()

    def get_marked_imgs_current_test(self):
        """Return indexes in self.imgs for images in selected series."""
        idxs = []
        if self.imgs:
            serno = self.list_series.currentRow()
            idxs = [i for i, img in enumerate(self.imgs)
                if self.imgs_series_numbers[i] == serno]
        return idxs

    def update_active_img(self):
        """Overwrite pixmap in memory with new active image, refresh GUI."""
        imgno = self.list_images.currentRow()
        serno = self.list_series.currentRow()
        if imgno > -1 and serno > -1:
            first_in_series = self.imgs_series_numbers.index(serno)
            self.gui.active_img_no = first_in_series + imgno
            self.active_img, _ = dcm.get_img(
                self.imgs[self.gui.active_img_no].filepath,
                frame_number=self.imgs[self.gui.active_img_no].frame_number,
                tag_infos=self.tag_infos)
            '''
            if self.active_img is not None:
                amin = round(np.amin(self.active_img))
                amax = round(np.amax(self.active_img))
                self.wid_window_level.min_wl.setRange(amin, amax)
                self.wid_window_level.max_wl.setRange(amin, amax)
                if len(np.shape(self.active_img)) == 2:
                    sz_acty, sz_actx = np.shape(self.active_img)
                else:
                    sz_acty, sz_actx, sz_actz = np.shape(self.active_img)
            self.wid_dcm_header.refresh_img_info(
                self.imgs[self.gui.active_img_no].info_list_general,
                self.imgs[self.gui.active_img_no].info_list_modality)
            '''

            self.refresh_img_display()
            self.refresh_results_display(update_table=False)
            self.refresh_selected_table_row()

    def update_current_test(self, reset_index=False, refresh_display=True):
        """Update when selected test change.

        Parameters
        ----------
        reset_index : bool
            Reset test index if mode change. Used in ui_main. Default is False
        """
        self.start_wait_cursor()
        widget = self.stack_test_tabs.currentWidget()
        if widget is not None:
            if hasattr(widget, 'currentIndex'):
                test_idx = widget.currentIndex()
                self.current_test = self.tests[test_idx]
                if self.active_img is not None and refresh_display:
                    self.update_roi()
                    self.refresh_results_display()
        self.stop_wait_cursor()

    def refresh_img_display(self):
        ui_main_methods.refresh_img_display(self)

    def refresh_selected_table_row(self):
        """Set selected results table row to the same as image selected file."""
        if self.current_test in self.results:
            if self.results[self.current_test] is not None:
                wid = self.tab_results.currentWidget()
                if isinstance(wid, ui_main_result_tabs.ResultTableWidget):
                    pass
                    '''
                    marked_imgs = self.get_marked_imgs_current_test()
                    if self.gui.active_img_no in marked_imgs:
                        idx = marked_imgs.index(self.gui.active_img_no)
                        self.wid_res_tbl.result_table.blockSignals(True)
                        self.wid_res_tbl_sup.result_table.blockSignals(True)
                        self.wid_res_tbl.result_table.selectRow(idx)
                        self.wid_res_tbl_sup.result_table.selectRow(idx)
                        self.wid_res_tbl.result_table.blockSignals(False)
                        self.wid_res_tbl_sup.result_table.blockSignals(False)
                    '''


    def update_roi(self, clear_results_test=False):
        ui_main_methods.update_roi(self, clear_results_test=clear_results_test)

    def display_errmsg(self, errmsg):
        """Display error in statusbar or as popup if long."""
        if errmsg:
            msg = ''
            if isinstance(errmsg, str):
                msg = errmsg
                details = []
            elif isinstance(errmsg, list):
                msg = 'Finished with issues. See details.'
                details = errmsg

            dlg = messageboxes.MessageBoxWithDetails(
                self, title='Finished with issues',
                msg=msg,
                details=details, icon=QMessageBox.Warning)
            dlg.exec()

    def refresh_results_display(self, update_table=True):
        """Update GUI for test results when results or selections change."""
        if self.current_test not in self.results:
            # clear all
            self.wid_res_tbl.result_table.clear()
            self.wid_res_tbl_sup.result_table.clear()
            self.wid_res_plot.plotcanvas.plot()
            self.wid_res_image.canvas.result_image_draw()
        else:
            # update only active
            wid = self.tab_results.currentWidget()
            if isinstance(wid, ui_main_result_tabs.ResultTableWidget) and update_table:
                try:
                    self.wid_res_tbl.result_table.fill_table(
                        col_labels=self.results[self.current_test]['headers'],
                        values_rows=self.results[self.current_test]['values'],
                        linked_image_list=self.results[
                            self.current_test]['pr_image'],
                        table_info=self.results[self.current_test]['values_info']
                        )
                except (KeyError, TypeError, IndexError):
                    self.wid_res_tbl.result_table.clear()
                try:
                    self.wid_res_tbl_sup.result_table.fill_table(
                        col_labels=self.results[self.current_test]['headers_sup'],
                        values_rows=self.results[self.current_test]['values_sup'],
                        linked_image_list=self.results[
                            self.current_test]['pr_image'],
                        table_info=self.results[
                            self.current_test]['values_sup_info']
                        )
                except (KeyError, TypeError, IndexError):
                    self.wid_res_tbl_sup.result_table.clear()
            elif isinstance(wid, ui_main_result_tabs.ResultPlotWidget):
                self.wid_res_plot.plotcanvas.plot()
            elif isinstance(wid, ui_main_result_tabs.ResultImageWidget):
                self.wid_res_image.canvas.result_image_draw()

    def start_wait_cursor(self):
        """Block mouse events by wait cursor."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        qApp.processEvents()

    def stop_wait_cursor(self):
        """Return to normal mouse cursor after wait cursor."""
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        qApp.processEvents()
