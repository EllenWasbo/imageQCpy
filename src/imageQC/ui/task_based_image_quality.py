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
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QAbstractItemView,
    QTabWidget, QStackedWidget,
    QLabel, QLineEdit, QPushButton, QAction, QSpinBox, QCheckBox, QListWidget,
    QComboBox, QFileDialog, QMessageBox
    )

# imageQC block start
from imageQC.scripts.input_main_auto import InputMain
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.ui.ui_main_image_widgets import ImageDisplayWidget
from imageQC.ui import open_multi
from imageQC.ui import reusable_widgets as uir
from imageQC.config import config_classes as cfc
from imageQC.config import config_func as cff
from imageQC.ui import messageboxes
from imageQC.scripts import input_main_auto
from imageQC.ui import ui_main_quicktest_paramset_select
from imageQC.ui.ui_main_test_tabs import ParamsTabCT
from imageQC.ui import ui_main_result_tabs
from imageQC.ui import ui_image_canvas
from imageQC.scripts.calculate_roi import get_rois
from imageQC.config.iQCconstants import ENV_ICON_PATH
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

        # widget need parameters from input_main_auto.py/InputMain and MainWindow
        self.lastload = time()
        self.fname = 'paramsets_CT_task_based'
        _, _, self.paramsets = cff.load_settings(fname=self.fname)
        self.current_modality = 'CT'
        self.current_test = 'TTF'
        self.current_paramset = cfc.ParamSetCT_TaskBased()
        self.current_quicktest = cfc.QuickTestTemplate()
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
        self.user_prefs = main.user_prefs
        self.tag_infos = main.tag_infos
        self.tag_patterns_special = main.tag_patterns_special
        self.save_blocked = main.save_blocked
        self.wid_image_display = ImageDisplayWidget(self)
        self.wid_paramset = ui_main_quicktest_paramset_select.SelectParamsetWidget(
            self, fname=self.fname)
        self.create_result_tabs()

        self.btn_locate_images = QPushButton('Locate images...')
        self.btn_locate_images.clicked.connect(self.open_multi)
        vlo.addWidget(self.btn_locate_images)
        vlo.addWidget(QLabel('Loaded series / images: '))
        self.lbl_n_loaded = QLabel('0 / 0')

        # How to locate the images - which is connected, which is MTF/NPS
        # identify MTF and NPS images based on
        # - MTF / NPS in seriesdescription
        # - manually select series
        # - z value range
        # - image content
        # - Assume Mercury phantom (both in same series)
        # - autoselect which MTF images to include/exclude based on variation in CT number for material with highest density
        # if delta_z between images larger than increment - regard as separate (e.g. Mercury)
        #Settings + Update button / list of series used for MTF / list of series used for NPS / show images for selected series
        vlo.addWidget(uir.LabelHeader('Locate MTF and NPS images', 3))
        hlo_detect = QHBoxLayout()
        vlo.addLayout(hlo_detect)

        vlo.addWidget(uir.HLine())

        # How to analyse
        vlo.addWidget(uir.LabelHeader('Analysis', 3))
        hlo_analyse = QHBoxLayout()
        vlo.addLayout(hlo_analyse)
        self.stack_test_tabs = QStackedWidget()
        self.tab_ct = ParamsTabCT(self, task_based=True)
        self.stack_test_tabs.addWidget(self.tab_ct)
        vlo.addWidget(self.stack_test_tabs)

        vlo.addWidget(uir.HLine())

        # Results / output
        vlo.addWidget(uir.LabelHeader('Results', 3))
        hlo_results = QHBoxLayout()
        vlo.addLayout(hlo_results)
        vlo.addWidget(self.tab_results)

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

    def open_multi(self):
        """Start open advanced dialog."""
        dlg = open_multi.OpenMultiDialog(self)
        res = dlg.exec()
        if res:
            if len(dlg.wid.open_imgs) > 0:
                img_infos = dlg.wid.open_imgs
                series_strings = [' '.join(img.series_list_strings)
                                  for img in img_infos]
                series_strings = list(set(series_strings))
                series_strings.sort()
                self.imgs = [[] for i in range(len(series_strings))]
                for img in img_infos:
                    ser = ' '.join(img.series_list_strings)
                    serno = series_strings.index(ser)
                    self.imgs[serno].append(img)
                self.lbl_n_loaded.setText(f'{len(self.imgs)} / {len(img_infos)}')

    def refresh_lists(self):
        """Refresh lists when tag pattern changed."""
        if self.path_input.text() != '':
            self.find_all_dcm()
        else:
            QMessageBox.warning(self, 'No folder selected',
                                'No folder selected so there is nothing to refresh.')

    def run_TTF(self):
        pass  # TODO

    def update_current_test(self, reset_index=False, refresh_display=True):
        if self.active_img is not None and refresh_display:
            self.update_roi()
            self.refresh_results_display()

    def update_roi(self, clear_results_test=False):
        """Recalculate ROI."""
        errmsg = None
        if self.active_img is not None:
            self.current_roi, errmsg = get_rois(
                self.active_img,
                self.gui.active_img_no, self)
        else:
            self.current_roi = None
        self.wid_image_display.canvas.roi_draw()
        self.display_errmsg(errmsg)
        if clear_results_test:
            if self.current_test in [*self.results]:
                self.results[self.current_test] = None
                self.refresh_results_display()

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
