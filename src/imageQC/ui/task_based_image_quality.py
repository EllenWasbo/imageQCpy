# -*- coding: utf-8 -*-
"""
Run TTF and NPS, d-prime on dataset.

@author: Ellen Wasbo
"""
import os
import copy
from time import time
from pathlib import Path
import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, qApp,
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QToolBar, QGroupBox,
    QTabWidget, QStackedWidget, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QAction, QListWidget, QComboBox, QLineEdit, QCheckBox,
    QDialogButtonBox, QMessageBox, QFileDialog
    )

# imageQC block start
from imageQC.ui import ui_main_methods
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.ui.ui_main_image_widgets import ImageDisplayWidget
from imageQC.ui import open_multi
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.tag_patterns import FormatDialog
from imageQC.config import config_classes as cfc
from imageQC.config import config_func as cff
from imageQC.scripts import dcm
from imageQC.ui import messageboxes
from imageQC.ui import ui_main_quicktest_paramset_select
from imageQC.ui.ui_main_test_tabs import ParamsTabCT
from imageQC.ui import ui_main_result_tabs
from imageQC.config.iQCconstants import ENV_ICON_PATH
from imageQC.scripts.mini_methods import get_uniq_ordered, get_included_tags
from imageQC.scripts.mini_methods_calculate import get_object_width_xy
from imageQC.scripts.calculate_qc import calculate_qc
# imageQC block end


class RangeDialog(ImageQCDialog):
    """Dialog to set range rule."""

    def __init__(self, task_main, range_input=None):#, parent=None):
        """Initialize RangeDialog.

        Parameters
        ----------
        task_main: TaskBasedImageQualityDialog
        range_input: list or None
            if None = add, else row as list for edit
        """
        super().__init__(parent=task_main)
        self.task_main = task_main
        self.setWindowTitle('Add/edit range rules')

        self.cbox_test = QComboBox()
        self.cbox_test.addItems(['TTF', 'NPS'])
        self.cbox_range_type = QComboBox()
        self.cbox_range_type.addItems(
            ['DICOM info', 'Max pixel value'])#, 'Phantom diameter'])
        self.cbox_tags = QComboBox()
        self.cbox_tags.addItems(self.task_main.CT_tags)
        self.txt_min = QLineEdit('None')
        self.txt_max = QLineEdit('None')
        self.txt_format = QLineEdit('')
        self.txt_format.setEnabled(False)

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        f_lo = QFormLayout()
        vlo.addLayout(f_lo)
        f_lo.addRow(QLabel('Test: '), self.cbox_test)
        f_lo.addRow(QLabel('Parameter type: '), self.cbox_range_type)
        f_lo.addRow(QLabel('Tag (if parameter is DICOM): '), self.cbox_tags)
        f_lo.addRow(QLabel('Minimum: '), self.txt_min)
        f_lo.addRow(QLabel('Minimum: '), self.txt_max)

        hlo_format = QHBoxLayout()
        vlo.addLayout(hlo_format)
        hlo_format.addWidget(QLabel('Format tag content for sorting: '))
        hlo_format.addWidget(self.txt_format)
        self.act_edit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Set format', self)
        self.act_edit.triggered.connect(self.edit_format)
        toolb = QToolBar()
        toolb.addAction(self.act_edit)
        hlo_format.addWidget(toolb)
        vlo.addWidget(uir.LabelItalic(
            'NB - mix of formats for the same tag in table might cause '
            'unexpected results'))

        btn_get_current = QPushButton(
            'Get min/max from all images')
        btn_get_current.clicked.connect(self.get_current_min_max)
        vlo.addWidget(btn_get_current)
        
        if range_input:
            self.cbox_range_type.setCurrentText(range_input[0])
            if len(range_input) == 5:
                self.cbox_tags.setCurrentText(range_input[1])
                self.txt_format.setText(range_input[-1])
            self.txt_min.setText(str(range_input[2]))
            self.txt_max.setText(str(range_input[3]))

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vlo.addWidget(buttons)

    def get_current_min_max(self):
        """Get and set min max value from dataset give selected type of parameter."""
        minimum = None
        maximum = None
        idx = self.cbox_range_type.currentIndex()
        if idx == 0:  # DICOM
            tag = self.cbox_tags.currentText()
            if tag not in self.task_main.tag_strings:
                self.task_main.get_image_values(
                    max_vals=False,
                    tag_patterns=[cfc.TagPatternFormat(
                        list_tags=[tag], list_format=[self.txt_format.text()])])
            strings = copy.deepcopy(self.task_main.tag_strings[tag])
            strings.sort()
            minimum = strings[0]
            maximum = strings[-1]
        elif idx == 1:
            if (len(self.task_main.max_pix_values) == 0
                and len(self.task_main.imgs_all) > 0):
                self.task_main.get_image_values(
                    max_vals=True, tag_patterns=None)
            minimum = str(np.min(self.task_main.max_pix_values))
            maximum = str(np.max(self.task_main.max_pix_values))
        self.txt_min.setText(minimum)
        self.txt_max.setText(maximum)

    def edit_format(self):
        """Set format string to acceptable string."""
        dlg = FormatDialog(format_string=self.txt_format.text(), parent=self)
        res = dlg.exec()
        if res:
            self.txt_format.setText(dlg.get_data())

    def get_range_info(self):
        """Get current selections and return.

        Returns
        -------
        range_info : list
            [test_string, dicom_tag or **max_pixel_value**, min, max]
        """
        range_info = [self.cbox_test.currentText()]
        idx = self.cbox_range_type.currentIndex()
        if idx == 0:  # DICOM
            range_info.append(self.cbox_tags.currentText())
        elif idx == 1:
            range_info.append('**max_pixel_value**')

        txt_min = self.txt_min.text()
        txt_max = self.txt_max.text()
        for txt in [txt_min, txt_max]:
            if txt in ['None', '']:
                range_info.append(None)
            else:
                try:
                    val = float(txt)
                except ValueError:
                    val = txt
                range_info.append(val)
        if idx == 0:
            range_info.append(self.txt_format.text())

        return range_info


class RangeWidget(QWidget):
    """Widget to set ranges to define NPS or TTF."""

    def __init__(self, task_main):
        """Initialize RangeWidget."""
        super().__init__()
        self.task_main = task_main
        self.table = RangeTableWidget(self)
        hlo = QHBoxLayout()
        self.setLayout(hlo)

        self.act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add rule', self)
        self.act_add.triggered.connect(self.add_row)
        self.act_edit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit selected rule', self)
        self.act_edit.triggered.connect(self.edit_row)
        self.act_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete rule', self)
        self.act_delete.triggered.connect(self.delete_row)
        self.act_auto_ranges = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}search.png'),
            'Autodetect rules', self)
        self.act_auto_ranges.triggered.connect(self.get_auto_ranges)
        toolb = QToolBar()
        toolb.addActions([self.act_add, self.act_edit, self.act_delete,
                          self.act_auto_ranges])
        #toolb.setOrientation(Qt.Vertical)
        hlo.addWidget(self.table)
        vlo_toolb = QVBoxLayout()
        hlo.addLayout(vlo_toolb)
        vlo_toolb.addWidget(toolb)
        btn_update_on_ranges = QPushButton('Set tests from ranges')
        btn_update_on_ranges.clicked.connect(
            self.task_main.set_marked_images_from_ranges)
        vlo_toolb.addWidget(btn_update_on_ranges)
        btn_all_ttf = QPushButton('All images TTF')
        btn_all_ttf.clicked.connect(self.task_main.set_all_ttf)
        vlo_toolb.addWidget(btn_all_ttf)

    def add_row(self):
        """Add row to table."""
        dlg = RangeDialog(self.task_main)
        res = dlg.exec()
        if res:
            info_row = dlg.get_range_info()
            rowno = 0
            if self.table.current_table is None:
                self.table.current_table = [info_row]
            else:
                sel = self.table.selectedIndexes()
                if len(sel) > 0:
                    rowno = sel[0].row()
                else:
                    rowno = self.table.rowCount()
                self.table.current_table.insert(rowno, info_row)
                self.task_main.current_paramset.ranges_table = copy.deepcopy(
                    self.table.current_table)
                self.task_main.wid_paramset.flag_edit(True)
            self.table.update_table()

    def edit_row(self):
        """Edit selected row."""
        sel = self.table.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            dlg = RangeDialog(self.task_main,# parent=self,
                              range_input=self.table.current_table[rowno])
            res = dlg.exec()
            if res:
                info_row = dlg.get_range_info()
                self.table.current_table[rowno] = info_row
                self.task_main.current_paramset.ranges_table = copy.deepcopy(
                    self.table.current_table)
                self.task_main.wid_paramset.flag_edit(True)
                self.table.update_table()

    def delete_row(self):
        """Delete row from table."""
        sel = self.table.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            self.table.current_table.pop(rowno)
            self.task_main.current_paramset.ranges_table = copy.deepcopy(
                self.table.current_table)
            self.task_main.wid_paramset.flag_edit(True)
            self.table.update_table()

    def get_auto_ranges(self):
        pass#TODO


class RangeTableWidget(QTableWidget):
    """Table widget displaying range rules."""

    def __init__(self, parent):
        """Initiate PositionTableWidget."""
        super().__init__()
        self.parent = parent
        self.current_table = None
        #self.cellChanged.connect(self.edit_current_table)
        self.update_table()

    def update_table(self):
        """Populate table with current table."""
        if self.current_table is None:  # not initiated yet
            self.current_table = (
                self.parent.task_main.current_paramset.ranges_table)

        self.blockSignals(True)
        self.clear()
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(
            ['Test', 'Parameter', 'min', 'max', 'format'])
        ch_w = self.parent.task_main.gui.char_width
        for i, width in enumerate([12, 30, 12, 12, 12]):
            self.setColumnWidth(i, width*ch_w)
        #self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        if self.current_table:
            self.setRowCount(len(self.current_table))
            for rowno, row in enumerate(self.current_table):
                for colno, val in enumerate(row):
                    twi = QTableWidgetItem(str(val))
                    twi.setTextAlignment(4)
                    twi.setFlags(twi.flags() ^ Qt.ItemIsEditable)
                    self.setItem(rowno, colno, twi)

            self.verticalHeader().setVisible(False)
            self.resizeRowsToContents()

        self.blockSignals(False)


class ExportDialog(ImageQCDialog):
    """GUI to select the output from calculated results."""
    def __init__(self, task_main):
        super().__init__(parent=task_main)
        self.task_main = task_main
        self.setWindowTitle('Export results')
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        hlo_browse = QHBoxLayout()
        vlo.addLayout(hlo_browse)
        lbl = QLabel('Selected folder for result files: ')
        hlo_browse.addWidget(lbl)
        self.path = QLineEdit()
        hlo_browse.addWidget(self.path)
        toolb = uir.ToolBarBrowse('Browse to folder')
        toolb.act_browse.triggered.connect(self.browse)
        hlo_browse.addWidget(toolb)

        hlo_select_all = QHBoxLayout()
        self.chk_select_all= QCheckBox('Select all')
        self.chk_select_all.stateChanged.connect(self.select_all)
        hlo_select_all.addWidget(self.chk_select_all)
        hlo_select_all.addStretch()
        vlo.addLayout(hlo_select_all)

        self.ttf = QGroupBox('TTF results')
        self.ttf.setCheckable(True)
        self.ttf.setChecked(False)
        self.ttf.setFont(uir.FontItalic())
        flo_ttf = QFormLayout()
        self.ttf.setLayout(flo_ttf)

        self.ttf_dcm = QCheckBox('')
        self.ttf_table = QCheckBox('')
        self.ttf_curves_gaussian = QCheckBox('')
        self.ttf_fit_gaussian = QCheckBox('')
        self.ttf_curves_discrete = QCheckBox('')
        flo_ttf.addRow(self.ttf_dcm,
                       QLabel('Table of condensed DICOM info pr group '
                              '(ttf_dcm.csv)'))
        flo_ttf.addRow(self.ttf_table,
                       QLabel('Result table pr group and material '
                              '(ttf_table.csv)'))
        flo_ttf.addRow(self.ttf_fit_gaussian,
                       QLabel('Supplement table (gaussian fit parameters) '
                              'pr group and material (ttf_table.csv)'))
        flo_ttf.addRow(QLabel(''), uir.LabelItalic(
            'Result- and supplement-tables combined if both selected.'))
        '''
        flo_ttf.addRow(self.ttf_curves_gaussian,
                       QLabel('TTF curves gaussian pr group and material '
                              '(ttf_gaussian.csv)'))
        flo_ttf.addRow(self.ttf_curves_discrete,
                       QLabel('TTF curves discrete pr group and material '
                              '(ttf_discrete.csv)'))
        '''

        self.nps = QGroupBox('NPS results')
        self.nps.setCheckable(True)
        self.nps.setChecked(False)
        self.nps.setFont(uir.FontItalic())
        flo_nps = QFormLayout()
        self.nps.setLayout(flo_nps)

        self.nps_dcm = QCheckBox('')
        self.nps_table = QCheckBox('')
        self.nps_curves_pr_image = QCheckBox('')
        self.nps_curves_average = QCheckBox('')
        flo_nps.addRow(self.nps_dcm,
                       QLabel('Table of condensed DICOM info pr group '
                              '(nps_table.csv)'))
        flo_nps.addRow(self.nps_table,
                       QLabel('Table of NPS results pr group (average) '
                              '(nps_table.csv)'))
        flo_nps.addRow(QLabel(''), uir.LabelItalic(
            'Tables above combined if both selected.'))
        '''
        flo_nps.addRow(self.nps_curves_average,
                       QLabel('NPS average curves pr group (nps_curves.csv)'))
        flo_nps.addRow(self.nps_curves_pr_image,
                       QLabel('NPS curves pr group and pr image '
                              '(nps_curves.csv)'))
        flo_nps.addRow(QLabel(''), uir.LabelItalic(
            'Curves above combined if both selected.'))
        '''

        '''
        self.dpr = QGroupBox('d-prime')
        self.dpr.setCheckable(True)
        self.dpr.setChecked(False)
        self.dpr.setFont(uir.FontItalic())
        flo_dpr = QFormLayout()
        self.dpr.setLayout(flo_dpr)

        self.dpr_dcm = QCheckBox('')
        flo_dpr.addRow(self.dpr_dcm, QLabel("Table of d' results pr group "
                                            "(d_prime.csv)"))
        '''
        hlo_csv = QHBoxLayout()
        vlo.addLayout(hlo_csv)
        hlo_csv.addWidget(QLabel('Decimal mark and separator for the csv files'))
        self.cbox_csv = QComboBox()
        self.cbox_csv.addItems([
            '. / , (decimal dot, comma separted)',
            ', / ; (decimal comma, semicolon separated'])
        idx = 0 if task_main.current_paramset.output.decimal_mark == '.' else 1
        self.cbox_csv.setCurrentIndex(idx)
        hlo_csv.addWidget(self.cbox_csv)

        vlo.addWidget(self.ttf)
        vlo.addWidget(self.nps)
        #vlo.addWidget(self.dpr)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.verify_accept)
        buttons.rejected.connect(self.reject)
        vlo.addWidget(buttons)

    def verify_accept(self):
        """Verify that path is given when ok-pressed."""
        if self.path.text() == '':
            QMessageBox.warning(self, 'No path given',
                'Please specify a path for the export.')
        else:
            self.accept()

    def browse(self):
        """Browse to result folder."""
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec():
            self.path.setText(dlg.selectedFiles()[0])

    def select_all(self):
        flag = True if self.chk_select_all.isChecked() else False
        self.ttf.setChecked(flag)
        self.ttf_dcm.setChecked(flag)
        self.ttf_table.setChecked(flag)
        #self.ttf_curves_gaussian.setChecked(flag)
        self.ttf_fit_gaussian.setChecked(flag)
        #self.ttf_curves_discrete.setChecked(flag)
        self.nps.setChecked(flag)
        self.nps_dcm.setChecked(flag)
        self.nps_table.setChecked(flag)
        #self.nps_curves_average.setChecked(flag)
        #self.nps_curves_pr_image.setChecked(flag)
        #self.dpr.setChecked(flag)
        #self.dpr_dcm.setChecked(flag)

    def get_export_selections(self):
        if self.cbox_csv.currentIndex() == 0:
            self.task_main.current_paramset.output.decimal_mark = '.'
        else:
            self.task_main.current_paramset.output.decimal_mark = ','
        if self.ttf.isChecked():
            ttf = [self.ttf_dcm.isChecked(), self.ttf_table.isChecked(),
                   self.ttf_curves_gaussian.isChecked(), self.ttf_fit_gaussian.isChecked(),
                   self.ttf_curves_discrete.isChecked()]
        else:
            ttf = None
        if self.nps.isChecked():
            nps = [self.nps_dcm.isChecked(), self.nps_table.isChecked(),
                   self.nps_curves_average.isChecked(), self.nps_curves_pr_image.isChecked()]
        else:
            nps = None
        '''
        if self.dpr.isChecked():
            dpr = [self.dpr_dcm.isChecked()]
        else:
        '''
        dpr = None

        if ttf is None and nps is None and dpr is None:
            path = ''
        else:
            path = self.path.text()
        return (path, ttf, nps, dpr)

class TaskBasedImageQualityDialog(ImageQCDialog):
    """GUI setup for the dialog window."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setWindowTitle('CT task based image quality automated analysis')
        self.run_all_active = False
        self.run_all_index = 0
        self.diameters = []  # hold diameter of object in imgs if calculated
        self.max_pix_values = []  # hold max pixel value pr img if calculated
        self.tag_strings = {}
        # dict holding dicom tag values pr image, key=attribute_name

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        vlo_right = QVBoxLayout()
        vlo_left = QVBoxLayout()
        hlo = QHBoxLayout()
        vlo.addLayout(hlo)
        hlo.addLayout(vlo_left)
        hlo.addLayout(vlo_right)

        _, self.CT_tags = get_included_tags(
                'CT', self.main.tag_infos, avoid_special_tags=False)

        # widget need parameters from input_main.py/InputMain and MainWindow
        self.current_modality = 'CT'
        self.current_test = 'DCM'
        self.tests = ['DCM', 'TTF', 'NPS', 'DPR']
        self.lastload = time()
        self.fname = 'paramsets_CT_task_based'
        _, _, self.paramsets = cff.load_settings(fname=self.fname)
        self.current_paramset = copy.deepcopy(self.paramsets[0])
        self.current_quicktest = cfc.QuickTestTemplate()

        self.imgs = []  # current selected group of images
        self.imgs_all = []  # all images
        self.imgs_group_numbers = []
        self.current_groups = ''
        self.group_strings = []

        self.results = {}  # results for active group of images, similart to main
        self.results_all = []  # results per group (list of dict)
        self.errmsgs = []
        self.current_group_indicators = []
        # string for each image if output set pr group with quicktest (paramset.output)
        self.automation_active = True
        self.active_img = None
        self.current_roi = None
        self.summed_img = None
        self.gui = copy.deepcopy(self.main.gui)
        self.gui.active_img_no = -1

        self.user_prefs = self.main.user_prefs
        self.tag_infos = self.main.tag_infos
        self.tag_patterns_special = self.main.tag_patterns_special
        self.save_blocked = self.main.save_blocked
        self.wid_image_display = ImageDisplayWidget(self, toolbar_right=False)
        self.wid_paramset = ui_main_quicktest_paramset_select.SelectParamsetWidget(
            self, fname=self.fname)
        self.wid_range = RangeWidget(self)
        self.wid_range.setMinimumHeight(200)

        self.create_result_tabs()

        min_height = 200
        min_width = 400
        self.list_groups = QListWidget()
        self.list_groups.setMinimumHeight(min_height)
        self.list_groups.setMinimumWidth(min_width)
        self.list_groups.currentRowChanged.connect(self.update_list_images)
        self.list_images = QListWidget()
        self.list_images.setMinimumHeight(min_height)
        self.list_images.setMinimumWidth(min_width)
        self.list_images.currentRowChanged.connect(self.update_active_img)

        self.btn_locate_images = QPushButton('Open images...')
        self.btn_locate_images.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}openMulti.png'))
        self.btn_locate_images.clicked.connect(self.open_multi)
        self.lbl_n_loaded = QLabel('0 / 0')

        hlo_load = QHBoxLayout()
        vlo_left.addLayout(hlo_load)
        hlo_load.addWidget(self.btn_locate_images)
        hlo_load.addWidget(QLabel('Loaded groups / images: '))
        hlo_load.addWidget(self.lbl_n_loaded)

        # How to locate the images - which is connected, which is MTF/NPS
        # identify MTF and NPS images based on
        # - Assume Mercury phantom (both in same group)
        # - z value range
        # - MTF / NPS in groupdescription
        # - manually select group
        # - image content

        # - autoselect which MTF images to include/exclude based on variation in CT number for material with highest density
        # if delta_z between images larger than increment - regard as separate (e.g. Mercury)
        #Settings + Update button / list of group used for MTF / list of group used for NPS / show images for selected group
        hlo_lists = QHBoxLayout()
        vlo_left.addLayout(hlo_lists)

        vlo_groups = QVBoxLayout()
        hlo_lists.addLayout(vlo_groups)
        vlo_groups.addWidget(uir.LabelHeader('Groups', 4))
        vlo_groups.addWidget(self.list_groups)

        vlo_images = QVBoxLayout()
        hlo_lists.addLayout(vlo_images)
        vlo_images.addWidget(uir.LabelHeader('Images in selected group', 4))
        vlo_images.addWidget(self.list_images)

        vlo_left.addWidget(self.wid_image_display)

        # How to analyse
        hlo_buttons = QHBoxLayout()
        vlo_right.addLayout(hlo_buttons)
        btn_run_all = QPushButton('Calculate all')
        btn_run_all.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}play.png'))
        btn_run_all.clicked.connect(self.run_all)
        btn_export_all = QPushButton('Export all results')
        btn_export_all.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}file.png'))
        btn_export_all.clicked.connect(self.export_all)
        hlo_buttons.addWidget(btn_run_all)
        hlo_buttons.addWidget(btn_export_all)
        
        vlo_right.addWidget(self.wid_paramset)
        vlo_right.addWidget(uir.LabelHeader(
            'Specify how to detect which images to use for '
            'TTF or NPS analysis', 4))
        vlo_right.addWidget(self.wid_range)
        self.stack_test_tabs = QStackedWidget()
        self.tab_ct = ParamsTabCT(self, task_based=True)
        self.stack_test_tabs.addWidget(self.tab_ct)
        self.tab_ct.create_tab_dpr()
        vlo_right.addWidget(self.stack_test_tabs)

        # Results / output
        vlo_right.addWidget(self.tab_results)
        self.update_paramset()

    def keyPressEvent(self, event):
        """Avoid Return actions."""
        if event.key() == Qt.Key_Return:
            pass
        else:
            super().keyPressEvent(event)

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
        input_pattern = self.current_paramset.group_tagpattern
        edit_grouping = True if len(self.imgs_all) > 0 else False
        dlg = open_multi.OpenMultiDialog(
            self, input_pattern=input_pattern, edit_grouping=edit_grouping)
        res = dlg.exec()
        if res:
            if len(dlg.wid.open_imgs) > 0:
                self.imgs_all = dlg.wid.open_imgs
                group_strings = [' '.join(
                    [s.strip() for s in img.series_list_strings])
                    for img in self.imgs_all]
                self.group_strings = get_uniq_ordered(group_strings)
                self.current_group = group_strings[0]
                self.imgs_group_numbers = []
                for imgno, img in enumerate(self.imgs_all):
                    ser = ' '.join(
                        [s.strip() for s in img.series_list_strings])
                    groupno = self.group_strings.index(ser)
                    self.imgs_group_numbers.append(groupno)
                    img.marked_quicktest = ['DCM']
                self.lbl_n_loaded.setText(
                    f'{len(self.group_strings)} / {len(self.imgs_all)}')
                self.blockSignals(True)
                self.list_groups.clear()
                self.list_groups.addItems(self.group_strings)
                self.list_groups.setCurrentRow(0)
                self.blockSignals(False)
                self.update_list_images(groupno=0)
        self.stop_wait_cursor()

        # reset value lists
        self.diameters = []  # hold diameter of object in imgs if calculated
        self.max_pix_values = []  # hold max pixel value pr img if calculated
        self.tag_strings = {}

    def update_list_images(self, groupno=None):
        """Fill list_images with all images in selected groups."""
        if groupno is None:
            groupno = self.list_groups.currentRow()
        if groupno >= 0:
            self.blockSignals(True)
            img_strings = []
            self.imgs = []
            for idx, img in enumerate(self.imgs_all):
                if self.imgs_group_numbers[idx] == groupno:
                    img_string = ' '.join(img.file_list_strings)
                    if len(img.marked_quicktest) > 1:
                        img_string = img_string + f' ({img.marked_quicktest[1]})'
                    img_strings.append(img_string)
                    self.imgs.append(img)

            self.list_images.clear()
            self.list_images.addItems(img_strings)
            self.list_images.setCurrentRow(0)
            self.blockSignals(False)
            if len(self.results_all) == self.list_groups.count():
                self.results = copy.deepcopy(self.results_all[groupno])
            self.refresh_results_display()
            self.update_active_img()

    def set_active_img(self, imgno):
        """Set active image programmatically.

        Parameters
        ----------
        imgno : int
            image number to set active
        """
        self.list_images.setCurrentRow(imgno)

    def get_image_values(self, max_vals=True, diameters=False,
                         tag_patterns=None):
        """Extract values to ranges pr image."""
        if tag_patterns is None:
            tag_patterns = []
        tag_strings_list = []
        if max_vals or diameters:  # read image array
            self.max_pix_values = []
            if diameters:
                self.diameters = []
            for img in self.imgs_all:
                img_array, tag_strings = dcm.get_img(
                    img.filepath, frame_number=img.frame_number,
                    tag_patterns=tag_patterns,
                    tag_infos=self.main.tag_infos)
                self.max_pix_values.append(np.max(img_array))
                if diameters and len(self.diameters) == 0:
                    widths = get_object_width_xy(
                        img_array, threshold_percent_max=10)
                    if None not in widths:
                        self.diameter.append(np.mean(widths) * img.pix[0])
                    else:
                        self.diameter.append(None)
                tag_strings_list.append(tag_strings)
        else:
            
            for img in self.imgs_all:
                tag_strings = dcm.get_tags(
                    img.filepath, frame_number=img.frame_number,
                    tag_patterns=tag_patterns,
                    tag_infos=self.main.tag_infos)
                tag_strings_list.append(tag_strings)

        if len(tag_strings_list) > 0:
            for i, tag in enumerate(tag_patterns[0].list_tags):
                if tag not in self.tag_strings:
                    self.tag_strings.update(
                        {tag: [img_values[i][0]
                               for img_values in tag_strings_list]})

    def get_within_range(self, string_values, minimum, maximum, format_string=''):
        """Find which string values are within min/max given format string.

        Parameters
        ----------
        string_values : list of str
            list of values formatted as strings
        minimum : str
            minimum value to test string_values agains (or None).
        maximum : str
            maximum value to test string_values against (or None)
        format_string : str, optional
            format_string as used in TagPatternFormat. The default is ''.

        Returns
        -------
        within_range_list : list of bool
        """
        values = string_values
        within_ranges_list = np.zeros(len(string_values), dtype=bool)
        within = []
        try:
            values = [float(val) for val in values]
            values = np.array(values)
            minimum = float(minimum)
            within = np.where(np.logical_and(
                values >= minimum, values <= maximum))
            within_ranges_list[within] = True
        except TypeError:
            pass
        if len(within) == 0:
            if minimum is not None:
                string_values.insert(0, minimum)
            if maximum is not None:
                string_values.append(maximum)
            sorted_idxs = np.argsort(string_values)
            sorted_strings = list(np.sort(string_values))
            index_min = None
            index_max = None
            if minimum is None:
                within_ranges_list[:] = True
            else:
                index_min = sorted_strings.index(minimum)
                within_ranges_list[sorted_idxs[index_min:]] = True
            if maximum is not None:
                index_max = sorted_strings.index(maximum)
                within_ranges_list[sorted_idxs[index_max:]] = False

        return within_ranges_list

    def set_marked_images_from_ranges(self):
        """Set marking for MTF and NPS based on defined ranges."""
        table = self.wid_range.table.current_table
        if table is not None:
            types = [row[1] for row in table]
            types_dicom = [row[1] for row in table if len(row) == 5]
            formats = [row[-1] for row in table if len(row) == 5]
            if len(types_dicom) > 0:
                tag_patterns = [cfc.TagPatternFormat(
                    list_tags=types_dicom,
                    list_format=formats)]
            else:
                tag_patterns = []
            max_vals = True if '**max_pixel_value**' in types else False
            if max_vals:
                if len(self.max_pix_values):
                    max_vals = False

            # update values if needed
            if max_vals or tag_patterns:
                self.get_image_values(max_vals=max_vals,
                                      tag_patterns=tag_patterns)

            # find indexes within ranges for ttf and nps
            idxs_ttf = np.ones(len(self.imgs_all), dtype=bool)
            idxs_nps = np.ones(len(self.imgs_all), dtype=bool)
            for row in table:
                within_range = None
                if row[1] == '**max_pixel_value**':
                    within_range = np.where(np.logical_and(
                        self.max_pix_values >= row[2],
                        self.max_pix_values <= row[3]))
                else:  # dicom tag range
                    within_range = self.get_within_range(
                        self.tag_strings[row[1]], row[2], row[3],
                        format_string=row[4])
                if within_range is not None:
                    if row[0] == 'TTF':
                        idxs_ttf = idxs_ttf * within_range
                    else:
                        idxs_nps = idxs_nps * within_range

            # add TTF or NPS to img marked_quicktest if within ranges
            if any(idxs_ttf + idxs_nps):
                for i, img in enumerate(self.imgs_all):
                    img.marked_quicktest = ['DCM']  # rest to default
                    if idxs_ttf[i]:
                        img.marked_quicktest.append('TTF')
                    elif idxs_nps[i]:
                        img.marked_quicktest.append('NPS')

            groupno = self.list_groups.currentRow()
            self.update_list_images(groupno=groupno)

    def set_all_ttf(self):
        """Mark all images for TTF analysis."""
        for img in self.imgs_all:
            img.marked_quicktest = ['DCM', 'TTF']
        groupno = self.list_groups.currentRow()
        self.update_list_images(groupno=groupno)

    def get_marked_imgs_current_test(self):
        """Return indexes in self.imgs for images in selected group."""
        idxs = []
        if self.imgs:
            idxs = [i for i, img in enumerate(self.imgs)
                if self.current_test in img.marked_quicktest]
        return idxs

    def update_active_img(self):
        """Overwrite pixmap in memory with new active image, refresh GUI."""
        self.gui.active_img_no = self.list_images.currentRow()
        groupno = self.list_groups.currentRow()
        if self.gui.active_img_no > -1 and groupno > -1:
            self.active_img, _ = dcm.get_img(
                self.imgs[self.gui.active_img_no].filepath,
                frame_number=self.imgs[self.gui.active_img_no].frame_number,
                tag_infos=self.tag_infos)

            self.refresh_img_display()
            #self.refresh_results_display(update_table=False)
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
                    marked_imgs = self.get_marked_imgs_current_test()
                    if self.gui.active_img_no in marked_imgs:
                        idx = marked_imgs.index(self.gui.active_img_no)
                        self.wid_res_tbl.result_table.blockSignals(True)
                        self.wid_res_tbl_sup.result_table.blockSignals(True)
                        self.wid_res_tbl.result_table.selectRow(idx)
                        self.wid_res_tbl_sup.result_table.selectRow(idx)
                        self.wid_res_tbl.result_table.blockSignals(False)
                        self.wid_res_tbl_sup.result_table.blockSignals(False)

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
                            self.current_test]['pr_image_sup'],
                        table_info=self.results[
                            self.current_test]['values_sup_info']
                        )
                except (KeyError, TypeError, IndexError):
                    self.wid_res_tbl_sup.result_table.clear()

            elif isinstance(wid, ui_main_result_tabs.ResultPlotWidget):
                self.wid_res_plot.plotcanvas.plot()
            elif isinstance(wid, ui_main_result_tabs.ResultImageWidget):
                self.wid_res_image.canvas.result_image_draw()

    def run_all(self):
        """Run all tests (DCM on all, TTF/NPS where set), DPR if both TTF,NPS in group."""
        if self.imgs_all:
            marked = [img.marked_quicktest for img in self.imgs_all]
            if 'TTF' not in marked and 'NPS' not in marked:
                self.set_marked_images_from_ranges()

            self.results_all = []
            current_test_before = self.current_test
            n_groups = self.list_groups.count()

            max_progress = 100 * n_groups  # %
            self.progress_modal = uir.ProgressModal(
                "Calculating...", "Cancel",
                0, max_progress, self, minimum_duration=0)

            self.run_all_active = True
            self.run_all_index = 0

            for groupno in range(n_groups):
                quicktest = cfc.QuickTestTemplate()
                self.run_all_index = groupno
                self.results = {}
                idxs = [i for i, img in enumerate(self.imgs_all)
                    if self.imgs_group_numbers[i] == groupno]
                self.imgs = []
                for idx, img in enumerate(self.imgs_all):
                    if idx in idxs:
                        quicktest.add_index(test_list=img.marked_quicktest)
                        self.imgs.append(img)

                self.current_quicktest = quicktest
                calculate_qc(self)
                self.progress_modal.setValue((groupno + 1) * 100)
                self.results_all.append(copy.deepcopy(self.results))

            self.run_all_active = False
            self.run_all_index = 0

            self.tab_ct.setCurrentIndex(self.tests.index(current_test_before))
            self.current_test = current_test_before

            # Results for selected group ready for display
            groupno = self.list_groups.currentRow()
            self.results = copy.deepcopy(self.results_all[groupno])
            self.refresh_results_display()
            if 'TTF' in self.results:
                self.refresh_img_display()
            self.progress_modal.setValue(max_progress)
            self.display_errmsg(self.errmsgs)

    def export_all(self):
        """Export all results to files."""
        def reduce_to_row(list_of_rows, idxs):
            """Reduce list of rows to one row.
            If column values not same, list values."""
            row = copy.deepcopy(list_of_rows[idxs[0]])
            for idx, row_this in enumerate(list_of_rows):
                if idx in idxs:
                    for colno, val in enumerate(row_this):
                        if val != row[colno]:
                            row[colno] = f'{row[colno]}/{val}'
            return row

        path = ''
        decimal_mark = self.current_paramset.output.decimal_mark
        sep = ';' if decimal_mark == ',' else ','
        # csv comma separated if decimal mark . else semicolonseparated
        ttf = None
        nps = None
        dpr = None
        if self.results_all:
            dlg = ExportDialog(self)
            res = dlg.exec()
            if res:
                path, ttf, nps, dpr = dlg.get_export_selections()
                if path:
                    if not os.access(path, os.W_OK):
                        path = ''
                        QMessageBox.warning(
                            self, 'Failed saving',
                            f'No writing permission for {path}')
        else:
            dlg = messageboxes.MessageBoxWithDetails(
                self, title='Missing results',
                msg='Calculate all first to generate results for export',
                icon=QMessageBox.Information)
            dlg.exec()

        if path:
            if ttf:
                dcm_ttf, table_ttf, curves_g, fit_g, curves_d = ttf
            else:
                dcm_ttf, table_ttf, curves_g, fit_g, curves_d = [False] * 5
            if nps:
                dcm_nps, table_nps, curves_avg, curves_img = nps
            else:
                dcm_nps, table_nps, curves_avg, curves_img = [False] * 4
            if dpr:
                pass  #TODO


            dcm_nps_table = None
            if dcm_ttf or dcm_nps:
                dcm_tables = {}
                if dcm_ttf:
                    dcm_tables['TTF'] = []
                if dcm_nps:
                    dcm_tables['NPS']  = []
                headers_dcm = ['Group'] + self.results_all[0]['DCM']['headers']
                for groupno, res in enumerate(self.results_all):
                    group_name = self.group_strings[groupno]
                    imgs_this = [img for i, img in enumerate(self.imgs_all)
                                 if groupno == self.imgs_group_numbers[i]]
                    dcm_values = res['DCM']['values']
                    for test in [*dcm_tables]:
                        idxs = [i for i, img in enumerate(imgs_this)
                            if test in img.marked_quicktest]
                        dcm_row = reduce_to_row(dcm_values, idxs)
                        dcm_tables[test].append([group_name] + dcm_row)
                for test, table in dcm_tables.items():
                    dataf = pd.DataFrame(table, columns=headers_dcm)
                    dataf = dataf.set_index('Group').T
                    if test == 'NPS' and table_nps:
                        dcm_nps_table = dataf
                    else:
                        path_this = Path(path) / f'{test.lower()}_dcm.csv'
                        dataf.to_csv(path_this, decimal=decimal_mark, sep=sep,
                                     index=False)

            if ttf:
                if table_ttf or fit_g:
                    # Combine result and suplement table pr group
                    headers = ['Group', 'Material']
                    suffixes = []
                    if table_ttf:
                        suffixes.append('')
                    if fit_g:
                        suffixes.append('_sup')
                    headers_set = False
                    table = []
                    for groupno, res in enumerate(self.results_all):
                        if 'TTF' in res:
                            group_name = self.group_strings[groupno]
                            row_pr_material = [
                                [group_name, row[0]] for row in res['TTF']['values']]
                            for suffix in suffixes:
                                ttf_values = res['TTF'][f'values{suffix}']
                                if headers_set is False:
                                    headers.extend(
                                        res['TTF'][f'headers{suffix}'][1:])
                                    if suffix == suffixes[-1]:
                                        headers_set = True
                                for rowno, row in enumerate(row_pr_material):
                                    row.extend(ttf_values[rowno][1:])
                            table.extend(row_pr_material)
                    dataf = pd.DataFrame(table, columns=headers)
                    dataf = dataf.set_index('Group').T
                    path_this = Path(path) / 'ttf_table.csv'
                    dataf.to_csv(path_this, decimal=decimal_mark, sep=sep)

                if curves_g or curves_d:
                    pass

            if nps:
                if table_nps:
                    headers = ['Group']
                    headers_set = False
                    table = []
                    for groupno, res in enumerate(self.results_all):
                        if 'NPS' in res:
                            group_name = self.group_strings[groupno]
                            table.append(
                                [group_name] + res['NPS']['values_sup'][0])
                            if headers_set is False:
                                headers.extend(
                                    res['NPS']['headers_sup'])
                                headers_set = True
                    dataf = pd.DataFrame(table, columns=headers)
                    dataf = dataf.set_index('Group').T
                    if dcm_nps_table is not None:
                        dataf = dcm_nps_table.append(dataf)
                    path_this = Path(path) / 'nps_table.csv'
                    dataf.to_csv(path_this, decimal=decimal_mark, sep=sep)

                if curves_avg or curves_img:
                    pass

            if dpr:
                pass

    def start_wait_cursor(self):
        """Block mouse events by wait cursor."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        qApp.processEvents()

    def stop_wait_cursor(self):
        """Return to normal mouse cursor after wait cursor."""
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        qApp.processEvents()
