# -*- coding: utf-8 -*-
"""User interface for test tabs in main window of imageQC.

@author: Ellen Wasbø
"""
import os
from dataclasses import fields
import numpy as np
import pandas as pd

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QFormLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QCheckBox, QRadioButton, QButtonGroup,
    QComboBox, QAction, QToolBar, QTableWidget, QTableWidgetItem,
    QMessageBox, QInputDialog
    )

# imageQC block start
from imageQC.config.iQCconstants import ENV_ICON_PATH, ALTERNATIVES
from imageQC.config import config_func as cff
from imageQC.config import config_classes as cfc
from imageQC.ui import reusable_widgets as uir
from imageQC.ui import messageboxes
from imageQC.ui.tag_patterns import TagPatternTreeTestDCM
from imageQC.scripts.calculate_qc import calculate_qc
# imageQC block end


class ParamsWidget(QWidget):
    """Generic super widget for test."""

    def __init__(self, parent, run_txt='Run test'):
        """Reusable run test button.

        Use self.hlo_top/.vlo_top and .hlo/.vlo as layouts for test widgets

        Parameters
        ----------
        parent : QWidget
            parent widget with run_current method (and QVBoxLayout vlo)
        run_txt : str, optionsl
            text on run button. Default is 'Run test'
        """
        super().__init__()
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        self.hlo_top = QHBoxLayout()  # if horizontal layout at top
        vlo.addLayout(self.hlo_top)
        self.vlo_top = QVBoxLayout()  # if vertical layout at top
        vlo.addLayout(self.vlo_top)
        self.hlo = QHBoxLayout()  # if horizontal second layout
        vlo.addLayout(self.hlo)
        self.vlo = QVBoxLayout()  # if vertital second layout
        vlo.addLayout(self.vlo)

        # run button at bottom
        vlo.addStretch()
        btn_run = QPushButton(run_txt)
        btn_run.setToolTip(
            'Run the current test on marked images (or all if none marked)')
        btn_run.clicked.connect(parent.run_current)
        vlo.addWidget(btn_run)


class ParamsTabCommon(QTabWidget):
    """Superclass for modality specific tests."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent
        self.flag_ignore_signals = False
        self.currentChanged.connect(self.main.update_current_test)

        self.create_tab_dcm()
        self.create_tab_roi()
        self.addTab(self.tab_dcm, "DCM")
        self.addTab(self.tab_roi, "ROI")

    def flag_edit(self, indicate_change=True):
        """Add star after cbox_paramsets to indicate any change from saved."""
        if indicate_change:
            self.main.wid_paramset.lbl_edit.setText('*')
        else:
            self.main.wid_paramset.lbl_edit.setText('')

    def update_displayed_params(self):
        """Display parameters according to current_paramset of main."""
        self.flag_ignore_signals = True
        paramset = self.main.current_paramset
        self.wid_dcm_pattern.current_template = paramset.dcm_tagpattern
        self.wid_dcm_pattern.update_data()

        # where attribute name of widget == dataclass attribute name of ParamSetXX
        attributes = fields(paramset)
        for field in attributes:
            reciever = getattr(self, field.name, None)
            content = (getattr(paramset, field.name, None)
                       if reciever is not None else None)
            if content is not None:
                if 'offset_xy' in field.name:
                    reciever.setText(f'{content[0]}, {content[1]}')
                elif field.type == 'int':
                    if hasattr(reciever, 'setCurrentIndex'):
                        reciever.setCurrentIndex(content)
                    elif hasattr(reciever, 'button'):
                        reciever.button(content).setChecked(True)
                    elif hasattr(reciever, 'setChecked'):
                        reciever.setChecked(content)
                    else:
                        reciever.setValue(content)
                elif field.type == 'float':
                    reciever.setValue(content)
                elif field.type == 'bool':
                    if hasattr(reciever, 'setChecked'):
                        reciever.setChecked(content)
                    else:  # info to programmer
                        print(f'Warning: Parameter {field.name} not set ',
                              '(ui_main_test_tabs.update_displayed_params)')
                elif field.type == 'str':
                    if hasattr(reciever, 'setText'):
                        reciever.setText(content)
                    else:  # info to programmer
                        print(f'Warning: Parameter {field.name} not set ',
                              '(ui_main_test_tabs.update_displayed_params)')

        if self.main.current_modality == 'Xray':
            self.update_NPS_independent_pixels()

        self.update_enabled()
        self.flag_ignore_signals = False

    def update_enabled(self):
        """Update enabled/disabled features."""
        paramset = self.main.current_paramset
        if paramset.roi_type == 0:
            self.roi_radius.setEnabled(True)
            self.roi_x.setEnabled(False)
            self.roi_y.setEnabled(False)
            self.roi_a.setEnabled(False)
        else:
            self.roi_radius.setEnabled(False)
            self.roi_x.setEnabled(True)
            self.roi_y.setEnabled(True)
            if paramset.roi_type == 1:
                self.roi_a.setEnabled(False)
            else:
                self.roi_a.setEnabled(True)
        # continues in subclasses if needed

    def make_param_odd_number(self, attribute='', update_roi=True,
                              clear_results=True, update_plot=True,
                              update_results_table=True, content=None):
        """Make sure number is odd number. Used for integers."""
        self.sender().blockSignals(True)
        prev_value = getattr(self.main.current_paramset, attribute)
        set_value = self.sender().value()
        if set_value > prev_value:
            max_val = self.sender().maximum()
            if set_value >= max_val:
                self.sender().setValue(max_val)
            else:
                new_val = 2 * (set_value // 2) + 1
                self.sender().setValue(new_val)
        else:
            min_val = self.sender().minimum()
            if set_value <= min_val:
                self.sender().setValue(min_val)
            else:
                new_val = 2 * (set_value // 2) - 1
                self.sender().setValue(new_val)
        self.sender().blockSignals(False)
        self.param_changed_from_gui(
            attribute=attribute, update_roi=update_roi,
            clear_results=clear_results, update_plot=update_plot,
            update_results_table=update_results_table, content=content)

    def param_changed_from_gui(self, attribute='', update_roi=True,
                               clear_results=True, update_plot=True,
                               update_results_table=True, content=None):
        """Update current_paramset with value from GUI.

        If changes found - update roi and delete results.

        Parameters
        ----------
        attribute : str
            attribute name in paramset
        update_roi : bool
            True (default) if rois have to be recalculated
        clear_results : bool
            True (default) if results have to be cleared
        update_plot : bool
            True (default) if plot settings affeccted
        update_results_table : bool
            True (default) if results table affected
        content : None
            Preset content. Default is None
        """
        if not self.flag_ignore_signals:
            if content is None:
                sender = self.sender()
                if hasattr(sender, 'setChecked'):
                    content = sender.isChecked()
                elif hasattr(sender, 'setValue'):
                    content = round(sender.value(), sender.decimals())
                    if sender.decimals() == 0:
                        content = int(content)
                elif hasattr(sender, 'setText'):
                    content = sender.text()
                elif hasattr(sender, 'setCurrentIndex'):  # QComboBox
                    content = sender.currentIndex()
                else:
                    content = None

            if content is not None:
                setattr(self.main.current_paramset, attribute, content)
                self.update_enabled()
                self.flag_edit(True)
                if update_roi:
                    self.main.update_roi()
                if clear_results:
                    self.clear_results_current_test()
                if ((update_plot or update_results_table)
                        and clear_results is False):
                    if attribute == 'mtf_gaussian':
                        self.update_values_mtf()
                    self.main.refresh_results_display()
                if all([self.main.current_modality == 'Xray',
                        self.main.current_test == 'NPS']):
                    self.update_NPS_independent_pixels()

    def clear_results_current_test(self):
        """Clear results of current test."""
        if self.main.current_test in [*self.main.results]:
            self.main.results[self.main.current_test] = None
            self.main.refresh_results_display()

    def update_values_mtf(self):
        """Update MTF table values when changing analytic vs discrete options."""
        if 'MTF' in self.main.results and self.main.modality in ['CT', 'Xray', 'MR']:
            if self.main.results['MTF']['pr_image']:
                details_dicts = self.main.results['MTF']['details_dict']
            else:
                details_dicts = [self.main.results['MTF']['details_dict']]
            try:
                mtf_gaussian = self.main.current_paramset.mtf_gaussian
                proceed = True
            except AttributeError:
                proceed = False
            if proceed:
                prefix = 'g' if mtf_gaussian else 'd'
                new_values = []
                for details_dict in details_dicts:
                    if isinstance(details_dict, dict):
                        details_dict = [details_dict]
                    new_values_this = details_dict[0][prefix + 'MTF_details']['values']
                    try:
                        new_values_this.extend(
                                details_dict[1][prefix + 'MTF_details']['values'])
                    except IndexError:
                        pass  # only if x and y dir
                    new_values.append(new_values_this)

                self.main.results['MTF']['values'] = new_values
                self.main.refresh_results_display()
                self.main.status_bar.showMessage('MTF tabular values updated', 1000)

    def update_NPS_independent_pixels(self):
        """Calculate independent pixels for NPS Xray."""
        nsub = self.main.current_paramset.nps_n_sub
        nroi = self.main.current_paramset.nps_roi_size
        self.nps_npix.setText(f'{1e-06*(((nsub*2-1)*nroi)**2):.2f} mill')

    def set_offset(self, attribute, reset=False):
        """Get last mouse click position and set as offset position for test.

        Parameters
        ----------
        attribute : str
            attribute name in paramset to be changed to offset [x,y]
        reset : bool
            Reset offset to [0,0]. Default is False.
        """
        if reset:
            pos = [0., 0.]
        else:
            sz_img_y, sz_img_x = np.shape(self.main.active_img)
            xpos = self.main.gui.last_clicked_pos[0] - 0.5 * sz_img_x
            ypos = self.main.gui.last_clicked_pos[1] - 0.5 * sz_img_y
            pos = [xpos, ypos]
        self.param_changed_from_gui(attribute=attribute, content=pos)
        self.update_displayed_params()

    def create_offset_widget(self, testcode):
        """Create widget where <testcode>_offset_xy and <testcode>_offset_mm."""
        testcode = testcode.lower()
        setattr(self, f'{testcode}_offset_xy', QLabel('0, 0'))
        setattr(self, f'{testcode}_offset_mm', BoolSelectTests(
            self, attribute=f'{testcode}_offset_mm',
            text_true='mm', text_false='pix'))
        tb_offset = QToolBar()
        btn_offset = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}selectArrow.png'),
            '''Left mouse click in image to set offset position, then fetch
            the position by clicking this button''', self)
        btn_offset_reset = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}reset.png'),
            '''Reset offset''', self)
        tb_offset.addActions(
            [btn_offset, btn_offset_reset])
        btn_offset.triggered.connect(
            lambda: self.set_offset(attribute=f'{testcode}_offset_xy'))
        btn_offset_reset.triggered.connect(
            lambda: self.set_offset(attribute=f'{testcode}_offset_xy', reset=True))
        tb_offset.addWidget(getattr(self, f'{testcode}_offset_xy'))
        tb_offset.addWidget(getattr(self, f'{testcode}_offset_mm'))
        wid_offset = QWidget()
        hlo_offset = QHBoxLayout()
        wid_offset.setLayout(hlo_offset)
        hlo_offset.addWidget(QLabel('Set extra offset'))
        hlo_offset.addWidget(tb_offset)
        setattr(self, f'wid_{testcode}_offset', wid_offset)

    def create_tab_dcm(self):
        """GUI of tab DCM."""
        self.tab_dcm = ParamsWidget(self, run_txt='Collect DICOM header information')
        self.wid_dcm_pattern = TagPatternTreeTestDCM(self)
        self.tab_dcm.vlo.addWidget(self.wid_dcm_pattern)

    def create_tab_roi(self):
        """GUI of tab ROI."""
        self.tab_roi = ParamsWidget(self, run_txt='Calculate ROI values')

        self.roi_type = QComboBox()
        self.roi_type.addItems(
            ['Circular', 'Rectangular', 'Rectangular rotated'])
        self.roi_type.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_type'))

        self.roi_radius = QDoubleSpinBox(decimals=1, minimum=0.1, maximum=10000,
                                         singleStep=0.1)
        self.roi_radius.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_radius'))
        self.roi_x = QDoubleSpinBox(decimals=1, minimum=0.1,  maximum=10000,
                                    singleStep=0.1)
        self.roi_x.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_x'))
        self.roi_y = QDoubleSpinBox(decimals=1, minimum=0.1,  maximum=10000,
                                    singleStep=0.1)
        self.roi_y.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_y'))
        self.roi_a = QDoubleSpinBox(decimals=1, minimum=0,  maximum=359.9,
                                    singleStep=0.1)
        self.roi_a.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_a'))

        self.create_offset_widget('roi')

        flo1 = QFormLayout()
        flo1.addRow(QLabel('ROI shape'), self.roi_type)
        flo1.addRow(QLabel('Radius of circular ROI (mm)'), self.roi_radius)
        self.tab_roi.hlo_top.addLayout(flo1)
        self.tab_roi.hlo_top.addStretch()
        hlo_size = QHBoxLayout()
        hlo_size.addWidget(QLabel('Rectangular ROI size width/height (mm)'))
        hlo_size.addWidget(self.roi_x)
        hlo_size.addWidget(QLabel('/'))
        hlo_size.addWidget(self.roi_y)
        hlo_size.addWidget(QLabel('  Rotation (degrees)'))
        hlo_size.addWidget(self.roi_a)
        hlo_size.addStretch()
        self.tab_roi.vlo.addLayout(hlo_size)
        hlo_offset = QHBoxLayout()
        hlo_offset.addWidget(self.wid_roi_offset)
        hlo_offset.addStretch()
        self.tab_roi.vlo.addLayout(hlo_offset)

    def create_tab_mtf(self):
        """GUI of tab MTF - common settings here."""
        self.tab_mtf = ParamsWidget(self, run_txt='Calculate MTF')

        self.mtf_type = QComboBox()
        self.mtf_type.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_type'))

        self.mtf_sampling_frequency = QDoubleSpinBox(
            decimals=3, minimum=0.001, singleStep=0.001)
        self.mtf_sampling_frequency.valueChanged.connect(
                    lambda: self.param_changed_from_gui(
                        attribute='mtf_sampling_frequency', update_roi=False,
                        clear_results=False, update_plot=False,
                        update_results_table=False))

        self.mtf_cut_lsf = QCheckBox('')
        self.mtf_cut_lsf.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_cut_lsf'))
        self.mtf_cut_lsf_w = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_cut_lsf_w.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_cut_lsf_w'))

        self.mtf_gaussian = BoolSelectTests(
            self, attribute='mtf_gaussian',
            text_true='Gaussian', text_false='Discrete',
            update_roi=False, clear_results=False, update_plot=False)

        self.mtf_plot = QComboBox()
        self.mtf_plot.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_plot',
                                                update_roi=False,
                                                clear_results=False))

    def create_tab_mtf_xray_mr(self):
        """GUI of tab MTF - common to Xray and MR."""
        self.mtf_roi_size_x = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_roi_size_x.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_roi_size_x'))
        self.mtf_roi_size_y = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_roi_size_y.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_roi_size_y'))

        self.mtf_auto_center = QGroupBox('Auto detect edge(s)')
        self.mtf_auto_center.setCheckable(True)
        self.mtf_auto_center.setFont(uir.FontItalic())
        self.mtf_auto_center.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_auto_center'))

        self.mtf_auto_center_type = QComboBox()
        self.mtf_auto_center_type.addItems(
            ['all detected edges', 'edge closest to image center'])
        self.mtf_auto_center_type.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_auto_center_type'))

        self.mtf_plot.addItems(['Edge position',
                                'Sorted pixel values', 'LSF', 'MTF'])

        self.create_offset_widget('mtf')

        hlo_size = QHBoxLayout()
        hlo_size.addWidget(QLabel('ROI width/height (mm)'))
        hlo_size.addWidget(self.mtf_roi_size_x)
        hlo_size.addWidget(QLabel('/'))
        hlo_size.addWidget(self.mtf_roi_size_y)

        vlo1 = QVBoxLayout()
        vlo1.addLayout(hlo_size)
        flo1 = QFormLayout()
        flo1.addRow(QLabel('Sampling freq. gaussian (mm-1)'),
                    self.mtf_sampling_frequency)
        vlo1.addLayout(flo1)
        vlo1.addWidget(self.wid_mtf_offset)
        self.tab_mtf.hlo.addLayout(vlo1)
        self.tab_mtf.hlo.addWidget(uir.VLine())
        vlo2 = QVBoxLayout()
        vlo2.addWidget(self.mtf_auto_center)
        hlo_gb_auto = QHBoxLayout()
        hlo_gb_auto.addWidget(QLabel('Use'))
        hlo_gb_auto.addWidget(self.mtf_auto_center_type)
        self.mtf_auto_center.setLayout(hlo_gb_auto)
        flo2 = QFormLayout()
        flo2.addRow(QLabel('Cut LSF tails'), self.mtf_cut_lsf)
        flo2.addRow(QLabel('    Cut at halfmax + (#FWHM)'), self.mtf_cut_lsf_w)
        vlo2.addLayout(flo2)
        flo3 = QFormLayout()
        flo3.addRow(QLabel('Table results from'), self.mtf_gaussian)
        flo3.addRow(QLabel('Plot'), self.mtf_plot)
        vlo2.addLayout(flo3)
        self.tab_mtf.hlo.addLayout(vlo2)

    def add_NPS_plot_settings(self):
        """Add common NPS settings and gui."""
        self.nps_sampling_frequency = QDoubleSpinBox(
            decimals=2, minimum=0.01, singleStep=0.01)
        self.nps_sampling_frequency.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_sampling_frequency'))
        self.nps_smooth_width = QDoubleSpinBox(
            decimals=2, minimum=0.01, singleStep=0.01)
        self.nps_smooth_width.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_smooth_width'))

        self.nps_normalize = QComboBox()
        self.nps_normalize.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_normalize',
                                                update_roi=False,
                                                clear_results=False))
        self.nps_normalize.addItems(['', 'Area under curve (AUC)',
                                     'Large area signal ^2 (LAS)'])
        self.nps_plot = QComboBox()
        self.nps_plot.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_plot',
                                                update_roi=False,
                                                clear_results=False))

        self.flo_nps_plot = QFormLayout()
        self.flo_nps_plot.addRow(QLabel('NPS sampling frequency (1/mm)'),
                                 self.nps_sampling_frequency)
        self.flo_nps_plot.addRow(QLabel('Smooth NPS by width (1/mm)'),
                                 self.nps_smooth_width)
        self.flo_nps_plot.addRow(QLabel('Normalize NPS curve by'), self.nps_normalize)
        self.flo_nps_plot.addRow(QLabel('Plot'), self.nps_plot)

    def run_tests(self):
        """Run all tests in current quicktest template."""
        self.main.wid_quicktest.get_current_template()
        self.main.current_quicktest = self.main.wid_quicktest.current_template
        calculate_qc(self.main)

    def run_current(self):
        """Run selected test."""
        tests = []
        marked_this = self.main.tree_file_list.get_marked_imgs_current_test()
        if len(marked_this) == 0:
            if self.main.wid_quicktest.gb_quicktest.isChecked():
                dlg = messageboxes.MessageBoxWithDetails(
                    self, title='No image marked',
                    msg=('QuickTests option is active, but no image is marked '
                         f'for the current test {self.main.current_test}'),
                    icon=QMessageBox.Warning)
                dlg.exec()
            else:
                tests = [[self.main.current_test]] * len(self.main.imgs)
        else:
            for img in range(len(self.main.imgs)):
                if img in marked_this:
                    tests.append([self.main.current_test])
                else:
                    tests.append([])
        self.main.current_quicktest.tests = tests
        calculate_qc(self.main)

        if len(marked_this) > 0:
            if self.main.gui.active_img_no not in marked_this:
                self.main.set_active_img(marked_this[0])


class ParamsTabCT(ParamsTabCommon):
    """Widget for CT tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_hom()
        self.create_tab_noi()
        self.create_tab_sli()
        self.create_tab_mtf()
        self.create_tab_ctn()
        self.create_tab_huw()
        self.create_tab_rin()
        self.create_tab_dim()
        self.create_tab_nps()

        self.addTab(self.tab_hom, "Homogeneity")
        self.addTab(self.tab_noi, "Noise")
        self.addTab(self.tab_sli, "Slice thickness")
        self.addTab(self.tab_mtf, "MTF")
        self.addTab(self.tab_ctn, "CT number")
        self.addTab(self.tab_huw, "HU water")
        self.addTab(self.tab_rin, "Ring artifacts")
        self.addTab(self.tab_dim, "Dimensions")
        self.addTab(self.tab_nps, "NPS")

    def update_enabled(self):
        """Update enabled/disabled features."""
        super().update_enabled()
        paramset = self.main.current_paramset

        if paramset.mtf_cut_lsf:
            self.mtf_cut_lsf_w.setEnabled(True)
            self.mtf_cut_lsf_w_fade.setEnabled(True)
        else:
            self.mtf_cut_lsf_w.setEnabled(False)
            self.mtf_cut_lsf_w_fade.setEnabled(False)

        if paramset.sli_type == 1:
            self.sli_ramp_distance.setEnabled(False)
        else:
            self.sli_ramp_distance.setEnabled(True)

    def update_sli_plot_options(self):
        """Update plot options for slice thickness."""
        self.sli_plot.clear()
        items = ['all']
        if self.sli_type.currentIndex() == 0:
            items.extend(['H1 upper', 'H2 lower', 'V1 left', 'V2 right'])
        elif self.sli_type.currentIndex() == 1:
            items.extend(['H1 upper', 'H2 lower',
                          'V1 left', 'V2 right', 'V1 inner', 'V2 inner'])
        else:
            items.extend(['V1 left', 'V2 right'])
        self.sli_plot.addItems(items)
        self.param_changed_from_gui(attribute='sli_type')

    def create_tab_hom(self):
        """GUI of tab Homogeneity."""
        self.tab_hom = ParamsWidget(self, run_txt='Calculate homogeneity')
        self.tab_hom.hlo_top.addWidget(uir.LabelItalic('Homogeneity'))

        self.hom_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.hom_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_size'))
        self.hom_roi_distance = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.hom_roi_distance.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_distance'))
        self.hom_roi_rotation = QDoubleSpinBox(
            decimals=1, minimum=-359.9, maximum=359.9, singleStep=0.1)
        self.hom_roi_rotation.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_rotation'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI radius (mm)'), self.hom_roi_size)
        flo.addRow(QLabel('ROI distance (mm)'), self.hom_roi_distance)
        flo.addRow(QLabel('Rotate ROI positions (deg)'), self.hom_roi_rotation)
        self.tab_hom.hlo.addLayout(flo)
        self.tab_hom.hlo.addStretch()

    def create_tab_noi(self):
        """GUI of tab Noise."""
        self.tab_noi = ParamsWidget(self, run_txt='Calculate noise')

        self.noi_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.noi_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='noi_roi_size'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI radius (mm)'), self.noi_roi_size)
        self.tab_noi.vlo_top.addSpacing(50)
        self.tab_noi.hlo.addLayout(flo)
        self.tab_noi.hlo.addStretch()

    def create_tab_sli(self):
        """GUI of tab Slice thickness."""
        self.tab_sli = ParamsWidget(self, run_txt='Calculate slice thickness')

        self.sli_type = QComboBox()
        self.sli_type.addItems(ALTERNATIVES['CT']['Sli'])
        self.sli_type.currentIndexChanged.connect(self.update_sli_plot_options)
        self.sli_ramp_distance = QDoubleSpinBox(decimals=1, minimum=0.1)
        self.sli_ramp_distance.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_ramp_distance'))
        self.sli_ramp_length = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.sli_ramp_length.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_ramp_length'))
        self.sli_background_width = QDoubleSpinBox(
            decimals=1, minimum=0.1, singleStep=0.1)
        self.sli_background_width.valueChanged.connect(
            lambda: self.param_changed_from_gui(
                attribute='sli_background_width'))
        self.sli_search_width = QDoubleSpinBox(decimals=0, minimum=1)
        self.sli_search_width.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_search_width'))
        self.sli_average_width = QDoubleSpinBox(decimals=0, minimum=1)
        self.sli_average_width.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_average_width'))
        self.sli_auto_center = QCheckBox('')
        self.sli_auto_center.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='sli_auto_center'))
        self.sli_plot = QComboBox()
        self.update_sli_plot_options()
        self.sli_plot.currentIndexChanged.connect(self.main.refresh_results_display)

        self.tab_sli.vlo_top.addStretch()
        hlo_type = QHBoxLayout()
        hlo_type.addWidget(QLabel('Ramp type'))
        hlo_type.addWidget(self.sli_type)
        hlo_type.addStretch()
        self.tab_sli.vlo_top.addLayout(hlo_type)

        flo1 = QFormLayout()
        flo1.addRow(QLabel('Center to ramp distance (mm)'), self.sli_ramp_distance)
        flo1.addRow(QLabel('Profile length (mm)'), self.sli_ramp_length)
        flo1.addRow(QLabel('Profile search margin (pix)'), self.sli_search_width)
        flo1.addRow(QLabel('Auto center'), self.sli_auto_center)
        flo2 = QFormLayout()
        flo2.addRow(
            QLabel('Within search margin, average over neighbour profiles (#)'),
            self.sli_average_width)
        flo2.addRow(QLabel('Background from profile outer (mm)'),
                    self.sli_background_width)
        flo2.addRow(QLabel('Plot image profiles'), self.sli_plot)
        self.tab_sli.hlo.addLayout(flo1)
        self.tab_sli.hlo.addWidget(uir.VLine())
        self.tab_sli.hlo.addLayout(flo2)

    def create_tab_ctn(self):
        """GUI of tab CT number."""
        self.tab_ctn = ParamsWidget(self, run_txt='Calculate CT numbers')

        self.ctn_table_widget = CTnTableWidget(self, self.main)
        if 'ParamSetCT' in str(type(self.main.current_paramset)):
            self.ctn_table_widget.table.update_table()
        self.ctn_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.ctn_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='ctn_roi_size'))
        self.ctn_search_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.ctn_search_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='ctn_search_size'))
        self.ctn_search = QCheckBox('')
        self.ctn_search.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='ctn_search'))
        self.ctn_auto_center = QCheckBox('')
        self.ctn_auto_center.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='ctn_auto_center'))

        flo1 = QFormLayout()
        flo1.addRow(QLabel('ROI radius (mm)'), self.ctn_roi_size)
        flo1.addRow(QLabel('Search for circular element'), self.ctn_search)
        flo1.addRow(QLabel('Search radius (mm)'), self.ctn_search_size)
        flo1.addRow(QLabel('Auto center'), self.ctn_auto_center)
        self.tab_ctn.hlo.addLayout(flo1)
        self.tab_ctn.hlo.addWidget(uir.VLine())
        self.tab_ctn.hlo.addWidget(self.ctn_table_widget)

    def create_tab_huw(self):
        """GUI of tab HU water."""
        self.tab_huw = ParamsWidget(self, run_txt='Calculate HU in water')

        self.huw_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.huw_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='huw_roi_size'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI radius (mm)'), self.huw_roi_size)
        self.tab_huw.vlo_top.addSpacing(50)
        self.tab_huw.hlo.addLayout(flo)
        self.tab_huw.hlo.addStretch()

    def create_tab_mtf(self):
        """GUI of tab MTF."""
        super().create_tab_mtf()

        self.mtf_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_roi_size'))
        self.mtf_background_width = QDoubleSpinBox(
            decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_background_width.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_background_width'))
        self.mtf_auto_center = QCheckBox('')
        self.mtf_auto_center.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_auto_center'))
        self.mtf_cut_lsf_w_fade = QDoubleSpinBox(
            decimals=1, minimum=0, singleStep=0.1)
        self.mtf_cut_lsf_w_fade.valueChanged.connect(
            lambda: self.param_changed_from_gui(
                attribute='mtf_cut_lsf_w_fade'))
        self.mtf_type.addItems(ALTERNATIVES['CT']['MTF'])
        self.mtf_plot.addItems(['Centered xy profiles',
                                'Sorted pixel values', 'LSF', 'MTF'])

        self.mtf_cy_pr_mm = BoolSelectTests(
            self, attribute='mtf_cy_pr_mm',
            text_true='cy/mm', text_false='cy/cm',
            update_roi=False, clear_results=False)
        self.create_offset_widget('mtf')

        self.tab_mtf.hlo_top.addWidget(
            uir.UnderConstruction(txt='Method wire not finished yet.'))
        vlo1 = QVBoxLayout()
        flo1 = QFormLayout()
        flo1.addRow(QLabel('MTF method'), self.mtf_type)
        flo1.addRow(QLabel('ROI radius (mm)'), self.mtf_roi_size)
        flo1.addRow(
            QLabel('Width of background (bead method)'), self.mtf_background_width)
        flo1.addRow(QLabel('Auto center ROI in max'), self.mtf_auto_center)
        flo1.addRow(QLabel('Sampling freq. gaussian (mm-1)'),
                    self.mtf_sampling_frequency)
        vlo1.addLayout(flo1)

        vlo1.addWidget(self.wid_mtf_offset)
        self.tab_mtf.hlo.addLayout(vlo1)
        self.tab_mtf.hlo.addWidget(uir.VLine())
        vlo2 = QVBoxLayout()
        flo2 = QFormLayout()
        flo2.addRow(QLabel('Cut LSF tails'), self.mtf_cut_lsf)
        flo2.addRow(QLabel('    Cut at halfmax + n*FWHM, n='), self.mtf_cut_lsf_w)
        flo2.addRow(
            QLabel('    Fade out within n*FWHM, n='), self.mtf_cut_lsf_w_fade)
        vlo2.addLayout(flo2)
        flo3 = QFormLayout()
        flo3.addRow(QLabel('Table results from'), self.mtf_gaussian)
        flo3.addRow(QLabel('Table results as'), self.mtf_cy_pr_mm)
        flo3.addRow(QLabel('Plot'), self.mtf_plot)
        vlo2.addLayout(flo3)
        self.tab_mtf.hlo.addLayout(vlo2)

    def create_tab_rin(self):
        """GUI of tab for Ring artefacts."""
        self.tab_rin = ParamsWidget(self, run_txt='Calculate radial profile')
        info_txt = '''
        Extract radial profile to quantify ring artifacts. Image center is assumed to
        be at center of scanner.<br>
        The radial profile is extracted by sorting pixels by distance from center.<br>
        Linear trend (or mean signal) is removed from the profile to calculate min/max
        as a measure of ring artifacts.<br>
        Values closer than 4 pix from center will always be ignored.
        '''
        self.tab_rin.hlo_top.addWidget(QLabel('Quantify ring artifacts'))
        self.tab_rin.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.rin_sigma_image = QDoubleSpinBox(decimals=1, minimum=0, singleStep=0.1)
        self.rin_sigma_image.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='rin_sigma_image'))
        self.rin_sigma_profile = QDoubleSpinBox(decimals=1, minimum=0, singleStep=0.1)
        self.rin_sigma_profile.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='rin_sigma_profile'))
        self.rin_range_start = QDoubleSpinBox(decimals=1, minimum=0, singleStep=0.1)
        self.rin_range_start.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='rin_range_start'))
        self.rin_range_stop = QDoubleSpinBox(decimals=1, minimum=0, singleStep=0.1)
        self.rin_range_stop.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='rin_range_stop'))
        self.rin_subtract_trend = BoolSelectTests(
            self, attribute='rin_subtract_trend',
            text_true='linear trend from profile', text_false='mean value of profile')

        flo = QFormLayout()
        flo.addRow(
            QLabel('Gaussian filter image before extracting profile. Sigma (mm) = '),
            self.rin_sigma_image)
        flo.addRow(
            QLabel('Gaussian filter radial profile before calculations. Sigma (mm) = '),
            self.rin_sigma_profile)
        flo.addRow(
            QLabel('Ignore profile data closer than (mm) to center (min. 4*pix)'),
            self.rin_range_start)
        flo.addRow(QLabel('Ignore profile data larger than (mm) from center'),
                   self.rin_range_stop)
        self.tab_rin.hlo.addLayout(flo)
        self.tab_rin.hlo.addStretch()
        hlo_subtract = QHBoxLayout()
        hlo_subtract.addWidget(QLabel('Subtract'))
        hlo_subtract.addWidget(self.rin_subtract_trend)
        hlo_subtract.addStretch()
        self.tab_rin.vlo.addLayout(hlo_subtract)

    def create_tab_dim(self):
        """GUI of tab Dim. Test distance of rods in Catphan."""
        self.tab_dim = ParamsWidget(self, run_txt='Calculate linear dimensions')
        self.tab_dim.vlo_top.addSpacing(50)
        self.tab_dim.vlo_top.addWidget(
            QLabel('Calculate distance between rods in Catphan'))
        self.tab_dim.vlo_top.addWidget(QLabel(
            'NB: difference from expected distance in supplement table'))

    def create_tab_nps(self):
        """GUI of tab NPS."""
        self.tab_nps = ParamsWidget(self, run_txt='Calculate NPS')
        self.tab_nps.hlo_top.addWidget(uir.LabelItalic('Noise Power Spectrum (NPS)'))

        self.nps_roi_size = QDoubleSpinBox(decimals=0,
                                           minimum=10, maximum=500, singleStep=1)
        self.nps_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_roi_size'))
        self.nps_roi_distance = QDoubleSpinBox(decimals=1, minimum=0, singleStep=0.1)
        self.nps_roi_distance.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_roi_distance'))
        self.nps_n_sub = QDoubleSpinBox(decimals=0, minimum=1, singleStep=1)
        self.nps_n_sub.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_n_sub'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI size (pix)'), self.nps_roi_size)
        flo.addRow(QLabel('Radius to center of ROIs (mm)'), self.nps_roi_distance)
        flo.addRow(QLabel('Number of ROIs'), self.nps_n_sub)

        self.add_NPS_plot_settings()
        self.nps_plot.addItems(['NPS pr image', 'NPS average all images',
                                'NPS pr image + average', 'NPS all images + average'])
        self.tab_nps.hlo.addLayout(flo)
        self.tab_nps.hlo.addWidget(uir.VLine())
        self.tab_nps.hlo.addLayout(self.flo_nps_plot)


class ParamsTabXray(ParamsTabCommon):
    """Copy for modality tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_hom()
        self.create_tab_noi()
        self.create_tab_mtf()
        self.create_tab_nps()
        self.create_tab_stp()
        self.create_tab_var()

        self.addTab(self.tab_hom, "Homogeneity")
        self.addTab(self.tab_noi, "Noise")
        self.addTab(self.tab_mtf, "MTF")
        self.addTab(self.tab_nps, "NPS")
        self.addTab(self.tab_stp, "STP")
        self.addTab(self.tab_var, "Variance")

    def create_tab_hom(self):
        """GUI of tab Homogeneity."""
        self.tab_hom = ParamsWidget(self, run_txt='Calculate homogeneity')

        self.hom_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.hom_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_size'))
        self.hom_roi_rotation = QDoubleSpinBox(
            decimals=1, minimum=-359.9, maximum=359.9, singleStep=0.1)
        self.hom_roi_rotation.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_rotation'))
        self.hom_roi_distance = QDoubleSpinBox(decimals=1, minimum=0, singleStep=0.1)
        self.hom_roi_distance.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_distance'))

        self.tab_hom.vlo_top.addSpacing(50)
        flo = QFormLayout()
        flo.addRow(QLabel('ROI radius (mm)'), self.hom_roi_size)
        flo.addRow(QLabel('Rotate ROI positions (deg)'), self.hom_roi_rotation)
        flo.addRow(QLabel('ROI distance (% from center)'), self.hom_roi_distance)
        self.tab_hom.hlo.addLayout(flo)
        alt_txt = [
            'Avg and stdev for each ROI',
            'Avg for each ROI + difference from average of all ROIs',
            'Avg for each ROI + % difference from average of all ROIs'
            ]
        gb_alternative = QGroupBox('Output to table')
        gb_alternative.setFont(uir.FontItalic())
        self.hom_tab_alt = QButtonGroup()
        lo_rb = QVBoxLayout()
        for btn_no, txt in enumerate(alt_txt):
            rbtn = QRadioButton(txt)
            self.hom_tab_alt.addButton(rbtn, btn_no)
            lo_rb.addWidget(rbtn)
            rbtn.clicked.connect(self.hom_tab_alt_changed)
        gb_alternative.setLayout(lo_rb)
        self.tab_hom.hlo.addWidget(gb_alternative)

        self.tab_hom.vlo.addWidget(uir.LabelItalic(
            'Same distance for all quadrants = % of shortest'
            'center-border distance.'))
        self.tab_hom.vlo.addWidget(uir.LabelItalic(
            'Leave distance empty to set ROIs at center of each qadrant.'))

    def create_tab_noi(self):
        """GUI of tab Noise."""
        self.tab_noi = ParamsWidget(self, run_txt='Calculate noise')

        self.noi_percent = QDoubleSpinBox(decimals=1,
                                          minimum=0.1, maximum=100., singleStep=0.1)
        self.noi_percent.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='noi_percent'))

        self.tab_noi.vlo_top.addSpacing(50)
        flo = QFormLayout()
        flo.addRow(QLabel('ROI width and height (% of image)'), self.noi_percent)
        self.tab_noi.hlo.addLayout(flo)
        self.tab_noi.hlo.addStretch()

    def create_tab_mtf(self):
        """GUI of tab MTF."""
        super().create_tab_mtf()
        self.create_tab_mtf_xray_mr()

    def create_tab_nps(self):
        """GUI of tab NPS."""
        self.tab_nps = ParamsWidget(self, run_txt='Calculate NPS')
        self.tab_nps.hlo_top.addWidget(uir.LabelItalic('Noise Power Spectrum (NPS)'))
        info_txt = '''
        The large area (combined all ROIs) will be trend corrected using a second order
        polynomial fit.<br>
        The resulting corrected large area can be visualized in the Result Image tab.
        <br>
        The horizontal and vertical 1d NPS curves are extracted from the 7 lines at each
        side of the axis (excluding the axis).<br>
        Also the values closest to 0 frequency are excluded from the resampled curves.
        <br>
        And the axis is set to zero in the NPS array (see Result Image tab).
        '''
        self.tab_nps.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.nps_roi_size = QDoubleSpinBox(decimals=0, minimum=22, maximum=10000)
        self.nps_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_roi_size'))
        self.nps_n_sub = QDoubleSpinBox(decimals=0, minimum=3)
        self.nps_n_sub.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_n_sub'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI size (pix)'), self.nps_roi_size)
        flo.addRow(QLabel('Total area to analyse (n x ROI size)^2, n = '),
                   self.nps_n_sub)
        self.tab_nps.hlo.addLayout(flo)
        self.add_NPS_plot_settings()
        self.nps_plot.addItems(['NPS pr image', 'NPS all images'])
        self.nps_plot_profile = QComboBox()
        self.nps_plot_profile.currentIndexChanged.connect(
            self.main.refresh_results_display)
        self.nps_plot_profile.addItems([
            'horizontal and vertical', 'horizontal', 'vertical', 'radial', 'all'])
        self.nps_show_image = QComboBox()
        self.nps_show_image.currentIndexChanged.connect(
            self.main.refresh_results_display)
        self.nps_show_image.addItems(['2d NPS', 'large area - trend subtracted'])
        self.flo_nps_plot.addRow(QLabel('Show profile'), self.nps_plot_profile)
        self.flo_nps_plot.addRow(QLabel('Result image'), self.nps_show_image)
        self.tab_nps.hlo.addWidget(uir.VLine())
        self.tab_nps.hlo.addLayout(self.flo_nps_plot)

        hlo_npix = QHBoxLayout()
        hlo_npix.addWidget(uir.LabelItalic(
            'Number of independent pixels/image = ((n*2-1)*roi size)^2 = '))
        self.nps_npix = QLabel('? mill    ')
        hlo_npix.addWidget(self.nps_npix)
        hlo_npix.addWidget(uir.LabelItalic('(preferrably > 4 mill)'))
        hlo_npix.addStretch()
        self.tab_nps.vlo.addLayout(hlo_npix)

    def create_tab_stp(self):
        """GUI of tab STP."""
        self.tab_stp = ParamsWidget(self, run_txt='Get mean in ROI')

        self.stp_roi_size = QDoubleSpinBox(decimals=1, minimum=1., singleStep=0.1)
        self.stp_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='stp_roi_size'))

        self.tab_stp.vlo_top.addWidget(uir.LabelItalic(
            'Signal Transfer Properties (STP):<br>'
            'Control that pixel values as proportional to dose.<br>'
            'Currently no change to the image (linearization) available.'))
        self.tab_stp.hlo.addWidget(QLabel('ROI size (mm)'))
        self.tab_stp.hlo.addWidget(self.stp_roi_size)
        self.tab_stp.hlo.addStretch()

    def create_tab_var(self):
        """GUI of tab Variance."""
        self.tab_var = ParamsWidget(self, run_txt='Calculate variance image(s)')

        self.var_roi_size = QDoubleSpinBox(decimals=1, minimum=1., singleStep=0.1)
        self.var_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='var_roi_size'))
        self.var_percent = QDoubleSpinBox(decimals=1,
                                          minimum=0.1, maximum=100., singleStep=0.1)
        self.var_percent.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='var_percent'))

        self.tab_var.vlo_top.addWidget(uir.LabelItalic(
            'The variance image can reveal artifacts in the image.<br>'
            'Adjust ROI size to find artifacts of different sizes.'))

        hlo_roi_size = QHBoxLayout()
        hlo_roi_size.addWidget(QLabel('ROI size (mm)'))
        hlo_roi_size.addWidget(self.var_roi_size)
        hlo_roi_size.addWidget(QLabel('if less than 3 pix, 3 pix will be used'))
        hlo_roi_size.addStretch()
        hlo_percent = QHBoxLayout()
        hlo_percent.addWidget(QLabel('ROI width and height (% of image)'))
        hlo_percent.addWidget(self.var_percent)
        hlo_percent.addStretch()
        self.tab_var.vlo.addLayout(hlo_roi_size)
        self.tab_var.vlo.addLayout(hlo_percent)

    def hom_tab_alt_changed(self):
        """Change alternative method (columns to display)."""
        self.param_changed_from_gui(
            attribute='hom_tab_alt', content=self.hom_tab_alt.checkedId())


class GroupBoxCorrectPointSource(QGroupBox):
    """Groupbox for correction of point source curvature."""

    def __init__(self, parent, testcode='Uni',
                 chk_pos_x=None, chk_pos_y=None,
                 chk_radius=None, wid_radius=None):
        super().__init__('Correct for point source curvature')
        testcode = testcode.lower()
        self.setCheckable(True)
        self.toggled.connect(
            lambda: parent.param_changed_from_gui(
                attribute=f'{testcode}_correct'))
        chk_pos_x.toggled.connect(
            lambda: parent.param_changed_from_gui(
                attribute=f'{testcode}_correct_pos_x'))
        chk_pos_y.toggled.connect(
            lambda: parent.param_changed_from_gui(
                attribute=f'{testcode}_correct_pos_y'))
        chk_radius.toggled.connect(
            lambda: parent.param_changed_from_gui(
                attribute=f'{testcode}_lock_radius'))
        wid_radius.valueChanged.connect(
            lambda: parent.param_changed_from_gui(
                attribute=f'{testcode}_radius'))

        vlo_gb = QVBoxLayout()
        self.setLayout(vlo_gb)
        hlo_xy = QHBoxLayout()
        hlo_xy.addWidget(QLabel('Fit point source position in'))
        hlo_xy.addWidget(chk_pos_x)
        hlo_xy.addWidget(chk_pos_y)
        vlo_gb.addLayout(hlo_xy)
        hlo_lock_dist = QHBoxLayout()
        vlo_gb.addLayout(hlo_lock_dist)
        hlo_lock_dist.addWidget(chk_radius)
        hlo_lock_dist.addWidget(wid_radius)
        hlo_lock_dist.addWidget(QLabel('mm'))


class ParamsTabNM(ParamsTabCommon):
    """Copy for modality tests."""

    def __init__(self, parent):
        super().__init__(parent)
        self.create_tab_uni()
        self.create_tab_sni()
        self.create_tab_mtf()
        self.create_tab_spe()
        self.create_tab_bar()
        self.addTab(self.tab_uni, "Uniformity")
        self.addTab(self.tab_sni, "SNI")
        self.addTab(self.tab_mtf, "Spatial resolution")
        self.addTab(self.tab_spe, "Scan speed")
        self.addTab(self.tab_bar, "Bar phantom")

    def create_tab_uni(self):
        """GUI of tab Uniformity."""
        self.tab_uni = ParamsWidget(self, run_txt='Calculate uniformity')
        self.tab_uni.vlo_top.addWidget(uir.UnderConstruction())

        self.uni_ufov_ratio = QDoubleSpinBox(
            decimals=2, minimum=0.1, maximum=1., singleStep=0.01)
        self.uni_ufov_ratio.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='uni_ufov_ratio'))
        self.uni_cfov_ratio = QDoubleSpinBox(
            decimals=2, minimum=0.1, maximum=1., singleStep=0.01)
        self.uni_cfov_ratio.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='uni_cfov_ratio'))

        self.uni_correct_pos_x = QCheckBox('x')
        self.uni_correct_pos_y = QCheckBox('y')
        self.uni_correct_radius_chk = QCheckBox('Lock source distance to')
        self.uni_correct_radius = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=5000, singleStep=0.1)
        self.uni_correct = GroupBoxCorrectPointSource(
            self, testcode='uni',
            chk_pos_x=self.uni_correct_pos_x, chk_pos_y=self.uni_correct_pos_y,
            chk_radius=self.uni_correct_radius_chk, wid_radius=self.uni_correct_radius)

        self.uni_sum_first = QCheckBox('Sum marked images before analysing sum')
        self.uni_sum_first.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='uni_sum_first'))

        self.uni_plot = QComboBox()
        self.uni_plot.addItems(['Uniformity result for all images',
                                'Curvature correction check'])
        self.uni_plot.currentIndexChanged.connect(
            self.main.wid_res_plot.plotcanvas.plot)

        self.uni_result_image = QComboBox()
        self.uni_result_image.addItems(
            [
                'Differential uniformity map',
                'Processed image (6.4mm pix, smoothed, corrected)',
                'Curvature corrected image',
                'Summed image (if sum marked)'
             ])
        self.uni_result_image.currentIndexChanged.connect(
            self.main.wid_res_image.canvas.result_image_draw)

        self.tab_uni.vlo_top.addWidget(QLabel('Based on NEMA NU-1 2007'))
        vlo_left = QVBoxLayout()
        self.tab_uni.hlo.addLayout(vlo_left)
        self.tab_uni.hlo.addWidget(uir.VLine())
        vlo_right = QVBoxLayout()
        self.tab_uni.hlo.addLayout(vlo_right)

        hlo_fov = QHBoxLayout()
        flo = QFormLayout()
        flo.addRow(QLabel('UFOV ratio'), self.uni_ufov_ratio)
        flo.addRow(QLabel('CFOV ratio'), self.uni_cfov_ratio)
        hlo_fov.addLayout(flo)
        hlo_fov.addSpacing(100)
        vlo_left.addLayout(hlo_fov)
        vlo_left.addWidget(self.uni_sum_first)

        vlo_right.addWidget(self.uni_correct)

        hlo_btm = QHBoxLayout()
        hlo_btm.addStretch()
        f_btm = QFormLayout()
        hlo_btm.addLayout(f_btm)
        f_btm.addRow(QLabel('Plot'), self.uni_plot)
        f_btm.addRow(QLabel('Result image'), self.uni_result_image)
        self.tab_uni.vlo.addLayout(hlo_btm)

    def create_tab_sni(self):
        """GUI of tab SNI."""
        self.tab_sni = ParamsWidget(self, run_txt='Calculate SNI')

        self.sni_area_ratio = QDoubleSpinBox(
            decimals=2, minimum=0.1, maximum=1., singleStep=0.01)
        self.sni_area_ratio.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sni_area_ratio'))

        self.sni_correct_pos_x = QCheckBox('x')
        self.sni_correct_pos_y = QCheckBox('y')
        self.sni_correct_radius_chk = QCheckBox('Lock source distance to')
        self.sni_correct_radius = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=5000, singleStep=0.1)
        self.sni_correct = GroupBoxCorrectPointSource(
            self, testcode='sni',
            chk_pos_x=self.sni_correct_pos_x, chk_pos_y=self.sni_correct_pos_y,
            chk_radius=self.sni_correct_radius_chk, wid_radius=self.sni_correct_radius)

        gb_eye_filter = QGroupBox('Human visual respose filter')
        self.sni_eye_filter_f = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=10, singleStep=0.1)
        self.sni_eye_filter_f.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sni_eye_filter_f'))
        self.sni_eye_filter_c = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=100, singleStep=1)
        self.sni_eye_filter_c.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sni_eye_filter_c'))
        self.sni_eye_filter_r = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=500, singleStep=1)
        self.sni_eye_filter_r.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sni_eye_filter_r'))

        self.sni_sum_first = QCheckBox('Sum marked images before analysing sum')
        self.sni_sum_first.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='sni_sum_first'))

        self.sni_plot = QComboBox()
        self.sni_plot.addItems(['SNI values',
                                'Power spectrums used to calculate SNI'])
        self.sni_plot.currentIndexChanged.connect(
            self.main.wid_res_plot.plotcanvas.plot)

        self.sni_result_image = QComboBox()
        self.sni_result_image.addItems(
            ['2d NPS', 'Curvature corrected image', 'Summed image (if sum marked)'])
        self.sni_result_image.currentIndexChanged.connect(
            self.main.wid_res_plot.plotcanvas.plot)

        self.tab_sni.vlo_top.addWidget(uir.UnderConstruction())
        self.tab_sni.vlo_top.addWidget(QLabel(
            'SNI = Structured Noise Index (J Nucl Med 2014; 55:169-174)'))

        vlo_left = QVBoxLayout()
        self.tab_sni.hlo.addLayout(vlo_left)
        self.tab_sni.hlo.addWidget(uir.VLine())
        vlo_right = QVBoxLayout()
        self.tab_sni.hlo.addLayout(vlo_right)

        flo = QFormLayout()
        flo.addRow(QLabel('Ratio of nonzero part of image to be analysed'),
                   self.sni_area_ratio)
        vlo_left.addLayout(flo)

        vlo_eye = QVBoxLayout()
        vlo_eye.addWidget(QLabel('filter = r<sup>f</sup> exp[-cr<sup>2</sup> ]'))
        hlo_eye = QHBoxLayout()
        vlo_eye.addLayout(hlo_eye)
        hlo_eye.addWidget(QLabel('f'))
        hlo_eye.addWidget(self.sni_eye_filter_f)
        hlo_eye.addWidget(QLabel('c'))
        hlo_eye.addWidget(self.sni_eye_filter_c)
        hlo_eye.addWidget(QLabel('display (mm)'))
        hlo_eye.addWidget(self.sni_eye_filter_r)
        gb_eye_filter.setLayout(vlo_eye)
        vlo_left.addWidget(gb_eye_filter)
        vlo_left.addWidget(self.sni_sum_first)

        vlo_right.addWidget(self.sni_correct)

        f_btm = QFormLayout()
        vlo_right.addLayout(f_btm)
        f_btm.addRow(QLabel('Plot'), self.sni_plot)
        f_btm.addRow(QLabel('Result image'), self.sni_result_image)

    def create_tab_mtf(self):
        """GUI of tab MTF."""
        super().create_tab_mtf()
        self.tab_mtf.vlo_top.addWidget(uir.UnderConstruction())

        self.mtf_type.addItems(ALTERNATIVES['NM']['MTF'])
        self.mtf_roi_size_x = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_roi_size_x.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_roi_size_x'))
        self.mtf_roi_size_y = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_roi_size_y.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_roi_size_y'))
        self.mtf_auto_center = QCheckBox('Auto center ROI on object signal')
        self.mtf_auto_center.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_auto_center'))
        self.mtf_plot.addItems(['Centered xy profiles', 'Line/Edge fit',
                                'Sorted pixel values', 'LSF', 'MTF'])

        hlo_size = QHBoxLayout()
        hlo_size.addWidget(QLabel('ROI width/height (mm)'))
        hlo_size.addWidget(self.mtf_roi_size_x)
        hlo_size.addWidget(QLabel('/'))
        hlo_size.addWidget(self.mtf_roi_size_y)

        vlo1 = QVBoxLayout()
        vlo1.addLayout(hlo_size)
        flo1 = QFormLayout()
        flo1.addRow(QLabel('MTF method'), self.mtf_type)
        flo1.addRow(QLabel('Sampling freq. gaussian (mm-1)'),
                    self.mtf_sampling_frequency)
        vlo1.addLayout(flo1)
        vlo1.addStretch()
        self.tab_mtf.hlo.addLayout(vlo1)
        self.tab_mtf.hlo.addWidget(uir.VLine())
        vlo2 = QVBoxLayout()
        vlo2.addWidget(self.mtf_auto_center)
        vlo2.addSpacing(50)
        flo2 = QFormLayout()
        flo2.addRow(QLabel('Cut LSF tails'), self.mtf_cut_lsf)
        flo2.addRow(QLabel('    Cut at halfmax + (#FWHM)'), self.mtf_cut_lsf_w)
        vlo2.addLayout(flo2)
        flo3 = QFormLayout()
        flo3.addRow(QLabel('Table results from'), self.mtf_gaussian)
        flo3.addRow(QLabel('Plot'), self.mtf_plot)
        vlo2.addLayout(flo3)
        self.tab_mtf.hlo.addLayout(vlo2)

    def create_tab_spe(self):
        """GUI of tab Scan speed."""
        self.tab_spe = ParamsWidget(self, run_txt='Calculate scan speed profile')

        self.tab_spe.hlo_top.addWidget(uir.LabelItalic(
            'Control uniform signal in longitudinal direction during planar whole body'
            ' scan (stable speed).'))
        info_txt = '''
        Find the average longitudinal profile to evaluate uniformity in planar
        whole body scans.<br>
        <br>
        Test descriptions:<br>
        - IAEA Human Health Series No.6:<a href=
        "https://www.iaea.org/publications/8119/quality-assurance-for-spect-systems">
        Quality Assurance for SPECT Systems (2009)</a>, chapter 3.2.2 Test of
        scan speed<br>
        - <a href="https://richtlijnendatabase.nl/gerelateerde_documenten/f/17299/Whole%20Body%20Gamma%20Camera.pdf">
        Richtlijnendatabase - Whole Body Gamma Camera</a> - Test: Whole body uniformity
        '''
        self.tab_spe.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.spe_avg = QDoubleSpinBox(
            decimals=0, minimum=1, maximum=1000, singleStep=1)
        self.spe_avg.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='spe_avg'))
        self.spe_height = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=200, singleStep=1)
        self.spe_height.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='spe_height'))
        self.spe_filter_w = QDoubleSpinBox(
            decimals=0, minimum=0, maximum=50, singleStep=1)
        self.spe_filter_w.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='spe_filter_w'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI width for averaging profile (pix)'), self.spe_avg)
        flo.addRow(QLabel('ROI height = profile length (cm)'), self.spe_height)
        flo.addRow(QLabel('Gaussian filter sigma (pix)'),
                   self.spe_filter_w)
        self.tab_spe.hlo.addLayout(flo)
        self.tab_spe.hlo.addStretch()

    def create_tab_bar(self):
        """GUI of tab Bar Phantom."""
        self.tab_bar = ParamsWidget(self, run_txt='Calculate MTF/FWHM from bar image')

        self.tab_bar.hlo_top.addWidget(uir.LabelItalic(
            'Retrieve MTF and FWHM from quadrant bar phantom'))
        info_txt = '''
        Based on Hander et al. Med Phys 24 (2) 1997;327-334<br>
        <br>
        MTF (f=1/(2*barwidth)) = SQRT(2 * ROvariance - ROImean) / ROImean<br>
        FWHM = barwidth * 4/pi * SQRT(LN(2)) * SQRT(LN(1/MTF))<br>
        <br>
        ROIs sorted by variance to automatically find widest and narrowest bars.
        '''
        self.tab_bar.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        hlo_gb = QHBoxLayout()
        for i in range(1, 5):
            setattr(self, f'bar_width_{i}', QDoubleSpinBox(
                decimals=1, minimum=0.1, maximum=10, singleStep=0.1))
            this_spin = getattr(self, f'bar_width_{i}')
            this_spin.valueChanged.connect(
                lambda: self.param_changed_from_gui(attribute=f'bar_width_{i}'))
            hlo_gb.addWidget(QLabel(f'{i}:'))
            hlo_gb.addWidget(this_spin)
            hlo_gb.addSpacing(50)

        self.bar_roi_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000, singleStep=0.1)
        self.bar_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='bar_roi_size'))

        gb_bar = QGroupBox('Bar widths (mm) (1 = widest, 4 = narrowest)')
        gb_bar.setLayout(hlo_gb)
        self.tab_bar.hlo.addWidget(gb_bar)
        self.tab_bar.hlo.addStretch()

        hlo_roi = QHBoxLayout()
        hlo_roi.addWidget(QLabel('ROI size (mm)'))
        hlo_roi.addWidget(self.bar_roi_size)
        hlo_roi.addStretch()
        self.tab_bar.vlo.addLayout(hlo_roi)


class ParamsTabSPECT(ParamsTabCommon):
    """Copy for modality tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_mtf()
        self.create_tab_con()

        self.addTab(self.tab_mtf, "Spatial resolution")
        self.addTab(self.tab_con, "Contrast")

    def create_tab_mtf(self):
        """GUI of tab MTF."""
        super().create_tab_mtf()

        self.tab_mtf.vlo_top.addWidget(uir.UnderConstruction())

        self.mtf_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_roi_size'))
        self.mtf_background_width = QDoubleSpinBox(
            decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_background_width.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_background_width'))
        self.mtf_auto_center = QCheckBox('')
        self.mtf_auto_center.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_auto_center'))
        self.mtf_cut_lsf_w_fade = QDoubleSpinBox(
            decimals=1, minimum=0, singleStep=0.1)
        self.mtf_cut_lsf_w_fade.valueChanged.connect(
            lambda: self.param_changed_from_gui(
                attribute='mtf_cut_lsf_w_fade'))
        self.mtf_type.addItems(ALTERNATIVES['SPECT']['MTF'])
        self.mtf_plot.addItems(['Centered xy profiles',
                                'Sorted pixel values', 'LSF', 'MTF'])

        vlo1 = QVBoxLayout()
        flo1 = QFormLayout()
        flo1.addRow(QLabel('MTF method'), self.mtf_type)
        flo1.addRow(QLabel('ROI radius (mm)'), self.mtf_roi_size)
        flo1.addRow(
            QLabel('Width of background (point method)'), self.mtf_background_width)
        flo1.addRow(QLabel('Auto center ROI in max'), self.mtf_auto_center)
        flo1.addRow(QLabel('Sampling freq. gaussian (mm-1)'),
                    self.mtf_sampling_frequency)
        vlo1.addLayout(flo1)

        self.tab_mtf.hlo.addLayout(vlo1)
        self.tab_mtf.hlo.addWidget(uir.VLine())
        vlo2 = QVBoxLayout()
        flo2 = QFormLayout()
        flo2.addRow(QLabel('Cut LSF tails'), self.mtf_cut_lsf)
        flo2.addRow(QLabel('    Cut at halfmax + n*FWHM, n='), self.mtf_cut_lsf_w)
        flo2.addRow(
            QLabel('    Fade out within n*FWHM, n='), self.mtf_cut_lsf_w_fade)
        vlo2.addLayout(flo2)
        flo3 = QFormLayout()
        flo3.addRow(QLabel('Table results from'), self.mtf_gaussian)
        flo3.addRow(QLabel('Plot'), self.mtf_plot)
        vlo2.addLayout(flo3)
        self.tab_mtf.hlo.addLayout(vlo2)

    def create_tab_con(self):
        """GUI of tab Contrast."""
        self.tab_con = ParamsWidget(self, run_txt='Calculate contrast')
        self.tab_con.vlo_top.addWidget(uir.UnderConstruction())

        #...


class ParamsTabPET(ParamsTabCommon):
    """Tab for PET tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_hom()
        self.create_tab_cro()

        self.addTab(self.tab_hom, "Homogeneity")
        self.addTab(self.tab_cro, "Cross Calibration")

    def create_tab_hom(self):
        """GUI of tab Homogeneity."""
        self.tab_hom = ParamsWidget(self, run_txt='Calculate homogeneity')
        self.tab_hom.vlo_top.addWidget(QLabel(
            'Calculate mean in ROIs and % difference from mean of all mean values.'))

        self.hom_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.hom_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_size'))
        self.hom_roi_distance = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.hom_roi_distance.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_distance'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI radius (mm)'), self.hom_roi_size)
        flo.addRow(QLabel('ROI distance (mm)'), self.hom_roi_distance)
        self.tab_hom.hlo.addLayout(flo)
        self.tab_hom.hlo.addStretch()

    def create_tab_cro(self):
        """GUI of tab Cross Calibration."""
        self.tab_cro = ParamsWidget(self, run_txt='Calculate calibration factor')
        info_txt = '''
        Based on Siemens PET-CT Cross calibration procedure with F-18.<br>
        Injected activity is read from DICOM header together with time of
        activity and scan start.<br>
        Option to set initial calibration factor different from 1.0 to skip part of
        resetting calibration factor before this test.<br>
        Images will be sorted by sliceposition during calculation.
        '''
        self.tab_cro.hlo_top.addWidget(uir.LabelItalic('PET Cross calibration F18'))
        self.tab_cro.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        # configurable settings
        self.cro_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.cro_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='cro_roi_size'))
        self.cro_volume = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=150000, singleStep=0.1)
        self.cro_volume.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='cro_volume'))

        self.cro_auto_select_slices = QGroupBox('Auto select slices')
        self.cro_auto_select_slices.setCheckable(True)
        self.cro_auto_select_slices.setFont(uir.FontItalic())
        self.cro_auto_select_slices.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='cro_auto_select_slices'))

        self.cro_percent_slices = QDoubleSpinBox(
            decimals=0, minimum=10, maximum=100, singleStep=1)
        self.cro_percent_slices.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='cro_percent_slices'))

        # input not configurable (will differ from time to time)
        self.cro_calibration_factor = QDoubleSpinBox(
            decimals=3, minimum=0.5, maximum=1.5)
        self.cro_calibration_factor.setValue(1.)
        self.cro_calibration_factor.valueChanged.connect(
            self.clear_results_current_test)

        flo0 = QFormLayout()
        flo0.addRow(QLabel('ROI size (mm)'), self.cro_roi_size)
        flo0.addRow(QLabel('Volume of container (ml)'), self.cro_volume)
        flo0.addRow(QLabel('Current calibration factor'), self.cro_calibration_factor)
        flo0.addRow(QLabel('           '),
                    uir.LabelItalic('(NB: Current calibration factor cannot be saved)'))
        self.tab_cro.hlo.addLayout(flo0)
        self.tab_cro.hlo.addStretch()

        hlo_auto_select = QHBoxLayout()
        self.cro_auto_select_slices.setLayout(hlo_auto_select)
        hlo_auto_select.addWidget(
            QLabel('Use percentage of images within FWHM of z-profile of ROIs'))
        hlo_auto_select.addWidget(self.cro_percent_slices)
        hlo_auto_select.addWidget(QLabel('%'))
        self.tab_cro.vlo.addWidget(self.cro_auto_select_slices)


class ParamsTabMR(ParamsTabCommon):
    """Tab for MR tests."""

    def __init__(self, parent):
        super().__init__(parent)
        self.create_tab_snr()
        self.create_tab_piu()
        self.create_tab_gho()
        self.create_tab_geo()
        self.create_tab_sli()
        self.create_tab_mtf()
        self.addTab(self.tab_snr, "SNR")
        self.addTab(self.tab_piu, "PIU")
        self.addTab(self.tab_gho, "Ghosting")
        self.addTab(self.tab_geo, "Geometric Distortion")
        self.addTab(self.tab_sli, "Slice thickness")
        self.addTab(self.tab_mtf, "MTF")

    def create_tab_snr(self):
        """GUI of tab SNR."""
        self.tab_snr = ParamsWidget(self, run_txt='Calculate SNR')

        self.tab_snr.hlo_top.addWidget(uir.LabelItalic('Signal to noise ratio (SNR)'))
        info_txt = '''
        Based on NEMA MS-1 2008<br>
        (SNR = S mean * sqrt(2) / stdev difference image)<br>
        <br>
        Center and size of phantom will be found from maximum x and y profiles.
        <br><br>
        Difference image is calculated from image2 - image1,
        image4 - image3 ...<br>
        If some images are marked, only marked images are considered.<br>
        If odd number of images, the last image will be ignored.<br>
        <br>
        Results in table and result image will be linked to the first of
        each two images.
        '''
        self.tab_snr.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.snr_roi_percent = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=100., singleStep=0.1)
        self.snr_roi_percent.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='snr_roi_percent'))

        self.snr_roi_cut_top = QDoubleSpinBox(
            decimals=0, minimum=0.)
        self.snr_roi_cut_top.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='snr_roi_cut_top'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI % of circular phantom'), self.snr_roi_percent)
        flo.addRow(QLabel('Cut top of ROI by (mm)'), self.snr_roi_cut_top)
        self.tab_snr.hlo.addLayout(flo)
        self.tab_snr.hlo.addStretch()

    def create_tab_piu(self):
        """GUI of tab PIU."""
        self.tab_piu = ParamsWidget(self, run_txt='Calculate PIU')

        self.tab_piu.hlo_top.addWidget(uir.LabelItalic(
            'Percent Integral Uniformity (PIU)'))
        info_txt = '''
        Based on NEMA MS-3 2008
        <br><br>
        Center and size of phantom will be found from maximum x and y profiles
        to generate the ROI.
        '''
        self.tab_piu.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.piu_roi_percent = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=100., singleStep=0.1)
        self.piu_roi_percent.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='piu_roi_percent'))

        self.piu_roi_cut_top = QDoubleSpinBox(
            decimals=0, minimum=0.)
        self.piu_roi_cut_top.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='piu_roi_cut_top'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI % of circular phantom'), self.piu_roi_percent)
        flo.addRow(QLabel('Cut top of ROI by (mm)'), self.piu_roi_cut_top)
        self.tab_piu.hlo.addLayout(flo)
        self.tab_piu.hlo.addStretch()

    def create_tab_gho(self):
        """GUI of tab Ghosting."""
        self.tab_gho = ParamsWidget(self, run_txt='Calculate Ghosting')

        self.tab_gho.hlo_top.addWidget(uir.LabelItalic('Percent Signal Ghosting'))
        info_txt = '''
        Based on ACR MR Quality Control Manual, 2015<br>
        Percent Signal Ghosting (PSG) = 100 * |(left + right)-(top+bottom)| /
        2*center<br><br>
        If optimized, center and size of phantom will be found from maximum
        x and y profiles.
        '''
        self.tab_gho.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.gho_roi_central = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.gho_roi_central.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='gho_roi_central'))
        self.gho_optimize_center = QCheckBox('')
        self.gho_optimize_center.toggled.connect(
            lambda: self.param_changed_from_gui(
                attribute='gho_optimize_center'))
        self.gho_roi_w = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.gho_roi_w.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='gho_roi_w'))
        self.gho_roi_h = QDoubleSpinBox(
            decimals=1, minimum=0.1, singleStep=0.1)
        self.gho_roi_h.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='gho_roi_h'))
        self.gho_roi_dist = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.gho_roi_dist.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='gho_roi_dist'))
        self.gho_roi_cut_top = QDoubleSpinBox(decimals=0, minimum=0.)
        self.gho_roi_cut_top.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='gho_roi_cut_top'))

        hlo1 = QHBoxLayout()
        flo1 = QFormLayout()
        flo1.addRow(QLabel('Radius of central ROI (mm)'), self.gho_roi_central)
        hlo_size = QHBoxLayout()
        hlo_size.addWidget(QLabel('Width / height of outer ROI (mm)'))
        hlo_size.addWidget(self.gho_roi_w)
        hlo_size.addWidget(QLabel('/'))
        hlo_size.addWidget(self.gho_roi_h)
        hlo_size.addStretch()
        hlo2 = QHBoxLayout()
        flo2 = QFormLayout()
        flo2.addRow(QLabel('Distance from image border (mm)'), self.gho_roi_dist)
        flo2.addRow(QLabel('Cut top of ROI by (mm)'), self.gho_roi_cut_top)
        flo2.addRow(QLabel('Optimize center'), self.gho_optimize_center)
        hlo1.addLayout(flo1)
        hlo1.addStretch()
        hlo2.addLayout(flo2)
        hlo2.addStretch()
        self.tab_gho.vlo.addLayout(hlo1)
        self.tab_gho.vlo.addLayout(hlo_size)
        self.tab_gho.vlo.addLayout(hlo2)

    def create_tab_geo(self):
        """GUI of tab Geometric Distortion."""
        self.tab_geo = ParamsWidget(self, run_txt='Calculate Geometric Distortion')

        info_txt = '''
        Based on NEMA MS-2 2008<br>
        Geometric distortion = 100 * ABS(measured width - true width) /
        true width<br>
        <br>
        Measured at 0, 90, 45, 135 degrees.<br>
        Center and width of phantom will be found from maximum profiles.<br>
        NB: Slice with grid not a good option for these measurements.
        Use a homogeneous slice.<br>
        A red circle of the actual width will be overlayed the image
        after calculating the distortion.
        '''
        self.tab_geo.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.geo_actual_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, singleStep=0.1)
        self.geo_actual_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='geo_actual_size'))
        self.geo_mask_outer = QDoubleSpinBox(
            decimals=0, minimum=0)
        self.geo_mask_outer.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='geo_mask_outer'))

        flo = QFormLayout()
        flo.addRow(QLabel('Actual width of phantom (mm)'), self.geo_actual_size)
        flo.addRow(QLabel('Mask outer image (mm)'), self.geo_mask_outer)
        self.tab_geo.hlo.addLayout(flo)
        self.tab_geo.hlo.addStretch()

    def create_tab_sli(self):
        """GUI of tab Slice thickness."""
        self.tab_sli = ParamsWidget(self, run_txt='Calculate Slice thickness')

        info_txt = '''
        Based on NEMA MS-5 2018 and ACR MR Quality Control Manual, 2015<br><br>
        Slice thickness ACR = 1/10 * harmonic mean of FWHM upper and lower =
        0.2 * upper * lower / (upper + lower)<br><br>
        FWHM will be calculated for the averaged profile within each ROI,
        max from medianfiltered profile.<br><br>
        If optimized, center of phantom will be found from maximum profiles.
        '''
        # alt: Slice thickness = tan (wedge angle) * FWHM<br><br>
        '''
        self.sli_tan_a = QDoubleSpinBox(decimals=3, minimum=0.)
        self.sli_tan_a.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_tan_a'))
        '''
        self.tab_sli.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.sli_ramp_length = QDoubleSpinBox(
            decimals=1, minimum=0., maximum=200., singleStep=0.1)
        self.sli_ramp_length.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_ramp_length'))
        self.sli_search_width = QDoubleSpinBox(decimals=0, minimum=0)
        self.sli_search_width.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_search_width'))
        self.sli_dist_lower = QDoubleSpinBox(decimals=1, singleStep=0.1)
        self.sli_dist_lower.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_dist_lower'))
        self.sli_dist_upper = QDoubleSpinBox(decimals=1, singleStep=0.1)
        self.sli_dist_upper.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_dist_upper'))
        self.sli_optimize_center = QCheckBox('')
        self.sli_optimize_center.toggled.connect(
            lambda: self.param_changed_from_gui(
                attribute='sli_optimize_center'))

        hlo_dist = QHBoxLayout()
        hlo_dist.addWidget(QLabel(
            'Profile distance from image center upper / lower (mm)'))
        hlo_dist.addWidget(self.sli_dist_upper)
        hlo_dist.addWidget(QLabel('/'))
        hlo_dist.addWidget(self.sli_dist_lower)
        hlo_dist.addStretch()
        self.tab_sli.vlo.addLayout(hlo_dist)

        flo1 = QFormLayout()
        flo1.addRow(QLabel('Profile length (mm)'), self.sli_ramp_length)
        flo1.addRow(QLabel('Profile search margin (pix)'), self.sli_search_width)
        # flo1.addRow(QLabel('Tangens of wedge angle'), self.sli_tan_a)
        flo1.addRow(QLabel('Optimize center'), self.sli_optimize_center)
        self.tab_sli.hlo.addLayout(flo1)
        self.tab_sli.hlo.addWidget(uir.VLine())
        vlo_right = QVBoxLayout()
        vlo_right.addStretch()
        self.sli_plot = QComboBox()
        self.sli_plot.addItems(['both', 'upper', 'lower'])
        self.sli_plot.currentIndexChanged.connect(self.main.refresh_results_display)
        flo2 = QFormLayout()
        flo2.addRow(QLabel('Plot image profiles'), self.sli_plot)
        vlo_right.addLayout(flo2)
        self.tab_sli.hlo.addLayout(vlo_right)

    def create_tab_mtf(self):
        """GUI of tab MTF."""
        super().create_tab_mtf()
        self.create_tab_mtf_xray_mr()


class BoolSelectTests(uir.BoolSelect):
    """Radiobutton group of two returning true/false as selected value."""

    def __init__(self, parent, attribute='',
                 text_true='True', text_false='False',
                 update_roi=True, clear_results=True, update_plot=True,
                 update_results_table=True):
        """Initialize BoolSelectTests.

        Parameters
        ----------
        parent : widget
            test widget containing this BoolSelect and param_changed
        attribute : str
            For use in param_changed of ParamsTabXX. The default is ''.
        text_true : str
            Text of true value
        text_false : str
            Text of false value
        """
        super().__init__(parent,
                         text_true=text_true, text_false=text_false)

        self.attribute = attribute
        self.update_roi = update_roi
        self.clear_results = clear_results
        self.update_plot = update_plot
        self.update_results_table = update_results_table

        self.btn_true.toggled.connect(
            lambda: self.parent.param_changed_from_gui(
                attribute=self.attribute,
                update_roi=self.update_roi, clear_results=self.clear_results,
                update_plot=self.update_plot,
                update_results_table=self.update_results_table,
                content=True)
            )
        self.btn_false.toggled.connect(
            lambda: self.parent.param_changed_from_gui(
                attribute=self.attribute,
                update_roi=self.update_roi, clear_results=self.clear_results,
                update_plot=self.update_plot,
                update_results_table=self.update_results_table,
                content=False)
            )


class CTnTableWidget(QWidget):
    """CT numbers table widget."""

    def __init__(self, parent, main):
        super().__init__()
        self.parent = parent
        self.main = main

        hlo = QHBoxLayout()
        self.setLayout(hlo)

        act_import = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}import.png'),
            'Import table from clipboard', self)
        act_import.triggered.connect(self.import_table)
        act_copy = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}copy.png'),
            'Copy table to clipboard', self)
        act_copy.triggered.connect(self.copy_table)
        act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add row', self)
        act_add.triggered.connect(self.add_row)
        act_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete row', self)
        act_delete.triggered.connect(self.delete_row)
        act_get_pos = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}selectArrow.png'),
            'Get position from last mouseclick in image', self)
        act_get_pos.triggered.connect(self.get_pos_mouse)
        toolb = QToolBar()
        toolb.addActions([act_import, act_copy, act_add, act_delete, act_get_pos])
        toolb.setOrientation(Qt.Vertical)
        hlo.addWidget(toolb)
        self.table = CTnTable(self.parent, self.main)
        hlo.addWidget(self.table)

    def import_table(self):
        """Import contents to table from clipboard or from predefined."""
        dlg = messageboxes.QuestionBox(
            parent=self.main, title='Import table',
            msg='Import table from...',
            yes_text='Clipboard',
            no_text='Predefined tables')
        res = dlg.exec()
        ctn_table = None
        if res:
            dataf = pd.read_clipboard()
            nrows, ncols = dataf.shape
            if ncols != 4:
                pass #TODO ask for separator / decimal or guess?
                errmsg = [
                    'Failed reading table from clipboard.',
                    'Expected 4 columns of data that are',
                    'separated by tabs as if copied to clipboard',
                    'from ImageQC or copied from Excel.'
                    ]
                dlg = messageboxes.MessageBoxWithDetails(
                    self, title='Failed reading table',
                    msg='Failed reading table. See details.',
                    details=errmsg, icon=QMessageBox.Warning)
                dlg.exec()
            else:
                ctn_table = cfc.HUnumberTable()
                ctn_table.materials = [
                    str(dataf.iat[row, 1]) for row in range(nrows)]
                ctn_table.pos_x = [
                    float(dataf.iat[row, 2]) for row in range(nrows)]
                ctn_table.pos_y = [
                    float(dataf.iat[row, 3]) for row in range(nrows)]
                ctn_table.relative_mass_density = [
                    float(dataf.iat[row, 4]) for row in range(nrows)]
        else:
            table_dict = cff.load_default_ct_number_tables()
            if len(table_dict) > 0:
                labels = [*table_dict]
                label, ok = QInputDialog.getItem(
                    self.main, "Select predefined table",
                    "Predefined tables:", labels, 0, False)
                if ok and label:
                    ctn_table = table_dict[label]

        if ctn_table is not None:
            self.main.current_paramset.ctn_table = ctn_table
            self.parent.flag_edit(True)
            self.table.update_table()

    def copy_table(self):
        """Copy contents of table to clipboard."""
        dict_2_pd = {
            'materials': self.main.current_paramset.ctn_table.materials,
            'pos_x': self.main.current_paramset.ctn_table.pos_x,
            'pos_y': self.main.current_paramset.ctn_table.pos_y,
            'Rel. mass density':
                self.main.current_paramset.ctn_table.relative_mass_density
            }
        dataf = pd.DataFrame(dict_2_pd)
        dataf.to_clipboard(index=False)
        self.main.status_bar.showMessage('Values in clipboard', 2000)

    def add_row(self):
        """Add row to table."""
        sel = self.table.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
        else:
            rowno = self.table.rowCount()
        self.main.current_paramset.ctn_table.materials.insert(rowno, '')
        self.main.current_paramset.ctn_table.pos_x.insert(rowno, 0)
        self.main.current_paramset.ctn_table.pos_y.insert(rowno, 0)
        self.main.current_paramset.ctn_table.relative_mass_density.insert(
            rowno, 0.0)
        self.parent.flag_edit(True)
        self.table.update_table()

    def delete_row(self):
        """Delete row from table."""
        sel = self.table.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            self.main.current_paramset.ctn_table.materials.pop(rowno)
            self.main.current_paramset.ctn_table.pos_x.pop(rowno)
            self.main.current_paramset.ctn_table.pos_y.pop(rowno)
            self.main.current_paramset.ctn_table.relative_mass_density.pop(
                rowno)
            self.parent.flag_edit(True)
            self.table.update_table()

    def get_pos_mouse(self):
        """Get position from last mouseclick in i mage."""
        sel = self.table.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            sz_acty, sz_actx = np.shape(self.main.active_img)
            image_info = self.main.imgs[self.main.gui.active_img_no]
            dx_pix = self.main.gui.last_clicked_pos[0] - 0.5 * sz_actx
            dxmm = round(dx_pix * image_info.pix[0], 1)
            dy_pix = self.main.gui.last_clicked_pos[1] - 0.5 * sz_acty
            dymm = round(dy_pix * image_info.pix[1], 1)
            self.main.current_paramset.ctn_table.pos_x[rowno] = dxmm
            self.main.current_paramset.ctn_table.pos_y[rowno] = dymm
            self.parent.flag_edit(True)
            self.table.update_table()


class CTnTable(QTableWidget):
    """CT numbers table.

    Parameters
    ----------
    parent : MainWindow
    """

    def __init__(self, parent, main):
        super().__init__()
        self.main = main
        self.parent = parent
        self.cellChanged.connect(self.edit_ctn_table)

    def edit_ctn_table(self, row, col):
        """Update HUnumberTable when cell edited."""
        val = self.item(row, col).text()
        if col > 0:
            val = float(val)
        if col == 0:
            self.main.current_paramset.ctn_table.materials[row] = val
        elif col == 1:
            self.main.current_paramset.ctn_table.pos_x[row] = val
        elif col == 2:
            self.main.current_paramset.ctn_table.pos_y[row] = val
        elif col == 3:
            self.main.current_paramset.ctn_table.relative_mass_density = val
        self.parent.flag_edit(True)
        self.parent.main.update_roi(clear_results_test=True)

    def update_table(self):
        """Populate table with current HUnumberTable."""
        self.blockSignals(True)
        self.clear()
        self.setColumnCount(4)
        n_rows = len(self.main.current_paramset.ctn_table.materials)
        self.setRowCount(n_rows)
        self.setHorizontalHeaderLabels(
            ['Material', 'x pos (mm)', 'y pos (mm)', 'Rel. mass density'])
        self.verticalHeader().setVisible(False)

        values = [
            self.main.current_paramset.ctn_table.materials,
            self.main.current_paramset.ctn_table.pos_x,
            self.main.current_paramset.ctn_table.pos_y,
            self.main.current_paramset.ctn_table.relative_mass_density]

        for col in range(4):
            this_col = values[col]
            for row in range(n_rows):
                twi = QTableWidgetItem(str(this_col[row]))
                if col > 0:
                    twi.setTextAlignment(4)
                self.setItem(row, col, twi)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.blockSignals(False)
        self.parent.main.update_roi(clear_results_test=True)
