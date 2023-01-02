# -*- coding: utf-8 -*-
"""User interface for test tabs in main window of imageQC.

@author: Ellen Wasbø
"""
import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QTreeWidget, QFormLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QComboBox,
    QAction, QToolBar,
    QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QMessageBox, QFileDialog, QInputDialog,
    )

# imageQC block start
from imageQC.config.iQCconstants import (
    ENV_ICON_PATH, VENDOR_FILE_OPTIONS, ALTERNATIVES
    )
import imageQC.config.config_func as cff
import imageQC.config.config_classes as cfc
import imageQC.ui.reusables as uir
import imageQC.scripts.read_vendor_QC_reports as rvr
from imageQC.scripts.calculate_qc import calculate_qc
# imageQC block end


class TagPatternTreeTestDCM(uir.TagPatternTree):
    """Widget for test DCM. Reusable for all modalities."""

    def __init__(self, parent):
        self.main = parent.main
        self.parent = parent
        self.fname = 'tag_patterns_format'

        tit = ('Extract data from DICOM header for each image '
               'using tag pattern.')
        super().__init__(
            parent, title=tit, typestr='format', editable=False)
        tb = QToolBar()
        tb.setOrientation(Qt.Vertical)
        actEdit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit tag pattern', self)
        actEdit.triggered.connect(self.edit)
        actImport = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}import.png'),
            'Import saved tag pattern', self)
        actImport.triggered.connect(self.import_tagpattern)
        actSaveAs = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Save tag pattern as Tag Pattern Format ...', self)
        actSaveAs.triggered.connect(self.save_as)

        if self.main.save_blocked:
            actSaveAs.setEnabled(False)
        tb.addActions([actEdit, actImport, actSaveAs])
        self.hLO.addWidget(tb)
        self.hLO.addStretch()

        self.currentTemplate = self.main.current_paramset.dcm_tagpattern

    def edit(self):
        """Edit tag pattern by dialog."""
        dlg = uir.TagPatternEditDialog(
            initial_pattern=self.currentTemplate,
            modality=self.main.current_modality,
            title='Edit tag pattern for test output',
            typestr='format',
            accept_text='Use',
            reject_text='Cancel')
        res = dlg.exec()
        if res:
            self.currentTemplate = dlg.get_pattern()
            self.update_data()
            self.main.current_paramset.dcm_tagpattern = self.currentTemplate
            self.parent.flag_edit(True)

    def save_as(self):
        """Save tag pattern as new tag pattern format."""
        text, ok = QInputDialog.getText(
            self, 'Save tag pattern as...', 'Label: ')
        if ok and text != '':
            ok, path, self.templates = cff.load_settings(fname='tag_patterns_format')
            currLabels = [x.label for x in self.templates[self.main.current_modality]]
            if text in currLabels:
                QMessageBox.warning(
                    self, 'Label already in use',
                    'This label is already in use.')
            else:
                newPattern = copy.deepcopy(self.currentTemplate)
                newPattern.label = text
                self.templates[self.main.current_modality].append(
                    newPattern)
                proceed = cff.test_config_folder(self)
                if proceed:
                    ok, path = cff.save_settings(
                        self.templates, fname='tag_patterns_format')
                    if ok is False:
                        QMessageBox.warning(
                            self, 'Failed saving',
                            f'Failed saving to {path}')

    def import_tagpattern(self):
        """Import tagpattern from tag_patterns_format."""
        ok, path, templates = cff.load_settings(fname='tag_patterns_format')
        currLabels = [x.label for x in templates[self.main.current_modality]]
        text, ok = QInputDialog.getItem(self, 'Select tag pattern',
                                        'Tag pattern format:', currLabels)
        print(f'selected text {text}')
        if ok and text != '':
            idx = currLabels.index(text)
            new_template = copy.deepcopy(
                templates[self.main.current_modality][idx])
            self.main.current_paramset.dcm_tagpattern = new_template
            self.parent.flag_edit(True)
            self.currentTemplate = new_template
            self.update_data()

    def update_data(self):
        """Update tablePattern with data from active parameterset."""
        self.tablePattern.clear()
        list_tags = self.currentTemplate.list_tags
        if len(list_tags) > 0:
            for rowno, tagname in enumerate(list_tags):
                infotext = self.currentTemplate.list_format[rowno]
                row_strings = [tagname, infotext]
                item = QTreeWidgetItem(row_strings)
                self.tablePattern.addTopLevelItem(item)


class TestTabCommon(QTabWidget):
    """Superclass for modality specific tests."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent
        self.flag_ignore_signals = False
        self.currentChanged.connect(self.main.update_current_test)

        self.create_tab_DCM()
        self.create_tab_ROI()

        self.addTab(self.tabDCM, "DCM")
        self.addTab(self.tabROI, "ROI")

    def flag_edit(self, indicate_change=True):
        """Add star after cbox_paramsets to indicate any change from saved."""
        if indicate_change:
            self.main.wParamset.lbl_edit.setText('*')
        else:
            self.main.wParamset.lbl_edit.setText('')

    def update_displayed_params(self):
        """Display parameters according to current_paramset of main."""
        self.flag_ignore_signals = True
        paramset = self.main.current_paramset

        self.wDCMpattern.currentTemplate = paramset.dcm_tagpattern
        self.wDCMpattern.update_data()
        self.roi_type.setCurrentIndex(paramset.roi_type)
        self.roi_radius.setValue(paramset.roi_radius)
        self.roi_x.setValue(paramset.roi_x)
        self.roi_y.setValue(paramset.roi_y)
        self.roi_a.setValue(paramset.roi_a)
        self.roi_offset_mm.setChecked(paramset.roi_offset_mm)
        self.roi_offset_xy.setText(
            f'{paramset.roi_offset_xy[0]}, {paramset.roi_offset_xy[1]}')

        #  continues in subclasses TestTab<modality>

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

        #  continues in subclasses

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
                    content = sender.value()
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
                    if self.main.current_test in [*self.main.results]:
                        self.main.results[self.main.current_test] = None
                        self.main.refresh_results_display()
                if ((update_plot or update_results_table)
                        and clear_results is False):
                    if attribute == 'mtf_gaussian':
                        self.update_values_MTF()
                    self.main.refresh_results_display()

    def update_values_MTF(self):
        """Update MTF table values when changing gaussian vs discrete options."""
        if 'MTF' in self.main.results:
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
                for ddno, dd in enumerate(details_dicts):
                    if isinstance(dd, dict):
                        dd = [dd]
                    new_values_this = dd[0][prefix + 'MTF_details']['values']
                    try:
                        new_values_this.extend(
                                dd[1][prefix + 'MTF_details']['values'])
                    except IndexError:
                        pass  # only if x and y dir
                    new_values.append(new_values_this)

                self.main.results['MTF']['values'] = new_values
                self.main.refresh_results_display()
                self.main.statusBar.showMessage('MTF tabular values updated', 1000)

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
            szImg_y, szImg_x = np.shape(self.main.active_img)
            xpos = self.main.vGUI.last_clicked_pos[0] - 0.5 * szImg_x
            ypos = self.main.vGUI.last_clicked_pos[1] - 0.5 * szImg_y
            pos = [xpos, ypos]
        self.param_changed_from_gui(attribute=attribute, content=pos)
        self.update_displayed_params()

    def create_tab_DCM(self):
        """GUI of tab DCM."""
        self.tabDCM = QWidget()
        lo = QVBoxLayout()

        self.wDCMpattern = TagPatternTreeTestDCM(self)
        lo.addWidget(self.wDCMpattern)

        self.btnRunDCM = QPushButton('Collect DICOM header information')
        self.btnRunDCM.setToolTip('Run test')
        self.btnRunDCM.clicked.connect(self.run_current)
        lo.addWidget(self.btnRunDCM)
        lo.addStretch()

        self.tabDCM.setLayout(lo)

    def create_tab_ROI(self):
        """GUI of tab ROI."""
        self.tabROI = QWidget()

        self.roi_type = QComboBox()
        self.roi_type.addItems(
            ['Circular', 'Rectangular', 'Rectangular rotated'])
        self.roi_type.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_type'))

        self.roi_radius = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.roi_radius.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_radius'))
        self.roi_x = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.roi_x.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_x'))
        self.roi_y = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.roi_y.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_y'))
        self.roi_a = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.roi_a.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_a'))

        self.roi_offset_xy = QLabel('0, 0')
        self.roi_offset_mm = BoolSelectTests(
            self, attribute='roi_offset_mm',
            text_true='mm', text_false='pix')
        tb_roi_offset = QToolBar()
        self.btn_roi_offset = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}selectArrow.png'),
            '''Left mouse click in image to set offset position, then fetch
            the position by clicking this button''', self)
        self.btn_roi_offset_reset = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}reset.png'),
            '''Reset offset''', self)
        tb_roi_offset.addActions(
            [self.btn_roi_offset, self.btn_roi_offset_reset])
        self.btn_roi_offset.triggered.connect(
            lambda: self.set_offset(attribute='roi_offset_xy'))
        self.btn_roi_offset_reset.triggered.connect(
            lambda: self.set_offset(attribute='roi_offset_xy', reset=True))
        tb_roi_offset.addWidget(self.roi_offset_xy)
        tb_roi_offset.addWidget(self.roi_offset_mm)

        vLO = QVBoxLayout()
        hLO = QHBoxLayout()
        f1 = QFormLayout()
        f1.addRow(QLabel('ROI shape'), self.roi_type)
        f1.addRow(QLabel('Radius of circular ROI (mm)'), self.roi_radius)
        hLO.addLayout(f1)
        hLO.addStretch()
        vLO.addLayout(hLO)
        hLO_size = QHBoxLayout()
        hLO_size.addWidget(QLabel('Rectangular ROI size width/height (mm)'))
        hLO_size.addWidget(self.roi_x)
        hLO_size.addWidget(QLabel('/'))
        hLO_size.addWidget(self.roi_y)
        hLO_size.addWidget(QLabel('  Rotation (degrees)'))
        hLO_size.addWidget(self.roi_a)
        hLO_size.addStretch()
        vLO.addLayout(hLO_size)
        hLO_offset = QHBoxLayout()
        hLO_offset.addWidget(QLabel('Set extra offset'))
        hLO_offset.addWidget(tb_roi_offset)
        hLO_offset.addStretch()
        vLO.addLayout(hLO_offset)
        vLO.addStretch()

        self.btnRunROI = QPushButton('Calculate ROI values')
        self.btnRunROI.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunROI)

        self.tabROI.setLayout(vLO)

    def run_tests(self):
        """Run all tests in current quicktest template."""
        self.main.wQuickTest.get_current_template()
        self.main.current_quicktest = self.main.wQuickTest.current_template
        calculate_qc(self.main)

    def run_current(self):
        """Run selected test."""
        tests = []
        marked_this = self.main.treeFileList.get_marked_imgs_current_test()
        if len(marked_this) == 0:
            tests = [[self.main.current_test]] * len(self.main.imgs)
        else:
            for im in range(len(self.main.imgs)):
                if im in marked_this:
                    tests.append([self.main.current_test])
                else:
                    tests.append([])
        self.main.current_quicktest.tests = tests
        '''
        n_marked = 0
        qt = True if self.main.wQuickTest.gbQT.isChecked() else False
        if qt:
            for img in self.main.imgs:
                if qt and self.main.current_test in img.marked_quicktest:
                    tests.append([self.main.current_test])
                    n_marked += 1
                else:
                    tests.append([])
        else:
            for img in self.main.imgs:
                if img.marked:
                    tests.append([self.main.current_test])
                    n_marked += 1
                else:
                    tests.append([])
        if n_marked > 0:
            self.main.current_quicktest.tests = tests
        else:
            self.main.current_quicktest.tests = [
                [self.main.current_test]] * len(self.main.imgs)
        '''
        calculate_qc(self.main)

        if self.main.vGUI.active_img_no not in marked_this:
            self.main.set_active_img(marked_this[0])


class TestTabDummyForCopy(TestTabCommon):
    """Copy for modality tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_Code()  # others than DCM and ROI - already created

        # Add tabs in same order as tests defined in iQCconstants
        self.addTab(self.tabCode, "???")

    def update_displayed_params(self):
        """Display parameters according to current_paramset of main."""
        super().update_displayed_params()
        paramset = self.main.current_paramset
        '''
        self.hom_roi_size.setValue(paramset.hom_roi_size)
        self.mtf_type.setCurrentIndex(paramset.mtf_type)
        self.mtf_gaussian.setChecked(paramset.mtf_gaussian)
        '''
        # all parameters....
        self.update_enabled()
        self.flag_ignore_signals = False

    def create_tab_Code(self):
        """GUI of tab Code."""
        self.tabCode = QWidget()
        vLO = QVBoxLayout()

        '''
        Example:
        hLO = QHBoxLayout()
        f = QFormLayout()
        self.noi_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1)
        self.noi_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='noi_roi_size'))

        f.addRow(QLabel('ROI radius (mm)'), self.noi_roi_size)
        hLO.addLayout(f)
        hLO.addStretch()

        vLO.addStretch()
        vLO.addLayout(hLO)

        self.btnRunNoi = QPushButton('Calculate noise')
        self.btnRunNoi.setToolTip('Run test')
        self.btnRunNoi.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunNoi)
        vLO.addStretch()
        '''

        self.tabCode.setLayout(vLO)


class TestTabCT(TestTabCommon):
    """Widget for CT tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_Hom()
        self.create_tab_Noi()
        self.create_tab_Sli()
        self.create_tab_MTF()
        self.create_tab_CTn()
        self.create_tab_HUw()

        self.addTab(self.tabHom, "Homogeneity")
        self.addTab(self.tabNoi, "Noise")
        self.addTab(self.tabSli, "Slice thickness")
        self.addTab(self.tabMTF, "MTF")
        self.addTab(self.tabCTn, "CT number")
        self.addTab(self.tabHUw, "HU water")

    def update_displayed_params(self):
        """Display parameters according to current_paramset of main."""
        super().update_displayed_params()
        paramset = self.main.current_paramset
        self.hom_roi_size.setValue(paramset.hom_roi_size)
        self.hom_roi_distance.setValue(paramset.hom_roi_distance)
        self.hom_roi_rotation.setValue(paramset.hom_roi_rotation)
        self.noi_roi_size.setValue(paramset.noi_roi_size)
        self.huw_roi_size.setValue(paramset.huw_roi_size)
        self.mtf_type.setCurrentIndex(paramset.mtf_type)
        self.mtf_roi_size.setValue(paramset.mtf_roi_size)
        self.mtf_background_width.setValue(paramset.mtf_background_width)
        self.mtf_plot.setCurrentIndex(paramset.mtf_plot)
        self.mtf_gaussian.setChecked(paramset.mtf_gaussian)
        self.mtf_cy_pr_mm.setChecked(paramset.mtf_cy_pr_mm)
        self.mtf_offset_mm.setChecked(paramset.mtf_offset_mm)
        self.mtf_auto_center.setChecked(paramset.mtf_auto_center)
        self.mtf_cut_lsf.setChecked(paramset.mtf_cut_lsf)
        self.mtf_cut_lsf_w.setValue(paramset.mtf_cut_lsf_w)
        self.mtf_cut_lsf_w_fade.setValue(paramset.mtf_cut_lsf_w_fade)
        self.mtf_offset_xy.setText(
            f'{paramset.mtf_offset_xy[0]}, {paramset.mtf_offset_xy[1]}')
        self.mtf_sampling_frequency.setValue(paramset.mtf_sampling_frequency)
        self.ctn_roi_size.setValue(paramset.ctn_roi_size)
        self.ctn_search_size.setValue(paramset.ctn_search_size)
        self.ctn_search.setChecked(paramset.ctn_search)
        self.ctn_table_widget.table.update_table()
        self.sli_ramp_distance.setValue(paramset.sli_ramp_distance)
        self.sli_ramp_length.setValue(paramset.sli_ramp_length)
        self.sli_background_width.setValue(paramset.sli_background_width)
        self.sli_search_width.setValue(paramset.sli_search_width)
        self.sli_average_width.setValue(paramset.sli_average_width)
        self.sli_type.setCurrentIndex(paramset.sli_type)
        #DELETE?self.sli_signal_low_density.setChecked(paramset.sli_signal_low_density)
        """
        rin_median_filter_w: int = 0  # in pix on image
        rin_smooth_filter_w: float = 1.  # in mm on radial profile
        rin_range: list[float] = field(default_factory=lambda: [5., 65.])
        # mm from center
        rin_subtract_trend: bool = True
        # True = subtract trend, False = subtract mean
        nps_roi_size: int = 50
        nps_roi_dist: float = 50.
        nps_n_sub: int = 20
        nps_plot_average: bool = True
        """
        self.update_enabled()
        self.flag_ignore_signals = False

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

    def update_Sli_plot_options(self):
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

    def create_tab_Hom(self):
        """GUI of tab Homogeneity."""
        self.tabHom = QWidget()
        vLO = QVBoxLayout()
        hLO = QHBoxLayout()
        f = QFormLayout()

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

        f.addRow(QLabel('ROI radius (mm)'), self.hom_roi_size)
        f.addRow(QLabel('ROI distance (mm)'), self.hom_roi_distance)
        f.addRow(QLabel('Rotate ROI positions (deg)'), self.hom_roi_rotation)
        hLO.addLayout(f)
        hLO.addStretch()

        vLO.addStretch()
        vLO.addLayout(hLO)

        self.btnRunHom = QPushButton('Calculate homogeneity')
        self.btnRunHom.setToolTip('Run test')
        self.btnRunHom.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunHom)
        vLO.addStretch()

        self.tabHom.setLayout(vLO)

    def create_tab_Noi(self):
        """GUI of tab Noise."""
        self.tabNoi = QWidget()
        vLO = QVBoxLayout()
        hLO = QHBoxLayout()
        f = QFormLayout()

        self.noi_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.noi_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='noi_roi_size'))

        f.addRow(QLabel('ROI radius (mm)'), self.noi_roi_size)
        hLO.addLayout(f)
        hLO.addStretch()

        vLO.addStretch()
        vLO.addLayout(hLO)

        self.btnRunNoi = QPushButton('Calculate noise')
        self.btnRunNoi.setToolTip('Run test')
        self.btnRunNoi.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunNoi)
        vLO.addStretch()

        self.tabNoi.setLayout(vLO)

    def create_tab_Sli(self):
        """GUI of tab Slice thickness."""
        self.tabSli = QWidget()

        self.sli_type = QComboBox()
        self.sli_type.addItems(ALTERNATIVES['CT']['Sli'])
        self.sli_type.currentIndexChanged.connect(self.update_Sli_plot_options)
        '''
        self.sli_signal_low_density = BoolSelectTests(
            self, attribute='sli_signal_low_density',
            text_true='higher',
            text_false='lower')
        '''

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

        self.sli_plot = QComboBox()
        self.update_Sli_plot_options()
        self.sli_plot.currentIndexChanged.connect(self.main.refresh_results_display)

        vLO = QVBoxLayout()
        hLO_type = QHBoxLayout()
        hLO_type.addWidget(QLabel('Ramp type'))
        hLO_type.addWidget(self.sli_type)
        hLO_type.addStretch()
        vLO.addLayout(hLO_type)
        '''
        hLO_dens = QHBoxLayout()
        hLO_dens.addWidget(QLabel('Ramp density is'))
        hLO_dens.addWidget(self.sli_signal_low_density)
        hLO_dens.addWidget(QLabel('than background'))
        hLO_dens.addStretch()
        vLO.addLayout(hLO_dens)
        '''

        f1 = QFormLayout()
        f1.addRow(QLabel('Center to ramp distance (mm)'),
                  self.sli_ramp_distance)
        f1.addRow(QLabel('Profile length (mm)'), self.sli_ramp_length)
        f1.addRow(QLabel('Profile search margin (pix)'), self.sli_search_width)
        f2 = QFormLayout()
        f2.addRow(QLabel('Within search margin, average over neighbour profiles (#)'),
                  self.sli_average_width)
        f2.addRow(QLabel('Background from profile outer (mm)'),
                  self.sli_background_width)
        f2.addRow(QLabel('Plot image profiles'), self.sli_plot)
        hLO1 = QHBoxLayout()
        hLO1.addLayout(f1)
        hLO1.addLayout(f2)
        hLO1.addStretch()
        vLO.addLayout(hLO1)

        self.btnRunSli = QPushButton('Calculate slice thickness')
        self.btnRunSli.setToolTip('Run test')
        self.btnRunSli.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunSli)
        vLO.addStretch()

        self.tabSli.setLayout(vLO)

    def create_tab_CTn(self):
        """GUI of tab CT number."""
        self.tabCTn = QWidget()
        vLO = QVBoxLayout()

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

        hLO = QHBoxLayout()
        f1 = QFormLayout()
        f1.addRow(QLabel('ROI radius (mm)'), self.ctn_roi_size)
        f1.addRow(QLabel('Search for circular element'), self.ctn_search)
        f1.addRow(QLabel('Search radius (mm)'), self.ctn_search_size)
        hLO.addLayout(f1)
        hLO.addWidget(uir.VLine())
        hLO.addWidget(self.ctn_table_widget)

        vLO.addLayout(hLO)

        self.btnRunCTn = QPushButton('Calculate CT numbers')
        self.btnRunCTn.setToolTip('Run test')
        self.btnRunCTn.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunCTn)
        vLO.addStretch()

        self.tabCTn.setLayout(vLO)

    def create_tab_HUw(self):
        """GUI of tab HU water."""
        self.tabHUw = QWidget()
        vLO = QVBoxLayout()
        hLO = QHBoxLayout()
        f = QFormLayout()

        self.huw_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.huw_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='huw_roi_size'))

        f.addRow(QLabel('ROI radius (mm)'), self.huw_roi_size)
        hLO.addLayout(f)
        hLO.addStretch()

        vLO.addStretch()
        vLO.addLayout(hLO)

        self.btnRunHUw = QPushButton('Calculate HU in water')
        self.btnRunHUw.setToolTip('Run test')
        self.btnRunHUw.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunHUw)
        vLO.addStretch()

        self.tabHUw.setLayout(vLO)

    def create_tab_MTF(self):
        """GUI of tab MTF."""
        self.tabMTF = QWidget()

        self.mtf_type = QComboBox()
        self.mtf_type.addItems(ALTERNATIVES['CT']['MTF'])
        #  ['bead', 'wire', 'circular edge'])
        self.mtf_type.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_type'))

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
        self.mtf_cut_lsf_w_fade = QDoubleSpinBox(
            decimals=1, minimum=0, singleStep=0.1)
        self.mtf_cut_lsf_w_fade.valueChanged.connect(
            lambda: self.param_changed_from_gui(
                attribute='mtf_cut_lsf_w_fade'))

        self.mtf_plot = QComboBox()
        self.mtf_plot.addItems(['Centered xy profiles',
                                'Sorted pixel values', 'LSF', 'MTF'])
        self.mtf_plot.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_plot',
                                                update_roi=False,
                                                clear_results=False))

        self.mtf_gaussian = BoolSelectTests(
            self, attribute='mtf_gaussian',
            text_true='Gaussian', text_false='Discrete',
            update_roi=False, clear_results=False, update_plot=False)

        self.mtf_cy_pr_mm = BoolSelectTests(
            self, attribute='mtf_cy_pr_mm',
            text_true='cy/mm', text_false='cy/cm',
            update_roi=False, clear_results=False)

        self.mtf_offset_xy = QLabel('0, 0')
        self.mtf_offset_mm = BoolSelectTests(
            self, attribute='mtf_offset_mm',
            text_true='mm', text_false='pix')
        tb_mtf_offset = QToolBar()
        self.btn_mtf_offset = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}selectArrow.png'),
            '''Left mouse click in image to set offset position, then fetch
            the position by clicking this button''', self)
        self.btn_mtf_offset_reset = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}reset.png'),
            '''Reset offset''', self)
        tb_mtf_offset.addActions(
            [self.btn_mtf_offset, self.btn_mtf_offset_reset])
        self.btn_mtf_offset.triggered.connect(
            lambda: self.set_offset(attribute='mtf_offset_xy'))
        self.btn_mtf_offset_reset.triggered.connect(
            lambda: self.set_offset(attribute='mtf_offset_xy', reset=True))
        tb_mtf_offset.addWidget(self.mtf_offset_xy)
        tb_mtf_offset.addWidget(self.mtf_offset_mm)

        vLO = QVBoxLayout()
        hLO = QHBoxLayout()
        vLO1 = QVBoxLayout()
        f1 = QFormLayout()
        f1.addRow(QLabel('MTF method'), self.mtf_type)
        f1.addRow(QLabel('ROI radius (mm)'), self.mtf_roi_size)
        f1.addRow(
            QLabel('Width of background (bead method)'), self.mtf_background_width)
        f1.addRow(QLabel('Auto center ROI in max'), self.mtf_auto_center)
        f1.addRow(QLabel('Sampling freq. gaussian (mm-1)'),
                  self.mtf_sampling_frequency)
        vLO1.addLayout(f1)
        hLO_offset = QHBoxLayout()
        hLO_offset.addWidget(QLabel('Set extra offset'))
        hLO_offset.addWidget(tb_mtf_offset)
        vLO1.addLayout(hLO_offset)
        hLO.addLayout(vLO1)
        hLO.addWidget(uir.VLine())
        vLO2 = QVBoxLayout()
        f2 = QFormLayout()
        f2.addRow(QLabel('Cut LSF tails'), self.mtf_cut_lsf)
        f2.addRow(QLabel('    Cut at halfmax + (#FWHM)'), self.mtf_cut_lsf_w)
        f2.addRow(
            QLabel('    Fade out within (#FWHM)'), self.mtf_cut_lsf_w_fade)
        vLO2.addLayout(f2)
        f3 = QFormLayout()
        f3.addRow(QLabel('Table results from'), self.mtf_gaussian)
        f3.addRow(QLabel('Table results as'), self.mtf_cy_pr_mm)
        f3.addRow(QLabel('Plot'), self.mtf_plot)
        vLO2.addLayout(f3)
        hLO.addLayout(vLO2)

        vLO.addLayout(hLO)

        self.btnRunMTF = QPushButton('Calculate MTF')
        self.btnRunMTF.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunMTF)

        self.tabMTF.setLayout(vLO)


class TestTabXray(TestTabCommon):
    """Copy for modality tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_Hom()
        self.create_tab_Noi()
        self.create_tab_MTF()
        self.create_tab_NPS()
        self.create_tab_Var()

        self.addTab(self.tabHom, "Homogeneity")
        self.addTab(self.tabNoi, "Noise")
        self.addTab(self.tabMTF, "MTF")
        self.addTab(self.tabNPS, "NPS")
        self.addTab(self.tabVar, "Variance")

    def update_displayed_params(self):
        """Display parameters according to current_paramset of main."""
        super().update_displayed_params()
        paramset = self.main.current_paramset

        self.hom_roi_size.setValue(paramset.hom_roi_size)
        self.hom_roi_rotation.setValue(paramset.hom_roi_rotation)
        self.hom_roi_distance.setValue(paramset.hom_roi_distance)
        self.hom_tab_alt.button(paramset.hom_tab_alt).setChecked(True)
        self.noi_percent.setValue(paramset.noi_percent)
        self.mtf_type.setCurrentIndex(paramset.mtf_type)
        self.mtf_roi_size_x.setValue(paramset.mtf_roi_size_x)
        self.mtf_roi_size_y.setValue(paramset.mtf_roi_size_y)
        self.mtf_plot.setCurrentIndex(paramset.mtf_plot)
        self.mtf_gaussian.setChecked(paramset.mtf_gaussian)
        self.mtf_offset_mm.setChecked(paramset.mtf_offset_mm)
        self.mtf_cut_lsf.setChecked(paramset.mtf_cut_lsf)
        self.mtf_cut_lsf_w.setValue(paramset.mtf_cut_lsf_w)
        self.mtf_offset_xy.setText(
            f'{paramset.mtf_offset_xy[0]}, {paramset.mtf_offset_xy[1]}')
        self.mtf_auto_center.setChecked(paramset.mtf_auto_center)
        self.mtf_sampling_frequency.setValue(paramset.mtf_sampling_frequency)
        self.nps_roi_size.setValue(paramset.nps_roi_size)
        self.nps_sub_size.setValue(paramset.nps_sub_size)
        npix_tot = ((self.nps_roi_size.value() * 2 - 1)
                    * self.nps_sub_size.value()) ** 2
        self.nps_npix.setText(f'{npix_tot}')
        self.var_roi_size.setValue(paramset.var_roi_size)

        self.update_enabled()
        self.flag_ignore_signals = False

    def create_tab_Hom(self):
        """GUI of tab Homogeneity."""
        self.tabHom = QWidget()
        vLO = QVBoxLayout()
        hLO = QHBoxLayout()
        f = QFormLayout()

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

        f.addRow(QLabel('ROI radius (mm)'), self.hom_roi_size)
        f.addRow(QLabel('Rotate ROI positions (deg)'), self.hom_roi_rotation)
        f.addRow(QLabel('ROI distance (% from center)'), self.hom_roi_distance)
        hLO.addLayout(f)

        alt_txt = [
            'Avg and stdev for each ROI',
            'Avg for each ROI + difference from average of all ROIs',
            'Avg for each ROI + % difference from average of all ROIs'
            ]

        self.gb_alternative = QGroupBox('Output to table')
        self.gb_alternative.setFont(uir.FontItalic())
        self.hom_tab_alt = QButtonGroup()
        lo = QVBoxLayout()
        for a, txt in enumerate(alt_txt):
            rb = QRadioButton(txt)
            self.hom_tab_alt.addButton(rb, a)
            lo.addWidget(rb)
            rb.clicked.connect(self.hom_tab_alt_changed)
        self.gb_alternative.setLayout(lo)
        hLO.addWidget(self.gb_alternative)

        vLO.addLayout(hLO)
        vLO.addWidget(uir.LabelItalic(
            'Same distance for all quadrants = % of shortest'
            'center-border distance.'))
        vLO.addWidget(uir.LabelItalic(
            'Leave distance empty to set ROIs at center of each qadrant.'))

        self.btnRunHom = QPushButton('Calculate homogeneity')
        self.btnRunHom.setToolTip('Run test')
        self.btnRunHom.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunHom)
        vLO.addStretch()

        self.tabHom.setLayout(vLO)

    def create_tab_Noi(self):
        """GUI of tab Noise."""
        self.tabNoi = QWidget()
        vLO = QVBoxLayout()
        hLO = QHBoxLayout()
        f = QFormLayout()

        self.noi_percent = QDoubleSpinBox(decimals=1,
                                          minimum=0.1, maximum=100., singleStep=0.1)
        self.noi_percent.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='noi_percent'))

        f.addRow(QLabel('ROI width and height (% of image)'), self.noi_percent)
        hLO.addLayout(f)
        hLO.addStretch()
        vLO.addLayout(hLO)

        self.btnRunNoi = QPushButton('Calculate noise')
        self.btnRunNoi.setToolTip('Run test')
        self.btnRunNoi.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunNoi)
        vLO.addStretch()

        self.tabNoi.setLayout(vLO)

    def create_tab_MTF(self):
        """GUI of tab MTF."""
        self.tabMTF = QWidget()

        self.mtf_type = QComboBox()
        self.mtf_type.addItems(['Exponential fit', 'Gaussian fit', 'No LSF fit'])
        self.mtf_type.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_type'))

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

        self.mtf_sampling_frequency = QDoubleSpinBox(
            decimals=3, minimum=0.001, singleStep=0.001)
        self.mtf_sampling_frequency.valueChanged.connect(
                    lambda: self.param_changed_from_gui(
                        attribute='mtf_sampling_frequency'))

        self.mtf_cut_lsf = QCheckBox('')
        self.mtf_cut_lsf.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_cut_lsf'))
        self.mtf_cut_lsf_w = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_cut_lsf_w.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_cut_lsf_w'))

        self.mtf_plot = QComboBox()
        self.mtf_plot.addItems(['Edge position',
                                'Sorted pixel values', 'LSF', 'MTF'])
        self.mtf_plot.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_plot',
                                                update_roi=False,
                                                clear_results=False))

        self.mtf_gaussian = BoolSelectTests(
            self, attribute='mtf_gaussian',
            text_true='Analytical', text_false='Discrete',
            update_roi=False, clear_results=False, update_plot=False)

        self.mtf_offset_xy = QLabel('0, 0')
        self.mtf_offset_mm = BoolSelectTests(
            self, attribute='mtf_offset_mm',
            text_true='mm', text_false='pix')
        tb_mtf_offset = QToolBar()
        self.btn_mtf_offset = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}selectArrow.png'),
            '''Left mouse click in image to set offset position, then fetch
            the position by clicking this button''', self)
        self.btn_mtf_offset_reset = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}reset.png'),
            '''Reset offset''', self)
        tb_mtf_offset.addActions(
            [self.btn_mtf_offset, self.btn_mtf_offset_reset])
        self.btn_mtf_offset.triggered.connect(
            lambda: self.set_offset(attribute='mtf_offset_xy'))
        self.btn_mtf_offset_reset.triggered.connect(
            lambda: self.set_offset(attribute='mtf_offset_xy', reset=True))
        tb_mtf_offset.addWidget(self.mtf_offset_xy)
        tb_mtf_offset.addWidget(self.mtf_offset_mm)

        vLO = QVBoxLayout()
        hLO = QHBoxLayout()
        hLO_size = QHBoxLayout()
        hLO_size.addWidget(QLabel('ROI width/height (mm)'))
        hLO_size.addWidget(self.mtf_roi_size_x)
        hLO_size.addWidget(QLabel('/'))
        hLO_size.addWidget(self.mtf_roi_size_y)

        vLO1 = QVBoxLayout()
        vLO1.addLayout(hLO_size)
        f1 = QFormLayout()
        f1.addRow(QLabel('MTF method'), self.mtf_type)
        f1.addRow(QLabel('Sampling freq. gaussian (mm-1)'),
                  self.mtf_sampling_frequency)
        vLO1.addLayout(f1)
        hLO_offset = QHBoxLayout()
        hLO_offset.addWidget(QLabel('Set extra offset'))
        hLO_offset.addWidget(tb_mtf_offset)
        vLO1.addLayout(hLO_offset)
        hLO.addLayout(vLO1)

        hLO.addWidget(uir.VLine())
        vLO2 = QVBoxLayout()
        vLO2.addWidget(self.mtf_auto_center)
        hLO_gb_auto = QHBoxLayout()
        hLO_gb_auto.addWidget(QLabel('Use'))
        hLO_gb_auto.addWidget(self.mtf_auto_center_type)
        self.mtf_auto_center.setLayout(hLO_gb_auto)
        f2 = QFormLayout()
        f2.addRow(QLabel('Cut LSF tails'), self.mtf_cut_lsf)
        f2.addRow(QLabel('    Cut at halfmax + (#FWHM)'), self.mtf_cut_lsf_w)
        vLO2.addLayout(f2)
        f3 = QFormLayout()
        f3.addRow(QLabel('Table results from'), self.mtf_gaussian)
        f3.addRow(QLabel('Plot'), self.mtf_plot)
        vLO2.addLayout(f3)
        hLO.addLayout(vLO2)

        vLO.addLayout(hLO)

        self.btnRunMTF = QPushButton('Calculate MTF')
        self.btnRunMTF.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunMTF)

        self.tabMTF.setLayout(vLO)

    def create_tab_NPS(self):
        """GUI of tab NPS."""
        self.tabNPS = QWidget()
        vLO = QVBoxLayout()
        hLO = QHBoxLayout()
        f = QFormLayout()

        self.nps_roi_size = QDoubleSpinBox(decimals=0, minimum=22)
        self.nps_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_roi_size'))
        self.nps_sub_size = QDoubleSpinBox(decimals=0, minimum=1)
        self.nps_sub_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_sub_size'))

        f.addRow(QLabel('ROI size (pix)'), self.nps_roi_size)
        f.addRow(QLabel('Total area to analyse (n x ROI size) n = '),
                 self.nps_sub_size)
        hLO.addLayout(f)
        hLO.addStretch()

        vLO.addLayout(hLO)
        hLO_npix = QHBoxLayout()
        hLO_npix.addWidget(uir.LabelItalic(
            '# of independent pixels/image (preferrably > 4 mill):'))
        self.nps_npix = QLabel('')
        hLO_npix.addWidget(self.nps_npix)
        vLO.addLayout(hLO_npix)

        self.btnRunNPS = QPushButton('Calculate NPS')
        self.btnRunNPS.setToolTip('Run test')
        self.btnRunNPS.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunNPS)
        vLO.addStretch()

        self.tabNPS.setLayout(vLO)

    def create_tab_Var(self):
        """GUI of tab Variance."""
        self.tabVar = QWidget()
        vLO = QVBoxLayout()

        self.var_roi_size = QDoubleSpinBox(decimals=1, minimum=1., singleStep=0.1)
        self.var_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='var_roi_size'))

        vLO.addWidget(uir.LabelItalic(
            'The variance image can reveal artifacts in the image.<br>'
            'Adjust ROI size to find artifacts of different sizes.'))

        hLO = QHBoxLayout()
        hLO.addWidget(QLabel('ROI size (mm)'))
        hLO.addWidget(self.var_roi_size)
        hLO.addWidget(QLabel('if less than 3 pix, 3 pix will be used'))
        hLO.addStretch()
        vLO.addLayout(hLO)

        vLO.addStretch()
        self.btnRunVar = QPushButton('Calculate variance image(s)')
        self.btnRunVar.setToolTip('Run test')
        self.btnRunVar.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunVar)

        vLO.addStretch()
        self.tabVar.setLayout(vLO)

    def hom_tab_alt_changed(self):
        """Change alternative method (columns to display)."""
        self.param_changed_from_gui(
            attribute='hom_tab_alt', content=self.hom_tab_alt.checkedId())


class TestTabNM(TestTabCommon):
    """Copy for modality tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_Uni()
        self.create_tab_SNI()
        self.create_tab_MTF()
        self.create_tab_Spe()
        self.create_tab_Bar()

        self.addTab(self.tabUni, "Uniformity")
        self.addTab(self.tabSNI, "SNI")
        self.addTab(self.tabMTF, "MTF")
        self.addTab(self.tabSpe, "Scan speed")
        self.addTab(self.tabBar, "Bar phantom")

    def update_displayed_params(self):
        """Display parameters according to current_paramset of main."""
        super().update_displayed_params()
        paramset = self.main.current_paramset

        self.uni_ufov_ratio.setValue(paramset.uni_ufov_ratio)
        self.uni_cfov_ratio.setValue(paramset.uni_cfov_ratio)
        self.uni_correct.setChecked(paramset.uni_correct)
        self.uni_correct_pos_x.setChecked(paramset.uni_correct_pos_x)
        self.uni_correct_pos_y.setChecked(paramset.uni_correct_pos_y)
        self.uni_correct_radius_chk.setChecked(paramset.uni_correct_radius != -1.)
        if paramset.uni_correct_radius != -1:
            self.uni_correct_radius.setValue(paramset.uni_correct_radius)
        self.uni_sum_first.setChecked(paramset.uni_sum_first)

        self.sni_area_ratio.setValue(paramset.sni_area_ratio)
        self.sni_correct.setChecked(paramset.sni_correct)
        self.sni_correct_pos_x.setChecked(paramset.sni_correct_pos_x)
        self.sni_correct_pos_y.setChecked(paramset.sni_correct_pos_y)
        self.sni_correct_radius_chk.setChecked(paramset.sni_correct_radius != -1.)
        if paramset.sni_correct_radius != -1:
            self.uni_correct_radius.setValue(paramset.sni_correct_radius)
        self.sni_sum_first.setChecked(paramset.sni_sum_first)
        self.sni_eye_filter_f.setValue(paramset.sni_eye_filter_f)
        self.sni_eye_filter_c.setValue(paramset.sni_eye_filter_c)
        self.sni_eye_filter_r.setValue(paramset.sni_eye_filter_r)
        '''
        self.mtf_type.setCurrentIndex(paramset.mtf_type)
        self.mtf_gaussian.setChecked(paramset.mtf_gaussian)
        '''
        # all parameters....
        '''

        mtf_type: int = 1
        mtf_roi_size: list[float] = field(default_factory=lambda: [20., 20.])
        mtf_plot: int = 4
        bar_roi_size: float = 50.
        bar_widths: list[float] = field(
            default_factory=lambda: [6.4, 4.8, 4.0, 3.2])
        spe_avg: int = 25
        spe_height: float = 100.
        spe_filter_w: int = 15
        '''
        self.update_enabled()
        self.flag_ignore_signals = False

    def create_tab_Uni(self):
        """GUI of tab Uniformity."""
        self.tabUni = QWidget()

        self.uni_ufov_ratio = QDoubleSpinBox(
            decimals=2, minimum=0.1, maximum=1., singleStep=0.01)
        self.uni_ufov_ratio.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='uni_ufov_ratio'))

        self.uni_cfov_ratio = QDoubleSpinBox(
            decimals=2, minimum=0.1, maximum=1., singleStep=0.01)
        self.uni_cfov_ratio.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='uni_cfov_ratio'))

        self.uni_correct = QGroupBox('Correct for point source curvature')
        self.uni_correct.setCheckable(True)
        self.uni_correct.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='uni_correct'))

        self.uni_correct_pos_x = QCheckBox('x')
        self.uni_correct_pos_x.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='uni_correct_pos_x'))

        self.uni_correct_pos_y = QCheckBox('y')
        self.uni_correct_pos_y.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='uni_correct_pos_y'))

        self.uni_correct_radius_chk = QCheckBox('Lock source distance to')
        self.uni_correct_radius = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=5000, singleStep=0.1)
        self.uni_correct_radius.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='uni_correct_radius'))

        self.uni_sum_first = QCheckBox('Sum marked images and analysing sum')
        self.uni_sum_first.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='uni_sum_first'))

        self.uni_plot = QComboBox()
        self.uni_plot.addItems(['Uniformity result for all images',
                                'Curvature correction check'])
        self.uni_plot.currentIndexChanged.connect(
            self.main.wResPlot.plotcanvas.plot)

        self.uni_result_image = QComboBox()
        self.uni_result_image.addItems(
            [
                'Differential uniformity map',
                'Processed image (6.4mm pix, smoothed, corrected)',
                'Curvature corrected image'
             ])
        self.uni_result_image.currentIndexChanged.connect(
            self.main.wResImage.canvas.result_image_draw)

        vLO = QVBoxLayout()
        vLO.addWidget(QLabel('Based on NEMA NU-1 2007'))
        hLO = QHBoxLayout()
        vLO.addLayout(hLO)

        vLO_left = QVBoxLayout()
        hLO.addLayout(vLO_left)
        hLO.addWidget(uir.VLine())
        vLO_right = QVBoxLayout()
        hLO.addLayout(vLO_right)

        hLO_fov = QHBoxLayout()
        f = QFormLayout()
        f.addRow(QLabel('UFOV ratio'), self.uni_ufov_ratio)
        f.addRow(QLabel('CFOV ratio'), self.uni_cfov_ratio)
        hLO_fov.addLayout(f)
        hLO_fov.addSpacing(100)
        vLO_left.addLayout(hLO_fov)
        vLO_left.addWidget(self.uni_sum_first)

        vLO_right.addWidget(self.uni_correct)
        hLO_xy = QHBoxLayout()
        hLO_xy.addWidget(QLabel('Fit point source position in'))
        hLO_xy.addWidget(self.uni_correct_pos_x)
        hLO_xy.addWidget(self.uni_correct_pos_y)
        vLO_gb = QVBoxLayout()
        self.uni_correct.setLayout(vLO_gb)
        vLO_gb.addLayout(hLO_xy)
        hLO_lock_dist = QHBoxLayout()
        vLO_gb.addLayout(hLO_lock_dist)
        hLO_lock_dist.addWidget(self.uni_correct_radius_chk)
        hLO_lock_dist.addWidget(self.uni_correct_radius)
        hLO_lock_dist.addWidget(QLabel('mm'))

        hLO_btm = QHBoxLayout()
        hLO_btm.addStretch()
        f_btm = QFormLayout()
        hLO_btm.addLayout(f_btm)
        f_btm.addRow(QLabel('Plot'), self.uni_plot)
        f_btm.addRow(QLabel('Result image'), self.uni_result_image)
        vLO.addLayout(hLO_btm)

        self.btnRunUni = QPushButton('Calculate uniformity')
        self.btnRunUni.setToolTip('Run test')
        self.btnRunUni.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunUni)
        vLO.addStretch()

        self.tabUni.setLayout(vLO)

    def create_tab_SNI(self):
        """GUI of tab SNI."""
        self.tabSNI = QWidget()

        self.sni_area_ratio = QDoubleSpinBox(
            decimals=2, minimum=0.1, maximum=1., singleStep=0.01)
        self.sni_area_ratio.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sni_area_ratio'))

        self.sni_correct = QGroupBox('Correct for point source curvature')
        self.sni_correct.setCheckable(True)
        self.sni_correct.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='sni_correct'))
        self.sni_correct_pos_x = QCheckBox('x')
        self.sni_correct_pos_x.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='sni_correct_pos_x'))
        self.sni_correct_pos_y = QCheckBox('y')
        self.sni_correct_pos_y.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='sni_correct_pos_y'))
        self.sni_correct_radius_chk = QCheckBox('Lock source distance to')
        self.sni_correct_radius = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=5000, singleStep=0.1)
        self.sni_correct_radius.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sni_correct_radius'))

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

        self.sni_sum_first = QCheckBox('Sum marked images and analysing sum')
        self.sni_sum_first.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='sni_sum_first'))

        self.sni_plot = QComboBox()
        self.sni_plot.addItems(['SNI values',
                                'Power spectrums used to calculate SNI'])
        self.sni_plot.currentIndexChanged.connect(
            self.main.wResPlot.plotcanvas.plot)

        self.sni_result_image = QComboBox()
        self.sni_result_image.addItems(
            ['2d NPS', 'Curvature corrected image'])
        self.sni_result_image.currentIndexChanged.connect(
            self.main.wResPlot.plotcanvas.plot)

        vLO = QVBoxLayout()
        vLO.addWidget(QLabel(
            'SNI = Structured Noise Index (J Nucl Med 2014; 55:169-174)'))
        hLO = QHBoxLayout()
        vLO.addLayout(hLO)

        vLO_left = QVBoxLayout()
        hLO.addLayout(vLO_left)
        hLO.addWidget(uir.VLine())
        vLO_right = QVBoxLayout()
        hLO.addLayout(vLO_right)

        f = QFormLayout()
        f.addRow(QLabel('Ratio of nonzero part of image to be analysed'),
                 self.sni_area_ratio)
        vLO_left.addLayout(f)

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
        vLO_left.addWidget(gb_eye_filter)
        vLO_left.addWidget(self.sni_sum_first)

        vLO_right.addWidget(self.sni_correct)
        hLO_xy = QHBoxLayout()
        hLO_xy.addWidget(QLabel('Fit point source position in'))
        hLO_xy.addWidget(self.sni_correct_pos_x)
        hLO_xy.addWidget(self.sni_correct_pos_y)
        vLO_gb = QVBoxLayout()
        self.sni_correct.setLayout(vLO_gb)
        vLO_gb.addLayout(hLO_xy)
        hLO_lock_dist = QHBoxLayout()
        vLO_gb.addLayout(hLO_lock_dist)
        hLO_lock_dist.addWidget(self.sni_correct_radius_chk)
        hLO_lock_dist.addWidget(self.sni_correct_radius)
        hLO_lock_dist.addWidget(QLabel('mm'))

        f_btm = QFormLayout()
        vLO_right.addLayout(f_btm)
        f_btm.addRow(QLabel('Plot'), self.sni_plot)
        f_btm.addRow(QLabel('Result image'), self.sni_result_image)

        self.btnRunSni = QPushButton('Calculate SNI')
        self.btnRunSni.setToolTip('Run test')
        self.btnRunSni.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunSni)
        vLO.addStretch()

        self.tabSNI.setLayout(vLO)

    def create_tab_MTF(self):
        """GUI of tab MTF."""
        self.tabMTF = QWidget()
        vLO = QVBoxLayout()

        #...

        self.tabMTF.setLayout(vLO)

    def create_tab_Spe(self):
        """GUI of tab Scan speed."""
        self.tabSpe = QWidget()
        vLO = QVBoxLayout()

        #...

        self.tabSpe.setLayout(vLO)

    def create_tab_Bar(self):
        """GUI of tab Bar Phantom."""
        self.tabBar = QWidget()
        vLO = QVBoxLayout()

        #...

        self.tabBar.setLayout(vLO)


class TestTabSPECT(TestTabCommon):
    """Copy for modality tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_MTF()
        self.create_tab_Con()

        self.addTab(self.tabMTF, "MTF")
        self.addTab(self.tabCon, "Contrast")

    def update_displayed_params(self):
        """Display parameters according to current_paramset of main."""
        super().update_displayed_params()
        paramset = self.main.current_paramset
        '''
        self.hom_roi_size.setValue(paramset.hom_roi_size)
        self.mtf_type.setCurrentIndex(paramset.mtf_type)
        self.mtf_gaussian.setChecked(paramset.mtf_gaussian)
        '''
        # all parameters....
        '''
        mtf_type: int = 1
        mtf_roi_size: float = 30.
        mtf_plot: int = 4
        mtf_3d: bool = True
        con_roi_size: float = 20.
        con_roi_dist: float = 58.
        '''
        self.update_enabled()
        self.flag_ignore_signals = False

    def create_tab_MTF(self):
        """GUI of tab MTF."""
        self.tabMTF = QWidget()
        vLO = QVBoxLayout()

        #...

        self.tabMTF.setLayout(vLO)

    def create_tab_Con(self):
        """GUI of tab Contrast."""
        self.tabCon = QWidget()
        vLO = QVBoxLayout()

        #...

        self.tabCon.setLayout(vLO)


class TestTabPET(TestTabCommon):
    """Tab for PET tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_Hom()
        self.create_tab_Cro()

        self.addTab(self.tabHom, "Homogeneity")
        self.addTab(self.tabCro, "Cross Calibration")

    def update_displayed_params(self):
        """Display parameters according to current_paramset of main."""
        super().update_displayed_params()
        paramset = self.main.current_paramset
        self.hom_roi_size.setValue(paramset.hom_roi_size)
        self.hom_roi_distance.setValue(paramset.hom_roi_distance)
        '''
        self.mtf_type.setCurrentIndex(paramset.mtf_type)
        self.mtf_gaussian.setChecked(paramset.mtf_gaussian)
        '''
        # all parameters....
        '''
        hom_roi_size: float = 10.
        hom_roi_distance: float = 55.
        cro_roi_size: float = 60.
        cro_volume: float = 0.
        '''
        self.update_enabled()
        self.flag_ignore_signals = False

    def create_tab_Hom(self):
        """GUI of tab Homogeneity."""
        self.tabHom = QWidget()
        vLO = QVBoxLayout()
        hLO = QHBoxLayout()
        f = QFormLayout()

        vLO.addWidget(QLabel(
            'Calculate mean in ROIs and % difference from mean of all mean values.'))

        self.hom_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.hom_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_size'))
        self.hom_roi_distance = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.hom_roi_distance.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_distance'))

        f.addRow(QLabel('ROI radius (mm)'), self.hom_roi_size)
        f.addRow(QLabel('ROI distance (mm)'), self.hom_roi_distance)
        hLO.addLayout(f)
        hLO.addStretch()

        vLO.addStretch()
        vLO.addLayout(hLO)

        self.btnRunHom = QPushButton('Calculate homogeneity')
        self.btnRunHom.setToolTip('Run test')
        self.btnRunHom.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunHom)
        vLO.addStretch()

        self.tabHom.setLayout(vLO)

    def create_tab_Cro(self):
        """GUI of tab Cross Calibration."""
        self.tabCro = QWidget()
        vLO = QVBoxLayout()

        #...

        self.tabCro.setLayout(vLO)


class TestTabMR(TestTabCommon):
    """Tab for MR tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_SNR()
        self.create_tab_PIU()
        self.create_tab_Gho()
        self.create_tab_Geo()
        self.create_tab_Sli()
        self.create_tab_ROI()

        self.addTab(self.tabSNR, "SNR")
        self.addTab(self.tabPIU, "PIU")
        self.addTab(self.tabGho, "Ghosting")
        self.addTab(self.tabGeo, "Geometric Distortion")
        self.addTab(self.tabSli, "Slice thickness")

    def update_displayed_params(self):
        """Display parameters according to current_paramset of main."""
        super().update_displayed_params()
        paramset = self.main.current_paramset

        self.snr_roi_percent.setValue(paramset.snr_roi_percent)
        self.snr_roi_cut_top.setValue(paramset.snr_roi_cut_top)
        self.piu_roi_percent.setValue(paramset.piu_roi_percent)
        self.piu_roi_cut_top.setValue(paramset.piu_roi_cut_top)
        self.gho_roi_central.setValue(paramset.gho_roi_central)
        self.gho_roi_w.setValue(paramset.gho_roi_w)
        self.gho_roi_h.setValue(paramset.gho_roi_h)
        self.gho_roi_dist.setValue(paramset.gho_roi_dist)
        self.gho_roi_cut_top.setValue(paramset.gho_roi_cut_top)
        self.gho_optimize_center.setChecked(paramset.gho_optimize_center)
        self.geo_actual_size.setValue(paramset.geo_actual_size)
        self.sli_tan_a.setValue(paramset.sli_tan_a)
        self.sli_roi_w.setValue(paramset.sli_roi_w)
        self.sli_roi_h.setValue(paramset.sli_roi_h)
        self.sli_dist_lower.setValue(paramset.sli_dist_lower)
        self.sli_dist_upper.setValue(paramset.sli_dist_upper)
        self.sli_optimize_center.setChecked(paramset.sli_optimize_center)

        self.update_enabled()
        self.flag_ignore_signals = False

    def create_tab_SNR(self):
        """GUI of tab SNR."""
        self.tabSNR = QWidget()
        vLO = QVBoxLayout()
        hLO_top = QHBoxLayout()
        hLO_top.addWidget(uir.LabelItalic('Signal to noise ratio (SNR)'))
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
        hLO_top.addWidget(uir.InfoTool(info_txt, parent=self.main))
        vLO.addLayout(hLO_top)

        hLO = QHBoxLayout()
        f = QFormLayout()

        self.snr_roi_percent = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=100., singleStep=0.1)
        self.snr_roi_percent.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='snr_roi_percent'))

        self.snr_roi_cut_top = QDoubleSpinBox(
            decimals=0, minimum=0.)
        self.snr_roi_cut_top.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='snr_roi_cut_top'))

        f.addRow(QLabel('ROI % of circular phantom'), self.snr_roi_percent)
        f.addRow(QLabel('Cut top of ROI by (mm)'), self.snr_roi_cut_top)
        hLO.addLayout(f)
        hLO.addStretch()

        vLO.addLayout(hLO)
        vLO.addStretch()

        self.btnRunSNR = QPushButton('Calculate SNR')
        self.btnRunSNR.setToolTip('Run test')
        self.btnRunSNR.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunSNR)
        vLO.addStretch()

        self.tabSNR.setLayout(vLO)

    def create_tab_PIU(self):
        """GUI of tab PIU."""
        self.tabPIU = QWidget()
        vLO = QVBoxLayout()
        hLO_top = QHBoxLayout()
        hLO_top.addWidget(uir.LabelItalic('Percent Integral Uniformity (PIU)'))
        info_txt = '''
        Based on NEMA MS-3 2008
        <br><br>
        Center and size of phantom will be found from maximum x and y profiles
        to generate the ROI.
        '''
        hLO_top.addWidget(uir.InfoTool(info_txt, parent=self.main))
        vLO.addLayout(hLO_top)

        hLO = QHBoxLayout()
        f = QFormLayout()

        self.piu_roi_percent = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=100., singleStep=0.1)
        self.piu_roi_percent.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='piu_roi_percent'))

        self.piu_roi_cut_top = QDoubleSpinBox(
            decimals=0, minimum=0.)
        self.piu_roi_cut_top.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='piu_roi_cut_top'))

        f.addRow(QLabel('ROI % of circular phantom'), self.piu_roi_percent)
        f.addRow(QLabel('Cut top of ROI by (mm)'), self.piu_roi_cut_top)
        hLO.addLayout(f)
        hLO.addStretch()

        vLO.addLayout(hLO)
        vLO.addStretch()

        self.btnRunPIU = QPushButton('Calculate PIU')
        self.btnRunPIU.setToolTip('Run test')
        self.btnRunPIU.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunPIU)
        vLO.addStretch()

        self.tabPIU.setLayout(vLO)

    def create_tab_Gho(self):
        """GUI of tab Ghosting."""
        self.tabGho = QWidget()
        vLO = QVBoxLayout()
        hLO_top = QHBoxLayout()
        hLO_top.addWidget(uir.LabelItalic('Percent Signal Ghosting'))
        info_txt = '''
        Based on ACR MR Quality Control Manual, 2015<br>
        Percent Signal Ghosting (PSG) = 100 * |(left + right)-(top+bottom)| /
        2*center<br><br>
        If optimized, center and size of phantom will be found from maximum
        x and y profiles.
        '''
        hLO_top.addWidget(uir.InfoTool(info_txt, parent=self.main))
        vLO.addLayout(hLO_top)

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

        hLO1 = QHBoxLayout()
        f1 = QFormLayout()
        f1.addRow(QLabel('Radius of central ROI (mm)'), self.gho_roi_central)
        hLO_size = QHBoxLayout()
        hLO_size.addWidget(QLabel('Width / height of outer ROI (mm)'))
        hLO_size.addWidget(self.gho_roi_w)
        hLO_size.addWidget(QLabel('/'))
        hLO_size.addWidget(self.gho_roi_h)
        hLO_size.addStretch()
        hLO2 = QHBoxLayout()
        f2 = QFormLayout()
        f2.addRow(QLabel('Distance from image border (mm)'), self.gho_roi_dist)
        f2.addRow(QLabel('Cut top of ROI by (mm)'), self.gho_roi_cut_top)
        f2.addRow(QLabel('Optimize center'), self.gho_optimize_center)
        hLO1.addLayout(f1)
        hLO1.addStretch()
        hLO2.addLayout(f2)
        hLO2.addStretch()
        vLO.addLayout(hLO1)
        vLO.addLayout(hLO_size)
        vLO.addLayout(hLO2)

        vLO.addStretch()

        self.btnRunGho = QPushButton('Calculate Ghosting')
        self.btnRunGho.setToolTip('Run test')
        self.btnRunGho.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunGho)
        vLO.addStretch()

        self.tabGho.setLayout(vLO)

    def create_tab_Geo(self):
        """GUI of tab Geometric Distortion."""
        self.tabGeo = QWidget()
        vLO = QVBoxLayout()
        hLO_top = QHBoxLayout()
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
        hLO_top.addWidget(uir.InfoTool(info_txt, parent=self.main))
        vLO.addLayout(hLO_top)

        hLO = QHBoxLayout()
        f = QFormLayout()

        self.geo_actual_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, singleStep=0.1)
        self.geo_actual_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='geo_actual_size'))

        f.addRow(QLabel('Actual width of phantom (mm)'), self.geo_actual_size)
        hLO.addLayout(f)
        hLO.addStretch()

        vLO.addLayout(hLO)
        vLO.addStretch()

        self.btnRunGeo = QPushButton('Calculate Geometric Distortion')
        self.btnRunGeo.setToolTip('Run test')
        self.btnRunGeo.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunGeo)
        vLO.addStretch()

        self.tabGeo.setLayout(vLO)

    def create_tab_Sli(self):
        """GUI of tab Slice thickness."""
        self.tabSli = QWidget()
        vLO = QVBoxLayout()
        info_txt = '''
        Based on NEMA MS-5 2018 and ACR MR Quality Control Manual, 2015<br><br>
        Slice thickness = tan (wedge angle) * FWHM<br><br>
        Slice thickness ACR = 1/10 * harmonic mean of FWHM upper and lower =
        0.2 * upper * lower / (upper + lower)<br><br>
        FWHM will be calculated for the averaged profile within each ROI,
        max from medianfiltered profile.<br><br>
        If optimized, center of phantom will be found from maximum profiles.
        '''
        hLO_top = QHBoxLayout()
        hLO_top.addWidget(uir.InfoTool(info_txt, parent=self.main))
        vLO.addLayout(hLO_top)

        self.sli_tan_a = QDoubleSpinBox(decimals=3, minimum=0.)
        self.sli_tan_a.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_tan_a'))

        self.sli_roi_w = QDoubleSpinBox(
            decimals=1, minimum=0., maximum=200., singleStep=0.1)
        self.sli_roi_w.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_roi_w'))
        self.sli_roi_h = QDoubleSpinBox(decimals=1, minimum=0., singleStep=0.1)
        self.sli_roi_h.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_roi_h'))

        self.sli_dist_lower = QDoubleSpinBox(decimals=1, minimum=0., singleStep=0.1)
        self.sli_dist_lower.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_dist_lower'))
        self.sli_dist_upper = QDoubleSpinBox(decimals=1, minimum=0., singleStep=0.1)
        self.sli_dist_upper.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_dist_upper'))

        self.sli_optimize_center = QCheckBox('')
        self.sli_optimize_center.toggled.connect(
            lambda: self.param_changed_from_gui(
                attribute='sli_optimize_center'))

        hLO_size = QHBoxLayout()
        hLO_size.addWidget(QLabel('ROI width / height (mm)'))
        hLO_size.addWidget(self.sli_roi_w)
        hLO_size.addWidget(QLabel('/'))
        hLO_size.addWidget(self.sli_roi_h)
        hLO_size.addStretch()
        vLO.addLayout(hLO_size)
        hLO_dist = QHBoxLayout()
        hLO_dist.addWidget(QLabel(
            'ROI distance from image center upper / lower (mm)'))
        hLO_dist.addWidget(self.sli_dist_upper)
        hLO_dist.addWidget(QLabel('/'))
        hLO_dist.addWidget(self.sli_dist_lower)
        hLO_dist.addStretch()
        vLO.addLayout(hLO_dist)

        hLO = QHBoxLayout()
        f1 = QFormLayout()
        f1.addRow(QLabel('tangens of wedge angle'), self.sli_tan_a)
        f1.addRow(QLabel('Optimize center'), self.sli_optimize_center)
        hLO.addLayout(f1)
        hLO.addStretch()
        vLO.addLayout(hLO)
        vLO.addStretch()

        self.btnRunSli = QPushButton('Calculate Slice thickness')
        self.btnRunSli.setToolTip('Run test')
        self.btnRunSli.clicked.connect(self.run_current)
        vLO.addWidget(self.btnRunSli)
        vLO.addStretch()

        self.tabSli.setLayout(vLO)


class TestTabVendor(QWidget):
    """Test tabs for vendor file analysis."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent

        self.selected = 0

        lo = QVBoxLayout()
        infotxt = '''
        imageQC can read and extract parameters from a set of
        vendor specific files (QC reports or exported data).<br>
        Select one of these file-types in the list and get the
        results tabulated.
        '''
        lo.addWidget(uir.LabelItalic(infotxt))
        lo.addWidget(uir.HLine())
        self.table_file_types = QTreeWidget()
        self.table_file_types.setColumnCount(1)
        self.table_file_types.setHeaderLabels(['Expected file type'])
        self.table_file_types.setMinimumWidth(300)
        self.table_file_types.setRootIsDecorated(False)
        self.table_file_types.currentItemChanged.connect(self.update_selected)
        lo.addWidget(self.table_file_types)
        self.btnOpen = QPushButton('Open and read vendor specific file')
        self.btnOpen.setIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}open.png'))
        self.btnOpen.clicked.connect(self.open_vendor_files)
        lo.addWidget(self.btnOpen)
        lo.addStretch()
        self.setLayout(lo)

        self.update_table()

    def update_displayed_params(self):
        pass #

    def update_selected(self, current, prev):
        """Update self.selected when selected in table.

        Parameters
        ----------
        current : QTreeWidgetItem
            Current item
        prev : QTreeWidgetItem
            Previous item
        """
        self.selected = self.table_file_types.indexOfTopLevelItem(current)

    def update_table(self, set_selected=0):
        """Update table based on mode selected.

        Parameters
        ----------
        set_selected : int, optional
            Set initially selected item. The default is 0.
        """
        self.table_file_types.clear()
        self.options = VENDOR_FILE_OPTIONS[self.main.current_modality]

        for option in self.options:
            item = QTreeWidgetItem([option])
            self.table_file_types.addTopLevelItem(item)

        self.selected = set_selected
        self.table_file_types.setCurrentItem(
            self.table_file_types.topLevelItem(self.selected))

    def open_vendor_files(self):
        """Open vendor files for analysis."""
        if len(VENDOR_FILE_OPTIONS[self.main.current_modality]) > 0:
            filetype = VENDOR_FILE_OPTIONS[
                self.main.current_modality][self.selected]
            res = {'status': False}

            if filetype == 'Siemens CT Constancy/Daily Reports (.pdf)':
                fname = QFileDialog.getOpenFileName(
                        self, 'Open Siemens CT QC report files',
                        filter="PDF files (*.pdf)")
                if len(fname[0]) > 0:
                    self.main.statusBar.showMessage(
                        'Reading pdf file... takes a few seconds')
                    self.main.start_wait_cursor()
                    #QApplication.setOverrideCursor(Qt.WaitCursor)
                    #qApp.processEvents()
                    txt = rvr.get_pdf_txt(fname[0])
                    self.main.statusBar.showMessage(
                        'Extracting data from pdf')
                    res = rvr.read_Siemens_CT_QC(
                        txt)
                    self.main.stop_wait_cursor()
                    #QApplication.restoreOverrideCursor()
                    self.main.statusBar.showMessage('Finished', 1000)

            elif filetype == 'GE QAP (.txt)':
                #TODO
                pass
            elif filetype == 'Siemens exported energy spectrum':
                #TODO
                pass
            elif filetype == 'Siemens PET-CT DailyQC Reports (.pdf)':
                fname = QFileDialog.getOpenFileName(
                        self, 'Open Siemens PET-CT QC report files',
                        filter="PDF files (*.pdf)")
                if len(fname[0]) > 0:
                    self.main.statusBar.showMessage(
                        'Reading pdf file... takes a few seconds')
                    self.main.start_wait_cursor()
                    #QApplication.setOverrideCursor(Qt.WaitCursor)
                    #qApp.processEvents()
                    txt = rvr.get_pdf_txt(fname[0])
                    self.main.statusBar.showMessage(
                        'Extracting data from pdf')
                    res = rvr.read_Siemens_PET_dailyQC(txt)
                    #QApplication.restoreOverrideCursor()
                    self.main.stop_wait_cursor()
                    self.main.statusBar.showMessage('Finished', 1000)
            elif filetype == 'Siemens PET-MR DailyQC Reports (.xml)':
                #TODO
                pass
            elif filetype == 'Philips MR PIQT / SPT report (.pdf)':
                #TODO
                pass
            elif filetype == 'Philips MR ACR report (.pdf)':
                #TODO
                pass

            if res['status']:
                self.main.wResTable.result_table.fill_table(
                    row_labels=res['headers'],
                    col_labels=[Path(fname[0]).stem],
                    values_cols=[res['values_txt']],
                    linked_image_list=False)

'''
class SpinRatio(QDoubleSpinBox):
    """QDoubleSpinbox for ratios (0-100)."""

    def __init__(self, parent, decimals=1):
        super().__init__()
        self.setRange(0.1, 100.)
        self.setDecimals(decimals)
        self.setSingleStep(0.1)
        if decimals == 2:
            self.setSingleStep(0.01)
        self.valueChanged.connect(self.parent.clear_results)


class SpinDegrees(QDoubleSpinBox):
    """QDoubleSpinbox for degrees (-359.9-359.9)."""

    def __init__(self, parent):
        super().__init__()
        self.setRange(-359.9, 359.9)
        self.setDecimals(1)
        self.valueChanged.connect(self.parent.clear_results)
'''


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
            For use in param_changed of TestTabXX. The default is ''.
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

        actImport = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}import.png'),
            'Import table from clipboard', self)
        actImport.triggered.connect(self.import_table)
        actCopy = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}copy.png'),
            'Copy table to clipboard', self)
        actCopy.triggered.connect(self.copy_table)
        actAdd = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add row', self)
        actAdd.triggered.connect(self.add_row)
        actDel = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete row', self)
        actDel.triggered.connect(self.delete_row)
        actGetPos = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}selectArrow.png'),
            'Get position from last mouseclick in image', self)
        actGetPos.triggered.connect(self.get_pos_mouse)
        tb = QToolBar()
        tb.addActions([actImport, actCopy, actAdd, actDel, actGetPos])
        tb.setOrientation(Qt.Vertical)
        hlo.addWidget(tb)
        self.table = CTnTable(self.parent, self.main)
        hlo.addWidget(self.table)

    def import_table(self):
        """Import contents to table from clipboard or from predefined."""
        dlg = uir.QuestionBox(
            parent=self.main, title='Import table',
            msg='Import table from...',
            yes_text='Clipboard',
            no_text='Predefined tables')
        res = dlg.exec()
        ctn_table = None
        if res:
            df = pd.read_clipboard()
            nrows, ncols = df.shape
            if ncols != 4:
                pass #TODO ask for separator / decimal or guess?
                errmsg = [
                    'Failed reading table from clipboard.',
                    'Expected 4 columns of data that are',
                    'separated by tabs as if copied to clipboard',
                    'from ImageQC or copied from Excel.'
                    ]
                QMessageBox.warning(
                    self, 'Failed reading table', "\n".join(errmsg))
            else:
                ctn_table = cfc.HUnumberTable()
                ctn_table.materials = [
                    str(df.iat[row, 1]) for row in range(nrows)]
                ctn_table.pos_x = [
                    float(df.iat[row, 2]) for row in range(nrows)]
                ctn_table.pos_y = [
                    float(df.iat[row, 3]) for row in range(nrows)]
                ctn_table.relative_mass_density = [
                    float(df.iat[row, 4]) for row in range(nrows)]
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
        df = pd.DataFrame(dict_2_pd)
        df.to_clipboard(index=False)
        self.main.statusBar.showMessage('Values in clipboard', 2000)

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
            szActy, szActx = np.shape(self.main.active_img)
            image_dict = self.main.imgs[self.main.vGUI.active_img_no]
            dx = self.main.vGUI.last_clicked_pos[0] - 0.5 * szActx
            dxmm = round(dx * image_dict.pix[0], 1)
            dy = self.main.vGUI.last_clicked_pos[1] - 0.5 * szActy
            dymm = round(dy * image_dict.pix[1], 1)
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
        #self.parent.flag_edit = False

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

        for c in range(4):
            this_col = values[c]
            for r in range(n_rows):
                twi = QTableWidgetItem(str(this_col[r]))
                if c > 0:
                    twi.setTextAlignment(4)
                self.setItem(r, c, twi)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.blockSignals(False)
        self.parent.main.update_roi(clear_results_test=True)
