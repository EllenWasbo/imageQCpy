# -*- coding: utf-8 -*-
"""User interface for test tabs in main window of imageQC.

@author: Ellen Wasbø
"""
import os
import copy
from dataclasses import fields
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QFormLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QCheckBox, QRadioButton, QButtonGroup,
    QComboBox, QAction, QToolBar, QTableWidget, QTableWidgetItem, QTimeEdit,
    QMessageBox, QInputDialog, QFileDialog, QDialogButtonBox, QHeaderView
    )

# imageQC block start
from imageQC.config.iQCconstants import (
    ENV_ICON_PATH, ALTERNATIVES, ALTERNATIVES_ROI, HALFLIFE, HEADERS, HEADERS_SUP)
from imageQC.config import config_func as cff
from imageQC.config import config_classes as cfc
from imageQC.ui import reusable_widgets as uir
from imageQC.ui import messageboxes
from imageQC.ui.tag_patterns import TagPatternTreeTestDCM
from imageQC.scripts.calculate_qc import calculate_qc
from imageQC.scripts.mini_methods import get_all_matches
from imageQC.ui.ui_dialogs import ImageQCDialog
# imageQC block end


flatfield_info_txt = '''
    Based on European guidelines for quality assurance in breast cancer screening
    and diagnosis<br>
    Fourth edition Supplements (2013)<br>
    <br>
    Variance matrix added as specified in the European guidelines.<br>
    <br>
    For Siemens equipment one courner may be filled with a high value and for <br>
    Hologic equiment an upper and lower rim may be filled with a high value.<br>
    Use the option to mask pixels with maximum value.

    Increasing the limit for ignoring an ROI based on masked pixels from 0 % <br>
    will cause the statistics of the unmasked pixels to be calculated.<br>
    This limit cannot be set higher than 95% (i.e. accepting calculation of <br>
    ROI statistics with as little as 5% of unmasked pixels left in an ROI).
    '''

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

    def __init__(self, parent, remove_roi_num=False):
        """Initiate tabs for tests available to all modalities.

        Parameters
        ----------
        parent : MainWindow or InputMainAuto
            InputMainAuto used for task_based
        remove_roi_num : bool, optional
            True for task_based_image_quality.py and modality SR. The default is False.
        """
        super().__init__()
        self.main = parent
        self.flag_ignore_signals = True
        self.currentChanged.connect(self.main.update_current_test)

        self.create_tab_dcm()
        self.addTab(self.tab_dcm, "DCM")
        if remove_roi_num is False:
            self.create_tab_roi()
            self.create_tab_num()
            self.addTab(self.tab_roi, "ROI")
            self.addTab(self.tab_num, "Num")

    def flag_edit(self, indicate_change=True):
        """Add star after cbox_paramsets to indicate any change from saved."""
        self.main.wid_paramset.flag_edit(indicate_change)

    def update_displayed_params(self):
        """Display parameters according to current_paramset of main."""
        self.flag_ignore_signals = True
        paramset = self.main.current_paramset
        self.wid_dcm_pattern.current_template = paramset.dcm_tagpattern
        self.wid_dcm_pattern.update_data()

        if hasattr(self, 'num_digit_label'):
            self.num_digit_label.clear()
            avail_digit_labels = [''] + [
                temp.label for temp in
                self.main.digit_templates[self.main.current_modality]]
            self.num_digit_label.addItems(avail_digit_labels)

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
                    elif hasattr(reciever, 'setCurrentText'):
                        if field.name == 'sni_ref_image':
                            reciever.setCurrentText(content)
                        elif field.name == 'num_digit_label':
                            reciever.setCurrentText(content)
                    else:  # info to programmer
                        print(f'Warning: Parameter {field.name} not set ',
                              '(ui_main_test_tabs.update_displayed_params)')
            else:
                if field.name == 'ctn_table':
                    self.ctn_table_widget.table.update_table()
                elif field.name == 'ttf_table':
                    self.ttf_table_widget.table.current_table = copy.deepcopy(
                        paramset.ttf_table)
                    self.ttf_table_widget.table.update_table()
                    self.update_ttf_plot_options()
                elif field.name == 'rec_table':
                    self.rec_table_widget.table.current_table = copy.deepcopy(
                        paramset.rec_table)
                    self.rec_table_widget.table.update_table()
                elif field.name == 'num_table':
                    self.num_table_widget.table.current_table = copy.deepcopy(
                        paramset.num_table)
                    self.num_table_widget.table.update_table()
                elif field.name == 'gho_table':
                    self.gho_table_widget.table.current_table = copy.deepcopy(
                        paramset.gho_table)
                    self.gho_table_widget.table.update_table()
                elif field.name == 'roi_table':
                    self.roi_table_widget.table.current_table = copy.deepcopy(
                        paramset.roi_table)
                    self.roi_table_widget.table.update_table()
                elif field.name == 'roi_use_table':
                    new_rect_pos = True if paramset.roi_use_table == 2 else False
                    if new_rect_pos != self.roi_table_widget.use_rectangle:
                        self.roi_table_widget.use_rectangle = new_rect_pos
                        self.roi_table_widget.update_on_rectangle_change()
                    if paramset.roi_use_table == 0:
                        self.roi_table_widget.setEnabled(False)
                    else:
                        self.roi_table_widget.setEnabled(True)
                elif field.name == 'sni_channels_table':
                    self.sni_channels_table_widget.update_table()

        if self.main.current_modality == 'Xray':
            self.update_NPS_independent_pixels()

        self.update_enabled()
        self.flag_ignore_signals = False

    def update_enabled(self):
        """Update enabled/disabled features."""
        paramset = self.main.current_paramset
        if hasattr(paramset, 'roi_type'):
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

    def param_changed_from_gui(self, attribute='', update_roi=True,
                               clear_results=True, update_plot=True,
                               update_results_table=True, content=None,
                               edit_ignore=False, make_odd=False):
        """Update current_paramset with value from GUI.

        If changes found - update roi and delete results.

        Parameters
        ----------
        attribute : str
            attribute name in paramset
        update_roi : bool, optional
            True (default) if rois have to be recalculated
        clear_results : bool, optional
            True (default) if results have to be cleared
        update_plot : bool, optional
            True (default) if plot settings affeccted
        update_results_table : bool, optional
            True (default) if results table affected
        content : None, optional
            Preset content. Default is None
        edit_ignore : bool, optional
            Avoid setting flag_edit to True. Default is False.
        make_odd : bool, optional
            Force integer to be odd number. Default is False
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
                        if make_odd:
                            if content % 2 == 0:
                                prev_value = getattr(
                                    self.main.current_paramset, attribute)
                                if content > prev_value:  # increasing value
                                    content += 1
                                else:
                                    content -= 1
                                self.blockSignals(True)
                                self.sender().setValue(content)
                                self.blockSignals(False)
                elif hasattr(sender, 'setText'):
                    content = sender.text()
                elif hasattr(sender, 'setCurrentIndex'):  # QComboBox
                    content = sender.currentIndex()
                    if attribute in ['sni_ref_image', 'num_digit_label']:
                        content = sender.currentText()
                else:
                    content = None

            if content is not None:
                if clear_results:
                    self.clear_results_current_test()
                setattr(self.main.current_paramset, attribute, content)
                self.update_enabled()
                if edit_ignore is False:
                    self.flag_edit(True)
                if update_roi:
                    self.main.update_roi()
                    if attribute.startswith('sni_'):
                        self.update_sni_roi_names()
                    elif attribute == 'mtf_type' and self.main.current_modality == 'NM':
                        if content == 2:
                            self.mtf_auto_center.setChecked(True)
                if ((update_plot or update_results_table)
                        and clear_results is False):
                    if attribute == 'mtf_gaussian':
                        self.update_values_mtf()
                    elif attribute == 'rec_type':
                        self.update_values_rec()
                    self.main.refresh_results_display()
                if attribute == 'roi_use_table':
                    new_rect_pos = True if content == 2 else False
                    if new_rect_pos != self.roi_table_widget.use_rectangle:
                        self.roi_table_widget.use_rectangle = new_rect_pos
                        self.roi_table_widget.update_on_rectangle_change()
                    if content == 0:
                        self.roi_table_widget.setEnabled(False)
                    else:
                        self.roi_table_widget.setEnabled(True)
                        self.set_offset('roi_offset_xy', reset=True)
                elif attribute in ['sni_type', 'sni_channels']:
                    alt = 0 if self.main.current_paramset.sni_type == 0 else 2
                    if self.main.current_paramset.sni_channels:
                        alt = alt + 1
                    self.main.current_paramset.sni_alt = alt
                    if attribute == 'sni_channels':
                        self.update_enabled()
                if all([self.main.current_modality == 'Xray',
                        self.main.current_test == 'NPS']):
                    self.update_NPS_independent_pixels()

                # changes that might affect output settings?
                # - alternative or headers changed
                log = []
                if attribute in [
                        'sli_type', 'mtf_type', 'rec_type', 'snr_type',
                        'roi_use_table', 'hom_tab_alt', 'sni_alt']:
                    _, log = cff.verify_output_alternative(
                        self.main.current_paramset, testcode=self.main.current_test)
                if log:
                    msg = ('Output settings are defined for this parameterset. '
                           'Changing this parameter will change the headers of '
                           'the Result table. Consider updating the output settings.')
                    dlg = messageboxes.MessageBoxWithDetails(
                        self, title='Warnings',
                        msg=msg,
                        info='See details',
                        icon=QMessageBox.Warning,
                        details=log)
                    dlg.exec()

    def clear_results_current_test(self):
        """Clear results of current test."""
        if self.main.current_test in [*self.main.results]:
            self.main.results[self.main.current_test] = None
            self.main.refresh_results_display()

    def update_values_mtf(self):
        """Update MTF table values when changing analytic vs discrete options."""
        if 'MTF' in self.main.results:
            #  and self.main.current_modality in ['CT', 'Xray', 'SPECT', 'MR']):
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
                    try:
                        new_values_this = details_dict[0][
                            prefix + 'MTF_details']['values']
                    except KeyError:
                        new_values_this = []
                    try:
                        new_values_this.extend(
                                details_dict[1][prefix + 'MTF_details']['values'])
                    except IndexError:
                        pass  # only if x and y dir
                    new_values.append(new_values_this)

                self.main.results['MTF']['values'] = new_values
                self.main.refresh_results_display()
                self.main.status_bar.showMessage('MTF tabular values updated', 1000)

    def update_values_rec(self):
        """Update Rec table values when changing type of values to display."""
        if 'Rec' in self.main.results:
            try:
                rec_type = self.main.current_paramset.rec_type
                proceed = True
            except AttributeError:
                proceed = False
            if proceed:
                if 'values' in self.main.results['Rec']:
                    self.main.results['Rec']['values'] = (
                        [self.main.results['Rec']['details_dict']['values'][rec_type]])
                    self.main.results['Rec']['headers'] = (
                        HEADERS[self.main.current_modality]['Rec'][
                            'alt' + str(rec_type)])
                    self.main.refresh_results_display()
                    self.main.status_bar.showMessage('Tabular values updated', 1000)

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
        pos = [0., 0.]
        if not reset:
            proceed = True
            if self.main.current_test == 'ROI':
                if self.main.current_paramset.roi_use_table > 0:
                    QMessageBox.warning(
                        self, 'Extra offset ignored',
                        'Extra offsets are ignored when table is used.')
                    proceed = False
            if proceed:
                sz_img_y, sz_img_x = np.shape(self.main.active_img)
                xpos = self.main.gui.last_clicked_pos[0] - 0.5 * sz_img_x
                ypos = self.main.gui.last_clicked_pos[1] - 0.5 * sz_img_y
                attr_mm = attribute[:3] + '_offset_mm'
                val_mm = getattr(self.main.current_paramset, attr_mm, False)
                if val_mm:
                    image_info = self.main.imgs[self.main.gui.active_img_no]
                    xpos = np.round(xpos * image_info.pix[0], decimals=1)
                    ypos = np.round(ypos * image_info.pix[1], decimals=1)
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
        self.tab_dcm.vlo.addWidget(QLabel(
            'Select tag in list above to plot values in tab "Results plot" below.'))

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
        self.roi_a = QDoubleSpinBox(decimals=1, minimum=-359.9, maximum=359.9,
                                    singleStep=0.1)
        self.roi_a.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_a'))
        self.roi_use_table = QComboBox()
        self.roi_use_table.addItems(ALTERNATIVES_ROI)
        self.roi_use_table.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(
                attribute='roi_use_table', update_roi=False))
        self.roi_table_widget = PositionWidget(
                self, self.main, table_attribute_name='roi_table')
        self.roi_table_widget.setEnabled(False)  # default irresponsive
        all_headers = HEADERS['CT']['ROI']['alt0'] + HEADERS_SUP['CT']['ROI']['alt0']
        self.roi_table_val = QComboBox()
        self.roi_table_val.addItems(all_headers)
        self.roi_table_val.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_table_val'))
        self.roi_table_val_sup = QComboBox()
        self.roi_table_val_sup.addItems(all_headers)
        self.roi_table_val_sup.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='roi_table_val_sup'))

        self.create_offset_widget('roi')

        vlo_left = QVBoxLayout()
        self.tab_roi.hlo.addLayout(vlo_left)

        flo1 = QFormLayout()
        flo1.addRow(QLabel('ROI shape'), self.roi_type)
        flo1.addRow(QLabel('Radius of circular ROI (mm)'), self.roi_radius)
        flo1.addRow(QLabel('Rectangular ROI width (mm)'), self.roi_x)
        flo1.addRow(QLabel('Rectangular ROI height (mm)'), self.roi_y)
        flo1.addRow(QLabel('ROI rotation (degrees)'), self.roi_a)
        vlo_left.addLayout(flo1)

        hlo_offset = QHBoxLayout()
        hlo_offset.addWidget(self.wid_roi_offset)
        hlo_offset.addStretch()
        vlo_left.addLayout(hlo_offset)

        vlo_right = QVBoxLayout()
        self.tab_roi.hlo.addLayout(vlo_right)

        vlo_right.addWidget(self.roi_use_table)
        vlo_right.addWidget(self.roi_table_widget)
        hlo_opt_val = QHBoxLayout()
        vlo_right.addLayout(hlo_opt_val)
        hlo_opt_val.addWidget(QLabel('Result table:'))
        hlo_opt_val.addWidget(self.roi_table_val)
        hlo_opt_val.addWidget(QLabel('Supplement table:'))
        hlo_opt_val.addWidget(self.roi_table_val_sup)

    def create_tab_num(self):
        """GUI of tab NUM."""
        self.tab_num = ParamsWidget(self, run_txt='Find numbers')

        self.tab_num.hlo_top.addWidget(uir.LabelItalic(
            'Read numbers in image (e.g. savescreens). '
            'Mark area with the number and add as ROI.'))
        info_txt = '''
        Make sure to have a Digit template fitting the font used in your image.<br>
        (A digit template can be defined and tested in Settings.)<br>
        <br>
        Set ROI by marking the area with the number and find the arrow-button to the<br>
        left of the table. Then the coordinates of the area will be added to <br>
        the ROI list. Label the ROIs to define the headers of the result table.
        '''
        self.tab_num.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.num_table_widget = PositionWidget(
                self, self.main, table_attribute_name='num_table',
                headers=['ROI label', '(x1,x2)', '(y1,y2)'],
                use_rectangle=True)
        self.num_digit_label = QComboBox()
        self.num_digit_label.setFixedWidth(250)
        self.num_digit_label.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='num_digit_label'))

        vlo_left = QVBoxLayout()
        self.tab_num.hlo.addLayout(vlo_left)

        flo1 = QFormLayout()
        flo1.addRow(QLabel('Digit template'), self.num_digit_label)
        vlo_left.addLayout(flo1)
        self.tab_num.hlo.addWidget(self.num_table_widget)

    def hom_get_coordinates(self):
        """Add coordinates of deviating pixels to results."""
        if 'Hom' in self.main.results:
            try:
                details_dict = self.main.results['Hom']['details_dict'][
                    self.main.gui.active_img_no]
            except (IndexError, KeyError):
                details_dict = None
            if details_dict:
                if 'deviating_pixel_coordinates' not in details_dict:
                    deviating_pixels = details_dict['deviating_pixels']
                    coords = []
                    idxs = np.where(deviating_pixels)
                    try:
                        ys = idxs[0]
                        xs = idxs[1]
                        coords = [(xs[i], ys[i]) for i in range(xs.size)]
                        details_dict['deviating_pixel_coordinates'] = coords
                    except IndexError:
                        pass
                else:
                    coords = details_dict['deviating_pixel_coordinates']
                if len(coords) == 0:
                    QMessageBox.information(
                        self, 'No deviating pixels found',
                        'Found no deviating pixels for the current image.')
                else:
                    question = 'Copy list of coordinates to clipboard?'
                    proceed = messageboxes.proceed_question(self, question)
                    if proceed:
                        df = pd.DataFrame(coords)
                        df.columns = ['x', 'y']
                        df.to_clipboard(index=False, excel=True)
                        self.main.status_bar.showMessage('Values in clipboard', 2000)
                self.main.refresh_results_display()
            else:
                QMessageBox.warning(
                    self, 'Calculate homogeneity first',
                    'Could not find results for current image. '
                    'Calculate homogeneity first.')

    def create_tab_hom_flatfield(self):
        """GUI for Mammo flatfield test (Hom) available also for Xray."""

        self.hom_roi_size_variance = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=300, singleStep=0.1)
        self.hom_roi_size_variance.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_size_variance'))
        self.hom_variance = QCheckBox()
        self.hom_variance.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='hom_variance'))
        self.hom_mask_max = QCheckBox()
        self.hom_mask_max.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='hom_mask_max'))
        self.hom_mask_outer_mm = QDoubleSpinBox(
            decimals=1, minimum=0., maximum=1000, singleStep=0.1)
        self.hom_mask_outer_mm.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_mask_outer_mm'))
        self.hom_ignore_roi_percent = QDoubleSpinBox(
            decimals=0, minimum=0., maximum=95, singleStep=1)
        self.hom_ignore_roi_percent.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_ignore_roi_percent'))
        self.hom_deviating_pixels = QDoubleSpinBox(
            decimals=0, minimum=1, singleStep=1)
        self.hom_deviating_pixels.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_deviating_pixels'))
        self.hom_deviating_rois = QDoubleSpinBox(
            decimals=0, minimum=1, singleStep=1)
        self.hom_deviating_rois.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_deviating_rois'))

        self.hom_result_image = QComboBox()
        self.hom_result_image.addItems(
            [
                'Average pr ROI map',
                'SNR pr ROI map',
                'Variance pr ROI map',
                'Average pr ROI (% difference from global average)',
                'SNR pr ROI (% difference from global SNR)',
                'Pixel values (% difference from global average)',
                'Deviating ROIs',
                'Deviating pixels',
                '# deviating pixels pr ROI'
             ])
        self.hom_result_image.currentIndexChanged.connect(
            self.main.wid_res_image.canvas.result_image_draw)

        self.flat_widget = QWidget()
        hlo_flat_widget = QHBoxLayout()
        self.flat_widget.setLayout(hlo_flat_widget)

        flo = QFormLayout()
        flo.addRow(QLabel('Calculate variance within each ROI'), self.hom_variance)
        flo.addRow(QLabel('     ROI size variance (mm)'), self.hom_roi_size_variance)
        flo.addRow(QLabel('Mask pixels with max values'), self.hom_mask_max)
        flo.addRow(QLabel('Ignore outer mm'), self.hom_mask_outer_mm)
        flo.addRow(QLabel('Ignore ROIs where more than (%) pixels masked'),
                   self.hom_ignore_roi_percent)
        hlo_flat_widget.addLayout(flo)
        hlo_flat_widget.addWidget(uir.VLine())
        vlo_right = QVBoxLayout()
        hlo_flat_widget.addLayout(vlo_right)
        flo_right = QFormLayout()
        flo_right.addRow(QLabel('Deviating pixels (% from average)'),
                         self.hom_deviating_pixels)
        flo_right.addRow(QLabel('Deviating ROIs (% from average)'),
                         self.hom_deviating_rois)
        vlo_right.addLayout(flo_right)
        hlo_res_img = QHBoxLayout()
        hlo_res_img.addWidget(QLabel('Result image'))
        hlo_res_img.addWidget(self.hom_result_image)
        vlo_right.addLayout(hlo_res_img)
        btn_get_coord = QPushButton('Get coordinates of deviating pixels')
        btn_get_coord.clicked.connect(self.hom_get_coordinates)
        vlo_right.addWidget(btn_get_coord)

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
        
    def create_tab_mtf_ct_spect_pet(self, modality='CT'):
        """GUI of tab MTF - common to CT/SPECT/PET."""
        self.mtf_roi_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000, singleStep=0.1)
        self.mtf_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_roi_size'))
        self.mtf_background_width = QDoubleSpinBox(
            decimals=1, minimum=0, maximum=1000, singleStep=0.1)
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
        self.mtf_type.addItems(ALTERNATIVES[modality]['MTF'])

        plot_items = ['Centered xy profiles',
                      'Sorted pixel values', 'LSF', 'MTF']
        if modality in ['SPECT', 'PET']:
            plot_items.extend(
                ['Line source max z-profile',
                 'Sliding window FWHM z-profile',
                 'Sliding window x/y offset z-profile'])
            self.mtf_line_tolerance = QDoubleSpinBox(
                decimals=0, minimum=0.1, singleStep=1)
            self.mtf_line_tolerance.valueChanged.connect(
                lambda: self.param_changed_from_gui(
                    attribute='mtf_line_tolerance'))
            self.mtf_sliding_window = QDoubleSpinBox(
                decimals=0, minimum=3, singleStep=1)
            self.mtf_sliding_window.valueChanged.connect(
                lambda: self.param_changed_from_gui(
                    attribute='mtf_sliding_window', make_odd=True))
        self.mtf_plot.addItems(plot_items)
        if modality == 'PET':
            self.blockSignals(True)
            self.mtf_plot.setCurrentIndex(5)
            self.blockSignals(False)

        if modality == 'CT':
            self.mtf_cy_pr_mm = BoolSelectTests(
                self, attribute='mtf_cy_pr_mm',
                text_true='cy/mm', text_false='cy/cm',
                update_roi=False, clear_results=False)
            self.create_offset_widget('mtf')

        vlo1 = QVBoxLayout()
        flo1 = QFormLayout()
        flo1.addRow(QLabel('MTF method'), self.mtf_type)
        flo1.addRow(QLabel('ROI radius (mm)'), self.mtf_roi_size)
        txt = 'bead/wire' if modality == 'CT' else 'point/line'
        flo1.addRow(
            QLabel(f'Width of background ({txt})'), self.mtf_background_width)
        flo1.addRow(QLabel('Auto center ROI in max'), self.mtf_auto_center)
        if modality in ['SPECT', 'PET']:
            flo1.addRow(QLabel('Linesource: ignore slices with max diff % from top 3 slices'),
                        self.mtf_line_tolerance)
            flo1.addRow(QLabel('Sliding window width (N slices)'),
                        self.mtf_sliding_window)
        flo1.addRow(QLabel('Sampling freq. gaussian (mm-1)'),
                    self.mtf_sampling_frequency)
        vlo1.addLayout(flo1)
        if modality == 'CT':
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
        if modality == 'CT':
            flo3.addRow(QLabel('Table results as'), self.mtf_cy_pr_mm)
        flo3.addRow(QLabel('Plot'), self.mtf_plot)
        vlo2.addLayout(flo3)
        self.tab_mtf.hlo.addLayout(vlo2)

    def create_tab_mtf_xray_mr(self):
        """GUI of tab MTF - common to Xray/Mammo and MR."""
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

        self.mtf_auto_center_mask_outer = QDoubleSpinBox(
            decimals=0, minimum=0, maximum=200, singleStep=1)
        self.mtf_auto_center_mask_outer.editingFinished.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_auto_center_mask_outer'))

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
        flo_gb_auto = QFormLayout()
        flo_gb_auto.addRow(QLabel('Use'), self.mtf_auto_center_type)
        flo_gb_auto.addRow(QLabel('Ignore outer mm'), self.mtf_auto_center_mask_outer)
        self.mtf_auto_center.setLayout(flo_gb_auto)
        flo2 = QFormLayout()
        flo2.addRow(QLabel('Cut LSF tails'), self.mtf_cut_lsf)
        flo2.addRow(QLabel('    Cut at halfmax + (#FWHM)'), self.mtf_cut_lsf_w)
        vlo2.addLayout(flo2)
        flo3 = QFormLayout()
        flo3.addRow(QLabel('Table results from'), self.mtf_gaussian)
        flo3.addRow(QLabel('Plot'), self.mtf_plot)
        vlo2.addLayout(flo3)
        self.tab_mtf.hlo.addLayout(vlo2)

    def create_tab_nps_xray(self):
        """GUI of tab NPS Xray and Mammo."""
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

    def create_tab_rin(self):
        """GUI of tab for Ring artefacts CT+SPECT."""
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
        self.nps_normalize.addItems(['', 'Area under curve (AUC)',
                                     'Large area signal ^2 (LAS)'])
        self.nps_normalize.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(
                attribute='nps_normalize', update_roi=False, clear_results=False))
        self.nps_plot = QComboBox()
        self.nps_plot.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(
                attribute='nps_plot', update_roi=False, clear_results=False))

        self.flo_nps_plot = QFormLayout()
        self.flo_nps_plot.addRow(QLabel('NPS sampling frequency (1/mm)'),
                                 self.nps_sampling_frequency)
        self.flo_nps_plot.addRow(QLabel('Smooth NPS by width (1/mm)'),
                                 self.nps_smooth_width)
        self.flo_nps_plot.addRow(QLabel('Normalize NPS curve by'), self.nps_normalize)
        self.flo_nps_plot.addRow(QLabel('Plot'), self.nps_plot)

    def run_current(self):
        """Run selected test."""
        tests = []
        marked_this = self.main.get_marked_imgs_current_test()
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
        try:
            self.main.wid_quicktest.get_current_template()
            self.main.current_quicktest = copy.deepcopy(
                self.main.wid_quicktest.current_template)
        except AttributeError:
            pass  # task based
        self.main.current_quicktest.tests = tests
        max_progress = 100  # %
        self.main.progress_modal = uir.ProgressModal(
            "Calculating...", "Cancel",
            0, max_progress, self, minimum_duration=0)
        calculate_qc(self.main)

        if len(marked_this) > 0 and hasattr(self.main, 'tree_file_list'):
            if self.main.gui.active_img_no not in marked_this:
                self.main.set_active_img(marked_this[0])


class ParamsTabCT(ParamsTabCommon):
    """Widget for CT tests."""

    def __init__(self, parent, task_based=False):
        """Initiate tabs for CT tests.

        Parameters
        ----------
        parent : MainWindow or InputMainAuto
            InputMainAuto used for task_based
        task_based : bool, optional
            True used for task_based_image_quality.py. The default is False.
        """
        super().__init__(parent, remove_roi_num=task_based)

        if task_based:
            self.create_tab_ttf()
            self.create_tab_nps()

            self.addTab(self.tab_ttf, "TTF")
            self.addTab(self.tab_nps, "NPS")
        else:
            self.create_tab_hom()
            self.create_tab_noi()
            self.create_tab_sli()
            self.create_tab_mtf()
            self.create_tab_ttf()
            self.create_tab_ctn()
            self.create_tab_huw()
            self.create_tab_rin()
            self.create_tab_dim()
            self.create_tab_nps()

            self.addTab(self.tab_hom, "Homogeneity")
            self.addTab(self.tab_noi, "Noise")
            self.addTab(self.tab_sli, "Slice thickness")
            self.addTab(self.tab_mtf, "MTF")
            self.addTab(self.tab_ttf, "TTF")
            self.addTab(self.tab_ctn, "CT number")
            self.addTab(self.tab_huw, "HU water")
            self.addTab(self.tab_rin, "Ring artifacts")
            self.addTab(self.tab_dim, "Dimensions")
            self.addTab(self.tab_nps, "NPS")

            self.flag_ignore_signals = False

    def update_enabled(self):
        """Update enabled/disabled features."""
        super().update_enabled()
        paramset = self.main.current_paramset

        try:
            if paramset.mtf_cut_lsf:
                self.mtf_cut_lsf_w.setEnabled(True)
                self.mtf_cut_lsf_w_fade.setEnabled(True)
            else:
                self.mtf_cut_lsf_w.setEnabled(False)
                self.mtf_cut_lsf_w_fade.setEnabled(False)
        except AttributeError:
            pass

        try:
            if paramset.ttf_cut_lsf:
                self.ttf_cut_lsf_w.setEnabled(True)
                self.ttf_cut_lsf_w_fade.setEnabled(True)
            else:
                self.ttf_cut_lsf_w.setEnabled(False)
                self.ttf_cut_lsf_w_fade.setEnabled(False)
        except AttributeError:
            pass

        try:
            if paramset.sli_type == 1:
                self.sli_ramp_distance.setEnabled(False)
            else:
                self.sli_ramp_distance.setEnabled(True)
        except AttributeError:
            pass

    def update_sli_plot_options(self):
        """Update plot options for slice thickness."""
        self.sli_plot.clear()
        items = ['all']
        tan_a = 0
        self.blockSignals(True)
        if self.sli_type.currentIndex() == 0:
            items.extend(['H1 upper', 'H2 lower', 'V1 left', 'V2 right'])
            dist = 38.
            tan_a = 0.42
        elif self.sli_type.currentIndex() == 1:
            items.extend(['H1 upper', 'H2 lower',
                          'V1 left', 'V2 right', 'V1 inner', 'V2 inner'])
            dist = 45.
        elif self.sli_type.currentIndex() == 2:
            items.extend(['V1 left', 'V2 right'])
            dist = 70.
        elif self.sli_type.currentIndex() == 3:
            dist = 0.
            tan_a = 0.42
        else:  # == 4
            items.extend(['H1 upper', 'H2 lower'])
            dist = 60.
            tan_a = 0.5
        self.main.current_paramset.sli_ramp_distance = dist
        self.sli_ramp_distance.setValue(dist)
        self.sli_tan_a.setValue(tan_a)
        self.sli_tan_a.setEnabled(tan_a > 0)
        self.blockSignals(False)
        self.sli_plot.addItems(items)
        self.param_changed_from_gui(attribute='sli_type')

    def update_ttf_plot_options(self):
        """Update plot options for TTF on materials change."""
        curr_select = self.ttf_plot_material.currentText()
        self.ttf_plot_material.clear()
        items = ['All'] + self.main.current_paramset.ttf_table.labels
        self.ttf_plot_material.addItems(items)
        if curr_select in items:
            self.ttf_plot_material.setCurrentText(curr_select)
        else:
            self.ttf_plot_material.setCurrentIndex(0)

    def create_tab_hom(self):
        """GUI of tab Homogeneity."""
        self.tab_hom = ParamsWidget(self, run_txt='Calculate homogeneity')
        self.tab_hom.hlo_top.addWidget(uir.LabelItalic('Homogeneity'))

        self.hom_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.hom_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_size'))
        self.hom_roi_distance = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000, singleStep=0.1)
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

        self.noi_roi_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000, singleStep=0.1)
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
        self.sli_ramp_distance = QDoubleSpinBox(decimals=1, minimum=0.)
        self.sli_ramp_distance.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_ramp_distance'))
        self.sli_tan_a = QDoubleSpinBox(decimals=2, minimum=0.)
        self.sli_tan_a.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_tan_a'))
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
        self.sli_median_filter = QDoubleSpinBox(decimals=0, minimum=0)
        self.sli_median_filter.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_median_filter'))
        self.sli_auto_center = QCheckBox('')
        self.sli_auto_center.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='sli_auto_center'))
        self.sli_plot = QComboBox()
        self.update_sli_plot_options()
        self.sli_plot.currentIndexChanged.connect(self.main.refresh_results_display)

        info_txt = '''
        When using the beaded ramp for CatPhan note that the inner ramps should be
        used for nominal slice thickness less than 2 mm.
        '''
        self.tab_sli.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))
        hlo_type = QHBoxLayout()
        hlo_type.addWidget(QLabel('Ramp type'))
        hlo_type.addWidget(self.sli_type)
        hlo_type.addStretch()
        self.tab_sli.vlo_top.addLayout(hlo_type)

        flo1 = QFormLayout()
        flo1.addRow(QLabel('Center to ramp distance (mm)'), self.sli_ramp_distance)
        flo1.addRow(QLabel('Tangens to ramp angle'), self.sli_tan_a)
        flo1.addRow(QLabel('Profile length (mm)'), self.sli_ramp_length)
        flo1.addRow(QLabel('Profile search margin (pix)'), self.sli_search_width)
        flo1.addRow(QLabel('Auto center'), self.sli_auto_center)
        flo2 = QFormLayout()
        flo2.addRow(
            QLabel('Within search margin, average over neighbour profiles (#)'),
            self.sli_average_width)
        flo2.addRow(
            QLabel('Median filter profiles, filter width (# pixels)'),
            self.sli_median_filter)
        flo2.addRow(QLabel('Background from profile outer (mm)'),
                    self.sli_background_width)
        flo2.addRow(QLabel('Plot image profiles'), self.sli_plot)
        self.tab_sli.hlo.addLayout(flo1)
        self.tab_sli.hlo.addWidget(uir.VLine())
        self.tab_sli.hlo.addLayout(flo2)

    def create_tab_ttf(self):
        """GUI of tab TTF."""
        self.tab_ttf = ParamsWidget(self, run_txt='Calculate TTF')
        self.tab_ttf.hlo_top.addWidget(uir.LabelItalic(
            'Task based transfer function (TTF)'))
        info_txt = '''
        Calculate MTF for circular disk for different materials (cylindric inserts).<br>
        Specify material names and approximate center of material inserts.<br>
        Material inserts from standard phantoms can be imported.<br>
        Calculations will be performed per seriesUIDs.
        '''
        self.tab_ttf.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.ttf_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.ttf_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='ttf_roi_size'))
        self.ttf_cut_lsf = QCheckBox('')
        self.ttf_cut_lsf.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='ttf_cut_lsf'))
        self.ttf_cut_lsf_w = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.ttf_cut_lsf_w.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='ttf_cut_lsf_w'))
        self.ttf_cut_lsf_w_fade = QDoubleSpinBox(decimals=1, minimum=0, singleStep=0.1)
        self.ttf_cut_lsf_w_fade.valueChanged.connect(
            lambda: self.param_changed_from_gui(
                attribute='ttf_cut_lsf_w_fade'))
        self.ttf_gaussian = BoolSelectTests(
            self, attribute='ttf_gaussian',
            text_true='Gaussian', text_false='Discrete',
            update_roi=False, clear_results=False, update_plot=False)
        self.ttf_sampling_frequency = QDoubleSpinBox(
            decimals=3, minimum=0.001, singleStep=0.001)
        self.ttf_sampling_frequency.valueChanged.connect(
                    lambda: self.param_changed_from_gui(
                        attribute='ttf_sampling_frequency', update_roi=False,
                        clear_results=False, update_plot=False,
                        update_results_table=False))

        headers = ['Material', 'pos x [mm]', 'pos y [mm]']
        self.ttf_table_widget = PositionWidget(
            self, self.main, table_attribute_name='ttf_table', headers=headers)
        self.ttf_plot_material = QComboBox()
        self.ttf_plot_material.addItems(
            ['All'] + self.main.current_paramset.ttf_table.labels)
        self.ttf_plot_material.currentIndexChanged.connect(
            self.main.wid_res_plot.plotcanvas.plot)
        self.ttf_plot = QComboBox()
        self.ttf_plot.addItems(['Centered xy profiles',
                                'Sorted pixel values', 'LSF', 'MTF'])
        self.ttf_plot.currentIndexChanged.connect(
            self.main.wid_res_plot.plotcanvas.plot)

        vlo1 = QVBoxLayout()
        flo1 = QFormLayout()
        flo1.addRow(QLabel('ROI radius (mm)'), self.ttf_roi_size)
        flo1.addRow(QLabel('Cut LSF tails'), self.ttf_cut_lsf)
        flo1.addRow(QLabel('    Cut at halfmax + n*FWHM, n='), self.ttf_cut_lsf_w)
        flo1.addRow(
            QLabel('    Fade out within n*FWHM, n='), self.ttf_cut_lsf_w_fade)
        flo1.addRow(QLabel('Table results from'), self.ttf_gaussian)
        flo1.addRow(QLabel('Sampling freq. gaussian (mm-1)'),
                    self.ttf_sampling_frequency)
        vlo1.addLayout(flo1)

        vlo2 = QVBoxLayout()
        hlo_plot = QHBoxLayout()
        hlo_plot.addWidget(QLabel('Plot'))
        hlo_plot.addWidget(self.ttf_plot)
        hlo_plot.addWidget(QLabel('for material'))
        hlo_plot.addWidget(self.ttf_plot_material)
        vlo2.addWidget(self.ttf_table_widget)
        vlo2.addLayout(hlo_plot)

        self.tab_ttf.hlo.addLayout(vlo1)
        self.tab_ttf.hlo.addWidget(uir.VLine())
        self.tab_ttf.hlo.addLayout(vlo2)

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
        self.ctn_plot = QComboBox()
        self.ctn_plot.addItems([
            'HU difference for set min/max',
            # 'HU difference for set min/max (%)',
            'CT number linearity'])
        self.ctn_plot.currentIndexChanged.connect(
            self.main.wid_res_plot.plotcanvas.plot)

        flo1 = QFormLayout()
        flo1.addRow(QLabel('ROI radius (mm)'), self.ctn_roi_size)
        flo1.addRow(QLabel('Search for circular element'), self.ctn_search)
        flo1.addRow(QLabel('Search radius (mm)'), self.ctn_search_size)
        flo1.addRow(QLabel('Auto center'), self.ctn_auto_center)
        flo1.addRow(QLabel('Plot'), self.ctn_plot)
        self.tab_ctn.hlo.addLayout(flo1)
        self.tab_ctn.hlo.addWidget(uir.VLine())
        self.tab_ctn.hlo.addWidget(self.ctn_table_widget)

    def create_tab_huw(self):
        """GUI of tab HU water."""
        self.tab_huw = ParamsWidget(self, run_txt='Calculate HU in water')

        self.huw_roi_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000, singleStep=0.1)
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
        self.create_tab_mtf_ct_spect_pet(modality='CT')

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
        self.nps_roi_distance = QDoubleSpinBox(
            decimals=1, minimum=0, maximum=1000, singleStep=0.1)
        self.nps_roi_distance.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_roi_distance'))
        self.nps_n_sub = QDoubleSpinBox(
            decimals=0, minimum=1, maximum=1000, singleStep=1)
        self.nps_n_sub.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='nps_n_sub'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI size (pix)'), self.nps_roi_size)
        flo.addRow(QLabel('Radius to center of ROIs (mm)'), self.nps_roi_distance)
        flo.addRow(QLabel('Number of ROIs'), self.nps_n_sub)

        self.add_NPS_plot_settings()
        self.nps_plot.addItems(
            ['NPS pr image', 'NPS average all images',
             'NPS pr image + average', 'NPS all images + average'])
        self.tab_nps.hlo.addLayout(flo)
        self.tab_nps.hlo.addWidget(uir.VLine())
        self.tab_nps.hlo.addLayout(self.flo_nps_plot)

    def create_tab_dpr(self):
        """GUI of tab d-prime used in task based window."""
        self.tab_dpr = ParamsWidget(self, run_txt='Calculate detectability')
        self.tab_dpr.hlo_top.addWidget(uir.LabelItalic('Detectability'))
        info_txt = '''
        Details on methods used can be found in AAPM TG-233<br>
        '''
        self.tab_dpr.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.dpr_contrast = QDoubleSpinBox(decimals=0, minimum=1, singleStep=1)
        self.dpr_contrast.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='dpr_contrast'))
        self.dpr_size = QDoubleSpinBox(decimals=0, minimum=1, singleStep=1)
        self.dpr_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='dpr_size'))
        self.dpr_designer = QCheckBox()
        self.dpr_designer.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='dpr_designer'))
        self.dpr_power = QDoubleSpinBox(decimals=2, minimum=0.25, maximum=2.,
                                        singleStep=0.05)
        self.dpr_power.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='dpr_power'))

        self.dpr_plot = QComboBox()
        self.dpr_plot.addItems(['..',
                                '...'])
        self.dpr_plot.currentIndexChanged.connect(
            self.main.wid_res_plot.plotcanvas.plot)

        self.dpr_result_image = QComboBox()
        self.dpr_result_image.addItems(
            [
                '--',
                '---',
             ])
        self.dpr_result_image.currentIndexChanged.connect(
            self.main.wid_res_image.canvas.result_image_draw)

        flo = QFormLayout()
        flo.addRow(QLabel('Task contrst (HU)'), self.dpr_contrast)
        flo.addRow(QLabel('Task radius (mm)'), self.dpr_size)
        flo.addRow(QLabel('Designer contrast profile'), self.dpr_designer)
        flo.addRow(QLabel('     sharpness constant'), self.dpr_power)

        flo2 = QFormLayout()
        flo2.addRow(QLabel('Plot'), self.dpr_plot)
        flo2.addRow(QLabel('Result image'), self.dpr_result_image)

        vlo1 = QVBoxLayout()
        vlo2 = QVBoxLayout()
        self.tab_dpr.hlo.addLayout(vlo1)
        self.tab_dpr.hlo.addWidget(uir.VLine())
        self.tab_dpr.hlo.addLayout(vlo2)
        vlo1.addWidget(QLabel('Task function (ideal image of signal):'))
        vlo1.addLayout(flo)
        vlo2.addLayout(flo2)
        self.tab_dpr.hlo.addStretch()

        self.addTab(self.tab_dpr, 'd-prime')


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

        self.flag_ignore_signals = False

    def update_enabled(self):
        """Update enabled/disabled features."""
        super().update_enabled()
        if self.main.current_modality == 'Xray':
            paramset = self.main.current_paramset
            if paramset.hom_tab_alt == 3:
                self.hom_roi_size_label.setText('ROI size (mm)')
                self.stack_hom.setCurrentIndex(1)
            else:
                self.hom_roi_size_label.setText('ROI radius (mm)')
                self.stack_hom.setCurrentIndex(0)

    def create_tab_hom(self):
        """GUI of tab Homogeneity."""
        self.tab_hom = ParamsWidget(self, run_txt='Calculate homogeneity')

        self.hom_tab_alt = QComboBox()
        self.hom_tab_alt.addItems(ALTERNATIVES['Xray']['Hom'])
        self.hom_tab_alt.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_tab_alt'))

        self.hom_roi_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=300, singleStep=0.1)
        self.hom_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_size'))
        self.hom_roi_size_label = QLabel('ROI radius (mm)')
        self.tab_hom.hlo_top.addWidget(self.hom_roi_size_label)
        self.tab_hom.hlo_top.addWidget(self.hom_roi_size)
        self.tab_hom.hlo_top.addSpacing(20)
        self.tab_hom.hlo_top.addWidget(QLabel('Method/output: '))
        self.tab_hom.hlo_top.addWidget(self.hom_tab_alt)
        info_txt = (
            'Method with central + quadrants ROI adapted from IPEM Report 32<br><br>'
            + 'Flat field test from Mammo:<br>'
            + flatfield_info_txt)
        self.tab_hom.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.hom_roi_rotation = QDoubleSpinBox(
            decimals=1, minimum=-359.9, maximum=359.9, singleStep=0.1)
        self.hom_roi_rotation.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_rotation'))
        self.hom_roi_distance = QDoubleSpinBox(
            decimals=1, minimum=0, maximum=1000, singleStep=0.1)
        self.hom_roi_distance.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_distance'))

        widget_hom_0 = QWidget()
        vlo_hom_0 = QVBoxLayout()
        widget_hom_0.setLayout(vlo_hom_0)
        flo = QFormLayout()
        flo.addRow(QLabel('Rotate ROI positions (deg)'), self.hom_roi_rotation)
        flo.addRow(QLabel('ROI distance (% from center)'), self.hom_roi_distance)
        hlo_hom_0 = QHBoxLayout()
        hlo_hom_0.addLayout(flo)
        hlo_hom_0.addStretch()
        vlo_hom_0.addLayout(hlo_hom_0)
        vlo_hom_0.addWidget(uir.LabelItalic(
            'Same distance for all quadrants = % of shortest'
            'center-border distance.'))
        vlo_hom_0.addWidget(uir.LabelItalic(
            'Leave distance empty to set ROIs at center of each qadrant.'))

        self.create_tab_hom_flatfield()

        self.stack_hom = QStackedWidget()
        self.stack_hom.addWidget(widget_hom_0)
        self.stack_hom.addWidget(self.flat_widget)
        self.tab_hom.hlo.addWidget(self.stack_hom)

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
        self.create_tab_nps_xray()

    def create_tab_stp(self):
        """GUI of tab STP."""
        self.tab_stp = ParamsWidget(self, run_txt='Get mean in ROI')

        self.stp_roi_size = QDoubleSpinBox(
            decimals=1, minimum=1., maximum=1000, singleStep=0.1)
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

        self.var_roi_size = QDoubleSpinBox(
            decimals=1, minimum=1., maximum=1000, singleStep=0.1)
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
        hlo_percent.addWidget(QLabel('Include % of image'))
        hlo_percent.addWidget(self.var_percent)
        hlo_percent.addStretch()
        self.tab_var.vlo.addLayout(hlo_roi_size)
        self.tab_var.vlo.addLayout(hlo_percent)


class ParamsTabMammo(ParamsTabCommon):
    """Copy for modality tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_sdn()
        self.create_tab_hom()
        self.create_tab_rlr()
        self.create_tab_gho()
        self.create_tab_mtf()
        self.create_tab_nps()

        self.addTab(self.tab_sdn, "SDNR")
        self.addTab(self.tab_hom, "Homogeneity")
        self.addTab(self.tab_rlr, "ROI left/right")
        self.addTab(self.tab_gho, "Ghost")
        self.addTab(self.tab_mtf, "MTF")
        self.addTab(self.tab_nps, "NPS")

        self.flag_ignore_signals = False

    def create_tab_sdn(self):
        """GUI of tab SDNR."""
        self.tab_sdn = ParamsWidget(self, run_txt='Calculate SDNR')

        self.tab_sdn.hlo_top.addWidget(uir.LabelItalic(
            'Signal-difference-to-noise ratio (SDNR)'))
        info_txt = '''
        Signal found from one central ROI and background from four ROIs at a distance
        from this.<br>
        Std background is the mean of the standard deviation of all four background
        ROIs.<br>
        <br>
        Based on European guidelines for quality assurance in breast cancer screening
        and diagnosis<br>
        <a href="https://op.europa.eu/en/publication-detail/-/publication/4e74ee9b-df80-4c91-a5fb-85efb0fdda2b">
        Fourth edition Supplements (2013)</a>
        '''
        self.tab_sdn.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.sdn_roi_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000., singleStep=0.1)
        self.sdn_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sdn_roi_size'))

        self.sdn_roi_dist = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000., singleStep=0.1)
        self.sdn_roi_dist.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sdn_roi_dist'))

        self.sdn_auto_center = QCheckBox()
        self.sdn_auto_center.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='sdn_auto_center'))

        self.sdn_auto_center_mask_outer = QDoubleSpinBox(
            decimals=0, minimum=0, maximum=200, singleStep=1)
        self.sdn_auto_center_mask_outer.editingFinished.connect(
            lambda: self.param_changed_from_gui(attribute='sdn_auto_center_mask_outer'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI size (mm)'), self.sdn_roi_size)
        flo.addRow(QLabel('Distance to background ROI (mm)'), self.sdn_roi_dist)
        flo.addRow(QLabel('Auto center ROI on object signal'), self.sdn_auto_center)
        flo.addRow(QLabel('If auto center, ignore outer mm'),
                   self.sdn_auto_center_mask_outer)

        self.tab_sdn.hlo.addLayout(flo)
        self.tab_sdn.hlo.addStretch()

    def create_tab_hom(self):
        """GUI of tab Homogeneity."""
        self.tab_hom = ParamsWidget(self, run_txt='Calculate Homogeneity')

        self.hom_roi_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=300, singleStep=0.1)
        self.hom_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_size'))
        self.hom_roi_size_label = QLabel('ROI size (mm)')
        self.tab_hom.hlo_top.addWidget(self.hom_roi_size_label)
        self.tab_hom.hlo_top.addWidget(self.hom_roi_size)
        self.tab_hom.hlo_top.addStretch()
        self.tab_hom.hlo_top.addWidget(uir.InfoTool(flatfield_info_txt, parent=self.main))

        self.create_tab_hom_flatfield()
        self.tab_hom.hlo.addWidget(self.flat_widget)

    def create_tab_rlr(self):
        """GUI of tab ROI left/right."""
        self.tab_rlr = ParamsWidget(self, run_txt='Calculate ROI values')

        self.rlr_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1,  maximum=10000,
                                           singleStep=0.1)
        self.rlr_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='rlr_roi_size'))
        self.rlr_x_mm = QDoubleSpinBox(decimals=1, minimum=0.1,  maximum=10000,
                                       singleStep=0.1)
        self.rlr_x_mm.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='rlr_x_mm'))

        self.rlr_relative_to_right = BoolSelectTests(
            self, attribute='rlr_relative_to_right',
            text_true='Right', text_false='Left')

        vlo_left = QVBoxLayout()
        self.tab_rlr.hlo.addLayout(vlo_left)

        flo1 = QFormLayout()
        flo1.addRow(QLabel('ROI width and height (mm)'), self.rlr_roi_size)
        flo1.addRow(QLabel('ROI distance to image border (mm)'), self.rlr_x_mm)
        flo1.addRow(QLabel('Distance to image border'),
                    self.rlr_relative_to_right)
        vlo_left.addLayout(flo1)

        self.tab_rlr.hlo.addStretch()

    def create_tab_gho(self):
        """GUI of tab Ghost factor."""
        self.tab_gho = ParamsWidget(self, run_txt='Calculate ghost factor')

        self.gho_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1,  maximum=10000,
                                           singleStep=0.1)
        self.gho_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='gho_roi_size'))

        self.gho_relative_to_right = BoolSelectTests(
            self, attribute='gho_relative_to_right',
            text_true='Right', text_false='Left')

        self.gho_table_widget = PositionWidget(
                self, self.main, table_attribute_name='gho_table')
        self.gho_table_widget.act_add.setVisible(False)
        self.gho_table_widget.act_delete.setVisible(False)
        self.gho_table_widget.act_import.setVisible(False)
        self.gho_table_widget.act_get_pos.setVisible(False)
        self.gho_table_widget.table.setFixedHeight(150)

        vlo_left = QVBoxLayout()
        self.tab_gho.hlo.addLayout(vlo_left)

        flo1 = QFormLayout()
        flo1.addRow(QLabel('ROI width and height (mm)'), self.gho_roi_size)
        flo1.addRow(QLabel('Pos x in table = distance to image border'),
                    self.gho_relative_to_right)
        vlo_left.addLayout(flo1)
        vlo_left.addWidget(self.gho_table_widget)

        self.tab_gho.hlo.addStretch()

    def create_tab_mtf(self):
        """GUI of tab MTF."""
        super().create_tab_mtf()
        self.create_tab_mtf_xray_mr()

    def create_tab_nps(self):
        """GUI of tab NPS."""
        self.create_tab_nps_xray()


class GroupBoxCorrectPointSource(QGroupBox):
    """Groupbox for correction of point source curvature."""

    def __init__(self, parent, testcode='uni',
                 chk_pos_x=None, chk_pos_y=None,
                 chk_radius=None, wid_radius=None):
        super().__init__('Correct for point source curvature')
        testcode = testcode.lower()
        self.parent = parent
        self.setCheckable(True)
        if testcode == 'sni':
            self.toggled.connect(
                lambda: parent.update_sni_display_options(attribute='sni_correct'))
        else:
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


class WidgetReferenceImage(QWidget):
    """Widget in NM SNI holding reference images."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.parent.sni_ref_image.currentIndexChanged.connect(
            lambda: self.parent.param_changed_from_gui(attribute='sni_ref_image'))
        self.ref_folder = Path(cff.get_config_folder()) / 'SNI_ref_images'
        self.update_reference_images()
        vlo_ref_image = QVBoxLayout()
        hlo_ref_image = QHBoxLayout()
        hlo_ref_image.addWidget(QLabel('Ref. image (optional):'))
        hlo_ref_image.addWidget(self.parent.sni_ref_image)
        act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add refernce image', self)
        act_add.triggered.connect(self.add_reference_image)
        act_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete reference image', self)
        act_delete.triggered.connect(self.delete_reference_image)
        act_info = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}info.png'),
            'Info about reference image', self)
        act_info.triggered.connect(self.info_reference_image)
        toolb = QToolBar()
        toolb.addActions([act_add, act_delete, act_info])
        hlo_ref_image.addWidget(toolb)
        vlo_ref_image.addLayout(hlo_ref_image)
        vlo_ref_image.addWidget(self.parent.sni_ref_image_fit)
        self.setLayout(vlo_ref_image)

    def update_reference_images(self, set_text=''):
        """Update reference images for NM SNI."""
        available_images = ['']
        if self.ref_folder.exists():
            filenames = [x.stem for x in self.ref_folder.glob('*')
                         if x.suffix == '.dcm']
            available_images.extend(filenames)
        self.parent.sni_ref_image.clear()
        self.parent.sni_ref_image.addItems(available_images)
        self.parent.sni_ref_image.setCurrentText(set_text)

    def add_reference_image(self):
        """Add reference images for NM SNI."""
        fname = QFileDialog.getOpenFileName(
            self, 'Select DICOM file to use as reference image for NM SNI test',
            filter="DICOM files (*.dcm *.IMA);;All files (*)")
        if fname[0] != '':
            src = fname[0]
            text, proceed = QInputDialog.getText(
                self, 'Rename copy?',
                'Rename the copy (how it will be listed)                      ',
                text=Path(fname[0]).stem)
            stem = text if proceed else Path(fname[0]).stem
            dest = self.ref_folder / (stem + '.dcm')
            if dest.exists():
                QMessageBox.warning(
                    self, 'Filename exist',
                    'Failed to add reference image. Set filename already exist')
            else:
                import shutil
                if not self.ref_folder.exists():
                    os.mkdir(self.ref_folder.resolve())
                shutil.copy2(src, dest.resolve())
                self.update_reference_images(set_text=stem)

    def delete_reference_image(self):
        """Delete reference images for NM SNI."""
        if self.parent.sni_ref_image.currentText() != '':
            filename = self.ref_folder / (
                self.parent.sni_ref_image.currentText() + '.dcm')
            if filename.exists():
                ref_images = [p.sni_ref_image for p in self.parent.main.paramsets]
                labels = [p.label for p in self.parent.main.paramsets]
                idxs = get_all_matches(
                    ref_images, self.parent.sni_ref_image.currentText())
                if len(idxs) > 1:
                    labels_used = [label for i, label in enumerate(labels) if i in idxs]
                    labels_used.remove(self.parent.main.current_paramset.label)
                    QMessageBox.warning(
                        self, 'Used in other templates',
                        'Failed to delete reference image. Used in other templates:\n'
                        f'{labels_used}')
                else:
                    question = (
                        f'Remove {self.wid_ref_image.currentText()}.dcm from the '
                        'list of available reference images?')
                    proceed = messageboxes.proceed_question(self, question)
                    if proceed:
                        os.remove(filename.resolve())
                        self.update_reference_images()

    def info_reference_image(self):
        """Info about reference images for NM SNI."""
        html_body_text = (
            'Estimating noise using a verified reference image.<br>'
            'Find an image acquired under the same circumstances, free of artifacts '
            'and use this as a reference '
            'to detect changes over time compared to the reference image.'
            )
        dlg = ImageQCDialog()
        dlg.setWindowTitle('Information')
        dlg.infotext = QLabel(f"""<html><head/><body>
                {html_body_text}
                </body></html>""")
        vlo = QVBoxLayout()
        vlo.addWidget(dlg.infotext)
        buttons = QDialogButtonBox.Ok
        dlg.buttonBox = QDialogButtonBox(buttons)
        dlg.buttonBox.accepted.connect(dlg.accept)
        vlo.addWidget(dlg.buttonBox)
        dlg.setLayout(vlo)
        dlg.exec()


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

        self.flag_ignore_signals = False

    def update_enabled(self):
        """Update enabled/disabled features."""
        super().update_enabled()
        if self.main.current_modality == 'NM':
            paramset = self.main.current_paramset
            if paramset.sni_type == 0:
                self.sni_roi_label.setVisible(False)
                self.sni_roi_size.setVisible(False)
                self.sni_roi_ratio.setVisible(False)
            else:
                self.sni_roi_label.setVisible(True)
                if paramset.sni_type == 1:
                    self.sni_roi_label.setText('ROI size ratio')
                    self.sni_roi_ratio.setVisible(True)
                    self.sni_roi_ratio.setEnabled(True)
                    self.sni_roi_size.setVisible(False)
                elif paramset.sni_type >= 2:
                    self.sni_roi_label.setText('ROI size')
                    self.sni_roi_size.setVisible(True)
                    self.sni_roi_size.setEnabled(True)
                    self.sni_roi_ratio.setVisible(False)
            if paramset.sni_channels:
                self.sni_channels_table_widget.setVisible(True)
                self.gb_eye_filter.setVisible(False)
                self.sni_plot_low.setVisible(True)
            else:
                self.sni_channels_table_widget.setVisible(False)
                self.gb_eye_filter.setVisible(True)
                self.sni_plot_low.setVisible(False)

    def update_sni_display_options(self, attribute=''):
        """Update plot and result image options for SNI."""
        self.sni_plot.clear()
        self.sni_result_image.clear()

        items_res_image = []
        if self.sni_type.currentIndex() == 0:
            items_plot = ['SNI values each ROI',
                          'SNI values all images',
                          'NPS all ROIs + filter',
                          'Filtered NPS and NPS structure, selected ROI']
            items_res_image = ['2d NPS for selected ROI']
        else:
            items_plot = [
                'SNI values each ROI',
                'SNI values all images',
                'Filtered NPS and NPS structure max+avg (small ROIs)',
                'Filtered NPS and NPS structure, selected ROI']
            items_res_image = ['SNI values map', '2d NPS for selected ROI']
        if self.sni_type.currentIndex() == 3:
            self.sni_roi_outside.setVisible(True)
            self.sni_roi_outside_label.setVisible(True)
        else:
            self.sni_roi_outside.setVisible(False)
            self.sni_roi_outside_label.setVisible(False)
        if self.sni_correct.isChecked() is True:
            items_res_image.append('Curvature corrected image')
            items_plot.append('Curvature correction check')
        if self.sni_sum_first.isChecked() is True:
            items_res_image.append('Summed image')
        self.sni_plot.addItems(items_plot)
        self.sni_plot.setCurrentIndex(0)
        self.sni_result_image.addItems(items_res_image)
        self.sni_result_image.setCurrentIndex(0)
        if attribute != '':
            self.param_changed_from_gui(attribute=attribute)

    def update_sni_roi_names(self):
        """Update ROI names to select based on existing ROIs."""
        self.sni_selected_roi.clear()
        if self.sni_type.currentIndex() == 0:
            items = ['L1', 'L2', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        else:
            items = ['L1', 'L2']
            try:
                for rowno, rois_row in enumerate(self.main.current_roi[3:]):
                    for colno, roi in enumerate(rois_row):
                        name = f'r{rowno}_c{colno}'
                        if roi is None:  # None if ignored
                            name = name + ' (ignored)'
                        items.append(name)
            except TypeError:
                pass
        self.sni_selected_roi.addItems(items)

    def create_tab_uni(self):
        """GUI of tab Uniformity."""
        self.tab_uni = ParamsWidget(self, run_txt='Calculate uniformity')

        self.uni_ufov_ratio = QDoubleSpinBox(
            decimals=2, minimum=0.1, maximum=1., singleStep=0.01)
        self.uni_ufov_ratio.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='uni_ufov_ratio'))
        self.uni_cfov_ratio = QDoubleSpinBox(
            decimals=2, minimum=0.1, maximum=1., singleStep=0.01)
        self.uni_cfov_ratio.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='uni_cfov_ratio'))
        self.uni_mask_corner = QDoubleSpinBox(
            decimals=1, minimum=0, maximum=50, singleStep=1)
        self.uni_mask_corner.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='uni_mask_corner'))

        self.uni_correct_pos_x = QCheckBox('x')
        self.uni_correct_pos_y = QCheckBox('y')
        self.uni_lock_radius = QCheckBox('Lock source distance to')
        self.uni_radius = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=5000, singleStep=0.1)
        self.uni_radius.valueChanged.connect(
            lambda: self.param_changed_from_gui(
                attribute='uni_radius', update_roi=False))
        self.uni_correct = GroupBoxCorrectPointSource(
            self, testcode='uni',
            chk_pos_x=self.uni_correct_pos_x, chk_pos_y=self.uni_correct_pos_y,
            chk_radius=self.uni_lock_radius, wid_radius=self.uni_radius)

        self.uni_sum_first = QCheckBox('Sum marked images before analysing sum')
        self.uni_sum_first.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='uni_sum_first'))

        self.uni_scale_factor = QComboBox()
        self.uni_scale_factor.addItems(
            ['Auto to 6.4mm/pix'] + [str(i) for i in range(1, 15)])
        self.uni_scale_factor.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='uni_scale_factor'))

        self.uni_plot = QComboBox()
        self.uni_plot.addItems(['Uniformity result for all images',
                                'Curvature correction check'])
        self.uni_plot.currentIndexChanged.connect(
            self.main.wid_res_plot.plotcanvas.plot)

        self.uni_result_image = QComboBox()
        self.uni_result_image.addItems(
            [
                'Differential uniformity map',
                'Processed image (6.4mm pix, smoothed) within UFOV',
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
        flo.addRow(QLabel('Mask corners of UFOV (mm)'), self.uni_mask_corner)
        hlo_fov.addLayout(flo)
        hlo_fov.addSpacing(100)
        vlo_left.addLayout(hlo_fov)
        vlo_left.addStretch()
        vlo_left.addWidget(self.uni_sum_first)
        hlo_scale = QHBoxLayout()
        vlo_left.addLayout(hlo_scale)
        hlo_scale.addWidget(QLabel('Scale by factor'))
        hlo_scale.addWidget(self.uni_scale_factor)
        info_txt = '''
        According to NEMA NU-1 the matrix should be scaled to 6.4mm/pix +/-30%<br>
        For Siemens gamma camera the "curvature corrected" dicom file have incorrect<br>
        pixel size defined in header (the original pixel size is not updated).<br>
        For calculating uniformity from these images you should use scale factor 3.<br>
        Original pix 0.6 mm/pix, correction matrix reduced from 1024 to 256 =
        2.4 mm/pix.<br>
        To get 6.4 +/- 30% mm/pix you could combine 2x2 or 3x3 pixels = scale factor
        2 or 3.
        '''
        hlo_scale.addWidget(uir.InfoTool(info_txt, parent=self.main))

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

        vlo_left = QVBoxLayout()
        self.tab_sni.hlo.addLayout(vlo_left)
        self.tab_sni.hlo.addWidget(uir.VLine())
        vlo_right = QVBoxLayout()
        self.tab_sni.hlo.addLayout(vlo_right)

        hlo_info = QHBoxLayout()
        vlo_left.addLayout(hlo_info)
        hlo_info.addWidget(QLabel('SNI = Structured Noise Index'))
        info_txt = '''
        Based on Nelson et al, J Nucl Med 2014; 55:169-174<br>
        SNI is a measure attempting to quantify the amount of structured noise in <br>
        the image. Noise Power Spectrum (NPS) is calculated for each ROI and the <br>
        expected quantum noise NPS is subtracted. A frequency filter is applied.<br>
        <br>
        The original suggestion by Nelson et al (2014) was to use two large and 6<br>
        small ROIs. There are different options for the small ROIs.<br>
        <br>
        For the option to correct for point source curvature, the quantum noise is <br>
        based on counts in image. For calibration images (Siemens) this is not <br>
        sufficient. For that case it is recommended to use a reference image to <br>
        estimate the expected noise. The reference image should be aquired under <br>
        the exact same conditions as the image to be analysed.
        '''
        hlo_info.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.sni_area_ratio = QDoubleSpinBox(
            decimals=2, minimum=0.1, maximum=1., singleStep=0.01)
        self.sni_area_ratio.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sni_area_ratio'))
        self.sni_ratio_dim = QComboBox()
        self.sni_ratio_dim.addItems(
            ['2d NPS', 'radial NPS profile'])
        self.sni_ratio_dim.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sni_ratio_dim'))
        self.sni_type = QComboBox()
        self.sni_type.addItems(ALTERNATIVES['NM']['SNI'])
        self.sni_type.currentIndexChanged.connect(
            lambda: self.update_sni_display_options(attribute='sni_type'))
        self.sni_roi_ratio = QDoubleSpinBox(
            decimals=2, minimum=0.05, maximum=1., singleStep=0.01)
        self.sni_roi_ratio.editingFinished.connect(
            lambda: self.param_changed_from_gui(attribute='sni_roi_ratio'))
        self.sni_roi_size = QDoubleSpinBox(
            decimals=0, minimum=16, maximum=1000, singleStep=1)
        self.sni_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sni_roi_size'))
        self.sni_roi_outside = QComboBox()
        self.sni_roi_outside.addItems(['ignore', 'move inside'])
        self.sni_roi_outside.currentIndexChanged.connect(
            lambda: self.update_sni_display_options(attribute='sni_roi_outside'))
        self.sni_roi_outside_label = QLabel('For ROIs partly outside large ROI')
        self.sni_scale_factor = QDoubleSpinBox(
            decimals=0, minimum=1, maximum=100, singleStep=1)
        self.sni_scale_factor.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sni_scale_factor'))

        self.sni_sampling_frequency = QDoubleSpinBox(
            decimals=3, minimum=0.001, singleStep=0.005)
        self.sni_sampling_frequency.valueChanged.connect(
                    lambda: self.param_changed_from_gui(
                        attribute='sni_sampling_frequency', update_roi=False,
                        clear_results=True))

        self.sni_correct_pos_x = QCheckBox('x')
        self.sni_correct_pos_y = QCheckBox('y')
        self.sni_lock_radius = QCheckBox('Lock source distance to')
        self.sni_radius = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=5000, singleStep=0.1)
        self.sni_radius.valueChanged.connect(
            lambda: self.param_changed_from_gui(
                attribute='sni_radius', update_roi=False))
        self.sni_ref_image = QComboBox()
        self.sni_ref_image_fit = QCheckBox(
            'If point source correction, use reference image to correct.')
        self.sni_ref_image_fit.toggled.connect(
            lambda: self.param_changed_from_gui(
                attribute='sni_ref_image_fit', update_roi=False))
        self.wid_ref_image = WidgetReferenceImage(self)
        self.sni_correct = GroupBoxCorrectPointSource(
            self, testcode='sni',
            chk_pos_x=self.sni_correct_pos_x, chk_pos_y=self.sni_correct_pos_y,
            chk_radius=self.sni_lock_radius, wid_radius=self.sni_radius)

        self.sni_channels = BoolSelectTests(
            self, attribute='sni_channels',
            text_true='Filter low/high channels',
            text_false='Human visual response filter',
            update_roi=False, clear_results=True, update_plot=False)

        self.gb_eye_filter = QGroupBox('Human visual respose filter')
        self.sni_eye_filter_c = QDoubleSpinBox(
            decimals=0, minimum=0, maximum=1000, singleStep=1)
        self.sni_eye_filter_c.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sni_eye_filter_c'))

        self.sni_channels_table_widget = SimpleTableWidget(
            self, 'sni_channels_table', ['Start frequency', 'Width', 'Flat top ratio'],
            row_labels=['Low', 'High'],
            min_vals=[0.0, None, 0.0], max_vals=[None, None, 1.0])

        self.sni_sum_first = QCheckBox('Sum marked images before analysing sum')
        self.sni_sum_first.toggled.connect(
            lambda: self.update_sni_display_options(attribute='sni_sum_first'))

        self.sni_plot = QComboBox()
        self.sni_result_image = QComboBox()
        self.sni_selected_roi = QComboBox()
        self.update_sni_display_options()
        self.sni_plot.currentIndexChanged.connect(
            self.main.wid_res_plot.plotcanvas.plot)
        self.sni_result_image.currentIndexChanged.connect(
            self.main.wid_res_image.canvas.result_image_draw)
        self.sni_selected_roi.addItems(['L1', 'L2'])
        self.sni_selected_roi.currentIndexChanged.connect(
            self.main.refresh_results_display)
        self.sni_show_labels = QCheckBox('Show ROI labels in image')
        self.sni_show_labels.setChecked(True)
        self.sni_show_labels.toggled.connect(
            self.main.wid_image_display.canvas.roi_draw)

        flo = QFormLayout()
        flo.addRow(QLabel('Ratio of nonzero part of image to be analysed'),
                   self.sni_area_ratio)
        flo.addRow(QLabel('Calculate SNI from ratio of integrals'),
                   self.sni_ratio_dim)
        flo.addRow(QLabel('Merge NxN pixels before analysing, N = '),
                   self.sni_scale_factor)
        vlo_left.addLayout(flo)

        hlo_type = QHBoxLayout()
        vlo_left.addLayout(hlo_type)
        hlo_type.addWidget(QLabel('Small ROIs'))
        hlo_type.addWidget(self.sni_type)
        self.sni_roi_label = QLabel('ROI size ratio')
        hlo_type.addWidget(self.sni_roi_label)
        hlo_type.addWidget(self.sni_roi_ratio)
        hlo_type.addWidget(self.sni_roi_size)
        hlo_outside = QHBoxLayout()
        vlo_left.addLayout(hlo_outside)
        hlo_outside.addSpacing(100)
        hlo_outside.addWidget(self.sni_roi_outside_label)
        hlo_outside.addWidget(self.sni_roi_outside)

        vlo_eye = QVBoxLayout()
        hlo_eye = QHBoxLayout()
        hlo_eye.addWidget(QLabel('V(r) = r<sup>1.3</sup> exp[-Cr<sup>2</sup> ]'))
        hlo_eye.addStretch()
        hlo_eye.addWidget(QLabel('C'))
        hlo_eye.addWidget(self.sni_eye_filter_c)
        vlo_eye.addLayout(hlo_eye)
        self.gb_eye_filter.setLayout(vlo_eye)

        vlo_left.addWidget(self.sni_channels)
        vlo_left.addWidget(self.gb_eye_filter)
        vlo_left.addWidget(self.sni_channels_table_widget)
        vlo_left.addWidget(self.sni_sum_first)

        vlo_right.addWidget(self.wid_ref_image)
        vlo_right.addWidget(self.sni_correct)

        self.sni_plot_low = uir.BoolSelect(
            self, text_true='low', text_false='high', text_label='Plot filter')
        self.sni_plot_low.btn_true.toggled.connect(self.main.refresh_results_display)

        f_btm = QFormLayout()
        vlo_right.addLayout(f_btm)
        f_btm.addRow(self.sni_show_labels, self.sni_plot_low)
        f_btm.addRow(QLabel('NPS sampling frequency (mm-1)'),
                     self.sni_sampling_frequency)
        f_btm.addRow(QLabel('Plot'), self.sni_plot)
        f_btm.addRow(QLabel('Result image'), self.sni_result_image)
        hlo_selected_roi = QHBoxLayout()
        vlo_right.addLayout(hlo_selected_roi)
        hlo_selected_roi.addWidget(QLabel('Selected ROI for plot/image'))
        hlo_selected_roi.addWidget(self.sni_selected_roi)

    def create_tab_mtf(self):
        """GUI of tab MTF."""
        super().create_tab_mtf()

        self.mtf_type.addItems(ALTERNATIVES['NM']['MTF'])
        self.mtf_roi_size_x = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_roi_size_x.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_roi_size_x'))
        self.mtf_roi_size_y = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.mtf_roi_size_y.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_roi_size_y'))
        self.mtf_auto_center = QCheckBox(
            'Auto center ROI on object signal (ignored if MTF method is edge)')
        self.mtf_auto_center.toggled.connect(
            lambda: self.param_changed_from_gui(attribute='mtf_auto_center'))
        self.mtf_plot.addItems(['Centered xy profiles', 'Line fit',
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
            '''Fail to update other than bar_width_4....
            this_spin.valueChanged.connect(
                lambda: self.param_changed_from_gui(attribute=f'bar_width_{i}'))
            '''
            hlo_gb.addWidget(QLabel(f'{i}:'))
            hlo_gb.addWidget(this_spin)
            hlo_gb.addSpacing(50)
        self.bar_width_1.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='bar_width_1'))
        self.bar_width_2.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='bar_width_2'))
        self.bar_width_3.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='bar_width_3'))
        self.bar_width_4.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='bar_width_4'))

        self.bar_roi_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000, singleStep=0.1)
        self.bar_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='bar_roi_size'))

        gb_bar = QGroupBox('Bar widths (mm) (1 = widest, 4 = narrowest)')
        gb_bar.setLayout(hlo_gb)
        self.tab_bar.hlo.addWidget(gb_bar)
        self.tab_bar.hlo.addStretch()

        hlo_roi = QHBoxLayout()
        hlo_roi.addWidget(QLabel('ROI radius (mm)'))
        hlo_roi.addWidget(self.bar_roi_size)
        hlo_roi.addStretch()
        self.tab_bar.vlo.addLayout(hlo_roi)


class ParamsTabSPECT(ParamsTabCommon):
    """Copy for modality tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_mtf()
        self.create_tab_rin()

        self.addTab(self.tab_mtf, "Spatial resolution")
        self.addTab(self.tab_rin, "Ring artifacts")

        self.flag_ignore_signals = False

    def create_tab_mtf(self):
        """GUI of tab MTF."""
        super().create_tab_mtf()
        self.create_tab_mtf_ct_spect_pet(modality='SPECT')


class ParamsTabPET(ParamsTabCommon):
    """Tab for PET tests."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_tab_hom()
        self.create_tab_cro()
        self.create_tab_rec()
        self.create_tab_mtf()

        self.addTab(self.tab_hom, "Homogeneity")
        self.addTab(self.tab_cro, "Cross Calibration")
        self.addTab(self.tab_rec, "Recovery Curve")
        self.addTab(self.tab_mtf, "Spatial resolution")

        self.flag_ignore_signals = False

    def create_tab_hom(self):
        """GUI of tab Homogeneity."""
        self.tab_hom = ParamsWidget(self, run_txt='Calculate homogeneity')
        self.tab_hom.vlo_top.addWidget(QLabel(
            'Calculate mean in ROIs and % difference from mean of all mean values.'))

        self.hom_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, singleStep=0.1)
        self.hom_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='hom_roi_size'))
        self.hom_roi_distance = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000, singleStep=0.1)
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
        self.cro_roi_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000, singleStep=0.1)
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
        self.cro_override_activity = QGroupBox(
            'Override injection time/activity in DICOM header')
        self.cro_override_activity.setCheckable(True)
        self.cro_override_activity.setChecked(False)
        self.cro_override_activity.setFont(uir.FontItalic())
        self.cro_override_activity.toggled.connect(self.clear_results_current_test)
        self.activity = QDoubleSpinBox(decimals=2, minimum=0., maximum=1000)
        self.activity.editingFinished.connect(self.clear_results_current_test)
        self.activity_resid = QDoubleSpinBox(decimals=2, minimum=0.)
        self.activity_resid.editingFinished.connect(self.clear_results_current_test)
        self.activity_time = QTimeEdit(self)
        self.activity_time.setDisplayFormat('hh:mm:ss')
        self.activity_time.editingFinished.connect(self.clear_results_current_test)
        self.activity_time_resid = QTimeEdit(self)
        self.activity_time_resid.setDisplayFormat('hh:mm:ss')
        self.activity_time_resid.editingFinished.connect(
            self.clear_results_current_test)

        flo0 = QFormLayout()
        flo0.addRow(QLabel('ROI radius (mm)'), self.cro_roi_size)
        flo0.addRow(QLabel('Volume of container (ml)'), self.cro_volume)
        flo0.addRow(QLabel('Current calibration factor'), self.cro_calibration_factor)
        flo0.addRow(QLabel('           '),
                    uir.LabelItalic('(NB: Current calibration factor cannot be saved)'))
        self.tab_cro.hlo.addLayout(flo0)
        self.tab_cro.hlo.addWidget(uir.VLine())

        hlo_act = QHBoxLayout()
        self.cro_override_activity.setLayout(hlo_act)
        flo1 = QFormLayout()
        flo2 = QFormLayout()
        hlo_act.addLayout(flo1)
        hlo_act.addLayout(flo2)
        flo1.addRow(QLabel('Activity (MBq)'), self.activity)
        flo2.addRow(QLabel('at'), self.activity_time)
        flo1.addRow(QLabel('Residual (MBq)'), self.activity_resid)
        flo2.addRow(QLabel('at'), self.activity_time_resid)
        self.tab_cro.hlo.addWidget(self.cro_override_activity)

        hlo_auto_select = QHBoxLayout()
        self.cro_auto_select_slices.setLayout(hlo_auto_select)
        hlo_auto_select.addWidget(
            QLabel('Use percentage of images within FWHM of z-profile of ROIs'))
        hlo_auto_select.addWidget(self.cro_percent_slices)
        hlo_auto_select.addWidget(QLabel('%'))
        self.tab_cro.vlo.addWidget(self.cro_auto_select_slices)

    def create_tab_rec(self):
        """GUI of tab Recovery Curve."""
        info_txt = '''
        Calculate Recovery curve from PET body phantom used for EARL accreditation.<br>
        Time of scan start is found from earliest acquisition time in images.<br>
        Images will be sorted by slice position during calculation and images to
        include are determined<br>
        from avarage in first background ROI and max in images (to find the hot spheres)
        .<br>
        Note that found scan start time and calculated activities at scan start can<br>
        be found in the supplement table.
        '''
        self.tab_rec = ParamsWidget(self, run_txt='Calculate recovery coefficients')
        self.rec_roi_size = QDoubleSpinBox(decimals=1, minimum=0.1, maximum=100)
        self.rec_roi_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='rec_roi_size'))
        self.rec_percent_slices = QDoubleSpinBox(decimals=0, minimum=50, maximum=100)
        self.rec_percent_slices.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='rec_percent_slices'))
        self.rec_sphere_percent = QDoubleSpinBox(decimals=0, minimum=10, maximum=100)
        self.rec_sphere_percent.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='rec_sphere_percent'))
        self.rec_table_widget = PositionWidget(
                self, self.main, table_attribute_name='rec_table',
                warn_output_add_delete=False)
        self.rec_plot = QComboBox()
        self.rec_plot.addItems(['Table values', 'z-profile slice selections'])
        self.rec_plot.currentIndexChanged.connect(
            self.main.wid_res_plot.plotcanvas.plot)
        self.rec_type = QComboBox()
        self.rec_type.addItems(ALTERNATIVES['PET']['Rec'])
        self.rec_type.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='rec_type',
                                                update_roi=False,
                                                clear_results=False))
        self.rec_earl = QComboBox()
        self.rec_earl.addItems(['None', 'EARL1', 'EARL2'])
        self.rec_earl.currentIndexChanged.connect(
            self.main.wid_res_plot.plotcanvas.plot)

        self.rec_act_sph = QDoubleSpinBox(decimals=2, minimum=0.)
        self.rec_act_sph.editingFinished.connect(self.clear_results_current_test)
        self.rec_act_sph_resid = QDoubleSpinBox(decimals=2, minimum=0.)
        self.rec_act_sph_resid.editingFinished.connect(self.clear_results_current_test)
        self.rec_vol_sph = QDoubleSpinBox(decimals=2, minimum=0., maximum=2000)
        self.rec_vol_sph.setValue(1000)
        self.rec_vol_sph.editingFinished.connect(self.clear_results_current_test)
        self.rec_act_bg = QDoubleSpinBox(decimals=2, minimum=0.)
        self.rec_act_bg.editingFinished.connect(self.clear_results_current_test)
        self.rec_act_bg_resid = QDoubleSpinBox(decimals=2, minimum=0.)
        self.rec_act_bg_resid.editingFinished.connect(self.clear_results_current_test)
        self.rec_background_volume = QDoubleSpinBox(
            decimals=0, minimum=0, maximum=20000)
        self.rec_background_volume.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='rec_background_volume',
                                                update_roi=False))
        self.rec_time_sph = QTimeEdit(self)
        self.rec_time_sph.setDisplayFormat('hh:mm:ss')
        self.rec_time_sph.editingFinished.connect(self.clear_results_current_test)
        self.rec_time_sph_resid = QTimeEdit(self)
        self.rec_time_sph_resid.setDisplayFormat('hh:mm:ss')
        self.rec_time_sph_resid.editingFinished.connect(self.clear_results_current_test)
        self.rec_time_bg = QTimeEdit(self)
        self.rec_time_bg.setDisplayFormat('hh:mm:ss')
        self.rec_time_bg.editingFinished.connect(self.clear_results_current_test)
        self.rec_time_bg_resid = QTimeEdit(self)
        self.rec_time_bg_resid.setDisplayFormat('hh:mm:ss')
        self.rec_time_bg_resid.editingFinished.connect(self.clear_results_current_test)

        vlo_left = QVBoxLayout()
        self.tab_rec.hlo.addLayout(vlo_left)
        self.tab_rec.hlo.addWidget(uir.VLine())
        vlo_right = QVBoxLayout()
        self.tab_rec.hlo.addLayout(vlo_right)

        vlo_left.addWidget(uir.LabelHeader('Stock solution spheres', 3))
        hlo_sph = QHBoxLayout()
        vlo_left.addLayout(hlo_sph)
        flo1 = QFormLayout()
        flo2 = QFormLayout()
        hlo_sph.addLayout(flo1)
        hlo_sph.addLayout(flo2)
        flo1.addRow(QLabel('Activity (MBq)'), self.rec_act_sph)
        flo2.addRow(QLabel('at'), self.rec_time_sph)
        flo1.addRow(QLabel('Residual (MBq)'), self.rec_act_sph_resid)
        flo2.addRow(QLabel('at'), self.rec_time_sph_resid)
        flo1.addRow(QLabel('Total volume (mL)'), self.rec_vol_sph)

        vlo_left.addWidget(uir.LabelHeader('Background', 3))
        hlo_bg = QHBoxLayout()
        vlo_left.addLayout(hlo_bg)
        flo1b = QFormLayout()
        flo2b = QFormLayout()
        hlo_bg.addLayout(flo1b)
        hlo_bg.addLayout(flo2b)
        flo1b.addRow(QLabel('Activity (MBq)'), self.rec_act_bg)
        flo2b.addRow(QLabel('at'), self.rec_time_bg)
        flo1b.addRow(QLabel('Residual (MBq)'), self.rec_act_bg_resid)
        flo2b.addRow(QLabel('at'), self.rec_time_bg_resid)
        flo1b.addRow(QLabel('Volume background (mL)'), self.rec_background_volume)

        hlo_avg_perc = QHBoxLayout()
        hlo_avg_perc.addWidget(QLabel('Average in sphere within threshold (%)'))
        hlo_avg_perc.addWidget(self.rec_sphere_percent)
        hlo_avg_perc.addWidget(uir.InfoTool(info_txt, parent=self.main))
        vlo_right.addLayout(hlo_avg_perc)

        vlo_right.addWidget(uir.LabelHeader('Background ROIs', 3))
        hlo_bg = QHBoxLayout()
        vlo_right.addLayout(hlo_bg)
        vlo_bg_roi = QVBoxLayout()
        vlo_bg_roi.addWidget(QLabel('ROI radius background'))
        flo_bg = QFormLayout()
        flo_bg.addRow(self.rec_roi_size, QLabel('mm'))
        vlo_bg_roi.addLayout(flo_bg)
        hlo_bg.addWidget(self.rec_table_widget)
        hlo_bg.addLayout(vlo_bg_roi)
        vlo_bg_roi.addWidget(QLabel('Use % of images within phantom'))
        flo_bg2 = QFormLayout()
        flo_bg2.addRow(self.rec_percent_slices, QLabel('%'))
        vlo_bg_roi.addLayout(flo_bg2)

        flo_plot = QFormLayout()
        flo_plot.addRow(QLabel('Table values'), self.rec_type)
        flo_plot.addRow(QLabel('Plot'), self.rec_plot)
        hlo_right = QHBoxLayout()
        vlo_right.addLayout(hlo_right)
        hlo_right.addLayout(flo_plot)
        hlo_right.addWidget(QLabel('EARL tolerances'))
        hlo_right.addWidget(self.rec_earl)

    def create_tab_mtf(self):
        """GUI of tab MTF."""
        super().create_tab_mtf()
        self.create_tab_mtf_ct_spect_pet(modality='PET')

    def get_Cro_activities(self):
        """Get override values for Cross calibration test."""
        if self.activity.value() > 0 and self.cro_override_activity.isChecked():
            if self.activity_resid.value() > 0:
                resid_time = self.activity_time_resid.time().toPyTime()
                act_time = self.activity_time.time().toPyTime()
                minutes = (
                    (resid_time.hour - act_time.hour) * 60 +
                    (resid_time.minute - act_time.minute) +
                    (resid_time.second - act_time.second) / 60
                    )
                act = self.activity.value() * np.exp(
                    -np.log(2)*minutes/HALFLIFE['F18'])
                act = act - self.activity_resid.value()
                act_time = resid_time
            else:
                act = self.activity.value()
                act_time = self.activity_time.time().toPyTime()

            rec_dict = {
                'activity_Bq': act, 'activity_time': act_time,
                }
        else:
            rec_dict = {}

        return rec_dict

    def get_Rec_activities(self):
        """Get values for Recovery curve test."""
        if all([
                self.rec_vol_sph.value() > 0,
                self.rec_act_sph.value() > 0,
                self.rec_act_bg.value() > 0
                ]):
            if self.rec_act_sph_resid.value() > 0:
                resid_time = self.rec_time_sph_resid.time().toPyTime()
                act_time = self.rec_time_sph.time().toPyTime()
                minutes = (
                    (resid_time.hour - act_time.hour) * 60 +
                    (resid_time.minute - act_time.minute) +
                    (resid_time.second - act_time.second) / 60
                    )
                act = self.rec_act_sph.value() * np.exp(
                    -np.log(2)*minutes/HALFLIFE['F18'])
                sph_act = act - self.rec_act_sph_resid.value()
                sph_time = self.rec_time_sph_resid.time().toPyTime()
            else:
                sph_act = self.rec_act_sph.value()
                sph_time = self.rec_time_sph.time().toPyTime()

            if self.rec_act_bg_resid.value() > 0:
                resid_time = self.rec_time_bg_resid.time().toPyTime()
                act_time = self.rec_time_bg.time().toPyTime()
                minutes = (
                    (resid_time.hour - act_time.hour) * 60 +
                    (resid_time.minute - act_time.minute) +
                    (resid_time.second - act_time.second) / 60
                    )
                bg = self.rec_act_bg.value() * np.exp(
                    -np.log(2)*minutes/HALFLIFE['F18'])
                bg_act = bg - self.rec_act_bg_resid.value()
                bg_time = self.rec_time_bg_resid.time().toPyTime()
            else:
                bg_act = self.rec_act_bg.value()
                bg_time = self.rec_time_bg.time().toPyTime()

            sph_act = sph_act * 1000000 / self.rec_vol_sph.value()
            bg_act = bg_act * 1000000 / self.rec_background_volume.value()
            rec_dict = {
                'sphere_Bq_ml': sph_act, 'sphere_time': sph_time,
                'background_Bq_ml': bg_act, 'background_time': bg_time,
                }
        else:
            rec_dict = {}

        return rec_dict


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

        self.flag_ignore_signals = False

    def create_tab_snr(self):
        """GUI of tab SNR."""
        self.tab_snr = ParamsWidget(self, run_txt='Calculate SNR')

        self.tab_snr.hlo_top.addWidget(uir.LabelItalic('Signal to noise ratio (SNR)'))
        info_txt = '''
        Based on NEMA MS-1 2008<br>
        Noise method 1 (subtraction image):
        (SNR = S mean / [stdev difference image) / sqrt(2)]<br>
        <br>
        Noise method 4 (noise from artifact free background):
        (SNR = S mean / [stdev background / 0.66]<br>
        <br>
        Center and size of phantom will be found from maximum x and y profiles.
        <br><br>
        Noise method 1:<br>
        Difference image is calculated from image2 - image1,
        image4 - image3 ...<br>
        If some images are marked, only marked images are considered.<br>
        If odd number of images, the last image will be ignored.<br>
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

        self.snr_type = QComboBox()
        self.snr_type.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='snr_type'))
        self.snr_type.addItems(ALTERNATIVES['MR']['SNR'])

        self.snr_background_size = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=200., singleStep=1)
        self.snr_background_size.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='snr_background_size'))

        self.snr_background_dist = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=200., singleStep=1)
        self.snr_background_dist.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='snr_background_dist'))

        flo = QFormLayout()
        flo.addRow(QLabel('ROI % of circular phantom area'), self.snr_roi_percent)
        flo.addRow(QLabel('Cut top of ROI by (mm)'), self.snr_roi_cut_top)
        flo.addRow(QLabel('SNR method'), self.snr_type)
        flo.addRow(QLabel('Background ROI width/height (mm)'),
                   self.snr_background_size)
        flo.addRow(QLabel('Background ROI distance from image border (mm)'),
                   self.snr_background_dist)

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
        flo.addRow(QLabel('ROI % of circular phantom area'), self.piu_roi_percent)
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

        self.gho_roi_central = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000, singleStep=0.1)
        self.gho_roi_central.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='gho_roi_central'))
        self.gho_optimize_center = QCheckBox('')
        self.gho_optimize_center.toggled.connect(
            lambda: self.param_changed_from_gui(
                attribute='gho_optimize_center'))
        self.gho_roi_w = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000, singleStep=0.1)
        self.gho_roi_w.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='gho_roi_w'))
        self.gho_roi_h = QDoubleSpinBox(
            decimals=1, minimum=0.1, maximum=1000, singleStep=0.1)
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
            decimals=1, minimum=0.1, maximum=300, singleStep=1)
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
        Slice thickness = tan(angle) * harmonic mean of FWHM upper and lower<br><br>
        (ACR phantom: tan(angle) = 1/10, i.e. 5.71 degrees)
        FWHM will be calculated for the averaged profile within each ROI,
        max from medianfiltered profile.<br><br>
        If optimized, center of phantom will be found from maximum profiles.
        '''

        self.tab_sli.hlo_top.addWidget(uir.InfoTool(info_txt, parent=self.main))

        self.sli_ramp_length = QDoubleSpinBox(
            decimals=1, minimum=0., maximum=200., singleStep=0.1)
        self.sli_ramp_length.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_ramp_length'))
        self.sli_average_width = QDoubleSpinBox(decimals=0, minimum=0)
        self.sli_average_width.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_average_width'))
        self.sli_dist_lower = QDoubleSpinBox(
            decimals=1, minimum=-100, singleStep=0.1)
        self.sli_dist_lower.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_dist_lower'))
        self.sli_dist_upper = QDoubleSpinBox(
            decimals=1, minimum=-100, singleStep=0.1)
        self.sli_dist_upper.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_dist_upper'))
        self.sli_optimize_center = QCheckBox('')
        self.sli_optimize_center.toggled.connect(
            lambda: self.param_changed_from_gui(
                attribute='sli_optimize_center'))
        self.sli_type = QComboBox()
        self.sli_type.addItems(ALTERNATIVES['MR']['Sli'])
        self.sli_type.currentIndexChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_type'))
        self.sli_tan_a = QDoubleSpinBox(decimals=3, minimum=0., maximum=1.)
        self.sli_tan_a.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_tan_a'))
        self.sli_sigma = QDoubleSpinBox(decimals=1, minimum=0)
        self.sli_sigma.valueChanged.connect(
            lambda: self.param_changed_from_gui(attribute='sli_sigma'))
        self.sli_background_width = QDoubleSpinBox(
            decimals=1, minimum=0.1, singleStep=0.1)
        self.sli_background_width.valueChanged.connect(
            lambda: self.param_changed_from_gui(
                attribute='sli_background_width'))

        hlo_dist = QHBoxLayout()
        hlo_dist.addWidget(QLabel(
            'Profile distance from image center upper / lower (mm)'))
        hlo_dist.addWidget(self.sli_dist_upper)
        hlo_dist.addWidget(QLabel('/'))
        hlo_dist.addWidget(self.sli_dist_lower)
        hlo_dist.addStretch()
        self.tab_sli.vlo.addLayout(hlo_dist)

        flo1 = QFormLayout()
        flo1.addRow(QLabel('Profile from'), self.sli_type)
        flo1.addRow(QLabel('Profile length (mm)'), self.sli_ramp_length)
        flo1.addRow(QLabel('Tangens of ramp/wedge angle'), self.sli_tan_a)
        flo1.addRow(QLabel('Optimize center'), self.sli_optimize_center)
        self.tab_sli.hlo.addLayout(flo1)
        self.tab_sli.hlo.addWidget(uir.VLine())
        vlo_right = QVBoxLayout()
        vlo_right.addStretch()
        self.sli_plot = QComboBox()
        self.sli_plot.addItems(['both', 'upper', 'lower'])
        self.sli_plot.currentIndexChanged.connect(self.main.refresh_results_display)
        flo2 = QFormLayout()
        flo2.addRow(QLabel('Average neighbour profiles (+/- #)'),
                    self.sli_average_width)
        flo2.addRow(QLabel('Gaussian smooth profile (sigma in pix)'), self.sli_sigma)
        flo2.addRow(QLabel('Background from profile outer (mm)'),
                    self.sli_background_width)
        flo2.addRow(QLabel('Plot profiles used to find FWHM'), self.sli_plot)
        vlo_right.addLayout(flo2)
        self.tab_sli.hlo.addLayout(vlo_right)

    def create_tab_mtf(self):
        """GUI of tab MTF."""
        super().create_tab_mtf()
        self.create_tab_mtf_xray_mr()


class ParamsTabSR(ParamsTabCommon):
    """Tab for SR DICOM header extraction."""

    def __init__(self, parent):
        super().__init__(parent, remove_roi_num=True)

        self.flag_ignore_signals = False


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


class SimpleTableWidget(QTableWidget):
    """Reusable table widget displaying a simple fixed size table of floats."""

    def __init__(self, parent, table_attribute_name,
                 column_labels, row_labels=None,
                 min_vals=None, max_vals=None):
        """Initiate SimpleTableWidget.

        Parameters
        ----------
        parent : ParamsTab_
        main : MainWindow
        table_attribute_name : str
            attribute name in main.current_paramset
        column_labels : list of str
        row_labels : list of str or None
        min_vals : list of floats or None
            minimum accepted value for each column (None if not set)
        max_vals : list of floats or None
            maximum accepted value for each column (None if not set)
        """
        super().__init__()
        self.parent = parent
        self.table_attribute_name = table_attribute_name
        self.current_table = getattr(
            self.parent.main.current_paramset, self.table_attribute_name, None)
        self.column_labels = column_labels
        self.row_labels = row_labels
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.cellChanged.connect(self.edit_current_table)
        self.setColumnCount(len(self.column_labels))
        self.setHorizontalHeaderLabels(self.column_labels)
        if self.row_labels:
            self.setRowCount(len(self.row_labels))
            self.setVerticalHeaderLabels(self.row_labels)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        if self.row_labels is None:
            self.verticalHeader().setVisible(False)
        self.resizeRowsToContents()
        self.update_table()

    def edit_current_table(self, row, col):
        """Update PositionTable when cell edited."""
        val = self.item(row, col).text()
        try:
            val = float(val)
        except ValueError:
            val = 0

        try:
            if val < self.min_vals[col]:
                val = self.min_vals[col]
        except (TypeError, IndexError):
            pass

        try:
            if val > self.max_vals[col]:
                val = self.max_vals[col]
        except (TypeError, IndexError):
            pass

        self.current_table[row][col] = val

        self.parent.flag_edit(True)
        setattr(self.parent.main.current_paramset, self.table_attribute_name,
                self.current_table)
        self.parent.clear_results_current_test()

    def update_table(self):
        """Populate table with current table."""
        if self.current_table is None:  # not initiated yet
            self.current_table = getattr(
                self.parent.main.current_paramset, self.table_attribute_name, None)
        else:
            setattr(self.parent.main.current_paramset, self.table_attribute_name,
                    self.current_table)
        if self.current_table:
            self.blockSignals(True)
            for rowidx, row in enumerate(self.current_table):
                for colidx, val in enumerate(row):
                    twi = QTableWidgetItem(str(val))
                    twi.setTextAlignment(4)
                    self.setItem(rowidx, colidx, twi)
            self.blockSignals(False)


def get_defined_output_for_columns(widget, paramset, attribute):
    """Get whether the test have an output defined in the paramset.

    Warn if columns defined for the output. Ask to proceed.
    Meant for dynamic tables.

    Parameters
    ----------
    widget : QWidget
        parent widget
    paramset : ParamSetXX
        as defined in config_classes.py
    attribute : str
        attribute name

    Returns
    -------
    proceed : bool
    """
    proceed = True
    log = []
    for key, sublist in paramset.output.tests.items():
        if key.lower() == attribute[:3]:
            for subno, sub in enumerate(sublist):
                if len(sub.columns) > 0:
                    log.append(str(sub))

    if len(log) > 0:
        question = ('Found output settings for this test with defined columns. '
                    'Changing columns may affect the output. Proceed?')
        proceed = messageboxes.proceed_question(widget, question, detailed_text=log)

    return proceed


class PositionWidget(QWidget):
    """Reusable table widget to hold user defined roi positions."""

    def __init__(self, parent, main, table_attribute_name='', headers=None,
                 use_rectangle=False, warn_output_add_delete=True):
        """Initialize PositionWidget.

        Parameters
        ----------
        parent : widget
            test widget containing this widget and param_changed
        main : MainWindow
        table_attribute_name : str
            attribute name of PositionTable or variants of this
        headers : list of str
            if None ['Label', 'pos x [mm]', 'pos y [mm]']
        use_rectangle : bool
            type of getting/using positions/coordinates
        """
        super().__init__()
        self.get_position_tooltips = [
            'Get position from last mouseclick in image',
            'Set selected ROI = marked area']
        self.get_position_icons = ['selectArrow', 'rectangle_select']
        self.use_rectangle = use_rectangle
        self.warn_output_add_delete = warn_output_add_delete
        self.parent = parent
        self.main = main
        if headers is None:
            self.headers = ['ROI label', 'pos x [mm]', 'pos y [mm]']
        else:
            self.headers = headers
        self.table = PositionTableWidget(
            self.parent, self.main,
            table_attribute_name, headers=copy.deepcopy(self.headers))
        self.ncols = len(self.table.headers)
        self.test_name = table_attribute_name[:3]

        hlo = QHBoxLayout()
        self.setLayout(hlo)

        self.act_import = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}import.png'),
            'Import table from clipboard or predefined tables', self)
        self.act_import.triggered.connect(self.import_table)
        self.act_copy = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}copy.png'),
            'Copy table to clipboard', self)
        self.act_copy.triggered.connect(self.copy_table)
        self.act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add row', self)
        self.act_add.triggered.connect(self.add_row)
        self.act_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete row', self)
        self.act_delete.triggered.connect(self.delete_row)
        self.act_get_pos = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}selectArrow.png'),
            self.get_position_tooltips[0], self)
        toolb = QToolBar()
        toolb.addActions([self.act_import, self.act_copy, self.act_add,
                          self.act_delete, self.act_get_pos])
        toolb.setOrientation(Qt.Vertical)
        hlo.addWidget(toolb)
        hlo.addWidget(self.table)
        self.update_on_rectangle_change(silent=True)

    def update_on_rectangle_change(self, silent=False):
        """Make changes to table and widget when setting selection changes."""
        if self.use_rectangle:
            self.main.wid_image_display.tool_rectangle.setChecked(True)
            self.act_get_pos.setToolTip(self.get_position_tooltips[1])
            self.act_get_pos.disconnect()
            self.act_get_pos.triggered.connect(self.get_active_rectangle)
            self.act_get_pos.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}{self.get_position_icons[1]}.png'))
            self.table.headers = ['ROI label', '(x1,x2)', '(y1,y2)']
        else:
            self.main.wid_image_display.tool_rectangle.setChecked(False)
            self.act_get_pos.setToolTip(self.get_position_tooltips[0])
            self.act_get_pos.disconnect()
            self.act_get_pos.triggered.connect(self.get_pos_mouse)
            self.act_get_pos.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}{self.get_position_icons[0]}.png'))
            self.table.headers = self.headers
        if self.table.current_table is not None and silent is False:
            if len(self.table.current_table.labels) > 0:
                reply = QMessageBox.question(
                    self, 'Copy current table to clipboard?',
                    'Table will be reset. '
                    'Copy content to clipboard to save current values?',
                    QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.copy_table()
                self.table.current_table = cfc.PositionTable()
                self.table.update_table()
                self.main.update_roi()
            else:
                self.table.update_table()

    def import_table(self):
        """Import contents to table from clipboard or from predefined."""
        proceed = True
        if self.warn_output_add_delete:
            proceed = get_defined_output_for_columns(
                self, self.main.current_paramset, self.table.table_attribute_name)
        if proceed:
            dlg = messageboxes.QuestionBox(
                parent=self.main, title='Import table',
                msg='Import table from...',
                yes_text='Clipboard',
                no_text='Predefined tables')
            proceed = dlg.exec()
        input_table = None
        if proceed:  # clipboard
            try:
                dataf = pd.read_clipboard()
                _, ncols = dataf.shape
                if ncols != self.ncols:
                    pass  # TODO ask for separator / decimal or guess?
                    if ncols > self.ncols:
                        input_table = self.validate_input_dataframe(
                            dataf.T[0:self.ncols].T)  # keep only first ncols of input
                    else:
                        errmsg = [
                            'Failed reading table from clipboard.',
                            f'Expected {self.ncols} columns of data that are',
                            'separated by tabs or as copied from Excel.'
                            ]
                        dlg = messageboxes.MessageBoxWithDetails(
                            self, title='Failed reading table',
                            msg='Failed reading table. See details.',
                            details=errmsg, icon=QMessageBox.Warning)
                        dlg.exec()
                else:
                    input_table = self.validate_input_dataframe(dataf)
            except pd.errors.ParserError as err:
                QMessageBox.warning(
                    self, 'Validation failed',
                    f'Trouble validating input table: {err}')
        else:  # predefined tables
            if self.test_name == 'num' and self.main.current_modality == 'NM':
                table_dict = cff.load_default_pos_tables(filename='Siemens_AutoQC')
                if len(table_dict) > 0:
                    labels = [*table_dict]
                    label, ok = QInputDialog.getItem(
                        self.main, "Select predefined table",
                        "Predefined tables:", labels, 0, False)
                    if ok and label:
                        input_table = table_dict[label]
            elif self.test_name == 'ttf':
                table_dict = cff.load_default_ct_number_tables()
                if len(table_dict) > 0:
                    labels = [*table_dict]
                    label, ok = QInputDialog.getItem(
                        self.main, "Select predefined table",
                        "Predefined tables:                     ", labels, 0, False)
                    if ok and label:
                        hu_tab = table_dict[label]
                        input_table = cfc.PositionTable(
                            labels=hu_tab.labels,
                            pos_x=hu_tab.pos_x, pos_y=hu_tab.pos_y)
            else:
                QMessageBox.information(
                    self, 'Missing predefined table',
                    'Sorry - no set of predefined tables exist yet.')

        if input_table is not None:
            self.table.current_table = input_table
            self.parent.flag_edit(True)
            self.table.update_table()

    def validate_input_dataframe(self, input_df):
        """Convert the input pandas dataframe to the current_table format."""
        def str_2_tuple(pos_string):
            pos_tuple = (0, 0)
            if len(pos_string) > 2:
                start_end = pos_string[1:-1].split(', ')
                pos_tuple = (int(start_end[0]), int(start_end[1]))
            return pos_tuple

        nrows, ncols = input_df.shape
        if nrows > 0:
            table = cfc.PositionTable()
            table.labels = [str(input_df.iat[row, 0]) for row in range(nrows)]
            try:
                if self.use_rectangle:  # pos is tuple
                    table.pos_x = [
                        str_2_tuple(input_df.iat[row, 1]) for row in range(nrows)]
                    table.pos_y = [
                        str_2_tuple(input_df.iat[row, 2]) for row in range(nrows)]
                else:
                    table.pos_x = [float(input_df.iat[row, 1]) for row in range(nrows)]
                    table.pos_y = [float(input_df.iat[row, 2]) for row in range(nrows)]
            except (ValueError, IndexError) as err:
                table = None
                QMessageBox.warning(
                    self, 'Validation failed',
                    f'Trouble validating input table: {err}')

        return table

    def copy_table(self):
        """Copy contents of table to clipboard."""
        dict_2_pd = {
            'labels': self.table.current_table.labels,
            'pos_x': self.table.current_table.pos_x,
            'pos_y': self.table.current_table.pos_y
            }
        dataf = pd.DataFrame(dict_2_pd)
        dataf.to_clipboard(index=False)
        self.main.status_bar.showMessage('Values in clipboard', 2000)

    def add_row(self):
        """Add row to table."""
        rowno = 0
        if self.table.current_table is None:
            self.table.current_table = cfc.PositionTable()
        else:
            sel = self.table.selectedIndexes()
            if len(sel) > 0:
                rowno = sel[0].row()
            else:
                rowno = self.table.rowCount()
        if self.use_rectangle:
            if self.main.active_img is not None:
                self.get_active_rectangle()
        else:
            if self.main.active_img is not None:
                self.get_pos_mouse()
            else:
                proceed = True
                if self.warn_output_add_delete:
                    proceed = get_defined_output_for_columns(
                        self, self.main.current_paramset,
                        self.table.table_attribute_name)
                if proceed:
                    self.table.current_table.add_pos(
                        label=f'ROI {rowno}', index=rowno, pos_x=0, pos_y=0)
                    self.parent.flag_edit(True)
                    self.table.update_table()

    def delete_row(self):
        """Delete row from table."""
        sel = self.table.selectedIndexes()
        if len(sel) > 0:
            proceed = True
            if self.warn_output_add_delete:
                proceed = get_defined_output_for_columns(
                    self, self.main.current_paramset, self.table.table_attribute_name)
            if proceed:
                rowno = sel[0].row()
                self.table.current_table.delete_pos(rowno)
                self.parent.flag_edit(True)
                self.table.update_table()

    def get_pos_mouse(self):
        """Get position from last mouseclick in i mage."""
        sel = self.table.selectedIndexes()
        add_row = False
        if len(sel) == 0:
            add_row = True
            rowno = self.table.rowCount()
        else:
            rowno = sel[0].row()
        if self.main.active_img is not None:
            sz_acty, sz_actx = np.shape(self.main.active_img)
            image_info = self.main.imgs[self.main.gui.active_img_no]
            in_mm = True  # used to be a choise left here if later as choise
            dx = self.main.gui.last_clicked_pos[0] - 0.5 * sz_actx
            dy = self.main.gui.last_clicked_pos[1] - 0.5 * sz_acty
            if in_mm:
                dx = round(dx * image_info.pix[0], 1)
                dy = round(dy * image_info.pix[1], 1)
            else:
                dx = int(dx)
                dy = int(dy)
            if add_row is False:
                self.table.current_table.pos_x[rowno] = dx
                self.table.current_table.pos_y[rowno] = dy
            else:
                proceed = True
                if self.warn_output_add_delete:
                    proceed = get_defined_output_for_columns(
                        self, self.main.current_paramset,
                        self.table.table_attribute_name)
                if proceed:
                    self.table.current_table.add_pos(
                        label=f'ROI {rowno}', pos_x=dx, pos_y=dy)
            self.parent.flag_edit(True)
            self.table.update_table()

    def get_active_rectangle(self):
        """Get coordinates af active rectangle in image."""
        if self.main.gui.delta_x != 0 or self.main.gui.delta_y != 0:
            QMessageBox.information(
                self, 'Offset is reset',
                'You have set an offset, this offset will be reset to prevent '
                'unexpected results defining these ROIs.')
            self.main.wid_center.reset_delta()
        sel = self.table.selectedIndexes()
        add_row = False
        if len(sel) == 0:
            add_row = True
            rowno = self.table.rowCount()
        else:
            rowno = sel[0].row()
        if self.main.active_img is not None:
            sz_y, sz_x = np.shape(self.main.active_img)
            x_tuple = (0, sz_x)
            y_tuple = (0, sz_y)
            for patch in self.main.wid_image_display.canvas.ax.patches:
                if patch.get_gid() == 'rectangle':
                    [x0, y0], [x1, y1] = patch.get_bbox().get_points()
                    x_tuple = (int(min([x0, x1])) + 1, int(max([x0, x1])) + 1)
                    y_tuple = (int(min([y0, y1])) + 1, int(max([y0, y1])) + 1)
            proceed = True
            if x_tuple == (0, sz_x) and y_tuple == (0, sz_y):
                question = (
                    'Did you mark the region of interest? '
                    'Proceed setting full image as ROI?')
                proceed = messageboxes.proceed_question(self, question)
            if proceed:
                if add_row is False:
                    self.table.current_table.pos_x[rowno] = x_tuple
                    self.table.current_table.pos_y[rowno] = y_tuple
                else:
                    proceed = True
                    if self.warn_output_add_delete:
                        proceed = get_defined_output_for_columns(
                            self, self.main.current_paramset,
                            self.table.table_attribute_name)
                    if proceed:
                        self.table.current_table.add_pos(
                            label=f'ROI {rowno}', pos_x=x_tuple, pos_y=y_tuple)
                self.parent.flag_edit(True)
                self.table.update_table()


class PositionTableWidget(QTableWidget):
    """Reusable table widget displaying positions ++."""

    def __init__(self, parent, main, table_attribute_name, headers):
        """Initiate PositionTableWidget.

        Parameters
        ----------
        parent : same parent as PositionWidget
        main : MainWindow
        table_attribute_name : str
            attribute name in main.current_paramset
        headers : list of str
            holding headers for each tuple parameter
        """
        super().__init__()
        self.main = main
        self.parent = parent
        self.table_attribute_name = table_attribute_name
        self.current_table = getattr(
            self.main.current_paramset, self.table_attribute_name, None)
        self.headers = headers
        self.cellChanged.connect(self.edit_current_table)
        if 'num' in table_attribute_name:
            self.cellClicked.connect(self.main.update_roi)

    def edit_current_table(self, row, col):
        """Update PositionTable when cell edited."""
        val = self.item(row, col).text()

        def str_2_tuple(pos_string):
            pos_tuple = (0, 10)
            if len(pos_string) > 2:
                start_end = pos_string[1:-1].split(', ')
                pos_tuple = (int(start_end[0]), int(start_end[1]))
            return pos_tuple

        if col > 0:
            if isinstance(self.current_table.pos_x[0], (tuple, list)):
                try:
                    val = str_2_tuple(val)
                except ValueError:
                    val = (0, 10)
            else:
                try:
                    val = float(val)
                except ValueError:
                    val = 0
        if col == 0:
            self.current_table.labels[row] = val
        elif col == 1:
            self.current_table.pos_x[row] = val
        elif col == 2:
            self.current_table.pos_y[row] = val
        self.parent.flag_edit(True)
        setattr(self.main.current_paramset, self.table_attribute_name,
                self.current_table)
        self.parent.main.update_roi(clear_results_test=True)
        if col == 0 and self.table_attribute_name == 'ttf_table':
            self.parent.update_ttf_plot_options()

    def update_table(self):
        """Populate table with current table."""
        if self.current_table is None:  # not initiated yet
            self.current_table = getattr(
                self.main.current_paramset, self.table_attribute_name, None)
        else:
            setattr(self.main.current_paramset, self.table_attribute_name,
                    self.current_table)
        self.blockSignals(True)
        self.clear()
        self.setColumnCount(len(self.headers))
        self.setHorizontalHeaderLabels(self.headers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        values = None
        n_rows = 0
        try:
            n_rows = len(self.current_table.labels)
            values = self.get_prepared_fill_values()
        except AttributeError:
            pass
        self.setRowCount(n_rows)

        if values is not None:
            for col in range(len(self.headers)):
                this_col = values[col]
                for row in range(n_rows):
                    twi = QTableWidgetItem(str(this_col[row]))
                    if col > 0:
                        twi.setTextAlignment(4)
                    self.setItem(row, col, twi)

            self.verticalHeader().setVisible(False)
            self.resizeRowsToContents()

        self.blockSignals(False)
        if self.table_attribute_name == 'ttf_table':
            self.parent.update_ttf_plot_options()

        self.parent.main.update_roi(clear_results_test=True)

    def get_prepared_fill_values(self):
        """Fill table with values in self.current_table."""
        values = [
            self.current_table.labels,
            self.current_table.pos_x,
            self.current_table.pos_y
            ]
        return values


class CTnTableWidget(QWidget):  # TODO PositionWidget
    """CT numbers table widget."""

    def __init__(self, parent, main):
        super().__init__()
        self.parent = parent
        self.main = main

        vlo = QVBoxLayout()
        self.setLayout(vlo)
        hlo = QHBoxLayout()
        vlo.addLayout(hlo)

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

        self.ctn_edit_linearity_unit = QPushButton('Rename linearity column')
        self.ctn_edit_linearity_unit.clicked.connect(self.rename_linearity_unit)
        vlo.addWidget(self.ctn_edit_linearity_unit)

    def import_table(self):
        """Import contents to table from clipboard or from predefined."""
        proceed = get_defined_output_for_columns(
            self, self.main.current_paramset, 'ctn_table')
        if proceed:
            dlg = messageboxes.QuestionBox(
                parent=self.main, title='Import table',
                msg='Import table from...',
                yes_text='Clipboard',
                no_text='Predefined tables')
            proceed = dlg.exec()
        ctn_table = None
        if proceed:
            dataf = pd.read_clipboard()
            nrows, ncols = dataf.shape
            if ncols != 6:
                pass  # TODO ask for separator / decimal or guess?
                errmsg = [
                    'Failed reading table from clipboard.',
                    'Expected 6 columns of data that are',
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
                ctn_table.labels = [
                    str(dataf.iat[row, 0]) for row in range(nrows)]
                ctn_table.pos_x = [
                    float(dataf.iat[row, 1]) for row in range(nrows)]
                ctn_table.pos_y = [
                    float(dataf.iat[row, 2]) for row in range(nrows)]
                ctn_table.min_HU = [
                    float(dataf.iat[row, 3]) for row in range(nrows)]
                ctn_table.max_HU = [
                    float(dataf.iat[row, 4]) for row in range(nrows)]
                ctn_table.linearity_axis = [
                    float(dataf.iat[row, 5]) for row in range(nrows)]
        else:
            table_dict = cff.load_default_ct_number_tables()
            if len(table_dict) > 0:
                labels = [*table_dict]
                label, ok = QInputDialog.getItem(
                    self.main, "Select predefined table",
                    "Predefined tables:                     ", labels, 0, False)
                if ok and label:
                    ctn_table = table_dict[label]

        if ctn_table is not None:
            self.main.current_paramset.ctn_table = ctn_table
            self.parent.flag_edit(True)
            self.table.update_table()

    def copy_table(self):
        """Copy conptents of table to clipboard."""
        dict_2_pd = {
            'labels': self.main.current_paramset.ctn_table.labels,
            'pos_x': self.main.current_paramset.ctn_table.pos_x,
            'pos_y': self.main.current_paramset.ctn_table.pos_y,
            'min_HU': self.main.current_paramset.ctn_table.min_HU,
            'max_HU': self.main.current_paramset.ctn_table.max_HU,
            self.main.current_paramset.ctn_table.linearity_unit:
                self.main.current_paramset.ctn_table.linearity_axis
            }
        proceed = True
        try:
            dataf = pd.DataFrame(dict_2_pd)
        except ValueError:  # as err:
            # might happen if not all same length
            self.main.current_paramset.ctn_table.fix_list_lengths()
            dict_2_pd = {
                'labels': self.main.current_paramset.ctn_table.labels,
                'pos_x': self.main.current_paramset.ctn_table.pos_x,
                'pos_y': self.main.current_paramset.ctn_table.pos_y,
                'min_HU': self.main.current_paramset.ctn_table.min_HU,
                'max_HU': self.main.current_paramset.ctn_table.max_HU,
                self.main.current_paramset.ctn_table.linearity_unit:
                    self.main.current_paramset.ctn_table.linearity_axis
                }
            dataf = pd.DataFrame(dict_2_pd)
            '''
            proceed = False
            QMessageBox.warning(
                self, 'Copy failed',
                f'Trouble copying table: {err}')
            '''

        if proceed:
            dataf.to_clipboard(index=False)
            self.main.status_bar.showMessage('Values in clipboard', 2000)

    def add_row(self):
        """Add row to table."""
        sel = self.table.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
        else:
            rowno = self.table.rowCount()
        proceed = get_defined_output_for_columns(
            self, self.main.current_paramset, 'ctn_table')
        if proceed:
            self.main.current_paramset.ctn_table.add_pos(label='undefined', index=rowno)
            self.parent.flag_edit(True)
            self.table.update_table()

    def delete_row(self):
        """Delete row from table."""
        sel = self.table.selectedIndexes()
        if len(sel) > 0:
            proceed = get_defined_output_for_columns(
                self, self.main.current_paramset, 'ctn_table')
            if proceed:
                rowno = sel[0].row()
                self.main.current_paramset.ctn_table.delete_pos(rowno)
                self.parent.flag_edit(True)
                self.table.update_table()

    def rename_linearity_unit(self):
        """Rename column for linearity check."""
        text, proceed = QInputDialog.getText(
            self, 'Rename linearity column',
            'Rename column used for linearity control against measured HU.',
            text=self.main.current_paramset.ctn_table.linearity_unit)
        if proceed:
            if text == '':
                '''
                QMessageBox.warning(
                    self, 'Rename ignored',
                    'Name cannot be empty string. Ignored.')
                '''
                pass
            else:
                self.main.current_paramset.ctn_table.linearity_unit = text
                self.parent.flag_edit(True)
                self.table.update_table()

    def get_pos_mouse(self):
        """Get position from last mouseclick in image."""
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
        n_rows = len(self.main.current_paramset.ctn_table.labels)
        if col > 0:
            try:
                val = float(val)
            except ValueError:
                if col > 2:
                    val = None
                else:
                    val = 0
        if col == 0:
            self.main.current_paramset.ctn_table.labels[row] = val
        elif col == 1:
            self.main.current_paramset.ctn_table.pos_x[row] = val
        elif col == 2:
            self.main.current_paramset.ctn_table.pos_y[row] = val
        elif col == 3:
            try:
                self.main.current_paramset.ctn_table.min_HU[row] = val
            except IndexError:
                self.main.current_paramset.ctn_table.min_HU = [0 for i in range(n_rows)]
                self.main.current_paramset.ctn_table.min_HU[row] = val
        elif col == 4:
            try:
                self.main.current_paramset.ctn_table.max_HU[row] = val
            except IndexError:
                self.main.current_paramset.ctn_table.max_HU = [0 for i in range(n_rows)]
                self.main.current_paramset.ctn_table.max_HU[row] = val
        elif col == 5:
            self.main.current_paramset.ctn_table.linearity_axis[row] = val
        self.parent.flag_edit(True)
        self.parent.main.update_roi(clear_results_test=True)

    def update_table(self):
        """Populate table with current HUnumberTable."""
        self.blockSignals(True)
        self.clear()
        self.setColumnCount(6)
        n_rows = len(self.main.current_paramset.ctn_table.labels)
        self.setRowCount(n_rows)
        self.setHorizontalHeaderLabels(
            ['Material', 'x pos (mm)', 'y pos (mm)',
             'Min HU', 'Max HU', self.main.current_paramset.ctn_table.linearity_unit])
        self.verticalHeader().setVisible(False)

        if len(self.main.current_paramset.ctn_table.min_HU) == 0:
            self.main.current_paramset.ctn_table.min_HU = [0 for i in range(n_rows)]
            self.main.current_paramset.ctn_table.max_HU = [0 for i in range(n_rows)]

        values = [
            self.main.current_paramset.ctn_table.labels,
            self.main.current_paramset.ctn_table.pos_x,
            self.main.current_paramset.ctn_table.pos_y,
            self.main.current_paramset.ctn_table.min_HU,
            self.main.current_paramset.ctn_table.max_HU,
            self.main.current_paramset.ctn_table.linearity_axis]

        for col in range(6):
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
