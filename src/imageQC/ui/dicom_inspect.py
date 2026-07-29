#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for RDSR inspection.

@author: Ellen Wasbo
"""
import os
import copy

import pandas
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QAction, QBrush, QColor
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QToolBar,
    QLabel, QLineEdit, QListWidget, QTextEdit, QSpinBox, QCheckBox,
    QPushButton, QMessageBox, QDialogButtonBox, QFileDialog
    )

# imageQC block start
from imageQC.config.iQCconstants import ENV_ICON_PATH
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.scripts import dcm
from imageQC.scripts.mini_methods import get_all_matches
# imageQC block end


class DicomInspectDialog(ImageQCDialog):
    """Dialog to explore content of Dicom file."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Read DICOM header')
        self.sample_filepath = QLineEdit()
        self.sample_filepath.textChanged.connect(self.read_file)
        self.btn_parse_sr = QPushButton('Parse ContentSequence of SR file')
        self.btn_parse_sr.setEnabled(False)
        self.btn_parse_sr.clicked.connect(self.parse_sr)
        self.list_tags = QListWidget()
        self.list_tags.itemClicked.connect(self.attribute_selected)
        self.lbl_tag_string = QLabel()
        self.list_sequences = QListWidget()
        self.spin_item = QSpinBox()
        self.spin_item.setEnabled(False)
        self.spin_item.setRange(1, 100)
        self.spin_item.setValue(1)
        self.spin_item.valueChanged.connect(self.update_item)
        self.lbl_n_items = QLabel()
        self.txt_content = QTextEdit('', self)
        self.chk_sort_name = QCheckBox('Sort attributes by name/keyword')
        self.chk_sort_name.toggled.connect(self.read_tags)
        self.chk_full_seq = QCheckBox('Display full sequence')
        self.chk_full_seq.setEnabled(False)
        self.chk_full_seq.toggled.connect(self.full_sequence_selected)

        self.sample_sequences = ['']  # selected sequence(s)
        self.sample_attribute_names = []  # attribute names (in selected sequence)
        self.sample_tags = []  # tags corresponding to attr_names above pydicom.tag.BaseTag
        self.pydict = None  # currently loaded pydicom.dataset.FileDataset
        self.current_dataset = None  # currently selected DataSet

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        vlo.addWidget(QLabel('Load DICOM file to fill the list of tags'))
        hlo_file = QHBoxLayout()
        vlo.addLayout(hlo_file)
        self.sample_filepath.setMinimumWidth(500)
        hlo_file.addWidget(self.sample_filepath)
        toolb = uir.ToolBarBrowse('Browse to select file')
        toolb.act_browse.triggered.connect(self.locate_file)
        act_dcm_dump = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}tags.png'),
            "View DICOM dump", self)
        act_dcm_dump.triggered.connect(self.dump_dicom)
        toolb.addAction(act_dcm_dump)
        hlo_file.addWidget(toolb)
        hlo_btns = QHBoxLayout()
        vlo.addLayout(hlo_btns)
        hlo_btns.addWidget(self.btn_parse_sr)
        hlo_btns.addStretch()
        vlo.addWidget(self.chk_sort_name)

        hlo_taglist = QHBoxLayout()
        vlo.addLayout(hlo_taglist)
        hlo_taglist.addWidget(
            QLabel('Tags/attributes from file: '))
        hlo_taglist.addWidget(self.list_tags)
        self.list_tags.setMinimumWidth(300)
        self.act_level_up = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
            "Get back to level above current sequence", self)
        self.act_level_up.triggered.connect(self.level_up)
        tb_sequence = QToolBar()
        tb_sequence.addAction(self.act_level_up)
        hlo_taglist.addWidget(tb_sequence)
        vlo.addWidget(uir.LabelItalic(
            'Click tags named with Sequence (blue) to list the elements '
            'within the sequence.'))
        vlo.addWidget(uir.HLine())

        vlo.addSpacing(20)
        hlo = QHBoxLayout()
        vlo.addLayout(hlo)
        f_lo = QFormLayout()
        hlo.addLayout(f_lo)
        self.list_sequences.setMaximumHeight(100)
        self.list_sequences.setEnabled(False)
        f_lo.addRow(QLabel('Selected sequence(s): '), self.list_sequences)
        f_lo.addRow(QLabel('Selected tag: '), self.lbl_tag_string)

        hlo_seq = QHBoxLayout()
        gbo_seq = QGroupBox('If sequence selected')
        gbo_seq.setLayout(hlo_seq)
        vlo.addWidget(gbo_seq)
        hlo_seq.addWidget(self.chk_full_seq)
        hlo_seq.addWidget(QLabel('Select item number: '))
        hlo_seq.addWidget(self.spin_item)
        hlo_seq.addWidget(QLabel('Total items: '))
        hlo_seq.addWidget(self.lbl_n_items)
    
        vlo.addWidget(QLabel('Content from file: '))
        self.txt_content.setReadOnly(True)
        self.txt_content.createStandardContextMenu()
        self.txt_content.setMinimumHeight(200)
        vlo.addWidget(self.txt_content)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close,
            Qt.Orientation.Horizontal, self)
        buttons.rejected.connect(self.reject)
        vlo.addWidget(buttons)

    def locate_file(self):
        """Locate sample DICOM file."""
        self.reset_selected_tag()
        fname = QFileDialog.getOpenFileName(
            self, 'Read DICOM file',
            filter="DICOM file (*.dcm *.IMA);;All files (*)")
        if fname[0] != '':
            self.sample_sequences = ['']
            self.sample_filepath.setText(fname[0])
            self.read_file()
        else:
            self.pydict = None
            self.current_dataset = None

    def read_file(self):
        """Read selected dicom file."""
        filename = self.sample_filepath.text()
        self.pydict = None
        self.reset_selected_tag()
        if filename != '':
            pyd, _, errmsg = dcm.read_dcm(filename)
            if pyd:
                self.pydict = pyd
                self.read_tags()
            else:
                if errmsg:
                    QMessageBox.information(
                        self, 'Failed reading DICOM', errmsg)
                self.sample_attribute_names = []
                self.sample_tags = []
        
        self.btn_parse_sr.setEnabled(False)
        if self.pydict:
            try:
                if self.pydict.Modality == 'SR':
                    self.btn_parse_sr.setEnabled(True)
            except AttributeError:
                pass

    def read_tags(self):
        """Fill list of tags."""
        self.sample_attribute_names = []
        self.sample_tags = []
        self.current_dataset = None

        if self.pydict:
            ds = copy.deepcopy(self.pydict)
            ds_seq = None
            item = None
            
            if self.sample_sequences[0] != '':
                for keyw_item in self.sample_sequences:
                    spl_str = keyw_item.split(' ')
                    keyword = spl_str[0]
                    item = None
                    if len(spl_str) > 1:
                        item = int(spl_str[1].split('/')[0]) - 1 # item/total, items start with 1
                    ds_seq = ds[keyword]
                    if len(ds_seq._value) == 1:
                        item = 0  # autoselect first item if only one
                    if item is not None:
                        try:
                            ds = ds[keyword][item]
                        except IndexError:
                            pass
            
            proceed_list_attr = True
            self.spin_item.blockSignals(True)
            self.chk_full_seq.blockSignals(True)
            if ds_seq:
                n_items = len(ds_seq._value)
                self.lbl_n_items.setText(str(n_items))
                self.spin_item.setRange(1, n_items)
                if item is None:
                    proceed_list_attr = False  # content full seq
                    self.spin_item.setEnabled(False)
                    self.spin_item.setValue(1)
                    self.chk_full_seq.setChecked(True)
                    self.chk_full_seq.setEnabled(True)
                    self.current_dataset = ds_seq
                    self.update_content('')
                else:
                    self.chk_full_seq.setChecked(False)
                    self.chk_full_seq.setEnabled(True)
                    self.spin_item.setValue(item + 1)
                    self.spin_item.setEnabled(True)
                    self.current_dataset = ds
            else:
                self.lbl_n_items.setText('')
                self.spin_item.setEnabled(False)
                self.chk_full_seq.setEnabled(False)
                self.chk_full_seq.setChecked(False)
                self.current_dataset = ds
            self.spin_item.blockSignals(False)
            self.chk_full_seq.blockSignals(False)

            if proceed_list_attr:
                self.sample_attribute_names = [elem.keyword for elem in ds if elem.tag.elem != 0]
                self.sample_tags = [elem.tag for elem in ds if elem.tag.elem != 0]
                no_keyword = get_all_matches(self.sample_attribute_names, '')  # Private tags / missing keyword
                for idx in no_keyword:
                    self.sample_attribute_names[idx] = f'{ds[self.sample_tags[idx]].name.replace(' ', '')} {str(self.sample_tags[idx])}'
            else:
                # content is full sequence
                self.list_tags.clear()
                self.list_tags.addItems(['Deselect "Display full sequence" to see content of specific item'])

        self.list_tags.blockSignals(True)
        self.list_tags.clear()
        if len(self.sample_attribute_names) > 0:
            if self.chk_sort_name.isChecked():  # sort by name
                dataf = pandas.DataFrame(
                    {'names': self.sample_attribute_names,
                     'idxs': [idx for idx in range(len(self.sample_tags))]})
                sorted_dataf = dataf.sort_values(by=['names'])
                self.sample_attribute_names = list(sorted_dataf['names'])
                sorted_sample_tags = [self.sample_tags[idx] for idx in sorted_dataf['idxs']]
                self.sample_tags = sorted_sample_tags

            self.list_tags.addItems(self.sample_attribute_names)
            brush = QBrush(QColor(110, 148, 192))
            for attrno, tag in enumerate(self.sample_tags):
                if ds[tag].VR == 'SQ':
                    self.list_tags.item(attrno).setBackground(brush)
            self.update_content('')
            self.list_tags.blockSignals(False)
        else:
            if self.pydict:
                self.list_tags.addItems(['Deselect "Display full sequence" to see content of specific item'])

    def level_up(self):
        """Go back to above sequence or deselect attribute"""
        prev_selection = ''
        sel = self.list_tags.selectedIndexes()
        attribute_selected = False if len(sel) == 0 else True
        if attribute_selected:
            rowno = sel[0].row()
            prev_selection = self.sample_attribute_names[rowno]
        else:
            prev_selection = self.sample_sequences[-1].split(' ')[0]
            if len(self.sample_sequences) == 1:
                self.sample_sequences = ['']
            else:
                self.sample_sequences.pop()
        self.reset_selected_tag()
        self.read_tags()
        self.set_sequences()
        if prev_selection:
            idx = self.sample_attribute_names.index(prev_selection)
            self.list_tags.scrollToItem(self.list_tags.item(idx))

    def set_sequences(self):
        """Update list of sequences according to self.sample_sequences."""
        self.list_sequences.clear()
        if self.sample_sequences[0] != '':
            self.list_sequences.addItems(self.sample_sequences)

    def reset_selected_tag(self):
        """Reset selected values, no currently selected tag after filling attribute list."""
        self.lbl_tag_string.setText('')
        self.txt_content.setPlainText('')
        self.lbl_n_items.setText('')
        self.spin_item.setEnabled(False)
        self.spin_item.setValue(1)

    def full_sequence_selected(self):
        """Checkbox Display full sequence (de)selected."""
        cur_seq = self.sample_sequences[-1].split(' ')[0]
        if self.chk_full_seq.isChecked():
            self.sample_sequences[-1] = cur_seq
        else:
            self.sample_sequences[-1] = f'{cur_seq} {self.spin_item.value()}/{self.lbl_n_items.text()}'
        self.set_sequences()
        self.read_tags()

    def update_item(self):
        """New item of sequence selected."""
        cur_seq = self.sample_sequences[-1].split(' ')[0]
        self.sample_sequences[-1] = f'{cur_seq} {self.spin_item.value()}/{self.lbl_n_items.text()}'
        self.set_sequences()
        self.read_tags()

    def attribute_selected(self):
        """Update attribute name."""
        self.reset_selected_tag()
        sel = self.list_tags.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            cur_text = self.sample_attribute_names[rowno]
            idx = self.sample_attribute_names.index(cur_text)
            tag = self.sample_tags[idx]
            if '(' in cur_text:  # missing keyword (tag numbar already in list)
                self.lbl_tag_string.setText(cur_text)
            else:
                self.lbl_tag_string.setText(f'({tag.group:04x},{tag.element:04x}) {cur_text}')

            if 'Sequence' in cur_text:
                if self.sample_sequences[0] == '':
                    self.sample_sequences[0] = cur_text
                else:
                    self.sample_sequences.append(cur_text)
                self.set_sequences()
                self.read_tags()
            else:
                self.update_content(cur_text)

    def update_content(self, attribute_name):
        """Display content of selected attribute or sequence."""
        if attribute_name:
            if '(' in attribute_name:  # private or missing keyword for other reasons?
                idx = self.sample_attribute_names.index(attribute_name)
                elem = self.current_dataset[self.sample_tags[idx]]
            else:
                elem = self.current_dataset[attribute_name]

            txt = str(elem)
            try:
                if elem.VR == 'UT':
                    if elem.value not in txt:
                        txt = txt + '\n\n' + f'(Value = {elem.value})'
            except:
                print(f'Failed printing {elem} of {attribute_name}')
            
            self.txt_content.setPlainText(txt)
        else:
            if 'Dataset' in str(type(self.current_dataset)):
                self.txt_content.setPlainText(str(self.current_dataset))
            else:
                txt = str(self.current_dataset._value)
                try:
                    if self.current_dataset.VR == 'SQ':
                        if len(self.current_dataset._value) > 1:
                            txt = ''
                            for itemno, item in enumerate(self.current_dataset._value):
                                txt = txt + f'\n\n<<<< item {itemno} >>>>\n' + str(item)
                except AttributeError:
                    pass
                self.txt_content.setPlainText(txt)

    def dump_dicom(self):
        """Dump dicom elements for file to text."""
        proceed = True
        if self.sample_filepath.text() == '':
            QMessageBox.information(self, 'Missing input', 'No file selected.')
            proceed = False
        if proceed:
            dcm.dump_dicom(self, filename=self.sample_filepath.text())

    def parse_sr(self):
        """Parse and display dose data."""
        output = []

        def parse_node(datasubset, level=0):
            for item in datasubset.ContentSequence:
                concept_name = ''
                if 'ConceptNameCodeSequence' in item:
                    if len(item.ConceptNameCodeSequence) > 0:
                        concept_name = item.ConceptNameCodeSequence[0].CodeMeaning

                value_type = getattr(item, 'ValueType', '')
                value = ''

                match value_type:
                    case 'NUM':
                        try:
                            meas_seq = item.MeasuredValueSequence[0]
                            num_val = meas_seq.NumericValue
                            unit = meas_seq.MeasurementUnitsCodeSequence[0].CodeMeaning if 'MeasurementUnitsCodeSequence' in meas_seq else ''
                            value = f"{num_val} {unit}"
                        except (AttributeError, IndexError):
                            pass
                    case 'CODE':
                        try:
                            value = item.ConceptCodeSequence[0].CodeMeaning
                        except (AttributeError, IndexError):
                            pass
                    case 'TEXT':
                        value = getattr(item, 'TextValue', '')
                    case 'DATETIME':
                        value = getattr(item, 'DateTime', '')
                    case 'CONTAINER':
                        value = '' #'[Container]'
                    case 'UIDREF':
                        value = getattr(item, 'UID', '')
                    case _:
                        value = f'(ValueType {value_type} not parsed)'

                output.append('\t'*level + f'{concept_name}: {value}')

                if 'ContentSequence' in item:
                    parse_node(item, level=level+1)

        if 'ContentSequence' in self.pydict:
            parse_node(self.pydict)
            txt = ('\n').join(output)
            self.txt_content.setPlainText(txt)
            
    def keyPressEvent(self, event):
        """Avoid close dialog on enter in widgets."""
        if event.key() == Qt.Key.Key_Return:
            pass
        else:
            super().keyPressEvent(event)

