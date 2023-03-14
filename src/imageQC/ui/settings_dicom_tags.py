#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for configuration settings.

@author: Ellen Wasbo
"""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QBrush, QColor
from PyQt5.QtWidgets import (
    QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QToolBar, QButtonGroup,
    QLabel, QLineEdit, QAction, QCheckBox, QListWidget, QComboBox,
    QAbstractScrollArea, QMessageBox, QDialogButtonBox, QFileDialog
    )
import pydicom

# imageQC block start
from imageQC.config.iQCconstants import QUICKTEST_OPTIONS, ENV_ICON_PATH
from imageQC.config import config_func as cff
from imageQC.config import config_classes as cfc
from imageQC.ui.settings_reusables import StackWidget, ToolBarImportIgnore
from imageQC.ui.tag_patterns import TagPatternWidget
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.ui import messageboxes
from imageQC.scripts import dcm
# imageQC block end


class DicomTagDialog(ImageQCDialog):
    """Dialog to add or edit tags for the DicomTagsWidget."""

    def __init__(self, tag_input=cfc.TagInfo()):
        super().__init__()
        self.tag_input = tag_input
        self.setWindowTitle('Add/edit DICOM tag')

        self.sample_filepath = QLineEdit()
        self.sample_filepath.textChanged.connect(self.get_all_tags_in_file)
        self.list_tags = QListWidget()
        self.list_tags.itemClicked.connect(self.attribute_selected)
        self.txt_attribute_name = QLineEdit(tag_input.attribute_name)
        self.tag_string = QLineEdit()
        self.tag_string.returnPressed.connect(self.correct_tag_input)
        self.list_sequences = QListWidget()
        self.cbox_value_id = QComboBox()
        self.cbox_value_id.currentIndexChanged.connect(self.update_value)
        self.lbl_tag_content = QLabel()
        self.btngr_modality = QButtonGroup(exclusive=False)
        self.txt_factor = QLineEdit(str(tag_input.factor))
        self.txt_factor.editingFinished.connect(self.validate_factor)
        self.txt_unit = QLineEdit(tag_input.unit)

        self.sample_attribute_names = []
        self.sample_sequences = ['']
        self.sample_tags = []
        self.pydict = None
        self.data_element = None

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        vlo.addWidget(QLabel('Read DICOM file to fill the list of tags'))
        hlo_file = QHBoxLayout()
        vlo.addLayout(hlo_file)
        self.sample_filepath.setMinimumWidth(500)
        hlo_file.addWidget(self.sample_filepath)
        toolb = uir.ToolBarBrowse('Browse to find sample file')
        toolb.act_browse.triggered.connect(self.locate_file)
        act_dcm_dump = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}tags.png'),
            "View DICOM dump", self)
        act_dcm_dump.triggered.connect(self.dump_dicom)
        toolb.addAction(act_dcm_dump)
        hlo_file.addWidget(toolb)

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
        self.act_level_up.setEnabled(False)
        tb_sequence = QToolBar()
        tb_sequence.addAction(self.act_level_up)
        hlo_taglist.addWidget(tb_sequence)
        vlo.addWidget(uir.LabelItalic(
            'Click tags named with Sequence to list the elements '
            'within the sequence.'))
        vlo.addWidget(uir.HLine())

        vlo.addSpacing(20)
        hlo = QHBoxLayout()
        vlo.addLayout(hlo)
        f_lo = QFormLayout()
        hlo.addLayout(f_lo)
        f_lo.addRow(QLabel('Attribute name: '), self.txt_attribute_name)
        f_lo.addRow(QLabel('Tag: '), self.tag_string)
        self.tag_string.setPlaceholderText('0000,0000')
        self.tag_string.setFixedWidth(140)
        self.tag_string.setText(
            f'{tag_input.tag[0][2:]:0>4},{tag_input.tag[1][2:]:0>4}')
        f_lo.addRow(QLabel('Value id if multivalue*:'), self.cbox_value_id)
        if tag_input.value_id != -1:
            if tag_input.value_id == -2:
                self.cbox_value_id.addItem('per frame')
            else:
                self.cbox_value_id.addItem(f'{tag_input.value_id}')
        f_lo.addRow(QLabel('Tag in sequence(s): '), self.list_sequences)
        self.list_sequences.setMaximumHeight(200)
        f_lo.addRow(QLabel('Multiply by: '), self.txt_factor)
        f_lo.addRow(QLabel('Unit: '), self.txt_unit)
        if tag_input.sequence[0] != '':
            for txt in tag_input.sequence:
                self.list_sequences.addItem(txt)
        f_lo.addRow(QLabel('Content from file: '), self.lbl_tag_content)
        vlo.addWidget(QLabel('* open sample file to fill options'))

        gb_modality = QGroupBox('Specified for modality...')
        gb_modality.setFont(uir.FontItalic())
        lo_mod = QVBoxLayout()
        listmod = [*QUICKTEST_OPTIONS]
        for idx, val in enumerate(listmod):
            chk = QCheckBox(val)
            self.btngr_modality.addButton(chk, idx)
            lo_mod.addWidget(chk)
            if val in tag_input.limited2mod:
                self.btngr_modality.button(idx).setChecked(True)
        gb_modality.setLayout(lo_mod)
        hlo.addWidget(gb_modality)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.verify_accept)
        buttons.rejected.connect(self.reject)
        vlo.addWidget(buttons)

    def verify_accept(self):
        """Validate."""
        if self.txt_attribute_name.text() == '':
            QMessageBox.warning(
                self, 'Missing info', 'Attribute name cannot be empty')
        else:
            self.accept()

    def locate_file(self):
        """Locate sample DICOM file."""
        fname = QFileDialog.getOpenFileName(
            self, 'Read DICOM file',
            filter="DICOM file (*.dcm);;All files (*)")
        if fname[0] != '':
            self.sample_sequences = ['']
            self.sample_filepath.setText(fname[0])
        else:
            self.pydict = None
            self.data_element = None

    def get_all_tags_in_file(self):
        """Get dicom tags to dropdownlist."""
        all_tags = []
        filename = self.sample_filepath.text()
        if filename != '':
            pd = {}
            try:
                pd = pydicom.dcmread(filename, stop_before_pixels=True)
            except pydicom.errors.InvalidDicomError:
                QMessageBox.information(
                    self, 'Failed reading DICOM',
                    'Could not read selected file as DICOM. InvalidDicomError')
            except FileNotFoundError:
                pass
            if len(pd) > 0:
                all_tags = dcm.get_all_tags_name_number(
                    pd, sequence_list=self.sample_sequences)
                self.pydict = pd
                self.get_tag_data()
                self.sample_attribute_names = all_tags['attribute_names']
                self.sample_tags = all_tags['tags']
            else:
                self.sample_attribute_names = []
                self.sample_tags = []

        self.list_tags.blockSignals(True)
        self.list_tags.clear()
        if len(self.sample_attribute_names) > 0:
            self.list_tags.addItems(self.sample_attribute_names)
            brush = QBrush(QColor(110, 148, 192))
            for attrno, attr_name in enumerate(self.sample_attribute_names):
                if 'Sequence' in attr_name:
                    self.list_tags.item(attrno).setBackground(brush)
        self.list_tags.blockSignals(False)

    def level_up(self):
        """Go back to above sequence."""
        if len(self.sample_sequences) == 1:
            self.sample_sequences = ['']
        else:
            self.sample_sequences.pop()
        if self.sample_sequences[0] == '':
            self.act_level_up.setEnabled(False)
        self.get_all_tags_in_file()
        self.set_sequences()

    def set_sequences(self):
        """Update list of sequences according to self.sample_sequences."""
        self.list_sequences.clear()
        if self.sample_sequences[0] != '':
            self.list_sequences.addItems(self.sample_sequences)

    def get_tag_data(self):
        """Update list of value_id options based on sample file and tag."""
        options = ['']
        if self.pydict is not None:
            taginfo_this = self.get_tag_info()
            data_element = dcm.get_tag_data(self.pydict, tag_info=taginfo_this)
            if isinstance(data_element, list):
                data_element = data_element[0]
            if data_element is not None:
                self.data_element = data_element
                if data_element.VM > 1:
                    options = [str(i) for i in range(data_element.VM)]
                    options.insert(0, 'all values')
                    frames = self.pydict.get('NumberOfFrames', -1)
                    if frames > 1:
                        options.insert(1, 'per frame')
        self.cbox_value_id.clear()
        self.cbox_value_id.addItems(options)
        self.update_value()

    def update_value(self):
        """Update lbl_tag_content according to current data_element."""
        val = ''
        if self.data_element is not None:
            if self.cbox_value_id.count() == 0:
                val = f'{self.data_element.value}'
            elif self.cbox_value_id.currentText() in ['all values', '']:
                val = f'{self.data_element.value}'
            elif self.cbox_value_id.currentText() == 'per frame':
                val = f'{self.data_element.value[0]} (frame 0)'
            else:
                selid = int(self.cbox_value_id.currentText())
                val = f'{self.data_element.value[selid]} (value {selid})'
        self.lbl_tag_content.setText(val)

    def attribute_selected(self):
        """Update attribute name."""
        sel = self.list_tags.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            cur_text = self.sample_attribute_names[rowno]

            if 'Sequence' in cur_text:
                if self.sample_sequences[0] == '':
                    self.sample_sequences[0] = cur_text
                else:
                    self.sample_sequences.append(cur_text)
                self.set_sequences()
                self.get_all_tags_in_file()
                self.act_level_up.setEnabled(True)
                self.cbox_value_id.clear()
            else:
                idx = self.sample_attribute_names.index(cur_text)
                self.txt_attribute_name.setText(cur_text)
                tag = self.sample_tags[idx]
                self.tag_string.setText(f'{tag.group:04x},{tag.element:04x}')
                self.set_sequences()
                self.get_tag_data()

    def str_to_tag(self, txt):
        """Convert string input to tag hex and back to string."""
        try:
            txt_group = hex(int('0x' + txt[0:4], 16))
            txt_elem = hex(int('0x' + txt[5:9], 16))
            tagstring = f'{txt_group},{txt_elem}'
        except ValueError:
            tagstring = '0000,0000'

        return tagstring

    def correct_tag_input(self, txt):
        """Correct string defining tag."""
        if len(txt) != 9:
            QMessageBox.warning(
                self, 'Unexpected tag format', 'Tag format expected as XXXX,XXXX.')
        else:
            self.tag_string.setText(self.str_to_tag(txt))

    def validate_factor(self):
        """Make sure factor is a float."""
        try:
            txt = self.txt_factor.text().replace(',', '.')  # accept , as decimal mark
            self.txt_factor.setText(str(float(txt)))
        except ValueError:
            self.txt_factor.setText('1.0')

    def get_tag_info(self):
        """Fill TagInfo with current values and return.

        Returns
        -------
        new_tag_info : TagInfo
        """
        tag_str = self.tag_string.text()
        tag_group = hex(int('0x' + tag_str[0:4], 16))
        tag_elem = hex(int('0x' + tag_str[5:9], 16))

        self.validate_factor()  # probably overkill
        new_tag_info = cfc.TagInfo(
            sort_index=self.tag_input.sort_index,
            attribute_name=self.txt_attribute_name.text(),
            tag=[tag_group, tag_elem],
            unit=self.txt_unit.text(),
            factor=float(self.txt_factor.text())
            )
        # get sequence from list as sample empty if input sequence (edit)
        if self.list_sequences.count() > 0:
            seq_list = []
            for i in range(self.list_sequences.count()):
                seq_list.append(self.list_sequences.item(i).text())
            if len(seq_list) > 0:
                new_tag_info.sequence = seq_list
        # value_id
        if self.cbox_value_id.count() > 0:
            cur_text = self.cbox_value_id.currentText()
            if cur_text == 'per frame':
                new_tag_info.value_id = -2
            elif cur_text in ['all', '']:
                new_tag_info.value_id = -1
            else:
                try:
                    new_tag_info.value_id = int(cur_text)
                except ValueError:
                    new_tag_info.value_id = -1

        checked_strings = []
        for idx in range(len([*QUICKTEST_OPTIONS])):
            if self.btngr_modality.button(idx).isChecked():
                checked_strings.append(
                    self.btngr_modality.button(idx).text())
        if (len(checked_strings) == 0
                or len(checked_strings) == len([*QUICKTEST_OPTIONS])):
            checked_strings = ['']
        new_tag_info.limited2mod = checked_strings

        return new_tag_info

    def dump_dicom(self):
        """Dump dicom elements for file to text."""
        proceed = True
        if self.sample_filepath.text() == '':
            QMessageBox.information(self, 'Missing input', 'No file selected.')
            proceed = False
        if proceed:
            dcm.dump_dicom(self, filename=self.sample_filepath.text())


class DicomTagsWidget(StackWidget):
    """Widget for Dicom Tags."""

    def __init__(self, dlg_settings):
        header = 'DICOM tags'
        subtxt = '''Customize list of available DICOM tags.<br>
        This information defines the available DICOM tags in ImageQC and how
        to read the tags.<br>
        NB: If a tag is "hidden" within a DICOM sequence,
        this sequence has to be defined (different variants might exist).<br>
        Variants of how to read the DICOM content is possible through
        duplicated attribute names.<br>
        When reading an attribute with variants, the first variant
        (by sort index) will be searched first, if not found,
        the next variant will be searched.<br>
        Some tags are protected as other parts of imageQC rely on them.
        Some of these are special tags, not reading a tag, but using other
        DICOM information.<br>
        The special tags are not available in all tag patterns.
        (Image sum for CountsAccumulated only available for test DCM).'''
        super().__init__(dlg_settings, header, subtxt)
        self.fname = 'tag_infos'
        self.indexes = []
        self.edited_names = []

        self.table_tags = QTreeWidget()
        self.table_tags.setColumnCount(4)
        self.table_tags.setHeaderLabels(
            ['Tag name', 'Tag number', 'Value id', 'Factor', 'unit',
             'Specific modalities', 'in sequence(s)', 'Sort index', 'Protected'])
        try:
            ch_w = self.dlg_settings.main.gui.char_width
        except AttributeError:
            ch_w = 12
        self.table_tags.setColumnWidth(0, 35*ch_w)
        self.table_tags.setColumnWidth(1, 14*ch_w)
        self.table_tags.setColumnWidth(2, 12*ch_w)
        self.table_tags.setColumnWidth(3, 14*ch_w)
        self.table_tags.setColumnWidth(4, 12*ch_w)
        self.table_tags.setColumnWidth(5, 20*ch_w)
        self.table_tags.setColumnWidth(6, 45*ch_w)
        self.table_tags.setColumnWidth(7, 14*ch_w)
        self.table_tags.setColumnWidth(8, 12*ch_w)
        self.sort_col_id = 7
        self.table_tags.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContents)
        self.table_tags.setSortingEnabled(True)
        self.table_tags.sortByColumn(self.sort_col_id, Qt.AscendingOrder)
        self.table_tags.setRootIsDecorated(False)
        self.table_tags.header().sectionClicked.connect(self.sort_header)

        hlo_modality = QHBoxLayout()
        gb_modality = QGroupBox('Tags specified for modality...')
        gb_modality.setFont(uir.FontItalic())
        self.btngr_modality_filter = QButtonGroup(exclusive=False)

        chk_all_modalities = QCheckBox()
        chk_all_modalities.clicked.connect(self.select_all_modalities)
        chk_all_modalities.setToolTip('Select (or deselect) all')
        chk_all_modalities.setFixedWidth(50)

        hlo = QHBoxLayout()
        listmod = ['General (not specified)']
        listmod.extend([*QUICKTEST_OPTIONS])
        for idx, val in enumerate(listmod):
            chk = QCheckBox(val)
            self.btngr_modality_filter.addButton(chk, idx)
            hlo.addWidget(chk)
            chk.clicked.connect(self.mode_changed)
            self.btngr_modality_filter.button(idx).setChecked(True)
        gb_modality.setLayout(hlo)
        gb_modality.setFixedWidth(1300)

        hlo_modality.addWidget(chk_all_modalities)
        hlo_modality.addWidget(gb_modality)
        hlo_modality.addStretch()
        self.vlo.addLayout(hlo_modality)

        hlo_table = QHBoxLayout()
        self.vlo.addLayout(hlo_table)
        hlo_table.addWidget(self.table_tags)

        if self.import_review_mode:
            toolb = ToolBarImportIgnore(self, typestr='tag')
        else:
            toolb = QToolBar()
            toolb.setOrientation(Qt.Vertical)
            act_add = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                'Add new tag to list', self)
            act_add.triggered.connect(self.add_tag)
            act_duplicate = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}copy.png'),
                'Duplicate tag to create variant', self)
            act_duplicate.triggered.connect(self.duplicate_tag)
            act_edit = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
                'Edit details for the selected tag', self)
            act_edit.triggered.connect(self.edit_tag)
            act_up = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
                'Move up', self)
            act_up.triggered.connect(self.move_up)
            act_down = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
                'Move down', self)
            act_down.triggered.connect(self.move_down)
            act_delete = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
                'Delete tag', self)
            act_delete.triggered.connect(self.delete)
            act_save = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
                'Save changes', self)
            act_save.triggered.connect(self.save)

            toolb.addActions([
                act_add, act_duplicate, act_edit, act_up, act_down,
                act_delete, act_save])

            if self.dlg_settings.main.save_blocked:
                act_add.setEnabled(False)
                act_edit.setEnabled(False)
                act_up.setEnabled(False)
                act_down.setEnabled(False)
                act_delete.setEnabled(False)
                act_save.setEnabled(False)

        hlo_table.addWidget(toolb)
        hlo_table.addStretch()

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def sort_header(self):
        """When clicked to sort column."""
        self.flag_edit()
        self.update_indexes()

    def update_indexes(self):
        """Update self.indexes according to displayed table."""
        self.indexes = []
        for row in range(self.table_tags.topLevelItemCount()):
            item = self.table_tags.topLevelItem(row)
            self.indexes.append(int(item.text(self.sort_col_id)))

    def update_data(self, set_selected_row=0,
                    keep_selection=False,
                    reset_sort_index=False):
        """Update table at start, when mode or config_folder change.

        Parameters
        ----------
        set_selected_row : int, optional
            Row number to highlight. The default is 0.
        keep_selection : bool, optional
            If true - keep same selected rowno as before.
        reset_sort_index : bool, optional
            reset sort index to order in self.templates. The default is False.
        """
        if keep_selection:
            sel = self.table_tags.selectedIndexes()
            if len(sel) > 0:
                set_selected_row = sel[0].row()

        if reset_sort_index:
            cff.taginfos_reset_sort_index(self.templates)
            self.flag_edit()

        self.table_tags.clear()
        checked_strings = []
        for idx in range(len([*QUICKTEST_OPTIONS])+1):
            if self.btngr_modality_filter.button(idx).isChecked():
                checked_strings.append(
                    self.btngr_modality_filter.button(idx).text())
                if idx == 0:
                    checked_strings[-1] = ''

        for temp in self.templates:
            # test if included in modality selection
            include = bool(set(checked_strings).intersection(temp.limited2mod))
            if include:
                row_strings = [
                    temp.attribute_name, '-', '-', '-', temp.unit,
                    ', '.join(temp.limited2mod), ', '.join(temp.sequence),
                    f'{temp.sort_index:03}', '-'
                    ]
                if temp.tag[1] == '':
                    row_strings[1] = temp.tag[0]
                else:
                    row_strings[1] = (f'{temp.tag[0][2:]:0>4},{temp.tag[1][2:]:0>4}')
                if temp.value_id != -1:
                    if temp.value_id == -2:
                        row_strings[2] = 'per frame'
                    elif temp.value_id == -3:
                        row_strings[2] = 'join'
                    else:
                        row_strings[2] = f'{temp.value_id}'
                if temp.factor != 1.0:
                    row_strings[3] = f'{temp.factor}'
                if temp.protected:
                    row_strings[8] = 'P'
                item = QTreeWidgetItem(row_strings)
                self.table_tags.addTopLevelItem(item)

        nrows = self.table_tags.topLevelItemCount()
        if set_selected_row < 0 or set_selected_row >= nrows:
            set_selected_row = 0
        if len(self.templates) > 0:
            self.table_tags.setCurrentItem(
                self.table_tags.topLevelItem(set_selected_row))
            self.update_indexes()

    def mode_changed(self):
        """Update table if modality selection changes."""
        self.update_data()

    def select_all_modalities(self):
        """Select all or none modalities."""
        checked_ids = [None] * (len([*QUICKTEST_OPTIONS]) + 1)
        for idx in range(len([*QUICKTEST_OPTIONS])+1):
            checked_ids[idx] = int(
                self.btngr_modality_filter.button(idx).isChecked())
        selection_all = not any(checked_ids)

        for idx in range(len([*QUICKTEST_OPTIONS])+1):
            self.btngr_modality_filter.button(idx).blockSignals(True)
            self.btngr_modality_filter.button(idx).setChecked(selection_all)
            self.btngr_modality_filter.button(idx).blockSignals(False)

        self.update_data()

    def get_idx_same_name(self, name='', mod_dict=False):
        """Get tag indexes of same attribute_name as name.

        Parameters
        ----------
        name : str
            attribute name to test
        mod_dict : bool, optional
            dict with list for each modality and general. The default is False.

        Returns
        -------
        idx_same_name : list of int or dict of lists of int
            indexes of taglist having same attribute name
            if mod=True {'CT': [], 'Xray': []...}
        """
        if mod_dict:
            idx_same_name = {mod: [] for mod in QUICKTEST_OPTIONS}
        else:
            idx_same_name = []
        for idx, temp in enumerate(self.templates):
            if temp.attribute_name == name:
                if mod_dict:
                    if temp.limited2mod == ['']:
                        for mod in idx_same_name:
                            idx_same_name[mod].append(idx)
                    else:
                        for mod in temp.limited2mod:
                            idx_same_name[mod].append(idx)
                else:
                    idx_same_name.append(idx)
        return idx_same_name

    def update_tag_info(self, tag_info, edit=False, index=-1):
        """Verify that a new tag can be added to the templates.

        Parameters
        ----------
        tag_info : TagInfo
            TagInfo from DicomTagDialog
        edit : bool, optional
            Edit tag was selected first. The default is False.
        index : int, optional
            -1 = add, else edit. The default is -1.
        """
        if edit and index > -1:  # if edit name and more than one - change all?
            old_temp = self.templates[index]
            old_name = self.templates[index].attribute_name
            new_name = tag_info.attribute_name
            if old_name != new_name:
                idx_same_name = self.get_idx_same_name(name=old_temp.attribute_name)
                if len(idx_same_name) > 1:
                    res = messageboxes.QuestionBox(
                        parent=self, title='Change name of all subelements',
                        msg=f'''{len(idx_same_name)} tags have name
                             {old_temp.attribute_name}.<br>
                             Change all these to {tag_info.attribute_name}?''',
                        yes_text='Yes, change all names',
                        no_text='No, change only this as an unlinked tag')
                    if res.exec():
                        # Change name of all other than index
                        for idx in idx_same_name:
                            if idx != index:
                                self.templates[idx].attribute_name = (
                                    tag_info.attribute_name)
                        self.edited_names.append([old_name, new_name])
                    # else some left with old name and tag references kept
                else:
                    self.edited_names.append([old_name, new_name])

        sel = self.table_tags.selectedIndexes()
        selrow = sel[0].row()
        if index == -1:  # add new
            temp_idx = self.indexes[selrow]
            self.templates.insert(temp_idx+1, tag_info)
            cff.taginfos_reset_sort_index(self.templates)
        else:  # edit
            new_name = tag_info.attribute_name
            self.templates[index] = tag_info

        self.flag_edit()
        self.update_data(set_selected_row=selrow)

    def add_tag(self):
        """Open Dialog to add tag."""
        dlg = DicomTagDialog()
        dlg.exec()
        newtag = dlg.get_tag_info()
        self.update_tag_info(newtag)

    def duplicate_tag(self):
        """Open Dialog to add tag with current tag settings."""
        sel = self.table_tags.selectedIndexes()
        selrow = sel[0].row()
        idx = self.indexes[selrow]
        dlg = DicomTagDialog(tag_input=self.templates[idx])
        dlg.exec()
        newtag = dlg.get_tag_info()
        self.update_tag_info(newtag)

    def edit_tag(self):
        """Open Dialog to edit tag."""
        sel = self.table_tags.selectedIndexes()
        selrow = sel[0].row()
        idx = self.indexes[selrow]
        if self.templates[idx].protected:
            QMessageBox.information(
                self, 'Protected tag',
                ('The selected tag is protected and can not be edited.')
            )
        else:
            dlg = DicomTagDialog(tag_input=self.templates[idx])
            res = dlg.exec()
            if res:
                edited_tag = dlg.get_tag_info()
                self.update_tag_info(edited_tag, edit=True, index=idx)

    def move_up(self):
        """Move tag before tag above."""
        sel = self.table_tags.selectedIndexes()
        selrow = sel[0].row()
        idx = self.indexes[selrow]
        if selrow != 0:
            idxprev = self.indexes[selrow-1]
            if idxprev > idx:
                QMessageBox.information(
                    self, 'Sorting prevented',
                    ('Press header "Sort index" to sort ascending '
                     + 'before moving tag or save to set sorting according '
                     + 'to current sorting.'))
            else:
                item_to_move = self.templates.pop(idx)
                self.templates.insert(idxprev, item_to_move)
                self.flag_edit()
                self.update_data(set_selected_row=selrow-1,
                                 reset_sort_index=True)

    def move_down(self):
        """Move tag after tag below."""
        sel = self.table_tags.selectedIndexes()
        selrow = sel[0].row()
        idx = self.indexes[selrow]
        if selrow != len(self.indexes)-1:
            idxnext = self.indexes[selrow+1]
            if idxnext < idx:
                QMessageBox.information(
                    self, 'Sorting prevented',
                    ('Press header "Sort index" to sort ascending '
                     + 'before moving tag or save to set sorting according '
                     + 'to current sorting.'))
            else:
                item_to_move = self.templates.pop(idxnext)
                self.templates.insert(idx, item_to_move)
                self.flag_edit()
                self.update_data(set_selected_row=selrow+1,
                                 reset_sort_index=True)

    def delete(self):
        """Delete selected tag."""
        sel = self.table_tags.selectedIndexes()
        selrow = sel[0].row()
        idx = self.indexes[selrow]
        if self.templates[idx].protected:
            QMessageBox.information(
                self, 'Protected tag',
                ('The selected tag is protected and can not be deleted.')
            )
        else:
            res = messageboxes.QuestionBox(
                parent=self, title='Delete?',
                msg=f'Delete tag {self.templates[idx].attribute_name}')
            if res.exec():
                idx_same_name = self.get_idx_same_name(
                    name=self.templates[idx].attribute_name, mod_dict=True)
                n_idx_same_mod = [len(idxs) for m, idxs in idx_same_name.items()]
                proceed = True
                if 1 in n_idx_same_mod:
                    self.tag_infos = self.templates
                    _, _, found_attributes = cff.get_taginfos_used_in_templates(
                        self)
                    for mod, attr in found_attributes.items():
                        if len(idx_same_name[mod]) == 1:
                            if self.templates[idx].attribute_name in attr:
                                proceed = False
                    if proceed is False:
                        QMessageBox.warning(
                            self, 'Tag in use',
                            'The selected tag is in use. Tag info can not be deleted.')
                if proceed:
                    self.templates.pop(idx)
                    self.flag_edit()
                    self.update_data(set_selected_row=selrow, reset_sort_index=True)

    def save(self):
        """Save current settings."""
        proceed = cff.verify_config_folder(self)
        if proceed:
            proceed, errmsg = cff.check_save_conflict(
                self.fname, self.lastload)
            if errmsg != '':
                proceed = messageboxes.proceed_question(self, errmsg)
            if proceed:
                # check if order of sort_index is ascending, if not ask to
                # save order as display
                sorted_indexes = sorted(self.indexes)
                if self.indexes != sorted_indexes:
                    res = messageboxes.QuestionBox(
                        parent=self, title='Save order?',
                        msg='Also save order of tags as displayed?')
                    if res.exec():
                        new_templates = []
                        for idx in self.indexes:
                            new_templates.append(self.templates[idx])
                        if len(self.indexes) < len(self.templates):
                            sort_desc = sorted(self.indexes, reverse=True)
                            for idx in sort_desc:
                                self.templates.pop(idx)
                            new_templates.extend(self.templates)
                        self.templates = new_templates
                        cff.taginfos_reset_sort_index(self.templates)

                log = []
                status = True
                # any changed attribute_names?
                # Update other templates using TagInfos
                if len(self.edited_names) > 0:
                    status, log = cff.attribute_names_used_in(
                        old_new_names=self.edited_names)

                ok = True
                if status:
                    ok_save, path = cff.save_settings(
                        self.templates, fname=self.fname)
                    if ok_save:
                        self.flag_edit(False)
                        self.edited_names = []
                        self.update_data(keep_selection=True)
                    else:
                        log.append(f'Failed saving to {path}')
                else:
                    log.append('Aborted saving tag_infos.yaml')

                if status and ok:
                    ico = QMessageBox.Information
                    tit = 'Changes saved'
                    if len(log) > 0:
                        txt = 'See details to view changes performed'
                    else:
                        txt = 'Changes saved to tag_infos.yaml'
                else:
                    ico = QMessageBox.Warning
                    tit = 'Issues on saving changes'
                    txt = 'Something went wrong while saving. See details.'
                dlg = messageboxes.MessageBoxWithDetails(
                    self, title=tit, msg=txt, details=log, icon=ico)
                dlg.exec()

    def mark_import(self, ignore=False):
        """If import review mode: Mark tag for import or ignore."""
        if not hasattr(self, 'marked'):  # initiate
            self.marked_ignore = []
            self.marked = []

        sel = self.table_tags.selectedIndexes()
        if len(sel) > 0:
            row = sel[0].row()
            item = self.table_tags.currentItem()
            if ignore:
                if row not in self.marked_ignore:
                    self.marked_ignore.append(row)
                if row in self.marked:
                    self.marked.remove(row)
                item.setBackground(0, QBrush(
                        QColor(203, 91, 76)))
            else:
                if row not in self.marked:
                    self.marked.append(row)
                if row in self.marked_ignore:
                    self.marked_ignore.remove(row)
                item.setBackground(0, QBrush(
                        QColor(127, 153, 85)))


class TagPatternSpecialWidget(StackWidget):
    """Setup for editing the special (protected) tag patterns."""

    def __init__(self, dlg_settings):
        header = 'Special tag patterns'
        subtxt = '''These tag patterns each have a specific function
        and can not be renamed or deleted, just edited.'''
        super().__init__(dlg_settings, header, subtxt,
                         typestr='tag pattern',
                         mod_temp=True, grouped=True)

        self.fname = 'tag_patterns_special'
        self.empty_template = cfc.TagPatternFormat()

        self.wid_tag_pattern = TagPatternWidget(self, typestr='format')
        if self.import_review_mode is False:
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_add)
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_rename)
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_duplicate)
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_up)
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_down)
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_delete)
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_move_modality)
        else:
            self.wid_tag_pattern.setEnabled(False)
        self.wid_mod_temp.list_temps.setFixedHeight(200)
        info_special = """<html><head/><body>
            <p><i><b>Annotate</b> define text for<br>
            image annotation.</i></p>
            <p><i><b>DICOM_display</b> define which<br>
            tags to display in the<br>
            DICOM header widget.</i></p>
            <p><i><b>File_list_display</b> define which<br>
            tags to display in the file list <br>
            if Tag pattern option is<br>
            selected instead of filepath.</i></p>
            </body></html>"""
        self.wid_mod_temp.vlo.addWidget(QLabel(info_special))
        self.wid_mod_temp.vlo.addStretch()
        self.hlo.addWidget(self.wid_tag_pattern)

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with the selected template."""
        self.wid_tag_pattern.wid_pattern.update_data()
        self.flag_edit(False)


class TagPatternFormatWidget(StackWidget):
    """Setup for creating tag patterns for formatting."""

    def __init__(self, dlg_settings):
        header = 'Tag patterns for formatting'
        subtxt = '''Tag patterns can be used for<br>
        - renaming DICOM files based on the DICOM tags<br>
        - exporting specific DICOM tags as tabular data'''
        super().__init__(dlg_settings, header, subtxt,
                         typestr='template',
                         mod_temp=True, grouped=True)

        self.fname = 'tag_patterns_format'
        self.empty_template = cfc.TagPatternFormat()

        self.wid_tag_pattern = TagPatternWidget(self, typestr='format')
        if self.import_review_mode:
            self.wid_tag_pattern.setEnabled(False)
        self.hlo.addWidget(self.wid_tag_pattern)

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with the selected template."""
        self.wid_tag_pattern.wid_pattern.update_data()
        self.flag_edit(False)


class TagPatternSortWidget(StackWidget):
    """Setup for creating tag patterns for sorting."""

    def __init__(self, dlg_settings):
        header = 'Tag patterns for sorting'
        subtxt = ('These tag patterns can be used for '
                  'sorting DICOM files based on the tags.')
        super().__init__(dlg_settings, header, subtxt,
                         typestr='template',
                         mod_temp=True, grouped=True)

        self.fname = 'tag_patterns_sort'
        self.empty_template = cfc.TagPatternSort()

        self.wid_tag_pattern = TagPatternWidget(self, typestr='sort')
        if self.import_review_mode:
            self.wid_tag_pattern.setEnabled(False)
        self.hlo.addWidget(self.wid_tag_pattern)

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with the selected template."""
        self.wid_tag_pattern.wid_pattern.update_data()
        self.flag_edit(False)


class RenamePatternWidget(StackWidget):
    """Setup for creating tag patterns for formatting."""

    def __init__(
            self, dlg_settings=None, initial_modality='CT',
            header=None, subtxt=None, editable=True):
        if header is None:
            header = 'Rename patterns'
        if subtxt is None:
            subtxt = (
                'Rename patterns combine two format - tag patterns to rename '
                'subfolders and files separately.<br>'
                'The subfolder tag pattern is also used when sorting files '
                'into subfolders (in Rename DICOM dialog or in the dialog '
                'with advanced open files options).'
                )
        super().__init__(dlg_settings, header, subtxt,
                         typestr='template',
                         mod_temp=True, grouped=True, editable=editable)

        self.fname = 'rename_patterns'
        self.empty_template = cfc.RenamePattern()
        self.current_modality = initial_modality

        self.wid_tag_pattern = TagPatternWidget(
            self, typestr='format', rename_pattern=True, editable=editable)
        if self.import_review_mode:
            self.wid_tag_pattern.setEnabled(False)
        self.hlo.addWidget(self.wid_tag_pattern)

        if editable:
            self.vlo.addWidget(uir.HLine())
            self.vlo.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with the selected template."""
        self.wid_tag_pattern.wid_pattern.update_data()
        self.wid_tag_pattern.wid_pattern2.update_data()
        self.flag_edit(False)
