#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface classes related to tag patterns.

@author: Ellen Wasbo
"""
from time import time
import copy
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QBrush, QColor, QPalette
from PyQt5.QtWidgets import (
    QWidget, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QToolBar, QAction, QComboBox, QLabel, QPushButton, QListWidget, QLineEdit,
    QTreeWidget, QTreeWidgetItem, QMessageBox, QInputDialog
    )

# imageQC block start
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.ui import messageboxes
from imageQC.scripts.mini_methods_format import get_format_strings
from imageQC.scripts.mini_methods import get_included_tags
from imageQC.config.iQCconstants import ENV_ICON_PATH
import imageQC.config.config_func as cff
# imageQC block end


class TagPatternTree(QWidget):
    """Widget for tag pattern and toolbar used in TagPatternWidget."""

    def __init__(self, parent, title='Tag pattern', typestr='sort',
                 list_number=1, editable=True):
        super().__init__()
        self.parent = parent
        self.parentabove = self.parent.parent
        self.typestr = typestr
        self.list_number = list_number

        self.hlo = QHBoxLayout()
        self.setLayout(self.hlo)

        if editable:
            vlo_push = QVBoxLayout()
            self.hlo.addLayout(vlo_push)
            vlo_push.addStretch()
            btn_push = QPushButton('>>')
            btn_push.clicked.connect(self.push_tag)
            vlo_push.addWidget(btn_push)
            vlo_push.addStretch()
            self.hlo.addSpacing(20)

        vlo = QVBoxLayout()
        self.hlo.addLayout(vlo)
        vlo.addWidget(uir.LabelItalic(title))
        self.table_pattern = QTreeWidget()
        self.table_pattern.setColumnCount(2)
        self.table_pattern.setColumnWidth(0, 200)
        if self.typestr == 'none':
            self.table_pattern.setColumnCount(1)
            self.table_pattern.setColumnWidth(0, 200)
            self.table_pattern.setHeaderLabels(['Tag'])
        else:
            self.table_pattern.setColumnCount(2)
            self.table_pattern.setColumnWidth(0, 200)
            self.table_pattern.setColumnWidth(1, 200)
        if self.typestr == 'sort':
            self.table_pattern.setHeaderLabels(['Tag', 'Sorting'])
        elif self.typestr == 'format':
            self.table_pattern.setHeaderLabels(['Tag', 'Format'])

        vlo.addWidget(self.table_pattern)
        if editable:
            self.table_pattern.setMinimumSize(400, 220)
        else:
            self.table_pattern.setMinimumWidth(400)
        self.table_pattern.setRootIsDecorated(False)

        palette = self.table_pattern.palette()
        palette.setColor(
            QPalette.Inactive, QPalette.Highlight,
            palette.color(QPalette.Active, QPalette.Highlight))
        palette.setColor(
            QPalette.Inactive, QPalette.HighlightedText,
            palette.color(QPalette.Active, QPalette.HighlightedText))
        self.table_pattern.setPalette(palette)

        if editable:
            toolb = QToolBar()
            toolb.setOrientation(Qt.Vertical)
            act_sort = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}sortAZ.png'),
                'ASC or DESC when sorting images', self)
            act_sort .triggered.connect(self.sort)
            act_format_out = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}format.png'),
                'Format output for selected tag', self)
            act_format_out.triggered.connect(self.format_output)
            act_up = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
                'Move tag(s) up in pattern list', self)
            act_up.triggered.connect(self.move_up)
            act_down = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
                'Move tag(s) down in pattern list', self)
            act_down.triggered.connect(self.move_down)
            act_delete = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
                'Delete selected tag(s) from pattern', self)
            act_delete.triggered.connect(self.delete)
            if self.typestr == 'sort':
                toolb.addActions([act_sort, act_up, act_down, act_delete])
            elif self.typestr == 'none':
                toolb.addActions([act_up, act_down, act_delete])
            else:
                toolb.addActions([act_format_out, act_up, act_down, act_delete])
            self.hlo.addWidget(toolb)

    def push_tag(self):
        """Button >> pressed - push selected tags into pattern."""
        rows = [index.row() for index in
                self.parent.list_tags.selectedIndexes()]
        if self.list_number == 2:
            tag_already = self.parentabove.current_template.list_tags2
        else:
            tag_already = self.parentabove.current_template.list_tags

        for row in rows:
            if self.parent.list_tags.item(row).text() not in tag_already:
                if self.list_number == 1:
                    self.parentabove.current_template.list_tags.append(
                        self.parent.list_tags.item(row).text())
                    if self.typestr in ['sort', 'none']:
                        self.parentabove.current_template.list_sort.append(
                            True)
                    else:
                        self.parentabove.current_template.list_format.append(
                            '')
                else:
                    self.parentabove.current_template.list_tags2.append(
                        self.parent.list_tags.item(row).text())
                    self.parentabove.current_template.list_format2.append('')
        self.update_data(set_selected=-1)
        try:
            self.parentabove.flag_edit()
        except AttributeError:
            pass

    def sort(self):
        """Change between ASC / DESC for selected tag."""
        sel = self.table_pattern.selectedIndexes()
        row = -1
        if len(sel) > 0:
            row = sel[0].row()
        if row > -1:
            self.parentabove.current_template.list_sort[row] = \
                not self.parentabove.current_template.list_sort[row]
            self.update_data(set_selected=row)
            try:
                self.parentabove.flag_edit()
            except AttributeError:
                pass

    def format_output(self):
        """Edit f-string for selected tag."""
        sel = self.table_pattern.selectedIndexes()
        row = -1
        if len(sel) > 0:
            row = sel[0].row()
        if row > -1:
            if self.list_number == 1:
                format_str = self.parentabove.current_template.list_format[row]
            else:
                format_str = self.parentabove.current_template.list_format2[
                    row]
            dlg = FormatDialog(format_string=format_str)
            res = dlg.exec()
            if res:
                new_str = dlg.get_data()
                if self.list_number == 1:
                    self.parentabove.current_template.list_format[
                        row] = new_str
                else:
                    self.parentabove.current_template.list_format2[
                        row] = new_str
                self.update_data(set_selected=row)
                try:
                    self.parentabove.flag_edit()
                except AttributeError:
                    pass
        else:
            QMessageBox.information(
                self, 'No tag selected',
                'Select a tag from the tag pattern to format.')

    def move_up(self):
        """Move tag up if possible."""
        sel = self.table_pattern.selectedIndexes()
        row = -1
        if len(sel) > 0:
            row = sel[0].row()
        if row > 0:
            if self.list_number == 1:
                popped_tag = \
                    self.parentabove.current_template.list_tags.pop(row)
                self.parentabove.current_template.list_tags.insert(
                    row - 1, popped_tag)
            else:
                popped_tag = \
                    self.parentabove.current_template.list_tags2.pop(row)
                self.parentabove.current_template.list_tags2.insert(
                    row - 1, popped_tag)

            if self.typestr == 'sort':
                popped_sort = self.parentabove.current_template.list_sort.pop(
                    row)
                self.parentabove.current_template.list_sort.insert(
                    row - 1, popped_sort)
            else:
                if self.list_number == 1:
                    popped_format = \
                        self.parentabove.current_template.list_format.pop(row)
                    self.parentabove.current_template.list_format.insert(
                        row - 1, popped_format)
                else:
                    popped_format = \
                        self.parentabove.current_template.list_format2.pop(row)
                    self.parentabove.current_template.list_format2.insert(
                        row - 1, popped_format)
            self.update_data(set_selected=row-1)
            try:
                self.parentabove.flag_edit()
            except AttributeError:
                pass

    def move_down(self):
        """Move tag down if possible."""
        sel = self.table_pattern.selectedIndexes()
        row = -1
        if len(sel) > 0:
            row = sel[0].row()
        if self.list_number == 1:
            n_tags = len(self.parentabove.current_template.list_tags)
        else:
            n_tags = len(self.parentabove.current_template.list_tags2)
        if row < n_tags-1:
            if self.list_number == 1:
                popped_tag = \
                    self.parentabove.current_template.list_tags.pop(row)
                self.parentabove.current_template.list_tags.insert(
                    row + 1, popped_tag)
            else:
                popped_tag = \
                    self.parentabove.current_template.list_tags2.pop(row)
                self.parentabove.current_template.list_tags2.insert(
                    row + 1, popped_tag)
            if self.typestr == 'sort':
                popped_sort = \
                    self.parentabove.current_template.list_sort.pop(row)
                self.parentabove.current_template.list_sort.insert(
                    row, popped_sort)
            else:
                if self.list_number == 1:
                    popped_format = \
                        self.parentabove.current_template.list_format.pop(row)
                    self.parentabove.current_template.list_format.insert(
                        row + 1, popped_format)
                else:
                    popped_format = \
                        self.parentabove.current_template.list_format2.pop(row)
                    self.parentabove.current_template.list_format2.insert(
                        row + 1, popped_format)
            self.update_data(set_selected=row+1)
            try:
                self.parentabove.flag_edit()
            except AttributeError:
                pass

    def delete(self):
        """Delete selected tag(s)."""
        sel = self.table_pattern.selectedIndexes()
        row = -1
        if len(sel) > 0:
            row = sel[0].row()
            if self.list_number == 1:
                self.parentabove.current_template.list_tags.pop(row)
            else:
                self.parentabove.current_template.list_tags2.pop(row)
            if self.typestr == 'sort':
                self.parentabove.current_template.list_sort.pop(row)
            else:
                if self.list_number == 1:
                    self.parentabove.current_template.list_format.pop(row)
                else:
                    self.parentabove.current_template.list_format2.pop(row)
            self.update_data(set_selected=-1)
            try:
                self.parentabove.flag_edit()
            except AttributeError:
                pass

    def update_data(self, set_selected=0):
        """Update table_pattern with data from current_template."""
        self.table_pattern.clear()
        if self.list_number == 1:
            list_tags = self.parentabove.current_template.list_tags
        else:
            list_tags = self.parentabove.current_template.list_tags2
        if len(list_tags) > 0:
            for rowno, tagname in enumerate(list_tags):

                if self.typestr == 'sort':
                    infotext = 'ASC' if \
                        self.parentabove.current_template.list_sort[rowno] \
                        else 'DESC'
                else:
                    if self.list_number == 1:
                        infotext = \
                            self.parentabove.current_template.list_format[
                                rowno]
                    else:
                        infotext = \
                            self.parentabove.current_template.list_format2[
                                rowno]
                row_strings = [tagname, infotext]
                item = QTreeWidgetItem(row_strings)
                if self.parent.lock_on_general is False:
                    try:
                        if tagname in self.parent.general_tags:
                            item.setForeground(
                                0, QBrush(QColor(110, 148, 192)))
                    except AttributeError:
                        pass  # ignore if editable = False
                self.table_pattern.addTopLevelItem(item)

            if set_selected == -1:
                set_selected = self.table_pattern.topLevelItemCount() - 1
            self.table_pattern.setCurrentItem(
                self.table_pattern.topLevelItem(set_selected))


class TagPatternWidget(QWidget):
    """Widget for setting the parameters for TagPattern.

    Parameters
    ----------
    parent : widget
    typestr : str
        'sort' if TagPatternSort, 'format' if TagPatternFormat
    lock_on_general : bool
        True if only general tag_infos (e.g. automation rename)
    rename_pattern : bool
        True = two patterns (subfolder + file)
    editable : bool
        False if editing not available (taglist and buttons not visible)
    """

    def __init__(self, parent, typestr='sort', lock_on_general=False,
                 rename_pattern=False, open_files_pattern=False,
                 editable=True):
        super().__init__()
        self.parent = parent
        self.typestr = typestr
        self.lock_on_general = lock_on_general
        self.rename_pattern = rename_pattern
        self.open_files_pattern = open_files_pattern

        hlo = QHBoxLayout()
        self.setLayout(hlo)

        vlo_taglist = QVBoxLayout()
        hlo.addLayout(vlo_taglist)

        if editable:
            vlo_taglist.addWidget(uir.LabelItalic('Available DICOM tags'))
            self.list_tags = QListWidget()
            self.list_tags.setSelectionMode(QListWidget.ExtendedSelection)
            self.list_tags.itemDoubleClicked.connect(self.double_click_tag)
            vlo_taglist.addWidget(self.list_tags)
            if self.lock_on_general:
                vlo_taglist.addWidget(uir.LabelItalic('General tags only'))
            else:
                vlo_taglist.addWidget(uir.LabelItalic('Blue font = general tags'))

            palette = self.list_tags.palette()
            palette.setColor(
                QPalette.Inactive, QPalette.Highlight,
                palette.color(QPalette.Active, QPalette.Highlight))
            palette.setColor(
                QPalette.Inactive, QPalette.HighlightedText,
                palette.color(QPalette.Active, QPalette.HighlightedText))
            self.list_tags.setPalette(palette)

        vlo_pattern = QVBoxLayout()
        hlo.addLayout(vlo_pattern)

        if self.rename_pattern:
            tit = 'Subfolder rename pattern'
        elif self.open_files_pattern:
            tit = 'Series indicator(s)'
        else:
            tit = 'Tag pattern'

        self.wid_pattern = TagPatternTree(
            self, title=tit, typestr=self.typestr, editable=editable)
        vlo_pattern.addWidget(self.wid_pattern)
        if self.rename_pattern or self.open_files_pattern:
            if editable:
                vlo_pattern.setSpacing(5)
            else:
                hlo.setSpacing(5)
            if self.rename_pattern:
                tit = 'File rename pattern'
            else:
                tit = 'File sort pattern'
            self.wid_pattern2 = TagPatternTree(
                self, title=tit, typestr=self.typestr,
                list_number=2, editable=editable)
            if editable:
                vlo_pattern.addWidget(self.wid_pattern2)
            else:
                hlo.addWidget(self.wid_pattern2)

    def fill_list_tags(self, modality, avoid_special_tags=False):
        """Find tags from tag_infos.yaml and fill list."""
        try:
            self.list_tags.clear()
            general_tags, included_tags = get_included_tags(
                modality, self.parent.tag_infos, avoid_special_tags=avoid_special_tags)
            self.list_tags.addItems(included_tags)
            if self.lock_on_general is False:
                for i in range(self.list_tags.count()):
                    if included_tags[i] in general_tags:
                        self.list_tags.item(i).setForeground(
                            QBrush(QColor(110, 148, 192)))
        except (RuntimeError, AttributeError):
            pass

    def double_click_tag(self):
        """Double click item = push item."""
        if self.rename_pattern is False:
            self.wid_pattern.push_tag()

    def update_data(self):
        """Fill pattern list."""
        self.wid_pattern.update_data()


class FormatDialog(ImageQCDialog):
    """Dialog to set format-string of tags in TagPatternFormat."""

    def __init__(self, format_string=''):
        super().__init__()
        self.setWindowTitle('Set format')
        self.cbox_decimals = QComboBox()
        self.cbox_padding = QComboBox()
        self.prefix = QLineEdit('')
        self.suffix = QLineEdit('')

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        hlo_decimals = QHBoxLayout()
        hlo_decimals.addWidget(QLabel('Number of decimals: '))
        dec_list = ['Auto'] + [str(i) for i in range(0, 10)]
        self.cbox_decimals.addItems(dec_list)
        self.cbox_decimals.setFixedWidth(100)
        hlo_decimals.addWidget(self.cbox_decimals)
        hlo_decimals.addStretch()
        vlo.addLayout(hlo_decimals)
        vlo.addSpacing(20)
        hlo_padding = QHBoxLayout()
        hlo_padding.addWidget(
            QLabel('0-padding (N characters): '))
        pad_list = ['Auto'] + [str(i) for i in range(2, 16)]
        self.cbox_padding.addItems(pad_list)
        self.cbox_padding.setFixedWidth(100)
        hlo_padding.addWidget(self.cbox_padding)
        hlo_padding.addStretch()
        vlo.addLayout(hlo_padding)

        hlo_prefix = QHBoxLayout()
        hlo_prefix.addWidget(QLabel("Prefix: "))
        hlo_prefix.addWidget(self.prefix)
        vlo.addLayout(hlo_prefix)
        hlo_suffix = QHBoxLayout()
        hlo_suffix.addWidget(QLabel("Suffix: "))
        hlo_suffix.addWidget(self.suffix)
        vlo.addLayout(hlo_suffix)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

        if format_string != '':
            prefix, format_str, suffix = get_format_strings(format_string)
            self.prefix.setText(prefix)
            self.suffix.setText(suffix)
            if '.' in format_str and 'f' in format_str:
                start = '.'
                end = 'f'
                dec_string = format_str[
                    format_str.find(start)+1:format_str.rfind(end)]
                self.cbox_decimals.setCurrentText(dec_string)
            if len(format_str) > 0:
                if format_str[0] == '0':
                    pos_dec = format_str.rfind('.')
                    if pos_dec != -1:
                        pad_string = format_str[1:pos_dec]
                    else:
                        pad_string = format_str[1:]
                    self.cbox_padding.setCurrentText(pad_string)

    def get_data(self):
        """Get formatting string.

        Returns
        -------
        return_string : str
            if prefix, suffix separated by | (prefix|formatstr|suffix)
            else only formatstr
            formatstr = string after {value in f-string to format the value
        """
        format_string = ''

        idx_dec = self.cbox_decimals.currentIndex()
        idx_pad = self.cbox_padding.currentIndex()
        if idx_dec > 0:
            format_string = '.' + self.cbox_decimals.currentText() + 'f'
        if idx_pad > 0:
            format_string = \
                '0' + self.cbox_padding.currentText() + format_string
        if format_string != '':
            format_string = ':' + format_string

        return_string = format_string
        if self.prefix != '' or self.suffix != '':
            return_string = '|'.join(
                [self.prefix.text(), format_string, self.suffix.text()])

        return return_string


class TagPatternEditDialog(ImageQCDialog):
    """Dialog for editing tag pattern."""

    def __init__(
            self, initial_pattern=None, modality='CT',
            title='', typestr='format',
            accept_text='Use', reject_text='Cancel',
            save_blocked=False):
        super().__init__()

        self.edited = False
        self.current_modality = modality
        self.status_label = QLabel('')
        self.current_template = initial_pattern
        self.current_labels = []

        if typestr == 'format':
            self.fname = 'tag_patterns_format'
        elif typestr == 'sort':
            self.fname = 'tag_patterns_sort'
        else:
            self.fname = ''

        self.setWindowTitle(title)

        vlo = QVBoxLayout()
        self.setLayout(vlo)
        hlo_temps = QHBoxLayout()
        vlo.addLayout(hlo_temps)
        hlo_temps.addWidget(QLabel('Select template:'))
        self.cbox_templist = QComboBox()
        self.cbox_templist.setFixedWidth(200)
        self.cbox_templist.activated.connect(
            self.update_clicked_template)
        if typestr != '':
            hlo_temps.addWidget(self.cbox_templist)
            toolb = QToolBar()
            hlo_temps.addWidget(toolb)
            act_add = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                'Add new tag pattern', self)
            act_add.triggered.connect(self.add)
            act_save = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
                'Save tag pattern', self)
            act_save.triggered.connect(self.save)
            if save_blocked:
                act_save.setEnabled(False)
            toolb.addActions([act_add, act_save])
            hlo_temps.addStretch()

        self.wid_tag_pattern = TagPatternWidget(self, typestr=typestr)
        vlo.addWidget(self.wid_tag_pattern)

        _, _, self.tag_infos = cff.load_settings(fname='tag_infos')
        self.update_from_yaml()

        vlo.addWidget(uir.HLine())
        vlo.addWidget(self.status_label)

        hlo_dlg_btns = QHBoxLayout()
        vlo.addLayout(hlo_dlg_btns)
        hlo_dlg_btns.addStretch()
        btn_close = QPushButton(accept_text)
        btn_close.clicked.connect(self.accept)
        hlo_dlg_btns.addWidget(btn_close)
        btn_cancel = QPushButton(reject_text)
        btn_cancel.clicked.connect(self.reject)
        hlo_dlg_btns.addWidget(btn_cancel)

    def update_from_yaml(self):
        """Refresh settings from yaml file."""
        self.lastload = time()
        _, _, self.templates = cff.load_settings(fname=self.fname)
        self.wid_tag_pattern.fill_list_tags(self.current_modality)
        self.refresh_templist()

    def refresh_templist(self, selected_id=0, selected_label=''):
        """Update the list of templates, and self.current...

        Parameters
        ----------
        selected_id : int, optional
            index to select in template list. The default is 0.
        selected_label : str, optional
            label to select in template list (override index)
            The default is ''.
        """
        self.current_labels = \
            [obj.label for obj
                in self.templates[self.current_modality]]

        if selected_label != '':
            tempno = self.current_labels.index(selected_label)
        else:
            tempno = selected_id
        tempno = max(tempno, 0)
        if tempno > len(self.current_labels)-1:
            tempno = len(self.current_labels)-1

        self.cbox_templist.blockSignals(True)
        self.cbox_templist.clear()
        self.cbox_templist.addItems(self.current_labels)
        if selected_label != '':
            self.update_current_template(selected_id=tempno)
            self.cbox_templist.setCurrentIndex(tempno)
        self.cbox_templist.blockSignals(False)
        self.wid_tag_pattern.update_data()

    def update_clicked_template(self):
        """Update data after new template selected (clicked)."""
        if self.edited:
            res = messageboxes.QuestionBox(
                self, title='Save changes?',
                msg='Save changes before changing template?')
            if res.exec():
                self.save(label=self.current_template.label)

        tempno = self.cbox_templist.currentIndex()
        self.update_current_template(selected_id=tempno)
        self.wid_tag_pattern.update_data()

    def update_current_template(self, selected_id=0):
        """Update self.current_template by label or id."""
        self.current_template = copy.deepcopy(
            self.templates[self.current_modality][selected_id])

    def get_pattern(self):
        """Get TagPattern from calling widget on Sort."""
        return self.current_template

    def add(self):
        """Add new template to list. Ask for new name and verify."""
        text, proceed = QInputDialog.getText(
            self, 'New label',
            'Name the new tag pattern')
        if proceed and text != '':
            if text in self.current_labels:
                QMessageBox.warning(
                    self, 'Label already in use',
                    'This label is already in use.')
            else:
                new_temp = copy.deepcopy(self.current_template)
                new_temp.label = text
                if self.templates[self.current_modality][0].label == '':
                    self.templates[self.current_modality][0] = new_temp
                else:
                    self.templates[self.current_modality].append(new_temp)

                self.current_template = new_temp
                self.current_labels.append(text)
                self.save(label=text)
                self.refresh_templist(selected_label=text)

    def save(self, label=None):
        """Save button pressed or specific save on label."""
        if self.current_template.label == '':
            self.add()
        else:
            if label is False or label is None:
                idx = self.cbox_templist.currentIndex()
            else:
                idx = self.current_labels.index(label)

            try:
                self.templates[self.current_modality][idx] = \
                    copy.deepcopy(self.current_template)
            except IndexError:
                pass

            proceed = cff.verify_config_folder(self)
            if proceed:
                proceed, errmsg = cff.check_save_conflict(
                    self.fname, self.lastload)
                if errmsg != '':
                    proceed = messageboxes.proceed_question(self, errmsg)
                if proceed:
                    save_ok, path = cff.save_settings(
                        self.templates, fname=self.fname)
                    if save_ok:
                        self.status_label.setText(
                            f'Changes saved to {path}')
                        self.flag_edit(False)
                    else:
                        QMessageBox.warning(
                            self, 'Failed saving', f'Failed saving to {path}')

    def flag_edit(self, flag=True):
        """Indicate some change."""
        if flag:
            self.edited = True
            self.status_label.setText('**Unsaved changes**')
        else:
            self.edited = False
            self.status_label.setText('')


class TagPatternTreeTestDCM(TagPatternTree):
    """Widget for test DCM. Reusable for all modalities."""

    def __init__(self, parent):
        self.main = parent.main
        self.parent = parent
        self.fname = 'tag_patterns_format'
        self.templates = None

        tit = ('Extract data from DICOM header for each image '
               'using tag pattern.')
        super().__init__(
            parent, title=tit, typestr='format', editable=False)
        toolb = QToolBar()
        toolb.setOrientation(Qt.Vertical)
        act_edit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit tag pattern', self)
        act_edit.triggered.connect(self.edit)
        act_import = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}import.png'),
            'Import saved tag pattern', self)
        act_import.triggered.connect(self.import_tagpattern)
        act_save = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Save tag pattern as Tag Pattern Format ...', self)
        act_save.triggered.connect(self.save_as)

        if self.main.save_blocked:
            act_save.setEnabled(False)
        toolb.addActions([act_edit, act_import, act_save])
        self.hlo.addWidget(toolb)
        self.hlo.addStretch()

        self.current_template = self.main.current_paramset.dcm_tagpattern

    def edit(self):
        """Edit tag pattern by dialog."""
        dlg = TagPatternEditDialog(
            initial_pattern=self.current_template,
            modality=self.main.current_modality,
            title='Edit tag pattern for test output',
            typestr='format',
            accept_text='Use',
            reject_text='Cancel')
        res = dlg.exec()
        if res:
            self.current_template = dlg.get_pattern()
            self.update_data()
            self.main.current_paramset.dcm_tagpattern = self.current_template
            self.parent.flag_edit(True)

    def save_as(self):
        """Save tag pattern as new tag pattern format."""
        text, proceed = QInputDialog.getText(
            self, 'Save tag pattern as...', 'Label: ')
        if proceed and text != '':
            _, _, self.templates = cff.load_settings(fname='tag_patterns_format')
            curr_labels = [x.label for x in self.templates[self.main.current_modality]]
            if text in curr_labels:
                QMessageBox.warning(
                    self, 'Label already in use',
                    'This label is already in use.')
            else:
                new_pattern = copy.deepcopy(self.current_template)
                new_pattern.label = text
                self.templates[self.main.current_modality].append(
                    new_pattern)
                proceed = cff.verify_config_folder(self)
                if proceed:
                    ok_save, path = cff.save_settings(
                        self.templates, fname='tag_patterns_format')
                    if ok_save is False:
                        QMessageBox.warning(
                            self, 'Failed saving',
                            f'Failed saving to {path}')

    def import_tagpattern(self):
        """Import tagpattern from tag_patterns_format."""
        _, _, templates = cff.load_settings(fname='tag_patterns_format')
        curr_labels = [x.label for x in templates[self.main.current_modality]]
        text, proceed = QInputDialog.getItem(self, 'Select tag pattern',
                                             'Tag pattern format:', curr_labels)
        if proceed and text != '':
            idx = curr_labels.index(text)
            new_template = copy.deepcopy(
                templates[self.main.current_modality][idx])
            self.main.current_paramset.dcm_tagpattern = new_template
            self.parent.flag_edit(True)
            self.current_template = new_template
            self.update_data()

    def update_data(self):
        """Update table_pattern with data from active parameterset."""
        self.table_pattern.clear()
        list_tags = self.current_template.list_tags
        if len(list_tags) > 0:
            for rowno, tagname in enumerate(list_tags):
                infotext = self.current_template.list_format[rowno]
                row_strings = [tagname, infotext]
                item = QTreeWidgetItem(row_strings)
                self.table_pattern.addTopLevelItem(item)
