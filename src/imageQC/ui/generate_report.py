#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface for dialog Generate report.

@author: Ellen Wasbo
"""
import copy
import os
import webbrowser
from io import BytesIO
import base64
from datetime import datetime
from time import time
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QToolBar,
    QLabel, QPushButton, QAction, QTreeWidget, QTreeWidgetItem,
    QComboBox, QPlainTextEdit, QSpinBox, QCheckBox, QLineEdit,
    QFileDialog, QInputDialog, QMessageBox
    )

# imageQC block start
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.ui import reusable_widgets as uir
from imageQC.ui import messageboxes
from imageQC.ui.ui_main_result_tabs import ResultPlotCanvas
from imageQC.ui import ui_main_result_tabs
from imageQC.scripts.calculate_qc import format_result_table, get_image_names
from imageQC.config import config_classes as cfc
from imageQC.config import config_func as cff
from imageQC.config.iQCconstants import (
    ENV_ICON_PATH, QUICKTEST_OPTIONS, USERNAME, VERSION)
# imageQC block end


# Equivalent A4 paper dimensions in pixels at 300 DPI 2480 pixels x 3508 pixels
DEFAULT_HEAD = '''
    <head>
        <style>
            @page {
                width: 2480px;
                height: 3508px;
                margin: 50px;
                @bottom-right {
                    padding: 10px;
                    content: "Page " counter(page) " / " counter(pages);}
                }
            @media print {
                .pagebreak { page-break-before: always; }
                :lastchild {page-break-after: auto;}
                }
            body {
                font-family: Arial, Verdana, sans-serif;
                font-size: 200%;
                width: 2480px; height: 3580px; margin: 50px;}
            table.test_header {
                width: 100%;
                color: white;
                text-align: left;
                background-color: black;
                border-radius:10px;
                padding: 5px;
                }
            table.element_frame {
                width: 100%;
                border: 1px solid #dddddd;
                text-align: left;
                padding: 3px;
                th {
                    color: white;
                    background-color: #6e94c0;
                    border-radius:10px;
                    padding: 5px;
                    }
                }
            table.result_table {
                border-collapse: collapse;
                border-radius:0px;}
            td.result_table, th.result_table {
                border: 1px solid #dddddd;
                text-align: center;
                padding: 3px;}
            tr.result_table:nth-child(even) {
                background-color: #dddddd;}
            table.image_table {
                border: 1px solid #dddddd;}
        </style>
    </head>
'''

class GenerateReportDialog(ImageQCDialog):
    """GUI setup for the Generate report dialog."""

    def __init__(self, main):
        super().__init__()
        self.fname = 'report_templates'
        self.main = main
        self.empty_template = cfc.ReportTemplate()
        self.current_template = copy.deepcopy(self.empty_template)
        self.variants = [
            'html_table_row', 'html_element', 'result_table',
            'result_plot', 'result_image', 'image']
        self.edited = False
        self.lbl_edit = QLabel('')
        self.plot_canvas = ResultPlotCanvas(self.main)
        self.wid_result_image = ui_main_result_tabs.ResultImageWidget(self.main)

        widget_test_tabs = self.main.stack_test_tabs.currentWidget()
        self.test_names = [
            widget_test_tabs.tabText(i)
            for i in range(widget_test_tabs.count())]
        self.test_codes = [*QUICKTEST_OPTIONS[self.main.current_modality]]

        self.setWindowTitle('Generate report')
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        hlo_select_template = QHBoxLayout()
        vlo.addLayout(hlo_select_template)
        self.cbox_template = QComboBox()
        self.cbox_template.currentIndexChanged.connect(
            self.update_current_template)
        self.cbox_template.setMinimumWidth(300)
        hlo_select_template.addWidget(QLabel('Select template'))
        hlo_select_template.addWidget(self.cbox_template)

        act_add_template = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Save current template as new template', self)
        act_add_template.triggered.connect(self.add_current_template)
        act_save_template = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Overwrite current template', self)
        act_save_template.triggered.connect(self.save_current_template)
        if main.save_blocked:
            act_save_template.setEnabled(False)
        act_settings = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}gears.png'),
            'Manage templates', self)
        act_settings.triggered.connect(
            lambda: main.run_settings(
                initial_view='Report templates',
                initial_template_label=self.current_template.label))

        toolb = QToolBar()
        toolb.addActions([act_add_template, act_save_template, act_settings])
        hlo_select_template.addWidget(toolb)
        btn_edit_head = QPushButton('Edit <head><style>')
        btn_edit_head.clicked.connect(self.edit_head)
        hlo_select_template.addWidget(btn_edit_head)
        hlo_select_template.addStretch()

        vlo.addWidget(QLabel(
            'Add result elements and standard text to a report template'))

        hlo_table = QHBoxLayout()
        vlo.addLayout(hlo_table)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(
            ['Type', 'Testcode', 'Header+Content', 'Note'])
        self.table.setColumnWidth(0, 170)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 500)
        self.table.setColumnWidth(3, 200)
        self.table.setMinimumHeight(500)
        self.table.setMinimumWidth(970)
        self.table.setAlternatingRowColors(True)
        hlo_table.addWidget(self.table)

        self.toolbar = QToolBar()
        self.toolbar.setOrientation(Qt.Vertical)
        hlo_table.addWidget(self.toolbar)
        act_clear = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}clear.png'),
            'Clear report template', self)
        act_clear.triggered.connect(self.clear_template)
        act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add element to report template', self)
        act_add.triggered.connect(
            lambda: self.add_element(duplicate=False))
        act_edit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit element', self)
        act_edit.triggered.connect(self.edit_element)
        act_duplicate = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}duplicate.png'),
            'Duplicate selected element', self)
        act_duplicate.triggered.connect(
            lambda: self.add_element(duplicate=True))
        act_note = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'Add (or edit) note to element', self)
        act_note.triggered.connect(self.add_note)
        act_up = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
            'Move element up', self)
        act_up.triggered.connect(self.move_element_up)
        act_down = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
            'Move element down', self)
        act_down.triggered.connect(self.move_element_down)
        act_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete element', self)
        act_delete.triggered.connect(self.delete_element)

        if main.save_blocked:
            act_save_template.setEnabled(False)
            act_add_template.setEnabled(False)

        self.toolbar.addActions(
            [act_clear, act_add, act_edit, act_duplicate, act_note,
             act_up, act_down, act_delete])

        hlo_dlg_btns = QHBoxLayout()
        vlo.addLayout(hlo_dlg_btns)
        hlo_dlg_btns.addWidget(self.lbl_edit)
        hlo_dlg_btns.addStretch()
        btn_generate = QPushButton('Generate report')
        btn_generate.clicked.connect(self.generate_report)
        hlo_dlg_btns.addWidget(btn_generate)
        btn_close = QPushButton('Close window')
        btn_close.clicked.connect(self.accept)
        hlo_dlg_btns.addWidget(btn_close)

        # update from yaml
        self.lastload = time()
        _, _, self.modality_dict = cff.load_settings(
            fname='report_templates')
        self.fill_template_list()

        self.tags_active, self.values_active = self.get_active_tags()

    def get_active_tags(self):
        tags = []
        values = []
        if len(self.main.imgs) > 0:
            img_info = self.main.imgs[self.main.gui.active_img_no]
            info_list = img_info.info_list_general
            info_list.extend(img_info.info_list_modality)
            list_tuples = [info.split(': ') for info in info_list]
            attr_list = [elem[0] for elem in list_tuples]
            value_list = [elem[1] for elem in list_tuples]
            sort_order = np.argsort(attr_list)
            tags = np.array(attr_list)[sort_order].tolist()
            values = np.array(value_list)[sort_order].tolist()
        return (tags, values)

    def get_tag_value(self, imgno, attribute):
        value_string = '-'
        if len(self.main.imgs) > imgno:
            img_info = self.main.imgs[imgno]
            info_list = img_info.info_list_general
            info_list.extend(img_info.info_list_modality)
            list_tuples = [info.split(': ') for info in info_list]
            attr_list = [elem[0] for elem in list_tuples]
            value_list = [elem[1] for elem in list_tuples]
            try:
                tag_no = attr_list.index(attribute)
                value_string = value_list[tag_no]
            except ValueError:
                pass
        return value_string

    def fill_template_list(self, set_label=''):
        """Fill list of templates for current modality."""
        self.cbox_template.blockSignals(True)
        self.cbox_template.clear()
        labels = [temp.label for temp
                  in self.modality_dict[self.main.current_modality]]
        self.cbox_template.addItems(labels)
        if set_label in labels:
            set_index = labels.index(set_label)
        else:
            set_index = 0
        self.cbox_template.setCurrentIndex(set_index)
        self.cbox_template.blockSignals(False)
        self.update_current_template()
        self.lastload = time()

    def flag_edit(self, flag=True):
        """Indicate some change."""
        if flag:
            self.edited = True
            self.lbl_edit.setText('*')
        else:
            self.edited = False
            self.lbl_edit.setText('')

    def get_selected_element(self):
        flattened_elements = []
        element_rows = []
        element_cols = []
        element_top_rows = []
        rowno = 0
        toprow = 0
        for i, element in enumerate(self.current_template.elements):
            if isinstance(element, list):
                pass  # already added
            elif element.variant == 'html_table_row':
                flattened_elements.append(element)
                element_list = self.current_template.elements[i + 1]
                flattened_elements.extend(element_list)
                element_rows.append(rowno)
                element_rows.extend([rowno + 1] * len(element_list))
                element_cols.append(None)
                element_cols.extend(list(range(len(element_list))))
                element_top_rows.extend([toprow] * (len(element_list) + 1))
                rowno += 2
                toprow += 1
            else:
                flattened_elements.append(element)
                element_rows.append(rowno)
                element_cols.append(None)
                element_top_rows.append(toprow)
                rowno += 1
                toprow += 1

        sel_idxs = self.table.selectedIndexes()
        sel = self.table.selectedItems()
        row = 0
        if len(sel) > 0:
            if sel[0].parent() is not None:  # child is selected
                toprow = self.table.indexOfTopLevelItem(sel[0].parent())
                row_parent = element_top_rows.index(toprow)
                row = row_parent + sel[0].parent().indexOfChild(sel[0]) + 1
            else:  # top level row
                toprow = sel_idxs[0].row()
                row = element_top_rows.index(toprow)
        else:
            row = element_rows[-1]

        try:
            res = (
                flattened_elements[row], element_rows[row], element_cols[row])
        except IndexError:
            res = (None, -1, None)
        return res

    def clear_template(self):
        self.current_template = copy.deepcopy(self.empty_template)
        self.fill_table()

    def add_element(self, duplicate=False):
        element = None
        if duplicate is False or duplicate is None:
            dlg = AddEditElementDialog(self, self.variants,
                                       element=cfc.ReportElement())
            if dlg.exec():
                element = dlg.get_element()
        sel_elem, row, col = self.get_selected_element()
        if duplicate:
            if sel_elem is None:
                duplicate = False

        if element or duplicate:
            sel_elem, row, col = self.get_selected_element()
            new_row = row + 1
            sel_is_table = False
            try:
                if sel_elem.variant == 'html_table_row':
                    new_row = row + 2
                    sel_is_table = True
            except AttributeError:
                pass

            if col is not None or sel_is_table:  # table element selected
                if duplicate:
                    if sel_is_table:  # duplicate html_table_row and sub elements
                        sel_table_copy = copy.deepcopy(
                            self.current_template.elements[row])
                        sel_table_copy2 = copy.deepcopy(
                            self.current_template.elements[row+1])
                        breakpoint()
                        self.current_template.elements.insert(
                            new_row, sel_table_copy)
                        self.current_template.elements.insert(
                            new_row+1, sel_table_copy2)
                    else:  # duplicate sub element
                        self.current_template.elements[row].insert(
                            col+1, copy.deepcopy(sel_elem))
                else:
                    if element.variant == 'html_table_row':
                        # force added after previous row
                        self.current_template.elements[new_row:new_row] = ([
                            element, []])
                    else:
                        dlg = messageboxes.QuestionBox(
                            parent=self, title='Add element',
                            msg='Add element to selected table or as next element?',
                            yes_text='Add to table',
                            no_text='Add as next element')
                        yes = dlg.exec()
                        if yes:
                            if col is None:
                                col = -1
                                row += 1
                            self.current_template.elements[row].insert(
                                col+1, element)
                        else:
                            self.current_template.elements.insert(
                                new_row, element)
            else:
                if duplicate:
                    self.current_template.elements.insert(
                        new_row, copy.deepcopy(sel_elem))
                else:
                    if element.variant == 'html_table_row':
                        self.current_template.elements[new_row:new_row] = ([
                            element, []])
                    else:
                        self.current_template.elements.insert(new_row, element)
            self.fill_table()
            self.flag_edit()

    def edit_element(self):
        sel_elem, row, col = self.get_selected_element()
        if row != -1:
            dlg = AddEditElementDialog(self, self.variants,
                                       element=sel_elem)
            if dlg.exec():
                element = dlg.get_element()
                if col is None:
                    self.current_template.elements[row] = element
                else:
                    self.current_template.elements[row][col] = element
                self.fill_table()
                self.flag_edit()

    def add_note(self):
        sel_elem, row, col = self.get_selected_element()
        if row != -1:
            dlg = AddEditNoteDialog(self, sel_elem)
            if dlg.exec():
                element = dlg.get_element()
                if col is None:
                    self.current_template.elements[row] = element
                else:
                    self.current_template.elements[row][col] = element
                self.fill_table()
                self.flag_edit()

    def move_element_up(self):
        sel_elem, row, col = self.get_selected_element()
        if col == 0:  # first row in html_table_row
            pass
        elif row == -1:
            pass
        else:
            if col is not None:  # in html_table_row
                table_list = self.current_template.elements[row]
                pop_elem = self.current_template.elements[row].pop(col)
                self.current_template.elements[row].insert(
                    col - 1, pop_elem)
            elif sel_elem.variant == 'html_table_row':
                table_elem = self.current_template.elements.pop(row)
                table_list = self.current_template.elements.pop(row)
                new_row = row - 1
                try:
                    if isinstance(
                            self.current_template.elements[row - 1], list):
                        new_row = row - 2
                except IndexError:
                    pass
                self.current_template.elements.insert(new_row, table_elem)
                self.current_template.elements.insert(new_row + 1, table_list)
            else:
                popelem = self.current_template.elements.pop(row)
                self.current_template.elements.insert(row - 1, popelem)
            self.fill_table()
            self.flag_edit()

    def move_element_down(self):
        sel_elem, row, col = self.get_selected_element()
        if row != -1:
            if col is not None:  # in html_table_row
                table_list = self.current_template.elements[row]
                if col >= len(table_list) - 1:
                    pass  # last in row
                else:
                    pop_elem = self.current_template.elements[row].pop(col)
                    self.current_template.elements[row].insert(
                        col + 1, pop_elem)
            elif sel_elem.variant == 'html_table_row':
                table_elem = self.current_template.elements.pop(row)
                table_list = self.current_template.elements.pop(row)
                new_row = row + 1
                try:
                    nextelem = self.current_template.elements[row + 1]
                    if nextelem.variant == 'html_table_row':
                        new_row = row + 2
                except IndexError:
                    pass
                self.current_template.elements.insert(new_row, table_elem)
                self.current_template.elements.insert(new_row + 1, table_list)
            else:
                popelem = self.current_template.elements.pop(row)
                self.current_template.elements.insert(row + 1, popelem)
            self.fill_table()
            self.flag_edit()

    def delete_element(self):
        sel_elem, row, col = self.get_selected_element()
        if row != -1:
            proceed = True
            is_table = False
            if sel_elem.variant == 'html_table_row':
                is_table = True
                if len(self.current_template.elements[row + 1]) > 0:
                    proceed = messageboxes.proceed_question(
                        self, 'Delete row of elements?')

            if proceed:
                if col is None:
                    self.current_template.elements.pop(row)
                    if is_table:
                        self.current_template.elements.pop(row)  # also delete list
                else:
                    self.current_template.elements[row].pop(col)
            self.fill_table()
            self.flag_edit()

    def edit_head(self):
        dlg = EditHeadDialog(self, self.current_template)
        if dlg.exec():
            self.current_template = dlg.get_template()
            self.flag_edit()

    def update_current_template(self):
        """Set current_template according to selected label."""
        if self.edited and self.current_template.label != '':
            self.ask_to_save_changes(before_select_new=True)

        self.main.start_wait_cursor()
        label = self.cbox_template.currentText()
        if label == '':
            self.current_template = copy.deepcopy(self.empty_template)
        else:
            try:
                template_id = self.cbox_template.currentIndex()
                self.current_template = copy.deepcopy(
                    self.modality_dict[self.main.current_modality][template_id])
            except IndexError:
                self.cbox_template.setCurrentIndex(0)
                self.current_template = copy.deepcopy(self.empty_template)
        self.fill_table()
        self.flag_edit(False)
        self.main.stop_wait_cursor()

    def add_current_template(self):
        """Add current template."""
        text, proceed = QInputDialog.getText(
            self, 'New label', 'Name the new template                  ')
        if proceed and text != '':
            templates = self.modality_dict[self.main.current_modality]
            current_labels = [obj.label for obj in templates]
            if text in current_labels:
                QMessageBox.warning(self, 'Label already in use',
                                    'This label is already in use.')
            else:
                self.current_template.label = text
                if templates[0].label == '':
                    self.modality_dict[self.main.current_modality] = [
                        copy.deepcopy(self.current_template)]
                else:
                    self.modality_dict[self.main.current_modality].append(
                        copy.deepcopy(self.current_template))
                self.save(new_added=True)

    def save_current_template(self, before_select_new=False):
        """Overwrite selected Paramset or QuickTest Template if any, else add new."""
        if self.cbox_template.currentText() == '':
            self.add_current_template()
        else:
            if before_select_new:
                # id corresponding to label of current_template (previous selected)
                label = self.current_template.label
                labels = [
                    temp.label for temp
                    in self.modality_dict[self.main.current_modality]]
                template_id = labels.index(label)
            else:
                template_id = self.cbox_template.currentIndex()
            self.modality_dict[
                self.main.current_modality][template_id] = copy.deepcopy(
                    self.current_template)
            self.save()

    def save(self, new_added=False):
        """Save to file."""
        proceed = cff.verify_config_folder(self)
        if proceed:
            proceed, errmsg = cff.check_save_conflict(self.fname, self.lastload)
            if errmsg != '':
                proceed = messageboxes.proceed_question(self, errmsg)
            if proceed:
                proceed, path = cff.save_settings(
                    self.modality_dict, fname=self.fname)
                if proceed:
                    self.lbl_edit.setText('')
                    self.lastload = time()
                    self.flag_edit(False)
                    if new_added:
                        self.fill_template_list(set_label=self.current_template.label)
                else:
                    QMessageBox.warning(
                        self, 'Failed saving', f'Failed saving to {path}')

    def ask_to_save_changes(self, before_select_new=False):
        """Ask user if changes to current parameter set should be saved."""
        reply = QMessageBox.question(
            self, 'Unsaved changes',
            f'Save changes to {self.fname}?',
            QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.save_current_template(before_select_new=before_select_new)
        else:
            self.flag_edit(False)

    def fill_table(self):
        """Fill table with original and generated names."""
        self.table.clear()
        for i, element in enumerate(self.current_template.elements):
            if isinstance(element, list):
                pass  #already added
            elif element.variant == 'html_table_row':
                top_item = QTreeWidgetItem([
                    'html_table_row', '', element.caption, element.note])
                for sub in self.current_template.elements[i + 1]:
                    row_strings = [sub.variant, sub.testcode,
                                   sub.caption + sub.text, sub.note]
                    child_item = QTreeWidgetItem(row_strings)
                    top_item.addChild(child_item)
                self.table.addTopLevelItem(top_item)
            else:
                row_strings = [element.variant, element.testcode,
                               element.caption + element.text, element.note]
                item = QTreeWidgetItem(row_strings)
                self.table.addTopLevelItem(item)
        self.table.expandAll()

    def frame_element(self, html_code, element):
        caption = ''
        testcode = ''
        if not isinstance(element, list):
            caption = element.caption
            testcode = element.testcode
        if caption == '' and testcode != '':
            if element.text != 'result_table':
                caption = element.text

        if caption:  # frame element
            html_frame = [
                '<br><table class="element_frame">',
                f'<tr><th>{caption}</th></tr>',
                f'<tr><td>{html_code}</td></tr>',
                '</table>']
            if element.note:
                html_note = f'<tr><td>{element.note}</td></tr>'
                idx = 2 if element.note_pos == 'before' else 3
                html_frame.insert(idx, html_note)
            html_code = "\n".join(html_frame)
        return html_code

    def generate_report(self):
        def get_header_pr_top_element():
            headers = []
            previous_code = ''
            for element in self.current_template.elements:
                if isinstance(element, list):
                    testcodes = []
                    for sub in element:
                        if sub.testcode != '':
                            if sub.testcode not in testcodes:
                                testcodes.append(sub.testcode)
                    this_code = testcodes
                else:
                    this_code = element.testcode
                if this_code == previous_code:
                    this_code = ''
                else:
                    previous_code = this_code
                headers.append(this_code)

            return headers

        def generate_test_header(testcode):
            test_header = ''
            if testcode != '':
                if isinstance(testcode, list):
                    names = []
                    for code in testcode:
                        idx = self.test_codes.index(code)
                        names.append(self.test_names[idx])
                    header = ' / '.join(names)
                else:
                    idx = self.test_codes.index(testcode)
                    header = self.test_names[idx]
                test_header = (
                    f'<br><table class="test_header"><tr><th>{header}'
                    '</th></tr></table>')
            return test_header

        fname, _ = QFileDialog.getSaveFileName(
            self, 'Choose a filename to save to', '',
            'HTML (*.html)', 'HTML (*.html)')
        if fname:
            max_progress = len(self.current_template.elements)
            progress_modal = uir.ProgressModal(
                "Generating report...", "Cancel",
                0, max_progress, self, minimum_duration=0)
            html_head = DEFAULT_HEAD if self.current_template.htmlhead == '' else self.current_template.htmlhead
            html = ['<!DOCTYPE html>','<html>', html_head]
            testcodes = get_header_pr_top_element()
            for i, element in enumerate(self.current_template.elements):
                progress_modal.setLabelText(
                    f'Generating report for element {i}/{max_progress}')
                if isinstance(element, list):
                    pass  # already added
                else:
                    html_header = ''
                    if element.variant == 'html_table_row':
                        html_this = self.add_table_of_html_elements(
                            self.current_template.elements[i + 1])
                        html_header = generate_test_header(testcodes[i + 1])
                    else:
                        html_this = self.add_html_element(element)
                        html_header = generate_test_header(testcodes[i])
                    if html_this:
                        if html_header:
                            html.append(html_header)
                        html_this = self.frame_element(html_this, element)
                        html.append(html_this)
                        progress_modal.setValue(i)
                    if progress_modal.wasCanceled():
                        break
            progress_modal.setValue(max_progress)
            html.extend(['</body>', '</html>'])
            with open(fname, "w") as html_file:
                html_file.write("\n".join(html))
            webbrowser.open(url='file://' + fname, new=1)

    def result_table_to_html(self, headers, values, pr_image, element):
        html_lines = []
        if len(values) > 0:
            html_lines.append(
                f'<table class="result_table" style="width:{element.width}%">')
            html_lines.append('<tr class="result_table">')
            image_names = []
            if pr_image and element.include_image_name == True:
                html_lines.append('<th class="result_table">Image</th>')
                image_names = get_image_names(self.main)
                marked_this = self.main.get_marked_imgs_current_test()
                image_names = [name for i, name in enumerate(image_names)
                               if i in marked_this]
            html_lines.extend([f'<th class="result_table">{header}</th>'
                               for header in headers])
            html_lines.append('</tr>')
            values_formatted = format_result_table(
                self.main, element.testcode, values, headers)
            for rowno, row in enumerate(values_formatted):
                html_lines.append('<tr class="result_table">')
                if pr_image and element.include_image_name == True:
                    html_lines.append(
                        f'<td class="result_table">{image_names[rowno]}</td>')
                html_lines.extend([f'<td class="result_table">{val}</td>'
                                   for val in row])
                html_lines.append('</tr>')
            html_lines.append('</table>')
        if len(html_lines) > 0:
            html_code = "\n".join(html_lines)
        else:
            html_code = ''
        return html_code

    def add_figure(self, element, width_px=0, image_name=''):
        html_code = ''
        buffer = BytesIO()
        if element.variant == 'result_plot':
            self.plot_canvas.plot(selected_text=element.text)
            self.plot_canvas.fig.savefig(buffer, format='png')
        elif element.variant == 'result_image':
            self.wid_result_image.canvas.result_image_draw(
                selected_text=element.text)
            self.wid_result_image.canvas.fig.savefig(buffer, format='png')
        elif element.variant == 'image':
            self.main.wid_image_display.canvas.fig.savefig(
                buffer, format='png')

        buffer.seek(0)
        img = base64.b64encode(buffer.getbuffer()).decode('utf-8')
        if width_px > 0:
            wtxt = f'width="{width_px}px" '
        else:
            wtxt = f'width="{element.width}%" '
        html_code = (
            f'<img {wtxt}src="data:image/png;base64,{img}">')
        if image_name != '' and element.include_image_name:
            html_code = f'{html_code}<br><div style="text-align: center;">{image_name}</div>'

        return html_code

    def convert_coded_text(self, text):
        if '#TODAY' in text:
            today = datetime.today().strftime("%d.%m.%Y")
            text = text.replace('#TODAY', today)
        text = text.replace('#USERNAME', USERNAME)
        text = text.replace('#VERSION', VERSION)
        if '#IMAGEQCICON' in text:
            icon_html = '<img src="https://github.com/EllenWasbo/imageQCpy/blob/main/src/imageQC/icons/iQC_icon128.png?raw=true">'
            text = text.replace('#IMAGEQCICON', icon_html)
        if '#DICOM[' in text:
            text_split = text.split('#DICOM[')
            for sub_txt in text_split[1:]:
                sub_txt_split = sub_txt.split(']')
                attr = sub_txt_split[0]
                nmb = sub_txt_split[1][1:]
                attr_no = self.tags_active.index(attr)
                val = '-'
                if nmb == 'active':
                    val = self.values_active[attr_no]
                else:
                    try:
                        imgno = int(nmb)
                        val = self.get_tag_value(imgno, attr)
                    except TypeError:
                        pass
                text = text.replace(f'#DICOM[{attr}][{nmb}]', val)
        return text

    def add_html_element(self, element, full_width=2480, margin=50):
        html_this = ''
        if element.variant == 'html_element':
            if '#' in element.text:
                html_this = self.convert_coded_text(element.text)
            else:
                html_this = element.text

        elif element.variant == 'result_table':
            self.main.update_current_test(
                reset_index=False, refresh_display=False,
                set_test=element.testcode)
            try:
                suffix = ''
                if element.text == 'Supplement table':
                    suffix = '_sup'
                testcode = element.testcode
                headers = self.main.results[testcode][f'headers{suffix}']
                values = self.main.results[testcode][f'values{suffix}']
                pr_image = self.main.results[testcode][f'pr_image{suffix}']
                html_this = self.result_table_to_html(
                    headers, values, pr_image, element)#.width, testcode)
            except KeyError:
                pass
        elif element.variant in ['result_plot', 'result_image']:
            self.main.update_current_test(
                reset_index=False, refresh_display=False,
                set_test=element.testcode)
            img_nos = []
            if element.all_images:
                marked_this = self.main.get_marked_imgs_current_test()
                if len(marked_this) == 0:
                    img_nos = list(range(len(self.main.imgs)))
                else:
                    img_nos = marked_this
            else:  # specific image_number:
                img_nos = [element.image_number]
            image_names = get_image_names(self.main)
            if len(img_nos) == 1:
                self.main.set_active_img(img_nos[0])
                html_this = self.add_figure(element,
                                            image_name=image_names[img_nos[0]])
            else:
                html_this = ['<table class="image_table"><tr>']
                total_width = element.width
                width_px = (full_width - 2*margin) * element.width / 100
                
                for img_no in img_nos:
                    html_this.append(
                        '<td valign="middle" align="center">')
                    self.main.set_active_img(img_no)
                    html_this.append(
                        self.add_figure(element, width_px=width_px,
                                        image_name=image_names[img_no]))
                    html_this.append('</td>')
                    total_width += element.width
                    if total_width > 100:
                        total_width = element.width
                        html_this.append('</tr><tr>')
                html_this.append('</tr></table>')
                html_this = "\n".join(html_this)
        elif element.variant == 'image':
            self.main.update_current_test(
                reset_index=False, refresh_display=False,
                set_test='DCM')  # no ROIs annotated
            if element.all_images:
                img_nos = list(range(len(self.main.imgs)))
            else:
                if element.image_number < len(self.main.imgs):
                    img_nos = [element.image_number]
                else:
                    img_nos = []

            image_names = get_image_names(self.main)
            if len(img_nos) == 1:
                self.main.set_active_img(img_nos[0])
                html_this = self.add_figure(element,
                                image_name=image_names[img_nos[0]])
            elif len(img_nos) > 1:
                html_this = ['<table class="image_table"><tr>']
                n_pr_row = element.width
                single_width = 100 / n_pr_row
                width_px = (full_width - 2*margin) * single_width / 100
                
                for idx, img_no in enumerate(img_nos):
                    html_this.append(
                        '<td valign="middle" align="center">')
                    self.main.set_active_img(img_no)
                    html_this.append(
                        self.add_figure(element, width_px=width_px,
                                        image_name=image_names[img_no]))
                    html_this.append('</td>')
                    next_col_no = (idx + 1) % n_pr_row
                    if next_col_no == 0:
                        html_this.append('</tr><tr>')
                html_this.append('</tr></table>')
                html_this = "\n".join(html_this)

        return html_this

    def add_table_of_html_elements(self, elements):
        html_this = ''
        if len(elements) > 0:
            html_this = []
            html_this.append('<table><tr>')
            for element in elements:
                html_this.append('<td>')
                html_this.append(self.add_html_element(element))
                html_this.append('</td>')
            html_this.append('</tr></table>')
            html_this = "\n".join(html_this)
        return html_this

    def generate_dicom_hash(self):
        if len(self.main.imgs) > 0:
            dlg = GenerateDicomHashDialog(self)
            dlg.exec()
        else:
            QMessageBox.warning(
                self, 'Missing DICOM info',
                'Please load at least one image of the desired modality '
                'to display the availabel DICOM header tags.')


class AddEditElementDialog(ImageQCDialog):
    """Dialog to add element to report."""

    def __init__(self, parent, variants, element=cfc.ReportElement()):
        super().__init__(parent=parent)
        self.parent = parent
        self.element = element
        self.variants = variants

        self.setWindowTitle('Add or edit element to report')
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        hlo_variant = QHBoxLayout()
        vlo.addLayout(hlo_variant)
        self.cbox_variant = QComboBox()
        self.cbox_variant.addItems(self.variants)
        self.cbox_variant.currentIndexChanged.connect(self.update_variant)
        hlo_variant.addWidget(QLabel('Element type'))
        hlo_variant.addWidget(self.cbox_variant)

        self.stack_variants = QStackedWidget()
        wid_table = QWidget()
        wid_html_element = QWidget()
        wid_result_element = QWidget()
        wid_image_element = QWidget()
        self.stack_variants.addWidget(wid_table)
        self.stack_variants.addWidget(wid_html_element)
        self.stack_variants.addWidget(wid_result_element)
        self.stack_variants.addWidget(wid_image_element)
        vlo.addWidget(self.stack_variants)

        # Stack table
        vlo_table = QVBoxLayout()
        wid_table.setLayout(vlo_table)
        self.txt_table_header = QLineEdit(self)
        hlo_table_header = QHBoxLayout()
        vlo_table.addLayout(hlo_table_header)
        hlo_table_header.addWidget(QLabel('Header'))
        hlo_table_header.addWidget(self.txt_table_header)
        vlo_table.addWidget(QLabel('Add table (one row) of elements'))

        # Stack html code
        vlo_html = QVBoxLayout()
        wid_html_element.setLayout(vlo_html)
        self.txt_header = QLineEdit(self)
        self.txt_html = QPlainTextEdit(self)
        self.txt_html.textChanged.connect(self.preview_html_element)
        hlo_header = QHBoxLayout()
        vlo_html.addLayout(hlo_header)
        hlo_header.addWidget(QLabel('Header'))
        hlo_header.addWidget(self.txt_header)
        vlo_html.addWidget(self.txt_html)
        btn_dicom_hash = QPushButton('Generate DICOM info content')
        btn_dicom_hash.clicked.connect(self.parent.generate_dicom_hash)
        vlo_html.addWidget(btn_dicom_hash)
        btn_add_pagebreak = QPushButton('Add pagebreak')
        btn_add_pagebreak.clicked.connect(self.add_pagebreak)
        vlo_html.addWidget(btn_add_pagebreak)
        btn_preview = QPushButton('Preview content')
        btn_preview.clicked.connect(self.preview_html_element)
        vlo_html.addWidget(btn_preview)
        self.lbl_preview = QLabel('')
        vlo_html.addWidget(self.lbl_preview)

        # Stack results
        vlo_result = QVBoxLayout()
        wid_result_element.setLayout(vlo_result)
        self.cbox_testcode = QComboBox()
        self.cbox_testcode.addItems(self.parent.test_codes)
        self.cbox_testcode.currentIndexChanged.connect(
            lambda: self.update_result_options(plot_or_image='plot'))
        self.cbox_testcode.currentIndexChanged.connect(
            lambda: self.update_result_options(plot_or_image='result_image'))
        self.cbox_table = QComboBox()
        self.cbox_table.addItems(['Result table','Supplement table'])
        self.cbox_plot = QComboBox()
        self.cbox_result_image = QComboBox()
        self.width_results = QSpinBox()
        self.width_results.setRange(3, 100)
        self.image_number_results = QSpinBox()
        if len(self.parent.main.imgs) == 0:
            maximg = 100
        else:
            maximg = len(self.parent.main.imgs) - 1
        self.image_number_results.setRange(0, maximg)
        self.all_images_results = QCheckBox()
        self.all_images_results.stateChanged.connect(self.update_all_checked)
        self.include_image_name_results = QCheckBox()
        flo_result = QFormLayout()
        vlo_result.addLayout(flo_result)
        flo_result.addRow(QLabel('Test:'), self.cbox_testcode)
        flo_result.addRow(QLabel('Result table:'), self.cbox_table)
        flo_result.addRow(QLabel('Result plot:'), self.cbox_plot)
        flo_result.addRow(QLabel('Result image:'), self.cbox_result_image)
        flo_result.addRow(QLabel('Results from each marked image'),
                          self.all_images_results)
        flo_result.addRow(QLabel('Result from image number:'),
                          self.image_number_results)
        flo_result.addRow(QLabel('Include image name(s)'),
                          self.include_image_name_results)
        flo_result.addRow(QLabel('Width (%):'), self.width_results)
        vlo_result.addWidget(uir.LabelItalic(
            'If "Result from each marked image" is checked with plot/image '
            'result, a table will be generated <br>'
            'adding results for each image with width indicating cell width.'))

        # Stack image
        vlo_image = QVBoxLayout()
        wid_image_element.setLayout(vlo_image)
        self.width_image = QSpinBox()
        flo_image = QFormLayout()
        vlo_image.addLayout(flo_image)
        self.image_number = QSpinBox()
        self.image_number.setRange(0, maximg)
        self.all_images = QCheckBox('')
        self.all_images.stateChanged.connect(self.update_all_checked)
        self.include_image_name = QCheckBox()
        self.lbl_width_image = QLabel('Number of images pr row:')
        flo_image.addRow(QLabel('All images'), self.all_images)
        flo_image.addRow(QLabel('Image number:'), self.image_number)
        flo_image.addRow(QLabel('Include image name(s)'),
                          self.include_image_name)
        flo_image.addRow(self.lbl_width_image, self.width_image)

        hlo_buttons_btm = QHBoxLayout()
        vlo.addLayout(hlo_buttons_btm)
        hlo_buttons_btm.addStretch()
        txt = 'Add' if element == cfc.ReportElement() else 'Edit'
        btn_add = QPushButton(f'{txt} element')
        btn_add.clicked.connect(self.accept)
        hlo_buttons_btm.addWidget(btn_add)
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(self.reject)
        hlo_buttons_btm.addWidget(btn_close)

        self.update_element()

    def update_element(self):
        idx = self.variants.index(self.element.variant)
        self.cbox_variant.setCurrentIndex(idx)

    def update_result_options(self, preset_text='', plot_or_image='plot'):
        widget_cbox = getattr(self, f'cbox_{plot_or_image}')
        widget_cbox.clear()
        testcode = self.cbox_testcode.currentText().lower()
        widget_params = self.parent.main.stack_test_tabs.currentWidget()
        attrib = f'{testcode}_{plot_or_image}'
        if testcode == 'hom':
            try:
                if self.parent.main.current_paramset.hom_tab_alt == 4:
                    attrib = attrib + '_aapm'
            except AttributeError:
                pass
        elif testcode == 'rec':
            attrib = 'rec_type'
        widget_options = getattr(widget_params, attrib, None)
        if widget_options is not None:
            options = [
                widget_options.itemText(i) for i
                in range(widget_options.count())]
            widget_cbox.addItems(options)
            if preset_text in options:
                widget_cbox.setCurrentText(preset_text)

    def update_variant(self):
        sel = self.cbox_variant.currentIndex()
        if self.element is not None:
            self.element.variant = self.variants[sel]
        if sel == 0:
            self.stack_variants.setCurrentIndex(0)
            self.txt_table_header.setText(self.element.caption)
        elif sel == 1:
            self.stack_variants.setCurrentIndex(1)
            self.txt_html.setPlainText(self.element.text)
            self.txt_header.setText(self.element.caption)
        elif sel == 5:
            self.stack_variants.setCurrentIndex(3)
            self.update_all_checked(initialize=True)
        else:
            self.stack_variants.setCurrentIndex(2)
            self.cbox_testcode.setCurrentText(self.element.testcode)
            self.cbox_table.setEnabled(False)
            self.cbox_plot.setEnabled(False)
            self.cbox_result_image.setEnabled(False)
            self.width_results.setValue(self.element.width)
            if sel == 2:
                self.cbox_table.setEnabled(True)
                self.all_images_results.setEnabled(False)
                self.image_number_results.setEnabled(False)
                self.include_image_name_results.setChecked(
                    self.element.include_image_name)
            else:
                self.update_all_checked(initialize=True)
                if sel == 3:
                    self.cbox_plot.setEnabled(True)
                    self.update_result_options(
                        self.element.text, plot_or_image='plot')
                elif sel == 4:
                    self.cbox_result_image.setEnabled(True)
                    self.update_result_options(
                        self.element.text, plot_or_image='result_image')

    def update_all_checked(self, initialize=False):
        """Update when all images (un)checked for variant 5, images."""
        # NB initialize is integer if not True...
        variant = self.cbox_variant.currentIndex()
        wid_all = self.all_images if variant == 5 else self.all_images_results
        wid_nmb = self.image_number if variant == 5 else self.image_number_results
        wid_incl = self.include_image_name if variant == 5 else self.include_image_name_results

        if initialize is True:
            wid_all.setEnabled(True)
            wid_all.setChecked(self.element.all_images)
        wid_nmb.setEnabled(not wid_all.isChecked())

        if initialize is True:
            wid_incl.setChecked(self.element.include_image_name)

        if wid_all.isChecked():
            if len(self.parent.main.imgs) == 0:
                maximg = 100
            else:
                maximg = len(self.parent.main.imgs) - 1
            wid_nmb.setRange(0, maximg)
            wid_nmb.setValue(self.element.image_number)
            if variant == 5:
                self.width_image.setRange(1, 10)
                self.lbl_width_image.setText('Number of images pr row:')
                if self.element.width > 10:
                    self.width_image.setValue(10)
                else:
                    if initialize is True:
                        self.width_image.setValue(self.element.width)
        else:
            if variant == 5:
                self.width_image.setRange(3, 100)
                self.lbl_width_image.setText('Width (%) in cell:')
                if initialize is True:
                    self.width_image.setValue(self.element.width)
                else:
                    self.width_image.setValue(100)

        wid_nmb.setValue(self.element.image_number)

    def add_pagebreak(self):
        txt = self.txt_html.toPlainText()
        txt = txt + '<div class="pagebreak"> </div>'
        self.txt_html.setPlainText(txt)

    def preview_html_element(self):
        sel = self.cbox_variant.currentIndex()
        if sel == 1:
            element = cfc.ReportElement(text=self.txt_html.toPlainText())
            html_this = self.parent.add_html_element(element)
            html = ['<!DOCTYPE html>', '<html>', '<body>',
                   html_this, '</body>', '</html>']
            self.lbl_preview.setText("\n".join(html))

    def get_element(self):
        sel = self.cbox_variant.currentIndex()
        self.element.variant = self.variants[sel]
        if sel == 0:
            self.element.caption = self.txt_table_header.text()
            self.element.text = ''
            self.element.testcode = ''
        elif sel == 1:
            self.element.text = self.txt_html.toPlainText()
            self.element.caption = self.txt_header.text()
            self.element.testcode = ''
        elif sel == 5:
            self.element.text = ''
            self.element.testcode = ''
            self.element.caption = 'Images'
            self.element.width = self.width_image.value()
            self.element.all_images = self.all_images.isChecked()
            self.element.image_number = self.image_number.value()
            self.element.include_image_name = self.include_image_name.isChecked()
        else:
            self.element.testcode = self.cbox_testcode.currentText()
            self.element.width = self.width_results.value()
            self.element.include_image_name = (
                self.include_image_name_results.isChecked())
            if sel == 2:
                self.element.text = self.cbox_table.currentText()
            else:
                self.element.all_images = self.all_images_results.isChecked()
                self.element.image_number = self.image_number_results.value()
                if sel == 3:
                    self.element.text = self.cbox_plot.currentText()
                elif sel == 4:
                    self.element.text = self.cbox_result_image.currentText()

        return self.element


class AddEditNoteDialog(ImageQCDialog):
    """Dialog to add/edit not of element to report."""

    def __init__(self, parent, element):
        super().__init__(parent=parent)
        self.element = element
        self.setWindowTitle('Add or edit note of element in report')
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        vlo.addWidget(uir.LabelItalic(
            'Use this note either as preset (saved) information for the '
            'current template'))
        vlo.addWidget(uir.LabelItalic(
            'or for specific comments to the current report and results.'))
        vlo.addWidget(QLabel('Note:'))
        self.note = QPlainTextEdit(self)
        self.note.setPlainText(self.element.note)
        vlo.addWidget(self.note)
        self.cbox_pos = QComboBox()
        self.cbox_pos.addItems(['before element','after element'])
        if self.element.note_pos == 'after':
            self.cbox_pos.setCurrentIndex(1)
        hlo_pos = QHBoxLayout()
        vlo.addLayout(hlo_pos)
        hlo_pos.addWidget(QLabel('Add note '))
        hlo_pos.addWidget(self.cbox_pos)
        hlo_pos.addStretch()
        hlo_buttons_btm = QHBoxLayout()
        vlo.addLayout(hlo_buttons_btm)
        btn_dicom_hash = QPushButton('Generate DICOM info content')
        btn_dicom_hash.clicked.connect(parent.generate_dicom_hash)
        hlo_buttons_btm.addWidget(btn_dicom_hash)
        hlo_buttons_btm.addStretch()
        txt = 'Add' if element.note == '' else 'Edit'
        btn_add = QPushButton(f'{txt} note')
        btn_add.clicked.connect(self.accept)
        hlo_buttons_btm.addWidget(btn_add)
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(self.reject)
        hlo_buttons_btm.addWidget(btn_close)

    def get_element(self):
        self.element.note = self.note.toPlainText()
        if self.cbox_pos.currentIndex() == 0:
            self.element.note_pos = 'before'
        else:
            self.element.note_pos = 'after'

        return self.element


class EditHeadDialog(ImageQCDialog):
    """Dialog to edit head of report."""

    def __init__(self, parent, template):
        super().__init__(parent=parent)
        self.template = template
        self.setWindowTitle('Edit <head> of html file template')
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        self.html = QPlainTextEdit(self)
        self.html.setMinimumSize(600, 600)
        if self.template.htmlhead == '':
            txt = DEFAULT_HEAD
        else:
            txt = self.template.htmlhead
        self.html.setPlainText(txt)
        vlo.addWidget(self.html)

        hlo_buttons_btm = QHBoxLayout()
        vlo.addLayout(hlo_buttons_btm)
        btn_dicom_hash = QPushButton('Generate DICOM info content')
        btn_dicom_hash.clicked.connect(parent.generate_dicom_hash)
        hlo_buttons_btm.addWidget(btn_dicom_hash)
        hlo_buttons_btm.addStretch()
        btn_add = QPushButton('Apply')
        btn_add.clicked.connect(self.accept)
        hlo_buttons_btm.addWidget(btn_add)
        btn_close = QPushButton('Cancel')
        btn_close.clicked.connect(self.reject)
        hlo_buttons_btm.addWidget(btn_close)

    def get_template(self):
        self.template.htmlhead = self.html.toPlainText()
        return self.template


class GenerateDicomHashDialog(ImageQCDialog):
    """Dialog to generate code to integrate DICOM header info in report."""

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.main = parent.main
        self.parent = parent
        self.setWindowTitle(
            'Generate code to integrate DICOM header info in report')
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        info_txt = (
            'Select a DICOM tag to include in the report and copy the '
            'generated hash code into html elements as text. The code will<br>'
            'be recognized during report generation based on the active '
            'image [active] (default) or set image number [zero based int].'
            '<br><br>'
            'Attribute value = #DICOM[attribute_name][active or integer]')
        hlo_top = QHBoxLayout()
        vlo.addLayout(hlo_top)
        hlo_top.addStretch()
        hlo_top.addWidget(uir.InfoTool(info_txt, parent=self))
        self.list_tags = uir.ListWidgetCheckable(
            texts=self.parent.tags_active)
        vlo.addWidget(uir.LabelItalic('Available DICOM tags'))
        vlo.addWidget(self.list_tags)
        self.chk_table = QCheckBox('Generate html-table for tags')
        vlo.addWidget(self.chk_table)
        btn_copy_code = QPushButton('Copy code for selected tag(s) to clipboard')
        btn_copy_code.clicked.connect(self.copy_code)
        vlo.addWidget(btn_copy_code)

        hlo_buttons_btm = QHBoxLayout()
        vlo.addLayout(hlo_buttons_btm)
        hlo_buttons_btm.addStretch()
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(self.reject)
        hlo_buttons_btm.addWidget(btn_close)

    def get_code(self):
        txt = ''
        sel_texts = self.list_tags.get_checked_texts()
        if self.chk_table.isChecked():
            if len(sel_texts) > 0:
                if len(sel_texts) == 1:
                    attr = sel_texts[0]
                    txt = ['<table>',
                           f'<tr><td>{attr}:</td><td>#DICOM[{attr}][active]</td></tr>',
                           '</table>']
                else:
                    txt = [f'<tr><td>{attr}:</td><td>#DICOM[{attr}][active]</td></tr>'
                           for attr in sel_texts]
                    txt.insert(0, '<table>')
                    txt.append('</table>')
        else:
            if len(sel_texts) > 0:
                if len(sel_texts) == 1:
                    attr = sel_texts[0]
                    txt = f'{attr}: #DICOM[{attr}][active]'
                else:
                    txt = [f'{attr}: #DICOM[{attr}][active]<br>'
                           for attr in sel_texts]
        return txt

    def copy_code(self):
        txt = self.get_code()
        if txt:
            if isinstance(txt, str):
                txt = [txt]
            dataf = pd.DataFrame(txt)
            dataf.to_clipboard(index=False, header=False)
