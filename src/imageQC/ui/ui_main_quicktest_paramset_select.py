#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for main window of imageQC - select Quicktest and paramset widgets.

@author: Ellen Wasbo
"""
import os
import copy
from time import time
import pandas as pd

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QGroupBox, QLabel, QComboBox, QAction, QToolBar,
    QMessageBox, QInputDialog
    )

# imageQC block start
from imageQC.ui import reusable_widgets as uir
from imageQC.ui import messageboxes
from imageQC.ui.ui_dialogs import QuickTestClipboardDialog
from imageQC.config import config_func as cff
from imageQC.config.iQCconstants import ENV_ICON_PATH
from imageQC.config import config_classes as cfc
from imageQC.scripts.calculate_qc import calculate_qc, quicktest_output
# imageQC block end


class SelectTemplateWidget(QWidget):
    """General widget for inheritance to QuickTest and Paramset widgets."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent
        self.edited = False
        self.lbl_edit = QLabel('')
        self.cbox_template = QComboBox()
        self.lastload = None

    def flag_edit(self, flag=True):
        """Indicate some change."""
        if flag:
            self.edited = True
            self.lbl_edit.setText('*')
        else:
            self.edited = False
            self.lbl_edit.setText('')

    def fill_template_list(self, set_label=''):
        """Fill list of templates for current modality."""
        self.cbox_template.blockSignals(True)
        self.cbox_template.clear()
        labels = [temp.label for temp in self.modality_dict[self.main.current_modality]]
        if len(labels) > 0:
            if self.fname == 'quicktest_templates':
                labels.insert(0, '')
            self.cbox_template.addItems(labels)
            if set_label in labels:
                set_index = labels.index(set_label)
            else:
                set_index = 0
            self.cbox_template.blockSignals(False)
            self.cbox_template.setCurrentIndex(set_index)
        else:
            self.cbox_template.blockSignals(False)
        self.lastload = time()

    def add_current_template(self):
        """Add current template."""
        if self.fname == 'quicktest_templates':
            self.get_current_template()
        elif 'paramsets' in self.fname:
            self.current_template = self.main.current_paramset
        text, proceed = QInputDialog.getText(self, 'New label', 'Name the new template')
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
        if self.modality_dict[self.main.current_modality][0].label == '':
            self.add_current_template()
        else:
            if before_select_new:
                # id corresponding to label of current_template (previous selected)
                if self.fname == 'quicktest_templates':
                    label = self.current_template.label
                else:  # self.fname == 'paramsets':
                    label = self.main.current_paramset.label
                labels = [
                    temp.label for temp
                    in self.modality_dict[self.main.current_modality]]
                template_id = labels.index(label)
            else:
                template_id = self.cbox_template.currentIndex()
            if self.fname == 'quicktest_templates':
                if not before_select_new:
                    template_id -= 1  # first is empty
                self.get_current_template()
            elif 'paramsets' in self.fname:
                self.current_template = self.main.current_paramset
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
                if self.fname == 'quicktest_templates':
                    proceed, path = cff.save_settings(
                        self.modality_dict, fname=self.fname)
                else:
                    proceed, path = cff.save_settings(
                        self.modality_dict[self.main.current_modality],
                        fname=self.fname)
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


class SelectQuickTestWidget(SelectTemplateWidget):
    """Widget for selecting and saving QuickTest templates."""

    def __init__(self, parent):
        super().__init__(parent)

        self.fname = 'quicktest_templates'
        self.modality_dict = self.main.quicktest_templates
        self.current_template = self.main.current_quicktest

        self.gb_quicktest = QGroupBox('QuickTest')

        h_lo = QHBoxLayout()
        self.setLayout(h_lo)

        self.gb_quicktest.setCheckable(True)
        self.gb_quicktest.setChecked(False)
        self.gb_quicktest.toggled.connect(self.update_current_template)
        self.gb_quicktest.setFont(uir.FontItalic())
        self.cbox_template.setMinimumWidth(150)
        self.cbox_template.currentIndexChanged.connect(self.update_current_template)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('QuickTest template'))
        hbox.addWidget(self.cbox_template)
        hbox.addWidget(self.lbl_edit)

        act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add current test pattern as QuickTest', self)
        act_add.triggered.connect(self.add_current_template)

        act_save = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Overwrite current test pattern as QuickTest', self)
        act_save.triggered.connect(self.save_current_template)
        if self.main.save_blocked:
            act_save.setEnabled(False)

        act_settings_qt = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}gears.png'),
            'Edit/manage QuickTest templates', self)
        act_settings_qt.triggered.connect(
            lambda: self.main.run_settings(initial_view='QuickTest templates'))

        act_exec_qt = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}play.png'),
            'Run QuickTest', self)
        act_exec_qt.triggered.connect(self.run_quicktest)

        act_clipboard_qt = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}copy.png'),
            'Copy QuickTest results to clipboard', self)
        act_clipboard_qt.triggered.connect(self.extract_results)

        toolb = QToolBar()
        toolb.addActions([act_add, act_save, act_settings_qt])
        toolb.addSeparator()
        toolb.addActions([act_exec_qt, act_clipboard_qt])

        hbox.addWidget(toolb)
        hbox.addStretch()
        self.gb_quicktest.setLayout(hbox)
        h_lo.addWidget(self.gb_quicktest)

        self.lastload = time()

        self.fill_template_list()

    def update_current_template(self):
        """Set current_template according to selected label."""
        if self.edited and self.current_template.label != '':
            self.ask_to_save_changes(before_select_new=True)

        template_id = self.cbox_template.currentIndex()
        if template_id == 0:
            self.current_template = cfc.QuickTestTemplate()
        else:
            self.current_template = copy.deepcopy(
                self.modality_dict[self.main.current_modality][template_id - 1])
        self.set_current_template_to_imgs()
        self.main.tree_file_list.update_file_list()
        self.flag_edit(False)

    def set_current_template_to_imgs(self):
        """Set image-dict values according to current template."""
        for imgno, img in enumerate(self.main.imgs):
            try:
                img.marked_quicktest = self.current_template.tests[imgno]
            except IndexError:
                img.marked_quicktest = []
                self.flag_edit(True)

            try:
                img.quicktest_image_name = \
                    self.current_template.image_names[imgno]
            except IndexError:
                img.quicktest_image_name = ''

            try:
                img.quicktest_group_name = \
                    self.current_template.group_names[imgno]
            except IndexError:
                img.quicktest_group_name = ''

    def get_current_template(self):
        """Fill current_template with values for imgs."""
        lbl = self.current_template.label
        self.current_template = cfc.QuickTestTemplate(label=lbl)
        for img in self.main.imgs:
            self.current_template.add_index(
                test_list=img.marked_quicktest,
                image_name=img.quicktest_image_name,
                group_name=img.quicktest_group_name
                )

    def run_quicktest(self):
        """Run quicktest with current settings."""
        self.get_current_template()
        if any(self.current_template.tests):
            self.main.current_quicktest = self.current_template
            calculate_qc(self.main)
        else:
            QMessageBox.information(
                self, 'No test specified',
                ('No image marked for testing. Double- or right-click images in file '
                 'list to mark for testing.'))

    def extract_results(self, skip_questions=False):
        """Extract result values according to paramset.output to clipboard."""
        proceed = True
        if skip_questions:  # for testing
            include_headers = self.main.current_paramset.output.include_header
            transpose_table = self.main.current_paramset.output.transpose_table
        else:
            dlg = QuickTestClipboardDialog(
                include_headers=self.main.current_paramset.output.include_header,
                transpose_table=self.main.current_paramset.output.transpose_table)
            res = dlg.exec()
            if res:
                include_headers, transpose_table = dlg.get_data()
            else:
                proceed = False

        if proceed:
            value_list, header_list = quicktest_output(self.main)
            date_str = self.main.imgs[0].acq_date  # yyyymmdd to dd.mm.yyyy
            date_str_out = date_str[6:8] + '.' + date_str[4:6] + '.' + date_str[0:4]
            value_list = [date_str_out] + value_list
            if include_headers:
                header_list = ['Date'] + header_list
                df = pd.DataFrame([value_list], columns=header_list)
                if transpose_table:
                    df = df.transpose().reset_index()
                    df.to_clipboard(index=False, excel=True, sep=None, header=None)
                else:
                    df.to_clipboard(index=False, excel=True, sep=None)
            else:
                df = pd.DataFrame([value_list])
                if transpose_table:
                    df = df.transpose()
                df.to_clipboard(index=False, excel=True, sep=None, header=None)
            self.main.status_bar.showMessage('Values in clipboard', 2000)


class SelectParamsetWidget(SelectTemplateWidget):
    """Widget for selecting and saving parameter sets."""

    def __init__(self, parent):
        super().__init__(parent)
        self.main = parent
        self.fname = 'paramsets_CT'
        self.modality_dict = {f'{self.main.current_modality}': self.main.paramsets}
        self.current_template = self.main.current_paramset

        h_lo = QHBoxLayout()
        self.setLayout(h_lo)

        h_lo.addWidget(QLabel('Parameter set:'))
        self.cbox_template.setMinimumWidth(150)
        self.cbox_template.currentIndexChanged.connect(self.update_current_template)
        h_lo.addWidget(self.cbox_template)
        h_lo.addWidget(self.lbl_edit)

        act_add_param = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Save current parameters as new parameter set', self)
        act_add_param.triggered.connect(self.add_current_template)

        act_save_param = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
            'Overwrite current parameter set', self)
        act_save_param.triggered.connect(self.save_current_template)
        if self.main.save_blocked:
            act_save_param.setEnabled(False)

        act_settings_param = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}gears.png'),
            'Manage parameter sets', self)
        act_settings_param.triggered.connect(
            lambda: self.main.run_settings(initial_view='Parameter sets'))

        toolb = QToolBar()
        toolb.addActions([act_add_param, act_save_param, act_settings_param])
        h_lo.addWidget(toolb)

        h_lo.addStretch()

        self.lastload = time()
        self.fill_template_list()

    def update_current_template(self):
        """Set current_template according to selected label."""
        if self.edited:
            self.ask_to_save_changes(before_select_new=True)

        self.main.update_paramset()