#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for configuration settings - part paramsets.

@author: Ellen Wasbo
"""
from __future__ import annotations

import os
import copy
from dataclasses import asdict

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QWidget, QTreeWidget, QTreeWidgetItem, QTabWidget,
    QVBoxLayout, QHBoxLayout,
    QToolBar, QLabel, QAction, QListWidget, QPushButton,
    QMessageBox
    )

# imageQC block start
from imageQC.config.iQCconstants import ENV_ICON_PATH
from imageQC.config import config_classes as cfc
from imageQC.ui.settings_reusables import (
    StackWidget, QuickTestOutputTreeView, ResultImageDefaultsTreeView)
from imageQC.ui.tag_patterns import TagPatternEditDialog
from imageQC.ui import reusable_widgets as uir
from imageQC.ui import messageboxes
from imageQC.scripts import dcm
# imageQC block end


class ParametersWidget(QWidget):
    """Widget for holding the parameters table for ParamSetsWidget."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        vlo_tables = QVBoxLayout()
        vlo_tables.addWidget(uir.LabelItalic(
            '''Differences between selected and active parameter set will be
            highlighted in red font.'''))
        self.table_params = QTreeWidget()
        self.table_params.setHeaderLabels(
            ['Parameter', 'Value in selected set',
             'Active value in main window'])
        vlo_tables.addWidget(self.table_params)
        self.setLayout(vlo_tables)


class ParametersOutputWidget(QWidget):
    """Widget for holding the output parameters table for ParamSetsWidget."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.list_group_by = QListWidget()
        self.wid_output_table = QuickTestOutputTreeView(self.parent)

        hlo = QHBoxLayout()
        vlo_general = QVBoxLayout()
        self.setLayout(hlo)
        hlo.addLayout(vlo_general)
        vlo_general.addWidget(uir.LabelItalic('Settings for text output:'))
        self.tb_copy = uir.ToolBarTableExport(self.parent, flag_edit=True)
        self.tb_copy.setOrientation(Qt.Horizontal)
        vlo_general.addWidget(self.tb_copy)
        vlo_general.addSpacing(10)
        vlo_general.addWidget(
            uir.LabelItalic('If calculate pr group, group by'))
        hlo_group = QHBoxLayout()
        vlo_general.addLayout(hlo_group)
        hlo_group.addWidget(self.list_group_by)
        self.list_group_by.setFixedWidth(200)
        self.tb_edit_group_by = uir.ToolBarEdit(tooltip='Edit list')
        vlo_general.addWidget(self.tb_edit_group_by)
        self.tb_edit_group_by.act_edit.triggered.connect(self.edit_group_by)

        vlo_table = QVBoxLayout()
        hlo.addLayout(vlo_table)
        vlo_table.addWidget(uir.LabelItalic(
            '''Add or edit settings to define output settings when QuickTest
            is used.<br>
            Default if no settings are defined for a test, all values from
            the results table will be printed.'''))
        hlo_table = QHBoxLayout()
        vlo_table.addLayout(hlo_table)
        hlo_table.addWidget(self.wid_output_table)
        self.wid_output_table.setFixedWidth(800)

        self.toolbar = QToolBar()
        self.toolbar.setOrientation(Qt.Vertical)
        hlo_table.addWidget(self.toolbar)
        act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add more settings', self)
        act_add.triggered.connect(self.wid_output_table.insert_row)
        act_edit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit selected row', self)
        act_edit.triggered.connect(self.wid_output_table.edit_row)
        act_up = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
            'Move up', self)
        act_up.triggered.connect(
            lambda: self.wid_output_table.move_sub(move_up=True))
        act_down = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
            'Move down', self)
        act_down.triggered.connect(
            lambda: self.wid_output_table.move_sub(move_up=False))
        act_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete selected row', self)
        act_delete.triggered.connect(self.wid_output_table.delete_row)
        self.toolbar.addActions([act_add, act_edit, act_up, act_down, act_delete])

        btn_test_output = QPushButton(
            'Test output with currently loaded images and QuickTest template.')
        btn_test_output.clicked.connect(self.test_output)
        vlo_table.addWidget(btn_test_output)

    def edit_group_by(self):
        """Edit parameters to group images by."""
        dlg = TagPatternEditDialog(
            initial_pattern=cfc.TagPatternFormat(
                list_tags=self.parent.current_template.output.group_by),
            modality=self.parent.current_modality,
            title='Group images by same DICOM header information (format ignored)',
            typestr='format',
            accept_text='Use',
            reject_text='Cancel',
            save_blocked=self.parent.save_blocked)
        res = dlg.exec()
        if res:
            pattern = dlg.get_pattern()
            self.parent.current_template.output.group_by = pattern.list_tags
            self.parent.flag_edit(True)
            self.list_group_by.clear()
            self.list_group_by.addItems(pattern.list_tags)
            main = self.parent.dlg_settings.main
            if len([*main.results]) > 0:
                ok_param = False
                if main.current_paramset.label == self.parent.current_template.label:
                    if main.current_modality == self.parent.current_modality:
                        ok_param = True
                if ok_param:
                    dlg = messageboxes.QuestionBox(
                        parent=self, title='Refresh group info?',
                        msg=('To test the output of current results with these changes,'
                             ' saving the parameterset and refreshing the group '
                             'information is needed.'),
                        yes_text='Save and refresh group information now',
                        no_text='No, skip save/refresh.')
                    res = dlg.exec()
                    if res:
                        self.parent.wid_mod_temp.save()
                        self.update_group_indicators()

    def update_group_indicators(self):
        """Update main.current_group_indicatiors according to output.group_by."""
        main = self.parent.dlg_settings.main
        main.current_group_indicators = ['' for i in range(len(main.imgs))]
        group_pattern = cfc.TagPatternFormat(
            list_tags=self.parent.current_template.output.group_by)
        for i, img in enumerate(main.imgs):
            tags = dcm.get_tags(
                img.filepath, frame_number=img.frame_number,
                tag_patterns=[group_pattern], tag_infos=main.tag_infos)
            if len(tags) > 0:
                main.current_group_indicators[i] = '_'.join(tags[0])
        QMessageBox.information(
            self, 'Group information updated',
            'Group tags updated according to the current settings.')

    def test_output(self):
        """Run QuickTest if set in main and current_paramset == selected paramset."""
        main = self.parent.dlg_settings.main
        if len(main.imgs) > 0:
            ok_param = False
            if main.current_paramset.label == self.parent.current_template.label:
                if main.current_modality == self.parent.current_modality:
                    ok_param = True

            ok_qt = False
            if main.wid_quicktest.gb_quicktest.isChecked():
                main.wid_quicktest.get_current_template()
                current_qt = main.wid_quicktest.current_template
                if current_qt is not None:
                    if any(current_qt.tests):
                        set_qt_tests = set(sum(current_qt.tests, []))
                        set_res = set([*main.results])
                        if len(set_res.difference(set_qt_tests)) == 0:
                            ok_qt = True
            if ok_qt and ok_param:
                # temporary set test output to main.current_paramset.output
                curr_output = copy.deepcopy(main.current_paramset.output)
                test_output = copy.deepcopy(self.parent.current_template.output)
                if not all([test_output.include_header, test_output.transpose_table]):
                    dlg = messageboxes.QuestionBox(
                        parent=self, title='Recommended test settings',
                        msg=('To better understand the output, the recommended settings'
                             ' for testing is with headers and transposed table.'),
                        yes_text='Use recommended settings for testing',
                        no_text='Keep settings for testing of parameter set as is')
                    res = dlg.exec()
                    if res:
                        test_output.include_header = True
                        test_output.transpose_table = True
                main.current_paramset.output = test_output
                main.wid_quicktest.extract_results(silent=True)
                main.current_paramset.output = curr_output
                # display output (extract from clipboard and display)
                main.display_clipboard(title='Export of current results in main window')
            else:
                msg = ('' if ok_param else
                       'The parameter set in main window need to be the same as here.')
                if ok_qt is False:
                    msg = msg + ' No QuickTest results available in main window.'
                QMessageBox.warning(self, 'No QuickTest defined ', msg)
        else:
            QMessageBox.warning(
                self, 'No images loaded',
                'No images loaded and calculated with QuickTest in main window.')


class ResultImageDefaultsWidget(QWidget):
    """Widget for holding the output parameters table for ParamSetsWidget."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.wid_defaults = ResultImageDefaultsTreeView(self.parent)

        hlo = QHBoxLayout()
        self.setLayout(hlo)
        vlo_general = QVBoxLayout()
        hlo.addLayout(vlo_general)
        vlo_general.addWidget(uir.LabelItalic(
            'Override default result image draw options'))
        vlo_general.addWidget(self.wid_defaults)

        self.toolbar = QToolBar()
        self.toolbar.setOrientation(Qt.Vertical)
        hlo.addWidget(self.toolbar)
        act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add override rule', self)
        act_add.triggered.connect(self.wid_defaults.insert_row)
        act_edit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit rule', self)
        act_edit.triggered.connect(self.wid_defaults.edit_row)
        act_up = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
            'Move up', self)
        act_up.triggered.connect(
            lambda: self.wid_defaults.move_sub(move_up=True))
        act_down = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
            'Move down', self)
        act_down.triggered.connect(
            lambda: self.wid_defaults.move_sub(move_up=False))
        act_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete rule', self)
        act_delete.triggered.connect(self.wid_defaults.delete_row)
        self.toolbar.addActions([act_add, act_edit, act_up, act_down, act_delete])


class ParamSetsWidget(StackWidget):
    """Widget holding paramsets settings."""

    def __init__(self, dlg_settings):
        header = 'Parameter sets - manager'
        subtxt = '''The parameter sets contain both parameters for
        test settings and for output settings when results are copied
        to clipboard or to file (when automation is used).<br>
        To edit the test settings use the main window to set and save
        the parameters.'''
        super().__init__(dlg_settings, header, subtxt,
                         temp_alias='parameterset',
                         mod_temp=True, grouped=True)
        self.main_current_modality = self.dlg_settings.main.current_modality
        self.main_current_paramset = self.dlg_settings.main.current_paramset

        self.fname = 'paramsets'
        self.empty_template = self.get_empty_paramset()

        if self.import_review_mode is False:
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_move_modality)
        self.wid_mod_temp.vlo.addWidget(
            QLabel('Selected parameterset used in Automation template:'))
        self.list_used_in = QListWidget()
        self.wid_mod_temp.vlo.addWidget(self.list_used_in)

        self.tabs = QTabWidget()
        self.wid_params = ParametersWidget(self)
        self.wid_output = ParametersOutputWidget(self)
        self.wid_result_image_defaults = ResultImageDefaultsWidget(self)
        if self.import_review_mode:
            self.wid_output.setEnabled(False)
            self.wid_result_image_defaults.setEnabled(False)

        self.tabs.addTab(self.wid_params, 'Parameters')
        self.tabs.addTab(self.wid_output, 'Output parameters')
        self.tabs.addTab(self.wid_result_image_defaults,
                         'Result image display overrides')

        self.hlo.addWidget(self.tabs)
        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def get_empty_paramset(self):
        """Get empty (default) paramset of current modality.

        Returns
        -------
        paramset
        """
        mod = self.current_modality
        class_ = getattr(cfc, f'ParamSet{mod}')
        paramset = class_()

        return paramset

    def update_data(self):
        """Update GUI with the selected paramset."""
        self.wid_output.wid_output_table.update_data()
        self.wid_output.tb_copy.parameters_output = self.current_template.output
        self.wid_output.tb_copy.update_checked()
        self.wid_output.list_group_by.clear()
        self.wid_output.list_group_by.addItems(
            self.current_template.output.group_by)
        self.wid_result_image_defaults.wid_defaults.update_data()

        self.wid_params.table_params.clear()
        font_bold = QFont()
        font_bold.setWeight(600)

        main_str = ''
        get_main = False
        if self.main_current_paramset is not None:
            if self.main_current_modality == (
                    self.current_modality):
                get_main = True
        for paramname, paramval in asdict(self.current_template).items():
            if paramname not in ['output', 'result_image_defaults']:
                if isinstance(paramval, dict):
                    item = QTreeWidgetItem(
                        self.wid_params.table_params, [paramname])
                    if get_main:
                        main_val = getattr(
                            self.main_current_paramset, paramname)
                    else:
                        main_val = None
                    for key, val in paramval.items():
                        if key != 'label':
                            if main_val is not None:
                                main_subval = getattr(main_val, key)
                                if isinstance(main_subval, list):
                                    main_subvals = [
                                        str(elem) for elem in main_subval]
                                    main_str = ' / '.join(main_subvals)
                                else:
                                    main_str = str(main_subval)
                            if isinstance(val, list):
                                vals = [
                                    str(elem) for elem in val]
                                str_val = ' / '.join(vals)
                            else:
                                str_val = str(val)
                            child = QTreeWidgetItem([key, str_val, main_str])
                            if str_val != main_str:
                                child.setForeground(2, QBrush(
                                    QColor(203, 91, 76)))
                                child.setFont(2, font_bold)
                            item.addChild(child)
                    item.setExpanded(True)
                else:
                    if get_main:
                        main_val = getattr(
                            self.main_current_paramset, paramname)
                        main_str = str(main_val)
                    item = QTreeWidgetItem(
                        self.wid_params.table_params,
                        [paramname, str(paramval), main_str])
                    if str(paramval) != main_str:
                        item.setForeground(2, QBrush(QColor(203, 91, 76)))
                        item.setFont(2, font_bold)
        for i in range(self.wid_params.table_params.columnCount()):
            self.wid_params.table_params.resizeColumnToContents(i)

        self.update_used_in()

    def update_used_in(self):
        """Update list of auto-templates where this template is used."""
        self.list_used_in.clear()
        if self.current_template.label != '':
            try:
                auto_labels = [
                    temp.label for temp in self.auto_templates[self.current_modality]
                    if temp.paramset_label == self.current_template.label
                    ]
            except KeyError:
                auto_labels = []
            if len(auto_labels) > 0:
                self.list_used_in.addItems(auto_labels)
