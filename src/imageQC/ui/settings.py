#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for configuration settings.

@author: Ellen Wasbo
"""
from __future__ import annotations

import os
from pathlib import Path
from time import ctime
from dataclasses import dataclass, field
import copy
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, qApp,
    QTreeWidget, QTreeWidgetItem, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox, QToolBar,
    QLabel, QLineEdit, QPushButton, QAction, QSpinBox, QCheckBox,
    QListWidget, QMessageBox, QDialogButtonBox, QFileDialog
    )

# imageQC block start
from imageQC.config.iQCconstants import (
    QUICKTEST_OPTIONS, CONFIG_FNAMES, ENV_ICON_PATH
    )
from imageQC.config import config_func as cff
from imageQC.config import config_classes as cfc
from imageQC.config.read_config_idl import ConfigIdl2Py
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.ui.settings_reusables import StackWidget, QuickTestTreeView
from imageQC.ui import settings_automation
from imageQC.ui import settings_dicom_tags
from imageQC.ui.settings_paramsets import ParamSetsWidget
from imageQC.ui import settings_digits
from imageQC.ui import reusable_widgets as uir
from imageQC.ui import messageboxes
from imageQC.scripts.mini_methods import get_all_matches
from imageQC.scripts.mini_methods_format import time_diff_string
# imageQC block end


class SettingsDialog(ImageQCDialog):
    """GUI setup for the settings dialog window."""

    def __init__(
            self, main, initial_view='User local settings', initial_template_label='',
            paramset_output=False, initial_modality=None,
            width1=200, width2=800, import_review_mode=False, exclude_paramsets=False):
        """Initiate Settings dialog.

        Parameters
        ----------
        main : MainWindow or ImportMain
            MainWindow if not import review mode
        initial_view : str, optional
            title of widget to select from tree. The default is 'User local settings'.
        initial_template_label : str, optional
            If a preset template on opening. The default is ''.
        paramset_output : bool, optional
            If initial view is paramset and open in output tab. The default is False.
        initial_modality : str, optional
            Modality as defined in imageQC. The default is None.
        width1 : int, optional
            width of tree widget. The default is 200.
        width2 : it, optional
            width of the right panel. The default is 800.
        import_review_mode : bool, optional
            special settings if reviewing settings to import. The default is False.
        exclude_paramsets : bool, optional
            if in import review mode and no paramset found (default set only)
        """
        super().__init__()
        self.main = main
        if initial_modality is None:
            self.initial_modality = self.main.current_modality
        else:
            self.initial_modality = initial_modality
        self.import_review_mode = import_review_mode
        if import_review_mode is False:
            self.setWindowTitle('Settings manager')
            self.width1 = round(self.main.gui.panel_width*0.3)
            self.width2 = round(self.main.gui.panel_width*1.7)
        else:
            self.width1 = width1
            self.width2 = width2
            self.setWindowTitle('Import review - configuration settings')

        hlo = QHBoxLayout()
        self.setLayout(hlo)

        self.tree_settings = QTreeWidget()
        hlo.addWidget(self.tree_settings)
        self.stacked_widget = QStackedWidget()
        hlo.addWidget(self.stacked_widget)

        self.tree_settings.setColumnCount(1)
        self.tree_settings.setFixedWidth(self.width1)
        self.tree_settings.setHeaderHidden(True)
        self.tree_settings.itemClicked.connect(self.change_widget)

        self.stacked_widget.setFixedWidth(self.width2)

        self.list_txt_item_widget = []

        def add_widget(parent=None, snake='', title='', widget=None,
                       exclude_if_empty=True):
            proceed = True
            if import_review_mode and exclude_if_empty:
                if getattr(self.main, snake, {}) == {}:
                    proceed = False
                if snake == 'paramsets' and exclude_paramsets:
                    proceed = False
            if proceed:
                setattr(self, f'item_{snake}', QTreeWidgetItem([title]))
                item = getattr(self, f'item_{snake}')
                if parent is None:
                    self.tree_settings.addTopLevelItem(item)
                else:
                    parent.addChild(item)
                setattr(self, f'widget_{snake}', widget)
                this_widget = getattr(self, f'widget_{snake}')
                self.stacked_widget.addWidget(this_widget)
                self.list_txt_item_widget.append((title, item, this_widget))

        if import_review_mode is False:
            add_widget(snake='user_settings', title='Local settings',
                       widget=UserSettingsWidget(self))
            add_widget(snake='shared_settings', title='Config folder',
                       widget=SharedSettingsWidget(self))
        else:
            add_widget(snake='shared_settings', title='Settings for import',
                       widget=SharedSettingsImportWidget(self),
                       exclude_if_empty=False)

        add_widget(parent=self.item_shared_settings, snake='dicom_tags',
                   title='DICOM tags',
                   widget=settings_dicom_tags.DicomTagsWidget(self),
                   exclude_if_empty=False)
        add_widget(parent=self.item_dicom_tags, snake='tag_patterns_special',
                   title='Special tag patterns',
                   widget=settings_dicom_tags.TagPatternSpecialWidget(self))
        add_widget(parent=self.item_dicom_tags, snake='tag_patterns_format',
                   title='Tag patterns - format',
                   widget=settings_dicom_tags.TagPatternFormatWidget(self))
        add_widget(parent=self.item_dicom_tags, snake='rename_patterns',
                   title='Rename patterns',
                   widget=settings_dicom_tags.RenamePatternWidget(self))
        add_widget(parent=self.item_dicom_tags, snake='tag_patterns_sort',
                   title='Tag patterns - sort',
                   widget=settings_dicom_tags.TagPatternSortWidget(self))

        add_widget(parent=self.item_shared_settings, snake='digit_templates',
                   title='Digit templates',
                   widget=settings_digits.DigitWidget(self))

        add_widget(parent=self.item_shared_settings, snake='paramsets',
                   title='Parameter sets / output',
                   widget=ParamSetsWidget(self))
        add_widget(parent=self.item_shared_settings, snake='quicktest_templates',
                   title='QuickTest templates',
                   widget=QuickTestTemplatesWidget(self))

        proceed = True
        if import_review_mode:
            if (
                    self.main.auto_common.import_path == ''
                    and self.main.auto_templates == {}
                    and self.main.auto_vendor_templates == {}
                    ):
                proceed = False
        if proceed:
            add_widget(parent=self.item_shared_settings, snake='auto_info',
                       title='Automation',
                       widget=settings_automation.AutoInfoWidget(self),
                       exclude_if_empty=False)

            proceed = True
            if import_review_mode:
                if self.main.auto_common.import_path == '':
                    proceed = False
            if proceed:
                add_widget(parent=self.item_auto_info, snake='auto_common',
                           title='Import settings',
                           widget=settings_automation.AutoCommonWidget(self),
                           exclude_if_empty=False)

            proceed = True
            if import_review_mode:
                if self.main.auto_templates == {}:
                    proceed = False
                else:
                    list_lens = [len(t) for mod, t
                                 in self.main.auto_templates.items()]
                    if max(list_lens) == 0:
                        proceed = False
            if proceed:
                add_widget(parent=self.item_auto_info, snake='auto_templates',
                           title='Templates DICOM',
                           widget=settings_automation.AutoTemplateWidget(self),
                           exclude_if_empty=False)

            proceed = True
            if import_review_mode:
                if self.main.auto_vendor_templates == {}:
                    proceed = False
                else:
                    list_lens = [len(t) for mod, t
                                 in self.main.auto_vendor_templates.items()]
                    if max(list_lens) == 0:
                        proceed = False
            if proceed:
                add_widget(parent=self.item_auto_info, snake='auto_vendor_templates',
                           title='Templates vendor files',
                           widget=settings_automation.AutoVendorTemplateWidget(self),
                           exclude_if_empty=False)

            add_widget(parent=self.item_auto_info, snake='dash_settings',
                       title='Dashboard settings',
                       widget=settings_automation.DashSettingsWidget(self),
                       exclude_if_empty=False)

            proceed = True
            if import_review_mode:
                if len(self.main.persons_to_notify) == 0:
                    proceed = False

            if proceed:
                add_widget(parent=self.item_auto_info, snake='persons_to_notify',
                           title='Persons to notify',
                           widget=settings_automation.PersonsToNotifyWidget(self),
                           exclude_if_empty=False)

        item, widget = self.get_item_widget_from_txt(initial_view)
        self.tree_settings.setCurrentItem(item)
        self.tree_settings.expandToDepth(2)
        self.tree_settings.resizeColumnToContents(0)
        self.stacked_widget.setCurrentWidget(widget)
        self.previous_selected_txt = initial_view
        self.current_selected_txt = initial_view

        if import_review_mode is False:
            widget.update_from_yaml(initial_template_label=initial_template_label)
            if paramset_output:
                if hasattr(widget, 'tabs'):
                    widget.tabs.setCurrentIndex(1)
        else:
            self.update_import_main()

    def get_item_widget_from_txt(self, txt):
        """Find tree item and stack widget based on item txt."""
        item = self.list_txt_item_widget[0][1]  # default
        widget = self.list_txt_item_widget[0][2]  # default
        for tiw in self.list_txt_item_widget:
            if tiw[0] == txt:
                item = tiw[1]
                widget = tiw[2]

        return (item, widget)

    def change_widget(self, item):
        """Update visible widget in stack when selection in tree change."""
        prevtxtitem = self.current_selected_txt
        item = self.tree_settings.indexFromItem(item)
        txtitem = item.data(Qt.DisplayRole)

        # Settings changed - saved? Go back to prev if regret leaving unchanged
        _, prev_widget = self.get_item_widget_from_txt(prevtxtitem)
        edited = False
        try:
            edited = getattr(prev_widget, 'edited')
        except AttributeError:
            pass

        proceed = True
        if edited:
            proceed = messageboxes.proceed_question(
                self, 'Proceed and loose unsaved changes?')

        if proceed:
            try:
                self.main.start_wait_cursor()
            except AttributeError:
                pass  # if ImportMain not MainWindow
            self.previous_selected_txt = self.current_selected_txt
            self.current_selected_txt = txtitem
            _, new_widget = self.get_item_widget_from_txt(txtitem)
            self.stacked_widget.setCurrentWidget(new_widget)
            new_widget.current_modality = prev_widget.current_modality
            if self.import_review_mode:
                if new_widget.grouped:
                    new_widget.wid_mod_temp.cbox_modality.setCurrentText(
                        new_widget.current_modality)
                    new_widget.update_modality()
            else:
                new_widget.update_from_yaml()
            try:
                self.main.stop_wait_cursor()
            except AttributeError:
                pass  # if ImportMain not MainWindow
        else:
            item, _ = self.get_item_widget_from_txt(
                self.previous_selected_txt)
            self.tree_settings.setCurrentItem(item)

    def closeEvent(self, event):
        """Test if unsaved changes before closing."""
        if self.import_review_mode:
            reply = QMessageBox.question(
                self, 'Cancel import?',
                'To finish import go to first page (Settings for import) and '
                'select what to include in the import. Proceed cancel import?',
                QMessageBox.Yes, QMessageBox.No)
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            prevtxtitem = self.current_selected_txt
            _, prev_widget = self.get_item_widget_from_txt(prevtxtitem)
            edited = False
            try:
                edited = getattr(prev_widget, 'edited')
            except AttributeError:
                pass
            if edited:
                reply = QMessageBox.question(
                    self, 'Unsaved changes',
                    'Close and loose unsaved changes?',
                    QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    event.accept()
                else:
                    event.ignore()
            else:
                event.accept()

    def import_from_yaml(self):
        """Import settings from another config folder."""
        def select_yaml_files_for_import(filenames):
            """Select which files to import from a config_folder."""
            dlg = SelectImportFilesDialog(filenames)
            if dlg.exec():
                filenames = dlg.get_checked_files()
            else:
                filenames = []
            return filenames

        dlg = QFileDialog(self, 'Locate config folder to import settings from')
        dlg.setFileMode(QFileDialog.Directory)
        filenames = []
        if dlg.exec():
            import_folder = dlg.selectedFiles()[0]
            if import_folder == cff.get_config_folder():
                QMessageBox.warning(
                    self, 'Select another folder',
                    'Cannot import from the current config folder.')
            else:
                filenames = [x.stem for x in Path(import_folder).glob('*')
                             if x.suffix == '.yaml']
                acceptable_filenames = [*CONFIG_FNAMES]
                acceptable_filenames.remove('active_users')
                acceptable_filenames.remove('last_modified')
                acceptable_filenames.remove('paramsets')
                acceptable_filenames.extend(
                    [f'paramsets_{m}' for m in QUICKTEST_OPTIONS])
                filenames = [filename for filename in filenames
                             if filename in acceptable_filenames]

                if len(filenames) == 0:
                    QMessageBox.warning(
                        self, 'No template files found',
                        'No template file found in the selected folder.')
                else:
                    filenames = select_yaml_files_for_import(filenames)

        if len(filenames) > 0:
            paramsets_imported = False
            import_main = ImportMain()
            _, _, default_paramsets = cff.load_settings(
                fname='paramsets', temp_config_folder='some_not_existing_folder')
            import_main.paramsets = default_paramsets
            for fname in filenames:
                if fname == 'tag_infos':
                    _, _, tag_infos_import = cff.load_settings(
                        fname='tag_infos', temp_config_folder=import_folder)
                    _, _, tag_infos = cff.load_settings(
                        fname='tag_infos')
                    tag_infos_new = cff.tag_infos_difference(
                        tag_infos_import, tag_infos)
                    import_main.tag_infos = tag_infos_new
                else:
                    _, _, temps = cff.load_settings(
                        fname=fname, temp_config_folder=import_folder)
                    if 'paramsets' in fname:
                        mod = fname.split('_')[1]
                        import_main.paramsets[mod] = temps
                        paramsets_imported = True
                    else:
                        setattr(import_main, fname, temps)

            if paramsets_imported:
                import_main.current_paramset = import_main.paramsets['CT'][0]
            dlg = SettingsDialog(
                import_main, initial_view='Config folder',
                width1=self.width1, width2=self.width2,
                import_review_mode=True, exclude_paramsets=(not paramsets_imported))
            res = dlg.exec()
            if res:
                import_main = dlg.get_marked()
                same_names = cff.import_settings(import_main)
                if same_names:
                    QMessageBox.information(
                        self, 'Information',
                        ('Imported one or more templates with same name as '
                         'templates already set. The imported template names '
                         'was marked with _import.'))
                # TODO if later option to import from the different widgets
                # _, widget = self.get_item_widget_from_txt(
                #    self.previous_selected_txt)
                self.widget_shared_settings.update_from_yaml()
                # TODO DELETE? fails? self.widget_shared_settings.verify_config_files()

    def update_import_main(self):
        """Update templates of all widgets according to import_main.

        Similar to settings_reusables.py - StackWidget def update_from_yaml.
        TODO: syncronize these two better...?
        """
        if self.main.tag_infos != []:
            self.widget_dicom_tags.templates = self.main.tag_infos
            self.widget_dicom_tags.update_data()
        list_dicts = [fname for fname, item in CONFIG_FNAMES.items()
                      if item['saved_as'] == 'modality_dict']
        list_dicts.append('paramsets')
        for snake in list_dicts:
            temps = getattr(self.main, snake, {})
            if temps != {}:
                try:
                    widget = getattr(self, f'widget_{snake}')
                    widget.templates = temps
                    try:
                        widget.current_template = temps[widget.current_modality][0]
                        if snake in [
                                'paramsets',
                                'quicktest_templates',
                                'persons_to_notify']:
                            widget.auto_templates = self.main.auto_templates
                            if snake == 'persons_to_notify':
                                widget.auto_vendor_templates = (
                                    self.main.auto_vendor_templates)
                        elif snake == 'digit_templates':
                            widget.paramsets = self.main.paramsets
                        elif snake == 'auto_templates':
                            widget.paramsets = self.main.paramsets
                            widget.quicktest_templates = self.main.quicktest_templates
                            widget.fill_lists()
                        elif snake == 'auto_vendor_templates':
                            widget.update_file_types()

                        widget.refresh_templist()
                    except IndexError:
                        pass
                except AttributeError:
                    pass

        if self.main.auto_common.import_path != '':
            widget = self.widget_auto_common
            widget.templates = self.main.auto_common
            widget.current_template = widget.templates.filename_pattern
            widget.update_data()
            widget.flag_edit(False)

        try:
            self.widget_persons_to_notify.templates = self.main.persons_to_notify
            self.widget_persons_to_notify.update_data()
        except AttributeError:
            pass
        try:
            self.widget_dash_settings.templates = self.main.dash_settings
            self.widget_dash_settings.update_data()
        except AttributeError:
            pass

    def mark_extra(self, widget, label_this, mod):
        """Also mark coupled templates within modality_dict."""
        all_labels = [
            temp.label for temp in widget.templates[mod]]
        if label_this in all_labels:
            idxs = get_all_matches(all_labels, label_this)
            try:
                marked_idxs = widget.marked[mod]
            except AttributeError:
                empty = {}
                for key in QUICKTEST_OPTIONS:
                    empty[key] = []
                widget.marked = empty
                widget.marked_ignore = copy.deepcopy(empty)
                marked_idxs = []
            for idx in idxs:
                if idx not in marked_idxs:
                    marked_idxs.append(idx)
                    if hasattr(widget.templates[mod][idx], 'num_digit_label'):
                        self.mark_digit_temps(widget.templates[mod][idx], mod=mod)
            widget.marked[mod] = marked_idxs

    def mark_qt_param(self, auto_template, mod='CT'):
        """Also mark used quicktest and paramset when auto_template is marked."""
        label_this = [auto_template.quicktemp_label, auto_template.paramset_label]
        for dict_no, snake in enumerate(['quicktest_templates', 'paramsets']):
            widget = getattr(self, f'widget_{snake}')
            self.mark_extra(widget, label_this[dict_no], mod)

    def mark_digit_temps(self, paramset, mod='CT'):
        """Also mark digit_template when paramset is marked."""
        if paramset.num_digit_label != '':
            widget = self.widget_digit_templates
            self.mark_extra(widget, paramset.num_digit_label, mod)

    def mark_persons(self, auto_template):
        """Also mark coupled persons to notify."""
        if len(auto_template.persons_to_notify) > 0:
            widget = self.widget_persons_to_notify
            all_labels = [temp.label for temp in widget.templates]
            for label_this in auto_template.persons_to_notify:
                if label_this in all_labels:
                    idxs = get_all_matches(all_labels, label_this)
                    try:
                        marked_idxs = widget.marked
                    except AttributeError:
                        widget.marked_ignore = []
                        marked_idxs = []
                    for idx in idxs:
                        if idx not in marked_idxs:
                            marked_idxs.append(idx)
                    widget.marked = marked_idxs

    def set_marked(self, marked=True, import_all=False):
        """Set type of marking to ImportMain."""
        self.main.marked = marked
        self.main.import_all = import_all
        self.accept()

    def get_marked(self):
        """Extract marked or not ignored templates and update ImportMain.

        Parameters
        ----------
        marked : bool
            True if import all marked, False if import all but ignored.
        """
        import_main = self.main
        marked = import_main.marked
        import_all = import_main.import_all
        if import_all is False:
            widget = self.widget_dicom_tags
            try:
                if marked:
                    if len(widget.marked) == 0:
                        import_main.tag_infos = []
                    else:
                        indexes = [widget.indexes[i] for i in widget.marked]
                        import_main.tag_infos = [
                            tag_info for tag_info
                            in import_main.tag_infos
                            if tag_info.sort_index in indexes
                            ]
                else:
                    if len(widget.marked_ignore) == 0:
                        pass
                    else:
                        ignore_ids = [widget.indexes[i] for i in widget.marked_ignore]
                        import_main.tag_infos = [
                            tag_info for tag_info
                            in import_main.tag_infos
                            if tag_info.sort_index not in ignore_ids
                            ]
            except AttributeError:
                pass  # marked not set

            list_dicts = [fname for fname, item in CONFIG_FNAMES.items()
                          if item['saved_as'] == 'modality_dict']
            list_dicts.append('paramsets')
            for snake in list_dicts:
                temps = getattr(import_main, snake, None)
                temps_new = {}
                try:
                    widget = getattr(self, f'widget_{snake}')
                    temps = getattr(import_main, snake)
                    marked_this = widget.marked if marked else widget.marked_ignore
                    for mod, marked_ids in marked_this.items():
                        if len(marked_ids) > 0:
                            if marked:
                                temps_new[mod] = []
                                for numb in marked_ids:
                                    temps_new[mod].append(temps[mod][numb])
                            else:
                                temps_new[mod] = temps[mod]
                                marked_ids.sort(reverse=True)
                                if len(marked_ids) > 0:
                                    for ign_id in marked_ids:
                                        del temps_new[mod][ign_id]
                        else:
                            if marked:
                                temps_new[mod] = []
                            else:
                                if mod in temps:
                                    temps_new[mod] = temps[mod]
                                else:
                                    temps_new[mod] = []
                    setattr(import_main, snake, temps_new)
                except AttributeError:
                    if marked:
                        setattr(import_main, snake, {})

            list_objects = [fname for fname, item in CONFIG_FNAMES.items()
                          if item['saved_as'] == 'object']
            list_objects.pop('last_modified')
            for snake in list_objects:
                proceed = True
                try:
                    widget = getattr(self, f'widget_{snake}')
                except AttributeError:
                    setattr(import_main, snake, None)
                    proceed = False
                if proceed:
                    try:
                        if marked is False and widget.marked_ignore:
                            setattr(import_main, snake, None)
                        elif marked and widget.marked is False:
                            setattr(import_main, snake, None)
                    except AttributeError:  # widget.marked not set
                        if marked:
                            setattr(import_main, snake, None)

        return import_main


class UserSettingsWidget(StackWidget):
    """Widget holding user settings."""

    def __init__(self, dlg_settings):
        """Initiate.

        Parameters
        ----------
        save_blocked : bool
            Block save button if user_preferences.yaml not available.
            Default is False.
        """
        header = 'Local settings'
        subtxt = '''Settings specific for the current user.<br>
        To be able to save any other settings, you will need to
        specify a config folder.<br>
        This config folder will hold all other settings and may
        be shared between users.<br>
        From start this may be an empty folder.'''
        super().__init__(dlg_settings, header, subtxt)

        self.config_folder = QLineEdit()
        self.lbl_user_prefs_path = QLabel()
        self.chk_dark_mode = QCheckBox()
        self.font_size = QSpinBox()

        self.vlo.addWidget(self.lbl_user_prefs_path)

        self.config_folder.setMinimumWidth(500)
        hlo_config_folder = QHBoxLayout()
        hlo_config_folder.addWidget(QLabel('Path to config folder:'))
        hlo_config_folder.addWidget(self.config_folder)
        toolbar = uir.ToolBarBrowse('Browse to find or initiate config folder')
        toolbar.act_browse.triggered.connect(
            lambda: self.locate_folder(self.config_folder))
        hlo_config_folder.addWidget(toolbar)
        self.vlo.addLayout(hlo_config_folder)
        self.vlo.addSpacing(50)

        hlo_mid = QHBoxLayout()
        vlo_1 = QVBoxLayout()
        hlo_mid.addLayout(vlo_1)
        self.vlo.addLayout(hlo_mid)

        gb_gui = QGroupBox('GUI settings')
        gb_gui.setFont(uir.FontItalic())
        vlo_gui = QVBoxLayout()
        self.font_size.setRange(5, 15)
        self.font_size.valueChanged.connect(self.flag_edit)
        hlo_font_size = QHBoxLayout()
        hlo_font_size.addWidget(QLabel('Set font size for GUI:'))
        hlo_font_size.addWidget(self.font_size)
        hlo_font_size.addWidget(QLabel('(Restart to update GUI)'))
        hlo_font_size.addStretch()
        vlo_gui.addLayout(hlo_font_size)
        hlo_dark_mode = QHBoxLayout()
        self.chk_dark_mode.clicked.connect(
            lambda: self.flag_edit(True))
        hlo_dark_mode.addWidget(QLabel('Dark mode'))
        hlo_dark_mode.addWidget(self.chk_dark_mode)
        hlo_dark_mode.addWidget(QLabel('(restart to update)'))
        hlo_dark_mode.addStretch()
        vlo_gui.addLayout(hlo_dark_mode)
        gb_gui.setLayout(vlo_gui)
        vlo_1.addWidget(gb_gui)
        vlo_1.addSpacing(50)

        hlo_mid.addStretch()

        btn_save_user_prefs = QPushButton('Save user preferences')
        btn_save_user_prefs.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}save.png'))
        btn_save_user_prefs.clicked.connect(self.save_user)
        if self.save_blocked:
            btn_save_user_prefs.setEnabled(False)
        self.vlo.addWidget(btn_save_user_prefs)

        self.vlo.addStretch()

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_from_yaml(self, initial_template_label=''):
        """Load settings from yaml and fill form."""
        _, path, self.user_prefs = cff.load_user_prefs()
        self.lbl_user_prefs_path.setText('User preferences saved in: ' + path)
        self.config_folder.setText(self.user_prefs.config_folder)
        self.font_size.setValue(self.user_prefs.font_size)
        self.chk_dark_mode.setChecked(self.user_prefs.dark_mode)
        self.flag_edit(False)

    def save_user(self):
        """Get current settings and save to yaml file."""
        if self.user_prefs.config_folder != self.config_folder.text():
            cff.remove_user_from_active_users()
        self.user_prefs.config_folder = self.config_folder.text()
        self.user_prefs.font_size = self.font_size.value()
        self.user_prefs.dark_mode = self.chk_dark_mode.isChecked()

        status_ok, path = cff.save_user_prefs(self.user_prefs, parentwidget=self)
        if status_ok:
            self.status_label.setText(f'Changes saved to {path}')
            self.flag_edit(False)
            cff.add_user_to_active_users()
        else:
            QMessageBox.Warning(self, 'Warning',
                                f'Failed to save changes to {path}')


class SharedSettingsWidget(StackWidget):
    """Widget for shared settings."""

    def __init__(self, dlg_settings):
        header = 'Config folder - shared settings'
        subtxt = '''Each of the sub-pages will display different settings
         saved in the config folder (specified in user settings).<br>
        Templates and settings will be saved as .yaml files. <br>
        Several users may link to the same config folder and
         share these settings.'''
        super().__init__(dlg_settings, header, subtxt)
        self.width1 = dlg_settings.main.gui.panel_width*0.3
        self.width2 = dlg_settings.main.gui.panel_width*1.7

        self.lbl_config_folder = QLabel('-- not defined --')
        self.list_files = QListWidget()

        hlo_cf = QHBoxLayout()
        self.vlo.addLayout(hlo_cf)
        hlo_cf.addWidget(QLabel('Config folder: '))
        hlo_cf.addWidget(self.lbl_config_folder)
        hlo_cf.addStretch()
        btn_locate_config = QPushButton('Locate new or exisiting config folder')
        btn_locate_config.clicked.connect(self.locate_config)
        btn_import = QPushButton(
            'Import from another config folder')
        btn_import.clicked.connect(self.dlg_settings.import_from_yaml)
        btn_import_idl_config = QPushButton(
            'Import config from IDL version of ImageQC')
        btn_import_idl_config.clicked.connect(self.import_idl_config)
        btn_verify_config = QPushButton((
            'Verify that the config files have the necessary connections defined'
            ' (for troubleshooting)'))
        btn_verify_config.clicked.connect(self.verify_config_files)
        self.vlo.addWidget(btn_locate_config)
        self.vlo.addWidget(btn_import)
        self.vlo.addWidget(btn_import_idl_config)
        self.vlo.addWidget(btn_verify_config)

        if self.save_blocked:
            btn_locate_config.setEnabled(False)
            btn_import_idl_config.setEnabled(False)

        self.vlo.addWidget(self.list_files)

    def update_from_yaml(self, initial_template_label=''):
        """Update settings from yaml file."""
        self.list_files.clear()

        path = cff.get_config_folder()
        if path != '':
            self.lbl_config_folder.setText(path)
            active_users = cff.get_active_users()
            self.list_files.addItem('Active users:')
            for user, lastsession in active_users.items():
                self.list_files.addItem(' '.join(['  ', user, lastsession]))
            self.list_files.addItem('')

            status_ok, path, last_modified = cff.load_settings(fname='last_modified')
            if status_ok:
                for cfn in CONFIG_FNAMES:
                    if cff.get_config_filename(cfn) != '':  # in case deleted
                        try:
                            res = getattr(last_modified, cfn)
                            if len(res) > 0:
                                self.list_files.addItem(cfn + ':')
                                string = ' '.join(
                                    ['    last edited by',
                                     res[0], time_diff_string(res[1]),
                                     '(', ctime(res[1]), ')'])
                                if len(res) > 2:  # with version number
                                    string = string + ' in version ' + res[2]
                                self.list_files.addItem(string)
                        except AttributeError:
                            pass
                    if cfn == 'paramsets':
                        for mod in [*QUICKTEST_OPTIONS]:
                            try:
                                res = getattr(last_modified, f'{cfn}_{mod}')
                                if len(res) > 0:
                                    self.list_files.addItem(f'{cfn}_{mod}:')
                                    string = ' '.join(
                                        ['    last edited by',
                                         res[0], time_diff_string(res[1]),
                                         '(', ctime(res[1]), ')'])
                                    if len(res) > 2:  # with version number
                                        string = string + ' in version ' + res[2]
                                    self.list_files.addItem(string)
                            except AttributeError:
                                pass
        else:
            self.lbl_config_folder.setText('-- not defined --')

    def locate_config(self):
        """Browse to config folder."""
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec():
            config_folder = dlg.selectedFiles()[0]
            self.change_config_user_prefs(config_folder)

    def change_config_user_prefs(self, folder):
        """Save new config folder and update."""
        _, _, user_prefs = cff.load_user_prefs()
        user_prefs.config_folder = os.path.normpath(folder)
        _, _ = cff.save_user_prefs(user_prefs, parentwidget=self)
        self.update_from_yaml()

    def verify_config_files(self):
        """Verify content of config files (that linked templates are defined)."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        qApp.processEvents()
        _, _, tag_infos = cff.load_settings(fname='tag_infos')
        _, _, tag_patterns_special = cff.load_settings(fname='tag_patterns_special')
        _, _, tag_patterns_format = cff.load_settings(fname='tag_patterns_format')
        _, _, tag_patterns_sort = cff.load_settings(fname='tag_patterns_sort')
        _, _, rename_patterns = cff.load_settings(fname='rename_patterns')
        _, _, digit_templates = cff.load_settings(fname='digit_templates')
        _, _, quicktest_templates = cff.load_settings(fname='quicktest_templates')
        _, _, paramsets = cff.load_settings(fname='paramsets')
        _, _, auto_common = cff.load_settings(fname='auto_common')
        _, _, auto_templates = cff.load_settings(fname='auto_templates')
        # _, _, dash_settings = cff.load_settings(fname='dash_settings')
        _, _, persons_to_notify = cff.load_settings(fname='persons_to_notify')

        all_temps = ImportMain(
            tag_infos=tag_infos,
            tag_patterns_special=tag_patterns_special,
            tag_patterns_format=tag_patterns_format,
            tag_patterns_sort=tag_patterns_sort,
            rename_patterns=rename_patterns,
            digit_templates=digit_templates,
            quicktest_templates=quicktest_templates,
            paramsets=paramsets,
            auto_common=auto_common,
            auto_templates=auto_templates,
            # dash_settings=dash_settings,
            persons_to_notify=persons_to_notify
            )

        status_ti, log_ti, _ = cff.get_taginfos_used_in_templates(all_temps)
        details = []
        if status_ti is False:
            details.append('Tag infos used but not defined:')
            details.extend(log_ti)
        status_au, log_au = cff.verify_auto_templates(all_temps)
        if status_au is False:
            details.append(
                'Automation templates linked to undefined parameter sets or QuickTest '
                'templates:')
            details.extend(log_au)
        status_param, log_param = cff.verify_paramsets(all_temps)
        if status_param is False:
            details.append(
                'Parameter sets linked to undefined Digit Templates')
            details.extend(log_param)
        if len(details) > 0:
            msg = 'Found issues with the templates. See details.'
            icon = QMessageBox.Warning
        else:
            msg = 'All templates are ok.'
            icon = QMessageBox.Information

        QApplication.restoreOverrideCursor()
        dlg = messageboxes.MessageBoxWithDetails(
            self, title='Results', msg=msg, details=details, icon=icon)
        dlg.exec()

    def import_idl_config(self):
        """Import config settings from IDL version of imageQC."""
        if self.lbl_config_folder.text() == '':
            QMessageBox.information(
                self,
                'Config path missing',
                'A config folder need to be defined to save the imported settings.'
                )
        else:
            fname = QFileDialog.getOpenFileName(
                self, 'Import config.dat from IDL version',
                filter="dat file (*.dat)")
            if fname[0] != '':
                _, _, self.tag_infos = cff.load_settings(fname='tag_infos')
                config_idl = ConfigIdl2Py(fname[0], self.tag_infos)
                if len(config_idl.errmsg) > 0:
                    dlg = messageboxes.MessageBoxWithDetails(
                        parent=self, title='Information',
                        msg='Imported parameters will now be presented for review.',
                        info='See details for warnings regarding the imported config file',
                        details=config_idl.errmsg)
                    dlg.exec()
                import_main = ImportMain(
                    tag_infos=config_idl.tag_infos_new,
                    rename_patterns=config_idl.rename_patterns,
                    paramsets=config_idl.paramsets,
                    quicktest_templates=config_idl.quicktest_templates,
                    auto_common=config_idl.auto_common,
                    auto_templates=config_idl.auto_templates,
                    auto_vendor_templates=config_idl.auto_vendor_templates
                    )
                if config_idl.paramsets:
                    import_main.current_paramset = config_idl.paramsets['CT'][0]
                dlg = SettingsDialog(
                    import_main, initial_view='Config folder',
                    width1=self.width1, width2=self.width2,
                    import_review_mode=True)
                res = dlg.exec()
                if res:
                    import_main = dlg.get_marked()
                    same_names = cff.import_settings(import_main)
                    if same_names:
                        QMessageBox.information(
                            self, 'Information',
                            ('Imported one or more templates with same name as '
                             'templates already set. The imported template names '
                             'was marked with _import.'))
                self.update_from_yaml()


class SharedSettingsImportWidget(StackWidget):
    """Widget to replace SharedSettingsWidget when import_review_mode."""

    def __init__(self, dlg_settings):
        header = 'Settings for import'
        subtxt = '''Mark templates for import or mark templates to ignore.<br>
        Then get back to this window to import according to your selections.'''
        super().__init__(dlg_settings, header, subtxt)
        btn_all = QPushButton('Import all')
        btn_all_but = QPushButton('Import all except for those marked to ignore')
        btn_marked = QPushButton('Import only marked')
        self.vlo.addWidget(btn_all)
        self.vlo.addWidget(btn_marked)
        self.vlo.addWidget(btn_all_but)
        self.vlo.addStretch()

        header_text = """<html><head/><body>
            <p><span style=\" font-size:20pt;color:gray\"><i>Review mode!</i></span></p>
            </body></html>"""
        info_text = """<html><head/><body>
            <p><span style=\" font-size:14pt;color:gray\"><i>Return to this tab
            (from tree list at your left hand) when you have decided what to include
            and not</i></span></p></body></html>"""
        self.vlo.addWidget(QLabel(header_text))
        self.vlo.addWidget(QLabel(info_text))
        self.vlo.addStretch()

        btn_all.clicked.connect(
            lambda: self.dlg_settings.set_marked(True, import_all=True))
        btn_marked.clicked.connect(
            lambda: self.dlg_settings.set_marked(True))
        btn_all_but.clicked.connect(
            lambda: self.dlg_settings.set_marked(False))


class QuickTestTemplatesWidget(StackWidget):
    """Widget holding QuickTest pattern settings."""

    def __init__(self, dlg_settings):
        header = 'QuickTest templates'
        subtxt = '''Which tests to perform on which images.<br>
        Optionally also define the label for the images and/or groups.<br>
        These labels will be used in headers when generating QuickTest output as defined
        in the parameter set either during automation or copy QuickTest results to
        clipboard.<br>
        If no label is set img0, img1.../group0, group1... will be used.<br>
        A group is by default images with the same seriesUID, but this can be edited in
        parameter set - output.'''
        super().__init__(dlg_settings, header, subtxt,
                         mod_temp=True, grouped=True)
        self.fname = 'quicktest_templates'
        self.empty_template = cfc.QuickTestTemplate()
        self.finished_init = False

        if self.import_review_mode is False:
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_move_modality)
        self.wid_mod_temp.vlo.addWidget(
            QLabel('Selected template used in Automation template:'))
        self.list_used_in = QListWidget()
        self.wid_mod_temp.vlo.addWidget(self.list_used_in)

        vlo = QVBoxLayout()
        self.hlo.addLayout(vlo)

        self.wid_test_table = QuickTestTreeView(self)
        vlo.addWidget(self.wid_test_table)

        hlo_nimgs = QHBoxLayout()
        hlo_nimgs.setAlignment(Qt.AlignLeft)
        vlo.addLayout(hlo_nimgs)
        hlo_nimgs.addWidget(QLabel('Minimum number of images expected: '))
        self.lbl_nimgs = QLabel('')
        hlo_nimgs.addWidget(self.lbl_nimgs)

        self.toolb = QToolBar()
        self.toolb.setOrientation(Qt.Vertical)
        self.hlo.addWidget(self.toolb)
        act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add new row after selected row', self)
        act_add.triggered.connect(self.wid_test_table.insert_empty_row)
        act_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete selected row', self)
        act_delete.triggered.connect(self.wid_test_table.delete_row)
        self.toolb.addActions([act_add, act_delete])

        if self.import_review_mode:
            self.wid_test_table.setEnabled(False)
            self.toolb.setEnabled(False)

        self.vlo.addWidget(uir.HLine())
        self.vlo.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with selected template."""
        self.wid_test_table.update_data()
        self.lbl_nimgs.setText(f'{len(self.current_template.tests)}')
        self.update_used_in()

    def update_used_in(self):
        """Update list of auto-templates where this template is used."""
        self.list_used_in.clear()
        if self.current_template.label != '':
            if self.current_modality in self.auto_templates:
                auto_labels = [
                    temp.label for temp in self.auto_templates[self.current_modality]
                    if temp.quicktemp_label == self.current_template.label
                    ]
                if len(auto_labels) > 0:
                    self.list_used_in.addItems(auto_labels)

    def get_current_template(self):
        """Get self.current_template."""
        self.current_template = self.wid_test_table.get_data()
        # save save space and read yaml time:
        if not any(self.current_template.image_names):
            self.current_template.image_names = []
        if not any(self.current_template.group_names):
            self.current_template.group_names = []


@dataclass
class ImportMain:
    """Class to replace MainWindow + hold imported templates when import_review_mode."""

    save_blocked: bool = True
    marked: bool = True
    include_all: bool = False
    current_modality: str = 'CT'
    current_paramset: dict = field(default_factory=dict)
    current_quicktest: dict = field(default_factory=dict)
    # converted from dict to paramset of correct modality when initialized
    tag_infos: list = field(default_factory=list)
    # TODO delete? tag_infos_new: list = field(default_factory=list)
    tag_patterns_special: dict = field(default_factory=dict)
    tag_patterns_format: dict = field(default_factory=dict)
    tag_patterns_sort: dict = field(default_factory=dict)
    rename_patterns: dict = field(default_factory=dict)
    digit_templates: dict = field(default_factory=dict)
    quicktest_templates: dict = field(default_factory=dict)
    paramsets: dict = field(default_factory=dict)
    auto_common: cfc.AutoCommon = field(default_factory=cfc.AutoCommon)
    auto_templates: dict = field(default_factory=dict)
    auto_vendor_templates: dict = field(default_factory=dict)
    dash_settings: cfc.DashSettings = field(default_factory=cfc.DashSettings)
    persons_to_notify: list = field(default_factory=list)


class SelectImportFilesDialog(ImageQCDialog):
    """Dialog to select files to import from."""

    def __init__(self, filenames):
        super().__init__()
        self.setWindowTitle('Select files to import from')
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        vlo.addWidget(QLabel('Select files to import from'))
        self.list_widget = uir.ListWidgetCheckable(
            texts=filenames,
            set_checked_ids=list(np.arange(len(filenames)))
            )
        vlo.addWidget(self.list_widget)
        self.btn_select_all = QPushButton('Deselect all')
        self.btn_select_all.clicked.connect(self.select_all)
        vlo.addWidget(self.btn_select_all)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        button_box = QDialogButtonBox(buttons)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        vlo.addWidget(button_box)

    def select_all(self):
        """Select or deselect all in list."""
        if self.btn_select_all.text() == 'Deselect all':
            set_state = Qt.Unchecked
            self.btn_select_all.setText('Select all')
        else:
            set_state = Qt.Checked
            self.btn_select_all.setText('Deselect all')

        for i in range(len(self.list_widget.texts)):
            item = self.list_widget.item(i)
            item.setCheckState(set_state)

    def get_checked_files(self):
        """Get list of checked testcode ids."""
        return self.list_widget.get_checked_texts()
