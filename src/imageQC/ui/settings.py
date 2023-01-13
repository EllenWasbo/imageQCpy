#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for configuration settings.

@author: Ellen Wasbo
"""
from __future__ import annotations

import os
from time import time, ctime
from dataclasses import asdict, dataclass, field
import copy
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, qApp, QSizePolicy,
    QDialog, QWidget, QTreeWidget, QTreeWidgetItem, QStackedWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QToolBar, QButtonGroup,
    QLabel, QLineEdit, QPushButton, QAction, QSpinBox, QCheckBox,
    QListWidget, QListWidgetItem, QComboBox,
    QAbstractScrollArea,
    QInputDialog, QMessageBox, QDialogButtonBox,
    QFileDialog
    )
import pydicom

# imageQC block start
from imageQC.config.iQCconstants import (
    QUICKTEST_OPTIONS, CONFIG_FNAMES, ENV_ICON_PATH,
    ENV_USER_PREFS_PATH, LOG_FILENAME,
    VENDOR_FILE_OPTIONS
    )
import imageQC.config.config_func as cff
import imageQC.config.config_classes as cfc
from imageQC.config.read_config_idl import ConfigIdl2Py
import imageQC.ui.reusables as uir
from imageQC.scripts.mini_methods_format import time_diff_string, valid_template_name
import imageQC.scripts.read_vendor_QC_reports as read_vendor_QC_reports
import imageQC.scripts.dcm as dcm
# imageQC block end


class SettingsDialog(uir.ImageQCDialog):
    """GUI setup for the settings dialog window."""

    def __init__(
            self, main, initial_view='User local settings',
            width1=200, width2=800,
            import_review_mode=False):
        super().__init__()
        self.main = main
        self.import_review_mode = import_review_mode
        if import_review_mode is False:
            self.setWindowTitle('Settings manager')
            width1 = self.main.vGUI.panel_width*0.3
            width2 = self.main.vGUI.panel_width*1.7
        else:
            self.setWindowTitle('Import review - configuration settings')

        hLO = QHBoxLayout()
        self.setLayout(hLO)

        self.treeSettings = QTreeWidget()
        hLO.addWidget(self.treeSettings)
        self.stacked_widget = QStackedWidget()
        hLO.addWidget(self.stacked_widget)

        self.treeSettings.setColumnCount(1)
        self.treeSettings.setFixedWidth(width1)
        self.treeSettings.setHeaderHidden(True)
        self.treeSettings.itemClicked.connect(self.change_widget)

        self.stacked_widget.setFixedWidth(width2)

        self.list_txt_item_widget = []
        if import_review_mode is False:
            txt = 'Local settings'
            self.item_user_settings = QTreeWidgetItem([txt])
            self.treeSettings.addTopLevelItem(self.item_user_settings)
            self.widget_user_settings = UserSettingsWidget(
                save_blocked=self.main.save_blocked)
            self.stacked_widget.addWidget(self.widget_user_settings)
            self.list_txt_item_widget = [
                (txt, self.item_user_settings, self.widget_user_settings)]

            txt = 'Config folder'
            self.item_shared_settings = QTreeWidgetItem([txt])
            self.treeSettings.addTopLevelItem(self.item_shared_settings)
            self.widget_shared_settings = SharedSettingsWidget(
                save_blocked=self.main.save_blocked, width1=width1, width2=width2)
            self.stacked_widget.addWidget(self.widget_shared_settings)
            self.list_txt_item_widget.append(
                (txt, self.item_shared_settings, self.widget_shared_settings))
        else:
            txt = 'Settings for import'
            self.item_shared_settings = QTreeWidgetItem([txt])
            self.treeSettings.addTopLevelItem(self.item_shared_settings)
            self.widget_shared_settings = SharedSettingsImportWidget(self)
            self.stacked_widget.addWidget(self.widget_shared_settings)
            self.list_txt_item_widget.append(
                (txt, self.item_shared_settings, self.widget_shared_settings))

        txt = 'DICOM tags'
        self.item_dicom_tags = QTreeWidgetItem([txt])
        self.item_shared_settings.addChild(self.item_dicom_tags)
        self.widget_dicom_tags = DicomTagsWidget(
            save_blocked=self.main.save_blocked,
            import_review_mode=import_review_mode)
        self.stacked_widget.addWidget(self.widget_dicom_tags)
        self.list_txt_item_widget.append(
            (txt, self.item_dicom_tags, self.widget_dicom_tags))

        proceed = True
        if import_review_mode:
            if self.main.tag_patterns_special == {}:
                proceed = False
        if proceed:
            txt = 'Special tag patterns'
            self.item_tag_patterns_special = QTreeWidgetItem([txt])
            self.item_dicom_tags.addChild(self.item_tag_patterns_special)
            self.widget_tag_patterns_special = TagPatternSpecialWidget(
                save_blocked=self.main.save_blocked,
                import_review_mode=import_review_mode)
            self.stacked_widget.addWidget(self.widget_tag_patterns_special)
            self.list_txt_item_widget.append(
                (txt, self.item_tag_patterns_special,
                 self.widget_tag_patterns_special))

        proceed = True
        if import_review_mode:
            if self.main.tag_patterns_format == {}:
                proceed = False
        if proceed:
            txt = 'Tag patterns - format'
            self.item_tag_patterns_format = QTreeWidgetItem([txt])
            self.item_dicom_tags.addChild(self.item_tag_patterns_format)
            self.widget_tag_patterns_format = TagPatternFormatWidget(
                save_blocked=self.main.save_blocked,
                import_review_mode=import_review_mode)
            self.stacked_widget.addWidget(self.widget_tag_patterns_format)
            self.list_txt_item_widget.append(
                (txt, self.item_tag_patterns_format,
                 self.widget_tag_patterns_format))

        proceed = True
        if import_review_mode:
            if self.main.rename_patterns == {}:
                proceed = False
        if proceed:
            txt = 'Rename patterns'
            self.item_rename_patterns = QTreeWidgetItem([txt])
            self.item_dicom_tags.addChild(self.item_rename_patterns)
            self.widget_rename_patterns = RenamePatternWidget(
                save_blocked=self.main.save_blocked,
                import_review_mode=import_review_mode)
            self.stacked_widget.addWidget(self.widget_rename_patterns)
            self.list_txt_item_widget.append(
                (txt, self.item_rename_patterns,
                 self.widget_rename_patterns))

        proceed = True
        if import_review_mode:
            if self.main.tag_patterns_sort == {}:
                proceed = False
        if proceed:
            txt = 'Tag patterns - sort'
            self.item_tag_patterns_sort = QTreeWidgetItem([txt])
            self.item_dicom_tags.addChild(self.item_tag_patterns_sort)
            self.widget_tag_patterns_sort = TagPatternSortWidget(
                save_blocked=self.main.save_blocked,
                import_review_mode=import_review_mode)
            self.stacked_widget.addWidget(self.widget_tag_patterns_sort)
            self.list_txt_item_widget.append(
                (txt, self.item_tag_patterns_sort, self.widget_tag_patterns_sort))

        proceed = True
        if import_review_mode:
            if self.main.paramsets == {}:
                proceed = False
        if proceed:
            txt = 'Parameter sets / output'
            self.item_paramsets = QTreeWidgetItem([txt])
            self.item_shared_settings.addChild(self.item_paramsets)
            self.widget_paramsets = ParamSetsWidget(
                save_blocked=self.main.save_blocked,
                main_current_paramset=self.main.current_paramset,
                main_current_modality=self.main.current_modality,
                import_review_mode=import_review_mode)
            self.stacked_widget.addWidget(self.widget_paramsets)
            self.list_txt_item_widget.append(
                (txt, self.item_paramsets, self.widget_paramsets))

        proceed = True
        if import_review_mode:
            if self.main.quicktest_templates == {}:
                proceed = False
        if proceed:
            txt = 'QuickTest templates'
            self.item_quicktest_templates = QTreeWidgetItem([txt])
            self.item_shared_settings.addChild(self.item_quicktest_templates)
            self.widget_quicktest_templates = QuickTestTemplatesWidget(
                save_blocked=self.main.save_blocked,
                import_review_mode=import_review_mode)
            self.stacked_widget.addWidget(self.widget_quicktest_templates)
            self.list_txt_item_widget.append(
                (txt, self.item_quicktest_templates,
                 self.widget_quicktest_templates))

        proceed = True
        if import_review_mode:
            if (
                    self.main.auto_common.import_path == ''
                    and self.main.auto_templates == {}
                    and self.main.auto_vendor_templates == {}
                    ):
                proceed = False
        if proceed:
            txt = 'Automation'
            self.item_auto_info = QTreeWidgetItem([txt])
            self.item_shared_settings.addChild(self.item_auto_info)
            self.widget_auto_info = AutoInfoWidget()
            self.stacked_widget.addWidget(self.widget_auto_info)
            self.list_txt_item_widget.append(
                (txt, self.item_auto_info, self.widget_auto_info))

            proceed = True
            if import_review_mode:
                if self.main.auto_common.import_path == '':
                    proceed = False
            if proceed:
                txt = 'Import settings'
                self.item_auto_common = QTreeWidgetItem([txt])
                self.item_auto_info.addChild(self.item_auto_common)
                self.widget_auto_common = AutoCommonWidget(
                    save_blocked=self.main.save_blocked,
                    import_review_mode=import_review_mode)
                self.stacked_widget.addWidget(self.widget_auto_common)
                self.list_txt_item_widget.append(
                    (txt, self.item_auto_common, self.widget_auto_common))

            proceed = True
            if import_review_mode:
                if self.main.auto_templates == {}:
                    proceed = False
                else:
                    list_lens = [len(t) for mod, t
                                 in self.main.auto_templates.items()]
                    if max(list_lens) > 0:
                        proceed = False
            if proceed:
                txt = 'Templates DICOM'
                self.item_auto_templates = QTreeWidgetItem([txt])
                self.item_auto_info.addChild(self.item_auto_templates)
                self.widget_auto_templates = AutoTemplateWidget(
                    save_blocked=self.main.save_blocked,
                    import_review_mode=import_review_mode)
                self.stacked_widget.addWidget(self.widget_auto_templates)
                self.list_txt_item_widget.append(
                    (txt, self.item_auto_templates, self.widget_auto_templates))

            proceed = True
            if import_review_mode:
                if self.main.auto_vendor_templates == {}:
                    proceed = False
                else:
                    list_lens = [len(t) for mod, t
                                 in self.main.auto_vendor_templates.items()]
                    if max(list_lens) > 0:
                        proceed = False
            if proceed:
                txt = 'Templates vendor files'
                self.item_auto_vendor_templates = QTreeWidgetItem([txt])
                self.item_auto_info.addChild(self.item_auto_vendor_templates)
                self.widget_auto_vendor_templates = AutoVendorTemplateWidget(
                    save_blocked=self.main.save_blocked,
                    import_review_mode=import_review_mode)
                self.stacked_widget.addWidget(self.widget_auto_vendor_templates)
                self.list_txt_item_widget.append(
                    (txt, self.item_auto_vendor_templates,
                     self.widget_auto_vendor_templates))

        item, widget = self.get_item_widget_from_txt(initial_view)
        self.treeSettings.setCurrentItem(item)
        self.treeSettings.expandToDepth(2)
        self.treeSettings.resizeColumnToContents(0)
        self.stacked_widget.setCurrentWidget(widget)
        self.previous_selected_txt = initial_view
        self.current_selected_txt = initial_view
        self.initial_modality = self.main.current_modality

        if import_review_mode is False:
            widget.update_from_yaml()
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

    def change_widget(self, item, col):
        """Update visible widget in stack when selection in tree change."""
        prevtxtitem = self.current_selected_txt
        item = self.treeSettings.indexFromItem(item)
        txtitem = item.data(Qt.DisplayRole)

        # Settings changed - saved? Go back to prev if regret leaving unchanged
        prev_item, prev_widget = self.get_item_widget_from_txt(prevtxtitem)
        edited = False
        try:
            edited = getattr(prev_widget, 'edited')
        except AttributeError:
            pass

        proceed = True
        if edited:
            proceed = uir.proceed_question(
                self, 'Proceed and loose unsaved changes?')

        if proceed:
            self.previous_selected_txt = self.current_selected_txt
            self.current_selected_txt = txtitem
            new_item, new_widget = self.get_item_widget_from_txt(txtitem)
            self.stacked_widget.setCurrentWidget(new_widget)
            new_widget.current_modality = prev_widget.current_modality
            if self.import_review_mode == False:
                new_widget.update_from_yaml()
        else:
            item, widget = self.get_item_widget_from_txt(
                self.previous_selected_txt)
            self.treeSettings.setCurrentItem(item)

    def closeEvent(self, event):
        """Test if unsaved changes before closing."""
        prevtxtitem = self.current_selected_txt
        prev_item, prev_widget = self.get_item_widget_from_txt(prevtxtitem)
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

    def update_import_main(self):
        """Update templates of all widgets according to import_main."""
        if self.main.tag_infos != []:
            self.widget_dicom_tags.templates = self.main.tag_infos
            self.widget_dicom_tags.update_data()
        list_dicts = [fname for fname, item in CONFIG_FNAMES.items()
                      if item['saved_as'] == 'modality_dict']
        for d in list_dicts:
            temps = getattr(self.main, d, {})
            try:
                widget = getattr(self, f'widget_{d}')
                widget.templates = temps
                try:
                    widget.current_template = temps['CT'][0]
                except IndexError:
                    pass
                widget.refresh_templist()
            except AttributeError:
                pass

        if self.main.auto_common.import_path != '':
            widget = self.widget_auto_common
            widget.templates = self.main.auto_common
            widget.current_template = widget.templates.filename_pattern
            widget.update_data()

    def set_marked(self, marked, import_all=False):
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
        include_all = import_main.include_all
        if include_all is False:
            list_dicts = [fname for fname, item in CONFIG_FNAMES.items()
                          if item['saved_as'] == 'modality_dict']

            widget = self.widget_dicom_tags
            try:
                if marked:
                    if len(widget.marked) == 0:
                        import_main.tag_infos = []
                    else:
                        import_main.tag_infos = import_main.tag_infos[
                            widget.marked]
                else:
                    if len(widget.marked_ignore) == 0:
                        pass
                    else:
                        ignore_ids = widget.marked_ignore
                        ignore_ids.sort(reverse=True)
                        for ign_id in ignore_ids:
                            del import_main.tag_infos[ign_id]
            except AttributeError:
                pass  # marked not set

            for d in list_dicts:
                temps = getattr(import_main, d, None)
                try:
                    widget = getattr(self, f'widget_{d}')
                    temps = getattr(import_main, d)
                    marked_this = widget.marked if marked else widget.marked_ignore
                    for mo, marked_ids in marked_this.items():
                        if len(marked_ids) == 0:
                            if marked:
                                temps[mo] = []
                        else:
                            if marked:
                                temps[mo] = temps[marked_ids]
                            else:
                                marked_ids.sort(reverse=True)
                                if len(marked_ids) > 0:
                                    for ign_id in marked_ids:
                                        del temps[mo][ign_id]
                except AttributeError:
                    pass  # marked not set

            try:
                widget = self.widget_auto_common
                if marked is False and widget.marked_ignore:
                    import_main.auto_common = None
            except AttributeError:
                import_main.auto_common = None  # marked not set

        return import_main


class StackWidget(QWidget):
    """Class for general widget attributes for the stacked widgets."""

    def __init__(self, header='', subtxt='', typestr='',
                 mod_temp=False, grouped=False, editable=True,
                 import_review_mode=False):
        """Initiate StackWidget.

        Parameters
        ----------
        header : str
            header text
        subtxt : str
            info text under header text
        typestr : str
            string to set type of data (parameterset or template)
            title over labels and used in tooltip
            for the buttons of ModTempSelector
        mod_temp : bool
            add ModTempSelector
        grouped : bool
            include modality selection for the ModTempSelector
        editable : bool
            False = hide toolbar for save/edit + hide edited
        """
        super().__init__()

        self.edited = False
        self.typestr = typestr
        self.mod_temp = mod_temp
        self.grouped = grouped
        self.import_review_mode = import_review_mode
        self.current_modality = 'CT'  # default get from latest selection
        self.status_label = QLabel('')

        self.vLO = QVBoxLayout()
        self.setLayout(self.vLO)
        if header != '':
            self.vLO.addWidget(uir.LabelHeader(header, 3))
        if subtxt != '':
            self.vLO.addWidget(uir.LabelItalic(subtxt))
        self.vLO.addWidget(uir.HLine())

        if self.mod_temp:
            self.hLO = QHBoxLayout()
            self.vLO.addLayout(self.hLO)
            self.wModTemp = ModTempSelector(
                self, editable=editable, import_review_mode=import_review_mode)
            self.hLO.addWidget(self.wModTemp)
            if self.grouped is False:
                self.wModTemp.lbl_modality.setVisible(False)
                self.wModTemp.cbox_modality.setVisible(False)

    def flag_edit(self, flag=True):
        """Indicate some change."""
        if flag:
            self.edited = True
            self.status_label.setText('**Unsaved changes**')
        else:
            self.edited = False
            self.status_label.setText('')

    def update_from_yaml(self):
        """Refresh settings from yaml file."""
        self.lastload = time()

        if hasattr(self, 'fname'):
            ok, path, self.templates = cff.load_settings(fname=self.fname)
            if 'patterns' in self.fname or self.fname == 'auto_templates':
                ok, path, self.tag_infos = cff.load_settings(fname='tag_infos')
            if self.fname == 'auto_templates':
                ok, path, self.paramsets = cff.load_settings(fname='paramsets')
                ok, path, self.quicktests = cff.load_settings(
                    fname='quicktest_templates')
                self.fill_lists()

            if self.grouped:
                self.wModTemp.cbox_modality.setCurrentText(
                    self.current_modality)
                if 'patterns' in self.fname:
                    try:
                        self.wTagPattern.fill_list_tags(self.current_modality)
                    except AttributeError:
                        pass  # ignore if editable == False

            if self.mod_temp:
                self.refresh_templist()
            else:
                self.update_data()

    def update_modality(self):
        """Refresh GUI after selecting modality (stack with ModTempSelector."""
        if self.edited:
            res = uir.QuestionBox(
                parent=self, title='Save changes?',
                msg='Save changes before changing modality?')
            if res.exec():
                self.wModTemp.save()

        self.current_modality = self.wModTemp.cbox_modality.currentText()

        if 'patterns' in self.fname:
            try:
                self.wTagPattern.fill_list_tags(self.current_modality)
            except AttributeError:
                pass  # ignore if editable = False
        elif self.fname == 'paramsets':
            self.empty_template = self.get_empty_paramset()
        elif self.fname == 'quicktest_templates':
            self.wTestTable.update_modality()
        elif self.fname == 'quicktest_output_templates':
            self.wOutputTable.update_modality()
        elif 'vendor' in self.fname:
            self.update_file_types()
        elif self.fname == 'auto_templates':
            self.fill_lists()

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
        if self.grouped:
            try:
                self.current_labels = \
                    [obj.label for obj
                     in self.templates[self.current_modality]]
            except KeyError:  # fx on import review mode from IDL SPECT
                self.current_labels = []
        else:
            self.current_labels = [obj.label for obj in self.templates]

        if selected_label != '':
            tempno = self.current_labels.index(selected_label)
        else:
            tempno = selected_id
        if tempno < 0:
            tempno = 0
        if tempno > len(self.current_labels)-1:
            tempno = len(self.current_labels)-1

        if len(self.current_labels) == 0:
            self.current_template = copy.deepcopy(self.empty_template)
        else:
            self.update_current_template(selected_id=tempno)

        self.wModTemp.list_temps.clear()
        if self.import_review_mode:
            self.refresh_templist_icons()
        else:
            self.wModTemp.list_temps.addItems(self.current_labels)
        self.wModTemp.list_temps.setCurrentRow(tempno)

        if 'auto' in self.fname and 'temp' in self.fname:
            if self.current_modality in self.templates:
                active = [obj.active for obj in self.templates[self.current_modality]]
                brush = QBrush(QColor(170, 170, 170))
                for i in range(len(active)):
                    if active[i] is False:
                        self.wModTemp.list_temps.item(i).setForeground(brush)
        self.update_data()

    def refresh_templist_icons(self):
        """Set green if marked for import and red if marked for ignore.

        Used if import review mode.
        """
        if hasattr(self, 'marked'):
            current_marked = self.marked[self.current_modality]
            current_ignore = self.marked_ignore[self.current_modality]
        else:
            current_marked = []
            current_ignore = []

        for i, label in enumerate(self.current_labels):
            if i in current_marked:
                icon = QIcon(f'{os.environ[ENV_ICON_PATH]}ok.png')
            elif i in current_ignore:
                icon = QIcon(f'{os.environ[ENV_ICON_PATH]}deleteRed.png')
            else:
                icon = QIcon()

            self.wModTemp.list_temps.addItem(QListWidgetItem(icon, label))

    def update_clicked_template(self):
        """Update data after new template selected (clicked)."""
        if self.edited:
            res = uir.QuestionBox(
                parent=self, title='Save changes?',
                msg='Save changes before changing template?')
            if res.exec():
                self.wModTemp.save(label=self.current_template.label)

        tempno = self.wModTemp.list_temps.currentIndex().row()
        self.update_current_template(selected_id=tempno)
        self.update_data()

    def update_current_template(self, selected_id=0):
        """Update self.current_template by label or id."""
        if self.grouped:
            self.current_template = copy.deepcopy(
                self.templates[self.current_modality][selected_id])
        else:
            self.current_template = copy.deepcopy(
                self.templates[selected_id])

    def set_empty_template(self):
        """Set default template when last in template list is deleted."""
        if self.grouped:
            if self.fname == 'paramsets':
                self.templates[self.current_modality] = [
                    self.get_empty_paramset()]
            else:
                self.templates[self.current_modality] = [
                    copy.deepcopy(self.empty_template)]
        else:
            self.templates = [copy.deepcopy(self.empty_template)]

    def locate_folder(self, widget):
        """Locate folder and set widget.text() to path.

        Parameters
        ----------
        widget : QLineEdit
            reciever of the path text
        """
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if widget.text() != '':
            dlg.setDirectory(widget.text())
        if dlg.exec():
            fname = dlg.selectedFiles()
            widget.setText(os.path.normpath(fname[0]))
            self.flag_edit()

    def locate_file(self, widget, title='Locate file',
                    filter_str='All files (*)', opensave=False):
        """Locate file and set widget.text() to path.

        Parameters
        ----------
        widget : QLineEdit
            reciever of the path text
        """
        if opensave:
            fname = QFileDialog.getSaveFileName(
                self, title, widget.text(), filter=filter_str)
        else:
            fname = QFileDialog.getOpenFileName(
                self, title, widget.text(), filter=filter_str)
        widget.setText(os.path.normpath(fname[0]))
        self.flag_edit()

    def get_data(self):
        """Update current_template into templates. Called by save."""
        if (hasattr(self.__class__, 'get_current_template')
            and callable(getattr(
                self.__class__, 'get_current_template'))):
            self.get_current_template()

    def add(self, label):
        """Add current_template or empty_template to templates."""
        '''
        if current is False:
            try:
                self.current_template = copy.deepcopy(
                    self.empty_template)
            except AttributeError:
                print('Missing empty template')
                print(self)
        '''
        self.current_template.label = label
        if self.grouped:
            if self.templates[self.current_modality][0].label == '':
                self.templates[
                    self.current_modality][0] = copy.deepcopy(
                        self.current_template)
            else:
                self.templates[self.current_modality].append(
                    copy.deepcopy(
                        self.current_template))
        else:
            if self.templates[0].label == '':
                self.templates[0] = copy.deepcopy(self.current_template)
            else:
                self.templates.append(copy.deepcopy(self.current_template))
        self.save()
        self.refresh_templist(selected_label=label)

    def duplicate(self, selected_id, new_label):
        """Duplicated template.

        Parameters
        ----------
        selected_id : int
            template number in list to duplicate
        new_label : str
            verified label of new template
        """
        if self.grouped:
            self.templates[self.current_modality].append(
                    copy.deepcopy(
                        self.templates[self.current_modality][selected_id]))
            self.templates[self.current_modality][-1].label = new_label
        else:
            self.templates.append(copy.deepcopy(self.templates[selected_id]))
            self.templates[-1].label = new_label
        self.save()
        self.refresh_templist(selected_label=new_label)

    def rename(self, newlabel):
        """Rename selected template."""
        tempno = self.current_labels.index(self.current_template.label)
        oldlabel = self.templates[self.current_modality][tempno].label
        self.current_template.label = newlabel
        if self.grouped:
            self.templates[self.current_modality][tempno].label = newlabel
        else:
            self.templates[tempno].label = newlabel

        save_more = False
        more = None
        more_fname = ''
        log = []
        if self.fname in ['paramset', 'quicktest_templates']:
            ok, path, auto_templates = cff.load_settings(
                fname='auto_templates')
            more_fname = 'auto_templates'
            mod = self.current_modality
            if path != '':
                if self.fname == 'paramset':
                    param_auto = cff.get_paramsets_used_in_auto_templates(
                        auto_templates)
                    log = [('Paramset label used in auto_templates. '
                            + 'Label updated.')]
                    param_labels, auto_labels = np.array(
                        param_auto[mod]).T.tolist()
                    if oldlabel in param_labels:
                        for i, pa in enumerate(param_labels):
                            if pa == oldlabel:
                                auto_templates[
                                    mod][i].paramset_label == newlabel
                                log.append(auto_labels[i])
                if self.fname == 'quicktest_templates':
                    qt_auto = cff.get_quicktest_used_in_auto_templates(
                        auto_templates)
                    log = [('QuickTest label used in auto_templates. '
                            + 'Label updated.')]
                    qt_labels, auto_labels = np.array(
                        qt_auto[mod]).T.tolist()
                    if oldlabel in qt_labels:
                        for i, qa in enumerate(qt_labels):
                            if qa == oldlabel:
                                auto_templates[
                                    mod][i].quicktemp_label == newlabel
                                log.append(auto_labels[i])
                if len(log) > 1:
                    save_more = True
                    more = auto_templates
                else:
                    log = ['Label not found used in auto_templates.']
        self.save(save_more=save_more, more=more,
                  more_fname=more_fname, log=log)
        self.refresh_templist(selected_label=newlabel)

    def save(self, save_more=False, more=None, more_fname='', log=[]):
        """Save template and other connected templates if needed.

        Parameters
        ----------
        save_more : bool, optional
            Connected templates to be saved exist. The default is False.
        more : template, optional
            Connected template to save. The default is None.
        more_fname : str, optional
            fname of connected template. The default is ''.
        log : list of str, optional
            Log from process of connected templates. The default is [].
        """
        proceed = cff.test_config_folder(self)
        if proceed:
            proceed, errmsg = cff.check_save_conflict(
                self.fname, self.lastload)
            if errmsg != '':
                proceed = uir.proceed_question(self, errmsg)
            if proceed:
                ok, path = cff.save_settings(
                    self.templates, fname=self.fname)
                if ok:
                    if save_more:
                        proceed, errmsg = cff.check_save_conflict(
                            more_fname, self.lastload)
                        if errmsg != '':
                            proceed = uir.proceed_question(self, errmsg)
                        if proceed:
                            ok, path = cff.save_settings(
                                more, fname=more_fname)
                            if ok:
                                msg = QMessageBox()
                                msg.setText('Related templates also updated. '
                                            'See details to view changes performed')
                                msg.setWindowTitle('Updated related templates')
                                msg.setIcon(QMessageBox.Information)
                                msg.setDetailedText('\n'.join(log))
                                msg.exec()
                    self.status_label.setText(
                        f'Changes saved to {path}')
                    self.flag_edit(False)
                    self.lastload = time()
                else:
                    QMessageBox.warning(
                        self, 'Failed saving', f'Failed saving to {path}')

        if 'auto' in self.fname and 'template' in self.fname:
            # Ensure refresh templist with foreground color for inactive templates.
            row = self.wModTemp.list_temps.currentRow()
            self.refresh_templist(selected_id=row)


class DummyStackWidgetForCopy(StackWidget):
    """...Copy this code as a template when starting on new stackwidget..."""

    def __init__(self, save_blocked=False, import_review_mode=False):
        header = ''
        subtxt = '''<br>'''
        # self.save_blocked = save_blocked # if mod_temp=True
        super().__init__(header, subtxt)
        # if list of templates
        super().__init__(header, subtxt,
                         typestr='template',
                         mod_temp=True, import_review_mode=False)
        # if grouped into modalities
        super().__init__(header, subtxt,
                         typestr='template',
                         mod_temp=True, grouped=True, import_review_mode=False)

        self.fname = '...fname... as key in CONFIG_FNAMES'
        # self.empty_template = object default temp

        # some GUI added to self.vLO (QVBoxLayout)
        #   if no templatelist else self.hLO (QHBoxLayout)

        self.vLO.addWidget(uir.HLine())
        self.vLO.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with self.current_template."""
        # if template-list, called by refresh_templist()
        # set values to GUI
        # if updated with saved data
        self.flag_edit(False)

    def get_current_template(self):
        # if template-list - used by get_data() in save()
        """Update current_template from GUI and save into self.templates."""
        if self.edited:
            # set self.current_template from GUI if not already updated by GUI
            pass


class ModTempSelector(QWidget):
    """Widget with modality selector, template selector and toolbar."""

    def __init__(self, parent, editable=True, import_review_mode=False):
        super().__init__()
        self.parent = parent
        self.setFixedWidth(400)

        self.vLO = QVBoxLayout()
        self.setLayout(self.vLO)
        hLOmodality = QHBoxLayout()
        self.vLO.addLayout(hLOmodality)
        self.lbl_modality = uir.LabelItalic('Modality')
        hLOmodality.addWidget(self.lbl_modality)
        self.cbox_modality = QComboBox()
        self.cbox_modality.addItems([*QUICKTEST_OPTIONS])
        self.cbox_modality.currentIndexChanged.connect(
            self.parent.update_modality)
        hLOmodality.addWidget(self.cbox_modality)
        hLOmodality.addStretch()
        self.vLO.addSpacing(10)
        self.vLO.addWidget(uir.LabelItalic(self.parent.typestr.title()+'s'))
        hLOlist = QHBoxLayout()
        self.vLO.addLayout(hLOlist)
        self.list_temps = QListWidget()
        self.list_temps.itemClicked.connect(
            self.parent.update_clicked_template)
        hLOlist.addWidget(self.list_temps)

        if import_review_mode:
            self.tb = QToolBar()
            self.tb.setOrientation(Qt.Vertical)
            hLOlist.addWidget(self.tb)
            self.actImport = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}ok.png'),
                'Mark ' + self.parent.typestr + ' for import', self)
            self.actImport.triggered.connect(self.mark_import)
            self.actIgnore = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}deleteRed.png'),
                'Mark ' + self.parent.typestr + ' to ignore', self)
            self.actIgnore.triggered.connect(
                lambda: self.mark_import(ignore=True))

            self.tb.addActions(
                [self.actImport, self.actIgnore])
        else:
            if editable:
                self.tb = QToolBar()
                self.tb.setOrientation(Qt.Vertical)
                hLOlist.addWidget(self.tb)
                self.actClear = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}clear.png'),
                    'Clear ' + self.parent.typestr + ' (reset to default)', self)
                self.actClear.triggered.connect(self.clear)
                self.actAdd = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                    'Add current values as new ' + self.parent.typestr, self)
                self.actAdd.triggered.connect(self.add)
                self.actSave = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
                    'Save current values to ' + self.parent.typestr, self)
                self.actSave.triggered.connect(self.save)
                self.actRename = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}rename.png'),
                    'Rename ' + self.parent.typestr, self)
                self.actRename.triggered.connect(self.rename)
                self.actDuplicate = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}duplicate.png'),
                    'Duplicate ' + self.parent.typestr, self)
                self.actDuplicate.triggered.connect(self.duplicate)
                actImport = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}import.png'),
                    'Import from .yaml', self)
                actImport.triggered.connect(self.import_from_yaml)
                self.actUp = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
                    'Move up', self)
                self.actUp.triggered.connect(self.move_up)
                self.actDown = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
                    'Move down', self)
                self.actDown.triggered.connect(self.move_down)
                self.actDel = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
                    'Delete ' + self.parent.typestr, self)
                self.actDel.triggered.connect(self.delete)

                if self.parent.save_blocked:
                    self.actClear.setEnabled(False)
                    self.actAdd.setEnabled(False)
                    self.actSave.setEnabled(False)
                    self.actDuplicate.setEnabled(False)
                    self.actRename.setEnabled(False)
                    actImport.setEnabled(False)
                    self.actUp.setEnabled(False)
                    self.actDown.setEnabled(False)
                    self.actDel.setEnabled(False)

                self.tb.addActions(
                    [self.actClear, self.actAdd, self.actSave, self.actDuplicate,
                     self.actRename, actImport, self.actUp,
                     self.actDown, self.actDel])

    def keyPressEvent(self, event):
        """Accept Delete key to delete templates."""
        if event.key() == Qt.Key_Delete:
            self.delete()
        else:
            super().keyPressEvent(event)

    def clear(self):
        """Clear template - set like empty_template."""
        try:
            lbl = self.parent.current_template.label
            self.parent.current_template = copy.deepcopy(
                self.parent.empty_template)
            self.parent.current_template.label = lbl
            self.parent.update_data()
        except AttributeError:
            print('Missing empty template')

    def add(self):
        """Add new template to list. Ask for new name and verify."""
        text, ok = QInputDialog.getText(
            self, 'New label',
            'Name the new ' + self.parent.typestr)
        text = valid_template_name(text)
        if ok and text != '':
            if text in self.parent.current_labels:
                QMessageBox.warning(
                    self, 'Label already in use',
                    'This label is already in use.')
            else:
                self.parent.add(text)

    def save(self, label=None):
        """Save button pressed or specific save on label."""
        if self.parent.current_template.label == '':
            self.add()
        else:
            if label is False or label is None:
                idx = self.list_temps.currentIndex().row()
            else:
                idx = self.parent.current_labels.index(label)
            self.parent.get_data()  # if get_current_template exist
            if self.parent.grouped:
                self.parent.templates[self.parent.current_modality][idx] = \
                    copy.deepcopy(self.parent.current_template)
            else:
                self.parent.templates[idx] = copy.deepcopy(
                    self.parent.current_template)
            self.parent.save()

    def rename(self):
        """Rename current template. Ask for new name and verify."""
        if self.parent.current_labels[0] == '':
            QMessageBox.warning(
                self, 'Empty list',
                'No template to rename.')
        else:
            sel = self.list_temps.currentItem()
            if sel is not None:
                currentText = sel.text()

                text, ok = QInputDialog.getText(
                    self, 'New label',
                    'Rename ' + self.parent.typestr,
                    text=currentText)
                text = valid_template_name(text)
                if ok and text != '' and currentText != text:
                    if text in self.parent.current_labels:
                        QMessageBox.warning(
                            self, 'Label already in use',
                            'This label is already in use.')
                    else:
                        self.parent.rename(text)

    def duplicate(self):
        """Duplicate template."""
        if self.parent.current_labels[0] == '':
            QMessageBox.warning(
                self, 'Empty list',
                'No template to duplicate.')
        else:
            proceed = True
            if self.parent.edited:
                res = uir.QuestionBox(
                    parent=self, title='Duplicate or add edited?',
                    msg='''Selected template has changed.
                    Add with current parameters or duplicate original?''',
                    yes_text='Add new with current parameter',
                    no_text='Duplicate original')
                if res.exec():
                    self.add()
                    proceed = False

            if proceed:  # duplicate original
                sel = self.list_temps.currentItem()
                currentText = sel.text()
                duplicate_id = self.parent.current_labels.index(currentText)

                text, ok = QInputDialog.getText(
                    self,
                    'New label',
                    'Name the new ' + self.parent.typestr)
                text = valid_template_name(text)
                if ok and text != '':
                    if text in self.parent.current_labels:
                        QMessageBox.warning(
                            self, 'Label already in use',
                            'This label is already in use.')
                    else:
                        self.parent.duplicate(duplicate_id, text)

    def move_up(self):
        """Move template up if possible."""
        row = self.list_temps.currentRow()
        if row > 0:
            if self.parent.grouped:
                popped_temp = self.parent.templates[
                    self.parent.current_modality].pop(row)
                self.parent.templates[
                    self.parent.current_modality].insert(row - 1, popped_temp)
            else:
                popped_temp = self.parent.templates.pop(row)
                self.parent.templates.insert(row - 1, popped_temp)
            self.parent.save()
            self.parent.refresh_templist(selected_id=row-1)

    def move_down(self):
        """Move template down if possible."""
        row = self.list_temps.currentRow()
        if row < len(self.parent.current_labels)-1:
            if self.parent.grouped:
                popped_temp = self.parent.templates[
                    self.parent.current_modality].pop(row)
                self.parent.templates[
                    self.parent.current_modality].insert(row + 1, popped_temp)
            else:
                popped_temp = self.parent.templates.pop(row)
                self.parent.templates.insert(row + 1, popped_temp)
            self.parent.save()
            self.parent.refresh_templist(selected_id=row+1)

    def import_from_yaml(self):
        """Import template(s) from yaml."""
        #TODO
        pass

    def delete(self, confirmed=False):
        """Delete template."""
        if self.parent.current_labels[0] == '':
            QMessageBox.warning(
                self, 'Empty list',
                'No template to delete.')
        else:
            if confirmed is False:
                res = QMessageBox.question(
                    self, 'Delete?', 'Delete selected template?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if res == QMessageBox.Yes:
                    confirmed = True
            if confirmed:
                row = self.list_temps.currentRow()
                if row < len(self.parent.current_labels):
                    if self.parent.grouped:
                        self.parent.templates[
                            self.parent.current_modality].pop(row)
                        if len(self.parent.templates[
                                self.parent.current_modality]) == 0:
                            self.parent.set_empty_template()
                    else:
                        self.parent.templates.pop(row)
                        if len(self.parent.templates) == 0:
                            self.parent.set_empty_template()

                    self.parent.save()
                    #TODO save_more - deleted something connected?
                    self.parent.refresh_templist(selected_id=row-1)

    def mark_import(self, ignore=False):
        """If import review mode: Mark template for import or ignore."""
        if not hasattr(self.parent, 'marked'):  # initiate
            if self.parent.grouped:
                empty = {}
                for key in QUICKTEST_OPTIONS:
                    empty[key] = []
            else:
                empty = []
            self.parent.marked_ignore = empty
            self.parent.marked = copy.deepcopy(empty)

        row = self.list_temps.currentRow()
        if self.parent.grouped:
            if ignore:
                if row not in self.parent.marked_ignore[self.parent.current_modality]:
                    self.parent.marked_ignore[self.parent.current_modality].append(row)
                if row in self.parent.marked[self.parent.current_modality]:
                    self.parent.marked[self.parent.current_modality].remove(row)
            else:
                if row not in self.parent.marked[self.parent.current_modality]:
                    self.parent.marked[self.parent.current_modality].append(row)
                if row in self.parent.marked_ignore[self.parent.current_modality]:
                    self.parent.marked_ignore[self.parent.current_modality].remove(row)
        else:
            if ignore:
                if row not in self.parent.marked_ignore:
                    self.parent.marked_ignore.append(row)
                if row in self.parent.marked:
                    self.parent.marked.remove(row)
            else:
                if row not in self.parent.marked:
                    self.parent.marked.append(row)
                if row in self.parent.marked_ignore:
                    self.parent.marked_ignore.remove(row)

        self.parent.refresh_templist()


class UserSettingsWidget(StackWidget):
    """Widget holding user settings."""

    def __init__(self, save_blocked=False):
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
        super().__init__(header, subtxt)

        self.config_folder = QLineEdit()
        self.chk_dark_mode = QCheckBox()
        self.font_size = QSpinBox()

        self.config_folder.setMinimumWidth(500)
        hLO_config_folder = QHBoxLayout()
        hLO_config_folder.addWidget(QLabel('Path to config folder:'))
        hLO_config_folder.addWidget(self.config_folder)
        tb = uir.ToolBarBrowse('Browse to find or initiate config folder')
        tb.actBrowse.triggered.connect(
            lambda: self.locate_folder(self.config_folder))
        hLO_config_folder.addWidget(tb)
        self.vLO.addLayout(hLO_config_folder)
        self.vLO.addSpacing(50)

        hLO_mid = QHBoxLayout()
        vLO_1 = QVBoxLayout()
        hLO_mid.addLayout(vLO_1)
        self.vLO.addLayout(hLO_mid)

        gb_gui = QGroupBox('GUI settings')
        gb_gui.setFont(uir.FontItalic())
        vlo_gui = QVBoxLayout()
        self.font_size.setRange(5, 15)
        self.font_size.valueChanged.connect(self.flag_edit)
        hLO_font_size = QHBoxLayout()
        hLO_font_size.addWidget(QLabel('Set font size for GUI:'))
        hLO_font_size.addWidget(self.font_size)
        hLO_font_size.addWidget(QLabel('(Restart to update GUI)'))
        hLO_font_size.addStretch()
        vlo_gui.addLayout(hLO_font_size)
        hLO_dark_mode = QHBoxLayout()
        self.chk_dark_mode.clicked.connect(
            lambda: self.flag_edit(True))
        hLO_dark_mode.addWidget(QLabel('Dark mode'))
        hLO_dark_mode.addWidget(self.chk_dark_mode)
        hLO_dark_mode.addWidget(QLabel('(restart to update)'))
        hLO_dark_mode.addStretch()
        vlo_gui.addLayout(hLO_dark_mode)
        gb_gui.setLayout(vlo_gui)
        vLO_1.addWidget(gb_gui)
        vLO_1.addSpacing(50)

        hLO_mid.addStretch()

        btn_save_user_prefs = QPushButton('Save user preferences')
        btn_save_user_prefs.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}save.png'))
        btn_save_user_prefs.clicked.connect(self.save_user)
        if save_blocked:
            btn_save_user_prefs.setEnabled(False)
        self.vLO.addWidget(btn_save_user_prefs)

        self.vLO.addStretch()

        self.vLO.addWidget(uir.HLine())
        self.vLO.addWidget(self.status_label)

    def update_from_yaml(self):
        """Load settings from yaml and fill form."""
        ok, path, self.user_prefs = cff.load_user_prefs()
        self.lbl_yaml_path = path
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

        ok, path = cff.save_user_prefs(self.user_prefs, parentwidget=self)
        if ok:
            self.status_label.setText(f'Changes saved to {path}')
            self.flag_edit(False)
            cff.add_user_to_active_users()
        else:
            QMessageBox.Warning(self, 'Warning',
                                f'Failed to save changes to {path}')


class SharedSettingsWidget(StackWidget):
    """Widget for shared settings."""

    def __init__(self, save_blocked=False, width1=200, width2=800):
        header = 'Config folder - shared settings'
        subtxt = '''Each of the sub-pages will display different settings
         saved in the config folder (specified in user settings).<br>
        Templates and settings will be saved as .yaml files. <br>
        Several users may link to the same config folder and
         share these settings.'''
        super().__init__(header, subtxt)
        self.width1 = width1
        self.width2 = width2

        self.lbl_config_folder = QLabel('-- not defined --')
        self.list_files = QListWidget()

        hLOcf = QHBoxLayout()
        self.vLO.addLayout(hLOcf)
        hLOcf.addWidget(QLabel('Config folder: '))
        hLOcf.addWidget(self.lbl_config_folder)
        hLOcf.addStretch()
        btnLocateConfig = QPushButton('Locate new or exisiting config folder')
        btnLocateConfig.clicked.connect(self.locate_config)
        btnImportIDLConfig = QPushButton(
            'Import config from IDL version of ImageQC')
        btnImportIDLConfig.clicked.connect(self.import_idl_config)
        self.vLO.addWidget(btnLocateConfig)
        self.vLO.addWidget(btnImportIDLConfig)

        if save_blocked:
            btnLocateConfig.setEnabled(False)
            btnImportIDLConfig.setEnabled(False)

        self.vLO.addWidget(self.list_files)

    def update_from_yaml(self):
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

            ok, path, last_modified = cff.load_settings(fname='last_modified')
            if ok:
                for cfn in [*CONFIG_FNAMES]:
                    if cff.get_config_filename(cfn) != '':  # in case deleted
                        try:
                            res = getattr(last_modified, cfn)
                            if len(res) > 0:
                                self.list_files.addItem(cfn + ':')
                                string = ' '.join(
                                    ['    last edited by',
                                     res[0], time_diff_string(res[1]),
                                     '(', ctime(res[1]), ')'])
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
        ok, path, user_prefs = cff.load_user_prefs()
        user_prefs.config_folder = os.path.normpath(folder)
        ok, path = cff.save_user_prefs(user_prefs, parentwidget=self)
        self.update_from_yaml()

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
                ok, path, self.tag_infos = cff.load_settings(fname='tag_infos')
                config_idl = ConfigIdl2Py(fname[0], self.tag_infos)
                if len(config_idl.errmsg) > 0:
                    QMessageBox.warning(
                        self, 'Warnings', '\n'.join(config_idl.errmsg))
                import_main = ImportMain(
                    tag_infos=config_idl.tag_infos_new,
                    rename_patterns=config_idl.rename_patterns,
                    paramsets=config_idl.paramsets,
                    quicktest_templates=config_idl.quicktest_templates,
                    auto_common=config_idl.auto_common,
                    auto_templates=config_idl.auto_templates,
                    auto_vendor_templates=config_idl.auto_vendor_templates
                    )
                if config_idl.paramsets != {}:
                    import_main.current_paramset = config_idl.paramsets['CT'][0]
                dlg = SettingsDialog(
                    import_main, initial_view='Config folder',
                    width1=self.width1, width2=self.width2,
                    import_review_mode=True)
                res = dlg.exec()
                if res:
                    import_main = dlg.get_marked()
                    cff.import_settings(import_main)
                self.update_from_yaml()


class SharedSettingsImportWidget(StackWidget):
    """Widget to replace SharedSettingsWidget when import_review_mode."""

    def __init__(self, settings_dialog):
        header = 'Settings for import'
        subtxt = '''Mark templates for import or mark templates to ignore.<br>
        Then get back to this window to import according to your selections.'''
        super().__init__(header, subtxt)
        self.dlg = settings_dialog
        btnAll = QPushButton('Import all templates')
        btnAllBut = QPushButton('Import all except for those marked to ignore')
        btnMarked = QPushButton('Import only marked templates')
        self.vLO.addWidget(btnAll)
        self.vLO.addWidget(btnMarked)
        self.vLO.addWidget(btnAllBut)
        self.vLO.addStretch()

        btnAll.clicked.connect(
            lambda: self.dlg.set_marked(True, import_all=True))
        btnMarked.clicked.connect(self.dlg.set_marked)
        btnAllBut.clicked.connect(
            lambda: self.dlg.set_marked(False))

    def update_from_import_main(self):
        breakpoint()  #TODO delete this method?
        pass


class DicomTagDialog(uir.ImageQCDialog):
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
        self.tagString = QLineEdit()
        self.tagString.returnPressed.connect(self.correct_tag_input)
        self.list_sequences = QListWidget()
        self.cbox_value_id = QComboBox()
        self.cbox_value_id.currentIndexChanged.connect(self.update_value)
        self.lbl_tag_content = QLabel()
        self.btngr_modality = QButtonGroup(exclusive=False)
        self.txt_unit = QLineEdit(tag_input.unit)

        self.sample_attribute_names = []
        self.sample_sequences = ['']
        self.sample_tags = []
        self.pydict = None
        self.data_element = None

        vLO = QVBoxLayout()
        self.setLayout(vLO)

        vLO.addWidget(QLabel(
            'Read DICOM file to fill the list of tags'))
        hLO_file = QHBoxLayout()
        vLO.addLayout(hLO_file)
        self.sample_filepath.setMinimumWidth(500)
        hLO_file.addWidget(self.sample_filepath)
        tb = uir.ToolBarBrowse('Browse to find sample file')
        tb.actBrowse.triggered.connect(self.locate_file)
        actDCMdump = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}tags.png'),
            "View DICOM dump", self)
        actDCMdump.triggered.connect(self.dump_dicom)
        tb.addAction(actDCMdump)
        hLO_file.addWidget(tb)

        hLO_taglist = QHBoxLayout()
        vLO.addLayout(hLO_taglist)
        hLO_taglist.addWidget(
            QLabel('Tags/attributes from file: '))
        hLO_taglist.addWidget(self.list_tags)
        self.list_tags.setMinimumWidth(300)
        self.actLevelUp = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
            "Get back to level above current sequence", self)
        self.actLevelUp.triggered.connect(self.level_up)
        self.actLevelUp.setEnabled(False)
        tb_sequence = QToolBar()
        tb_sequence.addAction(self.actLevelUp)
        hLO_taglist.addWidget(tb_sequence)
        vLO.addWidget(uir.LabelItalic(
            'Click tags named with Sequence to list the elements '
            'within the sequence.'))
        vLO.addWidget(uir.HLine())

        vLO.addSpacing(20)
        hLO = QHBoxLayout()
        vLO.addLayout(hLO)
        fLO = QFormLayout()
        hLO.addLayout(fLO)
        fLO.addRow(QLabel('Attribute name: '), self.txt_attribute_name)
        fLO.addRow(QLabel('Tag: '), self.tagString)
        self.tagString.setPlaceholderText('0000,0000')
        self.tagString.setFixedWidth(140)
        self.tagString.setText(
            f'{tag_input.tag[0][2:]:0>4},{tag_input.tag[1][2:]:0>4}')
        fLO.addRow(QLabel('Value id if multivalue*:'), self.cbox_value_id)
        if tag_input.value_id != -1:
            if tag_input.value_id == -2:
                self.cbox_value_id.addItem('per frame')
            else:
                self.cbox_value_id.addItem(f'{tag_input.value_id}')
        fLO.addRow(QLabel('Tag in sequence(s): '), self.list_sequences)
        self.list_sequences.setMaximumHeight(200)
        fLO.addRow(QLabel('Unit: '), self.txt_unit)
        if tag_input.sequence[0] != '':
            for txt in tag_input.sequence:
                self.list_sequences.addItem(txt)
        fLO.addRow(QLabel('Content from file: '), self.lbl_tag_content)
        vLO.addWidget(QLabel('* open sample file to fill options'))

        gb_modality = QGroupBox('Specified for modality...')
        gb_modality.setFont(uir.FontItalic())
        lo_mod = QVBoxLayout()
        listmod = [*QUICKTEST_OPTIONS]
        for idx, val in enumerate(listmod):
            cb = QCheckBox(val)
            self.btngr_modality.addButton(cb, idx)
            lo_mod.addWidget(cb)
            if val in tag_input.limited2mod:
                self.btngr_modality.button(idx).setChecked(True)
        gb_modality.setLayout(lo_mod)
        hLO.addWidget(gb_modality)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.verify_accept)
        buttons.rejected.connect(self.reject)
        vLO.addWidget(buttons)

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
            for a in range(len(self.sample_attribute_names)):
                if 'Sequence' in self.sample_attribute_names[a]:
                    self.list_tags.item(a).setBackground(brush)
        self.list_tags.blockSignals(False)

    def level_up(self):
        """Go back to above sequence."""
        if len(self.sample_sequences) == 1:
            self.sample_sequences = ['']
        else:
            self.sample_sequences.pop()
        if self.sample_sequences[0] == '':
            self.actLevelUp.setEnabled(False)
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
        rowno = sel[0].row()
        cur_text = self.sample_attribute_names[rowno]

        if 'Sequence' in cur_text:
            if self.sample_sequences[0] == '':
                self.sample_sequences[0] = cur_text
            else:
                self.sample_sequences.append(cur_text)
            self.set_sequences()
            self.get_all_tags_in_file()
            self.actLevelUp.setEnabled(True)
            self.cbox_value_id.clear()
        else:
            idx = self.sample_attribute_names.index(cur_text)
            self.txt_attribute_name.setText(cur_text)
            tag = self.sample_tags[idx]
            self.tagString.setText(f'{tag.group:04x},{tag.element:04x}')
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
                self, 'Unexpected tag format',
                'Tag format expected as XXXX,XXXX.')
        else:
            self.tagString.setText(self.str_to_tag(txt))

    def get_tag_info(self):
        """Fill TagInfo with current values and return.

        Returns
        -------
        new_tag_info : TagInfo
        """
        tag_str = self.tagString.text()
        tag_group = hex(int('0x' + tag_str[0:4], 16))
        tag_elem = hex(int('0x' + tag_str[5:9], 16))

        new_tag_info = cfc.TagInfo(
            sort_index=self.tag_input.sort_index,
            attribute_name=self.txt_attribute_name.text(),
            tag=[tag_group, tag_elem],
            unit=self.txt_unit.text()
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
            elif cur_text == 'all' or cur_text == '':
                new_tag_info.value_id = -1
            else:
                try:
                    new_tag_info.value_id = int(cur_text)
                except ValueError:
                    new_tag_info.value_id = -1

        checkedStrings = []
        for idx in range(len([*QUICKTEST_OPTIONS])):
            if self.btngr_modality.button(idx).isChecked():
                checkedStrings.append(
                    self.btngr_modality.button(idx).text())
        if (len(checkedStrings) == 0
                or len(checkedStrings) == len([*QUICKTEST_OPTIONS])):
            checkedStrings = ['']
        new_tag_info.limited2mod = checkedStrings

        return new_tag_info

    def dump_dicom(self):
        """Dump dicom elements for file to text."""
        proceed = True
        if self.sample_filepath.text() == '':
            QMessageBox.information(self, 'Missing input',
                                    'No file selected.')
            proceed = False
        if proceed:
            dcm.dump_dicom(self, filename=self.sample_filepath.text())


class DicomTagsWidget(StackWidget):
    """Widget for Dicom Tags."""

    def __init__(self, save_blocked=False, import_review_mode=False):
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
        the next variant will be searched.'''
        super().__init__(header, subtxt, import_review_mode=import_review_mode)
        self.fname = 'tag_infos'

        self.indexes = []
        self.edited_names = []

        self.table_tags = QTreeWidget()
        self.table_tags.setColumnCount(4)
        self.table_tags.setHeaderLabels(
            ['Tag name', 'Tag number', 'Value id', 'unit',
             'Specific modalities', 'in sequence(s)', 'Sort index'])
        self.table_tags.setColumnWidth(0, 400)
        self.table_tags.setColumnWidth(1, 150)
        self.table_tags.setColumnWidth(2, 120)
        self.table_tags.setColumnWidth(3, 150)
        self.table_tags.setColumnWidth(4, 200)
        self.table_tags.setColumnWidth(5, 400)
        self.table_tags.setColumnWidth(6, 120)
        self.table_tags.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContents)
        self.table_tags.setSortingEnabled(True)
        self.table_tags.sortByColumn(6, Qt.AscendingOrder)
        self.table_tags.setRootIsDecorated(False)
        self.table_tags.header().sectionClicked.connect(self.sort_header)

        hLO_modality = QHBoxLayout()
        gb_modality = QGroupBox('Tags specified for modality...')
        gb_modality.setFont(uir.FontItalic())
        self.btngr_modality_filter = QButtonGroup(exclusive=False)

        chk_all_modalities = QCheckBox()
        chk_all_modalities.clicked.connect(self.select_all_modalities)
        chk_all_modalities.setToolTip('Select (or deselect) all')
        chk_all_modalities.setFixedWidth(50)

        lo = QHBoxLayout()
        listmod = ['General (not specified)']
        listmod.extend([*QUICKTEST_OPTIONS])
        for idx, val in enumerate(listmod):
            cb = QCheckBox(val)
            self.btngr_modality_filter.addButton(cb, idx)
            lo.addWidget(cb)
            cb.clicked.connect(self.mode_changed)
            self.btngr_modality_filter.button(idx).setChecked(True)
        gb_modality.setLayout(lo)
        gb_modality.setFixedWidth(1200)

        hLO_modality.addWidget(chk_all_modalities)
        hLO_modality.addWidget(gb_modality)
        hLO_modality.addStretch()
        self.vLO.addLayout(hLO_modality)

        hLO_table = QHBoxLayout()
        self.vLO.addLayout(hLO_table)
        hLO_table.addWidget(self.table_tags)
        tb = QToolBar()
        tb.setOrientation(Qt.Vertical)
        hLO_table.addWidget(tb)
        hLO_table.addStretch()

        if import_review_mode:
            actImport = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}ok.png'),
                'Mark tag for import', self)
            actImport.triggered.connect(self.mark_import)
            actIgnore = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}deleteRed.png'),
                'Mark tag to ignore', self)
            actIgnore.triggered.connect(
                lambda: self.mark_import(ignore=True))

            tb.addActions(
                [actImport, actIgnore])

        else:
            actAdd = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                'Add new tag to list', self)
            actAdd.triggered.connect(self.add_tag)
            actDuplicate = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}copy.png'),
                'Duplicate tag to create variant', self)
            actDuplicate.triggered.connect(self.duplicate_tag)
            actEdit = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
                'Edit details for the selected tag', self)
            actEdit.triggered.connect(self.edit_tag)
            actUp = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
                'Move up', self)
            actUp.triggered.connect(self.move_up)
            actDown = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
                'Move down', self)
            actDown.triggered.connect(self.move_down)
            actDel = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
                'Delete tag', self)
            actDel.triggered.connect(self.delete)
            actSave = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
                'Save changes', self)
            actSave.triggered.connect(self.save)

            tb.addActions([
                actAdd, actDuplicate, actEdit, actUp, actDown, actDel, actSave])

            if save_blocked:
                actAdd.setEnabled(False)
                actEdit.setEnabled(False)
                actUp.setEnabled(False)
                actDown.setEnabled(False)
                actDel.setEnabled(False)
                actSave.setEnabled(False)

        self.vLO.addWidget(uir.HLine())
        self.vLO.addWidget(self.status_label)

    def sort_header(self):
        """When clicked to sort column."""
        self.flag_edit()
        self.update_indexes()

    def update_indexes(self):
        """Update self.indexes according to displayed table."""
        self.indexes = []
        for row in range(self.table_tags.topLevelItemCount()):
            item = self.table_tags.topLevelItem(row)
            self.indexes.append(int(item.text(6)))

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
            set_selected_row = sel[0].row()

        if reset_sort_index:
            cff.taginfos_reset_sort_index(self.templates)
            self.flag_edit()

        self.table_tags.clear()
        checkedStrings = []
        for idx in range(len([*QUICKTEST_OPTIONS])+1):
            if self.btngr_modality_filter.button(idx).isChecked():
                checkedStrings.append(
                    self.btngr_modality_filter.button(idx).text())
                if idx == 0:
                    checkedStrings[-1] = ''

        for tempid in range(len(self.templates)):
            # test if included in modality selection
            include = bool(
                set(checkedStrings).intersection(
                    self.templates[tempid].limited2mod))
            if include:
                row_strings = [
                    self.templates[tempid].attribute_name,
                    '-',
                    '-',
                    self.templates[tempid].unit,
                    ', '.join(self.templates[tempid].limited2mod),
                    ', '.join(self.templates[tempid].sequence),
                    f'{self.templates[tempid].sort_index:03}'
                    ]
                if self.templates[tempid].tag[1] == '':
                    row_strings[1] = self.templates[tempid].tag[0]
                else:
                    row_strings[1] = (
                            f'{self.templates[tempid].tag[0][2:]:0>4},'
                            f'{self.templates[tempid].tag[1][2:]:0>4}')
                if self.templates[tempid].value_id != -1:
                    if self.templates[tempid].value_id == -2:
                        row_strings[2] = 'per frame'
                    else:
                        row_strings[2] = f'{self.templates[tempid].value_id}'
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
        selection_all = False if any(checked_ids) else True

        for idx in range(len([*QUICKTEST_OPTIONS])+1):
            self.btngr_modality_filter.button(idx).blockSignals(True)
            self.btngr_modality_filter.button(idx).setChecked(selection_all)
            self.btngr_modality_filter.button(idx).blockSignals(False)

        self.update_data()

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
                idx_same_name = []
                for idx, temp in enumerate(self.templates):
                    if temp.attribute_name == old_temp.attribute_name:
                        idx_same_name.append(idx)
                if len(idx_same_name) > 1:
                    res = uir.QuestionBox(
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
            dlg.exec()
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
            res = uir.QuestionBox(
                parent=self, title='Delete?',
                msg=f'Delete tag {self.templates[idx].attribute_name}')
            if res.exec():
                self.templates.pop(idx)
                self.flag_edit()
                self.update_data(set_selected_row=selrow, reset_sort_index=True)

    def save(self):
        """Save current settings."""
        proceed = cff.test_config_folder(self)
        if proceed:
            proceed, errmsg = cff.check_save_conflict(
                self.fname, self.lastload)
            if errmsg != '':
                proceed = uir.proceed_question(self, errmsg)
            if proceed:
                # check if order of sort_index is ascending, if not ask to
                # save order as display
                sorted_indexes = sorted(self.indexes)
                if self.indexes != sorted_indexes:
                    res = uir.QuestionBox(
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
                    status, log = cff.correct_attribute_names(
                        old_new_names=self.edited_names)

                ok = True
                if status:
                    ok, path = cff.save_settings(
                        self.templates, fname=self.fname)
                    if ok:
                        self.flag_edit(False)
                        self.edited_names = []
                        self.update_data(keep_selection=True)
                    else:
                        log.append('Failed saving to {path}')
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
                msg = QMessageBox()
                msg.setText(txt)
                msg.setWindowTitle(tit)
                msg.setIcon(ico)
                if len(log) > 0:
                    msg.setDetailedText('\n'.join(log))
                msg.exec()

    def mark_import(self, ignore=False):
        """If import review mode: Mark tag for import or ignore."""
        if not hasattr(self, 'marked'):  # initiate
            self.marked_ignore = []
            self.marked = []

        sel = self.table_tags.selectedIndexes()
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

    def __init__(self, save_blocked=False, import_review_mode=False):
        header = 'Special tag patterns'
        subtxt = '''These tag patterns each have a specific function
        and can not be renamed or deleted, just edited.'''
        self.save_blocked = save_blocked
        super().__init__(header, subtxt,
                         typestr='tag pattern',
                         mod_temp=True, grouped=True,
                         import_review_mode=import_review_mode)

        self.fname = 'tag_patterns_special'
        self.empty_template = cfc.TagPatternFormat()

        self.wTagPattern = uir.TagPatternWidget(self, typestr='format')
        if import_review_mode is False:
            self.wModTemp.tb.removeAction(self.wModTemp.actAdd)
            self.wModTemp.tb.removeAction(self.wModTemp.actRename)
            self.wModTemp.tb.removeAction(self.wModTemp.actDuplicate)
            self.wModTemp.tb.removeAction(self.wModTemp.actUp)
            self.wModTemp.tb.removeAction(self.wModTemp.actDown)
            self.wModTemp.tb.removeAction(self.wModTemp.actDel)
        else:
            self.wTagPattern.setEnabled(False)
        self.wModTemp.list_temps.setFixedHeight(200)
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
        self.wModTemp.vLO.addWidget(QLabel(info_special))
        self.wModTemp.vLO.addStretch()
        self.hLO.addWidget(self.wTagPattern)

        self.vLO.addWidget(uir.HLine())
        self.vLO.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with the selected template."""
        self.wTagPattern.wPattern.update_data()
        self.flag_edit(False)


class TagPatternFormatWidget(StackWidget):
    """Setup for creating tag patterns for formatting."""

    def __init__(self, save_blocked=False, import_review_mode=False):
        header = 'Tag patterns for formatting'
        subtxt = '''Tag patterns can be used for<br>
        - renaming DICOM files based on the DICOM tags<br>
        - exporting specific DICOM tags as tabular data'''
        self.save_blocked = save_blocked
        super().__init__(header, subtxt,
                         typestr='template',
                         mod_temp=True, grouped=True,
                         import_review_mode=import_review_mode)

        self.fname = 'tag_patterns_format'
        self.empty_template = cfc.TagPatternFormat()

        self.wTagPattern = uir.TagPatternWidget(self, typestr='format')
        if import_review_mode:
            self.wTagPattern.setEnabled(False)
        self.hLO.addWidget(self.wTagPattern)

        self.vLO.addWidget(uir.HLine())
        self.vLO.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with the selected template."""
        self.wTagPattern.wPattern.update_data()
        self.flag_edit(False)


class TagPatternSortWidget(StackWidget):
    """Setup for creating tag patterns for sorting."""

    def __init__(self, save_blocked=False, import_review_mode=False):
        header = 'Tag patterns for sorting'
        subtxt = ('These tag patterns can be used for '
                  'sorting DICOM files based on the tags.')
        self.save_blocked = save_blocked
        super().__init__(header, subtxt,
                         typestr='template',
                         mod_temp=True, grouped=True,
                         import_review_mode=import_review_mode)

        self.fname = 'tag_patterns_sort'
        self.empty_template = cfc.TagPatternSort()

        self.wTagPattern = uir.TagPatternWidget(self, typestr='sort')
        if import_review_mode:
            self.wTagPattern.setEnabled(False)
        self.hLO.addWidget(self.wTagPattern)

        self.vLO.addWidget(uir.HLine())
        self.vLO.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with the selected template."""
        self.wTagPattern.wPattern.update_data()
        self.flag_edit(False)


class RenamePatternWidget(StackWidget):
    """Setup for creating tag patterns for formatting."""

    def __init__(
            self, initial_modality='CT',
            header=None, subtxt=None, editable=True, save_blocked=False,
            import_review_mode=False):
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
        self.save_blocked = save_blocked
        super().__init__(header, subtxt,
                         typestr='template',
                         mod_temp=True, grouped=True, editable=editable,
                         import_review_mode=import_review_mode)

        self.fname = 'rename_patterns'
        self.empty_template = cfc.RenamePattern()
        self.current_modality = initial_modality

        self.wTagPattern = uir.TagPatternWidget(
            self, typestr='format', rename_pattern=True, editable=editable)
        if import_review_mode:
            self.wTagPattern.setEnabled(False)
        self.hLO.addWidget(self.wTagPattern)

        if editable:
            self.vLO.addWidget(uir.HLine())
            self.vLO.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with the selected template."""
        self.wTagPattern.wPattern.update_data()
        self.wTagPattern.wPattern2.update_data()
        self.flag_edit(False)


class ParametersWidget(QWidget):
    """Widget for holding the parameters table for ParamSetsWidget."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        vLOtables = QVBoxLayout()
        vLOtables.addWidget(uir.LabelItalic(
            '''Differences between selected and active parameter set will be
            highlighted in red font.'''))
        self.table_params = QTreeWidget()
        self.table_params.setHeaderLabels(
            ['Parameter', 'Value in selected set',
             'Active value in main window'])
        vLOtables.addWidget(self.table_params)
        self.setLayout(vLOtables)


class ParametersOutputWidget(QWidget):
    """Widget for holding the output parameters table for ParamSetsWidget."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.list_group_by = QListWidget()
        self.wOutputTable = uir.QuickTestOutputTreeView(self.parent)

        hLO = QHBoxLayout()
        vLO_general = QVBoxLayout()
        self.setLayout(hLO)
        hLO.addLayout(vLO_general)
        vLO_general.addWidget(uir.LabelItalic('Settings for text output:'))
        self.tb_copy = uir.ToolBarTableExport(self.parent, flag_edit=True)
        self.tb_copy.setOrientation(Qt.Horizontal)
        vLO_general.addWidget(self.tb_copy)
        vLO_general.addSpacing(10)
        vLO_general.addWidget(
            uir.LabelItalic('If calculate pr group, group by'))
        hLO_group = QHBoxLayout()
        vLO_general.addLayout(hLO_group)
        hLO_group.addWidget(self.list_group_by)
        self.list_group_by.setFixedWidth(200)
        self.tb_edit_group_by = uir.ToolBarEdit(tooltip='Edit list')
        vLO_general.addWidget(self.tb_edit_group_by)
        self.tb_edit_group_by.actEdit.triggered.connect(self.edit_group_by)

        vLO_table = QVBoxLayout()
        hLO.addLayout(vLO_table)
        vLO_table.addWidget(uir.LabelItalic(
            '''Add or edit settings to define output settings when QuickTest
            is used.<br>
            Default if no settings are defined for a test, all values from
            the results table will be printed.'''))
        hLO_table = QHBoxLayout()
        vLO_table.addLayout(hLO_table)
        hLO_table.addWidget(self.wOutputTable)
        self.wOutputTable.setFixedWidth(800)

        self.tb = QToolBar()
        self.tb.setOrientation(Qt.Vertical)
        hLO_table.addWidget(self.tb)
        actAdd = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add more settings', self)
        actAdd.triggered.connect(self.wOutputTable.insert_row)
        actEdit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit selected row', self)
        actEdit.triggered.connect(self.wOutputTable.edit_row)
        actDel = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete selected row', self)
        actDel.triggered.connect(self.wOutputTable.delete_row)
        self.tb.addActions([actAdd, actEdit, actDel])

    def edit_group_by(self):
        #TODO
        pass


class ParamSetsWidget(StackWidget):
    """Widget holding paramsets settings."""

    def __init__(self, save_blocked=False,
                 main_current_paramset=None,
                 main_current_modality='CT',
                 import_review_mode=False):
        header = 'Parameter sets - manager'
        subtxt = '''The parameter sets contain both parameters for
        test settings and for output settings when results are copied
        to clipboard or to file (when automation is used).<br>
        To edit the test settings use the main window to set and save
        the parameters.'''
        self.save_blocked = save_blocked
        self.main_current_modality = main_current_modality
        self.main_current_paramset = main_current_paramset
        super().__init__(header, subtxt,
                         typestr='parameterset',
                         mod_temp=True, grouped=True,
                         import_review_mode=import_review_mode)
        self.fname = 'paramsets'
        self.empty_template = self.get_empty_paramset()

        self.tabs = QTabWidget()
        self.wParams = ParametersWidget(self)
        self.wOutput = ParametersOutputWidget(self)
        if import_review_mode:
            self.wOutput.setEnabled(False)

        self.tabs.addTab(self.wParams, 'Parameters')
        self.tabs.addTab(self.wOutput, 'Output parameters')

        self.hLO.addWidget(self.tabs)
        self.vLO.addWidget(uir.HLine())
        self.vLO.addWidget(self.status_label)

    def get_empty_paramset(self):
        """Get empty (default) paramset of current modality.

        Returns
        -------
        paramset
        """
        mod = self.current_modality
        paramset = None
        if mod == 'CT':
            paramset = cfc.ParamSetCT()
        elif mod == 'Xray':
            paramset = cfc.ParamSetXray()
        elif mod == 'NM':
            paramset = cfc.ParamSetNM()
        elif mod == 'SPECT':
            paramset = cfc.ParamSetSPECT()
        elif mod == 'PET':
            paramset = cfc.ParamSetPET()
        elif mod == 'MR':
            paramset = cfc.ParamSetMR()

        return paramset

    def update_data(self):
        """Update GUI with the selected paramset."""
        self.wOutput.wOutputTable.update_data()
        self.wOutput.tb_copy.parameters_output = self.current_template.output
        self.wOutput.tb_copy.update_checked()
        self.wOutput.list_group_by.clear()
        self.wOutput.list_group_by.addItems(
            self.current_template.output.group_by)

        self.wParams.table_params.clear()
        boldFont = QFont()
        boldFont.setWeight(600)

        main_str = ''
        get_main = False
        if self.main_current_paramset is not None:
            if self.main_current_modality == (
                    self.current_modality):
                get_main = True
        for paramname, paramval in asdict(self.current_template).items():
            if paramname != 'output':
                if isinstance(paramval, dict):
                    item = QTreeWidgetItem(
                        self.wParams.table_params, [paramname])
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
                                child.setFont(2, boldFont)
                            item.addChild(child)
                    item.setExpanded(True)
                else:
                    if get_main:
                        main_val = getattr(
                            self.main_current_paramset, paramname)
                        main_str = str(main_val)
                    item = QTreeWidgetItem(
                        self.wParams.table_params,
                        [paramname, str(paramval), main_str])
                    if str(paramval) != main_str:
                        item.setForeground(2, QBrush(QColor(203, 91, 76)))
                        item.setFont(2, boldFont)
        for i in range(self.wParams.table_params.columnCount()):
            self.wParams.table_params.resizeColumnToContents(i)


class QuickTestTemplatesWidget(StackWidget):
    """Widget holding QuickTest pattern settings."""

    def __init__(self, save_blocked=False, import_review_mode=False):
        header = 'QuickTest templates'
        subtxt = '''Which tests to perform on which images.<br>
        Optionally also define the label for the images and/or groups.<br>
        These labels will be used in headers when generating QuickTest output as defined
        in the parameter set either during automation or copy QuickTest results to
        clipboard.<br>
        If no label is set img0, img1.../group0, group1... will be used.<br>
        A group is by default images with the same seriesUID, but this can be edited in
        parameter set - output.'''
        self.save_blocked = save_blocked
        super().__init__(header, subtxt,
                         typestr='template',
                         mod_temp=True, grouped=True,
                         import_review_mode=import_review_mode)
        self.fname = 'quicktest_templates'
        self.empty_template = cfc.QuickTestTemplate()
        self.finished_init = False
        
        vLO = QVBoxLayout()
        self.hLO.addLayout(vLO)

        self.wTestTable = uir.QuickTestTreeView(self)
        vLO.addWidget(self.wTestTable)

        hLO_nimgs = QHBoxLayout()
        vLO.addLayout(hLO_nimgs)
        hLO_nimgs.addWidget(QLabel('Minimum number of images expected: '))
        self.lbl_nimgs = QLabel('')
        hLO_nimgs.addWidget(self.lbl_nimgs)

        self.tb = QToolBar()
        self.tb.setOrientation(Qt.Vertical)
        self.hLO.addWidget(self.tb)
        actAdd = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add new row after selected row', self)
        actAdd.triggered.connect(self.wTestTable.insert_empty_row)
        actDel = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete selected row', self)
        actDel.triggered.connect(self.wTestTable.delete_row)
        self.tb.addActions([actAdd, actDel])

        if import_review_mode:
            self.wTestTable.setEnabled(False)
            self.tb.setEnabled(False)

        self.vLO.addWidget(uir.HLine())
        self.vLO.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with selected template."""
        self.wTestTable.update_data()
        self.lbl_nimgs.setText(f'{len(self.current_template.tests)}')

    def get_current_template(self):
        """Get self.current_template."""
        self.current_template = self.wTestTable.get_data()


class AutoInfoWidget(StackWidget):
    """Widget holding information about automation."""

    def __init__(self):
        header = 'Automation'
        subtxt = (
            '''The main task for automation in imageQC is analysing constancy
            tests i.e. repeated tests with standardized output to follow trends.
            <br><br>
            <b>Import settings</b><br>
            ImageQC can be set to automatically sort (and rename) incoming images to
            the image pool into their respective folders defined in the Templates
            (DICOM).<br>
            Use the Import settings tab to set the details of this sorting process.
            <br><br>
            <b>Templates DICOM</b><br>
            These templates are meant for image based inputs.<br>
            Define settings for how to rename and sort incoming images into
             folders based on DICOM information.<br>
            Combine settings from parametersets and QuickTest templates to
            define how to analyse the images and where/how to output the
            results.
            <br><br>
            <b>Templates vendor reports</b><br>
            These templates are meant for vendor-report based inputs.<br>
            Define where to find the reports, type of report and output path.
            <br><br>
            Currently imageQC have no visualization tools for the trends.
            Until that is in place, use f.x. Excel or PowerBI to visualize
            the trends.
            '''
            )
        super().__init__(header, subtxt)
        self.vLO.addStretch()


class AutoDeleteDialog(uir.ImageQCDialog):
    """Dialog to set auto delete option."""

    def __init__(self, attribute, tag_infos, value='', search_path=''):
        """Initialize AutoDeleteDialog.

        Parameters
        ----------
        attribute : str
            DICOM attribute as in TagInfo
        tag_infos : list of TagInfo
        value : str
            Value of DICOM attribute (if edit). Default is ''
        """
        super().__init__()
        self.setWindowTitle('Auto delete option')

        self.attribute = attribute
        self.tag_infos = tag_infos
        self.txt_value = QLineEdit(value)
        self.search_path = search_path

        vLO = QVBoxLayout()
        self.setLayout(vLO)

        hLO = QHBoxLayout()
        vLO.addLayout(hLO)
        hLO.addWidget(QLabel(attribute))
        hLO.addWidget(self.txt_value)
        tb = QToolBar()
        self.actSearch = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}search.png'),
            'Get value from sample file', self)
        self.actSearch.triggered.connect(self.search_value)
        tb.addActions([self.actSearch])
        hLO.addWidget(tb)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vLO.addWidget(self.buttonBox)

    def get_value(self):
        """Get value of attribute.

        Return
        ------
        value: str
        """
        return self.txt_value.text()

    def search_value(self):
        """Search for attribute value in sample file."""
        value = ''
        fname = QFileDialog.getOpenFileName(
                self, 'Get attribute value from sample file',
                self.search_path,
                filter="DICOM file (*.dcm);;All files (*)")
        if len(fname[0]) > 0:
            tag_pattern_this = cfc.TagPatternFormat(
                list_tags=[self.attribute])#, list_format=[''])
            tags = dcm.get_tags(
                fname[0], tag_patterns=[tag_pattern_this],
                prefix_separator='', suffix_separator='',
                tag_infos=self.tag_infos)
            value = tags[0][0]
        self.txt_value.setText(value)


class AutoCommonWidget(StackWidget):
    """Widget holding common settings for automation."""

    def __init__(self, save_blocked=False, import_review_mode=False):
        header = 'Import settings for automation'
        subtxt = (
            'Define general settings for the process of importing and sorting'
            ' incoming files from the image pool.<br>'
            'If no matching automation template is found, the import'
            ' process will simply rename the files according to the naming'
            ' template defined here.'
            )
        super().__init__(header, subtxt, import_review_mode=import_review_mode)
        self.fname = 'auto_common'

        if import_review_mode:
            tb_marked = QToolBar()
            actImport = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}ok.png'),
                'Mark tag for import', self)
            actImport.triggered.connect(self.mark_import)
            actIgnore = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}deleteRed.png'),
                'Mark tag to ignore', self)
            actIgnore.triggered.connect(
                lambda: self.mark_import(ignore=True))

            tb_marked.addActions(
                [actImport, actIgnore])
            self.import_review_mark_txt = QLabel('Import and overwrite current')
            tb_marked.addWidget(self.import_review_mark_txt)
            hLO_import_tb = QHBoxLayout()
            hLO_import_tb.addStretch()
            hLO_import_tb.addWidget(tb_marked)
            hLO_import_tb.addStretch()
            self.vLO.addLayout(hLO_import_tb)

        self.import_path = QLineEdit()
        self.import_path.setMinimumWidth(500)
        hLO_import_path = QHBoxLayout()
        hLO_import_path.addWidget(QLabel('Image pool path:'))
        hLO_import_path.addWidget(self.import_path)
        tb = uir.ToolBarBrowse('Browse to find path')
        tb.actBrowse.triggered.connect(
            lambda: self.locate_folder(self.import_path))
        hLO_import_path.addWidget(tb)
        self.vLO.addLayout(hLO_import_path)

        hLO = QHBoxLayout()
        vLO_left = QVBoxLayout()
        hLO.addLayout(vLO_left)
        self.vLO.addLayout(hLO)

        vLO_left.addWidget(uir.LabelHeader(
            'Auto delete incoming files with either', 4))
        self.list_auto_delete = QListWidget()
        self.list_auto_delete.setFixedWidth(400)
        hLO_auto_delete = QHBoxLayout()
        hLO_auto_delete.addWidget(self.list_auto_delete)

        self.tb_auto_delete = QToolBar()
        self.tb_auto_delete.setOrientation(Qt.Vertical)
        self.btn_push_crit_delete = QPushButton('<<')
        self.btn_push_crit_delete.clicked.connect(self.push_auto_delete)
        self.act_pop_auto_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete selected row', self)
        self.act_pop_auto_delete.triggered.connect(self.pop_auto_delete)
        self.act_edit_auto_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit selected row', self)
        self.act_edit_auto_delete.triggered.connect(self.edit_auto_delete)
        self.tb_auto_delete.addActions([
            self.act_edit_auto_delete, self.act_pop_auto_delete])
        self.tb_auto_delete.addWidget(self.btn_push_crit_delete)
        hLO_auto_delete.addWidget(self.tb_auto_delete)
        vLO_left.addLayout(hLO_auto_delete)
        vLO_left.addSpacing(20)

        vLO_left.addWidget(uir.LabelHeader(
            'Append or overwrite log file', 4))
        self.cbox_log = QComboBox()
        self.cbox_log.addItems(['overwrite', 'append'])
        hLO_log = QHBoxLayout()
        vLO_left.addLayout(hLO_log)
        hLO_log.addWidget(self.cbox_log)
        tb_log = QToolBar()
        act_info_log = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}info.png'),
            'More about the log file', tb_log)
        act_info_log.triggered.connect(self.info_log)
        act_view_log = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'View log file', tb_log)
        act_view_log.triggered.connect(self.view_log)
        tb_log.addActions([act_info_log, act_view_log])
        hLO_log.addWidget(tb_log)
        vLO_left.addSpacing(20)

        vLO_left.addWidget(uir.LabelHeader(
            'Ignore (leave unsorted) if old images', 4))
        self.chk_ignore_since = QCheckBox('Yes, ignore if more than ')
        self.ignore_since = QSpinBox()
        self.ignore_since.setRange(1, 100)
        hLO_ignore_since = QHBoxLayout()
        vLO_left.addLayout(hLO_ignore_since)
        hLO_ignore_since.addWidget(self.chk_ignore_since)
        hLO_ignore_since.addWidget(self.ignore_since)
        hLO_ignore_since.addWidget(QLabel(' days old'))

        hLO.addWidget(uir.VLine())
        vLO_right = QVBoxLayout()
        hLO.addLayout(vLO_right)
        vLO_right.addWidget(uir.LabelHeader('Import rename pattern', 4))
        self.wTagPattern = uir.TagPatternWidget(
            self, typestr='format', lock_on_general=True)
        vLO_right.addWidget(self.wTagPattern)

        if import_review_mode:
            hLO_import_path.setEnabled(False)
            hLO.setEnabled(False)
            vLO_left.setEnabled(False)
        else:
            btn_save = QPushButton('Save general automation settings')
            btn_save.setIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'))
            btn_save.clicked.connect(self.save_auto_common)
            if save_blocked:
                btn_save.setEnabled(False)
            self.vLO.addWidget(btn_save)

        self.vLO.addWidget(uir.HLine())
        self.vLO.addWidget(self.status_label)

    def update_from_yaml(self):
        """Refresh settings from yaml file.

        Using self.templates as auto_common single template and
        self.current_template as TagPatternFormat to work smoothly
        with general code.
        """
        self.lastload = time()
        ok, path, self.templates = cff.load_settings(fname=self.fname)
        ok, path, self.tag_infos = cff.load_settings(fname='tag_infos')
        self.update_data()
        self.flag_edit(False)

    def update_data(self):
        """Fill GUI with current data."""
        self.current_template = self.templates.filename_pattern
        self.wTagPattern.fill_list_tags('')
        self.wTagPattern.update_data()
        self.import_path.setText(self.templates.import_path)
        self.ignore_since.setValue(self.templates.ignore_since)
        self.chk_ignore_since.setChecked(self.templates.ignore_since > 0)
        self.fill_auto_delete_list()

    def save_auto_common(self):
        """Get current settings and save to yaml file."""
        self.templates.import_path = self.import_path.text()
        if self.chk_ignore_since.isChecked():
            self.templates.ignore_since = self.ignore_since.value()
        else:
            self.templates.ignore_since = 0

        self.save()

    def fill_auto_delete_list(self):
        """Fill list of auto delete settings from current_template."""
        self.list_auto_delete.clear()
        if len(self.templates.auto_delete_criterion_attributenames) > 0:
            for i, attr in enumerate(
                    self.templates.auto_delete_criterion_attributenames):
                txt = self.templates.auto_delete_criterion_attributenames[i]
                txt = txt+' = '+self.templates.auto_delete_criterion_values[i]
                self.list_auto_delete.addItem(txt)

    def push_auto_delete(self):
        """Push currently selected DICOM tag to list of auto delete options."""
        sel_indexes = self.wTagPattern.listTags.selectedIndexes()
        rowno = sel_indexes[0].row()
        tag = self.wTagPattern.listTags.item(rowno).text()
        dlg = AutoDeleteDialog(
            tag, tag_infos=self.tag_infos,
            search_path=self.import_path.text())
        res = dlg.exec()
        if res:
            value = dlg.get_value()
            self.templates.auto_delete_criterion_attributenames.append(tag)
            self.templates.auto_delete_criterion_values.append(value)
            self.fill_auto_delete_list()

    def edit_auto_delete(self):
        """Edit selected auto delete option."""
        sel_indexes = self.list_auto_delete.selectedIndexes()
        if len(sel_indexes) > 0:
            rowno = sel_indexes[0].row()
            dlg = AutoDeleteDialog(
                self,
                self.templates.auto_delete_criterion_attributenames[rowno],
                value=self.templates.auto_delete_criterion_values[rowno],
                tag_infos=self.tag_infos,
                search_path=self.import_path.text())
            res = dlg.exec()
            if res:
                value = res.get_value()
                self.templates.auto_delete_criterion_values[rowno] = value
                self.fill_auto_delete_list()
        else:
            QMessageBox.information(
                self, 'No row selected', 'Select a row to delete.')

    def pop_auto_delete(self):
        """Delete selected auto-delete option."""
        sel_indexes = self.list_auto_delete.selectedIndexes()
        rows2delete = []
        for idx in sel_indexes:
            rows2delete.insert(0, idx.row())
        if len(rows2delete) > 0:
            for i in enumerate(rows2delete):
                self.templates.auto_delete_criterion_attributenames.pop(i)
                self.templates.auto_delete_criterion_values.pop(i)

    def info_log(self):
        """Show info about log."""
        text = [
            'A (local) log will be genereted during import from image pool and',
            'as automation templates are run. This log will be saved at the',
            'same location as the local user settings:',
            f'{ENV_USER_PREFS_PATH}',
            'The log may be rewritten each time import or automation is',
            'initiated or the log may append to the existing log.'
            ]
        dlg = uir.TextDisplay(
            self, '\n'.join(text), title='About the automation log',
            min_width=1000, min_height=300)

    def view_log(self):
        """Display log file contents."""
        if os.path.exists(os.path.join(ENV_USER_PREFS_PATH, LOG_FILENAME)):
            os.startfile(ENV_USER_PREFS_PATH)

    def mark_import(self, ignore=False):
        """If import review mode: Mark AutoCommon for import or ignore."""
        if ignore:
            self.marked = False
            self.marked_ignore = True
            self.import_review_mark_txt.setText('Ignore')
        else:
            self.marked = True
            self.marked_ignore = False
            self.import_review_mark_txt.setText('Import and overwrite current')

class AutoTemplateWidget(StackWidget):
    """Widget holding automation settings."""

    def __init__(self, save_blocked=False, import_review_mode=False):
        header = 'Automation templates DICOM'
        subtxt = '''The automation templates hold information on how to
         perform automated testing on DICOM images.<br>
        The automation template connect-parameter set and other templates
         to be able to use and reuse settings between similar test setups.'''
        self.save_blocked = save_blocked
        super().__init__(header, subtxt,
                         typestr='template',
                         mod_temp=True, grouped=True,
                         import_review_mode=import_review_mode)
        self.fname = 'auto_templates'
        self.empty_template = cfc.AutoTemplate()

        self.txt_input_path = QLineEdit('')
        self.txt_output_path = QLineEdit('')
        self.txt_statname = QLineEdit('')
        self.tree_crit = uir.DicomCritWidget(self)
        self.cbox_paramset = QComboBox()
        self.cbox_quicktest = QComboBox()
        self.list_sort_by = QListWidget()
        self.chk_archive = QCheckBox(
            'Move files to folder "Archive" when finished analysing.')
        self.chk_deactivate = QCheckBox('Deactivate template')

        vLOtemp = QVBoxLayout()
        self.hLO.addLayout(vLOtemp)

        hLOinput_path = QHBoxLayout()
        vLOtemp.addLayout(hLOinput_path)
        hLOinput_path.addWidget(QLabel('Input path  '))
        self.txt_input_path.textChanged.connect(self.flag_edit)
        self.txt_input_path.setMinimumWidth(500)
        hLOinput_path.addWidget(self.txt_input_path)
        tb = uir.ToolBarBrowse('Browse to find path')
        tb.actBrowse.triggered.connect(
            lambda: self.locate_folder(self.txt_input_path))
        hLOinput_path.addWidget(tb)

        hLOoutput_path = QHBoxLayout()
        vLOtemp.addLayout(hLOoutput_path)
        hLOoutput_path.addWidget(QLabel('Output path '))
        self.txt_output_path.textChanged.connect(self.flag_edit)
        self.txt_output_path.setMinimumWidth(500)
        hLOoutput_path.addWidget(self.txt_output_path)
        tb = uir.ToolBarBrowse('Browse to file')
        tb.actBrowse.triggered.connect(
            lambda: self.locate_file(
                self.txt_output_path, title='Locate output file',
                filter_str="Text file (*.txt)"))
        hLOoutput_path.addWidget(tb)
        act_output_view = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'Display output file', tb)
        act_output_view.triggered.connect(
            self.view_output_file)
        actNewFile = QAction(QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                             'Create an empty file', tb)
        actNewFile.triggered.connect(
            lambda: self.locate_file(
                self.txt_output_path, title='Locate output file',
                filter_str="Text file (*.txt)", opensave=True))
        tb.addActions([actNewFile, act_output_view])

        hLOtemp = QHBoxLayout()
        vLOtemp.addLayout(hLOtemp)

        gb_import_settings = QGroupBox('Import criteria from image pool')
        gb_import_settings.setFont(uir.FontItalic())
        vlo_import = QVBoxLayout()
        gb_import_settings.setLayout(vlo_import)
        hLOtemp.addWidget(gb_import_settings)

        hLOstatname = QHBoxLayout()
        hLOstatname.addWidget(QLabel('Station name'))
        hLOstatname.addWidget(self.txt_statname)
        self.txt_statname.textChanged.connect(self.flag_edit)
        self.txt_statname.setMinimumWidth(200)
        hLOstatname.addStretch()
        tb = QToolBar()
        act_get_statname = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}import.png'),
            'Get station name and DICOM criteria values from sample file', tb)
        act_get_statname.triggered.connect(self.get_station_name)
        tb.addActions([act_get_statname])
        hLOstatname.addWidget(tb)
        hLOstatname.addStretch()
        vlo_import.addLayout(hLOstatname)
        vlo_import.addWidget(QLabel(
            'If station name is left empty, at least one additional DICOM criterion'))
        vlo_import.addWidget(QLabel('need to be set for the template to be valid.'))
        vlo_import.addSpacing(20)
        vlo_import.addWidget(uir.LabelItalic(
            'Additional DICOM criteria'))
        vlo_import.addWidget(self.tree_crit)

        gb_analyse = QGroupBox('Image analysis settings')
        gb_analyse.setFont(uir.FontItalic())
        vlo_analyse = QVBoxLayout()
        gb_analyse.setLayout(vlo_analyse)
        hLOtemp.addWidget(gb_analyse)

        fLO_analyse = QFormLayout()
        vlo_analyse.addLayout(fLO_analyse)
        fLO_analyse.addRow(
            QLabel('Use parameter set: '),
            self.cbox_paramset)
        fLO_analyse.addRow(
            QLabel('Use QuickTest template: '),
            self.cbox_quicktest)
        vlo_analyse.addStretch()
        vlo_analyse.addWidget(uir.LabelItalic(
            'Sort images for each date/studyUID by:'))
        hlo_sort_list = QHBoxLayout()
        vlo_analyse.addLayout(hlo_sort_list)
        hlo_sort_list.addWidget(self.list_sort_by)
        self.list_sort_by.setMinimumWidth(300)
        self.tb_edit_sort_by = uir.ToolBarEdit(tooltip='Edit sort list')
        hlo_sort_list.addWidget(self.tb_edit_sort_by)
        self.tb_edit_sort_by.actEdit.triggered.connect(self.edit_sort_by)

        hlo_btm = QHBoxLayout()
        vLOtemp.addLayout(hlo_btm)
        vlo_chk = QVBoxLayout()
        hlo_btm.addLayout(vlo_chk)
        vlo_chk.addWidget(self.chk_archive)
        vlo_chk.addWidget(self.chk_deactivate)
        self.chk_archive.stateChanged.connect(
            lambda: self.flag_edit(True))
        self.chk_deactivate.stateChanged.connect(
            lambda: self.flag_edit(True))

        self.btn_move_modality = QPushButton('Move template to other modality')
        self.btn_move_modality.setToolTip(
            'Did you create the template in wrong modality folder?'
            )
        hlo_btm.addStretch()
        hlo_btm.addWidget(self.btn_move_modality)
        self.btn_move_modality.clicked.connect(self.move_modality)

        if import_review_mode:
            vLOtemp.setEnabled(False)

        self.vLO.addWidget(uir.HLine())
        self.vLO.addWidget(self.status_label)

    def update_data(self):
        """Refresh GUI after selecting template."""
        self.txt_input_path.setText(self.current_template.path_input)
        self.txt_output_path.setText(self.current_template.path_output)
        self.txt_statname.setText(self.current_template.station_name)
        self.tree_crit.update_data()
        if self.current_template.paramset_label != '':
            self.cbox_paramset.setCurrentText(
                self.current_template.paramset_label)
        else:
            self.cbox_paramset.setCurrentIndex(0)
        if self.current_template.quicktemp_label != '':
            self.cbox_quicktest.setCurrentText(
                self.current_template.quicktemp_label)
        else:
            self.cbox_quicktest.setCurrentIndex(0)
        self.fill_list_sort_by()
        self.chk_archive.setChecked(self.current_template.archive)
        self.chk_deactivate.setChecked(not self.current_template.active)
        self.flag_edit(False)

    def get_current_template(self):
        """Get self.current_template where not dynamically set."""
        self.current_template.path_input = self.txt_input_path.text()
        self.current_template.path_output = self.txt_output_path.text()
        self.current_template.station_name = self.txt_statname.text()
        self.current_template.paramset_label = self.cbox_paramset.currentText()
        self.current_template.quicktemp_label = self.cbox_quicktest.currentText()
        self.current_template.archive = self.chk_archive.isChecked()
        self.current_template.active = not self.chk_deactivate.isChecked()

    def fill_lists(self):
        """Fill all lists on modality change."""
        self.fill_list_paramsets()
        self.fill_list_quicktests()

    def fill_list_paramsets(self):
        """Find available paramsets and fill cbox."""
        self.cbox_paramset.clear()
        labels = [obj.label for obj in self.paramsets[self.current_modality]]
        self.cbox_paramset.addItems(labels)

    def fill_list_quicktests(self):
        """Fill list of QuickTest templates."""
        self.cbox_quicktest.clear()
        labels = [obj.label for obj in self.quicktests[self.current_modality]]
        self.cbox_quicktest.addItems(labels)

    def fill_list_sort_by(self):
        """Fill list of sort tags."""
        self.list_sort_by.clear()
        list_tags = self.current_template.sort_pattern.list_tags
        if len(list_tags) > 0:
            list_sort = self.current_template.sort_pattern.list_sort
            for i, tag in enumerate(list_sort):
                asc_desc_txt = '(ASC)' if list_sort[i] is True else '(DESC)'
                self.list_sort_by.addItem(' '.join([list_tags[i], asc_desc_txt]))

    def get_station_name(self):
        """Get station name (and dicom criteria values) from sample file."""
        def_path = self.txt_input_path.text()
        if hasattr(self, 'sample_filepath'):
            if self.sample_filepath != '':
                def_path = self.sample_filepath
        fname = QFileDialog.getOpenFileName(
                self, 'Get station name (+ dicom criteria values) from sample file',
                def_path,
                filter="DICOM file (*.dcm);;All files (*)")
        if len(fname[0]) > 0:
            self.sample_filepath = fname[0]
            tag_pattern_this = cfc.TagPatternFormat(
                list_tags=['StationName'])#, list_format=[''])
            for attr_name in self.current_template.dicom_crit_attributenames:
                tag_pattern_this.list_tags.append(attr_name)
                #tag_pattern_this.list_format.append('')
            tags = dcm.get_tags(
                fname[0], tag_patterns=[tag_pattern_this],
                tag_infos=self.tag_infos)
            self.current_template.station_name = tags[0][0]
            self.txt_statname.setText(tags[0][0])
            if len(tags[0]) > 1:
                for i in range(1, len(tags[0])):
                    self.current_template.dicom_crit_values[i-1] = tags[0][i]
                self.tree_crit.update_data()

    def view_output_file(self):
        """View output file as txt."""
        if os.path.exists(self.txt_output_path.text()):
            os.startfile(self.txt_output_path.text())

    def edit_sort_by(self):
        """Edit list to sort images by."""
        dlg = uir.TagPatternEditDialog(
            initial_pattern=self.current_template.sort_pattern,
            modality=self.current_modality,
            title='Sort images by DICOM header information',
            typestr='sort',
            accept_text='Use',
            reject_text='Cancel',
            save_blocked=self.save_blocked)
        res = dlg.exec()
        if res:
            sort_pattern = dlg.get_pattern()
            self.current_template.sort_pattern = sort_pattern
            self.fill_list_sort_by()

    def move_modality(self):
        """Move selected (saved) template to another modality."""
        if self.edited:
            QMessageBox.information(
                self, 'Save first',
                'Save changes to template before moving to another modality.',
            )
        else:
            new_mod, ok = QInputDialog.getItem(
                self, "Select new modality",
                "Modality", [*QUICKTEST_OPTIONS], 0, False)
            if ok and new_mod:
                if new_mod != self.current_modality:
                    labels_new_mod = \
                        [obj.label for obj
                         in self.templates[new_mod]]
                    label_this = self.current_template.label
                    if label_this in labels_new_mod:
                        text, ok = QInputDialog.getText(
                            self, 'Label exist',
                            'Rename the template')
                        if ok and text != '':
                            label_this = text
                        else:
                            label_this = ''
                    if label_this != '':
                        self.current_template.paramset_label = ''
                        self.current_template.quicktemp_label = ''
                        if self.templates[new_mod][0].label == '':
                            self.templates[
                                new_mod][0] = copy.deepcopy(
                                    self.current_template)
                        else:
                            self.templates[new_mod].append(
                                copy.deepcopy(
                                    self.current_template))
                        self.wModTemp.delete()
                        self.save()
                        self.wModTemp.cbox_modality.setCurrentText(new_mod)


class AutoVendorTemplateWidget(StackWidget):
    """Widget holding automation settings."""

    def __init__(self, save_blocked=False, import_review_mode=False):
        header = 'Automation templates vendor files'
        subtxt = '''The automation templates hold information on how to
         perform automated reading of vendor report files.<br>'''
        self.save_blocked = save_blocked
        super().__init__(header, subtxt,
                         typestr='template',
                         mod_temp=True, grouped=True,
                         import_review_mode=import_review_mode)
        self.fname = 'auto_vendor_templates'
        self.empty_template = cfc.AutoVendorTemplate()

        self.txt_input_path = QLineEdit('')
        self.txt_output_path = QLineEdit('')
        self.txt_statname = QLineEdit('')
        self.cbox_file_type = QComboBox()
        self.chk_archive = QCheckBox(
            'Archive files when analysed (Archive folder in input path).')
        self.chk_deactivate = QCheckBox('Deactivate template')

        vLOtemp = QVBoxLayout()
        self.hLO.addLayout(vLOtemp)

        hLOinput_path = QHBoxLayout()
        vLOtemp.addLayout(hLOinput_path)
        hLOinput_path.addWidget(QLabel('Input path '))
        self.txt_input_path.textChanged.connect(self.flag_edit)
        self.txt_input_path.setMinimumWidth(500)
        hLOinput_path.addWidget(self.txt_input_path)
        tb = uir.ToolBarBrowse('Browse to find path')
        tb.actBrowse.triggered.connect(
            lambda: self.locate_folder(self.txt_input_path))
        hLOinput_path.addWidget(tb)

        hLOoutput_path = QHBoxLayout()
        vLOtemp.addLayout(hLOoutput_path)
        hLOoutput_path.addWidget(QLabel('Output path '))
        self.txt_output_path.textChanged.connect(self.flag_edit)
        self.txt_output_path.setMinimumWidth(500)
        hLOoutput_path.addWidget(self.txt_output_path)
        tb = uir.ToolBarBrowse('Browse to file')
        tb.actBrowse.triggered.connect(
            lambda: self.locate_file(
                self.txt_output_path, title='Locate output file',
                filter_str="Text file (*.txt)"))
        hLOoutput_path.addWidget(tb)
        act_output_view = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}file.png'),
            'Display output file', tb)
        act_output_view.triggered.connect(
            self.view_output_file)
        actNewFile = QAction(QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                             'Create an empty file', tb)
        actNewFile.triggered.connect(
            lambda: self.locate_file(
                self.txt_output_path, title='Locate output file',
                filter_str="Text file (*.txt)", opensave=True))
        tb.addActions([actNewFile, act_output_view])

        hLOstatname = QHBoxLayout()
        hLOstatname.addWidget(QLabel('Station ID'))
        hLOstatname.addWidget(self.txt_statname)
        self.txt_statname.textChanged.connect(self.flag_edit)
        self.txt_statname.setMinimumWidth(300)
        tb = QToolBar()
        act_get_statname = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}import.png'),
            'Get station name from sample file', tb)
        act_get_statname.triggered.connect(
            self.get_station_name)
        tb.addActions([act_get_statname])
        hLOstatname.addWidget(tb)
        hLOstatname.addWidget(uir.LabelItalic(
            ('Only used for verification if station is set in vendor '
             + 'report file.')))
        hLOstatname.addStretch()
        vLOtemp.addLayout(hLOstatname)

        hLOoptions = QHBoxLayout()
        hLOoptions.addWidget(QLabel('Vendor file type: '))
        hLOoptions.addWidget(self.cbox_file_type)
        hLOoptions.addStretch()
        vLOtemp.addLayout(hLOoptions)
        vLOtemp.addStretch()
        vLOtemp.addWidget(self.chk_archive)
        self.chk_archive.stateChanged.connect(
            lambda: self.flag_edit(True))
        vLOtemp.addWidget(self.chk_deactivate)
        self.chk_deactivate.stateChanged.connect(
            lambda: self.flag_edit(True))
        vLOtemp.addStretch()

        if import_review_mode:
            vLOtemp.setEnabled(False)

        self.vLO.addWidget(uir.HLine())
        self.vLO.addWidget(self.status_label)

    def update_from_yaml(self):
        """Refresh settings from yaml file."""
        super().update_from_yaml()
        self.update_file_types()

    def update_data(self):
        """Refresh GUI after selecting template."""
        if self.current_template.file_type != '':
            self.cbox_file_type.setCurrentText(
                self.current_template.file_type)
        self.txt_input_path.setText(self.current_template.path_input)
        self.txt_output_path.setText(self.current_template.path_output)
        self.txt_statname.setText(self.current_template.station_name)
        self.chk_archive.setChecked(self.current_template.archive)
        self.chk_deactivate.setChecked(not self.current_template.active)

        self.flag_edit(False)

    def get_current_template(self):
        """Get self.current_template where not dynamically set."""
        self.current_template.path_input = self.txt_input_path.text()
        self.current_template.path_output = self.txt_output_path.text()
        self.current_template.station_name = self.txt_statname.text()
        self.current_template.active = not self.chk_deactivate.isChecked()

    def get_station_name(self):
        """Get station name from sample file.

        Return
        ------
        statname : str
            station name if found/defined in vendor report file
        """
        statname = ''
        file_type = self.cbox_file_type.currentText()

        open_title = ''
        file_filter = ''
        old_status = self.status_label.text()
        if self.current_modality == 'CT':
            open_title = 'Open Siemens CT QC report file'
            file_filter = "PDF file (*.pdf)"
        elif self.current_modality == 'PET':
            if 'pdf' in file_type:
                open_title = 'Open Siemens CT QC report file'
                file_filter = "PDF file (*.pdf)"
        else:
            pass

        if open_title != '':
            res = {'status': False}
            fname = QFileDialog.getOpenFileName(
                    self, open_title, filter=file_filter)
            if len(fname[0]) > 0:
                self.status_label.setText('Please wait while reading file....')
                QApplication.setOverrideCursor(Qt.WaitCursor)
                qApp.processEvents()
                if self.current_modality == 'CT':
                    txt = read_vendor_QC_reports.get_pdf_txt(fname[0])
                    res = read_vendor_QC_reports.read_Siemens_CT_QC(txt)
                    if res['status']:
                        if len(res['values_txt']) > 3:
                            statname = res['values_txt'][3]
                elif self.current_modality == 'PET' and 'pdf' in file_type:
                    txt = read_vendor_QC_reports.get_pdf_txt(fname[0])
                    res = read_vendor_QC_reports.read_Siemens_PET_dailyQC(txt)
                    if res['status']:
                        if len(res['values_txt']) > 1:
                            statname = res['values_txt'][1]
                QApplication.restoreOverrideCursor()
                self.status_label.setText(old_status)

        if statname == '':
            QMessageBox.information(
                self, 'Station ID not found',
                'No station ID defined or found for the selected file type.')
        else:
            self.txt_statname.setText(statname)

    def update_file_types(self):
        """Update list of file_types on modality change."""
        self.cbox_file_type.clear()
        self.cbox_file_type.addItems(
            VENDOR_FILE_OPTIONS[self.current_modality])

    def view_output_file(self):
        """View output file as txt."""
        if os.path.exists(self.txt_output_path.text()):
            os.startfile(self.txt_output_path.text())


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
    tag_infos_new: list = field(default_factory=list)
    tag_patterns_special: dict = field(default_factory=dict)
    tag_patterns_format: dict = field(default_factory=dict)
    tag_patterns_sort: dict = field(default_factory=dict)
    rename_patterns: dict = field(default_factory=dict)
    quicktest_templates: dict = field(default_factory=dict)
    paramsets: dict = field(default_factory=dict)
    quicktests: dict = field(default_factory=dict)
    auto_common: cfc.AutoCommon = field(default_factory=cfc.AutoCommon)
    auto_templates: dict = field(default_factory=dict)
    auto_vendor_templates: dict = field(default_factory=dict)
