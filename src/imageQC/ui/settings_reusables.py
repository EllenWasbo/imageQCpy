#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for configuration settings -  reusable classes.

@author: Ellen Wasbo
"""
import os
from time import time
import copy
import numpy as np

from PyQt6.QtCore import Qt, QModelIndex
from PyQt6.QtGui import QIcon, QAction, QBrush, QColor, QStandardItemModel
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QDialogButtonBox,
    QToolBar, QLabel, QLineEdit, QPushButton, QCheckBox, QDoubleSpinBox,
    QTreeWidget, QTreeWidgetItem, QTreeView,
    QListWidget, QListWidgetItem, QComboBox, QInputDialog, QMessageBox, QFileDialog
    )

# imageQC block start
from imageQC.config.iQCconstants import (
    QUICKTEST_OPTIONS, ENV_ICON_PATH, ALTERNATIVES, HEADERS, HEADERS_SUP,
    CALCULATION_OPTIONS)
from imageQC.config import config_func as cff
from imageQC.config import config_classes as cfc
from imageQC.ui import reusable_widgets as uir
from imageQC.ui import messageboxes
from imageQC.ui.ui_dialogs import ImageQCDialog, CmapSelectDialog
from imageQC.scripts.mini_methods import create_empty_file, get_included_tags
from imageQC.scripts.mini_methods_format import valid_template_name
# imageQC block end


class StackWidget(QWidget):
    """Class for general widget attributes for the stacked widgets."""

    def __init__(self, dlg_settings=None, header='', subtxt='', temp_alias='template',
                 mod_temp=False, grouped=False, editable=True):
        """Initiate StackWidget.

        Parameters
        ----------
        dlg_settings: QDialog
            parent of stackWidget
        header : str
            header text
        subtxt : str
            info text under header text
        temp_alias : str
            string to set type of data (parameterset or template +?)
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
        self.dlg_settings = dlg_settings
        self.temp_alias = temp_alias
        self.mod_temp = mod_temp
        self.grouped = grouped
        self.edited = False
        self.lastload = None
        self.templates = None
        try:
            self.import_review_mode = self.dlg_settings.import_review_mode
            self.save_blocked = self.dlg_settings.main.save_blocked
        except AttributeError:
            self.import_review_mode = False
            self.save_blocked = False
        try:
            self.current_modality = self.dlg_settings.initial_modality
        except AttributeError:
            self.current_modality = 'CT'
        self.status_label = QLabel('')

        self.vlo = QVBoxLayout()
        self.setLayout(self.vlo)
        if header != '':
            self.vlo.addWidget(uir.LabelHeader(header, 3))
        if subtxt != '':
            self.vlo.addWidget(uir.LabelItalic(subtxt))
        self.vlo.addWidget(uir.HLine())

        if self.mod_temp:
            self.hlo = QHBoxLayout()
            self.vlo.addLayout(self.hlo)
            self.wid_mod_temp = ModTempSelector(
                self, editable=editable, import_review_mode=self.import_review_mode)
            self.hlo.addWidget(self.wid_mod_temp)
            if self.grouped is False:
                self.wid_mod_temp.lbl_modality.setVisible(False)
                self.wid_mod_temp.cbox_modality.setVisible(False)

    def flag_edit(self, flag=True):
        """Indicate some change."""
        if flag:
            self.edited = True
            self.status_label.setText('**Unsaved changes**')
        else:
            self.edited = False
            self.status_label.setText('')

    def update_from_yaml(self, initial_template_label=''):
        """Refresh settings from yaml file."""
        self.lastload = time()

        if hasattr(self, 'fname'):
            _, _, self.templates = cff.load_settings(fname=self.fname)
            if 'patterns' in self.fname or self.fname == 'auto_templates':
                _, _, self.tag_infos = cff.load_settings(fname='tag_infos')
            elif self.fname in [
                    'paramsets', 'quicktest_templates',
                    'limits_and_plot_templates']:
                _, _, self.auto_templates = cff.load_settings(
                    fname='auto_templates')
                if self.fname in ['limits_and_plot_templates']:
                    _, _, self.auto_vendor_templates = cff.load_settings(
                        fname='auto_vendor_templates')
            elif self.fname == 'digit_templates':
                _, _, self.paramsets = cff.load_settings(fname='paramsets')

            if self.fname == 'auto_templates':
                _, _, self.auto_common = cff.load_settings(fname='auto_common')
                _, _, self.paramsets = cff.load_settings(fname='paramsets')
                _, _, self.quicktest_templates = cff.load_settings(
                    fname='quicktest_templates')
                self.fill_lists()

            if self.fname in ['auto_templates', 'auto_vendor_templates']:
                _, _, self.limits_and_plot_templates = cff.load_settings(
                    fname='limits_and_plot_templates')
                self.fill_list_limits_and_plot()

            if self.grouped:
                self.wid_mod_temp.cbox_modality.setCurrentText(
                    self.current_modality)
                if 'patterns' in self.fname:
                    avoid_special_tags = (self.fname == 'rename_patterns')
                    try:
                        self.wid_tag_pattern.fill_list_tags(
                            self.current_modality,
                            avoid_special_tags=avoid_special_tags)
                    except AttributeError:
                        pass  # ignore if editable == False

            if self.mod_temp:
                self.refresh_templist(selected_label=initial_template_label)
            else:
                self.update_data()

    def update_modality(self):
        """Refresh GUI after selecting modality (stack with ModTempSelector."""
        if self.edited and self.import_review_mode is False:
            if hasattr(self, 'current_template'):
                res = messageboxes.QuestionBox(
                    parent=self, title='Save changes?',
                    msg='Save changes before changing modality?')
                res.exec()
                if res.clickedButton() == res.yes:
                    self.wid_mod_temp.save()
            else:
                pass
                # TODO - quickfix to avoid error when change widget and modality not CT
                # cbox modality triggered and for some reason self.edited is True....
                # find better solution some time

        self.current_modality = self.wid_mod_temp.cbox_modality.currentText()

        try:
            self.dlg_settings.main.start_wait_cursor()
        except AttributeError:
            pass
        if 'patterns' in self.fname:
            avoid_special_tags = (self.fname == 'rename_patterns')
            try:
                self.wid_tag_pattern.fill_list_tags(
                    self.current_modality, avoid_special_tags=avoid_special_tags)
            except AttributeError:
                pass  # ignore if editable = False
        elif self.fname == 'paramsets':
            self.empty_template = self.get_empty_paramset()
        elif self.fname == 'quicktest_templates':
            self.wid_test_table.update_modality()
        elif 'vendor' in self.fname:
            self.update_file_types()
        elif self.fname == 'auto_templates':
            self.fill_lists()

        if self.fname in ['auto_templates', 'auto_vendor_templates']:
            self.fill_list_limits_and_plot()

        self.refresh_templist()
        try:
            self.dlg_settings.main.stop_wait_cursor()
        except AttributeError:
            pass

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
        tempno = max(tempno, 0)
        if tempno > len(self.current_labels)-1:
            tempno = len(self.current_labels)-1

        if len(self.current_labels) == 0:
            self.current_template = copy.deepcopy(self.empty_template)
        else:
            self.update_current_template(selected_id=tempno)

        self.wid_mod_temp.list_temps.blockSignals(True)
        self.wid_mod_temp.list_temps.clear()
        if self.import_review_mode:
            self.refresh_templist_icons()
        else:
            self.wid_mod_temp.list_temps.addItems(self.current_labels)
        self.wid_mod_temp.list_temps.setCurrentRow(tempno)
        self.wid_mod_temp.list_temps.blockSignals(False)

        if 'auto' in self.fname and 'temp' in self.fname:
            if self.current_modality in self.templates:
                active = [obj.active for obj in self.templates[self.current_modality]]
                brush = QBrush(QColor(170, 170, 170))
                for i, active_this in enumerate(active):
                    if active_this is False:
                        self.wid_mod_temp.list_temps.item(i).setForeground(brush)
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

            self.wid_mod_temp.list_temps.addItem(QListWidgetItem(icon, label))

    def update_clicked_template(self):
        """Update data after new template selected (clicked)."""
        if self.edited:
            res = messageboxes.QuestionBox(
                parent=self, title='Save changes?',
                msg='Save changes before changing template?')
            res.exec()
            if res.clickedButton() == res.yes:
                self.wid_mod_temp.save(label=self.current_template.label)
            else:
                self.flag_edit(False)

        tempno = self.wid_mod_temp.list_temps.currentIndex().row()
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
        dlg.setFileMode(QFileDialog.FileMode.Directory)
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
            fname, _ = QFileDialog.getSaveFileName(
                self, title, widget.text(), filter=filter_str)
            if fname != '':
                create_empty_file(fname, self, proceed=True)
        else:
            fname, _ = QFileDialog.getOpenFileName(
                self, title, widget.text(), filter=filter_str)
        if fname != '':
            widget.setText(os.path.normpath(fname))
        self.flag_edit()

    def get_data(self):
        """Update current_template into templates. Called by save."""
        if (hasattr(self.__class__, 'get_current_template')
            and callable(getattr(
                self.__class__, 'get_current_template'))):
            self.get_current_template()

    def add(self, label):
        """Add current_template or empty_template to templates."""
        self.get_data()  # if get_current_template exist
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
            if len(self.templates) == 0:
                self.templates = [copy.deepcopy(self.current_template)]
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
        more_fnames = None
        log = []
        mod = self.current_modality
        if self.fname in ['paramsets', 'quicktest_templates',
                          'limits_and_plot_templates']:
            if self.fname in ['limits_and_plot_templates']:
                if self.current_template.type_vendor:
                    more_fnames = ['auto_vendor_templates']
                else:
                    more_fnames = ['auto_templates']
            else:
                more_fnames = ['auto_templates']
            for more_fname in more_fnames:
                _, path, auto_templates = cff.load_settings(fname=more_fname)

                if path != '':
                    if self.fname == 'paramsets':
                        ref_attr = 'paramset_label'
                    elif self.fname == 'quicktest_templates':
                        ref_attr = 'quicktemp_label'
                    elif self.fname == 'limits_and_plot_templates':
                        ref_attr = 'limits_and_plot_label'
                    temp_auto = cff.get_ref_label_used_in_auto_templates(
                        auto_templates, ref_attr=ref_attr)
                    _, temp_labels = np.array(temp_auto[mod]).T.tolist()
                    changed = False

                    if oldlabel in temp_labels:
                        for i, temp in enumerate(temp_labels):
                            if temp == oldlabel:
                                setattr(auto_templates[mod][i], ref_attr, newlabel)
                                changed = True

                    if changed:
                        log.append(
                            f'{self.fname[:-1]} {oldlabel} used in {more_fname}. '
                            'Label updated.')
                        save_more = True
                        if more is None:
                            more = [auto_templates]
                        else:
                            more.append(auto_templates)

        elif self.fname == 'digit_templates':
            more_fname = f'paramsets_{mod}'
            more_fnames = [more_fname]
            _, path, paramsets = cff.load_settings(fname=more_fname)

            if path != '':
                digit_labels_used = [temp.num_digit_label for temp in paramsets]

                changed = False
                if oldlabel in digit_labels_used:
                    for i, temp in enumerate(digit_labels_used):
                        if temp == oldlabel:
                            paramsets[i].num_digit_label = newlabel
                            changed = True

                if changed:
                    log.append(
                        f'{self.fname[:-1]} {oldlabel} used in paramsets. '
                        'Label updated.')
                    save_more = True
                    more = [auto_templates]

        self.save(save_more=save_more, more=more,
                  more_fnames=more_fnames, log=log)
        self.refresh_templist(selected_label=newlabel)

    def move_modality(self):
        """Move selected (saved) template to another modality."""
        if self.edited:
            QMessageBox.information(
                self, 'Save first',
                'Save changes to template before moving to another modality.',
            )
        else:
            new_mod, proceed = QInputDialog.getItem(
                self, "Change modality",
                "Change modality to                  ", [*QUICKTEST_OPTIONS], 0, False)
            if proceed and new_mod:
                if new_mod != self.current_modality:
                    labels_new_mod = \
                        [obj.label for obj
                         in self.templates[new_mod]]
                    label_this = self.current_template.label
                    if label_this in labels_new_mod:
                        text, proceed = QInputDialog.getText(
                            self, 'Name exist',
                            'Name exist in target modality. Rename the template:')
                        if proceed and text != '':
                            label_this = text
                        else:
                            label_this = ''
                    if label_this != '' and label_this not in labels_new_mod:
                        self.current_template.label = label_this
                        if self.fname == 'auto_templates':
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
                        self.wid_mod_temp.delete()
                        self.save()
                        self.wid_mod_temp.cbox_modality.setCurrentText(new_mod)

    def save(self, save_more=False, more=None, more_fnames=None, log=[]):
        """Save template and other connected templates if needed.

        Parameters
        ----------
        save_more : bool, optional
            Connected templates to be saved exist. The default is False.
        more : list of templates, optional
            Connected templates to save. The default is None.
        more_fnames : list of str, optional
            fnames of connected templates. The default is None.
        log : list of str, optional
            Log from process of connected templates. The default is [].
        """
        def digit_templates_tolist(templates):
            for key, templist in templates.items():
                for tempno, temp in enumerate(templist):
                    for imgno, img in enumerate(temp.images):
                        if isinstance(img, np.ndarray):  # to list to save to yaml
                            templates[key][tempno].images[imgno] = img.tolist()

        proceed = cff.verify_config_folder(self)
        if proceed:
            if self.fname == 'paramsets':
                fname = f'paramsets_{self.current_modality}'
                templates = self.templates[self.current_modality]
            else:
                fname = self.fname
                templates = self.templates
            proceed, errmsg = cff.check_save_conflict(fname, self.lastload)
            if errmsg != '':
                proceed = messageboxes.proceed_question(self, errmsg)
            if proceed:
                if fname == 'digit_templates':
                    digit_templates_tolist(templates)
                ok_save, path = cff.save_settings(templates, fname=fname)
                if ok_save:
                    if save_more:
                        ok_save = []
                        for i, more_fname in enumerate(more_fnames):
                            proceed, errmsg = cff.check_save_conflict(
                                more_fname, self.lastload)
                            if errmsg != '':
                                proceed = messageboxes.proceed_question(self, errmsg)
                            if proceed:
                                ok_save_this, path = cff.save_settings(
                                    more[i], fname=more_fname)
                                ok_save.append(ok_save_this)
                        if len(ok_save) > 0:
                            if all(ok_save):
                                if log:
                                    dlg = messageboxes.MessageBoxWithDetails(
                                        self, title='Updated related templates',
                                        msg=('Related templates also updated. '
                                             'See details to view changes performed'),
                                        details=log, icon=QMessageBox.Icon.Information)
                                    dlg.exec()
                                if self.fname in ['paramsets', 'quicktest_templates']:
                                    self.update_from_yaml()
                                if self.fname == 'auto_vendor_templates':
                                    if 'limits_and_plot_templates' in more_fnames:
                                        self.fill_list_limits_and_plot()
                    self.status_label.setText(
                        f'Changes saved to {path}')
                    self.flag_edit(False)
                    self.lastload = time()
                    if self.fname == 'tag_patterns_special':
                        QMessageBox.information(
                            self, 'Saved',
                            'Changes will not have effect on already opened images.')
                else:
                    QMessageBox.warning(
                        self, 'Failed saving', f'Failed saving to {path}')

        if 'auto' in self.fname and 'template' in self.fname:
            # Ensure refresh templist with foreground color for inactive templates.
            row = self.wid_mod_temp.list_temps.currentRow()
            self.refresh_templist(selected_id=row)


class ModTempSelector(QWidget):
    """Widget with modality selector, template selector and toolbar."""

    def __init__(self, parent, editable=True, import_review_mode=False):
        super().__init__()
        self.parent = parent
        self.setFixedWidth(400)

        self.vlo = QVBoxLayout()
        self.setLayout(self.vlo)
        hlo_modality = QHBoxLayout()
        self.vlo.addLayout(hlo_modality)
        self.lbl_modality = uir.LabelItalic('Modality')
        hlo_modality.addWidget(self.lbl_modality)
        self.cbox_modality = QComboBox()
        self.cbox_modality.addItems([*QUICKTEST_OPTIONS])
        self.cbox_modality.currentIndexChanged.connect(
            self.parent.update_modality)
        self.cbox_modality.setFixedWidth(150)
        hlo_modality.addWidget(self.cbox_modality)
        hlo_modality.addStretch()
        self.vlo.addSpacing(10)
        self.vlo.addWidget(uir.LabelItalic(self.parent.temp_alias.title()+'s'))
        hlo_list = QHBoxLayout()
        self.vlo.addLayout(hlo_list)
        self.list_temps = QListWidget()
        self.list_temps.currentItemChanged.connect(self.parent.update_clicked_template)
        hlo_list.addWidget(self.list_temps)

        if import_review_mode:
            self.toolbar = ToolBarImportIgnore(self, temp_alias=self.parent.temp_alias)
            hlo_list.addWidget(self.toolbar)
        else:
            if editable:
                self.toolbar = QToolBar()
                self.toolbar.setOrientation(Qt.Orientation.Vertical)
                hlo_list.addWidget(self.toolbar)
                self.act_clear = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}clear.png'),
                    'Clear ' + self.parent.temp_alias + ' (reset to default)', self)
                self.act_clear.triggered.connect(self.clear)
                self.act_add = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                    'Add current values as new ' + self.parent.temp_alias, self)
                self.act_add.triggered.connect(self.add)
                self.act_save = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
                    'Save current values to ' + self.parent.temp_alias, self)
                self.act_save.triggered.connect(self.save)
                self.act_rename = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}rename.png'),
                    'Rename ' + self.parent.temp_alias, self)
                self.act_rename.triggered.connect(self.rename)
                self.act_duplicate = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}duplicate.png'),
                    'Duplicate ' + self.parent.temp_alias, self)
                self.act_duplicate.triggered.connect(self.duplicate)
                self.act_up = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
                    'Move up', self)
                self.act_up.triggered.connect(self.move_up)
                self.act_down = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
                    'Move down', self)
                self.act_down.triggered.connect(self.move_down)
                self.act_delete = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
                    'Delete ' + self.parent.temp_alias, self)
                self.act_delete.triggered.connect(self.delete)
                self.act_move_modality = QAction(
                    QIcon(f'{os.environ[ENV_ICON_PATH]}move_to.png'),
                    'Move ' + self.parent.temp_alias + ' to other modality', self)
                self.act_move_modality.triggered.connect(self.parent.move_modality)

                if self.parent.save_blocked:
                    self.act_clear.setEnabled(False)
                    self.act_add.setEnabled(False)
                    self.act_save.setEnabled(False)
                    self.act_duplicate.setEnabled(False)
                    self.act_rename.setEnabled(False)
                    self.act_up.setEnabled(False)
                    self.act_down.setEnabled(False)
                    self.act_delete.setEnabled(False)
                    self.act_move_modality.setEnabled(False)

                self.toolbar.addActions(
                    [self.act_clear, self.act_add, self.act_save, self.act_duplicate,
                     self.act_rename, self.act_up,
                     self.act_down, self.act_delete, self.act_move_modality])

    def keyPressEvent(self, event):
        """Accept Delete and arrow up/down key on list templates."""
        if event.key() == Qt.Key.Key_Delete:
            self.delete()
        else:
            super().keyPressEvent(event)

    def clear(self):
        """Clear template - set like empty_template."""
        try:
            self.parent.clear()
        except AttributeError:
            try:
                lbl = self.parent.current_template.label
                self.parent.current_template = copy.deepcopy(
                    self.parent.empty_template)
                self.parent.current_template.label = lbl
                self.parent.update_data()
                self.parent.flag_edit(True)
            except AttributeError:
                print('Missing empty template (method clear in ModTempSelector)')

    def add(self):
        """Add new template to list. Ask for new name and verify."""
        text, proceed = QInputDialog.getText(
            self, 'New name',
            'Name the new ' + self.parent.temp_alias + '                      ')
        # todo also ask if add as current or as empty
        text = valid_template_name(text)
        if proceed and text != '':
            if text in self.parent.current_labels:
                QMessageBox.warning(
                    self, 'Name already in use',
                    'This name is already in use.')
            else:
                self.parent.add(text)
        if self.parent.fname == 'digit_templates':
            self.parent.edit_template()

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
            proceed = True
            if self.parent.edited:
                res = messageboxes.QuestionBox(
                    parent=self, title='Rename edited?',
                    msg='''Selected template has changed.
                    Save changes before rename?''',
                    yes_text='Yes',
                    no_text='Cancel')
                res.exec()
                if res.clickedButton() == res.yes:
                    self.save()
                else:
                    proceed = False

            if proceed:
                sel = self.list_temps.currentItem()
                if sel is not None:
                    current_text = sel.text()

                    text, proceed = QInputDialog.getText(
                        self, 'New name',
                        'Rename ' + self.parent.temp_alias + '                      ',
                        text=current_text)
                    text = valid_template_name(text)
                    if proceed and text != '' and current_text != text:
                        if text in self.parent.current_labels:
                            QMessageBox.warning(
                                self, 'Name already in use',
                                'This name is already in use.')
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
                res = messageboxes.QuestionBox(
                    parent=self, title='Duplicate or add edited?',
                    msg='''Selected template has changed.
                    Add with current parameters or duplicate original?''',
                    yes_text='Add new with current parameter',
                    no_text='Duplicate original')
                res.exec()
                if res.clickedButton() == res.yes:
                    self.add()
                    proceed = False

            if proceed:  # duplicate original
                sel = self.list_temps.currentItem()
                current_text = sel.text()
                duplicate_id = self.parent.current_labels.index(current_text)

                text, proceed = QInputDialog.getText(
                    self, 'New name',
                    'Name the new ' + self.parent.temp_alias + '                      ',
                    text=f'{current_text}_')
                text = valid_template_name(text)
                if proceed and text != '':
                    if text in self.parent.current_labels:
                        QMessageBox.warning(
                            self, 'Name already in use',
                            'This name is already in use.')
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

    def delete(self, confirmed=False):
        """Delete template."""
        if self.parent.current_labels[0] == '':
            QMessageBox.warning(
                self, 'Empty list',
                'No template to delete.')
        else:
            qtext = ''
            if hasattr(self.parent, 'list_used_in'):
                if self.parent.list_used_in.count() > 0:
                    qtext = ' and all links to automation templates'

            if confirmed is False:
                res = QMessageBox.question(
                    self, 'Delete?', f'Delete selected template{qtext}?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if res == QMessageBox.StandardButton.Yes:
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

                    # also reset link to auto_templates
                    if qtext != '':
                        type_vendor = False
                        if self.parent.fname == 'limits_and_plot_templates':
                            type_vendor = self.parent.current_template.type_vendor
                        if type_vendor:
                            auto_labels = [
                                self.parent.list_used_in.item(i).text() for i
                                in range(self.parent.list_used_in.count())]
                            for temp in self.parent.auto_vendor_templates[
                                    self.parent.current_modality]:
                                if temp.label in auto_labels:
                                    temp.limits_and_plot_label = ''
                            auto_widget =\
                                self.parent.dlg_settings.widget_auto_vendor_templates
                            auto_widget.templates = copy.deepcopy(
                                self.parent.auto_vendor_templates)
                        else:
                            auto_labels = [
                                self.parent.list_used_in.item(i).text() for i
                                in range(self.parent.list_used_in.count())]
                            for temp in self.parent.auto_templates[
                                    self.parent.current_modality]:
                                if temp.label in auto_labels:
                                    if self.parent.fname == 'quicktest_templates':
                                        temp.quicktemp_label = ''
                                    elif self.parent.fname == 'limits_and_plot_templates':
                                        temp.limits_and_plot_label = ''
                                    else:
                                        temp.paramset_label = ''
                            auto_widget = self.parent.dlg_settings.widget_auto_templates
                            auto_widget.templates = copy.deepcopy(
                                self.parent.auto_templates)

                        auto_widget.lastload = self.parent.lastload
                        auto_widget.save()

                    self.parent.save()
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
                    if self.parent.fname in ['auto_templates', 'auto_vendor_templates']:
                        self.parent.dlg_settings.mark_qt_param_limits(
                            self.parent.current_template,
                            mod=self.parent.current_modality
                            )
                    if self.parent.fname == 'paramsets':
                        if self.parent.current_template.num_digit_label != '':
                            self.parent.dlg_settings.mark_digit_temps(
                                self.parent.current_template,
                                mod=self.parent.current_modality
                                )
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

        self.parent.refresh_templist(selected_id=row)


class ToolBarImportIgnore(QToolBar):
    """Toolbar with import or ignore buttons for import mode of dlg_settings."""

    def __init__(self, parent, temp_alias='template', orientation=Qt.Orientation.Vertical):
        """Initiate toolbar.

        Parameters
        ----------
        parent: widget with class method 'mark_import'
        temp_alias : str
            string to set type of data (parameterset or template)
        orientation: Qt.Orientation.Vertical/Horizontal
            Default is Qt.Orientation.Vertical
        """
        super().__init__()
        self.setOrientation(orientation)
        self.act_import = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}ok.png'),
            'Mark ' + temp_alias + ' for import', parent)
        self.act_import.triggered.connect(parent.mark_import)
        self.act_ignore = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}deleteRed.png'),
            'Mark ' + temp_alias + ' to ignore', parent)
        self.act_ignore.triggered.connect(
            lambda: parent.mark_import(ignore=True))

        self.addActions([self.act_import, self.act_ignore])


class QuickTestTreeView(QTreeView):
    """QTreeWidget for list of images marked for testing."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.update_model()
        self.setModel(self.model)
        self.selectionModel().currentChanged.connect(self.update_selimg)

    def update_model(self):
        """Set model headers based on current modality."""
        self.tests = QUICKTEST_OPTIONS[self.parent.current_modality]
        self.model = QStandardItemModel(0, len(self.tests) + 2, self.parent)
        self.model.setHeaderData(0, Qt.Orientation.Horizontal, "Image label")
        self.model.setHeaderData(1, Qt.Orientation.Horizontal, "Group label")

        for i, test in enumerate(self.tests):
            self.model.setHeaderData(i+2, Qt.Orientation.Horizontal, test)

        self.model.itemChanged.connect(self.parent.flag_edit)

    def update_modality(self):
        """Update model when modality change."""
        self.model.beginResetModel()
        self.model.clear()
        self.update_model()
        self.setModel(self.model)
        self.model.endResetModel()

    def update_data(self, set_selected=0):
        """Set data to self.parent.current_template.

        Parameters
        ----------
        set_selected : int
            Row number to set as selected when finished. Default is 0
        """
        self.model.beginResetModel()
        self.model.blockSignals(True)

        n_rows = self.model.rowCount()
        for i in range(n_rows):
            self.model.removeRow(n_rows-i-1, QModelIndex())

        temp = self.parent.current_template
        for imgno, img_tests in enumerate(temp.tests):
            self.model.insertRow(imgno)
            try:
                name = temp.image_names[imgno]
            except IndexError:
                name = ''
            self.model.setData(self.model.index(imgno, 0),
                               name, Qt.ItemDataRole.EditRole)
            try:
                name = temp.group_names[imgno]
            except IndexError:
                name = ''
            self.model.setData(self.model.index(imgno, 1),
                               name, Qt.ItemDataRole.EditRole)
            for testno, test in enumerate(self.tests):
                state = (Qt.CheckState.Checked if test in img_tests else Qt.CheckState.Unchecked)
                self.model.setData(self.model.index(imgno, testno+2),
                                   state, role=Qt.ItemDataRole.CheckStateRole)
                item = self.model.itemFromIndex(self.model.index(imgno, testno+2))
                item.setEditable(False)
                item.setCheckable(True)
        self.model.blockSignals(False)

        self.setColumnWidth(0, 170)
        self.setColumnWidth(1, 170)
        for i in range(len(self.tests)):
            self.setColumnWidth(i+2, 60)
        self.header().setStretchLastSection(False)
        self.model.endResetModel()
        self.setCurrentIndex(self.model.index(set_selected, 0))

    def update_selimg(self):
        """Update label showing which image number is selected."""
        self.parent.lbl_selimg.setText(f'{self.currentIndex().row()}')

    def insert_empty_row(self):
        """Insert empty row after selected or at end."""
        sel = self.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row() + 1
        else:
            rowno = self.model.rowCount()
        temp = self.get_data()
        if temp.tests == [[]] and temp.image_names == ['']:
            rowno = 0
        self.model.beginInsertRows(self.model.index(rowno, 0), rowno, rowno)
        self.model.insertRow(rowno)
        self.model.setData(self.model.index(rowno, 0), '', Qt.ItemDataRole.EditRole)
        self.model.setData(self.model.index(rowno, 1), '', Qt.ItemDataRole.EditRole)
        for testno in range(len(self.tests)):
            self.model.setData(self.model.index(rowno, testno+2),
                               Qt.CheckState.Unchecked, role=Qt.ItemDataRole.CheckStateRole)
            item = self.model.itemFromIndex(self.model.index(rowno, testno+2))
            item.setEditable(False)
            item.setCheckable(True)
        self.model.endInsertRows()
        self.parent.lbl_nimgs.setText(f'{self.model.rowCount()}')

    def delete_row(self):
        """Delete selected row."""
        sel = self.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            if self.model.rowCount() == 1:
                self.model.setData(
                    self.model.index(0, 0), '', Qt.ItemDataRole.EditRole)
                self.model.setData(
                    self.model.index(0, 1), '', Qt.ItemDataRole.EditRole)
                for testno in range(len(self.tests)):
                    self.model.setData(self.model.index(0, testno+2),
                                       Qt.CheckState.Unchecked, role=Qt.ItemDataRole.CheckStateRole)
                    item = self.model.itemFromIndex(self.model.index(0, testno+2))
                    item.setEditable(False)
                    item.setCheckable(True)
            else:
                temp = self.get_data()
                temp.tests.pop(rowno)
                temp.image_names.pop(rowno)
                temp.group_names.pop(rowno)
                self.parent.current_template = copy.deepcopy(temp)
                self.update_data(set_selected=rowno-1)
            self.parent.lbl_nimgs.setText(f'{self.model.rowCount()}')

    def get_data(self):
        """Read current settings as edited by user.

        Return
        ------
        temp : QuickTestTemplate
        """
        temp = cfc.QuickTestTemplate()
        temp.label = self.parent.current_template.label
        tests = []
        image_names = []
        group_names = []
        for imgno in range(self.model.rowCount()):
            item = self.model.itemFromIndex(self.model.index(imgno, 0))
            image_names.append(item.text())
            item = self.model.itemFromIndex(self.model.index(imgno, 1))
            group_names.append(item.text())
            img_tests = []
            for testno, test in enumerate(self.tests):
                item = self.model.itemFromIndex(self.model.index(imgno, testno+2))
                if item.checkState() == Qt.CheckState.Checked:
                    img_tests.append(test)
            tests.append(img_tests)
        temp.tests = tests
        temp.image_names = image_names
        temp.group_names = group_names

        return temp


class QuickTestOutputTreeView(QTreeView):
    """QTreeWidget for list of output settings."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.update_model()
        self.setModel(self.model)

    def update_model(self):
        """Initialize model with headers."""
        self.model = QStandardItemModel(0, 6, self.parent)
        self.model.setHeaderData(0, Qt.Orientation.Horizontal, "Test")
        self.model.setHeaderData(1, Qt.Orientation.Horizontal, "Alternative")
        self.model.setHeaderData(2, Qt.Orientation.Horizontal, "Columns")
        self.model.setHeaderData(3, Qt.Orientation.Horizontal, "Calculation")
        self.model.setHeaderData(4, Qt.Orientation.Horizontal, "Pr image or group")
        self.model.setHeaderData(5, Qt.Orientation.Horizontal, "Header_")

        self.model.itemChanged.connect(self.parent.flag_edit)

    def update_data(self, set_selected=0, set_selected_txt=''):
        """Set data to self.parent.current_template.output.

        Parameters
        ----------
        set_selected : int
            Row number to set as selected when finished. Default is 0
        """
        self.model.beginResetModel()
        self.model.blockSignals(True)

        n_rows = self.model.rowCount()
        for i in range(n_rows):
            self.model.removeRow(n_rows-i-1, QModelIndex())

        temp = self.parent.current_template.output
        row = 0
        if temp.tests != {}:
            for testcode, sett in temp.tests.items():
                for sub in sett:
                    self.model.insertRow(row)
                    self.model.setData(self.model.index(row, 0), testcode)
                    try:
                        text_alt = ALTERNATIVES[
                            self.parent.current_modality][
                                testcode][sub.alternative]
                    except IndexError:
                        # supplement table starting from 10
                        text_alt = ALTERNATIVES[
                            self.parent.current_modality][
                                testcode][sub.alternative - 10] + '(Sup. table)'
                    except KeyError:
                        text_alt = '-'
                        if sub.alternative >= 10:
                            text_alt = '(Sup. table)'
                    self.model.setData(self.model.index(row, 1), text_alt)
                    if len(sub.columns) > 0:
                        text_col = str(sub.columns)
                    else:
                        text_col = 'all'
                    self.model.setData(self.model.index(row, 2), text_col)
                    self.model.setData(self.model.index(row, 3), sub.calculation)
                    text_pr = 'Per group' if sub.per_group else 'Per image'
                    self.model.setData(self.model.index(row, 4), text_pr)
                    self.model.setData(self.model.index(row, 5), sub.label)
                    row += 1

        self.setColumnWidth(0, 70)
        self.setColumnWidth(3, 110)

        self.model.blockSignals(False)
        self.model.endResetModel()
        if set_selected_txt == '':
            self.setCurrentIndex(self.model.index(set_selected, 0))
        else:
            new_row_values = self.get_model_as_text_list()
            new_row_select = new_row_values.index(set_selected_txt)
            self.setCurrentIndex(self.model.index(new_row_select, 0))

    def get_model_as_text_list(self):
        """Return all rows as txt.

        Returns
        -------
        row_txts : list of str
        """
        row_txts = []
        for row in range(self.model.rowCount()):
            values_row = []
            for col in range(self.model.columnCount()):
                item = self.model.item(row, col)
                values_row.append(item.text())
            row_txts.append(' '.join(values_row))
        return row_txts

    def get_testcode_subno(self, rowno):
        """Get test_code and QuickTestOutputSub number from row number.

        Parameters
        ----------
        rowno : int
            row number in treeview

        Returns
        -------
        test_code : str
        subno : int
        """
        test_codes = []
        subnos = []
        for test_code, subs in self.parent.current_template.output.tests.items():
            subno = 0
            for _ in subs:
                test_codes.append(test_code)
                subnos.append(subno)
                subno += 1

        test_code = test_codes[rowno]
        subno = subnos[rowno]

        return (test_code, subno)

    def edit_row(self):
        """Edit selected row."""
        sel = self.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            code, subno = self.get_testcode_subno(rowno)
            dlg_sub = QuickTestOutputSubDialog(
                self.parent.current_template,
                qt_output_sub=self.parent.current_template.output.tests[code][subno],
                modality=self.parent.current_modality,
                initial_testcode=code)
            res = dlg_sub.exec()
            if res:
                new_sub = copy.deepcopy(dlg_sub.get_data())
                if new_sub is None:
                    QMessageBox.warning(
                        self.parent, 'Ignored',
                        'No table columns selected. Edit ignored.')
                else:
                    self.parent.current_template.output.tests[
                        code][subno] = new_sub
                    self.update_data()
                    self.parent.flag_edit(True)

    def insert_row(self):
        """Insert row after selected if same test_code, else end of same testcode."""
        sel = self.selectedIndexes()
        code = ''
        subno = -1
        if len(sel) > 0:
            rowno = sel[0].row()
            code, subno = self.get_testcode_subno(rowno)
        dlg_sub = QuickTestOutputSubDialog(
            self.parent.current_template,
            modality=self.parent.current_modality)
        res = dlg_sub.exec()
        if res:
            testcode = dlg_sub.get_testcode()
            new_sub = copy.deepcopy(dlg_sub.get_data())
            if new_sub is None:
                QMessageBox.warning(
                    self.parent, 'Ignored',
                    'No table columns selected. Ignored input.')
            else:
                if testcode == code:
                    self.parent.current_template.output.tests[
                        testcode].insert(subno + 1, new_sub)
                else:
                    try:
                        self.parent.current_template.output.tests[
                            testcode].append(new_sub)
                    except KeyError:
                        self.parent.current_template.output.tests[
                            testcode] = [new_sub]
                self.update_data()
                self.parent.flag_edit(True)

    def delete_row(self):
        """Delete selected row."""
        sel = self.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            code, subno = self.get_testcode_subno(rowno)
            self.parent.current_template.output.tests[code].pop(subno)
            self.update_data(set_selected=rowno - 1)
            self.parent.flag_edit(True)

    def move_sub(self, move_up=True):
        """If first in test, move test, else move sub within test."""
        def move_test(testcode):
            keys = [*self.parent.current_template.output.tests]
            testno = keys.index(testcode)
            popped = keys.pop(testno)
            if move_up:
                keys.insert(testno - 1, popped)
            else:
                keys.insert(testno + 1, popped)
            new_tests = {}
            for key in keys:
                new_tests[key] = self.parent.current_template.output.tests[key]
            self.parent.current_template.output.tests = new_tests

        sel = self.selectedIndexes()
        if len(sel) > 0:
            orig_row_values = self.get_model_as_text_list()
            rowno = sel[0].row()
            code, subno = self.get_testcode_subno(rowno)

            if move_up:
                if subno == 0:  # move test if not first
                    if rowno > 0:
                        move_test(code)
                else:
                    popped = self.parent.current_template.output.tests[code].pop(subno)
                    self.parent.current_template.output.tests[code].insert(
                        subno-1, popped)
            else:
                n_sub_this = len(self.parent.current_template.output.tests[code])
                if subno == n_sub_this - 1:  # last in current
                    if rowno < self.model.rowCount() - n_sub_this:  # not last test
                        move_test(code)
                else:
                    popped = self.parent.current_template.output.tests[code].pop(subno)
                    self.parent.current_template.output.tests[code].insert(
                        subno+1, popped)

            self.update_data(set_selected_txt=orig_row_values[rowno])
            self.parent.flag_edit(True)


class QuickTestOutputSubDialog(ImageQCDialog):
    """Dialog to set QuickTestOutputSub."""

    def __init__(self, paramset, qt_output_sub=None, modality='CT',
                 initial_testcode=''):
        """Initialize QuickTestOutputSubDialog.

        Parameters
        ----------
        paramset : object
            Paramset<mod> as defined in config_classes.py
        qt_output_sub : object, optional
            input QuickTestOutputSub. The default is None.
        modality : str, optional
            current modality from parent window. The default is 'CT'.
        initial_testcode : str, optional
            testcode selected from start (if edit existing sub)
        """
        super().__init__()
        self.setWindowTitle('QuickTestOutput details')

        if qt_output_sub is None:
            qt_output_sub = cfc.QuickTestOutputSub()
        self.qt_output_sub = qt_output_sub
        self.paramset = paramset
        self.modality = modality

        self.cbox_testcode = QComboBox()
        self.cbox_alternatives = QComboBox()
        self.cbox_table = QComboBox()
        self.cbox_table.addItems(['Result table', 'Supplement_table'])
        self.list_columns = QListWidget()
        self.cbox_calculation = QComboBox()
        self.cbox_calculation.currentIndexChanged.connect(
            self.update_suggested_header)
        self.chk_per_group = uir.BoolSelect(
            self, text_true='per group', text_false='per image')
        self.txt_header = QLineEdit('')

        self.cbox_testcode.addItems(QUICKTEST_OPTIONS[modality])
        if initial_testcode != '':
            self.cbox_testcode.setCurrentText(initial_testcode)
            self.cbox_testcode.setEnabled(False)
        else:
            self.cbox_testcode.setCurrentIndex(0)

        self.cbox_testcode.currentIndexChanged.connect(
            lambda: self.update_data(update_calculations=False))
        self.cbox_alternatives.currentIndexChanged.connect(
            lambda: self.update_data(
                update_alternatives=False, update_calculations=False))
        self.cbox_table.currentIndexChanged.connect(
            lambda: self.update_data(
                update_alternatives=False, update_calculations=False))

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        self.suplement_txt = 'Supplement table'

        flo = QFormLayout()
        flo.addRow(uir.LabelHeader('Test:', 4), self.cbox_testcode)
        flo.addRow(uir.LabelHeader('Alternative:', 4), self.cbox_alternatives)
        flo.addRow(uir.LabelHeader('Table:', 4), self.cbox_table)
        flo.addRow(uir.LabelHeader('Columns:', 4), self.list_columns)
        flo.addRow(uir.LabelHeader('Calculation:', 4), self.cbox_calculation)
        flo.addRow(uir.LabelItalic(
            'Calculation method ignored if any of the values are strings'))
        flo.addRow(QLabel(''), self.chk_per_group)
        flo.addRow(uir.LabelItalic(
            'Per image/group = per row where results already pr all selected '
            'images'))
        flo.addRow(QLabel(''))
        flo.addRow(uir.LabelHeader('Header:', 4),
                   self.txt_header)
        flo.addRow(uir.LabelItalic(
            'Default header is column title.'))
        flo.addRow(uir.LabelItalic(
            'Ignored if more than one output parameter per image or group.'))
        flo.addRow(uir.LabelItalic(
            'Used as: header_imagelabel or header_grouplabel'))

        vlo.addLayout(flo)

        buttons = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

        self.update_data(first=True)

    def update_data(self, update_alternatives=True, update_columns=True,
                    update_calculations=True, first=False):
        """Set visuals to input data and refresh lists if selections change.

        Parameters
        ----------
        update_alternatives : bool, optional
            Update list of alternatives. The default is True.
        update_columns : bool, optional
            Update list of columnheaders. The default is True.
        update_calculations : bool, optional
            Update list of calculation options. The default is True.
        first : bool, optional
            First time update = when dialog box pops up. The default is False.
        """
        testcode = self.cbox_testcode.currentText()

        if update_alternatives:  # fill text in alternatives cbox, set default if locked
            try:
                if testcode == 'SNI':
                    alts = [
                        '6 small ROI',
                        '6 small ROI low/high channel',
                        'grid/Siemens',
                        'grid/Siemens low/high channel']
                else:
                    alts = ALTERNATIVES[self.modality][testcode]
            except KeyError:
                alts = ['-']

            self.cbox_alternatives.blockSignals(True)
            self.cbox_alternatives.clear()
            self.cbox_alternatives.addItems(alts)
            if self.qt_output_sub.alternative < 9:
                self.cbox_alternatives.setCurrentIndex(
                    self.qt_output_sub.alternative)
            else:  # supplement table
                self.cbox_alternatives.setCurrentIndex(
                    self.qt_output_sub.alternative - 10)
            if self.cbox_alternatives.isEnabled() is False:
                # add mode, set to default according to parameters in set
                alt = cff.get_test_alternative(self.paramset, testcode)
                if alt is not None:
                    self.cbox_alternatives.setCurrentIndex(alt)
            self.cbox_alternatives.blockSignals(False)
        if update_columns:  # fill text in columns
            cols = []
            idx_alt = self.cbox_alternatives.currentIndex()
            if self.cbox_table.currentIndex() == 1:
                if testcode in HEADERS_SUP[self.modality]:
                    if 'altAll' in HEADERS_SUP[self.modality][testcode]:
                        cols = HEADERS_SUP[self.modality][testcode]['altAll']
                    elif 'alt0' in HEADERS_SUP[self.modality][testcode]:
                        try:
                            cols = HEADERS_SUP[
                                self.modality][testcode]['alt'+str(idx_alt)]
                        except KeyError:
                            if testcode == 'ROI':
                                cols = self.paramset.roi_table.labels
            else:
                try:
                    cols = HEADERS[self.modality][testcode]['alt'+str(idx_alt)]
                except KeyError:
                    if testcode == 'DCM':
                        cols = self.paramset.dcm_tagpattern.list_tags
                    elif testcode == 'CTn':
                        cols = self.paramset.ctn_table.labels
                    elif testcode == 'Num':
                        cols = self.paramset.num_table.labels
                    elif testcode == 'ROI':  # alt > 0, table
                        cols = self.paramset.roi_table.labels
                    elif 'altAll' in HEADERS[self.modality][testcode]:
                        cols = HEADERS[self.modality][testcode]['altAll']
            self.list_columns.clear()
            if len(cols) > 0:
                self.list_columns.addItems(cols)
                # set checkable
                subcols = self.qt_output_sub.columns
                if len(subcols) == 0:
                    subcols = [i for i in range(self.list_columns.count())]
                for i in range(self.list_columns.count()):
                    item = self.list_columns.item(i)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    if first:
                        if i in subcols:
                            item.setCheckState(Qt.CheckState.Checked)
                        else:
                            item.setCheckState(Qt.CheckState.Unchecked)
                    else:
                        item.setCheckState(Qt.CheckState.Checked)
        if update_calculations:  # fill and set default calculation option
            self.cbox_calculation.addItems(CALCULATION_OPTIONS)
            self.cbox_calculation.setCurrentText(
                self.qt_output_sub.calculation)
        if first:
            self.chk_per_group.setChecked(self.qt_output_sub.per_group)
            self.txt_header.setText(self.qt_output_sub.label)
            if self.cbox_testcode.isEnabled():  # add, not edit mode - alternatives lock
                self.cbox_alternatives.setEnabled(False)

    def update_suggested_header(self):
        """Suggest header text."""
        calc_opt_text = self.cbox_calculation.currentText()
        if calc_opt_text == '=':
            self.txt_header.setText('')
        else:
            self.txt_header.setText(
                f'{self.get_testcode()}_{calc_opt_text}')

    def get_testcode(self):
        """Get selected testcode.

        Return
        ------
        testcode: str
        """
        return self.cbox_testcode.currentText()

    def get_data(self):
        """Get settings from dialog as QuickTestOutputSub.

        Returns
        -------
        qtsub : QuickTestOutputSub
        """
        qtsub = cfc.QuickTestOutputSub()
        qtsub.label = self.txt_header.text()

        if self.cbox_table.currentIndex() == 1:
            qtsub.alternative = self.cbox_alternatives.currentIndex() + 10
        else:
            qtsub.alternative = self.cbox_alternatives.currentIndex()
        cols = []
        for i in range(self.list_columns.count()):
            if self.list_columns.item(i).checkState() == Qt.CheckState.Checked:
                cols.append(i)
        if len(cols) == self.list_columns.count():
            cols = []  # == all
        qtsub.columns = cols
        qtsub.calculation = self.cbox_calculation.currentText()
        if qtsub.calculation == '=' and self.chk_per_group.isChecked():
            qtsub.per_group = False
            QMessageBox.warning(
                self, 'Warning',
                ('The "per group" setting require some calculation to '
                 'extract one value from many. The calculation was set '
                 'to "=" which indicate that all values should be printed '
                 'as is. The "per group" setting was changed to "per image".')
                )
        else:
            qtsub.per_group = self.chk_per_group.isChecked()

        return qtsub

class ResultImageDefaultsTreeView(QTreeView):
    """QTreeWidget for list of override result image display settings."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.update_model()
        self.setModel(self.model)

    def update_model(self):
        """Initialize model with headers."""
        self.model = QStandardItemModel(0, 7, self.parent)
        self.model.setHeaderData(0, Qt.Orientation.Horizontal, "Test")
        self.model.setHeaderData(1, Qt.Orientation.Horizontal, "Selected text")
        self.model.setHeaderData(2, Qt.Orientation.Horizontal, "Set min")
        self.model.setHeaderData(3, Qt.Orientation.Horizontal, "Min")
        self.model.setHeaderData(4, Qt.Orientation.Horizontal, "Set max")
        self.model.setHeaderData(5, Qt.Orientation.Horizontal, "Max")
        self.model.setHeaderData(6, Qt.Orientation.Horizontal, "Colormap")

        self.model.itemChanged.connect(self.parent.flag_edit)

    def update_data(self, set_selected=0):
        """Set data to self.parent.current_template.output.

        Parameters
        ----------
        set_selected : int
            Row number to set as selected when finished. Default is 0
        """
        self.model.beginResetModel()
        self.model.blockSignals(True)

        n_rows = self.model.rowCount()
        for i in range(n_rows):
            self.model.removeRow(n_rows-i-1, QModelIndex())

        temp = self.parent.current_template.result_image_defaults
        row = 0
        if len(temp) > 0:
            for sub in temp:
                self.model.insertRow(row)
                self.model.setData(self.model.index(row, 0), sub.test)
                self.model.setData(self.model.index(row, 1), sub.selected_text)
                self.model.setData(self.model.index(row, 2), sub.set_min)
                self.model.setData(self.model.index(row, 3), sub.cmin)
                self.model.setData(self.model.index(row, 4), sub.set_max)
                self.model.setData(self.model.index(row, 5), sub.cmax)
                self.model.setData(self.model.index(row, 6), sub.cmap)
                row += 1

        self.setColumnWidth(0, 70)
        self.setColumnWidth(1, 200)
        self.setColumnWidth(6, 200)

        self.model.blockSignals(False)
        self.model.endResetModel()
        if set_selected:
            self.setCurrentIndex(self.model.index(set_selected, 0))

    def run_subdialog(self, subno, default_sub=None):
        main = self.parent.dlg_settings.main
        if main.current_modality == self.parent.current_modality:
            dlg_sub = ResultImageDefaultDialog(
                self.parent.current_template, default_sub=default_sub,
                modality=self.parent.current_modality, main=main)
            res = dlg_sub.exec()
            if res:
                new_sub = dlg_sub.get_data()
                def_list = self.parent.current_template.result_image_defaults
                if default_sub is None:
                    if subno == -1:
                        def_list.append(new_sub)
                    else:
                        def_list.insert(subno + 1, new_sub)
                else:
                    def_list[subno] = new_sub
                self.update_data(set_selected=subno)
                self.parent.flag_edit(True)
        else:
            QMessageBox.warning(self, 'Warning',
                                'Changes only possible when same modality '
                                'active in main window.')

    def insert_row(self):
        """Insert row after selected if any, else at end."""
        sel = self.selectedIndexes()
        subno = -1
        if len(sel) > 0:
            subno = sel[0].row()
        self.run_subdialog(subno, None)

    def edit_row(self):
        """Edit selected row."""
        sel = self.selectedIndexes()
        if len(sel) > 0:
            subno = sel[0].row()
            self.run_subdialog(
                subno,
                self.parent.current_template.result_image_defaults[subno])

    def delete_row(self):
        """Delete selected row."""
        sel = self.selectedIndexes()
        if len(sel) > 0:
            subno = sel[0].row()
            self.parent.current_template.result_image_defaults.pop(subno)
            self.update_data(set_selected=subno - 1)
            self.parent.flag_edit(True)

    def move_sub(self, move_up=True):
        """Move sub up or down."""
        sel = self.selectedIndexes()
        if len(sel) > 0:
            subno = sel[0].row()
            new_idx = subno

            move = False
            if move_up and subno > 0:
                move = True
                new_idx = subno - 1
            elif move_up is False:
                if subno < len(self.parent.current_template.result_image_defaults):
                    move = True
                    new_idx = subno + 1

            if move:
                popped = self.parent.current_template.result_image_defaults.pop(
                    subno)
                self.parent.current_template.result_image_defaults.insert(
                    new_idx, popped)

            self.update_data(set_selected=new_idx)
            self.parent.flag_edit(True)


class ResultImageDefaultDialog(ImageQCDialog):
    """Dialog to set QuickTestOutputSub."""

    def __init__(self, paramset, default_sub=None, modality='CT', main=None):
        """Initialize QuickTestOutputSubDialog.

        Parameters
        ----------
        paramset : object
            Paramset<mod> as defined in config_classes.py
        default_sub : object, optional
            input ResultImageDefaultSub. The default is None.
        modality : str, optional
            current modality from parent window. The default is 'CT'.
        initial_testcode : str, optional
            testcode selected from start (if edit existing sub)
        main : MainWindow
        """
        super().__init__()
        self.setWindowTitle('Override image display defaults')

        if default_sub is None:
            default_sub = cfc.ResultImageDefaultSub()
        self.default_sub = default_sub
        self.paramset = paramset
        self.modality = modality
        self.main_window = main

        self.cbox_testcode = QComboBox()
        self.cbox_selected_text = QComboBox()
        self.cmap = QLineEdit()
        self.chk_min = QCheckBox()
        self.chk_min.stateChanged.connect(self.update_chk_min)
        self.cmin = QDoubleSpinBox(
            decimals=2, minimum=-1000000, maximum=1000000, singleStep=1.)
        self.chk_max = QCheckBox()
        self.chk_max.stateChanged.connect(self.update_chk_max)
        self.cmax = QDoubleSpinBox(
            decimals=2, minimum=-1000000, maximum=1000000, singleStep=1.)

        self.cbox_testcode.addItems(QUICKTEST_OPTIONS[modality][3:])
        initial_testcode = self.default_sub.test
        if initial_testcode != '':
            self.cbox_testcode.setCurrentText(initial_testcode)
            self.cbox_testcode.setEnabled(False)
        else:
            self.cbox_testcode.setCurrentIndex(0)

        self.cbox_testcode.currentIndexChanged.connect(
            self.update_select_options)

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        flo = QFormLayout()
        flo.addRow(QLabel('Test:'), self.cbox_testcode)
        flo.addRow(QLabel('Selected text:'), self.cbox_selected_text)
        flo.addRow(QLabel('Set minimum:'), self.chk_min)
        flo.addRow(QLabel('Minimum:'), self.cmin)
        flo.addRow(QLabel('Set maximum:'), self.chk_max)
        flo.addRow(QLabel('Maximum:'), self.cmax)
        vlo.addLayout(flo)
        hlo_cmap = QHBoxLayout()
        hlo_cmap.addWidget(QLabel('Colormap:'))
        hlo_cmap.addWidget(self.cmap)
        btn_cmap = QPushButton('Select ...')
        btn_cmap.clicked.connect(self.select_cmap)
        hlo_cmap.addWidget(btn_cmap)
        vlo.addLayout(hlo_cmap)

        buttons = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.verify)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

        self.init = True
        self.update_data()
        self.init = False

    def update_chk_min(self):
        if self.chk_min.isChecked():
            self.cmin.setEnabled(True)
        else:
            self.cmin.setEnabled(False)

    def update_chk_max(self):
        if self.chk_max.isChecked():
            self.cmax.setEnabled(True)
        else:
            self.cmax.setEnabled(False)

    def update_select_options(self):
        self.cbox_selected_text.clear()
        testcode = self.cbox_testcode.currentText().lower()
        widget_params = self.main_window.stack_test_tabs.currentWidget()
        attrib = f'{testcode}_result_image'
        if testcode == 'hom':
            try:
                if self.main_window.current_paramset.hom_type == 4:
                    attrib = attrib + '_aapm'
            except AttributeError:
                pass
        widget_options = getattr(widget_params, attrib, None)
        if widget_options is not None:
            options = [
                widget_options.itemText(i) for i
                in range(widget_options.count())]
            self.cbox_selected_text.addItems(options)
            preset_text = '' if self.init is False else self.default_sub.selected_text
            if preset_text in options:
                self.cbox_selected_text.setCurrentText(preset_text)

    def update_data(self):
        """Set visuals to input data and refresh lists if selections change."""
        if self.init is True:
            self.cbox_testcode.setCurrentText(self.default_sub.test)
            self.update_select_options()
        self.cmap.setText(self.default_sub.cmap)
        self.chk_min.setChecked(self.default_sub.set_min)
        self.chk_max.setChecked(self.default_sub.set_max)
        self.cmin.setValue(self.default_sub.cmin)
        self.cmax.setValue(self.default_sub.cmax)
        self.update_chk_min()
        self.update_chk_max()

    def select_cmap(self):
        dlg = CmapSelectDialog(self)
        res = dlg.exec()
        if res:
            cmap = dlg.get_cmap()
            self.cmap.setText(cmap)

    def verify(self):
        verified = True
        if self.chk_min.isChecked() and self.chk_max.isChecked():
            if self.cmin.value() >= self.cmax.value():
                verified = False
                QMessageBox.warning(self, 'Warning',
                                    'Minimum have to be smaller than maximum.')
        if verified:
            self.accept()

    def get_data(self):
        """Get settings from dialog as QuickTestOutputSub.

        Returns
        -------
        default_sub : ResultImageDefaultSub
        """
        default_sub = cfc.ResultImageDefaultSub()
        default_sub.test = self.cbox_testcode.currentText()
        default_sub.selected_text = self.cbox_selected_text.currentText()
        default_sub.cmap = self.cmap.text()
        default_sub.set_min = self.chk_min.isChecked()
        default_sub.set_max = self.chk_max.isChecked()
        default_sub.cmin = self.cmin.value()
        default_sub.cmax = self.cmax.value()

        return default_sub

class DicomCritAddDialog(ImageQCDialog):
    """Dialog to add dicom criteria for automation."""

    def __init__(self, parent, attr_name='', value=''):
        super().__init__()
        self.parent = parent
        if attr_name == '':
            self.setWindowTitle('Add DICOM criteria')
        else:
            self.setWindowTitle('Edit DICOM criteria')

        self.cbox_tags = QComboBox()
        self.txt_value = QLineEdit('')

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        hlo_tags = QHBoxLayout()
        vlo.addLayout(hlo_tags)
        hlo_tags.addWidget(QLabel('Attribute name: '))
        hlo_tags.addWidget(self.cbox_tags)
        _, included_tags = get_included_tags(
            self.parent.parent.current_modality,
            self.parent.parent.tag_infos,
            avoid_special_tags=True)
        self.cbox_tags.addItems(included_tags)
        if attr_name != '':
            self.cbox_tags.setCurrentText(attr_name)
            self.txt_value.setText(value)

        hlo_values = QHBoxLayout()
        vlo.addLayout(hlo_values)
        hlo_values.addWidget(QLabel('Value string'))
        hlo_values.addWidget(self.txt_value)
        self.txt_value.setMinimumWidth(200)

        vlo.addWidget(QLabel('Leave value empty to get value from sample DICOM file'))
        vlo.addWidget(QLabel(
            'Wildcard possible: ? single character / * multiple characters'))

        buttons = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)

    def get_data(self):
        """Get attribute name and value.

        Returns
        -------
        attributename : str
        value : str

        """
        return (self.cbox_tags.currentText(), self.txt_value.text())


class DicomCritWidget(QWidget):
    """Widget for dicom_crit in automation templates + toolbar to edit."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.hlo = QHBoxLayout()
        self.setLayout(self.hlo)

        self.table_crit = QTreeWidget()
        self.table_crit.setColumnCount(2)
        self.table_crit.setColumnWidth(0, 200)
        self.table_crit.setColumnWidth(1, 150)
        self.table_crit.setHeaderLabels(['Attribute name', 'Value'])
        self.table_crit.setMinimumSize(350, 200)
        self.table_crit.setRootIsDecorated(False)
        self.hlo.addWidget(self.table_crit)

        toolb = QToolBar()
        toolb.setOrientation(Qt.Orientation.Vertical)
        act_add = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add criterion', self)
        act_add.triggered.connect(self.add)
        act_edit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit criterion', self)
        act_edit.triggered.connect(self.edit)
        act_delete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete selected criterion/row', self)
        act_delete.triggered.connect(self.delete)
        act_clear = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}clear.png'),
            'Clear table', self)
        act_clear.triggered.connect(self.clear)
        toolb.addActions([act_add, act_edit, act_delete, act_clear])
        self.hlo.addWidget(toolb)

    def add(self):
        """Add new criterion."""
        dlg = DicomCritAddDialog(self)
        res = dlg.exec()
        if res:
            attr_name, value = dlg.get_data()
            proceed = True
            if attr_name in self.parent.current_template.dicom_crit_attributenames:
                res = messageboxes.QuestionBox(
                    self.parent, title='Replace?',
                    msg='Attribute name already in table. Replace?')
                res.exec()
                if res.clickedButton() == res.no:
                    proceed = False
            if proceed:
                self.parent.current_template.dicom_crit_attributenames.append(
                    attr_name)
                self.parent.current_template.dicom_crit_values.append(value)
                self.update_data()
                self.parent.flag_edit()
                self.parent.get_sample_file_data()

    def edit(self):
        """Edit DICOM criterion."""
        sel = self.table_crit.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            attr = self.parent.current_template.dicom_crit_attributenames[
                rowno]
            val = self.parent.current_template.dicom_crit_values[rowno]
            dlg = DicomCritAddDialog(
                self, attr_name=attr, value=val)
            res = dlg.exec()
            if res:
                attr_name, value = dlg.get_data()
                already_other = copy.deepcopy(
                    self.parent.current_template.dicom_crit_attributenames)
                already_other.pop(rowno)
                if attr_name in already_other:
                    QMessageBox.warning(
                        self.parent, 'Ignored',
                        'Attribute name already in table. Edit ignored.')
                else:
                    self.parent.current_template.dicom_crit_attributenames[
                        rowno] = attr_name
                    self.parent.current_template.dicom_crit_values[rowno] = value
                    self.update_data()
                    self.parent.flag_edit()

    def delete(self):
        """Delete selected criterion."""
        sels = self.table_crit.selectedIndexes()
        if len(sels) > 0:
            sel_rows = [sel.row() for sel in sels]
            if len(sel_rows) > 0:
                sel_rows = list(set(sel_rows))  # remove duplicates
                if len(sel_rows) == len(
                        self.parent.current_template.dicom_crit_attributenames):
                    self.clear()
                else:
                    sel_rows.sort(reverse=True)
                    for row in sel_rows:
                        self.parent.current_template.dicom_crit_attributenames.pop(row)
                        self.parent.current_template.dicom_crit_values.pop(row)
                    self.update_data()
                    self.parent.flag_edit()

    def clear(self):
        """Clear all criteria."""
        self.parent.current_template.dicom_crit_attributenames = []
        self.parent.current_template.dicom_crit_values = []
        self.update_data()
        self.parent.flag_edit()

    def update_data(self, set_selected=0):
        """Update table_crit with data from current_template."""
        self.table_crit.clear()
        if len(self.parent.current_template.dicom_crit_attributenames) > 0:
            attr_names = self.parent.current_template.dicom_crit_attributenames
            vals = self.parent.current_template.dicom_crit_values
            for rowno, attr_name in enumerate(attr_names):
                row_strings = [attr_name, vals[rowno]]
                item = QTreeWidgetItem(row_strings)
                self.table_crit.addTopLevelItem(item)

            self.table_crit.setCurrentItem(
                self.table_crit.topLevelItem(set_selected))
