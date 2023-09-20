#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface for dialog Rename Dicom.

@author: Ellen Wasbo
"""
import os
from pathlib import Path
import copy
from time import time

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QWizard, QWizardPage,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QButtonGroup,
    QLabel, QCheckBox, QLineEdit, QPushButton, QRadioButton, QAction, QComboBox,
    QMessageBox, QFileDialog, QDialogButtonBox
    )

# imageQC block start
from imageQC.config.iQCconstants import ENV_ICON_PATH
import imageQC.config.config_func as cff
from imageQC.scripts.mini_methods_format import valid_template_name
from imageQC.config import config_classes as cfc
from imageQC.scripts import dcm
from imageQC.ui.reusable_widgets import (
    LabelMultiline, LabelItalic, ToolBarBrowse, ListWidgetCheckable)
from imageQC.ui import messageboxes
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.scripts.mini_methods import create_empty_file
# imageQC block end


class StartupPage(QWizardPage):
    """Page with list of tasks to fulfill to proceed."""

    def __init__(self, main, wizard, ok_config=False, ok_img=False, ok_paramset=False,
                 ok_quicktest=False, ok_results=False):
        super().__init__()
        self.main = main
        self.wizard = wizard

        self.setTitle('A walk-through to set automation templates')
        self.setSubTitle(
            ('Follow these steps to create an automation template'
             'based on the currently loaded images and settings.'))
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        gb_config = OkGroupBox(
            'Set config folder', ok=ok_config,
            txts=['Set the configuration folder for saving configuration settings.']
            )
        vlo.addWidget(gb_config)

        gb_img = OkGroupBox(
            'Load images', ok=ok_img,
            txts=[
                ('Acquire a set of representative images and save these to the path '
                 'where you want future new images to appear for this template.'),
                ('Load these images to the main window of imageQC before running this '
                 'wizard.'),
                ('Optionally sort the images by dicom header info (A-Z button) to '
                 'ensure same image order each time.')
                ]
            )
        vlo.addWidget(gb_img)

        txts = ['Set and save parameters on how to perform the tests.']
        if ok_paramset is False:
            txts.append('Unsaved changes detected.')
        gb_params = OkGroupBox(
            'Set test parameters', ok=ok_paramset, txts=txts
            )
        vlo.addWidget(gb_params)

        gb_qt = OkGroupBox(
            'Set QuickTest template', ok=ok_quicktest,
            txts=[
                ('Set which test to perform on which image i.e. activate QuickTest, '
                 'select an existing QuickTest template or create a new template by'),
                ('selecting images in the file list and right-click to set which tests '
                 'to perform. Save and name the new QuickTest template.')
                ]
            )
        vlo.addWidget(gb_qt)

        self.gb_res = OkGroupBox(
            'Optionally Run the QuickTest template', ok=ok_results,
            txts=[
                'Run the QuickTest template to get results and make possible '
                'output testing.'
                ]
            )
        if ok_quicktest and not ok_results:
            btn_run_quicktest = QPushButton('Run QuickTest')
            btn_run_quicktest.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}play.png'))
            btn_run_quicktest.clicked.connect(self.run_quicktest)
            self.gb_res.vlo.addWidget(btn_run_quicktest)
        self.btn_test_output = QPushButton('Test output')
        self.btn_test_output.setEnabled(ok_results)
        self.btn_test_output.clicked.connect(self.test_output)
        self.gb_res.vlo.addWidget(self.btn_test_output)
        self.btn_settings_output = QPushButton('Configure and test output')
        self.btn_settings_output.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}gears.png'))
        self.btn_settings_output.setEnabled(ok_results)
        self.btn_settings_output.clicked.connect(self.settings_output)
        self.gb_res.vlo.addWidget(self.btn_settings_output)
        vlo.addWidget(self.gb_res)

        if all([ok_img, ok_quicktest, ok_paramset]):
            vlo.addWidget(QLabel(
                'Continue to next step to verify and save template.'))
        else:
            vlo.addWidget(QLabel((
                'Cancel and come back to the wizard when all mandatory steps are '
                'fulfilled.')))

    def run_quicktest(self):
        """Run QuickTest and update flag."""
        self.main.wid_quicktest.run_quicktest()
        if self.main.results:
            self.btn_test_output.setEnabled(True)
            self.btn_settings_output.setEnabled(True)
            url_icon = f'{os.environ[ENV_ICON_PATH]}flag_ok.png'
            self.gb_res.setStyleSheet(
                f"""
                QGroupBox {{
                    margin-top: 10px;
                    }}
                QGroupBox::title {{
                    padding-top: 0px;
                    font-style: italic;
                    }}
                QGroupBox::indicator:checked {{
                    image: url({url_icon});
                    }}
                """
                )

    def test_output(self):
        """Display output with current settings."""
        test_output = copy.deepcopy(self.main.current_paramset.output)
        curr_output = copy.deepcopy(self.main.current_paramset.output)
        if not all([test_output.include_header, test_output.transpose_table]):
            QMessageBox.warning(
                self, 'Warning',
                ('Exported parameters will be shown with header and as a column'
                 f' even though parameterset {self.main.current_paramset.label} '
                 'sets this differently. With header and as a column is easier for '
                 'verification.'))
        test_output.include_header = True
        test_output.transpose_table = True
        self.main.current_paramset.output = test_output
        self.main.wid_quicktest.extract_results(silent=True)
        self.main.current_paramset.output = curr_output
        self.main.display_clipboard(
            title='Export values of current data in main window')

    def settings_output(self):
        """Open settings dialog with current paramset in output tab."""
        self.wizard.reject()
        self.main.run_settings(
            initial_view='Parameter sets / output',
            paramset_output=True,
            initial_template_label=self.main.current_paramset.label)


class OkGroupBox(QGroupBox):
    """Groupbox with ok/missing flag as checked symbol."""

    def __init__(self, title, ok=True, txts=[]):
        super().__init__(title)
        self.setCheckable(True)
        self.vlo = QVBoxLayout()
        self.setLayout(self.vlo)
        self.vlo.addWidget(LabelMultiline(txts=txts))
        url_icon = (f'{os.environ[ENV_ICON_PATH]}flag_ok.png' if ok
                    else f'{os.environ[ENV_ICON_PATH]}flag_missing.png')
        self.setStyleSheet(
            f"""
            QGroupBox {{
                margin-top: 10px;
                }}
            QGroupBox::title {{
                padding-top: 0px;
                font-style: italic;
                }}
            QGroupBox::indicator:checked {{
                image: url({url_icon});
                }}
            """
            )


class SelectOverrideDialog(ImageQCDialog):
    """Dialog to select what to override for selected template."""

    def __init__(self, saved_template):
        super().__init__()
        self.setWindowTitle('Override/keep settings?')
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        txt = '''
        <p>The wizard will not change these settings of the selected template:</p>
        <ul><li>Dicom criteria</li></ul>
        <p>These settings will be overridden by current values (if saved):</p>
        <ul>
        <li>Sort pattern</li>
        <li>Parameter set label</li>
        <li>Quicktest template label</li>
        </ul>
        <p>Set archiving and activation manually.</p>
        <p>
        '''
        html_txt = f"""<html><head/><body>{txt}</body></html>"""

        vlo.addWidget(QLabel(html_txt))
        vlo.addWidget(QLabel('Keep these saved values from the selected template:'))

        texts = [
            f'Input path: {saved_template.path_input}',
            f'Output path: {saved_template.path_output}',
            f'Station name: {saved_template.station_name}',
            ]
        self.list_widget = ListWidgetCheckable(
            texts=texts,
            set_checked_ids=[0, 1, 2]
            )
        vlo.addWidget(self.list_widget)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vlo.addWidget(self.buttonBox)


class SavePage(QWizardPage):
    """Page to finish and save."""

    def __init__(self, main, templates=None, lastload=None):
        super().__init__()
        self.main = main
        self.templates = templates
        self.lastload = lastload
        self.saved_label = ''
        self.already_labels = []
        saved_temp = cfc.AutoTemplate()
        try:
            self.already_labels = \
                [obj.label for obj
                 in self.templates[self.main.current_modality]]
            if len(self.already_labels) > 0:
                if self.already_labels[0] == '':
                    self.already_labels = []
                if self.main.gui.current_auto_template != '':
                    if self.main.gui.current_auto_template in self.already_labels:
                        idx = self.already_labels.index(
                            self.main.gui.current_auto_template)
                        saved_temp = self.templates[self.main.current_modality][idx]
        except KeyError:
            pass
        self.already_labels.insert(0, '(create new)')

        if len(self.main.imgs) > 0:
            statname = dcm.get_tags(
                self.main.imgs[0].filepath,
                tag_patterns=[cfc.TagPatternFormat(list_tags=['StationName'])],
                tag_infos=self.main.tag_infos)
            self.statname = statname[0][0]
            path_input = str(Path(self.main.imgs[0].filepath).parent)
        else:
            self.statname = ''
            path_input = ''

        self.initial_temp = cfc.AutoTemplate(
            label=saved_temp.label,
            path_input=path_input,
            path_output=saved_temp.path_output,
            station_name=self.statname,
            paramset_label=self.main.current_paramset.label,
            quicktemp_label=self.main.wid_quicktest.cbox_template.currentText(),
            archive=saved_temp.archive,
            active=saved_temp.active
            )
        sort_pattern_txt = '-'
        if self.main.current_sort_pattern is not None:
            self.initial_temp.sort_pattern = self.main.current_sort_pattern
            tags = self.main.current_sort_pattern.list_tags
            asc_txt = self.main.current_sort_pattern.list_sort
            tags_txt = []
            for i, tag in enumerate(tags):
                asc_txt = ('(ASC)' if self.main.current_sort_pattern.list_sort[i]
                           else '(DESC)')
                tags_txt.append(tag + asc_txt)
            sort_pattern_txt = ', '.join(tags_txt)

        self.setTitle('Finish and save template')
        self.setSubTitle('''Verify the parameters.''')
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        flo = QFormLayout()
        vlo.addLayout(flo)

        self.txt_label = QLineEdit(self.initial_temp.label)
        self.registerField("label*", self.txt_label)  # next not available before given
        self.cbox_labels = QComboBox()
        self.cbox_labels.addItems(self.already_labels)
        if self.initial_temp.label in self.already_labels:
            self.cbox_labels.setCurrentText(self.initial_temp.label)
        else:
            self.cbox_labels.setCurrentIndex(0)
        self.cbox_labels.currentIndexChanged.connect(self.update_selected_temp)
        hlo_label = QHBoxLayout()
        hlo_label.addWidget(QLabel('Create new or override existing template:'))
        hlo_label.addWidget(self.cbox_labels)
        hlo_label.addWidget(QLabel('Template name:'))
        hlo_label.addWidget(self.txt_label)
        vlo.addLayout(hlo_label)

        flo.addRow(QLabel('Modality:'), QLabel(self.main.current_modality))
        self.lbl_statname = QLabel(self.statname)
        flo.addRow(QLabel('Station name:'), self.lbl_statname)

        hlo_path_input = QHBoxLayout()
        vlo.addLayout(hlo_path_input)
        hlo_path_input.addWidget(QLabel('Input path '))
        self.txt_path_input = QLineEdit(self.initial_temp.path_input)
        self.txt_path_input.setMinimumWidth(300)
        hlo_path_input.addWidget(self.txt_path_input)
        toolb = ToolBarBrowse('Browse to find path')
        toolb.act_browse.triggered.connect(
            lambda: self.locate_folder(self.txt_path_input))
        hlo_path_input.addWidget(toolb)

        hlo_path_output = QHBoxLayout()
        vlo.addLayout(hlo_path_output)
        hlo_path_output.addWidget(QLabel('Output path '))
        self.txt_path_output = QLineEdit(self.initial_temp.path_output)
        self.txt_path_output.setMinimumWidth(300)
        hlo_path_output.addWidget(self.txt_path_output)
        toolb = ToolBarBrowse('Browse to file')
        toolb.act_browse.triggered.connect(
            lambda: self.locate_file(
                self.txt_path_output, title='Locate output file',
                filter_str="Text file (*.txt)"))
        hlo_path_output.addWidget(toolb)
        act_new_file = QAction(QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                               'Create an empty file', toolb)
        act_new_file.triggered.connect(
            lambda: self.locate_file(
                self.txt_path_output, title='Create output file',
                filter_str="Text file (*.txt)", opensave=True))
        toolb.addActions([act_new_file])

        flo2 = QFormLayout()
        vlo.addLayout(flo2)
        flo2.addRow(QLabel('Use parameter set:'),
                    QLabel(self.initial_temp.paramset_label))
        flo2.addRow(QLabel('Use QuickTest template:'),
                    QLabel(self.initial_temp.quicktemp_label))
        flo2.addRow(QLabel('Sort images by:'), QLabel(sort_pattern_txt))

        self.chk_archive = QCheckBox()
        self.chk_archive.setChecked(self.initial_temp.archive)
        flo2.addRow(QLabel('Archive images when analysed:'), self.chk_archive)
        self.chk_active = QCheckBox()
        self.chk_active.setChecked(self.initial_temp.active)
        flo2.addRow(QLabel('Set template active:'), self.chk_active)
        vlo.setSpacing(20)

        btn_save = QPushButton('Save automation template')
        btn_save.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}save.png'))
        btn_save.clicked.connect(self.save_template)
        if self.main.save_blocked:
            btn_save.setEnabled(False)
        vlo.addWidget(btn_save)

        self.lbl_verif_label = LabelItalic('', color='red')
        vlo.addWidget(self.lbl_verif_label)
        self.current_template = copy.deepcopy(self.initial_temp)

    def update_selected_temp(self):
        """Ask to update settings according to selected template."""
        if self.cbox_labels.currentIndex() == 0:
            self.current_template = copy.deepcopy(self.initial_template)
        else:
            self.current_template.label = self.cbox_labels.currentText()
            self.txt_label.setText(self.current_template.label)
            idx = self.cbox_labels.currentIndex() - 1
            saved_temp = self.templates[self.main.current_modality][idx]
            dlg = SelectOverrideDialog(saved_temp)
            res = dlg.exec()
            proceed = True
            if res:
                checked_ids = dlg.list_widget.get_checked_ids()
            else:
                proceed = False
                self.cbox_labels.setCurrentIndex(0)
            if proceed:
                # 0 input, 1 output, 2 station name
                if 0 in checked_ids:
                    self.current_template.path_input = saved_temp.path_input
                if 1 in checked_ids:
                    self.current_template.path_output = saved_temp.path_output
                if 2 in checked_ids:
                    self.current_template.station_name = saved_temp.station_name
                self.current_template.dicom_crit_attributenames = (
                    saved_temp.dicom_crit_attributenames)
                self.current_template.dicom_crit_values = saved_temp.dicom_crit_values

        self.txt_label.setText(self.current_template.label)
        self.lbl_statname.setText(self.current_template.station_name)
        self.txt_path_input.setText(self.current_template.path_input)
        self.txt_path_output.setText(self.current_template.path_output)

    def locate_folder(self, widget):
        """Select folder from FileDialog."""
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if widget.text() != '':
            dlg.setDirectory(widget.text())
        if dlg.exec():
            fname = dlg.selectedFiles()
            widget.setText(os.path.normpath(fname[0]))
            self.current_template.path_input = fname[0]

    def locate_file(self, widget, title='Locate file',
                    filter_str='All files (*)', opensave=False):
        """Select file from FileDialog."""
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
            self.current_template.path_output = fname

    def verify_new_label(self):
        """Ensure valid template label."""
        text = self.txt_label.text()
        if text != '':
            text = valid_template_name(text)
            self.txt_label.setText(text)
        else:
            self.lbl_verif_label.setText(
                'Please define a label for the automation template.')
        ok_label = True if text != '' else False

        return ok_label

    def save_template(self):
        """Verify and save current template."""
        proceed = self.verify_new_label()
        if proceed:
            fname = 'auto_templates'
            self.current_template.label = self.txt_label.text()
            self.current_template.archive = self.chk_archive.isChecked()
            self.current_template.active = self.chk_active.isChecked()
            if self.current_template.label in self.already_labels:
                proceed = messageboxes.proceed_question(
                    self,
                    f'Continue to override template {self.current_template.label}?')
                if proceed:
                    idx = self.already_labels.index(self.current_template.label)
                    self.templates[self.main.current_modality][
                        idx - 1] = self.current_template
            else:
                self.templates[self.main.current_modality].append(self.current_template)
                proceed = True

            if proceed:
                proceed, errmsg = cff.check_save_conflict(fname, self.lastload)
                if errmsg != '':
                    proceed = messageboxes.proceed_question(self, errmsg)
                if proceed:
                    ok_save, path = cff.save_settings(self.templates, fname=fname)
                    if ok_save:
                        self.saved_label = self.txt_label.text()
                        self.lbl_verif_label.setText(
                            'Template saved.'
                            ' Press Next for further options or Cancel to Close.')
                        self.lbl_verif_label.setStyleSheet("QLabel {color: darkgreen}")
                        self.main.gui.current_auto_template = (
                            self.current_template.label)
                    else:
                        QMessageBox.warning(
                            self, 'Failed saving', f'Failed saving to {path}')


class FinishPage(QWizardPage):
    """Page to finish or proceed working in settings."""

    def __init__(self, main):
        super().__init__()
        self.main = main

        self.setTitle('Edit details of templates in Settings manager')
        self.setSubTitle(
            '''Some details are only editable in the Settings manager.''')
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        vlo.addSpacing(50)
        alt_txts = [
            ('Finish and open Settings manager on automation templates '
             'to work on further details.'),
            ('Finish and open Settings manager on the selected Parameter set to work '
             'further on output settings (select the Output Settings tab there).'),
            ('Finish and open Settings manager to set the import settings for '
             'automation'),
            'Just finish'
            ]
        self.btngroup = QButtonGroup()
        lo_rb = QVBoxLayout()
        for btn_no, txt in enumerate(alt_txts):
            rbtn = QRadioButton(txt)
            self.btngroup.addButton(rbtn, btn_no)
            lo_rb.addWidget(rbtn)
        vlo.addLayout(lo_rb)


class AutomationWizard(QWizard):
    """GUI setup for the Automation assistant window."""

    def __init__(self, main):
        super().__init__()

        self.setWindowTitle('Automation wizard')
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOptions(
            QWizard.IndependentPages | QWizard.NoBackButtonOnStartPage)
        self.setPixmap(QWizard.WatermarkPixmap,
                       QPixmap(
                           f'{os.environ[ENV_ICON_PATH]}watermark_gears.png'))
        self.setPixmap(QWizard.LogoPixmap,
                       QPixmap(f'{os.environ[ENV_ICON_PATH]}iQC_icon128.png'))
        self.main = main

        ok_config_folder = cff.verify_config_folder(self)
        if ok_config_folder:
            # TODO delete? ok_common, path, auto_common = cff.load_settings(fname='auto_common')
            self.ok_auto, path, self.templates = cff.load_settings(
                fname='auto_templates')
            self.lastload = time()

        ok_img = (len(self.main.imgs) > 0)
        ok_paramset = not self.main.wid_paramset.edited
        current_qt = None
        if self.main.wid_quicktest.gb_quicktest.isChecked():
            self.main.wid_quicktest.get_current_template()
            current_qt = self.main.wid_quicktest.current_template
        ok_quicktest = False
        if current_qt is not None:
            if any(current_qt.tests):
                ok_quicktest = True
        ok_results = False
        if ok_quicktest:
            if len([*self.main.results]) > 0:
                set_qt_tests = set(sum(current_qt.tests, []))
                set_res = set([*self.main.results])
                if len(set_res.difference(set_qt_tests)) == 0:
                    ok_results = True

        self.addPage(StartupPage(
            self.main, self, ok_config=ok_config_folder, ok_img=ok_img,
            ok_paramset=ok_paramset,
            ok_quicktest=ok_quicktest, ok_results=ok_results))
        if all([ok_config_folder, ok_img, ok_quicktest, ok_paramset]):
            self.save_page = SavePage(
                self.main, templates=self.templates, lastload=self.lastload)
            self.addPage(self.save_page)
            self.finish_page = FinishPage(self.main)
            self.addPage(self.finish_page)

        self.finished = False
        self.button(QWizard.FinishButton).clicked.connect(self.finish_clicked)

    def finish_clicked(self):
        """Continue to Settings manager if specified."""
        try:
            select_id = self.finish_page.btngroup.checkedId()
        except AttributeError:
            select_id = None
        self.finished = True
        if select_id is not None:
            if select_id == 0:  # edit auto_template in Settings manager
                if self.save_page.saved_label != '':
                    self.main.run_settings(
                        initial_view='Templates DICOM',
                        initial_template_label=self.save_page.saved_label)
                else:
                    self.main.run_settings(initial_view='Templates DICOM')
            elif select_id == 1:  # edit paramset in Settings manager
                self.main.run_settings(
                    initial_view='Parameter sets / output',
                    paramset_output=True,
                    initial_template_label=self.main.current_paramset.label)
            elif select_id == 2:  # edit import settings in Settings manager
                self.main.run_settings(
                    initial_view='Import settings')
            else:
                pass
