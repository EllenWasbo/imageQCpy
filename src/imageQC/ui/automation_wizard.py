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
    QLabel, QCheckBox, QLineEdit, QPushButton, QRadioButton, QAction,
    QMessageBox, QFileDialog
    )

# imageQC block start
from imageQC.config.iQCconstants import ENV_ICON_PATH
import imageQC.config.config_func as cff
from imageQC.scripts.mini_methods_format import valid_template_name
import imageQC.ui.settings
from imageQC.config import config_classes as cfc
from imageQC.scripts import dcm
from imageQC.ui.reusable_widgets import LabelMultiline, LabelItalic, ToolBarBrowse
from imageQC.ui import messageboxes
from imageQC.scripts.mini_methods import create_empty_file
# imageQC block end


class StartupPage(QWizardPage):
    """Page with list of tasks to fulfill to proceed."""

    def __init__(self, main, ok_config=False, ok_img=False, ok_paramset=False,
                 ok_quicktest=False, ok_results=False):
        super().__init__()
        self.main = main

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
                 'to perform. Save and ame the new QuickTest template.')
                ]
            )
        vlo.addWidget(gb_qt)

        def test_output():
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

        gb_res = OkGroupBox(
            'Optionally Run the QuickTest template', ok=ok_results,
            txts=[
                'Run the QuickTest template in the main window to get results.',
                'Proof-testing of the output paramters may then be available.'
                ]
            )
        btn_test_output = QPushButton('Test output')
        btn_test_output.setEnabled(ok_results)
        btn_test_output.clicked.connect(test_output)
        gb_res.vlo.addWidget(btn_test_output)
        gb_res.vlo.addWidget(LabelItalic(
            ('You may change and further test the output settings from the Settings '
             'manager.')))
        vlo.addWidget(gb_res)

        if all([ok_img, ok_quicktest, ok_paramset]):
            vlo.addWidget(QLabel(
                'Continue to next step to verify and save template.'))
        else:
            vlo.addWidget(QLabel((
                'Cancel and come back to the wizard when all mandatory steps are '
                'fulfilled.')))


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


class SavePage(QWizardPage):
    """Page to finish and save."""

    def __init__(self, main, templates=None, lastload=None):
        super().__init__()
        self.main = main
        self.templates = templates
        self.lastload = lastload
        self.saved_label = ''
        try:
            self.already_labels = \
                [obj.label for obj
                 in self.templates[self.main.current_modality]]
            if len(self.already_labels) > 0:
                if self.already_labels[0] == '':
                    self.already_labels = None
        except KeyError:
            self.already_labels = None

        self.setTitle('Finish and save template')
        self.setSubTitle(
            '''Verify the parameters.''')
        vlo = QVBoxLayout()
        self.setLayout(vlo)

        flo = QFormLayout()
        vlo.addLayout(flo)

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
        sort_pattern_txt = '-'
        if self.main.current_sort_pattern is not None:
            tags = self.main.current_sort_pattern.list_tags
            asc_txt = self.main.current_sort_pattern.list_sort
            tags_txt = []
            for i, tag in enumerate(tags):
                asc_txt = ('(ASC)' if self.main.current_sort_pattern.list_sort[i]
                           else '(DESC)')
                tags_txt.append(tags[i] + asc_txt)
            sort_pattern_txt = ', '.join(tags_txt)

        flo.addRow(QLabel('Modality:'), QLabel(self.main.current_modality))
        flo.addRow(QLabel('Station name:'), QLabel(self.statname))

        hlo_path_input = QHBoxLayout()
        vlo.addLayout(hlo_path_input)
        hlo_path_input.addWidget(QLabel('Input path '))
        self.txt_path_input = QLineEdit(path_input)
        self.txt_path_input.setMinimumWidth(300)
        hlo_path_input.addWidget(self.txt_path_input)
        toolb = ToolBarBrowse('Browse to find path')
        toolb.act_browse.triggered.connect(
            lambda: self.locate_folder(self.txt_path_input))
        hlo_path_input.addWidget(toolb)

        hlo_path_output = QHBoxLayout()
        vlo.addLayout(hlo_path_output)
        hlo_path_output.addWidget(QLabel('Output path '))
        self.txt_path_output = QLineEdit('')
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
                    QLabel(self.main.current_paramset.label))
        flo2.addRow(QLabel('Use QuickTest template:'),
                    QLabel(self.main.wid_quicktest.cbox_template.currentText()))
        flo2.addRow(QLabel('Sort images by:'), QLabel(sort_pattern_txt))

        self.chk_archive = QCheckBox()
        self.chk_archive.setChecked(True)
        flo2.addRow(QLabel('Archive images when analysed:'), self.chk_archive)
        self.chk_active = QCheckBox()
        self.chk_active.setChecked(True)
        flo2.addRow(QLabel('Set template active:'), self.chk_active)
        vlo.setSpacing(20)

        self.txt_label = QLineEdit()
        self.registerField("label*", self.txt_label)  # next not available before given
        hlo_label = QHBoxLayout()
        hlo_label.addWidget(QLabel('Name the new template:'))
        hlo_label.addWidget(self.txt_label)
        btn_save = QPushButton('Save automation template')
        btn_save.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}save.png'))
        btn_save.clicked.connect(self.save_template)
        if self.main.save_blocked:
            btn_save.setEnabled(False)
        hlo_label.addWidget(btn_save)
        vlo.addLayout(hlo_label)
        self.lbl_verif_label = LabelItalic('', color='red')
        vlo.addWidget(self.lbl_verif_label)

    def locate_folder(self, widget):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if widget.text() != '':
            dlg.setDirectory(widget.text())
        if dlg.exec():
            fname = dlg.selectedFiles()
            widget.setText(os.path.normpath(fname[0]))

    def locate_file(self, widget, title='Locate file',
                    filter_str='All files (*)', opensave=False):
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

    def verify_new_label(self):
        text = self.txt_label.text()
        if text != '':
            text = valid_template_name(text)
            self.txt_label.setText(text)
        else:
            self.lbl_verif_label.setText(
                'Please define a new label for the automation template.')
        ok_label = True if text != '' else False
        if self.already_labels:
            if text in self.already_labels:
                ok_label = False
                if text != '':
                    self.lbl_verif_label.setText(
                        'Label already used. Set another.')
            else:
                self.lbl_verif_label.setText('')

        return ok_label

    def save_template(self):
        proceed = self.verify_new_label()
        if proceed:
            fname = 'auto_templates'
            # create new template and add to self.templates
            this_temp = cfc.AutoTemplate(
                label=self.txt_label.text(),
                path_input=self.txt_path_input.text(),
                path_output=self.txt_path_output.text(),
                station_name=self.statname,
                paramset_label=self.main.current_paramset.label,
                quicktemp_label=self.main.wid_quicktest.cbox_template.currentText(),
                archive=self.chk_archive.isChecked(),
                active=self.chk_active.isChecked()
                )
            if self.main.current_sort_pattern is not None:
                this_temp.sort_pattern = self.main.current_sort_pattern
            # TODO: delete_if_not_image: bool = False
            # TODO delete_if_too_many: bool = False
            if self.already_labels:
                self.templates[self.main.current_modality].append(this_temp)
            else:
                self.templates[self.main.current_modality] = [this_temp]

            proceed, errmsg = cff.check_save_conflict(fname, self.lastload)
            if errmsg != '':
                proceed = messageboxes.proceed_question(self, errmsg)
            if proceed:
                ok_save, path = cff.save_settings(self.templates, fname=fname)
                if ok_save:
                    self.saved_label = self.txt_label.text()
                    self.lbl_verif_label.setText(
                        'Template saved.')
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
            ok_common, path, auto_common = cff.load_settings(fname='auto_common')
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
            self.main, ok_config=ok_config_folder, ok_img=ok_img,
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
                    initial_template_label=self.main.current_paramset.label)
            elif select_id == 2:  # edit import settings in Settings manager
                self.main.run_settings(
                    initial_view='Import settings')
            else:
                pass
