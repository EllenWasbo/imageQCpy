# -*- coding: utf-8 -*-
"""User interface for test tabs vendor in main window of imageQC.

@author: Ellen Wasbø
"""
import os

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QMessageBox, QFileDialog)

# imageQC block start
from imageQC.config.iQCconstants import (
    ENV_ICON_PATH, VENDOR_FILE_OPTIONS
    )
from imageQC.ui import reusable_widgets as uir
from imageQC.ui import messageboxes
from imageQC.scripts.read_vendor_QC_reports import read_vendor_template
from imageQC.config.config_classes import AutoVendorTemplate
# imageQC block end


class ParamsTabVendor(QWidget):
    """Test tabs for vendor file analysis."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent

        self.selected = 0

        vlo = QVBoxLayout()
        infotxt = '''
        imageQC can read and extract parameters from a set of
        vendor specific files (QC reports or exported data).<br>
        Select one of these file-types in the list and get the
        results tabulated.
        '''
        vlo.addWidget(uir.LabelItalic(infotxt))
        vlo.addWidget(uir.HLine())
        self.table_file_types = QTreeWidget()
        self.table_file_types.setColumnCount(1)
        self.table_file_types.setHeaderLabels(['Expected file type'])
        self.table_file_types.setMinimumWidth(300)
        self.table_file_types.setRootIsDecorated(False)
        self.table_file_types.currentItemChanged.connect(self.update_selected)
        vlo.addWidget(self.table_file_types)
        btn_open = QPushButton('Open and read vendor specific file')
        btn_open.setIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}open.png'))
        btn_open.clicked.connect(self.open_vendor_files)
        vlo.addWidget(btn_open)
        vlo.addStretch()
        self.setLayout(vlo)

        self.update_table()

    def update_displayed_params(self):
        """Ignore updating displayed params."""
        pass

    def update_selected(self, current):
        """Update self.selected when selected in table."""
        self.selected = self.table_file_types.indexOfTopLevelItem(current)

    def update_table(self, set_selected=0):
        """Update table based on mode selected."""
        self.table_file_types.clear()
        self.options = VENDOR_FILE_OPTIONS[self.main.current_modality]

        for option in self.options:
            item = QTreeWidgetItem([option])
            self.table_file_types.addTopLevelItem(item)

        self.selected = set_selected
        self.table_file_types.setCurrentItem(
            self.table_file_types.topLevelItem(self.selected))

    def open_vendor_files(self):
        """Open vendor files for analysis."""
        if len(VENDOR_FILE_OPTIONS[self.main.current_modality]) > 0:
            file_type = VENDOR_FILE_OPTIONS[
                self.main.current_modality][self.selected]
            implemented_types = [
                'Siemens CT Constancy/Daily Reports (.pdf)',
                'Planmeca CBCT report (.html)',
                'Siemens PET-CT DailyQC Reports (.pdf)',
                'Siemens PET-MR DailyQC Reports (.xml)',
                'Siemens exported energy spectrum (.txt)',
                'Philips MR ACR report (.pdf)',
                'GE QAP (.txt)',
                'GE Mammo QAP (txt)',
                ]
            if file_type in implemented_types:
                file_suffix = file_type.split('(')[1]
                file_suffix = file_suffix.split(')')[0]
                if '.' not in file_suffix:
                    file_suffix = ''
                temp = AutoVendorTemplate(file_type=file_type, file_suffix=file_suffix)
                if file_suffix:
                    fnames = QFileDialog.getOpenFileNames(
                            self, f'Open {file_type}',
                            filter=f'{file_suffix[1:].upper()} files (*{file_suffix})')
                else:
                    fnames = QFileDialog.getOpenFileNames(self, f'Open {file_type}')
                if len(fnames[0]) > 0:
                    self.run_template(template=temp, files=fnames[0])
            else:
                #TODO
                QMessageBox.warning(
                    self, 'Not implemented yet', 'Sorry - not implemented yet')

    def run_template(self, template=None, files=None):
        """Run automation template for given files.

        Parameters
        ----------
        template : AutoVendorTemplate
            from config.config_classes
        files : list of str
            files to read
        """
        self.main.status_bar.showMessage('Reading vendor QC file(s)...')
        self.main.start_wait_cursor()

        results = []
        file_not_match = []
        res_failed = []
        for fno, file in enumerate(files):
            self.main.status_bar.showMessage(
                f'Reading file {fno+1}/{len(files)}: {file}')
            res_this = read_vendor_template(template=template, filepath=file)
            if res_this['status']:
                if len(results) > 0:
                    if res_this['headers'] == results[0]['headers']:
                        results.append(res_this)
                    else:
                        file_not_match.append(file)
                else:
                    results.append(res_this)
            else:
                res_failed.append(file)

        self.main.stop_wait_cursor()
        self.main.status_bar.showMessage('Finished', 1000)

        if len(results) > 0:
            self.main.results['vendor'] = {
                'headers': results[0]['headers'],
                'values': [res_this['values'] for res_this in results],
                }
            if 'details' in results[0]:
                self.main.results['vendor']['details'] = [
                    res_this['details'] for res_this in results]
            self.main.wid_res_tbl.result_table.fill_table(vendor=True)

            if len(file_not_match) > 0:
                dlg = messageboxes.MessageBoxWithDetails(
                    self, title='Files not matched',
                    msg=('Some files did not match the content of the first file.'
                         ' See details for files not matching first file.'),
                    details=file_not_match, icon=QMessageBox.Icon.Warning)
                dlg.exec()
        if len(res_failed) > 0:
            dlg = messageboxes.MessageBoxWithDetails(
                self, title='Files not recognized',
                msg=('Some files did not match the expected content for filetype '
                     f'{template.file_type}. See details for files not recognized.'),
                details=res_failed, icon=QMessageBox.Icon.Warning)
            dlg.exec()
