# -*- coding: utf-8 -*-
"""
To create phantom test data with anonymized equipment.

@author: ellen
"""
import sys
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication, QVBoxLayout, QPushButton
import pydicom


class filedialog(QWidget):
    def __init__(self, parent=None):
        super(filedialog, self).__init__(parent)

        layout = QVBoxLayout()
        self.btn = QPushButton("Locate file(s) to anonymize")
        self.btn.clicked.connect(self.anonymize_files)

        layout.addWidget(self.btn)
        self.setLayout(layout)

    def anonymize_files(self):
        fnames = QFileDialog.getOpenFileNames(
            self, 'Open DICOM files',
            filter="DICOM files (*.dcm);;All files (*)")
        filenames = fnames[0]
        if len(filenames) > 0:
            out_folder = ''
            p = Path(__file__).parent.parent / 'tests' / 'test_inputs'
            path_test_input = str(p)
            out_res = QFileDialog.getExistingDirectory(
                self, 'Save in folder', path_test_input)
            out_folder = out_res
            if out_folder != '':
                for filename in filenames:
                    pd = pydicom.dcmread(filename)
                    pd.remove_private_tags()
                    try:
                        pd.file_meta[0x2,0x16].value = 'NN'  # Source AET
                    except KeyError:
                        pass
                    pd['StationName'].value = 'NN'
                    pd['PatientID'].value = 'id'
                    pd['PatientName'].value = 'NN'
                    pd['PatientBirthDate'].value = '19000101'
                    
                    keys = ['InstitutionName',
                            'InstitutionAddress',
                            'ReferringPhysicianName',
                            'PhysiciansOfRecord',
                            'RequestingPhysician',
                            'OperatorsName'
                            ]
                    for key in keys:
                        try:
                            pd[key].value = ''
                        except KeyError:
                            pass

                    new_filename = ''
                    new_filename += pd['Modality'].value
                    new_filename += '_' + pd['SeriesDescription'].value
                    try:
                        new_filename += '_' + pd['ProtocolName'].value
                    except KeyError:
                        pass
                    new_filename += '_' + f"{pd['InstanceNumber'].value:03}"
                    new_filename += '.dcm'

                    new_path = Path(out_folder) / new_filename
                    pd.save_as(new_path)

def main():
    app = QApplication(sys.argv)
    ex = filedialog()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
