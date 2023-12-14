# -*- coding: utf-8 -*-
"""
Tests reading vendor QC reports.

@author: ewas
"""
from pathlib import Path

from imageQC.scripts import read_vendor_QC_reports

path_tests = Path(__file__).parent


def test_Siemens_PET_dailyQC():
    """Test reading Siemens PET dailyQC for different software versions."""
    file_names = ['Siemens_PET_DailyQC_VG60A.pdf',
                  'Siemens_PET_DailyQC_VG80B.pdf']
    expected_res = [
        ['21.02.2017', 'LAB19-PETCT', '', '', '',
         1.0, 103.4, 37.9, 33.5, 31610000.0, 0, 0, 0, -1.0, -5.7],
        ['30.05.2022', 'LAB19-PETCT', 'X', '', '',
         0.94, 103.112, 36.2, 31.617, 30790000.0, 0, 0, 0, -1.5, -3.0]
        ]

    for i, f in enumerate(file_names):
        p = path_tests / 'test_inputs' / 'vendor_QC_reports' / 'SiemensPET' / f
        txt = read_vendor_QC_reports.get_pdf_txt(p)
        res = read_vendor_QC_reports.read_Siemens_PET_dailyQC(txt)
        assert res['values'] == expected_res[i]


def test_Siemens_PET_dailyQC_xml():
    """Test reading Siemens PET dailyQC xml report (PET MR)."""
    file_names = ['Siemens_PET_DailyQC_MR.xml']
    expected_res = [
        ['02.08.2021', 1.0, 106.7, 52.8, 32.7, 19510000.0, 0, 0, 0]
        ]

    for i, f in enumerate(file_names):
        p = path_tests / 'test_inputs' / 'vendor_QC_reports' / 'SiemensPET' / f
        root = read_vendor_QC_reports.get_xml_tree(p)
        res = read_vendor_QC_reports.read_Siemens_PET_dailyQC_xml(root)
        assert res['values'] == expected_res[i]


def test_Siemens_CT_QC():
    """Test reading Siemens CT constancy and daily QC.

    Different software versions and CT models.
    """
    file_names = ['Siemens_CT_constancy_VA48A_DefinitionAS.pdf',
                  'Siemens_CT_constancy_VA48A_Edge_Norwegian.pdf',
                  'Siemens_CT_constancy_VA48A_Flash_Norwegian.pdf',
                  'Siemens_CT_constancy_VB20A_Force.pdf',
                  'Siemens_CT_Constancy_VB20A_Drive_Norwegian.pdf'
                  ]
    expected_res = [
        ['28.07.2016', 'SCM MSI 96233', 'SOMATOM Definition AS', '96233',
         '624261672', '-', 0.02, 0.6, -0.5, -0.06, 0.4, 0.44,
         3.69, 4.14, 4.72, 4.8, 4.89, 5.07, 3.3866666666666667,
         5.81, 3.19, 5.86, 11.37, 13.94, None, None, '', '', '', '', '', '', '', '',
         '', '', '', '', None, None, None, None],
        ['08.02.2017', 'CH Gundersen', 'SOMATOM Definition Edge', '83211',
         '276301531', '-', 0.14, 0.56, -0.43, 0.37, 0.73, 1.16,
         3.54, 4.13, 4.55, 4.68, 4.83, 5.24, 3.36, 5.756666666666668,
         3.13, 5.83, 11.37, 13.93, 17.165, 22.195, '', '', '', '',
         '', '', '', '', '', '', '', '', None, None, None, None],
        ['10.12.2015', 'Knut G', 'SOMATOM Definition Flash', '73537',
         '659121274', '660121274',  0.2, 0.69, -0.77, 0.22, 0.63, 1.83,
         3.56, 4.0, 4.71, 4.87, 4.98, 5.15, 3.28, 5.67, 3.01, 5.63,
         11.02, 13.72, 16.21, 22.05, 0.91, 1.99, -0.59, 0.08, 0.74,
         1.41, 3.79, 3.8, 5.04, 5.1, 4.97, 5.09, 3.28, 5.7, 3.33, 5.75],
        ['11.02.2021', 'Siemens v/Andreas', 'SOMATOM Force', '76311',
         '490282004', '674422072', -0.34, 0.44, -0.79, 1.43, 1.45, 2.14,
         3.5, 3.88, 4.84, 5.05, 4.83, 5.05, 3.3, 5.4, 3.2475, 5.8575,
         12.86, 15.79, 24.02, 28.786666666666665, -0.11, 1.84, -1.25,
         0.09, 2.52, 1.68, 3.9, 4.0, 4.84, 5.04, 4.91, 5.07, 3.3, 5.4, 3.28, 5.385],
        ['31.05.2022', 'Fysiker', 'SOMATOM Drive', '105158',
         '173312102', '-', 0.13, 0.53, -0.22, 0.88, 0.68, 0.94, 3.27,
         4.89, 4.57, 4.68, 4.88, 5.04, 3.56, 5.871666666666667, 3.1233333333333335,
         5.753333333333334, 11.713333333333333, 13.433333333333332,
         18.275, 23.225, 0.15, 0.9, -1.36, -0.17, 1.11, 2.69, 4.85,
         4.59, 4.79, 5.03, 4.82, 5.03, 3.5549999999999997, 5.871666666666667,
         3.4983333333333335, 5.82]]

    for i, f in enumerate(file_names):
        p = (path_tests / 'test_inputs' / 'vendor_QC_reports' / 'SiemensCT' / f)
        txt = read_vendor_QC_reports.get_pdf_txt(p)
        res = read_vendor_QC_reports.read_Siemens_CT_QC(txt)
        assert res['values'] == expected_res[i]


def test_Siemens_CT_QC_Intevo():
    """Test reading Siemens CT constancy and daily QC from Intevo (SPECT-CT).

    Different test files.
    """
    file_names = ['Siemens_CT_constancy_VB22A_Intevo_slice.pdf',
                  'Siemens_CT_constancy_VB22A_Intevo_homogeneity.pdf',
                  'Siemens_CT_constancy_VB22A_Intevo_noise.pdf',
                  'Siemens_CT_constancy_VB22A_Intevo_MTF.pdf']
    expected_res = [
        ['26.10.2021', 'Siemens v/Gundersen Stavanger Universitetssykehus',
         'Symbia Intevo 16', '98164', '289262083', '', '', '', '', '', '', '',
         '', 5.17, 5.18, 5.01, 5.04, '', '', '', '', '', ''],
        ['26.10.2021', 'Siemens v/Gundersen Stavanger Universitetssykehus',
         'Symbia Intevo 16', '98164', '289262083', -0.1, 0.77, 0.17, 0.51, 2.02, 2.2,
         '', '', '', '', '', '', '', '', '', '', '', ''],
        ['26.10.2021', 'Siemens v/Gundersen Stavanger Universitetssykehus',
         'Symbia Intevo 16', '98164', '289262083', '', '', '', '', '', '',
         4.4, 4.61, '', '', '', '', '', '', '', '', '', ''],
        ['26.10.2021', 'Siemens v/Gundersen Stavanger Universitetssykehus',
         'Symbia Intevo 16', '98164', '289262083', '', '', '', '', '', '', '',
         '', '', '', '', '', 3.7074999999999996, 6.942500000000001, 3.005,
         6.355, 9.71, 13.65]
        ]

    for i, f in enumerate(file_names):
        p = (path_tests / 'test_inputs' / 'vendor_QC_reports' / 'SiemensCT' /
             'Intevo' / f)
        txt = read_vendor_QC_reports.get_pdf_txt(p)
        res = read_vendor_QC_reports.read_Siemens_CT_QC(txt)
        assert res['values'] == expected_res[i]


def test_Siemens_CT_QC_Symia():
    """Not ready - return status false."""
    file_names = ['Siemens_CT_constancy_2007E_VA60A_Symbia_homogeneity.pdf',
                  'Siemens_CT_constancy_2007E_VA60C_Symbia_homogeneity_Norwegian.pdf',
                  'Siemens_CT_constancy_2013A_VB10B_Symbia.pdf',
                  'Siemens_CT_daily_2007E_VA60A_Symbia_noise.pdf',
                  'Siemens_CT_daily_2013A_VB10B_Symbia.pdf'
                  ]

    for i, f in enumerate(file_names):
        p = (path_tests / 'test_inputs' / 'vendor_QC_reports' / 'SiemensCT' /
             'Symbia' / f)
        txt = read_vendor_QC_reports.get_pdf_txt(p)
        res = read_vendor_QC_reports.read_Siemens_CT_QC(txt)
        assert res['values'] == []
        assert res['status'] is False


def test_Siemens_NM_energy_spectrum():
    file_name = (path_tests / 'test_inputs' / 'vendor_QC_reports' / 'SiemensNM' /
                 'eResDet1.txt')
    res = read_vendor_QC_reports.read_e_spectrum_Siemens_gamma_camera(
        file_name.resolve())
    assert res['values'] == [32706.0, 141.350159, 12.678160069552563, 8.96932848130193]


def test_Philips_MR_ACR_weekly():
    """Test reading Philips MR ACR Report (weekly)."""
    file_names = ['0123.pdf', '0223.pdf', '0323.pdf']
    expected_res = [
        ['06.01.2023', 'Weekly Tests', 0.2, 63886147, 1.2807,
         189.45, 189.45, 190.59, 189.21, '1', '0', '1', '0', 10, 10, 10, 10],
        ['13.01.2023', 'Weekly Tests', 0.3, 63886126, 1.2816,
         189.45, 189.45, 189.21, 190.59, '1', '0', '1', '0', 10, 10, 10, 10],
        ['20.01.2023', 'Weekly Tests', 0.2, 63886085, 1.2775,
         189.45, 189.45, 189.21, 190.59, '1', '0', '0', '0', 10, 10, 10, 10]
        ]

    for i, f in enumerate(file_names):
        p = path_tests / 'test_inputs' / 'vendor_QC_reports' / 'PhilipsMR' / f
        txt = read_vendor_QC_reports.get_pdf_txt(p)
        res = read_vendor_QC_reports.read_Philips_MR_ACR_report(txt)
        assert res['values'] == expected_res[i]
