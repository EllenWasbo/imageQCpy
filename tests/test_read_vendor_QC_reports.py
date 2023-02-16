# -*- coding: utf-8 -*-
"""
Tests reading vendor QC reports.

@author: ewas
"""
import os
from pathlib import Path

from imageQC.scripts import read_vendor_QC_reports

path_tests = Path(__file__).parent

def test_Siemens_PET_dailyQC():
    """Test reading Siemens PET dailyQC for different software versions."""
    dir_tests = os.path.dirname(__file__)
    file_names = ['Siemens_PET_DailyQC_VG60A.pdf',
                  'Siemens_PET_DailyQC_VG80B.pdf']
    expected_res = [
        ['21.02.2017', 'LAB19-PETCT', '', '', '', 
         1.0, 103.4, 37.9, 33.5, 31610000.0, 0, 0, 0, -1.0, -5.7],
        ['30.05.2022', 'LAB19-PETCT', 'X', '', '', '0.940000', '103.112',
         '36.2', '31.617', '3.079e+007', '0', '0', '0', '-1.5', '-3.0']
        ]

    for i, f in enumerate(file_names):
        p = os.path.join(dir_tests, 'test_inputs',
                         'vendor_QC_reports', 'SiemensPET', f)
        txt = read_vendor_QC_reports.get_pdf_txt(p)
        res = read_vendor_QC_reports.read_Siemens_PET_dailyQC(txt)
        breakpoint()
        assert res['values'] == expected_res[i]

def test_Siemens_PET_dailyQC_xml():
    """Test reading Siemens PET dailyQC xml report (PET MR)."""
    dir_tests = os.path.dirname(__file__)
    file_names = ['Siemens_PET_DailyQC_MR.xml']
    expected_res = [
        ['02.08.2021', 1.0, 106.7, 52.8, 32.7, '1.951e+007', '0', '0', '0']
        ]

    for i, f in enumerate(file_names):
        p = os.path.join(dir_tests, 'test_inputs',
                         'vendor_QC_reports', 'SiemensPET', f)
        root = read_vendor_QC_reports.get_xml_tree(p)
        res = read_vendor_QC_reports.read_Siemens_PET_dailyQC_xml(root)
        breakpoint()
        assert res['values'] == expected_res[i]

def test_Siemens_CT_QC():
    """Test reading Siemens CT constancy and daily QC.

    Different software versions and CT models.
    """
    dir_tests = os.path.dirname(__file__)
    file_names = ['Siemens_CT_constancy_VA48A_DefinitionAS.pdf',
                  'Siemens_CT_constancy_VA48A_Edge_Norwegian.pdf',
                  'Siemens_CT_constancy_VA48A_Flash_Norwegian.pdf',
                  'Siemens_CT_constancy_VB20A_Force.pdf',
                  'Siemens_CT_Constancy_VB20A_Drive_Norwegian.pdf'
                  ]
    expected_res = [
        ['28.07.2016', 'SCM MSI 96233', 'SOMATOM Definition AS', '96233',
         '624261672', '-', '0.02', '0.6', '-0.5', '-0.06', '0.4', '0.44',
         '3.69', '4.14', '4.72', '4.8', '4.89', '5.07', '3.39', '5.81',
         '3.19', '5.86', '11.37', '13.94', '', '', '', '', '', '', '', '',
         '', '', '', '', '', '', '', '', '', ''],
        ['08.02.2017', 'CH Gundersen', 'SOMATOM Definition Edge', '83211',
         '276301531', '-', '0.14', '0.56', '-0.43', '0.37', '0.73', '1.16',
         '3.54', '4.13', '4.55', '4.68', '4.83', '5.24', '3.36', '5.76',
         '3.13', '5.83', '11.37', '13.93', '17.16', '22.20', '', '', '', '',
         '', '', '', '', '', '', '', '', '', '', '', ''],
        ['10.12.2015', 'Knut G', 'SOMATOM Definition Flash', '73537',
         '659121274', '660121274', '0.2', '0.69', '-0.77', '0.22', '0.63',
         '1.83', '3.56', '4.0', '4.71', '4.87', '4.98', '5.15', '3.28',
         '5.67', '3.01', '5.63', '11.02', '13.72', '16.21', '22.05', '0.91',
         '1.99', '-0.59', '0.08', '0.74', '1.41', '3.79', '3.8', '5.04',
         '5.1', '4.97', '5.09', '3.28', '5.70', '3.33', '5.75'],
        ['11.02.2021', 'Siemens v/Andreas', 'SOMATOM Force', '76311',
         '490282004', '674422072', '-0.34', '0.44', '-0.79', '1.43', '1.45',
         '2.14', '3.5', '3.88', '4.84', '5.05', '4.83', '5.05', '3.30',
         '5.40', '3.25', '5.86', '12.86', '15.79', '24.02', '28.79', '-0.11',
         '1.84', '-1.25', '0.09', '2.52', '1.68', '3.9', '4.0', '4.84',
         '5.04', '4.91', '5.07', '3.30', '5.40', '3.28', '5.38'],
        ['31.05.2022', 'Fysiker', 'SOMATOM Drive', '105158',
         '173312102', '-', '0.13', '0.53', '-0.22', '0.88', '0.68', '0.94',
         '3.27', '4.89', '4.57', '4.68', '4.88', '5.04', '3.56', '5.87',
         '3.12', '5.75', '11.71', '13.43', '18.27', '23.23', '0.15', '0.9',
         '-1.36', '-0.17', '1.11', '2.69', '4.85', '4.59', '4.79', '5.03',
         '4.82', '5.03', '3.55', '5.87', '3.50', '5.82']]

    for i, f in enumerate(file_names):
        p = os.path.join(dir_tests, 'test_inputs',
                         'vendor_QC_reports', 'SiemensCT', f)
        txt = read_vendor_QC_reports.get_pdf_txt(p)
        res = read_vendor_QC_reports.read_Siemens_CT_QC(txt)
        breakpoint()
        assert res['values'] == expected_res[i]


def test_Siemens_CT_QC_Intevo():
    """Test reading Siemens CT constancy and daily QC from Intevo (SPECT-CT).

    Different test files.
    """
    dir_tests = os.path.dirname(__file__)
    file_names = ['Siemens_CT_constancy_VB22A_Intevo_slice.pdf',
                  'Siemens_CT_constancy_VB22A_Intevo_homogeneity.pdf',
                  'Siemens_CT_constancy_VB22A_Intevo_noise.pdf',
                  'Siemens_CT_constancy_VB22A_Intevo_MTF.pdf']
    expected_res = [
        ['26.10.2021', 'Siemens v/Gundersen Stavanger Universitetssykehus',
         'Symbia Intevo 16', '98164', '289262083', '', '', '', '', '', '', '',
         '', '5.17', '5.18', '5.01', '5.04', '', '', '', '', '', ''],
        ['26.10.2021', 'Siemens v/Gundersen Stavanger Universitetssykehus',
         'Symbia Intevo 16', '98164', '289262083', '-0.1', '0.77', '0.17',
         '0.51', '2.02', '2.2', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['26.10.2021', 'Siemens v/Gundersen Stavanger Universitetssykehus',
         'Symbia Intevo 16', '98164', '289262083', '', '', '', '', '', '',
         '4.4', '4.61', '', '', '', '', '', '', '', '', '', ''],
        ['26.10.2021', 'Siemens v/Gundersen Stavanger Universitetssykehus',
         'Symbia Intevo 16', '98164', '289262083', '', '', '', '', '', '', '',
         '', '', '', '', '', '3.71', '6.94', '3.00', '6.36', '9.71', '13.65']
        ]

    for i, f in enumerate(file_names):
        p = os.path.join(dir_tests, 'test_inputs',
                         'vendor_QC_reports', 'SiemensCT', 'Intevo', f)
        txt = read_vendor_QC_reports.get_pdf_txt(p)
        res = read_vendor_QC_reports.read_Siemens_CT_QC(txt)
        breakpoint()
        assert res['values'] == expected_res[i]


def test_Siemens_CT_QC_Symia():
    """Not ready - return status false."""
    dir_tests = os.path.dirname(__file__)
    file_names = ['Siemens_CT_constancy_2007E_VA60A_Symbia_homogeneity.pdf',
                  'Siemens_CT_constancy_2007E_VA60C_Symbia_homogeneity_Norwegian.pdf',
                  'Siemens_CT_constancy_2013A_VB10B_Symbia.pdf',
                  'Siemens_CT_daily_2007E_VA60A_Symbia_noise.pdf',
                  'Siemens_CT_daily_2013A_VB10B_Symbia.pdf'
                  ]

    for i, f in enumerate(file_names):
        p = os.path.join(dir_tests, 'test_inputs',
                         'vendor_QC_reports', 'SiemensCT', 'Symbia', f)
        txt = read_vendor_QC_reports.get_pdf_txt(p)
        res = read_vendor_QC_reports.read_Siemens_CT_QC(txt)
        assert res['values'] == []
        assert res['status'] is False


def test_Siemens_NM_energy_spectrum():
    file_name = (path_tests / 'test_inputs' / 'vendor_QC_reports' / 'SiemensNM' /
                 'eResDet1.txt')
    res = read_vendor_QC_reports.read_energy_spectrum_Siemens_gamma_camera(
        file_name.resolve())
    assert res['values'] == [32706.0, 141.350159, 12.678160069552563, 8.96932848130193]