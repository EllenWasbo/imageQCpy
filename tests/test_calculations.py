# -*- coding: utf-8 -*-
"""
Tests on calculating.

@author: ewas
"""
import yaml
import numpy as np
from pathlib import Path

import imageQC.scripts.dcm as dcm
import imageQC.config.config_classes as cfc
from imageQC.scripts.input_main_auto import InputMain
import imageQC.scripts.calculate_roi as calculate_roi
import imageQC.scripts.calculate_qc as calculate_qc
import imageQC.resources

'''during testing:
import matplotlib.pyplot as plt
imgplot = plt.imshow(image) / plt.scatter(x, y)
plt.show()
'''
path_tests = Path(__file__).parent
path_src_imageQC = path_tests.parent / 'src' / 'imageQC'


def read_tag_infos_from_yaml():
    """Get DICOM tags from tag_infos.yaml if tag_infos.yaml do not exist yet.

    Returns
    -------
    tag_infos : list of TagInfo
    """
    tag_infos = []
    file_path = path_src_imageQC / 'config_defaults' / 'tag_infos.yaml'
    with open(file_path, "r") as f:
        docs = yaml.safe_load_all(f)
        for doc in docs:
            tag_infos.append(cfc.TagInfo(**doc))

    return tag_infos


tag_infos = read_tag_infos_from_yaml()


def test_CTn():
    input_main = InputMain(
        current_modality='CT',
        current_test='CTn',
        current_paramset=cfc.ParamSetCT(),
        current_quicktest=cfc.QuickTestTemplate(tests=[['CTn']]),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'CT' / 'CTP404_FOV150' / 'CTP404_FOV150_001.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)

    assert round(input_main.results['CTn']['values'][0][0]) == 931
    assert round(input_main.results['CTn']['values'][0][7]) == -1006


def test_CT_Sli_axial():
    """Test Sli axial Catphan."""
    input_main = InputMain(
        current_modality='CT',
        current_test='Sli',
        current_paramset=cfc.ParamSetCT(),
        current_quicktest=cfc.QuickTestTemplate(tests=[['Sli']]),
        tag_infos=tag_infos,
        automation_active=False,
        )

    file_path = (
        path_tests / 'test_inputs' / 'CT' / 'CTP404_FOV150' / 'CTP404_FOV150_001.dcm')
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values10 = np.round(10.*np.array(input_main.results['Sli']['values'][0]))
    assert np.array_equal(
        values10, np.array([48., 44., 46., 46., 46., 46., -49.]))


def test_CT_Rin():
    input_main = InputMain(
        current_modality='CT',
        current_test='Rin',
        current_paramset=cfc.ParamSetCT(rin_sigma_image=2.),
        current_quicktest=cfc.QuickTestTemplate(tests=[['Rin']]),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = (
        path_tests / 'test_inputs' / 'CT' / 'CTP486' / 'CTP486_005.dcm')
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values = np.round(10*np.array(input_main.results['Rin']['values'][0]))
    assert np.array_equal(values, np.array([-3., 4.]))


def test_CT_Dim():
    input_main = InputMain(
        current_modality='CT',
        current_test='Dim',
        current_paramset=cfc.ParamSetCT(),
        current_quicktest=cfc.QuickTestTemplate(tests=[['Dim']]),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = (
        path_tests / 'test_inputs' / 'CT' / 'CTP404_FOV150' / 'CTP404_FOV150_001.dcm')
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values = np.round(np.array(input_main.results['Dim']['values'][0]))
    assert np.array_equal(
        values, np.array([50., 50., 50., 50., 71., 71.]))


def test_CT_Sli_helical():
    """Test Sli helical Catphan."""
    input_main = InputMain(
        current_modality='CT',
        current_test='Sli',
        current_paramset=cfc.ParamSetCT(sli_type=1),
        current_quicktest=cfc.QuickTestTemplate(tests=[['Sli']]),
        tag_infos=tag_infos,
        automation_active=False,
        )

    file_path = (
        path_tests / 'test_inputs' / 'CT' / 'CTP591_06mm' / 'CTP591_06mm_023.dcm')
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values10 = np.round(10.*np.array(input_main.results['Sli']['values'][0]))
    assert np.array_equal(values10, np.array([6., 17., 13., 11., 14.,  7.,  7.]))


def test_CT_MTF_bead():
    """Test MTF from circular edge on highest density (Teflon)."""
    input_main = InputMain(
        current_modality='CT',
        current_test='MTF',
        current_paramset=cfc.ParamSetCT(
            mtf_type=0,
            mtf_roi_size=3.,
            mtf_auto_center=True
            ),
        current_quicktest=cfc.QuickTestTemplate(tests=[['MTF']]),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'CT' / 'Wire_FOV50' / 'CT_wire_FOV50_001.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    assert len(input_main.results['MTF']['values'][0]) == 6
    values10 = np.round(10.*np.array(input_main.results['MTF']['values'][0]))
    assert np.array_equal(values10, np.array([11., 15., 19., 11., 15., 18.]))


def test_CT_Teflon_MTF_circular_edge_autocenter():
    """Test MTF from circular edge on highest density (Teflon)."""
    input_main = InputMain(
        current_modality='CT',
        current_test='MTF',
        current_paramset=cfc.ParamSetCT(
            mtf_auto_center=True,
            ),
        current_quicktest=cfc.QuickTestTemplate(tests=[['MTF']]),
        tag_infos=tag_infos,
        automation_active=False,
        )

    file_path = (path_tests / 'test_inputs' / 'CT' / 'CTP404_FOV150' /
                 'CTP404_FOV150_001.dcm')
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values100 = np.round(100.*np.array(input_main.results['MTF']['values'][0]))
    assert np.array_equal(values100, np.array([36., 66., 86.]))


def test_CT_MTF_circular_edge_not_optimal_offset():
    """Test MTF from circular edge on highest density (Teflon)."""
    file_path = (path_tests / 'test_inputs' / 'CT' / 'CTP404_FOV150' /
                 'CTP404_FOV150_001.dcm')
    dists = []
    vals = []

    centers = [[-105, 173], [-103, -173]]  # Teflon, Acrylic
    centerish = centers[1]
    diffs = [[0, 0], [6, 0], [4, 0], [2, 0], [0, 6], [0, 4], [0, 2], [6, 6]]
    offsets = [[centerish[0] + diff[0], centerish[1] + diff[1]] for diff in diffs]
    for offset in offsets:
        input_main = InputMain(
            current_modality='CT',
            current_test='MTF',
            current_paramset=cfc.ParamSetCT(
                mtf_auto_center=False,
                mtf_offset_xy=offset,
                ),
            current_quicktest=cfc.QuickTestTemplate(tests=[['MTF']]),
            tag_infos=tag_infos,
            automation_active=False,
            )
        img_infos, ignored_files = dcm.read_dcm_info(
            [file_path], GUI=False, tag_infos=input_main.tag_infos)
        input_main.imgs = img_infos
        calculate_qc.calculate_qc(input_main)

        dists.append(input_main.results['MTF']['details_dict'][0]['sorted_pixels_x'])
        vals.append(input_main.results['MTF']['details_dict'][0]['sorted_pixels'][0])

    # plt.plot(dists[0], vals[0], '.', markersize=2, color='red')
    values100 = np.round(100.*np.array(input_main.results['MTF']['values'][0]))
    assert values100[0] == 36.


def test_CT_Acrylic_MTF_circular_edge():
    """Test MTF from circular edge on lowest contrast (Acrylic)."""
    input_main = InputMain(
        current_modality='CT',
        current_test='MTF',
        current_paramset=cfc.ParamSetCT(
            mtf_offset_xy=[-115, -175],
            ),
        current_quicktest=cfc.QuickTestTemplate(tests=[['MTF'], ['MTF'], ['MTF']]),
        tag_infos=tag_infos,  # read_tag_infos_from_yaml(),
        automation_active=False,
        )

    file_path = path_tests / 'test_inputs' / 'CT' / 'CTP404_FOV150'
    file1 = file_path / 'CTP404_FOV150_001.dcm'
    file2 = file_path / 'CTP404_FOV150_002.dcm'
    file3 = file_path / 'CTP404_FOV150_003.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file1, file2, file3], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    assert len(input_main.results['MTF']['values'][0]) == 3


def test_CT_Polystyrene_MTF_circular_edge():
    """Test MTF from circular edge on negative contrast (Polystyrene)."""
    input_main = InputMain(
        current_modality='CT',
        current_test='MTF',
        current_paramset=cfc.ParamSetCT(
            mtf_offset_xy=[105, -175],
            ),
        current_quicktest=cfc.QuickTestTemplate(tests=[['MTF'], ['MTF'], ['MTF']]),
        tag_infos=tag_infos,  # read_tag_infos_from_yaml(),
        automation_active=False,
        )

    file_path = path_tests / 'test_inputs' / 'CT' / 'CTP404_FOV150'
    file1 = file_path / 'CTP404_FOV150_001.dcm'
    file2 = file_path / 'CTP404_FOV150_002.dcm'
    file3 = file_path / 'CTP404_FOV150_003.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file1, file2, file3], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    assert len(input_main.results['MTF']['values'][0]) == 3


def test_CT_NPS():
    input_main = InputMain(
        current_modality='CT',
        current_test='NPS',
        current_paramset=cfc.ParamSetCT(),
        current_quicktest=cfc.QuickTestTemplate(tests=[['NPS']]),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = (
        path_tests / 'test_inputs' / 'CT' / 'CTP486' / 'CTP486_005.dcm')
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values = np.round(10*np.array(input_main.results['NPS']['values'][0]))
    assert np.array_equal(values, np.array([3., 245., 463., 131.,  68.]))


def test_Xray_NPS():
    input_main = InputMain(
        current_modality='Xray',
        current_test='NPS',
        current_paramset=cfc.ParamSetXray(),
        current_quicktest=cfc.QuickTestTemplate(tests=[['NPS']]),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = (
        path_tests / 'test_inputs' / 'Xray' / 'hom.dcm')
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values = np.round(np.array(input_main.results['NPS']['values'][0]))
    assert np.array_equal(values, np.array([8.2e+01, 1.1e+03, 1.7e+01, 1.0e+00]))


def test_Xray_MTF_autocenter():
    """Test MTF from combined edges of auto_detected rectangle object."""
    input_main = InputMain(
        current_modality='Xray',
        current_test='MTF',
        current_paramset=cfc.ParamSetXray(mtf_auto_center=True, mtf_auto_center_type=0),
        current_quicktest=cfc.QuickTestTemplate(tests=[['MTF']]),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'Xray' / 'mtf.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    assert len(input_main.results['MTF']['values'][0]) == 6
    values10 = np.round(10.*np.array(input_main.results['MTF']['values'][0]))
    assert np.array_equal(values10, np.array([8., 6., 4., 3., 2., 12.]))


def test_Xray_Var():
    input_main = InputMain(
        current_modality='Xray',
        current_test='Var',
        current_paramset=cfc.ParamSetXray(),
        current_quicktest=cfc.QuickTestTemplate(tests=[['Var']]),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = (
        path_tests / 'test_inputs' / 'Xray' / 'hom.dcm')
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values = np.round(np.array(input_main.results['Var']['values'][0]))
    assert np.array_equal(values, np.array([ 31., 192.,  80.]))


def test_NM_uniformity():

    input_main = InputMain(
        current_modality='NM',
        current_test='Uni',
        current_paramset=cfc.ParamSetNM(
            uni_correct=True,
            uni_correct_pos_x=True,
            uni_correct_pos_y=True
            ),
        current_quicktest=cfc.QuickTestTemplate(tests=[['Uni'],['Uni']]),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'NM' / 'point_source_short_dist.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values = np.round(np.array(input_main.results['Uni']['values'][0]))
    assert np.array_equal(values, np.array([3., 2., 2., 1.]))


def test_NM_uniformity_sum():

    input_main = InputMain(
        current_modality='NM',
        current_test='Uni',
        current_paramset=cfc.ParamSetNM(uni_sum_first=True, uni_ufov_ratio=0.9),
        current_quicktest=cfc.QuickTestTemplate(
            tests=[['']] + [['Uni']]*65 + [['']] + [['Uni']]*65,
            ),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'NM' / 'sweep.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)

    assert round(input_main.results['Uni']['values'][0][0]) == 6


def test_NM_MTF_pointsource():
    tests = [[]] * 77
    tests[10] = ['MTF']
    input_main = InputMain(
        current_modality='NM',
        current_test='MTF',
        current_paramset=cfc.ParamSetNM(
            mtf_type=0, mtf_auto_center=True),
        current_quicktest=cfc.QuickTestTemplate(tests=tests),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'SPECT' / 'linesource_tomo_recon.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values = np.round(10*np.array(input_main.results['MTF']['values'][10]))
    assert np.array_equal(values, np.array([92., 169., 95., 174.]))


def test_NM_MTF_2_linesources():

    input_main = InputMain(
        current_modality='NM',
        current_test='MTF',
        current_paramset=cfc.ParamSetNM(
            mtf_type=2, mtf_auto_center=True,
            mtf_roi_size_x=40, mtf_roi_size_y=20),
        current_quicktest=cfc.QuickTestTemplate(tests=[['MTF']]),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'NM' / 'linesource_planar.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values = np.round(10*np.array(input_main.results['MTF']['values'][0]))
    assert np.array_equal(values, np.array([73., 134.,  74., 135.]))


def test_SPECT_MTF_linesource():
    """Test MTF 3d line."""
    input_main = InputMain(
        current_modality='SPECT',
        current_test='MTF',
        current_paramset=cfc.ParamSetSPECT(
            mtf_type=1,
            mtf_auto_center=True
            ),
        current_quicktest=cfc.QuickTestTemplate(tests=[['MTF']]*77),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'SPECT' / 'linesource_tomo_recon.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    assert len(input_main.results['MTF']['values'][0]) == 4
    values10 = np.round(10.*np.array(input_main.results['MTF']['values'][0]))
    assert np.array_equal(values10, np.array([93., 169.,  93., 170.]))


def test_PET_Cro():
    input_main = InputMain(
        current_modality='PET',
        current_test='Cro',
        current_paramset=cfc.ParamSetPET(),
        current_quicktest=cfc.QuickTestTemplate(tests=[['Cro']]*6),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'PET' / 'waterphantom'
    no = ['007', '010', '034', '035', '036']
    files = [file_path / f'PT_PETWB_{no[i]}.dcm' for i in range(5)]
    img_infos, ignored_files = dcm.read_dcm_info(
        files, GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values = np.round(np.array(input_main.results['Cro']['values'][0][3:]))
    assert np.array_equal(
        values,
        np.array([7.3000e+01, 1.1599e+04, 1.1835e+04, 1.0000e+00, 1.0000e+00]))


def test_MR_SNR():

    tests = [[''] for x in range(11)]
    tests[2] = ['SNR']
    tests[9] = ['SNR']
    input_main = InputMain(
        current_modality='MR',
        current_test='SNR',
        current_paramset=cfc.ParamSetMR(),
        current_quicktest=cfc.QuickTestTemplate(tests=tests),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = (
        path_tests / 'test_inputs' / 'MR' / 'ACR.dcm')
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values = np.round(np.array(input_main.results['SNR']['values'][2]))
    assert np.array_equal(
        values, np.array([1795., 1803., 1799., 52., 49.]))


def test_MR_Geo():

    tests = [[''] for x in range(11)]
    tests[5] = ['Geo']
    input_main = InputMain(
        current_modality='MR',
        current_test='Geo',
        current_paramset=cfc.ParamSetMR(),
        current_quicktest=cfc.QuickTestTemplate(tests=tests),
        tag_infos=tag_infos,
        automation_active=False
        )

    file_path = (
        path_tests / 'test_inputs' / 'MR' / 'ACR.dcm')
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    values = np.round(np.array(input_main.results['Geo']['values'][5]))
    assert np.array_equal(
        values[:4], np.array([189., 189., 189., 189.]))
