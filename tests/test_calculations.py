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
from imageQC.scripts.input_main_no_gui import InputMain
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


def test_CTn():
    input_main = InputMain(
        current_modality='CT',
        current_test='CTn',
        current_paramset=cfc.ParamSetCT(),
        current_quicktest=cfc.QuickTestTemplate(
            tests=[['CTn']],
            image_names=[''],
            group_names=['']
            ),
        tag_infos=read_tag_infos_from_yaml(),
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'CT' / 'CTP404_FOV150' / 'CTP404_FOV150_001.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)

    assert round(input_main.results['CTn']['values'][0][0]) == 931
    assert round(input_main.results['CTn']['values'][0][7]) == -1006


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
        current_quicktest=cfc.QuickTestTemplate(
            tests=[['MTF']],
            image_names=[''],
            group_names=['']
            ),
        tag_infos=read_tag_infos_from_yaml(),
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'CT' / 'Wire_FOV50' / 'CT_wire_FOV50_001.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    assert len(input_main.results['MTF']['values'][0]) == 6
    values10 = np.round(10.*np.array(input_main.results['MTF']['values'][0]))
    assert np.array_equal(values10, np.array([11., 15., 18., 11., 15., 19.]))


def test_CT_Teflon_MTF_circular_edge():
    """Test MTF from circular edge on highest density (Teflon)."""
    input_main = InputMain(
        current_modality='CT',
        current_test='MTF',
        current_paramset=cfc.ParamSetCT(
            mtf_auto_center=True,
            mtf_offset_xy=[-105, 175],
            ),
        current_quicktest=cfc.QuickTestTemplate(
            tests=[['MTF']],
            image_names=[''],
            group_names=[''],
            ),
        tag_infos=read_tag_infos_from_yaml(),
        automation_active=False,
        )

    file_path = path_tests / 'test_inputs' / 'CT' / 'CTP404_FOV150' / 'CTP404_FOV150_001.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    #TODO not tested yet
    assert round(input_main.results['MTF']['values'][0][0]) == 1
    assert round(input_main.results['MTF']['values'][0][1]) == 1


def test_CT_Acrylic_MTF_circular_edge():
    """Test MTF from circular edge on lowest contrast (Acrylic)."""
    input_main = InputMain(
        current_modality='CT',
        current_test='MTF',
        current_paramset=cfc.ParamSetCT(
            mtf_offset_xy=[-115, -175],
            ),
        current_quicktest=cfc.QuickTestTemplate(
            tests=[['MTF'], ['MTF'], ['MTF']],
            image_names=[['']] * 3,
            group_names=[['']] * 3,
            ),
        tag_infos=read_tag_infos_from_yaml(),
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
    #TODO not tested yet
    assert round(input_main.results['MTF']['values'][0][0]) == 1
    assert round(input_main.results['MTF']['values'][0][1]) == 1


def test_CT_Polystyrene_MTF_circular_edge():
    """Test MTF from circular edge on negative contrast (Polystyrene)."""
    input_main = InputMain(
        current_modality='CT',
        current_test='MTF',
        current_paramset=cfc.ParamSetCT(
            mtf_offset_xy=[105, -175],
            ),
        current_quicktest=cfc.QuickTestTemplate(
            tests=[['MTF'], ['MTF'], ['MTF']],
            image_names=[['']] * 3,
            group_names=[['']] * 3,
            ),
        tag_infos=read_tag_infos_from_yaml(),
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
    #TODO not tested yet
    assert round(input_main.results['MTF']['values'][0][0]) == 1
    assert round(input_main.results['MTF']['values'][0][1]) == 1


def test_Xray_MTF_autocenter():
    """Test MTF from combined edges of auto_detected rectangle object."""
    input_main = InputMain(
        current_modality='Xray',
        current_test='MTF',
        current_paramset=cfc.ParamSetXray(mtf_auto_center=True, mtf_auto_center_type=0),
        current_quicktest=cfc.QuickTestTemplate(
            tests=[['MTF']],
            image_names=[''],
            group_names=['']
            ),
        tag_infos=read_tag_infos_from_yaml(),
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'Xray' / 'mtf.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)
    assert len(input_main.results['MTF']['values'][0]) == 6
    values10 = np.round(10.*np.array(input_main.results['MTF']['values'][0]))
    assert np.array_equal(values10, np.array([11., 15., 18., 11., 15., 19.]))



def test_NM_uniformity():

    tag_infos = read_tag_infos_from_yaml()
    file_path = path_tests / 'test_inputs' / 'NM' / 'point_source_short_dist.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=tag_infos)
    image, tags = dcm.get_img(file_path, frame_number=0)
    image_dict = img_infos[0]

    roi_array = calculate_roi.get_ratio_NM(
        image, image_dict, ufov_ratio=0.95, cfov_ratio=0.75)
    res = calculate_qc.get_corrections_point_source(
            image, image_dict, roi_array[0],
            fit_x=True, fit_y=True, lock_z=-1.)
    values = calculate_qc.calculate_NM_uniformity(
        res['corrected_image'], roi_array, image_dict.pix[0])

    assert res['distance'] > 0


def test_NM_uniformity_sum():

    input_main = InputMain(
        current_modality='NM',
        current_test='Uni',
        current_paramset=cfc.ParamSetNM(uni_sum_first=True, uni_ufov_ratio=0.9),
        current_quicktest=cfc.QuickTestTemplate(
            tests=[['']] + [['Uni']]*65 + [['']] + [['Uni']]*65,
            image_names=[''] * 132,
            group_names=[''] * 132
            ),
        tag_infos=read_tag_infos_from_yaml(),
        automation_active=False
        )

    file_path = path_tests / 'test_inputs' / 'NM' / 'sweep.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=input_main.tag_infos)
    input_main.imgs = img_infos

    calculate_qc.calculate_qc(input_main)

    print(input_main.results['Uni']['values'])
    assert round(input_main.results['Uni']['values'][0][0]) == 6


def test_NM_SNI():

    tag_infos = read_tag_infos_from_yaml()
    file_path = path_tests / 'test_inputs' / 'NM' / 'point_source_short_dist.dcm'
    img_infos, ignored_files = dcm.read_dcm_info(
        [file_path], GUI=False, tag_infos=tag_infos)
    image, tags = dcm.get_img(file_path, frame_number=0)
    image_dict = img_infos[0]

    paramset = cfc.ParamSetNM()
    roi_array = calculate_roi.get_roi_SNI(
        image, image_dict, paramset)
    res = calculate_qc.get_corrections_point_source(
            image, image_dict, roi_array[0],
            fit_x=True, fit_y=True, lock_z=-1.)
    values = calculate_qc.calculate_NM_SNI(
        res['corrected_image'], roi_array, image_dict.pix[0])

    assert res['distance'] > 0