# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:03:20 2022

@author: ellen
"""
import os
import yaml
from pathlib import Path

import imageQC.config.config_classes as cfc
from imageQC.config.read_config_idl import ConfigIdl2Py


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


def test_read_config_idl():
    """Test reading config.dat from IDL version of ImageQC."""
    dir_tests = os.path.dirname(__file__)
    file_name = 'added_dcm_tags.dat'
    p = os.path.join(dir_tests, 'test_inputs', 'config_idl', file_name)
    res = ConfigIdl2Py(p, read_tag_infos_from_yaml())
    expected_new_tags = ['TEST1', 'TEST2', 'TESTCT', 'TESTMR']
    assert [nti.attribute_name for nti in res.tag_infos_new] == expected_new_tags
    expected_rename_temps = ['DEFAULT', 'TESTNAME']
    assert [rt.label for rt in res.rename_patterns['CT']] == expected_rename_temps

    file_name = 'config_stripped.dat'
    expected_paramsets_CT = [
        'CONFIGDEFAULT', 'GE_CT_PHANTOM', 'AUTO_DEFAULT', 'SIEMENS_CT',
        'CT_SIEM_AARLIG', 'AUTO_NO_FILENAME', 'SIEMENS_CT_NOFILENAME']
    p = os.path.join(dir_tests, 'test_inputs', 'config_idl', file_name)
    res = ConfigIdl2Py(p, read_tag_infos_from_yaml())
    assert [ps.label for ps in res.paramsets['CT']] == expected_paramsets_CT

    file_name = 'config_empty.dat'
    p = os.path.join(dir_tests, 'test_inputs', 'config_idl', file_name)
    res = ConfigIdl2Py(p, read_tag_infos_from_yaml())
    expected_paramsets_CT = ['CONFIGDEFAULT']
    assert [ps.label for ps in res.paramsets['CT']] == expected_paramsets_CT
