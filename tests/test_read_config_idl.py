# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:03:20 2022

@author: ellen
"""
import os

import imageQC.resources
from imageQC.config.iQCconstants import ENV_CONFIG_FOLDER
from imageQC.config.read_config_idl import ConfigIdl2Py

os.environ[ENV_CONFIG_FOLDER] = ''

def test_read_config_idl():
    """Test reading config.dat from IDL version of ImageQC."""
    dir_tests = os.path.dirname(__file__)
    file_names = ['test_config_tag2.dat']

    expected_res = []

    for i, f in enumerate(file_names):
        p = os.path.join(dir_tests, 'test_inputs', 'config_idl', f)
        res = ConfigIdl2Py(p)
        assert res == expected_res#[i]
