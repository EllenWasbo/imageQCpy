# -*- coding: utf-8 -*-
"""
Tests on automation.

@author: ewas
"""
import yaml
import numpy as np
from pathlib import Path

import imageQC.config.config_classes as cfc
from imageQC.scripts.input_main_auto import InputMain
import imageQC.scripts.calculate_qc as calculate_qc
import imageQC.resources
import imageQC.scripts.automation as automation

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


def test_validate_args():
    """Test reading codeline arguments for starting imageQC with automation."""

    sysargs = [
        ['...py', '-i', '-a'],
        ['...py', '-import', '-ndays=7', '-dicom'],
        ['...py', '-d', 'Xray', 'CT', 'NM/label1'],
        ['...py', '-a', 'Xray', 'CT', 'NM/label1', 'SPECT/sss', 'MR/mmm'],
        ['...py', '-f', 'lol', '///', 'pan/cakes', 'is=nice'],
    ]

    expected_res = [
        [True, ['-i', '-a'], -1, [], []],
        [True, ['-i', '-d'], 7, [], []],
        [True, ['-d'], -1, ['Xray', 'CT'], [['NM', 'label1']]],
        [False, ['-a'], -1, ['Xray', 'CT'],
         [['NM', 'label1'], ['SPECT', 'sss'], ['MR', 'mmm']]],
        [False, [], -1, [], []],
    ]

    for i in range(len(sysargs)):
        ok, args, ndays, mods, temps, msgs = automation.validate_args(sysargs[i])
        res = [ok, args, ndays, mods, temps]

        assert expected_res[i] == res
