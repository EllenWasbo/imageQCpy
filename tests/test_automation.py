# -*- coding: utf-8 -*-
"""
Tests on automation.

@author: ewas
"""
import imageQC.resources
import imageQC.scripts.automation as automation



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
