# -*- coding: utf-8 -*-
"""
Tests on GUI.

@author: ewas
"""
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from imageQC.scripts.artifact import Artifact, apply_artifacts
from imageQC.scripts.calculate_roi import get_rois
from imageQC.scripts.calculate_qc import calculate_2d, quicktest_output
import imageQC.scripts.dcm as dcm
from imageQC.scripts.input_main import InputMain
import imageQC.config.config_func as cff
import imageQC.config.config_classes as cfc
from imageQC.scripts.mini_methods import (
    get_all_matches, string_to_float, get_headers_first_values_in_path,
    find_files_prefix_suffix)
from imageQC.config.iQCconstants import ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER

n_imgs = 1
n_rep = 1
sim = 'sim flat'
corr_point = True if 'point' in sim else False
path_artifacts = r'C:\Users\ellen\CloudStation\ImageQCpy\DemoBilder\div_artifacts_alle_synlig.yaml'
path_applied = ''
path_output = r'C:\Users\ellen\CloudStation\ImageQCpy\DemoBilder\sim_SNI'
mod = 'NM'
decimal_mark = ','

auto_temp = cfc.AutoTemplate(
    label='test',
    path_input=r'C:\Users\ellen\CloudStation\ImageQCpy\DemoBilder\DemoBilder\NM\intevo_kalib_1rep',
    paramset_label='test', quicktemp_label='test', archive=False
    )

paramset = cfc.ParamSetNM(
    label='test',
    uni_ufov_ratio = 0.97, uni_correct = corr_point,
    sni_area_ratio = 0.97, sni_type = 2, sni_roi_size = 128,
    sni_sampling_frequency = 0.004, sni_ratio_dim = 0,
    sni_correct = corr_point,
    sni_channels = True,
    sni_channels_table = [[0.0, 0.1, 0.6], [0.05, 0.15, 0.6]],
    sni_alt = 3)

os.environ[ENV_USER_PREFS_PATH] = ''
os.environ[ENV_CONFIG_FOLDER] = ''

def load_artifacts():
    
    artifacts = []
    with open(path_artifacts, 'r') as file:
        docs = yaml.safe_load_all(file)
        for doc in docs:
            artifacts.append(Artifact(**doc))
    return artifacts


def load_artifacts_apply(artifact):
    all_applied = ['***set to zero***', sim, 'noise']
    if artifact:
        all_applied.insert(2, artifact)
    listoflists = [all_applied] * n_imgs

    '''
    listoflists = []
    with open(path_applied, 'r') as file:
        docs = yaml.safe_load_all(file)
        listoflists = [doc for doc in docs]
    '''

    return listoflists


def test_autores():
    _, _, tag_infos = cff.load_settings(fname='tag_infos')
    paramsets = {mod: [paramset]}
    artifacts = load_artifacts()

    p_input = Path(auto_temp.path_input)
    files = [x for x in p_input.glob('*') if x.is_file()]
    img_infos, _, _ = dcm.read_dcm_info(files, GUI=False, tag_infos=tag_infos)
    input_main = InputMain(
        imgs=img_infos,
        current_modality=mod,
        current_paramset=paramset,
        #current_quicktest=qt_template,
        digit_templates=None,
        tag_infos=tag_infos,
        artifacts=artifacts)
    output_headers, _ = get_headers_first_values_in_path(
        auto_temp.path_output)

    art_list = ['', 'rad30 x0 y0', 'line']

    image2d = np.zeros((1024, 1024), dtype='float64')
    artifacts_apply = load_artifacts_apply('')
    input_main.imgs[0].artifacts = artifacts_apply[0]
    image2d = apply_artifacts(
        image2d, input_main.imgs[0],
        input_main.artifacts, None, 0)

    input_main.current_test = 'Uni'
    roi_array_uni, _ = get_rois(image2d, 0, input_main)
    input_main.current_test = 'SNI'
    roi_array_sni, _ = get_rois(image2d, 0, input_main)

    count_values = [25, 50]#], 100]
    res_uni, res_sni = [], []
    for val in count_values:
        if val != 25:
            # set sim image value
            labels = [art.label for art in artifacts]
            idx = labels.index(sim)
            artifacts[idx].value = val
        for art in art_list:
            print(f'starting artifact {art}')
            if art == '':
                output_this = Path(path_output) / sim / f'val{val}' / 'ref.txt'
            else:
                output_this = Path(path_output) / sim / f'val{val}' / f'{art}.txt'
            with open(output_this, "w") as file:
                file.write('')
            auto_temp.path_output = output_this.resolve()
            artifacts_apply = load_artifacts_apply(art)
            input_main.imgs[0].artifacts = artifacts_apply[0]

            for i in range(n_rep):
                input_main.results = {}
                image2d = np.zeros((1024, 1024), dtype='float64')
                image2d = apply_artifacts(
                    image2d, input_main.imgs[0],
                    input_main.artifacts, None, 0)
                res_uni = calculate_2d(image2d, roi_array_uni, input_main.imgs[0],
                             mod, paramset, 'Uni', [0, 0, 0], None, None)
                res_sni = calculate_2d(image2d, roi_array_sni, input_main.imgs[0],
                             mod, paramset, 'SNI', [0, 0, 0], None, None)
    assert np.array_equal(
        np.round(res_uni.values), np.array([2., 2., 2., 2.]))
