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

n_imgs = 2
n_rep = 3
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
    sni_n_sample_noise = 5,
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

    art_list = [
        '',
        'rad30 x150 y0', 'rad30 x0 y0', 'rad30 x150 y130 s30',
        'rad15 x150 y0', 'rad15 x0 y0', 'rad15 x150 y130',
        'rad45 x150 y0', 'rad45 x0 y0', 'rad45 x150 y130',
        'pt3 x150 y0', 'pt3 x0 y0', 'pt3 x150 y130',
        'pt3+ x150 y0', 'pt3+ x0 y0', 'pt3+ x150 y130',
        'line']
    headers = ['IU_UFOV', 'DU_UFOV', 'IU_CFOV', 'DU_CFOV',
               'SNI L1 low', 'SNI L2 low', 'SNI S low max','SNI S low avg',
               'SNI L1 high', 'SNI L2 high', 'SNI S high max', 'SNI S high avg']
    n_metrics = len(headers)

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

            rows = []
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
                row = res_uni.values + res_sni.values
                rows.append([str(i) for i in row])
                print('.', sep='', end='', flush=True)

            with open(auto_temp.path_output, "w") as file:
                file.write('\t'.join(headers)+'\n')
                for row in rows:
                    file.write('\t'.join(row)+'\n')

    folder = Path(path_output) / sim
    res_path = folder / 'results.txt'
    res_rows = []
    cnt_folders = [x for x in folder.glob('*')
                   if (x.is_dir() and 'val' in str(x))]
    for folder_this in cnt_folders:
        ref_file = folder_this / 'ref.txt'
        if ref_file.exists():
            files = [x for x in folder_this.glob('*')
                     if (x.is_file() and 'ref' not in str(x))]
            df_ref = pd.read_csv(
                ref_file, sep='\t', decimal=decimal_mark, encoding='ISO-8859-1')
            ref_vals = df_ref.values.T.astype('float64')

            for file in files:
                df = pd.read_csv(
                    file, sep='\t', decimal=decimal_mark, encoding='ISO-8859-1')
                vals = df.values.T.astype('float64')
                for i, vals_this in enumerate(vals):
                    refs_this = ref_vals[i]
                    mean_this = np.mean(vals_this)
                    sd_this = np.std(vals_this)
                    sd_ref = np.std(refs_this)
                    mean_ref = np.mean(refs_this)
                    res_rows.append(
                        [file.parts[-2], file.parts[-1], headers[i],
                         str(mean_this), str(mean_ref), str(sd_this), str(sd_ref),
                         str((mean_this - mean_ref)/np.sqrt((sd_this**2 + sd_ref**2)/2))])
    headers = ['val', 'filename', 'metric', 'avg', 'avg ref',
               'std', 'std ref', 'Cohens d']

    with open(res_path, "w") as file:
        file.write('\t'.join(headers) + '\n')
        for row in res_rows:
            row_string = '\t'.join(row) + '\n'
            row_string = row_string.replace('.txt', '').replace('.', decimal_mark)
            file.write(row_string)

    res_max_path = folder / 'results_max_cohensd.txt'
    headers = ['val', 'filename', 'max Cohens d uni', 'max Cohens d sni']
    n_art = len(art_list) - 1
    res_max_rows = []
    for i, val in enumerate(count_values):
        for j in range(n_art):
            start = i*n_art*n_metrics + j*n_metrics
            cohens_this = [float(row[-1]) for row
                           in res_rows[start:start + n_metrics]]
            max_cohen_uni = np.max(cohens_this[:4])
            max_cohen_sni = np.max(cohens_this[4:])
            res_max_rows.append(
                [res_rows[start][0], res_rows[start][1].replace('.txt', ''),
                 str(max_cohen_uni), str(max_cohen_sni)])
    
    with open(res_max_path, "w") as file:
        file.write('\t'.join(headers) + '\n')
        for row in res_max_rows:
            row_string = '\t'.join(row) + '\n'
            row_string = row_string.replace('.', decimal_mark)
            file.write(row_string)

    param_list_uni = [f'{param}: {str(getattr(paramset, param))}' for param in dir(paramset) if 'uni' in param]
    param_list_sni = [f'{param}: {str(getattr(paramset, param))}' for param in dir(paramset) if 'sni' in param]
    params_path = folder / 'params.txt'
    with open(params_path, "w") as file:
        file.write('\n'.join(param_list_uni) + '\n')
        file.write('\n'.join(param_list_sni))

    print('Finsihed')
