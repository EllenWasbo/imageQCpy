# -*- coding: utf-8 -*-
"""
Tests on GUI.

@author: ewas
"""
import os
from pathlib import Path
import yaml

from imageQC.scripts.artifact import Artifact
import imageQC.config.config_func as cff
import imageQC.config.config_classes as cfc
from imageQC.scripts.automation import run_template
from imageQC.config.iQCconstants import ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER

n_imgs = 2
n_rep = 50
sim = 'sim flat'
val = 50
path_artifacts = r'C:\Users\ellen\CloudStation\ImageQCpy\DemoBilder\div_artifacts_alle_synlig.yaml'
path_applied = ''
path_output = r'C:\Users\ellen\CloudStation\ImageQCpy\DemoBilder\sim_SNI'
mod = 'NM'
auto_temp = cfc.AutoTemplate(
    label='test',
    path_input=r'C:\Users\ellen\CloudStation\ImageQCpy\DemoBilder\DemoBilder\NM\intevo_kalib_1rep',
    paramset_label='test', quicktemp_label='test', archive=False
    )
output_temp = cfc.QuickTestOutputTemplate(
    decimal_mark=',',
    tests={
        'Uni': [cfc.QuickTestOutputSub()],
        'SNI': [cfc.QuickTestOutputSub(alternative=3)]
        }
    )
paramset = cfc.ParamSetNM(
    label='test',
    output=output_temp,
    uni_ufov_ratio = 0.97, uni_correct = False,
    sni_area_ratio = 0.97, sni_type = 2, sni_roi_size = 128,
    sni_sampling_frequency = 0.004, sni_ratio_dim = 1,
    sni_correct = False,
    sni_n_sample_noise = 5,
    sni_channels = True,
    sni_channels_table = [[0.0, 0.1, 0.6], [0.05, 0.15, 0.6]],
    sni_alt = 3)
qt_template = cfc.QuickTestTemplate(
    label='test',
    tests=[['Uni', 'SNI'], ['Uni', 'SNI']])
os.environ[ENV_USER_PREFS_PATH] = ''
os.environ[ENV_CONFIG_FOLDER] = ''

def load_artifacts():
    
    artifacts = []
    with open(path_artifacts, 'r') as file:
        docs = yaml.safe_load_all(file)
        for doc in docs:
            artifacts.append(Artifact(**doc))
    if val != 25:
        labels = [art.label for art in artifacts]
        idx = labels.index(sim)
        artifacts[idx].value = val
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
    print('start')
    _, _, tag_infos = cff.load_settings(fname='tag_infos')
    print('loaded tag_infos')
    paramsets = {mod: [paramset]}
    qt_templates ={mod: [qt_template]}
    artifacts = load_artifacts()
    print('loaded artifacts')

    art_list = [
        '',
        'rad30 x150 y0', 'rad30 x0 y0', 'rad30 x150 y130 s30',
        'rad15 x150 y0', 'rad15 x0 y0', 'rad 15 x150 y130',
        'rad45 x150 y0', 'rad45 x0 y0', 'rad45 x150 y130',
        'pt3 x150 y0', 'pt3 x0 y0', 'pt3 x150 y130 ',
        'pt3+ x150 y0', 'pt3+ x0 y0', 'pt3+ x150 y130 ',
        'line']

    for art in art_list:
        output_this = Path(path_output) / sim / f'val{val}' / f'{art}.txt'
        with open(output_this, "w") as file:
            file.write('')
        auto_temp.path_output = output_this.resolve()
        artifacts_apply = load_artifacts_apply(art)
        for i in range(n_rep):
            _ = run_template(
                auto_temp, mod, paramsets, qt_templates,
                None, None, tag_infos, None, artifacts, artifacts_apply)
            print(f'{art} {i}')
