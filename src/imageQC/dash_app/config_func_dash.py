#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions used for configuration settings.

@author: Ellen Wasbo
"""
import os
from pathlib import Path
import yaml


def convert_OneDrive(path):
    """Test wether C:Users and OneDrive in path - replace username if not correct.

    Option to use shortcut paths in OneDrive to same sharepoint
    (shortcut name must be the same).

    Parameters
    ----------
    path_string : str

    Returns
    -------
    path : str
        as input or replaced username if OneDrive shortcut
    """
    if 'OneDrive' in path and 'C:\\Users' in path:
        username = os.getlogin()
        path_obj = Path(path)
        if path_obj.exists() is False:
            if username != path_obj.parts[2]:
                path = os.path.join(*path_obj.parts[:2], username, *path_obj.parts[3:])
    return path


def load_paramset_decimarks(modalities, config_path):
    """Load paramset (modality_dict) as dict from yaml file in config folder.

    Keep only labels and decimal mark for use in dash_app.

    Parameters
    ----------
    modalities : list of str
        available modalities
    config_path : str or Path
        path to config folder with yaml files

    Returns
    -------
    dict
        keys modality string
        items list of dict {label, decimal_mark}
    """
    path = Path(config_path)
    fnames = [f'paramsets_{m}' for m in modalities]
    paramsets = {
        modality: [] for modality in modalities}

    for idx, fname in enumerate(fnames):
        try:
            with open(path / f'{fname}.yaml', 'r') as file:
                doc = yaml.safe_load(file)
                for temp in doc:
                    #TODO decimalmark only
                    paramsets[idx].append(temp)

        except Exception as error:
            print(f'config_func_dash.py load_paramset {fname}: {str(error)}')

    return paramsets


def load_settings(fname, config_path):
    """Load settings as dict from yaml file in config folder.

    Parameters
    ----------
    fname : str
        yaml filename without folder and extension
    config_path: str or Path
        path to config folder

    Returns
    -------
    dict
    """
    settings = {}
    path = Path(config_path) / f'{fname}.yaml'
    
    if 'dash' in fname:
        try:
            with open(path, 'r') as file:
                settings = yaml.safe_load(file)
        except Exception as error:
            print(f'config_func_dash.py load_settings: {str(error)}')
    else:

        try:
            with open(path, 'r') as file:
                docs = yaml.safe_load(file)
                for mod, doc in docs.items():
                    settings[mod] = []
                    for temp in doc:
                        settings[mod].append(temp)
                        '''
                        if 'auto' in fname:
                            try:
                                for attr in [
                                        'path_input',
                                        'path_output',
                                        'path_warnings']:
                                    new_path = convert_OneDrive(
                                        getattr(settings[mod][-1], attr))
                                    setattr(settings[mod][-1], attr, new_path)
                            except (IndexError, KeyError, AttributeError):
                                pass
                        '''
        except Exception as error:
            print(f'config_func_dash.py load_settings: {str(error)}')

    return settings

