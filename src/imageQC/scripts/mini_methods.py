#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of small functions used in ImageQC.

@author: Ellen Wasbo
"""
import os
from fnmatch import fnmatch
from pathlib import Path
from PyQt5.QtWidgets import QMessageBox

# imageQC block start
from imageQC.config.iQCconstants import QUICKTEST_OPTIONS
from imageQC.ui import messageboxes
# imageQC block end


def string_to_float(string_value):
    """Convert string to float, accept comma as decimal-separator.

    Parameters
    ----------
    string_value : str

    Returns
    -------
    output : float or None
    """
    output = None
    if isinstance(string_value, str):
        string_value = string_value.replace(',', '.')
        try:
            output = float(string_value)
        except ValueError:
            pass
    return output


def get_uniq_ordered(input_list):
    """Get uniq elements of a group in same order as first appearance."""
    output_list = []
    for elem in input_list:
        if elem not in output_list:
            output_list.append(elem)
    return output_list


def get_all_matches(input_list, value, wildcards=False):
    """Get all matches of value in input_list.

    Parameters
    ----------
    input_list : list of object
    value : object
        Same type as input_list elements
    wildcards : bool, optional
        If true, use fnmatch to include wildcards */?. The default is False.

    Returns
    -------
    index_list : list of int
        list of indexes in input_list where value is found

    """
    index_list = []
    if wildcards and isinstance(value, str):
        index_list = [idx for idx, val in enumerate(input_list) if fnmatch(val, value)]
    else:
        index_list = [idx for idx, val in enumerate(input_list) if val == value]

    return index_list


def find_value_in_sublists(input_list, value):
    """Get all matches of value in nested input_list.

    Parameters
    ----------
    input_list : list of object
    value : str or number
        Same type as input_list elements

    Returns
    -------
    sublist_ids : list of int
        list of indexes of sublist in input_list where value is found

    """
    sublist_ids = []
    for i, sub in enumerate(input_list):
        if value in sub:
            sublist_ids.append(i)

    return sublist_ids


def get_included_tags(modality, tag_infos, avoid_special_tags=False):
    """Get tags from tag_infos general + modality specific.

    Parameters
    ----------
    modality : str
    tag_infos : TagInfos
    avoid_special_tags : bool, optional
        Flag to avoid tags with .tag[1] == ''. Default is False.

    Returns
    -------
    general_tags : list of str
        tag names for general tags
    included_tags : list of str
        tag names for all included tags
    """
    general_tags = []
    included_tags = []
    for tagid in range(len(tag_infos)):
        include = bool(
            set([modality, '']).intersection(
                tag_infos[tagid].limited2mod))
        if include and avoid_special_tags:
            if tag_infos[tagid].tag[1] == '':
                include = False
        if include:
            attr_name = tag_infos[tagid].attribute_name
            if attr_name not in included_tags:
                included_tags.append(attr_name)
            if tag_infos[tagid].limited2mod[0] == '':
                if attr_name not in general_tags:
                    general_tags.append(attr_name)

    return (general_tags, included_tags)


def create_empty_file(filepath, parent_widget, proceed_info_txt='', proceed=False):
    """Ask to create empty file if not existing path."""
    if not os.path.exists(filepath):
        if proceed is False:
            proceed = messageboxes.proceed_question(
                parent_widget, f'{proceed_info_txt} Proceed creating an empty file?')
        if proceed:
            try:
                with open(filepath, "w") as file:
                    file.write('')
            except (OSError, IOError) as error:
                QMessageBox.warning(
                    parent_widget, 'Error',
                    f'Failed creating the file {error}.')


def create_empty_folder(folderpath, parent_widget, proceed_info_txt=''):
    """Ask to create empty folder if not existing path."""
    if not os.path.exists(folderpath):
        proceed = messageboxes.proceed_question(
            parent_widget, f'{proceed_info_txt} Proceed creating an empty folder?')
        if proceed:
            try:
                Path(folderpath).mkdir(parents=True)
            except (NotADirectoryError, FileNotFoundError, OSError) as error:
                QMessageBox.warning(
                    parent_widget, 'Error',
                    f'Failed creating the folder {error}.')


def get_modality_index(modality_string):
    """Get index of given modality string.

    Parameters
    ----------
    modality_string : str
        modality as defined in imageQC (CT, Xray...)

    Returns
    -------
    int
        index of given modality
    """
    mods = [*QUICKTEST_OPTIONS]
    return mods.index(modality_string)


def get_headers_first_values_in_path(path):
    """Get headers and first row values from output path."""
    headers = []
    first_values = None
    if os.path.exists(path):
        with open(path) as f:
            headers = f.readline().strip('\n').split('\t')
            first_values = f.readline().strip('\n').split('\t')
        if len(headers) > 0:
            headers.pop(0)  # date not included
            if first_values is not None:
                first_values.pop(0)
    return headers, first_values
