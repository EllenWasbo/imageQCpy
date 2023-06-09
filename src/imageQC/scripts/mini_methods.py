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
                with open(filepath, "w") as f:
                    f.write('')
            except IOError as ex:
                QMessageBox.warning(
                    parent_widget, 'Error',
                    f'Failed creating the file {ex}.')


def create_empty_folder(folderpath, parent_widget, proceed_info_txt=''):
    """Ask to create empty folder if not existing path."""
    if not os.path.exists(folderpath):
        proceed = messageboxes.proceed_question(
            parent_widget, f'{proceed_info_txt} Proceed creating an empty folder?')
        if proceed:
            try:
                Path(folderpath).mkdir(parents=True)
            except (NotADirectoryError, FileNotFoundError) as ex:
                QMessageBox.warning(
                    parent_widget, 'Error',
                    f'Failed creating the folder {ex}.')


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
