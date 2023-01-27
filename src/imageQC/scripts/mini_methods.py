#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of small functions used in ImageQC.

@author: Ellen Wasbo
"""
import numpy as np
from fnmatch import fnmatch


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


def get_min_max_pos_2d(image, roi_array):
    """Get position of first min and max value in image.

    Parameters
    ----------
    image : np.array
    roi_array : np.array
        dtype bool

    Returns
    -------
    list
        min_idx, max_idx
    """
    arr = np.ma.masked_array(image, mask=np.invert(roi_array))
    min_idx = np.where(arr == np.min(arr))
    max_idx = np.where(arr == np.max(arr))

    return [
        [min_idx[0][0], min_idx[1][0]],
        [max_idx[0][0], max_idx[1][0]]
        ]