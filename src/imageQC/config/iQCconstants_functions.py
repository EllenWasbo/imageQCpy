#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions used for iQCconstants on startup.

@author: Ellen Wasbo
"""
from copy import deepcopy
from PyQt5.QtCore import QFile, QTextStream
import yaml

# imageQC block start
import imageQC.config.config_classes as cfc
import imageQC.resources  # needed for read_tag_infos
# imageQC block end


def empty_template_dict(quicktest_options, dummy=None):
    """Create empty dictionary for template sets.

    Parameters
    ----------
    quicktest_options : dict
        QUICKTEST_OPTIONS from iQCconstants to get modalities
    dummy : object
        dummy object of some class in config_classes

    Returns
    -------
    dict
        keys = modalities with quicktest as option, values []
    dummy
        default object to insert as first element
    """
    empty_dict = {}
    for key, val in quicktest_options.items():
        if len(val) > 0:
            empty_dict[key] = [dummy]

    return empty_dict


def read_tag_infos_from_yaml():
    """Get DICOM tags from tag_infos.yaml if tag_infos.yaml do not exist yet.

    Returns
    -------
    tag_infos : list of TagInfo
    """
    tag_infos = []
    f_text = ''

    file = QFile(":/config_defaults/tag_infos.yaml")
    file.open(QFile.ReadOnly | QFile.Text)
    f_text = QTextStream(file).readAll()

    if f_text != '':
        docs = yaml.safe_load_all(f_text)
        for doc in docs:
            tag_infos.append(cfc.TagInfo(**doc))
        # reset sort index if yaml changed manually
        for i, tag_info in enumerate(tag_infos):
            tag_info.sort_index = i

    return tag_infos


def set_tag_patterns_special_default(quicktest_options, tag_infos):
    """Set tag_patterns_special default if not defined(edited) yet.

    for each modalities, two labels: Annotate and DICOM_display
    Annotate - default z = for Sliceposition
    DICOM_display - show all available DICOM elements in tag_infos

    Parameters
    ----------
    quicktest_options : dict
        QUICKTEST_OPTIONS from iQCconstants to get modalities
    dummy : TagPatternFormat
        default object

    Returns
    -------
    tag_patterns : dict
        dict with modalities as key, and lists of TagPatternFormat
    """
    tag_patterns_special = empty_template_dict(
        quicktest_options, dummy=cfc.TagPatternFormat())
    all_modalities = [*quicktest_options]

    # initiate empty Annotate and DICOM_display
    tag_pattern_annot = cfc.TagPatternFormat()
    tag_pattern_annot.label = 'Annotate'
    tag_pattern_dicom_display = cfc.TagPatternFormat()
    tag_pattern_dicom_display.label = 'DICOM_display'
    tag_pattern_file_list = cfc.TagPatternFormat()
    tag_pattern_file_list.label = 'File_list_display'

    for mod in all_modalities:
        tag_patterns_special[mod][0] = deepcopy(tag_pattern_annot)
        tag_patterns_special[mod].append(deepcopy(tag_pattern_dicom_display))
        tag_patterns_special[mod].append(deepcopy(tag_pattern_file_list))

    # fill DICOM_display with all available tags
    for tag in tag_infos:
        if tag.limited2mod[0] == '':
            modalities = all_modalities
        else:
            modalities = tag.limited2mod
        for mod in modalities:
            if tag.attribute_name not in tag_patterns_special[mod][1].list_tags:
                tag_patterns_special[mod][1].list_tags.append(tag.attribute_name)
                if tag.attribute_name == 'PixelSpacing':
                    tag_patterns_special[mod][1].list_format.append('|:.3f|')
                else:
                    tag_patterns_special[mod][1].list_format.append('')
                # fill Annotate with z = for all using SliceLocation
                if tag.attribute_name == 'SliceLocation':
                    tag_patterns_special[mod][0].list_tags.append(
                        tag.attribute_name)
                    tag_patterns_special[mod][0].list_format.append(
                        'z = |.1f|')

    # default file list display
    tag_patterns_special['CT'][2].list_tags = [
        'AcquisitionNumber', 'SeriesNumber',
        'SeriesDescription', 'SliceLocation']
    tag_patterns_special['CT'][2].list_format = [
        'acq||', 'ser||', '', 'z=|:.1f|']
    tag_patterns_special['Xray'][2].list_tags = [
        'AcquisitionDate', 'AcquisitionTime', 'KVP', 'mAs']
    tag_patterns_special['Xray'][2].list_format = [
        '', '|:.0f|', '|:.1f|kVp', '|:.1f|mAs']
    tag_patterns_special['Mammo'][2].list_tags = [
        'AcquisitionDate', 'AcquisitionTime', 'AnodeTargetMaterial', 'FilterMaterial',
        'KVP', 'mAs']
    tag_patterns_special['Mammo'][2].list_format = [
        '', '|:.0f|', '', '', '|:.1f|kVp', '|:.1f|mAs']
    tag_patterns_special['NM'][2].list_tags = ['SeriesDescription']
    tag_patterns_special['NM'][2].list_format = ['']
    tag_patterns_special['SPECT'][2].list_tags = [
        'SeriesDescription', 'SliceLocation']
    tag_patterns_special['SPECT'][2].list_format = [
        '', 'z=|:.1f|']
    tag_patterns_special['PET'][2].list_tags = [
        'SeriesDescription', 'SliceLocation']
    tag_patterns_special['PET'][2].list_format = [
        '', 'z=|:.1f|']
    tag_patterns_special['MR'][2].list_tags = [
        'SeriesDescription', 'SliceLocation']
    tag_patterns_special['MR'][2].list_format = [
        '', 'z=|:.1f|']

    return tag_patterns_special


def set_auto_common_default():
    """Set default filename_pattern for AutoCommon."""
    filename_pattern = cfc.TagPatternFormat(
        list_tags=[
            'Modality', 'StationName', 'PatientID', 'AcquisitionDate',
            'AcquisitionTime',
            'SeriesDescription', 'ProtocolName', 'SeriesNumber', 'InstanceNumber'],
        list_format=['', '', '', '', '|:06|', '', '', '', '']
        )

    return cfc.AutoCommon(filename_pattern=filename_pattern)


def get_quicktest_output_default(modality):
    """Set default templates for QuickTest Output if no yaml file exist.

    On startup these are verified against available modalities and testcodes.
    Modalities and tests not defined will have default settings as defined
    in config_classes.

    Return
    ------
    dict
        dict for quicktest_output_template.tests
    """
    tests = {}
    if modality == 'CT':
        tests = {
            'Hom': [
                cfc.QuickTestOutputSub(label='Center_HU', columns=[4]),
                cfc.QuickTestOutputSub(
                    label='Min_diff_from_Center_HU',
                    columns=[5, 6, 7, 8], calculation='min',
                    per_group=True),
                cfc.QuickTestOutputSub(
                    label='Max_diff_from_Center_HU',
                    columns=[5, 6, 7, 8], calculation='max',
                    per_group=True)
                ],
            'Noi': [
                cfc.QuickTestOutputSub(
                    label='Max_noise', columns=[1], calculation='max', per_group=True)
                ],
            'Sli': [
                cfc.QuickTestOutputSub(
                    label='Avg_Slicethick', columns=[5], per_group=True),
                cfc.QuickTestOutputSub(
                    label='Avg_Slicethick_inner',
                    alternative=1,
                    columns=[1, 2, 3, 4], calculation='avg'),
                cfc.QuickTestOutputSub(
                    label='Avg_Slicethick_outer',
                    alternative=1,
                    columns=[5, 6], calculation='avg'),
                cfc.QuickTestOutputSub(
                    label='Avg_SlicethickGE',
                    alternative=2,
                    columns=[1, 2], calculation='avg')
                ],
            'MTF': [
                cfc.QuickTestOutputSub(
                    label='Avg_xy_MTF50', columns=[0, 3], calculation='avg'),
                cfc.QuickTestOutputSub(
                    label='Avg_xy_MTF10', columns=[1, 4], calculation='avg'),
                cfc.QuickTestOutputSub(
                    label='MTF50', alternative=1, columns=[0], per_group=True),
                cfc.QuickTestOutputSub(
                    label='MTF10', alternative=1, columns=[1], per_group=True),
                ]
            }

    return tests
