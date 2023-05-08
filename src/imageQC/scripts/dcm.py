#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts dealing with pydicom.

@author: Ellen Wasbo
"""
from __future__ import annotations
import warnings
import os
import pydicom
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox, QDialog, QTextEdit

# imageQC block start
from imageQC.config.iQCconstants import QUICKTEST_OPTIONS, ENV_ICON_PATH
import imageQC.scripts.mini_methods_format as mmf
from imageQC.config.config_classes import TagPatternFormat
# imageQC block end

pydicom.config.future_behavior(True)

warnings.filterwarnings(action='ignore', category=UserWarning)
# avoid not important DICOM warnings


@dataclass
class DcmInfo():
    """Class to keep a minimum number of image attributes."""

    filepath: str = ''
    modality: str = 'CT'  # as mode in imageQC
    modalityDCM: str = 'CT'  # as modality read from DICOM
    marked: bool = False
    marked_quicktest: list = field(default_factory=list)
    quicktest_image_name: str = ''  # for result export
    quicktest_series_name: str = ''  # for result export
    frame_number: int = -1
    pix: tuple = (-1., -1.)
    shape: tuple = (-1, -1)  # rows, columns
    slice_thickness: Optional[float] = None
    nm_radius: Optional[float] = None  # radius of gamma camera
    zpos: Optional[float] = None
    acq_date: str = ''
    studyUID: str = ''


@dataclass
class DcmInfoGui(DcmInfo):
    """Class to keep attributes for images when not automation. Info to GUI."""

    info_list_general: list = field(default_factory=list)
    # for DICOM header display general parameters
    info_list_modality: list = field(default_factory=list)
    # for DICOM header display modality specific parameters
    annotation_list: list = field(default_factory=list)
    # for image annotations
    file_list_strings: list = field(default_factory=list)
    # for file list display if not path
    series_list_strings: list = field(default_factory=list)
    # for seriesnumber / seriesdescription used in open multi
    window_width: int = 0
    window_center: int = 0


def fix_sop_class(elem, **kwargs):
    """Fix for Carestream DX unprocessed."""
    if elem.tag == 0x00020002:
        # DigitalXRayImageStorageForProcessing
        elem = elem._replace(value=b"1.2.840.10008.5.1.4.1.1.1.1.1")

    return elem


def read_dcm_info(filenames, GUI=True, tag_infos=[],
                  tag_patterns_special={}):
    """Read Dicom header into DcmInfo(Gui) objects when opening files.

    Parameters
    ----------
    filenames : list of str
        list of filenames to open
    GUI : bool, optional
        GUI version (True) or automation only (False). Default is True
        ignored if tag_patterns_special undefined
        and group_img_pattern is not None
    tag_infos : list of TagInfo
        holding info on possible tags to read and how
    tag_patterns_special : dict
        used if GUI is True
        holding info on which tags from tag_infos to read + formating
        for text display or annotation

    Returns
    -------
    list_of_DcmInfo : list of objects
        DcmInfo, DcmInfoGui or DcmInfoGroup objects
    ignored_files : list of str
        file_paths not accepted as dicom images
    """
    list_of_DcmInfo = []
    ignored_files = []

    for file in filenames:
        file_verified = True
        pd = {}
        try:
            pd = pydicom.dcmread(file, stop_before_pixels=True)
            columns = pd.get('Columns', 0)  # Columns>0 == image data
            if columns == 0:
                file_verified = False
        except pydicom.errors.InvalidDicomError:
            file_verified = False
        except AttributeError:
            pydicom.config.data_element_callback = fix_sop_class
            pd = pydicom.dcmread(file, stop_before_pixels=True)

        if file_verified:
            modalityDCM = pd.get('Modality', '')
            mod = get_modality(modalityDCM)['key']
            frames = pd.get('NumberOfFrames', None)
            attrib = {
                'filepath': str(file),
                'modality': mod,
                'modalityDCM': modalityDCM,
                'shape': (pd.get('Rows', 0), pd.get('Columns', 0)),
                'studyUID': (pd.get('StudyInstanceUID', ''))
                }

            slice_thickness, pix, slice_location, acq_date = get_dcm_info_list(
                    pd,
                    TagPatternFormat(list_tags=[
                        'SliceThickness', 'PixelSpacing', 'SliceLocation',
                        'AcquisitionDate']),
                    tag_infos,
                    prefix_separator='', suffix_separator='',
                    )
            attrib['slice_thickness'] = None
            if isinstance(slice_thickness, str):
                if '[' in slice_thickness:
                    slice_thickness = slice_thickness[1:-1]
                try:
                    attrib['slice_thickness'] = float(slice_thickness)
                except ValueError:
                    pass
            elif isinstance(slice_thickness, list):
                try:
                    attrib['slice_thickness'] = float(str(slice_thickness[1][0]))
                    # TODO? not assume same
                except ValueError:
                    pass
            if mod == 'NM':  # test if slicethickness - then SPECT
                if attrib['slice_thickness'] is not None:
                    attrib['modality'] = 'SPECT'
                    mod = 'SPECT'

            attrib['pix'] = [0, 0]
            if isinstance(pix, list):
                pix = str(pix[1][0])  # TODO? not assume same
            pix = pix[1:-1]
            pix = pix.split(',')
            if len(pix) == 2:
                try:
                    pix = [float(pix[0]), float(pix[1])]
                    attrib['pix'] = pix
                except ValueError:
                    pass

            if isinstance(acq_date, str):
                if len(acq_date) > 8:  # if AcquisitionDateTime is used
                    acq_date = acq_date[0:9]
            attrib['acq_date'] = acq_date

            nm_radius = []
            if mod == 'NM':
                # attrib nm_radius
                nm_radius = get_dcm_info_list(
                        pd,
                        TagPatternFormat(
                            list_tags=['RadialPosition'], list_format=['']),
                        tag_infos,
                        prefix_separator='', suffix_separator='',
                        )
                if nm_radius:
                    nm_radius = nm_radius[0]
                    if isinstance(nm_radius, str):
                        if '[' in nm_radius:
                            nm_radius = nm_radius[1:-1]
                            nm_radius = nm_radius.split(',')

            if GUI:
                ww = pd.get('WindowWidth', -1)
                wc = pd.get('WindowCenter', -1)
                # seen issue with window width and window center
                # listed as two identical values
                if str(type(ww)) == "<class 'pydicom.multival.MultiValue'>":
                    ww = int(ww[0])
                if str(type(wc)) == "<class 'pydicom.multival.MultiValue'>":
                    wc = int(wc[0])
                attrib['window_width'] = ww
                attrib['window_center'] = wc

            if frames is None:
                if attrib['slice_thickness'] is not None:
                    attrib['zpos'] = float(slice_location)  # try?
                try:
                    attrib['nm_radius'] = float(nm_radius)
                except (TypeError, ValueError):
                    pass
                if GUI:
                    attrib.update(get_dcm_gui_info_lists(
                        pd,
                        tag_infos=tag_infos,
                        tag_patterns_special=tag_patterns_special,
                        modality=mod))
                    dcm_obj = DcmInfoGui(**attrib)
                else:
                    dcm_obj = DcmInfo(**attrib)
                list_of_DcmInfo.append(dcm_obj)
            else:  # multiframe
                if GUI:
                    gui_info = get_dcm_gui_info_lists(
                        pd, tag_infos=tag_infos,
                        tag_patterns_special=tag_patterns_special,
                        modality=mod)
                frames = int(frames)
                for frame in range(frames):
                    attrib.update({'frame_number': frame})
                    if attrib['slice_thickness'] is not None:
                        try:
                            info_this = info_extract_frame(
                                frame, {'zpos': [slice_location]})
                            attrib['zpos'] = float(info_this['zpos'][0])
                        except (IndexError, ValueError):
                            attrib['zpos'] = frame * attrib['slice_thickness']
                    try:
                        info_this = info_extract_frame(
                            frame, {'nm_radius': [nm_radius]})
                        attrib['nm_radius'] = float(info_this['nm_radius'][0])
                    except (ValueError, IndexError):
                        pass
                    if GUI:
                        gui_info_this = info_extract_frame(frame, gui_info)
                        attrib.update(gui_info_this)
                        dcm_obj = DcmInfoGui(**attrib)
                    else:
                        dcm_obj = DcmInfo(**attrib)
                    list_of_DcmInfo.append(dcm_obj)
        else:
            ignored_files.append(file)

    return (list_of_DcmInfo, ignored_files)


def info_extract_frame(frame, info):
    """Extract frame number value from info dict.

    Parameters
    ----------
    frame : int
        frame number for multiframe dicom
    info : dict
        'key': [ str, str, [pre, vect, suff], str, str...]

    Returns
    -------
    out_dict : dict
        info for one frame only - all list of strings
    """
    out_dict = {}
    for key, item in info.items():
        out_dict[key] = []
        for tag in item:
            if isinstance(tag, list):
                if len(tag) == 3:
                    try:
                        out_dict[key].append(
                            ''.join([tag[0], str(tag[1][frame]), tag[2]]))
                        #TODO format for tag1....
                    except (IndexError, TypeError):
                        out_dict[key].append(
                            ''.join([tag[0], str(tag[1]), tag[2]]))

            else:
                out_dict[key].append(tag)

    return out_dict


def get_dcm_info_list(
        pydict, tag_pattern, tag_infos,
        prefix_separator=' ', suffix_separator=' ',
        attr_name_default_prefix=False, unit_default_suffix=False,
        not_found_text=''):
    """Get list of tag infos (prefix + formated value + suffix).

    Parameters
    ----------
    pydict : pydicom dataset
    tag_pattern : TagPatternFormat or TagPatternSort
    tag_infos : list of TagInfo
    prefix_separator : str
        separator between prefix and formated value. The default is ' '
    suffix_separator : str
        separator between formated value and suffix. The default is ' '
    attr_name_default_prefix : bool
        True=prefix '' causes prefix=attr_name:. The default is False.
    unit_default_suffix : bool
        True=suffix '' causes suffix=unit:. The default is False.
    not_found_text : str
        Text to set as value if not found

    Returns
    -------
    info_list : list of str
        list of prefix + formated value + suffix
        if multiframe list elements might be [prefix,[values],suffix]
    """
    info_list = []
    attribute_names = [obj.attribute_name for obj in tag_infos]

    def multiply_tagvalue(val, tag_info):
        multiplied_val = val
        if tag_info.factor != 1:
            try:
                if isinstance(val, list):
                    val = list(tag_info.factor * np.array(val, dtype=float))
                else:
                    multiplied_val = tag_info.factor * float(val)
            except ValueError:
                print((f'Tag {tag_info.attribute_name}: Failed multiplying '
                       f'{data_element.value} by factor {tag_info.factor}'
                       ))
        return multiplied_val

    for idx_pattern in range(len(tag_pattern.list_tags)):

        attribute_name = tag_pattern.list_tags[idx_pattern]
        tag_idx = [
            i for i, item in enumerate(attribute_names)
            if item == attribute_name]

        if hasattr(tag_pattern, 'list_format'):
            prefix, format_string, suffix = mmf.get_format_strings(
                tag_pattern.list_format[idx_pattern])
        else:
            prefix, format_string, suffix = ('', '', '')
        if prefix == '' and attr_name_default_prefix:
            prefix = f'{attribute_name}:'
        info = f'{prefix}{prefix_separator}{not_found_text}'

        for idx in tag_idx:
            if tag_infos[idx].tag[1] == '':
                data_element = get_special_tag_data(pydict, tag_info=tag_infos[idx])
            else:
                data_element = get_tag_data(pydict, tag_info=tag_infos[idx])
            if data_element is not None:
                if suffix == '' and unit_default_suffix:
                    suffix = f'{tag_infos[idx].unit}'
                if isinstance(data_element, list):
                    val = [elem.value for elem in data_element]
                    if tag_infos[idx].factor != 1.:
                        val = multiply_tagvalue(val, tag_infos[idx])
                    VM = 2  # > 1
                else:
                    val = data_element.value
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    if tag_infos[idx].factor != 1.:
                        val = multiply_tagvalue(val, tag_infos[idx])
                    VM = data_element.VM
                if VM > 1:
                    nframes = int(pydict.get('NumberOfFrames', -1))
                    value_id = tag_infos[idx].value_id
                    if value_id == -3:
                        if not isinstance(data_element, list):
                            value_id = -1
                        else:
                            VMs = [elem.VM for elem in data_element]
                            totVM = np.sum(np.array(VMs))
                            if totVM != nframes:
                                value_id = -1
                    if value_id == -2:  # per frame, check if possible
                        if data_element.VM != nframes:
                            value_id = -1
                    multi_frame = False
                    if value_id > -1:  # specific id, check if possible
                        if isinstance(val, list):
                            val = [x[value_id] for x in val]
                            multi_frame = True
                        else:
                            if value_id > data_element.VM:
                                value_id = -1
                    multi_val = False
                    if value_id == -1:
                        if not isinstance(val, list):
                            multi_val = True  # combine values to text
                        val_text = val
                    elif value_id == -2:
                        val_text = val
                    elif value_id == -3:
                        val_combined = []
                        try:
                            for elem in val:
                                val_combined.extend(elem)
                            val_text = val_combined
                        except TypeError:
                            val_text = val
                    else:
                        if multi_frame:
                            val_text = val
                        else:
                            val_text = val[value_id]

                    if format_string != '':
                        val_text = mmf.format_val(val, format_string)

                    if multi_val:
                        info = (
                            f'{prefix}{prefix_separator}'
                            f'{val_text}{suffix_separator}{suffix}'
                            )
                        info = info.replace("'", "")
                    else:
                        info = [
                            f'{prefix}{prefix_separator}',
                            val_text,
                            f'{suffix_separator}{suffix}'
                            ]
                else:
                    val_text = mmf.format_val(val, format_string)
                    info = (
                        f'{prefix}{prefix_separator}'
                        f'{val_text}{suffix_separator}{suffix}'
                        )

                break  # break for loop with same attr_name if found

        if isinstance(info, str):
            info.strip()
        info_list.append(info)
    return info_list


def get_dcm_gui_info_lists(
        pydict, tag_infos=[], tag_patterns_special={}, modality=''):
    """Get Dicom header info for display in Gui.

    Parameters
    ----------
    pydict : dict
        Dictionary from pydicom.dcmread without pixel data.
    tag_infos : list of TagInfo
        as defined in settings
    tag_patterns_special : dict
        as defined in settings (default in iQCconstants_functions.py)
    modality : str
        current modality

    Returns
    -------
    dict
        info_list_general as list
        info_list_modality as list
        annotation_list as list
        file_list_strings as list
        series_list_strings as list
        as expected by class DcmInfoGui
    """
    annotation_list = []
    info_list_general = []
    info_list_modality = []
    file_list_strings = []
    series_list_strings = []

    if len(tag_patterns_special) > 0:
        if modality != '':
            # annotation list
            annotation_list = get_dcm_info_list(
                pydict, tag_patterns_special[modality][0], tag_infos)

            # dicom header list
            dicom_display_pattern = tag_patterns_special[modality][1]
            attribute_names = [obj.attribute_name for obj in tag_infos]
            info_list = get_dcm_info_list(
                pydict, dicom_display_pattern, tag_infos,
                prefix_separator=' ', suffix_separator=' ',
                attr_name_default_prefix=True,
                unit_default_suffix=True,
                not_found_text='-not found-')
            for idx in range(len(dicom_display_pattern.list_tags)):
                attribute_name = dicom_display_pattern.list_tags[idx]
                if attribute_name in attribute_names:
                    tag_idx = attribute_names.index(attribute_name)
                    if tag_infos[tag_idx].limited2mod[0] == '':
                        info_list_general.append(info_list[idx])
                    else:
                        info_list_modality.append(info_list[idx])

            # file list strings
            file_list_strings = get_dcm_info_list(
                pydict, tag_patterns_special[modality][2], tag_infos)

        series_pattern = TagPatternFormat(
            list_tags=['SeriesNumber', 'SeriesDescription'],
            list_format=['|:04.0f|', ''])
        series_list_strings = get_dcm_info_list(
            pydict, series_pattern, tag_infos)

    return {
        'info_list_general': info_list_general,
        'info_list_modality': info_list_modality,
        'annotation_list': annotation_list,
        'file_list_strings': file_list_strings,
        'series_list_strings': series_list_strings
        }


def get_modality(modalityStr):
    """Get modality index by matching DICOM modality to modalities in imageQC.

    Parameters
    ----------
    modalityStr : str
        Modality from DICOM

    Returns
    -------
    dict
        {id: as int, key: as str)
    """
    variants = {
        'CT': 'CT',
        'DX': 'Xray', 'CR': 'Xray', 'DR': 'Xray', 'RG': 'Xray',
        'XA': 'Xray', 'RF': 'Xray', 'MG': 'Xray',
        'MG': 'Xray', 'PX': 'Xray',
        'NM': 'NM',
        'ST': 'SPECT',
        'PT': 'PET',
        'MR': 'MR'
        }
    # TODO: add mammo
    qtOptKey = variants.get(modalityStr, 'CT')

    return {'id': list(QUICKTEST_OPTIONS.keys()).index(qtOptKey),
            'key': qtOptKey}


def get_img(filepath, frame_number=-1, tag_patterns=[], tag_infos=None, NM_count=False,
            get_window_level=False):
    """Read pixmap from filepath int numpy array prepeared for display.

    Parameters
    ----------
    filepath : str
        full path of DICOM file
        - already verified as dicom, but file connection might get lost
    frame_number : int
        if multiframe, -1 if single frame dicom
    tag_patterns : list of TagPatternFormat, optional
        Optionally also read selected tags. Default is []
    tag_infos: list of TagInfo, optional
        Used together with tag_pattern. Default is None
    NM_count : bool, optional
        True if CountsAccumulated should be performed (if needed). Default is False
    get_window_level : bool, optional
        True if adding window min,max as list to tag_strings.
        Used in OpenAutomationDialog. Default is False

    Returns
    -------
    image : np.array
        pixelvalues
    tag_strings : list of list of str
        for each tag_pattern list dicom (formatted) values
        (+ windowlevel [min, max] if get_window_level = True
    """
    npout = None
    tag_strings = []
    pd = None
    try:
        pd = pydicom.dcmread(filepath)
    except pydicom.errors.InvalidDicomError:
        pass
    except FileNotFoundError:
        pass  # TODO show errormessage
    if pd is not None:
        pixarr = None
        try:
            pixarr = pd.pixel_array
        except AttributeError:
            pass

        overlay = None
        try:
            overlay = pd.overlay_array(0x6000)  # e.g. Patient Protocol Siemens CT
        except AttributeError:
            pass

        if pixarr is not None:
            slope, intercept = get_dcm_info_list(
                pd, TagPatternFormat(list_tags=['RescaleSlope', 'RescaleIntercept']),
                tag_infos)
            if frame_number == -1:
                slope_this = slope
                intercept_this = intercept
            else:
                try:
                    slope_this = slope[1][frame_number]
                    intercept_this = intercept[1][frame_number]
                except IndexError:
                    try:
                        slope_this = slope[1]
                        intercept_this = intercept[1]
                    except IndexError:
                        slope_this = slope
                        intercept_this = intercept
            try:
                slope = float(slope_this)
                intercept = float(intercept_this)
            except ValueError:
                slope = 1.
                intercept = 0.

            if frame_number > -1:
                ndim = pd.pixel_array.ndim
                if ndim == 3:
                    pixarr = pd.pixel_array[frame_number, :, :]
            else:
                pixarr = pd.pixel_array
            if pixarr is not None:
                npout = pixarr * slope + intercept

        if overlay is not None:
            if pixarr is not None:
                if npout.shape == overlay.shape:
                    npout = np.add(npout, overlay)
                else:
                    pass  # implement when example file exists
                    # overlay_offset at 0x6000, 0x0050
            else:
                npout = overlay

        orient = pd.get('PatientPosition', '')
        if orient == 'FFS':
            npout = np.fliplr(npout)

        if len(tag_patterns) > 0 and tag_infos is not None:
            tag_strings = read_tag_patterns(
                pd, tag_patterns, tag_infos, frame_number=frame_number)
            if NM_count:
                for pidx, pattern in enumerate(tag_patterns):
                    if 'CountsAccumulated' in pattern.list_tags:
                        idx = pattern.list_tags.index('CountsAccumulated')
                        if isinstance(tag_strings[pidx], dict):
                            this_list = tag_strings[pidx]['dummy']
                        else:
                            this_list = tag_strings[pidx]
                        if this_list[idx] in ['', '-']:
                            new_val = np.sum(pixarr)
                            if hasattr(pattern, 'list_format'):
                                new_val = mmf.format_val(
                                    new_val, pattern.list_format[idx])
                            if isinstance(tag_strings[pidx], dict):
                                tag_strings[pidx]['dummy'][idx] = new_val
                            else:
                                tag_strings[pidx][idx] = new_val

        if get_window_level:
            ww = pd.get('WindowWidth', -1)
            wc = pd.get('WindowCenter', -1)
            # seen issue with window width and window center
            # listed as two identical values
            if str(type(ww)) == "<class 'pydicom.multival.MultiValue'>":
                ww = int(ww[0])
                if str(type(wc)) == "<class 'pydicom.multival.MultiValue'>":
                    wc = int(wc[0])
            tag_strings.append([wc - ww / 2, wc + ww / 2])

    return (npout, tag_strings)


def get_tags(filepath, frame_number=-1, tag_patterns=[], tag_infos=None):
    """Read specific Dicom tags from file.

    Parameters
    ----------
    filepath : str
        full path of DICOM file
        - already verified as dicom, but file connection might get lost
    frame_number : int
        if multiframe, -1 if single frame dicom
    tag_patterns : list of TagPatternSort or TagPatternFormat
    tag_infos : list of TagInfo

    Returns
    -------
    tag_strings : list of list of str
        for each tag_pattern as input, list of (formatted) dicom tag values
    """
    tag_strings = []
    pd = None
    if len(tag_patterns) > 0 and tag_infos is not None:
        try:
            pd = pydicom.dcmread(filepath, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            pass
        if pd is not None:
            tag_strings = read_tag_patterns(
                pd, tag_patterns, tag_infos, frame_number=frame_number)

    return tag_strings


def read_tag_patterns(pd, tag_patterns, tag_infos, frame_number=-1):
    """Read tags in tag_patterns.

    Parameters
    ----------
    pd : pydicom dictionary
    tag_patterns : list of TagPatternFormat og TagPatternSort
    tag_infos : list of TagInfo

    Returns
    -------
    tag_lists : list of list of str
        list of dicom (formatted) values for each tag pattern
    """
    tag_string_lists = []
    for pattern in tag_patterns:
        tag_strings = get_dcm_info_list(
            pd, pattern, tag_infos,
            prefix_separator='', suffix_separator='')
        if frame_number > -1 and len(tag_strings) > 0:
            tag_strings = info_extract_frame(
                frame_number,  {'dummy': tag_strings})
        tag_string_lists.append(tag_strings)

    return tag_string_lists


def private_seq_str_to_tag(txt):
    """Convert string 'Private (GGGG,EEEE)' to tag [0xGGGG, 0xEEEE]."""
    txt_sub = txt.split('(')
    tag = None
    if len(txt_sub) > 1:
        txt = txt_sub[1]
        try:
            group = hex(int('0x' + txt[0:4], 16))
            elem = hex(int('0x' + txt[6:10], 16))
            tag = [group, elem]
        except ValueError:
            pass

    return tag


def get_element_in_sequence(dataset, sequence_string,
                            element_number=0, return_sequence=False):
    """Get element from sequence given a sequence as Key or 'Privat (GGGG,EEEE)'.

    Parameters
    ----------
    dataset : pydicom dict
    sequence_string : str
    element_number : int
        Default is 0
    return_sequence : bool
        True = do not extract element number but full sequence. Default is False

    Returns
    -------
    pydicom.DataElement or pydicom DataSet
    """
    seq = []
    elem = None
    if 'Private' in sequence_string:
        tag = private_seq_str_to_tag(sequence_string)
        if tag is not None:
            seq = dataset[tag]
    else:
        try:
            seq = dataset[sequence_string]
        except KeyError:
            pass

    if return_sequence:
        elem = seq
    else:
        try:
            elem = seq[element_number]
        except IndexError:
            pass

    return elem


def get_tag_data(pd, tag_info=None):
    """Read content of specific Dicom tag from file pydicom dictionary.

    Parameters
    ----------
    py : pydicom dict
    tag_info : TagInfo

    Returns
    -------
    data_element
    """
    data_element = None
    if pd is not None and tag_info is not None:
        seq = tag_info.sequence
        gr = tag_info.tag[0]
        el = tag_info.tag[1]
        nframes = int(pd.get('NumberOfFrames', -1))

        try:
            if seq[0] != '':
                pd_sub = get_element_in_sequence(pd, seq[0])
                if len(seq) > 1:
                    if seq[0] == 'PerFrameFunctionalGroupsSequence' and nframes > 1:
                        data_element = []
                        for f in range(nframes):
                            pd_sub = get_element_in_sequence(
                                pd, seq[0], element_number=f)
                            for i in range(1, len(seq)):
                                pd_sub = get_element_in_sequence(pd_sub, seq[i])
                            data_element.append(pd_sub[gr, el])
                    else:
                        pd_sub_final = get_element_in_sequence(
                            pd, seq[0], return_sequence=True)
                        pd_sub = pd_sub_final[0]
                        for i in range(1, len(seq)):
                            pd_sub_final = get_element_in_sequence(
                                pd_sub, seq[i], return_sequence=True)
                            pd_sub = pd_sub_final[0]
                else:
                    pd_sub_final = get_element_in_sequence(
                        pd, seq[0], return_sequence=True)

                if data_element is None:
                    if tag_info.value_id == -3:  # combine data from all sequences
                        try:
                            if len(pd_sub_final.value) > 1:
                                data_element = []
                                for i in range(len(pd_sub_final.value)):
                                    sub = pd_sub_final[i]
                                    data_element.append(sub[gr, el])
                            else:
                                data_element = pd_sub[gr, el]
                        except AttributeError:
                            data_element = None
                    else:
                        data_element = pd_sub[gr, el]
            else:
                data_element = pd[gr, el]
        except (KeyError, IndexError, TypeError):
            data_element = None

    return data_element


def get_special_tag_data(pd, tag_info=None):
    """Get DICOM information for data not specified as tag. tag_info.tag = ['...', ''].

    Parameters
    ----------
    py : pydicom dict
    tag_info : TagInfo

    Returns
    -------
    data_element
    """
    data_element = None
    if tag_info.attribute_name == 'FrameNumber':
        frames = int(pd.get('NumberOfFrames', -1))
        if frames > -1:
            data_element = pydicom.DataElement(
                0xffffffff, 'LO', list(np.arange(frames)))
    # elif tag_info.attribute_name == 'CountsAccumulated':# in get_img

    return data_element


def find_all_valid_dcm_files(
        path_str, parent_widget=None, grouped=True,
        search_subfolders=True):
    """Find all valid DICOM files in a directory.

    Parameters
    ----------
    path_str : str
        path to folder to test for dicom files
    parent_widget : widget
        if None, no error message given ()
    grouped : bool
        group files in lists for each (sub)folder. Default is True
    search_subfolders : bool
        search is recursive. Default is True.

    Returns
    -------
    folders : list of str
        list of folders holding the grouped files
    files : list of str
        if grouped: list of lists of str
        list of validated dicom files
    """
    dcm_files = []
    dcm_folders = []
    if path_str != '':
        p = Path(path_str)
        if p.is_dir():
            glob_string = '**/*' if search_subfolders else '*'
            dcm_files = [x for x in p.glob(glob_string) if x.suffix == '.dcm']
            if len(dcm_files) == 0:
                files = [x for x in p.glob(glob_string) if x.is_file()]
                dcm_files = []
                for file in files:
                    try:
                        pydicom.dcmread(file, specific_tags=[(0x8, 0x60)])
                        dcm_files.append(file)
                    except pydicom.errors.InvalidDicomError:
                        pass
            if len(dcm_files) == 0 and parent_widget is not None:
                QMessageBox.information(
                    parent_widget, 'No valid DICOM',
                    'Found no valid DICOM files in the given folder.')

            if len(dcm_files) > 0:
                file_parent = [Path(x).parent for x in dcm_files]
                if grouped:
                    grouped_files = []
                    for i, file in enumerate(dcm_files):
                        if file_parent[i] in dcm_folders:
                            grouped_files[-1].append(file)
                        else:
                            grouped_files.append([file])
                            dcm_folders.append(file_parent[i])
                    dcm_files = grouped_files
                else:
                    dcm_folders = list(set(file_parent))  # get unique folders

    else:
        if parent_widget is not None:
            QMessageBox.information(
                parent_widget, 'No path',
                'No folder to search for DICOM files.')

    return {
        'folders': dcm_folders,
        'files': dcm_files
        }


def sort_imgs(img_infos, tag_pattern_sort, tag_infos):
    """Sort images based on TagPatternSort.

    Parameters
    ----------
    img_infos : list of dict
        imgs from main
    tag_pattern_sort : TagPatternSort
    tag_infos : TagInfos

    Returns
    -------
    sorted_img_infos : list of dict

    """
    if len(tag_pattern_sort.list_tags) == 0:
        sorted_img_infos = img_infos
    else:
        infos = []
        for i in range(len(img_infos)):
            info = get_tags(img_infos[i].filepath,
                            frame_number=img_infos[i].frame_number,
                            tag_patterns=[tag_pattern_sort],
                            tag_infos=tag_infos)
            if isinstance(info[0], dict):
                infos.append(info[0]['dummy'])
            else:
                infos.append(info[0])
    
        df = {}
        for c, attr in enumerate(tag_pattern_sort.list_tags):
            col = [row[c] for row in infos]
            try:
                col = [float(row[c]) for row in infos]
            except ValueError:
                pass
            df[attr] = col
        df = pd.DataFrame(df)
        sorted_infos = df.sort_values(
            by=tag_pattern_sort.list_tags,
            ascending=tag_pattern_sort.list_sort)
    
        sorted_img_infos = []
        for idx in sorted_infos.index:
            sorted_img_infos.append(img_infos[idx])

    return sorted_img_infos


def dump_dicom(parent_widget, filename=''):
    """Dump dicom elements for file to text.

    Parameters
    ----------
    parent_widget : widget
        widget calling dicom_dump
    filename : str, optional
        The default is ''.
    """
    if filename != '':
        try:
            ds = pydicom.dcmread(filename)
            #TODO consider using TextDisplay from ui_dialogs
            dlg = QDialog(parent_widget)
            dlg.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
            dlg.setWindowFlags(dlg.windowFlags() | Qt.CustomizeWindowHint)
            dlg.setWindowFlags(
                dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            txtEdit = QTextEdit('', dlg)
            txtEdit.setPlainText(str(ds))
            txtEdit.setReadOnly(True)
            txtEdit.createStandardContextMenu()
            txtEdit.setMinimumWidth(1000)
            txtEdit.setMinimumHeight(800)
            dlg.setWindowTitle(filename)
            dlg.setMinimumWidth(1000)
            dlg.setMinimumHeight(800)
            dlg.show()

        except pydicom.errors.InvalidDicomError:
            QMessageBox.warning(
                parent_widget, 'Dicom read failed',
                'Failed to read selected file as dicom.')


def get_all_tags_name_number(pd, sequence_list=['']):
    """Find all (nested) attribute names.

    Parameters
    ----------
    pd: pydicom dataframe
    sequence_list : list of str
        not nested list!
        List of sequence attribute names (keyword e.g. 'PlanePositionSequence')
        if not ['']
        tags to find will be in pd[seq1][0][seq2][0]...

    Returns
    -------
    all_tags : dict
        {tags: [BaseTag],
         attribute_names: ['']}
    """
    def get_tags_also_private(dataset):
        list_name_tag = []
        attributes = dataset.dir()
        for attr in attributes:
            try:
                data_element = dataset[attr]
                list_name_tag.append((attr, data_element.tag))
            except KeyError:
                pass
        if len(attributes) != len(dataset):
            for elem in dataset:
                if elem.is_private:
                    seq_txt = 'Sequence' if 'SQ' in elem.VR else ''
                    list_name_tag.append(
                        (f'Private{seq_txt} {elem.tag}', elem.tag))
        return list_name_tag

    list_name_tag = []
    if sequence_list[0] == '':
        list_name_tag = get_tags_also_private(pd)
    else:
        pd_sub = get_element_in_sequence(pd, sequence_list[0])
        if pd_sub is not None:
            if len(sequence_list) == 1:
                list_name_tag = get_tags_also_private(pd_sub)
            else:
                for i in range(1, len(sequence_list)):
                    pd_sub = get_element_in_sequence(pd_sub, sequence_list[i])
                    if pd_sub is not None:
                        list_name_tag = get_tags_also_private(pd_sub)
                    else:
                        list_name_tag = []

    attribute_names = [tup[0] for tup in list_name_tag]
    tags = [tup[1] for tup in list_name_tag]

    return {'tags': tags, 'attribute_names': attribute_names}


def sum_marked_images(img_infos, included_ids, tag_infos):
    """Calculate sum of marked images.

    Parameters
    ----------
    img_infos : list of Dcm(Gui)Info
        as defined in scripts/dcm.py
    included_ids : list of int
        image ids to include in sum
    tag_infos : list of TagInfos

    Returns
    -------
    summed_img : np.ndarray
    errmsg : str
    """
    summed_img = None
    shape_first = None
    shape_failed = []
    errmsg = ''
    for idx, img_info in enumerate(img_infos):
        if idx in included_ids:
            image, tags = get_img(
                img_info.filepath,
                frame_number=img_info.frame_number, tag_infos=tag_infos
                )
            if summed_img is None:
                summed_img = image
                shape_first = image.shape
            else:
                if image.shape == shape_first:
                    summed_img = np.add(summed_img, image)
                else:
                    shape_failed.append(idx)
    if len(shape_failed) > 0:
        errmsg = ('Could not sum marked images due to different sizes. ' +
                  f'Image number {shape_failed} did not match the first marked image.')
        summed_img = None

    return (summed_img, errmsg)
