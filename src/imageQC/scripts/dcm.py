#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scripts dealing with pydicom.

@author: Ellen Wasbo
"""
from __future__ import annotations
import os
import pydicom
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox, QDialog, QTextEdit

# imageQC block start
from imageQC.config.iQCconstants import QUICKTEST_OPTIONS, ENV_ICON_PATH
from imageQC.scripts.mini_methods_format import get_format_strings
from imageQC.config.config_classes import TagPatternFormat
# imageQC block end

pydicom.config.future_behavior(True)


@dataclass
class DcmInfo():
    """Class to keep a minimum number of image attributes."""

    filepath: str = ''
    modality: str = 'CT'  # as mode in imageQC
    modalityDCM: str = 'CT'  # as modality read from DICOM
    marked: bool = True
    marked_quicktest: list = field(default_factory=list)
    quicktest_image_name: str = ''  # for result export
    quicktest_series_name: str = ''  # for result export
    frame_number: int = -1
    pix: tuple = (-1., -1.)
    shape: tuple = (-1, -1)  # rows, columns
    slice_thickness: float = -1.0
    nm_radius: float = -1.0  # radius of gamma camera
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
    zpos: Optional[float] = None


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
            frames = pd.get('NumberOfFrames', -1)
            attrib = {
                'filepath': str(file),
                'modality': mod,
                'modalityDCM': modalityDCM,
                'shape': (pd.get('Rows', 0), pd.get('Columns', 0)),
                'studyUID': (pd.get('StudyInstanceUID', ''))
                }

            slice_thickness, pix = get_dcm_info_list(
                    pd,
                    TagPatternFormat(
                        list_tags=['SliceThickness', 'PixelSpacing'],
                        list_format=['', '']),
                    tag_infos,
                    prefix_separator='', suffix_separator='',
                    )
            attrib['slice_thickness'] = -1.
            if '[' in slice_thickness:
                slice_thickness = slice_thickness[1:-1]
            try:
                attrib['slice_thickness'] = float(slice_thickness)
            except ValueError:
                pass
            attrib['pix'] = [0, 0]
            pix = pix[1:-1]
            pix = pix.split(',')
            if len(pix) == 2:
                try:
                    pix = [float(pix[0]), float(pix[1])]
                    attrib['pix'] = pix
                except ValueError:
                    pass

            acq_date = pd.get('AcquisitionDate', '')
            if acq_date == '':
                acq_date = pd.get('AcquisitionDateTime', '')
                if acq_date == '':
                    acq_date = acq_date[0:9]
            attrib['acq_date'] = acq_date

            nm_radius = []
            if mod in ['NM', 'SPECT']:
                #attrib nm_radius
                nm_radius = get_dcm_info_list(
                        pd,
                        TagPatternFormat(list_tags=['RadialPosition'], list_format=['']),
                        tag_infos,
                        prefix_separator='', suffix_separator='',
                        )
                if len(nm_radius) == 1:
                    nm_radius = nm_radius[0]
                if '[' in nm_radius[0]:
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

            if frames <= 1:
                if GUI:
                    zpos = pd.get('SliceLocation', None)
                    attrib['zpos'] = zpos
                    attrib.update(get_dcm_gui_info_lists(
                        pd,
                        tag_infos=tag_infos,
                        tag_patterns_special=tag_patterns_special,
                        modality=mod))
                    dcm_obj = DcmInfoGui(**attrib)
                else:
                    if len(nm_radius) > 0:
                        attrib['nm_radius'] = float(nm_radius)
                    dcm_obj = DcmInfo(**attrib)
                list_of_DcmInfo.append(dcm_obj)
            else:  # multiframe
                seq = pd.get('PerFrameFunctionalGroupsSequence', None)
                if GUI:
                    gui_info = get_dcm_gui_info_lists(
                        pd, tag_infos=tag_infos,
                        tag_patterns_special=tag_patterns_special,
                        modality=mod)
                for frame in range(frames):
                    attrib.update({'frame_number': frame})
                    if GUI:
                        if seq is not None:
                            try:
                                zpos = seq[frame][
                                    'PlanePositionSequence'][0][
                                        'ImagePositionPatient'].value[2]
                                attrib['zpos'] = zpos
                            except (IndexError, KeyError):
                                pass
                        gui_info_this = info_extract_frame(frame, gui_info)
                        attrib.update(gui_info_this)
                        dcm_obj = DcmInfoGui(**attrib)
                    else:
                        if len(nm_radius) > 0:
                            gui_info_this = info_extract_frame(
                                frame, {'nm_radius': nm_radius})
                            gui_info_this['nm_radius'] = float(
                                gui_info_this['nm_radius'][0])
                            attrib.update(gui_info_this)
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
        frame-infos as vectors

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
                    out_dict[key].append(
                        ''.join([tag[0], str(tag[1][frame]), tag[2]]))
                    #TODO format for tag1....
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

    for idx_pattern in range(len(tag_pattern.list_tags)):

        attribute_name = tag_pattern.list_tags[idx_pattern]
        tag_idx = [
            i for i, item in enumerate(attribute_names)
            if item == attribute_name]

        if hasattr(tag_pattern, 'list_format'):
            prefix, format_string, suffix = get_format_strings(
                tag_pattern.list_format[idx_pattern])
        else:
            prefix, format_string, suffix = ('', '', '')
        if prefix == '' and attr_name_default_prefix:
            prefix = f'{attribute_name}:'
        info = f'{prefix}{prefix_separator}{not_found_text}'

        for idx in tag_idx:
            data_element = get_tag_data(pydict, tag_info=tag_infos[idx])
            if data_element is not None:
                if suffix == '' and unit_default_suffix:
                    suffix = f'{tag_infos[idx].unit}'
                val = data_element.value
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                if data_element.VM > 1:
                    value_id = tag_infos[idx].value_id
                    if value_id == -2:  # per frame, check if possible
                        if data_element.VM != int(pydict.get('NumberOfFrames', -1)):
                            value_id = -1
                    if value_id > -1:  # specific id, check if possible
                        if value_id > data_element.VM:
                            value_id = -1
                    multi_val = False
                    if value_id == -1:
                        multi_val = True  # combine values to text
                        val_text = val
                    elif value_id == -2:
                        val_text = val
                    else:
                        val_text = val[value_id]
                    if format_string != '':
                        if isinstance(
                                val[0], str) and format_string[-1] == 'f':
                            try:
                                val = [float(x) for x in val]
                                val_text = [
                                    f'{x:{format_string}}' for x in val]
                            except ValueError:
                                pass
                        else:
                            val_text = [
                                f'{x:{format_string}}' for x in val]
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
                    val_text = val
                    if format_string != '':
                        if isinstance(val, str) and format_string[-1] == 'f':
                            try:
                                val = float(val)
                                val_text = f'{val:{format_string}}'
                            except ValueError:
                                pass
                        elif isinstance(val, str) and format_string[0] == '0':
                            n_first = int(format_string)
                            val_text = f'{val[:n_first]}'
                        else:
                            try:
                                val_text = f'{val:{format_string}}'
                            except TypeError:
                                val_text = '-'
                                pass
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


def get_img(filepath, frame_number=-1, tag_patterns=[], tag_infos=None):
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

    Returns
    -------
    image : np.array
        pixelvalues
    tag_strings : list of list of str
        for each tag_pattern list dicom (formatted) values
    """
    npout = None
    tag_strings = []
    pd = None
    try:
        pd = pydicom.dcmread(filepath)
    except pydicom.errors.InvalidDicomError:
        pass
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
            try:
                if frame_number == -1:
                    npout = pd.pixel_array * pd.get('RescaleSlope', 1.) \
                        + pd.get('RescaleIntercept', 0.)
                else:
                    npout = pd.pixel_array[frame_number, :, :] * pd.get('RescaleSlope', 1.) \
                        + pd.get('RescaleIntercept', 0.)
            except AttributeError:
                npout = pixarr

        if overlay is not None:
            if pixarr is not None:
                if npout.shape == overlay.shape:
                    npout = np.add(npout, overlay)
                else:
                    pass  # implement when example file exists
                    # overlay_offset at 0x6000, 0x0050
            else:
                npout = overlay

        if len(tag_patterns) > 0 and tag_infos is not None:
            tag_strings = read_tag_patterns(
                pd, tag_patterns, tag_infos, frame_number=frame_number)

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


def get_tag_data(pd, tag_info=None):
    """Read content of specific Dicom tag from file pydicom dictionary.

    Only first occurence of sequence

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
        try:
            if seq[0] != '':
                pd_sub = pd[seq[0]][0]
                if len(seq) > 1:
                    for i in range(1, len(seq)):
                        pd_sub = pd_sub[seq[i]][0]
                data_element = pd_sub[gr, el]
            else:
                data_element = pd[gr, el]
        except KeyError:
            pass
        except IndexError:
            pass

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
            dcm_files = [x for x in p.glob(glob_string) if x.suffix == 'dcm']
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
    infos = []
    for i in range(len(img_infos)):
        info = get_tags(img_infos.filepath[i],
                        frame_number=img_infos.frame_number[i],
                        tag_patterns=[tag_pattern_sort],
                        tag_infos=tag_infos)
        infos.append(info)
    print(infos)
    #sorted_img_infos = 
    pass
    #return sorted_img_infos


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
    tags = []
    attribute_names = []
    if sequence_list[0] == '':
        attributes = pd.dir()
        for attr in attributes:
            try:
                data_element = pd[attr]
                attribute_names.append(attr)
                tags.append(data_element.tag)
            except KeyError:
                pass
    else:
        try:
            pd_sub = pd[sequence_list[0]][0]
            if len(sequence_list) > 1:
                for i in range(1, len(sequence_list)):
                    pd_sub = pd_sub[sequence_list[i]][0]
            attributes = pd_sub.dir()
            for attr in attributes:
                data_element = pd_sub[attr]
                attribute_names.append(attr)
                tags.append(data_element.tag)
        except IndexError:
            pass  # e.g. SQ with zero elements
        #except KeyError:
        #    pass

    return {'tags': tags, 'attribute_names': attribute_names}


def sum_marked_images(img_infos, testcode=''):
    """Calculate sum of marked images.

    Parameters
    ----------
    img_infos : list of Dcm(Gui)Info
        as defined in scripts/dcm.py
    testcode : str
        apply if specific testcode and quicktest active

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
        include = False
        if testcode == '':
            include = img_info.marked
        else:
            include = testcode in img_info.marked_quicktest
        print(f'include {include} img_info.marked_quicktest {img_info.marked_quicktest}')
        if include:
            image, tags = get_img(
                img_info.filepath,
                frame_number=img_info.frame_number
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
