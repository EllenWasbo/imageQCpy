#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions used for configuration settings.

@author: Ellen Wasbo
"""
import os
import copy
from pathlib import Path
from time import time, ctime
from dataclasses import asdict
import numpy as np

import yaml
from PyQt5.QtCore import QFile, QIODevice, QTextStream
from PyQt5.QtWidgets import QMessageBox, QFileDialog

# imageQC block start
from imageQC.config.iQCconstants import (
    USERNAME, APPDATA, TEMPDIR, ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER,
    CONFIG_FNAMES, USER_PREFS_FNAME, QUICKTEST_OPTIONS, VERSION, ALTERNATIVES
    )
import imageQC.config.config_classes as cfc
from imageQC.ui import messageboxes
from imageQC.scripts.mini_methods import get_included_tags, get_all_matches
# imageQC block end


def calculate_version_difference(version_string, reference_version=None):
    """Calculate version difference for comparing versions.

    x.y.z-bw = x*1000000 + y*10000 + z*100 + w (i.e. max 100 for each subversion)

    Parameters
    ----------
    version_string : str
        version string to compare VERSION (current version) to
    reference_version : str or None
        to override VERSION

    Returns
    -------
    version_difference : int
        this_version - input version = if negative the current version is older
    """
    def calculate_version_number(vstring):
        version_number = 0
        if len(vstring) > 0:
            main_str = vstring
            beta_str = ''
            try:
                main_str, beta_str = vstring.split('-b')
            except ValueError:  # no beta or not valid main
                pass
            try:
                main_x, main_y, main_z = main_str.split('.')
                version_number = 1e6*int(main_x) + 1e4*int(main_y) + 1e2*int(main_z)
                if beta_str != '':
                    version_number = version_number + int(beta_str)
                else:
                    version_number += 100
                    # assume max b99 and let beta versions have lower number than
                    # final version without beta indicator
            except ValueError:
                print(f'version string {vstring} not as expected. Failed converting '
                      '(config_func - calculate_version_number).')

        return version_number

    if reference_version is None:
        reference_version = VERSION
    version_difference = (
        calculate_version_number(reference_version)
        - calculate_version_number(version_string))

    return version_difference


def version_control(input_main):
    """Compare version number of settings with current for updates.

    Parameters
    ----------
    input_main : MainWindow or InputMain
    """
    _, _, last_mod = load_settings(fname='last_modified')

    # tag infos
    res = getattr(last_mod, 'tag_infos')
    path = get_config_filename('tag_infos')
    if len(res) > 0 or path:
        if len(res) < 3:
            version_string = ''
        else:
            _, _, version_string = res

        version_diff = calculate_version_difference(version_string)
        if version_diff > 0:  # current version newer than saved tag_infos

            # compare protected tags in current versus saved version
            res = tag_infos_difference_default(input_main.tag_infos)
            change, added, adjusted, mammo_changes, new_tag_infos = res
            tags_in_added = [
                '\t' + str((tag.attribute_name, tag.tag, tag.sequence, tag.limited2mod))
                for tag in added]
            tags_adjusted = [
                '\t' + str((tag.attribute_name, tag.tag, tag.sequence, tag.limited2mod))
                for tag in adjusted]
            if change:
                if 'MainWindow' not in str(type(input_main)):
                    print('Warning: Current version of imageQC is newer than the '
                          'version used to save the DICOM tag settings. Consider '
                          'running imageQC GUI version to update settings.')
                else:
                    if len(tags_in_added):
                        tags_in_added.insert(0, 'Add tags:')
                    if len(tags_adjusted):
                        tags_adjusted.insert(
                            0, 'Changed protection, unit or modalities:')
                    mammo_msg = ['']
                    if mammo_changes:
                        mammo_msg = [
                            'Mammo added as modality. All tag infos marked for '
                            'Xray also marked for Mammo. '
                            'Some specific mammo tags added.']
                    res = messageboxes.QuestionBox(
                        parent=input_main, title='Update tag infos with new defaults?',
                        msg=(
                            'The current version of imageQC is newer than the '
                            'version previously used to save DICOM tag settings. '
                            'Found default tags missing in your saved version '
                            'and/or changes to protection, unit or modality settings. '
                            'Add missing and update current settings?'
                            ),
                        info='Find added and changed tags in details.',
                        details=(tags_in_added + tags_adjusted + mammo_msg),
                        msg_width=800
                        )
                    if res.exec():
                        if len(tags_in_added) > 0:
                            reply = QMessageBox.question(
                                input_main, 'Sort tags?',
                                'Added tags are currently at end of list. '
                                'Sort tags alphabetically?',
                                QMessageBox.Yes, QMessageBox.No)
                            if reply == QMessageBox.Yes:
                                new_tag_infos = sorted(
                                    new_tag_infos,
                                    key=lambda x: x.attribute_name.upper())
                        taginfos_reset_sort_index(new_tag_infos)
                        _, _ = save_settings(
                            new_tag_infos, fname='tag_infos')
                    else:
                        reply = QMessageBox.question(
                            input_main, 'Keep asking?',
                            'Ask again next time on startup?',
                            QMessageBox.Yes, QMessageBox.No)
                        if reply == QMessageBox.No:
                            # update version number
                            _, _ = save_settings(
                                input_main.tag_infos, fname='tag_infos')

    # paramset ROI offset_xy_mm False together with roi_use_table 1?
    _, _, paramsets = load_settings(fname='paramsets')
    warnings = []
    for mod, paramset_mod in paramsets.items():
        res = getattr(last_mod, f'paramsets_{mod}')
        proceed = False
        if len(res) == 3:
            _, _, version_string = res
            diff = calculate_version_difference(
                version_string, reference_version='3.0.6')
            if diff > 0:
                proceed = True
        else:
            proceed = True

        if proceed:
            for paramset in paramset_mod:
                try:
                    if paramset.roi_use_table == 1:
                        warnings.append(f'{mod}: {paramset.label}')
                except AttributeError:  # e.g. SR modality without ROI test
                    pass

    if len(warnings) > 0:
        dlg = messageboxes.MessageBoxWithDetails(
            input_main, title='Warnings',
            msg='For test ROI using table of ROIs with same shape the behaviour have '
            'been inconsistent in terms of indicating mm or pixels. This will now be '
            'in mm as the headers always have indicated. Some of your parameter sets '
            'make use of this option. Verify and possibly correct the use of the '
            'ROIs as intented for these parameter sets and save '
            'to avoid this warning to appear once again.',
            info='See details for which Parameter sets this might be an issue',
            icon=QMessageBox.Warning,
            details=warnings)
        dlg.exec()

    # remove version info if file missing
    fnames = [a for a in dir(last_mod) if not a.startswith('__')]
    changes = False
    for fname in fnames:
        res = getattr(last_mod, fname)
        path = get_config_filename(fname)
        if path == '' and len(res) > 0:
            setattr(last_mod, fname, [])
            changes = True
    if changes:
        _, _ = save_settings(last_mod, fname='last_modified')


def verify_config_folder(widget):
    """Test whether config folder exist, ask to create if not.

    Parameters
    ----------
    widget : QWidget
        calling widget

    Returns
    -------
    proceed : bool
        continue to save - config folder is ready
    """
    proceed = True
    if get_config_folder() == '':
        proceed = False
        quest = '''Config folder not specified.
        Do you want to locate or initate a config folder now?'''
        msg_box = QMessageBox(
            QMessageBox.Question,
            'Proceed?', quest,
            buttons=QMessageBox.Yes | QMessageBox.No,
            parent=widget
            )
        res = msg_box.exec_()
        if res == QMessageBox.Yes:
            proceed = True
        if proceed:
            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.Directory)
            if dlg.exec():
                fname = dlg.selectedFiles()
                os.environ[ENV_CONFIG_FOLDER] = fname[0]
                _, user_path, user_prefs = load_user_prefs()
                if user_path != '':
                    user_prefs.config_folder = os.environ[ENV_CONFIG_FOLDER]
                    _, _ = save_user_prefs(user_prefs, parentwidget=widget)

    return proceed


def get_active_users():
    """Get list of active usernames sharing the config folder.

    Returns
    -------
    active_users : dict
        dict from active_users.yaml
    """
    path = get_config_filename('active_users')
    active_users = {}

    if path != '':
        with open(path, 'r') as file:
            active_users = yaml.safe_load(file)

    return active_users


def add_user_to_active_users():
    """Add current user to yaml holding active users of the config folder."""
    path = os.environ[ENV_CONFIG_FOLDER]
    if path != '':
        active_users = {}
        path = get_config_filename('active_users', force=True)
        if os.path.exists(path):
            with open(path, 'r') as file:
                active_users = yaml.safe_load(file)

        active_users[USERNAME] = ctime()

        if os.access(Path(path).parent, os.W_OK):
            with open(path, 'w') as file:
                yaml.safe_dump(
                    active_users, file,
                    default_flow_style=None, sort_keys=False)


def remove_user_from_active_users():
    """Remove current user from yaml holding active users."""
    path = get_config_filename('active_users')
    if path != '':
        active_users = {}
        with open(path, 'r') as file:
            active_users = yaml.safe_load(file)

        active_users.pop(USERNAME, None)

        if os.access(Path(path).parent, os.W_OK):
            with open(path, 'w') as file:
                yaml.safe_dump(
                    active_users, file,
                    default_flow_style=None, sort_keys=False)


def init_user_prefs(path=APPDATA, config_folder=''):
    """Initiate empty local folder/file optionally with config_folder set.

    Parameters
    ----------
    path : str
        path to folder for saving user prefs
    config_folder : str
        path to config folder if specified

    Returns
    -------
    status: bool
        False if failed initiating user preferences and config folder
    path_out: str
        path (or '' if failed)
    errmsg: str
        error message
    """
    status = False
    errmsg = ''
    path_out = ''

    # initiate APPDATA or TEMPDIR imageQC if missing
    if os.path.exists(path) is False:
        if os.access(Path(path).parent, os.W_OK):
            os.mkdir(path)
        else:
            errmsg = '\n'.join(['Missing writing permission:',
                                str(Path(path).parent)])

    if errmsg == '':
        userpref = cfc.UserPreferences()
        userpref.config_folder = config_folder

        path_out = os.path.join(path, USER_PREFS_FNAME)

        if os.access(path, os.W_OK):
            with open(path_out, 'w') as file:
                yaml.safe_dump(
                    asdict(userpref), file,
                    default_flow_style=None, sort_keys=False)
            status = True
        else:
            errmsg = '\n'.join(['Missing writing permission:', path_out])
            path_out = ''

    if errmsg != '':
        errmsg = '\n'.join([errmsg,
                            'Saving settings is not possible.'])

    return (status, path_out, errmsg)


def verify_input_dict(dict_input, default_object):
    """Verify input from yaml if config classes change on newer versions.

    Remove old keywords from input.

    Parameters
    ----------
    dict_input : dict
        dictionary to verify
    default_object : object
        object to compare attributes vs dict_input keys

    Returns
    -------
    updated_dict : dict
        updated input_dict with valid keys
    """
    default_dict = asdict(default_object)
    actual_keys = [*default_dict]
    updated_dict = {k: v for k, v in dict_input.items() if k in actual_keys}

    # specific changes
    if default_object == cfc.HUnumberTable():
        if 'materials' in [*dict_input]:
            updated_dict['labels'] = dict_input['materials']
        if 'relative_mass_density' in [*dict_input]:
            updated_dict['linearity_axis'] = dict_input['relative_mass_density']
            updated_dict['linearity_unit'] = 'Rel. mass density'

    return updated_dict


def save_user_prefs(userpref, parentwidget=None):
    """Save user preferences to user_preferences.yaml file.

    Parameters
    ----------
    userpref : object of class UserPreferences
    parentwidget : widget
        parent for displaying messages

    Returns
    -------
    bool
        False if failed saving user preferences
    str
        file path to save to
    """
    status = False
    try:
        path = os.environ[ENV_USER_PREFS_PATH]
    except KeyError:
        if parentwidget is not None:
            quest = 'Save user_preferences.yaml in:'
            res = messageboxes.QuestionBox(
                parentwidget, title='Save as', msg=quest,
                yes_text=f'{APPDATA}', no_text=f'{TEMPDIR}')
            path_local = APPDATA if res.exec() == 0 else TEMPDIR
            _, path, userpref = init_user_prefs(
                    path=path_local, config_folder=os.environ[ENV_CONFIG_FOLDER])
        else:
            path = ''

    if path != '':
        if os.access(Path(path).parent, os.W_OK):
            with open(path, 'w') as file:
                yaml.safe_dump(
                    asdict(userpref), file,
                    default_flow_style=None, sort_keys=False)
            status = True
            os.environ[ENV_CONFIG_FOLDER] = userpref.config_folder

    return (status, path)


def load_user_prefs():
    """Load UserPreferences from yaml file.

    Returns
    -------
    bool
        True if yaml file found
    str
        file path where tried to load from
    UserPreferences
    """
    status = True
    userprefs = None
    path = ''
    try:
        path = os.environ[ENV_USER_PREFS_PATH]
    except KeyError:
        path = os.path.join(APPDATA, USER_PREFS_FNAME)
        if os.path.exists(path) is False:
            path = os.path.join(TEMPDIR, USER_PREFS_FNAME)  # try with TEMPDIR

    if os.path.exists(path):
        with open(path, 'r') as file:
            doc = yaml.safe_load(file)
            updated_doc = verify_input_dict(doc, cfc.UserPreferences())
            userprefs = cfc.UserPreferences(**updated_doc)

    if userprefs is None:
        status = False
        path = ''
        userprefs = cfc.UserPreferences()

    return (status, path, userprefs)


def get_config_folder():
    """Get config folder currently set.

    Returns
    -------
    str
        Config folder if exists else empty string.
    """
    try:
        path = os.environ[ENV_CONFIG_FOLDER]
        config_folder = path if os.path.exists(path) else ''
    except KeyError:
        config_folder = ''

    return config_folder


def get_config_filename(fname, force=False):
    """Verify if yaml file exists in config folder.

    Parameters
    ----------
    fname : str
        filename as defined in CONFIG_FNAMES (or + _<modality> if paramsets)
    force : bool
        force return filename even though it does not exist

    Returns
    -------
    str
        full path to yaml file if it exist, empty if not verified
    """
    path = ''
    if os.environ[ENV_CONFIG_FOLDER] != '':
        path_temp = os.path.join(
            os.environ[ENV_CONFIG_FOLDER], fname + '.yaml')
        path_temp = os.path.normpath(path_temp)
        if os.path.exists(path_temp):
            path = path_temp
        else:
            if force:
                path = path_temp

    return path


def load_default_dcm_test_tag_patterns():
    """Load default TagPatternsFormat for exporting DCM data (test DCM).

    Returns
    -------
    tag_patterns : dict of TagPatternFormat
        predefined defaults for tests DCM
    """
    settings = {}
    file = QFile(":/config_defaults/tag_patterns_test_dcm.yaml")
    file.open(QFile.ReadOnly | QFile.Text)
    f_text = QTextStream(file).readAll()
    docs = yaml.safe_load(f_text)
    for mod, doc in docs.items():
        for temp in doc:
            settings[mod] = cfc.TagPatternFormat(**temp)

    return settings


def load_default_ct_number_tables():
    """Load default HUnumbertables.

    Returns
    -------
    ct_number_tables : dict of HUnumberTable
    """
    ct_number_tables = {}
    file = QFile(":/config_defaults/CTnumberSensitometry.yaml")
    if file.open(QIODevice.ReadOnly | QFile.Text):
        data = QTextStream(file).readAll()
        file.close()
        docs = yaml.safe_load_all(data)
        for doc in docs:
            for key, ct_tab in doc.items():
                upd = verify_input_dict(ct_tab, cfc.HUnumberTable())
                ct_number_tables[key] = cfc.HUnumberTable(**upd)

    return ct_number_tables


def load_default_pos_tables(filename='Siemens_AutoQC'):
    """Load default PositionTables.

    Returns
    -------
    tables : dict of PositionTable
    """
    tables = {}
    file = QFile(f":/config_defaults/{filename}.yaml")
    if file.open(QIODevice.ReadOnly | QFile.Text):
        data = QTextStream(file).readAll()
        file.close()
        docs = yaml.safe_load_all(data)
        for doc in docs:
            for key, ct_tab in doc.items():
                upd = verify_input_dict(ct_tab, cfc.PositionTable())
                tables[key] = cfc.PositionTable(**upd)

    return tables


def load_cdmam(filename='cdmam'):
    """Load CDMAM diameters and thickness.

    Returns
    -------
    tables : dict
        keys: thickness, diameters
    """
    tables = {}
    file = QFile(f":/config_defaults/{filename}.yaml")
    if file.open(QIODevice.ReadOnly | QFile.Text):
        data = QTextStream(file).readAll()
        file.close()
        docs = yaml.safe_load_all(data)
        for doc in docs:
            for key, tab in doc.items():
                tables[key] = tab
    return tables

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
        path_obj = Path(path)
        if path_obj.exists() is False:
            if USERNAME != path_obj.parts[2]:
                path = os.path.join(*path_obj.parts[:2], USERNAME, *path_obj.parts[3:])
    return path


def get_modality_of_paramset(paramset):
    """Get modality_string for a parameterset.

    Parameters
    ----------
    paramset : object
        ParamSet<mod> as defined in config_classes.py

    Returns
    -------
    modality_string : str
    """
    class_name = type(paramset).__name__
    modality_string = class_name.replace('ParamSet', '')
    return modality_string


def load_paramsets(fnames, path):
    """Load paramsets (subprocess of load_settings).

    Parameters
    ----------
    fnames : list of str
        file base name of specific or all paramsets (paramset_<modality>)
    path : str
        parent path for paramset yaml files

    Returns
    -------
    settings : list or dict
        list if single modality, dict if all modalities
    return_default : bool
        True if no saved paramset found
    """
    return_default = False
    settings = {} if len(fnames) > 1 else []

    for fname_this in fnames:
        sett_this = []
        if len(fnames) > 1:
            path_this = str(Path(path) / f'{fname_this}.yaml')
        else:
            path_this = path
        if Path(path_this).exists():
            try:
                with open(path_this, 'r') as file:
                    docs = yaml.safe_load_all(file)
                    for doc in docs:
                        upd = verify_input_dict(doc['dcm_tagpattern'],
                                                cfc.TagPatternFormat())
                        doc['dcm_tagpattern'] = cfc.TagPatternFormat(
                            **upd)
                        if 'group_tagpattern' in doc:
                            upd = verify_input_dict(doc['group_tagpattern'],
                                                    cfc.TagPatternFormat())
                            doc['group_tagpattern'] = cfc.TagPatternFormat(
                                **upd)
                        tests = {}
                        for key, test in doc['output']['tests'].items():
                            tests[key] = []
                            for sub in test:
                                upd = verify_input_dict(
                                    sub, cfc.QuickTestOutputSub())
                                tests[key].append(
                                    cfc.QuickTestOutputSub(**upd))
                        try:
                            decimal_all = doc['output']['decimal_all']
                        except KeyError:
                            decimal_all = False
                        doc['output'] = cfc.QuickTestOutputTemplate(
                            include_header=doc[
                                'output']['include_header'],
                            transpose_table=doc[
                                'output']['transpose_table'],
                            decimal_mark=doc['output']['decimal_mark'],
                            decimal_all=decimal_all,
                            include_filename=doc[
                                'output']['include_filename'],
                            group_by=doc['output']['group_by'],
                            tests=tests)
                        modality = fname_this.split('_')[1]
                        if 'roi_table' in doc:
                            upd = verify_input_dict(doc['roi_table'],
                                                    cfc.PositionTable())
                            doc['roi_table'] = cfc.PositionTable(**upd)
                        if 'num_table' in doc:
                            upd = verify_input_dict(doc['num_table'],
                                                    cfc.PositionTable())
                            doc['num_table'] = cfc.PositionTable(**upd)
                        if modality == 'CT':
                            if 'ctn_table' in doc:
                                upd = verify_input_dict(doc['ctn_table'],
                                                        cfc.HUnumberTable())
                                doc['ctn_table'] = cfc.HUnumberTable(**upd)
                            if 'ttf_table' in doc:
                                upd = verify_input_dict(doc['ttf_table'],
                                                        cfc.PositionTable())
                                doc['ttf_table'] = cfc.PositionTable(**upd)
                        elif modality == 'Mammo':
                            upd = verify_input_dict(doc['gho_table'],
                                                    cfc.PositionTable())
                            doc['gho_table'] = cfc.PositionTable(**upd)
                        elif modality == 'PET':
                            if 'rec_table' in doc:
                                upd = verify_input_dict(
                                    doc['rec_table'], cfc.RecTable())
                                doc['rec_table'] = cfc.RecTable(**upd)

                        if 'task_based' in fname_this:
                            class_ = cfc.ParamSetCT_TaskBased
                        else:
                            class_ = getattr(cfc, f'ParamSet{modality}')
                        upd = verify_input_dict(doc, class_())
                        sett_this.append(class_(**upd))

                        if len(fnames) == 1:
                            settings = sett_this
                        else:
                            settings[modality] = sett_this
            except OSError as error:
                print(
                    f'config_func.py load_settings {fname_this}: '
                    f'{str(error)}')
                return_default = True
        else:
            return_default = True

    if return_default:
        default_tags_dcm = load_default_dcm_test_tag_patterns()
        if len(fnames) > 1:  # load as dict
            for mod in QUICKTEST_OPTIONS:
                if mod not in settings:
                    class_ = getattr(cfc, f'ParamSet{mod}')
                    settings[mod] = [class_(
                        dcm_tagpattern=default_tags_dcm[mod])]
        else:
            if 'task_based' in fnames[0]:
                settings = [cfc.ParamSetCT_TaskBased(
                    dcm_tagpattern=cfc.TagPatternFormat(
                        list_tags=[
                            'SeriesDescription', 'ProtocolName',
                            'KVP', 'mA', 'ExposureTime', 'ConvolutionKernel',
                            'CTDIvol', 'SliceThickness',
                            'AcquisitionDate', 'AcquisitionTime',
                            'InstanceNumber'],
                        list_format=[
                            '', '',
                            '|:.0f|', '|:.1f|', '|:.1f|', '',
                            '|:.2f|', '|:.1f|',
                            '', '|:.0f|',
                            '|:.0f|']
                        ),
                    group_tagpattern=cfc.TagPatternFormat(
                        list_tags=['ConvolutionKernel', 'mAs'],
                        list_format=['', '']
                        )
                    )]
            else:
                mod = fnames[0].split('_')[1]
                class_ = getattr(cfc, f'ParamSet{mod}')
                settings = [class_(dcm_tagpattern=default_tags_dcm[mod])]

    return settings


def load_settings(fname='', temp_config_folder=''):
    """Load settings from yaml file in config folder.

    Parameters
    ----------
    fname : str
        yaml filename without folder and extension
    temp_config_folder : str
        temporary config folder e.g. when import. Default is '' (ignored)

    Returns
    -------
    bool
        True if success
    str
        full path of file tried to load from
    object
        structured objects defined by the corresponding dataclass
    """
    status = False
    path = ''
    settings = None

    if fname != '':
        return_default = False
        if fname == 'paramsets':
            all_paramsets = True
            path = (
                get_config_folder() if temp_config_folder == '' else temp_config_folder)
        else:
            all_paramsets = False
            if temp_config_folder == '':
                path = get_config_filename(fname)
            else:
                path = str(Path(temp_config_folder) / f'{fname}.yaml')
        fname_ = 'paramsets' if 'paramsets' in fname else fname

        if path != '' or all_paramsets:
            if CONFIG_FNAMES[fname_]['saved_as'] == 'object_list':
                if fname_ != 'paramsets':
                    try:
                        with open(path, 'r') as file:
                            docs = yaml.safe_load_all(file)
                            settings = []
                            for doc in docs:
                                if fname == 'tag_infos':
                                    updated_doc = verify_input_dict(doc, cfc.TagInfo())
                                    settings.append(cfc.TagInfo(**updated_doc))
                        if fname == 'tag_infos':
                            taginfos_reset_sort_index(settings)
                    except OSError as error:
                        print(f'config_func.py load_settings {fname}: {str(error)}')
                        return_default = True

                else:  # paramsets
                    if all_paramsets:  # load as dict
                        fnames = [f'paramsets_{m}' for m in QUICKTEST_OPTIONS]
                    else:
                        fnames = [fname]
                    settings = load_paramsets(fnames, path)
                status = True

            elif CONFIG_FNAMES[fname_]['saved_as'] == 'modality_dict':
                settings = {
                    modality: [] for modality in [*QUICKTEST_OPTIONS]}
                try:
                    with open(path, 'r') as file:
                        docs = yaml.safe_load(file)
                        for mod, doc in docs.items():
                            settings[mod] = []
                            for temp in doc:
                                if fname == 'quicktest_templates':
                                    upd = verify_input_dict(
                                        temp, cfc.QuickTestTemplate())
                                    settings[mod].append(
                                        cfc.QuickTestTemplate(**upd))
                                elif fname == 'auto_templates':
                                    upd = verify_input_dict(
                                        temp['sort_pattern'], cfc.TagPatternSort())
                                    temp['sort_pattern'] = (
                                        cfc.TagPatternSort(**upd))
                                    upd = verify_input_dict(temp, cfc.AutoTemplate())
                                    settings[mod].append(cfc.AutoTemplate(**upd))
                                elif fname == 'auto_vendor_templates':
                                    upd = verify_input_dict(
                                        temp, cfc.AutoVendorTemplate())
                                    settings[mod].append(
                                        cfc.AutoVendorTemplate(**upd))
                                elif fname == 'rename_patterns':
                                    upd = verify_input_dict(
                                        temp, cfc.RenamePattern())
                                    settings[mod].append(
                                        cfc.RenamePattern(**upd))
                                elif fname == 'tag_patterns_sort':
                                    upd = verify_input_dict(
                                        temp, cfc.TagPatternSort())
                                    settings[mod].append(
                                        cfc.TagPatternSort(**upd))
                                elif fname == 'tag_patterns_format':
                                    upd = verify_input_dict(
                                        temp, cfc.TagPatternFormat())
                                    settings[mod].append(
                                        cfc.TagPatternFormat(**upd))
                                elif fname == 'tag_patterns_special':
                                    upd = verify_input_dict(
                                        temp, cfc.TagPatternFormat())
                                    settings[mod].append(
                                        cfc.TagPatternFormat(**upd))
                                elif fname == 'digit_templates':
                                    upd = verify_input_dict(
                                        temp, cfc.DigitTemplate())
                                    settings[mod].append(
                                        cfc.DigitTemplate(**upd))
                                elif fname == 'limits_and_plot_templates':
                                    upd = verify_input_dict(
                                        temp, cfc.LimitsAndPlotTemplate())
                                    settings[mod].append(
                                        cfc.LimitsAndPlotTemplate(**upd))
                                elif fname == 'report_templates':
                                    elements = []
                                    for elem in temp['elements']:
                                        if isinstance(elem, list):
                                            elements.append([])
                                            for sub_elem in elem:
                                                upd = verify_input_dict(
                                                    sub_elem, cfc.ReportElement())
                                                elements[-1].append(cfc.ReportElement(**upd))
                                        else:
                                            upd = verify_input_dict(
                                                elem, cfc.ReportElement())
                                            elements.append(cfc.ReportElement(**upd))
                                    upd = verify_input_dict(
                                        temp, cfc.ReportTemplate())
                                    temp = cfc.ReportTemplate(**upd)
                                    temp.elements = elements
                                    settings[mod].append(temp)

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
                    status = True
                except Exception as error:
                    print(f'config_func.py load_settings: {str(error)}')
                    return_default = True
                len_items = [len(mod_list) for key, mod_list in settings.items()]
                if 0 in len_items:
                    idx = len_items.index(0)
                    mod = [*QUICKTEST_OPTIONS][idx]
                    settings[mod] = copy.deepcopy(CONFIG_FNAMES[fname]['default'][mod])
            else:  # settings as one object
                try:
                    with open(path, 'r') as file:
                        doc = yaml.safe_load(file)
                        if fname == 'auto_common':
                            upd = verify_input_dict(
                                doc['filename_pattern'], cfc.TagPatternFormat())
                            doc['filename_pattern'] = cfc.TagPatternFormat(**upd)
                            upd = verify_input_dict(doc, cfc.AutoCommon())
                            settings = cfc.AutoCommon(**upd)
                            settings.import_path = convert_OneDrive(
                                settings.import_path)
                        elif fname == 'dash_settings':
                            upd = verify_input_dict(doc, cfc.DashSettings())
                            settings = cfc.DashSettings(**upd)
                        elif fname == 'last_modified':
                            upd = verify_input_dict(doc, cfc.LastModified())
                            settings = cfc.LastModified(**upd)
                    status = True
                except OSError as error:
                    print(f'config_func.py load_settings {fname}: {str(error)}')
                    return_default = True
        else:
            return_default = True

        if return_default:
            if 'paramsets' in fname:
                settings = load_paramsets([fname], '--')
            else:
                settings = CONFIG_FNAMES[fname]['default']
                if fname == 'tag_infos':
                    taginfos_reset_sort_index(settings)

    return (status, path, settings)


def check_save_conflict(fname, lastload):
    """Check if config file modified (by others) after last load.

    Parameters
    ----------
    fname : str
        yaml filename to check
    lastload : float
        epoch time of last load of the yaml file before trying to save.

    Returns
    -------
    proceed : bool
        proceed to save
    errmsg : str
        Errormessage if proceed False
    """
    status = True
    errmsg = ''
    path = get_config_filename(fname)
    if os.path.exists(path):
        if os.path.getmtime(path) > lastload:
            _, path, last_mod = load_settings(fname='last_modified')
            res = getattr(last_mod, fname)
            if len(res) == 2:
                user, modtime = res
                version_string = ''
            else:
                user, modtime, version_string = res

            version_difference = calculate_version_difference(version_string)
            if version_difference < 0:
                errmsg = errmsg + (
                    f' Current imageQC version {VERSION} is older than the one '
                    f'{user} used for saving {fname} ({version_string}).'
                    'saved parameters of new features might get lost if saved.'
                    )

            if user != USERNAME:
                if modtime > lastload:
                    errmsg = f'It seems that {user} is also editing this config file.'
                else:
                    errmsg = 'It seems that this file has been edited recently.'

                errmsg = (
                    errmsg +
                    '\nProceed saving and possibly overwrite changes done by others?')
            if errmsg != '':
                status = False

    return status, errmsg


def update_last_modified(fname=''):
    """Update last_modified.yaml."""
    _, _, last_mod = load_settings(fname='last_modified')
    setattr(last_mod, fname, [USERNAME, time(), VERSION])
    _, _ = save_settings(last_mod, fname='last_modified')


def save_settings(settings, fname=''):
    """Save settings to yaml file.

    Parameters
    ----------
    settings : object
        object of a settings dataclass
    fname : str
        filename without folder and extension

    Returns
    -------
    bool
        False if failed saving
    str
        filepath where tried to save
    """
    status = False
    path = ''

    def try_save(input_data, override_path=None):
        path_to_use = path
        status = False
        try_again = False
        if override_path is not None:  # for testing/troubleshooting
            path_to_use = override_path
        try:
            with open(path_to_use, 'w') as file:
                if isinstance(input_data, list):
                    yaml.safe_dump_all(
                        input_data, file, default_flow_style=None, sort_keys=False)
                else:
                    yaml.safe_dump(
                        input_data, file, default_flow_style=None, sort_keys=False)
            status = True
        except yaml.YAMLError:
            # try once more with eval(str(input_data))
            try_again = True
        except IOError as io_error:
            QMessageBox.warning(None, "Failed saving",
                                f'Failed saving to {path_to_use} {io_error}')
        if try_again:
            try:
                input_data = eval(str(input_data))
                with open(path_to_use, 'w') as file:
                    if isinstance(input_data, list):
                        yaml.safe_dump_all(
                            input_data, file, default_flow_style=None, sort_keys=False)
                    else:
                        yaml.safe_dump(
                            input_data, file, default_flow_style=None, sort_keys=False)
                status = True
            except yaml.YAMLError as yaml_error:
                QMessageBox.warning(None, 'Failed saving',
                                    f'Failed saving to {path_to_use} {yaml_error}')
        return status

    if fname != '':
        proceed = False
        fname_input = fname
        if fname == 'paramsets':
            all_paramsets = True
            path = get_config_folder()
            if os.access(Path(path), os.W_OK):
                proceed = True
        else:
            all_paramsets = False
            path = get_config_filename(fname, force=True)
            if os.access(Path(path).parent, os.W_OK):
                proceed = True
            if 'paramsets' in fname:
                fname = 'paramsets'
        if proceed:
            if CONFIG_FNAMES[fname]['saved_as'] == 'object_list':
                if all_paramsets:
                    status_all = []
                    folder = path
                    for mod, sett_this in settings.items():
                        listofdict = [asdict(temp) for temp in sett_this]
                        path = str(Path(folder) / f'paramset_{mod}.yaml')
                        status_this = try_save(listofdict)
                        status_all.append(status_this)
                        update_last_modified(f'paramset_{mod}')
                    path = folder  # output
                    status = all(status_all)
                else:
                    listofdict = [asdict(temp) for temp in settings]
                    status = try_save(listofdict)
            elif CONFIG_FNAMES[fname]['saved_as'] == 'modality_dict':
                temp_dict = {}
                for key, val in settings.items():
                    temp_dict[key] = [asdict(temp) for temp in val]
                if temp_dict:
                    status = try_save(temp_dict)
            else:
                status = try_save(asdict(settings))

        if fname != 'last_modified' and all_paramsets is False:
            update_last_modified(fname_input)

    return (status, path)


def import_settings(import_main):
    """Verify config settings to import. Avoid equal template names.

    Parameters
    ----------
    import_main : ImportMain
        as defined in settings.py

    Returns
    -------
    any_same_name : bool
    """
    any_same_name = False

    # tag_infos
    if import_main.tag_infos != []:
        fname = 'tag_infos'
        _, _, tag_infos = load_settings(fname=fname)
        tag_infos.extend(import_main.tag_infos)

        quest = (f'Importing {len(import_main.tag_infos)} DICOM tag settings. '
                 'By default these will be added to the end of tag-list. '
                 'Sort all tags by attribute name?')
        msg_box = QMessageBox(
            QMessageBox.Question,
            'Sort DICOM tags?', quest,
            buttons=QMessageBox.Yes | QMessageBox.No,
            parent=None
            )
        res = msg_box.exec_()
        if res == QMessageBox.Yes:
            tag_infos = sorted(tag_infos, key=lambda x: x.attribute_name.upper())

        taginfos_reset_sort_index(tag_infos)
        _, _ = save_settings(tag_infos, fname=fname)

    # templates using modality dictionary
    list_dicts = [fname for fname, item in CONFIG_FNAMES.items()
                  if item['saved_as'] == 'modality_dict']
    list_dicts.append('paramsets')
    for dict_string in list_dicts:
        new_temps = getattr(import_main, dict_string, {})
        if new_temps != {}:
            _, _, temps = load_settings(fname=dict_string)
            for key in temps:
                if key in new_temps:
                    if len(new_temps[key]) > 0:
                        old_labels = [temp.label for temp in temps[key]]
                        new_labels = [temp.label for temp in new_temps[key]]
                        if old_labels == ['']:
                            temps[key] = new_temps[key]
                        else:
                            for new_id in range(len(new_temps[key])):
                                if dict_string == 'tag_patterns_special':  # replace
                                    old_id = old_labels.index(new_labels[new_id])
                                    temps[key][old_id] = new_temps[key][new_id]
                                else:
                                    if new_labels[new_id] in old_labels:
                                        new_temps[key][new_id].label = (
                                            new_labels[new_id] + '_import')
                                        any_same_name = True
                                    temps[key].append(new_temps[key][new_id])
                        if dict_string == 'paramsets':
                            if new_labels[0] != '':
                                _, _ = save_settings(
                                    temps[key], fname=f'paramsets_{key}')
            if dict_string != 'paramsets':
                _, _ = save_settings(temps, fname=dict_string)

    # auto_common
    try:
        if import_main.auto_common.import_path != '':
            _, _ = save_settings(import_main.auto_common, fname='auto_common')
    except AttributeError:
        pass

    return any_same_name


def get_taginfos_used_in_templates(object_with_templates, specific_attribute=None):
    """Find all or one specific taginfo used in different templates.

    Parameters
    ----------
    object_with_templates : object
        some object with attributes named like template fnames as in iQCconstants.py
    specific_attribute : str, optional
        if only interest of one specific attribute. The default is None.

    Returns
    -------
    status : bool
        false if some taginfos used, but not defined in tag_infos
    log : list of str
        information log to be displayed to user
    found_attributes : dict
        list of attributes found for each modality
        or list of templates where special attribute is found
    """
    status = True
    log = []

    modalities = [*QUICKTEST_OPTIONS]
    defined_attributes = {}
    for mod in modalities:
        general_tags, included_tags = get_included_tags(
            mod, object_with_templates.tag_infos)
        defined_attributes[mod] = general_tags + included_tags

    found_attributes = {mod: [] for mod in modalities}
    fnames = ['tag_patterns_special', 'tag_patterns_format',
              'tag_patterns_sort', 'rename_patterns', 'paramsets',
              'auto_templates']
    for fname in fnames:
        if hasattr(object_with_templates, fname):
            templates = getattr(object_with_templates, fname)
        else:
            _, _, templates = load_settings(fname=fname)
        for mod in templates:
            for template in templates[mod]:
                if 'patterns' in fname:
                    if specific_attribute:
                        if specific_attribute in template.list_tags:
                            found_attributes[mod].append(f'{fname}: {template.label}')
                    else:
                        found_attributes[mod].extend(template.list_tags)
                if fname == 'rename_patterns':
                    if specific_attribute:
                        if specific_attribute in template.list_tags2:
                            found_attributes[mod].append(f'{fname}: {template.label}')
                    else:
                        found_attributes[mod].extend(template.list_tags2)
                if fname == 'paramsets':
                    if specific_attribute:
                        if specific_attribute in template.dcm_tagpattern.list_tags:
                            found_attributes[mod].append(
                                f'{fname}: {template.label} (DCM test pattern)')
                        if specific_attribute in template.output.group_by:
                            found_attributes[mod].append(
                                f'{fname}: {template.label} (Output group by)')
                    else:
                        found_attributes[mod].extend(
                            template.dcm_tagpattern.list_tags)
                        found_attributes[mod].extend(template.output.group_by)
                if fname == 'auto_templates':
                    if specific_attribute:
                        if specific_attribute in template.dicom_crit_attributenames:
                            found_attributes[mod].append(
                                f'{fname}: {template.label} (Dicom criteria)')
                        if specific_attribute in template.sort_pattern.list_tags:
                            found_attributes[mod].append(
                                f'{fname}: {template.label} (Sorting of images)')
                    else:
                        found_attributes[mod].extend(template.dicom_crit_attributenames)
                        found_attributes[mod].extend(template.sort_pattern.list_tags)

        fname = 'auto_common'
        if hasattr(object_with_templates, fname):
            template = getattr(object_with_templates, fname)
        else:
            _, _, template = load_settings(fname=fname)
        if specific_attribute:
            if specific_attribute in template.auto_delete_criterion_attributenames:
                found_attributes['General'].append(
                    'Automation import settings: Auto delete criteria')
            if specific_attribute in template.filename_pattern.list_tags:
                found_attributes['General'].append(
                    'Automation import settings: Rename template')
        else:
            found_attributes['CT'].extend(template.auto_delete_criterion_attributenames)
            found_attributes['CT'].extend(template.filename_pattern.list_tags)

    if specific_attribute is None:
        for mod, found_attr in found_attributes.items():
            missing = []
            for attr in found_attr:
                if attr not in defined_attributes[mod]:
                    missing.append(attr)
            if len(missing) > 0:
                missing = list(set(missing))
                log.append(f'{mod}: missing definition of DICOM tags named {missing}')
                status = False

    return (status, log, found_attributes)


def get_ref_label_used_in_auto_templates(auto_templates, ref_attr='paramset_label'):
    """For each paramset or quicktest template find which auto_templates using this.

    Parameters
    ----------
    auto_templates : dict
        key = modalitystring, value = AutoTemplate
    ref_attr : str
        'paramset_label' or 'quicktemp_label' or 'limits_and_plot_label'

    Returns
    -------
    templates_in_auto : dict
        key = modalitystring, value = [[.label, auto_template.label],...]
    """
    templates_in_auto = {}
    for mod in auto_templates:
        qt_param = []
        for temp in auto_templates[mod]:
            qt_param.append([temp.label, getattr(temp, ref_attr)])
        templates_in_auto[mod] = qt_param

    return templates_in_auto


def get_auto_labels_output_used_in_lim(
        auto_templates, auto_vendor_templates, limits_template,
        modality=''):
    """Get automation template labels and output path where limits template used.

    Parameters
    ----------
    auto_templates : dict
    auto_vendor_templates : dict
    limits_template : LimitsAndPlotTemplate
    modality : str

    Returns
    -------
    auto_labels : list of str
    output_paths : list of str
    """
    auto_labels = []
    output_paths = []
    if limits_template.label != '':
        mod_temps = []
        try:
            if limits_template.type_vendor:
                mod_temps = auto_vendor_templates[modality]
            else:
                mod_temps = auto_templates[modality]
        except KeyError:
            pass

        auto_labels = [
            temp.label for temp in mod_temps
            if temp.limits_and_plot_label == limits_template.label
            ]
        output_paths = [
            temp.path_output for temp in mod_temps
            if temp.limits_and_plot_label == limits_template.label
            ]

    return auto_labels, output_paths


def verify_auto_templates(main):
    """Verify all linked templates in auto_templates are defined.

    Also verify that persons to notify are defined (if any).
    """
    status = True
    log = []
    auto_fnames = ['auto_templates', 'auto_vendor_templates']
    for auto_fname in auto_fnames:
        if hasattr(main, auto_fname):
            linked_fnames = ['limits_and_plot_templates']
            linked_labels = ['limits_and_plot_label']
            if 'vendor' not in auto_fname:
                linked_fnames.extend(['paramsets', 'quicktest_templates'])
                linked_labels.extend(['paramset_label', 'quicktemp_label'])
            for i, fname in enumerate(linked_fnames):
                ref_attr = linked_labels[i]
                temp_in_auto = get_ref_label_used_in_auto_templates(
                    main.auto_templates, ref_attr=ref_attr)
                if hasattr(main, fname):
                    mod_dict = getattr(main, fname)
                    for mod, templist in mod_dict.items():
                        all_labels = [t.label for t in templist]
                        missing = []
                        auto_labels, temp_labels =\
                            np.array(temp_in_auto[mod]).T.tolist()
                        if auto_labels[0] != '':
                            for label in temp_labels:
                                if label not in all_labels:
                                    missing.append(label)
                            if len(missing) > 0:
                                if '' in missing:
                                    log.append(f'{mod}: {fname} not defined for '
                                               'some templates')
                                    missing.remove('')
                                if len(missing) > 0:
                                    log.append(
                                        f'{mod}: missing definition of '
                                        f'{fname} {missing}')
                                status = False

    return (status, log)


def verify_digit_templates(main):
    """Verify all digit templates in paramsets are defined.

    Parameters
    ----------
    main : ImportMain
        as defined in settings.py

    Returns
    -------
    status : bool
    log : list of str
    """
    status = True
    log = []
    if hasattr(main, 'paramsets'):
        fname = 'digit_templates'

        if hasattr(main, fname):
            mod_dict = getattr(main, fname)
            for mod, templist in mod_dict.items():
                all_digit_labels = [t.label for t in templist]
                missing = []
                used_in = []
                digit_in_params = [
                    [temp.label, temp.num_digit_label]
                    for temp in main.paramsets[mod] if temp.num_digit_label != '']
                if len(digit_in_params) > 0:
                    param_labels, digit_labels = np.array(digit_in_params).T.tolist()
                    for label in digit_labels:
                        if label not in all_digit_labels:
                            missing.append(label)
                            idxs = get_all_matches(digit_labels, label)
                            par_list = [param_labels[idx] for idx in idxs]
                            used_in.extend(par_list)
                    if len(missing) > 0:
                        if len(missing) > 0:
                            log.append(
                                f'{mod}: missing definition of {fname} {missing} '
                                f'used in paramset(s) {used_in}')
                        status = False

    return (status, log)


def get_test_alternative(paramset, testcode):
    """Get current alternative for settings in paramset for a given testcode.

    Parameters
    ----------
    paramset : object
        ParamSet<mod> as defined in config_classes.py
    testcode : str
        three letter string representing the test

    Returns
    -------
    alt : int or None
        alternative number specified for testcode
    """
    alt = None
    testcode = testcode.lower()
    if testcode in ['sli', 'mtf', 'rec', 'snr']:
        alt = getattr(paramset, f'{testcode}_type', None)
    elif testcode == 'roi':
        alt = getattr(paramset, 'roi_use_table', None)
    elif testcode == 'hom':
        alt = getattr(paramset, 'hom_tab_alt', None)
    elif testcode == 'sni':
        alt = paramset.sni_alt

    return alt


def verify_output_alternative(paramset, testcode=None):
    """Verify that the alternative correspond for output and params of a paramset.

    Parameters
    ----------
    paramset : ParamSetXX
        as defined in config_classes
    testcode : str, optional
        Test for a spesific testcode or all if None. The default is None.

    Returns
    -------
    status : bool
        False if issues found
    log : list of str
        Warnings to display
    """
    log = []
    for key, sublist in paramset.output.tests.items():
        alt = None
        proceed = True
        if testcode:
            if key.lower() != testcode.lower():
                proceed = False
        if proceed:
            alt = get_test_alternative(paramset, key)
            if alt is not None:
                for subno, sub in enumerate(sublist):
                    sub_alt = sub.alternative
                    if sub.alternative > 9:
                        sub_alt = sub.alternative - 10
                    if sub_alt != alt:
                        log.append(f'{key}: {sub}')
                        mod = get_modality_of_paramset(paramset)
                        try:
                            alt_out = ALTERNATIVES[mod][key][sub_alt]
                            alt_param = ALTERNATIVES[mod][key][alt]
                            log.append(
                                'Parameters used indicate alternative '
                                f'{alt}: {alt_param}')
                            log.append(
                                'Output settings indicate alternative '
                                f'{sub_alt}: {alt_out}')
                        except (KeyError, IndexError):
                            log.append(
                                f'Parameters used indicate alternative {alt}')
                            log.append(
                                f'Output settings indicate alternative {sub_alt}')

    status = False if len(log) > 0 else True

    return (status, log)


def verify_output_templates(paramset=None, main=None):
    """Verify output template alternatives correspond to set alternatives.

    Parameters
    ----------
    paramset : ParamSetXX, optional
        if verifying one paramset only
    main : ImportMain, optional
        if verifying all paramsets. The default is None.

    Returns
    -------
    status : bool
        False if anything to warn about.
    log : list of str
        Warnings to display
    """
    status = True
    log = []
    if main is None:
        if paramset is not None:
            status, log = verify_output_alternative(paramset)
    else:
        if hasattr(main, 'paramsets'):
            for mod, paramset_list in main.paramsets.items():
                for this_paramset in paramset_list:
                    status_this, log_this = verify_output_alternative(this_paramset)
                    if status_this is False:
                        log.append(f'Paramset {this_paramset.label}:')
                        log.extend(log_this)
                        status = False

    return (status, log)


def verify_limits_used_with_auto(main):
    """Validate headers of LimitsAndPlot templates to linked automation templates."""
    status = True
    log = []
    templates = main.limits_and_plot_templates
    mismatch_dict = {}
    for mod in [*templates]:
        mismatch_templates = []
        labels = [temp.label for temp in templates[mod]]
        if labels[0] != '':  # none defined
            for label in labels:
                idx = labels.index(label)
                this_template = templates[mod][idx]
                auto_labels, output_paths = get_auto_labels_output_used_in_lim(
                        main.auto_templates, main.auto_vendor_templates,
                        this_template, modality=mod)
                for auto_no, path in enumerate(output_paths):
                    headers_in_path = []
                    if os.path.exists(path):
                        with open(path) as file:
                            headers_in_path =\
                                file.readline().strip('\n').split('\t')
                    if len(headers_in_path) > 0:
                        headers_in_path.pop(0)  # remove date column
                    headers_in_limits = [
                        elem for sublist in this_template.groups
                        for elem in sublist]
                    if set(headers_in_path) != set(headers_in_limits):
                        if label not in mismatch_templates:
                            mismatch_templates.append(label)
        if len(mismatch_templates) > 0:
            mismatch_dict[mod] = mismatch_templates

    if mismatch_dict:
        status = False
        log = ['Found mismatches between headers in Limits & Plot templates '
               'when compared to output path header of automation templates '
               'where the Limits & plot template is used/linked.']
        log.extend([f'\t{mod}: {labels}' for mod, labels in mismatch_dict])

    return (status, log)


def taginfos_reset_sort_index(tag_infos):
    """Reset sort_index of list of TagInfo according to order of elements.

    Parameters
    ----------
    tag_infos : list of TagInfos
    """
    for i, tag_info in enumerate(tag_infos):
        tag_info.sort_index = i


def tag_infos_difference(tag_infos_import, tag_infos):
    """Compare imported tag_infos to current tag_infos and return tags to add.

    Parameters
    ----------
    tag_infos_import : list of TagInfo
        tag infos to be imported
    tag_infos : list of TagInfo
        current tag infos

    Returns
    -------
    tag_infos_new : list of TagInfo
        tags infos to import
    """
    old_attr = [tag.attribute_name for tag in tag_infos]
    import_attr = [tag.attribute_name for tag in tag_infos_import]
    tag_infos_new = []
    for attr in list(set(import_attr)):
        if attr not in old_attr:
            tag_infos_new.extend(
                [tag for tag in tag_infos_import if tag.attribute_name == attr])
        else:
            # find all old with this tag.attribute_name and compare tag and sequence
            old_tags_seqs = [
                (tag.tag, tag.sequence) for tag in tag_infos
                if tag.attribute_name == attr]
            old_tags = [tag_seq[0] for tag_seq in old_tags_seqs]
            old_seqs = [tag_seq[1] for tag_seq in old_tags_seqs]
            import_idxs = get_all_matches(import_attr, attr)
            for idx in import_idxs:
                diff_tag = False if tag_infos_import[idx].tag in old_tags else True
                diff_seq = False if tag_infos_import[idx].sequence in old_seqs else True
                if any([diff_tag, diff_seq]):
                    tag_infos_new.append(tag_infos_import[idx])
    return tag_infos_new


def tag_infos_difference_default(current_tag_infos):
    """Compare current tag_infos to default and return tags to adjusted current.

    New protections might have been added or new default tags might have been added.

    Parameters
    ----------
    current_tag_infos : list of TagInfo
        current tag infos

    Returns
    -------
    changes : bool
        True if changes available
    added_tags : list of TagInfo
        tags added
    adjusted_tags : list of TagInfo
        tags with changed protection
    updated_tag_infos : list of TagInfo
        updated tags infos
    """
    changes = False
    added_tags = []
    adjusted_tags = []
    current_attr = [tag.attribute_name for tag in current_tag_infos]
    default_tag_infos = CONFIG_FNAMES['tag_infos']['default']
    tag_attr = [*asdict(default_tag_infos[0])]
    unit_idx = tag_attr.index('unit')
    input_tag_infos = copy.deepcopy(current_tag_infos)

    # add Mammo to all tags with Xray if none with Mammo (new modality v3.0.6)
    mammo_changes = False
    idx_with_mammo = [tag.sort_index for tag in input_tag_infos
                      if 'Mammo' in tag.limited2mod]
    if len(idx_with_mammo) == 0:
        for tag in input_tag_infos:
            if 'Xray' in tag.limited2mod:
                tag.limited2mod.append('Mammo')
        changes = True
        mammo_changes = True

    updated_tag_infos = copy.deepcopy(input_tag_infos)
    default_attr = [tag.attribute_name for tag in default_tag_infos]
    for attr in list(set(default_attr)):
        if attr not in current_attr:
            new_tags = [tag for tag in default_tag_infos if tag.attribute_name == attr]
            updated_tag_infos.extend(new_tags)
            changes = True
            added_tags.extend(new_tags)
        else:
            # find all tags with attr as name and compare tag except protection
            curr_taginfos = [
                tag for tag in input_tag_infos if tag.attribute_name == attr]
            curr_vals = []
            for tag in curr_taginfos:
                this_vals = [val for key, val in asdict(tag).items()][1:]
                this_vals.pop(unit_idx-1)  # ignore unit
                curr_vals.append(this_vals)

            default_idxs = get_all_matches(default_attr, attr)
            for idx in default_idxs:
                def_vals = [
                    val for key, val in asdict(default_tag_infos[idx]).items()][1:]
                def_vals.pop(unit_idx-1)

                if def_vals not in curr_vals:
                    curr_wo_prot = [vals[:-1] for vals in curr_vals]
                    try:  # replace if only protection or unit differ
                        idx_curr_this = curr_wo_prot.index(def_vals[:-1])
                        curr_idxs = [
                            i for i, tag in enumerate(input_tag_infos)
                            if tag.attribute_name == attr]
                        curr_idx = curr_idxs[idx_curr_this]
                        updated_tag_infos[curr_idx].protected = def_vals[-1]
                        adjusted_tags.append(updated_tag_infos[curr_idx])
                    except ValueError:
                        updated_tag_infos.append(default_tag_infos[idx])
                        added_tags.append(default_tag_infos[idx])
                    changes = True

    return (changes, added_tags, adjusted_tags, mammo_changes, updated_tag_infos)


def attribute_names_used_in(old_new_names=[], name='', limited2mod=['']):
    """Change attribute_name in yaml files when tag_infos.yaml change name.

    Parameters
    ----------
    old_new_names : list of str, optional
        [[oldname1,newname1],[oldname2,newname2]...]. The default is [].
        if not [] == also save updated
    name: str, optional
        one attribute name to test

    Returns
    -------
    proceed : bool
        True if continue to save
    log : list of str
        progress-log
    """
    status = True
    oks = []
    log = []
    if old_new_names == []:
        if name != '':
            old_new_names = [[name, name]]

    if len(old_new_names) > 0:
        fnames = ['tag_patterns_special', 'tag_patterns_format',
                  'tag_patterns_sort', 'rename_patterns', 'paramsets',
                  'auto_templates']
        for fname in fnames:
            _, _, templates = load_settings(fname=fname)
            found_in = []
            for mod in templates:
                proceed = True
                if limited2mod != ['']:
                    if mod not in limited2mod:
                        proceed = False
                if proceed:
                    for template in templates[mod]:
                        found = []
                        for oldnew in old_new_names:
                            old_name, new_name = oldnew
                            if 'patterns' in fname:
                                if old_name in template.list_tags:
                                    for i in range(len(template.list_tags)):
                                        if template.list_tags[i] == old_name:
                                            template.list_tags[i] = new_name
                                            found = [mod, template.label]
                            if fname == 'rename_patterns':
                                if old_name in template.list_tags2:
                                    for i in range(len(template.list_tags2)):
                                        if template.list_tags2[i] == old_name:
                                            template.list_tags2[i] = new_name
                                            found = [mod, template.label]
                            if fname == 'paramsets':
                                if old_name in template.dcm_tagpattern.list_tags:
                                    list_tags = template.dcm_tagpattern.list_tags
                                    for i in range(len(list_tags)):
                                        if list_tags[i] == old_name:
                                            template.dcm_tagpattern.list_tags[
                                                i] = new_name
                                            if template.label != '':
                                                found = [mod, template.label]
                                            else:
                                                found = [mod, '(default)']
                                if old_name in template.output.group_by:
                                    list_group_by = template.output.group_by
                                    for i in range(len(list_group_by)):
                                        if list_group_by[i] == old_name:
                                            template.output.group_by[
                                                i] = new_name
                                            if template.label != '':
                                                found = [mod, template.label]
                                            else:
                                                found = [mod, '(default)']
                            if fname == 'auto_templates':
                                if old_name in template.dicom_crit_attributenames:
                                    attr_list = template.dicom_crit_attributenames
                                    i = attr_list.index(old_name)
                                    template.dicom_crit_attributenames[i] = new_name
                                    found = [mod, template.label]
                                if old_name in template.sort_pattern.list_tags:
                                    for i in range(len(
                                            template.sort_pattern.list_tags)):
                                        if template.sort_pattern.list_tags[
                                                i] == old_name:
                                            template.sort_pattern.list_tags[
                                                i] = new_name
                                            found = [mod, template.label]
                        if len(found) > 0:
                            found_in.append(found)

            if len(found_in) > 0:
                param_auto = {}
                if fname == 'paramsets':
                    # list all automation templates where paramset used
                    _, _, auto_templates = load_settings(
                        fname='auto_templates')
                    param_auto = get_ref_label_used_in_auto_templates(
                        auto_templates, ref_attr='paramset_label')

                if name == '':
                    log.append(f'{fname} with changed attribute_names:')
                else:
                    log.append(f'{fname} using attribute_name {name}:')
                for i in range(len(found_in)):
                    # modality: template.label
                    log.append(f'\t{found_in[i][0]}: {found_in[i][1]}')
                    if fname == 'paramsets':
                        used_in = []
                        for pa in param_auto[found_in[i][0]]:
                            if pa[0] == found_in[i][1]:
                                used_in.append(pa[1])
                        if len(used_in) > 0:
                            log.append(f'\t\t used in AutoTemplate: {used_in}')

                if name == '':  # save changes
                    if fname == 'paramsets':
                        changed_mod = [f[0] for f in found_in]
                        changed_mod = list(set(changed_mod))
                        ok_all = []
                        for mod in changed_mod:
                            ok_this, _ = save_settings(
                                templates[mod], fname=f'{fname}_{mod}')
                            ok_all.append(ok_this)
                        ok_this = all(ok_all)
                    else:
                        ok_this, _ = save_settings(templates, fname=fname)
                    oks.append(ok_this)
                    if ok_this:
                        log.append('Saved')
                    else:
                        log.append('Saving failed')
                else:
                    oks.append(False)
                log.append('---')
            else:
                if name == '':
                    log.append(f'{fname} - no change needed')
                else:
                    log.append(f'{fname} - attribute_name {name} not found')

        fname = 'auto_common'
        _, _, template = load_settings(fname=fname)
        found = False
        for oldnew in old_new_names:
            old_name, new_name = oldnew
            if old_name in template.auto_delete_criterion_attributenames:
                for i in range(len(template.auto_delete_criterion_attributenames)):
                    if template.auto_delete_criterion_attributenames[i] == old_name:
                        template.auto_delete_criterion_attributenames[i] = new_name
                        found = True
            if old_name in template.filename_pattern.list_tags:
                for i in range(len(template.filename_pattern.list_tags)):
                    if template.filename_pattern.list_tags[i] == old_name:
                        template.filename_pattern.list_tags[i] = new_name
                        found = True
        if found:
            if name == '':
                log.append(f'{fname} changed attribute_names')
                ok_this, _ = save_settings(template, fname=fname)
                oks.append(ok_this)
                if ok_this:
                    log.append('Saved')
                else:
                    log.append('Saving failed')
            else:
                log.append(f'{fname} - attribute_name {name} used')
                oks.append(False)
        else:
            if name == '':
                log.append(f'{fname} - no change needed')
            else:
                log.append(f'{fname} - attribute_name {name} not found')

    if len(oks) > 0:
        if False in oks:
            status = False

    return (status, log)


def verify_tag_infos(main):
    """Find all tags used and if all these are defined in main.tag_infos."""
    status = True
    log = []

    modalities = [*QUICKTEST_OPTIONS]
    defined_attributes = {}
    for mod in modalities:
        general_tags, included_tags = get_included_tags(mod, main.tag_infos)
        defined_attributes[mod] = general_tags + included_tags

    found_attributes = {mod: [] for mod in modalities}
    fnames = ['tag_patterns_special', 'tag_patterns_format',
              'tag_patterns_sort', 'rename_patterns', 'paramsets',
              'auto_templates']
    for fname in fnames:
        if hasattr(main, fname):
            templates = getattr(main, fname)
        else:
            _, _, templates = load_settings(fname=fname)
        for mod in templates:
            for template in templates[mod]:
                if 'patterns' in fname:
                    found_attributes[mod].extend(template.list_tags)
                if fname == 'rename_patterns':
                    found_attributes[mod].extend(template.list_tags2)
                if fname == 'paramsets':
                    found_attributes[mod].extend(
                        template.dcm_tagpattern.list_tags)
                    found_attributes[mod].extend(template.output.group_by)
                if fname == 'auto_templates':
                    found_attributes[mod].extend(template.dicom_crit_attributenames)
                    found_attributes[mod].extend(template.sort_pattern.list_tags)

        fname = 'auto_common'
        if hasattr(main, fname):
            template = getattr(main, fname)
        else:
            _, _, template = load_settings(fname=fname)
        found_attributes['CT'].extend(template.auto_delete_criterion_attributenames)
        found_attributes['CT'].extend(template.filename_pattern.list_tags)

    for mod, found_attr in found_attributes.items():
        missing = []
        for attr in found_attr:
            if attr not in defined_attributes[mod]:
                missing.append(attr)
        if len(missing) > 0:
            log.append(f'{mod}: missing definition of DICOM tags named {missing}')
            status = False

    return (status, log)


def get_icon_path(user_pref_dark_mode):
    """Get path for icons depending on darkmode settings."""
    path_icons = ':/icons/'
    if user_pref_dark_mode:
        path_icons = ':/icons_darkmode/'

    return path_icons
