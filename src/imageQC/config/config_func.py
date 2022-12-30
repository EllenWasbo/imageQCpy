#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions used for configuration settings.

@author: Ellen Wasbo
"""
import os
from pathlib import Path
from time import time, ctime
import yaml
from PyQt5.QtCore import QFile, QIODevice, QTextStream
from dataclasses import asdict

from PyQt5.QtWidgets import QMessageBox, QFileDialog

# imageQC block start
from imageQC.config.iQCconstants import (
    USERNAME, APPDATA, TEMPDIR, ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER,
    CONFIG_FNAMES, USER_PREFS_FNAME
    )
import imageQC.config.config_classes as cfc
import imageQC.ui.ui_dialogs as uid
# imageQC block end


def test_config_folder(widget):
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
        msgBox = QMessageBox(
            QMessageBox.Question,
            'Proceed?', quest,
            buttons=QMessageBox.Yes | QMessageBox.No,
            parent=widget
            )
        res = msgBox.exec_()
        if res == QMessageBox.Yes:
            proceed = True
        if proceed:
            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.Directory)
            if dlg.exec():
                fname = dlg.selectedFiles()
                os.environ[ENV_CONFIG_FOLDER] = fname[0]
                user_status, user_path, user_prefs = load_user_prefs()
                if user_path != '':
                    user_prefs.config_folder = os.environ[ENV_CONFIG_FOLDER]
                    ok, path = save_user_prefs(user_prefs, parentwidget=widget)

    return proceed


def get_active_users():
    """Get list of active usernames sharing the config folder."""
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

    Returns
    -------
    bool
        False if failed initiating user preferences and config folder
    str
        path (or '' if failed)
    str
        error message
    """
    status = False
    errmsg = ''
    p = ''

    # initiate APPDATA or TEMPDIR imageQC if missing
    if os.path.exists(path) is False:
        if os.access(Path(path).parent, os.W_OK):
            os.mkdir(path)
        else:
            errmsg = '\n'.join(['Missing writing permission:',
                                Path(path).parent])

    if errmsg == '':
        userpref = cfc.UserPreferences()
        userpref.config_folder = config_folder

        p = os.path.join(path, USER_PREFS_FNAME)

        if os.access(path, os.W_OK):
            with open(p, 'w') as file:
                yaml.safe_dump(
                    asdict(userpref), file,
                    default_flow_style=None, sort_keys=False)
            status = True
        else:
            errmsg = '\n'.join(['Missing writing permission:', p])
            p = ''

    if errmsg != '':
        errmsg = '\n'.join([errmsg,
                            'Saving settings is not possible.'])

    return (status, p, errmsg)


def verify_input_dict(dict_input, default_object):
    """Verify input from yaml if config classes change on newer versions.

    Remove old keywords from input.
    """
    default_dict = asdict(default_object)
    actual_keys = [*default_dict]
    updated_dict = {k: v for k, v in dict_input.items() if k in actual_keys}
    return updated_dict


def save_user_prefs(userpref, parentwidget=None):
    """Save user preferences to user_preferences.yaml file.

    Parameters
    ----------
    userpref : object of class UserPreferences

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
            quest = ('Save user_preferences.yaml in:')
            res = QuestionBox(
                parentwidget, title='Save as', msg=quest,
                yes_text=f'{APPDATA}', no_text=f'{TEMPDIR}')
            p = APPDATA if res.exec() == 0 else TEMPDIR
            stat, path, user_prefs = init_user_prefs(
                    path=p, config_folder=os.environ[ENV_CONFIG_FOLDER])
        else:
            path = ''

    if path != '':
        if os.access(Path(path).parent, os.W_OK):
            with open(path, 'w') as file:
                yaml.safe_dump(
                    asdict(userpref), file,
                    default_flow_style=None, sort_keys=False)
            status = True

    return (status, path)


def load_user_prefs():
    """Load yaml file.

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
        with open(path, 'r') as f:
            doc = yaml.safe_load(f)
            updated_doc = verify_input_dict(doc, cfc.UserPreferences())
            userprefs = cfc.UserPreferences(**updated_doc)

    if userprefs is None:
        status = False
        path = ''
        userprefs = cfc.UserPreferences()

    return (status, path, userprefs)


def get_config_folder():
    """Get config folder.

    Returns
    -------
    str
        Config folder if exists else empty string.
    """
    #config_folder = ''
    #user_prefs = UserPreferences()
    #p = os.path.join(APPDATA, USER_PREFS_FNAME)
    try:
        p = os.environ[ENV_CONFIG_FOLDER]
        config_folder = p if os.path.exists(p) else ''
    except KeyError:
        config_folder = ''
    '''
    if os.path.exists(p)
        with open(p, 'r') as f:
            doc = yaml.safe_load(f)
            user_prefs = UserPreferences(**doc)

        if os.path.exists(user_prefs.config_folder):
            config_folder = user_prefs.config_folder
    '''
    return config_folder


def get_config_filename(fname, force=False):
    """Verify if yaml file exists.

    Parameters
    ----------
    fname : str
        filename as defined in CONFIG_FNAMES
    force : bool
        force return filename even though it does not exist

    Returns
    -------
    str
        full path to yaml file if it exist, empty if not verified
    """
    path = ''
    """
    user_prefs_ok, user_path, userprefs = load_user_prefs()
    if user_prefs_ok:
        path_temp = os.path.join(
            userprefs.config_folder, fname + '.yaml')
    """
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
    """Load default TagPatterns format for exporting DCM data (test DCM).

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
                ct_number_tables[key] = cfc.HUnumberTable(**ct_tab)

    return ct_number_tables


def load_settings(fname=''):
    """Load settings from yaml file in config folder.

    Parameters
    ----------
    fname : str
        yaml filename without folder and extension

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
        path = get_config_filename(fname)
        if path != '':
            if CONFIG_FNAMES[fname]['saved_as'] == 'object_list':
                #try:
                with open(path, 'r') as file:
                    docs = yaml.safe_load_all(file)
                    settings = []
                    for doc in docs:
                        if fname == 'tag_infos':
                            settings.append(cfc.TagInfo(**doc))
                    if fname == 'tag_infos':
                        taginfos_reset_sort_index(settings)
                status = True
                #except:
                #    return_default = True
            elif CONFIG_FNAMES[fname]['saved_as'] == 'modality_dict':
                try:
                    with open(path, 'r') as file:
                        docs = yaml.safe_load(file)
                        settings = {}
                        for mod, doc in docs.items():
                            settings[mod] = []
                            for temp in doc:
                                if fname == 'paramsets':
                                    temp['dcm_tagpattern'] = (
                                        cfc.TagPatternFormat(
                                            **temp['dcm_tagpattern']))
                                    tests = {}
                                    for key, test in temp[
                                            'output']['tests'].items():
                                        tests[key] = []
                                        for sub in test:
                                            tests[key].append(
                                                cfc.QuickTestOutputSub(**sub))
                                    temp['output'] = (
                                        cfc.QuickTestOutputTemplate(
                                            include_header=temp[
                                                'output']['include_header'],
                                            transpose_table=temp[
                                                'output']['transpose_table'],
                                            decimal_mark=temp[
                                                'output']['decimal_mark'],
                                            include_filename=temp[
                                                'output']['include_filename'],
                                            tests=tests))
                                    if mod == 'CT':
                                        temp['ctn_table'] = (
                                            cfc.HUnumberTable(
                                                **temp['ctn_table']))
                                        settings[mod].append(
                                            cfc.ParamSetCT(**temp))
                                    elif mod == 'Xray':
                                        settings[mod].append(
                                            cfc.ParamSetXray(**temp))
                                    elif mod == 'NM':
                                        settings[mod].append(
                                            cfc.ParamSetNM(**temp))
                                    elif mod == 'SPECT':
                                        settings[mod].append(
                                            cfc.ParamSetSPECT(**temp))
                                    elif mod == 'PET':
                                        settings[mod].append(
                                            cfc.ParamSetPET(**temp))
                                    elif mod == 'MR':
                                        settings[mod].append(
                                            cfc.ParamSetMR(**temp))
                                elif fname == 'quicktest_templates':
                                    settings[mod].append(
                                        cfc.QuickTestTemplate(**temp))
                                elif fname == 'auto_templates':
                                    temp['sort_pattern'] = (
                                        cfc.TagPatternSort(
                                            **temp['sort_pattern']))
                                    settings[mod].append(
                                        cfc.AutoTemplate(**temp))
                                elif fname == 'auto_vendor_templates':
                                    settings[mod].append(
                                        cfc.AutoVendorTemplate(**temp))
                                elif fname == 'rename_patterns':
                                    settings[mod].append(
                                        cfc.RenamePattern(**temp))
                                elif fname == 'tag_patterns_sort':
                                    settings[mod].append(
                                        cfc.TagPatternSort(**temp))
                                elif fname == 'tag_patterns_format':
                                    settings[mod].append(
                                        cfc.TagPatternFormat(**temp))
                                elif fname == 'tag_patterns_special':
                                    settings[mod].append(
                                        cfc.TagPatternFormat(**temp))
                    status = True
                except:
                    return_default = True
            else:  # settings as one object
                #try:
                with open(path, 'r') as file:
                    doc = yaml.safe_load(file)
                    if fname == 'auto_common':
                        doc['filename_pattern'] = (
                            cfc.TagPatternFormat(
                                **doc['filename_pattern']))
                        settings = cfc.AutoCommon(**doc)
                    elif fname == 'last_modified':
                        settings = cfc.LastModified(**doc)
                status = True
                #except: what to except.... wait for error
                #    return_default = True
        else:
            return_default = True

        if return_default:
            settings = CONFIG_FNAMES[fname]['default']
            if fname == 'paramsets':
                default_tags_dcm = load_default_dcm_test_tag_patterns()
                for mod in default_tags_dcm:
                    settings[mod][0].dcm_tagpattern = default_tags_dcm[mod]
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
            ok, path, last_mod = load_settings(fname='last_modified')
            user, modtime = getattr(last_mod, fname)

            if user != USERNAME:
                if modtime > lastload:
                    errmsg = f'It seems that {user} is also editing this config file.'
                else:
                    errmsg = 'It seems that this file has been edited recently.'

                errmsg = (
                    errmsg +
                    '\nProceed saving and possibly overwrite changes done by others?')
                status = False

    return status, errmsg


def update_last_modified(fname=''):
    """Update last_modified.yaml."""
    # path = get_config_filename(fname, force=True)
    ok, path_last_mod, last_mod = load_settings(fname='last_modified')
    setattr(last_mod, fname, [USERNAME, time()])
    ok, path_last_mod = save_settings(last_mod, fname='last_modified')


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

    if fname != '':
        path = get_config_filename(fname, force=True)
        proceed = False
        if os.access(Path(path).parent, os.W_OK):
            proceed = True
        if proceed:
            breakpoint()  # sometimes fails by empty file - check when (p settings)
            if CONFIG_FNAMES[fname]['saved_as'] == 'object_list':
                try:
                    with open(path, 'w') as file:
                        yaml.safe_dump_all(
                            [asdict(temp) for temp in settings],
                            file, default_flow_style=None, sort_keys=False)
                    status = True
                except:
                    pass
            elif CONFIG_FNAMES[fname]['saved_as'] == 'modality_dict':
                try:
                    temp_dict = {}
                    for key, val in settings.items():
                        temp_dict[key] = [asdict(temp) for temp in val]
                    with open(path, 'w') as file:
                        yaml.safe_dump(
                            temp_dict,
                            file, default_flow_style=None, sort_keys=False)
                    status = True
                except:
                    pass
            else:
                try:
                    with open(path, 'w') as file:
                        yaml.safe_dump(
                            asdict(settings),
                            file, default_flow_style=None, sort_keys=False)
                    status = True
                except:
                    pass

        if fname != 'last_modified':
            update_last_modified(fname)

    return (status, path)


def import_settings(
        parent_widget=None,
        new_paramsets=None, new_quicktests=None,
        new_auto_common=None, new_auto_templates=None, new_auto_templates_vendor=None,
        new_tag_infos=None, new_tag_patterns_format=None
    ):
    """Import config settings."""
    if new_paramsets is not None:
        ok, path, paramsets = load_settings(fname='paramsets')
        if path != '':
            # check name conflict
            pass
    #TODO lots of


def get_paramsets_used_in_auto_templates(auto_templates):
    """For each paramset find which auto_templates using this.

    Parameters
    ----------
    auto_templates : dict
        key = modalitystring
        value = AutoTemplate

    Returns
    -------
    paramsets_in_auto : dict
        key = modalitystring
        value = [paramset.label, [auto_template.label, ...]]

    """
    paramsets_in_auto = {}
    for mod in auto_templates:
        auto_param = []
        for temp in auto_templates[mod]:
            auto_param.append([temp.label, temp.paramset_label])
        paramsets_in_auto[mod] = [auto_param]

    return paramsets_in_auto


def get_quicktest_used_in_auto_templates(auto_templates):
    """For each paramset find which auto_templates using this.

    Parameters
    ----------
    auto_templates : dict
        key = modalitystring
        value = AutoTemplate

    Returns
    -------
    quicktest_templates_in_auto : dict
        key = modalitystring
        value = [[quicktest.label, auto_template.label],...]

    """
    quicktest_templates_in_auto = {}
    for mod in auto_templates:
        qt_param = []
        for temp in auto_templates[mod]:
            qt_param.append([temp.label, temp.quicktemp_label])
        quicktest_templates_in_auto[mod] = [qt_param]

    return quicktest_templates_in_auto


def taginfos_reset_sort_index(tag_infos):
    """Reset sort_index of list of TagInfo according to order of elements.

    Parameters
    ----------
    tag_infos : list of TagInfos
    """
    for i, tag_info in enumerate(tag_infos):
        tag_info.sort_index = i


def correct_attribute_names(old_new_names=[]):
    """Change attribute_name in yaml files when tag_infos.yaml change name.

    Parameters
    ----------
    old_new_names : list of str, optional
        [[oldname1,newname1],[oldname2,newname2]...]. The default is [].

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
    if len(old_new_names) > 0:

        fnames = ['tag_patterns_special', 'tag_patterns_format',
                  'tag_patterns_sort', 'rename_patterns', 'paramsets',
                  'auto_templates']
        for fname in fnames:
            ok, path, templates = load_settings(fname=fname)
            found_in = []
            for mod in templates:
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
                            # TODO output.group_by?
                        if fname == 'auto_templates':
                            if old_name in template.dicom_crit_attributenames:
                                attr_list = template.dicom_crit_attributenames
                                i = attr_list.index(old_name)
                                template.dicom_crit_attributenames[i] = new_name
                                found = [mod, template.label]
                            if old_name in template.sort_pattern.list_tags:
                                for i in range(len(template.sort_pattern.list_tags)):
                                    if template.sort_pattern.list_tags[i] == old_name:
                                        template.sort_pattern.list_tags[i] = new_name
                                        found = [mod, template.label]
                    if len(found) > 0:
                        found_in.append(found)
            if len(found_in) > 0:
                param_auto = {}
                if fname == 'paramsets':
                    # list all automation templates where paramset used
                    ok, path, auto_templates = load_settings(
                        fname='auto_templates')
                    param_auto = get_paramsets_used_in_auto_templates(
                        auto_templates)

                log.append(f'{fname} with changed attribute_names:')
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

                ok, path = save_settings(templates, fname=fname)
                oks.append(ok)
                if ok:
                    log.append('Saved')
                else:
                    log.append('Saving failed')
                log.append('---')
            else:
                log.append(f'{fname} - no change needed')

        fname = 'auto_common'
        ok, path, template = load_settings(fname=fname)
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
            log.append(f'{fname} changed attribute_names')
            ok, path = save_settings(template, fname=fname)
            oks.append(ok)
            if ok:
                log.append('Saved')
            else:
                log.append('Saving failed')
        else:
            log.append(f'{fname} - no change needed')

    if len(oks) > 0:
        if False in oks:
            status = False

    return (status, log)


def reset_auto_template(auto_template=None, parent_widget=None):
    """Move all files in path_input/Archive to path_input.

    Parameters
    ----------
    auto_template : AutoTemplate
    """
    archive_path = Path(auto_template.path_input) / "Archive"
    if archive_path.exists():
        move_files = []
        move_dirs = []
        dirs = [x for x in archive_path.glob('*') if x.is_dir()]
        if len(dirs) > 0:
            dlg = uid.ResetAutoTemplateDialog(directories=dirs)
            res = dlg.exec()
            if res:
                idxs = dlg.get_idxs()
                move_dirs = [dirs[idx] for idx in idxs]
                if len(move_dirs) > 0:
                    for folder in move_dirs:
                        files = [x for x in folder.glob('*') if x.is_file()]
                        move_files.extend(files)
        else:
            files = [x for x in archive_path.glob('*') if x.is_file()]
            if len(files) > 0:
                dlg = uid.ResetAutoTemplateDialog(files=files)
                move_files = [files[idx] for idx in idxs]
            else:
                QMessageBox.information(
                    parent_widget, 'Found no files in Archive',
                    f'Found no files in Archive for template {auto_template.label}')

        if len(move_files) > 0:
            for file in move_files:
                file.rename(Path(auto_template.path_input) / file.name)
            if len(move_dirs) > 0:
                for folder in move_dirs:
                    if not any(folder.iterdir()):
                        folder.rmdir()
            QMessageBox.information(
                parent_widget, 'Finished moving files',
                f'{len(move_files)} files moved out of Archive.')
    else:
        QMessageBox.information(
            parent_widget, 'Found no Archive',
            f'Found no Archive to reset for the template {auto_template.label}')


def get_icon_path(user_pref_dark_mode):
    """Get path for icons depending on darkmode settings."""
    path_icons = ':/icons/'
    if user_pref_dark_mode:
        path_icons = ':/icons_darkmode/'

    return path_icons


class QuestionBox(QMessageBox):
    """QMessageBox with changed yes no text as options."""

    def __init__(
            self, parent=None, title='?', msg='?',
            yes_text='Yes', no_text='No', msg_width=500):
        """Initiate QuestionBox.

        Parameters
        ----------
        parent : widget, optional
            The default is None.
        title : str, optional
            The default is '?'.
        msg : str, optional
            Question text. The default is '?'.
        yes_text : TYPE, optional
            Text on yes button. The default is 'Yes'.
        no_text : TYPE, optional
            Text on no button. The default is 'No'.
        msg_width : int, optional
            Width of question label. The default is 500.

        Returns
        -------
        None.

        """
        super().__init__(
            QMessageBox.Question, title, msg, parent=parent)
        self.setIcon(QMessageBox.Question)
        self.setWindowTitle(title)
        self.setText(msg)
        self.addButton(no_text, QMessageBox.RejectRole)
        self.addButton(yes_text, QMessageBox.AcceptRole)
        self.setStyleSheet(
            f"""
            QPushButton {{
                padding: 5px;
                }}
            QLabel {{
                width: {msg_width}px;
                }}
            """)
