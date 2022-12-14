#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run automatic tests based on settings.

@author: Ellen Wasbø
"""
from __future__ import annotations

import os
from pathlib import Path
from fnmatch import fnmatch
import logging

import pydicom

# imageQC block start
import imageQC.config.config_func as cff
from imageQC.config.iQCconstants import QUICKTEST_OPTIONS
from imageQC.scripts.input_main_no_gui import InputMain
from imageQC.scripts.calculate_qc import calculate_qc, quicktest_output
import imageQC.scripts.read_vendor_QC_reports as read_vendor_QC_reports
import imageQC.scripts.dcm as dcm
from imageQC.scripts.mini_methods import get_all_matches
from imageQC.scripts.mini_methods_format import valid_path
import imageQC.config.config_classes as cfc
from imageQC.ui.reusables import proceed_question
# imageQC block end

pydicom.config.future_behavior(True)
logger = logging.getLogger(__name__)


def run_automation_non_gui(sysargv):
    """Run automation without GUI based on sys.argv."""
    def append_log(log_this, not_written=[]):
        if len(log_this) > 0:
            for msg in log_this:
                logger.info(msg)
        if len(not_written) > 0:
            logger.warning('Results not written to file:')
            for row in not_written:
                logger.info(row)

    logger.info('-----------------Automation started from command-----------------')
    ok, args, ndays, mods, temps = validate_args(sysargv)

    if ok:
        # load from yaml
        auto_common = None
        auto_templates = None
        auto_vendor_templates = None
        run_auto = False
        run_auto_vendor = False
        if '-i' in args or '-d' in args or '-a' in args:
            ok_common, path, auto_common = cff.load_settings(
                fname='auto_common')
            ok_tags, path, tag_infos = cff.load_settings(
                fname='tag_infos')
            ok_temps, path, auto_templates = cff.load_settings(
                fname='auto_templates')
            if '-d' in args or '-a' in args:
                if ok_temps:
                    run_auto = True
        if '-v' in args or '-a' in args:
            ok_temps_v, path, auto_vendor_templates = cff.load_settings(
                fname='auto_vendor_templates')
            if ok_temps_v:
                run_auto_vendor = True

        # import from image pool
        if '-i' in args and auto_common is None:
            logger.info(f'Could not find or read import settings from {path}')
        elif '-i' in args and auto_common is not None:
            if os.path.isdir(auto_common.import_path):
                log_import = import_incoming(
                    auto_common,
                    auto_templates,
                    tag_infos,
                    parent_widget=None,
                    ignore_since=ndays
                )
                append_log(log_import)
            else:
                logger.error('Path defined as image pool in import settings'
                             f'not valid dir: {auto_common.import_path}')

        # run templates
        modalities = [*QUICKTEST_OPTIONS]
        vendor = True  # first vendor templates if any
        for templates in [auto_vendor_templates, auto_templates]:
            run_this = run_auto_vendor if vendor else run_auto
            if run_this:
                labels = {}
                for mod in modalities:
                    arr = []
                    for temp in templates[mod]:
                        arr.append(temp.label)
                    labels[mod] = arr

                if vendor is False:
                    ok_params, path, paramsets = cff.load_settings(
                        fname='paramsets')
                    ok_qt, path, qt_templates = cff.load_settings(
                        fname='quicktest_templates')
                else:
                    paramsets = None
                    qt_templates = None

                if len(temps) > 0:
                    for t_id, mod_temp in enumerate(temps):
                        mod, templabel = mod_temp
                        if templabel in labels[mod]:
                            tempno = labels[mod].index(templabel)
                            template = templates[mod][tempno]
                            if vendor:
                                log_this, not_written = run_template_vendor(
                                    template, mod)
                            else:
                                log_this, not_written = run_template(
                                    template,
                                    mod,
                                    paramsets,
                                    qt_templates,
                                    tag_infos
                                )
                            append_log(log_this, not_written)
                        else:
                            v_txt = ('Automation template (vendor)' if vendor
                                     else 'Automation template')
                            logger.warning(f'{v_txt} {mod}/{templabel} not found.')
                if len(mods) == 0 and len(temps) == 0:
                    mods = modalities  # all if not specified
                if len(mods) > 0:
                    for mod in mods:
                        if labels[mod][0] != '':
                            for t_id, label in enumerate(labels[mod]):
                                if vendor:
                                    template = auto_vendor_templates[mod][t_id]
                                    log_this, not_written = run_template_vendor(
                                        template, mod)
                                else:
                                    template = auto_templates[mod][t_id]
                                    log_this, not_written = run_template(
                                        template,
                                        mod,
                                        paramsets,
                                        qt_templates,
                                        tag_infos
                                    )
                                append_log(log_this, not_written)
            vendor = False


def validate_args(sysargv):
    """Validate sys.argv and return list of interpreted args.

    Parameters
    ----------
    sysargv : sys.argv

    Returns
    -------
    status : bool
        True if validated for proceeding to next step.
    args : list of str
        List of arguments starting with -
    ndays : int
        Int after -ndays if present
    mods : list of str
        list of valid <modality> arguments
    temps : list of str
        list of <modality>/<templabel> arguments. <modality> validated
    msgs : list of str
        Error/warnings
    """
    args = []
    ndays = -1
    mods = []
    temps = []

    actmods = [*QUICKTEST_OPTIONS]

    status = True
    n_accepted_argcodes = 0  # number of accepted argument codes (-X)

    sysargv.pop(0)  # ignore imageQC.py
    for arg in sysargv:  # validate args
        if arg[0] == '-':
            argcode = arg[1]
            accepted_argcodes = 'inavd'
            if argcode in accepted_argcodes:
                if argcode == 'n':
                    argsplit = arg.split('=')
                    if len(argsplit) == 2:
                        try:
                            ndays = int(argsplit[1])
                        except ValueError:
                            logger.error('-ndays=XX could not convert XX to in.')
                            status = False
                else:
                    args.append(arg[0:2])   # keep short form only (-i not -import)
                    n_accepted_argcodes += 1
            else:
                logger.error(f'Argument {arg} not accepted.')
                status = False
        else:
            if '/' in arg:
                argsplit = arg.split('/')
                if len(argsplit) == 2:
                    if argsplit[0] in actmods:
                        if len(argsplit[1]) > 0:
                            temps.append(argsplit)  # [mod, templabel]
                        else:
                            logger.error(f'Argument {arg} not accepted.')
                            status = False
                    else:
                        logger.error(f'''Argument {arg} not accepted or valid as
                                      {argsplit[0]} not used as modality string in
                                      imageQC.''')
                        status = False
                else:
                    logger.error(
                        f'Argument {arg} not accepted or valid as <modality>/<temp>.')
                    status = False
            else:
                if arg in actmods:
                    mods.append(arg)
                else:
                    logger.error(
                        f'Argument {arg} not accepted or valid as modality.')
                    status = False

    if n_accepted_argcodes == 0:
        logger.error('Failed finding enough accepted arguments.')
        status = False
    else:
        n_a_v_d = args.count('-a') + args.count('-v') + args.count('-d')
        if n_a_v_d > 1:
            logger.error('Expected only one -a or -v or -d argument. Found more.')
            status = False
        if len(temps) > 0 and '-a' in args:
            logger.error('When specifying templates -v or -d has to be used, not -a.')
            status = False

    if status is False:
        print('python imageQC.py [-i [-ndays=n] [-a|-v|-d] [<mod>] [<mod>/<temp>]')

    return (status, args, ndays, mods, temps)


def import_incoming(auto_common, templates, tag_infos, parent_widget=None,
                    ignore_since=-1):
    """Import (rename and move) DICOM images if any in import_path of AutoCommon.

    Parameters
    ----------
    templates : modality_dict of AutoTemplate
    tag_infos : list of TagInfo
    parent_widget : widget or None
        Default is None
    ignore_since: int
        Override saved AutoCommon.ignore_since. Default is -1 (=not override)

    Returns
    -------
    import_log : list of str
        log messages
    """
    import_log = []
    proceed = False
    if os.path.exists(auto_common.import_path):
        p = Path(auto_common.import_path)
        files = [x for x in p.glob('**/*') if x.is_file()]
        if len(files) > 0:
            proceed = True
            if parent_widget is not None:
                proceed = proceed_question(
                    parent_widget,
                    f'Found {len(files)} files in import path. Proceed?'
                    )
        else:
            import_log.append(
                f'Found no new files in import path {p}')
    else:
        import_log.append(
            f'Failed to read import path {p}')

    if proceed:
        import_log.append(
            f'Found {len(files)} files in import path {p}')
        if ignore_since != -1:
            ignore_since = auto_common.ignore_since

        # get all station_names/dicom crits/input paths of automation_templates
        station_names = {}
        input_paths = {}
        dicom_crit_attributenames = {}
        dicom_crit_values = {}
        n_files_template = {}
        if templates is not None:
            nmod = len(templates)
            m = 0
            for mod, temps in templates.items():
                station_names[mod] = []
                for temp in temps:
                    if temp.active:
                        station_names[mod].append(temp.station_name)
                    else:
                        station_names[mod].append(None)
                input_paths[mod] = [temp.path_input for temp in temps]
                dicom_crit_attributenames[mod] = [
                    temp.dicom_crit_attributenames for temp in temps]
                dicom_crit_values[mod] = [
                    temp.dicom_crit_values for temp in temps]
                n_files_template[mod] = [0] * len(temps)
                if parent_widget is None:
                    m += 1
                    print_progress('Extracting template settings.', m, nmod)

        not_dicom_files = []  # index list of files not valid dicom
        no_template_match = []  # index list of files without template match
        multiple_match = []  # index list of file matching more than one templ
        no_input_path = []
        file_renames = [''] * len(files)  # new filenames generated from dicom
        file_new_folder = [auto_common.import_path] * len(files)  # input folder if set

        # prepare auto delete
        delete_files = []  # filenames to auto delete
        n_delete_crit = len(auto_common.auto_delete_criterion_attributenames)
        delete_pattern = cfc.TagPatternFormat(
            list_tags=auto_common.auto_delete_criterion_attributenames,
            list_format=['']*n_delete_crit)

        for f_id, file in enumerate(files):
            pd = {}
            try:
                pd = pydicom.dcmread(file, stop_before_pixels=True)
            except pydicom.errors.InvalidDicomError:
                not_dicom_files.append(f_id)

            if len(pd) > 0:
                # get modality and station name from file
                station_name = pd.get('StationName', '')
                modalityDCM = pd.get('Modality', '')
                mod = dcm.get_modality(modalityDCM)['key']

                # check for auto delete
                match_strings = dcm.get_dcm_info_list(
                    pd, delete_pattern, tag_infos,
                    prefix_separator='', suffix_separator='',
                    not_found_text='')
                delete_this = False
                for i, val in enumerate(auto_common.auto_delete_criterion_values):
                    if match_strings[i] == val:
                        delete_files.append(file)
                        delete_this = True
                        break

                if delete_this is False:
                    # generate new name
                    name_parts = dcm.get_dcm_info_list(
                        pd, auto_common.filename_pattern, tag_infos,
                        prefix_separator='', suffix_separator='',
                        not_found_text='')
                    new_name = "_".join(name_parts)
                    new_name = valid_path(new_name)  # + '.dcm'
                    file_renames[f_id] = new_name

                    # if station_name (+ dicom crit) - look for match
                    if mod in station_names:
                        if station_name in station_names[mod]:
                            match_idx = []
                            dcm_crit_only = False
                            idxs = get_all_matches(station_names[mod], station_name)
                            if len(idxs) == 0:
                                # try templates with empty station_name and dcm_crit
                                idxs = get_all_matches(station_names[mod], '')
                                dcm_crit_only = True
                            if len(idxs) > 0:
                                for idx in idxs:
                                    # Find matches to templates
                                    attr_names = dicom_crit_attributenames[mod][idx]
                                    if len(attr_names) > 0:
                                        tag_pattern = cfc.TagPatternFormat(
                                            list_tags=attr_names,
                                            list_format=[''] * len(attr_names))
                                        tag_values = dcm.get_dcm_info_list(
                                            pd, tag_pattern, tag_infos,
                                            prefix_separator='', suffix_separator='',
                                            not_found_text=''
                                            )
                                        if tag_values == dicom_crit_values[mod][idx]:
                                            match_idx.append(idx)
                                        else:
                                            # try fnmatch if wildcards
                                            fn_match = []
                                            for i, crit_val in enumerate(
                                                    dicom_crit_values[mod][idx]):
                                                if '?' in crit_val or '*' in crit_val:
                                                    if fnmatch(tag_values[i], crit_val):
                                                        fn_match.append(True)
                                                        pass
                                                    else:
                                                        fn_match.append(False)
                                                        break
                                            if len(fn_match) > 0 and all(fn_match):
                                                match_idx.append(idx)
                                    else:
                                        if dcm_crit_only is False:
                                            match_idx.append(idx)

                                if len(match_idx) > 1:
                                    multiple_match.append(f_id)
                                elif len(match_idx) == 0:
                                    no_template_match.append(f_id)
                                else:
                                    idx = match_idx[0]
                                    input_path = input_paths[mod][idx]
                                    if input_path == '':
                                        no_input_path.append(f_id)
                                    else:
                                        file_new_folder[f_id] = input_path
                                        n_files_template[mod][idx] += 1
                            else:
                                no_template_match.append(f_id)
                        else:
                            no_template_match.append(f_id)
                    else:
                        no_template_match.append(f_id)

            if parent_widget is None:
                print_progress('Reading DICOM header of files:', f_id + 1, len(files))

        # ensure uniq new file names
        fullpaths = [
            os.path.join(
                file_new_folder[f_id], file_renames[f_id]
                ) for f_id in range(len(files))]

        uniq_fullpaths = list(set(fullpaths))
        if len(uniq_fullpaths) < len(fullpaths):
            for path in uniq_fullpaths:
                n_equal = fullpaths.count(path)
                if n_equal > 1:
                    for i in range(n_equal):
                        idx = fullpaths.index(path)
                        fullpaths[idx] += f'_{i:03}'  # add suffix _XXX

        for f_id, file in enumerate(files):
            if file_renames[f_id] != '':
                status, errmsg = move_rename_file(
                    file,
                    fullpaths[f_id] + '.dcm',
                )
                if status is False:
                    import_log.append(errmsg)

        if templates is not None:
            for mod, temps in templates.items():
                temp_no = 0
                for temp in temps:
                    n_found = n_files_template[mod][temp_no]
                    if n_found > 0:
                        import_log.append(f'\t{mod}/{temp.label}: {n_found} new files')
                    temp_no += 1

        if len(not_dicom_files) > 0:
            import_log.append(
                f'{len(not_dicom_files)} files not valid DICOM.'
            )
        if len(no_template_match) > 0:
            import_log.append(
                f'{len(no_template_match)} files without match on any '
                'automation template. Left renamed in import path.'
            )
        if len(multiple_match) > 0:
            import_log.append(
                f'{len(multiple_match)} files matching more than one '
                'automation template. Left renamed in import path.'
            )
        if len(no_input_path) > 0:
            import_log.append(
                f'{len(no_input_path)} files matching automation '
                'template, but no input path defined for the template. '
                'Left renamed in import path.'
            )
        if len(delete_files) > 0:
            ndel = len(delete_files)
            for file in delete_files:
                try:
                    os.remove(file)
                except (PermissionError, OSError) as e:
                    import_log.append(f'Failed to delete {file}\n{e}')
                    ndel -= 1
            import_log.append(f'{len(delete_files)} files auto deleted.')

    return import_log


def verify_automation_possible(templates, templates_vendor):
    """Verify that any automation template exist.

    Parameters
    ----------
    templates : modality_dict of AutoTemplate
    templates_vendor : modality_dict of AutoVendorTemplate

    Returns
    -------
    status : bool
        True if any AutoTemplate found
    status_vendor : bool
        True if and AutoVendorTemplate found
    messages : list of str
        logg messages
    """
    status = False
    status_vendor = False
    messages = []

    for mod, temps in templates:
        labels = [temp.label for temp in temps]
        if len(labels) > 0 and '' not in labels:
            status = True
            break

    for mod, temps in templates_vendor:
        labels = [temp.label for temp in temps]
        if len(labels) > 0 and '' not in labels:
            status_vendor = True
            break

    if status is False and status_vendor is False:
        messages.append('No automation template defined.')

    return (status, status_vendor, messages)


def move_rename_file(filepath_orig, filepath_new):
    """Rename and move input file.

    Parameters
    ----------
    filepath_orig : str
        Original full path of a file.
    filepath_new : str
        New full path

    Returns
    -------
    status : bool
        True if rename succeeded
    errmsg : str

    """
    status = False
    errmsg = ''
    if os.path.exists(filepath_new):
        errmsg = f'Failed to move\n{filepath_orig}\nto {filepath_new}\nExists already.'
    else:
        try:
            filepath_orig.rename(filepath_new)
            status = True
        except (PermissionError, OSError) as e:
            errmsg = f'Failed to rename\n{filepath_orig}\nto {filepath_new}\n{e}'

    return (status, errmsg)


def run_template(auto_template, modality, paramsets, qt_templates, tag_infos,
                 select_files=False, parent_widget=None):
    """Find new images, generate img_dict, sort by date/sortpattern."""
    log = []
    not_written = []
    log_pre = f'Template {modality}/{auto_template.label}:'
    if os.path.exists(auto_template.path_input):
        p = Path(auto_template.path_input)
        files = [x for x in p.glob('**/*') if x.is_file()]
        if len(files) > 0:
            if parent_widget is None:
                print(f'Reading {len(files)} new files for template ',
                      f'{modality}/{auto_template.label} ...',
                      sep='', end='', flush=True)
            img_infos, ignored_files = dcm.read_dcm_info(
                files, GUI=False, tag_infos=tag_infos)

            if len(img_infos) > 0:
                qt_labels = [temp.label for temp in qt_templates[modality]]
                qt_idx = qt_labels.index(auto_template.quicktemp_label)
                param_labels = [temp.label for temp in paramsets[modality]]
                param_idx = param_labels.index(auto_template.paramset_label)
                paramset = paramsets[modality][param_idx]

                input_main_this = InputMain(
                    current_modality=modality,
                    current_paramset=paramset,
                    current_quicktest=qt_templates[modality][qt_idx],
                    tag_infos=tag_infos
                    )

                if parent_widget is None:
                    print(f'\rSorting {len(img_infos)} files for template ',
                          f'{modality}/{auto_template.label} ...',
                          sep='', end='', flush=True)

                # sort into groups of same acq_date, study uid
                date_uid_list = ['_'.join([info.acq_date, info.studyUID]) for
                                 info in img_infos]
                uniq_date_uids = list(set(date_uid_list))
                uniq_date_uids.sort()
                write_ok = os.access(auto_template.path_output, os.W_OK)
                if write_ok is False:
                    log.append(
                        f'\t No write permission to path {auto_template.path_output}')

                nd = len(uniq_date_uids)
                for d, uniq_date_uid in enumerate(uniq_date_uids):
                    if parent_widget is None:
                        print(f'\rAnalysing image set {d}/{nd} for template ',
                              f'{modality}/{auto_template.label} ...',
                              sep='', end='', flush=True)
                    date_str = (
                        uniq_date_uid[6:8] + '.'
                        + uniq_date_uid[4:6] + '.'
                        + uniq_date_uid[0:4]
                        )
                    img_infos_this = []
                    for i, img_info in enumerate(img_infos):
                        if uniq_date_uid == date_uid_list[i]:
                            img_infos_this.append(img_info)

                    input_main_this.imgs = img_infos_this
                    input_main_this.results = {}
                    input_main_this.current_group_indicators = []

                    calculate_qc(input_main_this)
                    value_list, header_list = quicktest_output(input_main_this)
                    header_list = ['Date'] + header_list
                    value_list = [date_str] + value_list
                    status, print_array = append_auto_res(
                        auto_template, header_list, value_list,
                        to_file=write_ok
                        )
                    if write_ok is False:
                        not_written.append(print_array)
                    else:
                        if auto_template.archive:
                            archive_files(input_main_this.imgs)
                print(f'\rFinished analysing template {modality}/{auto_template.label}',
                      '                                                              ')
                log.append(
                    'Finished analysing template ' +
                    f'{modality}/{auto_template.label} ({nd} sessions)'
                    )

    if len(log) > 0:
        log.insert(0, log_pre)
    return (log, not_written)


def run_template_vendor(auto_template, modality, pre_selected_files=[]):
    """Find new files and read."""
    log = []
    not_written = []
    log_pre = f'Template {modality}/{auto_template.label}:'

    if auto_template.file_type == '':
        log.append('\t File type not specified. Failed to proceed.')
    else:
        if len(pre_selected_files) > 0:
            files = pre_selected_files
        else:
            if auto_template.path_input == '':
                log.append('\t Input path not defined.')
            else:
                files = []
                p = Path(auto_template.path_input)
                if p.is_dir():
                    #glob_string = '**/*' if search_subfolders else '*'
                    #TODO - if GE QAP - search subfolder but not Archive
                    glob_string = '*'
                    files = [
                        x for x in p.glob(glob_string)
                        if x.suffix == auto_template.file_type
                        ]
            if len(files) > 0:
                write_ok = os.access(auto_template.path_output, os.W_OK)
                if write_ok is False:
                    log.append(
                        f'''\t Failed to write to output path
                        {auto_template.path_output}''')
                for file in files:
                    txt = read_vendor_QC_reports.get_pdf_txt(file)
                    res = read_vendor_QC_reports.read_Siemens_CT_QC(txt)

                    status, print_array = append_auto_res_vendor(
                        auto_template.path_output,
                        res, to_file=write_ok
                        )
                    if write_ok is False:
                        if len(not_written) == 0:
                            not_written.append(print_array)
                        else:
                            if len(not_written[-1]) == len(print_array[0]):
                                not_written.append(print_array[-1])  # ignore headers
                            else:
                                not_written.append(print_array)

    if len(log) > 0:
        log.insert(0, log_pre)
    return (log, not_written)


def append_auto_res(auto_template, headers, values, to_file=False):
    """Append test results to output path.

    Parameters
    ----------
    auto_template : AutoTemplate
    headers : list of str
    values : list of str
    to_file: bool
        True if output file verified. Default is False

    Returns
    -------
    status : bool
        False if append to file failed
    print_list : list of list
        values to print elsewhere if False
    """
    status = False
    print_list = [[]]

    if to_file:
        filesize = os.path.getsize(auto_template.path_output)
        if filesize == 0:
            print_list = [headers, values]
        else:
            print_list = [values]
        with open(auto_template.path_output, "a") as f:
            for row in print_list:
                f.write('\t'.join(row)+'\n')
        status = True
        print_list = [[]]

    return (status, print_list)


def append_auto_res_vendor(auto_template, result_dict, to_file=False):
    """Append test results to output path.

    Parameters
    ----------
    auto_template : AutoVendorTemplate
    output_template : QuickTestOutputTemplate
    result_dict : dict
        {<testcode>: {'headers': [], 'values': []}}
    to_file: bool
        True if output file verified. Default is False

    Returns
    -------
    status : bool
        False if append to file failed
    print_list : list of list
        [headers], [values] to print elsewhere if False
    """
    status = False
    print_list = [result_dict['headers']] + [result_dict['values']]

    if to_file:
        filesize = os.path.getsize(auto_template.path_output)
        if filesize > 0:
            print_list = print_list[1]
        with open(auto_template.path_output, "a") as f:
            for row in print_list:
                f.write('\t'.join(row))
        status = True
        print_list = []

    return (status, print_list)


def archive_files(input_main_imgs):
    """Move files into Archive folder."""
    all_files = [info.filepath for info in input_main_imgs]
    all_files = list(set(all_files))  # uniq filepaths if multiframe images
    first_file = Path(all_files[0])
    parent_path = first_file.parent
    archive_path = parent_path / 'Archive'
    archive_path.mkdir(exist_ok=True)
    if len(all_files) == 1:
        # move file directly to Archive
        first_file.rename(archive_path / first_file.name)
    else:
        # move files to Archive/yyyymmdd
        archive_path = archive_path / input_main_imgs[0].acq_date
        proceed = False
        try:
            archive_path.mkdir()
            proceed = True
        except FileExistsError:
            pass  # ignore if files already exists
        if proceed:
            for p in all_files:
                this_file = Path(p)
                this_file.rename(archive_path / this_file.name)


def print_progress(pretext, value, maxvalue, width=40):
    """Generate progress bar in print console."""
    ratio = value/maxvalue
    left = round(width * ratio)
    right = width - left

    tags = "#" * left
    spaces = " " * right
    percents = f"{(ratio*100):.0f}%"

    print(f'\r{pretext} [', tags, spaces, ']', percents, sep='', end='', flush=True)
    if value == maxvalue:
        print()
