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
from datetime import date, datetime
from time import time
import shutil
import webbrowser

# imageQC block start
import imageQC.config.config_func as cff
from imageQC.config.iQCconstants import QUICKTEST_OPTIONS
from imageQC.scripts.input_main import InputMain
from imageQC.scripts.calculate_qc import calculate_qc, quicktest_output
from imageQC.scripts import read_vendor_QC_reports
from imageQC.scripts import dcm
from imageQC.scripts.mini_methods import (
    get_all_matches, string_to_float, get_headers_first_values_in_path)
from imageQC.scripts.mini_methods_format import valid_path, val_2_str
import imageQC.config.config_classes as cfc
from imageQC.ui.messageboxes import proceed_question
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.settings_automation import DashWorker
# imageQC block end

logger = logging.getLogger('imageQC')


def run_automation_non_gui(args):
    """Run automation without GUI based on sys.argv."""
    actual_modalities = [*QUICKTEST_OPTIONS]

    def append_log(log_this, not_written=[]):
        """Append messages to log.

        Parameters
        ----------
        log_this : list of str
            messages to append to log
        not_written : list of str
            result-filepaths where results not written to file
        """
        if len(log_this) > 0:
            for msg in log_this:
                logger.info(msg)
        if len(not_written) > 0:
            logger.warning('Results not written to file:')
            for row in not_written:
                logger.info(row)

    def validate_modality_temps():
        """Validate modalities and template names.

        Returns
        -------
        status : bool
            True if validated for proceeding to next step.
        string: list of str
            List of arguments starting with -
        mods : list of str
            list of valid <modality> arguments
        temps : list of str
            list of <modality>/<templabel> arguments. <modality> validated
        msgs : list of str
            Error/warnings
        """
        modalities = []
        templates = []  # [modality, templatename]
        status = True

        for elem in args.modality_temp:
            if '/' in elem:
                elemsplit = elem.split('/')
                if len(elemsplit) == 2:
                    if elemsplit[0] in actual_modalities:
                        if len(elemsplit[1]) > 0:
                            templates.append(elemsplit)  # [mod, templabel]
                        else:
                            logger.error(f'Argument {elem} not accepted.')
                            status = False
                    else:
                        logger.error(f'Argument {elem} not accepted or valid as '
                                     f'{elemsplit[0]} not used as modality string in '
                                     'imageQC.')
                        status = False
                else:
                    logger.error(
                        f'Argument {elem} not accepted or valid as <modality>/<temp>.')
                    status = False
            else:
                if elem in actual_modalities:
                    modalities.append(elem)
                else:
                    logger.error(
                        f'Argument {elem} not accepted or valid as modality.')
                    status = False

        if status is False:
            print('python imageQC.py [-i [-ndays=n] [-a all/dicom/vendor] [-d/dash] '
                  '[<mod>] [<mod>/<temp>]')

        return (status, modalities, templates)

    logger.info('---------------imageQC automation started-----------------')
    ok_mod_temp = True
    if len(args.modality_temp) > 0:
        ok_mod_temp, set_modalities, set_templates = validate_modality_temps()
    else:
        set_modalities = []
        set_templates = []

    if ok_mod_temp:
        # load from yaml
        auto_common = None
        auto_templates = None
        auto_vendor_templates = None
        run_auto = False
        run_auto_vendor = False
        lastload_auto_common = time()
        ndays = args.ndays
        if args.import_images or args.auto in ['all', 'dicom']:
            _, path, auto_common = cff.load_settings(fname='auto_common')
            if ndays == -2:
                ndays = auto_common.ignore_since
            _, _, tag_infos = cff.load_settings(fname='tag_infos')
            ok_temps, _, auto_templates = cff.load_settings(fname='auto_templates')
            if args.auto in ['all', 'dicom']:
                if ok_temps:
                    run_auto = True
        if args.auto in ['all', 'vendor']:
            ok_temps_v, _, auto_vendor_templates = cff.load_settings(
                fname='auto_vendor_templates')
            if ok_temps_v:
                run_auto_vendor = True

        # import from image pool
        if args.import_images and auto_common is None:
            logger.info(f'Could not find or read import settings from {path}')
        elif args.import_images and auto_common is not None:
            if os.path.isdir(auto_common.import_path):
                import_status, log_import = import_incoming(
                    auto_common,
                    auto_templates,
                    tag_infos,
                    parent_widget=None,
                    ignore_since=ndays
                )
                append_log(log_import)
                if import_status:
                    fname = 'auto_common'
                    proceed_save, _ = cff.check_save_conflict(
                        fname, lastload_auto_common)
                    if proceed_save:
                        # save today as last import date to auto_common
                        auto_common.last_import_date = datetime.now().strftime(
                            "%d.%m.%Y %I:%M")
                        _, _ = cff.save_settings(auto_common, fname=fname)
            else:
                logger.error('Path defined as image pool in import settings'
                             f'not valid dir: {auto_common.import_path}')

        # run templates
        _, _, paramsets = cff.load_settings(fname='paramsets')
        _, _, qt_templates = cff.load_settings(fname='quicktest_templates')
        _, _, digit_templates = cff.load_settings(fname='digit_templates')
        _, _, limits_and_plot_templates = cff.load_settings(
            fname='limits_and_plot_templates')

        decimal_mark = '.'
        vendor = True  # first vendor templates if any
        warnings_all = []
        for templates in [auto_vendor_templates, auto_templates]:
            run_this = run_auto_vendor if vendor else run_auto
            if run_this:
                labels = {}
                for mod in actual_modalities:
                    arr = []
                    for temp in templates[mod]:
                        arr.append(temp.label)
                    labels[mod] = arr

                if len(set_templates) > 0:
                    for t_id, mod_temp in enumerate(set_templates):
                        warnings = []
                        mod, templabel = mod_temp
                        if vendor:
                            decimal_mark = paramsets[mod][0].output.decimal_mark
                        if templabel in labels[mod]:
                            tempno = labels[mod].index(templabel)
                            template = templates[mod][tempno]
                            if vendor:
                                log_this, warnings, not_written = run_template_vendor(
                                    template, mod,
                                    limits_and_plot_templates,
                                    decimal_mark=decimal_mark)
                            else:
                                log_this, warnings, not_written = run_template(
                                    template,
                                    mod,
                                    paramsets,
                                    qt_templates,
                                    digit_templates,
                                    limits_and_plot_templates,
                                    tag_infos
                                )
                            append_log(log_this, not_written)
                        else:
                            v_txt = ('Automation template (vendor)' if vendor
                                     else 'Automation template')
                            logger.warning(f'{v_txt} {mod}/{templabel} not found.')
                        if len(warnings) > 0:
                            paths_already = [msg_list[0] for msg_list in warnings_all]
                            if warnings[0] in paths_already:
                                idx = paths_already.index(warnings[0])
                                warnings_all[idx].extend(warnings[1:])
                            else:
                                warnings_all.append(warnings)
                if len(set_modalities) == 0 and len(set_templates) == 0:
                    set_modalities = actual_modalities  # all if not specified

                if len(set_modalities) > 0:
                    for mod in set_modalities:
                        if labels[mod][0] != '':
                            for t_id, label in enumerate(labels[mod]):
                                if vendor:
                                    template = auto_vendor_templates[mod][t_id]
                                    log_this, warnings, not_written =\
                                        run_template_vendor(
                                            template, mod,
                                            limits_and_plot_templates,
                                            decimal_mark=decimal_mark)
                                else:
                                    template = auto_templates[mod][t_id]
                                    log_this, warnings, not_written = run_template(
                                        template,
                                        mod,
                                        paramsets,
                                        qt_templates,
                                        digit_templates,
                                        limits_and_plot_templates,
                                        tag_infos
                                    )
                                append_log(log_this, not_written)
                                if len(warnings) > 0:
                                    paths_already = [
                                        msg_list[0] for msg_list in warnings_all]
                                    if warnings[0] in paths_already:
                                        idx = paths_already.index(warnings[0])
                                        warnings_all[idx].extend(warnings[1:])
                                    else:
                                        warnings_all.append(warnings)
            vendor = False
        if len(warnings_all) > 0:
            for warnings in warnings_all:
                proceed = os.path.exists(warnings[0])
                if proceed:
                    with open(warnings[0], "a") as file:
                        file.write('\n'.join(warnings[1:]))
                else:
                    append_log(['Failed adding warnings for violated limits to '
                               f'{warnings[0]}'])

        if args.dash:
            logger.info('------------Updating and displaying dashboard-------------')
            logger.info('NB! Dashboard runs as long as this program runs. '
                        'PRESS ENTER to EXIT program.')
            _, _, dash_settings = cff.load_settings(fname='dash_settings')
            dash_worker = DashWorker(dash_settings=dash_settings)
            dash_worker.start()
            url = f'http://{dash_settings.host}:{dash_settings.port}'
            webbrowser.open(url=url, new=1)
            _ = input(
                'Dashboard runs as long as this program runs. '
                'Press enter to exit program')


def import_incoming(auto_common, templates, tag_infos, parent_widget=None,
                    override_path=None, ignore_since=-1):
    """Import (rename and move) DICOM images if any in import_path of AutoCommon.

    Parameters
    ----------
    templates : modality_dict of AutoTemplate
    tag_infos : list of TagInfo
    parent_widget : widget or None
        Default is None
    override_path : str
        temporary import path if GUI
    ignore_since: int
        Override saved AutoCommon.ignore_since. Default is -1 (=none ignored)

    Returns
    -------
    status : bool
        proceeded to import
    import_log : list of str
        log messages
    """
    import_log = []
    proceed = False
    import_path = auto_common.import_path if override_path is None else override_path
    if os.path.exists(import_path):
        p_import = Path(import_path)
        files = [x for x in p_import.glob('**/*') if x.is_file()]
        if len(files) > 0:
            proceed = True
            if parent_widget is not None:
                proceed = proceed_question(
                    parent_widget,
                    f'Found {len(files)} files in import path. Proceed?'
                    )
        else:
            import_log.append(
                f'Found no new files in import path {p_import}')
    else:
        import_log.append(
            f'Failed to read import path {p_import}')

    if proceed:
        import_log.append(
            f'Found {len(files)} files in import path {p_import}')

        # get all station_names/dicom crits/input paths of automation_templates
        station_names = {}
        input_paths = {}
        dicom_crit_attributenames = {}
        dicom_crit_values = {}
        n_files_template = {}
        if templates is not None:
            nmod = len(templates)
            mod_no = 0
            for mod, temps in templates.items():
                station_names[mod] = []
                for temp in temps:
                    if temp.active and temp.label != '':
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
                    mod_no += 1
                    print_progress('Extracting template settings.', mod_no, nmod)

        not_dicom_files = []  # index list of files not valid dicom
        too_old_files = []  # older than ignore since if set
        no_date_files = []  # files without acq_date
        no_template_match = []  # index list of files without template match
        multiple_match = []  # index list of file matching more than one templ
        no_input_path = []
        file_renames = [''] * len(files)  # new filenames generated from dicom
        file_new_folder = [import_path] * len(files)  # input folder if set

        # prepare auto delete
        delete_files = []  # filenames to auto delete
        delete_rules = []  # auto delete rules used
        delete_pattern = cfc.TagPatternFormat(
            list_tags=auto_common.auto_delete_criterion_attributenames)

        today = date.today()

        progress_modal = None
        progress_max = len(files) * 2
        if parent_widget is not None:
            progress_modal = uir.ProgressModal(
                "Importing files...                            ", "Cancel",
                0, progress_max, parent_widget)  # 0-100 within each temp

        cancelled = False
        for f_id, file in enumerate(files):
            if parent_widget is not None:
                progress_modal.setValue(f_id)
                progress_modal.setLabelText(
                    f'Reading DICOM header of files: {f_id + 1} / {len(files)}')
            pyd, _, _ = dcm.read_dcm(file)
            if not pyd:
                not_dicom_files.append(f_id)
            else:
                # get modality and station name from file
                station_name = pyd.get('StationName', '')
                modality_dcm = pyd.get('Modality', '')
                mod = dcm.get_modality(modality_dcm)['key']

                # generate new name
                name_parts = dcm.get_dcm_info_list(
                    pyd, auto_common.filename_pattern, tag_infos,
                    prefix_separator='', suffix_separator='',
                    not_found_text='')
                delete_this = False
                ignore_this = False
                if any(name_parts):  # not valit if not
                    new_name = "_".join(name_parts)
                    new_name = valid_path(new_name)  # + '.dcm'

                    # check for auto delete
                    match_strings = dcm.get_dcm_info_list(
                        pyd, delete_pattern, tag_infos,
                        prefix_separator='', suffix_separator='',
                        not_found_text='')

                    for i, val in enumerate(auto_common.auto_delete_criterion_values):
                        if match_strings[i] == val:
                            match_rule = (
                                f'{auto_common.auto_delete_criterion_attributenames[i]}'
                                f' = {val}')
                            delete_files.append(file)
                            delete_rules.append(match_rule)
                            delete_this = True
                            break
                else:
                    not_dicom_files.append(f_id)
                    ignore_this = True

                if delete_this is False and ignore_this is False:
                    file_renames[f_id] = new_name

                    # older than ignore since?
                    proceed = True
                    if ignore_since > -1:
                        img_infos, _, _ = dcm.read_dcm_info(
                            [file], GUI=False, tag_infos=tag_infos)
                        try:
                            datestr = img_infos[0].acq_date
                            year = int(datestr[0:4])
                            month = int(datestr[4:6])
                            day = int(datestr[6:8])
                            acq_date = date(year, month, day)
                            diff = today - acq_date
                            if diff.days > ignore_since:
                                too_old_files.append(new_name)
                                proceed = False
                        except (IndexError, AttributeError, ValueError):
                            no_date_files.append(new_name)
                            proceed = False

                    # if station_name (+ dicom crit) - look for match
                    if proceed and mod in station_names:
                        if (
                                station_name in station_names[mod]
                                or '' in station_names[mod]):
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
                                            list_tags=attr_names)
                                        tag_values = dcm.get_dcm_info_list(
                                            pyd, tag_pattern, tag_infos,
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
                        if proceed:
                            no_template_match.append(f_id)

            if parent_widget is None:
                print_progress('Reading DICOM header of files:', f_id + 1, len(files))
            else:
                if progress_modal.wasCanceled():
                    cancelled = True
                    import_log.append('Import was cancelled.')
                    break

        if cancelled:
            if parent_widget is not None:
                progress_modal.close()
                parent_widget.status_label.clearMessage()
        else:
            if progress_modal is not None:
                progress_modal.setLabelText('Ensuring unique file names')
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

            if progress_modal is not None:
                progress_modal.setLabelText('Moving files')
            move_errors = []
            for f_id, file in enumerate(files):
                if progress_modal is not None:
                    progress_modal.setValue(int(len(files) + f_id * 0.5))
                if file_renames[f_id] != '':
                    new_path = (Path(file_new_folder[f_id]) != file.parent
                                or file_renames[f_id] != file.stem)
                    if new_path:
                        status, errmsg = move_rename_file(
                            file, fullpaths[f_id] + '.dcm')
                        if status is False:
                            move_errors.append(errmsg)
            if len(move_errors) > 0:
                import_log.append(f'Failed moving {len(move_errors)} files. '
                                  'See details to end of this log.')

            if progress_modal is not None:
                progress_modal.setLabelText('Generating log...')
                progress_modal.setValue(int(len(files) * 1.6))
            if templates is not None:
                import_log.append('\n')
                tot_found = 0
                for mod, temps in templates.items():
                    temp_no = 0
                    for temp in temps:
                        n_found = n_files_template[mod][temp_no]
                        if n_found > 0:
                            import_log.append(
                                f'\t{mod}/{temp.label}: {n_found} new files')
                            tot_found += n_found
                        temp_no += 1
                import_log.append(
                    f'\t In total {tot_found}/{len(files)} files matching templates')
                import_log.append('\n')

            if len(not_dicom_files) > 0:
                import_log.append(
                    f'{len(not_dicom_files)} files not valid DICOM.'
                )
            if len(too_old_files) > 0:
                import_log.append(
                    f'{len(too_old_files)} files older than specified limit. '
                    'Left renamed in import path.'
                )
            if len(no_date_files) > 0:
                import_log.append(
                    f'{len(no_date_files)} files without image content or acquisition '
                    'date. Left renamed in import path.'
                )
                import_log.append(
                    '\t Consider adding autodelete rules for these in import settings.'
                )
                for filename in no_date_files:
                    import_log.append(f'\t {filename}')
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
                if progress_modal is not None:
                    progress_modal.setLabelText('Auto deleting files...')
                    progress_modal.setValue(int(len(files) * 1.7))
                ndel = len(delete_files)
                del_files = []
                for file in delete_files:
                    try:
                        os.remove(file)
                        del_files.append(file)
                    except (PermissionError, OSError, FileNotFoundError) as err:
                        import_log.append(f'Failed to autodelete {file}\n{err}')
                        ndel -= 1
                import_log.append(
                    f'{len(delete_files)} files auto deleted according to import '
                    'settings.')
                rules = list(set(delete_rules))
                for rule in rules:
                    delete_rules.count(rule)
                    import_log.append(
                        f'\t {delete_rules.count(rule)} files where {rule}')

            if progress_modal is not None:
                progress_modal.setLabelText('Deleting empty folders...')
                progress_modal.setValue(int(len(files) * 1.9))
            if auto_common.auto_delete_empty_folders:
                dirs = [str(x) for x in p_import.glob('**/*') if x.is_dir()]
                dirs = sorted(dirs, key=len, reverse=True)
                for directory in dirs:
                    try:
                        Path(directory).rmdir()
                    except OSError:  # error if folder not empty
                        pass  # not empty
            if len(move_errors) > 0:
                import_log.append('Files failed moving:')
                import_log.extend(move_errors)
            if progress_modal is not None:
                progress_modal.close()

    return (proceed, import_log)


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

    for _, temps in templates:
        labels = [temp.label for temp in temps]
        if len(labels) > 0 and '' not in labels:
            status = True
            break

    for _, temps in templates_vendor:
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
        except (PermissionError, OSError) as err:
            if 'WinError 17' in str(err):
                shutil.move(filepath_orig, filepath_new)
                status = True
            else:
                errmsg = f'Failed to rename\n{filepath_orig}\nto {filepath_new}\n{err}'

    return (status, errmsg)


def test_limits(headers=None, values=None,
                limits_label='', limits_mod_templates=None):
    """Test limits and notify.

    Parameters
    ----------
    headers : list of str, optional
        QuickTest result headers without date. The default is None.
    values : list of str, optional
        QuickTest result values (one row). The default is None.
    limits_label : str, optional
        automation template set limits label. The default is ''.
    limits_mod_templates : TYPE, optional
        limits_and_plot_templates for current modality. The default is None.

    Return
    ------
    limits_crossed: bool
    msgs: list of str
    """
    limits_crossed = False
    proceed = False
    missing = []
    limits_template = None
    template_headers = []
    msgs = []
    if isinstance(values, list) and limits_label != '':
        limits = [temp.label for temp in limits_mod_templates]
        try:
            idx = limits.index(limits_label)
            limits_template = limits_mod_templates[idx]
            template_headers = [elem for sublist in limits_template.groups
                                for elem in sublist]
            missing = [header for header in headers if header not in template_headers]
            if len(missing) == len(headers):
                msgs.append('Headers in output file do not match headers defined in '
                            f'limits and plot template {limits_label}.')
            elif len(missing) > 0:
                msgs.append('Some column headers in limits and plot template '
                            f'{limits_label} do not match headers in output.')
                proceed = True
            else:
                proceed = True
        except ValueError:
            msgs.append(f'Failed finding limits and plot template named {limits_label}')

        if len(values) != len(headers):
            msgs.append('Number of extracted values do not match number of headers '
                        'in output file. Comparing to limits skipped.')
            proceed = False

    if proceed:
        for i, header in enumerate(headers):
            float_value = string_to_float(values[i])
            if float_value is not None:
                group_idx = limits_template.find_headers_group_index(header)
                if group_idx > -1:
                    limits = limits_template.groups_limits[group_idx]
                    if any(limits):
                        if limits[0] is not None:
                            if float_value < limits[0]:
                                limits_crossed = True
                                msgs.append(
                                    f'WARNING: {header} {float_value} lower than '
                                    f'set minimum {limits[0]}')
                        if limits[1] is not None:
                            if float_value > limits[1]:
                                limits_crossed = True
                                msgs.append(
                                    f'WARNING: {header} {float_value} higher than '
                                    f'set maximum {limits[1]}')

    return limits_crossed, msgs


def run_template(auto_template, modality, paramsets, qt_templates, digit_templates,
                 limits_and_plot_templates,
                 tag_infos, parent_widget=None):
    """Find new images, generate img_dict, sort by date/sortpattern.

    Parameters
    ----------
    auto_template : AutoTemplate
        as defined in config_classes.py
    modality : str
        modality as defined in imageQC
    paramsets : dict
        collection of available paramsets from .yaml
        keys = modalities
        values = list of ParamSetXX defined in config_classes.py
    qt_templates : dict
        collection of available quicktest templates from .yaml
        keys = modalities
        values = list of QuickTestTemplate defined in config_classes.py
    digit_templates: dict
        collection of available digit templates from .yaml
        keys = modalities
        values = list of DigitTemplate defined in config_classes.py
    limits_and_plot_templates: dict
        collection of available limits from .yaml
        keys = modalities
        values = list of LimitsAndPlotTemplate defined in config_classes.py
    tag_infos : list
        list of TagInfo as defined in config_classes.py
    parent_widget : widget, optional
        Widget to recieve messages. The default is None i.e. print messages to console.

    Returns
    -------
    log : list of str
        automation log - what has been performed
    warnings : list of str
        warnings when limits violated. first list element = log file path
    not_written : list of str
        results if failed writing results to file
    """
    log = []
    warnings = []
    not_written = []
    log_pre = f'Template {modality}/{auto_template.label}:'
    if auto_template.active is False:
        log.append('Deactivated template')
    else:
        proceed = os.path.exists(auto_template.path_input)
        if auto_template.quicktemp_label == '':
            proceed = False
        if auto_template.import_only:
            proceed = False
            log.append('Template marked as import only. No analysis.')
        if proceed:
            qt_labels = [temp.label for temp in qt_templates[modality]]
            qt_idx = qt_labels.index(auto_template.quicktemp_label)
            param_labels = [temp.label for temp in paramsets[modality]]
            proceed = True
            try:
                param_idx = param_labels.index(auto_template.paramset_label)
                paramset = paramsets[modality][param_idx]
            except ValueError:
                proceed = False
                log.append(
                    'No Paramset linked to the automation template. Template ignored.'
                    )
            if not any(qt_templates[modality][qt_idx].tests):
                proceed = False
                log.append(
                    ('The associatied Quicktest template have no tests specified. '
                     'Automation template ignored.'))
            if proceed:
                err_exist = False
                p_input = Path(auto_template.path_input)
                start_progress_val = 0
                if parent_widget is not None:
                    start_progress_val = parent_widget.progress_modal.value()
                if p_input.exists():
                    if parent_widget is not None:
                        parent_widget.progress_modal.setLabelText(
                            f'{auto_template.label}: Finding new files...')
                        parent_widget.progress_modal.setValue(start_progress_val + 2)
                    files = [x for x in p_input.glob('*') if x.is_file()]
                else:
                    files = []
                    err_exist = True
                if len(files) == 0:
                    if err_exist:
                        err_txt = (
                            f'{modality}/{auto_template.label}: '
                            f'Input path do not exist ({auto_template.path_input})')
                        if parent_widget is None:
                            print(err_txt)
                        else:
                            log.append(err_txt)
                else:
                    if parent_widget is None:
                        print(f'Reading {len(files)} new files for template ',
                              f'{modality}/{auto_template.label} ...',
                              sep='', end='', flush=True)
                    else:
                        parent_widget.progress_modal.setLabelText(
                            f'{auto_template.label}: Reading {len(files)} new files...')
                        parent_widget.progress_modal.setValue(start_progress_val + 3)
                    img_infos, _, warnings_dcm = dcm.read_dcm_info(
                        files, GUI=False, tag_infos=tag_infos)

                    if len(img_infos) > 0:
                        input_main_this = InputMain(
                            current_modality=modality,
                            current_paramset=paramset,
                            current_quicktest=qt_templates[modality][qt_idx],
                            digit_templates=digit_templates,
                            tag_infos=tag_infos
                            )
                        if len(warnings_dcm) > 0:
                            log.extend(warnings_dcm)
                        output_headers = []
                        if os.path.exists(auto_template.path_output):
                            output_headers, _ = get_headers_first_values_in_path(
                                auto_template.path_output)

                        if parent_widget is None:
                            print(f'\rSorting {len(img_infos)} files for template ',
                                  f'{modality}/{auto_template.label} ...',
                                  sep='', end='', flush=True)
                        else:
                            parent_widget.progress_modal.setLabelText(
                                f'{auto_template.label}: Sorting {len(img_infos)} '
                                'files by date')
                            parent_widget.progress_modal.setValue(
                                start_progress_val + 4)

                        # sort into groups of same acq_date, study uid
                        date_uid_list = ['_'.join([info.acq_date, info.studyUID]) for
                                         info in img_infos]
                        uniq_date_uids = list(set(date_uid_list))
                        uniq_date_uids.sort()
                        write_ok = os.access(auto_template.path_output, os.W_OK)
                        if write_ok is False:
                            log.append(
                                f'\t No write permission to {auto_template.path_output}'
                                )

                        n_sessions = len(uniq_date_uids)
                        finish_text = (
                            f'Finished analysing template ({n_sessions} sessions)')
                        if parent_widget is not None:
                            curr_val = parent_widget.progress_modal.value()
                            diff = curr_val - start_progress_val
                            sub_interval = int((100-diff)/n_sessions)
                            parent_widget.progress_modal.sub_interval = sub_interval

                        for ses_no, uniq_date_uid in enumerate(uniq_date_uids):
                            if parent_widget is None:
                                print(f'\rAnalysing image set {ses_no}/{n_sessions} '
                                      'for template ',
                                      f'{modality}/{auto_template.label} ...',
                                      sep='', end='', flush=True)
                            else:
                                if parent_widget.progress_modal.wasCanceled():
                                    finish_text = (
                                        'Analysis was cancelled after '
                                        f'{ses_no} sessions.'
                                        )
                                    break
                            date_str = (
                                uniq_date_uid[6:8] + '.'
                                + uniq_date_uid[4:6] + '.'
                                + uniq_date_uid[0:4]
                                )
                            img_infos_this = []
                            for i, img_info in enumerate(img_infos):
                                if uniq_date_uid == date_uid_list[i]:
                                    img_infos_this.append(img_info)

                            if len(auto_template.sort_pattern.list_tags) > 0:
                                img_infos_this, _ = dcm.sort_imgs(
                                    img_infos_this, auto_template.sort_pattern,
                                    tag_infos)

                            input_main_this.imgs = img_infos_this
                            # reset
                            input_main_this.results = {}
                            input_main_this.current_group_indicators = []
                            input_main_this.errmsgs = []

                            calculate_qc(
                                input_main_this, wid_auto=parent_widget,
                                auto_template_label=auto_template.label,
                                auto_template_session=f'Session {ses_no+1}/{n_sessions}'
                                )
                            value_list, header_list = quicktest_output(input_main_this)
                            if all([
                                    auto_template.limits_and_plot_label != '',
                                    output_headers]):
                                limits_crossed, msgs = test_limits(
                                    headers=output_headers, values=value_list,
                                    limits_label=auto_template.limits_and_plot_label,
                                    limits_mod_templates=limits_and_plot_templates[
                                        modality],
                                    )
                                if limits_crossed:
                                    log.extend(msgs)
                                    if auto_template.path_warnings != '':
                                        warnings.append(f'{date_str}:')
                                        warnings.extend(msgs)
                            header_list = ['Date'] + header_list
                            value_list = [date_str] + value_list

                            if len(input_main_this.errmsgs) > 0:
                                log.append(f'{date_str}: WARNING')
                                log.extend(input_main_this.errmsgs)

                            _, print_array = append_auto_res(
                                auto_template, header_list, value_list,
                                to_file=write_ok
                                )
                            if write_ok is False:
                                not_written.append(print_array)
                            else:
                                if auto_template.archive:
                                    errmsg = archive_files(
                                        input_main_imgs=input_main_this.imgs)
                                    if errmsg is not None:
                                        log.append(errmsg)
                        if parent_widget is None:
                            print(
                                '\rFinished analysing template                        ',
                                '                                                     ')
                        log.append(finish_text)

        else:
            if auto_template.import_only is False:
                if os.path.exists(auto_template.path_input) is False:
                    log.append(f'Input path not found ({auto_template.path_input})')
                if auto_template.quicktemp_label == '':
                    log.append(
                        ('No QuickTest template was linked to the automation template. '
                         'Template ignored'))
    if len(log) > 0:
        log.insert(0, log_pre)
    if len(warnings) > 0:
        warnings.insert(0, f'Template {auto_template.label}')
        warnings.insert(0, auto_template.path_warnings)
    return (log, warnings, not_written)


def run_template_vendor(auto_template, modality,
                        limits_and_plot_templates,
                        decimal_mark='.', parent_widget=None):
    """Find new files and read.

    Parameters
    ----------
    auto_template : AutoVendorTemplate
        as defined in config_classes.py
    modality : str
        modality as defined in imageQC
    limits_and_plot_templates: dict
        collection of available limits from .yaml
        keys = modalities
        values = list of LimitsAndPlotTemplate defined in config_classes.py
    decimal_mark : str
        . or ,
    parent_widget : widget, optional
        Widget to recieve messages. The default is None i.e. print messages to console.

    Returns
    -------
    log : list of str
        automation log - what has been performed
    warnings : list of str
        warnings when limits violated. first list element = log file path
    not_written : list of str
        results if failed writing results to file
    """
    log = []
    warnings = []
    not_written = []
    log_pre = f'Template {modality}/{auto_template.label}:'
    files = []
    n_ok = 0

    if auto_template.active is False:
        log.append('Deactivated template')
    else:
        if parent_widget is not None:
            parent_widget.progress_modal.setLabelText(f'{log_pre} Finding files...')
        proceed = os.path.exists(auto_template.path_input)
        if proceed is False:
            log.append('\t Input path not defined or do not exist.')
        if auto_template.file_type == '':
            proceed = False
            log.append('\t File type not specified. Failed to proceed.')
        if proceed:
            p_input = Path(auto_template.path_input)
            if p_input.is_dir():
                files = [
                    x for x in p_input.glob('*')
                    if x.suffix == auto_template.file_suffix
                    ]
                files.sort(key=lambda t: t.stat().st_mtime)
        if len(files) > 0:
            write_ok = os.access(auto_template.path_output, os.W_OK)
            if write_ok is False:
                log.append(
                    f'''\t Failed to write to output path
                    {auto_template.path_output}''')
            output_headers = []
            if os.path.exists(auto_template.path_output):
                output_headers, _ = get_headers_first_values_in_path(
                    auto_template.path_output)
            for fileno, file in enumerate(files):
                if parent_widget is not None:
                    if parent_widget.progress_modal.wasCanceled():
                        break
                    else:
                        parent_widget.progress_modal.setLabelText(
                            f'{log_pre} Extracting data from file '
                            f'{fileno}/{len(files)}...')
                        curr_val = parent_widget.progress_modal.value()
                        new_val = curr_val + int(100/len(files))
                        parent_widget.progress_modal.setValue(new_val)
                res = read_vendor_QC_reports.read_vendor_template(auto_template, file)
                if res is None:
                    log.append(f'\t Found no expected content in {file}')
                else:
                    status, print_array = append_auto_res_vendor(
                        auto_template.path_output,
                        res, to_file=write_ok, decimal_mark=decimal_mark
                        )
                    if status:
                        n_ok += 1
                    if auto_template.limits_and_plot_label != '' and output_headers:
                        limits_crossed, msgs = test_limits(
                                headers=output_headers,
                                values=print_array[0][1:],
                                limits_label=auto_template.limits_and_plot_label,
                                limits_mod_templates=limits_and_plot_templates[
                                    modality],
                                )
                        if limits_crossed:
                            log.extend(msgs)
                            if auto_template.path_warnings != '':
                                warnings.append(f'{print_array[-1][0]}:')  # date
                                warnings.extend(msgs)
                    if write_ok is False:
                        if len(not_written) == 0:
                            not_written.append(print_array)
                        else:
                            if len(not_written[-1]) == len(print_array[0]):
                                not_written.append(print_array[-1])  # ignore headers
                            else:
                                not_written.append(print_array)
                    else:
                        if auto_template.archive:
                            errmsg = archive_files(filepath=file)
                            if errmsg is not None:
                                log.append(errmsg)

    if len(log) > 0:
        log.insert(0, log_pre)
    else:
        if n_ok > 0:
            log = [log_pre, f'\tFinished reading {n_ok} files']
        else:
            log = [log_pre, '\tNo files read correctly']
    if len(warnings) > 0:
        warnings.insert(0, f'Template {auto_template.label}')
        warnings.insert(0, auto_template.path_warnings)
    return (log, warnings, not_written)


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
        with open(auto_template.path_output, "a") as file:
            for row in print_list:
                file.write('\t'.join(row)+'\n')
        status = True
        print_list = [[]]

    return (status, print_list)


def append_auto_res_vendor(output_path, result_dict, to_file=False, decimal_mark='.'):
    """Append test results to output path.

    Parameters
    ----------
    output_path : str
    result_dict : dict
        {<testcode>: {'headers': [], 'values': []}}
    to_file: bool
        True if output file verified. Default is False
    decimal_mark : str

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
        filesize = os.path.getsize(output_path)
        if filesize > 0:
            print_list = [print_list[1]]  # ignore header
        with open(output_path, "a") as file:
            for row in print_list:
                row_strings = val_2_str(
                    row, lock_format=True, decimal_mark=decimal_mark)
                file.write('\t'.join(row_strings) + '\n')
        status = True
        print_list = []

    return (status, print_list)


def archive_files(input_main_imgs=None, filepath=None):
    """Move files into Archive folder.

    Parameters
    ----------
    input_main_imgs : list of DcmInfo, optional
        The default is None.
    filepath : Path, optional
        file (vendor type file).The default is None.

    Returns
    -------
    errmsg : string or None
    """
    errmsg = None
    if input_main_imgs is not None:
        all_files = [info.filepath for info in input_main_imgs]
        all_files = list(set(all_files))  # uniq filepaths if multiframe images
        first_file = Path(all_files[0])
    if filepath is not None:
        all_files = [filepath]
        first_file = filepath
    parent_path = first_file.parent
    archive_path = parent_path / 'Archive'
    archive_path.mkdir(exist_ok=True)
    n_fail = 0
    if len(all_files) == 1:
        # move file directly to Archive
        try:
            first_file.rename(archive_path / first_file.name)
        except FileExistsError:
            n_fail += 1
    else:
        # move files to Archive/yyyymmdd - only if dicom
        archive_path = archive_path / input_main_imgs[0].acq_date
        archive_path.mkdir(exist_ok=True)

        for path in all_files:
            this_file = Path(path)
            try:
                this_file.rename(archive_path / this_file.name)
            except FileExistsError:
                n_fail += 1
    if n_fail > 0:
        errmsg = (
            f'\t{n_fail} files failed to archive, name already exist.'
            'Left in input path.')

    return errmsg


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
