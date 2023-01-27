#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of small functions used in ImageQC.

@author: Ellen Wasbo
"""
from time import time
from pathlib import Path
import re


def time_diff_string(seconds):
    """Return time difference as string from input epoch time.

    Parameters
    ----------
    seconds : int
        epoch time

    Returns
    -------
    time_string : str
        xx seconds/minutes/hours/days ago
    """
    time_string = "?"
    diff = time() - seconds
    if diff < 60:
        time_string = str(round(diff)) + ' seconds'
    elif diff < 3600:
        time_string = str(round(diff/60)) + ' minutes'
    elif diff < 3600 * 24:
        time_string = str(round(diff/60/60)) + ' hours'
    else:
        time_string = str(round(diff/60/60/24)) + ' days'

    return ' '.join([time_string, 'ago'])


def get_format_strings(format_string):
    """Extract parts of format_string from list_format of TagPatternFormat.

    Parameters
    ----------
    format_string : str
        '' or prefix|format_string|suffix

    Returns
    -------
    prefix : str
        text before value
    format_part : st
        :.. to place behind {val:...} in an f-string
    suffix : str
        text after value
    """
    prefix = ''
    format_part = ''
    suffix = ''
    if format_string != '':
        try:
            prefix, format_part, suffix = format_string.split('|')
            if len(format_part) > 0:
                if format_part[0] == ':':
                    format_part = format_part[1:]
        except ValueError:
            pass

    return (prefix, format_part, suffix)


def format_val(val, format_string):
    """Format a value or list of values using a format string."""
    val_text = val
    if format_string != '':
        if '|' in format_string:
            prefix, format_string, suffix = get_format_strings(format_string)
        if isinstance(val, list):
            last_format = format_string[-1]
            if not isinstance(val[0], float) and last_format == 'f':
                try:
                    val = [float(str(x)) for x in val]
                    val_text = [
                        f'{x:{format_string}}' for x in val]
                except ValueError:
                    val_text = [f'{x}' for x in val]
                except TypeError:
                    val_text = [f'{x}' for x in val]
            else:
                val_text = [f'{x:{format_string}}' for x in val]
        else:
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

    return val_text


def valid_path(input_string, folder=False):
    """Replace non-valid characters for filenames.

    Parameters
    ----------
    input_string : str
        string to become path (filename)
    folder : bool, optional
        avoid . in name if folder = True, default is False

    Returns
    -------
    valid_string : str

    """
    valid_string = re.sub(r'[^\.\w]', '_', input_string)
    if folder:
        valid_string = re.sub(r'[\.]', '_', valid_string)

    return valid_string


def generate_uniq_filepath(input_filepath, max_attempts=1000):
    """Generate new filepath_XXX.ext if already exists.

    Parameters
    ----------
    input_filepath : str
        path to check whether uniq
    max_attempts : int, optional
        _999 is max. The default is 1000.

    Returns
    -------
    uniq_path : str
        unique path based on input, empty string if failed
    """
    uniq_path = input_filepath
    p = Path(input_filepath)
    if p.exists():
        for i in range(max_attempts):
            new_p = p.parent / f'{p.stem}_{i:03}.{p.suffix}'
            if new_p.exists() is False:
                uniq_path = new_p.resolve()
                break
        if uniq_path == input_filepath:
            uniq_path = ''  # = failed
    return uniq_path


def get_dynamic_formatstring(val):
    """Set number of decimals based on value.

    Used in val_2_str.

    Parameters
    ----------
    val : float
        Value to check for number of decimals

    Return
    ------
    format_str : string
        string to be used for formatting (after :) eg '.2e'
    """
    format_string = ''
    type_string = 'f'
    decimals = ''
    val = abs(val)
    if val == 0:
        decimals = '.3'
    elif val <= 1.0:
        decimals = '.3'
        if val < 0.1:
            decimals = '.4'
            if val < 0.01:
                decimals = '.5'
                if val < 0.00001:
                    decimals = '.3'
                    type_string = 'e'
    elif val > 1.:
        decimals = '.3'
        if val > 10:
            decimals = '.2'
            if val > 100:
                decimals = '.1'
                if val > 1e+05:
                    decimals = '.0'

    if decimals != '' or type_string != 'f':
        format_string = decimals + type_string

    return format_string


def convert_lists_to_numbers(taglists, ignore_columns=[]):
    """Try convert to number.

    Parameters
    ----------
    taglists : list of lists
        extracted dicom parameters
    ignore_columns : list of ints
        columns to ignore (not convert to number = messes with zero-padding)

    Returns
    -------
    taglists : list of lists
    """
    for r, row in enumerate(taglists):
        for c, val in enumerate(row):
            if c not in ignore_columns:
                new_val = val
                try:
                    new_val = int(val)
                except ValueError:
                    try:
                        new_val = float(val)
                    except ValueError:
                        pass
                taglists[r][c] = new_val

    return taglists


def val_2_str(val_list, decimal_mark='.'):
    """Convert value to string with some rules of precision.

    Parameters
    ----------
    val_list : list of ints or floats or strings
        input to be converted
    decimal_mark : str
        Decimal mark of output. Default is '.'

    Returns
    -------
    string_list : list of strings
    """
    string_list = []
    type_list = []
    actual_vals = []
    for v, val in enumerate(val_list):
        if val is not None:
            actual_vals.append(val)
        else:
            val_list[v] = ''
        if isinstance(val, str):
            type_list.append('str')
        elif val is None:
            type_list.append('None')
        elif 'float' in str(type(val)):
            type_list.append('float')
        elif 'int' in str(type(val)) or 'long' in str(type(val)):
            type_list.append('int')
        else:
            type_list.append('?')

    if type_list.count('str') == len(actual_vals):
        string_list = val_list
    elif type_list.count('float') == len(actual_vals):
        max_val = max(actual_vals)
        format_string = get_dynamic_formatstring(max_val)
        for val in val_list:
            if val != '':
                string_list.append(f'{val:{format_string}}')
            else:
                string_list.append('')
        if decimal_mark != '.':
            string_list = [e.replace('.', ',') for e in string_list]
    elif type_list.count('int') == len(val_list):
        string_list = [str(val) for val in val_list]
    else:  # mix or other
        for i, val in enumerate(val_list):
            if type_list[i] == 'float':
                format_string = get_dynamic_formatstring(val)
                str_this = f'{val:{format_string}}'
                if decimal_mark != '.':
                    str_this.replace('.', ',')
                string_list.append(str_this)
            else:
                string_list.append(str(val))

    return string_list

def valid_template_name(text):
    """No slash or space in template names (confuse automation)."""
    return re.sub('[\s/]+', '_', text)
