#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions for extracting text from images

@author: Ellen Wasbo
"""
import numpy as np


def extract_char_blocks(nparr):
    """Chop numpy array assumed to contain a few chars.

    Parameters
    ----------
    nparr : numpy.2darray
        part of image with numbers

    Returns
    -------
    list_of_nparr : list of numpy.2darray
        list of images split into characters
    chopping_idxs : list of tuple
        (start_x, end_x) positions for chopping off the characters
    """
    def to_quadratic(arr):
        """Set array to quadratic by filling zeros."""
        szy, szx = arr.shape
        if szy >= szx:
            quad_arr = np.zeros((szy, szy))
            diff = szy - szx
            startx = diff // 2
            quad_arr[:, startx:startx+szx] = arr
        else:
            quad_arr = arr  # assumed never used as digits are higher than wide
        return quad_arr

    list_of_nparr = []
    chopping_idxs = []
    if len(nparr.shape) == 2 and np.min(nparr.shape) > 2:
        nparr = nparr - np.min(nparr)
        nparr = nparr / np.max(nparr)
        background = nparr[0][0]
        if background > 0:
            nparr = 1 - nparr
        # now signal from 0 to 1, background is assumed zero
        prof_x = np.sum(nparr, axis=0)
        signal = np.where(prof_x > 0)
        prof_x[signal[0]] = 1
        diff = np.diff(prof_x)
        start_char_pos = np.where(diff > 0)[0] + 1
        end_char_pos = np.where(diff < 0)[0] + 1

        if all([
                len(start_char_pos) > 0,
                len(end_char_pos) == len(start_char_pos)]):
            chopping_idxs = [
                (start_char_pos[i], end_char_pos[i])
                for i in range(len(start_char_pos))]
            prof_y = np.sum(nparr, axis=1)
            signal = np.where(prof_y > 0)
            prof_y[signal[0]] = 1
            diff = np.diff(prof_y)
            start_y = np.where(diff > 0)[0]
            end_y = np.where(diff < 0)[0]
            try:
                y0 = start_y[0] + 1
            except IndexError:
                y0 = 0
            try:
                y1 = end_y[0] + 1
            except IndexError:
                y1 = len(prof_y)

            for idxs in chopping_idxs:
                arr_quadratic = to_quadratic(nparr[y0:y1, idxs[0]:idxs[1]])
                list_of_nparr.append(arr_quadratic)

    return (list_of_nparr, chopping_idxs)


def compare_char_blocks_2_template(img_blocks, template):
    """Compare found image blocks to image blocks in DigitTemplate.

    Parameters
    ----------
    img_blocks : list of numpy.2darray
    template : DigitTemplate
        as defined in config/config_classes.py

    Returns
    -------
    number : float, int or None
        the string found converted to float, int or None if no match
    """
    template_images = []
    for img in template.images:
        if isinstance(img, list):
            img = np.array(img)
        template_images.append(img)  # list for saving, ensure numpy array before eval

    def match_block(img):
        char = None
        chars = [str(i) for i in range(10)] + ['.', '-']
        subtr = []
        for i, temp_img in enumerate(template_images):
            if isinstance(temp_img, np.ndarray):
                if img.shape == temp_img.shape:
                    if np.array_equal(img, temp_img):
                        char = chars[i]
                        break
                    else:
                        diff = np.subtract(img, temp_img)
                        subtr.append(np.abs(np.sum(diff)))
        if char is None and len(subtr) > 0:
            idx = np.where(subtr == np.min(subtr))
            char = chars[idx[0][0]]
        return char

    number = None
    if len(img_blocks) > 0:
        chars = []
        for block in img_blocks:
            chars.append(match_block(block))
        if all(chars):  # match on all to accept
            string = ''.join(chars)
            try:
                if '.' in string:
                    number = float(string)
                else:
                    number = int(string)
            except ValueError:
                pass

    return number
