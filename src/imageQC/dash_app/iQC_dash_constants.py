#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constants accessible for several modules within imageQC.

@author: Ellen Wasbø
"""
import sys
import os


LOCAL = True  # if False, use minio

if local:
    path = sys.path[0] + '/.env'
    print(path)
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
    except ImportError:
        print('Failed to import dotenv to read .env')
else:
    try:
        from minio import Minio
    except ImportError:
        print('Failed to import Minio')


ENV_CONFIG_FOLDER = 'IMAGEQC_CONFIG_FOLDER'



