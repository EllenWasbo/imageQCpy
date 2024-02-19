# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 19:43:14 2022

@author: ellen
"""

import glob
import os
from pathlib import Path

directory = Path(__file__).parent.parent / 'src'

search_string = "psutil"

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".py") and file != 'search_files.py':
            py_file = os.path.join(root, file)

            with open(py_file) as f:
                found = False
                lines = f.readlines()
                for l, line in enumerate(lines):
                    if search_string in line:
                        print(f'Line {l}')
                        found = True
                if found:
                    print(file)
                    print('-------------------------------')
print('Finished')

