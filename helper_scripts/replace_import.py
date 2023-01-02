# -*- coding: utf-8 -*-
"""
@author: EllenWasbo

Procedure to replace import statements.
<pkgname>. needed before each for use with virtual environment setup / pytest / codeline
If running the program via Spyder, this is not working well

Each block of imports from <pkgname> packages have a comment before and after:
    # <pkgname> block start
    import statements
    # <pkgname> block end

remove_pkgname_dot = True
will remove <pkgname>. from these statements
remove_pkgname_dot = False
will add <pkgname>. to these statements
"""

import glob
import os
from pathlib import Path
from tkinter import Tk
from tkinter import filedialog, messagebox

pkgname = 'imageQC'

def add_pkg(line):
    new_line = ''
    found_in_line = False
    if search_string not in line:
        if line[:4] in ['from', 'impo']:
            found_in_line = True
            if line[:4] == 'from':
                new_line = f'{line[:5]}{search_string}{line[5:]}'
            if line[:6] == 'import':
                new_line = f'{line[:7]}{search_string}{line[7:]}'
    return (found_in_line, new_line)

root_tk = Tk()
directory = filedialog.askdirectory(
    title=f'Locate src folder of {pkgname}',
    initialdir=str(Path(__file__).parent.parent))
if any(directory):
    res = messagebox.askyesnocancel(
        title=f'Remove or add?',
        message=f'Remove {pkgname}. (Yes) or add (No)')
    remove_pkgname_dot = res
    if res is not None:
        search_string = f'{pkgname}.'
        start_stop_string = f'# {pkgname} block'

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    py_file = os.path.join(root, file)
                    lines = []
                    found = []
                    start = False
                    with open(py_file) as f:
                        lines = f.readlines()
                        for l, line in enumerate(lines):
                            if start_stop_string in line:
                                if start:
                                    break
                                else:
                                    start = True
                            if start:
                                if remove_pkgname_dot is True:
                                    if search_string in line:
                                        found.append(l)
                                        lines[l] = line.replace(search_string, '')
                                else:
                                    found_l, new_line = add_pkg(line)
                                    if found_l:
                                        found.append(l)
                                        lines[l] = new_line

                    if len(found) > 0:
                        print(file)
                        print(f'at lines {found}')
                        print('-------------------------------')
                        with open(py_file, 'w') as f:
                            for lin in lines:
                                f.write(lin)
        print('Finished')

root_tk.destroy()