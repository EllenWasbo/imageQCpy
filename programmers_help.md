# Help for programmers

## Starting virt env from scratch
- conda create --name viQC
- conda activate viQC
- cd C:\....\imageQCpy
- pip install -e .

For using pytest:
- pip install pytest
- pip install pytest-qt

using virt env later
- conda activate viQC
- cd C:\...\imageQCpy
- python -m imageQC.imageQC # to run the program
- python -m pytest tests\test_calculations.py # example to test

To be able to use breakpoints:
- Find developer_mode in imageQC.py (currently at line 198) and set this variable from False to True.

To be able to plot during pytest or breakpoints:
- import matplotlib.pyplot as plt
- plt.plot([x,y]) or plt.imshow(some_array)
- plt.pause(.1)

## Help to find any text in code
To easily find out where a method or class is used or if any breakpoints are left, one could use imageQCpy/helper_scripts/search_files.py.
Edit the content of variable search_string and run the script. This will list all files and line numbers where the search_string is found within directory src.

## Update resources.py after changes
If adding icons or file-based default configuration settings, resources.qrc will have to be updated and a new resources.py need de be recreated.
If adding icons, a dark-mode icon is also needed. Create a separate virtual environment and import PySide6 (as PyQt6 no longer have pyrcc5 to convert
.qrc to .py.

- add file to folder icons or config_defaults and add file name to resouces.qrc
- with PySide6 installed:
	- cd C:\...\imageQCpy\src\imageQC
	- pyside6-rcc resources.qrc -o resources.py
- open the resources.py file and replace the PySide6 import to PyQt6

## Adding new modality
- add modality to all relevant dictionaries in config/iQCconstants.py
- add ParamSet<mod> in config/config_classes.py + add this to class ParamSet and add paramsets_<mod> in LastModified
- add ParamsTab<mod> in ui/ui_main_test_tabs.py
- add default DCM settings in config_defaults/tag_patterns_test_dcm.yaml (and re-run pyrcc5 as explained above)
- add Modality to dash_app (table_overview)
- consider adding variants to scripts/dcm.py - get_modality

## When adding new parameters and variants to a test
- if alternative to current test: add result table headers in config/iQCconstants.py HEADERS (and HEADERS_SUP if supplement data) (and optionally ALTERNATIVES)
- add configurable parameters in config/config_classes.py ParamSet&lt;Modality&gt; (or ParamsetCommon if available to all paramsets), starting with the testcode (three letters)
- else follow instructions for adding testcode (below)

## When adding new testcode (analyse type)
- add testcode (three letters at least first capital) in config/iQCconstants.py QUICKTEST_OPTIONS same order as tabs for the tests
- add result table headers in config/iQCconstants.py HEADERS (and HEADERS_SUP if supplement data) (and ALTERNATIVES if alternative headers)
- add configurable parameters in config/config_classes.py ParamSet&lt;Modality&gt; (or ParamsetCommon if available to all paramsets)
- add GUI in ui/ui_main_test_tabs.py TestTab&lt;Modality&gt; order of tabs should fit QUICKTEST_OPTIONS order
	- self.&lt;same as paramname in ParamSet&gt; as gui element
	- if same as for other modality consider adding (moving) create_tab_&lt;testcode&gt; to the Common tab
	- verify that type of widget supported by params_changed_from_gui and update_displayed_params
- add roi in scripts/calculate_roi.py (in get_rois method &lt;testcode&gt;)
- if more than four colored rois / other visualization needs: add visualization of roi in ui/ui_image_canvas - ImageCanvas.roi_draw()
- add test calculations to calculate_qc
- add test visualizations to plot canvas (ui/ui_main_result_tabs - ResultPlotCanvas)
	- and optionally result image canvas (ui/ui_image_canvas - ResultImageCanvas)
- if more ALTERNATIVES or dynamic HEADERS given in iQCconstants - note the comments above these parameters within iQCconstants.py
 
## When adding new vendor QA reports to read
- add code for reading the file-type in scripts/read_vendor_QC_reports.py (using the existing methods or method read_pdf_dummy as guide/example)
- add option to config/iQCconstants.py VENDOR_FILE_OPTIONS. Parenthesis at end of option string have to contain file extension
- add the same string to the list implemented_types in ui/ui_main_test_tabs_vendor.py
- add the same string to read_vendor_template (method in scripts/read_vendor_QC_reports.py) for pointing to the method for the specific file-type
 
## When adding new types of templates (settings)
- add to iQCconstants CONFIG_FNAMES
- add template type to config_classes + add to last_modified
- start with a widget (inheriting from StackWidget of settings_reusables) that will have a similar use
- create the widgets and add to SettingsDialog __init__ (settings.py)
- in config_func.py load_settings - add code to if fname == '&lt;the new fname&gt;'
- if more than only the fname to load:
	- search for "'patterns'" in  settings_reusables.py - you will probably have to make these places work for you new template
- if coupled in auto_template or paramset or tag_infos by labels/attribute names - consider warnings (verify_auto_template / 
- consider add to 
	- MainWindow (update settings) if affect processes in main window
	- ImportMain (settings.py) if settings should be available for import (if yaml file it should be)
	- syncronize StackWidget update_from_yaml with SettingsDialog import from yaml + update_import main
	- InputMain (input_main_auto.py) if affect processes in calculate_qc or run_automation_non_gui/open_automation, run_template
- if not modality_dict:
	- add code to SettingsDialog - update_import_main, get_marked
	- SharedSettings - verify_config_files
- if PositionTable to set column headers - settings_reusables.py QuickTestOutputSubDialog elif testcode == in update data

## Update requirements.txt
- cd to src-folder
- pipreqs 
- requirements will now be in the src-folder. Move it to folder above src.
- remove skimage...?
- remove charset_normalizer (only for pyinstaller)
- remove plotly/dash_core_components/dash_html_components (already included with dash)
- Copy also new content to setup.cfg

## Update pdf version of Wiki
- download wikidoc code from https://github.com/jobisoft/wikidoc
	- Replace wikidoc.py in wikidoc-master with the one in helper_scripts folder where the code is updated for python3 and some fixes for linking pages
- install exe files: 
	- pandoc https://pandoc.org/installing.html 
	- wkhtmltopdf https://wkhtmltopdf.org

- conda install -c anaconda git
Clone git from github
- git clone https://github.com/EllenWasbo/imageQCpy.wiki.git &lt;some path&gt;\imageQCpy_wiki
- or update with Pull and GitHub Desktop

- cd to wikidoc-master
- python wikidoc.py C:\Programfiler\wkhtmltopdf\bin\wkhtmltopdf.exe &lt;some path&gt;\\imageQCpy_wiki\

Note that code used by wikidoc are within the .md files of imageQCpy/wiki

## For building .exe
This method reduces from ~300 MB vs ~1,2 GB compared to doing this in conda environment:
- Install python 3.13 and choose to add path
- Create an empty folder (somewhere) called to_exe
- In cmd.exe (not from Anaconda):
	- cd to the to_exe folder
	- python -m venv iQC (creates a virtual environment)
	- iQC\Scripts\activate.bat (activates the venv)
- Create an empty subfolder in to_exe called imageQCpy. This folder should hold the input and output for pyinstaller.
- Copy into the empty imageQCpy folder src and all files directly from folder above src except .gitignore/.pylintrc
- Delete these folders in src: icons, config_defaults + all pycache/eggs folders
- Delete also resources.qrc
- In cmd.exe (not from Anaconda):
	- cd imageQCpy (the new stripped folder)
- pip install -e . 
	- (if error on cwd None, try this: pip install --upgrade pip setuptools wheel --user)
- pip install pyinstaller (or pip list to see if its already installed)
- maybe: pip uninstall pathlib (have had some troubles and this solved it)

pyinstaller -i="iQC_icon.ico" --clean --hidden-import=['charset_normalizer.md__mypyc'] --collect-submodules=pydicom --paths=src\imageQC src\imageQC\imageQC.py

- avoid -w (--windowed) to make argparse work running imageQC with arguments for automation. Drawback is that cmd window always shown.

Fixed missing module when using import-hidden by adding from charset_normalizer import md_mypyc

- Create an empty config_defaults folder in folder dist (result from pyinstaller process)
- Copy NM_Auto_QC from the original config_defaults to this new empty folder.
- Add latest pdf user manual to dist.

- zip content of folder dist to distribute imageQCpy_versionXXX.zip

To run the exe file from cmd.exe:
- cd .....\dist\imageQC
- imageQC &lt;arg1&gt; &lt;arg2&gt;
