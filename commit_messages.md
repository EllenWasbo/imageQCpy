# v3.0.8
_mon dd, 2024_

New functionalities:
- New buttons in image toolbar:
	- Colormap select
	- Show projection (maximum/minimum/average intensity projection) with additional option to plot values above from table (f.eks. mAs profile from extracted DCM values)
	- Set active area (rectangle). For defining ROIs and used when setting window level based on active area (min/max or avg/stdev)
- Added window level widget for result image.
- Added CT test TTF (task based MTF) - working on TTF/NPS/d' solution

Changes:
- Cancel when running automation templates now stops during one template running, not just between templates.
- Test Num:
	- considerably more robust for Siemens gamma camera savescreens that differ in screen size from day to day
	- default templates updated with larger ROIs to handle these day to day changes, ignoring parts of text at border of ROI

Bugfixes:
- fixed error when using image names for QuickTest where more images than expected are loaded (IndexError on set_names[i], calculate_qc.py line 158)


# v3.0.7
_Jan 26, 2024_

New functionalities:
- Added option to show all decimals in results table (last button in toolbar)
- Added test for Mammo where ROI of specific size can be placed relative to left or right image border.
- Dashboard:
	- Added legend per subplot and adjustable colorsettings (repeated for each subplot).
	- Added option to display dashboard from codeline (-d or --dash).

Changes:
- Reordered result table columns of Mammo test SDNR (signal before background) 

Fixes:
- Fixed rounding error for center when defining circular roi. Effect is that circular ROIs might be placed 1 pixel scewed compared to previous version.
- PET Recovery curve SUVpeak values now similar to EARL with finding 1cc with highest average rather than average of 1cc around max pixelvalue


# v3.0.6
_Jan 16, 2024_

New functionalities:
- Added Mammo as modality with a few tests and additional defualt dicom tags to read.
- Automation - functionality to move files out of archive now also include option to sort files by modification time, not only file name. Useful f.x for vendor QC files.

Fixes:
- RenameDicom have had some issues if there is a mix of files and folders directly in the selected path. This should no longer be a problem.
- Fixed incorrect behaviour if extra offset (test ROI and MTF).
- Fixed test ROI
	- tabulated ROIs with same shape - table used to show mm, but ROI effect was pixels if offset set to pixels
	- tabulated ROIs defined individually - mm removed from table headers - this option will always be pr pix, not mm
- Corrected tag Grid 0019,1166 to 0018,1166. This need to be corrected manually if tag info settings have been saved locally.
- Added cleanup code at startup to remove info about saved config files that has been removed manually from the config folder.
- Fixed nan result to zero for QuickTest output when all input values are zero.
- Fixes for Linux users (alternative locations for APPDATA and TEMPDIR).
- Fixes to Dashboard for trends. Handle more errors and more informative print messages to command window.

Security:
- Added functionality in GitHub to auto-detect potential security alerts for imported packages (Dependabot alerts). Added minimum version number criterias in requirements.txt according to these advices.
(werkzeug==2.3.8, numpy>=1.22, scipy>=1.10.0). Consider updating these packages if not running a full install when upgrading imageQC.
Werkzeug 3.0.1 had some issues so therefore set to ==2.3.8

Code structure:
- Reordered methods for the specific test - ordered alphabetically after test codes.

# v3.0.5
_Dec 14, 2023_

- Fixed bug with matplotlib 3.8.2 (for details see [Issue #4](https://github.com/EllenWasbo/imageQCpy/issues/4) on GitHub)
- Added option to analyse Spatial resolution from edge for modality NM
- Fixed updating ROI for each image during analysis (not only visually) when auto center is active
- Bugfixes when plotting with missing data / failed results
- Avoided paranthesis in file and folder names generated using RenameDICOM. Fixed bug where foldername generated dcm extension.
- Added option to read Philips MR ACR weekly report (pdf).

# v3.0.4
_Dec 04, 2023_

- added .IMA as DICOM extension variant (in addition to .dcm) - when searching for files and opening files
- bugfixes on AutoWizard (for details see  [Issue #3](https://github.com/EllenWasbo/imageQCpy/issues/3) on GitHub)
- bugfixes (catching more errors for Rename DICOM)
- bugfix when closing selected images

# v3.0.3
_Oct 13, 2023_

.exe + code

- Bugfixes

# v3.0.2
_Oct 11, 2023 - as .exe only_

- Updated readme.md
- Added content to Help menu.
- bugfixes LimitsAndPlot templates config

# v3.0.1
_Oct 05, 2023_

New options/functionalities:
- Added option to plot DCM parameters for test DCM for selected parameter i tag-list
- Removed the Contrast test for SPECT (never really implementet and not widely recommended)
- Added Ring artifact test for SPECT (same functionalities as for CT)
- Improvements to Limits and plot template validation and editing
Fixes/bugfixes:
- set colormap kept when another image is selected
- other small bugfixes

# v3.0.0_b16
_Sep 20, 2023_

New options/functionalities:
- If C:\Users and OneDrive in path defined in config folder settings, replace username with current to allow for shortcuts to sharepoint paths
- New option to view dashboard with trendlines (automation results).
- New option to set limits for automation results and append warnings to a specified txt file.
- New button in toolbar in automation dialog window: Option to open input path of selected template in file explorer
- Added calculation option to Parameterset output options: width (max-min) = to be able to extract max-min of result values
- included reading Siemens daily QC PET report from .xml (PET-MR) 
Fixes/bugfixes:
- more variants of Siemens CT Constancy and Quality Daily pdf files from SPECT CT (Intevo) can now be read.
- fixed bug when changing 'group by' in Parameterset output settings from SeriesUID. The changes now take effect. 
- other small bugfixes
 
# v3.0.0_b15 
_Aug 23, 2023_

New options/functionalities:
- new tools above file list:
	- move selected images to specific number in list
	- add dummy image (for use when QuickTest missing images
- when "Run in main window for selected files" (automation)
	- question to open output file also
	- question to add dummy images if missing
- added option to reload QuickTest template (new button in toolbar)
- added option to calculate MR SNR from single images (method 4 in NEMA MS 1-2008)
- open advanced option
	- Added option to edit the grouping of files
	- Added option to add all images in selected folder
Fixes/bugfixes:
- fixes to calculating slice thickness from GE CT phantom
- added tag 0018,1151 to find mA in e.g. GE CTs (in default dicom_tags, but not automatically updated if dicom tags edited i.e. local tag_infos.yaml)
- fixed error on test ROI when rotated rectangle combined with offcenter (large rotation angle revealed error)
- more modal progress bars and optimalization of existing
- paramsets files now listed in modified-log in Settings - Config folder
- other small bugfixes

# v3.0.0_b14 
_Aug 15, 2023_

New options/functionalities:
- Filter on modality possible in automation dialog
- automation DICOM templates can be marked as import only i.e. for files one just would like to collect or as a supplement for another template with the same input path if DICOM criteria collide
Fixes/bugfixes:
- fixes to modal progress bar during automation
- added warning when fewer images loaded than specified in current QuickTest template
- small bugfixes
- more validation of linked templates

# v3.0.0_b13
_Aug 8, 2023_

New options/functionalities:
- Running automation from codeline now finished (not full scale tested, yet)
- Zoom is now kept if image size is the same when selecting a new image
- Visualizing peak VOI for PET Recovery Curve
- Progress bar modal in Automation window - always option to Cancel/Stop unexpected long processes with loads of images
- Started to build settings for automation dashboard to display and warn if outside limits
Fixes/bugfixes:
- NM uniformity - not happy with imageQCpy giving larger numbers than IDL version and Siemens calculations
	- downscaling now closer to 6.4 mm pix (defined by NEMA). This will affect results as this effectivly is changed "smoothing"
		- option to set scaling factor manually instead of automatic
	- fixed error that only 4, not 5, pixels evaluated for the differential uniformity (affect results slightly)
	- fixed NEMA 2.4.4 "any pixels in the outer UFOV having less than 75% of mean in CFOV shall be set to zero" = not analysed
			if an outer row or column have values less than 75% of CFOV this row/col will be removed from ufov
- fixes to logging to file (automation)
- fixes to Recovery curve plotting (EARL tolerances). Auto set image to max in spheres after finished calculating.
- recognizing CBCT images from XA equipment as CT

# v3.0.0-b11__ 
_Jul 5, 2023_

New options/functionalities:
- NM SNI: option to use a grid of rois rather than the predefined 2 large, 6 small ROIs from the paper by Nelson et al 2013. SNI map available.
Fixes/bugfixes:
- Recovery curve - fixed errors finding spheres for full phantom
- fixes on automation with Digit Templates
- NM templates for Num with AutoQC from Siemens now included in config_defaults (forgotten to include in v3.0.0-b10)
- Dicom criteria now editable for automation templates
- other small fixes

# v3.0.0-b10
Jul 3, 2023

New options/functionalities:
- ROI test expanded to accept multiple ROIs. Min/max added as values to extract (in supplement table for single ROI).
- added test Num with 'Digit Templates' to read text from images based on user defined character image samples
Fixes/bugfixes:
- fixed error NM Bar phantom: changing bar widths now respond
- fixed missing data in quicktest results if all values of a row is 0
- fixed error in rename_dicom (in get_uniq_filenames)
- fixed error when SNI calculated from image in "portrait" mode (width < height)
- fixed error in visual xy coordinates with non-quadratic images and with offset (dx,dy other than zero)
- more robust detection of two line sources (NM) + avoid crashes
- more descriptive text in dialogbox when exporting QuickTest results (copy to clipboard)
- fixed bugs on copy/import ROI tables (CT numbers, Recovery Curve, Num)

# v3.0.0-b9
Jun 26, 2023

- Set pixelsize to 1x1 mm when not given to avoid ZeroDivisionError - warning given to user for these files
- Fixed error on rectangular ROI when color image (3d array, RGB)
- Sorted by zpos per series (if available) in open_multi (Advanced open files option)
- Fixes to Rename Dicom when splitting images into series without folder name template (using series UID)
- Fixed bugs in import dicom tags settings from another config folder
- Added functionality for version control. Version tag on each save. Warning if trying to save using older version. Update on default tags on startup with new version.
- Fixes to moving images up and down in list - updating results and keeping selected images, allowing for more than one to move
- Added option to use reference image for NM SNI (for use with calibration images Siemens)

# v3.0.0-b8
Jun 16, 2023

- result table now hide rows without results (images not marked for testing) - clicking image or table row will change highlight the corresponding table row or image
- copy selected part of result table now possible (Ctrl+C) or right-click and Copy selected from popup menu
- fixes for reading multiframe images CT
- changes to versions in requirements.txt (allow versions of pyqt5 and matplotlib that I have on my work-computer)
- fixed autocenter ROI MTF xray when inverted signal (high signal behind object)
- option to set min/max HU for CT number test and to change linearity parameter (name/unit text). Used to be relative mass density, now default values are relative e-density. HU-min max and e-density in preset (importable tables).

# v3.0.0-b7
Jun 13, 2023

fixed bug when deleting autodelete rules
fixed bug when slicelocation is missing
fixed bug on import of CT number predefined roi-tables, added HU min/max
added visual feedback on centering of circular disc for MTF
Recovery coefficient for PET ready for fine-tuning
added option to edit existing auto templates in wizard

# v3.0.0-b6
Jun 9, 2023

handle also matplotlib 3.7+
continued on Recovery Coefficients (PET),
fix on vendor files automation,
fix on quicktest output if alternative output types from same test,
bugfix on handeling deactivated templates,
warnings about extra offsets from config.dat (IDL) - please set again,
other bugfixes

# v3.0.0-b5 
Jun 6, 2023

fixes to progressbar during automation, fixes to NM uniformity with curvature correction, bugfixes - avoid crashes, continued on test for PET recovery curve

# v3.0.0-b4 
May 30, 2023

fixes to QuickTest output
added progress bar to automation window
 
# v3.0.0-b2 
May 23, 2023

fixed version number from last push
fixes to automation allowing to turn off ignore old images
fixes to QuickTest output
MR: added option to get slice thickness from wedge

# v3.0.0-b1 
May 22, 2023

small fixes / prep to future improvements
SNI with autoQC calibration on hold

# v3.0.20alpha 
May 12, 2023

bugfixes, fixes to SNI

# v3.0.19alpha
May 9, 2023

bugfixes + SNI fixes

# v3.0.18 alpha
May 8, 2023
 
bugfixes

# v3.0.17alpha
May 3, 2023

started adding PET test Recovery Curve - not finished
added - read_pdf_dummy function for easier adding other vendor QC PDF report files
bugfixes

# v3.0.16alpha
Apr 21, 2023

fixes to automation, fixed SNI calculations

# v3.0.15alpha
Apr 14, 2023

fixes to automation and quicktest
started on automation wizard

# v3.0.14alpha
Apr 4, 2023

bugfixes on quicktest output, point source correction NM flood, 3d line/wire source, handeling private dicom sequences

# v3.0.13alpha
Mar 14, 2023

rescale slope fixed for multiframe, fixes to MR tests, fixes to center of disc when offset

# v3.0.12alpha
Mar 6, 2023
Fixes on auto edge detection MTF xray/MR

# v3.0.11alpha
Mar 3, 2023

Many smaller fixes

# v3.0.10.2alpha
Feb 17, 2023
 
Small fixes to automation and a few other things

# v3.0.10alpha
Feb 16, 2023

added more tests (NPS and more) + other fixes

# v3.0.9alpha
Feb 7, 2023

Changed how paramsets are saved/loaded
Restructured/changed code towards PEP8
Added more tests
General fixes, smaller improvements
