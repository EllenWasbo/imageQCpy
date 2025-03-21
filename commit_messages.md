#v3.1.15
_Mar 21, 2025_

Set default window level for Xray and Mammo to center min/max (else from DICOM).

NM test SNI:
- temporary set option to set sampling noise for point source to 0 = estimate quantum noise from mean of corrected.
- added reference calibration AutoQC file to test files.

#v3.1.14
_Mar 03, 2025_

- NM test SNI - changed default settings again (working on optimization, probably not last changes):
	- sampling frequency 0.004 
	- SNI area ratio increased to 0.97
	- grid of ROIs with ROI size 128
	- if correct for point source default sampling noise 5 times for averaging NPS quantum noise

# v3.1.13
_Feb 27, 2025_

Changes:
- NM test SNI (options withdrawn as regarded not usedful - sorry if these options were in use with automation - recommended to recalculate previous results):
	- Option with ROIs placed as Siemens gamma camer PMTs removed (ROIs should be overlapping / large enough - this option no longer recommended
	- Option using ratio from integral of radial profiles no longer ignore negative values (no artefakts should be zero i.e. include both positive/negative noise)
	- Filter low/high set as default and default frequencies as two filters up to ~0.2 pr mm (approximate nyquist frequency with pixel size 2.5 mm) (0/0.1/0.6 + 0.05/0.15/0.6)
	- Added option to average quantum noise NPS over N repeated poisson noise simulations pr image (option only used and available for point source correction without reference image)

# v3.1.12
_Feb 24, 2025_

Changes:
- Generate report: Notes now also possible to be used per element within html_table_row element
- NM test SNI: speed up on calculations (removing repetitive steps)
- Plot widget - added toolbar button to set x/y-ranges to min/max in current data.

Fixes:
- Fixed another 'cannot remove Artist' error - different behaviour different matplotlib versions
- NM test SNI: Fixed errors in tabulated results

# v3.1.11
_Feb 19, 2025_

Changes to the CDMAM test:
- Speed up on calculations and roi display
- Different image orientations now possible and verified.
- Center reading option "Multiply center/corner detection matrix" removed (regarded as not reasonable option). Minimum center/corner detection matrix set to default.

Changes to Generate report:
- Added option to duplicate report elements
- Fixed error when selecting image number 0 for result_plot/result_image

Fixes:
- Fixed 'cannot remove Artist' error from v3.1.10

# v3.1.10
_Feb 12, 2025_

Changes:
- Generate report: Added option to include/exclude image name for plot/image. Image name now centered on plot/image
- Xray Homogeneity: Added more explanation to the result table content in the information dialogue.
- CDMAM: v4.0 - cropped cells 4pix=200um from cell-grid (border) to avoid that border pixel-values affecting result

Fixes:
- Annotations lagging for matplotlib 3.10 now fixed


# v3.1.9
_Feb 10, 2025_

Changes:
- Mammo - CDMAM: small fixes (less nan in result table by allowing for extrapolation) and adjusting progress-bar
- Generate report: Fixes to confusing behaviour single vs all images/results. Apologies if established templates need adjustments. 

Bugfixes:
- Xray/Mammo - Homogeneity: fixed error when generating QuickTest results (automation output and export to clipboard)

# v3.1.8
_Feb 05, 2025_

Changes:
- CDMAM:
	- New attempts to better fit to disc positions when these are inaccurately positioned. Now summing cells of different images (if available) to find positions from thickest discs of relatively small size.
	- Added option to correct for center disc found to min of detection matrix for centers and corners
	- Added option to set search radius for minimum average for the discs. Default is 3 pixels as hardcoded before.
	- Added template visualization (yellow) in cell-processed image in addition to found/not found (green/red) center and corner.
- Reading DICOM images with missing required tags - missing tags are guessed. Error message print to cmd-window before fix.
Bugfixes:
- Button to refresh image display no longer crashing the program...

# v3.1.7
_Feb 02, 2025_

New functionalities:
- NM: Added test Sweep AutoQC (under validation). This test takes projection images from the Extrinsic Sweep Verification test (Siemens, AutoQC) and calculates
	- resolution (FWHM) as a 2d map
	- linearity 2d map = difference from average x-position of line source pr image
	- independent calculation of uniformity from sum of images pr detector (as does the AutoQC vendor software)

# v3.1.6
_Jan 30, 2025_

Changes:
- Report generator: Display of single images now possible and fixes to single image results displays.
- CDMAM4.0: 
	- finetuned expected position of discs (based on median offset of center disc of 10 central thickest discs
	- restructured code, more code moved from calculate_qc.py to cdmam_methods.py

# v3.1.5
_ Jan 16, 2025_

New functionalities:
- Value of scatter plots by color shown in toolbar when mouse-over (e.g. detection matrix and similar of CDMAM test)
- CDMAM: Added options on how to handle whether center discs are found for the cells

Fixes
- More robust CDMAM v3.4 phantom reading of position
- CDMAM - Result plot with imported fraction.xls now updated when selected

# v3.1.4
_Jan 13, 2025_

Fixes from v3.1.3
- CDMAM plots sets aspect ratio to 1, caused subsequent plots to be strange if very different axis scales. The aspect ratio is now reset an all new plots.
- Fixed error when reading CDMAM v3.4 phantom y-position
- Fixed error on generate report: Adding images as element with specific number of images pr row - now corredtly adding new row after N images
- Fixed updating Result plot and image when new selection for specific tests where trouble when introduced report_generator and its flexibility from v3.1.3.

# v3.1.3
_Jan 6, 2025_

New functionalities:
- Generate html/pdf-report from the current result. Save setup as templates.
- Xray:
	- Added option to test Homogeneity according to AAPM TG150 (Flat field test). Find option in the Homogeneity test tab under Methods.
	- Added option to read focal spot size from star test pattern.
- Mammo
	- Improved functionalities for GE Mammo QAP tests:
		- Added button to import Mammo QAP files from Automation dialog. Export from modality include all files. The new import function ignore files already analysed.
		- Better explaination for the auto-templates from the info-button in Settings - Automation templates vendor files (Mammo)
		- Tolerance read from the QAP-files and fixed automatic link to the right tolerance-templates (Limit and plot templates).
	- Added test for analysing CDMAM v3.4 and v4.0 phantoms. Test is not fully validated yet.

Changes:
- Test Variance (Xray and Mammo):
	- Variance now calculated for two different ROI sizes to highlight artifacts of different dimensions.
	- Changed from defining ROI by % of image dimensions to outer margin (in mm).
	- These changes will affect result table. Apologies if you already used this for automation. Normally such changes are avoided if possible. 
- Window level widget: 
	- Added button to set window level to (min, max) of central part of image (half the width/height)
	- Lock window level (checkbox) changed to toggle-button with lock-icon.
- Crosshair always removed when test DCM is selected. I.e. quick option to remove annotations is to select test DCM.
- Automation dialog:
	- Modality filter changed from dropdown list to checkable list. Last used modality filter will be saved to user preferences.
- CT task based image analysis: improved export options
- Added prefilter sigma for gaussian fit in supplement table. See more info in Wiki appendix C.
- Added option for Limits and Plot templates to set limit based on difference to first or median of all previous values. Also comparing text result to first text parameter is an option.

Fixes:
- Disabled open images and mode change until all GUI is initialized to avoid crash if user is too quick to open non-CT images on start-up.
- When a small image widget due to screen resolution, there were an issue with the extend toolbar-button (>>) not showing all tools. Now fixed.
- Xray/Mammo Homogeneity (flat field test): Bug from v3.1.2 fixed - Variance map pr ROI now visible again 
- Removed warnings from pydicom v3.0.1 when reading DICOM images not perfectly following the standard 
- Handled more strange inputs and accepting closing images without crashing for PET test Recovery curves
- other small fixes

Code structure:
- gathered MTF methods from calculate_qc.py and mini_methods_calculate.py into a new file mtf_methods.py
- gathered NM SNI and uniformity methods from calculate_qc.py and mini_methods_calculate.py into a new file nm_methods.py


# v3.1.2
_Oct 29, 2024_

New functionalities:
- Added module for TTF/NPS calculation for CT (task based image analysis). Find module from File-menu. d-prime calculation not finished. 
- CT/SPECT/PET: Option to calculate z-resolution from wire/linesource (in ~x direction) and from slanted edge in z-direction. Not validated for CT and SPECT, yet.
- In the Advanced open files dialog: option to import saved Tag Pattern - Format for grouping images (including modality specific tags).
- Simulating artifacts:
	- Added option to generate 3d artifacts (not fully validated yet)
	- Added option to set full image to zero before adding artifacts
- Added Variance as test for Mammo. Similar to Variance test for Xray, but with option to mask max-values as for Homogeneity (flat field test).
- Added option in Edit Annotations dialog to turn off overlays. Default is still to include overlay in image. NB - during automation overlay is always on. No option, yet, to turn this off.

Changes:
- Changes to calculations of MTF from line source: more robust on how to interpret the direction of the source 
- Test CT / Slicethickness: Added Catphan 700 option (ignoring vertical ramps for axial and horizontal ramps for helical slicethickness compared to Catphan 500/600)
- Test NM/SNI: When point source correction and using reference image - both the reference image and the image to be analysed will now always be corrected separately 
allowing for difference in point source position.
- Test Xray / Variance: Now the pixels outside the set percentage to analyse is masked. Previously calculating ROIs centered at the border included pixels outside the border.

Fixes:
- Avoiding crash and output to automation results when CT Siemens constancy pdf-file content is shorter than useful content.
- Avoiding crashes when switching between gaussian and discrete tabulated results for MTF.
- Avoiding crashes when copying curves (error from changes in v3.1.1)
- Avoiding crash when adding more than 8 ROIs for the ROI test.
- Avoiding crash for matplotlib version 3.9+ when changing modality mode or resetting offset.
- CT test NPS, fixed average plot when NPS not calculated for all images.
- Handeling pydicom v3.0+ (change in how pixel data is read from DICOM)
- Some DICOM files cast InvalidDicomError and miss Transfer Syntax UID. Now fix with guess on Transfer Syntax. (Seen for some dental CBCT images). 

# v3.1.1
_Aug 07, 2024_

New functionalities:
- PET: Added spatial resolution (MTF) similar to SPECT. Added option to perform line source test with sliding window, listed results pr window (image group)
to allow for more variation over the line source. Tested with Ge-68 line sources for Siemens PET.
- Copied flatfield test (from Mammo) as option to Xray (variant within test Hom).
- Started to add option to read info from RDSR files. SR added as modality with DCM as only test. To be continued for reading structured data.

Changes:
- When copying curves/plots to clipboard, curves sharing the same x-values (i.e. one x-axis pr all curves in clipboard, not one pr curve)
- Added option to simulate gamma camera point source as artifact
- Added option to merge pixels before NM SNI calculations
- Alternative or dynamic headers of results tables:
	- Added validation to output settings vs parameter settings
- Output settings of parametersets more user friendly:
	- When adding new settings - the alternative is set and locked to the one defined in the parameterset (more logical)
	- When editing, the alternative is selectable (for flexibility with old settings)
	- When changing parameters that might affect related output settings - a warning is given.
	- Added to validation test in 'Verify config files ...' in Settings - Config folder

Fixes:
- NM test SNI: v3.1.0 lost connection to reference image, if used. Now restored.
- Adding artifacts:
	- Handeling adding noise when <=0 within noise shape (setting values to 1 if <=0 before adding noise).
	- Better handeling artifact name when editing of settings. Keep autogenerating or keep manually edited.
- Fixed order error in Settings - LimitsAndPlotTemplates when editing templates.
- In some combinations of actions (zoom and also maybe profile plot involved?) the image zoom have been unable to reset by the home button and also the refresh button. 
 Another force reset attempt is added hopefully to be able to reset using the refresh button if this happens. So far not succeeded to reproduce the situation.
- Fixed crash when Dicom Dump used in combination with replace_import script.
 

# v3.1.0
_Jun 21, 2024_

Dependancies:
- avoided numpy > 2.0 newly released. Will need some validation and probably migration 

Changes:
- Dialog to add artifacts with improved functionalities, including option to save artifacts to file for reuse.
- Added option to change angle of ramp for CT slice thickness if other phantoms used. Also added default settings for GE phantom with horizontal wire ramps (angle 45degrees).
- NM SNI:
	- Option to let SNI be ratio between integrals of 2d NPS or radial profiles, ignoring negative values. Significant difference and clearly more sensitive with latter testing with simulated artifacts.
	- Option to choose whether point source correction with reference image should fit position to reference image or current image, avoiding artifacts to affect the fit.
	- Option to design low/high filter instead of human visual filter. Experimental purpose.
- Selected image index now displayed when working with QuickTest templates in the Settings manager (benefit if working with more than a few images).

Bugfixes:
- Some MR tests failed finding ROI if high signal at image borders. This resulted in program crash.
- Some issues finding center of object within ROI fixed = improved/fixed CT MTF from circular edge with low contrast and finding phantom position for PET Recovery curves.
- Avoiding crash when setting window level from active rectangle and rectangle dragged in other direction than top-right ot lower-left

# v3.0.9
_Jun 11, 2024_

Dependancies:
- Changes to dependancies as recommended due to security updates related to dash. If using the dashboard options, these upgrades are recommended.
To upgrade werkzeug==3.0.3, also an upgrade of flask==3.0.3 was needed. Updated in requirements.txt/setup.cfg.

Changes/fixes:
- Test NM SNI: option to ignore low frequency by modifying the human visual response filter 
- Method get_radial_profile corrected by half a pixel to reflect the center position of the pixels, not the corner of the pixels. This method is used when calculating the radial profiles of NPS, test Radial profile and SNI. The change might slightly affect the results of these tests.
- Added tag 0008,0033 (Content Time) as alternative to AcquisitionTime if tag 0008,0032 is missing.

Bugfixes:
- handeled error when using matplotlib v3.9.0: AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'
- sampling frequency of NPS for test SNI now responsive

# v3.0.8
_Apr 23, 2024_

New functionalities:
- New buttons in image toolbar:
	- Colormap select
	- Show projection (maximum/average intensity projection) with additional option to plot values from results table (e.g. mAs profile from extracted DCM values)
	- Set active area (rectangle) for defining rectangular ROIs (previously defined by zoom functionality). The active area will also be used when setting window level based on active area (min/max or avg/stdev)
	- The edit-tool: option to display the axis for images (x/y position) (affect also results image when redrawn)
- Added option to save image (or result image) as .csv to further study the values in e.g. Excel
- Added window level widget for result image.
- Added colorbar to window level widget.
- Added option for CT slicethickness from Siemens CT constancy phantom. Also auto updates ramp distance to typical value when changing ramp type.
- Added CT test TTF (task based MTF) - working on automated TTF/NPS/d' solution
- Added option to read GE QAP Mammo result files and option to bulk-generate automation templates for reading the different file types.
- Added option to simulate added artifacts to images (from the File menu). Valueable for example to simulate artifacts and look on how NPS is affected.

Changes:
- Added image index first in file list
- Added option to set NPS sampling frequency for test SNI.
- Cancel when running automation templates now stops during sessions of one template running, not just between templates.
- More modal progress bars with option to Cancel.
- Test Num:
	- considerably more robust for Siemens gamma camera savescreens that differ in screen size from day to day
	- default templates updated with larger ROIs to handle these day to day changes, ignoring parts of text at left/right border of ROI
- Changed startup time (saved some update after gui presented)
- CT test Slice thickness:
	- Added option to median filter profiles before finding the FWHM (for noisy images - better option is probably to increase mAs)
- NM test Uniformity and SNI:
	- Searching for non-zero image now also ignore neighbour row/column of zero values according to NEMA NU-1 (i.e. smaller found active image by 1 pixel each direction)
	- Uniformity:
		- Several changes speed up and to get closer to NEMA NU-1. This highly affect the results with UFOV ratio close to 1, else minimal effect (as intended).
		- Added option to cut the corners as the largest errors in ufov often are found there which is generally not clinical relevant.
		- Default ufov_ratio is changed from 0.95 to 1
		- Added to supplement result table: scaled pixel size + center pixel count (after scaling)
		- Added warning if center pixel < 10000 (NEMA minimum)
	- SNI:
		- Added option to calculate SNI based on ROIs positioned in grid matching PMT positions for Siemens gamma camera
		- Two large ROIs for all grid options
- NM test MTF (Spatial resolution from two perpendicular lines):
	- x and y used to be mixed up (now: x results = results for profile found from line in y direction and vise versa)
- Mammo test Homogeneity:
   - Added option to not calculate variance-map (to speed up if not needed)
   - Added option show count of deviating pixels within each ROI to better detect where these are located, list coordinates of these pixels and highlight with circles
   - Added option to set limit for ignoring calculations of ROI if masked pixels higher than the limit (%). Previously always ignored if any pixels masked. Calculations are performed ignoring the masked pixels.  
   - Added option to mask outer rim of given mm.
- Mammo Ghost: Changed default ROI size to 20 mm to get the recommended 4 cm^2 ROIs. Also adjustet default position of ROI_1 by 5 mm.

Bugfixes:
- fixed error when using image names for QuickTest where more images than expected are loaded (IndexError on set_names[i], calculate_qc.py line 158)
- default spinbox maximum is 100. Increased this value for different test-parameters that were not yet specified with max > 100.
- avoiding crash when scrolling through images using arrow keys (therefore skips showing some images as signal is blocked while processing). 
- black and dark gray plot lines in dark-mode changed to white and light gray
- fixed error when using slicethickness test in automation. (AttributeError 'Gui' object has no attribute 'active_img_no'
- fixed error when locking source distance for NM Uniformity and SNI when correcting for point source curvature.
- logging to file avoided (avoiding permission errors) when user_preferences path not yet set
- other small fixes

Code structure:
- moved some methods from ui_main to ui_main_methods for better reuse of methods for coming task_based_image_quality dialog
- + some other changes to what is available from widgets_reusable to avoid import loops


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
