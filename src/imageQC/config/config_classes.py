#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code holding the data classes used for configurable settings.

@author: Ellen Wasbo
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class UserPreferences:
    """Class holding local settings."""

    config_folder: str = ''
    dark_mode: bool = False
    font_size: int = 8
    annotations_line_thick: int = 1
    annotations_font_size: int = 8


@dataclass
class LastModified:
    """Class holding [username, epoch-time] for last change of config files."""

    tag_infos: list = field(default_factory=list)
    tag_patterns_special: list = field(default_factory=list)
    tag_patterns_format: list = field(default_factory=list)
    tag_patterns_sort: list = field(default_factory=list)
    rename_patterns: list = field(default_factory=list)
    paramsets: list = field(default_factory=list)
    quicktest_templates: list = field(default_factory=list)
    quicktest_output_templates: list = field(default_factory=list)
    auto_common: list = field(default_factory=list)
    auto_templates: list = field(default_factory=list)


@dataclass
class TagInfo:
    """Tag for the tags list to choose from."""

    sort_index: int = 0  # for holding unique index of tag
    attribute_name: str = ''
    # If tags have same name the next is used if the first is not found
    tag: list[hex] = field(default_factory=lambda: ['0x0', '0x0'])
    # format [0xXX,0xXX] group, element
    value_id: int = -1
    # if tag multivalue - value id (-1) = use all, -2 == per frame
    # -3 == combine all occurences of tag in sequence before valid = -2
    #       (i.e.= pr frame pr detector (NM))
    sequence: list = field(default_factory=lambda: [''])
    # search for tag within defined sequences - default no  sequence= ['']
    # example: ['PerFrameFunctionalGroupsSequence','PlanePositionSequence']
    # if more than one option - duplicate attribute name
    #  - if first option not found, try next
    limited2mod: list = field(default_factory=lambda: [''])
    # modality (imageQC) where this tag is valid, '' = valid for all
    unit: str = ''
    protected: bool = False  # avoid deleting this tag


@dataclass
class TagPatternSort:
    """Pattern of dicom tags for sorting images.

    list_tags = labels from tag_infos.yaml (TagInfo)
    Used for
     - sorting in automation templates
     - sorting in file list in main window
    """

    label: str = ''
    list_tags: list = field(default_factory=list)  # list[str]
    list_sort: list = field(default_factory=list)  # list[bool]

    def add_tag(self, tag='', sort=True, index=-1):
        """Add element in each list at given index or append."""
        if index == -1:
            self.list_tags.append(tag)
            self.list_sort.append(sort)
        else:
            self.list_tags.insert(index, tag)
            self.list_sort.insert(index, sort)

    def delete_tag(self, index):
        """Delete tag element."""
        try:
            self.list_tags.pop(index)
            self.list_sort.pop(index)
        except IndexError:
            print(f'Failed deleting tag idx {index} from {self}')


@dataclass
class TagPatternFormat:
    """Pattern of dicom tags with formatting for output (export or filename).

    list_tags = attribute_name from tag_infos.yaml (TagInfo)
    list_format |-separated string
        prefix, format-string for numbers (:..), suffix
    Used for
     - extracting or displaying DICOM header data
         - test DCM - tag pattern defined in each subgroup of ParamSet
         - tag pattern defined in tag_patterns_special:
             - Image annotation
             - DICOM header widget
             - file list displayed as DICOM tags rather than filepath
     - exporting DICOM header data to table for all open files
         - choose pattern from list or define (temporay) new
     - importing for automation
         - created in AutoCommon and saved in auto_common.yaml
         - general options only
    """

    label: str = ''
    list_tags: list = field(default_factory=list)  # list[str]
    list_format: list = field(default_factory=list)  # list[str]

    def __post_init__(self):
        """Add empty list_format if not defined."""
        if len(self.list_tags) > len(self.list_format):
            diff = len(self.list_tags) - len(self.list_format)
            self.list_format.extend([''] * diff)

    def add_tag(self, tag='', format_string='', index=-1):
        """Add element in each list at given index or append."""
        if index == -1:
            self.list_tags.append(tag)
            self.list_format.append(format_string)
        else:
            self.list_tags.insert(index, tag)
            self.list_format.insert(index, format_string)

    def delete_tag(self, index):
        """Delete tag element."""
        try:
            self.list_tags.pop(index)
            self.list_format.pop(index)
        except IndexError:
            print(f'Failed deleting tag idx {index} from {self}')


@dataclass
class RenamePattern:
    """Pattern of dicom tags for renaming images.

    Used in RenameDICOM
    All tag_pattern... strings = label from tag_pattern_formats.yaml
    """

    label: str = ''
    # subfolder / group
    list_tags: list = field(default_factory=list)  # list[str]
    list_format: list = field(default_factory=list)  # list[str]
    # file / img
    list_tags2: list = field(default_factory=list)  # list[str]
    list_format2: list = field(default_factory=list)  # list[str]


@dataclass
class HUnumberTable:
    """Set of parameters regarding CT tests."""

    materials: list[str] = field(
        default_factory=lambda:
        ['Teflon', 'Delrin', 'Acrylic', 'Water', 'Polystyrene',
         'LDPE', 'PMP', 'Air'])
    relative_mass_density: list[float] = field(
        default_factory=lambda:
        [2.16, 1.41, 1.18, 1., 1.05, 0.92, 0.83, 0.])
    pos_x: list[float] = field(
        default_factory=lambda:
        [-28., -58., -28., 0., 28., 58., 28., 0.])
    pos_y: list[float] = field(
        default_factory=lambda:
        [50., 0., -50., -58., -50., 0., 50., 58.])


@dataclass
class QuickTestOutputTemplate:
    """Class for holding output templates."""

    include_header: bool = False
    transpose_table: bool = False
    decimal_mark: str = '.'
    include_filename: bool = False  # for quickTest
    group_by: list = field(default_factory=lambda: ['SeriesInstanceUID'])
    # if per_group is set in QuickTestOutputSub
    tests: dict = field(default_factory=lambda: {})
    # dict {<testcode>: [QuickTestOutputSub]}


@dataclass
class QuickTestOutputSub:
    """Class for holding details for element of QuickTestOutputTemplates."""

    label: str = ''  # header_ prefix when header included
    alternative: int = 0  # if supplement table starting with 10
    columns: list = field(default_factory=lambda: [])  # list of ints
    calculation: str = '='
    per_group: bool = False


@dataclass
class ParamSetCommon:
    """Set of paramaters used for all parameter sets."""

    label: str = ''
    output: QuickTestOutputTemplate = field(
        default_factory=QuickTestOutputTemplate)
    dcm_tagpattern: TagPatternFormat = field(default_factory=TagPatternFormat)
    roi_type: int = 0  # 0=circular, 1=rectangular, 2=rectangular with rotation
    roi_radius: float = 5.
    roi_x: float = 10.
    roi_y: float = 10.
    roi_a: float = 0.
    roi_offset_xy: list[float] = field(default_factory=lambda: [0., 0.])
    roi_offset_mm: bool = False  # False = pix, True = mm


@dataclass
class ParamSetCT(ParamSetCommon):
    """Set of parameters regarding CT tests."""

    hom_roi_size: float = 10.
    hom_roi_distance: float = 55.
    hom_roi_rotation: float = 0.
    noi_roi_size: float = 55.
    huw_roi_size: float = 55.
    mtf_type: int = 2  # 0=bead, 1=wire, 2=circular edge
    mtf_roi_size: float = 11.
    mtf_background_width: float = 1.  # used if bead method
    mtf_plot: int = 3  # default plot 0=xyprofiles, 1=sorted, 2=LSF, 3=MTF
    mtf_gaussian: bool = True  # True= gaussian fit, False = discrete FFT
    mtf_cy_pr_mm: bool = True  # True= cy/mm, False = cy/cm
    mtf_cut_lsf: bool = True
    mtf_cut_lsf_w: float = 3.  # lsf_w from halfmax x FWHM
    mtf_cut_lsf_w_fade: float = 1.  # fade out width from lsf_w x FWHM
    mtf_offset_xy: list[float] = field(default_factory=lambda: [0., 0.])
    mtf_offset_mm: bool = False  # False = pix, True = mm
    mtf_auto_center: bool = False
    mtf_sampling_frequency: float = 0.01  # mm-1 for gaussian
    ctn_roi_size: float = 3.
    ctn_search_size: float = 11.
    ctn_search: bool = True
    ctn_table: HUnumberTable = field(default_factory=HUnumberTable)
    sli_ramp_distance: float = 38.
    sli_ramp_length: float = 60.
    sli_background_width: float = 5.
    sli_search_width: int = 10
    sli_average_width: int = 1
    sli_type: int = 0  # 0=wire Catphan, 1=beaded Catphan helical, 2=GE phantom
    #sli_signal_low_density: bool = False
    # False = wire/ramp higher density than background
    rin_median_filter_w: int = 0  # in pix on image
    rin_smooth_filter_w: float = 1.  # in mm on radial profile
    rin_range: list[float] = field(default_factory=lambda: [5., 65.])
    # mm from center
    rin_subtract_trend: bool = True
    # True = subtract trend, False = subtract mean
    nps_roi_size: int = 50
    nps_roi_dist: float = 50.
    nps_n_sub: int = 20
    nps_plot_average: bool = True


@dataclass
class ParamSetXray(ParamSetCommon):
    """Set of parameters regarding Xray tests."""

    hom_roi_size: float = 10.
    hom_roi_distance: float = 0.
    # % of shortes center to edge distance, if zero = center of quadrants
    hom_roi_rotation: float = 0.
    # if non-zero - same distance to center for all (half if distance is zero)
    hom_tab_alt: int = 0  # alternatives for what to calculate in table
    noi_percent: int = 90
    mtf_type: int = 1  # exponential=0, gaussian=1, None(discrete)=2
    mtf_roi_size_x: int = 20.
    mtf_roi_size_y: int = 50.
    mtf_plot: int = 3
    mtf_gaussian: bool = True  # True= (gaussian/exp) fit, False = discrete FFT
    mtf_cut_lsf: bool = True
    mtf_cut_lsf_w: int = 3
    mtf_offset_xy: list[float] = field(default_factory=lambda: [0., 0.])
    mtf_offset_mm: bool = False  # False = pix, True = mm
    mtf_auto_center: bool = False
    mtf_auto_center_type: int = 0  # 0 all edges, 1 = most central edge
    mtf_sampling_frequency: float = 0.01  # mm-1 for gaussian
    nps_roi_size: int = 256
    nps_sub_size: int = 5
    stp_roi_size: float = 11.3
    var_roi_size: float = 2.0


@dataclass
class ParamSetNM(ParamSetCommon):
    """Set of parameters regarding NM tests."""

    uni_ufov_ratio: float = 0.95
    uni_cfov_ratio: float = 0.75
    uni_correct: bool = False
    uni_correct_pos_x: int = 0
    uni_correct_pos_y: int = 0
    uni_correct_radius: float = -1.
    uni_sum_first: bool = False
    sni_area_ratio: float = 0.9
    sni_correct: bool = False
    sni_correct_pos_x: int = 0
    sni_correct_pos_y: int = 0
    sni_correct_radius: float = -1.
    sni_sum_first: bool = False
    sni_eye_filter_f: float = 1.3
    sni_eye_filter_c: float = 28.
    sni_eye_filter_r: float = 65.  # in mm
    mtf_type: int = 1
    mtf_roi_size: list[float] = field(default_factory=lambda: [20., 20.])
    mtf_plot: int = 4
    bar_roi_size: float = 50.
    bar_widths: list[float] = field(
        default_factory=lambda: [6.4, 4.8, 4.0, 3.2])
    spe_avg: int = 25
    spe_height: float = 100.
    spe_filter_w: int = 15


@dataclass
class ParamSetSPECT(ParamSetCommon):
    """Set of parameters regarding SPECT tests."""

    mtf_type: int = 1
    mtf_roi_size: float = 30.
    mtf_plot: int = 4
    mtf_3d: bool = True
    con_roi_size: float = 20.
    con_roi_dist: float = 58.


@dataclass
class ParamSetPET(ParamSetCommon):
    """Set of parameters regarding PET tests."""

    hom_roi_size: float = 10.
    hom_roi_distance: float = 55.
    cro_roi_size: float = 60.
    cro_volume: float = 0.


@dataclass
class ParamSetMR(ParamSetCommon):
    """Set of parameters regarding MR tests."""

    snr_roi_percent: float = 75.
    snr_roi_cut_top: int = 0
    piu_roi_percent: float = 75.
    piu_roi_cut_top: int = 0
    gho_roi_central: float = 80.
    gho_roi_w: float = 40.
    gho_roi_h: float = 10.
    gho_roi_dist: float = 10.
    gho_optimize_center: bool = True
    gho_roi_cut_top: int = 0
    geo_actual_size: float = 190.
    sli_tan_a: float = 0.1
    sli_roi_w: float = 100.
    sli_roi_h: float = 3.
    sli_dist_lower: float = -2.5
    sli_dist_upper: float = 2.5
    sli_optimize_center: bool = True


@dataclass
class ParamSet:
    """Collection of parametersets.

    Used for resetting and for reading from IDL version.
    """

    CT: ParamSetCT = field(default_factory=ParamSetCT)
    Xray: ParamSetXray = field(default_factory=ParamSetXray)
    NM: ParamSetNM = field(default_factory=ParamSetNM)
    SPECT: ParamSetSPECT = field(default_factory=ParamSetSPECT)
    PET: ParamSetPET = field(default_factory=ParamSetPET)
    MR: ParamSetMR = field(default_factory=ParamSetMR)


@dataclass
class QuickTestTemplate:
    """Class for holding QuickTest templates - which images to test."""

    label: str = ''
    tests: list = field(default_factory=list)
    # nested list of str = testcode(s) for each image
    image_names: list = field(default_factory=list)
    # if name is '' then use img number
    group_names: list = field(default_factory=list)
    # if name is '' series <seriesnumber> as in dicom

    def add_index(self, test_list=[], image_name='', group_name='', index=-1):
        """Add element in each list at given index or append."""
        if index == -1:
            self.tests.append(test_list)
            self.image_names.append(image_name)
            self.group_names.append(group_name)
        else:
            self.tests.insert(index, test_list)
            self.image_names.insert(index, image_name)
            self.group_names.insert(index, group_name)

    '''in use?
    def remove_indexes(self, ids2remove=[]):
        """Remove element(s) in each list at given index(es)."""
        ids2remove.sort(reverse=True)
        for id_rem in ids2remove:
            self.tests.remove(id_rem)
            self.image_names.remove(id_rem)
            self.group_names.remove(id_rem)
    '''


@dataclass
class AutoCommon:
    """Class for holding common settings for automation."""

    import_path: str = ''
    log_mode: str = 'w'
    auto_continue: bool = False  # ignored if without GUI
    last_import_date: str = ''  # yyyymmdd
    ignore_since: int = 0  # ignore importing images older than X days
    auto_delete_criterion_attributenames: list[
        str] = field(default_factory=lambda: ['Modality'])
    auto_delete_criterion_values: list[
        str] = field(default_factory=lambda: ['SR'])
    filename_pattern: TagPatternFormat = field(
        default_factory=TagPatternFormat)


@dataclass
class AutoTemplate:
    """Dataclass for keeping information on how to perform automation."""

    label: str = ''
    path_input: str = ''
    path_output: str = ''
    station_name: str = ''
    dicom_crit_attributenames: list = field(default_factory=list)
    dicom_crit_values: list = field(default_factory=list)
    sort_pattern: TagPatternSort = field(default_factory=TagPatternSort)
    paramset_label: str = ''
    quicktemp_label: str = ''
    archive: bool = True
    delete_if_not_image: bool = False
    delete_if_too_many: bool = False
    active: bool = True


@dataclass
class AutoVendorTemplate:
    """Dataclass for automation on vender file analysis."""

    label: str = ''
    path_input: str = ''
    path_output: str = ''
    station_name: str = ''
    archive: bool = False
    file_type: str = ''
    active: bool = True

