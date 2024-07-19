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
    digit_templates: list = field(default_factory=list)
    paramsets_CT: list = field(default_factory=list)
    paramsets_Xray: list = field(default_factory=list)
    paramsets_Mammo: list = field(default_factory=list)
    paramsets_NM: list = field(default_factory=list)
    paramsets_SPECT: list = field(default_factory=list)
    paramsets_PET: list = field(default_factory=list)
    paramsets_MR: list = field(default_factory=list)
    paramsets_SR: list = field(default_factory=list)
    quicktest_templates: list = field(default_factory=list)
    auto_common: list = field(default_factory=list)
    auto_templates: list = field(default_factory=list)
    auto_vendor_templates: list = field(default_factory=list)
    dash_settings: list = field(default_factory=list)
    limits_and_plot_templates: list = field(default_factory=list)


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
    factor: float = 1.0  # multiply by factor (igored if 1.0)
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

    def __post_init__(self):
        """Add empty list_sort if not defined."""
        if len(self.list_tags) > len(self.list_sort):
            diff = len(self.list_tags) - len(self.list_sort)
            self.list_sort.extend([''] * diff)

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
class DigitTemplate:
    """Dataclass for text identify (digit) templates."""

    label: str = ''
    images: list = field(default_factory=list)
    # list of numpy arrays for the digits 0-9 and . and -
    active: bool = False  # flag to tell if it is ready for use (all digits set)


@dataclass
class PositionTable:
    """Set of labels and positions used for different test ROIs."""

    labels: list[str] = field(default_factory=list)
    pos_x: list[float] = field(default_factory=list)
    pos_y: list[float] = field(default_factory=list)

    def __post_init__(self):
        """Make sure labels if pos given."""
        if len(self.labels) < len(self.pos_x):
            diff = len(self.pos_x) - len(self.labels)
            self.labels.extend([''] * diff)

    def add_pos(self, label='', pos_x=0, pos_y=0, index=-1):
        """Add element in each list at given index or append."""
        if index == -1:
            self.labels.append(label)
            self.pos_x.append(pos_x)
            self.pos_y.append(pos_y)
        else:
            self.labels.insert(index, label)
            self.pos_x.insert(index, pos_x)
            self.pos_y.insert(index, pos_y)

    def delete_pos(self, index):
        """Delete element."""
        try:
            self.labels.pop(index)
            self.pos_x.pop(index)
            self.pos_y.pop(index)
        except IndexError:
            print(f'Failed deleting tag idx {index} from {self}')


@dataclass
class HUnumberTable:
    """Set of default materials and positions for CT test on HUnumbers."""

    labels: list[str] = field(
        default_factory=lambda:
        ['Teflon', 'Delrin', 'Acrylic', 'Water', 'Polystyrene',
         'LDPE', 'PMP', 'Air'])
    linearity_unit: str = 'Rel. e-density'
    linearity_axis: list[float] = field(
        default_factory=lambda:
        [1.868, 1.363, 1.147, 1., .998, 0.945, 0.853, 0.001])
    pos_x: list[float] = field(
        default_factory=lambda:
        [-28., -58., -28., 0., 28., 58., 28., 0.])
    pos_y: list[float] = field(
        default_factory=lambda:
        [50., 0., -50., -58., -50., 0., 50., 58.])
    min_HU: list[int] = field(default_factory=list)
    max_HU: list[int] = field(default_factory=list)

    def add_pos(self, label='', pos_x=0, pos_y=0, index=-1):
        """Add element in each list at given index or append."""
        if index == -1:
            self.labels.append(label)
            self.pos_x.append(pos_x)
            self.pos_y.append(pos_y)
            self.linearity_axis.append(0.)
            if len(self.min_HU) > 0:
                self.min_HU.append(0)
                self.max_HU.append(0)
        else:
            self.labels.insert(index, label)
            self.pos_x.insert(index, pos_x)
            self.pos_y.insert(index, pos_y)
            self.linearity_axis.insert(index, 0.)
            if len(self.min_HU) > 0:
                self.min_HU.insert(index, 0)
                self.max_HU.insert(index, 0)
        if len(self.min_HU) == 0:
            self.min_HU = [0 for i in range(len(self.labels))]
            self.max_HU = [0 for i in range(len(self.labels))]

    def delete_pos(self, index):
        """Delete element."""
        try:
            self.labels.pop(index)
            self.pos_x.pop(index)
            self.pos_y.pop(index)
            self.linearity_axis.pop(index)
            if len(self.min_HU) > 0:
                self.min_HU.pop(index)
                self.max_HU.pop(index)
        except IndexError:
            print(f'Failed deleting tag idx {index} from {self}')

    def fix_list_lengths(self):
        """Seen errors on list lengths."""
        n_labels = len(self.labels)
        for attr in ['pos_x', 'pos_y', 'min_HU', 'max_HU', 'linearity_axis']:
            this_list = getattr(self, attr)
            if len(this_list) != n_labels:
                if len(this_list) > n_labels:
                    setattr(self, attr, this_list[:n_labels])
                else:
                    extra = [0] * (n_labels - len(this_list))
                    setattr(self, attr, this_list + extra)


@dataclass
class RecTable(PositionTable):
    """Set of default background roi positions with test PET Recovery Curves."""

    def __post_init__(self):
        """Set default values."""
        if len(self.labels) == 0:
            self.labels = [str(i) for i in range(6)]
            self.pos_x = [-55, -110, -100, 100, 110, 55]
            self.pos_y = [75, 40, -25, -25, 40, 75]


@dataclass
class QuickTestOutputTemplate:
    """Class for holding output templates."""

    # NB if changing parameters here - fix in config/config_func.py:load_settings
    include_header: bool = False
    transpose_table: bool = False
    decimal_mark: str = '.'
    decimal_all: bool = False
    include_filename: bool = False  # for quickTest
    group_by: list = field(default_factory=lambda: ['SeriesInstanceUID'])
    # if per_group is set in QuickTestOutputSub
    tests: dict = field(default_factory=lambda: {})
    # dict {<testcode>: [QuickTestOutputSub]}


@dataclass
class QuickTestOutputSub:
    """Class for holding details for element of QuickTestOutputTemplates."""

    label: str = ''  # header_ prefix when header included
    alternative: int = 0  # supplement table starting with 10, -1 used for not defined
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
    roi_use_table: int = 0  # 0 = one point with offset, 1 = table of offsets, 2 zooms
    roi_table: PositionTable = field(default_factory=PositionTable)
    roi_table_val: int = 0
    # no in roi_headers + roi_headers_sup [average, stdev, min, max]
    roi_table_val_sup: int = 1  # same options as roi_table_val
    num_roi_size_x: int = 100
    num_roi_size_y: int = 50
    num_table: PositionTable = field(
        default_factory=lambda: PositionTable(
            pos_x=[(10, 20)], pos_y=[(10, 20)]))
    num_digit_label: str = ''


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
    ttf_roi_size: float = 11.
    ttf_gaussian: bool = True  # True= gaussian fit, False = discrete FFT
    ttf_cut_lsf: bool = True
    ttf_cut_lsf_w: float = 3.  # lsf_w from halfmax x FWHM
    ttf_cut_lsf_w_fade: float = 1.  # fade out width from lsf_w x FWHM
    ttf_sampling_frequency: float = 0.01  # mm-1 for gaussian
    ttf_table:  PositionTable = field(default_factory=PositionTable)
    ctn_roi_size: float = 3.
    ctn_search_size: float = 11.
    ctn_search: bool = True
    ctn_table: HUnumberTable = field(default_factory=HUnumberTable)
    ctn_auto_center: bool = False
    ctn_plot: int = 0  # 0=HU min max diff, 1=HU min max diff % 2=linearity
    sli_ramp_distance: float = 38.
    sli_ramp_length: float = 60.
    sli_background_width: float = 5.
    sli_search_width: int = 10
    sli_average_width: int = 1
    sli_median_filter: int = 0
    sli_type: int = 0
    # 0=wire Catphan, 1=beaded Catphan helical, 2=GE phantom, 3=Siemens phantom
    sli_tan_a: float = 0.42  # tangens of ramp angle
    sli_auto_center: bool = False
    rin_sigma_image: float = 0.  # sigma for gaussfilter of image
    rin_sigma_profile: float = 0.  # sigma for gaussfilter of radial profile
    rin_range_start: float = 5.  # mm from center
    rin_range_stop: float = 65.  # mm from center
    rin_subtract_trend: bool = True  # True = subtract trend, False = subtract mean
    nps_roi_size: int = 50
    nps_roi_distance: float = 50.
    nps_n_sub: int = 20
    nps_smooth_width: float = 0.05  # 1/mm
    nps_sampling_frequency: float = 0.01  # 1/mm
    nps_normalize: int = 0  # normalize curve by 0 = None, 1 = AUC, 2 = large area sign
    nps_plot: int = 0  # default plot 0=pr image, 1=avg, 2=pr image+avg, 3=all img+avg


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
    mtf_roi_size_x: int = 20.
    mtf_roi_size_y: int = 50.
    mtf_plot: int = 3
    mtf_gaussian: bool = True  # True= use gaussian fit, False = discrete FFT
    mtf_cut_lsf: bool = True
    mtf_cut_lsf_w: float = 3.
    mtf_offset_xy: list[float] = field(default_factory=lambda: [0., 0.])
    mtf_offset_mm: bool = False  # False = pix, True = mm
    mtf_auto_center: bool = False
    mtf_auto_center_type: int = 0  # 0 all edges, 1 = most central edge
    mtf_auto_center_mask_outer: int = 30  # mask outer mm
    mtf_sampling_frequency: float = 0.01  # mm-1 for gaussian
    nps_roi_size: int = 256
    nps_n_sub: int = 5
    nps_smooth_width: float = 0.05  # 1/mm
    nps_sampling_frequency: float = 0.01  # 1/mm
    nps_normalize: int = 0  # normalize curve by 0 = None, 1 = AUC, 2 = large area sign
    nps_plot: int = 0  # default plot 0=pr image, 1=avg, 2=pr image+avg, 3=all img+avg
    stp_roi_size: float = 11.3
    var_roi_size: float = 2.0
    var_percent: int = 90
    dea_percent: int = 100


@dataclass
class ParamSetMammo(ParamSetCommon):
    """Set of parameters regarding Mammo tests."""

    sdn_roi_size: float = 5.
    sdn_roi_dist: float = 10.
    sdn_auto_center: bool = True
    sdn_auto_center_mask_outer: int = 30  # mask outer mm
    hom_roi_size: float = 10.
    hom_variance: bool = True
    hom_roi_size_variance: float = 2.
    hom_mask_max: bool = False
    hom_mask_outer_mm: float = 0.
    hom_ignore_roi_percent: int = 0
    hom_deviating_pixels: float = 20.
    hom_deviating_rois: float = 15.
    rlr_roi_size: float = 5.
    rlr_relative_to_right: bool = True  # if false relative to left
    rlr_x_mm: float = 60.  # distance to left or right border
    gho_roi_size: float = 20.
    gho_relative_to_right: bool = True  # if false relative to left
    gho_table: PositionTable = field(
        default_factory=lambda: PositionTable(
            labels=['ROI_1', 'ROI_2', 'ROI_3'],
            pos_x=[25, 80, 80], pos_y=[30, 30, -30]))
    mtf_roi_size_x: int = 20.
    mtf_roi_size_y: int = 50.
    mtf_plot: int = 3
    mtf_gaussian: bool = True  # True= use gaussian fit, False = discrete FFT
    mtf_cut_lsf: bool = True
    mtf_cut_lsf_w: float = 3.
    mtf_offset_xy: list[float] = field(default_factory=lambda: [0., 0.])
    mtf_offset_mm: bool = False  # False = pix, True = mm
    mtf_auto_center: bool = False
    mtf_auto_center_type: int = 0  # 0 all edges, 1 = most central edge
    mtf_auto_center_mask_outer: int = 30  # mask outer mm
    mtf_sampling_frequency: float = 0.01  # mm-1 for gaussian
    nps_roi_size: int = 256
    nps_n_sub: int = 5
    nps_smooth_width: float = 0.05  # 1/mm
    nps_sampling_frequency: float = 0.01  # 1/mm
    nps_normalize: int = 0  # normalize curve by 0 = None, 1 = AUC, 2 = large area sign
    nps_plot: int = 0  # default plot 0=pr image, 1=avg, 2=pr image+avg, 3=all img+avg


@dataclass
class ParamSetNM(ParamSetCommon):
    """Set of parameters regarding NM tests."""

    uni_ufov_ratio: float = 1.
    uni_cfov_ratio: float = 0.75
    uni_mask_corner: float = 0.0  # mm to ignore in corners
    uni_correct: bool = False
    uni_correct_pos_x: bool = False
    uni_correct_pos_y: bool = False
    uni_lock_radius: bool = False
    uni_radius: float = 0.1
    uni_sum_first: bool = False
    uni_scale_factor: int = 0  # 0 = Auto, 1= no scale, 2... = scale factor
    sni_area_ratio: float = 0.9
    sni_type: int = 0  # 0 as Nelson 2014, 1= grid roi_ratio, 2 grid roi_size, 3 Siemens
    sni_roi_ratio: float = 0.2  # relative to sni_area defined by sni_area_ratio
    sni_roi_size: int = 128  # number of pixels
    sni_roi_outside: int = 0  # alternatives ignore/move
    sni_sampling_frequency: float = 0.01
    sni_ratio_dim: int = 0  # calculate ratio 2d integral (0) or radial profile (1)
    sni_correct: bool = False
    sni_correct_pos_x: bool = False
    sni_correct_pos_y: bool = False
    sni_lock_radius: bool = False
    sni_radius: float = 0.1
    sni_sum_first: bool = False
    sni_eye_filter_c: float = 28.
    sni_channels: bool = False   # use channels
    sni_channels_table: list[float] = field(
        default_factory=lambda: [[0.0, 0.15, 0.5], [0.1, 0.4, 0.5]])
    sni_scale_factor: int = 1  # 1 = no scale, 2 = merge 2x2
    sni_ref_image: str = ''  # file name (without path and extension)
    sni_ref_image_fit: bool = False  # True = curvature fit based on ref image
    sni_alt: int = 0  # alternative (HEADERS) - depende on _type and _channels
    mtf_type: int = 1  # [Point, line (default), Two lines, edge]
    mtf_roi_size_x: float = 50.
    mtf_roi_size_y: float = 50.
    mtf_plot: int = 4  # xyprofiles, line, sorted, LSF, MTF (default)
    mtf_gaussian: bool = True  # True= (gaussian/exp) fit, False = discrete FFT
    mtf_cut_lsf: bool = True
    mtf_cut_lsf_w: int = 3
    mtf_auto_center: bool = False
    mtf_sampling_frequency: float = 0.01  # mm-1 for gaussian
    bar_roi_size: float = 50.
    bar_width_1: float = 6.4
    bar_width_2: float = 4.8
    bar_width_3: float = 4.0
    bar_width_4: float = 3.2
    spe_avg: int = 200
    spe_height: float = 100.
    spe_filter_w: int = 0


@dataclass
class ParamSetSPECT(ParamSetCommon):
    """Set of parameters regarding SPECT tests."""

    mtf_type: int = 1  # 0=point, 1=line source, 2=line source, sliding window
    mtf_roi_size: float = 25.
    mtf_background_width: float = 5.
    mtf_line_tolerance: int = 10
    # ignore slices having max value differing more than % from mean of 3 highest max
    mtf_sliding_window: int = 3  # number of slices to use if line sliding window
    mtf_gaussian: bool = True  # True= gaussian fit, False = discrete FFT
    mtf_plot: int = 2  # default plot 0=xyprofiles, 1=edge, 2=sorted, 3=LSF, 4=MTF
    mtf_cut_lsf: bool = False
    mtf_cut_lsf_w: float = 3.  # lsf_w from halfmax x FWHM
    mtf_cut_lsf_w_fade: float = 1.  # fade out width from lsf_w x FWHM
    mtf_auto_center: bool = True
    mtf_3d: bool = True  # not used yet - assumed 3d for line and circ. edge
    mtf_sampling_frequency: float = 0.01  # mm-1 for gaussian
    rin_sigma_image: float = 0.  # sigma for gaussfilter of image
    rin_sigma_profile: float = 0.  # sigma for gaussfilter of radial profile
    rin_range_start: float = 5.  # mm from center
    rin_range_stop: float = 65.  # mm from center
    rin_subtract_trend: bool = True  # True = subtract trend, False = subtract mean


@dataclass
class ParamSetPET(ParamSetCommon):
    """Set of parameters regarding PET tests."""

    hom_roi_size: float = 10.
    hom_roi_distance: float = 55.
    cro_roi_size: float = 60.
    cro_volume: float = 6283.
    cro_auto_select_slices: bool = True
    cro_percent_slices: float = 75  # % within fwhm of signal profile to include
    rec_roi_size: float = 20.
    rec_type: int = 0  # 0 = RC avg, 1 = RC max, 2 = RC peak, 3,4,5 Bq/mL avg,max,peak
    rec_auto_select_slices: bool = True
    rec_background_full_phantom: bool = False
    rec_percent_slices: int = 90  # % within fwhm of background profile to include
    rec_table: RecTable = field(default_factory=RecTable)
    rec_sphere_diameters: list[float] = field(
        default_factory=lambda: [10., 13., 17., 22., 28., 37.])  # in mm NB increasing
    rec_sphere_dist: float = 57.  # distance center to center of spheres in mm
    rec_sphere_percent: int = 50  # % threshold to evaluate mean from
    rec_plot: int = 0  # 0 = rec max, 1 rec avg, 2 rec peak, 3 z-profile
    rec_earl: int = 1  # tolerances from 0 = None, 1 = EARL1, 2 = EARL2
    rec_background_volume: int = 9500
    mtf_type: int = 2  # 0=point, 1=line source, 2=line source, sliding window
    mtf_roi_size: float = 60.
    mtf_background_width: float = 5.
    mtf_line_tolerance: int = 50
    # ignore slices having max value differing more than % from mean of 3 highest max
    mtf_sliding_window: int = 5  # number of slices to use if line sliding window
    mtf_gaussian: bool = True  # True= gaussian fit, False = discrete FFT
    mtf_plot: int = 2  # default plot 0=xyprofiles, 1=edge, 2=sorted, 3=LSF, 4=MTF
    mtf_cut_lsf: bool = False
    mtf_cut_lsf_w: float = 3.  # lsf_w from halfmax x FWHM
    mtf_cut_lsf_w_fade: float = 1.  # fade out width from lsf_w x FWHM
    mtf_auto_center: bool = True
    mtf_3d: bool = True  # not used yet - assumed 3d for line and circ. edge
    mtf_sampling_frequency: float = 0.01  # mm-1 for gaussian

@dataclass
class ParamSetMR(ParamSetCommon):
    """Set of parameters regarding MR tests."""

    snr_roi_percent: float = 75.
    snr_roi_cut_top: int = 0
    snr_type: int = 0  # 0 from two images, from single image
    snr_background_size: float = 10.  # mm width/height
    snr_background_dist: float = 10.  # mm from image border
    piu_roi_percent: float = 75.
    piu_roi_cut_top: int = 0
    gho_roi_central: float = 80.
    gho_roi_w: float = 40.
    gho_roi_h: float = 10.
    gho_roi_dist: float = 10.
    gho_optimize_center: bool = True
    gho_roi_cut_top: int = 0
    geo_actual_size: float = 190.
    geo_mask_outer: float = 10.
    sli_type: int = 0  # 0 = ramp, 1 = wedge
    sli_tan_a: float = 0.1  # tan(angle), default as ACR phantom
    sli_sigma: int = 0  # gaussian blur of profile
    sli_ramp_length: float = 100.
    sli_background_width: float = 5.
    # sli_search_width: int = 0  # currently not in use
    sli_average_width: int = 0
    sli_median_filter: int = 0  # currently not in use for MR, CT only
    sli_dist_lower: float = -2.5
    sli_dist_upper: float = 2.5
    sli_optimize_center: bool = True
    mtf_roi_size_x: int = 20.
    mtf_roi_size_y: int = 20.
    mtf_plot: int = 3
    mtf_gaussian: bool = True  # True= use gaussian fit, False = discrete FFT
    mtf_cut_lsf: bool = True
    mtf_cut_lsf_w: float = 3.
    mtf_offset_xy: list[float] = field(default_factory=lambda: [0., 0.])
    mtf_offset_mm: bool = False  # False = pix, True = mm
    mtf_auto_center: bool = False
    mtf_auto_center_type: int = 0  # 0 all edges, 1 = most central edge
    mtf_auto_center_mask_outer: int = 10  # mask outer mm
    mtf_sampling_frequency: float = 0.01  # mm-1 for gaussian


@dataclass
class ParamSetSR:
    """Set of paramaters used for modality SR."""

    label: str = ''
    output: QuickTestOutputTemplate = field(
        default_factory=QuickTestOutputTemplate)
    dcm_tagpattern: TagPatternFormat = field(default_factory=TagPatternFormat)


@dataclass
class ParamSet:
    """Collection of parametersets.

    Used for resetting and for reading from IDL version.
    """

    CT: ParamSetCT = field(default_factory=ParamSetCT)
    Xray: ParamSetXray = field(default_factory=ParamSetXray)
    Mammo: ParamSetMammo = field(default_factory=ParamSetMammo)
    NM: ParamSetNM = field(default_factory=ParamSetNM)
    SPECT: ParamSetSPECT = field(default_factory=ParamSetSPECT)
    PET: ParamSetPET = field(default_factory=ParamSetPET)
    MR: ParamSetMR = field(default_factory=ParamSetMR)
    SR: ParamSetSR = field(default_factory=ParamSetSR)


@dataclass
class ParamSetCT_TaskBased:
    """Parameter set used for automated task based analysis."""

    label: str = ''
    output: QuickTestOutputTemplate = field(
        default_factory=QuickTestOutputTemplate)
    dcm_tagpattern: TagPatternFormat = field(default_factory=TagPatternFormat)
    ttf_roi_size: float = 11.
    ttf_gaussian: bool = True  # True= gaussian fit, False = discrete FFT
    ttf_cut_lsf: bool = True
    ttf_cut_lsf_w: float = 3.  # lsf_w from halfmax x FWHM
    ttf_cut_lsf_w_fade: float = 1.  # fade out width from lsf_w x FWHM
    ttf_sampling_frequency: float = 0.01  # mm-1 for gaussian
    ttf_table:  PositionTable = field(default_factory=PositionTable)
    zrange_table: PositionTable = field(
        default_factory=lambda: PositionTable(
            pos_x=[(-1000., 1000.)], pos_y=[(-1000., 1000.)]))
    #  zrange: pos_x = ttf (min, max), pos_y = nps zrange (min, max)
    nps_roi_size: int = 64
    nps_roi_distance_match_ttf: bool = True
    nps_roi_distance: float = 50.  # ignored if _match_ttf is True
    nps_n_sub: int = 65
    nps_smooth_width: float = 0.05  # 1/mm
    # nps_sampling_frequency: float = 0.01  # 1/mm, should match ttf_sampling_frequency
    nps_normalize: int = 0  # normalize curve by 0 = None, 1 = AUC, 2 = large area sign
    nps_plot: int = 0  # default plot 0=pr image, 1=avg, 2=pr image+avg, 3=all img+avg
    dpr_size: float = 10
    dpr_contrast: float = 10
    dpr_designer: bool = True  # False = rect func, True = designer contrast profile
    dpr_power: float = 1.


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

    def __post_init__(self):
        """Add empty image and group_names if not defined."""
        diff = len(self.tests) - len(self.image_names)
        if diff > 0 and len(self.image_names) > 0:
            self.image_names.extend([''] * diff)
        diff = len(self.tests) - len(self.group_names)
        if diff > 0 and len(self.group_names) > 0:
            self.group_names.extend([''] * diff)

    def add_index(self, test_list=[], image_name='', group_name='', index=-1):
        """Add element in each list at given index or append."""
        def add_name(name='', attribute='image_names'):
            """Add name or initiate name list.

            Parameters
            ----------
            name : str.
            attribute : str
                mage_names or group_names
            """
            new_list = getattr(self, attribute)  # get current
            if name != '':
                if len(new_list) == 0:
                    new_list = [''] * len(self.tests)
            if len(new_list) > 0:
                if index == -1 and len(new_list) < len(self.tests):
                    new_list.append('')
                new_list[index] = name
                setattr(self, attribute, new_list)

        if index == -1:
            self.tests.append(test_list)
            if len(self.image_names) > 0 or image_name != '':
                add_name(name=image_name, attribute='image_names')
            if len(self.group_names) > 0 or group_name != '':
                add_name(name=group_name, attribute='group_names')
        else:
            self.tests.insert(index, test_list)
            if len(self.image_names) > 0 or image_name != '':
                add_name(name=image_name, attribute='image_names')
            if len(self.group_names) > 0 or group_name != '':
                add_name(name=group_name, attribute='group_names')

    def remove_indexes(self, ids2remove=[]):
        """Remove element(s) in each list at given index(es)."""
        ids2remove.sort(reverse=True)
        for id_rem in ids2remove:
            self.tests.remove(id_rem)
            if len(self.image_names) > 0:
                self.image_names.remove(id_rem)
            if len(self.group_names) > 0:
                self.group_names.remove(id_rem)


@dataclass
class AutoCommon:
    """Class for holding common settings for automation."""

    import_path: str = ''
    log_mode: str = 'w'
    display_images: bool = True  # ignored if without GUI
    last_import_date: str = ''  # yyyymmdd
    ignore_since: int = -1  # ignore importing images older than X days, -1 if not used
    auto_delete_criterion_attributenames: list[
        str] = field(default_factory=lambda: ['Modality'])
    auto_delete_criterion_values: list[
        str] = field(default_factory=lambda: ['SR'])
    auto_delete_empty_folders: bool = False
    filename_pattern: TagPatternFormat = field(
        default_factory=TagPatternFormat)


@dataclass
class AutoTemplate:
    """Dataclass for keeping information on how to perform automation."""

    label: str = ''
    path_input: str = ''
    path_output: str = ''
    path_warnings: str = ''
    station_name: str = ''
    dicom_crit_attributenames: list = field(default_factory=list)
    dicom_crit_values: list = field(default_factory=list)
    sort_pattern: TagPatternSort = field(default_factory=TagPatternSort)
    paramset_label: str = ''
    quicktemp_label: str = ''
    limits_and_plot_label: str = ''
    archive: bool = True
    delete_if_not_image: bool = False
    delete_if_too_many: bool = False
    active: bool = True
    import_only: bool = False  # if used only during import supplement, not analyse


@dataclass
class AutoVendorTemplate:
    """Dataclass for automation on vendor file analysis."""

    label: str = ''
    path_input: str = ''
    path_output: str = ''
    path_warnings: str = ''
    station_name: str = ''
    limits_and_plot_label: str = ''
    archive: bool = False
    file_type: str = ''
    file_prefix: str = ''  # filter on start of filename (e.g. Mammo QAP)
    file_suffix: str = ''  # starting with .(dot) e.g. '.pdf'
    active: bool = True


@dataclass
class LimitsAndPlotTemplate:
    """Dataclass for the automation output limits and plot settings for dash_app."""

    label: str = ''
    type_vendor: bool = False  # True if used with automation vendor templates
    groups: list = field(default_factory=list)
    # list of headers to group = same limits and plot in same graph
    # ex. [['col a', 'col c'],['col b', 'col d'],['col e'], ['col f']]
    groups_limits: list = field(default_factory=list)  # list of lists
    # eg [min, max] * number of groups, default is [None, None] for each group
    # [textval, textval]textvalue to accept
    groups_ranges: list = field(default_factory=list)  # list of lists
    # min y, max y in display
    # eg [min, max] * number of groups, default is [None, None] for each group = Auto
    groups_hide: list = field(default_factory=list)  # list of bool
    # default is False for each group
    groups_title: list = field(default_factory=list)  # list of str (plot title)

    def __post_init__(self):
        """Add empty list_sort if not defined."""
        if len(self.groups) > len(self.groups_limits):
            diff = len(self.groups) - len(self.groups_limits)
            self.groups_limits.extend([[None, None] for i in range(diff)])
        if len(self.groups) > len(self.groups_ranges):
            diff = len(self.groups) - len(self.groups_ranges)
            self.groups_ranges.extend([[None, None] for i in range(diff)])
        if len(self.groups) > len(self.groups_hide):
            diff = len(self.groups) - len(self.groups_hide)
            self.groups_hide.extend([False for i in range(diff)])
        if len(self.groups) > len(self.groups_title):
            self.groups_title = [', '.join(group) for group in self.groups]

    def add_group(self, group=[], limits=[None, None],
                  ranges=[None, None], hide=False, title='', index=-1):
        """Add group element in each list at given index or append."""
        if index == -1:
            index = len(self.groups)
        self.groups.insert(index, group)
        self.groups_limits.insert(index, limits)
        self.groups_ranges.insert(index, ranges)
        self.groups_hide.insert(index, hide)
        if title == '':
            title = ', '.join(group)
        self.groups_title.insert(index, title)

    def delete_group(self, index):
        """Delete froup element."""
        try:
            self.groups.pop(index)
            self.groups_limits.pop(index)
            self.groups_ranges.pop(index)
            self.groups_hide.pop(index)
            self.groups_title.pop(index)
        except IndexError:
            print(f'Failed deleting group idx {index} from {self}')

    def remove_empty_groups(self):
        """Remove empty groups."""
        empty_group_idxs = [idx for idx, group in enumerate(self.groups)
                            if len(group) == 0]
        if len(empty_group_idxs) > 0:
            empty_group_idxs.reverse()
            for idx in empty_group_idxs:
                self.groups.pop(idx)
                self.groups_limits.pop(idx)
                self.groups_ranges.pop(idx)
                self.groups_hide.pop(idx)
                self.groups_title.pop(idx)

    def move_group(self, old_group_number, direction='up'):
        """Move group to new position (index).

        Parameters
        ----------
        old_group_number : int
            Original group number.
        direction : str
            'up'=lower number or 'down'= higher number (vertical list)
        """
        n_groups = len(self.groups)
        proceed = True
        if old_group_number == n_groups - 1 and direction == 'down':
            proceed = False
        if old_group_number == 0 and direction == 'up':
            proceed = False
        if proceed:
            group = self.groups.pop(old_group_number)
            limits = self.groups_limits.pop(old_group_number)
            ranges = self.groups_ranges.pop(old_group_number)
            hide = self.groups_hide.pop(old_group_number)
            title = self.groups_title.pop(old_group_number)
            new_group_number = (old_group_number - 1 if direction == 'up'
                                else old_group_number + 1)
            self.add_group(group=group, limits=limits, ranges=ranges,
                           hide=hide, title=title, index=new_group_number)

    def find_headers_group_index(self, header):
        """Return group index where header is found."""
        idx = -1
        for i, group in enumerate(self.groups):
            if header in group:
                idx = i
                break
        return idx

    def group_headers(self, headers):
        """Group headers."""
        if len(headers) > 1:
            # test if already grouped
            group_idxs = []
            for header in headers:
                group_idx = self.find_headers_group_index(header)
                if group_idx not in group_idxs:
                    group_idxs.append(group_idx)
            not_included_yet = []
            for idx in group_idxs:
                if len(self.groups[idx]) > 1:
                    for header in self.groups[idx]:
                        if header not in headers:
                            not_included_yet.append(header)
            if len(not_included_yet) > 0:
                pass
                # TODO ask to include rest of other group
                # or remove selected from other group
            group_to = group_idxs[0]
            group_idxs.pop(0)
            headers.pop(0)
            # move to first selected group
            for header_no, idx in enumerate(group_idxs):
                header = headers[header_no]
                self.groups[group_to].append(header)
                self.groups[idx].remove(header)
            self.groups_title[group_to] = ', '.join(headers)
            self.remove_empty_groups()

    def ungroup_headers(self, headers):
        """Ungroup headers."""
        if len(headers) > 0:
            for header in headers:
                group_idx = self.find_headers_group_index(header)
                self.groups[group_idx].remove(header)
                self.add_group(
                    group=[header],
                    limits=self.groups_limits[group_idx],
                    ranges=self.groups_ranges[group_idx],
                    hide=self.groups_hide[group_idx],
                    title=header,
                    index=group_idx+1
                    )
            self.remove_empty_groups()


@dataclass
class DashSettings:
    """Dataclass for dash settings (display of automation results)."""

    label: str = ''
    host: str = '127.0.0.1'
    port: int = 8050
    url_logo: str = ''
    header: str = 'Constancy controls'
    overview_table_headers: list[
        str] = field(default_factory=lambda: [
            'Template', 'Last results', 'Days since', 'Status'])
    days_since_limit: int = 30
    plot_height: int = 200
    colors: list[
        str] = field(default_factory=lambda: [
            '#000000', '#5165d5', '#a914a6', '#7f9955', '#efb412',
            '#97d2d1', '#b3303b'])
    override_css: bool = False  # TODO option to put css in config folder
