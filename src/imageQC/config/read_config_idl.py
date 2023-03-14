# -*- coding: utf-8 -*-
"""
Convert config.dat from IDL version of imageQC to config_classes of py version.

@author: EllenWasbo
"""
#import scipy.io
import numpy as np
import copy

# imageQC block start
import imageQC.config.modified_scipy_io_idl as mod_scipy_io_idl
import imageQC.config.config_classes as cfc
import imageQC.config.config_func as cff
from imageQC.config.iQCconstants_functions import empty_template_dict
from imageQC.config.iQCconstants import QUICKTEST_OPTIONS
# imageQC block end

# attribute name pair where IDL attribute name is not only uppercase of PY attributename
idl_py_attr = [
    ['InstitutionName', 'INSTITUTION'],
    ['ManufacturerModelName', 'MODELNAME'],
    ['SoftwareVersions', 'SWVERSION'],
    ['StudyDescription', 'STUDYDESCR'],
    ['SeriesDescription', 'SERIESNAME'],
    ['AcquisitionDate', 'ACQDATE'],
    ['AcquisitionTime', 'ACQTIME'],
    ['AcquisitionNumber', 'ACQNMB'],
    ['SeriesNumber', 'SERIESNMB'],
    ['InstanceNumber', 'IMGNO'],
    ['ExposureTime', 'EXPTIME'],
    ['mAs', 'MAS'],
    ['ExposureModulationType', 'EXMODTYPE'],
    ['ExposureControlMode', 'EXMODTYPE'],
    ['SID', 'SDD'],
    ['SliceThickness', 'SLICETHICK'],
    ['FocalSpots', 'FOCALSPOTSZ'],
    ['FilterType', 'FILTERADDON'],
    ['SpiralPitchFactor', 'PITCH'],
    ['TotalCollimationWidth', 'COLL'],
    ['ExposureIndex', 'EI'],
    ['ReconstructionDiameter', 'REKONFOV'],
    ['SliceLocation', 'ZPOS'],
    ['DataCollectionDiameter', 'FOV'],
    ['ConvolutionKernel', 'KERNEL'],
    ['ReconstructionMethod', 'RECONMETHOD'],
    ['AttenuationCorrectionMethod', 'ATTCORRMETHOD'],
    ['ScatterCorrectionMethod', 'SCACORRMETHOD'],
    ['CollimatorType', 'COLLTYPE'],
    ['NumberOfEnergyWindows', 'NEWINDOWS'],
    ['EnergyWindowName', 'EWINDOWNAME'],
    ['ActualFrameDuration', 'ACQFRAMEDURATION'],
    ['AcquisitionTerminationCondition', 'ACQTERMINATIONCOND'],
    ['Radiopharmaceutical', 'RADIOPHARMACA'],
    ['RadionuclideTotalDose', 'ADMDOSE'],
    ['RadiopharmaceuticalStartTime', 'ADMDOSETIME'],
    ['ScatterFractionFactor', 'SCATTERFRAC'],
    ['RadialPosition', 'RADIUS1'],
    ['AngularViewVector', 'ANGLE'],
    ['RepetitionTime', 'TR'],
    ['EchoTime', 'TE'],
    ['NumberOfAverages', 'NSA'],
    ['FlipAngle', 'FLIPANG'],
    ['SpacingBetweenSlices', 'SPACESLICE'],
    ['MRAcquisitionType', 'MRACQTYPE'],
    ['ScanningSequence', 'MRSCANSEQ'],
    ['SequenceVariant', 'MRSEQVARIANT'],
    ['ReceiveCoilName', 'RECCOILNAME'],
    ['TransmitCoilName', 'TRACOILNAME'],
    ['ImagingFrequency', 'IMGFREQ'],
    ['Columns', 'IMAGESIZE'],
    ['PixelSpacing', 'PIX'],
    ['PresentationIntentType', 'PRESTYPE'],
    ['StudyDate', 'STUDYDATETIME'],
    ['AcquisitionDate', 'IMGDATE'],
    ['SeriesInstanceUID', 'SERIESUID'],
    ['NumberOfFrames', 'NFRAMES'],
    ['WindowWidth', 'WWIDTH'],
    ['WindowCenter', 'WCENTER'],
    ]
idl_attr = [attr[1] for attr in idl_py_attr]
py_attr = [attr[0] for attr in idl_py_attr]


def try_decode(bytestr):
    """Decode and handle attributeError when empty input string."""
    try:
        return_string = bytestr.decode('utf-8')
    except AttributeError:
        return_string = ''

    return return_string


def mod_py(m):
    """Get modality string of py version either by index or idl string."""
    if isinstance(m, str):
        old_mod = ['CT', 'XRAY', 'NM', 'SPECT', 'PET', 'MR']
        idx = old_mod.index(m)
    else:
        idx = m
    new_mod = ['CT', 'Xray', 'NM', 'SPECT', 'PET', 'MR']
    return new_mod[idx]


class ConfigIdl2Py():
    """Class to convert IDL config.dat to config collection."""

    def __init__(self, fname, tag_infos):

        self.tag_infos = tag_infos
        self.tag_infos_labels = [
            tag_info.attribute_name for tag_info in self.tag_infos]
        self.tag_infos_tags = [tag_info.tag for tag_info in self.tag_infos]
        self.tag_infos_new = []  # holding new added tag infos
        self.paramsets = empty_template_dict(QUICKTEST_OPTIONS, dummy=[])
        self.quicktest_templates = {}
        self.quicktest_output_templates = {}
        filename_pattern = cfc.TagPatternFormat(
            list_tags=[
                'Modality', 'StationName', 'PatientID', 'AcquisitionDate',
                'AcquisitionTime',
                'SeriesDescription', 'ProtocolName', 'SeriesNumber', 'InstanceNumber'],
            list_format=['', '', '', '', '|:06|', '', '', '', '']
            )
        self.auto_common = cfc.AutoCommon(
            filename_pattern=filename_pattern,
            auto_delete_criterion_attributenames=[],
            auto_delete_criterion_values=[])
        self.auto_templates = {}
        self.auto_vendor_templates = {}
        self.dcm_additionals = empty_template_dict(
            QUICKTEST_OPTIONS, dummy=cfc.TagPatternFormat())
        self.rename_patterns = {}
        self.errmsg = []

        self.read_config_dat(fname)

    def read_config_dat(self, fname):
        """Read config.dat from IDL version of imageQC.

        Parameters
        ----------
        fname : str
            full path of config.dat
        """
        d = {}
        try:
            d = mod_scipy_io_idl.readsav(fname)
        except ValueError:
            self.errmsg.append(
                ('Error reading .dat file. Report the issue at '
                 'https://github.com/EllenWasbo/imageQCpy/issues.'))
        if 'configs' in d.keys():
            # rename temps
            if 'renametemp' in d.keys():
                if str(d['renametemp'].dtype) != 'int16':
                    self.convert_renametemp(self.as_dict(d['renametemp']))

            # qt_temps
            if 'quicktemp' in d.keys():
                if str(d['quicktemp'].dtype) != 'int16':
                    self.convert_quicktemp(self.as_dict(d['quicktemp']))

            # qt_out_temps
            if 'quicktout' in d.keys():
                self.convert_quicktout(self.as_dict(d['quicktout']))

            # common_config + params
            self.convert_configs(self.as_dict(d['configs']))

            # auto_temps
            if 'loadtemp' in d.keys():
                if str(d['loadtemp'].dtype) != 'int16':
                    self.convert_loadtemp(self.as_dict(d['loadtemp']))

        else:
            self.errmsg.append(
                'Could not find the expected content in the selected file.')

    def as_dict(self, rec):
        """Turn a numpy recarray record into a dict."""
        return {name: rec[name] for name in rec.dtype.names}

    def convert_configs(self, c):
        """Convert configS structure into objects.

        Parameters
        ----------
        c : dict
            configS in config.dat converted from IDL
        """
        cNames = [key for key, val in c.items()]

        # common_config in first dict
        cc = c[cNames[0]][0]
        ccNames = cc.dtype.names

        if 'AUTOCOMMON' in ccNames:
            ac = cc.AUTOCOMMON
            ac = ac[0]
            acNames = ac.dtype.names  # [key for key, val in ac.items()]
            if 'AUTOIMPORTPATH' in acNames:
                self.auto_common.import_path = try_decode(ac.AUTOIMPORTPATH[0])
            if 'AUTOCONTINUE' in acNames:
                self.auto_common.auto_continue = bool(ac.AUTOCONTINUE[0])
            if 'IGNORESINCE' in acNames:
                self.auto_common.ignore_since = ac.IGNORESINCE[0]

        empty_ParamSet = cfc.ParamSet()
        empty_ParamSet.CT.dcm_tagpattern = cfc.TagPatternFormat(
            label='default',
            list_tags=['KVP', 'mAs', 'CTDIvol', 'SoftwareVersions'],
            list_format=['|:.0f|', '|:.1f|', '|:.2f|', '']
            )
        empty_ParamSet.Xray.dcm_tagpattern = cfc.TagPatternFormat(
            label='default',
            list_tags=['KVP', 'mAs', 'ExposureIndex', 'DAP', 'SID', 'DetectorID'],
            list_format=['|:.1f|', '|:.2f|', '|:.1f|', '|:.3f|', '|:.1f|', '']
            )
        empty_ParamSet.NM.dcm_tagpattern = cfc.TagPatternFormat(
            label='default',
            list_tags=['CountsAccumulated', 'ActualFrameDuration'],
            list_format=['|:.0f|', '|:.0f|']
            )
        empty_ParamSet.MR.dcm_tagpattern = cfc.TagPatternFormat(
            label='default',
            list_tags=['ImagingFrequency', 'ReceiveCoilName', 'TransmitCoilName']
            )
        self.errmsg.extend(
            [
                '----------------',
                'Parametersets in IDL version included parameters for all modalities. '
                'The relavant parameters for each modality are now saved as '
                'separate modality specific templates.',
                '----------------',
                'Changes to filtering of CT test on ring artifacts. Please reestablish '
                'tolerance values.'
            ]
            )

        for i in range(1, len(c)):
            paramset = copy.deepcopy(empty_ParamSet)

            output = cfc.QuickTestOutputTemplate()
            cp = c[cNames[i]][0]

            cp_dict = self.as_dict(cp)
            param_names = [key for key, val in cp_dict.items()]

            if 'COPYHEADER' in param_names:
                output.include_header = bool(cp.COPYHEADER[0])
            if 'TRANSPOSETABLE' in param_names:
                output.transpose_table = bool(cp.TRANSPOSETABLE[0])
            if 'DECIMARK' in param_names:
                output.decimal_mark = cp.DECIMARK[0].decode('UTF-8')
            if 'INCLUDEFILENAME' in param_names:
                output.include_filename = bool(cp.INCLUDEFILENAME[0])

            for m in range(6):
                this_set = getattr(paramset, mod_py(m))
                this_set.label = cNames[i]
                this_set.output = copy.deepcopy(output)
            #todo delete
            '''
            paramset.CT.label = cNames[i]
            paramset.Xray.label = cNames[i]
            paramset.NM.label = cNames[i]
            paramset.SPECT.label = cNames[i]
            paramset.PET.label = cNames[i]
            paramset.MR.label = cNames[i]
            output = cfc.QuickTestOutputTemplate()
            
            paramset.CT.output = output
            paramset.Xray.output = output
            paramset.NM.output = output
            paramset.SPECT.output = output
            paramset.PET.output = output
            paramset.MR.output = output
            '''

            if 'AUTOIMPORTPATH' in param_names and i == 1:
                if isinstance(cp.AUTOIMPORTPATH[0], bytes):
                    self.auto_common.import_path = try_decode(
                        cp.AUTOIMPORTPATH[0])

            if 'LINTAB' in param_names:
                paramset.CT.ctn_table.materials = \
                    [elem.decode('UTF-8')
                     for elem in cp.LINTAB[0].MATERIALS[0]]
                paramset.CT.ctn_table.relative_mass_density = \
                    [round(elem.item(), 3) for elem in cp.LINTAB[0].RELMASSD[0]]
                paramset.CT.ctn_table.pos_x = \
                    [elem.item() for elem in cp.LINTAB[0].POSX[0]]
                paramset.CT.ctn_table.pos_y = \
                    [-elem.item() for elem in cp.LINTAB[0].POSY[0]]

            if 'HOMOGROISZ' in param_names:
                paramset.CT.hom_roi_size = cp.HOMOGROISZ[0]
            if 'HOMOGROIDIST' in param_names:
                paramset.CT.hom_roi_distance = cp.HOMOGROIDIST[0]
            if 'HOMOGROIROT' in param_names:
                paramset.CT.hom_roi_rotation = cp.HOMOGROIROT[0]
            if 'NOISEROISZ' in param_names:
                paramset.CT.noi_roi_size = cp.NOISEROISZ[0]
            if 'HUWATERROISZ' in param_names:
                paramset.CT.huw_roi_size = cp.HUWATERROISZ[0]
            if 'TYPEROI' in param_names:
                paramset.CT.roi_type = cp.TYPEROI[0]
            if 'ROIRAD' in param_names:
                paramset.CT.roi_radius = cp.ROIRAD[0]
            if 'ROIX' in param_names:
                paramset.CT.roi_x = cp.ROIX[0]
            if 'ROIY' in param_names:
                paramset.CT.roi_y = cp.ROIY[0]
            if 'ROIA' in param_names:
                paramset.CT.roi_a = cp.ROIA[0]
            if 'OFFXYROI' in param_names:
                paramset.CT.roi_offset_xy = list(cp.OFFXYROI[0].astype('float64'))
            if 'OFFXYROI_UNIT' in param_names:
                paramset.CT.roi_offset_mm = bool(cp.OFFXYROI_UNIT[0])
            if 'MTFTYPE' in param_names:
                paramset.CT.mtf_type = cp.MTFTYPE[0]
            if 'MTFROISZ' in param_names:
                paramset.CT.mtf_roi_size = cp.MTFROISZ[0]
            if 'PLOTMTF' in param_names:
                paramset.CT.mtf_plot = cp.PLOTMTF[0]
            if 'TABLEMTF' in param_names:
                paramset.CT.mtf_gaussian = bool(cp.TABLEMTF[0])
            if 'CYCLMTF' in param_names:
                paramset.CT.mtf_cy_pr_mm = bool(cp.CYCLMTF[0])
            if 'CUTLSF' in param_names:
                paramset.CT.mtf_cut_lsf = bool(cp.CUTLSF[0])
            if 'CUTLSF1' in param_names:
                paramset.CT.mtf_cut_lsf_w = float(cp.CUTLSF1[0])
            if 'CUTLSF2' in param_names:
                paramset.CT.mtf_cut_lsf_w_fade = float(cp.CUTLSF2[0])
            if 'OFFXYMTF' in param_names:
                paramset.CT.mtf_offset_xy = list(cp.OFFXYMTF[0].astype('float64'))
            if 'OFFXYMTF_UNIT' in param_names:
                paramset.CT.mtf_offset_mm = bool(cp.OFFXYMTF_UNIT[0])
            if 'SEARCHMAXMTF_ROI' in param_names:
                paramset.CT.mtf_search_max = bool(cp.SEARCHMAXMTF_ROI[0])
            if 'LINROIRAD' in param_names:
                paramset.CT.ctn_roi_size = cp.LINROIRAD[0]
            if 'LINROIRADS' in param_names:
                paramset.CT.ctn_search_size = cp.LINROIRADS[0]
            if 'RAMPDIST' in param_names:
                paramset.CT.sli_ramp_distance = cp.RAMPDIST[0]
            if 'RAMPLEN' in param_names:
                paramset.CT.sli_ramp_length = cp.RAMPLEN[0]
            if 'RAMPBACKG' in param_names:
                paramset.CT.sli_background_width = cp.RAMPBACKG[0]
            if 'RAMPSEARCH' in param_names:
                paramset.CT.sli_search_width = cp.RAMPSEARCH[0]
            if 'RAMPAVG' in param_names:
                paramset.CT.sli_average_width = cp.RAMPAVG[0]
            if 'RAMPTYPE' in param_names:
                paramset.CT.sli_type = cp.RAMPTYPE[0]
            if 'RINGSTOP' in param_names:
                rin_range = list(cp.RINGSTOP[0])
                paramset.CT.rin_range_start = float(rin_range[0])
                paramset.CT.rin_range_stop = float(rin_range[1])
            if 'RINGARTTREND' in param_names:
                paramset.CT.rin_subtract_trend = bool(cp.RINGARTTREND[0])
            if 'NPSROISZ' in param_names:
                paramset.CT.nps_roi_size = cp.NPSROISZ[0]
            if 'NPSROIDIST' in param_names:
                paramset.CT.nps_roi_distance = cp.NPSROIDIST[0]
            if 'NPSSUBNN' in param_names:
                paramset.CT.nps_n_sub = cp.NPSSUBNN[0]

            if 'HOMOGROISZX' in param_names:
                paramset.Xray.hom_roi_size = cp.HOMOGROISZX[0]
            if 'HOMOGROIDISTX' in param_names:
                paramset.Xray.hom_roi_distance = cp.HOMOGROIDISTX[0]
            if 'HOMOGROIROTX' in param_names:
                paramset.Xray.hom_roi_rotation = cp.HOMOGROIROTX[0]
            if 'ALTHOMOGX' in param_names:
                paramset.Xray.hom_tab_alt = cp.ALTHOMOGX[0]
            if 'NOISEXPERCENT' in param_names:
                paramset.Xray.noi_percent = cp.NOISEXPERCENT[0]
            if 'TYPEROIX' in param_names:
                paramset.Xray.roi_type = cp.TYPEROIX[0]
            if 'ROIXRAD' in param_names:
                paramset.Xray.roi_radius = cp.ROIXRAD[0]
            if 'ROIXX' in param_names:
                paramset.Xray.roi_x = cp.ROIXX[0]
            if 'ROIXY' in param_names:
                paramset.Xray.roi_y = cp.ROIXY[0]
            if 'ROIXA' in param_names:
                paramset.Xray.roi_a = cp.ROIXA[0]
            if 'OFFXYROIX' in param_names:
                paramset.Xray.roi_offset_xy = list(cp.OFFXYROIX[0].astype('float64'))
            if 'OFFXYROIX_UNIT' in param_names:
                paramset.Xray.roi_offset_mm = bool(cp.OFFXYROIX_UNIT[0])
            if 'MTFROISZX' in param_names:
                paramset.Xray.mtf_roi_size_x = cp.MTFROISZX[0][0]
                paramset.Xray.mtf_roi_size_y = cp.MTFROISZX[0][1]
            if 'PLOTMTFX' in param_names:
                paramset.Xray.mtf_plot = cp.PLOTMTFX[0]
            if 'TABLEMTFX' in param_names:
                paramset.Xray.mtf_gaussian = bool(cp.TABLEMTFX[0])
            if 'CUTLSFX' in param_names:
                paramset.Xray.mtf_cut_lsf = bool(cp.CUTLSFX[0])
            if 'CUTLSFX1' in param_names:
                paramset.Xray.mtf_cut_lsf_w = float(cp.CUTLSFX1[0])
            if 'OFFXYMTF_X' in param_names:
                paramset.Xray.mtf_offset_xy = list(cp.OFFXYMTF_X[0].astype('float64'))
            if 'OFFXYMTF_X_UNIT' in param_names:
                paramset.Xray.mtf_offset_mm = bool(cp.OFFXYMTF_X_UNIT[0])
            if 'NPSROISZX' in param_names:
                paramset.Xray.nps_roi_size = cp.NPSROISZX[0]
            if 'NPSSUBSZX' in param_names:
                paramset.Xray.nps_n_sub = cp.NPSSUBSZX[0]
            if 'NPSAVG' in param_names:
                paramset.Xray.nps_plot_average = cp.NPSAVG[0]
            if 'STPROISZ' in param_names:
                paramset.Xray.stp_roi_size = cp.STPROISZ[0]

            if 'UNIFAREARATIO' in param_names:
                paramset.NM.uni_ufov_ratio = cp.UNIFAREARATIO[0]
            if 'UNIFCORR' in param_names:
                paramset.NM.uni_correct = bool(cp.UNIFCORR[0])
            if 'UNIFCORRRAD' in param_names:
                paramset.NM.uni_radius = float(cp.UNIFCORRRAD[0])
                if paramset.NM.uni_radius > 0:
                    paramset.NM.uni_lock_radius = True
            if 'UNIFCORRPOS' in param_names:
                uni_correct_pos = list(cp.UNIFCORRPOS[0])
                paramset.NM.uni_correct_pos_x = not bool(uni_correct_pos[0])
                paramset.NM.uni_correct_pos_y = not bool(uni_correct_pos[1])
            if 'SNIAREARATIO' in param_names:
                paramset.NM.sni_area_ratio = cp.SNIAREARATIO[0]
            if 'SNICORR' in param_names:
                paramset.NM.sni_correct = bool(cp.SNICORR[0])
            if 'SNIFCORRRAD' in param_names:
                paramset.NM.sni_radius = float(cp.SNIFCORRRAD[0])
                if paramset.NM.sni_radius > 0:
                    paramset.NM.sni_lock_radius = True
            if 'SNIFCORRPOS' in param_names:
                sni_correct_pos = list(cp.SNICORRPOS[0])
                paramset.NM.sni_correct_pos_x = not bool(sni_correct_pos[0])
                paramset.NM.sni_correct_pos_y = not bool(sni_correct_pos[1])
            if 'SNI_FCD' in param_names:
                fcr = list(cp.SNI_FCD[0])
                paramset.NM.sni_eye_filter_f = fcr[0]
                paramset.NM.sni_eye_filter_c = fcr[1]
                paramset.NM.sni_eye_filter_r = fcr[2]
            # if 'PLOTSNI' in param_names:
            #     paramset.NM.sni_plot = cp.PLOTSNI[0]
            if 'MTFTYPENM' in param_names:
                paramset.NM.mtf_type = cp.MTFTYPENM[0]
            if 'MTFROISZNM' in param_names:
                sz = list(cp.MTFROISZNM[0])
                paramset.NM.mtf_roi_size_x = float(sz[0])
                paramset.NM.mtf_roi_size_y = float(sz[1])
            if 'PLOTMTFNM' in param_names:
                paramset.NM.mtf_plot = cp.PLOTMTFNM[0]
            if 'BARROISZ' in param_names:
                paramset.NM.bar_roi_size = cp.BARROISZ[0]
            if 'BARWIDTHS' in param_names:
                w = list(cp.BARWIDTHS[0])
                paramset.NM.bar_width_1 = float(w[0])
                paramset.NM.bar_width_2 = float(w[1])
                paramset.NM.bar_width_3 = float(w[2])
                paramset.NM.bar_width_4 = float(w[3])
            if 'SCANSPEEDAVG' in param_names:
                paramset.NM.spe_avg = cp.SCANSPEEDAVG[0]
            if 'SCANSPEEDHEIGHT' in param_names:
                paramset.NM.spe_height = cp.SCANSPEEDHEIGHT[0]
            if 'SCANSPEEDFILTW' in param_names:
                paramset.NM.spe_filter_w = cp.SCANSPEEDFILTW[0]

            if 'MTFTYPESPECT' in param_names:
                paramset.SPECT.mtf_type = cp.MTFTYPESPECT[0]
                if paramset.SPECT.mtf_type == 3:
                    paramset.SPECT.mtf_type = 2  # removed option 2 for edge
            if 'MTFROISZSPECT' in param_names:
                paramset.SPECT.mtf_roi_size = cp.MTFROISZSPECT[0]
            if 'PLOTMTFSPECT' in param_names:
                paramset.SPECT.mtf_plot = cp.PLOTMTFSPECT[0]
            if 'MTF3DSPECT' in param_names:
                paramset.SPECT.mtf_3d = bool(cp.MTF3DSPECT[0])
            if 'CONTRASTRAD1' in param_names:
                paramset.SPECT.con_roi_size = cp.CONTRASTRAD1[0]
            if 'CONTRASTRAD2' in param_names:
                paramset.SPECT.con_roi_dist = cp.CONTRASTRAD2[0]

            if 'HOMOGROISZPET' in param_names:
                paramset.PET.hom_roi_size = cp.HOMOGROISZPET[0]
            if 'HOMOGROIDISTPET' in param_names:
                paramset.PET.hom_roi_distance = cp.HOMOGROIDISTPET[0]
            if 'CROSSROISZ' in param_names:
                paramset.PET.cro_roi_size = cp.CROSSROISZ[0]
            if 'CROSSVOL' in param_names:
                paramset.PET.cro_volume = cp.CROSSVOL[0]

            if 'TYPEROIMR' in param_names:
                paramset.MR.roi_type = cp.TYPEROIMR[0]
            if 'ROIMRRAD' in param_names:
                paramset.MR.roi_radius = cp.ROIMRRAD[0]
            if 'ROIMRX' in param_names:
                paramset.MR.roi_x = cp.ROIMRX[0]
            if 'ROIMRY' in param_names:
                paramset.MR.roi_y = cp.ROIMRY[0]
            if 'ROIMRA' in param_names:
                paramset.MR.roi_a = cp.ROIMRA[0]
            if 'OFFXYROIMR' in param_names:
                paramset.MR.roi_offset_xy = list(cp.OFFXYROIMR[0].astype('float64'))
            if 'OFFXYROIMR_UNIT' in param_names:
                paramset.MR.roi_offset_mm = bool(cp.OFFXYROIMR_UNIT[0])
            if 'SNR_MR_ROI' in param_names:
                paramset.MR.snr_roi_percent = cp.SNR_MR_ROI[0]
            if 'SNR_MR_ROICUT' in param_names:
                paramset.MR.snr_roi_cut_top = cp.SNR_MR_ROICUT[0]
            if 'PIU_MR_ROI' in param_names:
                paramset.MR.piu_roi_percent = cp.PIU_MR_ROI[0]
            if 'PIU_MR_ROICUT' in param_names:
                paramset.MR.piu_roi_cut_top = cp.PIU_MR_ROICUT[0]
            if 'GHOST_MR_ROI' in param_names:
                vals = list(cp.GHOST_MR_ROI[0])
                paramset.MR.gho_roi_central = vals[0]
                paramset.MR.gho_roi_w = vals[1]
                paramset.MR.gho_roi_h = vals[2]
                paramset.MR.gho_roi_dist = vals[3]
                paramset.MR.gho_optimize_center = bool(vals[4])
            if 'GHOST_MR_ROICUT' in param_names:
                paramset.MR.gho_roi_cut_top = cp.GHOST_MR_ROICUT[0]
            if 'GD_MR_ACT' in param_names:
                paramset.MR.geo_actual_size = cp.GD_MR_ACT[0]
            if 'SLICE_MR_ROI' in param_names:
                vals = list(cp.SLICE_MR_ROI[0])
                paramset.MR.sli_tan_a = vals[0]
                paramset.MR.sli_ramp_length = vals[1]
                paramset.MR.sli_search_width = vals[2] // 2
                paramset.MR.sli_dist_lower = vals[3]
                paramset.MR.sli_dist_upper = vals[4]
                paramset.MR.sli_optimize_center = bool(vals[5])
            if 'PIU_MR_ROICUT' in param_names:
                paramset.MR.piu_roi_cut_top = cp.PIU_MR_ROICUT[0]

            vals = [elem.decode('UTF-8') for elem in cp.QTOUTTEMPS[0]]
            for m, val in enumerate(vals):
                if m != 3:  # Never implemented SPECT output template in idl
                    paramsetXX = getattr(paramset, mod_py(m))
                    paramsetXX.output.tests = (
                        self.quicktest_output_templates[mod_py(m)][val])
                    lbls_dcm_add = [
                        pattern.label for pattern in self.dcm_additionals[mod_py(m)]]
                    if val in lbls_dcm_add:
                        idx = lbls_dcm_add.index(val)
                        add_pattern = self.dcm_additionals[mod_py(m)][idx]
                        n_tags_before = len(paramsetXX.dcm_tagpattern.list_tags)
                        paramsetXX.dcm_tagpattern.list_tags.extend(
                            add_pattern.list_tags)
                        paramsetXX.dcm_tagpattern.list_format.extend(
                            add_pattern.list_format)
                        if 'DCM' in [*paramsetXX.output.tests]:
                            for attrno, attr in enumerate(add_pattern.list_tags):
                                paramsetXX.output.tests['DCM'].append(
                                    cfc.QuickTestOutputSub(
                                        label=attr,
                                        columns=[attrno + n_tags_before]
                                        )
                                    )
                    if cc.DEFCONFIGNO[0] == i:
                        self.paramsets[mod_py(m)].insert(0, paramsetXX)
                    else:
                        self.paramsets[mod_py(m)].append(paramsetXX)

        for mod in self.paramsets:
            if [] in self.paramsets[mod]:
                self.paramsets[mod].remove([])

    def convert_quicktemp(self, c):
        """Convert quickTemp struct in config.dat to python dict and update.

        Parameters
        ----------
        c : dict
            quickT (quicktest templates) in config.dat converted from IDL
        """
        qt = {}
        for m, mv in c.items():
            if mv[0].dtype != 'int16':  # -1 if undefined
                temps = self.as_dict(mv[0])
                tempsThis = []
                for tmp, tmpv in temps.items():
                    if tmpv[0].dtype != 'int16':
                        if len(tmpv[0].shape) == 2:
                            tempThis = self.convert_multiMark2py(m, tmpv[0])
                        else:  # len(shape) == 1, only one image for testing
                            tempThis = self.convert_multiMark2py(m, [tmpv[0]])
                        tempThis.label = tmp
                        tempsThis.append(tempThis)
                qt[mod_py(m)] = tempsThis
        self.quicktest_templates = qt

        return qt

    def convert_multiMark2py(self, modality, multimark):
        """Convert multiMark array from idl to dict - test indicators as key.

        Parameters
        ----------
        modality : str
            modality string as used in imageQC
        multimark : np.array
            one row for each image, one column for each test
            1 indicating that the test should be performed on the image

        Returns
        -------
        temp : QuickTestTemplate
        """
        temp = cfc.QuickTestTemplate()
        for row in multimark:
            tests_this = []
            if row.any():
                testnos = np.where(row)[0].tolist()
                testcodes = self.convert_quicktest_idl2py(modality, testnos)
                tests_this = testcodes
            temp.add_index(test_list=tests_this)

        return temp

    def convert_quicktest_idl2py(self, modality, test_nums=[], test_names=[]):
        """Convert test numbers for QuickTest in IDL version to strings.

        Parameters
        ----------
        modality : TYPE
            modality string to use in the convertion (modality as in imageQC)
        test_nums : list of int
                list with test integer indicatiors
        test_name : str
                idl test name

        Returns
        -------
        list_out : list of str
            list with 3-character test indicators
            empty string if conversion failed

        """
        # order of tests in ImageQC v2
        test_strings = [  # CT
            'Hom', 'Noi', 'Sli', 'MTF', 'CTn', 'HUw', 'DCM', 'ROI', 'Rin']
        if modality.upper() == 'XRAY':
            test_strings = ['STP', 'Hom', 'Noi', 'DCM', 'MTF', 'ROI']
        elif modality == 'NM':
            test_strings = ['Uni', 'SNI', 'DCM', 'Bar']
        elif modality == 'PET':
            test_strings = ['Hom']
        elif modality == 'MR':
            test_strings = ['DCM', 'SNR', 'PIU', 'Gho', 'Geo', 'Sli', 'ROI']

        failed = False
        if len(test_nums) > 0:
            list_out = []
            for num in test_nums:
                try:
                    list_out.append(test_strings[num])
                except IndexError:
                    pass

        if len(test_names) > 0:
            old_test_strings = [
                'HOMOG', 'NOISE', 'SLICETHICK', 'MTF', 'CTLIN',
                'HUWATER', 'EXP', 'ROI', 'RING']
            if modality.upper() == 'XRAY':
                old_test_strings = [
                    'STP', 'HOMOG', 'NOISE', 'EXP', 'MTF', 'ROI']
            elif modality == 'NM':
                old_test_strings = ['UNIF', 'SNI', 'ACQ', 'BAR']
            elif modality == 'PET':
                old_test_strings = ['HOMOG']
            elif modality == 'MR':
                old_test_strings = [
                    'DCM', 'SNR', 'PIU', 'GHOST', 'GEOMDIST',
                    'SLICETHICK', 'ROI']

            list_out = test_names.copy()
            for old_name in test_names:
                if old_name in old_test_strings:
                    id_test = old_test_strings.index(old_name)
                    list_out = list(map(lambda x: x.replace(
                        old_name, test_strings[id_test]), list_out))
                else:
                    failed = True
                    print(f'Failed to convert test type {old_name}')

        if failed:
            list_out = ['']

        return list_out

    def convert_quicktout(self, c):
        """Convert quickTout struct in config.dat to python dict.

        Update test names

        Parameters
        ----------
        c : dict
            quickTout (quicktest output temps) in config.dat converted from IDL
        """
        def calc_str(no):
            strs = ['=', 'min','max','mean','stdev','max abs']
            try:
                outstr = strs[no]
            except IndexError:
                outstr = '='
            return outstr

        qto = {}
        for m, mv in c.items():
            temps = self.as_dict(mv[0])
            tempsThis = {}
            for tmp, tmpv in temps.items():
                tests = self.as_dict(tmpv[0])
                testsThis = {}
                for tst, tstv in tests.items():
                    upd_test_key = self.convert_quicktest_idl2py(
                        m, test_names=[tst])
                    if len(upd_test_key) > 0:
                        if tstv[0] is not None:  # modified scipy.io._idl
                            if tstv[0].dtype != 'int16':
                                outputs = self.as_dict(tstv[0])
                                outputsThisTest = []
                                for key, v in outputs.items():
                                    if int(v[0].ALT[0]) == -1:
                                        try:
                                            dict_tags = self.as_dict(v[0].TAGS[0])
                                            dict_formats = self.as_dict(v[0].TAGFORMATS[0])
                                            tag_pattern = self.convert_tag_pattern(
                                                dict_tags, dict_formats)
                                            tag_pattern.label = tmp
                                            self.dcm_additionals[mod_py(m)].append(
                                                tag_pattern)
                                        except (KeyError, AttributeError):
                                            pass
                                    else:
                                        if v[0].COLUMNS[0].size == 1:
                                            col = [int(v[0].COLUMNS[0])]
                                        else:
                                            col = [int(x)
                                                   for x in list(v[0].COLUMNS[0])]
                                        outputsThisTest.append(
                                            cfc.QuickTestOutputSub(
                                                label=key,
                                                alternative=int(v[0].ALT[0]),
                                                columns=col,
                                                calculation=calc_str(int(v[0].CALC[0])),
                                                per_group=bool(v[0].PER_SERIES[0])
                                                )
                                            )
                                testsThis[upd_test_key[0]] = outputsThisTest
                            else:
                                testsThis[upd_test_key[0]] = [cfc.QuickTestOutputSub()]
                    else:
                        self.errmsg.append(
                            f'Test name {tst} for modality {m} no longer'
                            f'available. Ignored from output templates.')
                tempsThis[tmp] = testsThis
            qto[mod_py(m)] = tempsThis

        self.quicktest_output_templates = qto

    def convert_loadtemp(self, c):
        """Convert loadTemp struct in config.dat to python dict and updated.

        Parameters
        ----------
        c : dict
            loadtemp (automation templates) in config.dat converted from IDL
        """
        def get_filetype(vendor_alt):
            if len(vendor_alt) > 2:
                filetype = ''
                if vendor_alt[:3] == 'PDF':
                    filetype = 'pdf'
                elif vendor_alt[:3] == 'XML':
                    filetype = 'xml'
                else:
                    filetype = 'txt'  # GE_QAP

            return filetype
        auto_temps_mod = {}
        auto_temps_vendor_mod = {}
        n_unknown = 0
        for m, mv in c.items():
            if mv[0].dtype != 'int16':  # -1 if undefined
                temps = self.as_dict(mv[0])
                tempsThis = []
                tempsThis_vendor = []
                for tmp, tmpv in temps.items():
                    if tmpv[0].dtype != 'int16':
                        autos = self.as_dict(tmpv[0])
                        keys = [*autos]
                        vendor_alt = ''
                        if 'ALTERNATIVE' in keys:
                            if len(autos['ALTERNATIVE'][0]) > 0:
                                vendor_alt = try_decode(autos['ALTERNATIVE'][0])
                        if len(vendor_alt) > 0:
                            auto_temp = cfc.AutoVendorTemplate(label=tmp, archive=True)
                            for aut, autv in autos.items():
                                if aut == 'PATH':
                                    try:
                                        auto_temp.path_input = try_decode(autv[0][0])
                                    except IndexError:
                                        pass
                                elif aut == 'STATNAME':
                                    try:
                                        auto_temp.station_name = try_decode(autv[0][0])
                                    except IndexError:
                                        pass
                                elif aut == 'PATHAPP':
                                    try:
                                        auto_temp.path_output = try_decode(autv[0][0])
                                    except IndexError:
                                        pass
                                #elif aut == 'ARCHIVE':
                                #    auto_temp.archive = bool(autv[0])
                                elif aut == 'ALTERNATIVE':
                                    auto_temp.file_type = get_filetype(vendor_alt)
                                else:
                                    pass

                            tempsThis_vendor.append(auto_temp)
                        else:
                            auto_temp = cfc.AutoTemplate(label=tmp)
                            sort_list = []
                            sort_asc = []
                            for aut, autv in autos.items():
                                if aut == 'PATH':
                                    auto_temp.path_input = try_decode(autv[0][0])
                                elif aut == 'STATNAME':
                                    try:
                                        auto_temp.station_name = try_decode(autv[0][0])
                                    except IndexError:
                                        pass
                                elif aut == 'DCMCRIT' and autv[0].dtype != 'int16':
                                    if autv[0][0] != b'0000' and autv[0][1] != b'0000':
                                        dict_tags = {
                                            f'unknown{n_unknown}': [autv[0][:2]]}
                                        dict_formats = {}
                                        tag_pattern = self.convert_tag_pattern(
                                            dict_tags, dict_formats)
                                        if 'unknown' in tag_pattern.list_tags[0]:
                                            n_unknown += 1
                                        auto_temp.dicom_crit_attributenames = (
                                            tag_pattern.list_tags)
                                        auto_temp.dicom_crit_values = [
                                            try_decode(autv[0][2])
                                            ]
                                elif aut == 'SORTBY':
                                    if isinstance(autv[0], np.ndarray):
                                        sort_list = list(autv[0])
                                        sort_list = [x.decode('utf-8')
                                                     for x in sort_list]
                                    elif isinstance(autv[0], bytes):
                                        sort_list = [autv[0].decode('utf-8')]
                                elif aut == 'SORTASC' and len(sort_list) > 0:
                                    if isinstance(autv[0], np.ndarray):
                                        sort_asc = list(autv[0])
                                        sort_asc = [not bool(x) for x in sort_asc]
                                    else:
                                        sort_asc = [not bool(autv[0])]
                                elif aut == 'PARAMSET':
                                    auto_temp.paramset_label = try_decode(
                                        autv[0])
                                elif aut == 'QUICKTEMP':
                                    auto_temp.quicktemp_label = try_decode(
                                        autv[0])
                                elif aut == 'PATHAPP':
                                    try:
                                        auto_temp.path_output = try_decode(
                                            autv[0][0])
                                    except IndexError:
                                        pass
                                elif aut == 'ARCHIVE':
                                    auto_temp.archive = bool(autv[0])
                                elif aut == 'DELETEFILES':
                                    auto_temp.delete_if_not_image = bool(
                                        autv[0])
                                elif aut == 'DELETEFILESEND':
                                    auto_temp.delete_if_too_many = bool(
                                        autv[0])
                                else:
                                    pass
                            if len(sort_list) > 0:
                                tag_pattern = self.convert_sortBy2tag_sort_template(
                                    sort_list, sort_asc, m)
                                auto_temp.sort_pattern = tag_pattern

                            tempsThis.append(auto_temp)
                auto_temps_mod[mod_py(m)] = tempsThis
                auto_temps_vendor_mod[mod_py(m)] = tempsThis_vendor

        self.auto_templates = auto_temps_mod
        self.auto_vendor_templates = auto_temps_vendor_mod

    def convert_renametemp(self, c):
        """Convert renametemp structure into objects.

        Parameters
        ----------
        c : dict
            renametemp in config.dat converted from IDL
        """
        cNames = [key for key, val in c.items()]
        if 'TAGS' in cNames and 'TAGFORMATS' in cNames:
            # add additional tags if any found
            dict_tags = self.as_dict(c['TAGS'][0])
            dict_formats = self.as_dict(c['TAGFORMATS'][0])
            tag_pattern = self.convert_tag_pattern(dict_tags, dict_formats)
            idl_attr = [key for key in dict_tags]
            py_attr = tag_pattern.list_tags

            def tags_convert(subfolder_arr, allformats):
                """Convert template contents of idl renametemp to py version.

                Parameters
                ----------
                subfolder_arr : np.ndarray or bytes
                    content in ['TEMP'][0][index]['CAT' or 'FILE']
                allformats : list of str
                    format strings of all available in renametemp already converted

                Returns
                -------
                py_attr_names : list of str
                    list_tags to TagPattern
                formats : list of str
                    list_format to TagPattern
                """
                try:
                    subfolder_list = [elem.decode('utf-8') for elem in subfolder_arr]
                except AttributeError:
                    subfolder_list = [subfolder_arr.decode('utf-8')]
                subfolder_list = [elem.upper() for elem in subfolder_list]
                py_attr_names = []
                formats = []
                for name in subfolder_list:
                    idx = idl_attr.index(name)
                    py_attr_names.append(py_attr[idx])
                    formats.append(allformats[idx])
                return py_attr_names, formats

            if 'TEMP' in cNames:
                dict_temps = self.as_dict(c['TEMP'][0])
                self.rename_patterns = empty_template_dict(QUICKTEST_OPTIONS,
                                                           dummy=cfc.RenamePattern())
                for key, temp in dict_temps.items():
                    content = self.as_dict(temp[0])
                    dict_formats = self.as_dict(content['FORMATS'][0])
                    tag_pattern = self.convert_tag_pattern(dict_tags, dict_formats)
                    list_format_temp = tag_pattern.list_format
                    list_tags, list_format = tags_convert(
                        content['CAT'][0], list_format_temp)
                    list_tags2, list_format2 = tags_convert(
                        content['FILE'][0], list_format_temp)
                    self.rename_patterns['CT'].append(
                        cfc.RenamePattern(
                            label=key,
                            list_tags=list_tags, list_format=list_format,
                            list_tags2=list_tags2, list_format2=list_format2
                            )
                        )
                self.rename_patterns['CT'].pop(0)  # remove first=dummy
                self.errmsg.append(
                    'Found Rename templates in .dat file. IDL version did not '
                    'categorize these into modalities. All placed in modality CT. '
                    'After import is finished you could move templates to another '
                    'category (modality).')

    def convert_sortBy2tag_sort_template(
            self, sortBy, sortAsc, modality):
        """Convert sortBy+sortAsc from loadtemp in IDL version to TagPattern.

        TagPatternSort added to dictionary of sort patterns

        Parameters
        ----------
        sortBy : list of str
            strings according to DICOM parameters in IDL version
        sortAsc : list of int
            0/1 as False/True for sorting elements in sortBy desc/asc
        modality : str
            modality for the automation template
        """
        tag_pattern = cfc.TagPatternSort(
            list_sort=[bool(x) for x in sortAsc])
        sort_tags = []
        for idl_tag_name in sortBy:
            if idl_tag_name == 'FRAMENO':
                sort_tags.append('FrameNumber')
            else:
                try:
                    idx = idl_attr.index(idl_tag_name)
                    if idl_attr.count(idl_tag_name) == 2:
                        # EXMODTYPE
                        if modality == 'XRAY':
                            idx = idx + 1
                    sort_tags.append(py_attr[idx])
                except ValueError:
                    try:
                        tag_uppercase = [
                            attr.uppercase() for attr in self.tag_infos_labels]
                        idx = tag_uppercase.index(idl_tag_name)
                        sort_tags.append(self.tag_infos_labels[idx])
                    except ValueError:
                        sort_tags.append('')
                        self.errmsg.append(
                            f'Failed to convert idl attribute {idl_tag_name}')

        tag_pattern.list_tags = sort_tags
        if '' in sort_tags:
            delete_idxs = [idx for idx, tag in enumerate(sort_tags) if tag == '']
            delete_idxs.sort(reverse=True)
            for idx in delete_idxs:
                tag_pattern.delete_tag(idx)

        return tag_pattern

    def convert_format(self, arr):
        """Convert format string from idl to python f-string."""
        formatstr = arr.decode('UTF-8').replace('(', '').replace(')', '')
        origformatstr = formatstr
        if 'i' in formatstr:  # integer
            if formatstr == 'i0':
                formatstr = ''
            else:
                formatstr = formatstr.replace('i', ':')
        elif 'a' in formatstr:  # string a06 to :0>6
            if formatstr == 'a0':
                formatstr = ''
            else:
                if '0' in formatstr:
                    formatstr = formatstr.replace('a0', ':0>')
                else:  # slice string
                    formatstr = formatstr.replace('a', '[:') + ']'
        elif 'f' in formatstr:  # float f0.2 to :.2, f03.2 to :06.2 (:0 3+1+2 .2)
            if formatstr != 'f0':
                formatstr = formatstr.replace('f', '')
                spl = formatstr.split('.')
                new_formatstr = ''
                if spl[0] == '0' and int(spl[0]) > 0:
                    new_formatstr = '0'
                    if int(spl[0]) > 0:
                        tot_char = int(spl[0]) + 1 + int(spl[1])
                        new_formatstr = new_formatstr + f'{tot_char}'
                formatstr = f':{new_formatstr}.{spl[1]}'

        return formatstr

    def convert_tag_pattern(self, dict_tags, dict_formats):
        """Convert dicom tag dict from config.dat to python."""
        list_tags = []
        list_format = []

        for key, val in dict_tags.items():
            attribute_name = ''
            try:
                tag = [hex(val[0][0]), hex(val[0][1])]
            except TypeError:
                # assume if isinstance(val[0][0], bytes):
                tag = [hex(int(vv, 16)) for vv in val[0]]
            if tag in self.tag_infos_tags:
                idx = self.tag_infos_tags.index(tag)
                attribute_name = self.tag_infos_labels[idx]
            else:
                already_tags = [tag_info.tag for tag_info in self.tag_infos_new]
                if tag in already_tags:
                    idx = already_tags.index(tag)
                    attribute_name = self.tag_infos_new[idx].attribute_name
                else:
                    attribute_name = key
                    self.tag_infos_new.append(
                        cfc.TagInfo(
                            tag=tag,
                            attribute_name=attribute_name
                        )
                    )
            list_tags.append(attribute_name)

        if dict_formats != {}:
            for key, val in dict_formats.items():
                format_this = self.convert_format(val[0])
                list_format.append(format_this)
        else:
            list_format = [''] * len(list_tags)

        if len(list_tags) > 0:
            tag_pattern = cfc.TagPatternFormat()
            for i in range(len(list_tags)):
                tag_pattern.add_tag(tag=list_tags[i], format_string=list_format[i])

        return tag_pattern
