#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UI for image canvas. Used in main and for automation with GUI.

@author: Ellen Wasbo
"""
import numpy as np
from time import sleep

import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import patches
from PyQt5.QtWidgets import QMessageBox
from skimage.transform import resize

# imageQC block start
from imageQC.scripts.mini_methods_calculate import get_min_max_pos_2d
from imageQC.scripts.calculate_roi import (
    generate_SNI_Siemens_image, get_roi_circle)
from imageQC.config.iQCconstants import COLORS
# imageQC block end


def get_rotated_crosshair(szx, szy, delta_xya):
    """Get xydata for rotated crosshair.

    Parameters
    ----------
    szx : int
        image size x direction
    szy : int
        image size y direction
    delta_xya : tuple
        dx, dy, dangle - center offset relative to center in image

    Returns
    -------
    x1, x2, y1, y2 : int
        xydata for 2Dlines
    """
    tan_a = np.tan(np.deg2rad(delta_xya[2]))
    dy1 = tan_a*(szx*0.5 + delta_xya[0])
    dy2 = tan_a*(szx*0.5 - delta_xya[0])
    dx1 = tan_a*(szy*0.5 - delta_xya[1])
    dx2 = tan_a*(szy*0.5 + delta_xya[1])

    x1 = szx*0.5+delta_xya[0]-dx1
    x2 = szx*0.5+delta_xya[0]+dx2
    y1 = szy*0.5+delta_xya[1]-dy1
    y2 = szy*0.5+delta_xya[1]+dy2

    return (x1, x2, y1, y2)


class GenericImageCanvas(FigureCanvasQTAgg):
    """Canvas for display of image."""

    def __init__(self, parent, main):
        self.main = main
        self.parent = parent
        self.fig = matplotlib.figure.Figure(dpi=150)
        self.fig.subplots_adjust(0., 0., 1., 1.)
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.ax = self.fig.add_subplot(111)
        self.last_clicked_pos = (-1, -1)
        self.profile_length = 10  # assume click drag > length in pix = draw profile
        self.rectangle = patches.Rectangle((0, 0), 1, 1)
        self.current_image = None

        # default display
        self.img = self.ax.imshow(np.zeros((2, 2)))
        self.ax.clear()
        self.ax.axis('off')

        # intialize parameters to make pylint happy
        self.contours = []
        self.scatters = []
        self.linewidth = 2
        self.fontsize = 10
        self.cmap = 'gray'
        self.min_val = None
        self.max_val = None
        self.title = ''

    def profile_draw(self, x2, y2, pix=None):
        """Draw line for profile.

        Parameters
        ----------
        x2 : float
            endpoint x coordinate
        y2 : float
            endpoint y coordinate
        pix : float, optional
            pixelsize. The default is None.

        Returns
        -------
        plotstatus : bool
            True if plot was possible
        """
        self.profile_remove()
        plotstatus = False
        if self.last_clicked_pos != (-1, -1):
            x1 = self.last_clicked_pos[0]
            y1 = self.last_clicked_pos[1]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > self.profile_length:
                self.ax.add_artist(matplotlib.lines.Line2D(
                    [x1, x2], [y1, y2],
                    color='red', linewidth=self.main.gui.annotations_line_thick,
                    gid='profile'))
                self.draw_idle()
                plotstatus = True
        return plotstatus

    def profile_remove(self):
        """Clear profile line."""
        if hasattr(self.ax, 'lines'):
            for i, line in enumerate(self.ax.lines):
                if line.get_gid() == 'profile':
                    self.ax.lines[i].remove()
                    self.draw_idle()
                    break

    def rectangle_mark(self, x2, y2):
        """Mark rectangle in image for e.g. setting window level, draw ROI.

        Parameters
        ----------
        x2 : float
            endpoint x coordinate
        y2 : float
            endpoint y coordinate

        Returns
        -------
        status : bool
            True if setting rectangle was possible
        """
        self.rectangle_remove()
        status = False
        if self.last_clicked_pos != (-1, -1):
            x1 = self.last_clicked_pos[0]
            y1 = self.last_clicked_pos[1]
            width = x2 - x1
            height = y2 - y1

            if abs(width) > 5 and abs(height) > 5:
                self.rectangle = patches.Rectangle(
                    (x1, y1), width, height,
                    edgecolor='red', fill=False, gid='rectangle',
                    linewidth=self.main.gui.annotations_line_thick,
                    linestyle='dotted')
                self.ax.add_patch(self.rectangle)
                self.draw_idle()
                status = True
        return status

    def rectangle_remove(self):
        """Clear marked rectangle."""
        if len(self.ax.patches) > 0:
            for i, p in enumerate(self.ax.patches):
                if p.get_gid() == 'rectangle':
                    self.ax.patches[i].remove()
                    break
        self.draw_idle()

    def draw(self):
        """Avoid super().draw when figure collapsed by sliders."""
        try:
            super().draw()
        except ValueError:
            pass

    def draw_idle(self):
        """Avoid super().draw when figure collapsed by sliders."""
        try:
            super().draw_idle()
        except ValueError:
            pass


class ImageCanvas(GenericImageCanvas):
    """Canvas for drawing the active DICOM image."""

    def __init__(self, parent, main):
        super().__init__(parent, main)

    def img_is_missing(self):
        """Show message when pixel_data is missing."""
        try:
            cmap = self.ax.get_images()[0].cmap.name
        except (AttributeError, IndexError):
            cmap = 'gray'
        self.ax.clear()
        self.img = self.ax.imshow(np.zeros((100, 100)), cmap=cmap)
        self.ax.axis('off')
        at = matplotlib.offsetbox.AnchoredText(
            'Pixel data not found',
            prop=dict(size=14, color='gray'),
            frameon=False, loc='center')
        self.ax.add_artist(at)
        self.draw()

    def img_clear(self):
        """Clear image."""
        self.ax.clear()
        self.draw()

    def img_draw(self, auto=False, window_level=[], force_home=False):
        """Refresh image."""
        # keep previous zoom if same size image
        xlim = None
        ylim = None
        if force_home is False:
            try:
                prev_img = self.ax.get_images()[0].get_array()
                if prev_img.shape == self.main.active_img.shape:
                    xlim = self.ax.get_xlim()
                    ylim = self.ax.get_ylim()
            except (AttributeError, IndexError):
                pass
        try:
            cmap = self.ax.get_images()[0].cmap.name
        except (AttributeError, IndexError):
            cmap = 'gray'

        self.ax.clear()
        nparr = self.main.active_img
        if auto is False and hasattr(self.main, 'wid_window_level'):
            wl_min, wl_max = self.main.wid_window_level.tb_wl.get_min_max()
            self.main.wid_window_level.colorbar.colorbar_draw(cmap=cmap)
            annotate = self.main.gui.annotations
        else:
            annotate = False
            if len(window_level) > 0:
                wl_min, wl_max = window_level
            else:
                meanval = np.mean(self.main.active_img)
                stdval = np.std(self.main.active_img)
                wl_min = meanval-stdval
                wl_max = meanval+stdval

        self.img = self.ax.imshow(
            nparr, cmap=cmap, vmin=wl_min, vmax=wl_max)
        if xlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        if self.main.gui.show_axis is False:
            self.ax.axis('off')
            self.fig.subplots_adjust(.0, .0, 1., 1.)
        else:
            self.fig.subplots_adjust(.05, .05, 1., 1.)

        self.current_image = nparr

        if annotate:
            self.add_crosshair()

            # DICOM annotations
            try:
                tool_sum = self.parent.tool_sum.isChecked()
                if tool_sum:
                    marked_idxs = self.main.get_marked_imgs_current_test()
                    tool_sum = self.main.gui.active_img_no in marked_idxs
            except AttributeError:
                tool_sum = False
            if tool_sum:
                annot_text = (
                    ['Average image', ''] if self.main.average_img
                    else ['Summed image', '']
                    )
            else:
                try:
                    annot_text = self.main.imgs[
                        self.main.gui.active_img_no].annotation_list
                except IndexError:
                    annot_text = ''
            if annot_text:
                at = matplotlib.offsetbox.AnchoredText(
                    '\n'.join(annot_text),
                    prop=dict(size=self.main.gui.annotations_font_size, color='red'),
                    frameon=False, loc='upper left')
                self.ax.add_artist(at)

            self.roi_draw()
        else:
            self.draw()

    def remove_annotations(self, remove_crosshair=False):
        """Remove current annotations."""
        for contour in self.contours:
            try:
                for coll in contour.collections:
                    try:
                        coll.remove()
                    except ValueError:
                        pass
            except (AttributeError, NotImplementedError):
                try:
                    contour.remove()  # matplotlib v3.10(?), collections depricated
                except(AttributeError, NotImplementedError):
                    pass
        self.contours = []

        for scatter in self.scatters:
            try:
                scatter.remove()
            except (ValueError, NotImplementedError):
                pass
        self.scatters = []

        if hasattr(self.ax, 'lines'):
            n_lines = len(self.ax.lines)
            if remove_crosshair:
                for line in self.ax.lines:
                    line.remove()
            else:
                if n_lines > 2:
                    for i in range(n_lines - 2):
                        self.ax.lines[-1].remove()

        try:
            self.ax.get_legend().remove()
        except AttributeError:
            pass

        try:
            self.ax.texts.clear()
        except AttributeError:  # matplotlib 3.7+
            for txt in self.ax.texts:
                txt.remove()

        self.draw_idle()

    def add_crosshair(self):
        if self.current_image is not None and not self.main.automation_active:
            nparr = self.current_image
            szy, szx = np.shape(nparr)
            try:
                linewidth = self.main.gui.annotations_line_thick
            except AttributeError:
                linewidth = 1.
            if self.main.gui.delta_a == 0:
                self.ax.axhline(
                    y=szy*0.5 + self.main.gui.delta_y,
                    color='red', linewidth=linewidth, linestyle='--',
                    )
                self.ax.axvline(
                    x=szx*0.5 + self.main.gui.delta_x,
                    color='red', linewidth=linewidth, linestyle='--')
            else:
                x1, x2, y1, y2 = get_rotated_crosshair(
                    szx, szy,
                    (self.main.gui.delta_x,
                     self.main.gui.delta_y,
                     self.main.gui.delta_a)
                    )
                # NB keep these two lines as first and second in ax.lines
                self.ax.add_artist(matplotlib.lines.Line2D(
                    [0, szx], [y1, y2],
                    color='red', linewidth=linewidth, linestyle='--',
                    gid='axis1'))
                self.ax.add_artist(matplotlib.lines.Line2D(
                    [x1, x2], [szy, 0],
                    color='red', linewidth=linewidth, linestyle='--',
                    gid='axis2'))

    def roi_draw(self):
        """Update ROI countours on image."""
        remove_crosshair = (
            (self.main.current_test=='DCM')
            or (self.main.current_test=='CDM')
            or ('InputMain' in str(type(self.main)))
            )
        self.remove_annotations(remove_crosshair=remove_crosshair)
        if remove_crosshair is False:
            self.add_crosshair()

        annotate = True
        if hasattr(self.main, 'wid_window_level'):
            annotate = self.main.gui.annotations

        if self.main.current_roi is not None and annotate:
            try:
                self.linewidth = self.main.gui.annotations_line_thick
                self.fontsize = self.main.gui.annotations_font_size
            except AttributeError:
                pass  # default

            class_method = getattr(self, self.main.current_test, None)
            if class_method is not None:
                class_method()
            else:
                self.add_contours_to_all_rois()
        self.draw()
        if 'InputMain' in str(type(self.main)):
            sleep(.2)

    def add_contours_to_all_rois(self, colors=None, reset_contours=True,
                                 roi_indexes=None, filled=False,
                                 labels=None, labels_pos='upper_left',
                                 linestyles=None,
                                 hatches=None):
        """Draw all ROIs in self.main.current_roi (list) with specific colors.

        Parameters
        ----------
        colors : list of str, optional
            Default is None = all red
        reset_contours : bool, optional
            Default is True
        roi_indexes : list of int, optional
            roi indexes to draw. Default is None = all
        filled : bool
            if true used contourf (filled) instead
        labels : str, optional
            'upper_left' or 'center' relative to roi
        linestyles : list of str, optional
            Default is None == all solid
        hatches : list of str, optional
            Used if filled is True. Default is None.
        """
        this_roi = self.main.current_roi
        if not isinstance(self.main.current_roi, list):
            this_roi = [self.main.current_roi]

        if reset_contours:
            self.contours = []
        if colors is None:
            colors = ['red' for i in range(len(this_roi))]
        if linestyles is None:
            linestyles = ['solid' for i in range(len(this_roi))]
        if roi_indexes is None:
            roi_indexes = list(np.arange(len(this_roi)))

        for i, roi_no in enumerate(roi_indexes):
            color_no = i % len(colors)
            linestyle_no = i % len(linestyles)
            mask = np.where(this_roi[roi_no], 0, 1)
            contour = None
            if filled:
                if hatches is None:
                    contour = self.ax.contourf(
                        mask, levels=[0, 0.5], colors=colors[color_no], alpha=0.3)
                else:
                    hatch_no = i % len(hatches)
                    contour = self.ax.contourf(
                        mask, levels=[0, 0.5], colors='none',
                        hatches=hatches[hatch_no])
                    contour.collections[0].set_edgecolor(colors[color_no])
            else:
                try:
                    contour = self.ax.contour(
                        mask, levels=[0.9],
                        colors=colors[color_no], alpha=0.5, linewidths=self.linewidth,
                        linestyles=linestyles[linestyle_no])
                except TypeError:
                    pass
                except np.core._exeptions._ArrayMemoryExeption:
                    QMessageBox.warning(
                        self.main, 'Failed drawing ROI',
                        'There was a memory issue while drawing ROIs. '
                        'You may turn off annotations to avoid memory issues.')

            if labels:
                try:
                    label = labels[i]
                    mask_pos = np.where(mask == 0)
                    if labels_pos == 'center':
                        xpos = np.mean(mask_pos[1]) - 2
                        ypos = np.mean(mask_pos[0]) + 2
                    elif labels_pos == 'upper_left_inside':
                        xpos = np.min(mask_pos[1]) + 10
                        ypos = np.min(mask_pos[0]) + 30  # TODO better fit to image res.
                    else:  # upper left
                        xpos = np.min(mask_pos[1]) - 5
                        ypos = np.min(mask_pos[0]) - 5
                    if np.isfinite(xpos) and np.isfinite(ypos):
                        self.ax.text(xpos, ypos, label,
                                     fontsize=self.fontsize,
                                     color=colors[color_no])
                except (ValueError, IndexError):
                    pass
            if contour is not None:
                self.contours.append(contour)

    def Bar(self):
        """Draw Bar ROIs."""
        self.add_contours_to_all_rois(
            colors=COLORS,
            labels=[str(i+1) for i in range(4)]
            )

    def CDM(self):
        """Draw found lines."""
        if self.main.current_roi is not None:
            center_xs = self.main.current_roi['xs']
            center_ys = self.main.current_roi['ys']
            xs = center_xs.flatten()
            ys = center_ys.flatten()
            if 'include_array' in self.main.current_roi:
                include_array = self.main.current_roi['include_array']
                if include_array is not None:
                    include = include_array.flatten()
                    idxs = np.where(include == True)
                    xs = xs[idxs]
                    ys = ys[idxs]
            self.scatters = []
            scatter = self.ax.scatter(xs, ys, s=40, c='green', marker='+')
            self.scatters.append(scatter)
            if 'CDM' in self.main.results:
                idx_diam = self.main.tab_mammo.cdm_cbox_diameter.currentIndex()
                idx_thick = self.main.tab_mammo.cdm_cbox_thickness.currentIndex()
                x = center_xs[idx_thick, idx_diam]
                y = center_ys[idx_thick, idx_diam]
                scatter = self.ax.scatter(x, y, s=40, c='blue', marker='+')
                self.scatters.append(scatter)

    def CTn(self):
        """Draw CTn ROI."""
        ctn_table = self.main.current_paramset.ctn_table
        nroi = len(ctn_table.labels)
        self.add_contours_to_all_rois(
            labels=ctn_table.labels,
            roi_indexes=[i for i in range(nroi)]
            )

        if len(self.main.current_roi) == 2 * len(ctn_table.labels):
            # draw search rois
            self.add_contours_to_all_rois(
                colors=['blue' for i in range(nroi)],
                roi_indexes=[i for i in range(nroi, 2 * nroi)],
                reset_contours=False
                )

    def Dim(self):
        """Draw search ROI for rods and resulting centerpositions if any."""
        self.add_contours_to_all_rois(roi_indexes=[0, 1, 2, 3])
        if 'Dim' in self.main.results:
            try:
                details_dict = self.main.results['Dim'][
                    'details_dict'][self.main.gui.active_img_no]
                if 'centers_x' in details_dict:
                    xs = details_dict['centers_x']
                    ys = details_dict['centers_y']
                    for i in range(4):
                        self.ax.add_artist(matplotlib.lines.Line2D(
                            [xs[i-1], xs[i]], [ys[i-1], ys[i]],
                            color='r', linewidth=self.linewidth,
                            linestyle='dotted',
                            ))
            except (KeyError, TypeError):
                pass

    def Foc(self):
        missing_result = True
        if 'Foc' in self.main.results:
            try:
                details_dict = self.main.results['Foc'][
                    'details_dict'][self.main.gui.active_img_no]
                pix = self.main.imgs[self.main.gui.active_img_no].pix[0]
                if 'star_diameter_mm' in details_dict:
                    missing_result = False
                    off_xy = details_dict['off_xy']
                    radius = details_dict['star_diameter_mm'] / 2 / pix
                    roi_pattern = get_roi_circle(
                        self.main.active_img.shape, off_xy, radius)
                    mask = np.where(roi_pattern, 0, 1)
                    contour = self.ax.contour(
                        mask, levels=[0.9],
                        colors='blue', alpha=0.5,
                        linewidths=self.linewidth, linestyles='--')
                    self.contours.append(contour)
            except (KeyError, TypeError):
                pass

        if missing_result:
            self.add_contours_to_all_rois()

    def Gho(self):
        """Draw Ghosting ROIs."""
        try:
            labels = self.main.current_paramset.gho_table.labels
        except AttributeError:
            labels = None
        self.add_contours_to_all_rois(colors=COLORS, labels=labels)

    def Hom(self):
        """Draw Hom ROI."""
        flatfield = False
        if self.main.current_modality == 'Mammo':
            flatfield = True
        elif self.main.current_modality == 'Xray':
            if self.main.current_paramset.hom_tab_alt >= 3:
                flatfield = True

        if flatfield:
            colors = ['blue']
            idxs = [0]
            if self.main.current_roi[1] is not None:
                colors.append('blue')
                idxs.append(1)
            self.add_contours_to_all_rois(colors=colors, roi_indexes=idxs)
            if self.main.current_paramset.hom_mask_max:
                if self.main.current_roi[2] is not None:
                    self.add_contours_to_all_rois(
                        colors=['red'], roi_indexes=[2],
                        filled=True, hatches=['////'], reset_contours=False)
            if self.main.current_paramset.hom_mask_outer_mm > 0:
                mask = np.where(self.main.current_roi[-1], 0, 1)
                contour = self.ax.contour(
                    mask, levels=[0.9],
                    colors='blue', alpha=0.5, linewidths=self.linewidth,
                    linestyles='dotted')
                self.contours.append(contour)
        else:
            self.add_contours_to_all_rois(colors=COLORS)

    def MTF(self):
        """Draw MTF ROI."""
        if (
                self.main.current_modality in ['CT', 'SPECT', 'PET']
                and self.main.current_paramset.mtf_type in [0, 1]):
            # bead show background rim
            if self.main.current_roi[1].any():
                self.add_contours_to_all_rois(roi_indexes=[1])
            else:
                self.add_contours_to_all_rois(roi_indexes=[0])
        elif self.main.current_modality == 'NM':
            self.add_contours_to_all_rois(colors=['red', 'blue'])
        else:
            if isinstance(self.main.current_roi, list):
                if self.main.current_modality != 'CT':
                    cols = ['r', 'b', 'g', 'c']
                else:
                    cols = COLORS
                roi_indexes = list(range(len(self.main.current_roi) - 1))
                self.add_contours_to_all_rois(
                    roi_indexes=roi_indexes,
                    colors=cols)
                mask = np.where(self.main.current_roi[-1], 0, 1)
                contour = self.ax.contour(
                    mask, levels=[0.9],
                    colors='red', alpha=0.5, linewidths=self.linewidth,
                    linestyles='dotted')
                self.contours.append(contour)
            else:
                self.add_contours_to_all_rois(colors=COLORS)

            if 'MTF' in self.main.results and self.main.current_modality == 'CT':
                try:
                    if 'details_dict' in self.main.results['MTF']:
                        if self.main.current_paramset.mtf_type == 2:  # circular disc
                            roi_disc = self.main.results[
                                'MTF']['details_dict'][0]['found_disc_roi']
                            mask = np.where(roi_disc, 0, 1)
                            contour = self.ax.contour(
                                mask, levels=[0.9],
                                colors='red', alpha=0.5, linewidths=self.linewidth,
                                linestyles='dotted')
                            self.contours.append(contour)
                except (TypeError, KeyError, IndexError):
                    pass

    def NPS(self):
        """Draw NPS ROIs."""
        if self.main.current_modality == 'CT':
            self.add_contours_to_all_rois(
                colors=['w']*self.main.current_paramset.nps_n_sub, filled=True)
        elif self.main.current_modality in ['Xray', 'Mammo']:
            n_sub = self.main.current_paramset.nps_n_sub
            idxs = [0, n_sub*2-2, n_sub*2, n_sub*4, -(n_sub*2-1), -1]
            self.add_contours_to_all_rois(
                colors=['r']*6, filled=True, roi_indexes=idxs)
        else:
            pass

    def Num(self):
        """Draw  ROIs with labels."""
        labels = self.main.current_paramset.num_table.labels
        colors = ['r'] * len(labels)
        try:
            widget = self.main.stack_test_tabs.currentWidget()
            sel = widget.num_table_widget.table.selectedIndexes()
            idx = sel[0].row()
            colors[idx] = 'b'
        except (AttributeError, IndexError):
            pass
        self.add_contours_to_all_rois(colors=colors, labels=labels)

    def Pha(self):
        if 'Pha' in self.main.results:
            try:
                details_dict = self.main.results['Pha'][
                    'details_dict'][self.main.gui.active_img_no]
                pix = self.main.imgs[self.main.gui.active_img_no].pix[0]
                shape = self.main.imgs[self.main.gui.active_img_no].shape
                cnr_dicts = details_dict['cnr_results_pr_disc']
                center_xs = [elem['center_xy'][0] for elem in cnr_dicts]
                center_ys = [elem['center_xy'][1] for elem in cnr_dicts]
                center_xs = np.array(center_xs) + shape[1] / 2
                center_ys = np.array(center_ys) + shape[0] / 2
                self.scatters = []
                scatter = self.ax.scatter(
                    center_xs, center_ys, s=40, c='green', marker='x')
                self.scatters.append(scatter)
            except (KeyError, TypeError):
                pass

    def PIU(self):
        """Draw MR PIU ROI."""
        self.add_contours_to_all_rois()
        # display min, max pos
        self.scatters = []
        min_idx, max_idx = get_min_max_pos_2d(
            self.main.active_img, self.main.current_roi)
        scatter = self.ax.scatter(min_idx[1], min_idx[0], s=40,
                                  c='blue', marker='D')
        self.scatters.append(scatter)
        self.ax.text(min_idx[1], min_idx[0]+10,
                     'min', fontsize=self.fontsize, color='blue')
        scatter = self.ax.scatter(max_idx[1], max_idx[0], s=40,
                                  c='fuchsia', marker='D')
        self.scatters.append(scatter)
        self.ax.text(max_idx[1], max_idx[0]+10,
                     'max', fontsize=self.fontsize, color='fuchsia')

    def Rec(self):
        """Draw PET Rec ROI."""
        labels = self.main.current_paramset.rec_table.labels
        self.add_contours_to_all_rois(
            labels=labels,
            labels_pos='center',
            roi_indexes=[i for i in range(len(labels))])
        self.add_contours_to_all_rois(
            labels=['center'],
            roi_indexes=[len(labels)], reset_contours=False)
        if self.main.results:
            if 'Rec' in self.main.results:
                if self.main.results['Rec']:
                    if 'details_dict' in self.main.results['Rec']:
                        zpos_this = self.main.imgs[self.main.gui.active_img_no].zpos
                        dd = self.main.results['Rec']['details_dict']
                        if 'used_zpos' in dd:
                            if zpos_this in dd['used_zpos_spheres']:
                                idx = np.where(dd['used_zpos_spheres'] == zpos_this)
                                idx = idx[0][0]
                                if 'roi_spheres' in dd:
                                    for roi in dd['roi_spheres']:
                                        if roi[idx] is not None:
                                            mask = np.where(roi[idx], 0, 1)
                                            contour = self.ax.contour(
                                                mask, levels=[0.9],
                                                colors='blue', alpha=0.5,
                                                linewidths=self.linewidth)
                                            self.contours.append(contour)
                                    for roi in dd['roi_peaks']:
                                        if roi[idx] is not None:
                                            mask = np.where(roi[idx], 0, 1)
                                            contour = self.ax.contour(
                                                mask, levels=[0.9],
                                                colors='green', alpha=0.5,
                                                linewidths=self.linewidth)
                                            self.contours.append(contour)
                                    mask = np.where(
                                        np.zeros(self.main.active_img.shape) == 0)
                                    c1 = self.ax.contour(
                                        mask, colors='blue', alpha=0.5,
                                        linewidths=self.linewidth)
                                    h1, _ = c1.legend_elements()
                                    c2 = self.ax.contour(
                                        mask, colors='green', alpha=0.5,
                                        linewidths=self.linewidth)
                                    h2, _ = c2.legend_elements()
                                    self.ax.legend(
                                        [h1[0], h2[0]], ['Sphere VOI', 'Peak VOI'])

    def ROI(self):
        """Drow ROIs with labels if any."""
        if self.main.current_paramset.roi_use_table > 0:
            labels = self.main.current_paramset.roi_table.labels
            colors = COLORS
        else:
            labels = None
            colors = None
        self.add_contours_to_all_rois(labels=labels, colors=colors)

    def SDN(self):
        """Mammo SDNR ROIs."""
        if isinstance(self.main.current_roi, list):
            roi_indexes = list(range(5))
            self.add_contours_to_all_rois(roi_indexes=roi_indexes)
            if len(self.main.current_roi) == 6:
                mask = np.where(self.main.current_roi[-1], 0, 1)
                contour = self.ax.contour(
                    mask, levels=[0.9],
                    colors='blue', alpha=0.5, linewidths=self.linewidth,
                    linestyles='dotted')
                self.contours.append(contour)

    def Sli(self):
        """Draw Slicethickness search lines."""
        h_colors = ['b', 'lime']
        v_colors = ['c', 'r', 'm', 'darkorange']
        if self.main.current_modality == 'MR':
            search_margin = self.main.current_paramset.sli_average_width
        else:
            search_margin = self.main.current_paramset.sli_search_width
        background_length = self.main.current_paramset.sli_background_width
        try:
            pix = self.main.imgs[self.main.gui.active_img_no].pix
            background_length = background_length / pix[0]
        except (AttributeError, IndexError):  # with automation
            background_length = 0
        for l_idx, line in enumerate(self.main.current_roi['h_lines']):
            y1, x1, y2, x2 = line
            self.ax.add_artist(matplotlib.lines.Line2D(
                [x1, x2], [y1, y2],
                color=h_colors[l_idx], linewidth=self.linewidth,
                linestyle='dotted',
                ))
            self.ax.add_artist(matplotlib.lines.Line2D(
                [x1, x2], [y1 - search_margin, y2 - search_margin],
                color=h_colors[l_idx], linewidth=0.5*self.linewidth,
                ))
            self.ax.add_artist(matplotlib.lines.Line2D(
                [x1, x2], [y1 + search_margin, y2 + search_margin],
                color=h_colors[l_idx], linewidth=0.5*self.linewidth,
                ))
            self.ax.add_artist(matplotlib.lines.Line2D(
                [x1 + background_length, x1 + background_length],
                [y1 - search_margin, y1 + search_margin],
                color=h_colors[l_idx], linewidth=0.5*self.linewidth
                ))
            self.ax.add_artist(matplotlib.lines.Line2D(
                [x2 - background_length, x2 - background_length],
                [y2 - search_margin, y2 + search_margin],
                color=h_colors[l_idx], linewidth=0.5*self.linewidth
                ))
        for l_idx, line in enumerate(self.main.current_roi['v_lines']):
            y1, x1, y2, x2 = line
            self.ax.add_artist(matplotlib.lines.Line2D(
                [x1, x2], [y1, y2],
                color=v_colors[l_idx], linewidth=self.linewidth,
                linestyle='dotted',
                ))
            self.ax.add_artist(matplotlib.lines.Line2D(
                [x1 - search_margin, x2 - search_margin], [y1, y2],
                color=v_colors[l_idx], linewidth=0.5*self.linewidth
                ))
            self.ax.add_artist(matplotlib.lines.Line2D(
                [x1 + search_margin, x2 + search_margin], [y1, y2],
                color=v_colors[l_idx], linewidth=0.5*self.linewidth
                 ))
            self.ax.add_artist(matplotlib.lines.Line2D(
                [x1 - search_margin, x1 + search_margin],
                [y1 + background_length, y1 + background_length],
                color=v_colors[l_idx], linewidth=0.5*self.linewidth
                ))
            self.ax.add_artist(matplotlib.lines.Line2D(
                [x2 - search_margin, x2 + search_margin],
                [y2 - background_length, y2 - background_length],
                color=v_colors[l_idx], linewidth=0.5*self.linewidth
                ))

    def SNI(self):
        """Draw NM uniformity ROI."""
        roi_idx = -1
        small_start_idx = 3  # index of roi_array where small ROIs start
        labels = None
        show_labels = False
        try:
            show_labels = self.main.tab_nm.sni_show_labels.isChecked()
        except AttributeError:
            pass
        if show_labels:
            labels = ['L1', 'L2']
        self.add_contours_to_all_rois(
            colors=['red', 'blue'], roi_indexes=[1, 2], labels=labels,
            labels_pos='upper_left_inside')

        if self.main.current_paramset.sni_type == 0:
            self.add_contours_to_all_rois(
                reset_contours=False,
                colors=['red', 'blue', 'red', 'blue', 'red', 'blue'],
                roi_indexes=[i for i in range(3, 9)],
                filled=True)

            if show_labels:
                for i in range(6):
                    mask = np.where(self.main.current_roi[i+3], 0, 1)
                    colors = 'k'
                    mask_pos = np.where(mask == 0)
                    xpos = np.mean(mask_pos[1])
                    ypos = np.mean(mask_pos[0])
                    if np.isfinite(xpos) and np.isfinite(ypos):
                        self.ax.text(xpos-self.fontsize, ypos+self.fontsize,
                                     f'S{i+1}',
                                     fontsize=self.fontsize, color=colors)
        elif self.main.current_paramset.sni_type in [1, 2]:
            try:
                for row, col in [
                        (small_start_idx, 0),
                        (small_start_idx+1, 1), (-2, -2), (-1, -1)]:
                    mask = np.where(self.main.current_roi[row][col], 0, 1)
                    self.contours.append(
                        self.ax.contourf(mask, levels=[0, 0.5], colors='red', alpha=0.3))
                    if show_labels:
                        mask_pos = np.where(mask == 0)
                        xmin = np.min(mask_pos[1])
                        xmean = np.mean(mask_pos[1])
                        xpos = int(0.75*xmin + 0.25*xmean)
                        ypos = np.mean(mask_pos[0]) + 2
                        if np.isfinite(xpos) and np.isfinite(ypos):
                            colno = col
                            rowno = row
                            
                            if col < 0:
                                colno = len(self.main.current_roi[row]) + col
                            if row < 0:
                                rowno = len(self.main.current_roi) + row
                            self.ax.text(xpos, ypos, f'r{rowno-small_start_idx}_c{colno}',
                                         fontsize=self.fontsize, color='k')
            except (IndexError, TypeError):
                pass
        else:  # 3 Siemens
            for rowno, rois_row in enumerate(self.main.current_roi[small_start_idx:]):
                for colno, roi in enumerate(rois_row):
                    if roi is not None:  # None if ignored
                        mask = np.where(roi, 0, 1)
                        self.contours.append(
                            self.ax.contour(
                                mask, levels=[0.9], colors='red',
                                alpha=0.5, linewidths=self.linewidth))
                        if show_labels:
                            mask_pos = np.where(mask == 0)
                            xmin = np.min(mask_pos[1])
                            xmean = np.mean(mask_pos[1])
                            xpos = int(0.75*xmin + 0.25*xmean)
                            ypos = np.mean(mask_pos[0]) + 2
                            if np.isfinite(xpos) and np.isfinite(ypos):
                                self.ax.text(xpos, ypos, f'r{rowno}_c{colno}',
                                             fontsize=self.fontsize, color='red')
        if self.main.current_paramset.sni_type > 0:
            try:
                roi_idx = int(self.main.tab_nm.sni_selected_roi.currentIndex())
            except AttributeError:
                pass
        if roi_idx > 1 and self.main.results:  # not for L1 or L2
            plot_txt = self.main.tab_nm.sni_plot.currentText()
            img_txt = self.main.tab_nm.sni_result_image.currentText()
            if 'selected' in plot_txt or 'selected' in img_txt:
                flat_list = [
                    item for row in self.main.current_roi[3:]
                    for item in row]
                try:
                    selected_roi = flat_list[roi_idx - 2]
                except IndexError:
                    selected_roi = None

                if selected_roi is not None:
                    self.contours.append(
                        self.ax.contour(
                            np.where(selected_roi, 0, 1),
                            levels=[0.9], colors='b', linestyles='dotted',
                            alpha=0.5, linewidths=1.5 * self.linewidth))

    def SNR(self):
        """Draw MR SNR ROI(s)."""
        if self.main.current_paramset.snr_type == 0:
            self.add_contours_to_all_rois()
        else:
            self.add_contours_to_all_rois(colors=['red', 'lime'])

    def TTF(self):
        """Draw TTF ROIs."""
        #self.contours = []
        ttf_table = self.main.current_paramset.ttf_table
        self.add_contours_to_all_rois(labels=ttf_table.labels, colors=COLORS)
        if 'TTF' in self.main.results:
            try:
                if 'details_dict' in self.main.results['TTF']:
                    for details_this in self.main.results['TTF']['details_dict']:
                        roi_disc = details_this['found_disc_roi']
                        mask = np.where(roi_disc, 0, 1)
                        contour = self.ax.contour(
                            mask, levels=[0.9],
                            colors='red', alpha=0.5, linewidths=self.linewidth,
                            linestyles='dotted')
                        self.contours.append(contour)
            except (TypeError, KeyError, IndexError):
                pass

    def Uni(self):
        """Draw NM uniformity ROI."""
        self.add_contours_to_all_rois(colors=['red', 'blue'])

    def Var(self):
        #  [roi1, roi2, roi_mask_outer, roi_mask_max]
        self.add_contours_to_all_rois(
            colors=['c', 'm', 'b'], roi_indexes=[0, 1, 2],
            linestyles=['solid', 'solid', 'dotted'])
        if self.main.current_roi[3] is not None:
            self.add_contours_to_all_rois(
                colors=['red'], roi_indexes=[3],
                filled=True, hatches=['////'], reset_contours=False)


class ResultImageCanvas(GenericImageCanvas):
    """Canvas for display of results as image."""

    def __init__(self, parent, main):
        super().__init__(parent, main)

    def result_image_draw(self, selected_text=''):
        """Refresh result image."""
        self.ax.clear()
        self.current_image = None
        self.cmap = 'gray'
        self.min_val = None
        self.max_val = None
        self.min_val_default = None
        self.max_val_default = None
        self.title = ''
        self.positive_negative = False  # display positive as red, negative as blue
        self.set_min_max = False  # Force window level to be min-max at first display
        self.contours_to_add = []

        if self.main.current_test in self.main.results:
            if self.main.results[self.main.current_test] is not None:
                class_method = getattr(self, self.main.current_test, None)
                if class_method is not None:
                    if not isinstance(selected_text, str):
                        selected_text = ''
                    class_method(selected_text)

        if self.current_image is not None:
            if self.min_val is None or self.max_val is None:
                if self.current_image.dtype == bool:
                    self.min_val = False
                    self.max_val = True
                elif self.set_min_max:
                    self.min_val = np.min(self.current_image)
                    self.max_val = np.max(self.current_image)
                    if self.max_val == self.min_val:
                        self.max_val += 1
                else:
                    minval, maxval = (
                        self.parent.wid_window_level.tb_wl.calculate_min_max(
                            mode_string=''))
                    if self.min_val is None:
                        self.min_val = minval
                    if self.max_val is None:
                        self.max_val = maxval

            if self.positive_negative:
                if self.min_val != - self.max_val:
                    maxv = np.max(np.abs([self.min_val, self.max_val]))
                    self.min_val = -maxv
                    self.max_val = maxv
                self.cmap = 'coolwarm'
            self.parent.wid_window_level.positive_negative = self.positive_negative

            proceed = True
            try:
                self.img = self.ax.imshow(
                    self.current_image,
                    cmap=self.cmap, vmin=self.min_val, vmax=self.max_val)
                self.parent.image_title.setText(
                    '<html><head/><body><p>'+self.title+'</p></body></html>')
                contrast = self.max_val - self.min_val
            except TypeError:
                proceed = False
            if proceed:
                self.parent.wid_window_level.decimals = 0
                if contrast < 5:
                    self.parent.wid_window_level.decimals = 2
                elif contrast < 20:
                    self.parent.wid_window_level.decimals = 1
                try:
                    minimg = np.min(self.current_image)
                    maximg = np.max(self.current_image)
                    rmin = round(minimg * 10 ** self.parent.wid_window_level.decimals)
                    rmax = round(maximg * 10 ** self.parent.wid_window_level.decimals)
                    self.parent.wid_window_level.min_wl.setRange(rmin, rmax)
                    self.parent.wid_window_level.max_wl.setRange(rmin, rmax)
                    if self.min_val_default is None:
                        self.min_val_default = self.min_val
                    if self.max_val_default is None:
                        self.max_val_default = self.max_val
                    self.parent.wid_window_level.update_window_level(
                        self.min_val_default, self.max_val_default,
                        cmap=self.cmap)
                    self.parent.wid_window_level.canvas.plot(
                        self.current_image,
                        decimals=self.parent.wid_window_level.decimals)
                except ValueError:
                    pass

            flatfield = False
            if self.main.current_modality == 'Mammo':
                flatfield = True
            elif self.main.current_modality == 'Xray':
                if self.main.current_paramset.hom_tab_alt >= 3:
                    flatfield = True
            if flatfield:
                self.mark_pixels()

            if self.contours_to_add:
                try:
                    self.linewidth = self.main.gui.annotations_line_thick
                except AttributeError:
                    pass  # default
                for contour_to_add in self.contours_to_add:
                    mask = np.where(contour_to_add[0], 0, 1)
                    contour = self.ax.contour(
                        mask, levels=[0.9],
                        colors=contour_to_add[1], alpha=0.5,
                        linewidths=self.linewidth,
                        linestyles=contour_to_add[2])

        else:
            self.img = self.ax.imshow(np.zeros((100, 100)))
            at = matplotlib.offsetbox.AnchoredText(
                'No result to display',
                prop=dict(size=14, color='gray'),
                frameon=False, loc='center')
            self.ax.add_artist(at)
            self.parent.image_title.setText('')
            self.parent.wid_window_level.colorbar.fig.clf()
            self.parent.wid_window_level.canvas.fig.clear()
        if self.main.gui.show_axis is False:
            self.ax.axis('off')
            self.fig.subplots_adjust(.0, .0, 1., 1.)
        else:
            self.fig.subplots_adjust(.05, .05, 1., 1.)
        self.draw()

    def zoom_as_active_image(self):
        xlim = None
        ylim = None
        act_canvas = self.main.wid_image_display.canvas
        try:
            act_img = act_canvas.ax.get_images()[0].get_array()
            xlim = act_canvas.ax.get_xlim()
            ylim = act_canvas.ax.get_ylim()
            if self.current_image.shape != act_img.shape:
                if self.main.current_test in ['Hom', 'Var']:
                    if self.current_image.shape[0] < 0.5*act_img.shape[0]:
                        xlim, ylim = None, None
                    else:
                        # zoom relative to center (masked edge)
                        xlim = (np.array(xlim) - act_img.shape[1] / 2
                                + self.current_image.shape[1] / 2)
                        ylim = (np.array(ylim) - act_img.shape[0] / 2
                                + self.current_image.shape[0] / 2)
                else:
                    xlim, ylim = None, None
                if xlim is None:
                    QMessageBox.information(
                        self.parent, 'Not same size',
                        'Matched zoom not available when result image not '
                        'same size as active image.')
                    xlim, ylim = None, None
        except (AttributeError, IndexError):
            pass
        if xlim is not None and ylim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.draw_idle()

    def mark_pixels(self):
        if 'Hom' in self.main.results and self.main.current_test == 'Hom':
            if self.current_image.shape == self.main.active_img.shape:
                try:
                    details_dict = self.main.results['Hom']['details_dict'][
                        self.main.gui.active_img_no]
                except (IndexError, KeyError):
                    details_dict = None
                if details_dict:
                    if 'deviating_pixel_coordinates' in details_dict:
                        for coord in details_dict['deviating_pixel_coordinates']:
                            self.ax.add_patch(patches.Circle(
                                coord, radius=20, color='r', fill=False))
        elif 'Def' in self.main.results and self.main.current_test == 'Def':
            try:
                details_dict = self.main.results['Def']['details_dict']
            except KeyError:
                details_dict = None
            if details_dict:
                if 'mark_pixels' in details_dict:
                    for coord in details_dict['mark_pixels']:
                        self.ax.add_patch(patches.Circle(
                            coord, radius=20, color='r', fill=False))

    def set_cdm_cell_display(self, xpos, ypos):
        """Set result image to cell closes to xpos, ypos in image."""
        # find closest row, col
        # block signals, set diameter, and set thickness unblock signals
        try:
            details_dict = self.main.results['CDM']['details_dict'][
                self.main.gui.active_img_no]
        except (IndexError, KeyError):
            details_dict = None
        if details_dict:
            diff_xs = np.abs(self.main.current_roi['xs'] - xpos)
            diff_ys = np.abs(self.main.current_roi['ys'] - ypos)
            tolerance = 0.5 * self.main.current_roi['cell_width']
            min_xs = np.where(diff_xs < tolerance)
            min_ys = np.where(diff_ys < tolerance)
            row, col = None, None
            if min_xs[0].shape[0] > 0 and min_ys[0].shape[0] > 0:
                if self.main.current_roi['phantom'] == 40:
                    row = min_xs[0][0]
                    col = min_ys[1][0]
                else:
                    pos1 = list(zip(list(min_xs[0]), list(min_xs[1])))
                    pos2 = list(zip(list(min_ys[0]), list(min_ys[1])))
                    match = list(set(pos1).intersection(pos2))
                    row, col = match[0]
                if row is not None and col is not None:
                    self.main.tab_mammo.blockSignals(True)
                    self.main.tab_mammo.cdm_cbox_thickness.setCurrentIndex(row)
                    self.main.tab_mammo.cdm_cbox_diameter.setCurrentIndex(col)
                    self.main.tab_mammo.blockSignals(False)
                    self.result_image_draw()

    def get_default_display(self, testcode, sel_text, default_cmap='gray'):
        defaults = self.main.current_paramset.result_image_defaults
        self.found_default = False
        tests = [x.test for x in defaults]
        if testcode in tests:
            sel_texts = [x.selected_text for x in defaults
                         if x.test == testcode]
            if sel_text in sel_texts or sel_text == '':
                self.found_default = True
                obj = [x for x in defaults if x.test == testcode][0]
                if obj.set_min:
                    self.min_val_default = obj.cmin
                if obj.set_max:
                    self.max_val_default = obj.cmax
                self.cmap = obj.cmap
                self.positive_negative = False
        if self.found_default is False:
            self.cmap = default_cmap

    def CDM(self, sel_text):
        self.get_default_display('CDM', sel_text, 'gray')
        try:
            details_dict = self.main.results['CDM']['details_dict'][
                self.main.gui.active_img_no]
        except (IndexError, KeyError):
            details_dict = None
        if details_dict:
            idx_diam = self.main.tab_mammo.cdm_cbox_diameter.currentIndex()
            idx_thick = self.main.tab_mammo.cdm_cbox_thickness.currentIndex()
            res = details_dict['res_table'][idx_thick][idx_diam]
            if res:
                annotate = self.main.gui.annotations
                if annotate:
                    img_canvas = self.main.wid_image_display.canvas
                    try:
                        scatter_sel = img_canvas.scatters[-1]
                        scatter_sel.remove()
                        x = self.main.current_roi['xs'][idx_thick, idx_diam]
                        y = self.main.current_roi['ys'][idx_thick, idx_diam]
                        scatter = img_canvas.ax.scatter(
                            x, y, s=40, c='blue', marker='+')
                        img_canvas.scatters[1] = scatter
                        img_canvas.draw_idle()
                    except:
                        img_canvas.roi_draw()

                self.current_image = res['processed_sub']
                self.min_val = np.min(self.current_image)
                self.max_val = np.max(self.current_image)

                diameter = self.main.tab_mammo.cdm_cbox_diameter.currentText()
                thickness = self.main.results['CDM']['details_dict'][-1]['thickness']
                if self.main.current_roi['phantom'] == 40:
                    thick_txt = thickness[idx_thick][idx_diam]
                else:
                    thick_txt = self.main.tab_mammo.cdm_cbox_thickness.currentText()

                self.title = (
                    f'Processed sub-image averaged by kernel, disc with '
                    f'diameter {diameter} mm, thickess {thick_txt} &mu;m')

                if self.main.tab_mammo.cdm_chk_show_kernel.isChecked():
                    sz_sub = self.current_image.shape[0]
                    kernel = (details_dict['kernels'][idx_diam] > 0)
                    greens = np.zeros((sz_sub, sz_sub), dtype=bool)
                    reds = np.zeros((sz_sub, sz_sub), dtype=bool)
                    yellows = np.zeros((sz_sub, sz_sub), dtype=bool)

                    sz_k = kernel.shape[0]
                    radk = sz_k // 2

                    yk, xk = res['min_positions'][0]
                    if res['central_disc_found']:
                        greens[yk - radk:yk - radk + sz_k,
                               xk - radk:xk - radk + sz_k] = kernel
                    else:
                        reds[yk - radk:yk - radk + sz_k,
                             xk - radk:xk - radk + sz_k] = kernel

                    corner_idx = res['corner_index'][0]
                    if self.main.current_roi['phantom'] == 40:
                        corner_idx += 1

                    yk, xk = res['min_positions'][corner_idx]
                    y1 = max([0, yk - radk])
                    y2 = min([yk - radk + sz_k, sz_sub])
                    x1 = max([0, xk - radk])
                    x2 = min([xk - radk + sz_k, sz_sub])
                    ky1, ky2, kx1, kx2 = 0, sz_k, 0, sz_k
                    if greens[y1:y2,x1:x2].shape != kernel.shape:
                        if yk - radk < 0:
                            ky1 = - (yk - radk)
                        if xk - radk < 0:
                            kx1 = - (xk - radk)
                        if yk - radk + sz_k > sz_sub:
                            ky2 = sz_k - ((yk - radk + sz_k) - sz_sub)
                        if xk - radk + sz_k > sz_sub:
                            kx2 = sz_k - ((xk - radk + sz_k) - sz_sub)
                    try:
                        if details_dict['found_correct_corner'][idx_thick, idx_diam]:
                            greens[y1:y2,x1:x2] = kernel[ky1:ky2,kx1:kx2]
                        else:
                            reds[y1:y2,x1:x2] = kernel[ky1:ky2,kx1:kx2]
                    except ValueError:
                        pass

                    if res['end'] != 0:
                        for i in range(0,5):
                            yellows = yellows + details_dict['templates'][i][
                                res['start']:res['end'], 
                                res['start']:res['end']]
                    else:
                        for i in range(0,5):
                            yellows = yellows + details_dict['templates'][i]
                    
                    self.contours_to_add.append([yellows, 'yellow', '-'])
                    self.contours_to_add.append([greens, 'green', '-'])
                    self.contours_to_add.append([reds, 'red', '-'])

    def Def(self, sel_text):
        """Defective pixels."""
        try:
            details_dicts = self.main.results['Def']['details_dict']
        except KeyError:
            details_dicts = None
        if details_dicts:
            if sel_text == '':
                sel_txt = self.main.tab_xray.def_result_image.currentText()
            self.title = sel_txt

            try:
                def_cmap = 'viridis'
                if sel_txt == 'Pix == Avg of 8 neighbours':
                    self.current_image = details_dicts[
                        self.main.gui.active_img_no]['diff_neighbours_is_zero']
                    def_cmap = 'RdYlGn_r'
                elif sel_txt == 'Pix == Avg of 4 neighbours':
                    self.current_image = details_dicts[
                        self.main.gui.active_img_no]['diff_nearest_is_zero']
                    def_cmap = 'RdYlGn_r'
                elif sel_txt == '# pix == avg of 8 pr 1x1cm':
                    self.current_image = details_dicts[
                        self.main.gui.active_img_no]['n_pr_roi_neighbours']
                elif sel_txt == '# pix == avg of 4 pr 1x1cm':
                    self.current_image = details_dicts[
                        self.main.gui.active_img_no]['n_pr_roi_nearest']
                elif sel_txt == 'Fraction of images where avg of 8 neighbours':
                    self.current_image = details_dicts[
                        -1]['frac_diff_neighbours_is_zero']
                elif sel_txt == 'Fraction of images where avg of 4 neighbours':
                    self.current_image = details_dicts[
                        -1]['frac_diff_nearest_is_zero']

                self.get_default_display('Def', sel_text, def_cmap)
            except KeyError:
                pass

    def Foc(self, sel_text):
        try:
            details_dict = self.main.results['Foc']['details_dict'][
                self.main.gui.active_img_no]
            self.get_default_display('Foc', sel_text, 'viridis')
            self.title = 'Variance image cropped to star pattern'
            self.current_image = details_dict['variance_cropped']

            self.contours_to_add.append(
                [details_dict['roi_found_x'], 'r', '-'])
            self.contours_to_add.append(
                [details_dict['roi_found_y'], 'b', '-'])

        except (KeyError, IndexError):
            pass

    def Hom(self, sel_txt):
        """Prepare images of Mammo-Homogeneity."""
        flatfield = False
        if self.main.current_modality == 'Mammo':
            flatfield = True
        elif self.main.current_modality == 'Xray':
            if self.main.current_paramset.hom_tab_alt >= 3:
                flatfield = True

        if flatfield:
            def_cmap = 'viridis'
            try:
                details_dict = self.main.results['Hom']['details_dict'][
                    self.main.gui.active_img_no]
            except (IndexError, KeyError):
                details_dict = None
            if details_dict:
                aapm = False
                if self.main.current_modality == 'Mammo':
                    if sel_txt == '':
                        sel_txt = self.main.tab_mammo.hom_result_image.currentText()
                else:
                    if self.main.current_paramset.hom_tab_alt == 3:
                        if sel_txt == '':
                            sel_txt = self.main.tab_xray.hom_result_image.currentText()
                    else:
                        if sel_txt == '':
                            sel_txt = self.main.tab_xray.hom_result_image_aapm.currentText()
                        aapm = True
                self.title = sel_txt

                try:  # flatfield mammo and aapm variants
                    if sel_txt == 'Average pr ROI map':
                        self.current_image = details_dict['averages']
                    elif sel_txt == 'Noise pr ROI map':
                        self.current_image = details_dict['stds']
                    elif sel_txt == 'SNR pr ROI map':
                        if 'snrs' in details_dict:
                            self.current_image = details_dict['snrs']
                        elif 'stds' in details_dict:
                            self.current_image = np.divide(
                                details_dict['averages'], details_dict['stds'])
                    elif sel_txt == 'Average variance pr ROI map':
                        self.current_image = details_dict['variances']
                    elif sel_txt == 'Average pr ROI (% difference from global average)':
                        if 'diff_averages' in details_dict:
                            self.current_image = details_dict['diff_averages']
                        else:
                            overall_avg = np.mean(details_dict['averages'])
                            diff_avgs = details_dict['averages'] - overall_avg
                            self.current_image = 100 / overall_avg * diff_avgs
                        self.positive_negative = True
                        if aapm:
                            self.min_val_default = -5
                            self.max_val_default = 5
                    elif sel_txt == 'Noise pr ROI (% difference from average noise)':
                        overall_avg = np.mean(details_dict['stds'])
                        diff_avgs = details_dict['stds'] - overall_avg
                        self.current_image = 100 / overall_avg * diff_avgs
                        self.positive_negative = True
                    elif sel_txt == 'SNR pr ROI (% difference from global SNR)':
                        self.current_image = details_dict['diff_snrs']
                        self.positive_negative = True
                    elif sel_txt == 'SNR pr ROI (% difference from average SNR)':
                        snrs = np.divide(
                            details_dict['averages'], details_dict['stds'])
                        overall_avg = np.mean(snrs)
                        diff_avgs = snrs - overall_avg
                        self.current_image = 100 / overall_avg * diff_avgs
                        self.positive_negative = True
                    elif sel_txt == 'Local Uniformity map':
                        self.current_image = details_dict['local_uniformities']
                        if aapm:
                            self.min_val_default = -0.1
                            self.max_val_default = 0.1
                    elif sel_txt == 'Local Noise Uniformity map':
                        self.current_image = details_dict['local_noise_uniformities']
                    elif sel_txt == 'Local SNR Uniformity map':
                        self.current_image = details_dict['local_snr_uniformities']
                    elif sel_txt == 'Pixel values (% difference from global average)':
                        self.current_image = details_dict['diff_pixels']
                        self.positive_negative = True
                    elif sel_txt == 'Deviating ROIs':
                        self.current_image = details_dict['deviating_rois']
                        def_cmap = 'RdYlGn_r'
                    elif sel_txt == 'Deviating pixels':
                        self.current_image = details_dict['deviating_pixels']
                        def_cmap = 'RdYlGn_r'
                    elif sel_txt == '# deviating pixels pr ROI':
                        self.current_image = details_dict['deviating_pixel_clusters']
                        self.set_min_max = True
                    elif sel_txt == 'Anomalous pixels':
                        if details_dict['n_anomalous_pixels'] > 0:
                            self.current_image = details_dict['anomalous_pixels']
                            def_cmap = 'RdYlGn_r'
                    elif sel_txt == '# anomalous pixels pr ROI':
                        self.current_image = details_dict['n_anomalous_pixels_pr_roi']
                        self.set_min_max = True
                    self.get_default_display('Hom', sel_txt, def_cmap)

                except KeyError:
                    pass

    def NPS(self, sel_text):
        """Prepare result image for test NPS."""
        try:
            details_dict = self.main.results['NPS']['details_dict'][
                self.main.gui.active_img_no]
        except KeyError:
            details_dict = {}

        if self.main.current_modality == 'CT':
            self.title = '2d Noise Power Spectrum - average of ROIs'
            if 'NPS_array' in details_dict:
                self.current_image = details_dict['NPS_array']
        elif self.main.current_modality in ['Xray', 'Mammo']:
            if sel_text == '':
                test_widget = self.main.stack_test_tabs.currentWidget()
                sel_text = test_widget.nps_result_image.currentText()
            if 'NPS' in sel_text:
                self.title = '2d Noise Power Spectrum - average of all ROIs'
                if 'NPS_array' in details_dict:
                    self.current_image = details_dict['NPS_array']
            else:
                self.title = 'Large area correct for second order 2d polynomial trend'
                if 'trend_corrected_sub_matrix' in details_dict:
                    self.current_image = details_dict['trend_corrected_sub_matrix']
        self.get_default_display('NPS', sel_text, 'viridis')

    def Rin(self, sel_text):
        """Prepare result image for test Rin."""
        try:
            details_dict = self.main.results['Rin']['details_dict'][
                self.main.gui.active_img_no]
        except KeyError:
            details_dict = {}
        self.get_default_display('Rin', sel_text, 'viridis')
        if self.main.current_paramset.rin_sigma_image > 0:
            self.title = 'Gaussian filtered and masked image'
        else:
            self.title = 'Masked image'
        if 'processed_image' in details_dict:
            self.current_image = details_dict['processed_image']

    def SNI(self, sel_txt):
        """Prepare result image for test SNI."""
        if self.main.current_paramset.sni_sum_first:
            try:
                details_dict = self.main.results['SNI']['details_dict'][0]
            except KeyError:
                details_dict = {}
        else:
            try:
                details_dict = self.main.results['SNI']['details_dict'][
                    self.main.gui.active_img_no]
            except KeyError:
                details_dict = {}
        if details_dict:
            if sel_txt == '':
                sel_txt = self.main.tab_nm.sni_result_image.currentText()
            if 'Curvature' in sel_txt:
                self.title = 'Curvature corrected image'
                if 'corrected_image' in details_dict:
                    self.current_image = details_dict['corrected_image']
            elif 'Summed' in sel_txt:
                self.title = 'Summed image'
                if 'sum_image' in details_dict:
                    self.current_image = details_dict['sum_image']
            elif '2d NPS' in sel_txt:
                image_text = self.main.tab_nm.sni_result_image.currentText()
                roi_txt = self.main.tab_nm.sni_selected_roi.currentText()
                self.title = f'{image_text} for ROI {roi_txt}'
                roi_idx = self.main.tab_nm.sni_selected_roi.currentIndex()
                if self.main.current_paramset.sni_type > 0:
                    self.main.wid_image_display.canvas.roi_draw()
                try:
                    details_this = details_dict['pr_roi'][roi_idx]
                    self.current_image = details_this['NPS']
                except (IndexError, KeyError, TypeError):
                    self.current_image = None
            elif 'SNI values map' in sel_txt:
                self.title = 'SNI values map'
                if 'SNI_map' in details_dict:
                    show_filter_2 = False
                    if self.main.current_paramset.sni_channels:
                        if self.main.tab_nm.sni_plot_low.btn_false.isChecked():
                            show_filter_2 = True
                    suffix_2 = '_2' if show_filter_2 else ''
                    if self.main.current_paramset.sni_type == 3:  # Siemens
                        self.current_image = generate_SNI_Siemens_image(
                            details_dict[f'SNI_map{suffix_2}'])
                    else:
                        self.current_image = details_dict[f'SNI_map{suffix_2}']
                    self.min_val = 0
                    max_in_res = np.max([
                        row for row in self.main.results['SNI']['values']
                        if len(row) > 0])
                    if max_in_res > 0:
                        self.max_val = max_in_res
            self.get_default_display('SNI', sel_txt, 'viridis')

    def Swe(self, sel_text):
        try:
            details_dicts = self.main.results['Swe']['details_dict']
        except KeyError:
            details_dicts = []
        if sel_text == '':
            sel_text = self.main.tab_nm.swe_result_image.currentText()
        idx = 0 if '1' in sel_text else 1
        self.title = sel_text
        resize_x = details_dicts[0]['resize_x']

        def resize_aspect(img):
            res_img = resize(img, (img.shape[0], resize_x),
                             anti_aliasing=False, order=0)
            return res_img

        if 'FWHM' in sel_text:
            self.current_image = resize_aspect(
                details_dicts[idx]['fwhm_matrix'])
            self.min_val = np.nanmin(self.current_image[self.current_image>0])
            self.max_val = np.nanmax(self.current_image)
        elif 'line position' in sel_text:
            self.positive_negative = True
            self.current_image = resize_aspect(
                details_dicts[idx]['diff_matrix'])
            self.min_val = -3.
            self.max_val = 3.
        elif 'Sum detector' in sel_text:
            self.current_image = details_dicts[idx]['sum_matrix']
        elif 'Processed' in sel_text:
            self.current_image = details_dicts[idx]['ufov_matrix']
        elif 'Differential' in sel_text:
            self.current_image = details_dicts[idx]['du_matrix']
        self.get_default_display('Swe', sel_text, 'viridis')

    def Uni(self, sel_text):
        """Prepare result image for test Uni."""
        if self.main.current_paramset.uni_sum_first:
            try:
                details_dict = self.main.results['Uni']['details_dict'][0]
            except (KeyError, IndexError):
                details_dict = {}
        else:
            try:
                details_dict = self.main.results['Uni']['details_dict'][
                    self.main.gui.active_img_no]
            except (KeyError, IndexError):
                details_dict = {}

        if sel_text == '':
            sel_text = self.main.tab_nm.uni_result_image.currentText()

        set_min_max_avoid_zero = False
        if 'Differential' in  sel_text:
            self.title = 'Differential uniformity map in UFOV (max in x/y direction)'
            if 'du_matrix' in details_dict:
                self.current_image = details_dict['du_matrix']
                self.min_val = np.nanmin(self.current_image)
                self.max_val = np.nanmax(self.current_image)
        elif 'Processed' in sel_text:
            if 'pix_size' in details_dict:
                pix_sz = details_dict['pix_size']
                self.title = f'Processed image {pix_sz:0.2f} mm pr pix, UFOV part'
            else:
                self.title = 'Processed image, ~6.4 mm pr pix, UFOV part'
            if 'matrix_ufov' in details_dict:
                self.current_image = details_dict['matrix_ufov']
                set_min_max_avoid_zero = True
        elif 'Curvature corrected' in sel_text:
            self.title = 'Curvature corrected image'
            if 'corrected_image' in details_dict:
                self.current_image = details_dict['corrected_image']
                mean = np.mean(self.current_image[self.current_image != 0])
                stdev = np.std(self.current_image[self.current_image != 0])
                self.min_val = mean - stdev
                self.max_val = mean + stdev
        elif 'Summed' in sel_text:
            self.title = 'Summed image'
            if 'sum_image' in details_dict:
                self.current_image = details_dict['sum_image']
                set_min_max_avoid_zero = True
        if set_min_max_avoid_zero:
            max_val = np.nanmax(self.current_image)
            if max_val > 0:
                self.max_val = max_val
                non_zero = self.current_image[self.current_image != 0]
                self.min_val = np.nanmin(non_zero)
        self.get_default_display('Uni', sel_text, 'viridis')

    def Var(self, sel_text):
        """Prepare variance image."""
        try:
            details_dict = self.main.results['Var']['details_dict'][
                self.main.gui.active_img_no]
            self.title = 'Variance image'
            if self.main.current_modality == 'Mammo':
                if sel_text == '':
                    sel_text = self.main.tab_mammo.var_result_image.currentText()
            else:
                if sel_text == '':
                    sel_text = self.main.tab_xray.var_result_image.currentText()
            sel_idx = 0 if 'ROI 1' in sel_text else 1
            self.current_image = details_dict['variance_image'][sel_idx]
            self.get_default_display('Var', sel_text, 'viridis')

        except (KeyError, IndexError, TypeError):
            pass
