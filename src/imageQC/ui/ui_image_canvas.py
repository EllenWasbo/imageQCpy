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

# imageQC block start
from imageQC.scripts.mini_methods_calculate import get_min_max_pos_2d
from imageQC.scripts.calculate_roi import generate_SNI_Siemens_image
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
        self.profile_length = 20  # assume click drag > length in pix = draw profile
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

    def img_draw(self, auto=False, window_level=[]):
        """Refresh image."""
        # keep previous zoom if same size image
        xlim = None
        ylim = None
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
        if annotate:
            try:
                linewidth = self.main.gui.annotations_line_thick
            except AttributeError:
                linewidth = 1.
            # central crosshair
            szy, szx = np.shape(nparr)
            if self.main.gui.delta_a == 0:
                self.ax.axhline(
                    y=szy*0.5 + self.main.gui.delta_y,
                    color='red', linewidth=linewidth, linestyle='--')
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
                    pass
            at = matplotlib.offsetbox.AnchoredText(
                '\n'.join(annot_text),
                prop=dict(size=self.main.gui.annotations_font_size, color='red'),
                frameon=False, loc='upper left')
            self.ax.add_artist(at)
            self.roi_draw()
        else:
            self.draw()
        self.current_image = nparr

    def remove_annotations(self):
        """Remove current annotations."""
        for contour in self.contours:
            try:
                for coll in contour.collections:
                    try:
                        coll.remove()
                    except ValueError:
                        pass
            except AttributeError:
                pass

        for scatter in self.scatters:
            try:
                scatter.remove()
            except ValueError:
                pass
        if hasattr(self.ax, 'lines'):
            n_lines = len(self.ax.lines)
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

    def roi_draw(self):
        """Update ROI countours on image."""
        self.remove_annotations()

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
                                 linestyles='solid',
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
        linestyles : str, optional
            fx 'dotted'. Default is 'solid'
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
        if roi_indexes is None:
            roi_indexes = list(np.arange(len(this_roi)))

        for color_no, roi_no in enumerate(roi_indexes):
            mask = np.where(this_roi[roi_no], 0, 1)
            if filled:
                if hatches is None:
                    contour = self.ax.contourf(
                        mask, levels=[0, 0.5], colors=colors[color_no], alpha=0.3)
                else:
                    contour = self.ax.contourf(
                        mask, levels=[0, 0.5], colors='none',
                        hatches=hatches[color_no])
                    contour.collections[0].set_edgecolor(colors[color_no])
            else:
                contour = self.ax.contour(
                    mask, levels=[0.9],
                    colors=colors[color_no], alpha=0.5, linewidths=self.linewidth,
                    linestyles=linestyles)
            if labels:
                try:
                    label = labels[color_no]
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
            self.contours.append(contour)

    def Bar(self):
        """Draw Bar ROIs."""
        self.add_contours_to_all_rois(
            colors=COLORS,
            labels=[str(i+1) for i in range(4)]
            )

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
            if 'details_dict' in self.main.results['Dim']:
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

    def Gho(self):
        """Draw Ghosting ROIs."""
        try:
            labels = self.main.current_paramset.gho_table.labels
        except AttributeError:
            labels = None
        self.add_contours_to_all_rois(colors=COLORS, labels=labels)

    def Hom(self):
        """Draw Hom ROI."""
        if self.main.current_modality == 'Mammo':
            self.add_contours_to_all_rois(colors=['blue', 'blue'], roi_indexes=[0, 1])
            if self.main.current_paramset.hom_mask_max:
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
                self.main.current_modality in ['CT', 'SPECT']
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
                roi_indexes = list(range(len(self.main.current_roi) - 1))
                self.add_contours_to_all_rois(
                    roi_indexes=roi_indexes,
                    colors=COLORS)
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


class ResultImageCanvas(GenericImageCanvas):
    """Canvas for display of results as image."""

    def __init__(self, parent, main):
        super().__init__(parent, main)

    def result_image_draw(self):
        """Refresh result image."""
        self.ax.clear()
        self.current_image = None
        self.cmap = 'gray'
        self.min_val = None
        self.max_val = None
        self.title = ''
        self.positive_negative = False  # display positive as red, negative as blue

        if self.main.current_test in self.main.results:
            if self.main.results[self.main.current_test] is not None:
                class_method = getattr(self, self.main.current_test, None)
                if class_method is not None:
                    class_method()

        if self.current_image is not None:
            if self.min_val is None:
                self.min_val = np.amin(self.current_image)
            if self.max_val is None:
                self.max_val = np.amax(self.current_image)
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
                self.parent.image_title.setText(self.title)
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
                    rmin = round(self.min_val * 10 ** self.parent.wid_window_level.decimals)
                    rmax = round(self.max_val * 10 ** self.parent.wid_window_level.decimals)
                    self.parent.wid_window_level.min_wl.setRange(rmin, rmax)
                    self.parent.wid_window_level.max_wl.setRange(rmin, rmax)
                    self.parent.wid_window_level.update_window_level(
                        self.min_val, self.max_val, cmap=self.cmap)
                    self.parent.wid_window_level.canvas.plot(
                        self.current_image,
                        decimals=self.parent.wid_window_level.decimals)
                except ValueError:
                    pass
            if self.main.current_modality == 'Mammo':
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

    def Hom(self):
        """Prepare images of Mammo-Homogeneity."""
        if self.main.current_modality == 'Mammo':
            self.cmap = 'viridis'
            try:
                details_dict = self.main.results['Hom']['details_dict'][
                    self.main.gui.active_img_no]
            except (IndexError, KeyError):
                details_dict = None
            if details_dict:
                sel_txt = self.main.tab_mammo.hom_result_image.currentText()
                self.title = sel_txt
                try:
                    if sel_txt == 'Average pr ROI map':
                        self.current_image = details_dict['averages']
                    elif sel_txt == 'SNR pr ROI map':
                        self.current_image = details_dict['snrs']
                    elif sel_txt == 'Variance pr ROI map':
                        self.current_image = details_dict['variances']
                    elif sel_txt == 'Average pr ROI (% difference from global average)':
                        self.current_image = details_dict['diff_averages']
                        self.positive_negative = True
                    elif sel_txt == 'SNR pr ROI (% difference from global SNR)':
                        self.current_image = details_dict['diff_snrs']
                        self.positive_negative = True
                    elif sel_txt == 'Pixel values (% difference from global average)':
                        self.current_image = details_dict['diff_pixels']
                        self.positive_negative = True
                    elif sel_txt == 'Deviating ROIs':
                        self.current_image = details_dict['deviating_rois']
                        self.cmap = 'RdYlGn_r'
                    elif sel_txt == 'Deviating pixels':
                        self.current_image = details_dict['deviating_pixels']
                        self.cmap = 'RdYlGn_r'
                    elif sel_txt == '# deviating pixels pr ROI':
                        self.current_image = details_dict['deviating_pixel_clusters']
                except KeyError:
                    pass

    def NPS(self):
        """Prepare result image for test NPS."""
        try:
            details_dict = self.main.results['NPS']['details_dict'][
                self.main.gui.active_img_no]
        except KeyError:
            details_dict = {}
        self.cmap = 'viridis'

        if self.main.current_modality == 'CT':
            self.title = '2d Noise Power Spectrum - average of ROIs'
            if 'NPS_array' in details_dict:
                self.current_image = details_dict['NPS_array']
        elif self.main.current_modality in ['Xray', 'Mammo']:
            test_widget = self.main.stack_test_tabs.currentWidget()
            sel_text = test_widget.nps_show_image.currentText()
            if 'NPS' in sel_text:
                self.title = '2d Noise Power Spectrum - average of all ROIs'
                if 'NPS_array' in details_dict:
                    self.current_image = details_dict['NPS_array']
            else:
                self.title = 'Large area correct for second order 2d polynomial trend'
                if 'trend_corrected_sub_matrix' in details_dict:
                    self.current_image = details_dict['trend_corrected_sub_matrix']

    def Rin(self):
        """Prepare result image for test Rin."""
        try:
            details_dict = self.main.results['Rin']['details_dict'][
                self.main.gui.active_img_no]
        except KeyError:
            details_dict = {}
        self.cmap = 'viridis'
        if self.main.current_paramset.rin_sigma_image > 0:
            self.title = 'Gaussian filtered and masked image'
        else:
            self.title = 'Masked image'
        if 'processed_image' in details_dict:
            self.current_image = details_dict['processed_image']

    def SNI(self):
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
            self.cmap = 'viridis'
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
                    if self.main.current_paramset.sni_type == 3:  # Siemens
                        self.current_image = generate_SNI_Siemens_image(
                            details_dict['SNI_map'])
                    else:
                        self.current_image = details_dict['SNI_map']
                    self.min_val = 0
                    max_in_res = np.max([
                        row for row in self.main.results['SNI']['values']
                        if len(row) > 0])
                    if max_in_res > 0:
                        self.max_val = max_in_res

    def Uni(self):
        """Prepare result image for test Uni."""
        if self.main.current_paramset.uni_sum_first:
            try:
                details_dict = self.main.results['Uni']['details_dict'][0]
            except KeyError:
                details_dict = {}
        else:
            try:
                details_dict = self.main.results['Uni']['details_dict'][
                    self.main.gui.active_img_no]
            except KeyError:
                details_dict = {}
        self.cmap = 'viridis'
        type_img = self.main.tab_nm.uni_result_image.currentIndex()
        set_min_max_avoid_zero = False
        if type_img == 0:
            self.title = 'Differential uniformity map in UFOV (max in x/y direction)'
            if 'du_matrix' in details_dict:
                self.current_image = details_dict['du_matrix']
                self.min_val = np.nanmin(self.current_image)
                self.max_val = np.nanmax(self.current_image)
        elif type_img == 1:
            if 'pix_size' in details_dict:
                pix_sz = details_dict['pix_size']
                self.title = f'Processed image {pix_sz:0.2f} mm pr pix, UFOV part'
            else:
                self.title = 'Processed image, ~6.4 mm pr pix, UFOV part'
            if 'matrix_ufov' in details_dict:
                self.current_image = details_dict['matrix_ufov']
                set_min_max_avoid_zero = True
        elif type_img == 2:
            self.title = 'Curvature corrected image'
            if 'corrected_image' in details_dict:
                self.current_image = details_dict['corrected_image']
                mean = np.mean(self.current_image[self.current_image != 0])
                stdev = np.std(self.current_image[self.current_image != 0])
                self.min_val = mean - stdev
                self.max_val = mean + stdev
        elif type_img == 3:
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

    def Var(self):
        """Prepare variance image."""
        try:
            details_dict = self.main.results['Var']['details_dict'][
                self.main.gui.active_img_no]
            self.cmap = 'viridis'
            self.title = (
                f'Variance image of central {self.main.current_paramset.var_percent} %')
            self.current_image = details_dict['variance_image']
        except KeyError:
            pass
