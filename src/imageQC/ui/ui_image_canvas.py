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

# imageQC block start
from imageQC.scripts.mini_methods_calculate import get_min_max_pos_2d
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
        #self.setParent = parent
        self.ax = self.fig.add_subplot(111)
        self.last_clicked_pos = (-1, -1)
        self.profile_length = 20  # assume click drag > length in pix = draw profile
        self.current_image = None

        # default display
        self.img = self.ax.imshow(np.zeros((2, 2)))
        self.ax.cla()
        self.ax.axis('off')

        # intialize parameters to make pylint happy
        self.contours = []
        self.scatters = []
        self.linewidth = 2
        self.fontsize = 10
        self.current_image = None
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
                self.draw()
                plotstatus = True
        return plotstatus

    def profile_remove(self):
        """Clear profile line."""
        if hasattr(self.ax, 'lines'):
            for line in self.ax.lines:
                if line.get_gid() == 'profile':
                    line.remove()
                    break

    def draw(self):
        """Avoid super().draw when figure collapsed by sliders."""
        try:
            super().draw()
        except ValueError:
            pass


class ImageCanvas(GenericImageCanvas):
    """Canvas for drawing the active DICOM image."""

    def __init__(self, parent, main):
        super().__init__(parent, main)

    def img_is_missing(self):
        """Show message when pixel_data is missing."""
        self.ax.cla()
        self.img = self.ax.imshow(np.zeros((100, 100)))
        self.ax.axis('off')
        at = matplotlib.offsetbox.AnchoredText(
            'Pixel data not found',
            prop=dict(size=14, color='gray'),
            frameon=False, loc='center')
        self.ax.add_artist(at)
        self.draw()

    def img_draw(self, auto=False, window_level=[]):
        """Refresh image."""
        self.ax.cla()

        nparr = self.main.active_img
        if auto is False:
            wl_min, wl_max = self.main.wid_window_level.get_min_max()
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

        if len(np.shape(nparr)) == 2:
            self.img = self.ax.imshow(
                nparr, cmap='gray', vmin=wl_min, vmax=wl_max)
        elif len(np.shape(nparr)) == 3:
            # rgb to grayscale NTSC formula
            nparr = (0.299 * nparr[:, :, 0]
                     + 0.587 * nparr[:, :, 1]
                     + 0.114 * nparr[:, :, 2])
            self.img = self.ax.imshow(nparr, cmap='gray')
            annotate = False
        self.ax.axis('off')
        if annotate:
            # central crosshair
            szy, szx = np.shape(nparr)
            if self.main.gui.delta_a == 0:
                self.ax.axhline(
                    y=szy*0.5 + self.main.gui.delta_y,
                    color='red', linewidth=1., linestyle='--')
                self.ax.axvline(
                    x=szx*0.5 + self.main.gui.delta_x,
                    color='red', linewidth=1., linestyle='--')
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
                    color='red', linewidth=1., linestyle='--',
                    gid='axis1'))
                self.ax.add_artist(matplotlib.lines.Line2D(
                    [x1, x2], [szy, 0],
                    color='red', linewidth=1., linestyle='--',
                    gid='axis2'))
            # DICOM annotations
            marked_idxs = self.main.tree_file_list.get_marked_imgs_current_test()
            if (
                    self.parent.tool_sum.isChecked()
                    and self.main.gui.active_img_no in marked_idxs):
                annot_text = (
                    ['Average image', ''] if self.main.average_img
                    else ['Summed image', '']
                    )
            else:
                annot_text = self.main.imgs[
                    self.main.gui.active_img_no].annotation_list
            at = matplotlib.offsetbox.AnchoredText(
                '\n'.join(annot_text),
                prop=dict(size=self.main.gui.annotations_font_size, color='red'),
                frameon=False, loc='upper left')
            self.ax.add_artist(at)
            self.roi_draw()
        else:
            self.draw()
        self.current_image = nparr

    def roi_draw(self):
        """Update ROI countours on image."""
        for contour in self.contours:
            for coll in contour.collections:
                try:
                    coll.remove()
                except ValueError:
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

        self.ax.texts.clear()

        if self.main.current_roi is not None:
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
                                 roi_indexes=None, filled=False, hatches=None):
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
                    #TODO handle IndexError on hatches[color_no]
                    contour = self.ax.contourf(
                        mask, levels=[0, 0.5], colors='none',
                        hatches=hatches[color_no])
                    contour.collections[0].set_edgecolor(colors[color_no])
            else:
                contour = self.ax.contour(
                    mask, levels=[0.9],
                    colors=colors[color_no], alpha=0.5, linewidths=self.linewidth)
            self.contours.append(contour)

        #if hatches is not None:
         #   for i, collection in enumerate(self.contours.collections):
          #      collection.set_edgecolor(colors[i % len(colors)])

    def Hom(self):
        """Draw Hom ROI."""
        colors = ['red', 'blue', 'green', 'yellow', 'cyan']
        self.add_contours_to_all_rois(colors=colors)

    def CTn(self):
        """Draw CTn ROI."""
        self.contours = []
        ctn_table = self.main.current_paramset.ctn_table
        for i in range(len(ctn_table.materials)):
            mask = np.where(self.main.current_roi[i], 0, 1)
            contour = self.ax.contour(
                mask, levels=[0.9],
                colors='red', alpha=0.5, linewidths=self.linewidth)
            self.contours.append(contour)
            mask_pos = np.where(mask == 0)
            xpos = np.mean(mask_pos[1])
            ypos = np.mean(mask_pos[0])
            if np.isfinite(xpos) and np.isfinite(ypos):
                self.ax.text(xpos, ypos, ctn_table.materials[i],
                             fontsize=self.fontsize, color='red')

        if len(self.main.current_roi) == 2 * len(ctn_table.materials):
            # draw search rois
            nroi = len(ctn_table.materials)
            for i in range(nroi, 2 * nroi):
                mask = np.where(self.main.current_roi[i], 0, 1)
                contour = self.ax.contour(
                    mask, levels=[0.9],
                    colors='blue', alpha=0.5, linewidths=self.linewidth)
                self.contours.append(contour)

    def Sli(self):
        """Draw Slicethickness search lines."""
        h_colors = ['b', 'lime']
        v_colors = ['c', 'r', 'm', 'darkorange']
        search_margin = self.main.current_paramset.sli_search_width
        background_length = self.main.current_paramset.sli_background_width
        pix = self.main.imgs[self.main.gui.active_img_no].pix
        background_length = background_length / pix[0]
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
                    colors=['red', 'blue', 'green', 'cyan'])
                mask = np.where(self.main.current_roi[-1], 0, 1)
                contour = self.ax.contour(
                    mask, levels=[0.9],
                    colors='red', alpha=0.5, linewidths=self.linewidth,
                    linestyles='dotted')
                self.contours.append(contour)
            else:
                self.add_contours_to_all_rois(colors=['red', 'blue', 'green', 'cyan'])

    def Uni(self):
        """Draw NM uniformity ROI."""
        self.add_contours_to_all_rois(colors=['red', 'blue'])

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

    def NPS(self):
        """Draw NPS ROIs."""
        if self.main.current_modality == 'CT':
            self.add_contours_to_all_rois(
                colors=['w']*self.main.current_paramset.nps_n_sub, filled=True)
        elif self.main.current_modality == 'Xray':
            n_sub = self.main.current_paramset.nps_n_sub
            idxs = [0, n_sub*2-2, n_sub*2, n_sub*4, -(n_sub*2-1), -1]
            self.add_contours_to_all_rois(
                colors=['r']*6, filled=True, roi_indexes=idxs)
        else:
            pass

    def Bar(self):
        """Draw Bar ROIs."""
        self.contours = []
        labels = [str(i+1) for i in range(4)]
        colors = ['red', 'blue', 'green', 'cyan']
        for i in range(4):
            mask = np.where(self.main.current_roi[i], 0, 1)
            contour = self.ax.contour(
                mask, levels=[0.9],
                colors=colors[i], alpha=0.5, linewidths=self.linewidth)
            self.contours.append(contour)
            mask_pos = np.where(mask == 0)
            xpos = np.mean(mask_pos[1])
            ypos = np.mean(mask_pos[0])
            if np.isfinite(xpos) and np.isfinite(ypos):
                self.ax.text(xpos, ypos, labels[i],
                             fontsize=self.fontsize, color=colors[i])

    def SNI(self):
        """Draw NM uniformity ROI."""
        self.add_contours_to_all_rois(
            colors=['red', 'blue'], roi_indexes=[1, 2],
            filled=True, hatches=['//', '\\'])  # 2 large
        self.add_contours_to_all_rois(
            colors=['lime', 'cyan'], reset_contours=False,
            roi_indexes=[3, 4],
            filled=True, hatches=['|||', '---'])  # 2 first small, else only label

        for i in range(6):
            mask = np.where(self.main.current_roi[i+3], 0, 1)
            color = 'yellow'
            mask_pos = np.where(mask == 0)
            xpos = np.mean(mask_pos[1])
            ypos = np.mean(mask_pos[0])
            if np.isfinite(xpos) and np.isfinite(ypos):
                self.ax.text(xpos-self.fontsize, ypos+self.fontsize,
                             f'S{i+1}',
                             fontsize=self.fontsize, color=color)

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

    def Gho(self):
        """Draw MR Ghosting ROI."""
        colors = ['red', 'blue', 'green', 'yellow', 'cyan']
        self.add_contours_to_all_rois(colors=colors)


class ResultImageCanvas(GenericImageCanvas):
    """Canvas for display of results as image."""

    def __init__(self, parent, main):
        super().__init__(parent, main)

    def result_image_draw(self):
        """Refresh result image."""
        self.ax.cla()
        self.current_image = None
        self.cmap = 'gray'
        self.min_val = None
        self.max_val = None
        self.title = ''

        if self.main.current_test in self.main.results:
            if self.main.results[self.main.current_test] is not None:
                class_method = getattr(self, self.main.current_test, None)
                if class_method is not None:
                    class_method()

        if self.current_image is not None:
            if self.min_val is None:
                self.min_val = np.min(self.current_image)
            if self.max_val is None:
                self.max_val = np.max(self.current_image)
            self.img = self.ax.imshow(
                self.current_image,
                cmap=self.cmap, vmin=self.min_val, vmax=self.max_val)
            #self.ax.set_title(self.title)
            self.parent.image_title.setText(self.title)
        else:
            self.img = self.ax.imshow(np.zeros((100, 100)))
            at = matplotlib.offsetbox.AnchoredText(
                'No result to display',
                prop=dict(size=14, color='gray'),
                frameon=False, loc='center')
            self.ax.add_artist(at)
            self.parent.image_title.setText('')
        self.ax.axis('off')
        self.draw()

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
        elif self.main.current_modality == 'Xray':
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
        if type_img == 0:
            self.title = 'Differential uniformity map in UFOV (max in x/y direction)'
            if 'du_matrix' in details_dict:
                self.current_image = details_dict['du_matrix']
        elif type_img == 1:
            self.title = 'Processed image minimum 6.4 mm pr pix'
            if 'matrix' in details_dict:
                self.current_image = details_dict['matrix']
        elif type_img == 2:
            self.title = 'Curvature corrected image'
            if 'corrected_image' in details_dict:
                self.current_image = details_dict['corrected_image']
        elif type_img == 3:
            self.title = 'Summed image'
            if 'sum_image' in details_dict:
                self.current_image = details_dict['sum_image']

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
        self.cmap = 'viridis'
        type_img = self.main.tab_nm.sni_result_image.currentIndex()
        if type_img == 0:
            self.title = 'Curvature corrected image'
            if 'corrected_image' in details_dict:
                self.current_image = details_dict['corrected_image']
        elif type_img == 1:
            self.title = 'Summed image'
            if 'sum_image' in details_dict:
                self.current_image = details_dict['sum_image']
        elif type_img > 1:
            sel_text = self.main.tab_nm.sni_result_image.currentText()
            roi_txt = sel_text[-2:]
            self.title = f'2d NPS for {roi_txt}'
            roi_names = ['L1', 'L2', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
            roi_no = roi_names.index(roi_txt)
            details_this = details_dict['pr_roi'][roi_no]
            self.current_image = details_this['NPS']
