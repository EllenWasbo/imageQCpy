#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for main window of imageQC - image widgets part.

@author: Ellen Wasbo
"""
import os
import numpy as np
from skimage import draw

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QAction, QToolBar, QToolButton
    )

# imageQC block start
from imageQC.ui.ui_dialogs import (
    EditAnnotationsDialog, CmapSelectDialog, ProjectionPlotDialog)
from imageQC.ui import ui_image_canvas
from imageQC.ui.reusable_widgets import ImageNavigationToolbar
from imageQC.ui.plot_widgets import PlotDialog
from imageQC.config.iQCconstants import ENV_ICON_PATH
# imageQC block end


class GenericImageWidget(QWidget):
    """General image widget."""

    def __init__(self, parent, canvas):
        super().__init__()
        self.parent = parent
        self.canvas = canvas
        self.canvas.mpl_connect('motion_notify_event', self.image_on_move)
        self.canvas.mpl_connect('button_press_event', self.image_on_click)
        self.canvas.mpl_connect('button_release_event', self.image_on_release)
        self.tool_profile = QToolButton()
        self.tool_profile.setToolTip(
            'Toggle to plot image profile when click/drag line in image')
        self.tool_profile.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}profile.png'))
        self.tool_profile.clicked.connect(self.clicked_profile)
        self.tool_profile.setCheckable(True)
        self.tool_rectangle = QToolButton()
        self.tool_rectangle.setToolTip(
            'Toggle to activate option to mark active area when click/drag in image')
        self.tool_rectangle.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}rectangle.png'))
        self.tool_rectangle.clicked.connect(self.clicked_rectangle)
        self.tool_rectangle.setCheckable(True)
        self.tool_cmap = QToolButton()
        self.tool_cmap.setToolTip(
            'Select colormap for the image')
        self.tool_cmap.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}colorbar.png'))
        self.tool_cmap.clicked.connect(self.clicked_colormap)

        self.mouse_pressed = False

    def image_on_move(self, event):
        """Actions on mouse move event."""
        if self.mouse_pressed:
            if self.tool_profile.isChecked() or self.tool_rectangle.isChecked():
                if event.inaxes and len(event.inaxes.get_images()) > 0:
                    if self.canvas.last_clicked_pos != (-1, -1):
                        if self.tool_profile.isChecked():
                            _ = self.canvas.profile_draw(
                                round(event.xdata), round(event.ydata))
                        else:
                            _ = self.canvas.rectangle_mark(
                                round(event.xdata), round(event.ydata))

    def image_on_release(self, event, pix=None):
        """Actions when image canvas release mouse button."""
        if event.inaxes and len(event.inaxes.get_images()) > 0:
            self.mouse_pressed = False
            if self.tool_profile.isChecked():
                if self.canvas.last_clicked_pos != (-1, -1):
                    plotstatus = self.canvas.profile_draw(
                        round(event.xdata), round(event.ydata))
                    if plotstatus:
                        self.plot_profile(round(event.xdata), round(event.ydata))

    def image_on_click(self, event):
        """Actions when image canvas is clicked."""
        if event.inaxes and len(event.inaxes.get_images()) > 0:
            if self.tool_profile.isChecked():
                self.canvas.profile_remove()
                self.canvas.draw()
                self.canvas.last_clicked_pos = (
                    round(event.xdata), round(event.ydata))
                self.mouse_pressed = True
            elif self.tool_rectangle.isChecked():
                self.canvas.rectangle_remove()
                self.canvas.draw_idle()
                self.canvas.last_clicked_pos = (
                    round(event.xdata), round(event.ydata))
                self.mouse_pressed = True

    def clicked_profile(self):
        """Refresh image when deactivated profile."""
        if self.tool_profile.isChecked() is False:
            self.canvas.profile_remove()
            self.canvas.draw()

    def plot_profile(self, x2, y2, pix=None):
        """Pop up dialog window with plot of profile."""
        x1 = self.canvas.last_clicked_pos[0]
        y1 = self.canvas.last_clicked_pos[1]
        rr, cc = draw.line(y1, x1, y2, x2)
        profile = self.canvas.current_image[rr, cc]
        len_pix = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        len_pr_pix = len_pix / len(profile)
        if pix is not None:
            len_pr_pix = len_pr_pix * pix
            xtitle = 'pos (mm)'
        else:
            xtitle = 'pos (pix)'
        xvals = np.arange(len(profile)) * len_pr_pix
        dlg = PlotDialog(self.main, title='Image profile')
        dlg.plotcanvas.plot(xvals=[xvals], yvals=[profile],
                            xtitle=xtitle, ytitle='Pixel value',
                            title='', labels=['pixel_values'])
        dlg.exec()

    def clicked_rectangle(self):
        """Refresh image when deactivated profile."""
        if self.tool_rectangle.isChecked() is False:
            self.canvas.rectangle_remove()
            self.canvas.draw_idle()

    def clicked_colormap(self):
        """Display colormap dialog and update colormaps."""
        dlg = CmapSelectDialog(self)
        res = dlg.exec()
        if res:
            cmap = dlg.get_cmap()
            if 'Result' in str(type(self.canvas)):
                self.canvas.ax.get_images()[0].set_cmap(cmap)
                self.canvas.draw_idle()
                self.canvas.parent.wid_window_level.colorbar.colorbar_draw(cmap=cmap)
            else:
                try:
                    self.canvas.ax.get_images()[0].set_cmap(cmap)
                    self.canvas.draw_idle()
                except (AttributeError, IndexError):
                    pass
                self.parent.wid_window_level.colorbar.colorbar_draw(cmap=cmap)


class GenericImageToolbarPosVal(QToolBar):
    """Toolbar for showing cursor position and value."""

    def __init__(self, canvas, window):
        super().__init__()

        self.xypos = QLabel('')
        self.xypos.setMinimumWidth(500)
        self.addWidget(self.xypos)
        self.window = window

        canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        """When mouse cursor is moving in the canvas."""
        if event.inaxes and len(event.inaxes.get_images()) > 0:
            if hasattr(self.window, 'gui'):
                delta_x = self.window.gui.delta_x
                delta_y = self.window.gui.delta_y
            else:
                delta_x = 0
                delta_y = 0
            img = event.inaxes.get_images()[0].get_array()
            sz_img = img.shape
            max_abs = np.max(np.abs(img))
            xpos = event.xdata - 0.5 * sz_img[1] - delta_x
            ypos = event.ydata - 0.5 * sz_img[0] - delta_y
            xyval = event.inaxes.get_images()[0].get_cursor_data(event)
            try:
                if isinstance(xyval, np.ma.core.MaskedConstant):
                    txt = f'xy = ({xpos:.0f}, {ypos:.0f}), val = masked'
                elif max_abs < 10:
                    txt = f'xy = ({xpos:.0f}, {ypos:.0f}), val = {xyval:.3f}'
                else:
                    txt = f'xy = ({xpos:.0f}, {ypos:.0f}), val = {xyval:.1f}'
                self.xypos.setText(txt)
            except TypeError:
                self.xypos.setText('')
        else:
            self.xypos.setText('')


class ImageDisplayWidget(GenericImageWidget):
    """Image display widget."""

    def __init__(self, parent, toolbar_right=True):
        super().__init__(parent, ui_image_canvas.ImageCanvas(self, parent))
        self.main = parent

        tbimg = ImageNavigationToolbar(self.canvas, self.main)
        tbimg2 = GenericImageToolbarPosVal(self.canvas, self.main)
        hlo = QHBoxLayout()
        vlo_tb = QVBoxLayout()
        hlo.addLayout(vlo_tb)

        if toolbar_right:
            act_redraw = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}refresh.png'),
                'Force refresh on image if hiccups', self)
            act_redraw.triggered.connect(self.main.refresh_img_display)
            self.tool_sum = QToolButton()
            self.tool_sum.setToolTip(
                'Toggle to display sum of marked images, press again to display average.')
            self.tool_sum.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}sigma.png'))
            self.tool_sum.clicked.connect(self.clicked_sum)
            self.tool_sum.setCheckable(True)
            act_edit_annotations = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
                'Edit annotations', self)
            act_edit_annotations.triggered.connect(self.edit_annotations)
            self.tool_imgsize = QToolButton()
            self.tool_imgsize.setToolTip('Maximize image')
            self.tool_imgsize.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}layout_maximg.png'))
            self.tool_imgsize.clicked.connect(self.clicked_imgsize)
            self.tool_imgsize.setCheckable(True)
            act_projection_plot = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}projections.png'),
                'Show 3d projection and optionally plot values from result table', self)
            act_projection_plot.triggered.connect(self.projection_plot)
            tbimg.addAction(act_redraw)
            tbimg.addWidget(self.tool_cmap)
            tbimg.addWidget(self.tool_rectangle)
            tbimg.addAction(act_projection_plot)
            tbimg.addWidget(self.tool_profile)
            tbimg.addWidget(self.tool_sum)
            tbimg.addAction(act_edit_annotations)
            tbimg.addWidget(self.tool_imgsize)

        vlo_img = QVBoxLayout()
        vlo_img.addWidget(tbimg)
        vlo_img.addWidget(tbimg2)
        vlo_img.addWidget(self.canvas)

        self.setLayout(vlo_img)

        self.mouse_pressed = False

    def image_on_click(self, event):
        """Actions when image canvas is clicked."""
        super().image_on_click(event)
        if event.inaxes and len(event.inaxes.get_images()) > 0:
            self.main.gui.last_clicked_pos = (
                    round(event.xdata), round(event.ydata))
            if event.dblclick:
                self.main.wid_center.set_center_to_clickpos()

    def projection_plot(self):
        """Show 3d projection and plot."""
        if len(self.main.imgs) > 1:
            dlg = ProjectionPlotDialog(self.main)
            dlg.exec()
        else:
            self.main.status_bar.showMessage(
                'Not enough images loaded to extract a projection', 2000)

    def edit_annotations(self):
        """Pop up dialog to edit annotations settings."""
        dlg = EditAnnotationsDialog(
            annotations=self.main.gui.annotations,
            annotations_line_thick=self.main.gui.annotations_line_thick,
            annotations_font_size=self.main.gui.annotations_font_size)
        res = dlg.exec()
        if res:
            ann, line_thick, font_size = dlg.get_data()
            self.main.gui.annotations = ann
            self.main.gui.annotations_line_thick = line_thick
            self.main.gui.annotations_font_size = font_size
            if self.main.gui.active_img_no > -1:
                self.canvas.img_draw()

    def clicked_imgsize(self):
        """Maximize or reset image size."""
        if self.tool_imgsize.isChecked():
            self.tool_imgsize.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}layout_resetimg.png'))
            self.main.set_split_max_img()
        else:
            self.tool_imgsize.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}layout_maximg.png'))
            self.main.reset_split_max_img()

    def clicked_sum(self):
        """Activate or deactive display sum of marked images."""
        if self.main.summed_img is not None and self.main.average_img is False:
            self.main.average_img = True
            self.tool_sum.setChecked(True)
            self.tool_sum.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}xmean.png'))
        if self.tool_sum.isChecked():
            if self.main.average_img:
                self.main.update_summed_img(recalculate_sum=False)
            else:
                self.main.update_summed_img()
        else:
            self.main.reset_summed_img()
            self.canvas.img_draw()

    def plot_profile(self, x2, y2, pix=None):
        """Pop up dialog window with plot of profile."""
        pix = self.main.imgs[self.main.gui.active_img_no].pix[0]
        super().plot_profile(x2, y2, pix=pix)
