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
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QAction, QToolBar, QToolButton,
    QFileDialog, QMessageBox
    )
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

# imageQC block start
from imageQC.ui.ui_dialogs import EditAnnotationsDialog
from imageQC.ui import ui_image_canvas
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

        self.mouse_pressed = False

    def image_on_move(self, event):
        """Actions on mouse move event."""
        if self.mouse_pressed and self.tool_profile.isChecked():
            if event.inaxes and len(event.inaxes.get_images()) > 0:
                if self.canvas.last_clicked_pos != (-1, -1):
                    _ = self.canvas.profile_draw(
                        round(event.xdata), round(event.ydata))

    def image_on_release(self, event, pix=None):
        """Actions when image canvas release mouse button."""
        if event.inaxes and len(event.inaxes.get_images()) > 0:
            if self.tool_profile.isChecked():
                if self.canvas.last_clicked_pos != (-1, -1):
                    plotstatus = self.canvas.profile_draw(
                        round(event.xdata), round(event.ydata))
                    self.mouse_pressed = False
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
                if max_abs < 10:
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

    def __init__(self, parent):
        super().__init__(parent, ui_image_canvas.ImageCanvas(self, parent))
        self.main = parent

        tbimg = ImageNavigationToolbar(self.canvas, self.main)
        tbimg2 = GenericImageToolbarPosVal(self.canvas, self.main)
        hlo = QHBoxLayout()
        vlo_tb = QVBoxLayout()
        hlo.addLayout(vlo_tb)

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
        tbimg.addAction(act_redraw)
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


class ImageNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib navigation toolbar with some modifications."""

    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        for x in self.actions():
            if x.text() == 'Subplots':
                self.removeAction(x)

    def set_message(self, s):
        """Hide cursor position and value text."""
        pass

    # from https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/backends/backend_qt.py
    #  dirty fix to avoid crash on self.canvas.parent() TypeError
    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()

        # startpath = os.path.expanduser(mpl.rcParams['savefig.directory'])
        # start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = f'{name} ({exts_list})'
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)

        fname, filter = QFileDialog.getSaveFileName(
            self, 'Choose a filename to save to', '',
            filters, selectedFilter)
        if fname:
            try:
                self.canvas.figure.savefig(fname)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error saving file", str(e))
                #    _enum("QtWidgets.QMessageBox.StandardButton").Ok,
                #    _enum("QtWidgets.QMessageBox.StandardButton").NoButton)

'''Not in use anymore?
class ImageExtraToolbar(QToolBar):
    """Extra toolbar for showing more cursor position and value."""

    def __init__(self, canvas, window):
        super().__init__()

        self.xypos = QLabel('')
        self.xypos.setMinimumWidth(500)
        self.addWidget(self.xypos)
        try:
            self.delta_x = window.gui.delta_x
            self.delta_y = window.gui.delta_y
        except AttributeError:
            self.delta_x = 0
            self.delta_y = 0

        canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        """When mouse cursor is moving in the canvas."""
        if event.inaxes and len(event.inaxes.get_images()) > 0:
            sz_img = event.inaxes.get_images()[0].get_array().shape
            xpos = event.xdata - 0.5 * sz_img[1] + self.delta_x
            ypos = event.ydata - 0.5 * sz_img[0] + self.delta_y
            xyval = event.inaxes.get_images()[0].get_cursor_data(event)
            try:
                self.xypos.setText(
                    f'xy = ({xpos:.0f}, {ypos:.0f}), val = {xyval:.1f}')
            except TypeError:
                self.xypos.setText('')
        else:
            self.xypos.setText('')
'''
