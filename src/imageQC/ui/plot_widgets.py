#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic classes for plot in ImageQC.

@author: Ellen Wasbo
"""
import os

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QAction, QLabel,
    QInputDialog, QMessageBox)
import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)

# imageQC block start
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.scripts.mini_methods_format import val_2_str
from imageQC.config.iQCconstants import ENV_ICON_PATH
# imageQC block end


class PlotDialog(ImageQCDialog):
    """Dialog for plot."""

    def __init__(self, main, title=''):
        super().__init__()
        self.setWindowTitle(title)
        vlo = QVBoxLayout()
        self.setLayout(vlo)
        self.plotcanvas = PlotCanvas(main)
        vlo.addWidget(PlotWidget(main, self.plotcanvas))


class PlotWidget(QWidget):
    """Widget with plot."""

    def __init__(self, main, plotcanvas, include_min_max_button=False):
        super().__init__()
        self.main = main

        self.plotcanvas = plotcanvas
        tb_plot = PlotNavigationToolbar(self.plotcanvas, self)
        self.hlo = QHBoxLayout()
        vlo_tb = QVBoxLayout()
        self.hlo.addLayout(vlo_tb)

        tb_plot_copy = QToolBar()
        act_copy = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}copy.png'),
            'Copy curve as table to clipboard', self)
        act_copy.triggered.connect(self.copy_curves)
        if include_min_max_button:
            act_min_max = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}minmax.png'),
                'Set x and y ranges to min/max of content', self)
            act_min_max.triggered.connect(self.min_max_curves)
            tb_plot_copy.addActions([act_copy, act_min_max])
        else:
            tb_plot_copy.addActions([act_copy])
        tb_plot_copy.setOrientation(Qt.Vertical)

        vlo_tb.addWidget(tb_plot_copy)
        vlo_tb.addWidget(tb_plot)
        vlo_tb.addStretch()

        tb_message = PlotNavigationToolbarMessage(self.plotcanvas, self)
        vlo_plot = QVBoxLayout()
        vlo_plot.addWidget(tb_message)
        vlo_plot.addWidget(self.plotcanvas)
        self.hlo.addLayout(vlo_plot)
        self.setLayout(self.hlo)

    def min_max_curves(self):
        pass  # included in widgets with include_min_max_button = True

    def copy_curves(self):
        """Copy contents of curves to clipboard."""
        decimal_mark = self.main.current_paramset.output.decimal_mark
        headers = []
        values = []
        last_xvalues = None
        for curve in self.plotcanvas.curves:
            if 'tolerance' not in curve['label'] and 'nolegend' not in curve['label']:
                if 'xticks' in curve:
                    xvalues = curve['xticks']
                else:
                    xvalues = val_2_str(curve['xvals'], decimal_mark=decimal_mark)
                yvalues = val_2_str(curve['yvals'], decimal_mark=decimal_mark)
                if last_xvalues is not None:
                    np_xvalues = np.array(xvalues)
                    try:
                        if np.array_equal(np_xvalues, last_xvalues):
                            xvalues = None
                        else:
                            last_xvalues = np_xvalues
                    except (ValueError, AttributeError):
                        last_xvalues = np_xvalues
                else:
                    last_xvalues = np.array(xvalues)
                if xvalues is not None:
                    values.append(xvalues)
                    headers.append(self.plotcanvas.xtitle)
                values.append(yvalues)
                headers.append(curve['label'])
        if len(values) == 0:
            for bar_values in self.plotcanvas.bars:
                names = bar_values['names']
                yvalues = val_2_str(bar_values['values'], decimal_mark=decimal_mark)
                values.append(names)
                values.append(yvalues)
                headers.append(['names', 'values'])
        if len(values) == 0:
            try:
                labels = []
                for i, scatterdict in enumerate(self.plotcanvas.scatters):
                    if 'label' in scatterdict:
                        labels.append(scatterdict['label'])
                    else:
                        labels.append(f'plot {i}')
                if len(labels) > 1:
                    label, ok = QInputDialog.getItem(
                        self, "Select plot to copy to clipboard",
                        "Plot:", labels, 0, False)
                    if ok and label:
                        idx = labels.index(label)
                    else:
                        idx = -1
                else:
                    idx = 0
                if idx >= 0:
                    scatter = self.plotcanvas.scatters[idx]
                    if isinstance(scatter['color'], str):
                        arr = scatter['array']
                    else:
                        arr = scatter['color']
                    if 'xlabels' in scatterdict:
                        xlabels = scatterdict['xlabels']
                        ylabels = scatterdict['ylabels']
                        if decimal_mark == ',':
                            ylabels = [lbl.replace('.', ',') for lbl in ylabels]
                            xlabels = [lbl.replace('.', ',') for lbl in xlabels]
                        df = pd.DataFrame(arr, index=ylabels, columns=xlabels)
                        df.to_clipboard(excel=True, decimal=decimal_mark)
                    elif 'xs' in scatterdict:
                        color_values = val_2_str(scatter['color'],
                                            decimal_mark=decimal_mark)
                        xvalues = val_2_str(scatter['xs'],
                                            decimal_mark=decimal_mark)
                        yvalues = val_2_str(scatter['ys'],
                                            decimal_mark=decimal_mark)
                        values = [xvalues, yvalues, color_values]
                        headers = ['x', 'y', 'value']
                    self.main.status_bar.showMessage(
                        'Values in clipboard', 2000)
            except:
                QMessageBox.warning(self, 'Failed copy to clipboard',
                                    'Failed copying plot to clipboard.')

        if len(values) > 0:
            df = pd.DataFrame(values)
            df = df.transpose()
            df.columns = headers
            df.to_clipboard(index=False, excel=True)
            self.main.status_bar.showMessage('Values in clipboard', 2000)


class PlotNavigationToolbarMessage(QToolBar):
    """Toolbar to show values of scatter plot by color."""

    def __init__(self, canvas, window):
        super().__init__()
        self.window = window
        self.canvas = canvas
        self.xypos = QLabel('')
        self.xypos.setMinimumWidth(500)
        self.addWidget(self.xypos)

        canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        """When mouse cursor is moving in the canvas."""
        if event.inaxes:
            self.xypos.setText('')
            try:
                if len(self.canvas.scatters) > 0:
                    txt = ''
                    x = round(event.xdata)
                    y = round(event.ydata)
                    xlabel = self.canvas.scatters[0]['xlabels'][x]
                    ylabel = self.canvas.scatters[0]['ylabels'][y]
                    txt = f'x {xlabel}, y {ylabel} '
                    for scatter in self.canvas.scatters:
                        if not isinstance(scatter['color'], str):
                            val = scatter['color'][y][x]
                            if isinstance(val, float):
                                txt = txt + f' value = {val:.3f}'
                    self.xypos.setText(txt)
            except (AttributeError, KeyError, IndexError):
                pass


class PlotNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib navigation toolbar with some modifications."""

    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        for x in self.actions():
            if x.text() in ['Back', 'Forward', 'Pan', 'Subplots']:
                self.removeAction(x)
        self.setOrientation(Qt.Vertical)

    def set_message(self, s):
        """Hide cursor position and value text."""
        pass


class PlotCanvas(FigureCanvasQTAgg):
    """Canvas for display of results as plot."""

    def __init__(self, main):
        self.main = main
        if self.main.user_prefs.dark_mode:
            matplotlib.pyplot.style.use('dark_background')
        self.fig = matplotlib.figure.Figure(dpi=150)
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.ax = self.fig.add_subplot(111)
        self.title = ''
        self.xtitle = 'x'
        self.ytitle = 'y'
        self.default_range_x = [None, None]
        self.default_range_y = [None, None]
        self.legend_location = 'upper right'
        self.curves = []
        self.bars = []
        self.marked_this = []
        self.zpos_all = []

    def draw(self):
        """Avoid super().draw when figure collapsed by sliders."""
        try:
            super().draw()
        except ValueError:
            pass

    def plot(self, title='', xtitle='x', ytitle='y',
             xvals=[], yvals=[], labels=[]):
        """Refresh plot."""
        self.ax.clear()
        styles = ['k', 'g', 'b', 'r', 'p', 'c']
        self.curves = []
        self.xtitle = xtitle
        for i, vals in enumerate(yvals):
            self.curves.append(
                {'label': labels[i],
                 'xvals': xvals[i],
                 'yvals': vals,
                 'style': styles[i]
                 }
                )

        if len(self.curves) > 0:
            for curve in self.curves:
                self.ax.plot(curve['xvals'], curve['yvals'],
                             curve['style'], label=curve['label'])
            if len(self.curves) > 1:
                self.ax.legend(loc="upper right")
            if len(title) > 0:
                self.ax.suptitle(title)
                self.fig.subplots_adjust(0.15, 0.25, 0.95, 0.85)
            else:
                self.fig.subplots_adjust(0.15, 0.2, 0.95, .95)
            self.ax.set_xlabel(xtitle)
            self.ax.set_ylabel(ytitle)
            length = xvals[0][-1] - xvals[0][0]
            unit = ''
            if '(mm)' in xtitle:
                unit = ' mm'
            self.ax.set_title(f'Profile length: {length:.1f} {unit}')

        self.draw()
