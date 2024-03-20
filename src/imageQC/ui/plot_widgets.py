#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic classes for plot in ImageQC.

@author: Ellen Wasbo
"""
import os

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QAction
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

    def __init__(self, main, plotcanvas):
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
        tb_plot_copy.setOrientation(Qt.Vertical)
        tb_plot_copy.addActions([act_copy])

        vlo_tb.addWidget(tb_plot_copy)
        vlo_tb.addWidget(tb_plot)
        vlo_tb.addStretch()
        self.hlo.addWidget(self.plotcanvas)
        self.setLayout(self.hlo)

    def copy_curves(self):
        """Copy contents of curves to clipboard."""
        decimal_mark = self.main.current_paramset.output.decimal_mark
        headers = []
        values = []
        for curve in self.plotcanvas.curves:
            if 'tolerance' not in curve['label'] and 'nolegend' not in curve['label']:
                xvalues = val_2_str(curve['xvals'], decimal_mark=decimal_mark)
                yvalues = val_2_str(curve['yvals'], decimal_mark=decimal_mark)
                values.append(xvalues)
                values.append(yvalues)
                headers.append(f'{curve["label"]}_{self.plotcanvas.xtitle}')
                headers.append(curve['label'])
        if len(values) == 0:
            for bar_values in self.plotcanvas.bars:
                names = bar_values['names']
                yvalues = val_2_str(bar_values['values'], decimal_mark=decimal_mark)
                values.append(names)
                values.append(yvalues)
                headers.append(['names', 'values'])

        df = pd.DataFrame(values)
        df = df.transpose()
        df.columns = headers
        df.to_clipboard(index=False, excel=True)
        self.main.status_bar.showMessage('Values in clipboard', 2000)


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
        self.ax.cla()
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
