#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for main window of imageQC - result tabs part.

@author: Ellen Wasbo
"""
import os
import copy
import numpy as np
import pandas as pd

from PyQt5.QtGui import QIcon, QKeyEvent
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QAction, QToolBar, QToolButton,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QAbstractScrollArea,
    QMessageBox, QLabel, QSizePolicy
    )
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt

# imageQC block start
from imageQC.ui import ui_image_canvas
from imageQC.ui.ui_main_image_widgets import (
    GenericImageWidget, GenericImageToolbarPosVal, NavigationToolbar2QT)
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.plot_widgets import PlotWidget, PlotCanvas
from imageQC.config.iQCconstants import ENV_ICON_PATH
from imageQC.scripts import mini_methods_format as mmf
from imageQC.scripts.mini_methods_calculate import find_median_spectrum
# imageQC block end


class ResultTableWidget(QWidget):
    """Results table widget."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent

        hlo = QHBoxLayout()
        self.setLayout(hlo)
        vlo_tb = QVBoxLayout()
        hlo.addLayout(vlo_tb)

        act_copy = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}copy.png'),
            'Copy table to clipboard', self)
        act_copy.triggered.connect(self.copy_table)
        toolb = QToolBar()
        toolb.setOrientation(Qt.Vertical)
        toolb.addActions([act_copy])
        vlo_tb.addWidget(toolb)
        self.tb_copy = uir.ToolBarTableExport(
            self, parameters_output=self.main.current_paramset.output)
        vlo_tb.addWidget(self.tb_copy)
        vlo_tb.addStretch()

        self.result_table = ResultTable(parent=self, main=self.main)
        self.table_info = uir.LabelItalic('')
        vlo_table = QVBoxLayout()
        vlo_table.addWidget(self.table_info)
        vlo_table.addWidget(self.result_table)
        hlo.addLayout(vlo_table)

        toolb = QToolBar()
        self.tool_resultsize = ToolMaximizeResults(self.main)
        toolb.addWidget(self.tool_resultsize)
        toolb.setOrientation(Qt.Vertical)
        hlo.addWidget(toolb)

    def copy_table(self):
        """Copy contents of table to clipboard."""
        decimal_mark = '.'
        if self.tb_copy.tool_decimal.isChecked():
            decimal_mark = ','
        values = [
            mmf.val_2_str(
                col,
                decimal_mark=decimal_mark)
            for col in self.result_table.values]

        if self.tb_copy.tool_header.isChecked():  # insert headers
            if self.result_table.row_labels[0] == '':
                for i in range(len(values)):
                    values[i].insert(0, self.result_table.col_labels[i])
            else:
                # row headers true headers
                values.insert(0, self.result_table.row_labels)

        if self.tb_copy.tool_transpose.isChecked() is False:
            values = np.array(values).T.tolist()

        df = pd.DataFrame(values)
        df.to_clipboard(index=False, excel=True, header=None)
        self.main.status_bar.showMessage('Values in clipboard', 2000)


class ResultTable(QTableWidget):
    """Results table.

    Parameters
    ----------
    parent : MainWindow
        for link to active image
    """

    def __init__(self, parent=None, main=None):
        super().__init__()
        self.parent = parent
        self.main = main
        self.linked_image_list = True
        self.cellClicked.connect(self.cell_selected)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.values = [[]]  # always as columns, converted if input is rows
        self.row_labels = []
        self.col_labels = []
        self.installEventFilter(self)

    def eventFilter(self, source, event):
        """Handle arrow up/down events."""
        if isinstance(event, QKeyEvent):
            if event.type() == QEvent.KeyRelease:
                if event.key() in [Qt.Key_Up, Qt.Key_Down]:
                    self.cell_selected()
        return False

    def cell_selected(self):
        """Set new active image when current cell changed."""
        if self.linked_image_list:
            self.main.set_active_img(self.currentRow())

    def clear(self):
        """Also update visual table."""
        super().clear()
        self.parent.table_info.setText('')
        self.setRowCount(0)
        self.setColumnCount(0)

    def fill_table(self, row_labels=[], col_labels=[],
                   values_cols=[[]], values_rows=[[]],
                   linked_image_list=True, table_info='', vendor=False):
        """Populate table.

        Parameters
        ----------
        row_labels : list(str)
            if empty list, none will show
        col_labels : list(str)
            if empty list, numbers shown
        values_cols : list(list(str/float/int))
            one list for each column of values
        values_rows : list(list(str/float/int))
            one list for each row of values
        linked_image_list : bool
            selected table row also change the selection in image list.
            Default is True
        vendor : bool
            If vendor QC specific settings and test 'vendor'
            Default is False
        """
        self.parent.table_info.setText(table_info)

        decimal_mark = '.'
        if self.parent.tb_copy.tool_decimal.isChecked():
            decimal_mark = ','

        if vendor:
            try:
                row_labels = self.main.results['vendor']['headers']
                values_cols = self.main.results['vendor']['values']
                df = pd.DataFrame(values_cols)
                temp_rows = df.T
                txt_rows = []
                for i in range(len(row_labels)):
                    row_list = temp_rows.loc[i, :].values.flatten().tolist()
                    txt_rows.append(mmf.val_2_str(row_list, decimal_mark=decimal_mark))
                df = pd.DataFrame(txt_rows)
                values_cols = df.T.values.tolist()
                linked_image_list = False
            except KeyError:
                pass

        if len(row_labels) != 0:
            n_cols = len(values_cols)
            n_rows = len(row_labels)
        else:
            n_cols = len(col_labels)
            n_rows = len(values_rows)
        self.setColumnCount(n_cols)
        self.setRowCount(n_rows)

        self.row_labels = (
            row_labels if len(row_labels) > 0 else [''] * n_rows)
        self.col_labels = (
            col_labels if len(col_labels) > 0 else [
                str(i) for i in range(n_cols)])

        self.linked_image_list = linked_image_list
        self.setHorizontalHeaderLabels(self.col_labels)
        if len(row_labels) > 0:
            self.setVerticalHeaderLabels(self.row_labels)
            self.verticalHeader().setVisible(True)
        else:
            self.verticalHeader().setVisible(False)

        if len(col_labels) > 0:
            # convert rows to columns for better formatting -columns similar numbers
            for r in range(n_rows):
                if len(values_rows[r]) == 0:
                    values_rows[r] = [None] * n_cols
            values_cols = []
            for c in range(n_cols):
                values_cols.append([row[c] for row in values_rows])
        if len(values_cols[0]) > 0:
            for c in range(len(values_cols)):
                if vendor:
                    this_col = values_cols[c]  # formatted above
                else:
                    lock_format = False
                    if self.main.current_test == 'DCM':
                        if self.main.current_paramset.dcm_tagpattern.list_format[c] != '':
                            lock_format = True
                    this_col = mmf.val_2_str(
                        values_cols[c], decimal_mark=decimal_mark,
                        lock_format=lock_format)
                for r in range(len(this_col)):
                    twi = QTableWidgetItem(this_col[r])
                    twi.setTextAlignment(4)
                    self.setItem(r, c, twi)

        self.values = values_cols
        self.resizeColumnsToContents()
        self.resizeRowsToContents()


class ToolMaximizeResults(QToolButton):
    """Toolbutton with to maximize results panel."""

    def __init__(self, main):
        super().__init__()
        self.main = main

        self.setToolTip('Maximize')
        self.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}layout_maximg.png'))
        self.clicked.connect(
            lambda: self.main.clicked_resultsize(tool=self))
        self.setCheckable(True)


class ResultPlotWidget(PlotWidget):
    """Widget for display of results as plot."""

    def __init__(self, main, plotcanvas):
        super().__init__(main, plotcanvas)

        toolb = QToolBar()
        self.tool_resultsize = ToolMaximizeResults(main)
        toolb.addWidget(self.tool_resultsize)
        toolb.setOrientation(Qt.Vertical)
        self.hlo.addWidget(toolb)


class ResultPlotCanvas(PlotCanvas):
    """Canvas for display of results as plot."""

    def __init__(self, main):
        super().__init__(main)

    def plot(self):
        """Refresh plot."""
        self.ax.cla()
        self.title = ''
        self.xtitle = 'x'
        self.ytitle = 'y'
        self.default_range_x = [None, None]
        self.default_range_y = [None, None]
        self.legend_location = 'upper right'
        self.curves = []
        if self.main.current_test == 'vendor':
            try:
                _ = self.main.results['vendor']['details']
                self.vendor()
            except KeyError:
                pass
        else:
            self.zpos_all = [img.zpos for img in self.main.imgs]
            self.marked_this = self.main.tree_file_list.get_marked_imgs_current_test()

            if self.main.gui.active_img_no in self.marked_this:
                if self.main.current_test in self.main.results:
                    if self.main.results[self.main.current_test] is not None:
                        class_method = getattr(self, self.main.current_test, None)
                        if class_method is not None:
                            class_method()

        if len(self.curves) > 0:
            x_only_int = True
            for curve in self.curves:
                if 'markersize' in curve:
                    markersize = curve['markersize']
                else:
                    markersize = 6.
                if 'style' not in curve:
                    curve['style'] = '-'
                if 'color' in curve:
                    self.ax.plot(curve['xvals'], curve['yvals'],
                                 curve['style'], label=curve['label'],
                                 markersize=markersize,
                                 color=curve['color'])
                else:
                    self.ax.plot(curve['xvals'], curve['yvals'],
                                 curve['style'], label=curve['label'],
                                 markersize=markersize)
                if x_only_int:
                    xx = list(curve['xvals'])
                    if not isinstance(xx[0], int):
                        x_only_int = False
            if x_only_int:
                self.ax.xaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(integer=True))
            if len(self.curves) > 1:
                self.ax.legend(loc=self.legend_location)
            if len(self.title) > 0:
                self.ax.set_title(self.title)
                self.fig.subplots_adjust(0.15, 0.25, 0.95, 0.85)
            else:
                self.fig.subplots_adjust(0.15, 0.2, 0.95, .95)
            self.ax.set_xlabel(self.xtitle)
            self.ax.set_ylabel(self.ytitle)
            if None not in self.default_range_x:
                self.ax.set_xlim(self.default_range_x)
            if None not in self.default_range_y:
                self.ax.set_ylim(self.default_range_y)
        else:
            self.ax.axis('off')

        self.draw()

    def test_values_outside_yrange(self, yrange, limit_xrange=None):
        """Set yrange to min/max (=None) if values outside default range.

        Parameters
        ----------
        yrange: list
            [min_y, max_y] if any values outside - set to None = include values outside
        limit_xrange: list
            set to [min_x, max_x] to evaluate. Default is None (ignored).
        """
        def get_yvals_to_eval(curve):
            idxs = np.where(np.logical_and(
                curve['xvals'] > limit_xrange[0],
                curve['xvals'] < limit_xrange[1]))
            return curve['yvals'][idxs[0]]

        for curve in self.curves:
            if yrange[0] is not None:
                if limit_xrange is None:
                    if min(curve['yvals']) < yrange[0]:
                        yrange[0] = None
                else:
                    if min(get_yvals_to_eval(curve)) < yrange[0]:
                        yrange[0] = None
            if yrange[1] is not None:
                if limit_xrange is None:
                    if max(curve['yvals']) > yrange[1]:
                        yrange[1] = None
                else:
                    if max(get_yvals_to_eval(curve)) > yrange[1]:
                        yrange[1] = None
            if not any(yrange):
                break
        return yrange

    def ROI(self):
        """Prepare plot for test ROI."""
        xvals = []
        yvals = []
        for i, row in enumerate(self.main.results['ROI']['values']):
            if len(row) > 0:
                xvals.append(i)
                yvals.append(row[0])
        curve = {'label': 'Average',
                 'xvals': xvals,
                 'yvals': yvals,
                 'style': '-b'}
        self.curves.append(curve)
        self.xtitle = 'Image index'
        self.ytitle = 'Average pixel value'
        if self.main.current_modality in ['Xray', 'NM']:
            self.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
            self.ax.set_xticks(xvals)

    def Hom(self):
        """Prepare plot for test Hom."""
        img_nos = []
        xvals = []
        if self.main.current_modality == 'CT':
            self.title = 'Difference (HU) from center'
            yvals12 = []
            yvals15 = []
            yvals18 = []
            yvals21 = []
            for i, row in enumerate(self.main.results['Hom']['values']):
                if len(row) > 0:
                    img_nos.append(i)
                    xvals.append(self.zpos_all[i])
                    yvals12.append(row[5])
                    yvals15.append(row[6])
                    yvals18.append(row[7])
                    yvals21.append(row[8])
            self.curves.append(
                {'label': 'at 12', 'xvals': xvals,
                 'yvals': yvals12, 'style': '-b'})
            self.curves.append(
                {'label': 'at 15', 'xvals': xvals,
                 'yvals': yvals15, 'style': '-g'})
            self.curves.append(
                {'label': 'at 18', 'xvals': xvals,
                 'yvals': yvals18, 'style': '-y'})
            self.curves.append(
                {'label': 'at 21', 'xvals': xvals,
                 'yvals': yvals21, 'style': '-c'})
            self.xtitle = 'zpos (mm)'
            if None in xvals:
                xvals = img_nos
                self.xtitle = 'Image number'
            self.ytitle = 'Difference (HU)'
            tolmax = {'label': 'tolerance max',
                      'xvals': [min(xvals), max(xvals)],
                      'yvals': [4, 4],
                      'style': '--k'}
            tolmin = tolmax.copy()
            tolmin['label'] = 'tolerance min'
            tolmin['yvals'] = [-4, -4]
            self.curves.append(tolmin)
            self.curves.append(tolmax)
            self.default_range_y = self.test_values_outside_yrange([-6, 6])
        elif self.main.current_modality == 'PET':
            self.title = '% difference from mean of all means'
            yvalsC = []
            yvals12 = []
            yvals15 = []
            yvals18 = []
            yvals21 = []
            for i, row in enumerate(self.main.results['Hom']['values']):
                if len(row) > 0:
                    img_nos.append(i)
                    xvals.append(self.zpos_all[i])
                    yvalsC.append(row[5])
                    yvals12.append(row[6])
                    yvals15.append(row[7])
                    yvals18.append(row[8])
                    yvals21.append(row[9])
            self.curves.append(
                {'label': 'Center', 'xvals': xvals,
                 'yvals': yvalsC, 'style': '-r'})
            self.curves.append(
                {'label': 'at 12', 'xvals': xvals,
                 'yvals': yvals12, 'style': '-b'})
            self.curves.append(
                {'label': 'at 15', 'xvals': xvals,
                 'yvals': yvals15, 'style': '-g'})
            self.curves.append(
                {'label': 'at 18', 'xvals': xvals,
                 'yvals': yvals18, 'style': '-y'})
            self.curves.append(
                {'label': 'at 21', 'xvals': xvals,
                 'yvals': yvals21, 'style': '-c'})
            self.xtitle = 'zpos (mm)'
            if None in xvals:
                xvals = img_nos
                self.xtitle = 'Image number'
            self.ytitle = '% difference'
            tolmax = {'label': 'tolerance max',
                      'xvals': [min(xvals), max(xvals)],
                      'yvals': [5, 5],
                      'style': '--k'}
            tolmin = tolmax.copy()
            tolmin['label'] = 'tolerance min'
            tolmin['yvals'] = [-5, -5]
            self.curves.append(tolmin)
            self.curves.append(tolmax)
            self.default_range_y = self.test_values_outside_yrange([-7, 7])

    def CTn(self):
        """Prepare plot for test CTn."""
        self.title = 'CT linearity'
        self.ytitle = 'Relative mass density'
        yvals = self.main.current_paramset.ctn_table.relative_mass_density
        imgno = self.main.gui.active_img_no
        xvals = self.main.results['CTn']['values'][imgno]
        self.curves.append(
            {'label': 'HU mean', 'xvals': xvals,
             'yvals': yvals, 'style': '-bo'})
        fit_r2 = self.main.results['CTn']['values_sup'][imgno][0]
        fit_b = self.main.results['CTn']['values_sup'][imgno][1]
        fit_a = self.main.results['CTn']['values_sup'][imgno][2]
        yvals = fit_a * np.array(xvals) + fit_b
        self.curves.append(
            {'label': 'fitted', 'xvals': xvals,
             'yvals': yvals, 'style': 'b:'}
            )
        at = matplotlib.offsetbox.AnchoredText(
            f'$R^2$ = {fit_r2:.4f}', loc='lower right')
        self.ax.add_artist(at)
        self.xtitle = 'HU value'

    def HUw(self):
        """Prepare plot for test HUw."""
        xvals = []
        yvals = []
        img_nos = []
        for i, row in enumerate(self.main.results['HUw']['values']):
            if len(row) > 0:
                img_nos.append(i)
                xvals.append(self.zpos_all[i])
                yvals.append(row[0])
        curve = {'label': 'Average HU',
                 'xvals': xvals,
                 'yvals': yvals,
                 'style': '-r'}
        self.curves.append(curve)
        self.xtitle = 'zpos (mm)'
        if None in xvals:
            xvals = img_nos
            self.xtitle = 'Image number'
        self.ytitle = 'Average HU'
        tolmax = {'label': 'tolerance max',
                  'xvals': [min(xvals), max(xvals)],
                  'yvals': [4, 4],
                  'style': '--k'}
        tolmin = tolmax.copy()
        tolmin['yvals'] = [-4, -4]
        tolmin['label'] = 'tolerance min'
        self.curves.append(tolmin)
        self.curves.append(tolmax)
        self.default_range_y = self.test_values_outside_yrange([-6, 6])

    def Sli(self):
        """Prepare plot for test Sli."""
        if self.main.current_modality in ['CT', 'MR']:
            self.title = 'Profiles for slice thickness calculations'
            imgno = self.main.gui.active_img_no
            details_dict = self.main.results['Sli']['details_dict'][imgno]
            n_pix = len(details_dict['profiles'][0])
            xvals = [details_dict['dx'] * i for i in range(n_pix)]

            # ROI h_colors = ['b', 'lime']
            # ROI v_colors = ['c', 'r', 'm', 'darkorange']
            if self.main.current_modality == 'CT':
                if self.main.current_paramset.sli_type == 0:
                    colors = ['b', 'lime', 'c', 'r']
                elif self.main.current_paramset.sli_type == 1:
                    colors = ['b', 'lime', 'c', 'r', 'm', 'darkorange']
                elif self.main.current_paramset.sli_type == 2:
                    colors = ['c', 'r']
                if self.main.tab_ct.sli_plot.currentIndex() == 0:  # plot all
                    l_idxs = list(np.arange(len(details_dict['profiles'])))
                else:
                    l_idxs = [self.main.tab_ct.sli_plot.currentIndex() - 1]
            else:
                colors = ['b', 'lime']
                if self.main.tab_mr.sli_plot.currentIndex() == 0:  # plot both
                    l_idxs = list(np.arange(len(details_dict['profiles'])))
                else:
                    l_idxs = [self.main.tab_mr.sli_plot.currentIndex() - 1]

            for l_idx in l_idxs:
                self.curves.append({'label': details_dict['labels'][l_idx],
                                    'xvals': xvals,
                                    'yvals': details_dict['profiles'][l_idx],
                                    'color': colors[l_idx]})
                try:
                    self.curves.append({'label': details_dict['labels'][l_idx],
                                        'xvals': xvals,
                                        'yvals': details_dict[
                                            'envelope_profiles'][l_idx],
                                        'style': '--',
                                        'color': colors[l_idx]})
                except IndexError:
                    pass
                self.curves.append({
                    'label': '_nolegend_',
                    'xvals': [min(xvals), max(xvals)],
                    'yvals': [details_dict['background'][l_idx]] * 2,
                    'style': ':',
                    'color': colors[l_idx]})
                self.curves.append({
                    'label': '_nolegend_',
                    'xvals': [min(xvals), max(xvals)],
                    'yvals': [details_dict['peak'][l_idx]] * 2,
                    'style': ':',
                    'color': colors[l_idx]})
                self.curves.append({
                    'label': '_nolegend_',
                    'xvals': [details_dict['start_x'][l_idx],
                              details_dict['end_x'][l_idx]],
                    'yvals': [details_dict['halfpeak'][l_idx]] * 2,
                    'style': '--',
                    'color': colors[l_idx]})
            self.xtitle = 'pos (mm)'
            self.ytitle = 'HU' if self.main.current_modality == 'CT' else 'Pixel value'

    def MTF(self):
        """Prepare plot for test MTF."""
        imgno = self.main.gui.active_img_no
        rowno = imgno
        if self.main.results['MTF']['pr_image']:
            details_dicts = self.main.results['MTF']['details_dict'][imgno]
            if isinstance(details_dicts, dict):
                details_dicts = [details_dicts]
        else:
            details_dicts = self.main.results['MTF']['details_dict']
            rowno = 0

        def prepare_plot_MTF():
            if 'dMTF_details' in details_dicts[0]:
                nyquist_freq = 1/(2.*self.main.imgs[imgno].pix[0])
                try:
                    mtf_cy_pr_mm = self.main.current_paramset.mtf_cy_pr_mm
                except AttributeError:
                    mtf_cy_pr_mm = True
                if mtf_cy_pr_mm is False:
                    nyquist_freq = 10. * nyquist_freq

                self.xtitle = 'frequency [1/mm]' if mtf_cy_pr_mm else 'frequency [1/cm]'
                self.ytitle = 'MTF'

                colors = ['k', 'r']  # gaussian black, discrete red
                linestyles = ['-', '--']
                infotext = ['gaussian', 'discrete']
                prefix = ['g', 'd']
                suffix = [' x', ' y'] if len(details_dicts) >= 2 else ['']
                for ddno in range(2):
                    try:
                        dd = details_dicts[ddno]
                        proceed = True
                    except IndexError:
                        proceed = False
                    if proceed:
                        for no in range(len(prefix)):
                            key = f'{prefix[no]}MTF_details'
                            if key in dd:
                                dd_this = dd[key]
                                xvals = dd_this['MTF_freq']
                                if mtf_cy_pr_mm is False:
                                    xvals = 10. * xvals  # convert to /cm
                                yvals = dd_this['MTF']
                                self.curves.append({
                                    'label': infotext[no] + ' MTF' + suffix[ddno],
                                    'xvals': xvals,
                                    'yvals': yvals,
                                    'style': linestyles[ddno] + colors[no]
                                     })

                if 'MTF_filtered' in details_dicts[0]['gMTF_details']:
                    yvals = details_dicts[0]['gMTF_details']['MTF_filtered']
                    if yvals is not None:
                        xvals = details_dicts[0]['gMTF_details']['MTF_freq']
                        if mtf_cy_pr_mm is False:
                            xvals = 10. * xvals  # convert to /cm
                        yvals = details_dicts[0]['gMTF_details']['MTF_filtered']
                        self.curves.append({
                            'label': 'gaussian MTF pre-smoothed',
                            'xvals': xvals,
                            'yvals': yvals,
                            'style': '--',
                            'color': 'gray'
                             })

                self.default_range_y = self.test_values_outside_yrange(
                    [0, 1.3], limit_xrange=[0, nyquist_freq])
                self.default_range_x = [0, 1.1 * nyquist_freq]
                self.curves.append({
                    'label': '_nolegend_',
                    'xvals': [nyquist_freq, nyquist_freq],
                    'yvals': [0, 1.3],
                    'style': ':k'
                     })
                self.ax.text(0.9*nyquist_freq, 0.5, 'Nyquist frequency',
                             ha='left', size=8, color='gray')

                # MTF %
                if self.main.current_modality != 'NM':
                    values = self.main.results[self.main.current_test]['values'][rowno]
                    if self.main.current_modality == 'Xray':
                        yvals = [[.5, .5]]
                        xvals = [[0, values[-1]]]
                        yvals.extend([[0, values[i]] for i in range(5)])
                        xvals.extend([[.5, .5], [1, 1], [1.5, 1.5], [2, 2], [2.5, 2.5]])
                        for i in range(len(xvals)):
                            self.curves.append({
                                'label': '_nolegend_',
                                'xvals': xvals[i],
                                'yvals': yvals[i],
                                'style': ':k'
                                 })

                    else:
                        yvals = [[0, .5], [0, .1], [0, .02]]
                        factor = 10 if mtf_cy_pr_mm is False else 1
                        xvals = []
                        for i in range(3):
                            try:
                                val_this = factor*values[i]
                                xvals.append([val_this, val_this])
                            except TypeError:
                                xvals.append(None)
                        for i in range(len(xvals)):
                            if xvals[i] is not None:
                                self.curves.append({
                                    'label': '_nolegend_',
                                    'xvals': xvals[i],
                                    'yvals': yvals[i],
                                    'style': ':k'
                                     })

        def prepare_plot_LSF():
            self.xtitle = 'pos (mm)'
            self.ytitle = 'LSF'
            self.legend_location = 'upper left'

            linestyles = ['-', '--']
            suffix = [' x', ' y'] if len(details_dicts) >= 2 else ['']
            lbl_prefilter = ''
            prefilter = False
            if 'sigma_prefilter' in details_dicts[0]:
                if details_dicts[0]['sigma_prefilter'] > 0:
                    prefilter = True
                    lbl_prefilter = ' presmoothed'
            for ddno in range(2):
                try:
                    dd = details_dicts[ddno]
                    proceed = True
                except IndexError:
                    proceed = False
                if proceed:
                    xvals = dd['LSF_x']
                    yvals = dd['LSF']
                    self.curves.append({
                        'label': 'LSF' + suffix[ddno],
                        'xvals': xvals,
                        'yvals': yvals,
                        'style': linestyles[ddno] + 'r'
                         })
                    dd_this = dd['gMTF_details']
                    if prefilter:
                        xvals = dd_this['LSF_fit_x']
                        yvals = dd_this['LSF_prefit']
                        self.curves.append({
                            'label': f'LSF{lbl_prefilter}' + suffix[ddno],
                            'xvals': xvals,
                            'yvals': yvals,
                            'style': linestyles[ddno] + 'b'
                             })
                    xvals = dd_this['LSF_fit_x']
                    yvals = dd_this['LSF_fit']
                    self.curves.append({
                        'label': f'LSF{lbl_prefilter} - gaussian fit' + suffix[ddno],
                        'xvals': xvals,
                        'yvals': yvals,
                        'style': linestyles[ddno] + 'k'
                         })

                    if ddno == 0 and 'dMTF_details' in dd:
                        dd_this = dd['dMTF_details']
                        if 'cut_width' in dd_this:
                            cw = dd_this['cut_width']
                            if cw > 0:
                                minmax = [np.min(yvals), np.max(yvals)]
                                for x in [-1, 1]:
                                    self.curves.append({
                                        'label': '_nolegend_',
                                        'xvals': [x * cw] * 2,
                                        'yvals': minmax,
                                        'style': ':k'
                                        })
                                    self.ax.text(
                                        x * cw, np.mean(minmax), 'cut',
                                        ha='left', size=8, color='gray')
                                if 'cut_width_fade' in dd_this:
                                    cwf = dd_this['cut_width_fade']
                                    for x in [-1, 1]:
                                        self.curves.append({
                                            'label': '_nolegend_',
                                            'xvals': [x * cwf] * 2,
                                            'yvals': minmax,
                                            'style': ':k'
                                            })
                                        self.ax.text(
                                            x * cwf, np.mean(minmax), 'fade',
                                            ha='left', size=8, color='gray')
                                self.default_range_x = [-1.5*cw, 1.5*cw]

        def prepare_plot_sorted_pix():
            try:
                xvals = details_dicts[0]['sorted_pixels_x']
                proceed = True
            except KeyError:
                proceed = False
            sorted_pixels = []
            xvals = []
            if proceed is False:
                edge_details_dicts = []
                for i in range(len(details_dicts)):
                    if 'edge_details' in details_dicts[i]:
                        edge_details_dicts.append(details_dicts[i]['edge_details'])
                if len(edge_details_dicts) == 1:
                    try:
                        if isinstance(edge_details_dicts[0], list):
                            for i in range(len(edge_details_dicts[0])):
                                sorted_pixels.append(
                                    edge_details_dicts[0][i]['sorted_pixels'])
                                xvals.append(
                                    edge_details_dicts[0][i]['sorted_pixels_x'])
                        else:
                            sorted_pixels = edge_details_dicts[0]['sorted_pixels']
                            xvals = edge_details_dicts[0]['sorted_pixels_x']
                        proceed = True
                    except KeyError:
                        proceed = False
                elif len(edge_details_dicts) > 1:
                    for i in range(len(edge_details_dicts)):
                        if 'sorted_pixels' in edge_details_dicts[i][0]:
                            try:
                                xvals.append(
                                    edge_details_dicts[i][0]['sorted_pixels_x'])
                                sorted_pixels.append(
                                    edge_details_dicts[i][0]['sorted_pixels'])
                                proceed = True
                            except KeyError:
                                proceed = False

            if proceed:
                self.xtitle = 'pos (mm)'
                self.ytitle = 'Pixel value'
                xy_labels = False
                lsf_is_interp = False
                try:
                    mtf_type = self.main.current_paramset.mtf_type
                    if self.main.current_modality in ['CT', 'SPECT'] and mtf_type == 1:
                        self.ytitle = 'Summed pixel values'
                        xy_labels = True
                        lsf_is_interp = True
                except AttributeError:
                    pass
                if len(sorted_pixels) == 0:
                    if len(details_dicts) > 1:
                        for i in range(len(details_dicts)):
                            if 'sorted_pixels' in details_dicts[i]:
                                sorted_pixels.append(details_dicts[i]['sorted_pixels'])
                                xvals.append(details_dicts[i]['sorted_pixels_x'])
                    else:
                        sorted_pixels = details_dicts[0]['sorted_pixels']
                        xvals = details_dicts[0]['sorted_pixels_x']
                if not isinstance(sorted_pixels, list):
                    sorted_pixels = [sorted_pixels]
                if not isinstance(xvals, list):
                    xvals = [xvals]

                colors = ['r', 'b', 'g', 'c']
                dotcolors = ['darksalmon', 'cornflowerblue',
                             'mediumseagreen', 'paleturquoise']  # matching r,b,g,c
                if xy_labels:
                    suffix = [' x', ' y']
                else:
                    suffix = [f' {x}' for x in range(len(sorted_pixels))]
                for no, yvals in enumerate(sorted_pixels):
                    if len(xvals) == len(sorted_pixels):
                        xvals_this = xvals[no]
                    else:
                        xvals_this = xvals[0]

                    self.curves.append({
                        'label': f'Sorted pixels{suffix[no]}',
                        'xvals': xvals_this,
                        'yvals': yvals,
                        'style': '.',
                        'color': dotcolors[no % len(dotcolors)],
                        'markersize': 2.,
                         })

                if 'interpolated_x' in details_dicts[0]:
                    if len(details_dicts) > 1:
                        interpolated = []
                        xvals = []
                        for i in range(len(details_dicts)):
                            if 'interpolated' in details_dicts[i]:
                                interpolated.append(details_dicts[i]['interpolated'])
                                xvals.append(details_dicts[i]['interpolated_x'])
                    else:
                        interpolated = details_dicts[0]['interpolated']
                        xvals = details_dicts[0]['interpolated_x']
                        if not isinstance(interpolated, list):
                            interpolated = [interpolated]
                        if not isinstance(xvals, list):
                            xvals = [xvals]

                    for no, yvals in enumerate(interpolated):
                        if len(xvals) == len(interpolated):
                            xvals_this = xvals[no]
                            lbl = f'Interpolated{suffix[no]}'
                        else:
                            xvals_this = xvals[0]
                            lbl = 'Interpolated'
                        self.curves.append({
                                'label': lbl,
                                'xvals': xvals_this,
                                'yvals': yvals,
                                'style': '-',
                                'color': colors[no % len(colors)]
                                 })

                    if 'presmoothed' in details_dicts[0]:
                        yvals = details_dicts[0]['presmoothed']
                        self.curves.append({
                            'label': 'Presmoothed',
                            'xvals': xvals[0],
                            'yvals': yvals,
                            'style': '-b'
                             })

                if 'ESF' in details_dicts[0]:
                    if isinstance(details_dicts[0]['ESF'], list):
                        xvals = details_dicts[0]['ESF_x']
                        for no, ESF in enumerate(details_dicts[0]['ESF']):
                            lbl = f' {no}' if len(details_dicts[0]['ESF']) > 1 else ''
                            self.curves.append({
                                'label': f'ESF{lbl}',
                                'xvals': xvals,
                                'yvals': ESF,
                                'style': f'-{colors[no]}'
                                 })

                if lsf_is_interp:
                    for no in range(len(details_dicts)):
                        if 'LSF' in details_dicts[no]:
                            self.curves.append({
                                'label': f'Interpolated{suffix[no]}',
                                'xvals': details_dicts[no]['LSF_x'],
                                'yvals': details_dicts[no]['LSF'],
                                'style': f'-{colors[no]}'
                                 })

        def prepare_plot_centered_profiles():
            proceed = True
            if 'matrix' not in details_dicts[0]:
                proceed = False
            elif self.main.current_modality in ['CT', 'SPECT']:
                if self.main.current_paramset.mtf_type == 1:
                    proceed = False
            elif self.main.current_modality == 'NM':
                if self.main.current_paramset.mtf_type > 0:
                    proceed = False

            if proceed:
                self.xtitle = 'pos (mm)'
                self.ytitle = 'Pixel value'

                linestyles = ['-', '--']  # x, y
                colors = ['g', 'b', 'r', 'k', 'c', 'm']
                if len(details_dicts) == 2:
                    center_xy = [details_dicts[i]['center'] for i in range(2)]
                    submatrix = [details_dicts[0]['matrix']]
                else:
                    center_xy = details_dicts[0]['center_xy']
                    submatrix = details_dicts[0]['matrix']

                marked_imgs = self.main.tree_file_list.get_marked_imgs_current_test()
                pix = self.main.imgs[marked_imgs[0]].pix[0]

                for no, sli in enumerate(submatrix):
                    if self.main.results['MTF']['pr_image'] is False:
                        if no not in marked_imgs:
                            proceed = False
                    if proceed:
                        suffix = f' {no}' if len(submatrix) > 1 else ''
                        szy, szx = sli.shape
                        xvals1 = pix * (np.arange(szx) - center_xy[0])
                        yvals1 = sli[round(center_xy[1]), :]
                        self.curves.append({
                            'label': 'x' + suffix,
                            'xvals': xvals1,
                            'yvals': yvals1,
                            'style': linestyles[0] + colors[no % len(colors)]
                             })
                        xvals2 = pix * (np.arange(szy) - center_xy[1])
                        yvals2 = sli[:, round(center_xy[0])]
                        self.curves.append({
                            'label': 'y' + suffix,
                            'xvals': xvals2,
                            'yvals': yvals2,
                            'style': linestyles[1] + colors[no % len(colors)]
                             })

        def prepare_plot_edge_position():
            self.xtitle = 'pos (mm)'
            self.ytitle = 'ROI row index'

            if 'edge_details' in details_dicts[0]:
                eds = details_dicts[0]['edge_details']
                colors = ['r', 'b', 'g', 'c']
                deg = '\N{DEGREE SIGN}'
                txt_info = []
                for edno, ed in enumerate(eds):
                    lbl = f' {edno}' if len(eds) > 1 else ''
                    xvals = ed['edge_pos']
                    yvals = ed['edge_row']
                    self.curves.append({
                        'label': f'edge{lbl}',
                        'xvals': xvals,
                        'yvals': yvals,
                        'style': '.',
                        'color': 'darkgray',
                        'markersize': 2.,
                         })
                    xvals = ed['edge_fit_x']
                    yvals = ed['edge_fit_y']
                    self.curves.append({
                        'label': f'fit edge{lbl}',
                        'xvals': xvals,
                        'yvals': yvals,
                        'style': f'-{colors[edno]}'
                         })
                    lbl = f'{edno}: ' if len(eds) > 1 else ''
                    txt_info.append(
                        (f'{lbl}$R^2$ = {ed["edge_r2"]:.4f}, angle = '
                         f'{ed["angle"]:.2f}{deg}')
                        )

                at = matplotlib.offsetbox.AnchoredText(
                    '\n'.join(txt_info), loc='lower right')
                self.ax.add_artist(at)

        def prepare_plot_zprofile():
            self.xtitle = 'z position (mm)'
            self.ytitle = 'Max pixel value in ROI'

            if 'zpos_used' in details_dicts[-1]:
                common_details = details_dicts[-1]
                self.curves.append({
                    'label': 'Max in marked images',
                    'xvals': common_details['zpos_marked_images'],
                    'yvals': common_details['max_roi_marked_images'],
                    'style': '-b'})
                self.curves.append({
                    'label': 'Max in used images',
                    'xvals': common_details['zpos_used'],
                    'yvals': common_details['max_roi_used'],
                    'style': '-r'})

        test_widget = self.main.stack_test_tabs.currentWidget()
        try:
            sel_text = test_widget.mtf_plot.currentText()
        except AttributeError:
            sel_text = ''
        if sel_text == 'MTF':
            prepare_plot_MTF()
        elif sel_text == 'LSF':
            prepare_plot_LSF()
        elif sel_text == 'Sorted pixel values':
            prepare_plot_sorted_pix()
        elif sel_text == 'Centered xy profiles':
            prepare_plot_centered_profiles()
        elif sel_text in ['Edge position', 'Line fit']:
            prepare_plot_edge_position()
        elif 'z-profile' in sel_text:
            prepare_plot_zprofile()
        self.title = sel_text

    def Rin(self):
        """Prepare plot for test Rin."""
        imgno = self.main.gui.active_img_no
        self.title = 'Radial profile'
        self.ytitle = 'HU'
        self.xtitle = 'Position from center (mm)'
        try:
            details_dict = self.main.results['Rin']['details_dict'][imgno]
            xvals = details_dict['radial_profile_x']
            self.curves.append(
                {'label': 'Radial profile', 'xvals': xvals,
                 'yvals': details_dict['radial_profile'], 'style': '-b'})
            if 'radial_profile_smoothed' in details_dict:
                self.curves.append(
                    {'label': 'Radial profile smoothed', 'xvals': xvals,
                     'yvals': details_dict['radial_profile_smoothed'], 'style': '-k'})
            if 'radial_profile_trend' in details_dict:
                self.curves.append(
                    {'label': 'Radial profile trend', 'xvals': xvals,
                     'yvals': details_dict['radial_profile_trend'], 'style': '-r'})
            else:
                self.curves.append(
                    {'label': 'Radial profile mean', 'xvals': [xvals[0], xvals[-1]],
                     'yvals': [details_dict['mean_profile']] * 2, 'style': '-r'})
        except (KeyError, IndexError):
            pass

    def NPS(self):
        """Prepare plot for test NPS."""
        imgno = self.main.gui.active_img_no
        self.title = 'Noise Power Spectrum'
        self.ytitle = r'NPS ($\mathregular{mm^{2}}$)'
        self.xtitle = 'NPS (pr mm)'
        normalize = self.main.current_paramset.nps_normalize

        if self.main.current_modality == 'Xray':
            test_widget = self.main.stack_test_tabs.currentWidget()
            sel_text = test_widget.nps_plot_profile.currentText()
            profile_keys = []
            info_texts = []
            styles = []
            if sel_text == 'all':
                profile_keys = ['radial_profile', 'u_profile', 'v_profile']
                info_texts = [' radial', ' horizontal', ' vertical']
                styles = ['-b', '-m', '-c']
            else:
                if sel_text == 'radial':
                    profile_keys = ['radial_profile']
                    info_texts.append(' radial')
                    styles.append('-b')
                if 'horizontal' in sel_text:
                    profile_keys.append('u_profile')
                    info_texts.append(' horizontal')
                    styles.append('-m')
                if 'vertical' in sel_text:
                    profile_keys.append('v_profile')
                    info_texts.append(' vertical')
                    styles.append('-c')
        else:
            profile_keys = ['radial_profile']
            info_texts = ['']
            styles = ['-b']

        def plot_current_NPS():
            try:
                details_dict = self.main.results['NPS']['details_dict'][imgno]
                max_ys = []
                for i, prof_key in enumerate(profile_keys):
                    if normalize == 1:
                        AUC = details_dict[f'{prof_key}_AUC']
                        norm_factor = 1/AUC
                    elif normalize == 2:
                        norm_factor = 1/(details_dict['large_area_signal']**2)
                    else:
                        norm_factor = 1
                    yvals = norm_factor * details_dict[prof_key]
                    if prof_key == 'radial_profile':
                        xvals = details_dict['freq']
                    else:
                        xvals = details_dict['freq_uv']
                    self.curves.append(
                        {'label': f'img {imgno}{info_texts[i]}',
                         'xvals': xvals, 'yvals': yvals, 'style': styles[i]})
                    max_ys.append(np.max(yvals))

                if self.main.current_modality == 'CT':
                    self.curves.append({
                        'label': '_nolegend_',
                        'xvals': [
                            details_dict['median_freq'], details_dict['median_freq']],
                        'yvals': [0, norm_factor * details_dict['median_val']],
                        'style': '-b'})

                nyquist_freq = 1/(2.*self.main.imgs[imgno].pix[0])
                maxy = max(max_ys)
                self.curves.append({
                    'label': '_nolegend_',
                    'xvals': [nyquist_freq, nyquist_freq],
                    'yvals': [0, maxy],
                    'style': ':k'
                     })
                self.ax.text(
                    0.9*nyquist_freq, 0.5*maxy,
                    'Nyquist frequency', ha='left', size=8, color='gray')
            except (KeyError, IndexError):
                pass

        def plot_average_NPS():  # only for CT
            try:
                dicts = self.main.results['NPS']['details_dict']
                xvals = None
                yvals = None
                n_profiles = 0
                for i, details_dict in enumerate(dicts):
                    proceed = True
                    if details_dict:
                        if xvals is not None:
                            xvals_this = details_dict['freq']
                            if (xvals != xvals_this).all():
                                proceed = False
                        else:
                            xvals = details_dict['freq']
                    if proceed is False:
                        errmsg = 'Failed plotting average NPS. Not same pixel sizes.'
                        QMessageBox.information(self, 'Failed averaging', errmsg)
                        xvals = None
                        yvals = None
                        n_profiles = 0
                        break
                    else:
                        n_profiles += 1
                        if normalize == 1:
                            AUC = self.main.results['NPS']['values'][i][1]
                            norm_factor = 1/AUC
                        elif normalize == 2:
                            norm_factor = 1/(details_dict['large_area_signal']**2)
                        else:
                            norm_factor = 1
                        if yvals is None:
                            yvals = norm_factor * details_dict['radial_profile']
                        else:
                            yvals = yvals + norm_factor * details_dict['radial_profile']
                if n_profiles > 0:
                    y_avg = 1/n_profiles * yvals

                    self.curves.append(
                        {'label': 'average NPS', 'xvals': xvals, 'yvals': y_avg,
                         'style': '-k'})
                    median_frequency, median_val = find_median_spectrum(xvals, y_avg)
                    self.curves.append({
                        'label': '_nolegend_',
                        'xvals': [median_frequency, median_frequency],
                        'yvals': [0, median_val],
                        'style': '-k'})
            except (KeyError, IndexError):
                pass

        def plot_all_NPS():
            try:
                imgno = self.main.gui.active_img_no
                dicts = self.main.results['NPS']['details_dict']
                curve_this = None
                med_this = None
                for i, details_dict in enumerate(dicts):
                    if details_dict:
                        max_ys = []
                        for prof_key in profile_keys:
                            if normalize == 1:
                                AUC = details_dict[f'{prof_key}_AUC']
                                norm_factor = 1/AUC
                            elif normalize == 2:
                                norm_factor = 1/(details_dict['large_area_signal']**2)
                            else:
                                norm_factor = 1
                            yvals = norm_factor * details_dict[prof_key]
                            if prof_key == 'radial_profile':
                                xvals = details_dict['freq']
                            else:
                                xvals = details_dict['freq_uv']
                            color = 'r' if i == imgno else 'darkgray'
                            curve = {'label': f'img {i}',
                                     'xvals': xvals, 'yvals': yvals, 'color': color}
                            max_ys.append(np.max(yvals))

                            if i == imgno:
                                curve_this = curve
                            else:
                                self.curves.append(curve)
                            if self.main.current_modality == 'CT':
                                curve_med = {
                                    'label': '_nolegend_',
                                    'xvals': [details_dict['median_freq'],
                                              details_dict['median_freq']],
                                    'yvals': [0,
                                              norm_factor * details_dict['median_val']],
                                    'color': color}
                                if i == imgno:
                                    med_this = curve_med
                                else:
                                    self.curves.append(curve_med)

                if curve_this is not None:
                    self.curves.append(curve_this)
                if med_this is not None:
                    self.curves.append(med_this)
            except (KeyError, IndexError):
                pass

        test_widget = self.main.stack_test_tabs.currentWidget()
        try:
            sel_text = test_widget.nps_plot.currentText()
        except AttributeError:
            sel_text = ''
        if 'pr image' in sel_text:
            plot_current_NPS()
        if 'average' in sel_text:
            plot_average_NPS()
        if 'NPS all images' in sel_text:
            plot_all_NPS()
        self.title = sel_text

    def Uni(self):
        """Prepare plot for test Uni."""
        plot_idx = self.main.tab_nm.uni_plot.currentIndex()
        if plot_idx == 0:
            self.title = 'Uniformity result for all images'
            yvals_iu_ufov = []
            yvals_du_ufov = []
            yvals_iu_cfov = []
            yvals_du_cfov = []
            xvals = []
            for i, row in enumerate(self.main.results['Uni']['values']):
                if len(row) > 0:
                    xvals.append(i)
                    yvals_iu_ufov.append(row[0])
                    yvals_du_ufov.append(row[1])
                    yvals_iu_cfov.append(row[2])
                    yvals_du_cfov.append(row[3])
            if len(xvals) > 1:
                self.xtitle = 'Image number'

            self.curves.append(
                {'label': 'IU UFOV', 'xvals': xvals,
                 'yvals': yvals_iu_ufov, 'style': '-bo'})
            self.curves.append(
                {'label': 'DU UFOV', 'xvals': xvals,
                 'yvals': yvals_du_ufov, 'style': '-ro'})
            self.curves.append(
                {'label': 'IU CFOV', 'xvals': xvals,
                 'yvals': yvals_iu_cfov, 'style': ':bo'})
            self.curves.append(
                {'label': 'DU CFOV', 'xvals': xvals,
                 'yvals': yvals_du_cfov, 'style': ':ro'})
            self.xtitle = 'Image number'
            self.ytitle = 'Uniformity %'
            self.default_range_y = self.test_values_outside_yrange([0, 7])
        elif plot_idx == 1:
            self.title = 'Curvature correction check'
            imgno = self.main.gui.active_img_no
            details_dict = self.main.results['Uni']['details_dict'][imgno]
            if 'correction_matrix' in details_dict:
                # averaging central 10% rows/cols
                temp_img = self.main.active_img
                corrected_img = details_dict['corrected_image']
                sz_y, sz_x = corrected_img.shape
                nx = round(0.05 * sz_x)
                ny = round(0.05 * sz_y)
                xhalf = round(sz_x/2)
                yhalf = round(sz_y/2)
                prof_y = np.mean(temp_img[:, xhalf-nx:xhalf+nx], axis=1)
                prof_x = np.mean(temp_img[yhalf-ny:yhalf+ny, :], axis=0)
                corr_prof_y = np.mean(
                    corrected_img[:, xhalf-nx:xhalf+nx], axis=1)
                corr_prof_x = np.mean(
                    corrected_img[yhalf-ny:yhalf+ny, :], axis=0)
                self.curves.append({'label': 'Central 10% rows corrected',
                                    'xvals': np.arange(len(corr_prof_x)),
                                    'yvals': corr_prof_x,
                                    'style': 'r'})
                self.curves.append({'label': 'Central 10% rows original',
                                    'xvals': np.arange(len(prof_x)),
                                    'yvals': prof_x,
                                    'style': ':r'})
                self.curves.append({'label': 'Central 10% columns corrected',
                                    'xvals': np.arange(len(corr_prof_y)),
                                    'yvals': corr_prof_y,
                                    'style': 'b'})
                self.curves.append({'label': 'Central 10% columns original',
                                    'xvals': np.arange(len(prof_y)),
                                    'yvals': prof_y,
                                    'style': ':b'})
                self.xtitle = 'pixel number'
                self.ytitle = 'Average pixel value'
                self.legend_location = 'lower center'
            else:
                at = matplotlib.offsetbox.AnchoredText(
                    'No curvature correction applied',
                    prop=dict(size=self.main.gui.annotations_font_size,
                              color='red'),
                    frameon=False, loc='upper left')
                self.ax.add_artist(at)

    def SNI(self):
        """Prepare plot for test NM SNI test."""
        self.title = 'Scan speed profile'
        self.xtitle = 'Position (mm)'
        self.ytitle = 'Difference from mean %'
        imgno = self.main.gui.active_img_no
        details_dict = self.main.results['SNI']['details_dict'][imgno]
        styles = ['-r', '-b', '-lime', '-cyan', '-m', '--lime', '--cyan', '--m']
        for i in range(8):
            self.curves.append(
                {'label': 'profile', 'xvals': details_dict['freq'][i],
                 'yvals': details_dict['rNPS'][i], 'style': styles[i]
                 })

        # display eye filter
        #self.curves.append(
        #    {'label': 'Eye filter', 'xvals': details_dict['freq'][i],
        #     'yvals': details_dict['rNPS'][i], 'style': '.k'}
        #    )
        
        # quantum noise for L1

    def Spe(self):
        """Prepare plot for test NM Speed test."""
        self.title = 'Scan speed profile'
        self.xtitle = 'Position (mm)'
        self.ytitle = 'Difference from mean %'
        imgno = self.main.gui.active_img_no
        details_dict = self.main.results['Spe']['details_dict'][imgno]
        xvals = details_dict['profile_pos']
        self.curves.append(
            {'label': 'profile', 'xvals': xvals,
             'yvals': details_dict['diff_profile'], 'style': '-b'})
        tolmax = {'label': 'tolerance max',
                  'xvals': [min(xvals), max(xvals)],
                  'yvals': [5, 5],
                  'style': '--k'}
        tolmin = tolmax.copy()
        tolmin['yvals'] = [-5, -5]
        tolmin['label'] = 'tolerance min'
        self.curves.append(tolmin)
        self.curves.append(tolmax)

        self.default_range_y = self.test_values_outside_yrange([-7, 7])

    def Cro(self):
        """Prepare plot for test PET cross calibration."""
        self.title = 'z profile'
        self.xtitle = 'Slice position (mm)'
        self.ytitle = 'Average in ROI (Bq/ml)'
        details_dict = self.main.results['Cro']['details_dict']
        self.curves.append(
            {'label': 'all slices', 'xvals': details_dict['zpos'],
             'yvals': details_dict['roi_averages'], 'style': '-k'})
        self.curves.append(
            {'label': 'used slices', 'xvals': details_dict['used_zpos'],
             'yvals': details_dict['used_roi_averages'], 'style': '-r'})

    def vendor(self):
        """Prepare plot if vendor test results contain details."""
        # Currently only energy spectrum from Siemens gamma camera have this option.
        self.title = 'Energy spectrum'
        self.xtitle = 'Energy (keV)'
        self.ytitle = 'Counts'
        details_list = self.main.results['vendor']['details']
        colors = ['b', 'lime', 'c', 'r', 'm', 'darkorange']
        for i, details in enumerate(details_list):
            col_i = i % len(colors)
            self.curves.append(
                {'label': f'file {i}', 'xvals': details['curve_energy'],
                 'yvals': details['curve_counts'], 'color': colors[col_i]})
            self.curves.append(
                {'label': '_nolegend_', 'xvals': [details['keV_max']]*2,
                 'yvals': [0, np.max(details['curve_counts'])],
                 'style': ':', 'color': colors[col_i]})
            self.curves.append(
                {'label': '_nolegend_',
                 'xvals': details['keV_fwhm_start_stop'],
                 'yvals': [0.5*np.max(details['curve_counts'])]*2,
                 'style': ':', 'color': colors[col_i]})
            self.default_range_x = [
                details['keV_fwhm_start_stop'][0]*0.5,
                details['keV_fwhm_start_stop'][1]*1.5
                ]


class ResultImageWidget(GenericImageWidget):
    """Results image widget."""

    def __init__(self, main):
        canvas = ui_image_canvas.ResultImageCanvas(self, main)
        super().__init__(main, canvas)
        self.main = main
        toolb = ResultImageNavigationToolbar(self.canvas, self)
        hlo = QHBoxLayout()

        tbm = QToolBar()
        self.tool_resultsize = ToolMaximizeResults(self.main)
        tbm.addWidget(self.tool_resultsize)
        tbm.setOrientation(Qt.Vertical)
        tbm.addWidget(self.tool_profile)

        tb_top = QToolBar()
        tb_top.addWidget(GenericImageToolbarPosVal(self.canvas, self))
        self.image_title = QLabel()
        tb_top.addWidget(self.image_title)
        empty = QWidget()
        empty.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb_top.addWidget(empty)

        hlo.addWidget(toolb)
        vlo_mid = QVBoxLayout()
        vlo_mid.addWidget(tb_top)
        vlo_mid.addWidget(self.canvas)
        hlo.addLayout(vlo_mid)
        hlo.addWidget(tbm)
        self.setLayout(hlo)


class ResultImageNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib navigation toolbar with some modifications."""

    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        for act in self.actions():
            if act.text() in ['Back', 'Forward', 'Pan', 'Subplots']:
                self.removeAction(act)
        self.setOrientation(Qt.Vertical)

    def set_message(self, s):
        """Hide cursor position and value text by overriding method set_message."""
        pass