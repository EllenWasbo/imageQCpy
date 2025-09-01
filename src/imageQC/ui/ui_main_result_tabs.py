#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for main window of imageQC - result tabs part.

@author: Ellen Wasbo
"""
import os
import copy
import numpy as np
import pandas as pd

from PyQt6.QtGui import QIcon, QAction, QKeyEvent, QKeySequence
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QToolButton, QMenu,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QAbstractScrollArea,
    QSplitter, QLabel
    )
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt

# imageQC block start
from imageQC.ui import ui_image_canvas
from imageQC.ui.ui_main_image_widgets import (
    GenericImageWidget, GenericImageToolbarPosVal, ImageNavigationToolbar)
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.plot_widgets import PlotWidget, PlotCanvas
from imageQC.config.iQCconstants import ENV_ICON_PATH, COLORS
from imageQC.scripts import mini_methods_format as mmf
from imageQC.scripts.mini_methods_calculate import (
    find_median_spectrum, get_avg_NPS_curve)
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
        toolb.setOrientation(Qt.Orientation.Vertical)
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
        toolb.setOrientation(Qt.Orientation.Vertical)
        hlo.addWidget(toolb)

    def copy_table(self, row_range=None, col_range=None):
        """Copy contents of table to clipboard."""
        decimal_mark = '.'
        if self.tb_copy.tool_decimal.isChecked():
            decimal_mark = ','
        lock_format = self.tb_copy.tool_decimal_all.isChecked()
        values = [
            mmf.val_2_str(
                col,
                decimal_mark=decimal_mark,
                lock_format=lock_format)
            for col in self.result_table.values]

        if row_range is not None and col_range is not None:
            values = values[col_range[0]: col_range[1]+1]
            values = [col[row_range[0]: row_range[1]+1] for col in values]

        if self.tb_copy.tool_header.isChecked():  # insert headers
            if self.result_table.row_labels[0] == '':
                if col_range is None or col_range is False:
                    for i in range(len(values)):
                        values[i].insert(0, self.result_table.col_labels[i])
                else:
                    col_labels = self.result_table.col_labels[
                        col_range[0]:col_range[1]+1]
                    for i in range(len(values)):
                        values[i].insert(0, col_labels[i])
            else:
                # row headers true headers
                if row_range is None or row_range is False:
                    values.insert(0, self.result_table.row_labels)
                else:
                    values.insert(0, self.result_table.row_labels[
                        row_range[0]:row_range[1]+1])
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
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.values = [[]]  # always as columns, converted if input is rows
        self.row_labels = []
        self.col_labels = []
        self.installEventFilter(self)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.generate_ctxmenu)

    def generate_ctxmenu(self, pos):
        """Dropdown menu on right click."""
        menu = QMenu(self)
        act_copy = menu.addAction('Copy selected cells to clipboard (Ctrl+C)')
        act_copy.triggered.connect(self.copy_selection)
        menu.exec(self.mapToGlobal(pos))

    def eventFilter(self, source, event):
        """Handle arrow up/down events."""
        if isinstance(event, QKeyEvent):
            if event.type() == QEvent.Type.KeyRelease:
                if event.key() in [Qt.Key.Key_Up, Qt.Key.Key_Down]:
                    self.cell_selected()
            elif event.type() == QEvent.Type.KeyPress:
                if event == QKeySequence.StandardKey.Copy:
                    self.copy_selection()
                    return True
        return False

    def cell_selected(self):
        """Set new active image when current cell changed."""
        if self.linked_image_list:
            marked_imgs = self.main.get_marked_imgs_current_test()
            self.main.set_active_img(marked_imgs[self.currentRow()])

    def copy_selection(self):
        """Find which rows and columns to copy."""
        if self.selectedIndexes():
            rows = [index.row() for index in self.selectedIndexes()]
            cols = [index.column() for index in self.selectedIndexes()]
            self.parent.copy_table(row_range=[min(rows), max(rows)],
                                   col_range=[min(cols), max(cols)])

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
        lock_format = self.parent.tb_copy.tool_decimal_all.isChecked()

        def mtf_multiply_10(row):
            """Multiply MTF values by 10 to cy/cm (cy/mm default), accept None."""
            new_row = []
            try:
                new_row = list(10 * np.array(row))
            except TypeError:
                for val in row:
                    if val is not None:
                        new_row.append(10 * val)
                    else:
                        new_row.append(None)
            return new_row

        values_rows_copy = copy.deepcopy(values_rows)
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
            except (KeyError, TypeError):
                pass
        else:
            if linked_image_list:
                marked_imgs = self.main.get_marked_imgs_current_test()
                values_rows_copy = [
                    row for i, row in enumerate(values_rows_copy) if i in marked_imgs]
            if self.main.current_test == 'MTF' and self.main.current_modality == 'CT':
                if self.main.current_paramset.mtf_cy_pr_mm is False:
                    # factor 10 to get /cm instead of /mm
                    for i in range(len(values_rows_copy)):
                        values_rows_copy[i] = mtf_multiply_10(values_rows_copy[i])
        if len(row_labels) != 0:
            n_cols = len(values_cols)
            n_rows = len(row_labels)
        else:
            n_cols = len(col_labels)
            n_rows = len(values_rows_copy)
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
                if len(values_rows_copy[r]) == 0:
                    values_rows_copy[r] = [None] * n_cols
            values_cols = []
            for c in range(n_cols):
                values_cols.append([row[c] for row in values_rows_copy])
        if len(values_cols[0]) > 0:
            for c in range(len(values_cols)):
                if vendor:
                    this_col = values_cols[c]  # formatted above
                else:
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
        super().__init__(main, plotcanvas, include_min_max_button=True)

        toolb = QToolBar()
        self.tool_resultsize = ToolMaximizeResults(main)
        toolb.addWidget(self.tool_resultsize)
        toolb.setOrientation(Qt.Orientation.Vertical)
        self.hlo.addWidget(toolb)

    def min_max_curves(self):
        """Set ranges to min-max of x and y according to content."""
        self.plotcanvas.plot(ranges_min_max=True)


class ResultPlotCanvas(PlotCanvas):
    """Canvas for display of results as plot."""

    def __init__(self, main):
        super().__init__(main)
        self.color_k = 'k'
        self.color_gray = 'gray'
        self.color_darkgray = 'darkgray'
        if 'dark' in os.environ[ENV_ICON_PATH]:
            self.color_k = 'w'
            self.color_gray = 'whitesmoke'
            self.color_darkgray = 'lightgray'

    def plot(self, selected_text='', ranges_min_max=False):
        """Refresh plot.

        If selected_text is set = generate report specific selection.
        """
        if not isinstance(selected_text, str):
            selected_text = ''
        self.ax.clear()
        self.title = ''
        self.xtitle = 'x'
        self.ytitle = 'y'
        self.default_range_x = [None, None]
        self.default_range_y = [None, None]
        self.legend_location = 'best'
        self.bars = []
        self.curves = []
        self.scatters = []
        if self.main.current_test == 'vendor':
            try:
                _ = self.main.results['vendor']['details']
                self.vendor(selected_text)
            except KeyError:
                pass
        else:
            self.zpos_all = [img.zpos for img in self.main.imgs]
            self.marked_this = self.main.get_marked_imgs_current_test()
            if self.main.gui.active_img_no in self.marked_this:
                if self.main.current_test in self.main.results:
                    if self.main.results[self.main.current_test] is not None:
                        class_method = getattr(self, self.main.current_test, None)
                        if class_method is not None:
                            try:
                                class_method(selected_text)
                            except (KeyError, TypeError):
                                pass

        if len(self.curves) > 0:
            x_only_int = True
            self.ax.set_xscale('linear')  # reset if previous other than lin
            self.ax.set_yscale('linear')
            for curve in self.curves:
                if 'markersize' not in curve:
                    curve['markersize'] = 6.
                if 'style' not in curve:
                    curve['style'] = '-'
                if 'alpha' not in curve:  # TODO currently have no effect?
                    curve['alpha'] = 1.
                if 'fillstyle' not in curve:
                    curve['fillstyle'] = 'full'
                proceed = True
                try:
                    if curve['xvals'].size != curve['yvals'].size:
                        proceed = False
                except AttributeError:
                    try:
                        if len(curve['xvals']) != len(curve['yvals']):
                            proceed = False
                    except TypeError:
                        pass
                if proceed:
                    if 'color' in curve:
                        self.ax.plot(curve['xvals'], curve['yvals'],
                                     curve['style'], label=curve['label'],
                                     markersize=curve['markersize'],
                                     alpha=curve['alpha'],
                                     color=curve['color'],
                                     fillstyle=curve['fillstyle'])
                    else:
                        try:
                            self.ax.plot(curve['xvals'], curve['yvals'],
                                         curve['style'], label=curve['label'],
                                         markersize=curve['markersize'],
                                         alpha=curve['alpha'],
                                         fillstyle=curve['fillstyle'])
                        except ValueError:
                            self.ax.plot(curve['xvals'], curve['yvals'],
                                         color=curve['style'][1:-1],
                                         marker=curve['style'][-1],
                                         linestyle=curve['style'][0],
                                         label=curve['label'],
                                         markersize=curve['markersize'],
                                         alpha=curve['alpha'],
                                         fillstyle=curve['fillstyle'])
                    if 'xscale' in curve:
                        self.ax.set_xscale(curve['xscale'])
                    if 'yscale' in curve:
                        self.ax.set_yscale(curve['yscale'])

                if x_only_int:
                    xx = list(curve['xvals'])
                    if not isinstance(xx[0], int):
                        x_only_int = False

            if x_only_int:
                if 'xticks' in self.curves[0]:
                    self.ax.set_xticks(self.curves[0]['xvals'])
                    self.ax.set_xticklabels(self.curves[0]['xticks'], rotation=60,
                                            ha='right', rotation_mode='anchor')
                else:
                    self.ax.xaxis.set_major_locator(
                        matplotlib.ticker.MaxNLocator(integer=True))

            if len(self.curves) > 1:
                self.ax.legend(loc=self.legend_location)
            if ranges_min_max:
                self.default_range_x = [None, None]
                self.default_range_y = [None, None]
                for curve in self.curves:
                    min_x = np.min(curve['xvals'])
                    max_x = np.max(curve['xvals'])
                    min_y = np.min(curve['yvals'])
                    max_y = np.max(curve['yvals'])
                    if None in self.default_range_x:
                        self.default_range_x = [min_x, max_x]
                        self.default_range_y = [min_y, max_y]
                    else:
                        self.default_range_x[0] = min(
                            [min_x, self.default_range_x[0]])
                        self.default_range_x[1] = max(
                            [max_x, self.default_range_x[1]])
                        self.default_range_y[0] = min(
                            [min_y, self.default_range_y[0]])
                        self.default_range_y[1] = max(
                            [max_y, self.default_range_y[1]])

            if None not in self.default_range_x:
                try:
                    self.ax.set_xlim(self.default_range_x)
                except ValueError:
                    pass
            if None not in self.default_range_y:
                try:
                    self.ax.set_ylim(self.default_range_y)
                except ValueError:
                    pass
            self.ax.set_aspect('auto')
        elif len(self.bars) > 0:
            try:
                for bar in self.bars:
                    self.ax.bar(bar['names'], bar['values'])
                    self.ax.set_xticklabels(bar['names'], rotation=60,
                                            ha='right', rotation_mode='anchor')
                    self.ax.set_aspect('auto')
            except ValueError:
                pass
                # seen when in results, results and options change, #TODO better avoid
        elif len(self.scatters) > 0:
            first = self.scatters[0]
            if 'xlabels' in first:
                self.ax.set_xticks(np.arange(len(first['xlabels'])))
                self.ax.set_xticklabels(first['xlabels'], rotation=45)
                self.ax.set_yticks(np.arange(len(first['ylabels'])))
                self.ax.set_yticklabels(first['ylabels'])
            self.ax.set_aspect(1)
            for scatter in self.scatters:
                if 'array' in scatter:
                    arr = scatter['array']
                    if 'label' in scatter:
                        label = scatter['label']
                    else:
                        label = ''
                    r = np.arange(arr.shape[1])
                    p = np.arange(arr.shape[0])
                    R,P = np.meshgrid(r,p)
                    if isinstance(scatter['color'], str):
                        self.ax.scatter(
                            R[arr == True], P[arr == True],
                            s=scatter['size'],
                            marker=scatter['marker'],
                            color=scatter['color'], label=label)
                    else:
                        self.ax.scatter(
                            R[arr == True], P[arr == True],
                            s=scatter['size'],
                            marker=scatter['marker'], label=label,
                            c=scatter['color'][arr == True], cmap=scatter['cmap'],
                            norm=scatter['norm'])
                elif 'xs' in scatter:
                    self.ax.yaxis.set_inverted(True)
                    self.ax.scatter(scatter['xs'], scatter['ys'],
                                    s = scatter['size'],
                                    marker=scatter['marker'],
                                    c=scatter['color'],
                                    cmap=scatter['cmap'],
                                    norm=scatter['norm'])

        else:
            self.ax.axis('off')

        if len(self.curves) + len(self.bars) + len(self.scatters) > 0:
            self.ax.set_xlabel(self.xtitle)
            self.ax.set_ylabel(self.ytitle)
            if len(self.title) > 0:
                self.ax.set_title(self.title)
                self.fig.subplots_adjust(0.15, 0.25, 0.95, 0.85)
            else:
                self.fig.subplots_adjust(0.15, 0.2, 0.95, .95)

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
                    if len(curve['yvals']) > 0:
                        if min(curve['yvals']) < yrange[0]:
                            yrange[0] = None
                else:
                    try:
                        if min(get_yvals_to_eval(curve)) < yrange[0]:
                            yrange[0] = None
                    except ValueError:
                        pass
            if yrange[1] is not None:
                if limit_xrange is None:
                    if len(curve['yvals']) > 0:
                        if max(curve['yvals']) > yrange[1]:
                            yrange[1] = None
                else:
                    try:
                        if max(get_yvals_to_eval(curve)) > yrange[1]:
                            yrange[1] = None
                    except ValueError:
                        pass
            if not any(yrange):
                break
        return yrange

    def CDM(self, sel_text):
        """Prepare plot for test CDMAM."""
        imgno = self.main.gui.active_img_no
        details_dict = self.main.results['CDM']['details_dict'][imgno]
        cdmam_table_dict = self.main.results['CDM']['details_dict'][-1]
        xlabels = cdmam_table_dict['diameters']
        cbox = self.main.tab_mammo.cdm_cbox_thickness
        ylabels = [cbox.itemText(i) for i in range(cbox.count())]
        ylabels.reverse()

        include_array = None
        if 'include_array' in details_dict:
            if details_dict['include_array'] is not None:
                np.flipud(details_dict['include_array'])
        if include_array is None:
            include_array = np.ones(
                cdmam_table_dict['detection_matrix'].shape, dtype=bool)

        def prepare_found_corner_center_plot():
            self.title = 'Correctly found disc in corner (square) and at center (circle)'
            self.xtitle = 'Diameter (mm)'
            self.ytitle = r'Thickness ($\mu$m)'

            if 'include_array' in details_dict:
                if details_dict['include_array'] is not None:
                    self.scatters.append(
                        {'label': 'included',
                         'xlabels': xlabels, 'ylabels': ylabels,
                         'array': np.flipud(details_dict['include_array']),
                         'size': 30, 'marker': 'x', 'color': 'silver'})

            self.scatters.append(
                {'label': 'found correct corner',
                 'xlabels': xlabels, 'ylabels': ylabels,
                 'array': np.flipud(details_dict['found_correct_corner']),
                 'size': 100, 'marker': 's', 'color': 'green'})

            self.scatters.append(
                {'label': 'found center',
                 'xlabels': xlabels, 'ylabels': ylabels,
                 'array': np.flipud(details_dict['found_centers']),
                 'size': 30, 'marker': 'o', 'color': 'lightgreen'})

        def prepare_found_corrected_plot():
            if details_dict['corrected_neighbours'] is not None:
                
                if self.main.current_paramset.cdm_center_disc_option == 1:
                    self.title = ('Found discs in corner/center, corrected for '
                                  'nearest neighbours (blue background)')
                    uncorrected = (
                        details_dict['found_centers']
                        * details_dict['found_correct_corner'])
                else:
                    self.title = ('Found discs in corner, corrected for '
                                  'nearest neighbours (blue background)')
                    uncorrected = details_dict['found_correct_corner']

                any_corr = (
                    uncorrected.astype(int)
                    - details_dict['corrected_neighbours'].astype(int))
                any_corr = any_corr.astype(bool)

                self.xtitle = 'Diameter (mm)'
                self.ytitle = r'Thickness ($\mu$m)'

                if 'include_array' in details_dict:
                    if details_dict['include_array'] is not None:
                        self.scatters.append(
                            {'label': 'included',
                             'xlabels': xlabels, 'ylabels': ylabels,
                             'array': np.flipud(details_dict['include_array']),
                             'size': 30, 'marker': 'x', 'color': 'silver'})

                self.scatters.append(
                    {'label': 'found center',
                     'xlabels': xlabels, 'ylabels': ylabels,
                     'array': np.flipud(any_corr),
                     'size': 100, 'marker': 's', 'color': 'lightblue'})
                self.scatters.append(
                    {'label': 'found',
                     'xlabels': xlabels, 'ylabels': ylabels,
                     'array': np.flipud(details_dict['corrected_neighbours']),
                     'size': 30, 'marker': 'o', 'color': 'green'})

        def prepare_detection_matrix_plot(sel_text):
            self.title = 'Detection ratio based on all images'
            if 'corner' in sel_text:
                self.title = self.title + ' - corner discs'
                suff = '_corners'
            elif 'center' in sel_text:
                self.title = self.title +  ' - center discs'
                suff = '_centers'
            elif 'corrected' in sel_text:
                self.title = self.title + ' - corrected'
                if self.main.current_paramset.cdm_sigma > 0:
                    self.title = self.title + '/smoothed'
                suff = '_corrected'
            else:
                suff = ''

            self.xtitle = 'Diameter (mm)'
            self.ytitle = r'Thickness ($\mu$m)'

            self.scatters.append(
                {'xlabels': xlabels, 'ylabels': ylabels,
                 'array': np.flipud(include_array),
                 'size': 100, 'marker': 'o',
                 'color': np.flipud(cdmam_table_dict[
                     f'detection_matrix{suff}']),
                 'cmap': 'nipy_spectral_r',
                 'norm': matplotlib.colors.Normalize(vmin=0., vmax=2.)
                 })

        def prepare_comparison_vs_fraction_xls(sel_text):
            self.title = sel_text
            self.xtitle = 'Diameter (mm)'
            self.ytitle = r'Thickness ($\mu$m)'
            if 'diff' in sel_text:
                self.scatters.append(
                    {'xlabels': xlabels, 'ylabels': ylabels,
                     'array': np.flipud(include_array),
                     'size': 100, 'marker': 'o',
                     'color': np.flipud(cdmam_table_dict['diff_Fraction.xls']),
                     'cmap': 'coolwarm', #'vmin': -1., 'vmax': 1.
                     'norm': matplotlib.colors.Normalize(vmin=-1., vmax=1.)
                     })
            else:
                self.scatters.append(
                    {'xlabels': xlabels, 'ylabels': ylabels,
                     'array': np.flipud(include_array),
                     'size': 100, 'marker': 'o',
                     'color': np.flipud(cdmam_table_dict['Fraction.xls']),
                     'cmap': 'nipy_spectral_r',
                     'norm': matplotlib.colors.Normalize(vmin=0., vmax=2.)
                     })

        def prepare_psychometric_plot():
            self.title = 'Fitted psychometric curves'
            self.xtitle = r'Thickness ($\mu$m)'
            self.ytitle = 'Detection probability'
            psyc_res = cdmam_table_dict['psychometric_results']
            idx_diam = self.main.tab_mammo.cdm_cbox_diameter.currentIndex()
            for i, diam in enumerate(cdmam_table_dict['diameters']):
                c = COLORS[i % len(COLORS)]
                xvals = psyc_res['xs'][i]
                yvals = psyc_res['ys'][i]
                yfit = psyc_res['yfits'][i]
                if yfit is not None:
                    curve = {'label': f'Ø {diam}',
                             'xvals': xvals,
                             'yvals': yfit,
                             'style': ':', 'fillstyle': 'none', 'color': c}
                    if i == idx_diam:
                        curve['style'] = '-'
                    self.curves.append(curve)
                curve = {'label': '_no_legend_',
                         'xvals': xvals, 'xscale' : 'log',
                         'yvals': yvals,
                         'style': 'o', 'fillstyle': 'none', 'color': c}
                if i == idx_diam:
                    curve['fillstyle'] = 'full'
                self.curves.append(curve)
            self.curves.append({'label': '_no_legend_',
                      'xvals': [min(cdmam_table_dict['thickness']),
                                max(cdmam_table_dict['thickness'])],
                      'yvals': [0.65, 0.65],
                      'style': '-k' })
            self.default_range_y = [0.0, 1.1]

        def prepare_threshold_plot():
            self.title = 'Threshold thickness'
            self.xtitle = 'Diameter (mm)'
            self.ytitle = r'Thickness ($\mu$m)'

            psyc_res = cdmam_table_dict['psychometric_results']
            limits = cdmam_table_dict['EUREF_performance_limits']
            curve = {'label': 'fit to data',
                     'xvals': psyc_res['thickness_predicts_fit_d'],
                     'yvals': psyc_res['thickness_predicts_fit'],
                     'xscale' : 'log', 'yscale': 'log',
                     'style': ':b'}
            self.curves.append(curve)
            curve = {'label': 'predicted',
                     'xvals': psyc_res['thickness_predicts_fit_d'],
                     'yvals': psyc_res['thickness_predicts'],
                     'style': 'o', 'fillstyle': 'none', 'markersize': 3.}
            self.curves.append(curve)
            curve = {'label': 'acceptable',
                     'xvals': limits['diameters'],
                     'yvals': limits['acceptable_thresholds_thickness'],
                     'style': 'r'}
            self.curves.append(curve)
            curve = {'label': 'achievalble',
                     'xvals': limits['diameters'],
                     'yvals': limits['achievable_thresholds_thickness'],
                     'style': 'k'}
            self.curves.append(curve)
            self.default_range_x = [0.05, 2]
            self.default_range_y = [0.01, 5]

        def prepare_comparison_vs_inp_plot(sel_text):
            self.title = sel_text + '(imageQC found = green, inp found = orange)'
            self.xtitle = 'Diameter (mm)'
            self.ytitle = r'Thickness ($\mu$m)'

            if 'corner' in sel_text:
                self.scatters.append(
                    {'label': 'imageQC',
                     'xlabels': xlabels, 'ylabels': ylabels,
                     'array': np.flipud(details_dict['found_correct_corner']),
                     'size': 100, 'marker': 's', 'color': 'green'})
                if 'found_correct_corner_inp' in details_dict:
                    self.scatters.append(
                        {'label': 'CDCOM',
                         'xlabels': xlabels, 'ylabels': ylabels,
                         'array': np.flipud(details_dict['found_correct_corner_inp']),
                         'size': 50, 'marker': 's', 'color': 'orange'})
            if 'center' in sel_text:
                self.scatters.append(
                    {'label': 'imageQC',
                     'xlabels': xlabels, 'ylabels': ylabels,
                     'array': np.flipud(details_dict['found_centers']),
                     'size': 60, 'marker': 'o', 'color': 'green'})
                if 'found_centers_inp' in details_dict:
                    self.scatters.append(
                        {'label': 'CDCOM',
                         'xlabels': xlabels, 'ylabels': ylabels,
                         'array': np.flipud(details_dict['found_centers_inp']),
                         'size': 30, 'marker': 'o', 'color': 'orange'})

        if sel_text == '':
            test_widget = self.main.stack_test_tabs.currentWidget()
            try:
                sel_text = test_widget.cdm_plot.currentText()
            except AttributeError:
                sel_text = ''
        if sel_text == 'Found disc at center and in correct corner':
            prepare_found_corner_center_plot()
        elif sel_text == 'Found discs corrected for nearest neighbours':
            prepare_found_corrected_plot()
        elif 'CDCOM' in sel_text:
            prepare_comparison_vs_inp_plot(sel_text)
        elif 'Fraction.xls' in sel_text:
            prepare_comparison_vs_fraction_xls(sel_text)
        elif 'Detection matrix' in sel_text:
            prepare_detection_matrix_plot(sel_text)
        elif sel_text == 'Fitted psychometric curves':
            prepare_psychometric_plot()
        elif sel_text == 'Threshold thickness':
            prepare_threshold_plot()
        

    def Cro(self, sel_text):
        """Prepare plot for test PET cross calibration."""
        self.title = 'z profile'
        self.xtitle = 'Slice position (mm)'
        self.ytitle = 'Average in ROI (Bq/ml)'
        details_dict = self.main.results['Cro']['details_dict']
        self.curves.append(
            {'label': 'all slices', 'xvals': details_dict['zpos'],
             'yvals': details_dict['roi_averages'], 'style': '-' + self.color_k})
        self.curves.append(
            {'label': 'used slices', 'xvals': details_dict['used_zpos'],
             'yvals': details_dict['used_roi_averages'], 'style': '-r'})

    def CTn(self, sel_text):
        """Prepare plot for test CTn."""

        def prepare_plot_HU_min_max(percent=False):
            unit = '(%)' if percent else '(HU)'
            self.ytitle = f'Difference from set HU min/max {unit}'
            imgno = self.main.gui.active_img_no
            meas_vals = self.main.results['CTn']['values'][imgno]
            proceed = True
            try:
                diff_max = np.subtract(self.main.current_paramset.ctn_table.max_HU,
                                       meas_vals)
                diff_min = np.subtract(self.main.current_paramset.ctn_table.min_HU,
                                       meas_vals)
                if percent:
                    diff_max = 100. * np.divide(diff_max, meas_vals)
                    diff_min = 100. * np.divide(diff_min, meas_vals)
            except TypeError:
                self.main.status_bar.showMessage(
                    'Some set HU min or max is not valid', 2000, warning=True)
                proceed = False

            if proceed:
                self.curves.append({
                    'label': '_nolegend_',
                    'xvals': [np.min(meas_vals), np.max(meas_vals)],
                    'yvals': [0, 0],
                    'style': ':',
                    'color': 'blue'}
                    )
                self.curves.append(
                    {'label': 'set max - measured', 'xvals': meas_vals,
                     'yvals': diff_max, 'style': 'bv'})
                self.curves.append(
                    {'label': 'set min - measured', 'xvals': meas_vals,
                     'yvals': diff_min, 'style': 'b^'})
                if np.max(diff_min) > 0:
                    idxs = np.where(diff_min > 0)
                    xvals = [meas_vals[i] for i in idxs[0]]
                    yvals = [diff_min[i] for i in idxs[0]]
                    self.curves.append({
                        'label': '_nolegend_', 'xvals': xvals, 'yvals': yvals,
                        'style': 'r^'
                        })
                if np.min(diff_max) < 0:
                    idxs = np.where(diff_max < 0)
                    xvals = [meas_vals[i] for i in idxs[0]]
                    yvals = [diff_max[i] for i in idxs[0]]
                    self.curves.append({
                        'label': '_nolegend_', 'xvals': xvals, 'yvals': yvals,
                        'style': 'rv'
                        })
                self.xtitle = 'HU value'

        def prepare_plot_linear():
            self.ytitle = self.main.current_paramset.ctn_table.linearity_unit
            yvals = self.main.current_paramset.ctn_table.linearity_axis
            imgno = self.main.gui.active_img_no
            xvals = self.main.results['CTn']['values'][imgno]
            self.curves.append(
                {'label': 'HU mean', 'xvals': xvals,
                 'yvals': yvals, 'style': '-bo'})
            if self.main.results['CTn']['values_sup'][imgno][0] is not None:
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

        if sel_text == '':
            test_widget = self.main.stack_test_tabs.currentWidget()
            try:
                sel_text = test_widget.ctn_plot.currentText()
            except AttributeError:
                sel_text = ''
        if 'HU' in sel_text:
            # percent = True if '%' in sel_text else False
            prepare_plot_HU_min_max()  # percent=percent)
        else:
            prepare_plot_linear()
        self.title = sel_text

    def DCM(self, sel_text):
        """Prepare plot for test DCM."""
        widget = self.main.stack_test_tabs.currentWidget()
        param_no = widget.wid_dcm_pattern.current_select
        xvals = []
        yvals = []
        for i, row in enumerate(self.main.results['DCM']['values']):
            if len(row) > 0:
                xvals.append(i)
                yvals.append(row[param_no])
        proceed = True
        for yval in yvals:
            if isinstance(yval, str):
                proceed = False
                break
        if proceed:
            curve = {'label':
                     self.main.results['DCM']['headers'][param_no],
                     'xvals': xvals,
                     'yvals': yvals,
                     'style': '-bo'}
            self.curves.append(curve)
            self.xtitle = 'Image index'
            self.ytitle = self.main.results['DCM']['headers'][param_no]
            self.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
            self.ax.set_xticks(xvals)

    def Foc(self, sel_text):
        self.title = 'Variance map radial profiles processed and inverted'
        self.xtitle = 'distance to center (mm)'
        self.ytitle = '50.percentile - variance < 50.percentile'
        imgno = self.main.gui.active_img_no
        details_dict = self.main.results['Foc']['details_dict'][imgno]
        styles = ['-r', '-b']
        labels = ['x', 'y']
        profs = details_dict['profiles']
        xprofs = details_dict['profiles_dists']
        for i in [0, 1]:
            self.curves.append(
                {'label': labels[i], 'xvals': xprofs[i],
                 'yvals': profs[i], 'style': styles[i]})
            self.curves.append({
                'label': '_nolegend_',
                'xvals': [
                    details_dict['blur_diameter_xy'][i] / 2,
                    details_dict['blur_diameter_xy'][i] / 2],
                'yvals': [0, np.max(profs[i])],
                'style': '-'+styles[i]})

    def Hom(self, sel_text):
        """Prepare plot for test Hom."""
        if sel_text == '':
            try:
                test_widget = self.main.stack_test_tabs.currentWidget()
                sel_text = test_widget.hom_plot.currentText()
            except AttributeError:
                pass
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
                      'style': '--' + self.color_k}
            tolmin = tolmax.copy()
            tolmin['label'] = 'tolerance min'
            tolmin['yvals'] = [-4, -4]
            self.curves.append(tolmin)
            self.curves.append(tolmax)
            self.default_range_y = self.test_values_outside_yrange([-6, 6])
        elif self.main.current_modality == 'Xray':
            if self.main.current_paramset.hom_type == 4:  # flatfield aapm
                self.title = ('Profile average minus '
                              'average of neighbours in same profile')
                imgno = self.main.gui.active_img_no
                details_dict = self.main.results['Hom']['details_dict'][imgno]
                styles = ['-r', '-b']
                labels = ['Row', 'Column']
                profs = details_dict['diff_neighbours_profile_col_row']
                for i, prof in enumerate(profs):
                    self.curves.append(
                        {'label': labels[i], 'xvals': np.arange(prof.size),
                         'yvals': prof, 'style': styles[i]})
        elif self.main.current_modality == 'PET':
            if self.main.current_paramset.hom_type == 0:
                self.title = '% difference from mean of all means'
                yvals = [[] for j in range(5)]
                for i, row in enumerate(self.main.results['Hom']['values']):
                    if len(row) > 0:
                        img_nos.append(i)
                        xvals.append(self.zpos_all[i])
                        for j in range(5):
                            yvals[j].append(row[5+j])
                labels = ['Center', 'at 12', 'at 15', 'at 18', 'at 21']
                styles = ['-r', '-b', '-g', '-y', '-c']
                for j in range(5):
                    self.curves.append(
                        {'label': labels[j], 'xvals': xvals,
                         'yvals': yvals[j], 'style': styles[j]})
                self.xtitle = 'zpos (mm)'
                if None in xvals:
                    xvals = img_nos
                    self.xtitle = 'Image number'
                self.ytitle = '% difference'
                try:
                    tolmax = {'label': 'tolerance max',
                              'xvals': [min(xvals), max(xvals)],
                              'yvals': [5, 5],
                              'style': '--' + self.color_k}
                    tolmin = tolmax.copy()
                    tolmin['label'] = 'tolerance min'
                    tolmin['yvals'] = [-5, -5]
                    self.curves.append(tolmin)
                    self.curves.append(tolmax)
                except ValueError:
                    pass
                self.default_range_y = self.test_values_outside_yrange([-7, 7])
            else:
                details_dict = self.main.results['Hom']['details_dict']
                self.title = sel_text
                self.xtitle = 'zpos (mm)'
                xvals = details_dict['zpos_used']
                if sel_text == 'z-profile central ROI, slice selection':
                    self.curves.append(
                        {'label': 'All slices', 'xvals': details_dict['zpos'],
                         'yvals': details_dict['roi_averages'],
                         'style': '-' + self.color_k})
                    self.curves.append(
                        {'label': 'Used slices', 'xvals': xvals,
                         'yvals': details_dict['roi_averages_0'],
                         'style': '-r'})
                    self.ytitle = 'z-profile for Auto select slices'
                elif sel_text == 'Average in ROI pr slice':
                    labels = ['Center', 'at12', 'at15', 'at18', 'at21']
                    styles = ['-r', '-b', '-g', '-y', '-c']
                    for i in  range(5):
                        yvals = details_dict[f'roi_averages_{i}']
                        self.curves.append(
                            {'label': labels[i], 'xvals': xvals,
                             'yvals': yvals, 'style': styles[i]})
                    self.ytitle = 'Average pixel value'
                elif sel_text == 'Integral uniformity pr slice':
                    yvals = details_dict['uniformity_pr_slice']
                    self.ytitle = 'Integral uniformity (%)'
                    self.curves.append(
                        {'label': 'Integral uniformity', 'xvals': xvals,
                         'yvals': yvals, 'style': '-' + self.color_k})
                    tolmax = {'label': '_no_legend_',
                              'xvals': [min(xvals), max(xvals)],
                              'yvals': [5, 5],
                              'style': '--r'}
                    tolmin = tolmax.copy()
                    tolmin['yvals'] = [-5, -5]
                    self.curves.append(tolmin)
                    self.curves.append(tolmax)
                    self.default_range_y = self.test_values_outside_yrange([-7, 7])
                elif sel_text == 'Integral uniformity pr ROI':
                    yvals = details_dict['uniformity_pr_roi']
                    self.ytitle = 'Integral uniformity (%)'
                    self.xtitle = 'ROI'
                    self.curves.append(
                        {'label': 'Integral uniformity',
                         'xvals': list(range(5)),
                         'xticks': ['Center', 'at12', 'at15', 'at18', 'at21'],
                         'yvals': yvals, 'style': 'o'})
                    tolmax = {'label': '_no_legend_',
                              'xvals': [0, 4],
                              'yvals': [5, 5],
                              'style': '--r'}
                    tolmin = tolmax.copy()
                    tolmin['yvals'] = [-5, -5]
                    self.curves.append(tolmin)
                    self.curves.append(tolmax)
                    self.default_range_y = self.test_values_outside_yrange([-7, 7])
                elif sel_text == 'Non-uniformity pr slice':
                    self.curves.append(
                        {'label': 'Non-uniformity', 'xvals': xvals,
                         'yvals': details_dict['uniformity_pr_slice']})
                    self.ytitle = 'Non-uniformity %'
                elif 'Coefficient of Variation' in sel_text:
                    self.curves.append(
                        {'label': 'CoV', 'xvals': xvals,
                         'yvals': details_dict['cov_pr_slice']})
                    self.ytitle = 'CoV %'
                elif sel_text == 'Averages pr ROI for selected slice':
                    imgno = self.main.gui.active_img_no
                    if imgno in details_dict['image_indexes_used']:
                        sz = details_dict['roi_shape']
                        self.title =\
                            f'Averages pr ROI ({sz}x{sz}) for selected slice'
                        self.xtitle = 'pos x (pix)'
                        self.ytitle = 'pos y (pix)'
                        idx = np.where(
                            details_dict['image_indexes_used'] == imgno)[0][0]
                        values_this = details_dict['roi_averages_square'][idx]
                        self.scatters.append(
                            {'xs': self.main.current_roi[1],
                             'ys': self.main.current_roi[2],
                             'size': 100, 'marker': 'o',
                             'color': values_this,
                             'cmap': 'coolwarm',
                             'norm': matplotlib.colors.Normalize(
                                 vmin=0.95 * np.mean(values_this),
                                 vmax=1.05 * np.mean(values_this))
                             })


    def HUw(self, sel_text):
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
                  'style': '--' + self.color_k}
        tolmin = tolmax.copy()
        tolmin['yvals'] = [-4, -4]
        tolmin['label'] = 'tolerance min'
        self.curves.append(tolmin)
        self.curves.append(tolmax)
        self.default_range_y = self.test_values_outside_yrange([-6, 6])

    def MTF(self, sel_text):
        """Prepare plot for test MTF."""
        imgno = self.main.gui.active_img_no
        rowno = imgno
        if (self.main.results['MTF']['pr_image']
            or self.main.results['MTF']['pr_image_sup']):
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

                colors = [self.color_k, 'r']  # gaussian black, discrete red
                linestyles = ['-', '--', ':']
                infotext = ['gaussian', 'discrete']
                prefix = ['g', 'd']
                suffix = [' x', ' y', ' z'] if len(details_dicts) >= 2 else ['']
                if self.main.current_modality in ['CT', 'SPECT', 'PET']:
                    if self.main.current_paramset.mtf_type == 3:
                        suffix = [' line 1', ' line 2']
                for ddno in range(3):
                    try:
                        dd = details_dicts[ddno]
                        if f'{prefix[0]}MTF_details' in dd:
                            proceed = True
                        else:
                            proceed = False
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
                            'color': self.color_gray
                             })

                self.default_range_y = self.test_values_outside_yrange(
                    [0, 1.3], limit_xrange=[0, nyquist_freq])
                self.default_range_x = [0, 1.1 * nyquist_freq]
                self.curves.append({
                    'label': '_nolegend_',
                    'xvals': [nyquist_freq, nyquist_freq],
                    'yvals': [0, 1.3],
                    'style': ':' + self.color_k
                     })
                self.ax.text(0.9*nyquist_freq, 0.5, 'Nyquist frequency',
                             ha='left', size=8, color=self.color_gray)

                # MTF %
                if self.main.current_modality != 'NM':
                    values = self.main.results[self.main.current_test]['values'][rowno]
                    if self.main.current_modality in ['Xray', 'Mammo']:
                        yvals = [[.5, .5]]
                        xvals = [[0, values[-1]]]
                        yvals.extend([[0, values[i]] for i in range(5)])
                        if self.main.current_modality in 'Xray':
                            xvals.extend([
                                [.5, .5], [1, 1], [1.5, 1.5], [2, 2], [2.5, 2.5]])
                        else:
                            xvals.extend([
                                [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
                        for i in range(len(xvals)):
                            self.curves.append({
                                'label': '_nolegend_',
                                'xvals': xvals[i],
                                'yvals': yvals[i],
                                'style': ':' + self.color_k
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
                                    'style': ':' + self.color_k
                                     })

        def prepare_plot_LSF():
            self.xtitle = 'pos (mm)'
            self.ytitle = 'LSF'
            self.legend_location = 'upper left'

            linestyles = ['-', '--', ':']
            suffix = [' x', ' y', ' z'] if len(details_dicts) >= 2 else ['']
            if self.main.current_modality in ['CT', 'SPECT', 'PET']:
                if self.main.current_paramset.mtf_type == 3:
                    suffix = [' line 1', ' line 2']
            lbl_prefilter = ''
            prefilter = False
            if 'sigma_prefilter' in details_dicts[0]:
                if details_dicts[0]['sigma_prefilter'] > 0:
                    prefilter = True
                    lbl_prefilter = ' presmoothed'

            for ddno in range(3):
                try:
                    dd = details_dicts[ddno]
                    xvals = dd['LSF_x']  # TODO fix when this is error, expecting index
                    proceed = True
                except (IndexError, KeyError):
                    proceed = False
                except TypeError:
                    proceed = False  # TODO fix when this is error
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
                    curve_corrected = None
                    if prefilter:
                        xvals = dd_this['LSF_fit_x']
                        yvals = dd_this['LSF_prefit']
                        self.curves.append({
                            'label': f'LSF{lbl_prefilter}' + suffix[ddno],
                            'xvals': xvals,
                            'yvals': yvals,
                            'style': linestyles[ddno] + 'b'
                             })
                        if 'LSF_corrected' in dd_this:
                            yvals = dd_this['LSF_corrected']
                            if yvals is not None:
                                curve_corrected = {
                                    'label': (
                                        'LSF corrected - gaussian fit ' + suffix[ddno]),
                                    'xvals': xvals,
                                    'yvals': yvals,
                                    'style': '--',
                                    'color': 'green'
                                     }

                    xvals = dd_this['LSF_fit_x']
                    yvals = dd_this['LSF_fit']
                    if self.main.current_paramset.mtf_type in [0, 5, 6]:
                        # 2d or 3d point
                        # +/- 5x sigma devided on 100 x values params
                        try:
                            sigma = dd_this['LSF_fit_params'][1]
                            A = dd_this['LSF_fit_params'][0]
                            xvals = (5 * sigma)/100 * np.arange(100)
                            xvals = np.append(-xvals[::-1], xvals[1:])
                            yvals = A * np.exp(
                                -0.5 * (xvals ** 2) / (sigma ** 2))
                            if len(dd_this['LSF_fit_params']) == 4:
                                sigma2 = dd_this['LSF_fit_params'][3]
                                A2 = dd_this['LSF_fit_params'][2]
                                yvals = yvals + A2 * np.exp(
                                    -0.5 * (xvals ** 2) / (sigma2 ** 2))
                        except (KeyError, IndexError):
                            pass

                    self.curves.append({
                        'label': f'LSF{lbl_prefilter} - gaussian fit' + suffix[ddno],
                        'xvals': xvals,
                        'yvals': yvals,
                        'style': linestyles[ddno] + self.color_k
                         })
                    if curve_corrected:
                        self.curves.append(curve_corrected)

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
                                        'style': ':' + self.color_k
                                        })
                                    self.ax.text(
                                        x * cw, np.mean(minmax), 'cut',
                                        ha='left', size=8, color=self.color_gray)
                                if 'cut_width_fade' in dd_this:
                                    cwf = dd_this['cut_width_fade']
                                    for x in [-1, 1]:
                                        self.curves.append({
                                            'label': '_nolegend_',
                                            'xvals': [x * cwf] * 2,
                                            'yvals': minmax,
                                            'style': ':' + self.color_k
                                            })
                                        self.ax.text(
                                            x * cwf, np.mean(minmax), 'fade',
                                            ha='left', size=8, color=self.color_gray)
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
                    if self.main.current_modality == 'NM':  # TODO? make fit others
                        for i in range(len(edge_details_dicts[0])):
                            if 'sorted_pixels' in edge_details_dicts[0][i]:
                                try:
                                    xvals.append(
                                        edge_details_dicts[0][i]['sorted_pixels_x'])
                                    sorted_pixels.append(
                                        edge_details_dicts[0][i]['sorted_pixels'])
                                    proceed = True
                                except KeyError:
                                    proceed = False
                    else:
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
                line_labels = False
                lsf_is_interp = False
                try:
                    mtf_type = self.main.current_paramset.mtf_type
                    if self.main.current_modality in ['CT', 'SPECT', 'PET']:
                        if mtf_type >= 1:
                            self.ytitle = 'Summed pixel values'
                            xy_labels = True
                            lsf_is_interp = True
                            if mtf_type == 2 and self.main.current_modality == 'CT':
                                self.ytitle = 'Pixel value'
                                xy_labels = False
                            elif mtf_type == 3:
                                self.ytitle = 'Summed pixel values from front'
                                line_labels = True
                                xy_labels = False
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
                elif line_labels:
                    suffix = [' line1', ' line2']
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
                    proceed = True
                    try:
                        if self.main.current_paramset.mtf_type == 4:
                            proceed = False
                        elif self.main.current_paramset.mtf_type == 2:
                            if self.main.current_modality == 'CT':
                                proceed = False
                    except AttributeError:
                        pass
                    if proceed:
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
                if 'profile_xyz' not in details_dicts[-1]:
                    proceed = False
            elif self.main.current_modality in ['CT', 'SPECT', 'PET']:
                if self.main.current_paramset.mtf_type in [1, 4]:
                    proceed = False
            elif self.main.current_modality == 'NM':
                if self.main.current_paramset.mtf_type > 0:
                    proceed = False

            if proceed:
                self.xtitle = 'pos (mm)'
                self.ytitle = 'Pixel value - background mean'

                if 'profile_xyz' in details_dicts[-1]:
                    linestyles = ['-', '--', ':']
                    legend = ['x', 'y', 'z']
                    profiles = details_dicts[-1]['profile_xyz']
                    xvals = details_dicts[-1]['profile_xyz_dist']
                    for i in range(3):
                        self.curves.append({
                            'label': legend[i],
                            'xvals': xvals[i],
                            'yvals': profiles[i],
                            'style': linestyles[i] + 'k'
                             })
                else:
                    linestyles = ['-', '--']  # x, y
                    colors = ['g', 'b', 'r', self.color_k, 'c', 'm']
                    if len(details_dicts) == 2:
                        center_xy = [
                            details_dicts[i]['center'] for i in range(2)]
                        submatrix = [details_dicts[0]['matrix']]
                    else:
                        center_xy = details_dicts[0]['center_xy']
                        submatrix = details_dicts[0]['matrix']

                    marked_imgs = self.main.get_marked_imgs_current_test()
                    pix = self.main.imgs[marked_imgs[0]].pix[0]

                    for no, sli in enumerate(submatrix):
                        proceed = True
                        if self.main.results['MTF']['pr_image'] is False:
                            if no not in marked_imgs:
                                proceed = False
                        if proceed and sli is not None:
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
                        'color': self.color_darkgray,
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

        def prepare_plot_zprofile(sel_text):
            common_details = self.main.results['MTF']['details_dict'][-1]
            proceed = True if 'zpos_used' in common_details else False
            if 'average' in sel_text:
                if self.main.current_paramset.mtf_type != 4:
                    proceed = False
            elif 'ax ' in sel_text:  # max / Max
                if self.main.current_paramset.mtf_type == 4:
                    proceed = False

            if proceed:
                self.xtitle = 'z position (mm)'
                if 'ax z-profile' in sel_text or 'average z-profile' in sel_text:
                    word = 'Average' if 'average' in sel_text else 'Max'
                    self.ytitle = f'{word} pixel value in ROI'
                    self.curves.append({
                        'label': f'{word} in marked images',
                        'xvals': common_details['zpos_marked_images'],
                        'yvals': common_details['max_roi_marked_images'],
                        'style': '-b'})
                    if 'max_roi_used' in common_details:
                        self.curves.append({
                            'label': f'{word} in used images',
                            'xvals': common_details['zpos_used'],
                            'yvals': common_details['max_roi_used'],
                            'style': '-r'})
                    else:
                        self.curves[-1]['-style'] = '-r'  # all used
                    if 'max_roi_used_2' in common_details:
                        self.curves.append({
                            'label': 'Max in used images 2',
                            'xvals': common_details['zpos_used_2'],
                            'yvals': common_details['max_roi_used_2'],
                            'style': '-c'})
                elif 'FWHM' in sel_text:
                    if self.main.current_paramset.mtf_type == 2:
                        self.ytitle = 'FWHM pr sliding window'
                        yvals = [row[0] for row in
                            self.main.results[self.main.current_test][
                                'values_sup']]
                        if any(yvals):
                            self.curves.append({
                                'label': 'FWHM x',
                                'xvals': common_details['zpos_marked_images'],
                                'yvals': yvals,
                                'style': '-b'})
                        yvals = [row[1] for row in
                            self.main.results[self.main.current_test][
                                'values_sup']]
                        if any(yvals):
                            self.curves.append({
                                'label': 'FWHM y',
                                'xvals': common_details['zpos_marked_images'],
                                'yvals': yvals,
                                'style': '-r'})
                elif 'offset' in sel_text:
                    if self.main.current_paramset.mtf_type == 2:
                        self.ytitle = 'Offset pr image (mm from image center)'
                        yvals = [
                            row[2] for row
                            in self.main.results[self.main.current_test]['values_sup']]
                        if any(yvals):
                            self.curves.append({
                                'label': 'x',
                                'xvals': common_details['zpos_marked_images'],
                                'yvals': yvals,
                                'style': '-b'})
                        yvals = [
                            row[3] for row
                            in self.main.results[self.main.current_test]['values_sup']]
                        if any(yvals):
                            self.curves.append({
                                'label': 'y',
                                'xvals': common_details['zpos_marked_images'],
                                'yvals': yvals,
                                'style': '-r'})

        def prepare_plot_xyz_NEMA(sel_text):
            common_details = self.main.results['MTF']['details_dict'][-1]
            if sel_text[0] == 'x':
                idx = 0
            elif sel_text[0] == 'y':
                idx = 1
            else:
                idx = 2

            self.xtitle = 'position (mm)'
            self.ytitle = 'Pixel value'
            self.title = sel_text
            xvals = common_details['profile_xyz_dist'][idx]
            yvals = common_details['profile_xyz'][idx]
            self.curves.append({
                'label': 'pixel values',
                'xvals': xvals, 'yvals': yvals, 'style': '-k.'})
            xvals = common_details['NEMA_modified_profiles'][idx][0]
            yvals = common_details['NEMA_modified_profiles'][idx][1]
            self.curves.append({
                'label': 'parabolic fit',
                'xvals': xvals[2:-2], 'yvals': yvals[2:-2], 'style': ':k'})
            self.curves.append({
                'label': '_no_legend_', 'style': '-r',
                'xvals': [xvals[1], xvals[-2]],
                'yvals': [yvals[1], yvals[-2]]})
            self.ax.text(xvals[-2], yvals[1], '  FWHM',
                         ha='left', size=8)
            self.curves.append({
                'label': '_no_legend_', 'style': '-b',
                'xvals': [xvals[0], xvals[-1]],
                'yvals': [yvals[0], yvals[-1]]})
            self.ax.text(xvals[-1], yvals[0], '  FWTM',
                         ha='left', size=8)

        if sel_text == '':
            test_widget = self.main.stack_test_tabs.currentWidget()
            try:
                sel_text = test_widget.mtf_plot.currentText()
            except AttributeError:
                sel_text = ''
        try:
            if sel_text == 'MTF':
                prepare_plot_MTF()
            elif sel_text == 'LSF':
                prepare_plot_LSF()
            elif sel_text == 'Sorted pixel values':
                prepare_plot_sorted_pix()
            elif sel_text in ['Centered xy profiles', 'Centered xyz profiles']:
                prepare_plot_centered_profiles()
            elif sel_text in ['Edge position', 'Line fit']:
                prepare_plot_edge_position()
            elif 'z-profile' in sel_text:
                prepare_plot_zprofile(sel_text)
            elif 'profile with FWHM, FWTM and fit max' in sel_text:
                prepare_plot_xyz_NEMA(sel_text)
            self.title = sel_text
        except (TypeError, IndexError) as err:
            try:
                if self.main.developer_mode:
                    print(err)
                else:
                    pass
            except AttributeError:
                pass

    def NPS(self, sel_text):
        """Prepare plot for test NPS."""
        imgno = self.main.gui.active_img_no
        self.title = 'Noise Power Spectrum'
        self.ytitle = r'NPS ($\mathregular{mm^{2}}$)'
        self.xtitle = 'NPS (pr mm)'
        normalize = self.main.current_paramset.nps_normalize

        if self.main.current_modality in ['Xray', 'Mammo']:
            test_widget = self.main.stack_test_tabs.currentWidget()
            sel_prof = test_widget.nps_plot_profile.currentText()
            profile_keys = []
            info_texts = []
            styles = []
            if sel_prof  == 'all':
                profile_keys = ['radial_profile', 'u_profile', 'v_profile']
                info_texts = [' radial', ' horizontal', ' vertical']
                styles = ['-b', '-m', '-c']
            else:
                if sel_prof == 'radial':
                    profile_keys = ['radial_profile']
                    info_texts.append(' radial')
                    styles.append('-b')
                if 'horizontal' in sel_prof:
                    profile_keys.append('u_profile')
                    info_texts.append(' horizontal')
                    styles.append('-m')
                if 'vertical' in sel_prof:
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

                maxy = max(max_ys)
                if self.main.current_modality == 'CT':
                    self.curves.append({
                        'label': '_nolegend_',
                        'xvals': [
                            details_dict['median_freq'], details_dict['median_freq']],
                        'yvals': [0, norm_factor * details_dict['median_val']],
                        'style': '-b'})
                    self.ax.text(
                        1.05*details_dict['median_freq'], 0.5*maxy,
                        'Median', ha='left', size=8, color='b')

                nyquist_freq = 1/(2.*self.main.imgs[imgno].pix[0])
                self.curves.append({
                    'label': '_nolegend_',
                    'xvals': [nyquist_freq, nyquist_freq],
                    'yvals': [0, maxy],
                    'style': ':' + self.color_k
                     })
                self.ax.text(
                    0.9*nyquist_freq, 0.5*maxy,
                    'Nyquist frequency', ha='left', size=8, color=self.color_gray)
            except (KeyError, IndexError):
                pass

        def plot_average_NPS():  # only for CT
            try:
                xvals, y_avg, errmsg = get_avg_NPS_curve(
                    self.main.results['NPS'], normalize=normalize)
                self.curves.append(
                    {'label': 'average NPS', 'xvals': xvals, 'yvals': y_avg,
                     'style': '-' + self.color_k})
                median_frequency, median_val = find_median_spectrum(xvals, y_avg)
                self.curves.append({
                    'label': '_nolegend_',
                    'xvals': [median_frequency, median_frequency],
                    'yvals': [0, median_val],
                    'style': '-' + self.color_k})
                self.ax.text(
                    1.05*median_frequency, 0.3*median_val,
                    'Median', ha='left', size=8, color=self.color_k)
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
                            color = 'r' if i == imgno else self.color_darkgray
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
                                    self.ax.text(
                                        1.05*details_dict['median_freq'],
                                        0.5*np.max(max_ys),
                                        'Median', ha='left', size=8, color='r')
                                else:
                                    self.curves.append(curve_med)

                if curve_this is not None:
                    self.curves.append(curve_this)
                if med_this is not None:
                    self.curves.append(med_this)
            except (KeyError, IndexError):
                pass

        if sel_text == '':
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

    def Rec(self, sel_text):
        """Prepare plot for test PET recovery curve."""
        details_dict = self.main.results['Rec']['details_dict']
        test_widget = self.main.stack_test_tabs.currentWidget()

        def plot_Rec_curve(sel_text):
            """Plot Rec values together with EARL tolerances."""
            self.title = sel_text
            self.xtitle = 'Sphere diameter (mm)'
            roi_sizes = self.main.current_paramset.rec_sphere_diameters
            wid_rec_type = test_widget.rec_type
            rec_types = [wid_rec_type.itemText(i) for i
                         in range(wid_rec_type.count())]
            rec_type = rec_types.index(sel_text)
            subtr = -1 if rec_type < 3 else -2
            self.curves.append(
                {'label': 'measured values', 'xvals': roi_sizes,
                 'yvals': details_dict['values'][rec_type][:subtr], 'style': '-bo'})

            if rec_type < 3:
                self.ytitle = 'Recovery coefficient'
                idx = test_widget.rec_earl.currentIndex()
                if idx > 0:
                    proceed = True
                    if idx == 1 and rec_type == 2:
                        proceed = False
                    if roi_sizes != [10., 13., 17., 22., 28., 37.]:  # EARL tolerances
                        proceed = False
                    if proceed:
                        if idx == 1:  # EARL 1
                            yvals = [[.27, .44, .57, .63, .72, .76],  # lower A50
                                     [.43, .6, .73, .78, .85, .89],  # upper A50
                                     [.34, .59, .73, .83, .91, .95],  # lower max
                                     [.57, .85, 1.01, 1.09, 1.13, 1.16],  # upper max
                                     [None] * 6, [None] * 6  # peak
                                     ]
                        elif idx == 2:  # EARL 2
                            yvals = [[.39, .63, .76, .8, .82, .85],  # lower A50
                                     [.61, .86, .97, .99, .97, 1.],  # upper A50
                                     [.52, .85, 1., 1.01, 1.01, 1.05],  # lower max
                                     [.88, 1.22, 1.38, 1.32, 1.26, 1.29],  # upper max
                                     [.3, .46, 0.75, 0.9, 0.9, 0.9],  # lower peak
                                     [.43, .7, 0.98, 1.1, 1.1, 1.1]  # upper peak
                                     ]
                        idx_lower = 2 * (rec_type % 3)
                        tolmin = {'label': f'EARL{idx} lower',
                                  'xvals': roi_sizes,
                                  'yvals': yvals[idx_lower],
                                  'style': '--' + self.color_k}
                        tolmax = {'label': f'EARL{idx} upper',
                                  'xvals': roi_sizes,
                                  'yvals': yvals[idx_lower + 1],
                                  'style': '--' + self.color_k}
                        self.curves.append(tolmin)
                        self.curves.append(tolmax)
            else:
                self.ytitle = 'Image values (Bq/ml)'

        def plot_z_profile():
            """Plot z-profile of used slices."""
            self.title = 'z profile'
            self.xtitle = 'Slice position (mm)'
            self.ytitle = 'Pixel value'
            self.curves.append(
                {'label': 'first background ROI average, all',
                 'xvals': details_dict['zpos'],
                 'yvals': details_dict['roi_averages'], 'style': '-' + self.color_k})
            self.curves.append(
                {'label': 'first background ROI average, used slices',
                 'xvals': details_dict['used_zpos'],
                 'yvals': details_dict['used_roi_averages'], 'style': '-r'})
            self.curves.append(
                {'label': 'max in image, used slices spheres',
                 'xvals': details_dict['used_zpos_spheres'],
                 'yvals': details_dict['used_roi_maxs'], 'style': '-b'})

        if sel_text == '':
            try:
                sel_text = test_widget.rec_plot.currentText()
                if 'z-profile' in sel_text:
                    plot_z_profile()  # not an option if generate_report
                    sel_text = ''
                else:
                    sel_text = test_widget.rec_type.currentText()
            except AttributeError:
                sel_text = ''
        if sel_text != '':
            plot_Rec_curve(sel_text)

    def Rin(self, sel_text):
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
                     'yvals': details_dict['radial_profile_smoothed'],
                     'style': '-' + self.color_k})
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

    def ROI(self, sel_text):
        """Prepare plot for test ROI."""
        xvals = []
        yvals = []
        for i, row in enumerate(self.main.results['ROI']['values']):
            if len(row) > 0:
                xvals.append(i)
                if self.main.current_paramset.roi_use_table == 0:
                    yvals.append(row[0])
                else:
                    yvals.append(row)
        if self.main.current_paramset.roi_use_table == 0:
            curve = {'label': 'Average',
                     'xvals': xvals,
                     'yvals': yvals,
                     'style': '-b'}
            self.curves.append(curve)
            self.ytitle = 'Average pixel value'
        else:
            headers = self.main.results['ROI']['headers']
            for i in range(len(yvals[0])):
                curve = {'label': headers[i],
                         'xvals': xvals,
                         'yvals': [vals[i] for vals in yvals],
                         'style': '-',
                         'color': COLORS[i % len(COLORS)]}
                self.curves.append(curve)
            test_widget = self.main.stack_test_tabs.currentWidget()
            val_text = test_widget.roi_table_val.currentText()
            self.ytitle = val_text
        self.xtitle = 'Image index'
        self.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        self.ax.set_xticks(xvals)

    def Sli(self, sel_text):
        """Prepare plot for test Sli."""
        if self.main.current_modality in ['CT', 'MR']:
            self.title = 'Profiles for slice thickness calculations'
            imgno = self.main.gui.active_img_no
            details_dict = self.main.results['Sli']['details_dict'][imgno]

            # ROI h_colors = ['b', 'lime']
            # ROI v_colors = ['c', 'r', 'm', 'darkorange']
            if self.main.current_modality == 'CT':
                if self.main.current_paramset.sli_type == 0:
                    colors = ['b', 'lime', 'c', 'r']
                elif self.main.current_paramset.sli_type == 1:
                    colors = ['b', 'lime', 'c', 'r', 'm', 'darkorange']
                elif self.main.current_paramset.sli_type == 2:
                    colors = ['c', 'r']
                elif self.main.current_paramset.sli_type == 3:
                    colors = ['b']
                elif self.main.current_paramset.sli_type == 4:
                    colors = ['b', 'lime']
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
                n_pix = len(details_dict['profiles'][l_idx])
                xvals = [details_dict['dx'] * i for i in range(n_pix)]
                self.curves.append({
                    'label': details_dict['labels'][l_idx],
                    'xvals': xvals,
                    'yvals': details_dict['profiles'][l_idx],
                    'color': colors[l_idx]})

                try:
                    self.curves.append(
                        {'label': f'{details_dict["labels"][l_idx]} envelope',
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

    def SNI(self, sel_text):
        """Prepare plot for test NM SNI test."""
        imgno = self.main.gui.active_img_no
        if self.main.current_paramset.sni_sum_first:
            imgno = 0
        show_filter_2 = False
        if self.main.current_paramset.sni_channels:
            if self.main.tab_nm.sni_plot_low.btn_false.isChecked():
                show_filter_2 = True
        self.title = 'Calculations to get Structured Noise Index'
        self.ytitle = r'NPS ($\mathregular{mm^{2}}$)'
        self.xtitle = 'frequency (pr mm)'
        details_dict = self.main.results['SNI']['details_dict'][imgno]
        cbx = self.main.tab_nm.sni_selected_roi
        roi_names = [cbx.itemText(i) for i in range(cbx.count())]
        nyquist_freq = 1/(
            2.*self.main.imgs[imgno].pix[0]*self.main.current_paramset.sni_scale_factor)

        def plot_SNI_values_bar():
            """Plot SNI values as columns pr ROI."""
            try:
                if show_filter_2:
                    SNI_values = details_dict['SNI_values_2']
                else:
                    SNI_values = details_dict['SNI_values']
            except KeyError:
                SNI_values = []
            if SNI_values:
                self.bars.append({'names': roi_names, 'values': SNI_values})
                self.title = 'Structured Noise Index for each ROI in selected image'
                self.ytitle = 'SNI'
                self.xtitle = 'ROI'

        def plot_all_SNI_values():
            """Plot SNI values for all images pr ROI."""
            markers = ['.', 'o', 'v', 's', 'D', 'P']
            idxs = [i for i in range(len(roi_names))]
            for imgno, dd in enumerate(self.main.results['SNI']['details_dict']):
                try:
                    if show_filter_2:
                        SNI_values = dd['SNI_values_2']
                    else:
                        SNI_values = dd['SNI_values']
                except KeyError:
                    SNI_values = []
                if SNI_values:
                    self.title = 'Structured Noise Index for each ROI in all images'
                    self.ytitle = 'SNI'
                    self.xtitle = 'ROI'
                    c = COLORS[imgno % len(COLORS)]
                    m = markers[imgno % len(markers)]
                    self.curves.append(
                        {'label': f'Img {imgno}', 'xvals': idxs, 'xticks': roi_names,
                         'yvals': SNI_values, 'style': f'-{c}{m}'})

        def plot_SNI_result_table():
            """Plot SNI values as columns pr ROI."""
            self.title = 'Structured Noise Index, values from result table'
            self.ytitle = 'SNI'
            self.xtitle = 'Image number'
            labels = copy.deepcopy(self.main.results['SNI']['headers'])
            values = copy.deepcopy(self.main.results['SNI']['values'])
            img_nos = []
            colors = ['red', 'blue', 'green', 'cyan'] + [self.color_k] * 4
            styles = ['-', '-', '-', '-', '-', '--', ':', '-.']

            try:
                yvals_all = [[] for label in labels]
                for i, row in enumerate(values):
                    if len(row) > 0:
                        img_nos.append(i)
                        for j, val in enumerate(row):
                            yvals_all[j].append(val)
                if self.main.current_paramset.sni_alt == 0:
                    yvals_all.pop(0)  # do not include max in plot
                    labels.pop(0)
                xvals = img_nos
                for j, yvals in enumerate(yvals_all):
                    self.curves.append(
                        {'label': labels[j], 'xvals': xvals,
                         'yvals': yvals, 'color': colors[j], 'style': styles[j]})
            except (KeyError, IndexError):
                pass

        def plot_filtered_NPS():
            """Plot filtered NPS + NPS structure for selected ROI +quantum noise."""
            self.default_range_x = [0, nyquist_freq]
            roi_idx = self.main.tab_nm.sni_selected_roi.currentIndex()
            if self.main.current_paramset.sni_type > 0:
                self.main.wid_image_display.canvas.roi_draw()
            proceed = True
            try:
                details_dict_roi = details_dict['pr_roi'][roi_idx]
            except IndexError:
                proceed = False
            if proceed:
                name = self.main.tab_nm.sni_selected_roi.currentText()
                self.title = f'Filtered NPS and NPS structure for {name}'
                suffix_2 = '_2' if show_filter_2 else ''
                yvals = details_dict_roi[f'rNPS_filt{suffix_2}']
                xvals = details_dict_roi['freq']
                self.curves.append(
                    {'label': 'NPS with filter',
                     'xvals': xvals, 'yvals': yvals, 'style': '-b'})
                self.default_range_y = [0, 1.4 * np.max(yvals[2:])]

                self.curves.append(
                    {'label': 'NPS',
                     'xvals': xvals, 'yvals': details_dict_roi['rNPS'],
                     'style': ':b'})
                self.curves.append(
                    {'label': 'NPS structured noise with filter',
                     'xvals': xvals,
                     'yvals': details_dict_roi[f'rNPS_struct_filt{suffix_2}'],
                     'style': '-r'})
                self.curves.append(
                    {'label': 'NPS structured noise',
                     'xvals': xvals,
                     'yvals': details_dict_roi['rNPS_struct'],
                     'style': ':r'})

                if isinstance(details_dict_roi['quantum_noise'], float):
                    yvals = [details_dict_roi['quantum_noise']] * len(xvals)
                    self.curves.append(
                        {'label': 'NPS estimated quantum noise',
                         'xvals': xvals, 'yvals': yvals, 'style': ':' + self.color_k})
                else:
                    yvals = details_dict_roi['rNPS_quantum_noise']
                    self.curves.append(
                        {'label': 'NPS estimated quantum noise',
                         'xvals': xvals, 'yvals': yvals, 'style': ':' + self.color_k})

        def plot_all_NPS():
            """Plot NPS for all ROIs + filter."""
            if self.main.current_paramset.sni_type == 0:
                self.title = 'NPS all ROIs'
                colors = ['red', 'blue', 'green', 'cyan'] + [self.color_k] * 4
                styles = ['-', '-', '-', '-', '-', '--', ':', '-.']
                for roi_no in range(8):
                    details_dict_roi = details_dict['pr_roi'][roi_no]
                    yvals = details_dict_roi['rNPS']
                    if roi_no == 0:
                        self.default_range_y = [0, 1.4 * np.median(
                            details_dict_roi['rNPS_quantum_noise'])]
                    xvals = details_dict_roi['freq']
                    self.curves.append(
                        {'label': f'NPS ROI {roi_names[roi_no]}',
                         'xvals': xvals, 'yvals': yvals,
                         'color': colors[roi_no], 'style': styles[roi_no]})

                filter_curve = details_dict['eye_filter_large']
                suffix_2 = '_2' if show_filter_2 else ''
                yvals = filter_curve[f'V{suffix_2}'] * np.median(details_dict[
                    'pr_roi'][0]['rNPS'])
                self.curves.append(
                    {'label': 'Filter',
                     'xvals': filter_curve['r'], 'yvals': yvals,
                     'color': self.color_darkgray, 'style': '-'})
                self.default_range_x = [0, nyquist_freq]
            else:
                for roi_no, dd in enumerate(details_dict['pr_roi']):
                    yvals = dd['rNPS']
                    if roi_no == 0:
                        self.default_range_y = [0, 1.4 * np.median(
                            details_dict_roi['rNPS_quantum_noise'])]
                    xvals = dd['freq']
                    self.curves.append(
                        {'label': '_no_legend_',
                         'xvals': xvals, 'yvals': yvals})
                for suf in ['large', 'small']:
                    filter_curve = details_dict['eye_filter_' + suf]
                    yvals = filter_curve[f'V{suffix_2}'] * np.median(details_dict[
                        'pr_roi'][0]['rNPS'])
                    self.curves.append(
                        {'label': 'Filter',
                         'xvals': filter_curve['r'], 'yvals': yvals,
                         'color': self.color_darkgray, 'style': '-'})
                self.default_range_x = [0, nyquist_freq]

        def plot_filtered_max_avg():
            self.default_range_x = [0, nyquist_freq]
            suffix_2 = '_2' if show_filter_2 else ''
            xvals = details_dict['pr_roi'][2]['freq']  # freq for small ROIs
            yvals = details_dict[f'avg_rNPS_filt{suffix_2}_small']
            self.curves.append(
                {'label': 'avg NPS with filter',
                 'xvals': xvals, 'yvals': yvals, 'style': '-b'})
            yvals = details_dict[f'avg_rNPS_struct_filt{suffix_2}_small']
            self.curves.append(
                {'label': 'avg structured NPS with filter',
                 'xvals': xvals, 'yvals': yvals, 'style': '-r'})
            idx_max = details_dict[f'roi_max_idx_small{suffix_2}']
            yvals = details_dict['pr_roi'][idx_max + 2][f'rNPS_filt{suffix_2}']
            self.curves.append(
                {'label': 'max NPS with filter',
                 'xvals': xvals, 'yvals': yvals, 'style': ':b'})
            self.default_range_y = [0, 1.05 * np.max(yvals)]
            yvals = details_dict['pr_roi'][idx_max + 2][f'rNPS_struct_filt{suffix_2}']
            self.curves.append(
                {'label': 'max structured NPS with filter',
                 'xvals': xvals, 'yvals': yvals, 'style': ':r'})
            name = roi_names[idx_max + 2]
            self.title = f'Filtered NPS and NPS structure for {name} (max) and avg'

        def plot_filters():
            xvals = details_dict['eye_filter_small']['r']
            yvals = details_dict['eye_filter_small']['V']
            if 'V2' in details_dict['eye_filter_small']:
                labels = ['low', 'high']
            else:
                labels = ['Human visual response filter']
            self.curves.append(
                {'label': labels[0],
                 'xvals': xvals, 'yvals': yvals, 'style': '-k'})
            if len(labels) == 2:
                yvals = yvals = details_dict['eye_filter_small']['V2']
                self.curves.append(
                    {'label': labels[1],
                     'xvals': xvals, 'yvals': yvals, 'style': ':k'})
            self.title = 'Frequency filter(s)'
            self.ytitle = 'Ratio'
            self.default_range_y = [0, 1.1]
            self.default_range_x = [0, nyquist_freq]

        def plot_curve_corr_check():
            self.title = 'Curvature correction check'
            details_dict = self.main.results['SNI']['details_dict'][imgno]
            if 'corrected_image' in details_dict:
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
                if 'reference_estimated_noise' in details_dict:
                    ref = details_dict['reference_estimated_noise']
                    if ref.shape == corrected_img.shape:
                        prof_y = np.mean(ref[:, xhalf-nx:xhalf+nx], axis=1)
                        prof_x = np.mean(ref[yhalf-ny:yhalf+ny, :], axis=0)
                        self.curves.append({
                            'label': 'Central 10% rows ref. corr.',
                            'xvals': np.arange(len(prof_x)),
                            'yvals': prof_x,
                            'style': 'k'})
                        self.curves.append({
                            'label': 'Central 10% columns ref. corr.',
                            'xvals': np.arange(len(prof_y)),
                            'yvals': prof_y,
                            'style': ':k'})

        if sel_text == '':
            test_widget = self.main.stack_test_tabs.currentWidget()
            try:
                sel_text = test_widget.sni_plot.currentText()
            except AttributeError:
                sel_text = ''
        if 'SNI values each ROI' in sel_text:
            plot_SNI_values_bar()
        elif 'SNI values all images' in sel_text:
            if self.main.current_paramset.sni_type == 3:
                plot_all_SNI_values()
            else:
                plot_SNI_result_table()
        elif 'Filtered' in sel_text:
            if 'max' in sel_text:
                plot_filtered_max_avg()
            else:
                plot_filtered_NPS()
        elif 'all' in sel_text:
            plot_all_NPS()
        elif 'correction' in sel_text:
            plot_curve_corr_check()
        elif 'Filter(s)' in sel_text:
            plot_filters()

    def Spe(self, sel_text):
        """Prepare plot for test NM Speed test."""
        self.title = 'Scan speed profile'
        self.xtitle = 'Position (mm)'
        self.ytitle = 'Difference from mean %'
        imgno = self.main.gui.active_img_no
        details_dict = self.main.results['Spe']['details_dict'][imgno]
        if 'profile_pos' in details_dict:
            xvals = details_dict['profile_pos']
            self.curves.append(
                {'label': 'profile', 'xvals': xvals,
                 'yvals': details_dict['diff_profile'], 'style': '-b'})
            tolmax = {'label': 'tolerance max',
                      'xvals': [min(xvals), max(xvals)],
                      'yvals': [5, 5],
                      'style': '--' + self.color_k}
            tolmin = tolmax.copy()
            tolmin['yvals'] = [-5, -5]
            tolmin['label'] = 'tolerance min'
            self.curves.append(tolmin)
            self.curves.append(tolmax)

            self.default_range_y = self.test_values_outside_yrange([-7, 7])

    def TTF(self, sel_text):
        """Prepare plot for test TTF."""
        details_dicts = self.main.results['TTF']['details_dict']
        imgno = self.main.gui.active_img_no
        materials = self.main.current_paramset.ttf_table.labels

        def prepare_plot_MTF(idxs):
            nyquist_freq = 1/(2.*self.main.imgs[imgno].pix[0])
            self.xtitle = 'frequency [1/mm]'
            self.ytitle = 'MTF'
            colors = [self.color_k, 'r']  # gaussian black, discrete red
            linestyles = ['-', '--']
            infotext = ['gaussian', 'discrete']
            prefix = ['g', 'd']
            if len(idxs) > 1:
                cno = int(not self.main.current_paramset.ttf_gaussian)
                colors = [colors[cno]]
                linestyles = [linestyles[cno]]
                infotext = [infotext[cno]]
                prefix = [prefix[cno]]

            for m_idx in idxs:
                dd = details_dicts[m_idx]
                if 'dMTF_details' in dd:
                    for no in range(len(prefix)):
                        key = f'{prefix[no]}MTF_details'
                        if key in dd:
                            dd_this = dd[key]
                            xvals = dd_this['MTF_freq']
                            yvals = dd_this['MTF']
                            color = colors[no] if len(idxs) == 1 else COLORS[m_idx]
                            try:
                                self.curves.append({
                                    'label': infotext[no] + ' MTF ' + materials[m_idx],
                                    'xvals': xvals,
                                    'yvals': yvals,
                                    'color': color
                                     })
                            except IndexError:
                                pass

                if 'MTF_filtered' in dd['gMTF_details'] and len(idxs) == 1:
                    yvals = dd['gMTF_details']['MTF_filtered']
                    if yvals is not None:
                        xvals = dd['gMTF_details']['MTF_freq']
                        yvals = dd['gMTF_details']['MTF_filtered']
                        self.curves.append({
                            'label': 'gaussian MTF pre-smoothed',
                            'xvals': xvals,
                            'yvals': yvals,
                            'style': '--',
                            'color': self.color_gray
                             })

            if len(self.curves) > 0:
                self.default_range_y = self.test_values_outside_yrange(
                    [0, 1.3], limit_xrange=[0, nyquist_freq])
                self.default_range_x = [0, 1.1 * nyquist_freq]
                self.curves.append({
                    'label': '_nolegend_',
                    'xvals': [nyquist_freq, nyquist_freq],
                    'yvals': [0, 1.3],
                    'style': ':' + self.color_k
                     })
                self.ax.text(0.9*nyquist_freq, 0.5, 'Nyquist frequency',
                             ha='left', size=8, color=self.color_gray)

        def prepare_plot_LSF(idxs):
            self.xtitle = 'pos (mm)'
            self.ytitle = 'LSF'
            self.legend_location = 'upper left'

            lbl_prefilter = ''
            prefilter = False

            for m_idx in idxs:
                dd = details_dicts[m_idx]

                dd_this = dd['gMTF_details']
                curve_corrected = None
                if len(idxs) == 1:
                    if 'sigma_prefilter' in dd and len(idxs) == 1:
                        if dd['sigma_prefilter'] > 0:
                            prefilter = True
                            lbl_prefilter = ' presmoothed'

                    xvals = dd['LSF_x']
                    yvals = dd['LSF']
                    self.curves.append({
                        'label': 'LSF',
                        'xvals': xvals,
                        'yvals': yvals,
                        'style': 'r'
                         })
                    if prefilter:
                        xvals = dd_this['LSF_fit_x']
                        yvals = dd_this['LSF_prefit']
                        self.curves.append({
                            'label': f'LSF{lbl_prefilter}',
                            'xvals': xvals,
                            'yvals': yvals,
                            'style': '-b'
                             })
                        if 'LSF_corrected' in dd_this:
                            yvals = dd_this['LSF_corrected']
                            if yvals is not None:
                                curve_corrected = {
                                    'label': (
                                        'LSF corrected - gaussian fit'),
                                    'xvals': xvals,
                                    'yvals': yvals,
                                    'style': '--',
                                    'color': 'green'
                                     }

                xvals = dd_this['LSF_fit_x']
                yvals = dd_this['LSF_fit']
                color = self.color_k if len(idxs) == 1 else COLORS[m_idx]
                self.curves.append({
                    'label': f'LSF{lbl_prefilter} - gaussian fit',
                    'xvals': xvals,
                    'yvals': yvals,
                    'style': '-',
                    'color': color
                     })
                if curve_corrected:
                    self.curves.append(curve_corrected)

                if 'dMTF_details' in dd and len(idxs) == 1:
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
                                    'style': ':' + self.color_k
                                    })
                                self.ax.text(
                                    x * cw, np.mean(minmax), 'cut',
                                    ha='left', size=8, color=self.color_gray)
                            if 'cut_width_fade' in dd_this:
                                cwf = dd_this['cut_width_fade']
                                for x in [-1, 1]:
                                    self.curves.append({
                                        'label': '_nolegend_',
                                        'xvals': [x * cwf] * 2,
                                        'yvals': minmax,
                                        'style': ':' + self.color_k
                                        })
                                    self.ax.text(
                                        x * cwf, np.mean(minmax), 'fade',
                                        ha='left', size=8, color=self.color_gray)
                            self.default_range_x = [-1.5*cw, 1.5*cw]

        def prepare_plot_sorted_pix(idxs):
            self.xtitle = 'pos (mm)'
            self.ytitle = 'Pixel value'

            dot_colors = COLORS if len(idxs) > 0 else ['mediumseagreen']
            try:
                for m_idx in idxs:
                    dd = details_dicts[m_idx]
                    for slino, yvals in enumerate(dd['sorted_pixels']):
                        lbl = f'Sorted pixels {materials[m_idx]}' if slino == 0 \
                            else '_nolegend_'
                        self.curves.append({
                            'label': lbl,
                            'xvals': dd['sorted_pixels_x'],
                            'yvals': yvals,
                            'style': '.',
                            'color': dot_colors[m_idx % len(dot_colors)],
                            'markersize': 2.,
                            'alpha': 0.5
                             })

                    if 'interpolated_x' in dd and len(idxs) == 1:
                        self.curves.append({
                                'label': f'Interpolated {materials[m_idx]}',
                                'xvals': dd['interpolated_x'],
                                'yvals': dd['interpolated'],
                                'style': '--',
                                'color': COLORS[m_idx]
                                 })

                    if 'presmoothed' in dd and len(idxs) == 1:
                        self.curves.append({
                            'label': 'Presmoothed',
                            'xvals': dd['interpolated_x'],
                            'yvals': dd['presmoothed'],
                            'style': '-b'
                             })
            except IndexError:
                pass

        def prepare_plot_centered_profiles(idxs):
            m_idx = idxs[0]  # force only first
            center_xy = details_dicts[m_idx]['center_xy']
            submatrix = details_dicts[m_idx]['matrix']

            self.xtitle = 'pos (mm)'
            self.ytitle = 'Pixel value'

            marked_imgs = self.main.get_marked_imgs_current_test()
            pix = self.main.imgs[marked_imgs[0]].pix[0]

            for no, sli in enumerate(submatrix):
                proceed = True
                if no not in marked_imgs:
                    proceed = False
                if proceed and sli is not None:
                    suffix = f' {no}' if len(submatrix) > 1 else ''
                    szy, szx = sli.shape
                    xvals1 = pix * (np.arange(szx) - center_xy[0])
                    yvals1 = sli[round(center_xy[1]), :]
                    self.curves.append({
                        'label': 'x' + suffix,
                        'xvals': xvals1,
                        'yvals': yvals1,
                        'style': '-',
                        'color': COLORS[no % len(COLORS)]
                         })
                    xvals2 = pix * (np.arange(szy) - center_xy[1])
                    yvals2 = sli[:, round(center_xy[0])]
                    self.curves.append({
                        'label': 'y' + suffix,
                        'xvals': xvals2,
                        'yvals': yvals2,
                        'style': '--',
                        'color': COLORS[no % len(COLORS)]
                         })

        #TODO if sel_text == '':
        test_widget = self.main.stack_test_tabs.currentWidget()
        try:
            sel_text = test_widget.ttf_plot.currentText()
            sel_material_idx = test_widget.ttf_plot_material.currentIndex() - 1
        except AttributeError:
            sel_text = ''
            sel_material_idx = -1  # all
        if sel_material_idx == -1:
            sel_material_idxs = list(range(len(details_dicts)))
        else:
            sel_material_idxs = [sel_material_idx]

        txt_add = ''
        if sel_text == 'MTF':
            prepare_plot_MTF(sel_material_idxs)
        elif sel_text == 'LSF':
            prepare_plot_LSF(sel_material_idxs)
        elif sel_text == 'Sorted pixel values':
            prepare_plot_sorted_pix(sel_material_idxs)
        elif sel_text == 'Centered xy profiles':
            prepare_plot_centered_profiles(sel_material_idxs)
            txt_add = f' {materials[sel_material_idxs[0]]}'
        self.title = sel_text + txt_add

    def Uni(self, sel_text):
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
            try:
                details_dict = self.main.results['Uni']['details_dict'][imgno]
                if 'corrected_image' in details_dict:
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
            except IndexError:
                pass

    def vendor(self, sel_text):
        """Prepare plot if vendor test results contain details."""
        # Currently only energy spectrum from Siemens gamma camera have this option.
        self.title = 'Energy spectrum'
        self.xtitle = 'Energy (keV)'
        self.ytitle = 'Counts'
        details_list = self.main.results['vendor']['details']
        colors = ['b', 'r', 'c', 'm', 'lime', 'darkorange']
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

        tb_right_top = QToolBar()
        self.tool_resultsize = ToolMaximizeResults(self.main)
        tb_right_top.addWidget(self.tool_resultsize)
        tb_right_top.setOrientation(Qt.Orientation.Vertical)
        tb_right_top.addWidget(self.tool_profile)
        tb_right_top.addWidget(self.tool_cmap)
        tb_right_top.addWidget(self.tool_rectangle)
        tb_right_top.addWidget(self.tool_zoom_as_active_image)
        hlo_right = QHBoxLayout()
        hlo_right.addWidget(toolb)
        hlo_right.addWidget(tb_right_top)
        hlo_right.addStretch()

        self.image_title = QLabel()
        tb_top = QToolBar()
        tb_top.addWidget(GenericImageToolbarPosVal(self.canvas, self))

        vlo_left = QVBoxLayout()
        vlo_left.addWidget(self.image_title)
        vlo_left.addWidget(tb_top)
        vlo_left.addWidget(self.canvas)
        hlo.addLayout(vlo_left)
        hlo.addLayout(hlo_right)

        wid_image_toolbars = QWidget()
        wid_image_toolbars.setLayout(hlo)

        self.wid_window_level = uir.WindowLevelWidget(
            self, show_dicom_wl=False, show_lock_wl=False)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(wid_image_toolbars)
        self.splitter.addWidget(self.wid_window_level)
        hlo_ = QHBoxLayout()
        hlo_.addWidget(self.splitter)
        self.setLayout(hlo_)

        self.reset_split_sizes()

    def set_active_image_min_max(self, minval, maxval):
        """Update window level."""
        if self.canvas.current_image is not None:
            self.canvas.img.set_clim(vmin=minval, vmax=maxval)
            self.canvas.draw_idle()

    def reset_split_sizes(self):
        """Set and reset QSplitter sizes."""
        default_rgt_panel = self.main.gui.panel_width*0.8
        self.splitter.setSizes(
            [round(default_rgt_panel*0.7), round(default_rgt_panel*0.3)])

    def hide_window_level(self):
        """Set window level widget to zero width."""
        self.splitter.setSizes([round(self.main.gui.panel_width*0.8), 0])

    def zoom_as_active_image(self):
        self.canvas.zoom_as_active_image()


class ResultImageNavigationToolbar(ImageNavigationToolbar):
    """Matplotlib navigation toolbar with some modifications."""

    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        for act in self.actions():
            if act.text() in ['Customize']:
                self.removeAction(act)
        self.setOrientation(Qt.Orientation.Vertical)
