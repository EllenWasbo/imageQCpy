#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for main window of imageQC - left side widgets.

@author: Ellen Wasbo
"""
import os
import numpy as np
import pandas as pd

from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QSlider, QMenu, QAction, QToolBar, QToolButton,
    QMessageBox, QInputDialog, QTreeWidget, QTreeWidgetItem, QFileDialog,
    QDialogButtonBox
    )
import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# imageQC block start
from imageQC.ui.ui_image_canvas import get_rotated_crosshair
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.ui.tag_patterns import TagPatternEditDialog
from imageQC.config.iQCconstants import ENV_ICON_PATH, QUICKTEST_OPTIONS
from imageQC.config import config_classes as cfc
from imageQC.scripts import dcm
from imageQC.scripts import mini_methods_format as mmf
# imageQC block end


class TreeFileList(QTreeWidget):
    """QTreeWidget for list of images marked for testing."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent
        self.setColumnCount(3)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setHeaderLabels(['Image', 'Frame', 'Test'])
        self.setColumnWidth(0, round(0.8*self.main.gui.panel_width))
        self.setColumnWidth(1, 90)
        self.currentItemChanged.connect(self.main.update_active_img)
        self.itemDoubleClicked.connect(self.dbl_click_item)
        self.installEventFilter(self)
        self.setRootIsDecorated(False)

    def get_selected_imgs(self):
        """Get selected images in file list.

        Returns
        -------
        sel_rows : list of int
        """
        sel_rows = []
        last_selected = 0
        for sel in self.selectedIndexes():
            if sel.column() > -1:
                sel_rows.append(sel.row())
        if len(sel_rows) > 0:
            last_selected = sel_rows[-1]
            sel_rows = list(set(sel_rows))  # remove duplicates
        return (sel_rows, last_selected)

    def get_marked_imgs_current_test(self):
        """Get images (idx) marked for current test.

        Returns
        -------
        marked_img_ids : list of int
        """
        if self.main.wid_quicktest.gb_quicktest.isChecked():
            marked_img_ids = [
                i for i, im in enumerate(self.main.imgs)
                if self.main.current_test in im.marked_quicktest]
        else:
            marked_img_ids = [
                i for i, im in enumerate(self.main.imgs) if im.marked]
            if len(marked_img_ids) == 0:
                marked_img_ids = list(np.arange(len(self.main.imgs)))
        return marked_img_ids

    def update_file_list(self):
        """Populate tree with filepath/pattern and test indicators."""
        self.clear()
        quicktest_active = self.main.wid_quicktest.gb_quicktest.isChecked()
        if len(self.main.imgs) == 0:
            QTreeWidgetItem(self, [''] * 3)
        else:
            for img in self.main.imgs:
                if quicktest_active:
                    test_string = '+'.join(img.marked_quicktest)
                    if img.quicktest_image_name != '':
                        test_string += f' (Name: {img.quicktest_image_name})'
                    if img.quicktest_group_name != '':
                        test_string += f' (Group: {img.quicktest_group_name})'
                else:
                    test_string = 'x' if img.marked else ''
                frameno = f'{img.frame_number}' if img.frame_number > -1 else ''

                if self.main.cbox_file_list_display.currentIndex() == 0:
                    file_text = img.filepath
                else:
                    file_text = ' '.join(img.file_list_strings)
                QTreeWidgetItem(self, [file_text, frameno, test_string])
            self.main.lbl_n_loaded.setText(str(len(self.main.imgs)))
            self.setCurrentItem(self.topLevelItem(
                self.main.gui.active_img_no))

    def dragEnterEvent(self, event):
        """Handle drag enter event. Opening files by drag/drop."""
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Handle drag move event. Opening files by drag/drop."""
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Accept drop event with file paths."""
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            file_paths = []
            for url in event.mimeData().urls():
                file_paths.append(str(url.toLocalFile()))
            if len(file_paths) > 0:
                self.main.open_files(file_list=file_paths)
        else:
            event.ignore()

    def eventFilter(self, source, event):
        """Handle context menu events (rightclicks)."""
        if event.type() == QEvent.ContextMenu:
            ctx_menu = QMenu(self)
            if self.main.wid_quicktest.gb_quicktest.isChecked():
                act_edit_mark = QAction('Edit marked tests...')
                act_edit_mark.triggered.connect(
                    lambda: self.set_marking(qt_edit=True))
                ctx_menu.addActions([act_edit_mark])
            else:
                act_mark = QAction('Mark selected')
                act_mark.triggered.connect(self.set_marking)
                act_remove_mark_sel = QAction('Remove mark from selected')
                act_remove_mark_sel.triggered.connect(
                    lambda: self.set_marking(remove_mark=True))
                ctx_menu.addActions([act_mark, act_remove_mark_sel])
            act_remove_all_marks = QAction('Remove all marks')
            act_remove_all_marks.triggered.connect(self.clear_marking)
            act_select_inverse = QAction('Select inverse')
            act_select_inverse.triggered.connect(self.select_inverse)
            act_close_sel = QAction('Close selected')
            act_close_sel.triggered.connect(self.close_selected)
            ctx_menu.addActions(
                    [act_remove_all_marks, act_select_inverse, act_close_sel])
            if self.main.wid_quicktest.gb_quicktest.isChecked():
                ctx_menu.addSeparator()
                act_set_image_name = QAction('Set image name...')
                act_set_image_name.triggered.connect(
                    lambda: self.set_image_or_group_name(image_or_group='image'))
                act_set_group_name = QAction('Set group name...')
                act_set_group_name.triggered.connect(
                    lambda: self.set_image_or_group_name(image_or_group='group'))
                ctx_menu.addActions([act_set_image_name, act_set_group_name])
            ctx_menu.exec(event.globalPos())

        return False

    def select_inverse(self):
        """Select the inverse of the currently selected images."""
        selrows, _ = self.get_selected_imgs()
        self.blockSignals(True)
        for i in range(len(self.main.imgs)):
            if i in selrows:
                self.topLevelItem(i).setSelected(False)
            else:
                self.topLevelItem(i).setSelected(True)
        self.blockSignals(False)

    def close_selected(self):
        """Select inverse of the currently selected images."""
        selrows, last_selected = self.get_selected_imgs()
        if len(selrows) > 0:
            if len(selrows) == len(self.main.imgs):
                self.main.clear_all_images()
            else:
                selrows.sort(reverse=True)
                for row in selrows:
                    del self.main.imgs[row]
                self.main.update_results(deleted_idxs=selrows)
                if self.main.summed_img is not None:
                    self.main.reset_summed_img()
                if last_selected < len(self.main.imgs):
                    self.main.set_active_img(last_selected)
                else:
                    self.main.set_active_img(0)
                self.update_file_list()
                self.main.refresh_img_display()

    def clear_marking(self):
        """Remove all marks for testing from selected images."""
        for imgd in self.main.imgs:
            imgd.marked_quicktest = []
            imgd.marked = False
        if self.main.summed_img is not None:
            self.main.reset_summed_img()
        self.update_file_list()
        self.main.results = {}
        self.main.refresh_results_display()

    def set_marking(self, remove_mark=False, remove_all=False, qt_edit=False):
        """Set or remove mark for testing from selected images."""
        selrows, last_selected = self.get_selected_imgs()

        proceed = True
        test_codes = []
        if qt_edit:
            dlg = SelectTestcodeDialog(
                label='Select tests to run for selected images',
                modality=self.main.current_modality,
                current_tests=self.main.imgs[selrows[0]].marked_quicktest)
            res = dlg.exec()
            if res:
                test_codes = dlg.get_checked_testcodes()
            else:
                proceed = False

        if proceed:
            for sel in selrows:
                if qt_edit:
                    #tests_this = self.main.imgs[sel].marked_quicktest
                    self.main.imgs[sel].marked_quicktest = test_codes
                    self.main.wid_quicktest.flag_edit(True)
                elif remove_mark:
                    self.main.imgs[sel].marked = False
                else:
                    self.main.imgs[sel].marked = True

            if self.main.summed_img is not None:
                self.main.reset_summed_img()
            self.update_file_list()
            if remove_all:
                self.main.results = {}
            if remove_mark:
                self.main.update_results(deleted_idxs=selrows)
            else:
                self.main.results = {}
            self.main.set_active_img(last_selected)
            self.main.refresh_results_display()

    def dbl_click_item(self):
        """If double mouseclick on item - mark item for testing."""
        self.set_marking(qt_edit=self.main.wid_quicktest.gb_quicktest.isChecked())

    def move_file(self, direction=''):
        """Resort images. Move selected image to...

        Parameters
        ----------
        direction : str
            to where: top/down/up/bottom
        """
        if direction == 'top':
            insert_no = 0
        elif direction == 'down':
            insert_no = self.main.gui.active_img_no + 1
        elif direction == 'up':
            insert_no = self.main.gui.active_img_no - 1
        elif direction == 'bottom':
            insert_no = -1
        else:
            insert_no = -2
        if insert_no > -2:
            this = self.main.imgs.pop(self.main.gui.active_img_no)
            if insert_no == -1:
                self.main.imgs.append(this)
                self.main.gui.active_img_no = len(self.main.imgs) - 1
            else:
                self.main.imgs.insert(insert_no, this)
                self.main.gui.active_img_no = insert_no
            self.update_file_list()

    def set_image_or_group_name(self, image_or_group='image'):
        """Set name of selected image for QuickTestTemplate."""
        selrows, _ = self.get_selected_imgs()
        if len(selrows) != 1:
            QMessageBox.warning(
                self, 'Select one image', 'Select only one image')
        else:
            attribute = f'quicktest_{image_or_group}_name'
            current_name = getattr(self.main.imgs[selrows[0]], attribute, '')
            text, proceed = QInputDialog.getText(
                self, f'Set {image_or_group} name for selected image',
                'Name:                                                            ',
                text=current_name)
            if proceed:
                setattr(self.main.imgs[selrows[0]], attribute, text)
                self.main.wid_quicktest.flag_edit(True)
                self.update_file_list()


class SelectTestcodeDialog(ImageQCDialog):
    """Dialog to select tests."""

    def __init__(self, label='', modality='CT', current_tests=[]):
        super().__init__()
        self.setWindowTitle('Select tests')
        vLO = QVBoxLayout()
        self.setLayout(vLO)

        vLO.addWidget(QLabel(label))
        testcodes = QUICKTEST_OPTIONS[modality]
        idx_current_tests = [
            testcodes.index(current_test) for current_test in current_tests]
        self.list_widget = uir.ListWidgetCheckable(
            texts=testcodes,
            set_checked_ids=idx_current_tests
            )
        vLO.addWidget(self.list_widget)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vLO.addWidget(self.buttonBox)

    def get_checked_testcodes(self):
        """Get list of checked testcode ids."""
        return self.list_widget.get_checked_texts()


class DicomHeaderWidget(QWidget):
    """Holder for the Dicom header widget."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent
        hlo = QHBoxLayout()
        self.setLayout(hlo)
        vlo = QVBoxLayout()
        header = QLabel('DICOM header')
        header.setFont(uir.FontItalic())
        vlo.addWidget(header)
        QLabel()
        hlo.addLayout(vlo)
        tb1 = QToolBar()
        vlo.addWidget(tb1)
        tb2 = QToolBar()
        vlo.addWidget(tb2)
        vlo.addStretch()

        act_dcm_dump = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}tags.png'),
            "View DICOM dump", self)
        act_dcm_dump.triggered.connect(self.dump_dicom)
        act_dcm_clipboard = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}copy2clipboard.png'),
            "Send specified DICOM header information as table to clipboard",
            self)
        act_dcm_clipboard.triggered.connect(lambda: self.table_dicom('clipboard'))
        act_dcm_export = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}fileCSV.png'),
            "Save specified DICOM header information as .csv", self)
        act_dcm_export.triggered.connect(lambda: self.table_dicom('csv'))
        tb1.addActions([act_dcm_dump, act_dcm_clipboard])
        tb2.addAction(act_dcm_export)

        self.list_info_general = QTreeWidget(columnCount=1)
        self.list_info_general.setHeaderLabels(['General attributes'])
        self.list_info_general.setRootIsDecorated(False)
        hlo.addWidget(self.list_info_general)
        self.list_info_modality = QTreeWidget(columnCount=1)
        self.list_info_modality.setHeaderLabels(['Modality specific attributes'])
        self.list_info_modality.setRootIsDecorated(False)
        hlo.addWidget(self.list_info_modality)

    def table_dicom(self, output_type):
        """Extract dicom header info pr image according to TagPatternFormat.

        Parameters
        ----------
        output_type : str
            'csv' or 'clipboard'
        """
        if len(self.main.imgs) == 0:
            QMessageBox.information(self.main, 'No data',
                                    'Open any images to extract data from.')
        else:
            pattern = cfc.TagPatternFormat()
            finish_btn_txt = ('Save as .csv' if output_type == 'csv'
                              else 'Copy to clipboard')
            dlg = TagPatternEditDialog(
                initial_pattern=pattern,
                modality=self.main.current_modality,
                title='Extract DICOM header information',
                typestr='format',
                accept_text=finish_btn_txt,
                reject_text='Cancel',
                save_blocked=self.main.save_blocked)
            res = dlg.exec()  # returning TagPatternFormat
            if res:
                # generate table
                pattern = dlg.get_pattern()
                n_img = len(self.main.imgs)
                tag_lists = []
                for i in range(n_img):
                    self.main.status_bar.showMessage(
                        f'Reading DICOM header {i} of {n_img}')
                    tags = dcm.get_tags(
                        self.main.imgs[i].filepath,
                        frame_number=self.main.imgs[i].frame_number,
                        tag_patterns=[pattern],
                        tag_infos=self.main.tag_infos
                        )
                    tag_lists.append(tags[0])

                ignore_cols = []
                for idx, val in enumerate(pattern.list_format):
                    if len(val) > 2:
                        if val[2] == '0':
                            ignore_cols.append(idx)

                tag_lists = mmf.convert_lists_to_numbers(
                    tag_lists, ignore_columns=ignore_cols)

                df = {}
                for c, attr in enumerate(pattern.list_tags):
                    col = [row[c] for row in tag_lists]
                    df[attr] = col
                df = pd.DataFrame(df)

                deci_mark = self.main.current_paramset.output.decimal_mark
                if output_type == 'csv':
                    fname = QFileDialog.getSaveFileName(
                        self, 'Save data as',
                        filter="CSV file (*.csv)")
                    if fname[0] != '':
                        sep = ',' if deci_mark == '.' else ';'
                        try:
                            df.to_csv(fname[0], sep=sep,
                                      decimal=deci_mark, index=False)
                        except IOError as err:
                            QMessageBox.warning(self.main, 'Failed saving', err)
                elif output_type == 'clipboard':
                    df.to_clipboard(excel=True, decimal=deci_mark, index=False)
                    QMessageBox.information(self.main, 'Data in clipboard',
                                            'Data copied to clipboard.')

    def dump_dicom(self):
        """Dump dicom elements for active file to text."""
        proceed = True
        if self.main.gui.active_img_no < 0:
            QMessageBox.information(self, 'Missing input',
                                    'No file selected.')
            proceed = False
        if proceed:
            dcm.dump_dicom(self,
                           filename=self.main.imgs[
                               self.main.gui.active_img_no].filepath)

    def refresh_img_info(self, info_list_general, info_list_modality):
        """Refresh dicom header information for selected image."""
        self.list_info_general.clear()
        if len(info_list_general) > 0:
            for attr in info_list_general:
                QTreeWidgetItem(self.list_info_general, [attr])
        self.list_info_modality.clear()
        if len(info_list_modality) > 0:
            for attr in info_list_modality:
                QTreeWidgetItem(self.list_info_modality, [attr])


class CenterWidget(QGroupBox):
    """Widget with groupbox holding center/rotation display."""

    def __init__(self, parent):
        super().__init__('Center / rotation')
        self.main = parent
        self.setFont(uir.FontItalic())

        self.val_delta_x = QSpinBox()
        self.val_delta_y = QSpinBox()
        self.val_delta_a = QDoubleSpinBox()
        self.chk_delta_use = QCheckBox('Use offset')

        self.val_delta_x.setRange(-256, 256)
        self.val_delta_x.valueChanged.connect(self.update_delta)
        self.val_delta_y.setRange(-256, 256)
        self.val_delta_y.valueChanged.connect(self.update_delta)
        self.val_delta_a = QDoubleSpinBox()
        self.val_delta_a.setRange(-180., 180.)
        self.val_delta_a.setDecimals(1)
        self.val_delta_a.valueChanged.connect(self.update_delta)
        self.chk_delta_use.setChecked(True)

        self.val_delta_x.setFixedSize(110, 48)
        self.val_delta_x.setAlignment(Qt.AlignCenter)
        url_left = f'{os.environ[ENV_ICON_PATH]}arrowLeft.png'
        url_right = f'{os.environ[ENV_ICON_PATH]}arrowRight.png'
        css = f"""QSpinBox {{
                margin-left: 0px;
                border: 1px solid gray;
                border-radius: 1px;
                max-width: 20px;
                }}
            QSpinBox::up-button  {{
                subcontrol-origin: margin;
                subcontrol-position: center right;
                image: url({url_right});
                width: 20px;
                right: 1px;
                }}
            QSpinBox::down-button  {{
                subcontrol-origin: margin;
                subcontrol-position: center left;
                image: url({url_left});
                width: 20px;
                left: 1px;
                }}"""
        self.val_delta_x.setStyleSheet(css)

        self.val_delta_y.setFixedSize(64, 96)
        self.val_delta_y.setAlignment(Qt.AlignCenter)
        url_up = f'{os.environ[ENV_ICON_PATH]}arrowUp.png'
        url_down = f'{os.environ[ENV_ICON_PATH]}arrowDown.png'
        css = f"""QSpinBox {{
                border: 1px solid gray;
                border-radius: 3px;
                max-height: 20px;
                padding-top: 24px;
                padding-bottom: 24px;
                }}
            QSpinBox::up-button  {{
                subcontrol-origin: padding;
                subcontrol-position: center top;
                width: 30px;
                bottom: 5px;
                image: url({url_up});
                }}
            QSpinBox::down-button  {{
                subcontrol-origin: padding;
                subcontrol-position: center bottom;
                width: 30px;
                top: 5px;
                image: url({url_down});
                }}"""
        self.val_delta_y.setStyleSheet(css)

        toolb = QToolBar()
        act_search = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}search.png'),
            'Search geometric center of mass by threshold', self)
        act_search.triggered.connect(self.set_center_threshold)
        act_select = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}selectArrow.png'),
            'Set center to last mouse click position', self)
        act_select.triggered.connect(self.set_center_to_clickpos)
        act_reset = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}resetZero.png'),
            'Reset center and rotation', self)
        act_reset.triggered.connect(self.reset_delta)
        toolb.addActions([act_reset, act_select, act_search])

        hlo_delta = QHBoxLayout()
        hlo_delta.addWidget(QLabel('dx'))
        hlo_delta.addWidget(self.val_delta_x)
        hlo_delta.addSpacing(20)
        hlo_delta.addWidget(QLabel('dy'))
        hlo_delta.addWidget(self.val_delta_y)
        hlo_delta2 = QHBoxLayout()
        hlo_delta2.addWidget(QLabel('da'))
        hlo_delta2.addWidget(self.val_delta_a)
        hlo_delta2.addWidget(self.chk_delta_use)
        hlo_delta2.addStretch()

        vlo_center = QVBoxLayout()
        vlo_center.addLayout(hlo_delta)
        vlo_center.addLayout(hlo_delta2)
        vlo_center.addWidget(toolb)
        self.setLayout(vlo_center)

    def set_center_to_clickpos(self):
        """Set delta xy to last clicked position."""
        if self.main.active_img is not None:
            sz_acty, sz_actx = np.shape(self.main.active_img)
            self.val_delta_x.setValue(
                self.main.gui.last_clicked_pos[0] - 0.5 * sz_actx)
            self.val_delta_y.setValue(
                - (self.main.gui.last_clicked_pos[1] - 0.5 * sz_acty))
            self.update_delta()

    def set_center_threshold(self):
        """Set center position on geometric center of mass + thresholding."""
        if self.main.active_img is not None:
            num, proceed = QInputDialog.getInt(self, "Search center based on threshold",
                                               "Set threshold", value=0)
            if proceed:
                masked_img = np.where(self.main.active_img > num, 1, 0)
                if np.amax(masked_img) > 0:

                    center = [np.average(indices)
                              for indices in np.where(
                                      self.main.active_img > num)]
                    sz_act = np.shape(self.main.active_img)
                    self.val_delta_x.setValue(center[1] - sz_act[1]*0.5)
                    self.val_delta_y.setValue(center[0] - sz_act[0]*0.5)
                    self.update_delta()
        else:
            QMessageBox.information(self, 'Information',
                                    'No image loaded to threshold.')

    def update_delta(self):
        """Update delta x,y,a - make sure valid values."""
        self.main.gui.delta_x = self.val_delta_x.value()
        self.main.gui.delta_y = - self.val_delta_y.value()
        self.main.gui.delta_a = self.val_delta_a.value()

        if self.main.gui.annotations and self.main.active_img is not None:
            szy, szx = np.shape(self.main.active_img)
            if self.main.gui.delta_a == 0:
                self.main.wid_image_display.canvas.ax.lines[0].set_ydata(
                    y=szy * 0.5 + self.main.gui.delta_y)
                self.main.wid_image_display.canvas.ax.lines[1].set_xdata(
                    x=szx * 0.5 + self.main.gui.delta_x)
            else:
                x1, x2, y1, y2 = get_rotated_crosshair(
                    szx, szy,
                    (self.main.gui.delta_x, self.main.gui.delta_y,
                     self.main.gui.delta_a))
                self.main.wid_image_display.canvas.ax.lines[0].set_ydata(
                    [y1, y2])
                self.main.wid_image_display.canvas.ax.lines[1].set_xdata(
                    [x1, x2])

            self.main.wid_image_display.canvas.draw()
            self.main.update_roi()
            self.main.reset_results()

    def reset_delta(self):
        """Reset center displacement and rotation."""
        self.val_delta_x.setValue(0)
        self.val_delta_y.setValue(0)
        self.val_delta_a.setValue(0)
        self.update_delta()


class WindowLevelEditDialog(ImageQCDialog):
    """Dialog to set window level by numbers."""

    def __init__(self, min_max=[0, 0]):
        super().__init__()

        self.setWindowTitle('Edit window level')
        self.setMinimumHeight(400)
        self.setMinimumWidth(300)

        vLO = QVBoxLayout()
        self.setLayout(vLO)
        fLO = QFormLayout()
        vLO.addLayout(fLO)

        self.spin_min = QSpinBox()
        self.spin_min.setRange(-1000000, 1000000)
        self.spin_min.setValue(min_max[0])
        self.spin_min.editingFinished.connect(
            lambda: self.recalculate_others(sender='min'))
        fLO.addRow(QLabel('Minimum'), self.spin_min)

        self.spin_max = QSpinBox()
        self.spin_max.setRange(-1000000, 1000000)
        self.spin_max.setValue(min_max[1])
        self.spin_max.editingFinished.connect(
            lambda: self.recalculate_others(sender='max'))
        fLO.addRow(QLabel('Maximum'), self.spin_max)

        self.spin_center = QSpinBox()
        self.spin_center.setRange(-1000000, 1000000)
        self.spin_center.setValue(0.5*(min_max[0] + min_max[1]))
        self.spin_center.editingFinished.connect(
            lambda: self.recalculate_others(sender='center'))
        fLO.addRow(QLabel('Center'), self.spin_center)

        self.spin_width = QSpinBox()
        self.spin_width.setRange(0, 2000000)
        self.spin_width.setValue(min_max[1] - min_max[0])
        self.spin_width.editingFinished.connect(
            lambda: self.recalculate_others(sender='width'))
        fLO.addRow(QLabel('Width'), self.spin_width)

        self.chk_lock = QCheckBox('')
        self.chk_lock.setChecked(True)
        fLO.addRow(QLabel('Lock WL for all images'), self.chk_lock)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        vLO.addWidget(self.button_box)

    def accept(self):
        """Aoid close on enter if not ok button focus."""
        if self.button_box.button(QDialogButtonBox.Ok).hasFocus():
            if self.spin_width.value() == 0:
                QMessageBox.warning(
                    self, 'Warning',
                    'Window width should be larger than zero.')
            elif self.spin_min.value() >= self.spin_max.value():
                QMessageBox.warning(
                    self, 'Warning',
                    'Window max should be set larger than minimum.')
            else:
                super().accept()

    def recalculate_others(self, sender='min'):
        """Reset others based on input."""
        self.blockSignals(True)
        minval = self.spin_min.value()
        maxval = self.spin_max.value()
        width = self.spin_width.value()
        center = self.spin_center.value()
        if sender in ['min', 'max']:
            self.spin_center.setValue(round(0.5*(minval + maxval)))
            self.spin_width.setValue(maxval-minval)
        else:  # sender in ['center', 'width']:
            self.spin_min.setValue(center - round(0.5*width))
            self.spin_max.setValue(center + round(0.5*width))
        self.blockSignals(False)

    def get_min_max_lock(self):
        """Get min max values an lock setting as tuple."""
        return (
            self.spin_min.value(),
            self.spin_max.value(),
            self.chk_lock.isChecked()
            )


class WindowLevelWidget(QGroupBox):
    """Widget with groupbox holding WindowLevel display."""

    def __init__(self, parent):
        super().__init__('Window Level')
        self.main = parent
        self.setFont(uir.FontItalic())

        self.min_wl = QSlider(Qt.Horizontal)
        self.max_wl = QSlider(Qt.Horizontal)
        self.lbl_min_wl = QLabel('-200')
        self.lbl_max_wl = QLabel('200')
        self.canvas = WindowLevelHistoCanvas()

        self.max_wl.setRange(-200, 200)
        self.max_wl.setValue(200)
        self.min_wl.setRange(-200, 200)
        self.min_wl.setValue(-200)
        self.min_wl.sliderReleased.connect(self.correct_window_level_sliders)
        self.max_wl.sliderReleased.connect(self.correct_window_level_sliders)
        self.lbl_center = QLabel('0')
        self.lbl_width = QLabel('400')

        vlo_wl = QVBoxLayout()

        tb_wl = QToolBar()
        self.tool_min_max_wl = QToolButton()
        self.tool_min_max_wl.setToolTip("Set WL to [min,max] of active image")
        self.tool_min_max_wl.setIcon(
            QIcon(f'{os.environ[ENV_ICON_PATH]}minmax.png'))
        self.tool_min_max_wl.clicked.connect(
            lambda: self.clicked_window_level('min_max'))
        self.tool_range_wl = QToolButton()
        self.tool_range_wl.setToolTip(
            "Set WL to [mean-std,mean+std] of active image")
        self.tool_range_wl.setIcon(
            QIcon(f'{os.environ[ENV_ICON_PATH]}range.png'))
        self.tool_range_wl.clicked.connect(
            lambda: self.clicked_window_level('mean_stdev'))
        self.tool_dcm_wl = QToolButton()
        self.tool_dcm_wl.setToolTip(
            "Set WL as defined in the DICOM header of active image")
        self.tool_dcm_wl.setIcon(
            QIcon(f'{os.environ[ENV_ICON_PATH]}fileDCM.png'))
        self.tool_dcm_wl.clicked.connect(
            lambda: self.clicked_window_level('dcm'))
        self.tool_edit_wl = QToolButton()
        self.tool_edit_wl.setToolTip("Edit WL by numbers")
        self.tool_edit_wl.setIcon(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'))
        self.tool_edit_wl.clicked.connect(self.set_window_level_by_numbers)
        tb_wl.addWidget(self.tool_min_max_wl)
        tb_wl.addWidget(self.tool_range_wl)
        tb_wl.addWidget(self.tool_dcm_wl)
        tb_wl.addWidget(self.tool_edit_wl)
        self.chk_wl_update = QCheckBox('Lock WL')
        self.chk_wl_update.toggled.connect(self.update_window_level_mode)
        tb_wl.addWidget(self.chk_wl_update)

        hlo_slider = QHBoxLayout()
        vlo_min = QVBoxLayout()
        vlo_min.addSpacing(20)
        vlo_min.addWidget(self.lbl_min_wl)
        vlo_min.addStretch()
        hlo_slider.addLayout(vlo_min)
        vlo_slider = QVBoxLayout()
        vlo_slider.addWidget(self.min_wl)
        vlo_slider.addWidget(self.max_wl)
        vlo_slider.addWidget(self.canvas)
        hlo_slider.addLayout(vlo_slider)
        vlo_max = QVBoxLayout()
        vlo_max.addSpacing(20)
        vlo_max.addWidget(self.lbl_max_wl)
        vlo_max.addStretch()
        hlo_slider.addLayout(vlo_max)
        vlo_wl.addWidget(tb_wl)
        vlo_wl.addLayout(hlo_slider)
        hbox_cw = QHBoxLayout()
        hbox_cw.addStretch()
        hbox_cw.addWidget(QLabel('C: '))
        hbox_cw.addWidget(self.lbl_center)
        hbox_cw.addSpacing(20)
        hbox_cw.addWidget(QLabel('W: '))
        hbox_cw.addWidget(self.lbl_width)
        hbox_cw.addStretch()
        vlo_wl.addLayout(hbox_cw)

        self.setLayout(vlo_wl)

        self.update_window_level_mode()

    def update_window_level_mode(self):
        """Set and unset lock on window level when selecting a new image."""
        if self.chk_wl_update.isChecked():
            self.tool_min_max_wl.setCheckable(False)
            self.tool_range_wl.setCheckable(False)
            self.tool_dcm_wl.setCheckable(False)
        else:
            self.tool_min_max_wl.setCheckable(True)
            self.tool_range_wl.setCheckable(True)
            self.tool_dcm_wl.setCheckable(True)
            # default
            self.tool_range_wl.setChecked(True)
            self.set_window_level('mean_stdev')

    def get_min_max(self):
        """Get lower and upper window level based on image.

        Returns
        -------
        min_wl : int
            lower window level
        max_wl : TYPE
            upper window level
        """
        if self.chk_wl_update.isChecked() is False:
            if self.tool_min_max_wl.isChecked():
                self.set_window_level('min_max')
            elif self.tool_range_wl.isChecked():
                self.set_window_level('mean_stdev')
            else:
                self.set_window_level('dcm')

        return (self.min_wl.value(), self.max_wl.value())

    def clicked_window_level(self, arg):
        """When one of the window level toolbuttons is toggled."""
        if self.chk_wl_update.isChecked() is False:
            # unCheck others, check selected
            if arg == 'min_max':
                self.tool_min_max_wl.setChecked(True)
                self.tool_range_wl.setChecked(False)
                self.tool_dcm_wl.setChecked(False)
            elif arg == 'mean_stdev':
                self.tool_min_max_wl.setChecked(False)
                self.tool_range_wl.setChecked(True)
                self.tool_dcm_wl.setChecked(False)
            elif arg == 'dcm':
                self.tool_min_max_wl.setChecked(False)
                self.tool_range_wl.setChecked(False)
                self.tool_dcm_wl.setChecked(True)

        self.set_window_level(arg)

    def set_window_level_by_numbers(self):
        """Dialog box to set min/max or center/width and option to lock."""
        dlg = WindowLevelEditDialog(min_max=[self.min_wl.value(), self.max_wl.value()])
        res = dlg.exec()
        if res:
            minval, maxval, lock = dlg.get_min_max_lock()
            self.update_window_level(minval, maxval)
            if self.main.active_img is not None:
                self.main.wid_image_display.canvas.img.set_clim(
                    vmin=minval, vmax=maxval)
                self.main.wid_image_display.canvas.draw()
            self.chk_wl_update.setChecked(lock)
            self.update_window_level_mode()

    def set_window_level(self, arg):
        """Set window level based on active image conte t."""
        if self.main.active_img is not None:
            minval = 0
            maxval = 0
            if arg == 'dcm':
                imgno = self.main.gui.active_img_no
                if self.main.imgs[imgno].window_width > 0:
                    minval = self.main.imgs[imgno].window_center - \
                        0.5*self.main.imgs[imgno].window_width
                    maxval = self.main.imgs[imgno].window_center + \
                        0.5*self.main.imgs[imgno].window_width
            elif arg == 'min_max':
                minval = np.amin(self.main.active_img)
                maxval = np.amax(self.main.active_img)
            elif arg == 'mean_stdev':
                meanval = np.mean(self.main.active_img)
                stdval = np.std(self.main.active_img)
                minval = meanval-stdval
                maxval = meanval+stdval

            if maxval == minval:
                minval = np.amin(self.main.active_img)
                maxval = np.amax(self.main.active_img)

            minval = np.round(minval)
            maxval = np.round(maxval)

            self.update_window_level(minval, maxval)
            self.main.wid_image_display.canvas.img.set_clim(vmin=minval, vmax=maxval)
            self.main.wid_image_display.canvas.draw()

    def update_window_level(self, minval, maxval):
        """Update GUI for window level sliders."""
        self.min_wl.setValue(round(minval))
        self.max_wl.setValue(round(maxval))
        self.lbl_min_wl.setText(f'{minval:.0f}')
        self.lbl_max_wl.setText(f'{maxval:.0f}')
        self.lbl_center.setText(f'{0.5*(minval+maxval):.0f}')
        self.lbl_width.setText(f'{(maxval-minval):.0f}')

    def correct_window_level_sliders(self):
        """Make sure min_wl < max_wl after user input."""
        if self.max_wl.value() < self.min_wl.value():
            maxval = self.min_wl.value()
            self.update_window_level(self.max_wl.value(), maxval)
        else:
            self.update_window_level(self.min_wl.value(), self.max_wl.value())

        if self.main.active_img is not None:
            self.main.wid_image_display.canvas.img.set_clim(
                vmin=self.min_wl.value(), vmax=self.max_wl.value())
            self.main.wid_image_display.canvas.draw()


class WindowLevelHistoCanvas(FigureCanvasQTAgg):
    """Canvas for display of histogram for the active image."""

    def __init__(self):
        self.fig = matplotlib.figure.Figure(figsize=(2, 1))
        self.fig.subplots_adjust(0., 0., 1., 1.)
        FigureCanvasQTAgg.__init__(self, self.fig)

    def plot(self, nparr):
        """Refresh histogram."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        amin, amax = (np.amin(nparr), np.amax(nparr))
        try:
            hist, bins = np.histogram(nparr, bins=np.arange(
                amin, amax, (amax - amin)/100.))
            self.ax.plot(bins[:-1], hist)
            self.ax.axis('off')
            at_min = matplotlib.offsetbox.AnchoredText(
                f'Min:\n {amin:.0f}',
                prop=dict(size=12), frameon=False, loc='upper left')
            at_max = matplotlib.offsetbox.AnchoredText(
                f'Max:\n {amax:.0f}',
                prop=dict(size=12), frameon=False, loc='upper right')
            self.ax.add_artist(at_min)
            self.ax.add_artist(at_max)

            self.draw()
        except ValueError:
            pass