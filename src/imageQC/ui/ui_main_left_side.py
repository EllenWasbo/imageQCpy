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
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QMenu, QAction, QToolBar,
    QMessageBox, QInputDialog, QTreeWidget, QTreeWidgetItem, QFileDialog,
    QDialogButtonBox
    )

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
        self.setColumnCount(5)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setHeaderLabels(['Idx', 'Image', 'Frame', 'Test', 'Image name', 'Group name'])
        self.setColumnHidden(4, True)  # image name
        self.setColumnHidden(5, True)  # group name
        self.setColumnWidth(0, 50)
        self.setColumnWidth(1, round(0.55*self.main.gui.panel_width))
        self.setColumnWidth(2, 60)
        self.setColumnWidth(3, round(0.2*self.main.gui.panel_width))
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

    def update_file_list(self, set_selected=None, keep_active=False):
        """Populate tree with filepath/pattern and test indicators.

        Parameters
        ----------
        set_selected : list of int, optional
            Set selected image numbers. The default is None.
        keep_active : bool, optional
            Set True to avoid re-reading image data and other cascade actions.
            The default is False.
        """
        self.blockSignals(True)
        self.clear()
        self.blockSignals(False)
        quicktest_active = self.main.wid_quicktest.gb_quicktest.isChecked()
        self.setColumnHidden(4, not quicktest_active)  # image name
        self.setColumnHidden(5, not quicktest_active)  # group name
        if len(self.main.imgs) == 0:
            QTreeWidgetItem(self, [''] * 6)
        else:
            for idx, img in enumerate(self.main.imgs):
                image_name = ''
                group_name = ''
                if quicktest_active:
                    test_string = '+'.join(img.marked_quicktest)
                    image_name = img.quicktest_image_name
                    group_name = img.quicktest_group_name
                else:
                    test_string = 'x' if img.marked else ''
                frameno = f'{img.frame_number}' if img.frame_number > -1 else ''

                if img.filepath == '':
                    file_text = ' -- dummy -- '
                else:
                    if self.main.cbox_file_list_display.currentIndex() == 0:
                        file_text = img.filepath
                    else:
                        file_text = ' '.join(img.file_list_strings)
                QTreeWidgetItem(self, [str(idx), file_text, frameno, test_string,
                                       image_name, group_name])
            self.main.lbl_n_loaded.setText(str(len(self.main.imgs)))
            if keep_active is False:
                try:
                    self.main.gui.active_img_no = set_selected[-1]
                except (TypeError, IndexError):
                    pass

            if keep_active:  # avoid reading image from file once again
                self.blockSignals(True)
            self.setCurrentItem(self.topLevelItem(self.main.gui.active_img_no))
            if keep_active:
                self.blockSignals(False)

            if isinstance(set_selected, list):
                # force selected
                self.blockSignals(True)
                for i in range(len(self.main.imgs)):
                    if i in set_selected:
                        self.topLevelItem(i).setSelected(True)
                    else:
                        self.topLevelItem(i).setSelected(False)
                self.blockSignals(False)

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
                numbers = [i for i in range(len(self.main.imgs))]
                selrows.sort(reverse=True)
                for row in selrows:
                    del self.main.imgs[row]
                    del numbers[row]
                diff = np.array(numbers) - last_selected
                idx = np.where(diff > 0)
                try:
                    new_active = idx[0][0]
                except IndexError:
                    new_active = 0
                # delete results and roi if both 3d
                if self.main.current_test == 'Rec':
                    del self.main.results['Rec']
                    self.main.current_roi = None

                self.main.set_active_img(new_active)
                self.main.update_results(deleted_idxs=selrows)
                if self.main.summed_img is not None:
                    self.main.reset_summed_img()
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
                    self.main.imgs[sel].marked_quicktest = test_codes
                    self.main.wid_quicktest.flag_edit(True)
                elif remove_mark:
                    self.main.imgs[sel].marked = False
                else:
                    self.main.imgs[sel].marked = True

            if self.main.summed_img is not None:
                self.main.reset_summed_img()
            self.update_file_list(set_selected=selrows, keep_active=True)
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
        selrows, _ = self.get_selected_imgs()
        if len(selrows) > 0:
            if direction in ['top', 'down', 'up', 'bottom', 'to']:
                selrows_rev = sorted(selrows, reverse=True)
                selrows_ord = sorted(selrows)
                n_imgs = len(self.main.imgs)
                n_moved = len(selrows)
                popped_imgs = []
                order = [i for i in range(n_imgs)]
                set_selected = None
                for i in selrows_rev:
                    popped_imgs.insert(0, self.main.imgs.pop(i))
                    order.pop(i)
                if direction == 'to':
                    if len(selrows) == 1:
                        msg = 'Move selected image to index:'
                    else:
                        msg = 'Move selected images to index:'
                    num, proceed = QInputDialog.getInt(
                        self, "Move image(s) to...",
                        msg, value=0, min=0, max=len(self.main.imgs))
                    if proceed:
                        for i, selidx in enumerate(selrows_ord):
                            self.main.imgs.insert(num + i, popped_imgs[i])
                            order.insert(num + i, selidx)
                        set_selected = list(range(num, num + len(selrows), 1))
                elif direction == 'bottom':
                    self.main.imgs.extend(popped_imgs)
                    set_selected = [i for i in range(n_imgs-n_moved, n_imgs)]
                    order = order + selrows_ord
                elif direction == 'top':
                    self.main.imgs = popped_imgs + self.main.imgs
                    set_selected = [i for i in range(0, n_moved)]
                    order = selrows_ord + order
                else:
                    addidx = 1 if direction == 'down' else -1
                    for i, selidx in enumerate(selrows_ord):
                        self.main.imgs.insert(selidx + addidx, popped_imgs[i])
                        order.insert(selidx + addidx, selidx)
                    set_selected = list(np.array(selrows) + addidx)
                if set_selected is not None:
                    self.update_file_list(set_selected=set_selected)
                    if self.main.results:
                        self.main.update_results(sort_idxs=order)
                        self.main.refresh_selected_table_row()

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
                round(self.main.gui.last_clicked_pos[0] - 0.5 * sz_actx))
            self.val_delta_y.setValue(
                - round(self.main.gui.last_clicked_pos[1] - 0.5 * sz_acty))
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
                    y=[szy * 0.5 + self.main.gui.delta_y])
                self.main.wid_image_display.canvas.ax.lines[1].set_xdata(
                    x=[szx * 0.5 + self.main.gui.delta_x])
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
