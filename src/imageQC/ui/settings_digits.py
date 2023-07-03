#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User interface for configuration settings - text recognition.

@author: Ellen Wasbo
"""
import os
import copy
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QLabel, QLineEdit, QPushButton, QAction,
    QMessageBox, QDialogButtonBox, QFileDialog
    )
import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

# imageQC block start
from imageQC.config.iQCconstants import ENV_ICON_PATH
from imageQC.config import config_classes as cfc
from imageQC.ui.settings_reusables import StackWidget
from imageQC.ui import reusable_widgets as uir
from imageQC.ui.ui_dialogs import ImageQCDialog
from imageQC.scripts import dcm
from imageQC.scripts import digit_methods
# imageQC block end


class DigitWidget(StackWidget):
    """Widget holding settings for text recognition."""

    def __init__(
            self, dlg_settings=None, initial_modality='CT', editable=True):
        header = 'Identify text (digits) from images'
        subtxt = (
            'Some vendors have displays with text as numbers of interest.<br>'
            'imageQC provide the option to read numbers from ROIs by comparing '
            'the content in the ROIs to digit templates.<br>'
            'Some digit templates are provided by default '
            '(e.g. NM - Siemens savescreens)and you may add your own.'
            'Templates can be duplicated or moved between modalities.'
            )
        super().__init__(dlg_settings, header, subtxt,
                         typestr='template',
                         mod_temp=True, grouped=True, editable=editable
                         )
        self.fname = 'digit_templates'

        self.empty_template = cfc.DigitTemplate()
        self.current_template = self.empty_template
        self.current_modality = initial_modality

        self.wid_temp0 = QWidget(self)
        self.hlo.addWidget(self.wid_temp0)
        self.vlo_temp = QVBoxLayout()
        self.wid_temp = DigitTemplateWidget(self)
        self.vlo_temp.addWidget(self.wid_temp)
        self.wid_temp0.setLayout(self.vlo_temp)

        if not self.import_review_mode:
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_clear)
            self.wid_mod_temp.toolbar.removeAction(self.wid_mod_temp.act_save)
            act_edit = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
                'Edit template', self)
            act_edit.triggered.connect(self.edit_template)
            self.wid_mod_temp.toolbar.addAction(act_edit)

        if editable:
            self.vlo.addWidget(uir.HLine())
            self.vlo.addWidget(self.status_label)

    def update_data(self):
        """Update GUI with the selected template."""
        self.wid_temp.canvas.draw_template()
        self.flag_edit(False)

    def edit_template(self):
        """Start dialog to edit template."""
        sel = self.wid_mod_temp.list_temps.currentItem()
        if sel is not None:
            current_text = sel.text()
            if current_text != '':
                dlg = DigitTemplateEditDialog(
                    template_input=self.current_template,
                    tag_infos=self.dlg_settings.main.tag_infos)
                res = dlg.exec()
                if res:
                    self.current_template = dlg.get_template()
                    for i, img in enumerate(self.current_template.images):
                        if isinstance(img, np.ndarray):  # to list to save to yaml
                            self.current_template.images[i] = img.tolist()
                    idx = self.current_labels.index(self.current_template.label)
                    self.templates[self.current_modality][idx] = \
                        copy.deepcopy(self.current_template)
                    self.save()
                    self.update_data()
            else:
                QMessageBox.information(
                    self, 'Add first',
                    'No template selected. You might have to add one first.')
        else:
            QMessageBox.information(
                self, 'Missing selection',
                'No template selected. You might have to add one first.')


class DigitTemplateWidget(QWidget):
    """Widget for display current template."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.canvas = DigitTemplateCanvas(self.parent)
        self.hlo = QHBoxLayout()
        self.hlo.addWidget(self.canvas)
        self.setLayout(self.hlo)


class DigitTemplateCanvas(FigureCanvasQTAgg):
    """Canvas for display of digit-images."""

    def __init__(self, parent):
        self.parent = parent  # the StackWidget or edit dialog
        self.fig = matplotlib.figure.Figure(figsize=(.1, 3.5), dpi=150)
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.digit_names = [str(i) for i in range(10)] + ['.dot', '-neg']

        self.draw_template()

    def draw_template(self):
        """Draw the digits from current_template."""
        self.fig.clf()
        images = self.parent.current_template.images
        if len(images) == 0:
            images = [None] * len(self.digit_names)
        for i, image in enumerate(images):
            ax = self.fig.add_subplot(1, len(self.digit_names), i+1)
            ax.set(title=self.digit_names[i])
            ax.set_xticks(())
            ax.set_yticks(())
            if images[i] is not None:
                if isinstance(images[i], list):
                    images[i] = np.array(images[i])
                ax.imshow(images[i], cmap='gray')
            else:
                ax.imshow(np.zeros((2, 2)))
        self.draw()


class DigitTemplateEditDialog(ImageQCDialog):
    """Dialog to add or edit digit templates."""

    def __init__(self, template_input=cfc.DigitTemplate(), tag_infos=None):
        super().__init__()
        self.setWindowTitle('Add/edit digit template')

        self.current_template = copy.deepcopy(template_input)
        self.tag_infos = tag_infos
        self.wid_temp = DigitTemplateWidget(self)
        self.sample_filepath = QLabel()
        self.canvas = TextImageCanvas(self)
        self.txt_digits_in_zoom = QLineEdit()
        self.current_image = None
        self.chars = [str(i) for i in range(10)] + ['.', '-']

        vlo = QVBoxLayout()
        self.setLayout(vlo)

        hlo_top = QHBoxLayout()
        vlo.addLayout(hlo_top)
        toolb = QToolBar()
        act_open = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}open.png'),
            'Open DICOM file with text', self)
        act_open.triggered.connect(self.open_file)
        act_dump = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}tags.png'),
            'Display DICOM header of open file', self)
        act_dump.triggered.connect(self.dump_dicom)
        toolb.addActions([act_open, act_dump])
        hlo_top.addWidget(toolb)
        vlo.addWidget(self.sample_filepath)
        tb_img = TextImageNavigationToolbar(self.canvas, self)
        vlo.addWidget(tb_img)

        hlo_img = QHBoxLayout()
        hlo_img.addWidget(self.canvas)
        vlo.addLayout(hlo_img)
        vlo_actions = QVBoxLayout()
        hlo_img.addLayout(vlo_actions)
        vlo_actions.addWidget(QLabel('Digits in zoomed image part:'))
        vlo_actions.addWidget(self.txt_digits_in_zoom)
        btn_add_new = QPushButton('Add digits to template')
        btn_add_new.clicked.connect(self.add_new)
        vlo_actions.addWidget(btn_add_new)
        btn_test_current = QPushButton('Test read zoomed part')
        btn_test_current.clicked.connect(self.test_current)
        vlo_actions.addWidget(btn_test_current)
        vlo_actions.addStretch()

        vlo.addWidget(self.wid_temp)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vlo.addWidget(buttons)

    def get_template(self):
        """Return current template to StackWidget."""
        return self.current_template

    def get_subimg(self):
        """Return currently zoomed image part as numpy array."""
        xs = np.sort(self.canvas.ax.get_xlim())
        ys = np.sort(self.canvas.ax.get_ylim())
        sub_img = self.current_image[
            int(ys[0])+1:int(ys[1])+1, int(xs[0])+1:int(xs[1])+1]

        return sub_img

    def add_new(self):
        """Add new characters to template based on zoomed image and input string."""
        char_strings = self.txt_digits_in_zoom.text()
        sub_img = self.get_subimg()
        if sub_img.shape == self.current_image.shape:
            QMessageBox.warning(
                self, 'Zoom to number',
                'Please zoom to the part of the image with a number and type '
                'the number there in the text field.')
        elif len(char_strings) == 0:
            QMessageBox.warning(
                self, 'Set the number',
                'The text string is empty. Please zoom to a number in the image '
                'and type the number there in the text field.')
        else:
            char_imgs, chop_idxs = digit_methods.extract_char_blocks(sub_img)
            if len(char_imgs) == 0:
                QMessageBox.warning(
                    self, 'Failed finding character',
                    'Failed finding characters in the zoomed image. '
                    'Clean enough background? A little margin around?')
            else:
                self.show_chopping(chop_idxs)
                if len(char_imgs) != len(char_strings):
                    QMessageBox.warning(
                        self, 'Mismatch',
                        f'Number of characters in input text {len(char_strings)} do '
                        'not match number of characters found in zoomed image '
                        f'{len(char_imgs)}.')
                else:
                    errchars = []
                    for no, char in enumerate(char_strings):
                        if char in self.chars:
                            idx = self.chars.index(char)
                            if len(self.current_template.images) == 0:
                                self.current_template.images = [
                                    None for i in range(len(self.chars))]
                            self.current_template.images[idx] = char_imgs[no]
                        else:
                            errchars.append(char)
                    self.wid_temp.canvas.draw_template()
                    if len(errchars) > 0:
                        QMessageBox.warning(
                            self, 'Characters not allowed',
                            f'Characters {errchars} not accepted. Digits, decimal mark '
                            '(.) or minus (-) only.')

    def test_current(self):
        """Test template on zoomed image."""
        sub_img = self.get_subimg()
        if sub_img.shape == self.current_image.shape:
            QMessageBox.warning(
                self, 'Zoom to number',
                'Please zoom to the part of the image with a number.')
        #elif not self.current_template.images[:-1].all():
        #    QMessageBox.warning(
        #        self, 'Missing digits',
        #        'Please add template image to all digits before testing.')
        else:
            char_imgs, chop_idxs = digit_methods.extract_char_blocks(sub_img)
            if len(char_imgs) == 0:
                QMessageBox.warning(
                    self, 'Failed finding digit(s)',
                    'Failed finding digits in the zoomed image. '
                    'Clean enough background? A little margin around?')
            else:
                self.show_chopping(chop_idxs)
                digit = digit_methods.compare_char_blocks_2_template(
                    char_imgs, self.current_template)
                if digit is None:
                    QMessageBox.warning(
                        self, 'No match',
                        'Found no match between the blocks and the template.')
                else:
                    QMessageBox.information(
                        self, 'Found digit(s)',
                        f'Found number in zoomed part: {digit}')

    def show_chopping(self, idxs):
        """Show chopping lines for the characters."""
        xs = np.sort(self.canvas.ax.get_xlim())
        self.canvas.clear_annotations()
        for x_start_stop in idxs:
            self.canvas.ax.axvline(
                x=x_start_stop[0]+int(xs[0]), color='green', linewidth=1)
            self.canvas.ax.axvline(
                x=x_start_stop[1]+int(xs[0])+1, color='blue', linewidth=1)
        self.canvas.draw()

    def open_file(self):
        """Locate sample DICOM file."""
        fname = QFileDialog.getOpenFileName(
            self, 'Read DICOM file',
            filter="DICOM file (*.dcm);;All files (*)")
        if fname[0] != '':
            self.sample_filepath.setText(fname[0])
            self.current_image, _ = dcm.get_img(fname[0], tag_infos=self.tag_infos)
            self.canvas.img_draw()

    def dump_dicom(self):
        """Dump dicom elements for file to text."""
        proceed = True
        if self.sample_filepath.text() == '':
            QMessageBox.information(self, 'Missing input', 'Open a DICOM file first.')
            proceed = False
        if proceed:
            dcm.dump_dicom(self, filename=self.sample_filepath.text())


class TextImageCanvas(FigureCanvasQTAgg):
    """Canvas for display of image."""

    def __init__(self, parent):
        self.parent = parent
        self.fig = matplotlib.figure.Figure(dpi=150)
        self.fig.subplots_adjust(0., 0., 1., 1.)
        FigureCanvasQTAgg.__init__(self, self.fig)

        self.ax = self.fig.add_subplot(111)
        self.current_image = None

        # default display
        self.ax.cla()
        self.ax.axis('off')

    def img_draw(self):
        """Refresh image."""
        self.ax.cla()
        nparr = self.parent.current_image
        if nparr is not None:
            self.img = self.ax.imshow(nparr, cmap='gray')
            self.ax.axis('off')
        self.draw()

    def clear_annotations(self):
        """Remove the chopping lines."""
        if hasattr(self.ax, 'lines'):
            n_lines = len(self.ax.lines)
            for i in range(n_lines - 2):
                self.ax.lines[-1].remove()


class TextImageNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib navigation toolbar with some modifications."""

    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        for x in self.actions():
            if x.text() in ['Subplots', 'Save']:
                self.removeAction(x)
