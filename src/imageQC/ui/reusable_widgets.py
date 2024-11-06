#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface classes for different uses and reuses in ImageQC.

@author: Ellen Wasbo
"""
import os
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import (
    qApp, QWidget, QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QFrame,
    QToolBar, QAction, QComboBox, QRadioButton, QButtonGroup, QToolButton,
    QLabel, QPushButton, QListWidget, QLineEdit, QCheckBox, QGroupBox, QSlider,
    QProgressDialog, QProgressBar, QStatusBar, QFileDialog, QMessageBox
    )

import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

# imageQC block start
from imageQC.config.iQCconstants import ENV_ICON_PATH
# imageQC block end


class UnderConstruction(QWidget):
    """Under construction widget to display for tests not finished."""

    def __init__(self, txt=''):
        super().__init__()
        hlo = QHBoxLayout()
        self.setLayout(hlo)
        toolb = QToolBar()
        act_warn = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}warning.png'), '', self)
        toolb.addActions([act_warn])
        hlo.addWidget(toolb)
        if txt == '':
            txt = ('Sorry - under construction. Do not trust the results. '
                   'Run test might cause crash or breakpoint.')
        hlo.addWidget(LabelItalic(txt))
        hlo.addStretch()


class LabelItalic(QLabel):
    """Label with preset italic font."""

    def __init__(self, txt, color=None):
        self.color = color
        html_txt = self.convert_to_html(txt)
        super().__init__(html_txt)
        self.setStyleSheet(f'QLabel {{color:{self.color}}}')

    def convert_to_html(self, txt):
        """Add html code to input text."""
        html_txt = f"""<html><head/><body>
        <p><i>{txt}</i></p>
        </body></html>"""
        return html_txt

    def setText(self, txt):
        """Override setText to include formatting."""
        html_txt = self.convert_to_html(txt)
        super().setText(html_txt)


class LabelMultiline(QLabel):
    """Label as multiline."""

    def __init__(self, txts=[]):
        txt = ''
        for this_txt in txts:
            txt = txt + f'<p>{this_txt}</p>'
        html_txt = f"""<html><head/><body>{txt}</body></html>"""
        super().__init__(html_txt)


class LabelHeader(QLabel):
    """Label as header at some level."""

    def __init__(self, txt, level):
        html_txt = f"""<html><head/><body>
            <h{level}><i>{txt}</i></h{level}>
            </body></html>"""
        super().__init__(html_txt)


class FontItalic(QFont):
    """Set italic font."""

    def __init__(self):
        super().__init__()
        self.setItalic(True)


class InfoTool(QToolBar):
    """ToolBar with popup message box with html-formated information."""

    def __init__(self, html_body_text='', parent=None):
        """Initiate.

        Parameters
        ----------
        html_body_text : str
            text between <html><head/><body>   </body></html>
        """
        super().__init__()
        self.parent = parent
        self.html_body_text = html_body_text
        self.btn_info = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}info.png'),
            '''Detailed information about this test''', self)
        self.addActions([self.btn_info])
        self.btn_info.triggered.connect(self.display_info_popup)

    def display_info_popup(self):
        """Popup information."""
        dlg = QDialog(self.parent)
        dlg.setWindowTitle('Information')
        dlg.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        dlg.setWindowFlags(dlg.windowFlags() | Qt.CustomizeWindowHint)
        dlg.setWindowFlags(
            dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        dlg.infotext = QLabel(f"""<html><head/><body>
                {self.html_body_text}
                </body></html>""")
        dlg.infotext.setOpenExternalLinks(True)

        vlo = QVBoxLayout()
        vlo.addWidget(dlg.infotext)
        buttons = QDialogButtonBox.Ok
        dlg.buttonBox = QDialogButtonBox(buttons)
        dlg.buttonBox.accepted.connect(dlg.accept)
        vlo.addWidget(dlg.buttonBox)
        dlg.setLayout(vlo)

        dlg.exec()


class PushButtonWithIcon(QPushButton):
    """Styled PushButton."""

    def __init__(self, text, icon_filename, iconsize=32,
                 align='center', width=-1):
        """Compact definition of Pushbuttons with icon and formatting.

        Parameters
        ----------
        text : str
        icon_filename : str
            file base name in folder icons without extension
        iconsize : int.
            The default is 32.
        align : str, optional
            Text alignment. The default is 'center'.
        size : int, optional
            Size of button. Ignored if -1. Default is -1.
        """
        super().__init__(
            text=f'  {text}',
            icon=QIcon(f'{os.environ[ENV_ICON_PATH]}{icon_filename}.png'))
        self.setStyleSheet(f'text-align:{align};')
        self.setIconSize(QSize(iconsize, iconsize))
        if width > 0:
            self.setFixedWidth(width)


class HLine(QFrame):
    """Class for hline used frequently in the widgets."""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setLineWidth(1)


class VLine(QFrame):
    """Class for vline used frequently in the widgets."""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.VLine)
        self.setLineWidth(1)


class ProgressBar(QProgressBar):
    """Redefine QProgressBar to set style."""

    def __init__(self, parent_widget):
        super().__init__(parent_widget)
        self
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            """
            QProgressBar {
                border-radius: 10px;
                }
            QProgressBar:chunk {
                background-color: #6e94c0;
                border-radius :10px;
                }
            """
            )


class ProgressModal(QProgressDialog):
    """Redefine QProgressDialog to set wanted behaviour."""

    def __init__(self, text, cancel_text, start, stop, parent,
                 minimum_duration=200, hide_cancel=False):
        super().__init__(text, cancel_text, start, stop, parent)
        self.setWindowModality(Qt.WindowModal)
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle('Wait while processing...')
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setMinimumDuration(minimum_duration)
        self.setAutoClose(True)
        if hide_cancel:
            ch = self.findChildren(QPushButton)
            ch[0].hide()
        self.setStyleSheet(
            """
            QProgressBar {
                border-radius: 10px;
                width: 400px;
                }
            QProgressBar:chunk {
                background-color: #6e94c0;
                border-radius :10px;
                }
            """
            )
        self.sub_interval = 0  # used to communicate subprosess range within setRange


class ToolBarBrowse(QToolBar):
    """Toolbar for reuse with search button."""

    def __init__(self, browse_tooltip='', clear=False):
        super().__init__()
        self.act_browse = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}open.png'),
            browse_tooltip, self)
        self.addActions([self.act_browse])
        if clear:
            self.act_clear = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}clear.png'),
                'Clear', self)
            self.addActions([self.act_clear])


class ToolBarEdit(QToolBar):
    """Toolbar for reuse with edit button."""

    def __init__(self, tooltip='',
                 edit_button=True, add_button=False, delete_button=False):
        super().__init__()
        if edit_button:
            self.act_edit = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
                tooltip, self)
            self.addAction(self.act_edit)
        if add_button:
            self.act_add = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                'Add', self)
            self.addAction(self.act_add)
        if delete_button:
            self.act_delete = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
                'Delete', self)
            self.addAction(self.act_delete)


class ToolBarTableExport(QToolBar):
    """Toolbar for reuse with setting table export options."""

    def __init__(self, parent, parameters_output=None, flag_edit=False):
        """Initialize ToolBarTableExport.

        Parameters
        ----------
        parent : obj
            parent having flag_edit function
        parameters_output : obj, optional
            QuickTestOuput template. The default is None.
        flag_edit : bool, optional
            Set option to flag parent as edited. The default is False.
        """
        super().__init__()

        self.setOrientation(Qt.Vertical)
        self.parent = parent
        self.parameters_output = parameters_output
        self.flag_edit = flag_edit

        self.tool_transpose = QToolButton()
        self.tool_transpose.setToolTip(
            "Toggle to transpose table when export or copy")
        self.tool_transpose.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}table_transpose.png'))
        self.tool_transpose.clicked.connect(self.clicked_transpose)
        self.tool_transpose.setCheckable(True)

        self.tool_header = QToolButton()
        self.tool_header.setToolTip(
            "Toggle to include header when export or copy")
        self.tool_header.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}table_no_headers.png'))
        self.tool_header.clicked.connect(self.clicked_header)
        self.tool_header.setCheckable(True)

        self.tool_decimal = QToolButton()
        self.tool_decimal.setToolTip(
            'Set decimal mark to comma or point when export or copy.')
        self.tool_decimal.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}decimal_point.png'))
        self.tool_decimal.clicked.connect(self.clicked_decimal)
        self.tool_decimal.setCheckable(True)

        self.tool_decimal_all = QToolButton()
        self.tool_decimal_all.setToolTip(
            "Toggle to show all decimals")
        self.tool_decimal_all.setIcon(QIcon(
            f'{os.environ[ENV_ICON_PATH]}decimal_all.png'))
        self.tool_decimal_all.clicked.connect(self.clicked_decimal_all)
        self.tool_decimal_all.setCheckable(True)

        self.addWidget(self.tool_transpose)
        self.addWidget(self.tool_header)
        self.addWidget(self.tool_decimal)
        self.addWidget(self.tool_decimal_all)

        if self.parameters_output is not None:
            self.update_checked()

    def update_checked(self, icon_only=False):
        """Update toggled status and icons according to user_prefs.

        Parameters
        ----------
        icon_only : bool, optional
            Do not change user_prefs template, only display.
            The default is False.
        """
        if icon_only is False:
            self.tool_transpose.setChecked(self.parameters_output.transpose_table)
            self.tool_header.setChecked(self.parameters_output.include_header)
            if self.parameters_output.decimal_mark == ',':
                self.tool_decimal.setChecked(True)
            else:
                self.tool_decimal.setChecked(False)
            self.tool_decimal_all.setChecked(self.parameters_output.decimal_all)

        if self.parameters_output.include_header:
            self.tool_header.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}table_headers.png'))
        else:
            self.tool_header.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}table_no_headers.png'))
        if self.parameters_output.decimal_mark == ',':
            self.tool_decimal.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}decimal_comma.png'))
        else:
            self.tool_decimal.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}decimal_point.png'))

    def clicked_transpose(self):
        """Actions when transpose table button clicked."""
        self.parameters_output.transpose_table = (
            self.tool_transpose.isChecked()
            )
        if self.flag_edit:
            self.parent.flag_edit(True)

    def clicked_header(self):
        """Actions when include header button clicked."""
        self.parameters_output.include_header = (
            self.tool_header.isChecked()
            )
        self.update_checked(icon_only=True)
        if self.flag_edit:
            self.parent.flag_edit(True)

    def clicked_decimal(self):
        """Actions when decimal mark button clicked."""
        if self.parameters_output.decimal_mark == ',':
            self.parameters_output.decimal_mark = '.'
        else:
            self.parameters_output.decimal_mark = ','
        self.update_checked(icon_only=True)
        try:
            self.parent.main.refresh_results_display()  # if main window
        except AttributeError:
            pass
        if self.flag_edit:
            self.parent.flag_edit(True)

    def clicked_decimal_all(self):
        """Actions when decimal_all button clicked."""
        self.parameters_output.decimal_all = self.tool_decimal_all.isChecked()
        self.update_checked(icon_only=True)
        try:
            self.parent.main.refresh_results_display()  # if main window
        except AttributeError:
            pass
        if self.flag_edit:
            self.parent.flag_edit(True)


class ToolBarWindowLevel(QToolBar):
    """Toolbar for reuse with setting window level."""

    def __init__(self, parent):
        """Initialize.

        Parameters
        ----------
        parent : WindowLevelWidget
        """
        super().__init__()
        self.parent = parent

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
        self.tool_min_max_center_wl = QToolButton()
        self.tool_min_max_center_wl.setToolTip("Set WL to [min,max] of central part of active image")
        self.tool_min_max_center_wl.setIcon(
            QIcon(f'{os.environ[ENV_ICON_PATH]}minmax_center.png'))
        self.tool_min_max_center_wl.clicked.connect(
            lambda: self.clicked_window_level('min_max_center'))
        self.tool_edit_wl = QToolButton()
        self.tool_edit_wl.setToolTip("Edit WL by numbers")
        self.tool_edit_wl.setIcon(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'))
        self.tool_edit_wl.clicked.connect(self.set_window_level_by_numbers)
        self.addWidget(self.tool_min_max_wl)
        self.addWidget(self.tool_min_max_center_wl)
        self.addWidget(self.tool_range_wl)
        self.act_tool_dcm_wl = self.addWidget(self.tool_dcm_wl)
        self.addWidget(self.tool_edit_wl)
        self.act_wl_update = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}unlocked.png'),
            "Lock window level")
        self.act_wl_update.setCheckable(True)
        self.act_wl_update.triggered.connect(self.update_window_level_mode)
        self.addAction(self.act_wl_update)
        #self.chk_wl_update = QCheckBox('Lock WL')
        #self.chk_wl_update.toggled.connect(self.update_window_level_mode)
        #self.act_chk_wl_update = self.addWidget(self.chk_wl_update)

    def update_window_level_mode(self):
        """Set and unset lock on window level when selecting a new image."""
        if self.act_wl_update.isChecked():
            self.tool_min_max_wl.setCheckable(False)
            self.tool_min_max_center_wl.setCheckable(False)
            self.tool_range_wl.setCheckable(False)
            self.tool_dcm_wl.setCheckable(False)
            self.act_wl_update.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}locked.png'))
        else:
            self.tool_min_max_wl.setCheckable(True)
            self.tool_min_max_center_wl.setCheckable(True)
            self.tool_range_wl.setCheckable(True)
            self.tool_dcm_wl.setCheckable(True)
            # default
            self.tool_range_wl.setChecked(True)
            self.set_window_level('mean_stdev')
            self.act_wl_update.setIcon(QIcon(
                f'{os.environ[ENV_ICON_PATH]}unlocked.png'))

    def get_min_max(self):
        """Get lower and upper window level based on image.

        Returns
        -------
        min_wl : int
            lower window level
        max_wl : TYPE
            upper window level
        """
        if self.act_wl_update.isChecked() is False:
            if self.tool_min_max_wl.isChecked():
                self.set_window_level('min_max')
            elif self.tool_min_max_center_wl.isChecked():
                self.set_window_level('min_max_center')
            elif self.tool_range_wl.isChecked():
                self.set_window_level('mean_stdev')
            else:
                self.set_window_level('dcm')

        return self.parent.read_min_max()

    def clicked_window_level(self, arg):
        """When one of the window level toolbuttons is toggled.

        Parameters
        ----------
        arg : str
            type of window level 'dcm', 'min_max', 'mean_stdev'
        """
        if self.act_wl_update.isChecked() is False:
            # unCheck others, check selected
            if arg == 'min_max':
                self.tool_min_max_wl.setChecked(True)
                self.tool_min_max_center_wl.setChecked(False)
                self.tool_range_wl.setChecked(False)
                self.tool_dcm_wl.setChecked(False)
            elif arg == 'min_max_center':
                self.tool_min_max_wl.setChecked(False)
                self.tool_min_max_center_wl.setChecked(True)
                self.tool_range_wl.setChecked(False)
                self.tool_dcm_wl.setChecked(False)
            elif arg == 'mean_stdev':
                self.tool_min_max_wl.setChecked(False)
                self.tool_min_max_center_wl.setChecked(False)
                self.tool_range_wl.setChecked(True)
                self.tool_dcm_wl.setChecked(False)
            elif arg == 'dcm':
                self.tool_min_max_wl.setChecked(False)
                self.tool_min_max_center_wl.setChecked(False)
                self.tool_range_wl.setChecked(False)
                self.tool_dcm_wl.setChecked(True)

        self.set_window_level(arg)

    def set_window_level_by_numbers(self):
        """Dialog box to set min/max or center/width and option to lock."""
        factor = 1 / 10 ** self.parent.decimals
        try:
            from imageQC.ui.ui_dialogs import WindowLevelEditDialog
        except:
            from ui.ui_dialogs import WindowLevelEditDialog
        dlg = WindowLevelEditDialog(
            min_max=[
                factor * self.parent.min_wl.value(),
                factor * self.parent.max_wl.value()
                ],
            decimals=self.parent.decimals,
            show_lock_wl=self.act_wl_update.isVisible(),
            positive_negative=self.parent.positive_negative)
        res = dlg.exec()
        if res:
            minval, maxval, lock = dlg.get_min_max_lock()
            self.act_wl_update.setChecked(lock)
            self.parent.update_window_level(minval, maxval)
            if lock:
                self.update_window_level_mode()

    def set_window_level(self, arg, set_tools=False):
        """Set window level based on active image content.

        Parameters
        ----------
        arg : str
            type of window level 'dcm', 'min_max', 'min_max_center', 'mean_stdev'
        set_tools : bool, optional
            Also set the tool-buttons to change. Default is False.
        """
        if set_tools:
            self.tool_min_max_wl.setChecked(arg == 'min_max')
            self.tool_min_max_center_wl.setChecked(arg == 'min_max_center')
            self.tool_range_wl.setChecked(arg == 'mean_stdev')
            self.tool_dcm_wl.setChecked(arg == 'dcm')
        try:
            image = self.parent.parent.active_img
        except AttributeError:
            try:
                image = self.parent.parent.canvas.current_image
            except AttributeError:
                image = None
        if image is not None:
            minval = 0
            maxval = 0
            if arg == 'dcm':
                try:
                    imgno = self.parent.parent.gui.active_img_no
                    img_info = self.parent.parent.imgs[imgno]
                    if img_info.window_width > 0:
                        minval = img_info.window_center - 0.5*img_info.window_width
                        maxval = img_info.window_center + 0.5*img_info.window_width
                except (AttributeError, IndexError):
                    pass
            else:
                image_sub = image
                if arg == 'min_max_center':
                    sz_x, sz_y = image.shape
                    image_sub = image[sz_y//4:-sz_y//4,sz_x//4:-sz_x//4]
                else:
                    patches = []
                    try:
                        if self.parent.parent.wid_image_display.tool_rectangle.isChecked():
                            patches = self.parent.parent.wid_image_display.canvas.ax.patches
                    except AttributeError:
                        try:
                            if self.parent.parent.tool_rectangle.isChecked():
                                patches = self.parent.parent.canvas.ax.patches
                        except AttributeError:
                            pass
                    for patch in patches:
                        if patch.get_gid() == 'rectangle':
                            [x0, y0], [x1, y1] = patch.get_bbox().get_points()
                            x_tuple = (int(min([x0, x1])), int(max([x0, x1])) + 1)
                            y_tuple = (int(min([y0, y1])), int(max([y0, y1])) + 1)
                            image_sub = image[y_tuple[0]:y_tuple[1], x_tuple[0]:x_tuple[1]]

                if 'min_max' in arg:
                    minval = np.amin(image_sub)
                    maxval = np.amax(image_sub)
                elif arg == 'mean_stdev':
                    meanval = np.mean(image_sub)
                    stdval = np.std(image_sub)
                    minval = meanval-stdval
                    maxval = meanval+stdval

            if maxval == minval:
                minval = np.amin(image)
                maxval = np.amax(image)

            minval = np.round(minval)
            maxval = np.round(maxval)
            if maxval == minval:
                maxval = minval + 1

            self.parent.update_window_level(minval, maxval)

    def get_window_level_mode(self):
        """Get current window level mode.

        Returns
        -------
        mode_string : str
            min_max, mean_stdev, dcm or ''
        """
        if self.tool_min_max_wl.isChecked():
            mode_string = 'min_max'
        elif self.tool_min_max_center_wl.isChecked():
            mode_string = 'min_max_center'
        elif self.tool_range_wl.isChecked():
            mode_string = 'mean_stdev'
        elif self.tool_dcm_wl.isChecked():
            mode_string = 'dcm'
        else:
            mode_string = ''
        return mode_string


class WindowLevelWidget(QGroupBox):
    """Widget with groupbox holding WindowLevel display."""

    def __init__(self, parent, show_dicom_wl=True, show_lock_wl=True):
        """Initialize.

        Parameters
        ----------
        parent : MainWindow or ResultImageWidget
            need method set_active_image_min_max(minval, maxval)
        show_dicom_wl : bool, optional
            Show button dicom window level. The default is True.
        show_lock_wl : bool, optional
            Show checkbox to lock window level. The default is True.
        """
        super().__init__('Window Level')
        self.parent = parent
        self.setFont(FontItalic())
        self.tb_wl = ToolBarWindowLevel(self)
        self.tb_wl.setOrientation(Qt.Horizontal)
        self.tb_wl.act_tool_dcm_wl.setVisible(show_dicom_wl)
        self.tb_wl.act_wl_update.setVisible(show_lock_wl)
        self.min_wl = QSlider(Qt.Horizontal)
        self.max_wl = QSlider(Qt.Horizontal)
        self.lbl_min_wl = QLabel('-200')
        self.lbl_max_wl = QLabel('200')
        self.canvas = WindowLevelHistoCanvas(self)

        self.max_wl.setRange(-200, 200)
        self.max_wl.setValue(200)
        self.min_wl.setRange(-200, 200)
        self.min_wl.setValue(-200)
        self.min_wl.sliderReleased.connect(
            lambda: self.correct_window_level_sliders(sender='min'))
        self.max_wl.sliderReleased.connect(
            lambda: self.correct_window_level_sliders(sender='max'))
        self.decimals = 0  # 1 if difference < 10, 2 if difference < 1
        self.positive_negative = False  # True if center always should be zero
        self.lbl_center = QLabel('0')
        self.lbl_width = QLabel('400')
        self.colorbar = ColorBar(slider_min=self.min_wl, slider_max=self.max_wl)
        self.colorbar.setMaximumHeight(30)

        vlo_wl = QVBoxLayout()
        hlo_slider = QHBoxLayout()
        vlo_min = QVBoxLayout()
        vlo_min.addSpacing(20)
        vlo_min.addWidget(self.lbl_min_wl)
        vlo_min.addStretch()
        hlo_slider.addLayout(vlo_min)
        vlo_slider = QVBoxLayout()
        vlo_slider.addWidget(self.min_wl)
        vlo_slider.addWidget(self.max_wl)
        vlo_slider.addWidget(self.colorbar)
        vlo_slider.addWidget(self.canvas)
        hlo_slider.addLayout(vlo_slider)
        vlo_max = QVBoxLayout()
        vlo_max.addSpacing(20)
        vlo_max.addWidget(self.lbl_max_wl)
        vlo_max.addStretch()
        hlo_slider.addLayout(vlo_max)
        vlo_wl.addWidget(self.tb_wl)
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

        self.tb_wl.update_window_level_mode()

    def read_min_max(self):
        """Get current min max."""
        return (self.min_wl.value() / (10 ** self.decimals),
                self.max_wl.value() / (10 ** self.decimals))

    def read_range(self):
        """Get current slider range."""
        return (self.min_wl.minimum() / (10 ** self.decimals),
                self.max_wl.maximum() / (10 ** self.decimals))

    def update_window_level(self, minval, maxval, cmap=''):
        """Update GUI for window level sliders and labels + active image.

        Parameters
        ----------
        minval : float
        maxval : float
        """
        proceed = True
        try:
            self.min_wl.setValue(round(minval * 10 ** self.decimals))
            self.max_wl.setValue(round(maxval * 10 ** self.decimals))
        except ValueError:
            proceed = False
        if proceed:
            formatstr = f'.{self.decimals}f'
            formatstrmax = '.2e' if maxval > 99999 else formatstr
            formatstrmin = '.2e' if minval > 99999 else formatstr
            self.lbl_min_wl.setText(f'{minval:{formatstrmin}}')
            self.lbl_max_wl.setText(f'{maxval:{formatstrmax}}')
            self.lbl_center.setText(f'{0.5*(minval+maxval):{formatstrmax}}')
            self.lbl_width.setText(f'{(maxval-minval):{formatstrmax}}')

            self.parent.set_active_image_min_max(minval, maxval)
            self.colorbar.colorbar_draw(cmap=cmap)

    def correct_window_level_sliders(self, sender='min'):
        """Make sure min_wl < max_wl after user input."""
        if self.max_wl.value() < self.min_wl.value():
            maxval = self.min_wl.value()
            minval = self.max_wl.value()
        else:
            minval = self.min_wl.value()
            maxval = self.max_wl.value()
        if self.positive_negative:
            if sender == 'min':
                maxval = - minval
            else:
                minval = - maxval
        minval = minval / (10 ** self.decimals)
        maxval = maxval / (10 ** self.decimals)
        self.update_window_level(minval, maxval)


class WindowLevelHistoCanvas(FigureCanvasQTAgg):
    """Canvas for display of histogram for the active image."""

    def __init__(self, parent):
        self.fig = matplotlib.figure.Figure(figsize=(2, 1))
        self.fig.subplots_adjust(0., 0., 1., 1.)
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.parent = parent

    def plot(self, nparr, decimals=0):
        """Refresh histogram."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        amin, amax = (np.amin(nparr), np.amax(nparr))
        try:
            hist, bins = np.histogram(nparr, bins=np.arange(
                amin, amax, (amax - amin)/100.))
            self.ax.plot(bins[:-1], hist)
            self.ax.axis('off')
            formatstr = f'.{decimals}f'
            formatstrmax = '.2e' if amax > 99999 else formatstr
            formatstrmin = '.2e' if amin > 99999 else formatstr
            at_minmax = matplotlib.offsetbox.AnchoredText(
                f'max: {amax:{formatstrmax}} \n min: {amin:{formatstrmin}}',
                prop=dict(size=12, weight='light'), frameon=False, loc='upper right')
            self.ax.add_artist(at_minmax)

            if all([self.parent.min_wl, self.parent.max_wl]):
                factor = 1 / 10 ** self.parent.decimals
                range_max = factor * self.parent.max_wl.maximum()
                range_min = factor * self.parent.min_wl.minimum()

                full = range_max - range_min
                if full > 0:
                    min_ratio = (amin - range_min) / full
                    max_ratio = (amax - range_min) / full
                    if max_ratio == min_ratio:
                        max_ratio = min_ratio + 0.01
                    self.fig.subplots_adjust(min_ratio, 0., max_ratio, 1.)

            self.draw()
        except ValueError:
            pass


class ColorBar(FigureCanvasQTAgg):
    """Canvas for colorbar."""

    def __init__(self, slider_min=None, slider_max=None):
        self.fig = matplotlib.figure.Figure(figsize=(2, 0.5))
        self.fig.subplots_adjust(0., 0., 1., 1.)
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.slider_min = slider_min
        self.slider_max = slider_max
        self.cmap = 'gray'

    def colorbar_draw(self, cmap=''):
        """Draw or update colorbar."""
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        if cmap == '':
            cmap = self.cmap
        else:
            self.cmap = cmap
        if cmap:
            try:
                _ = matplotlib.colorbar.ColorbarBase(
                    ax, cmap=matplotlib.pyplot.cm.get_cmap(cmap),
                    orientation='horizontal')
            except AttributeError:  # from matplotlib v 3.9.0
                _ = matplotlib.colorbar.ColorbarBase(
                    ax, cmap=matplotlib.pyplot.get_cmap(cmap),
                    orientation='horizontal')
            if all([self.slider_min, self.slider_max]):
                range_max = self.slider_max.maximum()
                range_min = self.slider_min.minimum()
                set_min = self.slider_min.value()
                set_max = self.slider_max.value()
                full = range_max - range_min
                if full > 0:
                    min_ratio = (set_min - range_min) / full
                    max_ratio = (set_max - range_min) / full
                    if max_ratio == min_ratio:
                        max_ratio = min_ratio + 0.01
                    self.fig.subplots_adjust(min_ratio, 0., max_ratio, 1.)
        ax.axis('off')
        self.draw()


class ListWidgetCheckable(QListWidget):
    """Checkable list widget."""

    def __init__(self, texts=[], set_checked_ids=[]):
        super().__init__()
        self.texts = texts
        self.addItems(self.texts)

        for i in range(len(self.texts)):
            item = self.item(i)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if i in set_checked_ids:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

    def get_checked_ids(self):
        """Get checked ids from list."""
        checked_ids = []
        for i in range(self.count()):
            if self.item(i).checkState() == Qt.Checked:
                checked_ids.append(i)
        return checked_ids

    def get_checked_texts(self):
        """Get checked strings from list."""
        checked_texts = []
        for i, txt in enumerate(self.texts):
            if self.item(i).checkState() == Qt.Checked:
                checked_texts.append(txt)
        return checked_texts

    def set_checked_texts(self, set_texts):
        """Set checked strings in list."""
        for i, text in enumerate(self.texts):
            item = self.item(i)
            if text in set_texts:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)


class CheckCell(QCheckBox):
    """CheckBox for use in TreeWidget cells."""

    def __init__(self, parent, initial_value=True):
        super().__init__()
        self.setStyleSheet('''QCheckBox {
            margin-left:50%;
            margin-right:50%;
            }''')
        self.setChecked(initial_value)
        self.parent = parent
        self.clicked.connect(self.parent.flag_edit)


class LineCell(QLineEdit):
    """LineEdit for use in TreeWidget cells."""

    def __init__(self, parent, initial_text=''):
        super().__init__()
        self.parent = parent
        self.setText(initial_text)
        self.textEdited.connect(self.parent.flag_edit)


class PushColorCell(QPushButton):
    """Button to show and edit colors from cell widget in table."""

    def __init__(self, parent, initial_color='#000000', row=-1, col=-1):
        super().__init__(initial_color)
        self.setStyleSheet(
            f'QPushButton{{background-color: {initial_color};}}')
        self.parent = parent
        self.clicked.connect(
            lambda: self.parent.color_edit(self.row, self.col))
        self.row = row
        self.col = col


class BoolSelect(QWidget):
    """Radiobutton group of two returning true/false as selected value."""

    def __init__(self, parent, text_true='True', text_false='False',
                 text_label=''):
        """Initialize BoolSelect.

        Parameters
        ----------
        parent : widget
            test widget containing this BoolSelect and param_changed
        text_true : str
            Text of true value
        text_false : str
            Text of false value
        text_label : str
            Text to show before radiobuttons
        """
        super().__init__()
        self.parent = parent

        self.btn_true = QRadioButton(text_true)
        self.btn_true.setChecked(True)
        self.btn_false = QRadioButton(text_false)

        hlo = QHBoxLayout()
        if text_label:
            hlo.addWidget(QLabel(text_label))
        group = QButtonGroup()
        group.setExclusive(True)
        group.addButton(self.btn_true)
        group.addButton(self.btn_false)
        hlo.addWidget(self.btn_true)
        hlo.addWidget(self.btn_false)
        self.setLayout(hlo)

    def setChecked(self, value=True):
        """Set BoolSelect to input bool. Mimic QCheckBox behaviour.

        Parameters
        ----------
        value : bool, optional
            set value. The default is True.
        """
        self.btn_true.setChecked(value)
        if value is True:
            self.btn_false.setChecked(False)
        else:
            self.btn_false.setChecked(True)

    def isChecked(self):
        """Make BoolSelect return as if QCheckBox.

        Returns
        -------
        bool
            True if true_value is set.
        """
        return self.btn_true.isChecked()


class CheckComboBox(QComboBox):
    """Class for checkable QComboBox."""

    # https://learndataanalysis.org/how-to-create-checkable-combobox-widget-pyqt5-tutorial/
    def __init__(self):
        super().__init__()
        self.edited = False
        self.view().pressed.connect(self.itemPressed)

    def setItemChecked(self, index, checked=False):
        """Set checkstate of list index.

        Parameters
        ----------
        index : int
            list index to set checkstate for
        checked : bool, optional
            The default is False.
        """
        item = self.model().item(index, self.modelColumn())
        if checked:
            item.setCheckState(Qt.Checked)
        else:
            item.setCheckState(Qt.Unchecked)

    def itemPressed(self, index):
        """Handle item pressed."""
        item = self.model().itemFromIndex(index)

        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
        else:
            item.setCheckState(Qt.Checked)
        self.edited = True

    def hidePopup(self):
        if not self.edited:
            super().hidePopup()
        self.edited = False

    def itemChecked(self, index):
        item = self.model().item(index, self.modelColumn())
        return item.checkState() == Qt.Checked

    def get_ids_checked(self):
        """Get array of checked ids."""
        ids = []
        for i in range(self.count()):
            if self.itemChecked(i):
                ids.append(i)

        return ids


class StatusBar(QStatusBar):
    """Tweeks to QStatusBar."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent
        self.setStyleSheet("QStatusBar{padding-left: 8px;}")
        self.default_color = self.palette().window().color().name()
        self.message = QLabel('')
        self.message.setAlignment(Qt.AlignCenter)
        self.addWidget(self.message, 1)
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.clearMessage)

    def showMessage(self, txt, timeout=0, warning=False):
        """Set background color when message is shown."""
        if warning:
            self.setStyleSheet("QStatusBar{background:#efb412;}")
            timeout = 2000
        else:
            self.setStyleSheet("QStatusBar{background:#6e94c0;}")
        self.message.setText(txt)
        if timeout > 0:
            self.timer.start(timeout)
        else:
            self.timer.start()
        qApp.processEvents()

    def clearMessage(self):
        """Reset background and clear message."""
        self.setStyleSheet(
            "QStatusBar{background:" + self.default_color + ";}")
        self.message.setText('')
        qApp.processEvents()


class StatusLabel(QWidget):
    """Widget with QLabel - to make it look like StatusBar."""

    def __init__(self, parent):
        super().__init__()
        self.main = parent
        self.default_color = self.palette().window().color().name()
        self.setStyleSheet("QWidget{background-color:" + self.default_color + ";}")
        lo = QHBoxLayout()
        self.message = QLabel('')
        self.message.setStyleSheet("QLabel{padding-left: 8px;}")
        self.message.setAlignment(Qt.AlignCenter)
        self.setLayout(lo)
        lo.addWidget(self.message)

    def showMessage(self, txt):
        """Set background color when message is shown."""
        self.setStyleSheet("QWidget{background-color:#6e94c0;}")
        self.message.setText(txt)
        qApp.processEvents()

    def clearMessage(self):
        """Reset background and clear message."""
        self.setStyleSheet(
            "QWidget{background-color:" + self.default_color + ";}")
        self.message.setText('')
        qApp.processEvents()


class ImageNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib navigation toolbar with some modifications."""

    def __init__(self, canvas, window, remove_customize=False):
        super().__init__(canvas, window)
        remove_list = ['Subplots']
        if remove_customize:
            remove_list.append('Customize')
        for x in self.actions():
            if x.text() in remove_list:
                self.removeAction(x)

    def set_message(self, s):
        """Hide cursor position and value text."""
        pass

    # from https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/backends/backend_qt.py
    #  dirty fix to avoid crash on self.canvas.parent() TypeError
    # also added .csv save
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
            filter_this = f'{name} ({exts_list})'
            if default_filetype in exts:
                selectedFilter = filter_this
            filters.append(filter_this)
        filters.insert(0, '*.csv')  # TODO - also .npy (np.save(fname, arr))
        filters = ';;'.join(filters)

        fname, filter_selected = QFileDialog.getSaveFileName(
            self, 'Choose a filename to save to', '',
            filters, selectedFilter)
        if fname:
            if filter_selected == '*.csv':
                try:
                    current_image = self.canvas.ax.get_images()[0].get_array()
                    np.savetxt(fname, current_image, delimiter=',')
                except (AttributeError, IndexError) as e:
                    QMessageBox.critical(
                        self, "Error saving file", str(e))
            else:
                try:
                    self.canvas.figure.savefig(fname)
                except Exception as e:
                    QMessageBox.critical(
                        self, "Error saving file", str(e))
