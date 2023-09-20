#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface classes for different uses and reuses in ImageQC.

@author: Ellen Wasbo
"""
import os

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import (
    qApp, QWidget, QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QFrame,
    QToolBar, QAction, QComboBox, QRadioButton, QButtonGroup, QToolButton,
    QLabel, QPushButton, QListWidget, QLineEdit, QCheckBox,
    QProgressDialog, QProgressBar, QStatusBar
    )

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
            txt = ('Sorry - Test is under construction. Do not trust the results. '
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


class PushButtonRounded(QPushButton):
    """Styled PushButton."""

    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet(
            """
            QPushButton {
                border-style: solid;
                border-width: 2px;
                border-color: #888888;
                border-radius: 10px;
                padding: 6px;
                }
            QPushButton:hover {
                background-color: #888888;
                }
            QPushButton:pressed {
                background-color: #999999;
                }
            """
            )


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

        self.addWidget(self.tool_transpose)
        self.addWidget(self.tool_header)
        self.addWidget(self.tool_decimal)

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
            self.tool_transpose.setChecked(
                self.parameters_output.transpose_table)
            self.tool_header.setChecked(
                self.parameters_output.include_header)
            if self.parameters_output.decimal_mark == ',':
                self.tool_decimal.setChecked(True)
            else:
                self.tool_decimal.setChecked(False)

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


class BoolSelect(QWidget):
    """Radiobutton group of two returning true/false as selected value."""

    def __init__(self, parent, text_true='True', text_false='False'):
        """Initialize BoolSelect.

        Parameters
        ----------
        parent : widget
            test widget containing this BoolSelect and param_changed
        text_true : str
            Text of true value
        text_false : str
            Text of false value
        """
        super().__init__()
        self.parent = parent

        self.btn_true = QRadioButton(text_true)
        self.btn_true.setChecked(True)
        self.btn_false = QRadioButton(text_false)

        hlo = QHBoxLayout()
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
