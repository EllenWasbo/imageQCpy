#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User interface classes for different uses and reuses in ImageQC.

@author: Ellen Wasbo
"""
from time import time
import copy
import os

import pandas as pd
from PyQt5.QtCore import Qt, QModelIndex, QTimer
from PyQt5.QtGui import (
    QIcon, QFont, QBrush, QColor, QPalette, QStandardItemModel
    )
from PyQt5.QtWidgets import (
    QWidget, QDialog, QDialogButtonBox,
    QVBoxLayout, QHBoxLayout, QFrame, QFormLayout,
    QToolBar, QAction, QComboBox, QRadioButton, QButtonGroup, QToolButton,
    QLabel, QPushButton, QListWidget, QLineEdit, QCheckBox, QTextEdit,
    QTreeWidget, QTreeWidgetItem, QTreeView,
    QMessageBox, QProgressDialog, QInputDialog,
    QStatusBar, qApp
    )
import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)

# imageQC block start
from imageQC.scripts.mini_methods_format import val_2_str, get_format_strings
from imageQC.scripts.mini_methods import get_included_tags
from imageQC.config.iQCconstants import (
    ENV_ICON_PATH, QUICKTEST_OPTIONS,
    ALTERNATIVES, CALCULATION_OPTIONS, HEADERS, HEADERS_SUP
    )
import imageQC.config.config_func as cff
import imageQC.config.config_classes as cfc
# imageQC block end


def proceed_question(widget, question,
                     msg_width=500, detailed_text='', info_text=''):
    """Ask a question whether to proceed with some process.

    Parameters
    ----------
    widget : QWidget
    question : str
    msg_width : int
        label width in pixels. Default is 500.
    detailed_text : str
        add detailed text if not empty string. Default is empty string.
    info_text : str
        subtext after question. Default is empty string.

    Returns
    -------
    proceed : bool
        yes = true
    """
    proceed = False

    msgBox = QMessageBox(
        QMessageBox.Question,
        'Proceed?', question,
        buttons=QMessageBox.Yes | QMessageBox.No,
        parent=widget
        )
    msgBox.setDefaultButton(QMessageBox.No)
    if detailed_text != '':
        msgBox.setDetailedText(detailed_text)
    if info_text != '':
        msgBox.setInformativeText(info_text)
    msgBox.setStyleSheet(
        f"""
        QPushButton {{
            padding: 5px;
            }}
        QLabel {{
            width: {msg_width}px;
            }}
        """)
    msgBox.exec_()
    reply = msgBox.standardButton(msgBox.clickedButton())
    if reply == QMessageBox.Yes:
        proceed = True

    return proceed


def add_to_modality_dict(
        templates_dict, modality, new_template, parent_widget=None):
    """Add template to template modality dict.

    Parameters
    ----------
    templates_dict : dict
        modality_dict of templates
    modality: str
    new_template : object
    parent_widget : widget

    Returns
    -------
    status: bool
        True if succeeded adding
    """
    text, ok = QInputDialog.getText(
        parent_widget, 'New label', 'Name the new template')
    current_labels = []
    status = False
    if ok and text != '':
        current_labels = \
            [obj.label for obj
             in templates_dict[modality]]
        if text in current_labels:
            if parent_widget is not None:
                QMessageBox.warning(
                    parent_widget, 'Label already in use',
                    'This label is already in use.')
        else:
            new_template.label = text
            if templates_dict[modality][0].label == '':
                templates_dict[modality][0] = copy.deepcopy(new_template)
            else:
                templates_dict[modality].append(copy.deepcopy(new_template))
            status = True

    return status


class ImageQCDialog(QDialog):
    """Dialog for reuse with imageQC icon and flags."""

    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)


class LabelItalic(QLabel):
    """Label with preset italic font."""

    def __init__(self, txt):
        html_txt = f"""<html><head/><body>
        <p><i>{txt}</i></p>
        </body></html>"""
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
    """Message box with formated information."""

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

        vLO = QVBoxLayout()
        vLO.addWidget(dlg.infotext)
        buttons = QDialogButtonBox.Ok
        dlg.buttonBox = QDialogButtonBox(buttons)
        dlg.buttonBox.accepted.connect(dlg.accept)
        vLO.addWidget(dlg.buttonBox)
        dlg.setLayout(vLO)

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


class TextDisplay(ImageQCDialog):
    """QDialog with QTextEdit.setPlainText to display text."""

    def __init__(self, parent_widget, text, title='',
                 read_only=True,
                 min_width=1000, min_height=1000):
        super().__init__()
        txtEdit = QTextEdit('', self)
        txtEdit.setPlainText(text)
        txtEdit.setReadOnly(read_only)
        txtEdit.createStandardContextMenu()
        txtEdit.setMinimumWidth(min_width)
        txtEdit.setMinimumHeight(min_height)
        self.setWindowTitle(title)
        self.setMinimumWidth(min_width)
        self.setMinimumHeight(min_height)
        self.show()


class ProgressModal(QProgressDialog):
    """Redefine QProgressDialog to set wanted behaviour."""

    def __init__(self, text, cancel_text, start, stop, parent):
        super().__init__(text, cancel_text, start, stop, parent)
        self.setWindowModality(Qt.WindowModal)
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle('Wait while processing...')
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setMinimumDuration(200)
        self.setAutoClose(True)


class ToolBarBrowse(QToolBar):
    """Toolbar for reuse with search button."""

    def __init__(self, browse_tooltip='', clear=False):
        super().__init__()
        self.actBrowse = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}open.png'),
            browse_tooltip, self)
        self.addActions([self.actBrowse])
        if clear:
            self.actClear = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}clear.png'),
                'Clear', self)
            self.addActions([self.actClear])


class ToolBarEdit(QToolBar):
    """Toolbar for reuse with edit button."""

    def __init__(self, tooltip=''):
        super().__init__()
        self.actEdit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            tooltip, self)
        self.addActions([self.actEdit])


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
        if self.flag_edit:
            self.parent.flag_edit(True)


class QuestionBox(QMessageBox):
    """QMessageBox with changed yes no text as options."""

    def __init__(
            self, parent=None, title='?', msg='?',
            yes_text='Yes', no_text='No', msg_width=500):
        """Initiate QuestionBox.

        Parameters
        ----------
        parent : widget, optional
            The default is None.
        title : str, optional
            The default is '?'.
        msg : str, optional
            Question text. The default is '?'.
        yes_text : TYPE, optional
            Text on yes button. The default is 'Yes'.
        no_text : TYPE, optional
            Text on no button. The default is 'No'.
        msg_width : int, optional
            Width of question label. The default is 500.

        Returns
        -------
        None.

        """
        super().__init__(
            QMessageBox.Question, title, msg, parent=parent)
        self.setIcon(QMessageBox.Question)
        self.setWindowTitle(title)
        self.setText(msg)
        self.setTextFormat(Qt.RichText)
        self.addButton(no_text, QMessageBox.RejectRole)
        self.addButton(yes_text, QMessageBox.AcceptRole)
        self.setStyleSheet(
            f"""
            QPushButton {{
                padding: 5px;
                }}
            QLabel {{
                width: {msg_width}px;
                }}
            """)


class TagPatternTree(QWidget):
    """Widget for tag pattern and toolbar used in TagPatternWidget."""

    def __init__(self, parent, title='Tag pattern', typestr='sort',
                 list_number=1, editable=True):
        super().__init__()
        self.parent = parent
        self.parentabove = self.parent.parent
        self.typestr = typestr
        self.list_number = list_number

        self.hLO = QHBoxLayout()
        self.setLayout(self.hLO)

        if editable:
            vLOpush = QVBoxLayout()
            self.hLO.addLayout(vLOpush)
            vLOpush.addStretch()
            btnPush = QPushButton('>>')
            btnPush.clicked.connect(self.push_tag)
            vLOpush.addWidget(btnPush)
            vLOpush.addStretch()
            self.hLO.addSpacing(20)

        vLO = QVBoxLayout()
        self.hLO.addLayout(vLO)
        vLO.addWidget(LabelItalic(title))
        self.tablePattern = QTreeWidget()
        self.tablePattern.setColumnCount(2)
        self.tablePattern.setColumnWidth(0, 200)
        if self.typestr == 'none':
            self.tablePattern.setColumnCount(1)
            self.tablePattern.setColumnWidth(0, 200)
            self.tablePattern.setHeaderLabels(['Tag'])
        else:
            self.tablePattern.setColumnCount(2)
            self.tablePattern.setColumnWidth(0, 200)
            self.tablePattern.setColumnWidth(1, 200)
        if self.typestr == 'sort':
            self.tablePattern.setHeaderLabels(['Tag', 'Sorting'])
        elif self.typestr == 'format':
            self.tablePattern.setHeaderLabels(['Tag', 'Format'])

        vLO.addWidget(self.tablePattern)
        if editable:
            self.tablePattern.setMinimumSize(400, 220)
        else:
            self.tablePattern.setMinimumWidth(400)
        self.tablePattern.setRootIsDecorated(False)

        palette = self.tablePattern.palette()
        palette.setColor(
            QPalette.Inactive, QPalette.Highlight,
            palette.color(QPalette.Active, QPalette.Highlight))
        palette.setColor(
            QPalette.Inactive, QPalette.HighlightedText,
            palette.color(QPalette.Active, QPalette.HighlightedText))
        self.tablePattern.setPalette(palette)

        if editable:
            tb = QToolBar()
            tb.setOrientation(Qt.Vertical)
            actSort = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}sortAZ.png'),
                'ASC or DESC when sorting images', self)
            actSort.triggered.connect(self.sort)
            actFormatOut = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}format.png'),
                'Format output for selected tag', self)
            actFormatOut.triggered.connect(self.format_output)
            actUp = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}moveUp.png'),
                'Move tag(s) up in pattern list', self)
            actUp.triggered.connect(self.move_up)
            actDown = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}moveDown.png'),
                'Move tag(s) down in pattern list', self)
            actDown.triggered.connect(self.move_down)
            actDelete = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
                'Delete selected tag(s) from pattern', self)
            actDelete.triggered.connect(self.delete)
            if self.typestr == 'sort':
                tb.addActions([actSort, actUp, actDown, actDelete])
            elif self.typestr == 'none':
                tb.addActions([actUp, actDown, actDelete])
            else:
                tb.addActions([actFormatOut, actUp, actDown, actDelete])
            self.hLO.addWidget(tb)

    def push_tag(self):
        """Button >> pressed - push selected tags into pattern."""
        rows = [index.row() for index in
                self.parent.listTags.selectedIndexes()]
        if self.list_number == 2:
            tagAlready = self.parentabove.current_template.list_tags2
        else:
            tagAlready = self.parentabove.current_template.list_tags

        for row in rows:
            if self.parent.listTags.item(row).text() not in tagAlready:
                if self.list_number == 1:
                    self.parentabove.current_template.list_tags.append(
                        self.parent.listTags.item(row).text())
                    if self.typestr in ['sort', 'none']:
                        self.parentabove.current_template.list_sort.append(
                            True)
                    else:
                        self.parentabove.current_template.list_format.append(
                            '')
                else:
                    self.parentabove.current_template.list_tags2.append(
                        self.parent.listTags.item(row).text())
                    self.parentabove.current_template.list_format2.append('')
        self.update_data(set_selected=len(rows)+1)
        self.parentabove.flag_edit()

    def sort(self):
        """Change between ASC / DESC for selected tag."""
        sel = self.tablePattern.selectedIndexes()
        row = -1
        if len(sel) > 0:
            row = sel[0].row()
        if row > -1:
            self.parentabove.current_template.list_sort[row] = \
                not self.parentabove.current_template.list_sort[row]
            self.update_data(set_selected=row)
            self.parentabove.flag_edit()

    def format_output(self):
        """Edit f-string for selected tag."""
        sel = self.tablePattern.selectedIndexes()
        row = -1
        if len(sel) > 0:
            row = sel[0].row()
        if row > -1:
            if self.list_number == 1:
                format_str = self.parentabove.current_template.list_format[row]
            else:
                format_str = self.parentabove.current_template.list_format2[
                    row]
            dlg = FormatDialog(
                self, format_string=format_str)
            res = dlg.exec()
            if res:
                new_str = dlg.get_data()
                if self.list_number == 1:
                    self.parentabove.current_template.list_format[
                        row] = new_str
                else:
                    self.parentabove.current_template.list_format2[
                        row] = new_str
                self.update_data(set_selected=row)
                self.parentabove.flag_edit()
        else:
            QMessageBox.information(
                self, 'No tag selected',
                'Select a tag from the tag pattern to format.')

    def move_up(self):
        """Move tag up if possible."""
        sel = self.tablePattern.selectedIndexes()
        row = -1
        if len(sel) > 0:
            row = sel[0].row()
        if row > 0:
            if self.list_number == 1:
                popped_tag = \
                    self.parentabove.current_template.list_tags.pop(row)
                self.parentabove.current_template.list_tags.insert(
                    row - 1, popped_tag)
            else:
                popped_tag = \
                    self.parentabove.current_template.list_tags2.pop(row)
                self.parentabove.current_template.list_tags2.insert(
                    row - 1, popped_tag)

            if self.typestr == 'sort':
                popped_sort = self.parentabove.current_template.list_sort.pop(
                    row)
                self.parentabove.current_template.list_sort.insert(
                    row - 1, popped_sort)
            else:
                if self.list_number == 1:
                    popped_format = \
                        self.parentabove.current_template.list_format.pop(row)
                    self.parentabove.current_template.list_format.insert(
                        row - 1, popped_format)
                else:
                    popped_format = \
                        self.parentabove.current_template.list_format2.pop(row)
                    self.parentabove.current_template.list_format2.insert(
                        row - 1, popped_format)
            self.update_data(set_selected=row-1)
            self.parentabove.flag_edit()

    def move_down(self):
        """Move tag down if possible."""
        sel = self.tablePattern.selectedIndexes()
        row = -1
        if len(sel) > 0:
            row = sel[0].row()
        if self.list_number == 1:
            n_tags = len(self.parentabove.current_template.list_tags)
        else:
            n_tags = len(self.parentabove.current_template.list_tags2)
        if row < n_tags-1:
            if self.list_number == 1:
                popped_tag = \
                    self.parentabove.current_template.list_tags.pop(row)
                self.parentabove.current_template.list_tags.insert(
                    row + 1, popped_tag)
            else:
                popped_tag = \
                    self.parentabove.current_template.list_tags2.pop(row)
                self.parentabove.current_template.list_tags2.insert(
                    row + 1, popped_tag)
            if self.typestr == 'sort':
                popped_sort = \
                    self.parentabove.current_template.list_sort.pop(row)
                self.parentabove.current_template.list_sort.insert(
                    row, popped_sort)
            else:
                if self.list_number == 1:
                    popped_format = \
                        self.parentabove.current_template.list_format.pop(row)
                    self.parentabove.current_template.list_format.insert(
                        row + 1, popped_format)
                else:
                    popped_format = \
                        self.parentabove.current_template.list_format2.pop(row)
                    self.parentabove.current_template.list_format2.insert(
                        row + 1, popped_format)
            self.update_data(set_selected=row+1)
            self.parentabove.flag_edit()

    def delete(self):
        """Delete selected tag(s)."""
        sel = self.tablePattern.selectedIndexes()
        row = -1
        if len(sel) > 0:
            row = sel[0].row()
            if self.list_number == 1:
                self.parentabove.current_template.list_tags.pop(row)
            else:
                self.parentabove.current_template.list_tags2.pop(row)
            if self.typestr == 'sort':
                self.parentabove.current_template.list_sort.pop(row)
            else:
                if self.list_number == 1:
                    self.parentabove.current_template.list_format.pop(row)
                else:
                    self.parentabove.current_template.list_format2.pop(row)
            self.update_data(set_selected=row)
            self.parentabove.flag_edit()

    def update_data(self, set_selected=0):
        """Update tablePattern with data from current_template."""
        self.tablePattern.clear()
        if self.list_number == 1:
            list_tags = self.parentabove.current_template.list_tags
        else:
            list_tags = self.parentabove.current_template.list_tags2
        if len(list_tags) > 0:
            for rowno, tagname in enumerate(list_tags):

                if self.typestr == 'sort':
                    infotext = 'ASC' if \
                        self.parentabove.current_template.list_sort[rowno] \
                        else 'DESC'
                else:
                    if self.list_number == 1:
                        infotext = \
                            self.parentabove.current_template.list_format[
                                rowno]
                    else:
                        infotext = \
                            self.parentabove.current_template.list_format2[
                                rowno]
                row_strings = [tagname, infotext]
                item = QTreeWidgetItem(row_strings)
                if self.parent.lock_on_general is False:
                    try:
                        if tagname in self.parent.general_tags:
                            item.setForeground(
                                0, QBrush(QColor(110, 148, 192)))
                    except AttributeError:
                        pass  # ignore if editable = False
                self.tablePattern.addTopLevelItem(item)

            self.tablePattern.setCurrentItem(
                self.tablePattern.topLevelItem(set_selected))


class TagPatternWidget(QWidget):
    """Widget for setting the parameters for TagPattern.

    Parameters
    ----------
    parent : widget
    typestr : str
        'sort' if TagPatternSort, 'format' if TagPatternFormat
    lock_on_general : bool
        True if only general tag_infos (e.g. automation rename)
    rename_pattern : bool
        True = two patterns (subfolder + file)
    editable : bool
        False if editing not available (taglist and buttons not visible)
    """

    def __init__(self, parent, typestr='sort', lock_on_general=False,
                 rename_pattern=False, open_files_pattern=False,
                 editable=True):
        super().__init__()
        self.parent = parent
        self.typestr = typestr
        self.lock_on_general = lock_on_general
        self.rename_pattern = rename_pattern
        self.open_files_pattern = open_files_pattern

        hLO = QHBoxLayout()
        self.setLayout(hLO)

        vLOtaglist = QVBoxLayout()
        hLO.addLayout(vLOtaglist)

        if editable:
            vLOtaglist.addWidget(LabelItalic('Available DICOM tags'))
            self.listTags = QListWidget()
            self.listTags.setSelectionMode(QListWidget.ExtendedSelection)
            self.listTags.itemDoubleClicked.connect(self.double_click_tag)
            vLOtaglist.addWidget(self.listTags)
            if self.lock_on_general:
                vLOtaglist.addWidget(LabelItalic('General tags only'))
            else:
                vLOtaglist.addWidget(LabelItalic('Blue font = general tags'))

            palette = self.listTags.palette()
            palette.setColor(
                QPalette.Inactive, QPalette.Highlight,
                palette.color(QPalette.Active, QPalette.Highlight))
            palette.setColor(
                QPalette.Inactive, QPalette.HighlightedText,
                palette.color(QPalette.Active, QPalette.HighlightedText))
            self.listTags.setPalette(palette)

        vLO_pattern = QVBoxLayout()
        hLO.addLayout(vLO_pattern)

        if self.rename_pattern:
            tit = 'Subfolder rename pattern'
        elif self.open_files_pattern:
            tit = 'Series indicator(s)'
        else:
            tit = 'Tag pattern'

        self.wPattern = TagPatternTree(
            self, title=tit, typestr=self.typestr, editable=editable)
        vLO_pattern.addWidget(self.wPattern)
        if self.rename_pattern or self.open_files_pattern:
            if editable:
                vLO_pattern.setSpacing(5)
            else:
                hLO.setSpacing(5)
            if self.rename_pattern:
                tit = 'File rename pattern'
            else:
                tit = 'File sort pattern'
            self.wPattern2 = TagPatternTree(
                self, title=tit, typestr=self.typestr,
                list_number=2, editable=editable)
            if editable:
                vLO_pattern.addWidget(self.wPattern2)
            else:
                hLO.addWidget(self.wPattern2)

    def fill_list_tags(self, modality):
        """Find tags from tag_infos.yaml and fill list."""
        try:
            self.listTags.clear()
            general_tags, included_tags = get_included_tags(
                modality, self.parent.tag_infos)
            self.listTags.addItems(included_tags)
            if self.lock_on_general is False:
                for i in range(self.listTags.count()):
                    if included_tags[i] in general_tags:
                        self.listTags.item(i).setForeground(
                            QBrush(QColor(110, 148, 192)))
        except (RuntimeError, AttributeError):
            pass

    def double_click_tag(self, item):
        """Double click item = push item."""
        if self.rename_pattern is False:
            self.wPattern.push_tag()

    def update_data(self):
        """Fill pattern list."""
        self.wPattern.update_data()


class FormatDialog(ImageQCDialog):
    """Dialog to set format-string of tags in TagPatternFormat."""

    def __init__(self, parent, format_string='', DICOM_display=False):
        super().__init__()
        self.setWindowTitle('Set format')
        self.cbox_decimals = QComboBox()
        self.cbox_padding = QComboBox()
        self.prefix = QLineEdit('')
        self.suffix = QLineEdit('')

        vLO = QVBoxLayout()
        self.setLayout(vLO)

        hLO_decimals = QHBoxLayout()
        hLO_decimals.addWidget(QLabel('Number of decimals: '))
        dec_list = ['Auto'] + [str(i) for i in range(0, 10)]
        self.cbox_decimals.addItems(dec_list)
        self.cbox_decimals.setFixedWidth(100)
        hLO_decimals.addWidget(self.cbox_decimals)
        hLO_decimals.addStretch()
        vLO.addLayout(hLO_decimals)
        vLO.addSpacing(20)
        hLO_padding = QHBoxLayout()
        hLO_padding.addWidget(
            QLabel('0-padding (N characters): '))
        pad_list = ['Auto'] + [str(i) for i in range(2, 16)]
        self.cbox_padding.addItems(pad_list)
        self.cbox_padding.setFixedWidth(100)
        hLO_padding.addWidget(self.cbox_padding)
        hLO_padding.addStretch()
        vLO.addLayout(hLO_padding)

        hLO_prefix = QHBoxLayout()
        hLO_prefix.addWidget(QLabel("Prefix: "))
        hLO_prefix.addWidget(self.prefix)
        vLO.addLayout(hLO_prefix)
        hLO_suffix = QHBoxLayout()
        hLO_suffix.addWidget(QLabel("Suffix: "))
        hLO_suffix.addWidget(self.suffix)
        vLO.addLayout(hLO_suffix)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vLO.addWidget(self.buttonBox)

        if format_string != '':
            prefix, format_str, suffix = get_format_strings(format_string)
            self.prefix.setText(prefix)
            self.suffix.setText(suffix)
            if '.' in format_str and 'f' in format_str:
                start = '.'
                end = 'f'
                dec_string = format_str[
                    format_str.find(start)+1:format_str.rfind(end)]
                self.cbox_decimals.setCurrentText(dec_string)
            if len(format_str) > 0:
                if format_str[0] == '0':
                    pos_dec = format_str.rfind('.')
                    if pos_dec != -1:
                        pad_string = format_str[1:pos_dec]
                    else:
                        pad_string = format_str[1:]
                    self.cbox_padding.setCurrentText(pad_string)

    def get_data(self):
        """Get formatting string.

        Returns
        -------
        return_string : str
            if prefix, suffix separated by | (prefix|formatstr|suffix)
            else only formatstr
            formatstr = string after {value in f-string to format the value
        """
        format_string = ''

        idx_dec = self.cbox_decimals.currentIndex()
        idx_pad = self.cbox_padding.currentIndex()
        if idx_dec > 0:
            format_string = '.' + self.cbox_decimals.currentText() + 'f'
        if idx_pad > 0:
            format_string = \
                '0' + self.cbox_padding.currentText() + format_string
        if format_string != '':
            format_string = ':' + format_string

        return_string = format_string
        if self.prefix != '' or self.suffix != '':
            return_string = '|'.join(
                [self.prefix.text(), format_string, self.suffix.text()])

        return return_string


class TagPatternEditDialog(QDialog):
    """Dialog for editing tag pattern for test DCM."""

    def __init__(
            self, initial_pattern=None, modality='CT',
            title='', typestr='format',
            accept_text='Use', reject_text='Cancel',
            save_blocked=False):
        super().__init__()

        self.edited = False
        self.current_modality = modality
        self.status_label = QLabel('')
        self.current_template = initial_pattern

        if typestr == 'format':
            self.fname = 'tag_patterns_format'
        elif typestr == 'sort':
            self.fname = 'tag_patterns_sort'
        else:
            self.fname = ''

        self.setWindowTitle(title)
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        vLO = QVBoxLayout()
        self.setLayout(vLO)
        hLO_temps = QHBoxLayout()
        vLO.addLayout(hLO_temps)
        hLO_temps.addWidget(QLabel('Select template:'))
        self.cbox_templist = QComboBox()
        self.cbox_templist.setFixedWidth(200)
        self.cbox_templist.activated.connect(
            self.update_clicked_template)
        if typestr != '':
            hLO_temps.addWidget(self.cbox_templist)
            tb = QToolBar()
            hLO_temps.addWidget(tb)
            actAdd = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
                'Add new tag pattern', self)
            actAdd.triggered.connect(self.add)
            actSave = QAction(
                QIcon(f'{os.environ[ENV_ICON_PATH]}save.png'),
                'Save tag pattern', self)
            actSave.triggered.connect(self.save)
            if save_blocked:
                actSave.setEnabled(False)
            tb.addActions([actAdd, actSave])
            hLO_temps.addStretch()

        self.wTagPattern = TagPatternWidget(self, typestr=typestr)
        vLO.addWidget(self.wTagPattern)

        ok, path, self.tag_infos = cff.load_settings(fname='tag_infos')
        self.update_from_yaml()

        vLO.addWidget(HLine())
        vLO.addWidget(self.status_label)

        hLOdlgBtns = QHBoxLayout()
        vLO.addLayout(hLOdlgBtns)
        hLOdlgBtns.addStretch()
        btnClose = QPushButton(accept_text)
        btnClose.clicked.connect(self.accept)
        hLOdlgBtns.addWidget(btnClose)
        btnCancel = QPushButton(reject_text)
        btnCancel.clicked.connect(self.reject)
        hLOdlgBtns.addWidget(btnCancel)

    def update_from_yaml(self):
        """Refresh settings from yaml file."""
        self.lastload = time()
        ok, path, self.templates = cff.load_settings(fname=self.fname)
        self.wTagPattern.fill_list_tags(self.current_modality)
        self.refresh_templist()

    def refresh_templist(self, selected_id=0, selected_label=''):
        """Update the list of templates, and self.current...

        Parameters
        ----------
        selected_id : int, optional
            index to select in template list. The default is 0.
        selected_label : str, optional
            label to select in template list (override index)
            The default is ''.
        """
        self.current_labels = \
            [obj.label for obj
                in self.templates[self.current_modality]]

        if selected_label != '':
            tempno = self.current_labels.index(selected_label)
        else:
            tempno = selected_id
        if tempno < 0:
            tempno = 0
        if tempno > len(self.current_labels)-1:
            tempno = len(self.current_labels)-1

        self.cbox_templist.blockSignals(True)
        self.cbox_templist.clear()
        self.cbox_templist.addItems(self.current_labels)
        if selected_label != '':
            self.update_current_template(selected_id=tempno)
            self.cbox_templist.setCurrentIndex(tempno)
        self.cbox_templist.blockSignals(False)
        self.wTagPattern.update_data()

    def update_clicked_template(self):
        """Update data after new template selected (clicked)."""
        if self.edited:
            res = QuestionBox(
                self, title='Save changes?',
                msg='Save changes before changing template?')
            if res.exec():
                self.save(label=self.current_template.label)

        tempno = self.cbox_templist.currentIndex()
        self.update_current_template(selected_id=tempno)
        self.wTagPattern.update_data()

    def update_current_template(self, selected_id=0):
        """Update self.current_template by label or id."""
        self.current_template = copy.deepcopy(
            self.templates[self.current_modality][selected_id])

    def get_pattern(self):
        """Get TagPattern from calling widget on Sort."""
        return self.current_template

    def add(self):
        """Add new template to list. Ask for new name and verify."""
        text, ok = QInputDialog.getText(
            self, 'New label',
            'Name the new tag pattern')
        if ok and text != '':
            if text in self.current_labels:
                QMessageBox.warning(
                    self, 'Label already in use',
                    'This label is already in use.')
            else:
                new_temp = copy.deepcopy(self.current_template)
                new_temp.label = text
                if self.templates[self.current_modality][0].label == '':
                    self.templates[self.current_modality][0] = new_temp
                else:
                    self.templates[self.current_modality].append(new_temp)

                self.current_template = new_temp
                self.current_labels.append(text)
                self.save(label=text)
                self.refresh_templist(selected_label=text)

    def save(self, label=None):
        """Save button pressed or specific save on label."""
        if self.current_template.label == '':
            self.add()
        else:
            if label is False or label is None:
                idx = self.cbox_templist.currentIndex()
            else:
                idx = self.current_labels.index(label)

            self.templates[self.current_modality][idx] = \
                copy.deepcopy(self.current_template)

            proceed = cff.test_config_folder(self)
            if proceed:
                proceed, errmsg = cff.check_save_conflict(
                    self.fname, self.lastload)
                if errmsg != '':
                    proceed = proceed_question(self, errmsg)
                if proceed:
                    ok, path = cff.save_settings(
                        self.templates, fname=self.fname)
                    if ok:
                        self.status_label.setText(
                            f'Changes saved to {path}')
                        self.flag_edit(False)
                    else:
                        QMessageBox.warning(
                            self, 'Failed saving', f'Failed saving to {path}')

    def flag_edit(self, flag=True):
        """Indicate some change."""
        if flag:
            self.edited = True
            self.status_label.setText('**Unsaved changes**')
        else:
            self.edited = False
            self.status_label.setText('')


class QuickTestTreeView(QTreeView):
    """QTreeWidget for list of images marked for testing."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.update_model()
        self.setModel(self.model)

    def update_model(self):
        """Set model headers based on current modality."""
        self.tests = QUICKTEST_OPTIONS[self.parent.current_modality]
        self.model = QStandardItemModel(0, len(self.tests) + 2, self.parent)
        self.model.setHeaderData(0, Qt.Horizontal, "Image label")
        self.model.setHeaderData(1, Qt.Horizontal, "Group label")

        for i in range(len(self.tests)):
            self.model.setHeaderData(i+2, Qt.Horizontal, self.tests[i])

        self.model.itemChanged.connect(self.parent.flag_edit)

    def update_modality(self):
        """Update model when modality change."""
        self.model.beginResetModel()
        self.model.clear()
        self.update_model()
        self.setModel(self.model)
        self.model.endResetModel()

    def update_data(self, set_selected=0):
        """Set data to self.parent.current_template.

        Parameters
        ----------
        set_selected : int
            Row number to set as selected when finished. Default is 0
        """
        self.model.beginResetModel()
        self.model.blockSignals(True)

        n_rows = self.model.rowCount()
        for i in range(n_rows):
            self.model.removeRow(n_rows-i-1, QModelIndex())

        temp = self.parent.current_template
        for im, img_tests in enumerate(temp.tests):
            self.model.insertRow(im)
            self.model.setData(self.model.index(im, 0),
                               temp.image_names[im], Qt.ItemIsEditable)
            self.model.setData(self.model.index(im, 1),
                               temp.group_names[im], Qt.ItemIsEditable)
            for t in range(len(self.tests)):
                state = (Qt.Checked if self.tests[t]
                         in img_tests else Qt.Unchecked)
                self.model.setData(self.model.index(im, t+2),
                                   state, role=Qt.CheckStateRole)
                item = self.model.itemFromIndex(self.model.index(im, t+2))
                item.setEditable(False)
                item.setCheckable(True)
        self.model.blockSignals(False)

        self.setColumnWidth(0, 170)
        self.setColumnWidth(1, 170)
        for i in range(len(self.tests)):
            self.setColumnWidth(i+2, 60)
        self.header().setStretchLastSection(False)
        self.model.endResetModel()
        self.setCurrentIndex(self.model.index(set_selected, 0))

    def insert_empty_row(self):
        """Insert empty row after selected or at end."""
        sel = self.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row() + 1
        else:
            rowno = self.model.rowCount()
        temp = self.get_data()
        if temp.tests == [[]] and temp.image_names == ['']:
            rowno = 0
        self.model.beginInsertRows(self.model.index(rowno, 0), rowno, rowno)
        self.model.insertRow(rowno)
        self.model.setData(self.model.index(rowno, 0), '', Qt.ItemIsEditable)
        self.model.setData(self.model.index(rowno, 1), '', Qt.ItemIsEditable)
        for t in range(len(self.tests)):
            self.model.setData(self.model.index(rowno, t+2),
                               Qt.Unchecked, role=Qt.CheckStateRole)
            item = self.model.itemFromIndex(self.model.index(rowno, t+2))
            item.setEditable(False)
            item.setCheckable(True)
        self.model.endInsertRows()
        self.parent.nimgs.setText(f'{self.model.rowCount()}')

    def delete_row(self):
        """Delete selected row."""
        sel = self.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            if self.model.rowCount() == 1:
                self.model.setData(
                    self.model.index(0, 0), '', Qt.ItemIsEditable)
                self.model.setData(
                    self.model.index(0, 1), '', Qt.ItemIsEditable)
                for t in range(len(self.tests)):
                    self.model.setData(self.model.index(0, t+2),
                                       Qt.Unchecked, role=Qt.CheckStateRole)
                    item = self.model.itemFromIndex(self.model.index(0, t+2))
                    item.setEditable(False)
                    item.setCheckable(True)
            else:
                temp = self.get_data()
                temp.tests.pop(rowno)
                temp.image_names.pop(rowno)
                temp.group_names.pop(rowno)
                self.parent.current_template = copy.deepcopy(temp)
                self.update_data(set_selected=rowno-1)
            self.parent.nimgs.setText(f'{self.model.rowCount()}')

    def get_data(self):
        """Read current settings as edited by user.

        Return
        ------
        temp : QuickTestTemplate
        """
        temp = cfc.QuickTestTemplate()
        temp.label = self.parent.current_template.label
        tests = []
        image_names = []
        group_names = []
        for im in range(self.model.rowCount()):
            item = self.model.itemFromIndex(self.model.index(im, 0))
            image_names.append(item.text())
            item = self.model.itemFromIndex(self.model.index(im, 1))
            group_names.append(item.text())
            img_tests = []
            for t in range(len(self.tests)):
                item = self.model.itemFromIndex(self.model.index(im, t + 2))
                if item.checkState() == Qt.Checked:
                    img_tests.append(self.tests[t])
            tests.append(img_tests)
        temp.tests = tests
        if image_names == []:
            image_names = ['']
        if group_names == []:
            group_names = ['']
        temp.image_names = image_names
        temp.group_names = group_names

        return temp


class QuickTestOutputTreeView(QTreeView):
    """QTreeWidget for list of output settings."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.update_model()
        self.setModel(self.model)

    def update_model(self):
        """Initialize model with headers."""
        self.model = QStandardItemModel(0, 6, self.parent)
        self.model.setHeaderData(0, Qt.Horizontal, "Test")
        self.model.setHeaderData(1, Qt.Horizontal, "Alternative")
        self.model.setHeaderData(2, Qt.Horizontal, "Columns")
        self.model.setHeaderData(3, Qt.Horizontal, "Calculation")
        self.model.setHeaderData(4, Qt.Horizontal, "Pr image or group")
        self.model.setHeaderData(5, Qt.Horizontal, "Header_")

        self.model.itemChanged.connect(self.parent.flag_edit)

    def update_data(self, set_selected=0):
        """Set data to self.parent.current_template.output.

        Parameters
        ----------
        set_selected : int
            Row number to set as selected when finished. Default is 0
        """
        self.model.beginResetModel()
        self.model.blockSignals(True)

        n_rows = self.model.rowCount()
        for i in range(n_rows):
            self.model.removeRow(n_rows-i-1, QModelIndex())

        temp = self.parent.current_template.output
        r = 0
        if temp.tests != {}:
            for testcode, sett in temp.tests.items():
                for s, sub in enumerate(sett):
                    self.model.insertRow(r)
                    self.model.setData(self.model.index(r, 0), testcode)
                    try:
                        text_alt = ALTERNATIVES[
                            self.parent.current_modality][
                                testcode][sub.alternative]
                    except IndexError:
                        # supplement table starting from 10
                        text_alt = ALTERNATIVES[
                            self.parent.current_modality][
                                testcode][sub.alternative - 10]
                    except KeyError:
                        text_alt = '-'
                    self.model.setData(self.model.index(r, 1), text_alt)
                    if len(sub.columns) > 0:
                        text_col = str(sub.columns)
                    else:
                        text_col = 'all'
                    self.model.setData(self.model.index(r, 2), text_col)
                    self.model.setData(self.model.index(r, 3), sub.calculation)
                    text_pr = 'Per group' if sub.per_group else 'Per image'
                    self.model.setData(self.model.index(r, 4), text_pr)
                    self.model.setData(self.model.index(r, 5), sub.label)
                    r += 1

        self.setColumnWidth(0, 70)
        self.setColumnWidth(3, 110)

        self.model.blockSignals(False)
        self.model.endResetModel()
        self.setCurrentIndex(self.model.index(set_selected, 0))

    def get_testcode_subno(self, rowno):
        """Get test_code and QuickTestOutputSub number from row number.

        Parameters
        ----------
        rowno : int
            row number in treeview

        Returns
        -------
        test_code : str
        subno : int
        """
        test_codes = []
        subnos = []
        for test_code, subs in self.parent.current_template.output.tests.items():
            s = 0
            for sub in subs:
                test_codes.append(test_code)
                subnos.append(s)
                s += 1

        test_code = test_codes[rowno]
        subno = subnos[rowno]

        return (test_code, subno)

    def edit_row(self):
        """Edit selected row."""
        sel = self.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            code, subno = self.get_testcode_subno(rowno)
            dlg_sub = QuickTestOutputSubDialog(
                self.parent.current_template,
                qt_output_sub=self.parent.current_template.output.tests[code][subno],
                modality=self.parent.current_modality,
                initial_testcode=code)
            res = dlg_sub.exec()
            if res:
                new_sub = copy.deepcopy(dlg_sub.get_data())
                if new_sub is None:
                    QMessageBox.warning(
                        self.parent, 'Ignored',
                        'No table columns selected. Edit ignored.')
                else:
                    self.parent.current_template.output.tests[
                        code][subno] = new_sub
                    self.update_data()
                    self.parent.flag_edit(True)

    def insert_row(self):
        """Insert row after selected if same test_code, else end of same testcode."""
        sel = self.selectedIndexes()
        code = ''
        subno = -1
        if len(sel) > 0:
            rowno = sel[0].row()
            code, subno = self.get_testcode_subno(rowno)
        dlg_sub = QuickTestOutputSubDialog(
            self.parent.current_template,
            modality=self.parent.current_modality)
        res = dlg_sub.exec()
        if res:
            testcode = dlg_sub.get_testcode()
            new_sub = copy.deepcopy(dlg_sub.get_data())
            if new_sub is None:
                QMessageBox.warning(
                    self.parent, 'Ignored',
                    'No table columns selected. Ignored input.')
            else:
                if testcode == code:
                    self.parent.current_template.output.tests[
                        testcode].insert(subno + 1, new_sub)
                else:
                    try:
                        self.parent.current_template.output.tests[
                            testcode].append(new_sub)
                    except KeyError:
                        self.parent.current_template.output.tests[
                            testcode] = [new_sub]
                self.update_data()
                self.parent.flag_edit(True)

    def delete_row(self):
        """Delete selected row."""
        sel = self.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            code, subno = self.get_testcode_subno(rowno)
            self.parent.current_template.output.tests[code].pop(subno)
            self.update_data(set_selected=rowno - 1)
            self.parent.flag_edit(True)


class QuickTestOutputSubDialog(QDialog):
    """Dialog to set QuickTestOutputSub."""

    def __init__(self, paramset, qt_output_sub=None, modality='CT',
                 initial_testcode=''):
        """Initialize QuickTestOutputSubDialog.

        Parameters
        ----------
        paramset : dict
            paramsets
        qt_output_sub : object, optional
            input QuickTestOutputSub. The default is None.
        modality : str, optional
            current modality from parent window. The default is 'CT'.
        initial_testcode : str, optional
            testcode selected from start (if edit existing sub)
        """
        super().__init__()

        self.setWindowTitle('QuickTestOutput details')
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        if qt_output_sub is None:
            qt_output_sub = cfc.QuickTestOutputSub()
        self.qt_output_sub = qt_output_sub
        self.paramset = paramset
        self.modality = modality

        self.cbox_testcode = QComboBox()
        self.cbox_alternatives = QComboBox()
        self.cbox_table = QComboBox()
        self.cbox_table.addItems(['Result table', 'Supplement_table'])
        self.list_columns = QListWidget()
        self.cbox_calculation = QComboBox()
        self.chk_per_group = BoolSelect(
            self, text_true='per group', text_false='per image')
        self.txt_header = QLineEdit('')

        self.cbox_testcode.addItems(QUICKTEST_OPTIONS[modality])
        if initial_testcode != '':
            self.cbox_testcode.setCurrentText(initial_testcode)
            self.cbox_testcode.setEnabled(False)
        else:
            self.cbox_testcode.setCurrentIndex(0)

        self.cbox_testcode.currentIndexChanged.connect(
            lambda: self.update_data(update_calculations=False))
        self.cbox_alternatives.currentIndexChanged.connect(
            lambda: self.update_data(
                update_alternatives=False, update_calculations=False))
        self.cbox_table.currentIndexChanged.connect(
            lambda: self.update_data(
                update_alternatives=False, update_calculations=False))

        vLO = QVBoxLayout()
        self.setLayout(vLO)

        self.suplement_txt = 'Supplement table'

        fLO = QFormLayout()
        fLO.addRow(QLabel('Test:'), self.cbox_testcode)
        fLO.addRow(QLabel('Alternative:'), self.cbox_alternatives)
        fLO.addRow(QLabel('Table:'), self.cbox_table)
        fLO.addRow(QLabel('Columns:'), self.list_columns)
        fLO.addRow(QLabel('Calculation:'), self.cbox_calculation)
        fLO.addRow(LabelItalic(
            '    ignored if any of the values are strings'))
        fLO.addRow(QLabel(''), self.chk_per_group)
        fLO.addRow(QLabel('Header:'), self.txt_header)
        fLO.addRow(LabelItalic(
            '    header_imagelabel or header_grouplabel, ignored if = and > 1 column'))

        vLO.addLayout(fLO)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vLO.addWidget(self.buttonBox)

        self.update_data(first=True)

    def update_data(self, update_alternatives=True, update_columns=True,
                    update_calculations=True, first=False):
        """Set visuals to input data and refresh lists if selections change.

        Parameters
        ----------
        update_alternatives : bool, optional
            Update list of alternatives. The default is True.
        update_columns : bool, optional
            Update list of columnheaders. The default is True.
        update_calculations : bool, optional
            Update list of calculation options. The default is True.
        first : bool, optional
            First time update. The default is False.
        """
        testcode = self.cbox_testcode.currentText()

        if update_alternatives:  # fill text in alternatives cbox
            try:
                alts = ALTERNATIVES[self.modality][testcode]
            except KeyError:
                alts = ['-']
            '''
            TODO delete? new solution not fully tested
            try:
                altsup = HEADERS_SUP[self.modality][testcode]#['altSup']
                if len(altsup) > 0:
                    # add alternative sup not just sup if not supAll
                    if alts[0] == '-':
                        alts = ['Results table', self.suplement_txt]
                    else:
                        alts.append(self.suplement_txt)
            except KeyError:
                pass
            '''
            self.cbox_alternatives.blockSignals(True)
            self.cbox_alternatives.clear()
            self.cbox_alternatives.addItems(alts)
            if self.qt_output_sub.alternative < 9:
                self.cbox_alternatives.setCurrentIndex(
                    self.qt_output_sub.alternative)
            else:  # supplement table
                self.cbox_alternatives.setCurrentIndex(
                    self.qt_output_sub.alternative - 10)
            self.cbox_alternatives.blockSignals(False)
        if update_columns:  # fill text in columns
            cols = []
            idx_alt = self.cbox_alternatives.currentIndex()
            if self.cbox_table.currentIndex() == 1:
                if testcode in HEADERS_SUP[self.modality]:
                    if 'altAll' in HEADERS_SUP[self.modality][testcode]:
                        cols = HEADERS_SUP[self.modality][testcode]['altAll']
                    elif 'alt0' in HEADERS_SUP[self.modality][testcode]:
                        cols = HEADERS_SUP[self.modality][testcode]['alt'+str(idx_alt)]
            else:
                try:
                    cols = HEADERS[self.modality][testcode]['alt'+str(idx_alt)]
                except KeyError:
                    if testcode == 'DCM':
                        cols = self.paramset.dcm_tagpattern.list_tags
                    elif testcode == 'CTn':
                        cols = self.paramset.ctn_table.materials
            self.list_columns.clear()
            if len(cols) > 0:
                self.list_columns.addItems(cols)
                # set checkable
                subcols = self.qt_output_sub.columns
                if len(subcols) == 0:
                    subcols = [i for i in range(self.list_columns.count())]
                for i in range(self.list_columns.count()):
                    item = self.list_columns.item(i)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    if first:
                        if i in subcols:
                            item.setCheckState(Qt.Checked)
                        else:
                            item.setCheckState(Qt.Unchecked)
                    else:
                        item.setCheckState(Qt.Checked)
        if update_calculations:  # fill and set default calculation option
            self.cbox_calculation.addItems(CALCULATION_OPTIONS)
            self.cbox_calculation.setCurrentText(
                self.qt_output_sub.calculation)
        if first:
            self.chk_per_group.setChecked(self.qt_output_sub.per_group)
            self.txt_header.setText(self.qt_output_sub.label)

    def get_testcode(self):
        """Get selected testcode.

        Return
        ------
        testcode: str
        """
        return self.cbox_testcode.currentText()

    def get_data(self):
        """Get settings from dialog as QuickTestOutputSub.

        Returns
        -------
        qtsub : QuickTestOutputSub
        """
        qtsub = cfc.QuickTestOutputSub()
        qtsub.label = self.txt_header.text()

        if self.cbox_table.currentIndex() == 1:
            qtsub.alternative = self.cbox_alternatives.currentIndex() + 10
        else:
            qtsub.alternative = self.cbox_alternatives.currentIndex()
        cols = []
        for i in range(self.list_columns.count()):
            if self.list_columns.item(i).checkState() == Qt.Checked:
                cols.append(i)
        qtsub.columns = cols
        qtsub.calculation = self.cbox_calculation.currentText()
        qtsub.per_group = self.chk_per_group.isChecked()

        if len(cols) == 0:
            qtsub = None

        return qtsub


class DicomCritAddDialog(QDialog):
    """Dialog to add dicom criteria for automation."""

    def __init__(self, parent, attr_name='', value=''):
        super().__init__()
        self.parent = parent

        self.setWindowTitle('Add DICOM criteria')
        self.setWindowIcon(QIcon(f'{os.environ[ENV_ICON_PATH]}iQC_icon.png'))
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.cbox_tags = QComboBox()
        self.txt_value = QLineEdit('')

        vLO = QVBoxLayout()
        self.setLayout(vLO)

        hLO_tags = QHBoxLayout()
        vLO.addLayout(hLO_tags)
        hLO_tags.addWidget(QLabel('Attribute name: '))
        hLO_tags.addWidget(self.cbox_tags)
        general_tags, included_tags = get_included_tags(
            self.parent.parent.current_modality, self.parent.parent.tag_infos)
        self.cbox_tags.addItems(included_tags)
        if attr_name != '':
            self.cbox_tags.setCurrentText(attr_name)
            self.txt_value.setText(value)

        hLO_values = QHBoxLayout()
        vLO.addLayout(hLO_values)
        hLO_values.addWidget(QLabel('Value string'))
        hLO_values.addWidget(self.txt_value)
        self.txt_value.setMinimumWidth(200)

        vLO.addWidget(QLabel('Leave value empty to get value from sample DICOM file'))
        vLO.addWidget(QLabel(
            'Wildcard possible: ? single character / * multiple characters'))

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vLO.addWidget(self.buttonBox)

    def get_data(self):
        """Get attribute name and value.

        Returns
        -------
        attributename : str
        value : str

        """
        return (self.cbox_tags.currentText(), self.txt_value.text())


class DicomCritWidget(QWidget):
    """Widget for dicom_crit in automation templates + toolbar to edit."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.hLO = QHBoxLayout()
        self.setLayout(self.hLO)

        self.tableCrit = QTreeWidget()
        self.tableCrit.setColumnCount(2)
        self.tableCrit.setColumnWidth(0, 200)
        self.tableCrit.setColumnWidth(1, 150)
        self.tableCrit.setHeaderLabels(['Attribute name', 'Value'])
        self.tableCrit.setMinimumSize(350, 200)
        self.tableCrit.setRootIsDecorated(False)
        self.hLO.addWidget(self.tableCrit)

        tb = QToolBar()
        tb.setOrientation(Qt.Vertical)
        actAdd = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}add.png'),
            'Add criterion', self)
        actAdd.triggered.connect(self.add)
        actEdit = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}edit.png'),
            'Edit criterion', self)
        actEdit.triggered.connect(self.edit)
        actDelete = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}delete.png'),
            'Delete selected criterion/row', self)
        actDelete.triggered.connect(self.delete)
        actClear = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}clear.png'),
            'Clear table', self)
        actClear.triggered.connect(self.clear)
        tb.addActions([actAdd, actDelete, actClear])
        self.hLO.addWidget(tb)

    def add(self):
        """Add new criterion."""
        dlg = DicomCritAddDialog(self)
        res = dlg.exec()
        if res:
            attr_name, value = dlg.get_data()
            proceed = True
            if attr_name in self.parent.current_template.dicom_crit_attributenames:
                res = QuestionBox(
                    self.parent, title='Replace?',
                    msg='Attribute name already in table. Replace?')
                if res.exec():
                    proceed = True
                else:
                    proceed = False
            if proceed:
                self.parent.current_template.dicom_crit_attributenames.append(
                    attr_name)
                self.parent.current_template.dicom_crit_values.append(value)
                self.update_data()
                self.parent.flag_edit()

    def edit(self):
        """Edit DICOM criterion."""
        sel = self.tableCrit.selectedIndexes()
        if len(sel) > 0:
            rowno = sel[0].row()
            dlg = DicomCritAddDialog(self, attr_name='', value='')
            res = dlg.exec()
            if res:
                attr_name, value = dlg.get_data()
                already_other = self.parent.current_template.dicom_crit_attributenames
                already_other.pop(rowno)
                if attr_name in already_other:
                    QMessageBox.warning(
                        self.parent, 'Ignored',
                        'Attribute name already in table. Edit that row.')
                else:
                    self.parent.current_template.dicom_crit_attributenames[
                        rowno] = attr_name
                    self.parent.current_template.dicom_crit_values[rowno] = value
                    self.update_data()
                    self.parent.flag_edit()
        else:
            self.add()

    def delete(self):
        """Delete selected criterion."""
        sels = self.tableCrit.selectedIndexes()
        if len(sels) > 0:
            if len(sels) == len(self.parent.current_template.dicom_crit_attributenames):
                self.clear()
            else:
                delrows = [sel.row() for sel in sels]
                delrows.sort(reverse=True)
                row = 0
                for row in delrows:
                    self.parent.current_template.dicom_crit_attributenames.pop(row)
                    self.parent.current_template.dicom_crit_values.pop(row)
                self.update_data(set_selected=row)
                self.parent.flag_edit()

    def clear(self):
        """Clear all criteria."""
        self.parent.current_template.dicom_crit_attributenames = []
        self.parent.current_template.dicom_crit_values = []
        self.update_data()
        self.parent.flag_edit()

    def update_data(self, set_selected=0):
        """Update tableCrit with data from current_template."""
        self.tableCrit.clear()
        if len(self.parent.current_template.dicom_crit_attributenames) > 0:
            attr_names = self.parent.current_template.dicom_crit_attributenames
            vals = self.parent.current_template.dicom_crit_values
            for rowno, attr_name in enumerate(attr_names):
                row_strings = [attr_name, vals[rowno]]
                item = QTreeWidgetItem(row_strings)
                self.tableCrit.addTopLevelItem(item)

            self.tableCrit.setCurrentItem(
                self.tableCrit.topLevelItem(set_selected))


class CheckCell(QCheckBox):
    """CheckBox for use in TreeWidget cells."""

    def __init__(self, parent, initial_value=True):
        super().__init__()
        '''
        self.setStyleSheet(QCheckBox {
            padding-left: 15px;
            padding-bottom: 2px;
            })
        '''
        self.setStyleSheet('''QCheckBox {
            margin-left:50%;
            margin-right:50%;
            }''')
        self.setChecked(initial_value)
        self.parent = parent
        self.clicked.connect(self.parent.flag_edit)
    '''
    def focusInEvent(self, event):
        """Trigger cell selection changed event."""
        self.parent.cell_selection_changed()
        super().focusInEvent(event)
    '''


class LineCell(QLineEdit):
    """LineEdit for use in TreeWidget cells."""

    def __init__(self, parent, initial_text=''):
        super().__init__()
        self.parent = parent
        self.setText(initial_text)
        self.textEdited.connect(self.parent.flag_edit)
    '''
    def focusInEvent(self, event):
        """Trigger cell selection changed event."""
        self.parent.cell_selection_changed()
        super().focusInEvent(event)
    '''


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

        hLO = QHBoxLayout()
        group = QButtonGroup()
        group.setExclusive(True)
        group.addButton(self.btn_true)
        group.addButton(self.btn_false)
        hLO.addWidget(self.btn_true)
        hLO.addWidget(self.btn_false)
        self.setLayout(hLO)

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
        """Handle item pressed.

        Parameters
        ----------
        index : int
            index pressed.
        """
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
        """Get array of checked ids.

        Return
        ------
        ids: list of int
        """
        ids = []
        for i in range(self.count()):
            if self.itemChecked(i):
                ids.append(i)

        return ids


class PlotDialog(ImageQCDialog):
    """Dialog for plot."""

    def __init__(self, main, title=''):
        super().__init__()
        self.setWindowTitle(title)
        vLO = QVBoxLayout()
        self.setLayout(vLO)
        self.plotcanvas = PlotCanvas(main)
        vLO.addWidget(PlotWidget(main, self.plotcanvas))


class PlotWidget(QWidget):
    """Widget with plot."""

    def __init__(self, main, plotcanvas):
        super().__init__()
        self.main = main

        self.plotcanvas = plotcanvas
        tbPlot = PlotNavigationToolbar(self.plotcanvas, self)
        self.hlo = QHBoxLayout()
        vlo_tb = QVBoxLayout()
        self.hlo.addLayout(vlo_tb)

        tbPlot_copy = QToolBar()
        actCopy = QAction(
            QIcon(f'{os.environ[ENV_ICON_PATH]}copy.png'),
            'Copy curve as table to clipboard', self)
        actCopy.triggered.connect(self.copy_curves)
        tbPlot_copy.setOrientation(Qt.Vertical)
        tbPlot_copy.addActions([actCopy])

        vlo_tb.addWidget(tbPlot_copy)
        vlo_tb.addWidget(tbPlot)
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

        df = pd.DataFrame(values)
        df = df.transpose()
        df.columns = headers
        df.to_clipboard(index=False, excel=True)
        self.main.statusBar.showMessage('Values in clipboard', 2000)


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
            self.ax.set_title(f'Profile lengt:h {length:.1f} {unit}')

        self.draw()


class ResetAutoTemplateDialog(ImageQCDialog):
    """Dialog to move directories/files in input_path/Archive to input_path."""

    def __init__(self, parent_widget, files=[], directories=[]):
        super().__init__()
        self.setWindowTitle('Reset Automation template')
        self.setMinimumHeight(300)
        self.setMinimumWidth(300)
        files_or_folders = 'files'
        if len(files) > 0:
            self.list_elements = [file.name for file in files]
        else:
            self.list_elements = [folder.name for folder in directories]
            files_or_folders = 'folders'

        vLO = QVBoxLayout()
        self.setLayout(vLO)

        self.list_file_or_dirs = QListWidget()
        self.list_file_or_dirs.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_file_or_dirs.addItems(self.list_elements)

        vLO.addWidget(QLabel(
            'Move files out of Archive to regard these files as incoming.'))
        vLO.addStretch()
        vLO.addWidget(QLabel(f'List of {files_or_folders} in Archive:'))
        vLO.addWidget(self.list_file_or_dirs)
        vLO.addStretch()
        hLO_buttons = QHBoxLayout()
        vLO.addLayout(hLO_buttons)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        vLO.addWidget(self.buttonBox)


    def get_idxs(self):
        """Return selected elements in list.

        Returns
        -------
        idxs : list of int
            selected indexes
        """
        idxs = []
        for sel in self.list_file_or_dirs.selectedIndexes():
            idxs.append(sel.row())

        return idxs


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
