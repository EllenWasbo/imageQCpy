#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QMessageBoxes with specific settings for different uses and reuses in ImageQC.

@author: Ellen Wasbo
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QLabel, QPushButton

# imageQC block start
# imageQC block end


def proceed_question(widget, question,
                     msg_width=500, detailed_text=None, info_text=''):
    """Ask a question whether to proceed with some process.

    Parameters
    ----------
    widget : QWidget
    question : str
    msg_width : int
        label width in pixels. Default is 500.
    detailed_text : str or list of str
        add detailed text if not empty string. Default is empty string.
    info_text : str
        subtext after question. Default is empty string.

    Returns
    -------
    proceed : bool
        yes = true
    """
    proceed = False

    msg_box = QMessageBox(
        QMessageBox.Icon.Question,
        'Proceed?', question,
        buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        parent=widget
        )
    msg_box.setDefaultButton(QMessageBox.StandardButton.No)
    if detailed_text is not None:
        if isinstance(detailed_text, list):
            detailed_text = '\n'.join(detailed_text)
        msg_box.setDetailedText(detailed_text)
    if info_text != '':
        msg_box.setInformativeText(info_text)
    msg_box.setStyleSheet(
        f"""
        QPushButton {{
            padding: 5px;
            }}
        QLabel {{
            width: {msg_width}px;
            }}
        """)
    msg_box.exec()
    reply = msg_box.standardButton(msg_box.clickedButton())
    if reply == QMessageBox.StandardButton.Yes:
        proceed = True

    return proceed


class MessageBoxWithDetails(QMessageBox):
    """QMessageBox with details and richtext for shorter definition when reused."""

    def __init__(self, parent=None, title='', msg='', info='', details=[],
                 msg_width=700, icon=QMessageBox.Icon.Information):
        super().__init__()
        self.setIcon(icon)
        self.setWindowTitle(title)
        self.setText(msg)
        if info != '':
            self.setInformativeText(info)
        self.setTextFormat(Qt.TextFormat.RichText)
        if details != []:
            self.setDetailedText('\n'.join(details))
        _qlabels = self.findChildren(QLabel)
        _qlabels[1].setFixedWidth(msg_width)
        '''
        self.setStyleSheet(
            f"""
            QPushButton {{
                padding: 5px;
                }}
            QLabel {{
                width: {msg_width}px;
                }}
            """)
        '''


class QuestionBox(QMessageBox):
    """QMessageBox with changed yes no text as options."""

    def __init__(
            self, parent=None, title='?', msg='?',
            yes_text='Yes', no_text='No', msg_width=500,
            info='', details=[], default_yes=False, cancel=False):
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
        cancel : bool, optional
            if True = include Cancel button. The default is False.

        Returns
        -------
        None.

        """
        super().__init__(
            QMessageBox.Icon.Question, title, msg, parent=parent)
        self.setIcon(QMessageBox.Icon.Question)
        self.setWindowTitle(title)
        self.setText(msg)
        self.setTextFormat(Qt.TextFormat.RichText)
        if info != '':
            self.setInformativeText(info)
        if details != []:
            self.setDetailedText('\n'.join(details))
        _qlabels = self.findChildren(QLabel)
        _qlabels[1].setFixedWidth(msg_width)
        self.no = self.addButton(no_text, QMessageBox.ButtonRole.NoRole)
        self.yes = self.addButton(yes_text, QMessageBox.ButtonRole.YesRole)
        if cancel:
            self.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
        _qbuttons = self.findChildren(QPushButton)
        if default_yes:
            _qbuttons[0].setAutoDefault(False)
            _qbuttons[0].setDefault(False)
            _qbuttons[1].setAutoDefault(True)
            _qbuttons[1].setDefault(True)
        else:
            _qbuttons[1].setAutoDefault(False)
            _qbuttons[1].setDefault(False)
            _qbuttons[0].setAutoDefault(True)
            _qbuttons[0].setDefault(True)
        self.setStyleSheet(
            f"""
            QPushButton {{
                padding: 5px;
                }}
            QLabel {{
                width: {msg_width}px;
                }}
            """)
