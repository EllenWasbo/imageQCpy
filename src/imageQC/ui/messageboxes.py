#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QMessageBoxes with specific settings for different uses and reuses in ImageQC.

@author: Ellen Wasbo
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QLabel, QPushButton

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
        QMessageBox.Question,
        'Proceed?', question,
        buttons=QMessageBox.Yes | QMessageBox.No,
        parent=widget
        )
    msg_box.setDefaultButton(QMessageBox.No)
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
    msg_box.exec_()
    reply = msg_box.standardButton(msg_box.clickedButton())
    if reply == QMessageBox.Yes:
        proceed = True

    return proceed


class MessageBoxWithDetails(QMessageBox):
    """QMessageBox with details and richtext for shorter definition when reused."""

    def __init__(self, parent=None, title='', msg='', info='', details=[],
                 msg_width=700, icon=QMessageBox.Information):
        super().__init__()
        self.setIcon(icon)
        self.setWindowTitle(title)
        self.setText(msg)
        if info != '':
            self.setInformativeText(info)
        self.setTextFormat(Qt.RichText)
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
            info='', details=[], default_yes=False):
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
        if info != '':
            self.setInformativeText(info)
        if details != []:
            self.setDetailedText('\n'.join(details))
        _qlabels = self.findChildren(QLabel)
        _qlabels[1].setFixedWidth(msg_width)
        self.addButton(no_text, QMessageBox.RejectRole)
        self.addButton(yes_text, QMessageBox.AcceptRole)
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
