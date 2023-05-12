#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""imageQC - startup and GUI of MainWindow.

@author: EllenWasbo
url: https://github.com/EllenWasbo/imageQC
"""
import sys
import os
from pathlib import Path
import logging

from PyQt5.QtGui import QPixmap, QFont, QFontMetrics, QPalette, QColor
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtCore import Qt

# imageQC block start
import imageQC.resources
import imageQC.ui.ui_main as ui_main
from imageQC.ui.ui_dialogs import StartUpDialog
import imageQC.config.config_func as cff
from imageQC.config.iQCconstants import (
    ENV_ICON_PATH, ENV_USER_PREFS_PATH, ENV_CONFIG_FOLDER, LOG_FILENAME
    )
import imageQC.scripts.automation as automation
# imageQC block end


def prepare_debug():
    """Set a tracepoint in PDB that works with Qt."""
    # https://stackoverflow.com/questions/1736015/debugging-a-pyqt4-app
    from PyQt5.QtCore import pyqtRemoveInputHook
    import pdb
    import sys
    pyqtRemoveInputHook()
    # set up the debugger
    debugger = pdb.Pdb()
    debugger.reset()
    # custom next to get outside of function scope
    debugger.do_next(None)  # run the next command
    users_frame = sys._getframe().f_back  # frame where user invoked `pyqt_set_trace()`
    debugger.interaction(users_frame, None)


if __name__ == '__main__':
    # prepare_debug()  # TODO - activate (not needen when not debugging)
    user_prefs_status, user_prefs_path, user_prefs = cff.load_user_prefs()
    # verify that config_folder exists
    if user_prefs.config_folder != '':
        if not os.path.exists(user_prefs.config_folder):
            print(
                f'Config folder do not exist.({user_prefs.config_folder})', flush=True)
            print('Config folder will be unlinked.', flush=True)
            user_prefs.config_folder = ''

    os.environ[ENV_USER_PREFS_PATH] = user_prefs_path
    os.environ[ENV_ICON_PATH] = cff.get_icon_path(user_prefs.dark_mode)
    os.environ[ENV_CONFIG_FOLDER] = user_prefs.config_folder

    log_mode = 'w'
    if os.environ[ENV_CONFIG_FOLDER] == '':
        ok, path, auto_common = cff.load_settings(fname='auto_common')
        if ok:
            log_mode = auto_common.log_mode
    parent_folder = Path(user_prefs_path).parent
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(
                os.path.join(parent_folder, LOG_FILENAME), mode=log_mode),
            logging.StreamHandler(sys.stdout)
        ],
    )

    if len(sys.argv) > 1:
        if os.environ[ENV_CONFIG_FOLDER] == '':
            print('Config folder not specified. Run GUI version to configure imageQC.')
        else:
            automation.run_automation_non_gui(sys.argv)

        sys.exit('Program exits')
    else:
        print('imageQC is starting up...', flush=True)
        # to set taskbar icon correctly for windows
        try:
            from ctypes import windll  # Only exists on Windows.
            myappid = 'sus.imageQC.app.3'
            windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except ImportError:
            pass

        app = QApplication(sys.argv)
        screen = app.primaryScreen()
        sz = screen.geometry()

        splash_img = QPixmap(':/icons/iQC_splash.png')
        splash = QSplashScreen(
            splash_img, Qt.WindowStaysOnTopHint)
        splash.show()

        app.setStyle('Fusion')
        if user_prefs.dark_mode:
            gb_background = 'background-color: #484848;'
            hover_background = 'background-color: #585858;'
            pressed_background = 'background-color: #686868;'
        else:
            gb_background = 'background-color: #e7e7e7;'
            hover_background = 'background-color: #d7d7d7;'
            pressed_background = 'background-color: #c7c7c7;'
        app.setStyleSheet(
            f"""QSplitter::handle:horizontal {{
                width: 4px;
                background-color: #6e94c0;
                }}
            QSplitter::handle:vertical {{
                height: 4px;
                background-color: #6e94c0;
                }}
            QWidget {{
                padding: 2px;
                }}
            QGroupBox {{
                {gb_background}
                border-radius: 5px;
                border: 1px solid grey;
                margin-top: 10px;
                }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding-left: 10px;
                padding-top: -7px;
                font-style: italic;
                }}
            QPushButton {{
                border-style: solid;
                border-width: 2px;
                border-color: #888888;
                border-radius: 10px;
                padding: 6px;
                {gb_background}
                }}
            QPushButton::hover {{
                {hover_background}
                }}
            QPushButton:pressed {{
                {pressed_background}
                }}""")
        myFont = QFont()
        myFont.setPointSize(user_prefs.font_size)
        app.setFont(myFont)
        font_metric = QFontMetrics(myFont)
        char_width = font_metric.averageCharWidth()

        if user_prefs.dark_mode:
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(25, 25, 25))
            palette.setColor(
                QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, Qt.black)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(
                QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.black)
            app.setPalette(palette)

        if os.environ[ENV_USER_PREFS_PATH] == '':
            dlg = StartUpDialog()
            dlg.show()
            splash.finish(dlg)
            dlg.exec()
        w = ui_main.MainWindow(scX=sz.width(), scY=sz.height(), char_width=char_width)
        w.show()
        splash.finish(w)
        app.exec()
