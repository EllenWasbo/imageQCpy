#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initiating and running the dash app.

@author: Ellen Wasbo
"""
from __future__ import annotations
from dataclasses import dataclass, field
import sys
import logging
import pandas as pd
try:
    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output
    import dash_bootstrap_components as dbc
    from waitress import serve
except (ImportError, ModuleNotFoundError) as err:
    print(f'Warning: {err}')
    print('To run the dash application you will need to install dash '
          '+ dash_bootstrap_components + waitress.')
    if "cannot import name 'json' from 'itsdangerous'" in str(err):
        print("You might have to downgrade itsdangerous: "
              "pip install itsdangerous==2.0.1 --force-reinstall")
    if "cannot import name 'BaseResponse' from 'werkzeug.wrappers'" in str(err):
        print("You might have to downgrade werkzeug: "
              "pip install werkzeug==2.0.3 --force-reinstall")


from PyQt5.QtWidgets import QMessageBox

# imageQC block start
from imageQC.config import config_func as cff
# imageQC block end


def tab_overview(modality_dict):
    cols = []
    for mod, temp_list in modality_dict.items():
        cols.append(dbc.Col(
            html.Div(f'Table of {mod}')
            ))
    #return html.Div([dbc.Row(cols)])
    return html.Div("test")#dbc.Row(cols))


def tab_modality(temp_list):
    return html.Div([
        dbc.Row([
            dbc.Col(
                html.Div("Template list")
                ),
            dbc.Col(
                html.Div("Template content")
                ),
            ]),
        ])


def build_tabs(modality_dict):
    list_of_tabs = [dbc.Tab(html.Div("test"), label="Overview")]
    for mod, temp_list in modality_dict.items():
        list_of_tabs.append(dbc.Tab(label=mod))
    return html.Div([dbc.Tabs(list_of_tabs)])


def generate_layout(dash_app, dash_settings, modality_dict=None):
    """Build the overall layout structure."""
    dash_app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.Img(src=dash_settings.url_logo), width=2),
            dbc.Col(html.H1(dash_settings.header)),
            ]),
        dbc.Row(build_tabs(modality_dict)),
        ])


@dataclass
class Template:
    """Class holding template settings and values."""

    label: str = ''
    visualization_label: str = ''
    min_max: list = field(default_factory=list)
    data: dict = field(default_factory=dict)


def get_data():
    """Extract data from result files combined with automation templates."""
    modality_dict = {}
    _, _, auto_templates = cff.load_settings(fname='auto_templates')
    _, _, auto_vendor_templates = cff.load_settings(fname='auto_vendor_templates')
    all_templates = [auto_templates, auto_vendor_templates]
    for at in all_templates:
        for mod, template_list in at.items():
            if mod not in [*modality_dict]:
                modality_dict[mod] = []
            for temp in template_list:
                if temp.label != '' and temp.path_output != '' and temp.active:
                    proceed = True
                    try:
                        if temp.import_only:
                            proceed = False
                    except AttributeError:
                        pass

                    df = None
                    #breakpoint()
                    #df = pd.read_csv(temp.path_output, sep='/t')
                    #if df.index.size > 1:
                    temp_this = Template(
                        label=temp.label,
                        visualization_label=temp.visualization_label,
                        min_max=temp.min_max,
                        data=df)
                    modality_dict[mod].append(temp_this)
    return modality_dict


def run_dash_app(widget=None):
    #logger = logging.getLogger('imageQC')
    #logger.setLevel(logging.ERROR)

    #try:
    dash_app = dash.Dash(
        __name__, suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.YETI])
    '''
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
    )
    '''
    modality_dict = get_data()
    #modality_dict = {}
    dash_settings = widget.templates
    generate_layout(dash_app, dash_settings, modality_dict=modality_dict)
    serve(dash_app.server, host=dash_settings.host, port=dash_settings.port)
    '''
    except:
        QMessageBox.warning(
            widget, 'Failed running dash',
            'Probably due to missing installed packages (dash + waitress).')
    '''

