#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initiating and running the dash app.

@author: Ellen Wasbo
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from waitress import serve


dash_app = dash.Dash(__name__, suppress_callback_exceptions=True, use_pages=True)

dash_app.layout = html.Div([
    html.Div(
        [
            dcc.Link(f"{page['name']}  ", href=page["relative_path"])
            for page in dash.page_registry.values()
        ], className="row",
    ),
    # html.Div(['Header and logo']),
    dash.page_container
])

def run_dash_app(host='127.0.0.2', port=8082):
#if __name__ == '__main__':
    #logger = logging.getLogger('imageQC')
    #logger.setLevel(logging.ERROR)
    serve(dash_app.server, host=host, port=port)

