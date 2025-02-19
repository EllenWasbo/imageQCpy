#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone app for displaying trend data from automated results.
Prepared for either local host or for data saved in minio-buckets

@author: Ellen Wasbo
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
from datetime import date, datetime
import numpy as np
import pandas as pd
try:
    import dash
    # TODO from dash import dcc, html when bootstrap ready
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output, ALL
    import dash_bootstrap_components as dbc
    from waitress import serve
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    try:
        from dash import ctx
    except:
        from dash import callback_context
except (ImportError, ModuleNotFoundError) as err:
    print(f'Warning: {err}')
    print('To run the dash application you will need to install dash '
          '+ dash_bootstrap_components + waitress.')
    if "cannot import name 'json' from 'itsdangerous'" in str(err):
        print('You might have to downgrade itsdangerous: '
              'pip install itsdangerous==2.0.1 --force-reinstall')
    if "cannot import name 'BaseResponse' from 'werkzeug.wrappers'" in str(err):
        print('You might have to downgrade werkzeug: '
              'pip install werkzeug==2.0.3 --force-reinstall')

# imageQC block start
from imageQC.dash_app import config_func_dash as cffd
# imageQC block end


@dataclass
class LimitsAndPlotTemplate:
    """Copy of dataclass from imageQC/config/config_classes. Read only part."""

    label: str = ''
    type_vendor: bool = False
    groups: list = field(default_factory=list)
    groups_limits: list = field(default_factory=list)
    groups_ranges: list = field(default_factory=list)
    groups_hide: list = field(default_factory=list)
    groups_title: list = field(default_factory=list)


@dataclass
class Template:
    """Class holding template settings and values."""

    label: str = ''
    limits_and_plot_template: LimitsAndPlotTemplate = field(
        default_factory=LimitsAndPlotTemplate)
    data: dict = field(default_factory=dict)
    newest_date: str = 'xx.xx.xxxx'  # last data found in results
    days_since: int = -1  # days since newest date (on update)
    status: int = 0  # [0 = ok, 1 = failed, 2 = watch]


def status_to_text(status):
    """Convert status integer to text."""
    if status == 0:
        txt = 'ok'
    elif status == 1:
        txt = 'failed'
    else:
        txt = 'watch'
    return txt


def get_data(config_path):
    """Extract data from result files combined with automation templates.

    Parameters
    ----------
    config_path: Path
        path to config_folder

    Returns
    -------
    modality_dict: dict
        keys = modalities as defined in imageQCpy
        items = list of Templates defined above
        ignored modalities and templates with no results
    """
    print('Reading data from result files...')

    # Read confiuuration files to find result file paths and plot-settings
    
    auto_templates = cffd.load_settings('auto_templates', config_path)
    if auto_templates:
        modalities = [*auto_templates]
        paramsets = cffd.load_paramset_decimarks(
            modalities, config_path)
    else:
        paramsets = {}
    auto_vendor_templates = cffd.load_settings(
        'auto_vendor_templates', config_path)
    if auto_templates or auto_vendor_templates:
        lim_plots = cffd.load_settings(
            'limits_and_plot_templates', config_path)
    else:
        lim_plots = {}

    modality_dict = {}
    all_templates = [auto_templates, auto_vendor_templates]
    n_processed_files = 0
    for auto_no, template in enumerate(all_templates):
        for mod, template_list in template.items():
            param_labels = []
            decimarks = []
            lim_labels = []
            if mod not in [*modality_dict]:
                modality_dict[mod] = []
            try:
                lim_labels = [lim.label for lim in lim_plots[mod]]
                if auto_no == 0:
                    param_labels = [paramset.label for paramset in paramsets[mod]]
                    decimarks = [paramset.output.decimal_mark
                                 for paramset in paramsets[mod]]
                else:
                    decimarks = [paramsets[mod][0].output.decimal_mark]
            except KeyError:
                pass
            for temp in template_list:
                if all([
                        temp.label != '',
                        temp.path_output != '',
                        temp.active]):
                    proceed = True
                    try:
                        if temp.import_only:
                            proceed = False
                        elif temp.paramset_label == '':
                            proceed = False
                        elif temp.quicktemp_label == '':
                            proceed = False
                    except AttributeError:
                        pass

                    if proceed:
                        proceed = False
                        dataframe = None
                        try:
                            if hasattr(temp, 'paramset_label'):
                                idx_paramset = param_labels.index(temp.paramset_label)
                            else:
                                idx_paramset = 0
                            dataframe = pd.read_csv(
                                temp.path_output, sep='\t',
                                decimal=decimarks[idx_paramset],
                                parse_dates=[0],
                                dayfirst=True,
                                on_bad_lines='error',
                                encoding='ISO-8859-1')
                            if dataframe.index.size > 1:
                                proceed = True
                        except FileNotFoundError as ferror:
                            print('FileNotFoundError                                ')
                            print(temp.path_output)
                            print(ferror)
                        except OSError as oerror:
                            print('OSError - could not read file                    ')
                            print(temp.path_output)
                            print(oerror)
                        except pd.errors.EmptyDataError as error:
                            print(temp.path_output)
                            print(error)
                        except pd.errors.ParserError:
                            n_headers = 0
                            with open(temp.path_output) as file:
                                first_line = file.readline()
                                n_headers = len(first_line.split(sep='\t'))
                            if n_headers > 0:
                                try:
                                    dataframe = pd.read_csv(
                                        temp.path_output, sep='\t',
                                        decimal=decimarks[idx_paramset],
                                        parse_dates=[0],
                                        dayfirst=True,
                                        on_bad_lines='error',
                                        usecols=list(range(n_headers)),
                                        encoding='ISO-8859-1')
                                    if dataframe.index.size > 1:
                                        proceed = True
                                except Exception as error:
                                    print(f'Failed reading {temp.path_output}')
                                    print(str(error))
                        if proceed:
                            print(f'Reading results for {mod}/{temp.label}',
                                  end='\r', flush=True)
                            dataframe.dropna(
                                how='all', inplace=True)  # ignore empty rows
                            date_header = dataframe.columns[0]
                            dataframe = dataframe.sort_values(by=[date_header])
                            try:
                                first_row_val = dataframe[date_header].iloc[-1]
                            except IndexError:
                                first_row_val = ''
                            if isinstance(first_row_val, str):
                                newest_date = 'error'
                                days_since = -1000
                            else:
                                newest_date = first_row_val.date()
                                days_since = (date.today() - newest_date).days
                            temp_this = Template(
                                label=temp.label,
                                limits_and_plot_template=temp.limits_and_plot_label,
                                data=dataframe,
                                newest_date=f'{newest_date}',
                                days_since=days_since
                                # TODO status=
                                )
                            create_empty = True
                            if temp.limits_and_plot_label != '':
                                if temp.limits_and_plot_label in lim_labels:
                                    idx_lim = lim_labels.index(
                                        temp.limits_and_plot_label)
                                    lim_temp = lim_plots[mod][idx_lim]
                                    create_empty = False
                            if create_empty:
                                lim_temp = LimitsAndPlotTemplate(groups=[
                                    [col] for col in dataframe.columns[1:]])
                            temp_this.limits_and_plot_template = lim_temp
                            modality_dict[mod].append(temp_this)
                            n_processed_files += 1

    print(f'Processed {n_processed_files} result files.'
          '                                                                          ')

    # remove empty modality from list dict
    len_temp_lists = [len(template_list)
                      for mod, template_list in modality_dict.items()]
    mods_orig = [*modality_dict]
    for i in range(len(modality_dict)):
        if len_temp_lists[-(i+1)] == 0:
            modality_dict.pop(mods_orig[-(i+1)])
    return modality_dict


def run_dash_app(dash_settings=None):
    """Update content in dashboard to display automation results.

    Parameters
    ----------
    dash_settings : config.config_classes.DashSettings
    """
    logger = logging.getLogger('imageQC')
    logger.setLevel(logging.ERROR)

    modality_dict = {}
    #try:
    app = dash.Dash(
        __name__, suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.YETI])
    modality_dict = get_data()
    '''
    except Exception as error:
        print('Failed running dash')
        print(error)
    '''

    def layout():
        """Build the overall layout structure."""
        return dbc.Container([
            dcc.Store(id='results'),
            dbc.Row([
                dbc.Col(html.H1(dash_settings.header)),
                dbc.Col(html.Img(src=dash_settings.url_logo)),
                ]),
            dbc.Row(html.Div([
                dbc.Tabs([
                    dbc.Tab(tab_overview(), label='Overview', tab_id='overview'),
                    dbc.Tab(tab_results(),
                            label='Results pr template', tab_id='results'),
                    ],
                    id='tabs',
                    active_tab='overview',
                    )
                ])),
            ])

    def table_overview(modality):
        if modality in modality_dict:
            table_data = {
                dash_settings.overview_table_headers[0]:
                    [temp.label for temp in modality_dict[modality]],
                dash_settings.overview_table_headers[1]:
                    [temp.newest_date for temp in modality_dict[modality]],
                dash_settings.overview_table_headers[2]:
                    [temp.days_since for temp in modality_dict[modality]],
                # TODO 'Status': [status_to_text(temp.status)
                #           for temp in modality_dict[modality]],
                }
            dataframe = pd.DataFrame(table_data)
            data = dataframe.to_dict('records')
            content = html.Div([
                dbc.Button(
                    modality, color='secondary',
                    id={
                        'type': 'overview_modality_button',
                        'index': modality
                        },
                    n_clicks=0,
                    className='d-grid gap-2 col-6 mx-auto'
                    ),
                dash.dash_table.DataTable(
                    id={
                        'type': 'overview_modality_table',
                        'index': modality
                        },
                    columns=[{'name': header, 'id': header} for header
                             in dash_settings.overview_table_headers[:3]],
                    data=data,
                    style_data_conditional=[
                       {
                           'if': {
                               'filter_query': (
                                   '{' + dash_settings.overview_table_headers[2] + '} '
                                   '> ' + str(dash_settings.days_since_limit)),
                               'column_id': dash_settings.overview_table_headers[2]
                           },
                           'color': 'red'
                           },
                       ],
                    style_header={
                        'color': 'white',
                        'backgroundColor': '#799DBF',
                        'fontWeight': 'bold',
                        },
                    style_cell={'padding-right': '10px', 'padding-left': '10px',
                                'text-align': 'center'},
                    ),
                ])
        else:
            content = html.Div()
        return content

    def tab_overview():
        return html.Div([
            dbc.Row([
                dbc.Col(table_overview('CT')),
                dbc.Col(table_overview('Xray')),
                dbc.Col(table_overview('Mammo')),
                dbc.Col([
                    dbc.Row(table_overview('NM')),
                    dbc.Row(table_overview('SPECT')),
                    dbc.Row(table_overview('PET')),
                    ]),
                dbc.Col(table_overview('MR')),
                ]),
            dbc.Row(dbc.Alert(
                f'Last update {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}',
                color='light')),
            ], style={'marginBottom': 50, 'marginTop': 25}
            )

    def tab_results():
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Label('Select modality'),
                        dbc.RadioItems(
                            options=[
                                {'label': mod, 'value': i}
                                for i, mod in enumerate([*modality_dict])],
                            value=0,
                            id='modality_select'),
                    ]),
                    html.Hr(),
                    html.Div([
                        dbc.Label('Select template'),
                        dbc.RadioItems(
                            options=update_template_options(),
                            value=0,
                            id='template_select'),
                    ])
                    ],
                    width=2,
                ),
                dbc.Col([
                    html.Div(id='template_graphs')
                    ]),
                ]),
            ], style={'marginBottom': 50, 'marginTop': 25})

    def update_template_options(modality_value=0):
        try:
            mod = [*modality_dict][modality_value]
            template_list = modality_dict[mod]
        except (IndexError, KeyError):
            template_list = []
        return [{'label': temp.label, 'value': i}
                for i, temp in enumerate(template_list)]

    @app.callback(
        Output('tabs', 'active_tab'),
        Output('modality_select', 'value'),
        [Input({'type': 'overview_modality_button', 'index': ALL}, 'n_clicks')],
    )
    def go_to_modality(n_clicks):
        mod_value = 0
        try:
            if ctx.triggered_id:
                mod_value = [*modality_dict].index(ctx.triggered_id.index)
        except NameError:
            triggered_id = callback_context.triggered[0]['prop_id'].split('.')
            if triggered_id[0]:
                dd = eval(triggered_id[0])
                if isinstance(dd, dict):
                    if 'index' in dd:
                        mod_value = [*modality_dict].index(dd['index'])

        return 'results', mod_value

    @app.callback(
        Output('template_select', 'options'),
        Output('template_select', 'value'),
        [
            Input('modality_select', 'value'),
        ],
    )
    def on_modality_select(modality_value):
        return update_template_options(modality_value=modality_value), 0

    @app.callback(
        Output('template_graphs', 'children'),
        [
            Input('modality_select', 'value'),
            Input('template_select', 'value'),
        ],
    )
    def on_template_select(modality_value, template_value):
        proceed = True
        try:
            mod = [*modality_dict][modality_value]
        except IndexError:
            proceed = False
        if proceed:
            #colorlist = [
            #    '#000000', '#5165d5', '#a914a6', '#7f9955', '#efb412',
            #    '#97d2d1', '#b3303b']
            colorlist = dash_settings.colors
            data = modality_dict[mod][template_value].data
            lim_plots = modality_dict[mod][template_value].limits_and_plot_template
            titles = [title for i, title in enumerate(lim_plots.groups_title)
                      if lim_plots.groups_hide[i] is False]
            n_rows = len(titles)
            fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                                subplot_titles=titles)
            rowno = 1
            for group_idx, group in enumerate(lim_plots.groups):
                if lim_plots.groups_hide[group_idx] is False:
                    for lineno, header in enumerate(group):
                        fig.add_trace(
                            go.Scatter(
                                x=data[data.columns[0]], y=data[header],
                                line_color=colorlist[lineno % len(colorlist)],
                                name=header, mode='lines+markers', showlegend=True,
                                legendgroup=str(group_idx)
                                ),
                            row=rowno, col=1,
                            )
                    if any(lim_plots.groups_limits[group_idx]):
                        lims = lim_plots.groups_limits[group_idx]
                        if isinstance(lims[0], str):
                            if lims[0] == 'text':
                                lims = [None, None]
                            elif lims[0] == 'relative_first':
                                first_val = data[header][0]
                                tol = first_val * 0.01 * lims[1]
                                lims = [first_val - tol, first_val + tol]
                            else:  # 'relative_median'
                                med_val = np.median(data[header][:-1])
                                tol = med_val * 0.01 * lims[1]
                                lims = [med_val - tol, med_val + tol]
                        for lim in lims:
                            if lim is not None:
                                fig.add_hline(y=lim, line_dash='dash', line_color='red',
                                              row=rowno, col=1)
                    if any(lim_plots.groups_ranges[group_idx]):
                        set_range = lim_plots.groups_ranges[group_idx]
                        # TODO if None in set_range:
                        fig.update_yaxes(range=set_range, row=rowno, col=1)
                    rowno += 1
            fig.update_xaxes(
                dtick="M1",
                tickformat="%b\n%Y",
                ticklabelmode="period")
            fig_height = dash_settings.plot_height * n_rows
            tracegroupgap = (
                fig.layout.yaxis.domain[1] - fig.layout.yaxis.domain[0]) * fig_height
            fig.update_layout(margin=dict(t=100, b=0, l=0, r=0))
            fig.update_layout(height=fig_height,
                              title_text=modality_dict[mod][template_value].label,
                              legend_tracegroupgap=tracegroupgap)
            template_content = html.Div([
                dcc.Graph(figure=fig),
                ])
        else:
            template_content = html.Div([])
        return template_content

    try:
        app.layout = layout
        serve(app.server, host=dash_settings.host, port=dash_settings.port)
    except Exception as error:
        print(error)

    logger = logging.getLogger('imageQC')
    logger.setLevel(logging.INFO)

def imageQC_dash_local_host(config_path):
    modality_data_dict = get_
    pass

def imageQC_dash_minio():
    pass

if __name__ == '__main__':
    local = False
    config_path = ''
    try:
        # run on client from imageQC
        config_path = os.environ['IMAGEQC_CONFIG_FOLDER']
        local = True
    except KeyError:
        pass

    if local:
        path = sys.path[0] + '/.env'
        print(path)
        try:
            from dotenv import load_dotenv
            load_dotenv(path)
        except ImportError:
            print('Failed to import dotenv to read .env')
        imageQC_dash_local_host(config_path)
    else:
        # run on server independently
        imageQC_dash_minio()

