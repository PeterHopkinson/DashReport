# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:30:55 2024

@author: PeterH
"""

from dash import Dash, html, dcc, dash_table, callback, Output, Input
#from dash_table.Format import Format, Group, Scheme, Symbol https://community.plotly.com/t/dash-table-formatting-decimal-place/34975/3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
print('imports loaded')

relative_filepath = 'NHSdata.xlsx'
orange = (247, 110, 0)
deep_red = (242, 79, 79)
filepath = os.getcwd() + '\\' + relative_filepath
column_definitions = {'Year': pd.Int64Dtype(),
                      'OADR (Old Age Dependancy Ratio)': float,
                      'NHS Inputs': float,
                      'Quality Adjusted Output': float,
                      'Unadjusted Output': float,
                      'Quality Adjusted Productivity': float,
                      'Unadjusted Productivity': float,
                      'Total Healthcare Expenditure (£, nominal)': pd.Int64Dtype(),
                      'Government Financed Expenditure (£, nominal)': pd.Int64Dtype(),
                      'Total Expenditure Per Person (£, nominal)': pd.Int64Dtype(),
                      'Government Expenditure Per Person (£, nominal)': pd.Int64Dtype(),
                      'Total Healthcare Expenditure (£, real terms)': pd.Int64Dtype(),
                      'Government Financed Expenditure (£, real terms)': pd.Int64Dtype(),
                      'Total Expenditure Per Person (£, real terms)': pd.Int64Dtype(),
                      'Government Expenditure Per Person (£, real terms)': pd.Int64Dtype(),
                      'CPIH (Consumer Prices Index with Housing costs)': float,
                      'CPI (Consumer Prices Index)': float,
                      'Population': pd.Int64Dtype(),
                      '0-9 Population': pd.Int64Dtype(),
                      '65+ Population': pd.Int64Dtype(),
                      '80+ Population': pd.Int64Dtype(),
                      '85+ Population': pd.Int64Dtype(),
                      'RTT Total Waiting': pd.Int64Dtype(),
                      'RTT Median Wait Time (weeks)': float
                      }
non_year_columns = {key:value for key, value in column_definitions.items() if key not in ('Year',)}
measures = {key:value for key, value in column_definitions.items() if key not in ('Year',)}
measures['None'] = None

spreadsheet = pd.read_excel(filepath, sheet_name=None)
data = spreadsheet['Data']
data.set_index('Year')

min_year = int(data['Year'].min())
max_year = int(data['Year'].max())

mean_values = data.mean() * data.notnull()
mean_yearly_deviation = data.diff().abs().mean()
spc_upper = mean_values + (3 * mean_yearly_deviation * data.notnull())
spc_lower = mean_values - (3 * mean_yearly_deviation * data.notnull())

#regression = data[['Year']].copy()
regression = pd.DataFrame(index=['gradient', 'constant'])
for measure1 in list(column_definitions.keys()):
    for measure2 in list(column_definitions.keys()):
        mask = data[measure1].notnull() & data[measure2].notnull()
        masked_data = data[mask]
        x = masked_data[measure2]
        y = masked_data[measure1]
        n = x.size # = y.size
        mean_x = x.mean()
        mean_y = y.mean()
        deviation_xy = (y * x).sum() - (n * mean_y * mean_x)
        deviation_xx = (x * x).sum() - (n * mean_x * mean_x)
        gradient = deviation_xy / deviation_xx
        constant = mean_y - (gradient * mean_x)
        line_of_best_fit = constant + (gradient * x)
        regression.loc[['gradient', 'constant'], f'{measure1} -- {measure2}'] = gradient, constant
regression = regression.copy()
print('Completed Regression')

###
# Page - DATA
tab_data =  dcc.Tab(label='Data', children=[
    dash_table.DataTable(data.to_dict('records'), columns=[{"name": i, "id": i} for i in data.columns], id='data-table'),
    ])

"""@callback(
    Output('data-table', 'columns'),
    Input('data-year-slider', 'value'),
    Input('data-column-dropdown', 'value')) #State('data-table', 'columns'))
def update_data_table(year, selected_column, columns):
    if selected_column == 'None': columns = [{"name": i, "id": i} for i in data.columns] # show all
    else: columns = ['Year', selected_column]
    table = dash_table.DataTable(data.to_dict('records'), columns)
    return table
"""

###
# Page - COMPARISON
tab_comparison = dcc.Tab(label='Comparison', children=[
    html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Year'),
            dcc.RangeSlider(min=min_year, max=max_year, value=[min_year, max_year], id='comparison-year-slider',
                            marks={i: '{}'.format(i) for i in range(min_year, max_year, 10)})
            ]),
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Left Axis'),
            dcc.Dropdown(list(non_year_columns.keys()), 'RTT Total Waiting', id='comparison-measure1-dropdown'),
            ]),
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Right Axis'),
            dcc.Dropdown(list(measures.keys()), 'None', id='comparison-measure2-dropdown'),
            ])
        ]),
    dcc.Graph(id='comparison-graph'),
    ])

@callback(
    Output('comparison-graph', 'figure'),
    Input('comparison-year-slider', 'value'),
    Input('comparison-measure1-dropdown', 'value'),
    Input('comparison-measure2-dropdown', 'value'))
def update_comparison_graph(year, measure1, measure2):
    if measure2 == 'None':
        fig = px.line(data[(data['Year']>=year[0]) & (data['Year']<=year[1])], x='Year', y=measure1, markers=True)
    else:
        masked_data = data[(data['Year']>=year[0]) & (data['Year']<=year[1])]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=masked_data['Year'], y=masked_data[measure1], mode='lines+markers', name=measure1), secondary_y=False)
        fig.add_trace(go.Scatter(x=masked_data['Year'], y=masked_data[measure2], mode='lines+markers', name=measure2), secondary_y=True)
        fig.update_xaxes(title_text='Year')
        fig.update_yaxes(title_text=measure1, secondary_y=False)
        fig.update_yaxes(title_text=measure2, secondary_y=True)
    return(fig)

###
# Page - SPC
tab_spc = dcc.Tab(label='SPC', children=[
    html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Year'),
            dcc.RangeSlider(min=min_year, max=max_year, value=[min_year, max_year], id='spc-year-slider',
                            marks={i: '{}'.format(i) for i in range(min_year, max_year, 10)})
            ]),
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Y-Axis'),
            dcc.Dropdown(list(non_year_columns.keys()), 'RTT Total Waiting', id='spc-measure-dropdown'),
            ])
        ]),
    dcc.Graph(id='spc-graph'),
    ])

@callback(
    Output('spc-graph', 'figure'),
    Input('spc-year-slider', 'value'),
    Input('spc-measure-dropdown', 'value'))
def update_spc_graph(year, measure):
    fig = go.Figure()
    #plot = px.line(data[(data['Year']>=year[0]) & (data['Year']<=year[1])], x='Year', y=measure, markers=True)
    mask = (data['Year']>=year[0]) & (data['Year']<=year[1]) & data[measure].notnull()
    fig.add_trace(go.Scatter(x=data[mask]['Year'], y=mean_values[mask][measure], mode='lines', line=dict(color='black'), name='Mean'))
    fig.add_trace(go.Scatter(x=data[mask]['Year'], y=spc_upper[mask][measure], mode='lines', line=dict(color='black', dash='dash'), name='Upper Control'))
    fig.add_trace(go.Scatter(x=data[mask]['Year'], y=spc_lower[mask][measure], mode='lines', line=dict(color='black', dash='dash'), name='Lower Control'))
    fig.add_trace(go.Scatter(x=data[mask]['Year'], y=data[mask][measure], mode='lines+markers', name=measure))
    return fig

###
# Page - REGRESSION
tab_regression = dcc.Tab(label='Regression', children=[
    html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Year'),
            dcc.RangeSlider(min=min_year, max=max_year, value=[min_year, max_year], id='regression-year-slider',
                            marks={i: '{}'.format(i) for i in range(min_year, max_year, 10)})
            ]),
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Left Axis'),
            dcc.Dropdown(list(column_definitions.keys()), 'RTT Total Waiting', id='regression-measure1-dropdown'),
            ]),
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Right Axis'),
            dcc.Dropdown(list(column_definitions.keys()), 'Population', id='regression-measure2-dropdown'),
            ])
        ]),
    dcc.Graph(id='regression-graph'),
    ])

@callback(
    Output('regression-graph', 'figure'),
    Input('regression-year-slider', 'value'),
    Input('regression-measure1-dropdown', 'value'),
    Input('regression-measure2-dropdown', 'value'))
def update_regression_graph(year, measure1, measure2):
    regression_name = f'{measure1} -- {measure2}'
    masked_data = data[data[measure1].notnull() & data[measure2].notnull()]
    x = masked_data[measure2]
    y = masked_data[measure1]
    line_of_best_fit = regression.loc['constant', regression_name] + (x * regression.loc['gradient', regression_name])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data Points'))
    fig.add_trace(go.Scatter(x=x, y=line_of_best_fit, mode='lines', name='Line of Best Fit'))
    fig.update_xaxes(title_text=measure2)
    fig.update_yaxes(title_text=measure1)
    return(fig)

###
# Page - MODELS
tab_models = dcc.Tab(label='Linear Prediction Models', children=[
    html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Year'),
            dcc.RangeSlider(min=min_year, max=max_year, value=[min_year, max_year], id='models-year-slider',
                            marks={i: '{}'.format(i) for i in range(min_year, max_year, 10)})
            ]),
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Model 1'),
            dcc.Dropdown(list(column_definitions.keys()), 'Population', id='models-measure1-dropdown'),
            ]),
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Model 2'),
            dcc.Dropdown(list(column_definitions.keys()), '80+ Population', id='models-measure2-dropdown'),
            ]),
        html.Div(style={'padding': 10, 'flex': 1}, children=[
            html.Label('Model 3'),
            dcc.Dropdown(list(column_definitions.keys()), 'OADR (Old Age Dependancy Ratio)', id='models-measure3-dropdown'),
            ])
        ]),
    dcc.Graph(id='models-graph'),
    ])

@callback(
    Output('models-graph', 'figure'),
    Input('models-year-slider', 'value'),
    Input('models-measure1-dropdown', 'value'),
    Input('models-measure2-dropdown', 'value'),
    Input('models-measure3-dropdown', 'value'))
def update_models_graph(year, measure1, measure2, measure3):
    fig = go.Figure()
    x = data['Year']
    for measure in (measure1, measure2, measure3):
        regression_name = f'RTT Total Waiting -- {measure}'
        line_of_best_fit = regression.loc['constant', regression_name] + (data[measure] * regression.loc['gradient', regression_name])
        fig.add_trace(go.Scatter(x=x, y=line_of_best_fit, mode='lines', name=measure))
    fig.add_trace(go.Scatter(x=x, y=data['RTT Total Waiting'], mode='markers', line=dict(color='black'), name='Recorded Figure'))
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='RTT Total Waiting')
    return(fig)

###
# App Layout

app = Dash()

app.layout = [
    html.Div(children='', id='vertical-line-1', style={'width':'4 px', 'height':'100%', 'top':'0', 'left':'10', 'background-color':f'rgb{deep_red}'}),
    html.Div(id='vertical-line-2', style={'width':'12 px', 'height':'100%', 'top':'0', 'left':'20', 'background-color':f'rgb{orange}'}),
    html.Div(id='horizontal-line-1', style={'width':'100%', 'height':'4 px', 'top':'10', 'left':'0', 'background-color':f'rgb{deep_red}'}),
    html.H1(children='Dash Demo', id='Header', style={'textAlign':'left'}),
    dcc.Tabs([tab_comparison, tab_spc, tab_regression, tab_models, tab_data]),
    ]

if __name__ == '__main__':
    #app.run(debug=True)
    app.run_server(host="0.0.0.0", port="8050")