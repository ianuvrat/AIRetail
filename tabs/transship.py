#Import library

from __future__ import print_function
from ortools.graph import pywrapgraph

from pulp import *
import pandas as pd
import numpy as np
from app import app, server, dbc, dcc, de
from app import app
import dash_html_components as html
import plotly.graph_objs as go
import dash
import plotly.express as px
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
from dash.exceptions import PreventUpdate
import dash_table
import pathlib
import dash_daq as daq
from datetime import date, timedelta
import dash_auth


#import cufflinks as cf

# Lottie
url = "https://assets2.lottiefiles.com/packages/lf20_ftcfknxp.json"
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))
# Mapbox token
token = 'pk.eyJ1IjoiaWFudXZyYXQiLCJhIjoiY2tvNWQ1a2FkMHB6MTJ2cXdmeGt3MTdldyJ9.5vxPVvQdr6NL8hxVCZ1ecg'
#---------------------------------------------------------------
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()
df = pd.read_csv(DATA_PATH.joinpath("transship_data.csv"))
#_--------------------------------------------------------------

# Cleaning
dff = df.drop(['location_description','is_internal'],axis=1)
# Filtering US data
dff = dff[dff['country']=='US']

supply_input_types = ['number','number','number','number','number','number','number','number','number']
demand_input_types = ['number','number','number','number','number','number']
alert0 = dbc.Alert("Message : Please feed the demand and supply units and select a scenario to simulate! ", color="info",  dismissable=False),

alert1 = dbc.Alert("Alert : Demand and Supply do not match but model found the optimized result. ",
                  color="warning",
                  dismissable=True),
alert2 = dbc.Alert("Success : All constraints were met and model found the optimized result ",
                  color="success",
                  dismissable=True),

my_elements_2 = [# Nodes elements
                                    {'data': {'id': 'dc1', 'label': 'Minneapolis (DC1)'},
                                     'position': {'x': -100, 'y': -250},
                                     'selected': True
                                     },

                                    {'data': {'id': 'dc2', 'label': 'Denver (DC2)'},
                                     'position': {'x': -100, 'y': -150},
                                     'selected': True
                                     },

                                    {'data': {'id': 'dc3', 'label': 'Kansas City (DC3)'},
                                     'position': {'x': -100, 'y': -50},
                                     'selected': True
                                     },

                                    {'data': {'id': 'dc4', 'label': 'Portland (DC4)'},
                                     'position': {'x': -100, 'y': 50},
                                     'locked': True,
                                     'selected': True
                                     },

                                    {'data': {'id': 'dc5', 'label': 'Las Vegas (DC5)'},
                                     'position': {'x': -100, 'y': 150},
                                     'locked': True,
                                      'selected': True
                                     },

                                    {'data': {'id': 'w1', 'label': 'Atlanta (W1)'},
                                     'position': {'x': 100, 'y': -200},
                                     'locked': True,
                                     'selected': True,
                                     'classes': 'red'

                                     },

                                    {'data': {'id': 'w2', 'label': 'New Haven (W2)'},
                                     'position': {'x': 100, 'y': -100},
                                     'grabbable': True,
                                     'selected': True,
                                     'classes': 'red'

                                     },

                                    {'data': {'id': 'w3', 'label': 'Siracusa (W3)'},
                                     'position': {'x': 100, 'y': 0},
                                     'selected': True,
                                     'classes': 'red'

                                     },

                                    {'data': {'id': 'w4', 'label': 'Sacramento (W4)'},
                                     'position': {'x': 100, 'y': 100},
                                     'selected': True,
                                     'classes': 'red'

                                     },

                                    {'data': {'id': 'r1', 'label': 'New York (R1)'},
                                     'position': {'x': 300, 'y': -250},
                                     'selectable': True,
                                     'selected': True,
                                     'classes': 'green'
                                     },

                                    {'data': {'id': 'r2', 'label': 'Washington DC(R2)'},
                                     'position': {'x': 300, 'y': -150},
                                     'selectable': True,
                                     'selected': True,
                                     'classes': 'green'

                                     },

                                    {'data': {'id': 'r3', 'label': 'Chicago (R3)'},
                                     'position': {'x': 300, 'y': -70},
                                     'selectable': True,
                                     'selected': True,
                                     'classes': 'green'

                                     },

                                    {'data': {'id': 'r4', 'label': 'Boston (R4)'},
                                     'position': {'x': 300, 'y': 10},
                                     'selectable': True,
                                     'selected': True,
                                     'classes': 'green'

                                     },

                                    {'data': {'id': 'r5', 'label': 'San Francisco(R5)'},
                                     'position': {'x': 300, 'y':80},
                                     'selectable': True,
                                     'selected': True,
                                     'classes': 'green'

                                     },

                                    {'data': {'id': 'r6', 'label': 'Seattle(R6)'},
                                     'position': {'x': 300, 'y': 150},
                                     'selectable': True,
                                     'selected': True,
                                     'classes': 'green'

                                     },

                 # Edge elements
                                    {'data': {'source': 'dc1', 'target': 'r1'}, 'classes': 'green'},
                                    {'data': {'source': 'dc1', 'target': 'w1'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc1', 'target': 'w2'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc1', 'target': 'w3'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc1', 'target': 'w4'}, 'classes': 'blue'},
#                                    {'data': {'source': 'dc1', 'target': 'r6'}},
                                    {'data': {'source': 'dc2', 'target': 'w1'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc2', 'target': 'w2'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc2', 'target': 'w3'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc2', 'target': 'w4'}, 'classes': 'blue'},
#                                    {'data': {'source': 'dc2', 'target': 'r5'}},
#                                    {'data': {'source': 'dc2', 'target': 'r6'}},
                                    {'data': {'source': 'dc3', 'target': 'w1'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc3', 'target': 'w2'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc3', 'target': 'w3'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc3', 'target': 'w4'}, 'classes': 'blue'},
#                                    {'data': {'source': 'dc3', 'target': 'r5'}},
#                                    {'data': {'source': 'dc3', 'target': 'r6'}},
                                    {'data': {'source': 'dc4', 'target': 'w1'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc4', 'target': 'w2'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc4', 'target': 'w3'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc4', 'target': 'w4'}, 'classes': 'blue'},
#                                    {'data': {'source': 'dc4', 'target': 'r5'}},
#                                    {'data': {'source': 'dc4', 'target': 'r6'}},
                                    {'data': {'source': 'dc5', 'target': 'r6'}, 'classes': 'green'},
                                    {'data': {'source': 'dc5', 'target': 'w1'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc5', 'target': 'w2'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc5', 'target': 'w3'}, 'classes': 'blue'},
                                    {'data': {'source': 'dc5', 'target': 'w4'}, 'classes': 'blue'},
#                                    {'data': {'source': 'dc5', 'target': 'r6'}},
                                    {'data': {'source': 'w1', 'target': 'r1'}, 'classes': 'red'},
                                    {'data': {'source': 'w1', 'target': 'r2'},'classes': 'red'},
                                    {'data': {'source': 'w1', 'target': 'r3'},'classes': 'red'},
                                    {'data': {'source': 'w1', 'target': 'r4'},'classes': 'red'},
                                    {'data': {'source': 'w1', 'target': 'r5'},'classes': 'red'},
                                    {'data': {'source': 'w1', 'target': 'r6'},'classes': 'red'},
                                    {'data': {'source': 'w2', 'target': 'r1'},'classes': 'red'},
                                    {'data': {'source': 'w2', 'target': 'r2'},'classes': 'red'},
                                    {'data': {'source': 'w2', 'target': 'r3'},'classes': 'red'},
                                    {'data': {'source': 'w2', 'target': 'r4'},'classes': 'red'},
                                    {'data': {'source': 'w2', 'target': 'r5'},'classes': 'red'},
                                    {'data': {'source': 'w2', 'target': 'r6'},'classes': 'red'},
                                    {'data': {'source': 'w3', 'target': 'r1'},'classes': 'red'},
                                    {'data': {'source': 'w3', 'target': 'r2'},'classes': 'red'},
                                    {'data': {'source': 'w3', 'target': 'r3'},'classes': 'red'},
                                    {'data': {'source': 'w3', 'target': 'r4'},'classes': 'red'},
                                    {'data': {'source': 'w3', 'target': 'r5'},'classes': 'red'},
                                    {'data': {'source': 'w3', 'target': 'r6'},'classes': 'red'},
                                    {'data': {'source': 'w4', 'target': 'r1'},'classes': 'red'},
                                    {'data': {'source': 'w4', 'target': 'r2'},'classes': 'red'},
                                    {'data': {'source': 'w4', 'target': 'r3'},'classes': 'red'},
                                    {'data': {'source': 'w4', 'target': 'r4'},'classes': 'red'},
                                    {'data': {'source': 'w4', 'target': 'r5'},'classes': 'red'},
                                    {'data': {'source': 'w4', 'target': 'r6'},'classes': 'red'},
                                    ]

# App Layout ----------------------------------------------------------------------------------------------------------------------------------------
layout = dbc.Container([
    html.Br(),
                #Supply Info
                dbc.Row([

                    # i button for supply
                    dbc.Col([
                        dbc.Button("i", id="popover-bottom-target0", color="warning", outline=True),
                    ], className="text-left", width={'size': 1.5, 'offset': 0, 'order': 0}),
                    dbc.Popover([
                        dbc.PopoverBody(
                            "Feed the units of supply for each facility. Feed 0, if any facility is 'Closed' / 'Not Operable'."),
                                ],
                        id="popover0",
                        target="popover-bottom-target0",  # needs to be the same as dbc.Button id
                        trigger="hover",
                        placement="bottom",
                        is_open=False,
                                ),

                    # supply Label
                    dbc.Col([
                        html.Label("Total DC Supply:",className="font-weight-bold text-primary"),
                            ],className="text-left", width={'size': 1.5, 'offset': 0, 'order': 1}),

                    # supply Total
                    dbc.Col([
                        html.Div(id='sum_supply2',className="font-weight-bold text-primary"),
                            ],className="text-left", width={'size': 3, 'offset': 0, 'order': 2}),
                ]),


                dbc.Row([
                    # DC Labels
                    html.Label(
                        "__Minneapolis__Denver__Kansas City__Portland __ Las Vegas "),
                        ]),

    #Supply Inputs
                dbc.Row([

                    dbc.Col([
                            dcc.Input(id='s1_ip',type='number',inputMode='numeric',value=80,placeholder="DC1", debounce=True,  min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s2_ip',type='number',inputMode='numeric',value=80,placeholder="DC2", debounce=True,  min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s3_ip',type='number',inputMode='numeric',value=60,placeholder="DC3", debounce=True,  min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s4_ip',type='number',inputMode='numeric',value=80,placeholder="DC4", debounce=True,  min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s5_ip',type='number',inputMode='numeric',value=60,placeholder="DC5", debounce=True,  min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            ],className="text-left", width={'size': 6.5, 'offset': 0, 'order': 0}),

                    # Min Cost output
                    dbc.Col([
                        dcc.Loading(children=[html.Div(id='output_state2')], color="#119DFF", type="dot",
                                    fullscreen=False),
                            ], className="text-primary font-weight-bold text-left border-white", width={'size': 3, 'offset': 4, 'order': 1}),

                   # Model Status output
                    dbc.Col([
                        dcc.Loading(children=[html.Div(id='model_status2')], color="#119DFF", type="default",
                                    fullscreen=False),
                            ], className="text-info font-weight-bold text-left border-primary", width={'size': 0, 'offset': 0, 'order': 2}),

                        ]),

    html.Br(),
                # Warehouse Info
                dbc.Row([

                    # i button for Warehouse
                    dbc.Col([
                        dbc.Button("i", id="popover-bottom-target1", color="warning", outline=True),
                    ], className="text-left", width={'size': 1.5, 'offset': 0, 'order': 0}),
                    dbc.Popover([
                        dbc.PopoverBody(
                            "If no stock reservation (Cross-Docking), feed 0."),
                    ],
                        id="popover1",
                        target="popover-bottom-target1",  # needs to be the same as dbc.Button id
                        trigger="hover",
                        placement="bottom",
                        is_open=False,
                    ),

                    # Warehouse Label
                    dbc.Col([
                        html.Label("Total Warehouse Reservation:", className="font-weight-bold text-primary"),
                    ], className="text-left", width={'size': 1.5, 'offset': 0, 'order': 0}),

                    # Warehouse total
                    dbc.Col([
                        html.Div(id='sum_warehouse', className="font-weight-bold text-primary"),
                    ], className="text-left", width={'size': 1, 'offset': 0, 'order': 1}),

                ]),

                # Labels
                dbc.Row([
                    html.Label(
                        "___ Atlanta ___ New Haven ___Siracusa ___Sacramento"),

                ]),

                dbc.Row([
                    # Warehouse Inputs
                    dbc.Col([
                        dcc.Input(id='s6_ip', type="number", inputMode='numeric', value=0, placeholder="W1", debounce=True,min=-1000, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                        dcc.Input(id='s7_ip', type="number", inputMode='numeric', value=0, placeholder="W2",debounce=True, min=-1000, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on',disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                        dcc.Input(id='s8_ip', type="number", inputMode='numeric', value=0, placeholder="W3", debounce=True,min=-1000, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False,readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                        dcc.Input(id='s9_ip', type="number", inputMode='numeric', value=0, placeholder="W4", debounce=True, min=-1000, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),


                    # i button for scenario
                    dbc.Col([
                        dbc.Button("i", id="popover-bottom-target", color="info", outline=True),
                    ], className="text-left", width={'size': 1.5, 'offset': 1, 'order': 1}),
                    dbc.Popover([
                        dbc.PopoverHeader("All about Cross-Docking / Warehouse Reservation?"),
                        dbc.PopoverBody(

                            "CROSS-DOCKING - A logistics procedure where products from a supplier are distributed directly to a customer/ retail chain with marginal to no handling or storage time.\n"
                            
                            "\n\nWAREHOUSE RESERVATION - Gives the user the ability to decide whether to reserve some stock of units in any intermediate facility(s). ")
                    ],
                        id="popover",
                        target="popover-bottom-target",  # needs to be the same as dbc.Button id
                        trigger="hover",
                        placement="bottom",
                        is_open=False,
                    ),

                    # Warehouse Reservation
                    dbc.Col([
                        html.Label(['Scenario:'], className="font-weight-bold text-primary",
                                   style={'font-weight': 'bold'}),
                        dcc.RadioItems(
                            id='resv_id',
                            options=[
                                {'label': 'Cross-Docking', 'value': 'noresv'},
                                {'label': 'Warehouse Reservation ', 'value': 'resv'},
                            ],
                            value='noresv',
                            style={"width": "50%"},
                        ),
                    ], className="text-left", width={'size': 4, 'offset': 0, 'order': 2}),

                    # Alert,
                    dbc.Col([
                         html.Div(id="the_alert_2", children=[]),
                     ], className="text-left", width={'size': 3, 'offset': 0, 'order': 3}),
                ]),

                #Demand Info
                dbc.Row([

                    # i button for retail store
                    dbc.Col([
                        dbc.Button("i", id="popover-bottom-target2", color="warning", outline=True),
                            ], className="text-left", width={'size': 1.5, 'offset': 0, 'order': 0}),
                    dbc.Popover([
                        dbc.PopoverBody(
                            "Feed the demand for each retail store, feed 0 where there is no demand."),
                                ],
                        id="popover2",
                        target="popover-bottom-target2",  # needs to be the same as dbc.Button id
                        trigger="hover",
                        placement="bottom",
                        is_open=False,
                                ),


                    dbc.Col([
                        html.Label("Total Retail Demand:",className="font-weight-bold text-primary"),
                    ], className="text-left", width={'size': 1.5, 'offset': 0, 'order': 0}),

                    dbc.Col([
                        html.Div(id='sum_demand2',className="font-weight-bold text-primary"),
                    ], className="text-left", width={'size': 1, 'offset': 0, 'order': 1}),
                        ]),

                dbc.Row([
                    html.Label(
                        " __ New York __WashingtonDC __Chicago __ Boston__ San Francisco __ Seattle "),
                ]),

                dbc.Row([
                    # Demand Inputs
                    dbc.Col([
                            dcc.Input(id='d1_ip',type="number",inputMode='numeric',value=60,placeholder="New York",debounce=True, min=-1000, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='d2_ip',type="number",inputMode='numeric',value=60,placeholder="Washington DC", debounce=True, min=-1000, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='d3_ip',type="number",inputMode='numeric',value=60,placeholder="Chicago", debounce=True, min=-1000, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='d4_ip',type="number",inputMode='numeric',value=60,placeholder="Boston", debounce=True, min=-1000, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='d5_ip',type="number",inputMode='numeric',value=60,placeholder="San Francisco", debounce=True, min=-1000, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='d6_ip',type="number",inputMode='numeric',value=60,placeholder="Seattle", debounce=True, min=-1000, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            ],className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

                    # Submit Button
                    dbc.Col([
                            html.Button(id='submit_button', n_clicks=0, children='Submit'),
                            ],className="text-left", width={'size': 1, 'offset': 0, 'order': 1}),

                    # Capacity and Cost Input
                    dbc.Col([
                        html.Label('Truck Cap.'),
                        dcc.Input(id='cap_ip', type='number', inputMode='numeric', value=50, placeholder="Capacity",
                                  debounce=True, min=0, max=2000, step=50, minLength=0, maxLength=50, autoComplete='on',
                                  disabled=False, readOnly=False, required=True, size="30"),
                        # , persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 1, 'offset': 4, 'order': 1}),
                    dbc.Col([
                        html.Label('Unit Cost'),
                        dcc.Input(id='cst_ip', type='number', inputMode='numeric', value=700, placeholder="Unit Cost",
                                  debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on',
                                  disabled=False, readOnly=False, required=True, size="30"),
                        # , persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 1, 'offset': 0, 'order': 2}),

                ]),

    html.Br(),
                # Map Mode radio items
                html.Div([
                            html.Label(['Map View:'],style={'font-weight': 'bold'},className="font-weight-bold text-primary"),
                            dcc.RadioItems(
                                id='map_mode',
                                options=[
                                         {'label': 'OpenStreet ', 'value': 'ostreet'},
                                         {'label': 'Satellite ', 'value': 'sat'},
                                         {'label': 'Water ', 'value': 'water'},
                                         {'label': 'Dark ', 'value': 'dak'},
                                         {'label': 'States ', 'value': 'street'},
                                         ],
                                value='ostreet',
                                style={"width": "50%"},
                            ),
                        ]),

                dbc.Row([
                    # Map Scatter Box
                    dbc.Col([
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Label("Geographic Locations",className="font-weight-bold text-primary"),
                        dcc.Loading(children=[dcc.Graph(id="mymap2",config={ 'displayModeBar': False })],color="#119DFF", type="cube", fullscreen=False),
                            ], className="text-left", width={'size': 4, 'offset': 0, 'order': 0}),

                    # Dash-Cytoscape
                    dbc.Col([
                        html.Div([
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            cyto.Cytoscape(
                                id='org-chart',
                                layout={'name': 'preset'},
                                minZoom=1,
                                maxZoom=1,
                                style={'width': '150%', 'height': '600px'},
                                stylesheet= [

                                        # Group selectors for NODES
                                         {
                                          'selector': 'node',
                                           'style': {
                                                    'label': 'data(label)'
                                                    }
                                          },

                                        # Class selectors (green)
                                        {
                                          'selector': '.green',
                                           'style': {
                                                      'background-color': 'green',
                                                      'line-color': 'green'
                                                    }
                                        },
                                    # Class selectors (red)
                                    {
                                        'selector': '.red',
                                        'style': {
                                            'background-color': 'red',
                                            'line-color': 'red'
                                        }
                                    },
                                    # Class selectors (blue)
                                    {
                                        'selector': '.blue',
                                        'style': {
                                            'background-color': 'blue',
                                            'line-color': 'blue'
                                        }
                                    },

                                ],
                                elements= my_elements_2
                                         )
                                ]),
                            ], className="text-left", width={'size': 4, 'offset': 3, 'order': 0}),
                        ]),


                dbc.Row([
                    # Dash datatable (Routes)
                    dbc.Col([
                        html.Label("Desired Routes and Units to transfer",className="font-weight-bold text-primary"),
                        html.Div(id="output-datatable_1", style={'width': '100vh', 'height': '50vh'}),
                            ],className="text-left", width={'size': 6, 'offset': 0, 'order': 0}),

                    # Bar Graph
                    dbc.Col([
                        html.Label("Demand Fulfilment", className="font-weight-bold text-primary"),
                        html.Div(id='bar-container', style={'width': '100vh', 'height': '100vh'}),
                            ],className="text-left", width={'size': 6, 'offset': 0, 'order': 1})

                        ]),

],fluid=True),

# ------------------------------------------------------------------------
# Map
@app.callback(
    Output('mymap2', 'figure'),
    Input('map_mode', 'value'),
            )
def update_graph(view):
    # Graph
    fig = px.scatter_mapbox(dff, lat="latitude", lon="longitude",
                            hover_name="location_id",
                            hover_data=["city"],
                            color="location_type",
                            zoom=3.5, height=500, width=950,
                            # size= "city",
                            opacity=1,
                            )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_traces(marker_symbol="circle"),
    fig.update_traces(marker_size=20),
    fig.update_traces(showlegend=True),
    # fig.update_traces(marker={'color': "red"})
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1, xanchor="auto", x=0),
                      legend_title_text='')
    if view == 'ostreet':
        fig.update_layout(mapbox_style="open-street-map", mapbox_accesstoken=token)
    elif view == 'sat':
        fig.update_layout(mapbox_style="satellite", mapbox_accesstoken=token)
    elif view == 'water':
        fig.update_layout(mapbox_style="stamen-watercolor", mapbox_accesstoken=token)
    elif view == 'dak':
        fig.update_layout(mapbox_style="carto-darkmatter", mapbox_accesstoken=token)
    elif view == 'street':
        fig.update_layout(mapbox_style="streets", mapbox_accesstoken=token)


    return fig



# Model Deployment
# ------------------------------------------------------------------------
@app.callback(
    [Output('output_state2', 'children'), # Min. cost
     Output('model_status2', 'children'), # optimal
     Output('sum_supply2', 'children'),  # tot supply
     Output('sum_warehouse', 'children'),  # tot reservation
     Output('sum_demand2', 'children'),  # tot demand
     Output("output-datatable_1", "children"), # dash datatable output
     Output("the_alert_2", "children")],      #alert message

    [Input('submit_button',  'n_clicks')],
    [State('resv_id', 'value')],
    [State('cap_ip', 'value')],
    [State('cst_ip', 'value')],
    [State('s1_ip', 'value'),State('s2_ip', 'value'),State('s3_ip', 'value'),State('s4_ip', 'value'),State('s5_ip', 'value')],
    [State('s6_ip', 'value'),State('s7_ip', 'value'),State('s8_ip', 'value'),State('s9_ip', 'value')],
    [State('d1_ip', 'value'),State('d2_ip', 'value'),State('d3_ip', 'value'),State('d4_ip', 'value'),State('d5_ip', 'value'),State('d6_ip', 'value')]
            )
def update_data(num_clicks, resv_val,
                 cap, cst,
                 s1,s2,s3,s4,s5,
                 s6,s7,s8,s9,
                 d1,d2,d3,d4,d5,d6):

#    if s1==s2==s3==s4==s5==s6==s7==s8==s9==None and d1==d2==d3==d4==d5==d6==None:
    if num_clicks == 0:
#        raise PreventUpdate
        return "","","", "", "", "", alert0


    elif num_clicks>0:

        # Define 4 parallel arrays: start_nodes, end_nodes, capacities, and unit costs between each pair.
        start_nodes = [0,  0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8]
        end_nodes =   [5,  6,  7,  8,  9,  5,  6,  7,  8,  5,  6,  7,  8,  5,  6,  7,  8,  5,  6,  7,  8,  14, 9,  10, 11, 12, 13, 14, 9,  10, 11, 12, 13, 14, 9,  10, 11, 12, 13, 14, 9,  10, 11, 12, 13, 14]

        print("capacities")
        capacities =  [cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap, cap]
        print(capacities)

        print("Cost")
        unit_costs =  [cst+100, cst+83, cst+136, cst+134, cst+136, cst+185, cst+174, cst+173, cst+142, cst+174, cst+173, cst+172, cst+173, cst+127, cst+182, cst+162, cst+162, cst+173, cst+172,cst+173, cst+172, cst+138, cst+192, cst+128, cst+162, cst+183, cst+126, cst+174, cst+174, cst+193, cst+173, cst+172, cst+172, cst+126, cst+183, cst+131, cst+162, cst+162, cst+184, cst+183, cst+137, cst+135, cst+184, cst+126, cst+173, cst+147]
        print(unit_costs)
        # Define an array of supplies at each node.

        if resv_val == "noresv":
            supplies = [s1, s2, s3, s4, s5, s6, s7, s8, s9, -d1, -d2, -d3, -d4, -d5, -d6]
            print("No Warehouse Reservation")
            print(supplies)
        elif resv_val == "resv":
            supplies = [s1, s2, s3, s4, s5, -s6, -s7, -s8, -s9, -d1, -d2, -d3, -d4, -d5, -d6]
            print("Warehouse Reservation")
            print(supplies)
        # supplies = [s1, s2, s3, s4, s5, s6, s7, s8, s9, -d1, -d2, -d3, -d4, -d5, -d6]

#        supplies = [s1, s2, s3, s4, s5, -s6, -s7, -s8, -s9, -d1, -d2, -d3, -d4, -d5, -d6]


        tot_supply = s1 + s2 + s3 + s4 + s5
        print("Total Supply", tot_supply)

        warehouse_resv = s6 + s7 + s8 + s9
        print("Total Reservation", warehouse_resv)


        tot_demand = (d1+d2+d3+d4+d5+d6)
        print("Total Demand", tot_demand)

        # Balanced/Unbalanced Problem
#--------------------------------------------------------------------------------------------------------
        if tot_supply == tot_demand+warehouse_resv:
            print("Balanced Problem")
            # Instantiate a SimpleMinCostFlow solver.
            model_2 = pywrapgraph.SimpleMinCostFlow()

            # Add each arc.
            for i in range(0, len(start_nodes)):
                model_2.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], unit_costs[i])

            # Add node supplies.
            for i in range(0, len(supplies)):
                model_2.SetNodeSupply(i, supplies[i])

            # Find the minimum cost flow between 2 nodes.
            if model_2.Solve() == model_2.OPTIMAL:
                optimal = "Optimal"
                source_list = []
                dest_list = []
                flow_list = []
                capacity_list = []
                cost_list = []

                for i in range(model_2.NumArcs()):
                    cost = model_2.Flow(i) * model_2.UnitCost(i)
                    if cost > 0:
                        source_list.append(model_2.Tail(i)),
                        dest_list.append(model_2.Head(i)),
                        flow_list.append(model_2.Flow(i)),
                        capacity_list.append(model_2.Capacity(i)),
                        cost_list.append(cost)

            # Source Encoding
            for n, i in enumerate(source_list):
                if i == 0:
                    source_list[n] = "Minneapolis"
                elif i == 1:
                    source_list[n] = "Denver"
                elif i == 2:
                    source_list[n] = "Kansas City"
                elif i == 3:
                    source_list[n] = "Portland"
                elif i == 4:
                    source_list[n] = "Las Vegas"
                elif i == 5:
                    source_list[n] = "Atlanta"
                elif i == 6:
                    source_list[n] = "New Haven"
                elif i == 7:
                    source_list[n] = "Siracusa"
                elif i == 8:
                    source_list[n] = "Sacramento"
                elif i == 9:
                    source_list[n] = "New York"
                elif i == 10:
                    source_list[n] = "Washington DC"
                elif i == 11:
                    source_list[n] = "Chicago"
                elif i == 12:
                    source_list[n] = "Boston"
                elif i == 13:
                    source_list[n] = "San Francisco"
                elif i == 14:
                    source_list[n] = "Seattle"

            # Destinations Encoding
            for n, i in enumerate(dest_list):
                if i == 0:
                    dest_list[n] = "Minneapolis"
                elif i == 1:
                    dest_list[n] = "Denver"
                elif i == 2:
                    dest_list[n] = "Kansas City"
                elif i == 3:
                    dest_list[n] = "Portland"
                elif i == 4:
                    dest_list[n] = "Las Vegas"
                elif i == 5:
                    dest_list[n] = "Atlanta"
                elif i == 6:
                    dest_list[n] = "New Haven"
                elif i == 7:
                    dest_list[n] = "Siracusa"
                elif i == 8:
                    dest_list[n] = "Sacramento"
                elif i == 9:
                    dest_list[n] = "New York"
                elif i == 10:
                    dest_list[n] = "Washington DC"
                elif i == 11:
                    dest_list[n] = "Chicago"
                elif i == 12:
                    dest_list[n] = "Boston"
                elif i == 13:
                    dest_list[n] = "San Francisco"
                elif i == 14:
                    dest_list[n] = "Seattle"

            # Dataframe result
            new_df = pd.DataFrame(
                {'Source': source_list, 'Destination': dest_list, 'Flow': flow_list, 'Capacity': capacity_list,
                 'Cost': cost_list})
            del source_list, dest_list, flow_list, capacity_list, cost_list

            new_df["Capacity Utilization %"] = (new_df["Flow"] / new_df["Capacity"]) * 100
            new_df["Unused Capacity"] = (new_df["Capacity"] - new_df["Flow"])


            print('Minimum cost:', model_2.OptimalCost())
            print(new_df.head(50))

            table = html.Div([
                dash_table.DataTable(
                    id='datatable_id',
                    data=new_df.to_dict("records"),
                    columns=[
                        {"name": i, "id": i, "deletable": False, "selectable": True, "hideable": True} for i in
                        new_df.columns
                    ],

                    editable=True,  # allow editing of data inside all cells
                    filter_action="native",  # allow filtering of data by user ('native') or not ('none')
                    sort_action="native",  # enables data to be sorted per-column by user or not ('none')
                    sort_mode="multi",  # sort across 'multi' or 'single' columns
                    column_selectable="multi",  # allow users to select 'multi' or 'single' columns
                    row_selectable="multi",  # allow users to select 'multi' or 'single' rows
                    row_deletable=False,  # choose if user can delete a row (True) or not (False)
                    selected_columns=[],  # ids of columns that user selects
                    selected_rows=[],  # indices of rows that user selects
                    page_action='native',
                    style_data={
                      'whiteSpace' : 'normal',
                        'height' : 'auto'
                                },
                    style_table={'overflowX': 'scroll'},
                    style_cell={
                            'height': 'auto',
                            # all three widths are needed
                            'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                            'whiteSpace': 'normal'
                        },
#                    fixed_columns={'headers': True, 'data': 2},  # digit = number of columns fixed
                    fixed_rows={'headers': True, 'data': 0},
                    virtualization=False,
                    export_columns='all',  # 'all' or 'visible
                    export_format='xlsx',  # 'csv or 'none' or 'xlsx'
                    style_data_conditional=([
                        {
                            'if': {
                                'filter_query': '{Flow} = {Capacity}',
                                'column_id': 'Source'
                            },
                            'backgroundColor': 'black',
                            'color': 'red',
                            'fontWeight': 'bold',
                        },

                        {
                            'if': {
                                'filter_query': '{Flow} = {Capacity}',
                                'column_id': 'Destination'
                                 },
                            'backgroundColor': 'black',
                            'color': 'red',
                            'fontWeight': 'bold',
                        },

                        {
                            'if': {
                                'filter_query': '{Capacity Utilization %} = 100',
                                'column_id': 'Capacity Utilization %'
                                 },
                            'backgroundColor': 'black',
                            'color': 'red',
                            'fontWeight': 'bold',
                        },

                                        ]),
                    style_cell_conditional=[  # align text columns to left. By default they are aligned to right
                        {
                            'if': {'column_id': c},
                            'textAlign': 'left'
                        } for c in ['Source', 'Destination']

                    ],
                ),
            ])

            # Total Min. Cost
            res = model_2.OptimalCost()
            print("Total transportation cost = ", model_2.OptimalCost())
            print("Maximum Flow", model_2.MaximumFlow())

            return "Total Minimum Cost $ {}".format(int(res)), optimal, tot_supply, warehouse_resv, tot_demand, table, alert2


#-----------------------------------------------------------------------------------------------------------
        elif tot_supply < (tot_demand+warehouse_resv):
            short = (tot_demand+warehouse_resv)-tot_supply
            print("Excess Demand:", short)

            # Instantiate a SimpleMinCostFlow solver.
            model_2 = pywrapgraph.SimpleMinCostFlow()

            # Add each arc.
            for i in range(0, len(start_nodes)):
                model_2.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], unit_costs[i])

            # Add node supplies.
            for i in range(0, len(supplies)):
                model_2.SetNodeSupply(i, supplies[i])

            # Find the minimum cost flow between 2 nodes.
            if model_2.SolveMaxFlowWithMinCost() == model_2.OPTIMAL:
                optimal = "Optimal"
                source_list = []
                dest_list = []
                flow_list = []
                capacity_list = []
                cost_list = []

                for i in range(model_2.NumArcs()):
                    cost = model_2.Flow(i) * model_2.UnitCost(i)
                    if cost > 0:
                        source_list.append(model_2.Tail(i)),
                        dest_list.append(model_2.Head(i)),
                        flow_list.append(model_2.Flow(i)),
                        capacity_list.append(model_2.Capacity(i)),
                        cost_list.append(cost)

            # Lists
            print("Sources", source_list)
            print("Destinations", dest_list)
            print("Flow", flow_list)
            print("Capacity", capacity_list)
            print("Cost", cost_list)

            # Source Encoding
            for n, i in enumerate(source_list):
                if i == 0:
                    source_list[n] = "Minneapolis"
                elif i == 1:
                    source_list[n] = "Denver"
                elif i == 2:
                    source_list[n] = "Kansas City"
                elif i == 3:
                    source_list[n] = "Portland"
                elif i == 4:
                    source_list[n] = "Las Vegas"
                elif i == 5:
                    source_list[n] = "Atlanta"
                elif i == 6:
                    source_list[n] = "New Haven"
                elif i == 7:
                    source_list[n] = "Siracusa"
                elif i == 8:
                    source_list[n] = "Sacramento"
                elif i == 9:
                    source_list[n] = "New York"
                elif i == 10:
                    source_list[n] = "Washington DC"
                elif i == 11:
                    source_list[n] = "Chicago"
                elif i == 12:
                    source_list[n] = "Boston"
                elif i == 13:
                    source_list[n] = "San Francisco"
                elif i == 14:
                    source_list[n] = "Seattle"

            # Destinations Encoding
            for n, i in enumerate(dest_list):
                if i == 0:
                    dest_list[n] = "Minneapolis"
                elif i == 1:
                    dest_list[n] = "Denver"
                elif i == 2:
                    dest_list[n] = "Kansas City"
                elif i == 3:
                    dest_list[n] = "Portland"
                elif i == 4:
                    dest_list[n] = "Las Vegas"
                elif i == 5:
                    dest_list[n] = "Atlanta"
                elif i == 6:
                    dest_list[n] = "New Haven"
                elif i == 7:
                    dest_list[n] = "Siracusa"
                elif i == 8:
                    dest_list[n] = "Sacramento"
                elif i == 9:
                    dest_list[n] = "New York"
                elif i == 10:
                    dest_list[n] = "Washington DC"
                elif i == 11:
                    dest_list[n] = "Chicago"
                elif i == 12:
                    dest_list[n] = "Boston"
                elif i == 13:
                    dest_list[n] = "San Francisco"
                elif i == 14:
                    dest_list[n] = "Seattle"

            # Dataframe result
            new_df = pd.DataFrame(
                {'Source': source_list, 'Destination': dest_list, 'Flow': flow_list, 'Capacity': capacity_list,
                 'Cost': cost_list})
            del source_list, dest_list, flow_list, capacity_list, cost_list

            new_df["Capacity Utilization %"] = (new_df["Flow"] / new_df["Capacity"]) * 100
            new_df["Unused Capacity"] = (new_df["Capacity"] - new_df["Flow"])


            print('Minimum cost:', model_2.OptimalCost())
            print(new_df.head(50))

            table = html.Div([
                dash_table.DataTable(
                    id='datatable_id',
                    data=new_df.to_dict("records"),
                    columns=[
                        {"name": i, "id": i, "deletable": False, "selectable": True, "hideable": True} for i in
                        new_df.columns
                    ],

                    editable=True,  # allow editing of data inside all cells
                    filter_action="native",  # allow filtering of data by user ('native') or not ('none')
                    sort_action="native",  # enables data to be sorted per-column by user or not ('none')
                    sort_mode="multi",  # sort across 'multi' or 'single' columns
                    column_selectable="multi",  # allow users to select 'multi' or 'single' columns
                    row_selectable="multi",  # allow users to select 'multi' or 'single' rows
                    row_deletable=False,  # choose if user can delete a row (True) or not (False)
                    selected_columns=[],  # ids of columns that user selects
                    selected_rows=[],  # indices of rows that user selects
                    page_action='native',
#                    style_cell={'whiteSpace': 'normal', 'minWidth': 95, 'maxWidth': 95, 'width': 95},
                    fixed_rows={'headers': True, 'data': 0},
                    style_table={'overflowX': 'scroll'},
                    style_cell={
                        'height': 'auto',
                        # all three widths are needed
                        'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                        'whiteSpace': 'normal'
                               },
                    # fixed_columns={'headers': True, 'data': 2},  # digit = number of columns fixed

                    virtualization=False,
                    export_columns='all',  # 'all' or 'visible
                    export_format='xlsx',  # 'csv or 'none' or 'xlsx'
                    style_data_conditional=([
                        {
                            'if': {
                                'filter_query': '{Flow} = {Capacity}',
                                'column_id': 'Source'
                            },
                            'backgroundColor': 'black',
                            'color': 'red',
                            'fontWeight': 'bold',
                        },
                        {
                            'if': {
                                'filter_query': '{Flow} = {Capacity}',
                                'column_id': 'Destination'
                            },
                            'backgroundColor': 'black',
                            'color': 'red',
                            'fontWeight': 'bold',
                        },

                        {
                            'if': {
                                'filter_query': '{Capacity Utilization %} = 100',
                                'column_id': 'Capacity Utilization %'
                            },
                            'backgroundColor': 'black',
                            'color': 'red',
                            'fontWeight': 'bold',
                        },

                    ]),
                    style_cell_conditional=[  # align text columns to left. By default they are aligned to right
                        {
                            'if': {'column_id': c},
                            'textAlign': 'left'
                        } for c in ['Source', 'Destination']
                    ],
                ),
            ])

            # Total Min. Cost
            res = model_2.OptimalCost()
            print("Total transportation cost = ", model_2.OptimalCost())
            print("Maximum Flow", model_2.MaximumFlow())


            return "Total Minimum Cost $ {}".format(int(res)), optimal, tot_supply,warehouse_resv ,tot_demand, table, alert1


#---------------------------------------------------------------------------------------------------
        elif tot_supply > (tot_demand+warehouse_resv):
            excess = tot_supply-tot_demand
            print("Excess Supply:", excess)

            # Instantiate a SimpleMinCostFlow solver.
            model_2 = pywrapgraph.SimpleMinCostFlow()

            # Add each arc.
            for i in range(0, len(start_nodes)):
                model_2.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], unit_costs[i])

            # Add node supplies.
            for i in range(0, len(supplies)):
                model_2.SetNodeSupply(i, supplies[i])

            # Find the minimum cost flow between 2 nodes.
            if model_2.SolveMaxFlowWithMinCost() == model_2.OPTIMAL:
                optimal = "Optimal"
                source_list = []
                dest_list = []
                flow_list = []
                capacity_list = []
                cost_list = []

                for i in range(model_2.NumArcs()):
                    cost = model_2.Flow(i) * model_2.UnitCost(i)
                    if cost > 0:
                        source_list.append(model_2.Tail(i)),
                        dest_list.append(model_2.Head(i)),
                        flow_list.append(model_2.Flow(i)),
                        capacity_list.append(model_2.Capacity(i)),
                        cost_list.append(cost)
            # else:
            #     print('Demand and Supply do not match but model found the optimized solution')
            #     return "", "", "", "", "","", alert1

            # Lists
            print("Sources", source_list)
            print("Destinations", dest_list)
            print("Flow", flow_list)
            print("Capacity", capacity_list)
            print("Cost", cost_list)

            # Source Encoding
            for n, i in enumerate(source_list):
                if i == 0:
                    source_list[n] = "Minneapolis"
                elif i == 1:
                    source_list[n] = "Denver"
                elif i == 2:
                    source_list[n] = "Kansas City"
                elif i == 3:
                    source_list[n] = "Portland"
                elif i == 4:
                    source_list[n] = "Las Vegas"
                elif i == 5:
                    source_list[n] = "Atlanta"
                elif i == 6:
                    source_list[n] = "New Haven"
                elif i == 7:
                    source_list[n] = "Siracusa"
                elif i == 8:
                    source_list[n] = "Sacramento"
                elif i == 9:
                    source_list[n] = "New York"
                elif i == 10:
                    source_list[n] = "Washington DC"
                elif i == 11:
                    source_list[n] = "Chicago"
                elif i == 12:
                    source_list[n] = "Boston"
                elif i == 13:
                    source_list[n] = "San Francisco"
                elif i == 14:
                    source_list[n] = "Seattle"

            # Destinations Encoding
            for n, i in enumerate(dest_list):
                if i == 0:
                    dest_list[n] = "Minneapolis"
                elif i == 1:
                    dest_list[n] = "Denver"
                elif i == 2:
                    dest_list[n] = "Kansas City"
                elif i == 3:
                    dest_list[n] = "Portland"
                elif i == 4:
                    dest_list[n] = "Las Vegas"
                elif i == 5:
                    dest_list[n] = "Atlanta"
                elif i == 6:
                    dest_list[n] = "New Haven"
                elif i == 7:
                    dest_list[n] = "Siracusa"
                elif i == 8:
                    dest_list[n] = "Sacramento"
                elif i == 9:
                    dest_list[n] = "New York"
                elif i == 10:
                    dest_list[n] = "Washington DC"
                elif i == 11:
                    dest_list[n] = "Chicago"
                elif i == 12:
                    dest_list[n] = "Boston"
                elif i == 13:
                    dest_list[n] = "San Francisco"
                elif i == 14:
                    dest_list[n] = "Seattle"

            # Dataframe result
            new_df = pd.DataFrame(
                {'Source': source_list, 'Destination': dest_list, 'Flow': flow_list, 'Capacity': capacity_list,
                 'Cost': cost_list})
            del source_list, dest_list, flow_list, capacity_list, cost_list

            new_df["Capacity Utilization %"] = (new_df["Flow"] / new_df["Capacity"]) * 100
            new_df["Unused Capacity"] = (new_df["Capacity"] - new_df["Flow"])


            print('Minimum cost:', model_2.OptimalCost())
            print(new_df.head(50))

            table = html.Div([
                dash_table.DataTable(
                    id='datatable_id',
                    data=new_df.to_dict("records"),
                    columns=[
                        {"name": i, "id": i, "deletable": False, "selectable": True, "hideable": True} for i in
                        new_df.columns
                    ],

                    editable=True,  # allow editing of data inside all cells
                    filter_action="native",  # allow filtering of data by user ('native') or not ('none')
                    sort_action="native",  # enables data to be sorted per-column by user or not ('none')
                    sort_mode="multi",  # sort across 'multi' or 'single' columns
                    column_selectable="multi",  # allow users to select 'multi' or 'single' columns
                    row_selectable="multi",  # allow users to select 'multi' or 'single' rows
                    row_deletable=False,  # choose if user can delete a row (True) or not (False)
                    selected_columns=[],  # ids of columns that user selects
                    selected_rows=[],  # indices of rows that user selects
                    page_action='native',
#                    style_cell={'whiteSpace': 'normal', 'minWidth': 95, 'maxWidth': 95, 'width': 95},
                    style_table={'overflowX': 'scroll'},
                    style_cell={
                        'height': 'auto',
                        # all three widths are needed
                        'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                        'whiteSpace': 'normal'
                               },
                    # fixed_columns={'headers': True, 'data': 2},  # digit = number of columns fixed

                    fixed_rows={'headers': True, 'data': 0},
                    virtualization=False,
                    export_columns='all',  # 'all' or 'visible
                    export_format='xlsx',  # 'csv or 'none' or 'xlsx'
                    style_data_conditional=([
                        {
                            'if': {
                                'filter_query': '{Flow} = {Capacity}',
                                'column_id': 'Source'
                            },
                            'backgroundColor': 'black',
                            'color': 'red',
                            'fontWeight': 'bold',
                        },

                        {
                            'if': {
                                'filter_query': '{Flow} = {Capacity}',
                                'column_id': 'Destination'
                            },
                            'backgroundColor': 'black',
                            'color': 'red',
                            'fontWeight': 'bold',
                        },

                        {
                            'if': {
                                'filter_query': '{Capacity Utilization %} = 100',
                                'column_id': 'Capacity Utilization %'
                            },
                            'backgroundColor': 'black',
                            'color': 'red',
                            'fontWeight': 'bold',
                        },

                    ]),
                    style_cell_conditional=[  # align text columns to left. By default they are aligned to right
                        {
                            'if': {'column_id': c},
                            'textAlign': 'left'
                        } for c in ['Source', 'Destination']
                    ],
                ),
            ])

            # Total Min. Cost
            res = model_2.OptimalCost()
            print("Total transportation cost = ", model_2.OptimalCost())
            print("Maximum Flow", model_2.MaximumFlow())


            return "Total Minimum Cost $ {}".format(int(res)), optimal, tot_supply,warehouse_resv ,tot_demand, table, alert1


# Create bar chart
@app.callback(
    Output(component_id='bar-container', component_property='children'),
    [Input(component_id='datatable_id', component_property="derived_virtual_data"),
     Input(component_id='datatable_id', component_property='derived_virtual_selected_rows'),
     Input(component_id='datatable_id', component_property='derived_virtual_selected_row_ids'),
     Input(component_id='datatable_id', component_property='selected_rows'),
     Input(component_id='datatable_id', component_property='derived_virtual_indices'),
     Input(component_id='datatable_id', component_property='derived_virtual_row_ids'),
     Input(component_id='datatable_id', component_property='active_cell'),
     Input(component_id='datatable_id', component_property='selected_cells')]
)
def update_bar(all_rows_data, slctd_row_indices, slct_rows_names, slctd_rows,
               order_of_rows_indices, order_of_rows_names, actv_cell, slctd_cell):

    dff_new = pd.DataFrame(all_rows_data)
    print(dff_new.head(100))

    # used to highlight selected countries on bar chart
    colors = ['#7FDBFF' if i in slctd_row_indices else '#0074D9'
              for i in range(len(dff_new))]

    if "Destination" in dff_new and "Cost" in dff_new:
        return [
            dcc.Graph(id='bar-chart',
                      figure=px.bar(
                          data_frame=dff_new,
                          x="Destination",
                          y='Cost',
                          color= "Source",
                          text="Flow",
                          #textposition='auto',
                          labels={"Destination": "Retail Store/Warehouses", "Cost": "Transportation Cost"},
                        ).update_layout(showlegend=True, xaxis={'categoryorder': 'total ascending'})
 #                     .update_traces(marker_color=colors)
                        .update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="auto", x=0)) #,legend_title_text='DCs/Warehouses')
                        .update_traces(texttemplate='%{text:.2s}', textposition='inside')
                        #.update_layout(xaxis_tickangle=-45)
                        #.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                        #.update_layout(barmode='group')

                )
            ]


#------------------------------------------------
# No token = 'open-street-map', 'white-bg', 'carto-positron', 'carto-darkmatter', 'stamen- terrain', 'stamen-toner', 'stamen-watercolor'
# Token =     'basic', 'streets', 'outdoors', 'light', 'dark', 'satellite', 'satellite-streets'
#------------------------------------------------


# supply = {'S1': 80, 'S2': 80, 'S3': 60, 'S4': 80, 'S5': 60,}
# Warehouse'S6': 0, 'S7': 0, 'S8': 0, 'S9': 0}
# demand = {'D1': 60, 'D2': 50, 'D3': 60, 'D4': 60, 'D5': 60, 'D6': 60}
