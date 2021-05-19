#Import library---------------
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


#Lottie and Mapbox tokens-------------------------------------
url = "https://assets2.lottiefiles.com/packages/lf20_ftcfknxp.json"
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))
# Mapbox token
token = 'pk.eyJ1IjoiaWFudXZyYXQiLCJhIjoiY2tvNWQ1a2FkMHB6MTJ2cXdmeGt3MTdldyJ9.5vxPVvQdr6NL8hxVCZ1ecg'
#---------------------------------------------------------------------

# #Data-----------------------------
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()
df = pd.read_csv(DATA_PATH.joinpath("sc_location_md.csv"))

# Cleaning -------------------------------------------------------------
dff = df.drop(['location_description','is_internal'],axis=1)
# Filtering US data
dff = dff[dff['country']=='US']

supply_input_types = ['number','number','number','number','number','number','number','number','number']
demand_input_types = ['number','number','number','number','number','number']
alert0 = dbc.Alert("Message : Please feed the demand and supply units and select a solver to simulate! ", color="info",  dismissable=False),
alert1 = dbc.Alert("Alert : All constraints were not met due to  Incorrect values ", color="danger",  dismissable=False),
alert2 = dbc.Alert("Success : All constraints were met and model found the optimized result ", color="success", dismissable=True),

my_elements = [# Nodes elements
                                    {'data': {'id': 'dc 1', 'label': 'Minneapolis (DC1)'},
                                     'position': {'x': -100, 'y': -200},
                                     'classes': 'blue',
                                     'selected': True
                                     },

                                    {'data': {'id': 'dc 2', 'label': 'Denver (DC2)'},
                                     'position': {'x': -100, 'y': -150},
                                     'selected': True
                                     },

                                    {'data': {'id': 'dc 3', 'label': 'Kansas City (DC3)'},
                                     'position': {'x': -100, 'y': -100},
                                     'selected': True
                                     },

                                    {'data': {'id': 'dc 4', 'label': 'Portland (DC4)'},
                                     'position': {'x': -100, 'y': -50},
                                     'locked': True,
                                     'selected': True
                                     },

                                    {'data': {'id': 'dc 5', 'label': 'Las Vegas (DC5)'},
                                     'position': {'x': -100, 'y': 0},
                                     'locked': True,
                                      'selected': True
                                     },

                                    {'data': {'id': 'dc 6', 'label': 'Atlanta (DC6)'},
                                     'position': {'x': -100, 'y': 50},
                                     'locked': True,
                                     'selected': True
                                     },

                                    {'data': {'id': 'dc 7', 'label': 'New Haven (DC7)'},
                                     'position': {'x': -100, 'y': 100},
                                     'grabbable': True,
                                     'selected': True
                                     },

                                    {'data': {'id': 'dc 8', 'label': 'Siracusa (DC8)'},
                                     'position': {'x': -100, 'y': 150},
                                     'selected': True
                                     },

                                    {'data': {'id': 'dc 9', 'label': 'Sacramento (DC9)'},
                                     'position': {'x': -100, 'y': 200},
                                     'selected': True
                                     },

                                    {'data': {'id': 'r1', 'label': 'New York (R1)'},
                                     'position': {'x': 300, 'y': -250},
                                     'selectable': True,
                                     'selected': True,
                                     'classes': 'red'
                                     },

                                    {'data': {'id': 'r2', 'label': 'Washington DC(R2)'},
                                     'position': {'x': 300, 'y': -150},
                                     'selectable': True,
                                     'selected': True,
                                     'classes': 'red'

                                     },

                                    {'data': {'id': 'r3', 'label': 'Chicago (R3)'},
                                     'position': {'x': 300, 'y': -50},
                                     'selectable': True,
                                     'selected': True,
                                     'classes': 'red'
                                     },

                                    {'data': {'id': 'r4', 'label': 'Boston (R4)'},
                                     'position': {'x': 300, 'y': 50},
                                     'selectable': True,
                                     'classes': 'red',
                                     'selected': True
                                     },

                                    {'data': {'id': 'r5', 'label': 'San Francisco(R5)'},
                                     'position': {'x': 300, 'y':150},
                                     'selectable': True,
                                     'classes': 'red',
                                     'selected': True
                                     },

                                    {'data': {'id': 'r6', 'label': 'Seattle(R6)'},
                                     'position': {'x': 300, 'y': 250},
                                     'selectable': True,
                                     'classes': 'red',
                                     'selected': True
                                     },

                 # Edge elements
                                    {'data': {'source': 'dc 1', 'target': 'r1'}},
                                    {'data': {'source': 'dc 1', 'target': 'r2'}},
                                    {'data': {'source': 'dc 1', 'target': 'r3'}},
                                    {'data': {'source': 'dc 1', 'target': 'r4'}},
                                    {'data': {'source': 'dc 1', 'target': 'r5'}},
                                    {'data': {'source': 'dc 1', 'target': 'r6'}},
                                    {'data': {'source': 'dc 2', 'target': 'r1'}},
                                    {'data': {'source': 'dc 2', 'target': 'r2'}},
                                    {'data': {'source': 'dc 2', 'target': 'r3'}},
                                    {'data': {'source': 'dc 2', 'target': 'r4'}},
                                    {'data': {'source': 'dc 2', 'target': 'r5'}},
                                    {'data': {'source': 'dc 2', 'target': 'r6'}},
                                    {'data': {'source': 'dc 3', 'target': 'r1'}},
                                    {'data': {'source': 'dc 3', 'target': 'r2'}},
                                    {'data': {'source': 'dc 3', 'target': 'r3'}},
                                    {'data': {'source': 'dc 3', 'target': 'r4'}},
                                    {'data': {'source': 'dc 3', 'target': 'r5'}},
                                    {'data': {'source': 'dc 3', 'target': 'r6'}},
                                    {'data': {'source': 'dc 4', 'target': 'r1'}},
                                    {'data': {'source': 'dc 4', 'target': 'r2'}},
                                    {'data': {'source': 'dc 4', 'target': 'r3'}},
                                    {'data': {'source': 'dc 4', 'target': 'r4'}},
                                    {'data': {'source': 'dc 4', 'target': 'r5'}},
                                    {'data': {'source': 'dc 4', 'target': 'r6'}},
                                    {'data': {'source': 'dc 5', 'target': 'r1'}},
                                    {'data': {'source': 'dc 5', 'target': 'r2'}},
                                    {'data': {'source': 'dc 5', 'target': 'r3'}},
                                    {'data': {'source': 'dc 5', 'target': 'r4'}},
                                    {'data': {'source': 'dc 5', 'target': 'r5'}},
                                    {'data': {'source': 'dc 5', 'target': 'r6'}},
                                    {'data': {'source': 'dc 6', 'target': 'r1'}},
                                    {'data': {'source': 'dc 6', 'target': 'r2'}},
                                    {'data': {'source': 'dc 6', 'target': 'r3'}},
                                    {'data': {'source': 'dc 6', 'target': 'r4'}},
                                    {'data': {'source': 'dc 6', 'target': 'r5'}},
                                    {'data': {'source': 'dc 6', 'target': 'r6'}},
                                    {'data': {'source': 'dc 7', 'target': 'r1'}},
                                    {'data': {'source': 'dc 7', 'target': 'r2'}},
                                    {'data': {'source': 'dc 7', 'target': 'r3'}},
                                    {'data': {'source': 'dc 7', 'target': 'r4'}},
                                    {'data': {'source': 'dc 7', 'target': 'r5'}},
                                    {'data': {'source': 'dc 7', 'target': 'r6'}},
                                    {'data': {'source': 'dc 8', 'target': 'r1'}},
                                    {'data': {'source': 'dc 8', 'target': 'r2'}},
                                    {'data': {'source': 'dc 8', 'target': 'r3'}},
                                    {'data': {'source': 'dc 8', 'target': 'r4'}},
                                    {'data': {'source': 'dc 8', 'target': 'r5'}},
                                    {'data': {'source': 'dc 8', 'target': 'r6'}},
                                    {'data': {'source': 'dc 9', 'target': 'r1'}},
                                    {'data': {'source': 'dc 9', 'target': 'r2'}},
                                    {'data': {'source': 'dc 9', 'target': 'r3'}},
                                    {'data': {'source': 'dc 9', 'target': 'r4'}},
                                    {'data': {'source': 'dc 9', 'target': 'r5'}},
                                    {'data': {'source': 'dc 9', 'target': 'r6'}},
                                    ]

# App Layout ----------------------------------------------------------------------------------------------------------------------------------------
layout = dbc.Container([
    html.Br(),

               #Supply Info
                dbc.Row([

                    # i button for DC Supply
                    dbc.Col([
                        dbc.Button("i", id="popover-bottom-target4", color="warning", outline=True),
                    ], className="text-left", width={'size': 1.5, 'offset': 0, 'order': 0}),
                    dbc.Popover([
#                        dbc.PopoverHeader("All about Cost/Profit Matrix"),
                        dbc.PopoverBody(
                            "Feed the units of supply for each facility. Feed 0, if any facility is 'Closed' / 'Not Operable'."),
                    ],
                        id="popover4",
                        target="popover-bottom-target4",  # needs to be the same as dbc.Button id
                        trigger="hover",
                        placement="bottom",
                        is_open=False,
                    ),

                    # Supply Label
                    dbc.Col([
                        html.Label("Total DC Supply:",className="font-weight-bold text-primary"),
                            ],className="text-left", width={'size': 1.5, 'offset': 0, 'order': 1}),

                    # Supply total
                    dbc.Col([
                        html.Div(id='sum_supply',className="font-weight-bold text-primary"),
                            ],className="text-left", width={'size': 1, 'offset': 0, 'order': 2}),
                        ]),

                dbc.Row([
                    html.Label(
                        " Minneapolis __ Denver __Kansas City __Portland__Las Vegas __ Atlanta __ New Haven __ Siracusa __Sacramento")
                ]),

    #Supply Inputs
                dbc.Row([
                    dbc.Col([
                            dcc.Input(id='s1_ip',type='number',inputMode='numeric',value=1200,placeholder="DC1", debounce=True,  min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s2_ip',type='number',inputMode='numeric',value=800,placeholder="DC2", debounce=True,  min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s3_ip',type='number',inputMode='numeric',value=800,placeholder="DC3", debounce=True,  min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s4_ip',type='number',inputMode='numeric',value=450,placeholder="DC4", debounce=True,  min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s5_ip',type='number',inputMode='numeric',value=400,placeholder="DC5", debounce=True,  min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s6_ip',type='number',inputMode='numeric',value=400,placeholder="DC6", debounce=True,  min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s7_ip',type='number',inputMode='numeric',value=350,placeholder="DC7", debounce=True,  min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s8_ip',type='number',inputMode='numeric',value=200,placeholder="DC8", debounce=True,  min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='s9_ip',type='number',inputMode='numeric',value=None,placeholder="DC9", debounce=True,  min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),#, persistence=True, persistence_type='local')
                            ],className="text-left", width={'size': 6.5, 'offset': 0, 'order': 0}),
                    # zMin Cost
                    dbc.Col([
                        dcc.Loading(children=[html.Div(id='output_state')], color="#119DFF", type="dot",
                                    fullscreen=False),
                            ], className="text-primary font-weight-bold text-left border-white", width={'size': 2, 'offset': 8, 'order': 1}),

                   # Model Status
                    dbc.Col([
                        dcc.Loading(children=[html.Div(id='model_status')], color="#119DFF", type="default",
                                    fullscreen=False),
                            ], className="text-info font-weight-bold text-left border-primary", width={'size': 2, 'offset': 0, 'order': 2}),

                        ]),


    html.Br(),

                #Demand Info
                dbc.Row([

                    # i button for retail demand
                    dbc.Col([
                        dbc.Button("i", id="popover-bottom-target5", color="warning", outline=True),
                    ], className="text-left", width={'size': 1.5, 'offset': 0, 'order': 0}),
                    dbc.Popover([
#                        dbc.PopoverHeader("All about Cost/Profit Matrix"),
                        dbc.PopoverBody(
                            "Feed the demand for each retail store, feed 0 where there is no demand."),
                    ],
                        id="popover5",
                        target="popover-bottom-target5",  # needs to be the same as dbc.Button id
                        trigger="hover",
                        placement="bottom",
                        is_open=False,
                    ),

                    dbc.Col([
                        html.Label("Total Retail Demand:",className="font-weight-bold text-primary"),
                    ], className="text-left", width={'size': 1.5, 'offset': 0, 'order': 0}),

                    dbc.Col([
                        html.Div(id='sum_demand',className="font-weight-bold text-primary"),
                    ], className="text-left", width={'size': 1, 'offset': 0, 'order': 1}),
                    ]),

                dbc.Row([
                    html.Label(
                        " New York __Washingtn DC__Chicago __Boston__ San Francisco__Seattle "),
                        ]),

                    dbc.Row([
                    # Demand Inputs
                    dbc.Col([
                            dcc.Input(id='d1_ip',type="number",inputMode='numeric',value=800,placeholder="R1",debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='d2_ip',type="number",inputMode='numeric',value=850,placeholder="R2", debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='d3_ip',type="number",inputMode='numeric',value=900,placeholder="R3", debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='d4_ip',type="number",inputMode='numeric',value=1200,placeholder="R4", debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='d5_ip',type="number",inputMode='numeric',value=900,placeholder="R5", debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            dcc.Input(id='d6_ip',type="number",inputMode='numeric',value=750,placeholder="R6", debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),#, persistence=True, persistence_type='local'),
                            ],className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

                    # Submit Button
                    dbc.Col([
                            html.Button(id='submit_button', n_clicks=0, children='Submit'),
                            ],className="text-left", width={'size': 1, 'offset': 0, 'order': 1}),

                    # i button
                    dbc.Col([
                            dbc.Button("i", id="popover-bottom-target", color="info", outline=True),
                            ], className="text-left", width={'size': 0, 'offset': 0, 'order': 2}),

                            dbc.Popover([
                                    dbc.PopoverHeader("All about Solver Objective:"),
                                    dbc.PopoverBody(
                                        "MINIMIZE COST - If user wants to feed in costs (associated to each route)  to find Minimum Transportation Cost.\n"
                                        "\n\nMAXIMIZE PROFIT - If the user wants to feed in the profits (associated to each route) and want Maximum Overall Profit. "),
                                        ],
                                    id="popover",
                                    target="popover-bottom-target",  # needs to be the same as dbc.Button id
                                    trigger="hover",
                                    placement="bottom",
                                    is_open=False,
                                        ),

                    # Objective Scenario Radio Buttons
                    dbc.Col([
                        html.Label(['Solver Objective:'],className="font-weight-bold text-primary", style={'font-weight': 'bold'}),
                        dcc.RadioItems(
                            id='obj',
                            options=[
                                {'label': 'Minimize Cost ', 'value': 'cost_val'},
                                {'label': 'Maximize Profit ', 'value': 'profit_val'},

                            ],
                            value='cost_val',
                            style={"width": "50%"},
                        ),
                    ], className="text-left", width={'size': 3, 'offset': 0, 'order': 3}),

                    # Alert Message,
                    dbc.Col([
                              html.Div(id="the_alert", children=[]),
                            ], className="text-left", width={'size': 3, 'offset': 0, 'order': 4}),
                ]),


    html.Br(),

                dbc.Row([

                    # i button for cost matrix
                    dbc.Col([
                        dbc.Button("i", id="popover-bottom-target3", color="warning", outline=True),
                    ], className="text-left", width={'size': 1.5, 'offset': 0, 'order': 0}),
                    dbc.Popover([
                        dbc.PopoverHeader("All about Cost/Profit Matrix"),
                        dbc.PopoverBody(
                            "Feed unit cost associated to each route between a source and a destination and select solver 'Minimize'. \nIn case user have unit sales/profit associated to each route, select solver 'Maximize' "),
                    ],
                        id="popover3",
                        target="popover-bottom-target3",  # needs to be the same as dbc.Button id
                        trigger="hover",
                        placement="bottom",
                        is_open=False,
                    ),

                    dbc.Col([
                        html.Label(['Feed Cost/Profit associated to routes:'], className="font-weight-bold text-primary",style={'font-weight': 'bold'}),
                            ])
                        ]),

                dbc.Row([
                    dbc.Col([
                        html.Label("New York | Washington | Chicago |  ___Boston | SanFrancisco  | Seattle"),
                    ],className="text-left", width={'size': 0, 'offset': 1, 'order': 0}),
                        ]),

                    # Minneapolis
                    dbc.Col([
                        html.Label("Minniapolis-"),
                        dcc.Input(id='s1r1', type="number", inputMode='numeric', value=35, placeholder="R1",debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on',disabled=False, readOnly=False, required=True, size="30"),# , persistence=True, persistence_type='local'),
                        dcc.Input(id='s1r2', type="number", inputMode='numeric', value=30, placeholder="R2",debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on',disabled=False, readOnly=False, required=True, size="30"),# , persistence=True, persistence_type='local'),
                        dcc.Input(id='s1r3', type="number", inputMode='numeric', value=40, placeholder="R3",debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on',disabled=False, readOnly=False, required=True, size="30"),# , persistence=True, persistence_type='local'),
                        dcc.Input(id='s1r4', type="number", inputMode='numeric', value=32, placeholder="R4",debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on',disabled=False, readOnly=False, required=True, size="30"),# , persistence=True, persistence_type='local'),
                        dcc.Input(id='s1r5', type="number", inputMode='numeric', value=34, placeholder="R5",debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"), # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s1r6', type="number", inputMode='numeric', value=31, placeholder="R6",debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on',disabled=False, readOnly=False, required=True, size="30"), # , persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

                    # Denver
                    dbc.Col([
                        html.Label("Denver------"),
                        dcc.Input(id='s2r1', type="number", inputMode='numeric', value=37, placeholder="R1", debounce=True, min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s2r2', type="number", inputMode='numeric', value=40, placeholder="R2", debounce=True, min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s2r3', type="number", inputMode='numeric', value=42, placeholder="R3", debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s2r4', type="number", inputMode='numeric', value=25, placeholder="R4", debounce=True, min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s2r5', type="number", inputMode='numeric', value=23, placeholder="R5", debounce=True, min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s2r6', type="number", inputMode='numeric', value=28, placeholder="R6", debounce=True, min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),  # , persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

                    # Kansas
                    dbc.Col([
                        html.Label("Kansas City-"),
                        dcc.Input(id='s3r1', type="number", inputMode='numeric', value=40, placeholder="R1", debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s3r2', type="number", inputMode='numeric', value=15, placeholder="R2", debounce=True, min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s3r3', type="number", inputMode='numeric', value=20, placeholder="R3", debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s3r4', type="number", inputMode='numeric', value=28, placeholder="R4", debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s3r5', type="number", inputMode='numeric', value=29, placeholder="R5", debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s3r6', type="number", inputMode='numeric', value=32, placeholder="R6", debounce=True, min=0, max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,required=True, size="30"),  # , persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

                    # Portland
                    dbc.Col([
                        html.Label("Portland----"),
                        dcc.Input(id='s4r1', type="number", inputMode='numeric', value=40, placeholder="R1", debounce=True, min=0,max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False, required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s4r2', type="number", inputMode='numeric', value=15, placeholder="R2", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s4r3', type="number", inputMode='numeric', value=20, placeholder="R3", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s4r4', type="number", inputMode='numeric', value=28, placeholder="R4", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s4r5', type="number", inputMode='numeric', value=29, placeholder="R5", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s4r6', type="number", inputMode='numeric', value=39, placeholder="R6", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

                    # Las Vegas
                    dbc.Col([
                        html.Label("Las Vegas---"),
                        dcc.Input(id='s5r1', type="number", inputMode='numeric', value=40, placeholder="R1", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s5r2', type="number", inputMode='numeric', value=15, placeholder="R2", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s5r3', type="number", inputMode='numeric', value=20, placeholder="R3", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s5r4', type="number", inputMode='numeric', value=28, placeholder="R4", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s5r5', type="number", inputMode='numeric', value=37, placeholder="R5", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s5r6', type="number", inputMode='numeric', value=35, placeholder="R6", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

                    # Atlanta
                    dbc.Col([
                        html.Label("Atlanta------"),
                        dcc.Input(id='s6r1', type="number", inputMode='numeric', value=28, placeholder="R1", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s6r2', type="number", inputMode='numeric', value=43, placeholder="R2", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s6r3', type="number", inputMode='numeric', value=29, placeholder="R3", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s6r4', type="number", inputMode='numeric', value=34, placeholder="R4", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s6r5', type="number", inputMode='numeric', value=22, placeholder="R5", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s6r6', type="number", inputMode='numeric', value=23, placeholder="R6", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

                    # New Haven
                    dbc.Col([
                        html.Label("New Haven-"),
                        dcc.Input(id='s7r1', type="number", inputMode='numeric', value=32, placeholder="R1", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s7r2', type="number", inputMode='numeric', value=41, placeholder="R2", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s7r3', type="number", inputMode='numeric', value=40, placeholder="R3", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s7r4', type="number", inputMode='numeric', value=26, placeholder="R4", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s7r5', type="number", inputMode='numeric', value=45, placeholder="R5", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s7r6', type="number", inputMode='numeric', value=34, placeholder="R6", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

                    # Siracusa
                    dbc.Col([
                        html.Label("Siracusa----"),
                        dcc.Input(id='s8r1', type="number", inputMode='numeric', value=21, placeholder="R1", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s8r2', type="number", inputMode='numeric', value=34, placeholder="R2", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s8r3', type="number", inputMode='numeric', value=22, placeholder="R3", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s8r4', type="number", inputMode='numeric', value=43, placeholder="R4", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s8r5', type="number", inputMode='numeric', value=30, placeholder="R5", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s8r6', type="number", inputMode='numeric', value=21, placeholder="R6", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

                    # Sacramento
                    dbc.Col([
                        html.Label("Sacramento"),
                        dcc.Input(id='s9r1', type="number", inputMode='numeric', value=40, placeholder="R1", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s9r2', type="number", inputMode='numeric', value=15, placeholder="R2", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s9r3', type="number", inputMode='numeric', value=20, placeholder="R3", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s9r4', type="number", inputMode='numeric', value=28, placeholder="R4", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s9r5', type="number", inputMode='numeric', value=40, placeholder="R5", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                        dcc.Input(id='s9r6', type="number", inputMode='numeric', value=21, placeholder="R6", debounce=True, min=0,
                                  max=2000, step=1, minLength=0, maxLength=50, autoComplete='on', disabled=False, readOnly=False,
                                  required=True, size="30"),  # , persistence=True, persistence_type='local'),
                    ], className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),




    html.Br(),

                # Map Mode radio items
                html.Div([
                    html.Label(['Map View:'], style={'font-weight': 'bold'}, className="font-weight-bold text-primary"),
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
                        dcc.Loading(children=[dcc.Graph(id="mymap",config={ 'displayModeBar': False })],color="#119DFF", type="cube", fullscreen=False),
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
                                elements= my_elements
                                         )
                                ]),
                            ], className="text-left", width={'size': 4, 'offset': 3, 'order': 0}),
                        ]),


                dbc.Row([
                    # Dash datatable (sensitivity Analysis)
                    dbc.Col([
                        html.Label("Sensitivity Analysis", className="font-weight-bold text-primary"),
                        html.Div(id="output-datatable2", style={'width': '80vh', 'height': '50vh'}),
                            ],className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

                    # Dash datatable (Routes)
                    dbc.Col([
                        html.Label("Desired Routes and Units to transfer",className="font-weight-bold text-primary"),
                        html.Div(id="output-datatable", style={'width': '100vh', 'height': '50vh'}),
                            ],className="text-left", width={'size': 0, 'offset': 1, 'order': 0}),
                ]),

html.Br(),
html.Br(),
html.Br(),
html.Br(),
html.Br(),
html.Br(),
html.Br(),




                dbc.Row([
                    # Bar Graph
                    dbc.Col([
                        html.Label("Change in Cost", className="font-weight-bold text-primary"),
                        html.Div(id='sensitivity_bar', style={'width': '200vh', 'height': '100vh'}),
                    ], className="text-left", width={'size': 12, 'offset': 0, 'order': 1})

                ])








],fluid=True),




# Callbacks ------------------------------------------------------------------------

# i Button
@app.callback(
    Output("popover", "is_open"),
    [Input("popover-bottom-target", "n_clicks")],
    [State("popover", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



# Map
@app.callback(
    Output('mymap', 'figure'),
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
    [Output('output_state', 'children'),
     Output('model_status', 'children'),
     Output('sum_supply', 'children'),
     Output('sum_demand', 'children'),
     Output("output-datatable", "children"),
     Output("output-datatable2", "children"),
     Output("the_alert", "children")],

    [Input('submit_button',  'n_clicks')],
    [State('obj', 'value')], # Objective
    [State('s1_ip', 'value'),State('s2_ip', 'value'),State('s3_ip', 'value'),State('s4_ip', 'value'),State('s5_ip', 'value'),State('s6_ip', 'value'),State('s7_ip', 'value'),State('s8_ip', 'value'),State('s9_ip', 'value')], # supply
    [State('d1_ip', 'value'),State('d2_ip', 'value'),State('d3_ip', 'value'),State('d4_ip', 'value'),State('d5_ip', 'value'),State('d6_ip', 'value')],# demand
    [State('s1r1', 'value'), State('s1r2', 'value'), State('s1r3', 'value'), State('s1r4', 'value'),State('s1r5', 'value'), State('s1r6', 'value')], # cost
    [State('s2r1', 'value'), State('s2r2', 'value'), State('s2r3', 'value'), State('s2r4', 'value'), State('s2r5', 'value'),State('s2r6', 'value')],
    [State('s3r1', 'value'), State('s3r2', 'value'), State('s3r3', 'value'), State('s3r4', 'value'), State('s3r5', 'value'),State('s3r6', 'value')],
    [State('s4r1', 'value'), State('s4r2', 'value'), State('s4r3', 'value'), State('s4r4', 'value'), State('s4r5', 'value'),State('s4r6', 'value')],
    [State('s5r1', 'value'), State('s5r2', 'value'), State('s5r3', 'value'), State('s5r4', 'value'),State('s5r5', 'value'),State('s5r6', 'value')],
    [State('s6r1', 'value'), State('s6r2', 'value'), State('s6r3', 'value'), State('s6r4', 'value'),State('s6r5', 'value'),State('s6r6', 'value')],
    [State('s7r1', 'value'), State('s7r2', 'value'), State('s7r3', 'value'), State('s7r4', 'value'), State('s7r5', 'value'),State('s7r6', 'value')],
    [State('s8r1', 'value'), State('s8r2', 'value'), State('s8r3', 'value'), State('s8r4', 'value'),State('s8r5', 'value'),State('s8r6', 'value')],
    [State('s9r1', 'value'), State('s9r2', 'value'), State('s9r3', 'value'), State('s9r4', 'value'), State('s9r5', 'value'),State('s9r6', 'value')],

)
def update_data(num_clicks, val ,
                 s1,s2,s3,s4,s5,s6,s7,s8,s9,
                 d1,d2,d3,d4,d5,d6,
                 s1r1,s1r2,s1r3,s1r4,s1r5,s1r6,
                 s2r1,s2r2,s2r3,s2r4,s2r5,s2r6,
                 s3r1,s3r2,s3r3,s3r4,s3r5,s3r6,
                 s4r1,s4r2,s4r3,s4r4,s4r5,s4r6,
                 s5r1,s5r2,s5r3,s5r4,s5r5,s5r6,
                 s6r1,s6r2,s6r3,s6r4,s6r5,s6r6,
                 s7r1,s7r2,s7r3,s7r4,s7r5,s7r6,
                 s8r1,s8r2,s8r3,s8r4,s8r5,s8r6,
                 s9r1,s9r2,s9r3,s9r4,s9r5,s9r6,
                ):

    if s1==s2==s3==s4==s5==s6==s7==s8==s9==None and d1==d2==d3==d4==d5==d6==None or s9 ==None :
#        raise PreventUpdate
        return "", "", "", "", "", "", alert0

    elif num_clicks>=0:

        # Sources and Destinations: List
        Supply = ['Minneapolis', 'Denver', 'Kansas City', 'Portland', 'Las Vegas', 'Atlanta', 'New Haven', 'Siracusa', 'Sacramento']
        Demand = ['New York', 'Washington DC', 'Chicago', 'Boston', 'San Francisco', 'Seattle']

        # Supply from each node
        supply = {'Minneapolis': s1, 'Denver': s2, 'Kansas City': s3, 'Portland': s4, 'Las Vegas': s5, 'Atlanta': s6, 'New Haven': s7, 'Siracusa': s8, 'Sacramento': s9}
        print('Supply', supply)

        tot_supply = sum(supply.values())
        print("Total Supply", tot_supply)

        # Demand for each node
        demand = {'New York': d1, 'Washington DC': d2, 'Chicago': d3, 'Boston': d4, 'San Francisco': d5, 'Seattle': d6}
        print('Demand', demand)

        tot_demand = sum(demand.values())
        print("Total Demand", tot_demand)
        # Cost for all Supply-Demand arcs
        cost = {'Minneapolis': {'New York': s1r1, 'Washington DC': s1r2, 'Chicago': s1r3, 'Boston': s1r4, 'San Francisco': s1r5, 'Seattle': s1r6},
                'Denver':      {'New York': s2r1, 'Washington DC': s2r2, 'Chicago': s2r3, 'Boston': s2r4, 'San Francisco': s2r5, 'Seattle': s2r6},
                'Kansas City': {'New York': s3r1, 'Washington DC': s3r2, 'Chicago': s3r3, 'Boston': s3r4, 'San Francisco': s3r5, 'Seattle': s3r6},
                'Portland':     {'New York':s4r1, 'Washington DC': s4r2, 'Chicago': s4r3, 'Boston': s4r4, 'San Francisco': s4r5, 'Seattle': s4r6},
                'Las Vegas':    {'New York':s5r1, 'Washington DC': s5r2, 'Chicago': s5r3, 'Boston': s5r4, 'San Francisco': s5r5, 'Seattle': s5r6},
                'Atlanta':      {'New York':s6r1, 'Washington DC': s6r2, 'Chicago': s6r3, 'Boston': s6r4, 'San Francisco': s6r5,'Seattle': s6r6},
                'New Haven':    {'New York': s7r1, 'Washington DC': s7r2, 'Chicago': s7r3, 'Boston': s7r4, 'San Francisco': s7r5,'Seattle': s7r6},
                'Siracusa':     {'New York': s8r1, 'Washington DC': s8r2, 'Chicago': s8r3, 'Boston': s8r4, 'San Francisco': s8r5, 'Seattle': s8r6},
                'Sacramento':   {'New York': s9r1, 'Washington DC': s9r2, 'Chicago': s9r3, 'Boston': s9r4, 'San Francisco': s9r5,'Seattle': s9r6},
                }

        # Balanced/Unbalanced Problem
        if tot_supply == tot_demand:
            print("Balanced Problem")

        elif tot_supply < tot_demand:               # appending dummy warehouse
            short = tot_demand-tot_supply
            print("Excess Supply:", short)
            Supply.append("Dummy_Warehouse")        #Supply list
            supply["Dummy_Warehouse"] = short      #Supply dict (key:value)
            print('Supply', supply)
            cost = {'Minneapolis': {'New York': s1r1, 'Washington DC': s1r2, 'Chicago': s1r3, 'Boston': s1r4, 'San Francisco': s1r5, 'Seattle': s1r6},
                    'Denver':      {'New York': s2r1, 'Washington DC': s2r2, 'Chicago': s2r3, 'Boston': s2r4, 'San Francisco': s2r5, 'Seattle': s2r6},
                    'Kansas City': {'New York': s3r1, 'Washington DC': s3r2, 'Chicago': s3r3, 'Boston': s3r4, 'San Francisco': s3r5, 'Seattle': s3r6},
                    'Portland':     {'New York':s4r1, 'Washington DC': s4r2, 'Chicago': s4r3, 'Boston': s4r4, 'San Francisco': s4r5, 'Seattle': s4r6},
                    'Las Vegas':    {'New York':s5r1, 'Washington DC': s5r2, 'Chicago': s5r3, 'Boston': s5r4, 'San Francisco': s5r5, 'Seattle': s5r6},
                    'Atlanta':      {'New York':s6r1, 'Washington DC': s6r2, 'Chicago': s6r3, 'Boston': s6r4, 'San Francisco': s6r5,'Seattle': s6r6},
                    'New Haven':    {'New York': s7r1, 'Washington DC': s7r2, 'Chicago': s7r3, 'Boston': s7r4, 'San Francisco': s7r5,'Seattle': s7r6},
                    'Siracusa':     {'New York': s8r1, 'Washington DC': s8r2, 'Chicago': s8r3, 'Boston': s8r4, 'San Francisco': s8r5, 'Seattle': s8r6},
                    'Sacramento':   {'New York': s9r1, 'Washington DC': s9r2, 'Chicago': s9r3, 'Boston': s9r4, 'San Francisco': s9r5,'Seattle': s9r6},
                    'Dummy_Warehouse': {'New York': 0, 'Washington DC': 0, 'Chicago': 0, 'Boston': 0,'San Francisco': 0, 'Seattle': 0},

                    }

        elif tot_supply > tot_demand:               # appending dummy retailer
            excess = tot_supply-tot_demand
            print("Excess Supply:", excess)
            Demand.append("Dummy_Retailer")        #Demand list
            demand["Dummy_Retailer"] = excess      #Demand dict (key:value)
            print('Demand', demand)
            cost = {'Minneapolis': {'New York': s1r1, 'Washington DC': s1r2, 'Chicago': s1r3, 'Boston': s1r4, 'San Francisco': s1r5, 'Seattle': s1r6, 'Dummy_Retailer': 0},
                    'Denver': {'New York': s2r1, 'Washington DC': s2r2, 'Chicago': s2r3, 'Boston': s2r4, 'San Francisco': s2r5, 'Seattle': s2r6, 'Dummy_Retailer': 0},
                    'Kansas City': {'New York': s3r1, 'Washington DC': s3r2, 'Chicago': s3r3, 'Boston': s3r4, 'San Francisco': s3r5, 'Seattle': s3r6, 'Dummy_Retailer': 0},
                    'Portland': {'New York': s4r1, 'Washington DC': s4r2, 'Chicago': s4r3, 'Boston': s4r4, 'San Francisco': s4r5, 'Seattle': s4r6, 'Dummy_Retailer': 0},
                    'Las Vegas': {'New York': s5r1, 'Washington DC': s5r2, 'Chicago': s5r3, 'Boston': s5r4,'San Francisco': s5r5, 'Seattle': s5r6, 'Dummy_Retailer': 0},
                    'Atlanta': {'New York': s6r1, 'Washington DC': s6r2, 'Chicago': s6r3, 'Boston': s6r4, 'San Francisco': s6r5, 'Seattle': s6r6, 'Dummy_Retailer': 0},
                    'New Haven': {'New York': s7r1, 'Washington DC': s7r2, 'Chicago': s7r3, 'Boston': s7r4, 'San Francisco': s7r5, 'Seattle': s7r6, 'Dummy_Retailer': 0},
                    'Siracusa': {'New York': s8r1, 'Washington DC': s8r2, 'Chicago': s8r3, 'Boston': s8r4,'San Francisco': s8r5, 'Seattle': s8r6, 'Dummy_Retailer': 0},
                    'Sacramento': {'New York': s9r1, 'Washington DC': s9r2, 'Chicago': s9r3, 'Boston': s9r4,'San Francisco': s9r5, 'Seattle': s9r6, 'Dummy_Retailer': 0},
                    }

        # Setting Model
        if val=="cost_val":
            model = LpProblem("Transportation", LpMinimize)
        elif val=="profit_val":
            model = LpProblem("Transportation", LpMaximize)

        # Routes or Flows
        routes = [(i, j) for i in Supply for j in Demand]
        print("Total Routes", len(routes))


        # Defining Decision Variables
        X = LpVariable.dicts("Route", (Supply, Demand), 0)

        # Defining Objective Function
        model += lpSum(X[i][j] * cost[i][j] for (i, j) in routes)
        # model

        # Demand Constraints
        for j in Demand:
            model += lpSum(X[i][j] for i in Supply) >= demand[j]
        # Supply Constraints
        for i in Supply:
            model += lpSum(X[i][j] for j in Demand) <= supply[i]





        # Evaluate Model
        if model.solve() == 1:
            print("Model Feasibility:", model.solve())
            optimiality = LpStatus[model.status]
            print("Status:", optimiality)

            # Decision Varibles and associated costs
            name_list = []
            val_list = []
            for v in model.variables():
                if v.varValue > 0:
                    name_list.append(v.name)
                    val_list.append(v.varValue)
            cost_df = pd.DataFrame({'Route': name_list, 'Units': val_list})
            del name_list, val_list
            print(cost_df.head(20))
            # ------------------------------------------------------------
            # for v in model.variables():
            #     if v.varValue > 0:
            #         print('Cost for', v.name, "=", v.varValue)

            table = html.Div(
                [
                    dash_table.DataTable(
                        id='datatable_id',
                        data=cost_df.to_dict("records"),
                        columns=[
                            {'name': 'Route', 'id': 'Route', 'type': 'text', 'editable': False, 'selectable': True},
                            {'name': 'Units', 'id': 'Units', 'type': 'numeric', 'editable': False, "hideable": True, 'selectable': True},
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
                        style_cell={'whiteSpace': 'normal', 'minWidth': 95, 'maxWidth': 95, 'width': 95},
                        fixed_rows={'headers': True, 'data': 0},
                        virtualization=False,
                        export_columns='all',  # 'all' or 'visible
                        export_format='xlsx',  # 'csv or 'none' or 'xlsx'
                        style_data_conditional=(
                            [

                                {
                                    'if': {  # # Cost > 0
                                        'filter_query': '{Units} > 0',
                                        'column_id': 'Units'
                                    },
                                    'fontWeight': 'bold',
                                    'color': 'blue',
                                },

                            ]
                                            )
                    ),
                ]
            )


            # Total Min. Cost
            res = value(model.objective)
            print("Total transportation cost = ", value(model.objective))

            # Sensitiviy Analysis
            '''
            cname = name of the constraint
            cinfo = info about the constraint such as shadow price and slack variable
            '''

            if tot_supply == tot_demand:
                entity = ['Demand', 'Demand', 'Demand', 'Demand', 'Demand', 'Demand',
                          'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', ]

                facility = ["New York", "Washington DC", "Chicago", "Boston", "San Francisco", "Seattle",
                            "Minneapolis", "Denver", "Kansas City", "Portland", "Las Vegas", "Atlanta", "New Haven", "Siracusa", "Sacramento"]
                # print('Balanced Transportation Problem')
                SA = [{'Constraint No. ': cname, 'Slack Values': cinfo.slack, 'Shadow price': cinfo.pi} for cname, cinfo in model.constraints.items()]
                sensitivity_df = pd.DataFrame(SA)
                sensitivity_df['Facility'] = facility
                sensitivity_df['Entity'] = entity
                print(sensitivity_df.head(20))

            elif tot_supply < tot_demand:
                entity = ['Demand', 'Demand', 'Demand', 'Demand', 'Demand', 'Demand',
                          'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply','Dummy Supply' ]

                facility = ["New York", "Washington DC", "Chicago", "Boston", "San Francisco", "Seattle",
                            "Minneapolis", "Denver", "Kansas City", "Portland", "Las Vegas", "Atlanta", "New Haven", "Siracusa", "Sacramento", 'Dummy Supplier']

                SA = [{'Constraint No. ': cname, 'Slack Values': cinfo.slack, 'Shadow price': cinfo.pi} for cname, cinfo in model.constraints.items()]
                sensitivity_df = pd.DataFrame(SA)
                sensitivity_df['Facility'] = facility
                sensitivity_df['Entity'] = entity
                print(sensitivity_df.head(20))

            elif tot_supply > tot_demand:
                # print("Excess Supply:", tot_supply - tot_demand)
                entity = ['Demand', 'Demand', 'Demand', 'Demand', 'Demand', 'Demand', 'Dummy Demand',
                          'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply' ]

                facility = ["New York", "Washington DC", "Chicago", "Boston", "San Francisco", "Seattle", 'Dummy Retailer' ,
                            "Minneapolis", "Denver", "Kansas City", "Portland", "Las Vegas", "Atlanta", "New Haven", "Siracusa", "Sacramento"]

                SA = [{'Constraint No. ': cname, 'Slack Values': cinfo.slack, 'Shadow price': cinfo.pi} for cname, cinfo in model.constraints.items()]
                sensitivity_df = pd.DataFrame(SA)
                sensitivity_df['Facility'] = facility
                sensitivity_df['Entity'] = entity
                print(sensitivity_df.head(20))


            table2 = html.Div(
                [
                    dash_table.DataTable(
                        id='datatable_id2',
                        data=sensitivity_df.to_dict("records"),
                        columns=[
                            {"name": i, "id": i, "deletable": False, "selectable": True, "hideable": True}  for i in sensitivity_df.columns
                                ],
                        tooltip_delay=0,  # 1000
                        tooltip_duration=None,  # 2000
                        # column headers
                        tooltip_header={
                            'Slack Values': 'Amount of resource that is "unused" for each constraint',
                            'Shadow price': 'Change in amount of "Total Transportation Cost" for each unit change in Demand/Supply',
                        },
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
                        style_cell={'whiteSpace': 'normal', 'minWidth': 95, 'maxWidth': 95, 'width': 95},
                        fixed_rows={'headers': True, 'data': 0},
                        virtualization=False,
                        export_columns='all',  # 'all' or 'visible
                        export_format='xlsx',  # 'csv or 'none' or 'xlsx'
                        style_data_conditional=(
                            [
                                {
                                    'if': {
                                        'filter_query': '{Entity} = "Dummy Demand"',
                                    },
                                    'backgroundColor': '#FFFF00',
                                },
                                {
                                    'if': {
                                        'filter_query': '{Entity} = "Dummy Supply"',
                                    },
                                    'backgroundColor': '#FFFF00',
                                },

                                {
                                    'if': {  # Cost = 0
                                        'filter_query': '{Shadow price} > 0',
                                        'column_id': 'Shadow price'
                                    },
                                    #'backgroundColor': 'red',
                                    'fontWeight': 'bold',
                                    'color': 'red',
                                },
                                {
                                    'if': {  # # Cost > 0
                                        'filter_query': '{Shadow price} < 0',
                                        'column_id': 'Shadow price'
                                    },
                                    'fontWeight': 'bold',
                                    'color': 'blue',
                                },

                            ]
                        )
                    ),
                ]
            )

#            print(model)
            return "Total Min. Cost $ {}".format(int(res)), optimiality, tot_supply, tot_demand,  table,table2, alert2



        elif model.solve() ==-1:
            print("Model Feasibility:", model.solve())
            optimiality = LpStatus[model.status]
            print("Status:", optimiality)

            # Decision Varibles and associated costs
            name_list = []
            val_list = []
            for v in model.variables():
                if v.varValue > 0:
                    name_list.append(v.name)
                    val_list.append(v.varValue)
            cost_df = pd.DataFrame({'Route': name_list, 'Units': val_list})
            del name_list, val_list
            # ------------------------------------------------------------
            # for v in model.variables():
            #     if v.varValue > 0:
            #         print('Cost for', v.name, "=", v.varValue)

            table = html.Div(
                [
                    dash_table.DataTable(
                        id='datatable_id',
                        data=cost_df.to_dict("records"),
                        columns=[
                            {'name': 'Route', 'id': 'Route', 'type': 'text', 'editable': False, 'selectable': True},
                            {'name': 'Units', 'id': 'Units', 'type': 'numeric', 'editable': False, "hideable": True,
                             'selectable': True},
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
                        style_cell={'whiteSpace': 'normal', 'minWidth': 95, 'maxWidth': 95, 'width': 95},
                        fixed_rows={'headers': True, 'data': 0},
                        virtualization=False,
                        export_columns='all',  # 'all' or 'visible
                        export_format='xlsx',  # 'csv or 'none' or 'xlsx'
                        style_data_conditional=(
                            [
                                # {
                                #     'if': {  # Cost = 0
                                #         'filter_query': '{Units} = 0',
                                #         'column_id': 'Cost'
                                #     },
                                #     'backgroundColor': 'gray',
                                # },
                                {
                                    'if': {
                                        'filter_query': '{Units} > 0',
                                        'column_id': 'Units'
                                    },
                                    'fontWeight': 'bold',
                                    'color': 'blue',
                                },
                            ]
                        )
                    ),
                ]
            )

            # Total Min. Cost
            res = value(model.objective)
            print("Total transportation cost = ", value(model.objective))

            # Sensitiviy Analysis
            '''
            cname = name of the constraint
            cinfo = info about the constraint such as shadow price and slack variable
            '''
            SA = [{'Constraint No. ': cname, 'Slack Values': cinfo.slack, 'Shadow price': cinfo.pi} for cname, cinfo in model.constraints.items()]
#
            # sensitivity_df = pd.DataFrame(SA, index=["Demand 1","Demand 2","Demand 3","Demand 4","Demand 5","Demand 6",
            #                                          "Supply 1","Supply 2","Supply 3","Supply 4","Supply 5","Supply 6","Supply 7","Supply 8","Supply 9"])

            entity = ['Demand', 'Demand', 'Demand', 'Demand', 'Demand', 'Demand',
                      'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', 'Supply', ]
            facility = ["New York", "Washington DC", "Chicago", "Boston", "San Francisco", "Seattle",
                        "Minneapolis", "Denver", "Kansas City", "Portland", "Las Vegas", "Atlanta", "New Haven",
                        "Siracusa", "Sacramento"]

            sensitivity_df = pd.DataFrame(SA)
            sensitivity_df['Facility'] = facility
            sensitivity_df['Entity'] = entity

#            sensitivity_df['Constraint Name'] = sensitivity_df.index
            print(sensitivity_df.head(20))

            table2 = html.Div(   #sensitivity Analysis
                [
                    dash_table.DataTable(
                        id='datatable_id2',
                        data=sensitivity_df.to_dict("records"),
                        columns=[
                            {"name": i, "id": i, "deletable": False, "selectable": True, "hideable": True}  for i in sensitivity_df.columns
                                ],
                        tooltip_delay=0,  # 1000
                        tooltip_duration=None,  # 2000
                        # column headers
                        tooltip_header={
                            'Slack Values': 'Amount of resource that is "unused" for each constraint',
                            'Shadow price': 'Change in amount of "Total Transportation Cost" for each unit change in Demand/Supply',
                        },
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
                        style_cell={'whiteSpace': 'normal', 'minWidth': 95, 'maxWidth': 95, 'width': 95},
                        fixed_rows={'headers': True, 'data': 0},
                        virtualization=False,
                        export_columns='all',  # 'all' or 'visible
                        export_format='xlsx',  # 'csv or 'none' or 'xlsx'
                        style_data_conditional=(
                            [

                                {
                                    'if': {
                                        'filter_query': '{Entity} = "Dummy Demand"',
                                    },
                                    'backgroundColor': '#FFFF00',
                                },

                                {
                                    'if': {  # Cost = 0
                                        'filter_query': '{Shadow price} > 0',
                                        'column_id': 'Shadow price'
                                    },
                                    #'backgroundColor': 'red',
                                    'fontWeight': 'bold',
                                    'color': 'red',
                                },
                                {
                                    'if': {  # # Cost > 0
                                        'filter_query': '{Shadow price} < 0',
                                        'column_id': 'Shadow price'
                                    },
                                    'fontWeight': 'bold',
                                    'color': 'blue',
                                },
                            ]
                        )
                    ),
                ]
            )

            return "Total Minimum Cost $ {}".format(int(res)), optimiality, tot_supply, tot_demand,  table,table2, alert1





# Create bar chart
@app.callback(
    Output(component_id='sensitivity_bar', component_property='children'),
    [Input(component_id='datatable_id2', component_property="derived_virtual_data"),
     Input(component_id='datatable_id2', component_property='derived_virtual_selected_rows'),
     Input(component_id='datatable_id2', component_property='derived_virtual_selected_row_ids'),
     Input(component_id='datatable_id2', component_property='selected_rows'),
     Input(component_id='datatable_id2', component_property='derived_virtual_indices'),
     Input(component_id='datatable_id2', component_property='derived_virtual_row_ids'),
     Input(component_id='datatable_id2', component_property='active_cell'),
     Input(component_id='datatable_id2', component_property='selected_cells')]
)
def update_bar(all_rows_data, slctd_row_indices, slct_rows_names, slctd_rows,
               order_of_rows_indices, order_of_rows_names, actv_cell, slctd_cell):

    dff_bar = pd.DataFrame(all_rows_data)
    print(dff_bar.head(100))
    # used to highlight selected countries on bar chart
    colors = ['#7FDBFF' if i in slctd_row_indices else '#0074D9'
              for i in range(len(dff_bar))]

    if "Facility" in dff_bar and "Shadow price" in dff_bar:
        return [
            dcc.Graph(id='bar-chart',
                      figure=px.bar(
                          data_frame=dff_bar,
                          x="Facility",
                          y='Shadow price',
                          color="Entity",
                          text="Shadow price",
                          # textposition='auto',
                          #labels={"Destination": "Retail Store/Warehouses", "Cost": "Transportation Cost"},
                      ).update_layout(showlegend=True, xaxis={'categoryorder': 'total ascending'})
                      #                     .update_traces(marker_color=colors)
                      .update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="auto",
                                                 x=0))  # ,legend_title_text='DCs/Warehouses')
                      .update_traces(texttemplate='%{text:.2s}', textposition='inside')
                      # .update_layout(xaxis_tickangle=-45)
                      # .update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                      # .update_layout(barmode='group')

                      )
        ]

#------------------------------------------------
# No token = 'open-street-map', 'white-bg', 'carto-positron', 'carto-darkmatter', 'stamen- terrain', 'stamen-toner', 'stamen-watercolor'
# Token =     'basic', 'streets', 'outdoors', 'light', 'dark', 'satellite', 'satellite-streets'
#------------------------------------------------

# supply = {'S1': 1200, 'S2': 800, 'S3': 800, 'S4': 450, 'S5': 400, 'S6': 400, 'S7': 350, 'S8': 200, 'S9': 800}
# demand = {'D1': 800, 'D2': 850, 'D3': 900, 'D4': 1200, 'D5': 900, 'D6': 750}

