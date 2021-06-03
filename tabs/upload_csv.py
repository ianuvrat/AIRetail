from __future__ import print_function
from ortools.graph import pywrapgraph

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

import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.figure_factory as ff

import pandas as pd
import numpy as np



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))
# Mapbox token
token = 'pk.eyJ1IjoiaWFudXZyYXQiLCJhIjoiY2tvNWQ1a2FkMHB6MTJ2cXdmeGt3MTdldyJ9.5vxPVvQdr6NL8hxVCZ1ecg'

alert99 = dbc.Alert("Data uploaded! Select a date and submit ", color="info",  dismissable=True),
alert98 = dbc.Alert("Upload Purchase Orders data!", color="warning",  dismissable=True),


#---------------------------------------------------------------
# get relative locations from data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()
dff = pd.read_csv(DATA_PATH.joinpath("sc_location_md.csv"))
dfff =pd.read_csv(DATA_PATH.joinpath("sc_transportation_lanes.csv"))
#_--------------------------------------------------------------
# Cleaning
df_loc = dff.drop(['location_description','is_internal', 'size'],axis=1)
#dropping unwanted columns
df_lane = dfff.drop(['priority','ordering_calendar_id','missing_supplier','missing_transportation'],axis=1)
#creating id col.
df_lane['id'] = df_lane['material_id'] + '_' + df_lane['location_from_id'] +'_' + df_lane['location_to_id']
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

layout = dbc.Container([

html.Br(),

    dbc.Row([

        # Alert,
        dbc.Col([
            html.Div(id="alert_id2", children=[]),
        ], className="text-left", width={'size': 3, 'offset': 0, 'order': 0}),


        dbc.Col([
            dcc.Dropdown(
                id='report-dpdn',
                multi=False,
                searchable=False,
                clearable=False,
                disabled=False,
                options=[
                    {'label': 'CSV', 'value': 'csv'},
                    {'label': 'Excel', 'value': 'excel'},
                    {'label': 'Text', 'value': 'txt'}
                ],
                value='csv'),
        ], className="text-left", width={'size': 1, 'offset': 1, 'order': 1}),

        # upload button
        dbc.Col([
            html.H5("Upload file"),
            dcc.Upload(
                id="upload-data",
                children=html.Div(["Drag/Drop or ", html.A("Select File")]),
                style={
                    "width": "100%",
                    "height": "45px",
                    "lineHeight": "40px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
                # Allow multiple files to be uploaded
                multiple=False,
            )
        ], className="text-left", width={'size': 2, 'offset': 0, 'order': 2}),

        # 1 Datepicker
        dbc.Col([
            html.H5("Datepicker"),
            dcc.DatePickerSingle(
                id='my-date-picker-single',
                # min_date_allowed=date(1995, 8, 5),
                # max_date_allowed=date(2050, 9, 19),
                # initial_visible_month=date(2020, 1, 14),
                # date=date(2020, 1, 14)
                date='2020-01-14',
                display_format='D-MM-YYYY',
                with_portal=True,
            ),
        ], className="text-left", width={'size': 1.5, 'offset': 0, 'order': 3}),


    ]),



    dbc.Row([

        # Submit Button
        dbc.Col([
            html.Button(id='submit_id', n_clicks=0, children='Submit'),
        ], className="text-left", width={'size': 1, 'offset': 3, 'order': 1}),

        # Alert,
        dbc.Col([
            html.Div(id="alert_id", children=[]),
        ], className="text-left", width={'size': 3, 'offset': 0, 'order': 0}),
    ]),


    dbc.Row([

        # 2 Selected Date Display
        dbc.Col([
            html.Div(id='output-container-date-picker-single', className="text-danger font-weight-bold text-left border-primary")
                ] , className="text-left", width={'size': 3, 'offset': 9, 'order': 1}),
            ]),

    # 5  Uploaded Dash datatable
   dcc.Loading(children=[html.Div(id="output-data-upload", style={'width': '190vh', 'height': '100vh'})], color="#119DFF",
                    type="default", fullscreen=False),

    #html.Div(id='output-data-upload', style={'width': '190vh', 'height': '100vh'}),

html.Br(),
html.Br(),
html.Br(),
html.Br(),
html.Br(),


    dbc.Row([

        dbc.Col([
            # Map Mode radio items
            html.Div([
                html.Label(['Map View:'], style={'font-weight': 'bold'}, className="font-weight-bold text-primary"),
                dcc.RadioItems(
                    id='map_mode',
                    options=[

                        {'label': 'Toner', 'value': 'toner'},
                        {'label': 'OpenStreet ', 'value': 'ostreet'},
                        {'label': 'Satellite ', 'value': 'sat'},
                        {'label': 'Region ', 'value': 'region'},
                        {'label': 'Dark ', 'value': 'dak'},
                        {'label': 'States ', 'value': 'street'},
                    ],
                    value='toner',
                    style={"width": "50%"},
                ),
            ]),

        ] , className="text-left", width={'size': 7, 'offset': 0, 'order': 1}),

        # i button
        dbc.Col([
            dbc.Button("i", id="popover-bottom-target0", color="warning", outline=True),
                ], className="text-left", width={'size': 1, 'offset': 4, 'order': 0}),
        dbc.Popover([
            dbc.PopoverBody(
                "Select a map layout preview "),
                    ],
            id="popover0",
            target="popover-bottom-target0",  # needs to be the same as dbc.Button id
            trigger="hover",
            placement="bottom",
            is_open=False,
                ),

    ]),


    html.H5("Summay for selected day"),
html.Br(),
html.Br(),

    dbc.Row([

        # 3 Summary Outputs
        dbc.Col([
            html.Table([
                html.Tr([html.Td(['Supply Nodes'], className="p-1 border border-primary border-right-0 bg-warning font-weight-bold text-left border-primary"),
                         html.Td(id='s_node', className=" text-dark font-weight-bold text-left" )]),
html.Br(),
                html.Tr([html.Td(['Demand Nodes'], className="p-1 border border-primary border-right-0 bg-warning font-weight-bold text-left border-primary"),
                         html.Td(id='d_node', className=" text-dark font-weight-bold text-left")]),
html.Br(),
                html.Tr([html.Td(['Unique PO_items'], className="p-1 border border-primary border-right-0 bg-warning font-weight-bold text-left border-primary"),
                         html.Td(id='po_item_cat', className=" text-dark font-weight-bold text-left")]),
html.Br(),
                html.Tr([html.Td(['Material Flow'], className="p-1 border border-primary border-right-0 bg-warning font-weight-bold text-left border-primary"),
                         html.Td(id='mat_flow', className=" text-dark font-weight-bold text-left")]),
html.Br(),

                html.Tr([html.Td(['Tot. POs'], className="p-1 border border-primary border-right-0 bg-warning font-weight-bold text-left border-primary"),
                         html.Td(id='tot_po', className=" text-dark font-weight-bold text-left")]),
html.Br(),
html.H5('Google OR tools results'),
                html.Tr([html.Td(['Max. PO flow'], className="p-1 border border-primary border-right-0 bg-success font-weight-bold text-left border-primary"),
                         html.Td(id='max_flow', className=" text-dark font-weight-bold text-left")]),
                html.Tr([html.Td(['Min. Cost: $'], className="p-1 border border-primary border-right-0 bg-success font-weight-bold text-left border-primary"),
                         html.Td(id='min_cost', className=" text-dark font-weight-bold text-left")]),
                    ]),
        ], className="text-left", width={'size': 0, 'offset': 0, 'order': 0}),

        # Map Scatter Box
        dbc.Col([
            dcc.Loading(children=[dcc.Graph(id="mymap3", config={'displayModeBar': True})], color="#119DFF",
                        type="cube", fullscreen=False),
        ], className="text-left", width={'size': 8, 'offset': 1, 'order': 1}),

    ]),


    dbc.Row([

        # 6  Final Dash datatable
        dbc.Col([
            html.H5("Optimized PO flow", className="font-weight-bold text-danger"),
            html.Div(id='output-data-upload2'),
        ], className="text-left", width={'size': 12, 'offset': 0, 'order': 0}),

    ]),



    dbc.Row([

        # Sankey
        dbc.Col([
            dcc.Loading(children=[dcc.Graph(id="sanky-fig", config={'displayModeBar': True})], color="#119DFF",
                        type="default", fullscreen=False),
                ], className="text-left", width={'size': 6, 'offset': 0, 'order': 0}),

        # Bar-Container-2 (Cost vs Demand Fulfilment)
        dbc.Col([
                html.Div(id='bar-container2'),
                ], className="text-left", width={'size': 6, 'offset': 0, 'order': 1}),
            ]),


    dbc.Row([

        # Shipment Summary
        dbc.Col([
            html.Div(id='bar_fig'),
        ], className="text-left", width={'size': 12, 'offset': 0, 'order': 0})
    ]),


    dbc.Row([

        # 6  Comparision table (Cost vs Lead time)
        dbc.Col([
            html.H5("Cost vs Lead time vs PO Flows vs Carriers", className="font-weight-bold text-danger"),
            html.Div(id='table_fig'),
                ], className="text-left", width={'size': 6, 'offset': 0, 'order': 0}),

    ]),

    html.Br(),

    # # 5  Uploaded Dash datatable
    # html.Div(id='output-data-upload', style={'width': '190vh', 'height': '100vh'}),

], fluid=True)
#____________________________________________________________________________________________________________________________________________________________________________________________________________________

#-----------------------Parsing uploaded data and returning df------------------------------------
def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),parse_dates=['date_delivery_requested'])

            df_po = df

            # dropping unwanted columns
            df_po = df_po.drop(['pr_item', 'po_type', 'po_category', 'po_line', 'po_del_flag', 'po_item_del_flag', 'po_type'], axis=1)

            # Formaating Date Columns
            df_po['date_delivery_requested'] = pd.to_datetime(df_po['date_delivery_requested'])
            df_po['date_delivery_requested'] = pd.to_datetime(df_po['date_delivery_requested']).dt.date

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

            df_po = df

            # dropping unwanted columns
            df_po = df_po.drop(['pr_item', 'po_type', 'po_category', 'po_line', 'po_del_flag', 'po_item_del_flag', 'po_type'], axis=1)

            # Formaating Date Columns
            df_po['date_delivery_requested'] = pd.to_datetime(df_po['date_delivery_requested'])
            df_po['date_delivery_requested'] = pd.to_datetime(df_po['date_delivery_requested']).dt.date


    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df_po

#-----------------------Uploaded Datatable callback------------------------------------
@app.callback(
              [Output('output-data-upload', 'children'),
               Output("alert_id", "children"),
               Output("alert_id2", "children")],


              Input('upload-data', 'contents'),
              State('upload-data', 'filename')
            )
def parse_output(contents, filename):
    if contents is None and filename is None:
        return "", "", alert98
#        raise PreventUpdate


    elif contents is not None and filename is not None:
        df_po = parse_data(contents, filename)  # calling fn.

        table = html.Div(
            [
                html.H5("Uploaded CSV data"),
                dash_table.DataTable(
                    id='datatable_id',
                    data=df_po.to_dict("records"),
                    columns=[
                        {"name": i, "id": i, "deletable": False, "selectable": True, "hideable": True} for i in
                        df_po.columns
                    ],
                    editable=False,  # allow editing of data inside all cells
                    filter_action="native",  # allow filtering of data by user ('native') or not ('none')
                    sort_action="native",  # enables data to be sorted per-column by user or not ('none')
                    sort_mode="multi",  # sort across 'multi' or 'single' columns
                    column_selectable="multi",  # allow users to select 'multi' or 'single' columns
                    row_selectable="multi",  # allow users to select 'multi' or 'single' rows
                    row_deletable=False,  # choose if user can delete a row (True) or not (False)
                    selected_columns=[],  # ids of columns that user selects
                    selected_rows=[],  # indices of rows that user selects
                    page_action='native',

                    style_cell={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        #                        'minWidth': 180, 'maxWidth': 180, 'width': 180,
                    },

                    style_header={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        #                        'minWidth': 180, 'maxWidth': 180, 'width': 180,
                    },

                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'minWidth': 250, 'maxWidth': 250, 'width': 250,
                    },

                    style_table={'overflowX': 'scroll'},

                    export_columns='all',  # 'all' or 'visible
                    export_format='xlsx',  # 'csv or 'none' or 'xlsx'
                    fixed_rows={'headers': True, 'data': 0},
                    virtualization=False,
                ),
            ]
        )

        return table, alert99, ""
#---------------------------------------------------------------
#---------------------Datepicker callback------------------------
@app.callback(
    Output('output-container-date-picker-single', 'children'),
    Output('s_node', 'children'),
    Output('d_node', 'children'),
    Output('po_item_cat', 'children'),
    Output('mat_flow', 'children'),
    Output('tot_po', 'children'),
    Output('max_flow', 'children'),
    Output('min_cost', 'children'),
    Output('output-data-upload2', 'children'),
    Output('mymap3', 'figure'),
    Output('sanky-fig', 'figure'),

    [Input('map_mode', 'value'),
    Input('submit_id', 'n_clicks'),
    Input('my-date-picker-single', 'date'),
    Input('upload-data', 'contents'),
    # State('cap_id', 'value'),
    State('upload-data', 'filename')]
)


def process_update(view, num_clicks, date_value, contents,filename):
    string_prefix = 'You have selected: '

    # if contents is None or filename is None or date_value is None:
    #     raise PreventUpdate

    if num_clicks == 0:
        raise PreventUpdate

    #    if date_value is not None:
    else:
        date_object = date.fromisoformat(date_value)
        date_string = date_object.strftime('%B %d, %Y')
        selected_day = date_object


        df_po = parse_data(contents, filename)  # Function call for uploaded data
        print("Total Records in uploaded csv")
        print("Total unique PO numbers", df_po['po_number'].nunique())
        print("Total unique PO items", df_po['po_item'].nunique())
        print("Total unique raw material", df_po['material_id'].nunique())
        print("total unique supply nodes", df_po["location_from_id"].nunique())
        print("total unique demand nodes", df_po["location_to_id"].nunique())
        print("Total records", len(df_po),'\n')

        # Filtering df for a selected date
        df_selected_day = df_po[df_po["date_delivery_requested"] == selected_day]

        res1 = df_selected_day["location_from_id"].nunique(),  # supply nodes count
        res2 = df_selected_day["location_to_id"].nunique(),    # demand nodes count
        res3 = df_selected_day["po_item"].nunique(),           # unique PO Item count
        res4 = df_selected_day["material_id"].nunique(),       # unique materials count

        print("Selected Day:", selected_day)
        print("total supply nodes", df_selected_day["location_from_id"].nunique())
        print("total demand nodes", df_selected_day["location_to_id"].nunique())
        print("total unique PO category", df_selected_day["po_item"].nunique())
        print("total material Flow", df_selected_day["material_id"].nunique())

        # print(df_selected_day.head())


        # Facilities involved in the selected day
        s_nodes = []
        e_nodes = []

        for s in df_selected_day['location_from_id'].unique():
            s_nodes.append(s)
        for e in df_selected_day['location_to_id'].unique():
            e_nodes.append(e)



        # POs supplies at 'supply' nodes
        print("total supply nodes", df_selected_day["location_from_id"].nunique())
        print("Soure Facilities\n", s_nodes)

        # POs Demand at 'demand' nodes
        print("total demand nodes", df_selected_day["location_to_id"].nunique())
        print("\nEnd Facilities\n", e_nodes)

        # Main df
        main = pd.DataFrame(df_selected_day.groupby(['material_id', 'location_from_id', 'location_to_id']).agg({'po_number': 'count'}).reset_index())

        # rename cols.
        main.rename(columns={'po_number': '#POs'}, inplace=True)

        main['id'] = main['material_id'] + '_' + main['location_from_id'] + '_' + main['location_to_id']

        # Adding lead time and cost (according to transport lanes)
        capacity_lst = []
        lead_time_lst = []
        unit_cost_lst = []

        for id1 in main['id']:
            for id2 in df_lane['id']:
                if id1 == id2:
                    w = df_lane.loc[df_lane['id'] == id1, 'capacity'].iloc[0]  # capacity
                    x = df_lane.loc[df_lane['id'] == id1, 'fixed_transportation_cost'].iloc[0]     # fixed cost
                    y = df_lane.loc[df_lane['id'] == id1, 'transportation_cost_per_unit'].iloc[0]  # variable cost
                    z = df_lane.loc[df_lane['id'] == id1, 'lead_time'].iloc[0]                     # lead time cost

                    capacity_lst.append(w)
                    unit_cost_lst.append(x + y)
                    lead_time_lst.append(z)

        capacity_lst = [int(item) for item in capacity_lst]
        main['capacity'] = capacity_lst
        main['unit_costs'] = unit_cost_lst
        main['lead_time'] = lead_time_lst
        main = main.drop(['id'], axis=1)
        print("main df")
        print(main.info())

        print(main.head(100))


        #Supplies
        df_supply = pd.DataFrame(main.groupby(['location_from_id'])['#POs'].sum().reset_index(name='PO_supplies'))
        #Demands
        df_demand = pd.DataFrame(main.groupby(['location_to_id'])['#POs'].sum().reset_index(name='PO_demand'))

        ## Encoding facilities

        temp = len(s_nodes)
        key1 = s_nodes
        value1 = range(len(s_nodes))
        key2 = e_nodes
        value2 = range(temp, temp + len(e_nodes))

        d1 = dict(zip(key1, value1))
        d2 = dict(zip(key2, value2))
        print(d1)
        print(d2)

        main = main.replace({"location_from_id": d1, "location_to_id": d2})

        # start-end nodes / unit cost / supply-demand
        start_nodes = []
        end_nodes = []
        unit_costs = []
        supplies = []
        mat_lst = []
        capacities = capacity_lst

        # start nodes
        for s_node in main['location_from_id']:
            start_nodes.append(s_node)
        # end nodes
        for e_node in main['location_to_id']:
            end_nodes.append(e_node)
        # supplies at supply nodes
        for s in df_supply['PO_supplies']:
            supplies.append(s)
        # demands at demand nodes (taken as -ve supplies)
        for s in df_demand['PO_demand']:
            supplies.append(-s)
        # unit costs
        for c in main["unit_costs"]:
            unit_costs.append(c)


        # material list
        for mat in main['material_id']:
            mat_lst.append(mat)

        print("start_nodes", start_nodes)
        print(len(start_nodes))

        print("end_nodes", end_nodes)
        print(len(end_nodes))

        print("capacities", capacities)
        print(len(capacities))

        print("unit_costs", unit_costs)
        print(len(capacities))

        print("supplies", supplies)
        print(len(supplies))

        print('sum of supplies ', sum(supplies))

        print("Materials", mat_lst)
        print(len(mat_lst))

        #-----------------OR tools-------------------
        # Instantiate a SimpleMinCostFlow solver.
        model = pywrapgraph.SimpleMinCostFlow()

        # Add each arc.
        for i in range(0, len(start_nodes)):
            model.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                capacities[i], unit_costs[i])
        # Add node supplies.
        for i in range(0, len(supplies)):
            model.SetNodeSupply(i, supplies[i])

        print(model.SolveMaxFlowWithMinCost())

        # Find the minimum cost flow between node 0 and node 4.
        if model.SolveMaxFlowWithMinCost() == model.OPTIMAL:
            print('Minimum cost:', model.OptimalCost())
            print('')
            print('  Arc    Flow / Capacity  Cost')
            for i in range(model.NumArcs()):
                cost = model.Flow(i) * model.UnitCost(i)
                print('%1s -> %1s   %3s  / %3s       %3s' % (
                    model.Tail(i),
                    model.Head(i),
                    model.Flow(i),
                    model.Capacity(i),
                    cost))
        else:
            print('There was an issue with the min cost flow input.')

        print(model.MaximumFlow())

        # Find the minimum cost flow between two nodes
        if model.SolveMaxFlowWithMinCost() == model.OPTIMAL:
            optimal = "Optimal"
            source_list = []
            dest_list = []
            flow_list = []
            capacity_list = []
            cost_list = []

            for i in range(model.NumArcs()):
                cost = model.Flow(i) * model.UnitCost(i)
                #   if cost > 0:
                source_list.append(model.Tail(i)),
                dest_list.append(model.Head(i)),
                flow_list.append(model.Flow(i)),
                capacity_list.append(model.Capacity(i)),
                cost_list.append(cost)

        # Lists
        print("Sources", source_list)
        print("Destinations", dest_list)
        print("Flow", flow_list)
        print("Capacity", capacity_list)
        print("Cost", cost_list)

        # Dataframe result
        new_df = pd.DataFrame({'material_id': mat_lst,'Source': source_list, 'Destination': dest_list, 'PO Flow': flow_list, 'Flow Capacity': capacity_list, 'Cost': cost_list, 'Lead time': lead_time_lst})
        #del source_list, dest_list, flow_list, capacity_list, cost_list

        # Load (assuming 1 truck/flight can carry 2 POs)
        new_df['trucks/flights'] = np.ceil(new_df["PO Flow"] / 2)

        # capacity utilization
        new_df["Capacity Utilization %"] = (new_df["PO Flow"] / new_df["Flow Capacity"]) * 100
        new_df["Unused Capacity"] = (new_df["Flow Capacity"] - new_df["PO Flow"])

        #Decoding Facilities
        d1_rev = {value1: key1 for (key1, value1) in d1.items()}
        d2_rev = {value2: key2 for (key2, value2) in d2.items()}

        print(d1_rev)
        print(d2_rev)

        new_df = new_df.replace({"Source": d1_rev, "Destination": d2_rev})
        print(new_df.head(100))

        table2 = html.Div(
            [
                # html.H5("Optimized PO flow", className="font-weight-bold text-danger"),
                dash_table.DataTable(
                    id='datatable_id_2',
                    data=new_df.to_dict("records"),
                    columns=[
                        {"name": i, "id": i, "deletable": False, "selectable": True, "hideable": True} for i in
                        new_df.columns
                    ],
                    editable=False,  # allow editing of data inside all cells
                    filter_action="native",  # allow filtering of data by user ('native') or not ('none')
                    sort_action="native",  # enables data to be sorted per-column by user or not ('none')
                    sort_mode="multi",  # sort across 'multi' or 'single' columns
                    column_selectable="multi",  # allow users to select 'multi' or 'single' columns
                    row_selectable="multi",  # allow users to select 'multi' or 'single' rows
                    row_deletable=False,  # choose if user can delete a row (True) or not (False)
                    selected_columns=[],  # ids of columns that user selects
                    selected_rows=[],  # indices of rows that user selects
                    page_action='native',


                    style_cell={
                        'whiteSpace': 'normal',
                        'height': 'auto',
#                        'minWidth': 180, 'maxWidth': 180, 'width': 180,
                    },

                    style_header={
                        'whiteSpace': 'normal',
                        'height': 'auto',
#                        'minWidth': 180, 'maxWidth': 180, 'width': 180,
                    },

                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'minWidth': 250, 'maxWidth': 250, 'width': 250,
                    },

                    style_table={'overflowX': 'scroll'},

                    fixed_rows={'headers': True, 'data': 0},
                    virtualization=False,
                    export_columns='all',  # 'all' or 'visible
                    export_format='xlsx',  # 'csv or 'none' or 'xlsx'

                    style_data_conditional=([
                        {
                            'if': {
                                'filter_query': '{PO Flow} = {Flow Capacity}',
                                'column_id': 'Source'
                            },
                            'backgroundColor': 'orange',
                            'color': 'white',
                            'fontWeight': 'bold',
                        },

                        {
                            'if': {
                                'filter_query': '{PO Flow} = {Flow Capacity}',
                                'column_id': 'Destination'
                            },
                            'backgroundColor': 'orange',
                            'color': 'white',
                            'fontWeight': 'bold',
                        },

                        {
                            'if': {
                                'filter_query': '{PO Flow} > 0',
                                'column_id': 'PO Flow'
                            },
                            #'backgroundColor': 'black',
                            'color': 'blue',
                            'fontWeight': 'bold',
                        },

                        {
                            'if': {
                                'filter_query': '{trucks/flights} > 0',
                                'column_id': 'trucks/flights'
                            },
                            # 'backgroundColor': 'black',
                            'color': 'red',
                            'fontWeight': 'bold',
                        },

                        {
                            'if': {
                                'filter_query': '{Capacity Utilization %} = 100',
                                'column_id': 'Capacity Utilization %'
                            },
                            'backgroundColor': 'orange',
                            'color': 'white',
                            'fontWeight': 'bold',
                        },

                    ]),

                ),
            ]
        )

# # #            Generating map for selected day
# # #------------------------------------------------------------------------------------------------------------
        location_id_lst = s_nodes
        for var in e_nodes:
            location_id_lst.append(var)
        lat_lst = []
        long_lst = []
        city_lst = []
        country_lst = []
        loc_type_lst = []
        for loc in location_id_lst:
            for Loc in df_loc['location_id']:
                if loc == Loc:
                    lat = df_loc.loc[df_loc['location_id'] == loc, 'latitude'].iloc[0]
                    lat_lst.append(lat)
                    lon = df_loc.loc[df_loc['location_id'] == loc, 'longitude'].iloc[0]
                    long_lst.append(lon)
                    city = df_loc.loc[df_loc['location_id'] == loc, 'city'].iloc[0]
                    city_lst.append(city)
                    country = df_loc.loc[df_loc['location_id'] == loc, 'country'].iloc[0]
                    country_lst.append(country)
                    loc_type = df_loc.loc[df_loc['location_id'] == loc, 'location_type'].iloc[0]
                    loc_type_lst.append(loc_type)


        dict_loc = {'location_id': location_id_lst, 'location_type': loc_type_lst, 'latitude': lat_lst, 'longitude': long_lst, 'city': city_lst, 'country': country_lst}
        df_loc_day = pd.DataFrame(dict_loc)

        print(df_loc_day.info())

#------------------------------------------------------------------------------------------------------------------------------
        fig = px.scatter_mapbox(df_loc_day, lat="latitude", lon="longitude",
                                hover_name="location_id",
                                hover_data=["city", "country"],
                                color="location_type",
                                zoom=1,
                                # size= "city",
                                opacity=1,
                                )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.update_traces(marker_symbol="circle"),
        fig.update_traces(marker_size=15),
        fig.update_traces(showlegend=True),
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1, xanchor="auto", x=0),
                          legend_title_text='')
        if view == 'ostreet':
            fig.update_layout(mapbox_style="carto-positron", mapbox_accesstoken=token)
        if view == 'toner':
            fig.update_layout(mapbox_style="stamen-toner", mapbox_accesstoken=token)
        elif view == 'sat':
            fig.update_layout(mapbox_style="satellite", mapbox_accesstoken=token)
        elif view == 'region':
            fig.update_layout(mapbox_style="stamen-terrain", mapbox_accesstoken=token)
        elif view == 'dak':
            fig.update_layout(mapbox_style="carto-darkmatter", mapbox_accesstoken=token)
        elif view == 'street':
            fig.update_layout(mapbox_style="streets", mapbox_accesstoken=token)

        # fig.update_layout(mapbox_style="carto-darkmatter", mapbox_accesstoken=token)

        #------------------------------------------------------------------------------------------------------------
        res5 = main['#POs'].sum(),
        res6 = model.MaximumFlow(),
        res7 = model.OptimalCost(),

        print('Total POs in the selected day:', main['#POs'].sum())
        print('Maximum PO Flow:', model.MaximumFlow())
        print('Minimum cost: $', model.OptimalCost())

        #---------------------------------------------------------------------------------------------------------
        # Sankey Diagram
        fig2 = go.Figure(data=[go.Sankey(

            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=location_id_lst,
                color = "blue"),

            link = dict(
                source=start_nodes,
                target = end_nodes,
                value = flow_list
            ))])

        fig2.update_layout(title_text="Purchase Order Flows ", font_size=15)
        fig2.update_traces(node_color= 'orange', selector = dict(type='sankey'))
        # ---------------------------------------------------------------------------------------------------------


        return string_prefix + date_string, res1,res2,res3,res4,res5,res6,res7, table2, fig, fig2
#---------------------------------------------------------------

# Create bar chart
@app.callback(
    Output(component_id='bar-container2', component_property='children'),
    Output(component_id='bar_fig', component_property='children'),
    Output(component_id='table_fig', component_property='children'),



    [Input(component_id='datatable_id_2', component_property="derived_virtual_data"),
     Input(component_id='datatable_id_2', component_property='derived_virtual_selected_rows'),
     Input(component_id='datatable_id_2', component_property='derived_virtual_selected_row_ids'),
     Input(component_id='datatable_id_2', component_property='selected_rows'),
     Input(component_id='datatable_id_2', component_property='derived_virtual_indices'),
     Input(component_id='datatable_id_2', component_property='derived_virtual_row_ids'),
     Input(component_id='datatable_id_2', component_property='active_cell'),
     Input(component_id='datatable_id_2', component_property='selected_cells')]
)
def update_viz(all_rows_data, slctd_row_indices, slct_rows_names, slctd_rows,
               order_of_rows_indices, order_of_rows_names, actv_cell, slctd_cell):

    df_bar = pd.DataFrame(all_rows_data)

    df_table =  df_bar.filter(['material_id','Cost','Lead time', 'PO Flow', 'trucks/flights'])
    print((df_table.head(100)))

    colors = ['#7FDBFF' if i in slctd_row_indices else '#0074D9'
              for i in range(len(df_bar))]
    colorscale = [[0, '#4d004c'], [.5, '#f2e5ff'], [1, '#ffffff']]

    if "Destination" in df_bar and "Cost" in df_bar and 'Lead time' in df_bar:

        fig_bar = dcc.Graph(id='bar-chart',
                      figure=px.bar(
                          data_frame=df_bar,
                          x="Destination",
                          y='PO Flow',
                          color= "Cost",
                          text="PO Flow",
                          title="Cost vs Demand Fulfilment",
                          hover_data=['Source'],
                          labels={"Cost": "Transportation Cost"},
                        )
                        .update_layout(showlegend=True, xaxis={'categoryorder': 'total ascending'})
                        .update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="auto", x=0)) #,legend_title_text='DCs/Warehouses')
                        .update_traces(texttemplate='%{text:.2s}', textposition='inside')
                        #.update_layout(xaxis_tickangle=-45)
                        #.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                        #.update_layout(barmode='group')
                            )

        fig_facet_bar = dcc.Graph(id='bar-chart2',
                      figure=px.bar(
                          data_frame=df_bar,
                          y="Capacity Utilization %",
                          x='trucks/flights',
                          #size='PO Flow',
                          text='PO Flow',
                          color= "Source",
                          title="Shipment: Capacity Utilizations/ Lead time/ POs",
                          hover_data=['Source', 'Destination', 'Lead time'],
                          facet_col="Destination",
                          #labels={"Cost": "Transportation Cost"},
                          )
                         .update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="auto", x=0)) #,legend_title_text='DCs/Warehouses')
                         .update_traces(showlegend=False),

                          )


        fig_table = dcc.Graph(id='table-chart',
                            figure=ff.create_table(df_table, colorscale=colorscale))

        return fig_bar, fig_facet_bar, fig_table

