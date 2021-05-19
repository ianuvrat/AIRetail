from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app, server, dbc
from tabs import intro, transport, transship


style = {'maxWidth': '960px', 'margin': 'auto'}

app.layout = dbc.Container([
    dbc.Row([
            dbc.Col(html.H1("Route To Retail",
                            className='text-center text-uppercase text-primary mb-2'), #Space has to be defined when mutliple parameters need to be assigned
                            width=12)]),
    dcc.Tabs(id='tabs', value='tab-intro', children=[
        dcc.Tab(label='Unconstrained Network', value='tab-transport'),
        dcc.Tab(label='Capacity Constrained Network', value='tab-transship'),
        # dcc.Tab(label='Inventory Projection', value='tab-projection'),

    ]),
    html.Div(id='tabs-content')
],fluid=True)


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-intro': return intro.layout
    elif tab == 'tab-transport': return transport.layout
    elif tab == 'tab-transship':return transship.layout
    # elif tab == 'tab-projection': return projection.layout

#--------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)