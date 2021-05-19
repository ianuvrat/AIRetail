from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_extensions as de

from app import app,dbc

url = "https://assets2.lottiefiles.com/packages/lf20_SkhtL8.json"
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))

center={
  'display': 'block',
  'width': '70%',
  'margin-left':'auto',
  'margin-right':'35%',
  'height': '50%'
}

layout = [
        html.Br(),
        dcc.Markdown(""" Accenture Applied Intelligence - Retail Network Optimization""",
                     className="text-center font-weight-normal text-primary mb-5"),

        html.Br(),
        html.Br(),

        dbc.Row([
            #Lottie
            dbc.Col([
                    html.Div(de.Lottie(options=options, width="70%", height="50%", url=url)),
                    ],className="text-left", width={'size': 6, 'offset': 0, 'order': 0}),

            #Slideshow container
            dbc.Col([
                    html.Div(id="slideshow-container", children=[ html.Div(id="image"),
                                                      dcc.Interval(id='interval', interval=2000)
                                                       ],style={'padding-bottom':'10px'}),
                    ]),
              ]),


        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        # html.Br(),

        # html.Br(),
        # html.Br(),
        # html.Br(),
        # html.Br(),


        dcc.Markdown("""Created by Anuvrat""",
                     className="text-left font-weight-normal text-primary")
        ]



################################################Callback for Slideshow#####################################


@app.callback(Output('image', 'children'),
              [Input('interval', 'n_intervals')])
def display_image(n):
    if n == None or n % 2 == 0:
        img = html.Img(src=app.get_asset_url('a.JPG'),style=center)
    else:
        img = html.Img(src=app.get_asset_url('b.JPG'),style=center)
#        img = "None"
    return img

