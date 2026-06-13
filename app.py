from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

from pages import home, prediction, insert_data


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

server = app.server


app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
    ]
)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/prediction":
        return prediction.layout()

    if pathname == "/insert-data":
        return insert_data.layout()

    return home.layout()


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)