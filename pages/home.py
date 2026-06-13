from dash import html, dcc


def layout():
    return html.Div(
        className="home-page",
        children=[
            html.Div(
                className="home-overlay",
                children=[
                    html.Div(
                        className="home-hero-card",
                        children=[
                            html.Div("Copper Image Web", className="home-badge"),

                            html.H1(
                                "Copper Particle Detection Platform",
                                className="home-title",
                            ),

                            html.P(
                                "Optical analysis and calibration data collection for Eddy Current Separation samples.",
                                className="home-subtitle",
                            ),

                            html.Div(
                                className="home-options-grid",
                                children=[
                                    dcc.Link(
                                        html.Div(
                                            className="home-option-card",
                                            children=[
                                                html.Div("01", className="home-option-number"),
                                                html.H2("Prediction"),
                                                html.P(
                                                    "Upload an image and estimate copper area percentage and copper mass percentage using the calibrated models."
                                                ),
                                                html.Div("Open prediction module →", className="home-option-action"),
                                            ],
                                        ),
                                        href="/prediction",
                                        className="home-option-link",
                                    ),

                                    dcc.Link(
                                        html.Div(
                                            className="home-option-card",
                                            children=[
                                                html.Div("02", className="home-option-number"),
                                                html.H2("Insert Data"),
                                                html.P(
                                                    "Upload an image and add hand-sorting data to support future model calibration."
                                                ),
                                                html.Div("Open data module →", className="home-option-action"),
                                            ],
                                        ),
                                        href="/insert-data",
                                        className="home-option-link",
                                    ),
                                ],
                            ),

                            html.Div(
                                "Summer School Study Project · Eddy Current Separation",
                                className="home-footer",
                            ),
                        ],
                    )
                ],
            )
        ],
    )