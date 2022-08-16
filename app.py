"""Dash app for crossfit movement classifier with MoViNets. """

import base64
import json
from pathlib import Path

import boto3
import dash
import dash_bootstrap_components as dbc
from botocore.exceptions import NoCredentialsError
from dash import Input, Output, State, dcc, html

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.JOURNAL],
    update_title="Crossfit Movement",
)
app.title = "Crossfit Movement Classifier"
server = app.server


# Load the label movements for the dropdown of samples
try:
    labels = Path("assets/labels.txt").read_text().splitlines()
except Exception as exc:
    print(f"labels.txt couldn't get loaded: {exc}")


# Get access to the lambda function
client = boto3.client("lambda", region_name="us-east-1")


upload_field = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                dcc.Upload(
                    id="upload-clip",
                    children=[html.Div(["Drag and Drop or ", html.A("Select Files")])],
                    style={
                        "width": "100%",
                        "border": "25%",
                        "lineHeight": "50px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "margin": "10px",
                    },
                ),
            ),
        ),
        # dbc.Row(html.Div(id="output-video")),
    ]
)


# TODO: Este video debería salir como respuesta a id="output-video"?
video = dbc.Container(
    children=[
        dbc.Row(dbc.Col(dcc.Loading(html.Div(id="video-loaded")))),
    ],
    style={
        "text-align": "center",
        "margin-left": "12px",
        "margin-right": "24px",
    },
    id="video-window",
)


def prepare_video(contents):
    """Gets the contents passed when uploading a video
    and prepares it as bytes to be sent to aws lambda.

    contents is the input passed to play_video callback.
    """
    # To see the contents passed when uploading a video, print here:
    # contents[:200] are a str of the form:
    # data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABkpxtZGF0AAACrwYF//+r3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE1NSByMjkxNyAwYTg0ZDk4IC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcH
    # Only the video content is passed, splitting by the ,
    video = base64.b64encode(contents.split(",")[1].encode("utf-8")).decode(
        "utf8"
    )
    # Update, send the video encoded as is
    # video = contents.split(",")[1]
    # The content must be decoded to be sent as bytes
    print(f"video type and length: {type(video)}, {len(video)}")
    print(f"video slice: {video[:50]}")
    return bytes(json.dumps({"video": video}), "utf-8")


@app.callback(Output("video-loaded", "children"), [Input("upload-clip", "contents")])
def play_video(contents):
    """contents represents directly the video as uploaded by the user."""
    return html.Video(controls=True, id="movie_player", autoPlay=True, src=contents)


video_container = dbc.Card(
    children=[
        html.Div(upload_field),
        html.Div(id="video-loaded"),
    ],
    style={"textAlign": "center", "horizontal-align": "middle", "offset": "10"},
)

# TODO: Meter loading en el botón para la predicción:
# https://dash.plotly.com/dash-core-components/loading
submit_button = dbc.Container(
    # dcc.Loading(
    dbc.Row(
        dbc.Col(
            dbc.Button(
                "Submit Video",
                id="submit-video-button",
                style={
                    "margin-top": "12px",
                    "margin-bottom": "12px",
                    "horizontal-align": "center",
                    "vertical-align": "center",
                },
                color="success",
            )
        )
    )
)


def get_prediction(video):
    """Calls the aws lambda function with the video passed and returns the predictions."""
    try:
        prediction_response = client.invoke(
            FunctionName="movinet_for_crossfit",
            Payload=video,
        )
        prediction = prediction_response["body"]["prediction"]

    except NoCredentialsError as exc:
        print(f"Credential errors when calling the lambda function: {exc}")
        prediction = (("ERROR", -1.0),) * 5

    print(f"prediction: {prediction}")

    return prediction


def get_table(prediction):
    table_header = [html.Thead(html.Tr([html.Th("Movement"), html.Th("Probability")]))]

    def get_table_body(prediction):
        """Uses the prediction obtained from the model. List of dict with movement and probability."""
        rows = []
        for pred in prediction:
            rows.append(html.Tr([html.Td(pred[0]), html.Td(pred[1])]))
        return [html.Tbody(rows)]

    table_body = get_table_body(prediction)
    return dbc.Table(
        table_header + table_body,
        style={"width": "60%", "textAlign": "center", "horizontal-align": "middle"},
    )


# @app.callback(Output("video-loaded", "children"), [Input("submit-video-button", "contents")])
@app.callback(
    Output("prediction-table", "children"),
    [Input("submit-video-button", "n_clicks")],
    State("upload-clip", "contents"),
)
def call_lambda(n_clicks, contents):
    """contents represents directly the video as uploaded by the user."""
    # To see the contents passed when uploading a video, print here:
    # contents[:200] are a str of the form:
    # data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABkpxtZGF0AAACrwYF//+r3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE1NSByMjkxNyAwYTg0ZDk4IC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcH
    video = prepare_video(contents)
    prediction = get_prediction(video)
    # TODO: Call the aws lambda function

    return get_table(prediction)


prediction = dbc.Card(
    children=dbc.Row(
        [
            submit_button,
            dcc.Loading(
                id="loading-1",
                type="default",
                children=html.Div(
                    id="prediction-table",
                    style={
                        "textAlign": "center",
                        "horizontal-align": "middle",
                        "padding-left": "35%",
                        "padding-right": "10%",
                    },
                ),
            ),
        ]
    ),
    style={
        "textAlign": "center",
        "horizontal-align": "middle",
        "vertical-align": "middle",
    },
)

# Tabs
tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Clip prediction", tab_id="tab-1"),
                dbc.Tab(label="Examples", tab_id="tab-2"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
        html.Div(id="content"),
    ]
)


tab_prediction = dbc.Row([video_container, prediction])


video_example = dbc.Container(
    children=[
        dbc.Row(
            dbc.Col(
                html.Video(
                    controls=True,
                    id="movie_player",
                    autoPlay=True,
                    src=app.get_asset_url("thruster_32.mp4"),
                )
            )
        ),
    ],
    style={
        "text-align": "center",
        "margin-left": "12px",
        "margin-right": "24px",
    },
    id="video-example",
)

tab_examples = dbc.Card(
    dbc.CardBody(
        [
            html.P("Select a movement to see a demo", className="card-text"),
            dcc.Dropdown(labels, labels[0], id="dropdown-movements"),
            html.Div(id="video-example"),
        ]
    )
)


@app.callback(
    Output("video-example", "children"), [Input("dropdown-movements", "value")]
)
def play_video_example(input_value):
    map_movements = {
        "bar-facing burpee": "bar-facing burpee_1.mp4",
        "chest-to-bar": "chest-to-bar_46.mp4",
        "deadlift": "deadlift_64.mp4",
        "double-unders": "double-unders_73.mp4",
        "ghd": "ghd_79.mp4",
        "ohs": "ohs_142.mp4",
        "power clean": "power clean_262.mp4",
        "shspu": "shspu_291.mp4",
        "thruster": "thruster_32.mp4",
    }
    return dbc.Container(
        children=[
            dbc.Row(
                dbc.Col(
                    html.Video(
                        controls=True,
                        id="movie_player",
                        autoPlay=True,
                        src=app.get_asset_url(map_movements[input_value]),
                    )
                )
            ),
        ],
        style={
            "text-align": "center",
            "margin-left": "12px",
            "margin-right": "24px",
        },
        id="video-window",
    )


tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(
                    tab_prediction, label="Clip prediction", tab_id="tab-prediction"
                ),
                dbc.Tab(tab_examples, label="Examples", tab_id="tab-examples"),
            ],
            id="tabs",
            active_tab="tab-prediction",
        ),
    ]
)


app.layout = dbc.Container(
    [
        html.H1(
            "Crossfit Movement classifier",
            style={
                "text-align": "center",
                "margin-top": "24px",
                "margin-bottom": "24px",
            },
        ),
        html.Hr(),
        html.H5(
            children="Upload a clip to detect the movement",
            style={
                "text-align": "center",
                "margin-top": "12px",
                "margin-bottom": "24px",
            },
        ),
        tabs
        # dbc.Row([video_container, prediction])
    ]
)


if __name__ == "__main__":
#    app.run_server(debug=True)  # When debugging
    app.run_server(host='0.0.0.0', port="80")
