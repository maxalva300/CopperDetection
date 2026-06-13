from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import json
import plotly.graph_objects as go
import cv2
from dash import Input, Output, State, dcc, html, no_update, callback
import dash_bootstrap_components as dbc

from config import MODEL_FILES, OVERLAY_DIR, REPORT_DIR, UPLOAD_DIR, DYNAMIC_GLOBAL_MODEL_FILE
from copper_core import (
    analyze_image,
    bgr_to_data_uri,
    decode_dash_upload,
    load_model,
    save_uploaded_image,
)
from database import insert_calibration_record, export_records_to_excel, fetch_all_records, count_records
from model_updater import recalibrate_global_model, apply_dynamic_global_correction
def detect_group_from_filename(filename: str) -> str:
    name = filename.upper()

    run_match = re.search(r"RUN\s*(\d+)", name)
    run = f"RUN{run_match.group(1)}" if run_match else None

    method = None
    if "_POL_" in name:
        method = "POL"
    elif "_BAR_" in name:
        method = "BAR"

    if "_NCP_" in name and run and method:
        return f"{run}_{method}_NCP"
    if "_CP_" in name and method:
        return f"{method}_CP"
    return "GLOBAL"


def get_model_path_for_filename(filename: str):
    group = detect_group_from_filename(filename)
    return group, MODEL_FILES.get(group, MODEL_FILES["GLOBAL"])


def result_card(label: str, value: str, help_text: str | None = None):
    children = [
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
    ]
    if help_text:
        children.append(html.Div(help_text, className="metric-help"))
    return html.Div(children, className="metric-card")

def create_calibration_plot():
    try:
        df = fetch_all_records()

        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_dark",
                title="No calibration records available yet",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
            )
            return fig

        df = df.dropna(subset=["predicted_cu_mass_pct", "real_cu_pct"])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["predicted_cu_mass_pct"],
                y=df["real_cu_pct"],
                mode="markers",
                text=df["filename"],
                marker=dict(size=10),
                name="Calibration records",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Predicted Cu: %{x:.2f}%<br>"
                    "Real Cu: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

        min_val = min(df["predicted_cu_mass_pct"].min(), df["real_cu_pct"].min())
        max_val = max(df["predicted_cu_mass_pct"].max(), df["real_cu_pct"].max())

        margin = max((max_val - min_val) * 0.10, 5)
        x0 = max(0, min_val - margin)
        x1 = min(100, max_val + margin)

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[x0, x1],
                mode="lines",
                name="Ideal line y = x",
                line=dict(dash="dash"),
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            title="Predicted Cu % vs Real Cu %",
            xaxis_title="Predicted Cu mass %",
            yaxis_title="Real Cu mass % from hand sorting",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(l=50, r=30, t=60, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        fig.update_xaxes(range=[x0, x1], gridcolor="rgba(255,255,255,0.12)")
        fig.update_yaxes(range=[x0, x1], gridcolor="rgba(255,255,255,0.12)")

        return fig

    except Exception as exc:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            title=f"Could not load calibration plot: {exc}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
        )
        return fig

def layout():
    return html.Div(
        className="page-shell",
        children=[
            html.Div(
                className="header-container",
                children=[
                    html.Img(src="/assets/MetsoLogo.png", className="logo"),
                    html.Div(
                        children=[
                            html.H1("Insert Calibration Data", className="main-title"),
                            html.Div(
                                "Upload an image and add hand-sorting data to support future model calibration",
                                className="main-subtitle",
                            ),
                        ],
                        className="title-block",
                    ),
                    dcc.Link("Home", href="/", className="local-badge"),
                ],
            ),

            html.Div(
                className="content-container",
                children=[
                    html.Div(
                        className="left-panel glass-card",
                        children=[
                            html.H2("1. Upload image", className="section-title"),
                            html.P(
                                "Upload one sample image. The app will run the current prediction model first.",
                                className="section-description",
                            ),

                            dcc.Upload(
                                id="insert-upload-image",
                                className="upload-box",
                                children=html.Div(
                                    [
                                        html.Div("Drag and drop an image here", className="upload-main-text"),
                                        html.Div("or click to select a JPG / PNG file", className="upload-sub-text"),
                                    ]
                                ),
                                accept="image/*",
                                multiple=False,
                            ),

                            html.Div(id="insert-upload-status", className="status-box"),

                            html.H2("2. Hand-sorting data", className="section-title compact"),
                            html.P(
                                "Enter the real measured values from hand sorting. These values will later be stored in the calibration database.",
                                className="section-description",
                            ),

                            html.Label("Total sample mass (g)", className="input-label"),
                            dcc.Input(
                                id="input-total-mass",
                                type="number",
                                placeholder="Example: 38.60",
                                className="data-input",
                            ),

                            html.Label("Copper mass (g)", className="input-label"),
                            dcc.Input(
                                id="input-copper-mass",
                                type="number",
                                placeholder="Example: 20.30",
                                className="data-input",
                            ),

                            html.Label("Reject mass (g)", className="input-label"),
                            dcc.Input(
                                id="input-reject-mass",
                                type="number",
                                placeholder="Example: 18.30",
                                className="data-input",
                            ),

                            html.Label("Notes", className="input-label"),
                            dcc.Textarea(
                                id="input-notes",
                                placeholder="Optional comments about sample, lighting, process condition, etc.",
                                className="data-textarea",
                            ),

                            html.Button(
                                "Calculate calibration error",
                                id="calculate-error-button",
                                className="primary-button full-width-button",
                                disabled=True,
                            ),
                            html.Button(
                                "Add to calibration database",
                                id="save-calibration-button",
                                className="secondary-button full-width-button",
                                disabled=True,
                            ),
                            
                            html.Button(
                                "Download calibration database",
                                id="download-database-button",
                                className="secondary-button full-width-button",
                            ),
                            html.Button(
                                "Recalibrate Global Model",
                                id="recalibrate-global-button",
                                className="primary-button full-width-button",
                            ),

html.Div(id="recalibration-status", className="status-box"),

dcc.Download(id="download-calibration-database"),

                            html.Div(id="insert-data-status", className="status-box"),
                            html.Div(id="save-data-status", className="status-box"),
                        ],
                    ),

                    html.Div(
                        className="right-panel",
                        children=[
                            html.Div(
                                className="glass-card results-card",
                                children=[
                                    html.H2("Prediction and calibration results", className="section-title"),
                                    html.Div(id="insert-metrics-container", className="metrics-grid"),
                                ],
                            ),

                            html.Div(
                                className="glass-card results-card",
                                children=[
                                    html.H2("Calibration plot", className="section-title"),
                                    html.P(
                                        "Open this plot to compare model predictions against real hand-sorting data stored in the calibration database.",
                                        className="section-description",
                                    ),

                                    html.Button(
                                        "Show calibration plot",
                                        id="toggle-calibration-plot-button",
                                        className="secondary-button full-width-button",
                                        n_clicks=0,
                                    ),

                                    html.Div(
                                        id="calibration-plot-wrapper",
                                        style={"display": "none"},
                                        children=[
                                            html.Button(
                                                "Refresh calibration plot",
                                                id="refresh-calibration-plot-button",
                                                className="secondary-button full-width-button",
                                            ),
                                            dcc.Graph(
                                                id="calibration-plot",
                                                figure=create_calibration_plot(),
                                                config={"displayModeBar": False},
                                            ),
                                        ],
                                    ),
                                ],
                            ),

                            html.Div(
                                className="image-grid",
                                children=[
                                    html.Div(
                                        className="image-card glass-card",
                                        children=[
                                            html.H3("Original image", className="image-title"),
                                            html.Div(
                                                id="insert-original-image-container",
                                                className="image-placeholder",
                                                children="Upload an image to display it here.",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="image-card glass-card",
                                        children=[
                                            html.H3("Processed image", className="image-title"),
                                            html.Div(
                                                id="insert-overlay-image-container",
                                                className="image-placeholder",
                                                children="Copper overlay will appear here.",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            dcc.Store(id="insert-analysis-store"),
            dcc.Store(id="calibration-comparison-store"),
            html.Div("Copper Image Web · Insert Data Module", className="footer"),
        ],
    )


@callback(
    Output("insert-upload-status", "children"),
    Output("insert-original-image-container", "children"),
    Output("insert-overlay-image-container", "children"),
    Output("insert-metrics-container", "children"),
    Output("insert-analysis-store", "data"),
    Output("calculate-error-button", "disabled"),
    Input("insert-upload-image", "contents"),
    State("insert-upload-image", "filename"),
    prevent_initial_call=True,
)
def run_insert_prediction(contents, filename):
    if contents is None or filename is None:
        return no_update, no_update, no_update, no_update, no_update, True

    try:
        detected_group, model_path = get_model_path_for_filename(filename)
        model = load_model(model_path)
        model["group"] = detected_group

        uploaded_path = save_uploaded_image(contents, filename, UPLOAD_DIR)
        bgr = decode_dash_upload(contents)
        overlay, stats = analyze_image(bgr, model)
        if detected_group == "GLOBAL" and DYNAMIC_GLOBAL_MODEL_FILE.exists():
            with open(DYNAMIC_GLOBAL_MODEL_FILE, "r", encoding="utf-8") as f:
                dynamic_model = json.load(f)

            stats = apply_dynamic_global_correction(stats, dynamic_model)
            model["calibration_metrics"] = dynamic_model.get("calibration_metrics", {})
            model["dynamic_version"] = dynamic_model.get("version", "GLOBAL_v2")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(filename).stem
        overlay_path = OVERLAY_DIR / f"{stem}_insert_overlay_{detected_group}_{timestamp}.jpg"
        cv2.imwrite(str(overlay_path), overlay)

        original_img = html.Img(src=contents, className="preview-image")
        overlay_img = html.Img(src=bgr_to_data_uri(overlay), className="preview-image")

        metrics = [
            result_card("Detected type", detected_group, "Selected automatically from filename"),
            result_card("Predicted Cu area %", f"{stats['copper_%_area']:.3f} %"),
            result_card("Predicted Cu mass %", f"{stats['copper_%_mass_estimate']:.3f} %"),
            result_card("Copper particles", f"{stats['copper_particle_count']}"),
        ]
        if stats.get("global_dynamic_model_applied"):
            metrics.append(
                result_card(
                    "Global model version",
                    model.get("dynamic_version", "GLOBAL_v2"),
                    "Dynamic correction applied after GLOBAL v1 prediction",
                )
            )
            metrics.append(
                result_card(
                    "Original GLOBAL v1 prediction",
                    f"{stats['copper_%_mass_estimate_original_global_v1']:.3f} %",
                    "Prediction before dynamic correction",
                )
            )

        store_data = {
            "filename": filename,
            "detected_group": detected_group,
            "model_file": str(model_path),
            "uploaded_path": str(uploaded_path),
            "overlay_path": str(overlay_path),
            "predicted_cu_area_pct": float(stats["copper_%_area"]),
            "predicted_cu_mass_pct": float(stats["copper_%_mass_estimate"]),
            "copper_particle_count": int(stats["copper_particle_count"]),
            "material_pixels": int(stats["material_pixels"]),
            "copper_pixels": int(stats["copper_pixels"]),
        }

        status = html.Div(
            [
                html.Div(f"Loaded: {filename}", className="status-success"),
                html.Div(f"Detected group: {detected_group}", className="status-muted"),
                html.Div("Prediction completed. You can now enter hand-sorting data.", className="status-muted"),
            ]
        )

        return status, original_img, overlay_img, metrics, store_data, False

    except Exception as exc:
        error = html.Div(f"Error: {exc}", className="status-error")
        return error, "Could not display image.", "Could not generate overlay.", [], None, True


@callback(
    Output("insert-data-status", "children"),
    Output("insert-metrics-container", "children", allow_duplicate=True),
    Output("calibration-comparison-store", "data"),
    Output("save-calibration-button", "disabled"),
    Input("calculate-error-button", "n_clicks"),
    State("insert-analysis-store", "data"),
    State("input-total-mass", "value"),
    State("input-copper-mass", "value"),
    State("input-reject-mass", "value"),
    State("input-notes", "value"),
    prevent_initial_call=True,
)
def calculate_calibration_error(n_clicks, data, total_mass, copper_mass, reject_mass, notes):
    if not data:
        return (
            html.Div("Please upload an image first.", className="status-error"),
            no_update,
            None,
            True,
        )

    if total_mass is None or copper_mass is None:
        return (
            html.Div("Please enter at least total sample mass and copper mass.", className="status-error"),
            no_update,
            None,
            True,
        )

    if total_mass <= 0:
        return (
            html.Div("Total sample mass must be greater than zero.", className="status-error"),
            no_update,
            None,
            True,
        )

    real_cu_pct = (float(copper_mass) / float(total_mass)) * 100.0
    predicted_cu_pct = float(data["predicted_cu_mass_pct"])
    error_pct_points = predicted_cu_pct - real_cu_pct
    abs_error = abs(error_pct_points)

    mass_balance_delta = None
    if reject_mass is not None:
        mass_balance_delta = float(total_mass) - float(copper_mass) - float(reject_mass)

    metrics = [
        result_card("Detected type", data["detected_group"]),
        result_card("Predicted Cu mass %", f"{predicted_cu_pct:.3f} %"),
        result_card("Real Cu mass %", f"{real_cu_pct:.3f} %", "Calculated from hand-sorting data"),
        result_card("Prediction error", f"{error_pct_points:+.3f} %-points", "Prediction minus real value"),
        result_card("Absolute error", f"{abs_error:.3f} %-points"),
        result_card("Total mass", f"{float(total_mass):.3f} g"),
        result_card("Copper mass", f"{float(copper_mass):.3f} g"),
    ]

    if reject_mass is not None:
        metrics.append(result_card("Reject mass", f"{float(reject_mass):.3f} g"))
        metrics.append(result_card("Mass balance delta", f"{mass_balance_delta:+.3f} g", "Total - Cu - Reject"))

    comparison_data = {
        **data,
        "total_mass_g": float(total_mass),
        "copper_mass_g": float(copper_mass),
        "reject_mass_g": float(reject_mass) if reject_mass is not None else None,
        "real_cu_pct": float(real_cu_pct),
        "error_pct_points": float(error_pct_points),
        "absolute_error_pct_points": float(abs_error),
        "mass_balance_delta_g": float(mass_balance_delta) if mass_balance_delta is not None else None,
        "notes": notes or "",
    }

    status = html.Div(
        [
            html.Div("Calibration comparison calculated successfully.", className="status-success"),
            html.Div("You can now add this record to the calibration database.", className="status-muted"),
        ]
    )

    return status, metrics, comparison_data, False

@callback(
    Output("save-data-status", "children"),
    Input("save-calibration-button", "n_clicks"),
    State("calibration-comparison-store", "data"),
    prevent_initial_call=True,
)
def save_calibration_data(n_clicks, comparison_data):
    if not comparison_data:
        return html.Div("Please calculate the calibration error before saving.", className="status-error")

    try:
        record = {
            "filename": comparison_data.get("filename"),
            "detected_group": comparison_data.get("detected_group"),
            "model_file": comparison_data.get("model_file"),

            "uploaded_path": comparison_data.get("uploaded_path"),
            "overlay_path": comparison_data.get("overlay_path"),

            "predicted_cu_area_pct": comparison_data.get("predicted_cu_area_pct"),
            "predicted_cu_mass_pct": comparison_data.get("predicted_cu_mass_pct"),

            "total_mass_g": comparison_data.get("total_mass_g"),
            "copper_mass_g": comparison_data.get("copper_mass_g"),
            "reject_mass_g": comparison_data.get("reject_mass_g"),

            "real_cu_pct": comparison_data.get("real_cu_pct"),
            "error_pct_points": comparison_data.get("error_pct_points"),
            "absolute_error_pct_points": comparison_data.get("absolute_error_pct_points"),
            "mass_balance_delta_g": comparison_data.get("mass_balance_delta_g"),

            "copper_particle_count": comparison_data.get("copper_particle_count"),
            "copper_pixels": comparison_data.get("copper_pixels"),
            "material_pixels": comparison_data.get("material_pixels"),

            "notes": comparison_data.get("notes", ""),
            "validated": 0,
        }


        record_id = insert_calibration_record(record)
        total_records = count_records()

        return html.Div(
            [
                html.Div("Calibration record saved successfully.", className="status-success"),
                html.Div(f"Record ID: {record_id}", className="status-muted"),
                html.Div(f"Total records in database: {total_records}", className="status-muted"),
                html.Div("The data is now stored in the calibration database.", className="status-muted"),
            ]
        )

    except Exception as exc:
        return html.Div(f"Error while saving data: {exc}", className="status-error")

@callback(
    Output("download-calibration-database", "data"),
    Input("download-database-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_calibration_database(n_clicks):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = REPORT_DIR / f"calibration_database_export_{timestamp}.xlsx"

    export_records_to_excel(output_path)

    return dcc.send_file(str(output_path))
@callback(
    Output("recalibration-status", "children"),
    Input("recalibrate-global-button", "n_clicks"),
    prevent_initial_call=True,
)
def recalibrate_global_callback(n_clicks):
    try:
        model = recalibrate_global_model(min_records=3)

        a = model["correction"]["a"]
        b = model["correction"]["b"]
        rmse = model["calibration_metrics"]["rmse_pct_points"]
        mae = model["calibration_metrics"]["mae_pct_points"]
        n_records = model["source_records"]

        return html.Div(
            [
                html.Div("GLOBAL model recalibrated successfully.", className="status-success"),
                html.Div(f"New version: {model['version']}", className="status-muted"),
                html.Div(f"Records used: {n_records}", className="status-muted"),
                html.Div(f"Correction: corrected Cu % = {a:.4f} × predicted Cu % + {b:.4f}", className="status-muted"),
                html.Div(f"RMSE: {rmse:.3f} %-points | MAE: {mae:.3f} %-points", className="status-muted"),
                html.Div("The updated model was saved in the data folder.", className="status-muted"),
            ]
        )

    except Exception as exc:
        return html.Div(f"Recalibration error: {exc}", className="status-error")
@callback(
    Output("calibration-plot", "figure"),
    Input("refresh-calibration-plot-button", "n_clicks"),
    Input("save-calibration-button", "n_clicks"),
    Input("recalibrate-global-button", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_calibration_plot(n_refresh, n_save, n_recalibrate):
    return create_calibration_plot()

@callback(
    Output("calibration-plot-wrapper", "style"),
    Output("toggle-calibration-plot-button", "children"),
    Input("toggle-calibration-plot-button", "n_clicks"),
)
def toggle_calibration_plot(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block", "marginTop": "16px"}, "Hide calibration plot"

    return {"display": "none"}, "Show calibration plot"