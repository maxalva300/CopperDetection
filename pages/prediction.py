from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import json
import cv2
from dash import Input, Output, State, dcc, html, no_update, callback
import dash_bootstrap_components as dbc

from config import MODEL_FILES, OVERLAY_DIR, REPORT_DIR, UPLOAD_DIR, DYNAMIC_GLOBAL_MODEL_FILE
from model_updater import apply_dynamic_global_correction
from copper_core import (
    analyze_image,
    bgr_to_data_uri,
    decode_dash_upload,
    load_model,
    save_single_result_excel,
    save_uploaded_image,
)


def detect_group_from_filename(filename: str) -> str:
    """
    v7 model selection:
      RUN1_POL_CP_img1.jpg   -> POL_CP
      RUN2_POL_NCP_img4.jpg  -> RUN2_POL_NCP
      RUN3_BAR_CP_img1.jpg   -> BAR_CP
      RUN4_BAR_NCP_img4.jpg  -> RUN4_BAR_NCP
    """
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


def metric_card(label: str, value: str, help_text: str | None = None) -> html.Div:
    children = [
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
    ]
    if help_text:
        children.append(html.Div(help_text, className="metric-help"))
    return html.Div(children, className="metric-card")


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
                            html.H1("Copper Particle Detection", className="main-title"),
                            html.Div(
                                "Optical analysis for Eddy Current Separation samples",
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
                                "Upload one sample image. The app will automatically select the calibrated model from the filename.",
                                className="section-description",
                            ),
                            dcc.Upload(
                                id="upload-image",
                                className="upload-box",
                                children=html.Div(
                                    [
                                        html.Div(
                                            "Drag and drop an image here",
                                            className="upload-main-text",
                                        ),
                                        html.Div(
                                            "or click to select a JPG / PNG file",
                                            className="upload-sub-text",
                                        ),
                                    ]
                                ),
                                accept="image/*",
                                multiple=False,
                            ),
                            html.Div(id="upload-status", className="status-box"),
                            html.H2("2. Automatic model", className="section-title compact"),
                            html.Div(
                                id="model-status",
                                className="model-box",
                                children="CP uses POL_CP/BAR_CP. NCP uses run-specific models: RUN1_POL_NCP, RUN2_POL_NCP, RUN3_BAR_NCP, RUN4_BAR_NCP.",
                            ),
                            html.P(
                                "Example: RUN2_POL_NCP_img4.jpg uses RUN2_POL_NCP, while RUN3_BAR_CP_img1.jpg uses BAR_CP.",
                                className="small-note",
                            ),
                            html.Div(
                                className="download-row",
                                children=[
                                    html.Button(
                                        "Download overlay",
                                        id="download-overlay-button",
                                        className="primary-button",
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "Download Excel",
                                        id="download-excel-button",
                                        className="secondary-button",
                                        disabled=True,
                                    ),
                                ],
                            ),
                            dcc.Download(id="download-overlay"),
                            dcc.Download(id="download-excel"),
                        ],
                    ),
                    html.Div(
                        className="right-panel",
                        children=[
                            html.Div(
                                className="glass-card results-card",
                                children=[
                                    html.H2("Results", className="section-title"),
                                    html.Div(id="metrics-container", className="metrics-grid"),
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
                                                id="original-image-container",
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
                                                id="overlay-image-container",
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
            dcc.Store(id="analysis-store"),
            html.Div("Copper Image Web · Summer School Study Project", className="footer"),
        ],
    )


@callback(
    Output("upload-status", "children"),
    Output("model-status", "children"),
    Output("original-image-container", "children"),
    Output("overlay-image-container", "children"),
    Output("metrics-container", "children"),
    Output("analysis-store", "data"),
    Output("download-overlay-button", "disabled"),
    Output("download-excel-button", "disabled"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    prevent_initial_call=True,
)
def run_analysis(contents, filename):
    if contents is None or filename is None:
        return no_update, no_update, no_update, no_update, no_update, no_update, True, True

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
        overlay_path = OVERLAY_DIR / f"{stem}_overlay_{detected_group}_{timestamp}.jpg"
        excel_path = REPORT_DIR / f"{stem}_results_{detected_group}_{timestamp}.xlsx"

        cv2.imwrite(str(overlay_path), overlay)
        save_single_result_excel(stats, filename, excel_path)

        original_img = html.Img(src=contents, className="preview-image")
        overlay_img = html.Img(src=bgr_to_data_uri(overlay), className="preview-image")

        metrics = [
            metric_card("Detected type", detected_group, "Selected automatically from filename"),
            metric_card("Cu area %", f"{stats['copper_%_area']:.3f} %", "Detected copper pixels / material pixels"),
            metric_card("Estimated Cu mass %", f"{stats['copper_%_mass_estimate']:.3f} %", "Calibrated model estimate"),
            metric_card("Material in image", f"{stats['material_%_of_image']:.3f} %", "Non-background pixels"),
            metric_card("Copper particles", f"{stats['copper_particle_count']}", "Connected components"),
            metric_card("Copper pixels", f"{stats['copper_pixels']:,}"),
            metric_card("Material pixels", f"{stats['material_pixels']:,}"),
        ]
        if stats.get("global_dynamic_model_applied"):
            metrics.append(
                metric_card(
                    "Global model version",
                    model.get("dynamic_version", "GLOBAL_v2"),
                    "Dynamic correction applied after GLOBAL v1 prediction",
                )
            )
            metrics.append(
                metric_card(
                    "Original GLOBAL v1 prediction",
                    f"{stats['copper_%_mass_estimate_original_global_v1']:.3f} %",
                    "Prediction before dynamic correction",
                )
            )
        rmse = model.get("calibration_metrics", {}).get("rmse_pct_points")
        if rmse is not None:
            metrics.append(
                metric_card(
                    "Model RMSE",
                    f"{float(rmse):.3f} %-points",
                    "Calibration error for selected model",
                )
            )

        store_data = {
            "filename": filename,
            "detected_group": detected_group,
            "model_file": str(model_path),
            "uploaded_path": str(uploaded_path),
            "overlay_path": str(overlay_path),
            "excel_path": str(excel_path),
            "stats": stats,
        }

        status = html.Div(
            [
                html.Div(f"Loaded: {filename}", className="status-success"),
                html.Div(f"Detected group: {detected_group}", className="status-muted"),
                html.Div("Analysis completed successfully.", className="status-muted"),
            ]
        )

        model_status = html.Div(
            [
                html.Div(f"Detected sample model: {detected_group}", className="status-success"),
                html.Div("Automatic model selected from filename", className="status-muted"),
            ]
        )

        return status, model_status, original_img, overlay_img, metrics, store_data, False, False

    except Exception as exc:
        error = html.Div(f"Error: {exc}", className="status-error")
        return error, no_update, "Could not display image.", "Could not generate overlay.", [], None, True, True


@callback(
    Output("download-overlay", "data"),
    Input("download-overlay-button", "n_clicks"),
    State("analysis-store", "data"),
    prevent_initial_call=True,
)
def download_overlay(n_clicks, data):
    if not data or not data.get("overlay_path"):
        return no_update
    return dcc.send_file(data["overlay_path"])


@callback(
    Output("download-excel", "data"),
    Input("download-excel-button", "n_clicks"),
    State("analysis-store", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, data):
    if not data or not data.get("excel_path"):
        return no_update
    return dcc.send_file(data["excel_path"])