from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from config import DATA_DIR
from database import fetch_all_records


DYNAMIC_GLOBAL_MODEL = DATA_DIR / "calibrated_copper_model_GLOBAL_dynamic.json"


def recalibrate_global_model(min_records: int = 3) -> dict:
    """
    Creates a dynamic GLOBAL correction model using saved calibration records.

    The first version applies a post-prediction correction:

        corrected_cu_pct = a * predicted_cu_mass_pct + b

    It does not retrain image segmentation. It recalibrates the relationship
    between image-derived prediction and real hand-sorting Cu%.
    """

    df = fetch_all_records()

    if df.empty:
        raise ValueError("No calibration records found in the database.")

    # For this first version, only use GLOBAL records.
    df = df[df["detected_group"] == "GLOBAL"].copy()

    # Use only valid numeric records.
    df = df.dropna(subset=["predicted_cu_mass_pct", "real_cu_pct"])

    if len(df) < min_records:
        raise ValueError(
            f"At least {min_records} GLOBAL calibration records are required. "
            f"Current valid GLOBAL records: {len(df)}."
        )

    x = df["predicted_cu_mass_pct"].astype(float).to_numpy()
    y = df["real_cu_pct"].astype(float).to_numpy()

    # Linear correction y = a*x + b
    a, b = np.polyfit(x, y, deg=1)

    y_pred = a * x + b
    errors = y_pred - y

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    model = {
        "model_type": "dynamic_global_post_correction",
        "version": "GLOBAL_v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_records": int(len(df)),
        "correction": {
            "formula": "corrected_cu_pct = a * predicted_cu_mass_pct + b",
            "a": float(a),
            "b": float(b),
        },
        "calibration_metrics": {
            "rmse_pct_points": rmse,
            "mae_pct_points": mae,
        },
        "notes": (
            "This model recalibrates the GLOBAL mass estimate using validated "
            "hand-sorting data stored in calibration_data.db. It does not change "
            "the image segmentation criteria."
        ),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(DYNAMIC_GLOBAL_MODEL, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=4)

    return model


def apply_dynamic_global_correction(stats: dict, model: dict) -> dict:
    """
    Applies GLOBAL_v2 post-correction to stats produced by copper_core.analyze_image.
    """

    correction = model.get("correction", {})
    a = float(correction.get("a", 1.0))
    b = float(correction.get("b", 0.0))

    original_pred = float(stats["copper_%_mass_estimate"])
    corrected_pred = a * original_pred + b

    # Keep result in physically meaningful range.
    corrected_pred = max(0.0, min(100.0, corrected_pred))

    stats = dict(stats)
    stats["copper_%_mass_estimate_original_global_v1"] = original_pred
    stats["copper_%_mass_estimate"] = corrected_pred
    stats["global_dynamic_model_applied"] = True
    stats["global_dynamic_formula"] = f"{a:.6f} * x + {b:.6f}"

    return stats