from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import pandas as pd


DEFAULT_MODEL: Dict[str, Any] = {
    "model_name": "default_model",
    "group": "GLOBAL",
    "green_lower": [30, 40, 30],
    "green_upper": [95, 255, 255],
    "h_upper": 22,
    "s_lower": 60,
    "v_lower": 40,
    "rg_ratio": 1.05,
    "rb_ratio": 1.10,
    "redness_min": 0.0,
    "min_particle_area": 30,
    "morph_kernel": 3,
    "dark_value_cutoff": 25,
    "copper_overlay_alpha": 0.65,
    "mass_model": {
        "type": "density_k",
        "rho_copper": 8.96,
        "rho_other": 2.0,
        "k_area_factor": 1.0,
        "post_correction_a": 1.0,
        "post_correction_b": 0.0,
        "sat_base": None,
        "sat_max": None,
        "sat_k": None,
    },
}


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = base.copy()
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def normalize_model(raw: Dict[str, Any], model_path: Path | None = None) -> Dict[str, Any]:
    """Normalize JSON models generated during this project into one runtime format."""
    model = _deep_merge(DEFAULT_MODEL, raw)

    seg = raw.get("segmentation_parameters", {}) or {}
    if seg:
        for key in [
            "green_lower", "green_upper", "copper_lower_a", "copper_upper_a",
            "copper_lower_b", "copper_upper_b", "dark_value_cutoff", "morph_kernel",
            "overlay_alpha", "copper_overlay_alpha",
        ]:
            if key in seg:
                model[key] = seg[key]

        if "rg_ratio" in seg:
            model["rg_ratio"] = seg["rg_ratio"]
        if "rb_ratio" in seg:
            model["rb_ratio"] = seg["rb_ratio"]
        if "redness_min" in seg:
            model["redness_min"] = seg["redness_min"]
        if "redness_index_min" in seg:
            model["redness_min"] = seg["redness_index_min"]
        if "min_copper_area_px" in seg:
            model["min_particle_area"] = seg["min_copper_area_px"]
        if "min_particle_area" in seg:
            model["min_particle_area"] = seg["min_particle_area"]
        if "overlay_alpha" in seg:
            model["copper_overlay_alpha"] = seg["overlay_alpha"]

        if "copper_upper_a" in seg and len(seg["copper_upper_a"]) >= 1:
            model["h_upper"] = seg["copper_upper_a"][0]
        if "copper_lower_a" in seg and len(seg["copper_lower_a"]) >= 3:
            model["s_lower"] = seg["copper_lower_a"][1]
            model["v_lower"] = seg["copper_lower_a"][2]

    # Older linear model format.
    if "linear_correction_model" in raw:
        lin = raw["linear_correction_model"] or {}
        model["mass_model"] = {
            "type": "linear_correction",
            "a": lin.get("a", lin.get("linear_a", 1.0)),
            "b": lin.get("b", lin.get("linear_b", 0.0)),
        }

    # v3-v8 calibration scripts store mass model here.
    if "mass_model" in raw:
        model["mass_model"] = _deep_merge(DEFAULT_MODEL["mass_model"], raw["mass_model"] or {})

    mm = model.get("mass_model", {}) or {}
    if "post_a" in mm and "post_correction_a" not in mm:
        mm["post_correction_a"] = mm["post_a"]
    if "post_b" in mm and "post_correction_b" not in mm:
        mm["post_correction_b"] = mm["post_b"]
    if "k" in mm and "k_area_factor" not in mm:
        mm["k_area_factor"] = mm["k"]
    model["mass_model"] = mm

    if model_path is not None:
        model["model_file"] = model_path.name
    return model


def load_model(model_path: str | Path | None = None) -> Dict[str, Any]:
    if model_path is None:
        return DEFAULT_MODEL.copy()

    model_path = Path(model_path)
    if not model_path.exists():
        model = DEFAULT_MODEL.copy()
        model["model_file"] = f"{model_path.name} not found; using default"
        return model

    with model_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return normalize_model(raw, model_path=model_path)


def _get_number(model: Dict[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        if key in model and model[key] is not None:
            return float(model[key])
    return float(default)


def decode_dash_upload(contents: str) -> np.ndarray:
    if not contents or "," not in contents:
        raise ValueError("Invalid uploaded image content.")
    _, b64_data = contents.split(",", 1)
    image_bytes = base64.b64decode(b64_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV could not decode the uploaded image.")
    return bgr


def bgr_to_data_uri(bgr: np.ndarray, ext: str = ".jpg") -> str:
    ok, buffer = cv2.imencode(ext, bgr)
    if not ok:
        raise ValueError("Could not encode image for display.")
    encoded = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if ext.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def save_uploaded_image(contents: str, filename: str, upload_dir: str | Path) -> Path:
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    out_path = upload_dir / safe_name
    _, b64_data = contents.split(",", 1)
    out_path.write_bytes(base64.b64decode(b64_data))
    return out_path


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(bool)

    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8, dtype=np.uint8)
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def _threshold_array(model: Dict[str, Any], key: str, fallback: list[int]) -> np.ndarray:
    return np.array(model.get(key, fallback), dtype=np.uint8)


def segment_copper(bgr: np.ndarray, model: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    green_lower = _threshold_array(model, "green_lower", [30, 40, 30])
    green_upper = _threshold_array(model, "green_upper", [95, 255, 255])

    h_upper = int(_get_number(model, "h_upper", "h_up", default=22))
    s_lower = int(_get_number(model, "s_lower", "s_lo", default=60))
    v_lower = int(_get_number(model, "v_lower", "v_lo", default=40))

    copper_lower_a = _threshold_array(model, "copper_lower_a", [0, s_lower, v_lower])
    copper_upper_a = _threshold_array(model, "copper_upper_a", [h_upper, 255, 255])
    copper_lower_b = _threshold_array(model, "copper_lower_b", [170, s_lower, v_lower])
    copper_upper_b = _threshold_array(model, "copper_upper_b", [179, 255, 255])

    rg_ratio = _get_number(model, "rg_ratio", "r_g_ratio", default=1.05)
    rb_ratio = _get_number(model, "rb_ratio", "r_b_ratio", default=1.10)
    redness_min = _get_number(model, "redness_min", "redness_index_min", default=0.0)
    min_particle_area = int(_get_number(model, "min_particle_area", "min_area", "min_copper_area_px", default=30))
    morph_kernel = int(_get_number(model, "morph_kernel", default=3))
    dark_cutoff = int(_get_number(model, "dark_value_cutoff", default=25))

    bg_mask = cv2.inRange(hsv, green_lower, green_upper) > 0
    very_dark = hsv[..., 2] < dark_cutoff
    material = (~bg_mask) & (~very_dark)

    hsv_cu_a = cv2.inRange(hsv, copper_lower_a, copper_upper_a) > 0
    hsv_cu_b = cv2.inRange(hsv, copper_lower_b, copper_upper_b) > 0
    hsv_copper = hsv_cu_a | hsv_cu_b

    b = bgr[..., 0].astype(np.float32) + 1.0
    g = bgr[..., 1].astype(np.float32) + 1.0
    r = bgr[..., 2].astype(np.float32) + 1.0
    rgb_ratio_filter = (r / g >= rg_ratio) & (r / b >= rb_ratio)

    if redness_min > 0:
        redness_index = r / (r + g + b)
        red_filter = redness_index >= redness_min
    else:
        red_filter = np.ones_like(material, dtype=bool)

    copper = hsv_copper & rgb_ratio_filter & red_filter & material

    if morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        material_u8 = (material.astype(np.uint8)) * 255
        copper_u8 = (copper.astype(np.uint8)) * 255
        material = cv2.morphologyEx(material_u8, cv2.MORPH_OPEN, kernel) > 0
        material = cv2.morphologyEx((material.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, kernel) > 0
        copper = cv2.morphologyEx(copper_u8, cv2.MORPH_OPEN, kernel) > 0
        copper = cv2.morphologyEx((copper.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, kernel) > 0
        copper = copper & material

    copper = remove_small_components(copper, min_particle_area)
    copper = copper & material
    return material, copper


def make_red_overlay(bgr: np.ndarray, copper_mask: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    out = bgr.copy()
    red = np.full_like(bgr, (0, 0, 255), dtype=np.uint8)
    out[copper_mask] = ((1.0 - alpha) * out[copper_mask] + alpha * red[copper_mask]).astype(np.uint8)
    return out


def density_mass_pct_from_area(area_pct_copper: float, rho_other: float, rho_copper: float = 8.96, k_area_factor: float = 1.0) -> float:
    a_cu = float(np.clip(area_pct_copper / 100.0, 0.0, 1.0))
    a_other = max(0.0, 1.0 - a_cu)
    effective_cu = max(0.0, k_area_factor * a_cu)
    m_cu = rho_copper * effective_cu
    m_other = rho_other * a_other
    total = m_cu + m_other
    return 0.0 if total <= 0 else float((m_cu / total) * 100.0)


def saturated_area_prediction(area_pct: float, base: float, sat_max: float, sat_k: float) -> float:
    area = max(0.0, float(area_pct))
    pred = base + (sat_max - base) * (1.0 - np.exp(-sat_k * area))
    return float(np.clip(pred, 0, 100))


def predict_mass_pct(area_pct: float, model: Dict[str, Any]) -> Dict[str, float | str]:
    mm = model.get("mass_model", {}) or {}
    model_type = str(mm.get("type", "density_k"))

    if model_type == "linear_correction":
        a = float(mm.get("a", mm.get("linear_a", 1.0)))
        b = float(mm.get("b", mm.get("linear_b", 0.0)))
        pred = float(np.clip(a * area_pct + b, 0, 100))
        return {
            "prediction_type": model_type,
            "density_based_cu_mass_pct": pred,
            "copper_%_mass_estimate": pred,
        }

    rho_copper = float(mm.get("rho_copper", 8.96))
    rho_other = float(mm.get("rho_other", 2.0))
    k_area_factor = float(mm.get("k_area_factor", 1.0))
    density_pred = density_mass_pct_from_area(area_pct, rho_other, rho_copper, k_area_factor)

    if model_type == "ncp_saturated_area":
        base = mm.get("sat_base")
        sat_max = mm.get("sat_max")
        sat_k = mm.get("sat_k")
        if base is not None and sat_max is not None and sat_k is not None:
            pred = saturated_area_prediction(area_pct, float(base), float(sat_max), float(sat_k))
            return {
                "prediction_type": model_type,
                "density_based_cu_mass_pct": float(density_pred),
                "copper_%_mass_estimate": pred,
                "sat_base": float(base),
                "sat_max": float(sat_max),
                "sat_k": float(sat_k),
                "rho_copper_used": rho_copper,
                "rho_other_used": rho_other,
                "k_area_factor_used": k_area_factor,
            }

    post_a = float(mm.get("post_correction_a", mm.get("post_a", 1.0)))
    post_b = float(mm.get("post_correction_b", mm.get("post_b", 0.0)))
    pred = float(np.clip(post_a * density_pred + post_b, 0, 100))
    return {
        "prediction_type": model_type,
        "density_based_cu_mass_pct": float(density_pred),
        "copper_%_mass_estimate": pred,
        "rho_copper_used": rho_copper,
        "rho_other_used": rho_other,
        "k_area_factor_used": k_area_factor,
        "post_correction_a": post_a,
        "post_correction_b": post_b,
    }


def particle_statistics(copper_mask: np.ndarray) -> Dict[str, float]:
    mask_u8 = copper_mask.astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, n_labels)]
    if not areas:
        return {
            "copper_particle_count": 0,
            "mean_copper_particle_area_px": 0.0,
            "max_copper_particle_area_px": 0,
        }
    return {
        "copper_particle_count": len(areas),
        "mean_copper_particle_area_px": float(np.mean(areas)),
        "max_copper_particle_area_px": int(np.max(areas)),
    }


def analyze_image(bgr: np.ndarray, model: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    material, copper = segment_copper(bgr, model)

    n_total = int(bgr.shape[0] * bgr.shape[1])
    n_material = int(material.sum())
    n_copper = int(copper.sum())

    material_pct = (n_material / n_total) * 100.0 if n_total else 0.0
    copper_area_pct = (n_copper / n_material) * 100.0 if n_material else 0.0

    pred_info = predict_mass_pct(copper_area_pct, model)
    alpha = float(model.get("copper_overlay_alpha", model.get("overlay_alpha", 0.65)))
    overlay = make_red_overlay(bgr, copper, alpha=alpha)

    stats = {
        "model_group": model.get("group", model.get("model_name", "UNKNOWN")),
        "model_file": model.get("model_file", "default"),
        "material_%_of_image": round(material_pct, 3),
        "copper_%_area": round(copper_area_pct, 3),
        "copper_%_mass_estimate": round(float(pred_info["copper_%_mass_estimate"]), 3),
        "density_based_cu_mass_pct": round(float(pred_info.get("density_based_cu_mass_pct", pred_info["copper_%_mass_estimate"])), 3),
        "prediction_type": pred_info.get("prediction_type", "unknown"),
        "total_pixels": n_total,
        "material_pixels": n_material,
        "copper_pixels": n_copper,
    }

    for key in [
        "rho_copper_used", "rho_other_used", "k_area_factor_used",
        "post_correction_a", "post_correction_b", "sat_base", "sat_max", "sat_k",
    ]:
        if key in pred_info:
            stats[key] = round(float(pred_info[key]), 6)

    stats.update(particle_statistics(copper))
    return overlay, stats


def save_single_result_excel(stats: Dict[str, Any], filename: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"filename": filename, **stats}
    df = pd.DataFrame([row])
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="image_result", index=False)
    return output_path
