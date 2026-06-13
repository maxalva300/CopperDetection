from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
OVERLAY_DIR = RESULTS_DIR / "overlays"
REPORT_DIR = RESULTS_DIR / "reports"

# Fallback model used only when the filename cannot be mapped to a calibrated group.
MODEL_FILE = DATA_DIR / "calibrated_copper_model.json"
DYNAMIC_GLOBAL_MODEL_FILE = DATA_DIR / "calibrated_copper_model_GLOBAL_dynamic.json"
# v7 strategy:
#   CP  -> grouped by method: POL_CP, BAR_CP
#   NCP -> grouped by run + method: RUN1_POL_NCP, RUN2_POL_NCP, RUN3_BAR_NCP, RUN4_BAR_NCP
MODEL_FILES = {
    "POL_CP": DATA_DIR / "calibrated_model_POL_CP.json",
    "BAR_CP": DATA_DIR / "calibrated_model_BAR_CP.json",
    "RUN1_POL_NCP": DATA_DIR / "calibrated_model_RUN1_POL_NCP.json",
    "RUN2_POL_NCP": DATA_DIR / "calibrated_model_RUN2_POL_NCP.json",
    "RUN3_BAR_NCP": DATA_DIR / "calibrated_model_RUN3_BAR_NCP.json",
    "RUN4_BAR_NCP": DATA_DIR / "calibrated_model_RUN4_BAR_NCP.json",
    "GLOBAL": MODEL_FILE,
}

for folder in [DATA_DIR, UPLOAD_DIR, RESULTS_DIR, OVERLAY_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
