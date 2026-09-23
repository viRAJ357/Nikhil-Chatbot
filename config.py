"""
config.py — Central configuration for Nikhil AI Chatbot
"""
from pathlib import Path

# ─── Directories ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
MODEL_DIR     = BASE_DIR / "model"
EDA_DIR       = BASE_DIR / "eda"
STATIC_DIR    = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

for d in [DATA_DIR, MODEL_DIR, EDA_DIR, STATIC_DIR, TEMPLATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME   = "clinc/clinc_oos"
DATASET_CONFIG = "plus"

# ─── Preprocessing ────────────────────────────────────────────────────────────
MAX_FEATURES  = 20_000
NGRAM_RANGE   = (1, 2)
MIN_DF        = 2
SUBLINEAR_TF  = True

# ─── Model ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40   # below this → graceful fallback
MAX_HISTORY          = 30     # max conversation turns in memory

# ─── Saved Artifacts ──────────────────────────────────────────────────────────
VECTORIZER_PATH    = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
BEST_MODEL_PATH    = MODEL_DIR / "best_model.joblib"
METRICS_PATH       = MODEL_DIR / "metrics.json"
MODEL_INFO_PATH    = MODEL_DIR / "model_info.json"
EDA_RESULTS_PATH   = MODEL_DIR / "eda_results.json"

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST    = "0.0.0.0"
API_PORT    = 8000
API_TITLE   = "Nikhil AI Chatbot API"
API_VERSION = "1.0.0"

# ─── LLM Fallback ─────────────────────────────────────────────────────────────
import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
