"""
train.py — Full training pipeline for Nikhil AI Chatbot
=========================================================
Dataset  : CLINC150 / plus  (HuggingFace)
Models   : Logistic Regression (baseline)
           Linear SVM (calibrated)
           Random Forest
Splits   : Dataset's built-in train / validation / test
Outputs  : best_model.joblib · tfidf_vectorizer.joblib
           label_encoder.joblib · metrics.json · model_info.json
"""

import json
import time
import random
import logging
import warnings
import numpy as np
from typing import Any, Dict, List, Tuple

import joblib
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)

from config import (
    RANDOM_SEED, DATASET_NAME, DATASET_CONFIG,
    BEST_MODEL_PATH, METRICS_PATH, MODEL_INFO_PATH,
    CONFIDENCE_THRESHOLD,
)
from preprocess import PreprocessingPipeline

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DIVIDER = "─" * 68


# ──────────────────────────────────────────────────────────────────────────────
def load_splits() -> Tuple[List, List, List, List, List, List]:
    """Download CLINC150 and return (train/val/test) texts + labels."""
    logger.info(f"Downloading {DATASET_NAME}/{DATASET_CONFIG} from HuggingFace …")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    def _extract(split):
        texts  = [ex["text"]   for ex in ds[split]]
        labels = [ds[split].features["intent"].int2str(ex["intent"]) for ex in ds[split]]
        return texts, labels

    tr_t, tr_l = _extract("train")
    vl_t, vl_l = _extract("validation")
    te_t, te_l = _extract("test")

    logger.info(
        f"Split sizes → Train: {len(tr_t):,}  Val: {len(vl_t):,}  Test: {len(te_t):,}"
    )
    return tr_t, tr_l, vl_t, vl_l, te_t, te_l


# ──────────────────────────────────────────────────────────────────────────────
def build_models() -> Dict[str, Any]:
    return {
        "Logistic Regression (Baseline)": LogisticRegression(
            C=4.0, max_iter=1000, solver="lbfgs",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(
            LinearSVC(C=0.8, max_iter=2000, random_state=RANDOM_SEED), cv=3,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=2,
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
def _metrics(model, X, y_enc) -> Dict:
    y_pred = model.predict(X)
    return {
        "accuracy"         : round(float(accuracy_score(y_enc, y_pred)),                               4),
        "precision_macro"  : round(float(precision_score(y_enc, y_pred, average="macro",  zero_division=0)), 4),
        "recall_macro"     : round(float(recall_score   (y_enc, y_pred, average="macro",  zero_division=0)), 4),
        "f1_macro"         : round(float(f1_score       (y_enc, y_pred, average="macro",  zero_division=0)), 4),
        "f1_weighted"      : round(float(f1_score       (y_enc, y_pred, average="weighted", zero_division=0)), 4),
    }, y_pred


# ──────────────────────────────────────────────────────────────────────────────
def train_compare(pipeline, X_tr, y_tr, X_vl, y_vl):
    """Train every model, pick best by val Macro-F1."""
    models     = build_models()
    results    = {}
    best_model = None
    best_name  = ""
    best_f1    = -1.0

    print(f"\n{DIVIDER}")
    print("  MODEL COMPARISON  —  VALIDATION SET")
    print(DIVIDER)
    print(f"  {'Model':<38} {'Acc':>7} {'F1-Mac':>8} {'F1-Wt':>8}  {'Time':>6}")
    print(f"  {'-'*65}")

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        vm, _ = _metrics(model, X_vl, y_vl)
        results[name] = {**vm, "train_time_sec": round(elapsed, 2)}

        marker = " ◀" if vm["f1_macro"] > best_f1 else ""
        print(
            f"  {name:<38} {vm['accuracy']:>7.4f} "
            f"{vm['f1_macro']:>8.4f} {vm['f1_weighted']:>8.4f}  {elapsed:>5.1f}s{marker}"
        )

        if vm["f1_macro"] > best_f1:
            best_f1    = vm["f1_macro"]
            best_model = model
            best_name  = name

    print(f"\n  ✅  Best: {best_name}  (Val Macro-F1 = {best_f1:.4f})")
    return best_model, best_name, results


# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'━'*68}")
    print("  🤖  NIKHIL AI CHATBOT  —  TRAINING PIPELINE")
    print(f"{'━'*68}\n")

    # 1 ── Load dataset
    tr_t, tr_l, vl_t, vl_l, te_t, te_l = load_splits()
    num_intents = len(set(tr_l) | set(vl_l) | set(te_l))
    logger.info(f"Total unique intents: {num_intents}")

    # 2 ── Preprocessing  (fit ONLY on train — no leakage)
    pipeline = PreprocessingPipeline()
    X_tr, y_tr = pipeline.fit_transform(tr_t, tr_l)
    X_vl       = pipeline.transform(vl_t);  y_vl = pipeline.encode_labels(vl_l)
    X_te       = pipeline.transform(te_t);  y_te = pipeline.encode_labels(te_l)

    logger.info(f"X_train {X_tr.shape}  X_val {X_vl.shape}  X_test {X_te.shape}")

    # 3 ── Train & compare
    best_model, best_name, comparison = train_compare(
        pipeline, X_tr, y_tr, X_vl, y_vl
    )

    # 4 ── Final evaluation on UNTOUCHED test set
    test_m, y_pred = _metrics(best_model, X_te, y_te)

    print(f"\n{DIVIDER}")
    print("  FINAL EVALUATION  —  TEST SET  (never seen during training)")
    print(DIVIDER)
    print(f"  Accuracy     : {test_m['accuracy']:.4f}  ({test_m['accuracy']*100:.2f}%)")
    print(f"  Precision    : {test_m['precision_macro']:.4f}")
    print(f"  Recall       : {test_m['recall_macro']:.4f}")
    print(f"  F1 Macro     : {test_m['f1_macro']:.4f}")
    print(f"  F1 Weighted  : {test_m['f1_weighted']:.4f}")
    print(DIVIDER)

    # Per-class report + error analysis
    label_names = pipeline.classes
    report      = classification_report(y_te, y_pred, target_names=label_names, output_dict=True)
    per_cls_f1  = {k: v["f1-score"] for k, v in report.items() if k in label_names}
    worst       = sorted(per_cls_f1.items(), key=lambda x: x[1])[:10]

    print("\n  ⚠  Worst-performing intents (F1):")
    for cls, f1 in worst:
        bar = "█" * int(f1 * 20)
        print(f"    {cls:<35} {bar:<20}  {f1:.3f}")

    # Correct / incorrect sample analysis
    pred_labels = pipeline.decode_labels(y_pred)
    incorrect   = [(t, p, g) for t, p, g in zip(te_t, pred_labels, te_l) if p != g]
    correct     = [(t, p, g) for t, p, g in zip(te_t, pred_labels, te_l) if p == g]
    print(f"\n  Correct  : {len(correct):,}  ({len(correct)/len(te_t)*100:.1f}%)")
    print(f"  Incorrect: {len(incorrect):,}  ({len(incorrect)/len(te_t)*100:.1f}%)")

    print("\n  Sample mis-classifications:")
    for txt, pred, true in incorrect[:5]:
        print(f"    Text : {txt}")
        print(f"    Pred : {pred}  |  True: {true}\n")

    # 5 ── Save artifacts
    pipeline.save()
    joblib.dump(best_model, BEST_MODEL_PATH)
    logger.info(f"Model saved → {BEST_MODEL_PATH}")

    # 6 ── Save metrics JSON
    all_metrics = {
        "best_model"     : best_name,
        "test_metrics"   : test_m,
        "comparison"     : comparison,
        "per_class_report": {k: v for k, v in report.items() if isinstance(v, dict)},
        "dataset_info"   : {
            "name"        : DATASET_NAME,
            "config"      : DATASET_CONFIG,
            "train_size"  : len(tr_t),
            "val_size"    : len(vl_t),
            "test_size"   : len(te_t),
            "num_intents" : num_intents,
        },
        "error_analysis" : {
            "total"       : len(te_t),
            "correct"     : len(correct),
            "incorrect"   : len(incorrect),
            "error_rate"  : round(len(incorrect) / len(te_t), 4),
        },
    }
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2))

    # 7 ── Save model-info JSON
    MODEL_INFO_PATH.write_text(json.dumps({
        "model_name"           : best_name,
        "framework"            : "scikit-learn",
        "dataset"              : f"{DATASET_NAME}/{DATASET_CONFIG}",
        "num_classes"          : len(label_names),
        "vocabulary_size"      : len(pipeline.vectorizer.vocabulary_),
        "confidence_threshold" : CONFIDENCE_THRESHOLD,
        "test_accuracy"        : test_m["accuracy"],
        "test_f1_macro"        : test_m["f1_macro"],
        "random_seed"          : RANDOM_SEED,
    }, indent=2))

    print(f"\n  ✅  All artifacts saved to ./model/")
    print(f"  ▶   Run:  python app.py   to start the chatbot!\n")


if __name__ == "__main__":
    main()
