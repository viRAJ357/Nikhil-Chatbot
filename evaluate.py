"""
evaluate.py — Standalone evaluation on untouched test set
Loads saved model & pipeline, prints full metrics + error analysis.
"""

import json
import logging
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
import joblib

from config import DATASET_NAME, DATASET_CONFIG, BEST_MODEL_PATH, METRICS_PATH
from preprocess import PreprocessingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIV = "─" * 68


def main():
    # 1 — Load test set
    logger.info("Loading dataset …")
    ds         = load_dataset(DATASET_NAME, DATASET_CONFIG)
    test_texts = [ex["text"] for ex in ds["test"]]
    test_labels= [ds["test"].features["intent"].int2str(ex["intent"]) for ex in ds["test"]]

    # 2 — Load pipeline & model
    pipeline = PreprocessingPipeline.load()
    model    = joblib.load(BEST_MODEL_PATH)

    # 3 — Transform
    X_test = pipeline.transform(test_texts)
    y_test = pipeline.encode_labels(test_labels)

    # 4 — Predict
    y_pred     = model.predict(X_test)
    y_proba    = model.predict_proba(X_test)
    y_conf     = np.max(y_proba, axis=1)
    pred_labels= pipeline.decode_labels(y_pred)
    label_names= pipeline.classes

    # 5 — Metrics
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, average="macro",    zero_division=0)
    rec    = recall_score   (y_test, y_pred, average="macro",    zero_division=0)
    f1_mac = f1_score       (y_test, y_pred, average="macro",    zero_division=0)
    f1_wt  = f1_score       (y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{DIV}")
    print("  EVALUATION REPORT  —  TEST SET")
    print(DIV)
    print(f"  Model          : {joblib.load(BEST_MODEL_PATH).__class__.__name__}")
    print(f"  Test samples   : {len(test_texts):,}")
    print(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (M)  : {prec:.4f}")
    print(f"  Recall    (M)  : {rec:.4f}")
    print(f"  F1 Macro       : {f1_mac:.4f}")
    print(f"  F1 Weighted    : {f1_wt:.4f}")
    print(f"  Avg Confidence : {y_conf.mean():.4f}")
    print(DIV)

    # Per-class
    report   = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    per_cls  = [(k, v) for k, v in report.items() if k in label_names]
    by_f1    = sorted(per_cls, key=lambda x: x[1]["f1-score"])

    print("\n  ⬇  Bottom 10 intents by F1:")
    for cls, v in by_f1[:10]:
        bar = "█" * int(v["f1-score"] * 20)
        print(f"    {cls:<35} {bar:<20}  {v['f1-score']:.3f}  (n={int(v['support'])})")

    print("\n  ⬆  Top 10 intents by F1:")
    for cls, v in by_f1[-10:][::-1]:
        bar = "█" * int(v["f1-score"] * 20)
        print(f"    {cls:<35} {bar:<20}  {v['f1-score']:.3f}  (n={int(v['support'])})")

    # Correct / incorrect
    correct   = [(t, p, g) for t, p, g, in zip(test_texts, pred_labels, test_labels) if p == g]
    incorrect = [(t, p, g) for t, p, g  in zip(test_texts, pred_labels, test_labels) if p != g]
    print(f"\n  Correct  : {len(correct):,}  ({len(correct)/len(test_texts)*100:.1f}%)")
    print(f"  Incorrect: {len(incorrect):,}  ({len(incorrect)/len(test_texts)*100:.1f}%)")

    print("\n  ✔  Sample correct:")
    for t, p, g in correct[:3]:
        print(f"    [{g}] → {t}")

    print("\n  ✘  Sample incorrect:")
    for t, p, g in incorrect[:3]:
        print(f"    Text: {t}")
        print(f"    Pred: {p}  |  True: {g}\n")


if __name__ == "__main__":
    main()
