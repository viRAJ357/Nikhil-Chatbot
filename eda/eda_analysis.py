"""
eda/eda_analysis.py — Complete EDA for CLINC150 dataset
Run standalone: python eda/eda_analysis.py
Saves results to model/eda_results.json
"""
import json, sys, logging
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datasets import load_dataset
from config import DATASET_NAME, DATASET_CONFIG, MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
DIV = "─" * 65

def run():
    log.info("Downloading dataset …")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    def to_df(split):
        return pd.DataFrame({
            "text"  : [e["text"] for e in ds[split]],
            "intent": [ds[split].features["intent"].int2str(e["intent"]) for e in ds[split]],
            "split" : split,
        })

    tr = to_df("train"); vl = to_df("validation"); te = to_df("test")
    df = pd.concat([tr, vl, te], ignore_index=True)
    df["word_count"] = df["text"].str.split().str.len()
    df["char_count"] = df["text"].str.len()

    print(f"\n{DIV}\n  📊  CLINC150 EDA REPORT\n{DIV}")

    # ── Shape & columns ───────────────────────────────────────────────────────
    print(f"\n  Shape          : {df.shape}")
    print(f"  Columns        : {list(df.columns)}")
    print(f"  Train          : {len(tr):,}  ({len(tr)/len(df)*100:.1f}%)")
    print(f"  Validation     : {len(vl):,}  ({len(vl)/len(df)*100:.1f}%)")
    print(f"  Test           : {len(te):,}  ({len(te)/len(df)*100:.1f}%)")
    print(f"  Unique intents : {df['intent'].nunique()}")

    # ── Missing / Duplicates ──────────────────────────────────────────────────
    missing = df.isna().sum()
    dups    = df.duplicated(subset=["text"]).sum()
    print(f"\n  Missing values :\n{missing.to_string()}")
    print(f"  Duplicate texts: {dups}  ({dups/len(df)*100:.2f}%)")

    # ── Text length distribution ───────────────────────────────────────────────
    wc = df["word_count"]
    print(f"\n  Word Count Stats:")
    for stat, val in [("min",wc.min()),("max",wc.max()),("mean",round(wc.mean(),2)),
                      ("std",round(wc.std(),2)),("p25",wc.quantile(.25)),
                      ("p50",wc.quantile(.50)),("p75",wc.quantile(.75)),("p95",wc.quantile(.95))]:
        print(f"    {stat:<6}: {val}")

    # ── Class distribution ────────────────────────────────────────────────────
    counts = df["intent"].value_counts()
    print(f"\n  Intent Distribution:")
    print(f"    Min samples/intent : {counts.min()}")
    print(f"    Max samples/intent : {counts.max()}")
    print(f"    Mean samples/intent: {counts.mean():.1f}")
    print(f"    Std                : {counts.std():.1f}")

    # ── In-scope vs out-of-scope ──────────────────────────────────────────────
    ins = df[df["intent"] != "oos"]; oos = df[df["intent"] == "oos"]
    print(f"\n  Class Balance:")
    print(f"    In-scope    : {len(ins):,}  ({len(ins)/len(df)*100:.1f}%)")
    print(f"    Out-of-scope: {len(oos):,}  ({len(oos)/len(df)*100:.1f}%)")

    # ── Top/bottom intents ────────────────────────────────────────────────────
    print(f"\n  Top 10 intents:")
    for intent, cnt in counts.head(10).items():
        print(f"    {intent:<35} {cnt}")

    # ── Word frequency ────────────────────────────────────────────────────────
    words = " ".join(df["text"]).lower().split()
    freq  = Counter(words).most_common(20)
    print(f"\n  Top 20 words:")
    for w, c in freq:
        print(f"    {w:<20} {c:,}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "total_samples"  : int(len(df)),
        "train_size"     : int(len(tr)),
        "val_size"       : int(len(vl)),
        "test_size"      : int(len(te)),
        "unique_intents" : int(df["intent"].nunique()),
        "missing_values" : missing.to_dict(),
        "duplicates"     : int(dups),
        "word_count"     : {
            "min": int(wc.min()), "max": int(wc.max()),
            "mean": float(round(wc.mean(),2)), "std": float(round(wc.std(),2)),
            "p25": float(wc.quantile(.25)), "p50": float(wc.quantile(.50)),
            "p75": float(wc.quantile(.75)), "p95": float(wc.quantile(.95)),
        },
        "intent_distribution": {
            "min": int(counts.min()), "max": int(counts.max()),
            "mean": float(round(counts.mean(),2)),
        },
        "class_balance"  : {"in_scope": int(len(ins)), "out_of_scope": int(len(oos))},
        "top_intents"    : counts.head(20).to_dict(),
        "top_words"      : dict(freq),
        "intent_list"    : sorted(df["intent"].unique().tolist()),
    }
    out = MODEL_DIR / "eda_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n  ✅  EDA saved → {out}")
    return result

if __name__ == "__main__":
    run()
