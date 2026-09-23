"""
predict.py — Inference engine
Usage: python predict.py "What is my account balance?"
"""
import sys, json, logging, numpy as np, joblib
from config import BEST_MODEL_PATH, CONFIDENCE_THRESHOLD
from preprocess import PreprocessingPipeline

logging.basicConfig(level=logging.WARNING)


class Predictor:
    """Loads model once; call .predict(text) repeatedly."""

    def __init__(self):
        self.pipeline    = PreprocessingPipeline.load()
        self.model       = joblib.load(BEST_MODEL_PATH)
        self.label_names = self.pipeline.classes

    def predict(self, text: str) -> dict:
        X      = self.pipeline.transform([text])
        proba  = self.model.predict_proba(X)[0]
        idx    = int(np.argmax(proba))
        conf   = float(proba[idx])

        top5   = np.argsort(proba)[::-1][:5]
        return {
            "intent"      : self.label_names[idx],
            "confidence"  : round(conf, 4),
            "is_confident": conf >= CONFIDENCE_THRESHOLD,
            "top_k"       : [
                {"intent": self.label_names[i], "confidence": round(float(proba[i]), 4)}
                for i in top5
            ],
        }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is my account balance?"
    p = Predictor()
    print(json.dumps(p.predict(query), indent=2))
