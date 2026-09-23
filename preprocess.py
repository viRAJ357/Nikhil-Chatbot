"""
preprocess.py — Reproducible NLP preprocessing pipeline
Handles: cleaning · tokenisation · lemmatisation · TF-IDF · label encoding
The fitted pipeline is saved to disk so identical transforms are applied at inference.
"""

import re
import logging
import joblib
import numpy as np
from typing import List, Tuple

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    RANDOM_SEED, MAX_FEATURES, NGRAM_RANGE, MIN_DF, SUBLINEAR_TF,
    VECTORIZER_PATH, LABEL_ENCODER_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _download_nltk():
    for res in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


_download_nltk()


# ──────────────────────────────────────────────────────────────────────────────
class TextPreprocessor:
    """
    Step 1 of the pipeline: raw text → cleaned, lemmatised string.

    Design note on stopwords: we intentionally KEEP stopwords because
    words like 'what', 'how', 'when', 'my' are discriminative signals
    for intent classification (e.g. "what is my balance" vs "how do I
    transfer money").
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)           # URLs
        text = re.sub(r"\S+@\S+", " ", text)                   # emails
        text = re.sub(r"[^a-z0-9\s']", " ", text)             # special chars
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def lemmatise(self, text: str) -> str:
        tokens = word_tokenize(text)
        return " ".join(self.lemmatizer.lemmatize(t) for t in tokens)

    def __call__(self, text: str) -> str:
        return self.lemmatise(self.clean(text))

    def batch(self, texts: List[str]) -> List[str]:
        return [self(t) for t in texts]


# ──────────────────────────────────────────────────────────────────────────────
class PreprocessingPipeline:
    """
    Full pipeline:  raw text  →  TF-IDF sparse matrix  /  encoded label
    ─────────────────────────────────────────────────────────────────────
    Usage
    -----
    # Training
    pipeline = PreprocessingPipeline()
    X_train, y_train = pipeline.fit_transform(train_texts, train_labels)

    # Inference / validation / test
    X = pipeline.transform(texts)
    y = pipeline.encode_labels(labels)

    # Persistence
    pipeline.save()
    pipeline = PreprocessingPipeline.load()
    """

    def __init__(self):
        self.cleaner = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF,
            sublinear_tf=SUBLINEAR_TF,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    # ── Fit + Transform ───────────────────────────────────────────────────────
    def fit_transform(
        self,
        texts: List[str],
        labels: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Fitting pipeline on {len(texts):,} samples …")
        cleaned = self.cleaner.batch(texts)
        X = self.vectorizer.fit_transform(cleaned)
        y = self.label_encoder.fit_transform(labels)
        self.is_fitted = True
        logger.info(f"Vocabulary size : {len(self.vectorizer.vocabulary_):,}")
        logger.info(f"Num classes     : {len(self.label_encoder.classes_)}")
        return X, y

    # ── Transform only ────────────────────────────────────────────────────────
    def transform(self, texts: List[str]) -> np.ndarray:
        self._check_fitted()
        return self.vectorizer.transform(self.cleaner.batch(texts))

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        self._check_fitted()
        return self.label_encoder.transform(labels)

    def decode_labels(self, encoded: np.ndarray) -> List[str]:
        self._check_fitted()
        return list(self.label_encoder.inverse_transform(encoded))

    @property
    def classes(self) -> List[str]:
        return list(self.label_encoder.classes_)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self):
        joblib.dump(self.vectorizer,    VECTORIZER_PATH)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        logger.info(f"Pipeline saved → {MODEL_DIR_REF()}")

    @classmethod
    def load(cls) -> "PreprocessingPipeline":
        obj = cls()
        obj.vectorizer    = joblib.load(VECTORIZER_PATH)
        obj.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        obj.is_fitted     = True
        logger.info("Pipeline loaded from disk.")
        return obj

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform().")


def MODEL_DIR_REF():
    from config import MODEL_DIR
    return MODEL_DIR


if __name__ == "__main__":
    p = TextPreprocessor()
    sample = "What's my current account balance right now?"
    print(f"Original : {sample}")
    print(f"Processed: {p(sample)}")
