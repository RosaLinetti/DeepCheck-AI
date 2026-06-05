"""
Lightweight classifier for plagiarism detection.

Uses a Logistic Regression model trained on simple features like
cosine similarity, length ratio, and lexical overlap.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# Training Data
# Each row: [cosine_score, length_ratio, lexical_overlap]
# Labels:   0 = original, 1 = suspicious, 2 = plagiarised

TRAINING_FEATURES = np.array([
    # original
    [0.10, 0.90, 0.00],
    [0.15, 0.80, 0.01],
    [0.20, 0.70, 0.02],
    [0.25, 0.60, 0.03],
    [0.30, 0.95, 0.05],
    [0.35, 0.85, 0.04],

    # summary / synthesis examples
    [0.55, 0.20, 0.05],
    [0.60, 0.25, 0.08],
    [0.65, 0.18, 0.06],
    [0.50, 0.30, 0.09],
    [0.58, 0.28, 0.07],

    # suspicious
    [0.50, 0.80, 0.10],
    [0.55, 0.75, 0.12],
    [0.60, 0.70, 0.15],
    [0.62, 0.85, 0.18],
    [0.65, 0.90, 0.20],
    [0.68, 0.78, 0.22],

    # plagiarised
    [0.80, 0.95, 0.55],
    [0.85, 0.90, 0.60],
    [0.90, 0.88, 0.65],
    [0.92, 0.96, 0.70],
    [0.95, 0.99, 0.80],
    [0.78, 0.92, 0.50],
])

TRAINING_LABELS = np.array([
    0, 0, 0, 0, 0, 0,    # original
    0, 0, 0, 0, 0,       # summary / synthesis examples
    1, 1, 1, 1, 1, 1,    # suspicious
    2, 2, 2, 2, 2, 2,    # plagiarised
])

LABEL_MAP = {0: "original", 1: "suspicious", 2: "plagiarised"}


# Model

class PlagiarismClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = LogisticRegression(max_iter=1000, random_state=42)
        self._train()

    def _train(self):
        X = self.scaler.fit_transform(TRAINING_FEATURES)
        self.model.fit(X, TRAINING_LABELS)

    def predict(
        self,
        cosine_score: float,
        length_ratio: float,
        lexical_overlap: float,
    ) -> Tuple[str, float]:
        """
        Predicts verdict and confidence for a single chunk comparison.

        Returns:
            (verdict_label, confidence) e.g. ("suspicious", 0.81)
        """
        # ── FIXED: Summary Guardrail Override ────────────────────────────────
        # If vocabulary overlap and length match are both low, it's an original 
        # summary. Catch it cleanly before the model can make a false prediction.
        if lexical_overlap < 0.12 and length_ratio < 0.35:
            return "original", float(round(1.0 - lexical_overlap, 4))

        features = np.array([[cosine_score, length_ratio, lexical_overlap]])
        features_scaled = self.scaler.transform(features)

        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = float(round(float(np.max(probabilities)), 4))

        return LABEL_MAP[prediction], confidence


# Module-level singleton — trained once on startup
_classifier = PlagiarismClassifier()


def classify_chunk(
    cosine_score: float,
    length_ratio: float,
    lexical_overlap: float,
) -> Tuple[str, float]:
    """Public interface for the classifier."""
    return _classifier.predict(cosine_score, length_ratio, lexical_overlap)