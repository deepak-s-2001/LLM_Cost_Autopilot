import json
from pathlib import Path

import joblib
import pandas as pd

from app.classifier.features import extract_features

_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_MODEL_PATH = _ARTIFACT_DIR / "classifier.joblib"
_FEATURE_COLUMNS_PATH = _ARTIFACT_DIR / "feature_columns.json"

_model = None
_feature_columns = None


def _load() -> None:
    global _model, _feature_columns
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
        _feature_columns = json.loads(_FEATURE_COLUMNS_PATH.read_text())


def classify_prompt(prompt: str, context: str | None = None) -> tuple[int, float]:
    tier, confidence, _tier_probs, _features = classify_prompt_detailed(prompt, context)
    return tier, confidence


def classify_prompt_detailed(prompt: str, context: str | None = None) -> tuple[int, float, dict[str, float], dict]:
    _load()
    features = extract_features(prompt, context)
    row = pd.DataFrame([[features[col] for col in _feature_columns]], columns=_feature_columns)
    probs = _model.predict_proba(row)[0]
    classes = _model.classes_
    best_idx = probs.argmax()
    tier_probabilities = {str(int(cls)): float(p) for cls, p in zip(classes, probs)}
    return int(classes[best_idx]), float(probs[best_idx]), tier_probabilities, features
