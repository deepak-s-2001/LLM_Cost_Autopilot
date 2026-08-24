import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.classifier.features import FEATURE_COLUMNS, extract_features
from app.logging.db import get_connection

DATA_PATH = Path(__file__).resolve().parent / "data" / "labeled_prompts.csv"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "classifier.joblib"
FEATURE_COLUMNS_PATH = ARTIFACT_DIR / "feature_columns.json"


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        extract_features(prompt, context if isinstance(context, str) else None)
        for prompt, context in zip(df["prompt"], df["context"])
    ]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def load_training_examples_from_db() -> tuple[pd.DataFrame, pd.Series]:
    conn = get_connection()
    rows = conn.execute("SELECT features_json, tier_label FROM training_examples").fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=int)
    records = [json.loads(row["features_json"]) for row in rows]
    X = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    y = pd.Series([row["tier_label"] for row in rows])
    return X, y


def build_dataset(include_training_examples: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    X, y = build_feature_frame(df), df["tier"]
    if include_training_examples:
        extra_X, extra_y = load_training_examples_from_db()
        if not extra_X.empty:
            X = pd.concat([X, extra_X], ignore_index=True)
            y = pd.concat([y, extra_y], ignore_index=True)
    return X, y


def train_and_save(X: pd.DataFrame, y: pd.Series) -> tuple[str, object, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    }

    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        fitted[name] = (model, acc)
        print(f"\n=== {name} ===")
        print(f"Held-out accuracy: {acc:.3f}")
        print(classification_report(y_test, preds))
        print("Confusion matrix (rows=true, cols=predicted), classes:", sorted(y.unique()))
        print(confusion_matrix(y_test, preds, labels=sorted(y.unique())))

    # Defaults to logistic regression over the higher-scoring random forest, since RF's ~100% held-out accuracy on this small templated dataset is template memorization, not real generalization (verified against fresh prompts).
    best_name, (best_model, best_acc) = "logistic_regression", fitted["logistic_regression"]
    print(f"\nSelected {best_name} (accuracy={best_acc:.3f}) as the saved classifier.")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(FEATURE_COLUMNS))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature columns to {FEATURE_COLUMNS_PATH}")
    return best_name, best_model, best_acc


def main() -> None:
    X, y = build_dataset(include_training_examples=False)
    train_and_save(X, y)


if __name__ == "__main__":
    main()
