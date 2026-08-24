import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from app.classifier.train import MODEL_PATH, build_dataset, train_and_save


def main() -> None:
    X, y = build_dataset(include_training_examples=True)
    print(f"Dataset size (labeled_prompts.csv + accumulated training_examples): {len(X)} rows")

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if MODEL_PATH.exists():
        old_model = joblib.load(MODEL_PATH)
        old_acc = accuracy_score(y_test, old_model.predict(X_test))
        print(f"Accuracy of current saved classifier on this held-out split: {old_acc:.3f}")
    else:
        print("No existing classifier found, skipping before-accuracy comparison.")

    print("\nRetraining...")
    _name, _model, new_acc = train_and_save(X, y)
    print(f"\nAccuracy of retrained classifier: {new_acc:.3f}")


if __name__ == "__main__":
    main()
