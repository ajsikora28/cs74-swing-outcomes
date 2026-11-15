from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import PITCH_FEATURES

ROOT = Path.cwd()
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "metrics"


def load_data():
    X_train = pd.read_csv(DATA / "train_x.csv")
    y_train = pd.read_csv(DATA / "train_y.csv")["swing_result"]
    X_val = pd.read_csv(DATA / "validation_x.csv")
    y_val = pd.read_csv(DATA / "validation_y.csv")["swing_result"]
    return X_train, y_train, X_val, y_val


def make_preprocessor(X_train, features):
    categorical = X_train[features].select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical = [f for f in features if f not in categorical]

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])


def evaluate_model(name, y_true, probabilities, classes):
    predictions = classes[np.argmax(probabilities, axis=1)]

    one_hot = (
        np.asarray(y_true)[:, None] == classes[None, :]
    ).astype(float)

    brier = np.mean(
        np.sum((probabilities - one_hot) ** 2, axis=1)
    )

    return {
        "model": name,
        "features": "Pitch Only",
        "log_loss": log_loss(y_true, probabilities, labels=classes),
        "brier_score": brier,
        "accuracy": accuracy_score(y_true, predictions)
    }


def main():
    X_train, y_train, X_val, y_val = load_data()

    pitch_features = [
        f for f in PITCH_FEATURES
        if f not in ["stand", "p_throws", "same_hand"]
    ]

    preprocessor = make_preprocessor(X_train, pitch_features)
    classes = np.unique(y_train)

    frequencies = (
        y_train.value_counts(normalize=True)
        .reindex(classes, fill_value=0)
        .to_numpy()
    )

    baseline_probs = np.tile(frequencies, (len(y_val), 1))

    results = [
        evaluate_model(
            "Frequency Baseline",
            y_val,
            baseline_probs,
            classes
        )
    ]

    models = [
        ("Logistic Regression L2", LogisticRegression(
            penalty="l2",
            max_iter=1000,
            random_state=42
        )),
        ("Logistic Regression L1", LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=1000,
            random_state=42
        ))
    ]

    for name, classifier in models:
        model = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier)
        ])

        print(f"Training {name}...")

        model.fit(X_train[pitch_features], y_train)

        probabilities = model.predict_proba(
            X_val[pitch_features]
        )

        model_classes = model.named_steps[
            "classifier"
        ].classes_

        results.append(
            evaluate_model(
                name,
                y_val,
                probabilities,
                model_classes
            )
        )

    comparison = pd.DataFrame(results)

    RESULTS.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        RESULTS / "logreg_pitch_validation.csv",
        index=False
    )

    print("\nAugust Validation Results:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
