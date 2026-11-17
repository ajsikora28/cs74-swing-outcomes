from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

# Allow running from anywhere, using either `python src/file.py` or `python -m src.file`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import PITCH_FEATURES, CATEGORICAL_FEATURES
DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "results" / "metrics"


def load_data():
    
    X_train = pd.read_csv(DATA_DIR / "train_x.csv")
    y_train = pd.read_csv(DATA_DIR / "train_y.csv")["swing_result"]
    X_val = pd.read_csv(DATA_DIR / "validation_x.csv")
    y_val = pd.read_csv(DATA_DIR / "validation_y.csv")["swing_result"]
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("The feature and label files have mismatched row counts")
    if not set(y_val.unique()).issubset(set(y_train.unique())):
        raise ValueError("August contains a class absent from training")
    return X_train, y_train, X_val, y_val


def prepare_data(X_train, X_val, scale_numbers):
    
    from sklearn.preprocessing import StandardScaler

    features = list(PITCH_FEATURES)  # Includes stand, p_throws, same_hand.
    missing = [f for f in features if f not in X_train or f not in X_val]
    if missing:
        raise ValueError("Required columns are missing: " + str(missing))

    categorical = [f for f in CATEGORICAL_FEATURES if f in features]
    numeric = [f for f in features if f not in categorical]
    train = X_train[features].copy()
    val = X_val[features].copy()

    # Numeric imputation: the August data does not determine these medians.
    for c in numeric:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        val[c] = pd.to_numeric(val[c], errors="coerce")
    medians = train[numeric].median().fillna(0)
    train[numeric] = train[numeric].fillna(medians)
    val[numeric] = val[numeric].fillna(medians)

    for c in categorical:
        train[c] = train[c].fillna("missing").astype(str)
        val[c] = val[c].fillna("missing").astype(str)

    # Create dummy columns; match August columns to the TRAINING columns.
    train_encoded = pd.get_dummies(train, columns=categorical, dtype=float)
    val_encoded = pd.get_dummies(val, columns=categorical, dtype=float)
    val_encoded = val_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    if scale_numbers:
        scaler = StandardScaler()
        # Ensure float dtype before writing standardized values into integer columns.
        train_encoded[numeric] = train_encoded[numeric].astype(float)
        val_encoded[numeric] = val_encoded[numeric].astype(float)
        train_encoded[numeric] = scaler.fit_transform(train_encoded[numeric])
        val_encoded[numeric] = scaler.transform(val_encoded[numeric])

    return train_encoded, val_encoded


def evaluate(name, y_true, probabilities, classes, **extra):
    
    classes = np.asarray(classes)
    predicted = classes[np.argmax(probabilities, axis=1)]
    one_hot = (np.asarray(y_true)[:, None] == classes[None, :]).astype(float)
    return {
        "model": name,
        "features": "Pitch Only (incl. handedness)",
        **extra,
        "log_loss": log_loss(y_true, probabilities, labels=classes),
        "brier_score": np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)),
        "accuracy": accuracy_score(y_true, predicted),
    }

# Small, explainable tuning comparison; all models use the same training/val splits.
CONFIGURATIONS = [
    ("RF depth=10, leaf=20", 10, 20),
    ("RF depth=12, leaf=20", 12, 20),
    ("RF depth=14, leaf=10", 14, 10),
]


def main():
    X_train, y_train, X_val, y_val = load_data()
    X_train_ready, X_val_ready = prepare_data(X_train, X_val, scale_numbers=False)
    # sklearn's trees handle float32 efficiently and don't need standardization.
    X_train_ready = X_train_ready.astype(np.float32)
    X_val_ready = X_val_ready.astype(np.float32)
    print("Training rows:", len(X_train_ready), "Validation rows:", len(X_val_ready))
    print("Encoded features:", X_train_ready.shape[1])

    results = []
    for name, depth, leaf in CONFIGURATIONS:
        print("Training", name, flush=True)
        model = RandomForestClassifier(
            n_estimators=100, max_depth=depth, min_samples_leaf=leaf,
            max_samples=0.7, random_state=42, n_jobs=-1,
        )
        model.fit(X_train_ready, y_train)
        probabilities = model.predict_proba(X_val_ready)
        results.append(evaluate(name, y_val, probabilities, model.classes_,
                                max_depth=depth, min_samples_leaf=leaf))
        print(f"  Validation log loss: {results[-1]['log_loss']:.6f}", flush=True)

    table = pd.DataFrame(results)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "rf_pitch_validation.csv"
    table.to_csv(path, index=False)
    print("\nRANDOM FOREST AUGUST VALIDATION COMPARISON:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    best = table.sort_values("log_loss").iloc[0]
    print(f"\nBest RF configuration: {best['model']} (log loss {best['log_loss']:.6f})")
    print("Saved:", path)
    return table


if __name__ == "__main__":
    main()

