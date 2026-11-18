from features import PITCH_FEATURES, PITCH_CATEGORICAL_FEATURES, CONTEXT_FEATURES, CONTEXT_CATEGORICAL_FEATURES
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import pandas as pd

train_x = pd.read_csv("data/processed/train_x.csv")
train_x_pitch = train_x[PITCH_FEATURES].copy()
train_x_context = train_x[CONTEXT_FEATURES].copy()
train_y = pd.read_csv("data/processed/train_y.csv")["swing_result"]
validation_x = pd.read_csv("data/processed/validation_x.csv")
validation_x_pitch = validation_x[PITCH_FEATURES].copy()
validation_x_context = validation_x[CONTEXT_FEATURES].copy()
validation_y = pd.read_csv("data/processed/validation_y.csv")["swing_result"]

def encode(train, val, categorical):
    train_encoded = pd.get_dummies(train, columns=categorical, dtype=float)
    val_encoded = pd.get_dummies(val, columns=categorical, dtype=float)
    val_encoded = val_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    return train_encoded, val_encoded

train_pitch_encoded, val_pitch_encoded = encode(train_x_pitch, validation_x_pitch, PITCH_CATEGORICAL_FEATURES)
train_context_encoded, val_context_encoded = encode(train_x_context, validation_x_context, CONTEXT_CATEGORICAL_FEATURES)

results = []
for d in [10, 12, 14]:
    for leaf in [10, 20]:
        rf_pitch = RandomForestClassifier(n_estimators=100, max_depth=d, min_samples_leaf=leaf, max_samples=0.7, random_state=42, n_jobs=-1).fit(train_pitch_encoded, train_y)
        rf_context = RandomForestClassifier(n_estimators=100, max_depth=d, min_samples_leaf=leaf, max_samples=0.7, random_state=42, n_jobs=-1).fit(train_context_encoded, train_y)

        pitch_loss = log_loss(validation_y, rf_pitch.predict_proba(val_pitch_encoded), labels=rf_pitch.classes_)
        context_loss = log_loss(validation_y, rf_context.predict_proba(val_context_encoded), labels=rf_context.classes_)

        results.append({"max_depth": d, "min_samples_leaf": leaf, "pitch_loss": pitch_loss, "context_loss": context_loss})
        print(f"depth={d}, leaf={leaf}: pitch {pitch_loss:.4f} | context {context_loss:.4f}", flush=True)

results = pd.DataFrame(results)
results.to_csv("results/metrics/rf_validation.csv", index=False)
print("\n", results.to_string(index=False))
