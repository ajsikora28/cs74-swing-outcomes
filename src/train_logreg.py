## LogisticRegression
from features import PITCH_FEATURES, PITCH_CATEGORICAL_FEATURES, CONTEXT_FEATURES, CONTEXT_CATEGORICAL_FEATURES
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


# Scale numeric features (fit on train only) and one-hot encode categorical features
def encode(train, val, categorical):
    numeric = [c for c in train.columns if c not in categorical]
    scaler = StandardScaler()
    train[numeric] = scaler.fit_transform(train[numeric])
    val[numeric] = scaler.transform(val[numeric])

    train_encoded = pd.get_dummies(train, columns=categorical, dtype=float)
    val_encoded = pd.get_dummies(val, columns=categorical, dtype=float)
    val_encoded = val_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    return train_encoded, val_encoded

train_pitch_encoded, val_pitch_encoded = encode(train_x_pitch, validation_x_pitch, PITCH_CATEGORICAL_FEATURES)
train_context_encoded, val_context_encoded = encode(train_x_context, validation_x_context, CONTEXT_CATEGORICAL_FEATURES)


# Tune penalty and C for each model
results = []
for penalty in ["l1", "l2"]:
    for c in [0.1, 1, 10]:
        solver = "saga" if penalty == "l1" else "lbfgs"
        logreg_pitch = LogisticRegression(penalty=penalty, C=c, solver=solver, max_iter=1000, tol=1e-3, random_state=42).fit(train_pitch_encoded, train_y)
        logreg_context = LogisticRegression(penalty=penalty, C=c, solver=solver, max_iter=1000, tol=1e-3, random_state=42).fit(train_context_encoded, train_y)

        pitch_loss = log_loss(validation_y, logreg_pitch.predict_proba(val_pitch_encoded), labels=logreg_pitch.classes_)
        context_loss = log_loss(validation_y, logreg_context.predict_proba(val_context_encoded), labels=logreg_context.classes_)

        results.append({"penalty": penalty, "C": c, "pitch_loss": pitch_loss, "context_loss": context_loss})
        print(f"{penalty.upper()}, C={c}: pitch {pitch_loss:.4f} | context {context_loss:.4f}", flush=True)

results = pd.DataFrame(results)
results.to_csv("results/metrics/logreg_validation.csv", index=False)
print("\n", results.to_string(index=False))
