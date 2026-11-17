from features import BASE_FEATURES, PITCH_FEATURES, PITCH_CATEGORICAL_FEATURES, PITCH_GROUPS, CONTEXT_FEATURES, CONTEXT_CATEGORICAL_FEATURES, CONTEXT_GROUPS
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
import time

train_x = pd.read_csv("data/processed/train_x.csv")
train_y = pd.read_csv("data/processed/train_y.csv")["swing_result"]
validation_x = pd.read_csv("data/processed/validation_x.csv")
validation_y = pd.read_csv("data/processed/validation_y.csv")["swing_result"]

# Base run
results = []
train_x_base = train_x[BASE_FEATURES]
validation_x_base = validation_x[BASE_FEATURES]
base_cats = BASE_FEATURES

hgb_base = HistGradientBoostingClassifier(random_state=10, learning_rate=0.05, max_iter=1000, max_depth=5, max_leaf_nodes=None, early_stopping=False, categorical_features=base_cats).fit(train_x_base, train_y)
base_losses = []
for probs in hgb_base.staged_predict_proba(validation_x_base):
    base_losses.append(log_loss(validation_y, probs, labels=hgb_base.classes_))

base_index = np.argmin(base_losses)
base_loss = base_losses[base_index]

results = []
results.append({"type": "base", "group": "base", "max_depth": 5, "base_iters": base_index + 1, "base_loss": base_loss})
        
pd.DataFrame(results).to_csv("results/metrics/hgb_feature_analysis.csv", index=False)



# Loop with each feature group
start_time = time.perf_counter()
for cat in list(PITCH_GROUPS) + list(CONTEXT_GROUPS):
    if cat in PITCH_GROUPS:
        leave_one_out_features = []
        for feature in PITCH_FEATURES:
            if feature not in PITCH_GROUPS[cat]:
                leave_one_out_features.append(feature)
        one_only_features = BASE_FEATURES + PITCH_GROUPS[cat]
    else:
        leave_one_out_features = []
        for feature in CONTEXT_FEATURES:
            if feature not in CONTEXT_GROUPS[cat]:
                leave_one_out_features.append(feature)
        one_only_features = PITCH_FEATURES + CONTEXT_GROUPS[cat]

    train_x_leave_one_out = train_x[leave_one_out_features]
    leave_one_out_cats = [c for c in CONTEXT_CATEGORICAL_FEATURES if c in leave_one_out_features]
    train_x_one_only = train_x[one_only_features]
    one_only_cats = [c for c in CONTEXT_CATEGORICAL_FEATURES if c in one_only_features]
    validation_x_leave_one_out = validation_x[leave_one_out_features]
    validation_x_one_only = validation_x[one_only_features]

    hgb_leave_one_out = HistGradientBoostingClassifier(random_state=10, learning_rate=0.05, max_iter=1000, max_depth=5, max_leaf_nodes=None, early_stopping=False, categorical_features=leave_one_out_cats).fit(train_x_leave_one_out, train_y)
    hgb_one_only = HistGradientBoostingClassifier(random_state=10, learning_rate=0.05, max_iter=1000, max_depth=5, max_leaf_nodes=None, early_stopping=False, categorical_features=one_only_cats).fit(train_x_one_only, train_y)

    leave_one_out_losses = []
    for probs in hgb_leave_one_out.staged_predict_proba(validation_x_leave_one_out):
        leave_one_out_losses.append(log_loss(validation_y, probs, labels=hgb_leave_one_out.classes_))

    one_only_losses = []
    for probs in hgb_one_only.staged_predict_proba(validation_x_one_only):
        one_only_losses.append(log_loss(validation_y, probs, labels=hgb_one_only.classes_))

    leave_one_out_index = np.argmin(leave_one_out_losses)
    leave_one_out_loss = leave_one_out_losses[leave_one_out_index]
    one_only_index = np.argmin(one_only_losses)
    one_only_loss = one_only_losses[one_only_index]

    if cat in PITCH_GROUPS:
        pc = "pitch"
    else:
        pc = "context"

    results.append({"type": pc, "group": cat, "max_depth": 5, "leave_one_out_iters": leave_one_out_index + 1, "leave_one_out_loss": leave_one_out_loss,
                    "one_only_iters": one_only_index + 1, "one_only_loss": one_only_loss})
        
    pd.DataFrame(results).to_csv("results/metrics/hgb_feature_analysis.csv", index=False)

    print(f"[{len(results) - 1}/{len(PITCH_GROUPS) + len(CONTEXT_GROUPS)}] {pc} '{cat}' | "
          f"leave one out {leave_one_out_loss:.4f} ({leave_one_out_index + 1} trees) | "
          f"one only {one_only_loss:.4f} ({one_only_index + 1} trees) | "
          f"{(time.perf_counter() - start_time) / 60:.0f} min elapsed", flush=True)

results = pd.DataFrame(results)