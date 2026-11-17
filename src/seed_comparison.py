# Run the full context model with 2 other seeds to see how different they are
from features import BASE_FEATURES, PITCH_FEATURES, PITCH_CATEGORICAL_FEATURES, PITCH_GROUPS, CONTEXT_FEATURES, CONTEXT_CATEGORICAL_FEATURES, CONTEXT_GROUPS
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

train_x = pd.read_csv("data/processed/train_x.csv")[CONTEXT_FEATURES]
train_y = pd.read_csv("data/processed/train_y.csv")["swing_result"]
validation_x = pd.read_csv("data/processed/validation_x.csv")[CONTEXT_FEATURES]
validation_y = pd.read_csv("data/processed/validation_y.csv")["swing_result"]

results_10 = pd.read_csv("results/metrics/hgb_tuning_results.csv")
row_match = results_10[(results_10["max_depth"] == 5) & (results_10["learning_rate"] == 0.05)]
loss_10 = row_match["best_context_loss"].iloc[0]
iters_10 = row_match["best_context_iters"].iloc[0]

hgb_20 = HistGradientBoostingClassifier(random_state=20, learning_rate=0.05, max_iter=1000, max_depth=5, max_leaf_nodes=None, early_stopping=False, categorical_features=CONTEXT_CATEGORICAL_FEATURES).fit(train_x, train_y)
hgb_30 = HistGradientBoostingClassifier(random_state=30, learning_rate=0.05, max_iter=1000, max_depth=5, max_leaf_nodes=None, early_stopping=False, categorical_features=CONTEXT_CATEGORICAL_FEATURES).fit(train_x, train_y)


losses_20 = []
for probs in hgb_20.staged_predict_proba(validation_x):
    losses_20.append(log_loss(validation_y, probs, labels=hgb_20.classes_))

losses_30 = []
for probs in hgb_30.staged_predict_proba(validation_x):
    losses_30.append(log_loss(validation_y, probs, labels=hgb_30.classes_))

index_30 = np.argmin(losses_30)
loss_30 = losses_30[index_30]

index_20 = np.argmin(losses_20)
loss_20 = losses_20[index_20]

results = []
results.append({"seed": 10, "loss": loss_10, "iters": iters_10})
results.append({"seed": 20, "loss": loss_20, "iters": index_20 + 1})
results.append({"seed": 30, "loss": loss_30, "iters": index_30 + 1})

results = pd.DataFrame(results)
print(results)
results.to_csv("results/metrics/other_seeds.csv", index=False)


