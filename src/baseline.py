## Baseline model using probabilities of swing results given pitch type, handedness, and in/out of zone

import pandas as pd
from evaluate import evaluate_model

train_x = pd.read_csv("data/processed/train_x.csv")
train_y = pd.read_csv("data/processed/train_y.csv")["swing_result"]
validation_x = pd.read_csv("data/processed/validation_x.csv")
validation_y = pd.read_csv("data/processed/validation_y.csv")["swing_result"]

train = pd.concat([train_x, train_y], axis=1)

probs = train.groupby(["pitch_type","in_zone","same_hand"],as_index=False).agg(
    no_swing = ("swing_result", lambda x: (x == "no_swing").mean()),
    swing_and_miss = ("swing_result", lambda x: (x == "swing_and_miss").mean()),
    foul = ("swing_result", lambda x: (x == "foul").mean()),
    hard_contact = ("swing_result", lambda x: (x == "hard_contact").mean()),
    weak_contact = ("swing_result", lambda x: (x == "weak_contact").mean())
)

# Find any rare pitch types
pitch_type_counts = train["pitch_type"].value_counts()
all_pitch_types = pitch_type_counts.index.to_list()
rare_pitch_types = pitch_type_counts[pitch_type_counts < 100].index.to_list()

# Create an overall probs df for rare pitch types
overall_probs = train.groupby(["in_zone","same_hand"],as_index=False).agg(
    no_swing = ("swing_result", lambda x: (x == "no_swing").mean()),
    swing_and_miss = ("swing_result", lambda x: (x == "swing_and_miss").mean()),
    foul = ("swing_result", lambda x: (x == "foul").mean()),
    hard_contact = ("swing_result", lambda x: (x == "hard_contact").mean()),
    weak_contact = ("swing_result", lambda x: (x == "weak_contact").mean())
)
overall_probs["pitch_type"] = "rare_pitch"

all_probs = pd.concat([probs, overall_probs], axis=0)

classes = ["no_swing", "swing_and_miss", "foul", "hard_contact", "weak_contact"]

# data must contain columns pitch_type, in_zone, and same_hand
def baseline_probs(data_x):
    data = data_x.copy()
    predicted_probs = pd.DataFrame()
    data.loc[(data["pitch_type"].isin(rare_pitch_types) | ~data["pitch_type"].isin(all_pitch_types)), "pitch_type"] = "rare_pitch"
    predicted = data.merge(all_probs, how="left", on=["pitch_type", "in_zone", "same_hand"])[classes]
    # Claude Opus 5.5 generated the following lines to handle unseen combinations and
    # eliminate zero probabilities/renormalize
    fallback = data.merge(overall_probs, how="left", on=["in_zone", "same_hand"])[classes]
    predicted = predicted.fillna(fallback)
    predicted = predicted.clip(lower=1e-4)
    predicted = predicted.div(predicted.sum(axis=1), axis=0)
    return predicted

# Make sure this only runs when this file is run
if __name__ == "main":
    logloss_v, brier_v, accuracy_v = evaluate_model(validation_y, baseline_probs(validation_x))
    logloss_t, brier_t, accuracy_t = evaluate_model(train_y, baseline_probs(train_x))

    results = []
    results.append({"log_loss": logloss_v, "brier": brier_v, "accuracy": accuracy_v})

    print("Train log loss:", logloss_t)
    print("Train Brier score:", brier_t)
    print("Train Accuracy:", accuracy_t)
    print("Validation log loss:", logloss_v)
    print("Validation Brier score:", brier_v)
    print("Validation Accuracy:", accuracy_v)

    pd.DataFrame(results).to_csv("results/metrics/baseline_metrics.csv", index=False)
