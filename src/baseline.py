import pandas as pd

train_x = pd.read_csv("data/processed/train_x.csv")
train_y = pd.read_csv("data/processed/train_y.csv")

train = pd.concat([train_x, train_y], axis=1)

probs = train.groupby(["pitch_type","in_zone","same_hand"],as_index=False).agg(
    no_swing = ("swing_result", lambda x: (x == "no swing").mean()),
    swing_and_miss = ("swing_result", lambda x: (x == "swing and miss").mean()),
    foul = ("swing_result", lambda x: (x == "foul").mean()),
    hard_contact = ("swing_result", lambda x: (x == "hard contact").mean()),
    weak_contact = ("swing_result", lambda x: (x == "weak contact").mean())
)

# Find any rare pitch types
pitch_type_counts = train["pitch_type"].value_counts()
all_pitch_types = pitch_type_counts.index.to_list()
rare_pitch_types = pitch_type_counts[pitch_type_counts < 100].index.to_list()
print(rare_pitch_types)

# Create an overall probs df for rare pitch types
overall_probs = train.groupby(["in_zone","same_hand"],as_index=False).agg(
    no_swing = ("swing_result", lambda x: (x == "no swing").mean()),
    swing_and_miss = ("swing_result", lambda x: (x == "swing and miss").mean()),
    foul = ("swing_result", lambda x: (x == "foul").mean()),
    hard_contact = ("swing_result", lambda x: (x == "hard contact").mean()),
    weak_contact = ("swing_result", lambda x: (x == "weak contact").mean())
)
overall_probs["pitch_type"] = "rare_pitch"

all_probs = pd.concat([probs, overall_probs], axis=0)

results = ["no_swing", "swing_and_miss", "foul", "hard_contact", "weak_contact"]

# data must contain columns pitch_type, in_zone, and same_hand
def baseline_probs(data_x):
    data = data_x.copy()
    predicted_probs = pd.DataFrame()
    data.loc[(data["pitch_type"].isin(rare_pitch_types) | ~data["pitch_type"].isin(all_pitch_types)), "pitch_type"] = "rare_pitch"
    predicted_probs = data.merge(all_probs, how="left", on=["pitch_type","in_zone","same_hand"])
    return predicted_probs[results]
