import pandas as pd
import numpy as np

month_dates = [['2026-03-01', '2026-03-31'],
               ['2026-04-01', '2026-04-30'],
               ['2026-05-01', '2026-05-31'],
               ['2026-06-01', '2026-06-30'],
               ['2026-07-01', '2026-07-31'],
               ['2026-08-01', '2026-08-31'],
               ['2026-09-01', '2026-10-31']]

data = pd.DataFrame()
for start_dt, end_dt in month_dates:
    data = pd.concat([data, pd.read_csv(f"data/raw/statcast_{start_dt[0:4]}_{start_dt[5:7]}.csv")], ignore_index=True)

# Filter out rows with missing descriptions, pitchouts, automatic balls, bunts, and balls in play with no launch speed
data = data[~(data["description"].isna()) & 
            ~(data["description"].str.contains("pitchout")) & 
            ~(data["description"].str.contains("automatic")) &
            ~(data["description"].str.contains("bunt")) &
            ~((data["description"].str.contains("hit_into_play")) & (data["launch_speed"].isna()))]

conditions = [
    (data["description"].str.contains("ball")) | (data["description"].str.contains("called_strike")) | (data["description"].str.contains("hit_by_pitch")),
    (data["description"].str.contains("swinging_strike")) | (data["description"].str.contains("foul_tip")),
    (data["description"].str.contains("foul")),
    (~data["launch_speed"].isna()) & (data["launch_speed"] >= 95),
    (~data["launch_speed"].isna()) & (data["launch_speed"] < 95)
]

values = ["no swing", "swing and miss", "foul", "hard contact", "weak contact"]

data["swing_result"] = np.select(conditions, values, default="unknown")

data = data[["game_date","pitch_type","release_speed","release_spin_rate","spin_axis","release_extension","release_pos_x","release_pos_z",
             "pfx_x","pfx_z","plate_x","plate_z","on_3b","on_2b","on_1b","outs_when_up","balls","strikes","bat_score_diff","inning","stand","p_throws","swing_result"]]

bases = ["on_3b","on_2b","on_1b"]
for base in bases:
    data[base] = data[base].notna().astype(int)

# Split into train, validation, and test sets based on game month
train_data = data[data["game_date"].str[5:7].isin(["03","04", "05", "06", "07"])]
validation_data = data[data["game_date"].str[5:7] == "08"]
test_data = data[data["game_date"].str[5:7].isin(["09", "10"])]

train_x = train_data.drop(columns=["swing_result"])
train_y = train_data["swing_result"]
validation_x = validation_data.drop(columns=["swing_result"])
validation_y = validation_data["swing_result"]
test_x = test_data.drop(columns=["swing_result"])
test_y = test_data["swing_result"]

train_x.to_csv("data/processed/train_x.csv", index=False)
train_y.to_csv("data/processed/train_y.csv", index=False)
validation_x.to_csv("data/processed/validation_x.csv", index=False)
validation_y.to_csv("data/processed/validation_y.csv", index=False)
test_x.to_csv("data/processed/test_x.csv", index=False)
test_y.to_csv("data/processed/test_y.csv", index=False)
