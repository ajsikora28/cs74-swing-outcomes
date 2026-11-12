import pybaseball as pyb
import numpy as np

pyb.cache.enable()
data = pyb.statcast(start_dt='2026-05-01', end_dt='2026-05-31')

# file = data.to_csv('statcast_may_2026.csv', index=False)

data = data[~(data["description"].isna()) & 
            ~(data["description"].str.contains("pitchout")) & 
            ~(data["description"].str.contains("automatic")) &
            ~((data["description"].str.contains("hit_into_play")) & (data["launch_speed"].isna()))]

conditions = [
    (data["description"].str.contains("ball")) | (data["description"].str.contains("called_strike")) | (data["description"].str.contains("hit_by_pitch")),
    (data["description"].str.contains("foul")),
    (data["description"].str.contains("swinging_strike")) | (data["description"].str.contains("missed_bunt")),
    (~data["launch_speed"].isna()) & (data["launch_speed"] >= 95),
    (~data["launch_speed"].isna()) & (data["launch_speed"] < 95)
]

values = ["no swing", "foul", "swing and miss", "hard contact", "weak contact"]

data["swing_result"] = np.select(conditions, values, default="unknown")

counts = data["swing_result"].value_counts()
counts = (counts / counts.sum()) * 100
print(counts)
print("n = ", len(data))



