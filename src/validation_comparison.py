import pandas as pd

hgb = pd.read_csv("results/metrics/hgb_settings.csv")[["model","features","val_log_loss","val_brier","val_accuracy"]]
logreg = pd.read_csv("results/metrics/logreg_settings.csv")[["model","features","val_log_loss","val_brier","val_accuracy"]]
rf = pd.read_csv("results/metrics/rf_settings.csv")[["model","features","val_log_loss","val_brier","val_accuracy"]]
baseline = pd.read_csv("results/metrics/baseline_metrics.csv")

# Match up the column names
baseline["model"] = "baseline"
baseline["features"] = "pitch type, handedness, zone"
baseline = baseline.rename(columns={"log_loss": "val_log_loss", "brier": "val_brier", "accuracy": "val_accuracy"})

final = pd.concat([hgb, logreg, rf, baseline], ignore_index=True)
final = final.sort_values(["val_log_loss"])

final.to_csv("results/metrics/validation_comparison.csv", index=False)