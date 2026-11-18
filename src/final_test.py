import pandas as pd
from features import PITCH_FEATURES, PITCH_CATEGORICAL_FEATURES, CONTEXT_FEATURES, CONTEXT_CATEGORICAL_FEATURES
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate_model
from baseline import create_baseline_table, baseline_probs

# Read in all of the data, combine train and validation
train_x = pd.read_csv("data/processed/train_x.csv")
validation_x = pd.read_csv("data/processed/validation_x.csv")
train_y = pd.read_csv("data/processed/train_y.csv")["swing_result"]
validation_y = pd.read_csv("data/processed/validation_y.csv")["swing_result"]
full_train_x = pd.concat([train_x, validation_x], ignore_index=True)
full_train_y = pd.concat([train_y, validation_y], ignore_index=True)
train_x_pitch = full_train_x[PITCH_FEATURES].copy()
train_x_context = full_train_x[CONTEXT_FEATURES].copy()

test_x = pd.read_csv("data/processed/test_x.csv")
test_x_pitch = test_x[PITCH_FEATURES].copy()
test_x_context = test_x[CONTEXT_FEATURES].copy()
test_y = pd.read_csv("data/processed/test_y.csv")["swing_result"]

# Read in all of the model settings
hgb = pd.read_csv("results/metrics/hgb_settings.csv")[["features","max_depth","learning_rate","max_iter"]]
hgb_pitch = hgb[hgb["features"] == "pitch"].iloc[0]
hgb_context = hgb[hgb["features"] == "context"].iloc[0]
logreg = pd.read_csv("results/metrics/logreg_settings.csv")[["features","penalty","C"]]
logreg_pitch = logreg[logreg["features"] == "pitch"].iloc[0]
logreg_context = logreg[logreg["features"] == "context"].iloc[0]
rf = pd.read_csv("results/metrics/rf_settings.csv")[["features","max_depth","min_samples_leaf","n_estimators"]]
rf_pitch = rf[rf["features"] == "pitch"].iloc[0]
rf_context = rf[rf["features"] == "context"].iloc[0]

# Scale numeric features for logreg (fit on train only) and one-hot encode categorical features
# Modified from code written by Claude Opus 5.5
def encode(model, train, test, categorical):
    if model == "logreg":
        numeric = [c for c in train.columns if c not in categorical]
        scaler = StandardScaler()
        train[numeric] = scaler.fit_transform(train[numeric])
        test[numeric] = scaler.transform(test[numeric])

    train_encoded = pd.get_dummies(train, columns=categorical, dtype=float)
    test_encoded = pd.get_dummies(test, columns=categorical, dtype=float)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    return train_encoded, test_encoded

logreg_train_pitch_encoded, logreg_test_pitch_encoded = encode("logreg", train_x_pitch.copy(), test_x_pitch.copy(), PITCH_CATEGORICAL_FEATURES)
logreg_train_context_encoded, logreg_test_context_encoded = encode("logreg", train_x_context.copy(), test_x_context.copy(), CONTEXT_CATEGORICAL_FEATURES)
rf_train_pitch_encoded, rf_test_pitch_encoded = encode("rf", train_x_pitch.copy(), test_x_pitch.copy(), PITCH_CATEGORICAL_FEATURES)
rf_train_context_encoded, rf_test_context_encoded = encode("rf", train_x_context.copy(), test_x_context.copy(), CONTEXT_CATEGORICAL_FEATURES)


# Train the final models
print("Fitting baseline")
all_probs, overall_probs, rare_pitch_types, all_pitch_types = create_baseline_table(pd.concat([full_train_x, full_train_y], axis=1))

print("Fitting HGB pitch")
final_hgb_pitch = HistGradientBoostingClassifier(random_state=10,
                                                 learning_rate=hgb_pitch["learning_rate"],
                                                 max_iter=int(hgb_pitch["max_iter"]),
                                                 max_depth=int(hgb_pitch["max_depth"]),
                                                 max_leaf_nodes=None,
                                                 early_stopping=False,
                                                 categorical_features=PITCH_CATEGORICAL_FEATURES).fit(train_x_pitch, full_train_y)
print("Fitting HGB context")
final_hgb_context = HistGradientBoostingClassifier(random_state=10,
                                                   learning_rate=hgb_context["learning_rate"],
                                                   max_iter=int(hgb_context["max_iter"]),
                                                   max_depth=int(hgb_context["max_depth"]),
                                                   max_leaf_nodes=None,
                                                   early_stopping=False,
                                                   categorical_features=CONTEXT_CATEGORICAL_FEATURES).fit(train_x_context, full_train_y)

print("Fitting logreg pitch")
final_logreg_pitch = LogisticRegression(l1_ratio=1 if logreg_pitch["penalty"] == "L1" else 0,
                                        C=logreg_pitch["C"],
                                        solver="saga" if logreg_pitch["penalty"] == "L1" else "lbfgs",
                                        max_iter=1000,
                                        random_state=42).fit(logreg_train_pitch_encoded, full_train_y)
print("Fitting logreg context")
final_logreg_context = LogisticRegression(l1_ratio=1 if logreg_context["penalty"] == "L1" else 0,
                                        C=logreg_context["C"],
                                        solver="saga" if logreg_context["penalty"] == "L1" else "lbfgs",
                                        max_iter=1000,
                                        random_state=42).fit(logreg_train_context_encoded, full_train_y)

print("Fitting RF pitch")
final_rf_pitch = RandomForestClassifier(n_estimators=int(rf_pitch["n_estimators"]),
                                        max_depth=int(rf_pitch["max_depth"]),
                                        min_samples_leaf=int(rf_pitch["min_samples_leaf"]),
                                        max_samples=0.7,
                                        random_state=42,
                                        n_jobs=-1).fit(rf_train_pitch_encoded, full_train_y)
print("Fitting RF context")
final_rf_context = RandomForestClassifier(n_estimators=int(rf_context["n_estimators"]),
                                          max_depth=int(rf_context["max_depth"]),
                                          min_samples_leaf=int(rf_context["min_samples_leaf"]),
                                          max_samples=0.7,
                                          random_state=42,
                                          n_jobs=-1).fit(rf_train_context_encoded, full_train_y)

# Get final probabilities
final_baseline_probs = baseline_probs(test_x, all_probs, overall_probs, rare_pitch_types, all_pitch_types)
final_baseline_probs.to_csv("results/predictions/baseline_test_probs.csv", index=False)
final_hgb_pitch_probs = pd.DataFrame(final_hgb_pitch.predict_proba(test_x_pitch),columns=final_hgb_pitch.classes_)
final_hgb_pitch_probs.to_csv("results/predictions/hgb_pitch_test_probs.csv", index=False)
final_hgb_context_probs = pd.DataFrame(final_hgb_context.predict_proba(test_x_context),columns=final_hgb_context.classes_)
final_hgb_context_probs.to_csv("results/predictions/hgb_context_test_probs.csv", index=False)
final_logreg_pitch_probs = pd.DataFrame(final_logreg_pitch.predict_proba(logreg_test_pitch_encoded),columns=final_logreg_pitch.classes_)
final_logreg_pitch_probs.to_csv("results/predictions/logreg_pitch_test_probs.csv", index=False)
final_logreg_context_probs = pd.DataFrame(final_logreg_context.predict_proba(logreg_test_context_encoded),columns=final_logreg_context.classes_)
final_logreg_context_probs.to_csv("results/predictions/logreg_context_test_probs.csv", index=False)
final_rf_pitch_probs = pd.DataFrame(final_rf_pitch.predict_proba(rf_test_pitch_encoded),columns=final_rf_pitch.classes_)
final_rf_pitch_probs.to_csv("results/predictions/rf_pitch_test_probs.csv", index=False)
final_rf_context_probs = pd.DataFrame(final_rf_context.predict_proba(rf_test_context_encoded),columns=final_rf_context.classes_)
final_rf_context_probs.to_csv("results/predictions/rf_context_test_probs.csv", index=False)

# Evaluate all of the models
hgb_pitch_logloss, hgb_pitch_brier, hgb_pitch_accuracy = evaluate_model(test_y, final_hgb_pitch_probs)
hgb_context_logloss, hgb_context_brier, hgb_context_accuracy = evaluate_model(test_y, final_hgb_context_probs)
logreg_pitch_logloss, logreg_pitch_brier, logreg_pitch_accuracy = evaluate_model(test_y, final_logreg_pitch_probs)
logreg_context_logloss, logreg_context_brier, logreg_context_accuracy = evaluate_model(test_y, final_logreg_context_probs)
rf_pitch_logloss, rf_pitch_brier, rf_pitch_accuracy = evaluate_model(test_y, final_rf_pitch_probs)
rf_context_logloss, rf_context_brier, rf_context_accuracy = evaluate_model(test_y, final_rf_context_probs)
baseline_logloss, baseline_brier, baseline_accuracy = evaluate_model(test_y, final_baseline_probs)


results = pd.DataFrame([{"model":"baseline", "features":"pitch type, handedness, zone",
                         "log_loss":baseline_logloss, "brier":baseline_brier, "accuracy":baseline_accuracy},
                        {"model":"hgb", "features":"pitch",
                         "log_loss":hgb_pitch_logloss, "brier":hgb_pitch_brier, "accuracy":hgb_pitch_accuracy},
                        {"model":"hgb", "features":"context",
                         "log_loss":hgb_context_logloss, "brier":hgb_context_brier, "accuracy":hgb_context_accuracy},
                        {"model":"logreg", "features":"pitch",
                         "log_loss":logreg_pitch_logloss, "brier":logreg_pitch_brier, "accuracy":logreg_pitch_accuracy},
                        {"model":"logreg", "features":"context",
                         "log_loss":logreg_context_logloss, "brier":logreg_context_brier, "accuracy":logreg_context_accuracy},
                        {"model":"rf", "features":"pitch",
                         "log_loss":rf_pitch_logloss, "brier":rf_pitch_brier, "accuracy":rf_pitch_accuracy},
                        {"model":"rf", "features":"context",
                         "log_loss":rf_context_logloss, "brier":rf_context_brier, "accuracy":rf_context_accuracy}
                       ])

results.to_csv("results/metrics/final_comparison.csv", index=False)