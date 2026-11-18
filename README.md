# Predicting MLB Swing Results Using Pitch Data

AJ Sikora, Bryce Thomas — CS 74 Final Project

## Problem
Predict the probability distribution of a swing result (no swing, swing and miss, foul,
hard-hit ball ≥95 mph, non-hard-hit ball <95 mph) from a pitch's physical characteristics,
and test whether adding game context (count, outs, baserunners, score, inning) improves that prediction. Also, understand which of those features are most helpful in improving on a baseline prediction that includes only handedness, in/out of zone, and pitch type.

## Data
2026 MLB regular season Statcast data, pulled with `pybaseball`.
- Train: March-July
- Validation: August
- Test: September

Raw data is not stored in this repo (too large). Processed train, validate, and test datasets are in `data/processed/`.

## Setup
This code was written with Python 3.12.3. Create a virtual environment and install the required packages by running
```
pip install -r requirements.txt
```
Run all scripts from the project root.

## Reproducing the results
1. `python src/fetch_data.py`: downloads raw 2026 Statcast data to `data/raw/`
2. `python src/preprocess.py`: removes unwanted rows, creates labels, and writes train/val/test to `data/processed/`
3. `python src/baseline.py`: creates the initial baseline model and saves results on validation to `results/metrics/baseline_metrics.csv`
4. `python src/train_logreg.py`, `python src/train_rf.py`, `python src/train_hgb.py`: must set RUN_TUNING to True to run the model tuning (warning: gradient boosting may take several hours). With RUN_TUNING = False, the script relies on the existing tuning results and only refits the final models. Tune the logistic regression, random forest, and gradient boosted models and write the results to their respective `results/metrics/{model}_tuning_results.csv` files. Write the final model settings for each pitch and context model to their respective `results/metrics/{model}_settings.csv` files. `train_rf.py` also saves results from comparing two values of `n_estimators` on each model to `rf_n_estimators.csv`. Tuning figures are written to `results/figures/logreg_tuning_plot.png`, `results/figures/rf_tuning_heatmap.png`, and `results/figures/hgb_tuning_heatmap.png`
5. `python src/validation_comparison.py`: saves the validation results for the best version of each model to `results/metrics/validation_comparison.csv`
6. `python src/seed_comparison.py`: tests the HGB model with other random seeds to understand how variable loss is based on the seed and saves results to `results/metrics/other_seeds.csv`
7. `python src/hgb_feature_analysis.py`: runs feature analysis on HGB, the best model, and saves results to `results/metrics/hgb_feature_analysis.csv`
8. `python src/feature_figures.py`: creates `results/metrics/hgb_feature_analysis_table.csv` and `results/figures/hgb_feature_analysis.png`
9. `python src/final_test.py`: runs the final models on the test data, saving all predictions to `results/predictions/` and results comparison to `results/metrics/final_comparison.csv`
10. `python src/final_figure.py`: creates the final model comparison plot and saves to `results/figures/improvement_over_baseline.png`

All metrics are written to `results/metrics/`, and figures are written to `results/figures/`. Final predictions are written to `results/predictions/`.

## Other file information
- `src/features.py`: feature lists shared by all models
- `src/evaluate.py`: stores a function used to evaluate the models on log loss, Brier score, and accuracy

## Generative AI usage
Claude Opus 5.5 was used to help outline and format this README file, though we wrote the main content ourselves. For other AI usage in this project, see comments throughout in addition to the Generative AI Usage section in the final report.
