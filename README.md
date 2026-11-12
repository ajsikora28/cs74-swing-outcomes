# Predicting MLB Swing Results Using Pitch Data

AJ Sikora, Bryce Thomas — CS 74 Final Project

## Problem
Predict the probability distribution of a swing result (no swing, swing and miss, foul,
hard-hit ball ≥95 mph, non-hard-hit ball <95 mph) from a pitch's physical characteristics,
and test whether adding game context (count, outs, baserunners, handedness) improves prediction.

## Data
2026 MLB regular season Statcast data, pulled with `pybaseball`.
- Train: March–July
- Validation: August
- Test: September

Raw data is not stored in this repo (too large). Processed splits are in `data/processed/`.

## Setup
```
pip install -r requirements.txt
```

## Reproducing the results
1. `python src/fetch_data.py` — downloads raw Statcast data to `data/raw/`
2. `python src/preprocess.py` — builds labels and writes train/val/test to `data/processed/`
3. `python src/baseline.py`
4. `python src/train_logreg.py`, `python src/train_rf.py`, `python src/train_hgb.py`

Metrics are written to `results/metrics/`, figures to `results/figures/`.

## Repository layout
- `src/features.py` — feature lists shared by all models
- `src/evaluate.py` — metrics shared by all models
- `notebooks/` — exploration only; final results come from `src/`

## Generative AI usage
See `AI_USAGE.md`.
