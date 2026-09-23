# LSTM Forecasting for Mauna Loa CO2

[![CI](https://github.com/devissaputra/lstm_time_series/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/lstm_time_series/actions/workflows/ci.yml)

![Project overview](assets/01_cover.svg)

A leakage-aware sequence-forecasting experiment that asks a harder question than “can an LSTM fit a time series?”:

> **Can the LSTM beat simple forecasting baselines on a chronological hold-out set?**

## Data and split

The project uses the Mauna Loa atmospheric CO2 series distributed with statsmodels.

- weekly resampling with interpolation
- 24 previous weeks → next-week target
- 2,260 windows
- first 80% for training, final 20% for testing
- normalization fitted only on the training period

No future observations are used to estimate the training mean or standard deviation.

## Models

![Forecasting pipeline](assets/02_data_pipeline.svg)

Three approaches are compared:

1. **Persistence:** predict the next value as the last observed value.
2. **Ridge autoregression:** fit a linear model to the 24-lag window.
3. **LSTM:** 32 hidden units followed by a linear head.

The LSTM uses Adam, learning rate 0.005, batch size 64, and 20 epochs.

## Recorded results

![Sequence model](assets/03_data_or_model.svg)

| Model | RMSE ↓ | MAE ↓ |
|---|---:|---:|
| Persistence | 0.5135 | 0.4042 |
| **Ridge** | **0.4641** | **0.3546** |
| LSTM | 1.0700 | 0.8540 |

![Chronological evaluation](assets/04_evaluation_or_results.svg)

The neural model does **not** win. Ridge is strongest on this setup, and even persistence beats the LSTM. That is a useful result: recurrent complexity is not automatically valuable when a smooth, highly autocorrelated series can be forecast well from recent lags.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

Generated outputs are written under `results/`.

## Test

```bash
pip install pytest
pytest
```

CI runs model-shape, preprocessing, baseline, and one-epoch smoke tests. The full 20-epoch experiment is not retrained on every commit.

## Engineering details

- chronological split, never random time-series splitting
- train-only normalization
- persistence and Ridge baselines
- deterministic PyTorch seed and DataLoader generator
- import-safe experiment module
- behavioural tests and GitHub Actions
- generated plots separated from curated portfolio graphics

## Limits

This is one historical series and one forecast horizon. A stronger study would add seasonal/ARIMA baselines, rolling-origin evaluation, repeated initialization, uncertainty intervals, longer horizons, and hyperparameter selection on a validation period.
