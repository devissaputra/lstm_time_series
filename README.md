# LSTM Forecasting for Mauna Loa CO₂

This experiment compares persistence, ridge autoregression, and a compact LSTM for next-week Mauna Loa CO₂ forecasting. Missing weeks now use past-only forward filling, and the corrected 20-epoch experiment has been rerun successfully in GitHub Actions. On the 452-week chronological holdout, ridge achieves RMSE 0.4639 ppm, persistence 0.5135 ppm, and the LSTM 1.0751 ppm. The recurrent model therefore does not justify its added complexity in this fixed-seed setup.

[![CI](https://github.com/devissaputra/lstm_time_series/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/lstm_time_series/actions/workflows/ci.yml)

## Read the evidence

- [Calculation guide](CALCULATIONS.md)
- [Working paper](paper/paper.md)
- [Data](DATA.md) and [reproduction instructions](REPRODUCIBILITY.md)
- [Successful corrected run](https://github.com/devissaputra/lstm_time_series/actions/runs/36235069744) and [run provenance](results/review_status.json)

![Study overview](assets/review_overview.svg)

![Calculation and corrected evidence](assets/review_calculations.svg)

## Research question

Can a compact recurrent model improve next-week forecasts over persistence and a linear autoregression when the evaluation respects observation time?

## Data and temporal boundaries

The source is the Mauna Loa CO₂ series bundled with statsmodels. Weekly observations are resampled and missing weeks are forward-filled using only a preceding observation. The earlier linear interpolation step could borrow a future measurement and has been removed. Older interpolation-based scores are superseded by the corrected run linked above.

Each input contains the previous 24 weekly values. The resulting 2,260 windows are split into 1,808 training and 452 test windows in time order. The mean and standard deviation are calculated only from observations available through the final training target.

MAE and RMSE are calculated after reversing training-fitted normalization and are reported in ppm. Later test predictions use earlier observed test values, so this is one-step rolling evaluation rather than a recursive forecast from one origin. One seed, fixed hyperparameters and one test era limit generalization.

## Comparators

Persistence repeats the last observed concentration. Ridge regression uses the same 24-lag input with alpha 1.0. The LSTM has 32 hidden units and a linear output layer; it trains for 20 epochs with Adam, learning rate 0.005 and batch size 64. These settings are fixed, not selected on the test set.

## Corrected results

| Model | RMSE (ppm) ↓ | MAE (ppm) ↓ |
|---|---:|---:|
| persistence | 0.5135 | 0.4042 |
| ridge | 0.4639 | 0.3538 |
| lstm | 1.0751 | 0.8599 |

Ridge has the smallest error under both metrics. The result is useful precisely because the more complex model does not win. It does not imply that all recurrent models underperform on atmospheric time series.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
python scripts/build_review_figures.py
python scripts/build_review_figures.py --check
```

The runner saves unrounded metrics and forecast plots. Install pytest and run `PYTHONPATH=. python -m pytest -q` to run the existing tests. A regression test supplies a missing week between values 1 and 100 and verifies that the loader fills it with 1, not an interpolated future-dependent value.

## Validation and limitations

The corrected full experiment and CI both succeeded in GitHub Actions. No repeated-seed confidence interval is claimed. This benchmark still needs additional forecast origins, independently chosen validation periods, seasonal/statistical baselines and sensitivity to missing-data policy before supporting a stronger comparative claim.
