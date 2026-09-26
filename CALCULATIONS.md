# Calculation guide

## Question, inputs and protocol

This experiment compares persistence, ridge autoregression, and a compact LSTM for next-week Mauna Loa CO₂ forecasting. Missing weeks now use past-only forward filling, and the corrected 20-epoch experiment has been rerun successfully in GitHub Actions. On the 452-week chronological holdout, ridge achieves RMSE 0.4639 ppm, persistence 0.5135 ppm, and the LSTM 1.0751 ppm. The recurrent model therefore does not justify its added complexity in this fixed-seed setup.

The input is a 24-week window of past observed or past-only forward-filled concentrations. The 1,808 training and 452 test windows retain time order. Ridge and the LSTM receive identical lag inputs, and all reported errors return to the original ppm scale.

## Error calculations

For n test targets, observed values y and forecasts f:

- MAE = sum(abs(f - y)) / n. Each miss contributes in direct proportion to its magnitude.
- RMSE = sqrt(sum((f - y)^2) / n). Squaring penalizes larger misses more strongly.
- Normalization uses z = (value - training mean) / training SD. Forecasts are returned to ppm using f = predicted z × training SD + training mean before scoring.
- Persistence sets the next forecast to the most recent input value. Ridge estimates a regularized linear combination of the 24 inputs. The LSTM trains on normalized mean squared error; that training loss is not the reported test RMSE.

## Reproduced evidence

| Model | RMSE (ppm) ↓ | MAE (ppm) ↓ |
|---|---:|---:|
| persistence | 0.5135 | 0.4042 |
| ridge | 0.4639 | 0.3538 |
| lstm | 1.0751 | 0.8599 |

The unrounded values are in [results/metrics.json](results/metrics.json). [Run provenance](results/review_status.json) identifies the source commit, artifact and [successful workflow](https://github.com/devissaputra/lstm_time_series/actions/runs/36235069744). The old interpolation run is superseded.

## Interpretation boundaries

MAE and RMSE are calculated after reversing training-fitted normalization and are reported in ppm. Later test predictions use earlier observed test values, so this is one-step rolling evaluation rather than a recursive forecast from one origin. One seed, fixed hyperparameters and one test era limit generalization.

## Implementation map

- [Data loader and training-only normalization](src/run_experiment.py)
- [Causal missingness regression test](tests/test_causal_missingness.py)
- [Full experiment and baseline checks](tests/test_experiment.py)

## Figure regeneration

```bash
python scripts/build_review_figures.py
python scripts/build_review_figures.py --check
```

The generator reads the selected JSON paths in [docs/figure_spec.json](docs/figure_spec.json). It displays exact source values rounded for readability, with a source-file hash prefix for traceability.
