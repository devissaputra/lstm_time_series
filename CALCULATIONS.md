# Calculation guide

## Question and evidence

Can a recurrent model improve weekly CO₂ forecasts?

Mauna Loa weekly CO₂ from statsmodels; 24-week input windows.

**Status:** CORRECTED PIPELINE | neural results require regeneration.

## Design

Past-only forward fill; training-only normalization; chronological holdout; persistence, ridge and LSTM.

## Calculation and interpretation

`RMSE = sqrt(mean((forecast-observed)^2)); MAE = mean(|forecast-observed|).`

The original results used linear interpolation before splitting. The corrected loader uses forward fill to avoid future borrowing. Historical scores are preserved for traceability but cannot substantiate the corrected protocol until a full rerun.

## Evidence table

Corrected-run status; historical scores are not reused. Full precision below is for traceability, not a claim of measurement precision.

| Quantity | Value | Unit / meaning | JSON path |
|---|---:|---|---|
| Corrected full run complete | False | boolean | `corrected_full_run_complete` |

Source: [results/review_status.json](results/review_status.json). Values resolve directly from this file when figures are regenerated.

This repository compares persistence, ridge autoregression, and an LSTM for weekly Mauna Loa CO₂ forecasting with 24-week input windows. Review identified future-information borrowing in the original interpolation step, which has been replaced with past-only forward filling. The earlier numerical comparison is now explicitly historical; the corrected neural experiment still requires a full rerun before its performance can be presented as verified.

## Verification performed in this review

The existing suite requires unavailable dependencies; no full-suite pass is claimed. The complete data/model experiment was not rerun in this review. The old results used interpolation and are superseded for the corrected protocol; PyTorch is unavailable here.

The figure-generation check verifies agreement between the selected source values and SVGs. It does not validate the raw dataset, fitted model, identification assumptions, or external generalization.

```bash
python scripts/build_review_figures.py
python scripts/build_review_figures.py --check
```

## Implementation map

Follow these functions to inspect each transformation. Validation helpers and private functions remain visible in the linked modules.

| Function | Purpose / documented behavior |
|---|---|
| [`load_weekly_series`](src/run_experiment.py#L34) | Inspect the explicit implementation and its callers. |
| [`prepare_windows`](src/run_experiment.py#L47) | Inspect the explicit implementation and its callers. |
| [`error_metrics`](src/run_experiment.py#L72) | Inspect the explicit implementation and its callers. |
| [`train_lstm`](src/run_experiment.py#L81) | Inspect the explicit implementation and its callers. |
| [`run_experiment`](src/run_experiment.py#L115) | Inspect the explicit implementation and its callers. |
| [`main`](src/run_experiment.py#L189) | Inspect the explicit implementation and its callers. |
| [`forward`](src/run_experiment.py#L29) | Inspect the explicit implementation and its callers. |

## What remains before a stronger research claim

The original results used linear interpolation before splitting. The corrected loader uses forward fill to avoid future borrowing. Historical scores are preserved for traceability but cannot substantiate the corrected protocol until a full rerun. A successful software test is not validation of a scientific construct. New experiments should state their split unit, comparator, outcome, uncertainty procedure and failure criteria before examining final test results.
