# LSTM Time-Series Forecasting: Scientific-Style Technical Report

**Status:** reproducible portfolio report, not peer reviewed.  
**Difficulty:** ★★★★  
**Dataset:** Mauna Loa atmospheric CO₂ dataset via statsmodels

## Abstract
This project studies a concrete AI Engineering problem using a real public dataset and a fully inspectable pipeline. The project focuses on LSTM, time series, forecasting, temporal split. Its central engineering goal is to make data preparation, model fitting, evaluation, and limitations reproducible rather than treating the model as a black box.

## 1. Research objective
Forecast real atmospheric CO₂ observations from lagged historical windows using an LSTM.

## 2. Data
The dataset is **Mauna Loa atmospheric CO₂ dataset via statsmodels**. Provenance and the original reference are documented in [`DATA.md`](../DATA.md).

## 3. Method
The implemented pipeline is:
1. Load CO2 series
2. Interpolate gaps
3. Window sequences
4. LSTM
5. Chronological forecast

## 4. Evaluation
**Primary metric(s):** RMSE / MAE.  
**Validation design:** chronological hold-out.  
The experiment saves machine-readable metrics and visual diagnostics so claims can be traced to an executable run.

## 5. Results
Generated metrics:
```json
{
  "rmse": 5.781065940856934,
  "mae": 5.34670877456665,
  "window_weeks": 24,
  "n_windows": 2260
}
```

## 6. Limitations and validity
Key concern: nonstationarity. Benchmark performance on one dataset does not imply universal performance. The project is intended to demonstrate research engineering discipline and to provide a base for stronger comparative studies.

## 7. Reproducibility
Run `python src/run_experiment.py` from the repository root after installing `requirements.txt`.

## 8. Next research extension
Add repeated cross-validation or temporal/external validation, stronger baselines, hyperparameter sensitivity, confidence intervals, and a domain-specific error analysis.

## References
- Dataset/reference page: https://www.statsmodels.org/stable/datasets/generated/co2.html
