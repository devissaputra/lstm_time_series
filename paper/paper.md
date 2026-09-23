# LSTM Forecasting for Mauna Loa CO2

## Abstract

This experiment evaluates a compact LSTM against persistence and Ridge autoregression for next-week forecasting of the Mauna Loa CO2 series. All evaluation is chronological and normalization is fitted only on the training period.

## Method

Twenty-four previous weekly observations are used to predict the next value. The first 80% of 2,260 windows are used for training and the final 20% for evaluation. The LSTM has 32 hidden units and trains for 20 epochs with Adam.

## Results

| Model | RMSE | MAE |
|---|---:|---:|
| Persistence | 0.5135 | 0.4042 |
| Ridge | 0.4641 | 0.3546 |
| LSTM | 1.0700 | 0.8540 |

## Interpretation

Ridge gives the best held-out performance, and persistence also outperforms the LSTM. The result demonstrates why sequence models should be compared against simple time-series baselines before their complexity is justified.

## Limitations

A stronger study would use rolling-origin evaluation, seasonal and autoregressive statistical baselines, repeated neural-network seeds, validation-based tuning, and uncertainty intervals.
