# Portfolio Summary

## LSTM Forecasting for Mauna Loa CO2

I use 24 weeks of historical CO2 measurements to forecast the next weekly value with a 32-unit LSTM.

The split is chronological, and normalization is fitted only on the training period.

### Images

![Project overview](assets/01_cover.svg)

![Forecasting pipeline](assets/02_data_pipeline.svg)

![Sequence model](assets/03_data_or_model.svg)

![Chronological evaluation](assets/04_evaluation_or_results.svg)

**Key result:** RMSE 4.4853 and MAE 4.3378 on 452 held-out windows after the preprocessing fix.
