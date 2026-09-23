# Web Portfolio Card

## LSTM Time-Series Forecasting

**Track:** AI Engineering  
**Difficulty:** ★★★★  
**Dataset:** Mauna Loa atmospheric CO₂ dataset via statsmodels  
**Quick description:** Forecast real weekly CO₂ observations from 24-week historical windows using a compact LSTM.

### Suggested website image gallery

![Cover](assets/01_cover.svg)

![Time-series pipeline](assets/02_data_pipeline.svg)

![Sequence model](assets/03_data_or_model.svg)

![Chronological hold-out evaluation](assets/04_evaluation_or_results.svg)

### Suggested portfolio copy
This project trains a compact LSTM to forecast the next weekly Mauna Loa CO₂ observation from the preceding 24 weeks. The model uses a 32-unit recurrent state, an 80/20 chronological split, and one-step-ahead evaluation with RMSE and MAE. The repository also documents a preprocessing limitation in the current implementation: normalization statistics are computed on the full series and should be restricted to the training period in a stricter experiment.
