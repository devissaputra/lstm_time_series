# Data

This project uses the Mauna Loa atmospheric CO2 dataset distributed with statsmodels.

Source documentation: https://www.statsmodels.org/stable/datasets/generated/co2.html

The raw observations are converted to weekly means and missing weekly values are interpolated.

The forecasting setup uses:

- 24 weeks of history per input window;
- the following week as the target;
- 2,260 total windows;
- the first 1,808 windows for training;
- the final 452 windows for chronological evaluation.

Normalization statistics are calculated from the training period only.
