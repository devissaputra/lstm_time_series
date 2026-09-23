# LSTM Forecasting for Mauna Loa CO2

## Question

Can the previous 24 weeks of atmospheric CO2 measurements predict the next weekly value with a small LSTM?

## Data

I use the Mauna Loa CO2 series from statsmodels. I resample it to weekly means and interpolate missing weekly values.

I create 2,260 sliding windows. Each input contains 24 weeks and the target is the following week.

The first 1,808 windows are used for training and the final 452 are held out in chronological order.

Normalization statistics are fitted on the training period only.

## Method

The model contains one LSTM layer with 32 hidden units followed by a linear output layer.

Training uses Adam with learning rate 0.005 and mean-squared error loss for 20 epochs.

## Results

After fixing the preprocessing so the hold-out period is not used to calculate normalization statistics, the recorded run produced:

| Metric | Result |
|---|---:|
| RMSE | 4.4853 |
| MAE | 4.3378 |

## Interpretation

The model follows the later CO2 series reasonably well in this small experiment, but the error values are hard to judge without baselines.

The next step should not be a larger neural network. It should be a fair comparison against simple forecasting methods.

## Limitations

The evaluation uses one chronological split. The series also contains long-term trend and seasonal structure, so a single LSTM result does not tell me which part of the signal the model is actually exploiting.

Useful baselines would include last-value prediction, seasonal naive forecasting, moving averages, and autoregressive models.

## Reproduce

```bash
pip install -r requirements.txt
python src/run_experiment.py
```
