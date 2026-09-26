# Does a compact LSTM improve next-week Mauna Loa CO₂ forecasts?

Working methodological report. Devis Saputra.

## Abstract

A recurrent architecture should be evaluated against credible simple forecasts rather than judged from its training fit. This experiment compares a 32-unit LSTM with persistence and ridge autoregression on weekly Mauna Loa CO₂ observations distributed with statsmodels. A review identified future borrowing in an earlier interpolation step; the corrected protocol uses past-only forward filling and was rerun in GitHub Actions. All models use a chronological holdout, and trainable models use the same 24-week inputs. Ridge achieves test RMSE 0.4639 ppm, compared with 0.5135 for persistence and 1.0751 for the LSTM. These single-seed results support the simpler model in this setup, with no claim of universal architectural superiority.

## Question and rationale

The question is whether recurrent nonlinear modeling provides an advantage beyond recent lagged observations for one-week prediction. The series is smooth and strongly autocorrelated, making persistence a necessary comparator. Ridge adds a fitted multivariate lag relationship while limiting coefficient size. Comparing both with the LSTM distinguishes gains from fitting a lag model from gains attributable to recurrent complexity.

## Data and preprocessing

The data are the Mauna Loa atmospheric CO₂ series available through `statsmodels.datasets.co2`. Values are resampled weekly. Missing values are forward-filled from preceding observations; no later observation is consulted. This matters because interpolation over the full series can use information unavailable at a forecast date, including within a nominally held-out test era.

A 24-week sliding window predicts the following week. There are 2,260 windows, of which the first 1,808 are used for training and the final 452 for testing. Normalization parameters use observations through the final training target only. The test period is evaluated one week at a time using observed history available at each forecast origin. This is not a recursive multi-step forecast launched once at the start of the test era.

## Models and fixed settings

Persistence predicts the last value in the input window. Ridge regression uses alpha 1.0 on the standardized 24-lag inputs. The LSTM has one recurrent layer with 32 hidden units and a linear output head. It trains for 20 epochs with Adam, learning rate 0.005 and batch size 64. Seed 42 controls the neural initialization and data-loader ordering. There is no test-driven checkpoint or hyperparameter selection in this experiment.

## Outcome definitions

MAE is the mean absolute forecast error; RMSE is the square root of mean squared error. Predictions and targets are transformed back from normalized units before calculation, so both metrics are reported in ppm. RMSE gives larger misses greater weight than MAE. The two metrics agree on the model ordering in this run.

## Results

| Model | RMSE (ppm) ↓ | MAE (ppm) ↓ |
|---|---:|---:|
| persistence | 0.5135 | 0.4042 |
| ridge | 0.4639 | 0.3538 |
| lstm | 1.0751 | 0.8599 |

The best baseline is ridge. The LSTM has more than twice ridge’s RMSE in the corrected run. This is evidence against adding this particular recurrent model under the specified training setup. It does not isolate whether optimization, model size, regularization or data structure explains the gap.

## Validity and next study

One seed and one chronological test era limit the comparison. No confidence interval is estimated, and no claim of statistical significance is made. A stronger follow-up should predefine validation-based tuning, compare seasonal and statistical autoregressive models, evaluate multiple forecast origins and repeat neural initializations. The missing-data policy should also be varied using temporally valid alternatives. Hyperparameters must be selected before inspecting final test performance.

## Reproducibility

The [corrected full run](https://github.com/devissaputra/lstm_time_series/actions/runs/36235069744) and CI succeeded. The exact source commit and artifact are in [run provenance](../results/review_status.json). The [calculation guide](../CALCULATIONS.md) documents the arithmetic and figure-generation paths. The repository’s prior interpolation-based results are superseded. This is a working report, not a peer-reviewed publication.

![Design and interpretation](../assets/review_overview.svg)

![Calculation evidence](../assets/review_calculations.svg)

## References

- Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780. DOI: 10.1162/neco.1997.9.8.1735.
- Seabold, S., and Perktold, J. (2010). Statsmodels: Econometric and Statistical Modeling with Python. Proceedings of the 9th Python in Science Conference.
