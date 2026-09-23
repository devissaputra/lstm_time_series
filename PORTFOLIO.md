# LSTM Forecasting for Mauna Loa CO2

**Focus:** leakage-aware sequence forecasting with strong baselines.

A 24-week LSTM forecast is compared with persistence and Ridge autoregression on a chronological hold-out period. Normalization is fitted only on values available during training.

Ridge is strongest in the recorded experiment with RMSE 0.4641, followed by persistence at 0.5135. The LSTM reaches RMSE 1.0700. The repository keeps that negative result because recurrent complexity has to beat simple lag-based models before it is justified.

The project includes deterministic PyTorch training, train-only preprocessing, behavioural tests, GitHub Actions CI, and separate generated diagnostics.
