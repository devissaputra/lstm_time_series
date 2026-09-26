# LSTM Forecasting for Mauna Loa CO₂

This experiment compares persistence, ridge autoregression, and a compact LSTM for next-week Mauna Loa CO₂ forecasting. Missing weeks now use past-only forward filling, and the corrected 20-epoch experiment has been rerun successfully in GitHub Actions. On the 452-week chronological holdout, ridge achieves RMSE 0.4639 ppm, persistence 0.5135 ppm, and the LSTM 1.0751 ppm. The recurrent model therefore does not justify its added complexity in this fixed-seed setup.

[Calculations and evidence](CALCULATIONS.md)
