# LSTM Forecasting for Mauna Loa CO2

This repository compares persistence, ridge autoregression, and an LSTM for weekly Mauna Loa CO₂ forecasting with 24-week input windows. Review identified future-information borrowing in the original interpolation step, which has been replaced with past-only forward filling. The earlier numerical comparison is now explicitly historical; the corrected neural experiment still requires a full rerun before its performance can be presented as verified.

See [CALCULATIONS.md](CALCULATIONS.md) for evidence and verification scope.
