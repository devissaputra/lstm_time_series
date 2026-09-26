> **Protocol correction:** missing weekly values now use past-only forward fill. Previously recorded interpolation-based scores are historical; a full corrected run is pending.

# Reproducing the experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The weekly CO2 series is converted into 24-week windows and split chronologically. Normalization statistics are calculated only from observations available by the end of the final training window.

The full run compares persistence, Ridge autoregression, and an LSTM trained for 20 epochs. PyTorch uses seed 42, one CPU thread, and a seeded DataLoader generator.

Outputs:

- `results/metrics.json`
- `results/figures/forecast_comparison.png`
- `results/figures/training_loss.png`

The CI test suite runs a one-epoch smoke test rather than the full 20-epoch training job. Exact neural-network values may vary slightly across PyTorch builds.
