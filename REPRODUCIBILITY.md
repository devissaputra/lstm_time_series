# Reproducing the Experiment

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The script sets the PyTorch seed to 42 and uses one CPU thread.

The CO2 series is resampled weekly and interpolated. It is converted into 24-week windows, then split chronologically: 80% for training and 20% for evaluation.

Normalization is fitted only on the training period. The same training mean and standard deviation are then applied to the held-out period.

The model trains for 20 epochs with Adam at learning rate 0.005.

Results are written to `results/metrics.json`. Record package versions if you need exact numerical reproduction.
