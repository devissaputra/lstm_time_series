from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from statsmodels.datasets import co2

torch.manual_seed(42)
torch.set_num_threads(1)

series = (
    co2.load_pandas()
    .data["co2"]
    .resample("W")
    .mean()
    .interpolate()
    .astype("float32")
)

values = series.to_numpy()
window = 24
n_windows = len(values) - window
split = int(0.8 * n_windows)

# Fit normalization only on values that belong to the training period.
train_end = split + window
train_mean = float(values[:train_end].mean())
train_std = float(values[:train_end].std())
scaled = (values - train_mean) / train_std

X = np.array(
    [scaled[i:i + window] for i in range(n_windows)],
    dtype="float32",
)[:, :, None]
y = np.array(
    [scaled[i + window] for i in range(n_windows)],
    dtype="float32",
)[:, None]

train_loader = DataLoader(
    TensorDataset(torch.tensor(X[:split]), torch.tensor(y[:split])),
    batch_size=64,
    shuffle=False,
)

class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1])

model = LSTMForecaster()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

losses = []
for _ in range(20):
    batch_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    losses.append(float(np.mean(batch_losses)))

with torch.no_grad():
    forecast = model(torch.tensor(X[split:])).numpy().ravel()
    forecast = forecast * train_std + train_mean
    observed = y[split:].ravel() * train_std + train_mean

rmse = float(np.sqrt(np.mean((forecast - observed) ** 2)))
mae = float(np.mean(np.abs(forecast - observed)))

results = {
    "rmse": rmse,
    "mae": mae,
    "window_weeks": window,
    "n_windows": int(n_windows),
    "train_windows": int(split),
    "test_windows": int(n_windows - split),
}

Path("results").mkdir(exist_ok=True)
Path("results/metrics.json").write_text(json.dumps(results, indent=2))

plt.figure(figsize=(9, 5))
plt.plot(series.index, series.values)
plt.xlabel("Date")
plt.ylabel("CO2")
plt.title("Mauna Loa weekly CO2 series")
plt.tight_layout()
plt.savefig("assets/03_data_or_model.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(observed, label="observed")
plt.plot(forecast, label="forecast")
plt.xlabel("Held-out step")
plt.ylabel("CO2")
plt.title("LSTM chronological hold-out forecast")
plt.legend()
plt.tight_layout()
plt.savefig("assets/04_evaluation_or_results.png", dpi=150)
plt.close()

print(json.dumps(results, indent=2))
