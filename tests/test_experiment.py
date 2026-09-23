from pathlib import Path

import numpy as np
import torch

from src.run_experiment import (
    LSTMForecaster,
    prepare_windows,
    run_experiment,
)


def test_model_output_shape():
    model = LSTMForecaster()
    output = model(torch.zeros(5, 24, 1))
    assert output.shape == (5, 1)


def test_normalization_uses_training_period_only():
    values = np.arange(50, dtype="float32")
    _, _, split, mean, _ = prepare_windows(values, window=5, train_fraction=0.8)
    expected_end = split + 5
    assert mean == float(values[:expected_end].mean())


def test_one_epoch_smoke_run(tmp_path):
    result = run_experiment(tmp_path, epochs=1, make_plots=False)
    assert set(["persistence", "ridge", "lstm"]).issubset(result)
    assert result["train_windows"] < result["n_windows"]
    for model_name in ["persistence", "ridge", "lstm"]:
        assert result[model_name]["rmse"] >= 0
        assert result[model_name]["mae"] >= 0


def test_repository_structure():
    root = Path(__file__).resolve().parents[1]
    assert (root / ".github/workflows/ci.yml").exists()
    assert (root / "paper/paper.md").exists()
