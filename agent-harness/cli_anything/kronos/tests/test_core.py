"""Unit tests for Kronos CLI core modules — synthetic data, no external deps."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


class TestSession:
    """Test session state management."""

    def test_initial_state(self):
        from cli_anything.kronos.core.session import Session
        s = Session()
        assert s.has_model() is False
        st = s.status()
        assert st["model"] is None
        assert st["n_predictions"] == 0

    def test_set_model(self):
        from cli_anything.kronos.core.session import Session
        s = Session()
        s.set_model("kronos-small", "/path/to/model", "cpu", 512)
        assert s.has_model()
        assert s.status()["model"] == "kronos-small"

    def test_record_prediction(self):
        from cli_anything.kronos.core.session import Session
        s = Session()
        s.record_prediction({"rows": 120, "lookback": 400})
        assert s.status()["n_predictions"] == 1
        assert s.last_prediction["rows"] == 120

    def test_undo_redo(self):
        from cli_anything.kronos.core.session import Session
        s = Session()
        s.set_model("kronos-small", None, "cpu", 512)
        s.record_prediction({"rows": 100})
        s.record_prediction({"rows": 200})
        assert s.status()["n_predictions"] == 2

        r = s.undo()
        assert r["action"] == "undo"
        assert s.status()["n_predictions"] == 1

        r = s.redo()
        assert r["action"] == "redo"
        assert s.status()["n_predictions"] == 2

    def test_save_load(self, tmp_path):
        from cli_anything.kronos.core.session import Session
        s = Session()
        s.set_model("kronos-base", "/models/Kronos-base", "cpu", 512)
        s.record_prediction({"rows": 50})
        save_path = tmp_path / "session.json"
        s.save(str(save_path))

        s2 = Session()
        s2.load(str(save_path))
        assert s2.has_model()
        assert s2.model_name == "kronos-base"
        assert s2.status()["n_predictions"] == 1


class TestBackendHelpers:
    """Test backend utility functions with mocks."""

    def test_resolve_device_cpu(self):
        with patch("cli_anything.kronos.utils.kronos_backend.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            from cli_anything.kronos.utils.kronos_backend import _resolve_device
            assert _resolve_device() == "cpu"

    def test_resolve_device_cuda(self):
        with patch("cli_anything.kronos.utils.kronos_backend.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            from cli_anything.kronos.utils.kronos_backend import _resolve_device
            assert _resolve_device() == "cuda:0"

    def test_get_model_list(self):
        from cli_anything.kronos.utils.kronos_backend import _AVAILABLE_MODELS
        models = list(_AVAILABLE_MODELS.keys())
        assert "kronos-small" in models
        assert "kronos-base" in models
        assert "kronos-mini" in models


class TestPredictionCore:
    """Test prediction logic with mocked predictor."""

    def _make_mock_predictor(self):
        mock = MagicMock()
        mock.predict.return_value = MagicMock()
        mock.predict_batch.return_value = []
        return mock

    def test_predict_run_with_mock(self, tmp_path):
        from cli_anything.kronos.core.predict import run_predict
        import pandas as pd

        # Create a minimal CSV
        df = pd.DataFrame({
            "timestamps": pd.date_range("2024-01-01", periods=500, freq="5min"),
            "open": [100.0] * 500,
            "high": [101.0] * 500,
            "low": [99.0] * 500,
            "close": [100.5] * 500,
            "volume": [1000.0] * 500,
            "amount": [100500.0] * 500,
        })
        data_path = tmp_path / "test.csv"
        df.to_csv(data_path, index=False)

        mock_predictor = self._make_mock_predictor()
        mock_pred_df = pd.DataFrame({
            "open": [101.0] * 10,
            "high": [102.0] * 10,
            "low": [100.0] * 10,
            "close": [101.5] * 10,
            "volume": [1000.0] * 10,
            "amount": [101500.0] * 10,
        })
        mock_predictor.predict.return_value = mock_pred_df

        result = run_predict(
            predictor=mock_predictor,
            data_path=str(data_path),
            lookback=100,
            pred_len=10,
            output_path=str(tmp_path / "out.csv"),
            verbose=False,
        )
        assert result["rows"] == 10
        assert result["pred_len"] == 10
        assert os.path.exists(tmp_path / "out.csv")

    def test_predict_batch_with_mock(self, tmp_path):
        from cli_anything.kronos.core.predict import run_batch_predict
        import pandas as pd

        for i in range(3):
            df = pd.DataFrame({
                "timestamps": pd.date_range("2024-01-01", periods=500, freq="5min"),
                "open": [100.0 + i] * 500,
                "high": [101.0 + i] * 500,
                "low": [99.0 + i] * 500,
                "close": [100.5 + i] * 500,
                "volume": [1000.0] * 500,
                "amount": [100500.0] * 500,
            })
            df.to_csv(tmp_path / f"stock_{i}.csv", index=False)

        mock_predictor = self._make_mock_predictor()
        mock_pred = pd.DataFrame({
            "open": [101.0] * 5,
            "high": [102.0] * 5,
            "low": [100.0] * 5,
            "close": [101.5] * 5,
            "volume": [1000.0] * 5,
            "amount": [101500.0] * 5,
        })
        mock_predictor.predict_batch.return_value = [mock_pred] * 3

        os.makedirs(tmp_path / "out", exist_ok=True)
        results = run_batch_predict(
            predictor=mock_predictor,
            data_dir=str(tmp_path),
            lookback=100,
            pred_len=5,
            output_dir=str(tmp_path / "out"),
            verbose=False,
        )
        assert len(results) == 3
        assert all(r["rows"] == 5 for r in results)


class TestBacktestCore:
    """Test backtest logic with mocked predictor."""

    def test_backtest_run_with_mock(self, tmp_path):
        from cli_anything.kronos.core.backtest import run_backtest
        import pandas as pd

        df = pd.DataFrame({
            "timestamps": pd.date_range("2024-01-01", periods=600, freq="5min"),
            "open": [100.0] * 600,
            "high": [101.0] * 600,
            "low": [99.0] * 600,
            "close": [100.5] * 600,
            "volume": [1000.0] * 600,
            "amount": [100500.0] * 600,
        })
        data_path = tmp_path / "test.csv"
        df.to_csv(data_path, index=False)

        # Mock the entire backtest by patching model loading
        with patch("cli_anything.kronos.core.backtest.run_backtest") as mock_run:
            mock_run.return_value = {
                "symbol": "TEST",
                "data_path": str(data_path),
                "lookback": 100,
                "pred_len": 20,
                "n_windows": 2,
                "avg_direction_accuracy": 0.55,
                "windows": [],
            }
            result = mock_run()
            assert result["n_windows"] == 2
