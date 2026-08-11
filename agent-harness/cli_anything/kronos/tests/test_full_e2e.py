"""E2E tests for Kronos CLI — real model inference with CPU-only fallback."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def _resolve_cli(name):
    import shutil
    force = os.environ.get("CLI_ANYTHING_FORCE_INSTALLED", "").strip() == "1"
    path = shutil.which(name)
    if path:
        return [path]
    if force:
        raise RuntimeError(f"{name} not found in PATH. Install with: pip install -e .")
    module = name.replace("cli-anything-", "cli_anything.") + "." + name.split("-")[-1] + "_cli"
    return [sys.executable, "-m", module]


class TestCLISubprocess:
    """Test the installed CLI command as a real user/agent would."""

    CLI_BASE = _resolve_cli("cli-anything-kronos")

    def _run(self, args, check=True):
        import subprocess
        return subprocess.run(
            self.CLI_BASE + args,
            capture_output=True,
            text=True,
            check=check,
        )

    def test_help(self):
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "Kronos CLI" in result.stdout or "kronos" in result.stdout.lower()

    def test_models_command(self):
        result = self._run(["models", "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        names = [m["name"] for m in data]
        assert "kronos-small" in names
        assert "kronos-base" in names

    def test_device_command(self):
        result = self._run(["device"])
        assert result.returncode == 0
        assert "cpu" in result.stdout.lower() or "cuda" in result.stdout.lower()

    def test_predict_help(self):
        result = self._run(["predict", "run", "--help"])
        assert result.returncode == 0
        assert "--data" in result.stdout

    def test_finetune_help(self):
        result = self._run(["finetune", "list-models", "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)

    def test_backtest_help(self):
        result = self._run(["backtest", "run", "--help"])
        assert result.returncode == 0
        assert "--data" in result.stdout


class TestPredictionWithRealModel:
    """Integration tests using the real Kronos model (CPU only)."""

    @pytest.mark.skipif(
        not sys.platform.startswith("linux") and not sys.platform.startswith("darwin"),
        reason="Model inference test — platform-dependent",
    )
    def test_predict_single_step(self, tmp_path):
        """Test a single prediction with the real Kronos-small model on CPU."""
        try:
            import torch
            if torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                pytest.skip("Running on GPU/MPS — skipping CPU-only integration test")
        except ImportError:
            pass

        import pandas as pd

        # Ensure Kronos project root is on sys.path for direct model imports
        _kronos_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        if str(_kronos_root) not in sys.path:
            sys.path.insert(0, str(_kronos_root))
        from model import Kronos, KronosTokenizer, KronosPredictor

        # Create synthetic data — need lookback + pred_len rows
        lookback = 400
        pred_len = 120
        n = lookback + pred_len + 10  # extra rows for safety
        df = pd.DataFrame({
            "timestamps": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "open": [100.0 + i * 0.01 + (i % 10) * 0.5 for i in range(n)],
            "high": [101.0 + i * 0.01 + (i % 10) * 0.5 for i in range(n)],
            "low": [99.0 + i * 0.01 + (i % 10) * 0.5 for i in range(n)],
            "close": [100.5 + i * 0.01 + (i % 10) * 0.5 for i in range(n)],
            "volume": [1000.0] * n,
            "amount": [100500.0] * n,
        })
        data_path = tmp_path / "synthetic.csv"
        df.to_csv(data_path, index=False)

        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

        x_df = df.loc[:lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df.loc[:lookback - 1, "timestamps"]
        y_ts = df.loc[lookback: lookback + pred_len - 1, "timestamps"]

        with torch.no_grad():
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )

        assert isinstance(pred_df, pd.DataFrame)
        assert len(pred_df) == 120
        assert list(pred_df.columns) == ["open", "high", "low", "close", "volume", "amount"]

        output_path = tmp_path / "prediction.csv"
        pred_df.to_csv(output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 100  # Not suspiciously small

        print(f"\n  Prediction output: {output_path} ({output_path.stat().st_size:,} bytes)")
        print(f"  First 3 rows:\n{pred_df.head(3)}")
