"""Kronos backend — model loading, prediction, and finetuning wrappers."""

from __future__ import annotations

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_AVAILABLE_MODELS = {
    "kronos-mini": {
        "model": "NeoQuasar/Kronos-mini",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
        "params": "4.1M",
    },
    "kronos-small": {
        "model": "NeoQuasar/Kronos-small",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
        "params": "24.7M",
    },
    "kronos-base": {
        "model": "/run/media/onai/MyDisk/Work/Kronos/models/Kronos-base",
        "tokenizer": "/run/media/onai/MyDisk/Work/Kronos/models/Kronos-Tokenizer-base",
        "max_context": 512,
        "params": "102.3M",
    },
}

# Cache singleton for loaded models
_model_cache: dict[str, dict] = {}


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    name: str,
    local_path: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple:
    """Load a Kronos model + tokenizer. Returns (predictor, model_meta).

    Args:
        name: Model identifier — one of 'kronos-mini', 'kronos-small', 'kronos-base'
              or a HuggingFace repo id / local path.
        local_path: Optional local path to override HuggingFace download.
        device: Device string ('cpu', 'cuda:0', 'mps'). Auto-detected if None.

    Returns:
        (predictor, meta_dict)
    """
    cache_key = f"{name}:{device or _resolve_device()}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if device is None:
        device = _resolve_device()

    # Resolve model identifier
    if local_path and os.path.isdir(local_path):
        model_path = local_path
        tokenizer_path = local_path
    elif name in _AVAILABLE_MODELS:
        model_path = _AVAILABLE_MODELS[name]["model"]
        tokenizer_path = _AVAILABLE_MODELS[name]["tokenizer"]
    else:
        model_path = name
        tokenizer_path = name

    click.echo(f"  Loading model from {model_path} on {device} ...", err=True)

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError:
        # Try importing from the Kronos project root (when running inside the repo)
        kronos_root = Path(__file__).resolve().parent.parent.parent.parent
        if kronos_root.name == "Kronos" and kronos_root.exists():
            sys.path.insert(0, str(kronos_root))
            from model import Kronos, KronosTokenizer, KronosPredictor
        else:
            raise ImportError(
                "Kronos model package not found. "
                "Install Kronos with: pip install -e /path/to/Kronos"
            )

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(model_path)
    max_ctx = _AVAILABLE_MODELS.get(name, {}).get("max_context", 512)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_ctx)

    meta = {
        "name": name,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "device": device,
        "max_context": max_ctx,
        "params": _AVAILABLE_MODELS.get(name, {}).get("params", "unknown"),
    }

    _model_cache[cache_key] = (predictor, meta)
    return predictor, meta


def get_model_list() -> list[dict]:
    """Return info about all built-in models."""
    out = []
    for key, info in _AVAILABLE_MODELS.items():
        out.append({
            "name": key,
            "model_repo": info["model"],
            "tokenizer_repo": info["tokenizer"],
            "max_context": info["max_context"],
            "params": info["params"],
        })
    return out


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw K-line DataFrame to expected columns."""
    df = df.copy()
    # Normalize column names
    col_map = {
        "open": "open", "high": "high", "low": "low", "close": "close",
        "volume": "volume", "amount": "amount",
    }
    lower = {k.lower(): v for k, v in col_map.items()}
    for old, new in lower.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns and "volume" in df.columns:
        df["amount"] = df["volume"] * df[required].mean(axis=1)
    if "amount" not in df.columns:
        df["amount"] = 0.0

    # Ensure timestamps column
    if "timestamps" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "timestamps"})
    if "timestamps" not in df.columns:
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "timestamps"})
        df["timestamps"] = pd.date_range(start=df["timestamps"].min(), periods=len(df), freq="5min")

    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    return df


def run_predict(
    predictor,
    data_path: str,
    lookback: int,
    pred_len: int,
    T: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    sample_count: int = 1,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Run a single prediction and return result dict."""
    df = pd.read_csv(data_path)
    df = _prepare_df(df)

    if lookback >= len(df):
        lookback = len(df) - 1
        click.echo(f"  Adjusted lookback to {lookback} (data has {len(df)} rows)", err=True)

    x_df = df.loc[: lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_ts = df.loc[: lookback - 1, "timestamps"]

    # y_timestamp: next pred_len points after lookback
    y_end = lookback + pred_len
    if y_end > len(df):
        y_end = len(df)
    y_ts = df.loc[lookback: y_end - 1, "timestamps"]

    if len(y_ts) != pred_len:
        click.echo(
            f"  Warning: y_timestamp has {len(y_ts)} entries, expected {pred_len}. "
            "Truncating pred_len to available data.",
            err=True,
        )
        pred_len = len(y_ts)

    with torch.no_grad():
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )

    result: dict = {
        "data_path": data_path,
        "lookback": lookback,
        "pred_len": int(pred_len),
        "rows": len(pred_df),
        "columns": list(pred_df.columns),
        "sample_count": sample_count,
        "T": T,
        "top_p": top_p,
        "top_k": top_k,
        "head": pred_df.head(5).to_dict(orient="records"),
    }

    if output_path:
        pred_df.to_csv(output_path, index=True)
        result["output_path"] = str(output_path)
        result["output_rows"] = len(pred_df)

    return result


def run_batch_predict(
    predictor,
    data_dir: str,
    lookback: int,
    pred_len: int,
    T: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    sample_count: int = 1,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    """Run batch prediction across CSV files in a directory."""
    dir_path = Path(data_dir)
    csv_files = sorted(dir_path.glob("*.csv"))
    if not csv_files:
        raise click.ClickException(f"No CSV files found in {data_dir}")

    df_list, x_ts_list, y_ts_list = [], [], []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = _prepare_df(df)
        lb = min(lookback, len(df) - 1)
        x_df = df.loc[: lb - 1, ["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df.loc[: lb - 1, "timestamps"]
        y_end = min(lb + pred_len, len(df))
        y_ts = df.loc[lb: y_end - 1, "timestamps"]
        if len(y_ts) < pred_len:
            pred_len_actual = len(y_ts)
        else:
            pred_len_actual = pred_len
        df_list.append(x_df)
        x_ts_list.append(x_ts)
        y_ts_list.append(y_ts)

    with torch.no_grad():
        pred_dfs = predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_ts_list,
            y_timestamp_list=y_ts_list,
            pred_len=pred_len_actual,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )

    results = []
    for i, (csv_file, pred_df) in enumerate(zip(csv_files, pred_dfs)):
        rec = {
            "source_file": str(csv_file.name),
            "lookback": lb,
            "pred_len": int(pred_len_actual),
            "rows": len(pred_df),
            "columns": list(pred_df.columns),
            "head": pred_df.head(3).to_dict(orient="records"),
        }
        if output_dir:
            out_path = Path(output_dir) / f"{csv_file.stem}_pred.csv"
            pred_df.to_csv(out_path, index=True)
            rec["output_path"] = str(out_path)
        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Finetune
# ---------------------------------------------------------------------------

def run_finetune_tokenizer(
    config_path: str,
    resume: bool = False,
) -> dict:
    """Fine-tune the tokenizer on custom data."""
    try:
        from finetune_csv.config_loader import ConfigLoader
        from finetune_csv.finetune_tokenizer import finetune_tokenizer_main
    except ImportError:
        # Try from Kronos root
        kronos_root = Path(__file__).resolve().parent.parent.parent.parent
        if kronos_root.name == "Kronos" and kronos_root.exists():
            sys.path.insert(0, str(kronos_root))
            from finetune_csv.config_loader import ConfigLoader
            from finetune_csv.finetune_tokenizer import finetune_tokenizer_main
        else:
            raise ImportError(
                "Finetune modules not found. Run from the Kronos project root "
                "or set PYTHONPATH accordingly."
            )

    cfg = ConfigLoader(config_path)
    cfg_dict = cfg.config

    with tempfile.TemporaryDirectory() as tmpdir:
        result = finetune_tokenizer_main(cfg_dict, resume=resume, output_dir=tmpdir)

    out = {
        "config_path": config_path,
        "resume": resume,
        "trained_tokenizer_path": result.get("save_path", ""),
        "final_loss": result.get("final_loss", None),
    }
    return out


def run_finetune_predictor(
    config_path: str,
    resume: bool = False,
) -> dict:
    """Fine-tune the predictor on custom data."""
    try:
        from finetune_csv.config_loader import ConfigLoader
        from finetune_csv.finetune_base_model import finetune_base_model_main
    except ImportError:
        kronos_root = Path(__file__).resolve().parent.parent.parent.parent
        if kronos_root.name == "Kronos" and kronos_root.exists():
            sys.path.insert(0, str(kronos_root))
            from finetune_csv.config_loader import ConfigLoader
            from finetune_csv.finetune_base_model import finetune_base_model_main
        else:
            raise ImportError("Finetune modules not found. Run from Kronos project root.")

    cfg = ConfigLoader(config_path)
    cfg_dict = cfg.config

    with tempfile.TemporaryDirectory() as tmpdir:
        result = finetune_base_model_main(cfg_dict, resume=resume, output_dir=tmpdir)

    out = {
        "config_path": config_path,
        "resume": resume,
        "trained_predictor_path": result.get("save_path", ""),
        "final_loss": result.get("final_loss", None),
    }
    return out


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_backtest(
    config_path: str,
    symbol: str,
    data_path: str,
    lookback: int = 400,
    pred_len: int = 120,
    output_path: Optional[str] = None,
) -> dict:
    """Run a backtest using the Kronos predictor."""
    try:
        from finetune_csv.config_loader import ConfigLoader
    except ImportError:
        kronos_root = Path(__file__).resolve().parent.parent.parent.parent
        if kronos_root.name == "Kronos" and kronos_root.exists():
            sys.path.insert(0, str(kronos_root))
            from finetune_csv.config_loader import ConfigLoader
        else:
            raise ImportError("Finetune modules not found.")

    cfg = ConfigLoader(config_path)
    cfg_dict = cfg.config

    # Override key parameters
    cfg_dict["lookback_window"] = lookback
    cfg_dict["predict_window"] = pred_len
    cfg_dict["data_path"] = data_path
    cfg_dict["symbol"] = symbol

    # Use built-in predictor for backtest
    from model import Kronos, KronosTokenizer, KronosPredictor

    model_name = cfg_dict.get("model_name", "NeoQuasar/Kronos-small")
    tokenizer_name = cfg_dict.get("tokenizer_name", "NeoQuasar/Kronos-Tokenizer-base")
    device = _resolve_device()

    click.echo(f"  Loading model {model_name} for backtest ...", err=True)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg_dict.get("max_context", 512))

    df = pd.read_csv(data_path)
    df = _prepare_df(df)

    # Simple forward-test backtest
    total = len(df)
    step = pred_len
    results_records = []
    for start in range(lookback, total - pred_len, step):
        end = start + pred_len
        x_df = df.loc[start - lookback: start - 1, ["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df.loc[start - lookback: start - 1, "timestamps"]
        y_ts = df.loc[start: end - 1, "timestamps"]

        with torch.no_grad():
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=min(pred_len, len(y_ts)),
                T=cfg_dict.get("inference_T", 0.6),
                top_p=cfg_dict.get("inference_top_p", 0.9),
                sample_count=cfg_dict.get("inference_sample_count", 5),
                verbose=False,
            )

        actual = df.loc[start: end - 1, "close"].values
        predicted = pred_df["close"].values[: len(actual)]

        # Simple direction accuracy
        if len(actual) > 1 and len(predicted) > 1:
            actual_dir = np.diff(actual) > 0
            pred_dir = np.diff(predicted) > 0
            accuracy = float(np.mean(actual_dir == pred_dir))
        else:
            accuracy = None

        results_records.append({
            "start_idx": int(start),
            "end_idx": int(end),
            "actual_close_last": float(actual[-1]) if len(actual) else None,
            "predicted_close_last": float(predicted[-1]) if len(predicted) else None,
            "direction_accuracy": accuracy,
        })

    out = {
        "symbol": symbol,
        "data_path": data_path,
        "lookback": lookback,
        "pred_len": pred_len,
        "n_windows": len(results_records),
        "avg_direction_accuracy": (
            round(np.mean([r["direction_accuracy"] for r in results_records if r["direction_accuracy"] is not None]), 4)
            if results_records
            else None
        ),
        "windows": results_records[:10],  # first 10 windows
    }

    if output_path:
        import json as _json
        with open(output_path, "w") as f:
            _json.dump(out, f, indent=2, default=str)
        out["output_path"] = output_path

    return out
