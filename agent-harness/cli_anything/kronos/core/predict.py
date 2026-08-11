"""Core prediction module — single and batch prediction commands."""

from __future__ import annotations

import os
from typing import Optional

import click

from cli_anything.kronos.utils.kronos_backend import (
    load_model,
    run_predict,
    run_batch_predict,
    _resolve_device,
)
from cli_anything.kronos.core.session import Session


def _get_session() -> Session:
    from cli_anything.kronos.kronos_cli import _session
    return _session  # type: ignore


@click.group()
def predict():
    """Prediction commands — forecast future K-line sequences."""
    pass


@predict.command("info")
@click.option("--name", "model_name", default="kronos-small", help="Model identifier (kronos-mini, kronos-small, kronos-base, or HF repo id)")
@click.option("--device", default=None, help="Device: cpu, cuda:0, mps (auto-detected if omitted)")
@click.pass_context
def predict_info(ctx, model_name, device):
    """Load and inspect a model without running prediction."""
    sess = _get_session()
    predictor, meta = load_model(model_name, device=device or _resolve_device())
    sess.set_model(meta["name"], meta["model_path"], meta["device"], meta["max_context"])
    click.echo(f"Model loaded: {meta['name']} ({meta['params']} params) on {meta['device']}")
    click.echo(f"  max_context: {meta['max_context']}")
    click.echo(f"  model_repo:  {meta['model_path']}")
    click.echo(f"  tokenizer:   {meta['tokenizer_path']}")
    return {"model": meta}


@predict.command()
@click.option("--data", "data_path", required=True, help="Path to input CSV with OHLCV columns")
@click.option("--lookback", "-l", default=400, help="Historical context length (default: 400)")
@click.option("--pred-len", "pred_len", default=120, help="Number of steps to predict (default: 120)")
@click.option("--model", "-m", "model_name", default=None, help="Model name (overrides session model)")
@click.option("--device", default=None, help="Device override")
@click.option("--T", "temperature", default=1.0, help="Sampling temperature (default: 1.0)")
@click.option("--top-p", "top_p", default=0.9, help="Nucleus sampling threshold (default: 0.9)")
@click.option("--top-k", "top_k", default=0, help="Top-k filtering (default: 0 = off)")
@click.option("--sample-count", "sample_count", default=1, help="Number of samples to average (default: 1)")
@click.option("--output", "-o", "output_path", default=None, help="Output CSV path")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
@click.option("--verbose", is_flag=True, default=True, help="Show progress (default: on)")
@click.pass_context
def run(ctx, data_path, lookback, pred_len, model_name, device, temperature, top_p, top_k, sample_count, output_path, json_out, verbose):
    """Run a single K-line prediction."""
    sess = _get_session()

    # Resolve model
    resolve_model = model_name or sess.model_name or "kronos-small"
    resolve_device = device or sess.device or _resolve_device()
    predictor, meta = load_model(resolve_model, device=resolve_device)
    if sess.model_name != resolve_model or sess.device != resolve_device:
        sess.set_model(meta["name"], meta["model_path"], meta["device"], meta["max_context"])

    if not os.path.isfile(data_path):
        raise click.ClickException(f"Data file not found: {data_path}")

    result = run_predict(
        predictor=predictor,
        data_path=data_path,
        lookback=lookback,
        pred_len=pred_len,
        T=temperature,
        top_p=top_p,
        top_k=top_k,
        sample_count=sample_count,
        output_path=output_path,
        verbose=verbose,
    )
    sess.record_prediction(result)

    if json_out:
        import json as _json
        click.echo(_json.dumps(result, indent=2, default=str))
    else:
        click.echo(f"\nPrediction complete: {result['rows']} rows, {result['pred_len']} steps")
        if output_path:
            click.echo(f"  Output: {output_path}")
        click.echo(f"\nFirst 5 rows:")
        for row in result["head"]:
            click.echo(f"  {row}")
    return result


@predict.command("batch")
@click.option("--data-dir", "data_dir", required=True, help="Directory containing input CSV files")
@click.option("--lookback", "-l", default=400, help="Historical context length (default: 400)")
@click.option("--pred-len", "pred_len", default=120, help="Number of steps to predict (default: 120)")
@click.option("--model", "-m", "model_name", default=None, help="Model name")
@click.option("--device", default=None, help="Device override")
@click.option("--T", "temperature", default=1.0, help="Sampling temperature")
@click.option("--top-p", "top_p", default=0.9, help="Nucleus sampling threshold")
@click.option("--top-k", "top_k", default=0, help="Top-k filtering")
@click.option("--sample-count", "sample_count", default=1, help="Samples to average")
@click.option("--output-dir", "output_dir", default=None, help="Output directory for predictions")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
@click.option("--verbose", is_flag=True, default=True, help="Show progress")
@click.pass_context
def batch(ctx, data_dir, lookback, pred_len, model_name, device, temperature, top_p, top_k, sample_count, output_dir, json_out, verbose):
    """Batch-predict across all CSV files in a directory."""
    sess = _get_session()
    resolve_model = model_name or sess.model_name or "kronos-small"
    resolve_device = device or sess.device or _resolve_device()
    predictor, meta = load_model(resolve_model, device=resolve_device)
    if sess.model_name != resolve_model or sess.device != resolve_device:
        sess.set_model(meta["name"], meta["model_path"], meta["device"], meta["max_context"])

    results = run_batch_predict(
        predictor=predictor,
        data_dir=data_dir,
        lookback=lookback,
        pred_len=pred_len,
        T=temperature,
        top_p=top_p,
        top_k=top_k,
        sample_count=sample_count,
        output_dir=output_dir,
        verbose=verbose,
    )
    for r in results:
        sess.record_prediction(r)

    if json_out:
        import json as _json
        click.echo(_json.dumps(results, indent=2, default=str))
    else:
        click.echo(f"\nBatch prediction complete: {len(results)} files processed")
        for r in results:
            click.echo(f"  {r['source_file']} → {r['rows']} rows" + (f"  → {r.get('output_path', '')}" if r.get("output_path") else ""))
    return results
