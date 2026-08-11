"""Backtest commands — evaluate prediction performance on historical data."""

from __future__ import annotations

import os
from typing import Optional

import click

from cli_anything.kronos.utils.kronos_backend import run_backtest


@click.group()
def backtest():
    """Backtest prediction strategies on historical K-line data."""
    pass


@backtest.command()
@click.option("--data", "data_path", required=True, help="Path to historical CSV data")
@click.option("--symbol", "symbol", required=True, help="Symbol identifier (e.g. 600580)")
@click.option("--config", "config_path", default="./finetune_csv/configs/config_ali09988_candle-5min.yaml", help="Finetune config path")
@click.option("--lookback", "-l", default=400, help="Lookback window (default: 400)")
@click.option("--pred-len", "pred_len", default=120, help="Prediction length (default: 120)")
@click.option("--output", "-o", "output_path", default=None, help="Output JSON path")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
def run(data_path, symbol, config_path, lookback, pred_len, output_path, json_out):
    """Run a forward-test backtest on historical data."""
    if not os.path.isfile(data_path):
        raise click.ClickException(f"Data file not found: {data_path}")

    click.echo(f"Backtesting {symbol} on {data_path} ...", err=True)
    result = run_backtest(
        config_path=config_path,
        symbol=symbol,
        data_path=data_path,
        lookback=lookback,
        pred_len=pred_len,
        output_path=output_path,
    )

    if json_out:
        import json as _json
        click.echo(_json.dumps(result, indent=2, default=str))
    else:
        click.echo(f"\nBacktest complete: {result['n_windows']} windows")
        click.echo(f"  Avg direction accuracy: {result.get('avg_direction_accuracy', 'N/A')}")
        if output_path:
            click.echo(f"  Results saved to: {output_path}")
        click.echo(f"\nFirst {min(3, len(result['windows']))} windows:")
        for w in result["windows"][:3]:
            click.echo(f"  [{w['start_idx']}:{w['end_idx']}] "
                       f"actual={w['actual_close_last']:.2f} "
                       f"pred={w['predicted_close_last']:.2f} "
                       f"acc={w['direction_accuracy']}")
    return result


@backtest.command("summary")
@click.option("--input", "input_path", required=True, help="Path to backtest result JSON (from --output)")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
def summary(input_path, json_out):
    """Show a summary of a previous backtest result."""
    import json as _json
    if not os.path.isfile(input_path):
        raise click.ClickException(f"Result file not found: {input_path}")
    with open(input_path) as f:
        result = _json.load(f)

    out = {
        "symbol": result.get("symbol"),
        "data_path": result.get("data_path"),
        "lookback": result.get("lookback"),
        "pred_len": result.get("pred_len"),
        "n_windows": result.get("n_windows"),
        "avg_direction_accuracy": result.get("avg_direction_accuracy"),
    }
    if json_out:
        click.echo(_json.dumps(out, indent=2))
    else:
        click.echo(f"Symbol:        {out['symbol']}")
        click.echo(f"Data:          {out['data_path']}")
        click.echo(f"Lookback:      {out['lookback']}")
        click.echo(f"Pred length:   {out['pred_len']}")
        click.echo(f"Windows:       {out['n_windows']}")
        click.echo(f"Dir. accuracy: {out['avg_direction_accuracy']}")
    return out
