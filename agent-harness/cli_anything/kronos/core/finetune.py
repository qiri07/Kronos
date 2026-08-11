"""Finetune commands — tokenizer and predictor training workflows."""

from __future__ import annotations

import os
from typing import Optional

import click

from cli_anything.kronos.utils.kronos_backend import (
    run_finetune_tokenizer,
    run_finetune_predictor,
)


@click.group()
def finetune():
    """Finetune Kronos models on custom data."""
    pass


@finetune.command("list-models")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
def list_models(json_out):
    """List available pre-trained Kronos models."""
    try:
        from cli_anything.kronos.utils.kronos_backend import _AVAILABLE_MODELS
    except ImportError:
        _AVAILABLE_MODELS = {}

    models = []
    for key, info in _AVAILABLE_MODELS.items():
        models.append({
            "name": key,
            "model_repo": info["model"],
            "tokenizer_repo": info["tokenizer"],
            "max_context": info["max_context"],
            "params": info["params"],
        })

    if json_out:
        import json as _json
        click.echo(_json.dumps(models, indent=2))
    else:
        click.echo("Available Kronos models:")
        for m in models:
            click.echo(f"  {m['name']:15s}  {m['params']:8s}  max_ctx={m['max_context']}  model={m['model_repo']}")
    return models


@finetune.command("train-tokenizer")
@click.option("--config", "config_path", required=True, help="Path to YAML finetune config")
@click.option("--resume", is_flag=True, help="Resume from existing checkpoint")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
def train_tokenizer(config_path, resume, json_out):
    """Fine-tune the tokenizer on custom K-line data."""
    if not os.path.isfile(config_path):
        raise click.ClickException(f"Config file not found: {config_path}")

    click.echo(f"Finetuning tokenizer from {config_path}{' (resume)' if resume else ''} ...", err=True)
    result = run_finetune_tokenizer(config_path, resume=resume)

    if json_out:
        import json as _json
        click.echo(_json.dumps(result, indent=2, default=str))
    else:
        click.echo(f"  Tokenizer saved to: {result.get('trained_tokenizer_path', 'N/A')}")
        if result.get("final_loss") is not None:
            click.echo(f"  Final loss: {result['final_loss']:.6f}")
    return result


@finetune.command("train-predictor")
@click.option("--config", "config_path", required=True, help="Path to YAML finetune config")
@click.option("--resume", is_flag=True, help="Resume from existing checkpoint")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
def train_predictor(config_path, resume, json_out):
    """Fine-tune the predictor on custom K-line data."""
    if not os.path.isfile(config_path):
        raise click.ClickException(f"Config file not found: {config_path}")

    click.echo(f"Finetuning predictor from {config_path}{' (resume)' if resume else ''} ...", err=True)
    result = run_finetune_predictor(config_path, resume=resume)

    if json_out:
        import json as _json
        click.echo(_json.dumps(result, indent=2, default=str))
    else:
        click.echo(f"  Predictor saved to: {result.get('trained_predictor_path', 'N/A')}")
        if result.get("final_loss") is not None:
            click.echo(f"  Final loss: {result['final_loss']:.6f}")
    return result


@finetune.command("train")
@click.option("--config", "config_path", required=True, help="Path to YAML finetune config (runs both tokenizer and predictor)")
@click.option("--resume", is_flag=True, help="Resume from existing checkpoints")
@click.option("--skip-tokenizer", is_flag=True, help="Skip tokenizer training")
@click.option("--skip-predictor", is_flag=True, help="Skip predictor training")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
def train(config_path, resume, skip_tokenizer, skip_predictor, json_out):
    """Run full finetune pipeline (tokenizer + predictor)."""
    results = {}
    if not skip_tokenizer:
        click.echo("=== Training Tokenizer ===", err=True)
        results["tokenizer"] = train_tokenizer(config_path, resume, json_out=False)
    if not skip_predictor:
        click.echo("=== Training Predictor ===", err=True)
        results["predictor"] = train_predictor(config_path, resume, json_out=False)

    if json_out:
        import json as _json
        click.echo(_json.dumps(results, indent=2, default=str))
    else:
        click.echo("\nFinetune complete.")
    return results
