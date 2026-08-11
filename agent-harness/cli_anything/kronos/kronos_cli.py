#!/usr/bin/env python3
"""Kronos agent-native CLI — prediction, finetuning, and backtesting for K-line data.

Usage:
    # Predict
    cli-anything-kronos predict run --data ./data.csv --lookback 400 --pred-len 120 --json

    # List models
    cli-anything-kronos finetune list-models

    # Backtest
    cli-anything-kronos backtest run --data ./data.csv --symbol 600580

    # Interactive REPL
    cli-anything-kronos
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

import click

# Ensure the Kronos project root is in path when running as a module
_KRONOS_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if _KRONOS_ROOT.name == "Kronos" and _KRONOS_ROOT.exists():
    sys.path.insert(0, str(_KRONOS_ROOT))

from cli_anything.kronos.core.session import Session
from cli_anything.kronos.core.predict import predict
from cli_anything.kronos.core.finetune import finetune
from cli_anything.kronos.core.backtest import backtest
from cli_anything.kronos.utils.repl_skin import ReplSkin, make_skin
from cli_anything.kronos.utils.kronos_backend import _AVAILABLE_MODELS, _resolve_device


# ---------------------------------------------------------------------------
# Global session
# ---------------------------------------------------------------------------
_session: Optional[Session] = None
_json_output = False


def get_session() -> Session:
    global _session
    if _session is None:
        _session = Session()
    return _session


def output(data, message: str = "") -> None:
    if _json_output:
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        if message:
            click.echo(message)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (list, dict)):
                    click.echo(f"  {k}:")
                    _print_nested(v, indent=4)
                else:
                    click.echo(f"  {k}: {v}")
        elif isinstance(data, list):
            for item in data:
                click.echo(f"  - {item}")
        else:
            click.echo(str(data))


def _print_nested(obj, indent: int = 4) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                click.echo(" " * indent + f"{k}:")
                _print_nested(v, indent + 2)
            else:
                click.echo(" " * indent + f"{k}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            click.echo(" " * indent + f"- {item}")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def _build_repl_command_table() -> dict:
    """Build a command table for the REPL help display."""
    skin = make_skin("kronos")
    session = get_session()
    tables = {
        "predict": [
            ("run", "--data FILE --lookback N --pred-len N"),
            ("batch", "--data-dir DIR"),
            ("info", "--name MODEL"),
        ],
        "finetune": [
            ("list-models", "List available models"),
            ("train", "--config YAML"),
            ("train-tokenizer", "--config YAML"),
            ("train-predictor", "--config YAML"),
        ],
        "backtest": [
            ("run", "--data FILE --symbol SYM"),
            ("summary", "--input RESULTS.json"),
        ],
        "session": [
            ("status", "Show current session state"),
            ("save", "[PATH]"),
            ("load", "PATH"),
            ("undo", "Undo last action"),
            ("redo", "Redo last action"),
            ("clear", "Clear session"),
        ],
        "model": [
            ("list", "List available models"),
            ("load", "--name NAME"),
            ("info", "--name NAME"),
        ],
    }
    return tables


def _run_repl() -> None:
    skin = make_skin("kronos")
    session = get_session()
    pt_session = skin.create_prompt_session()

    skin.print_banner()

    # Load session if exists
    session_path = Path.home() / ".cli-anything-kronos-session.json"
    if session_path.exists():
        try:
            session.load(str(session_path))
            skin.info(f"Session loaded from {session_path}")
        except Exception as e:
            skin.warning(f"Could not load session: {e}")

    cmd_table = _build_repl_command_table()
    skin.help(cmd_table)

    while True:
        try:
            line = skin.get_input(pt_session, prompt="kronos> ").strip()
        except (EOFError, KeyboardInterrupt):
            skin.print_goodbye()
            break

        if not line:
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("exit", "quit", "q"):
            # Auto-save session
            try:
                session.save(str(session_path))
                skin.success(f"Session saved to {session_path}")
            except Exception:
                pass
            skin.print_goodbye()
            break

        elif cmd == "help":
            skin.help(cmd_table)

        elif cmd == "status":
            skin.table(
                ["Key", "Value"],
                [[k, str(v)] for k, v in session.status().items()]
            )

        elif cmd == "save":
            save_path = args.strip() or str(session_path)
            session.save(save_path)
            skin.success(f"Session saved to {save_path}")

        elif cmd == "load":
            try:
                session.load(args.strip())
                skin.success("Session loaded")
            except Exception as e:
                skin.error(str(e))

        elif cmd == "undo":
            result = session.undo()
            if result:
                skin.success(result["action"])
            else:
                skin.warning("Nothing to undo")

        elif cmd == "redo":
            result = session.redo()
            if result:
                skin.success(result["action"])
            else:
                skin.warning("Nothing to redo")

        elif cmd == "clear":
            session = Session()
            skin.success("Session cleared")

        elif cmd == "model":
            if args.strip() == "list":
                models = []
                for key, info in _AVAILABLE_MODELS.items():
                    models.append([key, info["params"], str(info["max_context"]), info["model"]])
                skin.table(["Name", "Params", "MaxCtx", "ModelRepo"], models)
            elif args.strip().startswith("load"):
                name = args.split(None, 1)[1].strip() if " " in args else "kronos-small"
                predictor, meta = __import__("cli_anything.kronos.utils.kronos_backend", fromlist=["load_model"]).load_model(name, device=_resolve_device())
                session.set_model(meta["name"], meta["model_path"], meta["device"], meta["max_context"])
                skin.success(f"Model loaded: {meta['name']} ({meta['params']}) on {meta['device']}")
            else:
                skin.warning("Usage: model list | model load <name>")

        else:
            skin.error(f"Unknown command: {cmd}. Type 'help' for available commands.")


# ---------------------------------------------------------------------------
# CLI Group
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.version_option(version="1.0.0", prog_name="cli-anything-kronos")
@click.option("--json", "json_flag", is_flag=True, help="Global JSON output mode")
@click.pass_context
def cli(ctx, json_flag):
    """Kronos CLI — agent-native interface for financial K-line prediction.

    Kronos is a foundation model for financial candlestick (K-line) sequences,
    trained on data from over 45 global exchanges.

    Quick start:
      cli-anything-kronos predict run --data ./data.csv --lookback 400 --pred-len 120
      cli-anything-kronos finetune list-models
      cli-anything-kronos backtest run --data ./data.csv --symbol 600580
    """
    global _json_output
    _json_output = json_flag
    if ctx.invoked_subcommand is None:
        _run_repl()


# Register subcommands
cli.add_command(predict, "predict")
cli.add_command(finetune, "finetune")
cli.add_command(backtest, "backtest")


# ---------------------------------------------------------------------------
# Model info command (top-level)
# ---------------------------------------------------------------------------

@cli.command("models")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
def models_cmd(json_out):
    """List all available Kronos models."""
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
        click.echo(json.dumps(models, indent=2))
    else:
        click.echo("Available Kronos models:")
        for m in models:
            click.echo(f"  {m['name']:15s}  {m['params']:8s}  max_ctx={m['max_context']}")
    return models


@cli.command("device")
def device_cmd():
    """Show the resolved inference device."""
    dev = _resolve_device()
    click.echo(f"Device: {dev}")
    if "cuda" in dev:
        import torch
        click.echo(f"  GPU: {torch.cuda.get_device_name(0)}")
    return {"device": dev}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    main()
