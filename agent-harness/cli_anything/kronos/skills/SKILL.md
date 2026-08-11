---
name: "cli-anything-kronos"
description: "Agent-native CLI for Kronos — financial K-line prediction, finetuning, and backtesting."
---

# cli-anything-kronos

Agent-native CLI for **Kronos**, a foundation model for financial candlestick (K-line) prediction.

## Commands

### predict
- `predict run --data FILE --lookback N --pred-len N [--json]` — Single stock prediction
- `predict batch --data-dir DIR [--json]` — Batch prediction across directory
- `predict info --name MODEL` — Load and inspect a model

### finetune
- `finetune list-models [--json]` — List available models
- `finetune train --config YAML [--json]` — Full finetune pipeline
- `finetune train-tokenizer --config YAML [--json]` — Train tokenizer
- `finetune train-predictor --config YAML [--json]` — Train predictor

### backtest
- `backtest run --data FILE --symbol SYM [--json]` — Forward-test backtest
- `backtest summary --input JSON [--json]` — Summarize backtest results

### model
- `models [--json]` — List available models
- `device` — Show resolved inference device

### REPL
- Run `cli-anything-kronos` with no subcommand for interactive mode
- Commands: `model list/load`, `predict run`, `backtest run`, `status`, `save`, `load`, `undo`, `redo`, `exit`

## Data Format
CSV with columns: `timestamps`, `open`, `high`, `low`, `close` (required), `volume`, `amount` (optional)

## JSON Output
All commands support `--json` for machine-readable output.

## Models
- `kronos-mini`: 4.1M params, max_context=2048
- `kronos-small`: 24.7M params, max_context=512
- `kronos-base`: 102.3M params, max_context=512
