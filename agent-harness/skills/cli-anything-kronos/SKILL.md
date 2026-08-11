---
name: "cli-anything-kronos"
description: "Agent-native CLI for Kronos — a foundation model for financial K-line (candlestick) prediction. Supports single/batch prediction, finetuning, and backtesting with structured JSON output."
---

# cli-anything-kronos

Agent-native CLI for **Kronos**, the first open-source foundation model for financial
candlestick (K-line) sequences, trained on data from 45+ global exchanges.

## Installation

```bash
# Install the CLI harness
pip install -e /path/to/Kronos/agent-harness

# Or from CLI-Hub (when available)
cli-hub install kronos
```

**Prerequisites:** Python 3.10+, PyTorch 2.0+, and the Kronos library installed.

## Command Groups

### `predict` — Forecast K-line sequences

```bash
# Single stock prediction
cli-anything-kronos predict run \
  --data ./data/stock.csv \
  --lookback 400 \
  --pred-len 120 \
  --model kronos-small \
  --json

# Batch predict all CSVs in a directory
cli-anything-kronos predict batch \
  --data-dir ./data/ \
  --pred-len 60 \
  --output-dir ./predictions/ \
  --json

# Inspect a model without running prediction
cli-anything-kronos predict info --name kronos-base --json
```

**predict run options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--data` | *(required)* | Input CSV path with OHLCV data |
| `--lookback` | 400 | Historical context length |
| `--pred-len` | 120 | Number of steps to predict |
| `--model` | `kronos-small` | Model name or HF repo id |
| `--T` | 1.0 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--top-k` | 0 | Top-k filtering (0 = off) |
| `--sample-count` | 1 | Number of samples to average |
| `--output` | — | Output CSV path |
| `--json` | — | Machine-readable JSON output |

### `finetune` — Train and manage models

```bash
# List available models
cli-anything-kronos finetune list-models --json

# Train tokenizer on custom data
cli-anything-kronos finetune train-tokenizer \
  --config ./configs/my_config.yaml \
  --json

# Train predictor on custom data
cli-anything-kronos finetune train-predictor \
  --config ./configs/my_config.yaml \
  --json

# Full pipeline (tokenizer + predictor)
cli-anything-kronos finetune train \
  --config ./configs/my_config.yaml \
  --json
```

### `backtest` — Evaluate prediction performance

```bash
# Run forward-test backtest
cli-anything-kronos backtest run \
  --data ./data/history.csv \
  --symbol 600580 \
  --lookback 400 \
  --pred-len 120 \
  --output ./backtest_result.json \
  --json

# Summarize previous backtest
cli-anything-kronos backtest summary --input ./backtest_result.json --json
```

### `model` — Model management

```bash
# List all available models
cli-anything-kronos models --json

# Show resolved device
cli-anything-kronos device
```

### REPL (interactive mode)

```bash
cli-anything-kronos
```

Inside REPL:
```
kronos> model list
kronos> model load kronos-small
kronos> predict run --data ./stock.csv --lookback 400 --pred-len 120
kronos> backtest run --data ./history.csv --symbol 09988
kronos> status
kronos> save
kronos> exit
```

## Model Catalog

| Model | Params | Max Context | HuggingFace Repo |
|-------|--------|-------------|------------------|
| `kronos-mini` | 4.1M | 2048 | NeoQuasar/Kronos-mini |
| `kronos-small` | 24.7M | 512 | NeoQuasar/Kronos-small |
| `kronos-base` | 102.3M | 512 | NeoQuasar/Kronos-base |

Custom models: pass any HuggingFace repo id or local path to `--model`.

## Data Format

Input CSV must contain:
- `open`, `high`, `low`, `close` — OHLC prices (required)
- `timestamps` — datetime column (auto-generated if absent)
- `volume`, `amount` — optional (auto-filled if absent)

Example:
```csv
timestamps,open,high,low,close,volume,amount
2024-01-01 09:30:00,100.5,101.2,99.8,100.9,1500000,150750000
2024-01-01 09:35:00,100.9,101.5,100.6,101.2,1200000,121440000
...
```

## JSON Output Examples

### Prediction
```json
{
  "data_path": "./data/stock.csv",
  "lookback": 400,
  "pred_len": 120,
  "rows": 120,
  "columns": ["open", "high", "low", "close", "volume", "amount"],
  "sample_count": 1,
  "T": 1.0,
  "top_p": 0.9,
  "head": [
    {"timestamps": "2024-02-01T09:30:00", "open": 101.2, "high": 102.0, ...}
  ]
}
```

### Models
```json
[
  {"name": "kronos-small", "params": "24.7M", "max_context": 512, "model_repo": "NeoQuasar/Kronos-small"},
  {"name": "kronos-base", "params": "102.3M", "max_context": 512, "model_repo": "NeoQuasar/Kronos-base"}
]
```

### Backtest
```json
{
  "symbol": "09988",
  "n_windows": 4,
  "avg_direction_accuracy": 0.5833,
  "windows": [
    {"start_idx": 400, "end_idx": 520, "actual_close_last": 102.5, "predicted_close_last": 103.1, "direction_accuracy": 0.6}
  ]
}
```

## Agent Usage Guidelines

1. **Always use `--json`** for programmatic consumption
2. **Load the model once** — the CLI caches loaded models in-memory
3. **Use `predict info`** to verify model availability before running predictions
4. **Batch predict** is faster than individual calls for multiple stocks
5. **Backtest results** include direction accuracy — a useful signal for strategy evaluation
6. **REPL mode** maintains session state across commands (model, predictions, history)
7. **Session persistence**: `save` / `load` in REPL to resume across sessions

## Error Handling

All commands exit with non-zero status on failure. JSON output includes error details:
```json
{"error": "Data file not found: ./missing.csv"}
```

Common errors:
- `Data file not found` — check `--data` path
- `Missing required columns` — CSV must have `open`, `high`, `low`, `close`
- `Model not found` — use `models` to see available options
- `CUDA not available` — fall back to CPU or install PyTorch with CUDA

## See Also

- Kronos paper: https://arxiv.org/abs/2508.02739
- HuggingFace models: https://huggingface.co/NeoQuasar
- CLI-Anything: https://github.com/HKUDS/CLI-Anything
