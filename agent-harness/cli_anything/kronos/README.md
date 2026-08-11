# Kronos CLI — Agent-Native Harness

## What is Kronos?

Kronos is a **foundation model for financial K-line (candlestick) sequences**,
trained on data from over 45 global exchanges. It uses a two-stage framework:
a specialized tokenizer quantizes continuous OHLCV data into hierarchical discrete
tokens, then a decoder-only Transformer is pre-trained autoregressively on those
tokens.

This CLI wraps the Kronos model library with structured commands for prediction,
finetuning, and backtesting — all with `--json` output for agent consumption.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (with CUDA/MPS support optional)
- The Kronos project installed: `pip install -e /path/to/Kronos`

### Install this CLI

```bash
cd agent-harness
pip install -e .
```

### Verify

```bash
cli-anything-kronos --help
cli-anything-kronos models
cli-anything-kronos device
```

## Quick Start

### Predict a single stock

```bash
cli-anything-kronos predict run \
  --data ./data/HK_ali_09988_kline_5min_all.csv \
  --lookback 400 \
  --pred-len 120 \
  --output ./prediction_out.csv \
  --json
```

### Batch predict all CSVs in a directory

```bash
cli-anything-kronos predict batch \
  --data-dir ./data/ \
  --lookback 400 \
  --pred-len 60 \
  --output-dir ./predictions/ \
  --json
```

### List available models

```bash
cli-anything-kronos finetune list-models
cli-anything-kronos models
```

### Run a backtest

```bash
cli-anything-kronos backtest run \
  --data ./data/HK_ali_09988_kline_5min_all.csv \
  --symbol 09988 \
  --lookback 400 \
  --pred-len 120 \
  --output ./backtest_result.json \
  --json
```

### Interactive REPL

```bash
cli-anything-kronos
```

Inside the REPL:
```
kronos> model list
kronos> model load kronos-small
kronos> predict run --data ./data.csv --lookback 400 --pred-len 120
kronos> backtest run --data ./data.csv --symbol 09988
kronos> status
kronos> save
kronos> exit
```

## Command Groups

### `predict`
| Command | Description |
|---------|-------------|
| `predict run` | Single stock prediction from CSV |
| `predict batch` | Batch prediction across directory of CSVs |
| `predict info` | Load and inspect a model |

### `finetune`
| Command | Description |
|---------|-------------|
| `finetune list-models` | List available pre-trained models |
| `finetune train` | Full finetune pipeline (tokenizer + predictor) |
| `finetune train-tokenizer` | Train tokenizer only |
| `finetune train-predictor` | Train predictor only |

### `backtest`
| Command | Description |
|---------|-------------|
| `backtest run` | Forward-test backtest on historical data |
| `backtest summary` | Summarize a previous backtest result |

### `session` (REPL only)
| Command | Description |
|---------|-------------|
| `status` | Show current model and prediction state |
| `save [PATH]` | Save session to file |
| `load PATH` | Load session from file |
| `undo` | Undo last action |
| `redo` | Redo last action |
| `clear` | Clear session state |

## JSON Output

Every command supports `--json` for machine-readable output. Agents should prefer
`--json` mode for programmatic consumption.

Example prediction output:
```json
{
  "data_path": "./data.csv",
  "lookback": 400,
  "pred_len": 120,
  "rows": 120,
  "columns": ["open", "high", "low", "close", "volume", "amount"],
  "sample_count": 1,
  "T": 1.0,
  "top_p": 0.9,
  "head": [{"timestamps": "...", "open": 100.5, ...}, ...]
}
```

## Model Catalog

| Model | Params | Max Context | HuggingFace Repo |
|-------|--------|-------------|------------------|
| kronos-mini | 4.1M | 2048 | NeoQuasar/Kronos-mini |
| kronos-small | 24.7M | 512 | NeoQuasar/Kronos-small |
| kronos-base | 102.3M | 512 | NeoQuasar/Kronos-base |

## Data Format

Input CSV must contain at minimum:
- `open`, `high`, `low`, `close` — OHLC price columns
- `timestamps` — datetime column (auto-generated if missing)

Optional columns:
- `volume` — trading volume (default: 0)
- `amount` — trading amount (default: volume × mean price)

## Finetuning

Finetuning requires a YAML config file. See `Kronos/finetune_csv/configs/` for
examples. Key fields:

```yaml
model_name: "NeoQuasar/Kronos-small"
tokenizer_name: "NeoQuasar/Kronos-Tokenizer-base"
data_path: "./data/your_data.csv"
lookback_window: 400
predict_window: 120
max_context: 512
epochs: 30
batch_size: 32
save_path: "./outputs/models/my_finetune"
```

Run finetuning:
```bash
cli-anything-kronos finetune train --config ./my_config.yaml --json
```

## Notes for AI Agents

- **Every command supports `--json`** — always use this for programmatic use
- **Model loading is cached** — loading the same model twice reuses the in-memory instance
- **Session state is persisted** across REPL commands via undo/redo
- **Batch prediction** requires all input files to have consistent historical lengths
- **Backtest** uses a simple forward-test walk-forward approach; for production
  backtesting, use the Kronos `run_backtest_kronos.py` script directly
