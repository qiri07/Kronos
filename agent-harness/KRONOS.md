# KRONOS CLI — Agent Harness Analysis

## Architecture Summary

Kronos is a **decoder-only foundation model for financial K-line (candlestick)
sequences**, pre-trained on data from 45+ global exchanges. It uses a two-stage
framework:

1. **KronosTokenizer**: A Transformer autoencoder + Binary Spherical Quantizer
   (BSQuantizer) that compresses continuous OHLCV data into hierarchical discrete
   tokens (s1 pre-tokens + s2 post-tokens).
2. **Kronos**: An autoregressive Transformer that predicts the next token in the
   sequence, enabling multi-step price forecasting.

The CLI wraps `KronosPredictor` — a high-level API that handles data normalization,
prediction, and inverse normalization — with structured Click commands.

```
┌─────────────────────────────────────────────────────┐
│              cli-anything-kronos                    │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────────┐ │
│  │  predict │ │ finetune │ │     backtest        │ │
│  │  run/batch│ │ train    │ │  run/summary        │ │
│  └────┬─────┘ └────┬─────┘ └──────────┬──────────┘ │
│       │             │                  │            │
│  ┌────┴─────────────┴──────────────────┴─────┐     │
│  │         KronosPredictor                   │     │
│  │  (normalize → encode → autoregressive     │     │
│  │   decode → inverse-normalize)             │     │
│  └──────────────────┬────────────────────────┘     │
│                     │                              │
│  ┌──────────────────┴────────────────────────┐     │
│  │  Kronos (Transformer) + KronosTokenizer   │     │
│  │  + BSQuantizer + HierarchicalEmbedding    │     │
│  └───────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

### Core Domains

| Domain | Module | Key Operations |
|--------|--------|----------------|
| Prediction | `core/predict.py` | Single prediction, batch prediction, model info |
| Finetuning | `core/finetune.py` | List models, train tokenizer, train predictor, full pipeline |
| Backtest | `core/backtest.py` | Forward-test backtest, result summary |
| Session | `core/session.py` | Model state, prediction history, undo/redo |
| Backend | `utils/kronos_backend.py` | Model loading, prediction engine, finetune wrappers |
| REPL | `kronos_cli.py` | Interactive mode, command routing, ReplSkin |

### Model Registry

Three built-in models, all loadable from HuggingFace:

| Name | Params | Max Context | Repo |
|------|--------|-------------|------|
| kronos-mini | 4.1M | 2048 | NeoQuasar/Kronos-mini |
| kronos-small | 24.7M | 512 | NeoQuasar/Kronos-small |
| kronos-base | 102.3M | 512 | NeoQuasar/Kronos-base |

Custom models (HuggingFace repo ids or local paths) are also supported.

### Data Flow

1. **Input**: CSV with OHLCV data + timestamps
2. **Normalization**: Zero-mean, unit-variance per-feature, clipped to ±5
3. **Tokenization**: KronosTokenizer.encode() → discrete token pairs (s1, s2)
4. **Prediction**: Autoregressive Transformer decoding → token sequences
5. **Inverse**: KronosTokenizer.decode() → normalized prices → denormalized

### Backend Integration

The CLI directly imports and calls the Kronos Python library:
- `model.Kronos`, `model.KronosTokenizer`, `model.KronosPredictor`
- `finetune_csv.config_loader.ConfigLoader`
- `finetune_csv.finetune_tokenizer.finetune_tokenizer_main`
- `finetune_csv.finetune_base_model.finetune_base_model_main`

No subprocess wrapping is needed — the CLI is a thin structured interface
around the native Python API.

### Rendering/Output Gap: None

Kronos outputs are pandas DataFrames (CSV), not visual artifacts. No rendering
gap exists. Output is always structured tabular data.

### Model as Hard Dependency

The Kronos library must be installed and importable. The CLI resolves the path
by checking the parent directory tree for a `Kronos/` project root and adding it
to `sys.path`.
