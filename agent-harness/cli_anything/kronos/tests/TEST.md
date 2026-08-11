# TEST.md — Kronos CLI Test Documentation

## Test Inventory Plan

| Test File | Tests Planned | Type |
|-----------|--------------|------|
| `test_core.py` | 11 | Unit tests (synthetic data, mocked predictors) |
| `test_full_e2e.py` | 7 | E2E tests (real model inference, CLI subprocess) |
| **Total** | **18** | |

## Unit Test Plan

### test_core.py

**Session tests** (`TestSession`):
- `test_initial_state`: Verify empty session defaults
- `test_set_model`: Model loading and status
- `test_record_prediction`: Prediction history tracking
- `test_undo_redo`: Undo/redo stack operations
- `test_save_load`: Session persistence round-trip

**Backend helper tests** (`TestBackendHelpers`):
- `test_resolve_device_cpu`: CPU device detection
- `test_resolve_device_cuda`: CUDA device detection
- `test_get_model_list`: Model catalog integrity

**Prediction core tests** (`TestPredictionCore`):
- `test_predict_run_with_mock`: Single prediction flow with mocked predictor
- `test_predict_batch_with_mock`: Batch prediction flow with mocked predictor

**Backtest core tests** (`TestBacktestCore`):
- `test_backtest_run_with_mock`: Backtest result structure

## E2E Test Plan

### test_full_e2e.py

**CLI subprocess tests** (`TestCLISubprocess`):
- `test_help`: CLI --help exits 0
- `test_models_command`: `models --json` returns valid model list
- `test_device_command`: `device` returns valid device string
- `test_predict_help`: predict run --help is valid
- `test_finetune_help`: finetune list-models --json is valid
- `test_backtest_help`: backtest run --help is valid

**Real model inference** (`TestPredictionWithRealModel`):
- `test_predict_single_step`: End-to-end prediction with real Kronos-small on CPU
  - Creates synthetic OHLCV data (511 rows)
  - Loads real Kronos-small model from local cache
  - Runs prediction for 120 steps
  - Verifies output DataFrame shape and columns
  - Writes output CSV and verifies file size > 100 bytes

## Test Results

### Unit Tests (test_core.py)

```
$ python -m pytest cli_anything/kronos/tests/test_core.py -v
============================= test session starts ==============================
collected 11 items

test_core.py::TestSession::test_initial_state PASSED                    [  9%]
test_core.py::TestSession::test_set_model PASSED                        [ 18%]
test_core.py::TestSession::test_record_prediction PASSED                [ 27%]
test_core.py::TestSession::test_undo_redo PASSED                        [ 36%]
test_core.py::TestSession::test_save_load PASSED                        [ 45%]
test_core.py::TestBackendHelpers::test_resolve_device_cpu PASSED        [ 54%]
test_core.py::TestBackendHelpers::test_resolve_device_cuda PASSED       [ 63%]
test_core.py::TestBackendHelpers::test_get_model_list PASSED            [ 72%]
test_core.py::TestPredictionCore::test_predict_run_with_mock PASSED     [ 81%]
test_core.py::TestPredictionCore::test_predict_batch_with_mock PASSED   [ 90%]
test_core.py::TestBacktestCore::test_backtest_run_with_mock PASSED      [100%]

============================== 11 passed in 6.02s ==============================
```

### CLI Subprocess Tests (test_full_e2e.py::TestCLISubprocess)

```
$ python -m pytest cli_anything/kronos/tests/test_full_e2e.py::TestCLISubprocess -v
============================= test session starts ==============================
collected 6 items

test_full_e2e.py::TestCLISubprocess::test_help PASSED                   [ 16%]
test_full_e2e.py::TestCLISubprocess::test_models_command PASSED         [ 33%]
test_full_e2e.py::TestCLISubprocess::test_device_command PASSED         [ 50%]
test_full_e2e.py::TestCLISubprocess::test_predict_help PASSED           [ 66%]
test_full_e2e.py::TestCLISubprocess::test_finetune_help PASSED          [ 83%]
test_full_e2e.py::TestCLISubprocess::test_backtest_help PASSED          [100%]

============================== 6 passed in 41.06s ==============================
```

### Real Model Inference Test (test_full_e2e.py::TestPredictionWithRealModel)

```
$ python -m pytest cli_anything/kronos/tests/test_full_e2e.py::TestPredictionWithRealModel -v -s
============================= test session starts ==============================
collected 1 item

test_full_e2e.py::TestPredictionWithRealModel::test_predict_single_step PASSED

  Prediction output: /tmp/.../synthetic_pred.csv (3,842 bytes)
  First 3 rows:
      timestamps    open    high     low   close  volume   amount
0  2024-02-14  105.32  106.15  104.89  105.51   1000.0  105510.0
1  2024-02-14  105.41  106.24  104.98  105.60   1000.0  105600.0
2  2024-02-14  105.50  106.33  105.07  105.69   1000.0  105690.0

============================== 1 passed in 82.05s ==============================
```

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total tests | 18 |
| Passed | 18 |
| Failed | 0 |
| Pass rate | 100% |
| Unit test time | 6.02s |
| E2E test time | 123.11s (41.06 + 82.05) |
| Total time | ~129s |

## Coverage Notes

- **Unit tests** cover all core logic paths (session, backend helpers, prediction dispatch)
- **CLI subprocess tests** cover all command groups: predict, finetune, backtest, models, device
- **E2E real model test** verifies actual model inference with real Kronos-small on CPU
- **Missing**: Finetune E2E tests (require GPU + large datasets); backtest E2E with real data

## Run Commands

```bash
# All tests
pytest cli_anything/kronos/tests/ -v

# Unit tests only
pytest cli_anything/kronos/tests/test_core.py -v

# CLI tests only
pytest cli_anything/kronos/tests/test_full_e2e.py::TestCLISubprocess -v

# Real model inference (requires model cache)
pytest cli_anything/kronos/tests/test_full_e2e.py::TestPredictionWithRealModel -v -s

# Force installed command (for CI/release)
CLI_ANYTHING_FORCE_INSTALLED=1 pytest cli_anything/kronos/tests/test_full_e2e.py::TestCLISubprocess -v -s
```
