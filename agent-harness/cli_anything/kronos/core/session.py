"""Kronos CLI - Session management with model state and prediction history."""

from __future__ import annotations

import json
import os
import copy
from typing import Any, Optional
from datetime import datetime


def _locked_save_json(path: str, data: Any, **dump_kwargs) -> None:
    """Atomically write JSON with exclusive file locking."""
    try:
        f = open(path, "r+")
    except FileNotFoundError:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        f = open(path, "w")
    with f:
        _locked = False
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            _locked = True
        except (ImportError, OSError):
            pass
        try:
            f.seek(0)
            f.truncate()
            json.dump(data, f, **dump_kwargs, default=str)
            f.flush()
        finally:
            if _locked:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class Session:
    """Manages Kronos session state with undo/redo for prediction contexts."""

    MAX_UNDO = 30

    def __init__(self):
        self.model_name: Optional[str] = None
        self.model_path: Optional[str] = None
        self.device: Optional[str] = None
        self.max_context: Optional[int] = None
        self.last_prediction: Optional[dict] = None
        self.predictions: list[dict] = []
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._modified: bool = False

    # -- State queries --------------------------------------------------------

    def has_model(self) -> bool:
        return self.model_name is not None

    def status(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "max_context": self.max_context,
            "n_predictions": len(self.predictions),
            "modified": self._modified,
            "undo_depth": len(self._undo_stack),
            "redo_depth": len(self._redo_stack),
        }

    # -- Model management -----------------------------------------------------

    def set_model(self, name: str, path: Optional[str], device: str, max_context: int) -> None:
        self._undo_stack.append({
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "max_context": self.max_context,
        })
        self._redo_stack.clear()
        self.model_name = name
        self.model_path = path
        self.device = device
        self.max_context = max_context
        self._modified = True

    # -- Prediction history ---------------------------------------------------

    def record_prediction(self, result: dict) -> None:
        self._undo_stack.append({
            "predictions": copy.deepcopy(self.predictions),
            "last_prediction": copy.deepcopy(self.last_prediction),
        })
        self._redo_stack.clear()
        self.predictions.append(result)
        self.last_prediction = result
        self._modified = True

    def undo(self) -> Optional[dict]:
        if not self._undo_stack:
            return None
        state = self._undo_stack.pop()
        self._redo_stack.append({
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "max_context": self.max_context,
            "predictions": copy.deepcopy(self.predictions),
            "last_prediction": copy.deepcopy(self.last_prediction),
        })
        self.model_name = state.get("model_name")
        self.model_path = state.get("model_path")
        self.device = state.get("device")
        self.max_context = state.get("max_context")
        self.predictions = state.get("predictions", [])
        self.last_prediction = state.get("last_prediction")
        self._modified = bool(self._undo_stack) or bool(self._redo_stack)
        return {"action": "undo", "n_predictions_left": len(self.predictions)}

    def redo(self) -> Optional[dict]:
        if not self._redo_stack:
            return None
        state = self._redo_stack.pop()
        self._undo_stack.append({
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "max_context": self.max_context,
            "predictions": copy.deepcopy(self.predictions),
            "last_prediction": copy.deepcopy(self.last_prediction),
        })
        self.model_name = state.get("model_name")
        self.model_path = state.get("model_path")
        self.device = state.get("device")
        self.max_context = state.get("max_context")
        self.predictions = state.get("predictions", [])
        self.last_prediction = state.get("last_prediction")
        self._modified = bool(self._undo_stack) or bool(self._redo_stack)
        return {"action": "redo", "n_predictions": len(self.predictions)}

    # -- Persistence ----------------------------------------------------------

    def save(self, path: str) -> None:
        _locked_save_json(path, self._to_dict())

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Session file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        self.model_name = data.get("model_name")
        self.model_path = data.get("model_path")
        self.device = data.get("device")
        self.max_context = data.get("max_context")
        self.predictions = data.get("predictions", [])
        self.last_prediction = data.get("last_prediction")

    def _to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "max_context": self.max_context,
            "predictions": self.predictions,
            "last_prediction": self.last_prediction,
            "modified": self._modified,
            "saved_at": datetime.now().isoformat(),
        }
