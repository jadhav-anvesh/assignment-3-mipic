"""Logging and experiment tracking utilities for V2."""

import json
from pathlib import Path
from typing import Any, Dict, List
import time


class ExperimentLogger:
    """Logs progress to both terminal and a structured train_log.json."""

    def __init__(self, log_path: str | Path | None = None, verbose: bool = True):
        self.log_path = Path(log_path) if log_path else None
        self.verbose = verbose
        self.history: List[Dict[str, Any]] = []
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def log_step(self, step_data: Dict[str, Any]) -> None:
        step_data["timestamp"] = time.time()
        self.history.append(step_data)
        if self.log_path:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

    def save(self) -> None:
        if self.log_path:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

