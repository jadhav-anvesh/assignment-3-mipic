"""Checkpointing and model state persistence for reproducible runs."""

from pathlib import Path
from typing import Any, Dict, Optional
import torch


def save_initial_weights(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    """Save initial model weights to ensure identical starting conditions across all runs."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_initial_weights(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    """Load pre-saved initial model weights."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Initial weights checkpoint not found: {path}")
    state_dict = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    model.load_state_dict(state_dict)


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save full training checkpoint with optimizer, scheduler, and metadata."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "metadata": metadata or {},
    }
    torch.save(payload, save_path)

