"""Seed and reproducibility utilities for V2 experiments."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

