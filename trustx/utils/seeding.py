"""Reproducibility helpers."""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
