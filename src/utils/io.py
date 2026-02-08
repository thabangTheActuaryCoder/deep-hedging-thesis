"""I/O utilities for saving/loading data and configs."""
import os
import json
import hashlib
import torch
import numpy as np


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(data, path):
    """Save data as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_tensor(tensor, path):
    """Save a tensor to disk."""
    ensure_dir(os.path.dirname(path))
    torch.save(tensor, path)


def load_tensor(path):
    """Load a tensor from disk."""
    return torch.load(path, weights_only=True)


def save_checkpoint(state_dict, path):
    """Save model checkpoint."""
    ensure_dir(os.path.dirname(path))
    torch.save(state_dict, path)


def load_checkpoint(path, device="cpu"):
    """Load model checkpoint."""
    return torch.load(path, map_location=device, weights_only=True)


def compute_split_hash(train_idx, val_idx, test_idx):
    """Hash split indices for reproducibility tracking."""
    data = np.concatenate([train_idx, val_idx, test_idx]).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _json_default(obj):
    """Default JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)
