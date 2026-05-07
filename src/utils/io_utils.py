"""File I/O, checkpoint save/load, and serialization utilities."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data as a JSON file.

    Parameters
    ----------
    data : Any
        JSON-serializable data to save.
    path : str or Path
        Destination file path.
    indent : int
        JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    Any
        Deserialized data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(obj: Any, path: str | Path) -> None:
    """Save a Python object as a pickle checkpoint.

    Parameters
    ----------
    obj : Any
        Object to pickle.
    path : str or Path
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_checkpoint(path: str | Path) -> Any:
    """Load a Python object from a pickle checkpoint.

    Parameters
    ----------
    path : str or Path
        Path to the checkpoint file.

    Returns
    -------
    Any
        Deserialized object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return the Path object.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        The created or existing directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pytorch_model(model: Any, path: str | Path) -> None:
    """Save a PyTorch model's state dict.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to save.
    path : str or Path
        Destination file path (should end in .pt or .pth).
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_pytorch_model(model: Any, path: str | Path, device: str = "cpu") -> Any:
    """Load a PyTorch model's state dict.

    Parameters
    ----------
    model : torch.nn.Module
        Model architecture to load weights into.
    path : str or Path
        Path to the saved state dict.
    device : str
        Device to map the checkpoint to.

    Returns
    -------
    torch.nn.Module
        Model with loaded weights.
    """
    import torch

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model
