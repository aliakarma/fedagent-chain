"""Configuration loading and management utilities using Hydra and OmegaConf."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML configuration file as an OmegaConf DictConfig.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    DictConfig
        The loaded configuration object.

    Examples
    --------
    >>> cfg = load_config("configs/experiment/fedagent_chain_full.yaml")
    >>> print(cfg.federated.n_rounds)
    20
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)  # type: ignore[return-value]


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple OmegaConf configs, with later configs overriding earlier ones.

    Parameters
    ----------
    *configs : DictConfig
        Configuration objects to merge, in order of increasing priority.

    Returns
    -------
    DictConfig
        Merged configuration object.
    """
    base = OmegaConf.create({})
    for cfg in configs:
        base = OmegaConf.merge(base, cfg)  # type: ignore[assignment]
    return base


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert an OmegaConf DictConfig to a plain Python dictionary.

    Parameters
    ----------
    cfg : DictConfig
        OmegaConf configuration object.

    Returns
    -------
    dict
        Plain Python dictionary with all interpolations resolved.
    """
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore


def get_git_commit_hash() -> str:
    """Return the current git commit hash for reproducibility logging.

    Returns
    -------
    str
        Short git commit hash, or 'unknown' if not in a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"
