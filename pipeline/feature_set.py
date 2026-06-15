"""Feature set management — load, validate, and list YAML feature set definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from rl_hmm import FeatureRegistry

# Root of RL-GHMM project
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Default search path for feature set YAMLs
_SETS_DIR = _PROJECT_ROOT / "rl_hmm" / "features" / "sets"


@dataclass
class FeatureSet:
    """Represents a feature set loaded from a YAML file."""
    name: str
    description: str
    features: List[str]
    signal_interval: str = "15m"
    early_minutes: int = 0
    data_path: str = "data/btc_5m_20240401_20260111.csv"
    db_path: Optional[str] = None
    exchange: Optional[str] = None
    symbol: Optional[str] = None
    extra: dict = field(default_factory=dict)


_VALID_INTERVALS = {"5m", "15m", "30m", "1h"}


def _get_sets_dir() -> Path:
    """Return the feature sets directory."""
    d = _SETS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_feature_sets() -> List[str]:
    """List all available feature set names (YAML files without extension)."""
    sets_dir = _get_sets_dir()
    return sorted(
        f.stem for f in sets_dir.glob("*.yaml") if f.is_file()
    )


def load_feature_set(name: str) -> FeatureSet:
    """Load and validate a feature set by name.

    Looks for {name}.yaml in the feature sets directory.
    Validates that all feature names exist in the registry.
    """
    yaml_path = _get_sets_dir() / f"{name}.yaml"
    if not yaml_path.exists():
        available = ", ".join(list_feature_sets())
        raise FileNotFoundError(
            f"Feature set '{name}' not found. Available: {available}"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "features" not in data:
        raise ValueError(f"Invalid feature set YAML: {yaml_path} — missing 'features' key")

    # Validate feature names against the registry
    available_features = set(FeatureRegistry.get_all_features())
    unknown = [f for f in data["features"] if f not in available_features]
    if unknown:
        raise ValueError(
            f"Unknown features in '{name}': {unknown}. "
            f"Available: {sorted(available_features)}"
        )

    # Validate signal_interval
    interval = data.get("signal_interval", "15m")
    if interval not in _VALID_INTERVALS:
        raise ValueError(
            f"Invalid signal_interval '{interval}' in '{name}'. "
            f"Must be one of {_VALID_INTERVALS}"
        )

    extra = {k: v for k, v in data.items()
             if k not in ("name", "description", "features", "signal_interval",
                          "early_minutes", "data_path", "db_path",
                          "exchange", "symbol")}

    return FeatureSet(
        name=data.get("name", name),
        description=data.get("description", ""),
        features=data["features"],
        signal_interval=interval,
        early_minutes=data.get("early_minutes", 0),
        data_path=data.get("data_path", "data/btc_5m_20240401_20260111.csv"),
        db_path=data.get("db_path"),
        exchange=data.get("exchange"),
        symbol=data.get("symbol"),
        extra=extra,
    )
