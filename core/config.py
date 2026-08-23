"""Configuration management: YAML loading, environment overrides, defaults."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "server": {"host": "0.0.0.0", "port": 8000, "workers": 4},
    "search": {"top_k": 10, "min_score": 0.05, "rerank": True},
    "crawler": {"max_depth": 3, "delay_ms": 500, "user_agent": "DeepSearchBot/0.1"},
    "index": {"chunk_size": 512, "chunk_overlap": 64, "vector_dim": 384},
}


@dataclass
class Config:
    """Typed application configuration with nested sections."""

    server: dict = field(default_factory=lambda: dict(DEFAULTS["server"]))
    search: dict = field(default_factory=lambda: dict(DEFAULTS["search"]))
    crawler: dict = field(default_factory=lambda: dict(DEFAULTS["crawler"]))
    index: dict = field(default_factory=lambda: dict(DEFAULTS["index"]))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load config from YAML file, apply env overrides, fall back to defaults."""
        data: dict[str, Any] = {}
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        cfg = cls()
        for section, values in cfg.__dict__.items():
            values.update(data.get(section, {}))
            env_prefix = f"DEEPSEARCH_{section.upper()}_"
            for key in list(values):
                env_val = os.getenv(env_prefix + key.upper())
                if env_val is not None:
                    values[key] = _coerce(env_val)
        return cfg

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Fetch a value using dot notation, e.g. ``search.top_k``."""
        node: Any = self
        for part in dotted_key.split("."):
            node = getattr(node, part, None) if isinstance(node, Config) else (node or {}).get(part)
            if node is None:
                return default
        return node


def _coerce(raw: str) -> Any:
    """Convert an environment string to int, float, bool, or keep as str."""
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


_config: Config | None = None


def get_config(path: str | Path | None = None) -> Config:
    """Return the process-wide singleton configuration."""
    global _config
    if _config is None:
        _config = Config.load(path or os.getenv("DEEPSEARCH_CONFIG"))
    return _config
