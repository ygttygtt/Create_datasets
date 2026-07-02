"""Configuration loader with env-var substitution."""

import os
import re
from pathlib import Path
from typing import Any

import yaml


def _substitute_env(value: Any) -> Any:
    """Recursively substitute ${VAR} patterns in config values."""
    if isinstance(value, str):
        def _replacer(match: re.Match) -> str:
            var_name = match.group(1)
            env_val = os.environ.get(var_name, "")
            if not env_val:
                print(f"[WARN] Env var '{var_name}' is not set, using empty string.")
            return env_val
        return re.sub(r'\$\{(\w+)\}', _replacer, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def load_config(config_path: str | Path) -> dict:
    """Load and validate a YAML config file, substituting env vars."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = _substitute_env(raw)
    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    """Basic validation of required config sections."""
    required_sections = ["llm", "rate_limit", "generation", "quality"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing required section: [{section}]")

    llm = config["llm"]
    if not llm.get("api_key"):
        print("[WARN] llm.api_key is empty — API calls will fail. "
              "Set the env var or use --dry-run for testing.")
    if not llm.get("model"):
        raise ValueError("llm.model is required.")

    gen = config["generation"]
    valid_formats = ("llamafactory", "sharegpt", "dpo")
    if gen.get("output_format") not in valid_formats:
        raise ValueError(
            f"Unknown output_format: {gen.get('output_format')}. "
            f"Use one of: {', '.join(valid_formats)}."
        )
    valid_modes = ("sft", "dpo")
    if gen.get("mode", "sft") not in valid_modes:
        raise ValueError(
            f"Unknown mode: {gen.get('mode')}. Use 'sft' or 'dpo'."
        )

    # Set defaults for optional fields
    config.setdefault("templates", {})
