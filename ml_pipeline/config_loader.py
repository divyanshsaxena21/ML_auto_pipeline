import json
from pathlib import Path
from typing import Any, Dict
from .logging_utils import get_logger

logger = get_logger("config")

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

SUPPORTED_SUFFIXES = {".yml", ".yaml", ".json"}

class ConfigError(Exception):
    pass

def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ConfigError(f"Unsupported config extension: {p.suffix}; use .yml/.yaml/.json")
    logger.info(f"Loading config file {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise ConfigError("PyYAML not installed. Install with: pip install ml-autopipeline[config]")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or '{}')
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping/object")
    return data

def merge_config(cli_args: Dict[str, Any], cfg: Dict[str, Any], precedence: str = "cli") -> Dict[str, Any]:
    """Merge config with CLI args. Precedence 'cli' keeps CLI provided values.

    Only includes keys present either in cfg or cli_args.
    """
    result = dict(cfg)
    for k, v in cli_args.items():
        if precedence == "cli" and v is not None:
            result[k] = v
        elif k not in result:
            result[k] = v
    return result
