from .eda import load_data, basic_report
from .sampling import check_imbalance, apply_smote
from .training import train_models

__all__ = [
    "load_data",
    "basic_report",
    "check_imbalance",
    "apply_smote",
    "train_models",
    "__version__"
]

# Keep a single source of truth by importing from top-level package if available
try:
    from ml_autopipeline import __version__  # type: ignore  # circular import safe at runtime
except Exception:  # pragma: no cover
    __version__ = "0.1.0"
