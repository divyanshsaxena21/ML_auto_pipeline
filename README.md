# ML AutoPipeline

Automated Machine Learning pipeline for beginners.

Features:
- Dataset loading and basic exploratory data analysis (EDA)
- Class imbalance detection and SMOTE sampling
- Model training on Logistic Regression, Random Forest, and SVM
- Performance metrics: Accuracy, Precision, Recall, F1 Score
- Simple CLI interface

## Installation

```bash
pip install ml-autopipeline
```

## Quick Start (CLI)

```bash
ml-autopipeline --file path/to/data.csv --target target_column --apply_smote
```

## Python Usage

```python
from ml_autopipeline import load_data, basic_report, check_imbalance, apply_smote, train_models

# Load data
import pandas as pd
# df = load_data("your.csv")  # or construct a dataframe directly

report = basic_report(df)
print(report["columns"])      # column names

imbalance = check_imbalance(df, "target")
print(imbalance["is_imbalanced"])

X = pd.get_dummies(df.drop(columns=["target"]))
y = df["target"]

if imbalance["is_imbalanced"]:
    X, y = apply_smote(X, y)

results = train_models(X, y)
print(results)
```

## Provided Functions

| Function | Description |
|----------|-------------|
| `load_data(path)` | Load CSV into a pandas DataFrame |
| `basic_report(df)` | Return shape, columns, missing values, dtypes, head |
| `check_imbalance(df, target)` | Report class distribution and imbalance flag |
| `apply_smote(X, y)` | Oversample minority classes using SMOTE |
| `train_models(X, y)` | Train Logistic Regression, Random Forest, SVM and return metrics |

## Versioning
A single source of truth for the version lives in `ml_autopipeline.__version__`.

## Development Setup

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell

pip install --upgrade pip
pip install -e .[dev]
pytest
```

## Build & Publish

```bash
pip install build twine
python -m build
# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*
# After verification
python -m twine upload dist/*
```

## License
MIT
