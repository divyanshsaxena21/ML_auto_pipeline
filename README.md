# ML AutoPipeline

Automated Machine Learning pipeline for beginners.

Features:
- Dataset loading and basic exploratory data analysis (EDA)
- Class imbalance detection and SMOTE sampling
- Model training on Logistic Regression, Random Forest, and SVM
- Performance metrics: Accuracy, Precision, Recall, F1 Score
- Optional extended metrics (confusion matrix, ROC AUC)
- Structured logging (plain or JSON) + timing + optional progress bars
- Config-driven pipeline (YAML/JSON) & CLI flags

## Installation

```bash
pip install ml-autopipeline
# With optional extras
pip install "ml-autopipeline[all]"     # progress + config
pip install "ml-autopipeline[progress]" # just tqdm
pip install "ml-autopipeline[config]"   # just PyYAML
```

## Quick Start (CLI)

```bash
ml-autopipeline --file path/to/data.csv --target target_column --apply_smote
```

Add verbosity:
```bash
ml-autopipeline --file data.csv --target label -v      # INFO
ml-autopipeline --file data.csv --target label -vv     # DEBUG
```

Extended metrics & progress bar (needs extras installed):
```bash
ml-autopipeline --file data.csv --target label --extended-metrics --progress
```

JSON logs to file:
```bash
ml-autopipeline --file data.csv --target label --json-logs --log-file run.log
```

Use a config file (values can be overridden by CLI):
```yaml
# config.yml
file: data.csv
target: label
apply_smote: true
extended_metrics: true
progress: true
verbose: 1
```
Run with config:
```bash
ml-autopipeline --config config.yml
```

## Python Usage

```python
from ml_autopipeline import (
    load_data, basic_report, check_imbalance, apply_smote, train_models
)

# df = load_data("your.csv")
report = basic_report(df)
imbalance = check_imbalance(df, "target")
X = pd.get_dummies(df.drop(columns=["target"]))
y = df["target"]
if imbalance["is_imbalanced"]:
    X, y = apply_smote(X, y)
results = train_models(X, y, extended_metrics=True, show_progress=True)
print(results)
```

## Provided Functions

| Function | Description |
|----------|-------------|
| `load_data(path)` | Load CSV into a pandas DataFrame |
| `basic_report(df)` | Return shape, columns, missing values, dtypes, head |
| `check_imbalance(df, target)` | Report class distribution and imbalance flag |
| `apply_smote(X, y)` | Oversample minority classes using SMOTE / fallback strategy |
| `train_models(X, y, ...)` | Train models and return metrics (+ optional extended metrics) |

Extended metrics keys (when enabled):
- `confusion_matrix`: 2D list
- `roc_auc` or `roc_auc_ovr_weighted` (if probabilities available)

## Logging & Timing
- Default log level: WARNING
- `-v` -> INFO, `-vv` -> DEBUG
- `--json-logs` produces JSON per line
- `--log-file FILE` duplicates logs to file
- Timing for major steps included (split, fit per model)

## Config Precedence
1. CLI arguments (if provided)
2. Config file values

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev,all]
pytest
```

## Build & Publish

```bash
pip install build twine
python -m build
python -m twine upload --repository testpypi dist/*
# After verification
python -m twine upload dist/*
```

## License
MIT
