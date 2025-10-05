# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-10-05
### Added
- Progress bar support with optional `tqdm` (install via extra `[progress]`).
- Extended metrics: confusion matrix and ROC AUC (when probabilities available) behind `--extended-metrics`.
- JSON logging output (`--json-logs`) and log file support (`--log-file`).
- Timing utilities (`timing.py`) for data split and per-model fit durations.
- Config-driven execution via YAML/JSON (`--config`), with precedence to CLI args.
- Optional extras groups: `progress`, `config`, `all`.

### Changed
- Training pipeline now uses timing contexts and optional progress bars.
- Logging system centralized with structured configuration.

### Fixed
- More graceful handling of edge cases around sampling and model training instrumentation.

## [0.1.0] - 2025-10-05
- Initial release with EDA, SMOTE handling, and baseline model training.

---
Format based on Keep a Changelog, adhering to semantic versioning.
