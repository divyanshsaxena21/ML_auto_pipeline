import argparse
import pandas as pd
from ml_autopipeline import load_data, basic_report, check_imbalance, apply_smote, train_models
from ml_pipeline.logging_utils import configure_logging, get_logger
from ml_pipeline.config_loader import load_config, merge_config, ConfigError

logger = get_logger("cli")


def parse_args():
    parser = argparse.ArgumentParser(description="ML Auto-Pipeline CLI")
    parser.add_argument('--file', type=str, help='CSV file path', required=False)
    parser.add_argument('--target', type=str, help='Target column name', required=False)
    parser.add_argument('--apply_smote', action='store_true', help='Apply SMOTE sampling')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (-v, -vv for more)')
    parser.add_argument('--config', type=str, help='YAML/JSON config file specifying arguments')
    parser.add_argument('--log-file', type=str, help='Path to log file (appended)')
    parser.add_argument('--json-logs', action='store_true', help='Emit logs in JSON format')
    parser.add_argument('--progress', action='store_true', help='Show training progress bar (requires tqdm)')
    parser.add_argument('--extended-metrics', action='store_true', help='Include confusion matrix and ROC AUC when possible')
    return parser.parse_args()


def verbosity_to_level(vcount: int) -> int:
    if vcount >= 2:
        return 10  # DEBUG
    if vcount == 1:
        return 20  # INFO
    return 30  # WARNING


def main():
    args = parse_args()
    cli_dict = vars(args)

    # Load and merge configuration if provided
    if args.config:
        try:
            cfg = load_config(args.config)
        except ConfigError as e:
            raise SystemExit(f"Config error: {e}")
        merged = merge_config(cli_dict, cfg, precedence="cli")
    else:
        merged = cli_dict

    # Validate required fields after merge
    required_fields = ['file', 'target']
    missing = [f for f in required_fields if not merged.get(f)]
    if missing:
        raise SystemExit(f"Missing required arguments after merge: {missing}. Provide via CLI or config file.")

    level = verbosity_to_level(merged.get('verbose', 0))
    configure_logging(level=level, json_logs=merged.get('json_logs'), log_file=merged.get('log_file'))
    logger.info("Logger configured")
    if merged.get('config'):
        logger.info(f"Loaded config file: {merged['config']}")

    df = load_data(merged['file'])
    logger.info(f"Loaded dataset shape={df.shape}")

    eda = basic_report(df)
    logger.info("Generated basic EDA report")
    if level <= 20:
        logger.info(f"Columns: {eda['columns']}")
        logger.info(f"Missing values: {eda['missing_values']}")
        logger.info(f"Data types: {eda['data_types']}")
    if level <= 10:
        logger.debug(f"Head: {pd.DataFrame(eda['head'])}")

    imbalance_report = check_imbalance(df, merged['target'])
    logger.info(f"Class distribution: {imbalance_report['class_distribution']}")
    logger.info(f"Imbalance ratio: {imbalance_report['imbalance_ratio']:.2f}")
    if imbalance_report['is_imbalanced']:
        logger.warning("Dataset is imbalanced")

    X = pd.get_dummies(df.drop(columns=[merged['target']]))
    y = df[merged['target']]

    if imbalance_report['is_imbalanced'] and merged.get('apply_smote'):
        logger.info("Applying SMOTE oversampling")
        X, y = apply_smote(X, y)
        logger.info(f"Post-sampling distribution: {pd.Series(y).value_counts().to_dict()}")

    logger.info("Training models ...")
    results = train_models(
        X,
        y,
        show_progress=merged.get('progress', False),
        extended_metrics=merged.get('extended_metrics', False),
    )
    for model_name, metrics in results.items():
        logger.info(f"Model: {model_name}")
        for metric, score in metrics.items():
            logger.info(f"  {metric}: {score}")

if __name__ == "__main__":
    main()
