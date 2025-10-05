import argparse
import pandas as pd
from ml_autopipeline import load_data, basic_report, check_imbalance, apply_smote, train_models
from ml_pipeline.logging_utils import configure_logging, get_logger

logger = get_logger("cli")

def main():
    parser = argparse.ArgumentParser(description="ML Auto-Pipeline CLI")
    parser.add_argument('--file', type=str, help='CSV file path', required=True)
    parser.add_argument('--target', type=str, help='Target column name', required=True)
    parser.add_argument('--apply_smote', action='store_true', help='Apply SMOTE sampling')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (-v, -vv for more)')

    args = parser.parse_args()

    # Verbosity mapping
    if args.verbose >= 2:
        level = 10  # DEBUG
    elif args.verbose == 1:
        level = 20  # INFO
    else:
        level = 30  # WARNING (quiet default)
    configure_logging(level=level)
    logger.info("Logger configured")

    df = load_data(args.file)
    logger.info(f"Loaded dataset shape={df.shape}")

    eda = basic_report(df)
    logger.info("Generated basic EDA report")
    if level <= 20:
        logger.info(f"Columns: {eda['columns']}")
        logger.info(f"Missing values: {eda['missing_values']}")
        logger.info(f"Data types: {eda['data_types']}")
    if level <= 10:
        logger.debug(f"Head: {pd.DataFrame(eda['head'])}")

    imbalance_report = check_imbalance(df, args.target)
    logger.info(f"Class distribution: {imbalance_report['class_distribution']}")
    logger.info(f"Imbalance ratio: {imbalance_report['imbalance_ratio']:.2f}")
    if imbalance_report['is_imbalanced']:
        logger.warning("Dataset is imbalanced")

    X = pd.get_dummies(df.drop(columns=[args.target]))
    y = df[args.target]

    if imbalance_report['is_imbalanced'] and args.apply_smote:
        logger.info("Applying SMOTE oversampling")
        X, y = apply_smote(X, y)
        logger.info(f"Post-sampling distribution: {pd.Series(y).value_counts().to_dict()}")

    logger.info("Training models ...")
    results = train_models(X, y)
    for model_name, metrics in results.items():
        logger.info(f"Model: {model_name}")
        for metric, score in metrics.items():
            logger.info(f"  {metric}: {score:.4f}")

if __name__ == "__main__":
    main()
