import argparse
import pandas as pd
from ml_autopipeline import load_data, basic_report, check_imbalance, apply_smote, train_models

def main():
    parser = argparse.ArgumentParser(description="ML Auto-Pipeline CLI")
    parser.add_argument('--file', type=str, help='CSV file path', required=True)
    parser.add_argument('--target', type=str, help='Target column name', required=True)
    parser.add_argument('--apply_smote', action='store_true', help='Apply SMOTE sampling')

    args = parser.parse_args()

    df = load_data(args.file)
    print(f"\nLoaded dataset with shape: {df.shape}")

    eda = basic_report(df)
    print("\n--- Basic EDA Report ---")
    print(f"Columns: {eda['columns']}")
    print(f"Missing values:\n{eda['missing_values']}")
    print(f"Data types:\n{eda['data_types']}")
    print(f"Sample data (first 5 rows):\n{pd.DataFrame(eda['head'])}")

    imbalance_report = check_imbalance(df, args.target)
    print("\n--- Imbalance Check ---")
    print(f"Class distribution: {imbalance_report['class_distribution']}")
    print(f"Imbalance ratio: {imbalance_report['imbalance_ratio']:.2f}")
    if imbalance_report['is_imbalanced']:
        print("Warning: Dataset is imbalanced.")
    else:
        print("Dataset class distribution is balanced.")

    X = pd.get_dummies(df.drop(columns=[args.target]))
    y = df[args.target]

    if imbalance_report['is_imbalanced'] and args.apply_smote:
        print("\nApplying SMOTE oversampling...")
        X, y = apply_smote(X, y)
        print("After sampling, class distribution:")
        print(pd.Series(y).value_counts())

    print("\n--- Training Models ---")
    results = train_models(X, y)
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.4f}")

if __name__ == "__main__":
    main()
