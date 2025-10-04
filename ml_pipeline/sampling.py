from imblearn.over_sampling import SMOTE

def check_imbalance(df, target_col):
    counts = df[target_col].value_counts()
    imbalance_ratio = counts.min() / counts.max()
    is_imbalanced = imbalance_ratio < 0.5
    return {
        "class_distribution": counts.to_dict(),
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": is_imbalanced
    }

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
