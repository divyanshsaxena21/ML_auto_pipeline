from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter

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
    """Apply SMOTE with graceful fallback when the minority class is too small.

    SMOTE requires at least k_neighbors + 1 samples in the minority class. The
    default k_neighbors=5 therefore needs at least 6 samples. The test dataset
    includes only a single minority example, so plain SMOTE fails. We handle:

    - minority count == 1: fall back to RandomOverSampler to duplicate samples
      up to the majority class size.
    - 2 <= minority count <= 5: reduce k_neighbors to minority_count - 1.
    - otherwise: use standard SMOTE (k_neighbors=5).
    """
    counts = Counter(y)
    if not counts:
        return X, y  # empty input safeguard
    minority_count = min(counts.values())
    majority_count = max(counts.values())

    # Nothing to do if already balanced or trivially small
    if minority_count == majority_count:
        return X, y

    # If only one minority sample, SMOTE cannot work; fallback to simple duplication
    if minority_count < 2:
        ros = RandomOverSampler(random_state=42)
        return ros.fit_resample(X, y)

    # Adjust k_neighbors dynamically to avoid ValueError
    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    return smote.fit_resample(X, y)
