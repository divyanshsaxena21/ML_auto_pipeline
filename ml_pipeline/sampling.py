from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
from .logging_utils import get_logger

logger = get_logger("sampling")

def check_imbalance(df, target_col):
    logger.info(f"Checking class imbalance for target='{target_col}'")
    counts = df[target_col].value_counts()
    imbalance_ratio = counts.min() / counts.max()
    is_imbalanced = imbalance_ratio < 0.5
    logger.debug(f"Class distribution: {counts.to_dict()} ratio={imbalance_ratio:.3f} is_imbalanced={is_imbalanced}")
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
        logger.warning("apply_smote called with empty labels; returning inputs unchanged")
        return X, y  # empty input safeguard
    minority_count = min(counts.values())
    majority_count = max(counts.values())
    logger.info(f"Applying sampling strategy minority={minority_count} majority={majority_count}")

    # Nothing to do if already balanced or trivially small
    if minority_count == majority_count:
        logger.info("Data already balanced; skipping SMOTE")
        return X, y

    # If only one minority sample, SMOTE cannot work; fallback to simple duplication
    if minority_count < 2:
        logger.warning("Minority class has only 1 sample; using RandomOverSampler fallback")
        ros = RandomOverSampler(random_state=42)
        return ros.fit_resample(X, y)

    # Adjust k_neighbors dynamically to avoid ValueError
    k_neighbors = min(5, minority_count - 1)
    logger.debug(f"Using SMOTE with k_neighbors={k_neighbors}")
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    return smote.fit_resample(X, y)
