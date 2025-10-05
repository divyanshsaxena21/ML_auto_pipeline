from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
from sklearn.exceptions import ConvergenceWarning
from .logging_utils import get_logger
from .timing import timed

logger = get_logger("training")

try:  # optional dependency
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # fallback

def _compute_extended_metrics(y_true, y_pred, model, y_proba, average="weighted"):
    metrics = {}
    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
    except Exception as e:  # pragma: no cover
        logger.debug(f"Could not compute confusion matrix: {e}")
    # ROC AUC (only if probabilities and binary or multiclass)
    if y_proba is not None:
        try:
            if y_proba.ndim == 1:  # binary as scores
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            else:
                # multiclass: use one-vs-rest
                metrics["roc_auc_ovr_weighted"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
        except Exception as e:  # pragma: no cover
            logger.debug(f"Could not compute ROC AUC: {e}")
    return metrics

def train_models(
    X,
    y,
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = True,
    lr_max_iter: int = 2000,
    scale_linear_models: bool = True,
    svm_kernel: str = "rbf",
    svm_probability: bool = False,
    show_progress: bool = False,
    extended_metrics: bool = False,
):
    """Train a suite of baseline models and return evaluation metrics.

    Parameters:
        X, y: Features and target
        test_size: Fraction reserved for test split
        random_state: Seed for reproducibility
        stratify: Whether to stratify by y (ignored if False)
        lr_max_iter: Max iterations for Logistic Regression (increase to reduce convergence warnings)
        scale_linear_models: If True, apply StandardScaler before LR and SVM
        svm_kernel: Kernel for SVC
        svm_probability: Enable probability estimates (slower)
        show_progress: If True and tqdm available, show a progress bar for models
        extended_metrics: If True, include confusion matrix and ROC AUC (if possible)
    """
    logger.info(
        f"Starting training pipeline test_size={test_size} stratify={stratify} scale_linear_models={scale_linear_models} extended_metrics={extended_metrics}"
    )
    stratify_arg = y if stratify else None
    with timed("data_split"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
        )
    logger.debug(
        f"Train shape={getattr(X_train, 'shape', None)} Test shape={getattr(X_test, 'shape', None)}"
    )

    # Pipelines with optional scaling
    if scale_linear_models:
        logger.debug("Building pipelines with StandardScaler for LR & SVM")
        lr_model = Pipeline([
            ("scaler", StandardScaler(with_mean=False) if hasattr(X_train, "sparse") else StandardScaler()),
            ("clf", LogisticRegression(max_iter=lr_max_iter))
        ])
        svm_model = Pipeline([
            ("scaler", StandardScaler(with_mean=False) if hasattr(X_train, "sparse") else StandardScaler()),
            ("clf", SVC(kernel=svm_kernel, probability=svm_probability))
        ])
    else:
        lr_model = LogisticRegression(max_iter=lr_max_iter)
        svm_model = SVC(kernel=svm_kernel, probability=svm_probability)

    models = {
        "Logistic Regression": lr_model,
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "SVM": svm_model,
    }

    iterable = models.items()
    if show_progress and tqdm is not None:
        iterable = tqdm(iterable, desc="Training models", total=len(models))

    results = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for name, model in iterable:
            with timed(f"fit_{name.replace(' ', '_').lower()}"):
                logger.info(f"Fitting model: {name}")
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                # Probabilities or decision scores
                proba = None
                if extended_metrics:
                    if hasattr(model, "predict_proba"):
                        try:
                            proba = model.predict_proba(X_test)
                        except Exception:  # pragma: no cover
                            proba = None
                    elif hasattr(model, "decision_function"):
                        try:
                            proba = model.decision_function(X_test)
                        except Exception:  # pragma: no cover
                            proba = None
                base_metrics = {
                    "Accuracy": accuracy_score(y_test, preds),
                    "Precision": precision_score(y_test, preds, average='weighted', zero_division=0),
                    "Recall": recall_score(y_test, preds, average='weighted', zero_division=0),
                    "F1 Score": f1_score(y_test, preds, average='weighted', zero_division=0)
                }
                if extended_metrics:
                    ext = _compute_extended_metrics(y_test, preds, model, proba)
                    base_metrics.update(ext)
                results[name] = base_metrics
                logger.info(
                    f"Completed {name}: Acc={base_metrics['Accuracy']:.3f} F1={base_metrics['F1 Score']:.3f}"
                )
    logger.info("Training pipeline complete")
    return results
