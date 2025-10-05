from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
from sklearn.exceptions import ConvergenceWarning
from .logging_utils import get_logger

logger = get_logger("training")

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
    """
    logger.info(
        f"Starting training pipeline test_size={test_size} stratify={stratify} scale_linear_models={scale_linear_models}"
    )
    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    logger.debug(f"Train shape={getattr(X_train, 'shape', None)} Test shape={getattr(X_test, 'shape', None)}")

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

    results = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for name, model in models.items():
            logger.info(f"Fitting model: {name}")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = {
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, preds, average='weighted', zero_division=0),
                "F1 Score": f1_score(y_test, preds, average='weighted', zero_division=0)
            }
            logger.info(
                f"Completed {name}: Acc={results[name]['Accuracy']:.3f} F1={results[name]['F1 Score']:.3f}"
            )
    logger.info("Training pipeline complete")
    return results
