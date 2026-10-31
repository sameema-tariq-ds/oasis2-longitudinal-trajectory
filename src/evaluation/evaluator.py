
"""Model evaluation and diagnostic visualization."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def _get_random_forest(model):
    """Extract RandomForestClassifier from a model or pipeline."""

    if isinstance(model, RandomForestClassifier):
        return model

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if isinstance(step, RandomForestClassifier):
                return step

    raise TypeError(
        "No RandomForestClassifier found in the supplied model."
    )


def _get_feature_names(model, fallback_features):
    """Retrieve feature names after preprocessing."""

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "get_feature_names_out"):
                try:
                    return list(step.get_feature_names_out())
                except (ValueError, AttributeError):
                    logger.warning(
                        "Unable to retrieve transformed feature names."
                    )

    return list(fallback_features)


def validate_evaluation_data(
    model,
    clustering_model,
    train_trajectory_matrix,
    test_df,
    feature_predictors,
):
    """Validate models and datasets before evaluation."""

    if test_df.empty:
        raise ValueError("Test dataset is empty.")

    if train_trajectory_matrix.empty:
        raise ValueError("Training trajectory matrix is empty.")

    if "Trajectory_Cluster" not in train_trajectory_matrix.columns:
        raise ValueError(
            "Training trajectory matrix lacks Trajectory_Cluster."
        )

    if "Subject ID" not in test_df.columns:
        raise ValueError("Subject ID column is missing from test data.")

    if "Visit" not in test_df.columns:
        raise ValueError("Visit column is missing from test data.")

    missing_features = set(feature_predictors).difference(test_df.columns)

    if missing_features:
        raise ValueError(
            f"Missing baseline predictors: {sorted(missing_features)}"
        )

    if not hasattr(clustering_model, "predict"):
        raise TypeError("Clustering model must implement predict().")

    if not hasattr(model, "predict"):
        raise TypeError("Classifier must implement predict().")

    logger.info("Evaluation input validation completed.")


def _prepare_test_trajectories(
    test_df,
    train_trajectory_matrix,
    tracking_feature,
):
    """Create test trajectories using the training visit columns."""

    if test_df.duplicated(["Subject ID", "Visit"]).any():
        raise ValueError(
            "Duplicate Subject ID and Visit combinations in test data."
        )

    test_matrix = test_df.pivot(
        index="Subject ID",
        columns="Visit",
        values=tracking_feature,
    )

    visit_columns = [
        col for col in train_trajectory_matrix.columns
        if col != "Trajectory_Cluster"
    ]

    test_matrix = test_matrix.reindex(columns=visit_columns)

    test_matrix = test_matrix.ffill(axis=1).bfill(axis=1)

    if test_matrix.empty or test_matrix.isna().any().any():
        raise ValueError(
            "Test trajectory matrix is empty or contains missing values."
        )

    if not np.isfinite(test_matrix.to_numpy(dtype=float)).all():
        raise ValueError(
            "Test trajectory matrix contains non-finite values."
        )

    return test_matrix


def _save_confusion_matrix(y_true, y_pred, class_labels, output_dir):
    """Save confusion matrix."""

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_labels,
    )

    labels = [f"Cluster {value}" for value in class_labels]

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_title("Held-out Confusion Matrix")
    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("Actual Cluster")

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def _save_roc_curve(
    y_true,
    y_probabilities,
    positive_class,
    output_dir,
):
    """Save binary classification ROC curve."""

    binary_labels = (y_true == positive_class).astype(int)

    if binary_labels.nunique() != 2:
        logger.warning(
            "ROC curve skipped: both classes are not present."
        )
        return

    fpr, tpr, _ = roc_curve(binary_labels, y_probabilities)
    auc_score = roc_auc_score(binary_labels, y_probabilities)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"Random Forest (AUC = {auc_score:.3f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random baseline",
    )

    ax.set_title("Held-out ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close(fig)

    logger.info("ROC-AUC: %.4f", auc_score)


def _save_feature_importance(
    model,
    feature_predictors,
    output_dir,
):
    """Save Random Forest feature importance."""

    forest = _get_random_forest(model)

    importances = forest.feature_importances_

    feature_names = _get_feature_names(
        model,
        feature_predictors,
    )

    if len(feature_names) != len(importances):
        logger.warning(
            "Feature names do not match importance dimensions."
        )
        feature_names = [
            f"Feature {i + 1}" for i in range(len(importances))
        ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
    )

    ax.set_title("Baseline Feature Importance")
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Predictor")

    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close(fig)


def _save_trajectory_plot(
    trajectory_matrix,
    tracking_feature,
    output_dir,
):
    """Save individual subject trajectories grouped by cluster."""

    visit_columns = [
        col for col in trajectory_matrix.columns
        if col != "Trajectory_Cluster"
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in trajectory_matrix.iterrows():
        cluster = row["Trajectory_Cluster"]

        ax.plot(
            range(1, len(visit_columns) + 1),
            row[visit_columns].to_numpy(dtype=float),
            alpha=0.25,
            linewidth=1.2,
            label=f"Cluster {cluster}",
        )

    ax.set_title(f"Longitudinal Trajectories ({tracking_feature})")
    ax.set_xlabel("Visit Sequence")
    ax.set_ylabel("Normalized Brain Volume")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_clusters.png", dpi=300)
    plt.close(fig)


def evaluate_model(
    model,
    clustering_model,
    train_trajectory_matrix,
    test_df,
    feature_predictors,
    tracking_feature="nWBV",
    output_dir=Path("reports/figures"),
):
    """
    Evaluate the baseline classifier using held-out subjects.

    Returns:
        dict: Classification report, predictions and evaluation metrics.
    """

    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        validate_evaluation_data(
            model,
            clustering_model,
            train_trajectory_matrix,
            test_df,
            feature_predictors,
        )

        # 1. Assign test subjects to training-defined clusters.
        test_trajectory_matrix = _prepare_test_trajectories(
            test_df,
            train_trajectory_matrix,
            tracking_feature,
        )

        test_clusters = clustering_model.predict(
            test_trajectory_matrix
        )

        test_labels = pd.Series(
            test_clusters,
            index=test_trajectory_matrix.index,
            name="True_Cluster",
        )

        # 2. Extract baseline features.
        test_baseline = test_df.loc[
            test_df["Visit"] == 1
        ].copy()

        if test_baseline["Subject ID"].duplicated().any():
            raise ValueError(
                "Multiple baseline records found for a test subject."
            )

        test_baseline = test_baseline.set_index("Subject ID")

        common_subjects = test_baseline.index.intersection(
            test_labels.index
        )

        if common_subjects.empty:
            raise ValueError(
                "No matching baseline subjects and trajectory labels."
            )

        test_baseline = test_baseline.loc[common_subjects]

        X_test = test_baseline[feature_predictors]
        y_test = test_labels.loc[common_subjects]

        # 3. Predictions.
        y_pred = model.predict(X_test)

        class_labels = np.asarray(
            sorted(pd.unique(y_test))
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=class_labels,
            zero_division=0,
        )

        logger.info("Classification report:\n%s", report)

        print("\n--- HELD-OUT SUBJECT-LEVEL EVALUATION ---")
        print(report)

        # 4. Generate diagnostic figures.
        sns.set_theme(style="whitegrid")

        _save_confusion_matrix(
            y_test,
            y_pred,
            class_labels,
            output_dir,
        )

        _save_feature_importance(
            model,
            feature_predictors,
            output_dir,
        )

        _save_trajectory_plot(
            train_trajectory_matrix,
            tracking_feature,
            output_dir,
        )

        auc_score = None

        if len(class_labels) == 2 and hasattr(model, "predict_proba"):
            model_classes = list(model.classes_)
            positive_class = model_classes[1]
            positive_index = model_classes.index(positive_class)

            y_probabilities = model.predict_proba(X_test)[
                :, positive_index
            ]

            binary_labels = (y_test == positive_class).astype(int)

            if binary_labels.nunique() == 2:
                auc_score = roc_auc_score(
                    binary_labels,
                    y_probabilities,
                )

                _save_roc_curve(
                    y_test,
                    y_probabilities,
                    positive_class,
                    output_dir,
                )
        else:
            logger.warning(
                "ROC evaluation skipped: binary classification "
                "or probability predictions unavailable."
            )

        logger.info(
            "Evaluation figures saved to: %s",
            output_dir.resolve(),
        )

        return {
            "classification_report": report,
            "y_true": y_test,
            "y_pred": y_pred,
            "roc_auc": auc_score,
            "output_dir": output_dir,
        }

    except (ValueError, TypeError, KeyError):
        logger.exception("Model evaluation failed.")
        raise

    except Exception:
        logger.exception("Unexpected evaluation error.")
        raise
