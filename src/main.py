
"""Main entry point for the OASIS-2 pipeline."""

import logging
from pathlib import Path

from logging_config import setup_logging
from data_processing.preprocessing import (
    split_and_impute_dataset,
)
from models.trajectory import (
    build_and_fit_trajectories,
)
from models.forecaster import (
    train_baseline_forecaster,
)
from evaluation.evaluator import (
    evaluate_model,
)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print(PROJECT_ROOT)

DATA_PATH = PROJECT_ROOT / "data" / "oasis_longitudinal.xlsx"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures"

TRACKING_FEATURE = "nWBV"

FEATURE_PREDICTORS = [
    "Age",
    "M/F",
    "EDUC",
    "SES",
    "eTIV",
    "ASF",
]


def main() -> None:
    """Execute the complete OASIS-2 pipeline."""

    try:
        setup_logging()

        logger.info("Starting OASIS-2 longitudinal analysis.")

        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}"
            )

        # 1. Load, split and preprocess.
        train_df, test_df = split_and_impute_dataset(
            str(DATA_PATH)
        )

        logger.info(
            "Data split completed. Train: %d rows | Test: %d rows",
            len(train_df),
            len(test_df),
        )

        # 2. Fit trajectory clustering on training subjects.
        train_trajectory_matrix, clustering_model = (
            build_and_fit_trajectories(
                train_df,
                tracking_metric=TRACKING_FEATURE,
            )
        )

        # 3. Train baseline classifier.
        classifier = train_baseline_forecaster(
            train_df,
            train_trajectory_matrix,
            FEATURE_PREDICTORS,
        )

        # 4. Evaluate on held-out subjects.
        results = evaluate_model(
            model=classifier,
            clustering_model=clustering_model,
            train_trajectory_matrix=train_trajectory_matrix,
            test_df=test_df,
            feature_predictors=FEATURE_PREDICTORS,
            tracking_feature=TRACKING_FEATURE,
            output_dir=FIGURE_PATH,
        )

        logger.info(
            "Pipeline completed successfully. ROC-AUC: %s",
            results["roc_auc"],
        )

        print(f"\nFigures saved to: {FIGURE_PATH}")

    except Exception:
        logger.exception("OASIS-2 pipeline execution failed.")
        raise


if __name__ == "__main__":
    main()
