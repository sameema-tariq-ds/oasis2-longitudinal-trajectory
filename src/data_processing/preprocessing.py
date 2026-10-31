
"""Subject-level splitting and dual-tiered missing-value imputation."""

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .ingestion import load_dataset

logger = logging.getLogger(__name__)


def split_and_impute_dataset(
    file_path: str,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, split, and impute longitudinal OASIS-2 data.

    Processing stages:
        1. Load and validate input dataset.
        2. Perform subject-level stratified train/test split.
        3. Verify subject-level isolation.
        4. Apply within-subject forward/backward filling.
        5. Apply training-derived population median imputation.
        6. Perform final validation.

    Args:
        file_path: Path to the raw OASIS-2 Excel file.
        random_state: Random seed for reproducible splitting.

    Returns:
        Tuple containing cleaned training and testing DataFrames.
    """

    logger.info("Starting preprocessing pipeline.")

    try:
        # --------------------------------------------------
        # Phase 1: Data Ingestion and Validation
        # --------------------------------------------------

        df = load_dataset(file_path)

        logger.info(
            "Dataset loaded successfully: %d rows, %d columns.",
            len(df),
            len(df.columns),
        )

        # --------------------------------------------------
        # Phase 2: Subject-Level Train/Test Split
        # --------------------------------------------------

        logger.info("Starting stratified subject-level splitting.")

        sgkf = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        )

        train_idx, test_idx = next(
            sgkf.split(
                X=df,
                y=df["Group"],
                groups=df["Subject ID"],
            )
        )

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        logger.info(
            "Split completed: %d training records and "
            "%d testing records.",
            len(train_df),
            len(test_df),
        )

        # --------------------------------------------------
        # Phase 3: Subject-Level Isolation Validation
        # --------------------------------------------------

        train_subjects = set(train_df["Subject ID"].unique())
        test_subjects = set(test_df["Subject ID"].unique())

        overlapping_subjects = train_subjects.intersection(
            test_subjects
        )

        if overlapping_subjects:
            raise RuntimeError(
                "Subject-level data leakage detected. "
                f"Overlapping subjects: {len(overlapping_subjects)}"
            )

        logger.info(
            "Subject isolation verified: %d training subjects, "
            "%d testing subjects.",
            len(train_subjects),
            len(test_subjects),
        )

        # --------------------------------------------------
        # Phase 4: Chronological Sorting
        # --------------------------------------------------

        if "Visit" in df.columns:

            train_df = train_df.sort_values(
                by=["Subject ID", "Visit"]
            )

            test_df = test_df.sort_values(
                by=["Subject ID", "Visit"]
            )

            logger.info("Data sorted chronologically by visit.")

        else:
            logger.warning(
                "Visit column not found. Ensure observations are "
                "already chronologically ordered before imputation."
            )

        # --------------------------------------------------
        # Phase 5: Dual-Tiered Imputation
        # --------------------------------------------------

        imputation_columns = ["SES", "MMSE"]

        # Tier 1: Within-Subject Forward and Backward Filling

        logger.info(
            "Starting Tier 1: Within-subject imputation."
        )

        for col in imputation_columns:

            train_missing_before = train_df[col].isna().sum()
            test_missing_before = test_df[col].isna().sum()

            train_df[col] = (
                train_df.groupby("Subject ID")[col]
                .transform(lambda x: x.ffill().bfill())
            )

            test_df[col] = (
                test_df.groupby("Subject ID")[col]
                .transform(lambda x: x.ffill().bfill())
            )

            logger.info(
                "Tier 1 imputation for %s completed. "
                "Training missing: %d -> %d; "
                "Testing missing: %d -> %d.",
                col,
                train_missing_before,
                train_df[col].isna().sum(),
                test_missing_before,
                test_df[col].isna().sum(),
            )

        # Tier 2: Population Median Imputation
        # Calculate medians exclusively from training data.

        logger.info(
            "Starting Tier 2: Training-derived median imputation."
        )

        for col in imputation_columns:

            training_median = train_df[col].median()

            if pd.isna(training_median):
                raise ValueError(
                    f"Unable to calculate training median "
                    f"for column '{col}'."
                )

            train_missing_before = train_df[col].isna().sum()
            test_missing_before = test_df[col].isna().sum()

            train_df[col] = train_df[col].fillna(
                training_median
            )

            test_df[col] = test_df[col].fillna(
                training_median
            )

            logger.info(
                "Tier 2 imputation for %s completed. "
                "Training median: %.3f; "
                "Training values filled: %d; "
                "Testing values filled: %d.",
                col,
                training_median,
                train_missing_before,
                test_missing_before,
            )

        # --------------------------------------------------
        # Phase 6: Final Validation
        # --------------------------------------------------

        for col in imputation_columns:

            if train_df[col].isna().any():
                raise RuntimeError(
                    f"Missing values remain in training column '{col}'."
                )

            if test_df[col].isna().any():
                raise RuntimeError(
                    f"Missing values remain in testing column '{col}'."
                )

        # Verify subject isolation once again.
        assert train_subjects.isdisjoint(test_subjects)

        logger.info(
            "Preprocessing completed successfully."
        )

        logger.info(
            "Final training shape: %s | Testing shape: %s",
            train_df.shape,
            test_df.shape,
        )

        return train_df, test_df

    except Exception:
        logger.exception(
            "Preprocessing pipeline failed."
        )
        raise
