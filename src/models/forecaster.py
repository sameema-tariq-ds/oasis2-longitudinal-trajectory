
"""Baseline feature extraction and class-balanced trajectory forecasting."""

import logging
from typing import List

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


logger = logging.getLogger(__name__)


def train_baseline_forecaster(
    train_df: pd.DataFrame,
    trajectory_matrix: pd.DataFrame,
    feature_cols: List[str],
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = None,
    min_samples_leaf: int = 2,
) -> Pipeline:
    """
    Train a class-balanced Random Forest to predict trajectory clusters
    using baseline (Visit == 1) subject characteristics.

    Processing stages:
        1. Validate input DataFrames and feature columns.
        2. Extract baseline observations.
        3. Validate subject-level uniqueness.
        4. Align baseline features with trajectory cluster labels.
        5. Separate numeric and categorical features.
        6. Impute missing values and encode categorical features.
        7. Train a class-balanced Random Forest classifier.

    Args:
        train_df: Cleaned long-form training partition.
        trajectory_matrix: Subject-level trajectory matrix containing
            the Trajectory_Cluster target.
        feature_cols: Baseline predictor columns.
        random_state: Random seed for reproducibility.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth. None allows unrestricted growth.
        min_samples_leaf: Minimum samples required at each leaf.

    Returns:
        Fitted sklearn Pipeline containing preprocessing and
        RandomForestClassifier.

    Raises:
        ValueError: If inputs, features, or labels are invalid.
        TypeError: If input objects have incorrect types.
        RuntimeError: If model training fails.
    """

    logger.info("Starting baseline trajectory forecasting.")

    try:
        # --------------------------------------------------
        # Phase 1: Input Validation
        # --------------------------------------------------

        if not isinstance(train_df, pd.DataFrame):
            raise TypeError("train_df must be a pandas DataFrame.")

        if not isinstance(trajectory_matrix, pd.DataFrame):
            raise TypeError(
                "trajectory_matrix must be a pandas DataFrame."
            )

        if train_df.empty:
            raise ValueError("Training DataFrame is empty.")

        if trajectory_matrix.empty:
            raise ValueError("Trajectory matrix is empty.")

        if not isinstance(feature_cols, list) or not feature_cols:
            raise ValueError(
                "feature_cols must be a non-empty list."
            )

        if len(feature_cols) != len(set(feature_cols)):
            raise ValueError(
                "feature_cols contains duplicate column names."
            )

        if not isinstance(random_state, int):
            raise ValueError("random_state must be an integer.")

        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")

        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")

        if max_depth is not None and max_depth < 1:
            raise ValueError(
                "max_depth must be a positive integer or None."
            )

        required_train_columns = ["Subject ID", "Visit"]
        required_target_column = ["Trajectory_Cluster"]

        missing_train = [
            col for col in required_train_columns
            if col not in train_df.columns
        ]

        missing_target = [
            col for col in required_target_column
            if col not in trajectory_matrix.columns
        ]

        if missing_train:
            raise ValueError(
                f"Missing training columns: {missing_train}"
            )

        if missing_target:
            raise ValueError(
                f"Missing trajectory target columns: {missing_target}"
            )

        missing_features = [
            col for col in feature_cols
            if col not in train_df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Requested feature columns not found: {missing_features}"
            )

        # Prevent identifier, time, and target leakage.
        prohibited_features = {
            "Subject ID",
            "Visit",
            "Trajectory_Cluster",
        }

        invalid_features = prohibited_features.intersection(feature_cols)

        if invalid_features:
            raise ValueError(
                "Features contain prohibited identifier, time, "
                f"or target columns: {sorted(invalid_features)}"
            )

        logger.info("Input validation completed successfully.")

        # --------------------------------------------------
        # Phase 2: Extract Baseline Records
        # --------------------------------------------------

        logger.info("Extracting baseline observations (Visit == 1).")

        baseline_records = train_df.loc[
            train_df["Visit"] == 1
        ].copy()

        if baseline_records.empty:
            raise ValueError(
                "No baseline observations found for Visit == 1."
            )

        if baseline_records["Subject ID"].isna().any():
            raise ValueError(
                "Baseline records contain missing Subject IDs."
            )

        duplicate_baselines = baseline_records.duplicated(
            subset=["Subject ID"],
            keep=False,
        )

        if duplicate_baselines.any():
            raise ValueError(
                "Multiple baseline records found for one or more "
                "subjects. Each subject must have one baseline record."
            )

        logger.info(
            "Baseline extraction completed: %d subjects.",
            len(baseline_records),
        )

        # --------------------------------------------------
        # Phase 3: Extract Trajectory Labels
        # --------------------------------------------------

        logger.info("Extracting trajectory cluster labels.")

        labels_frame = trajectory_matrix[
            ["Trajectory_Cluster"]
        ].copy()

        if labels_frame.index.has_duplicates:
            raise ValueError(
                "Trajectory matrix contains duplicate subject indices."
            )

        labels_frame = labels_frame.reset_index()

        if "Subject ID" not in labels_frame.columns:
            raise ValueError(
                "Trajectory matrix index must represent Subject ID."
            )

        if labels_frame["Trajectory_Cluster"].isna().any():
            raise ValueError(
                "Trajectory cluster labels contain missing values."
            )

        # --------------------------------------------------
        # Phase 4: Align Baseline Features and Target Labels
        # --------------------------------------------------

        logger.info("Aligning baseline features with trajectory labels.")

        processed_ml_data = pd.merge(
            baseline_records,
            labels_frame,
            on="Subject ID",
            how="inner",
            validate="one_to_one",
        )

        if processed_ml_data.empty:
            raise ValueError(
                "No matching subjects between baseline records "
                "and trajectory labels."
            )

        unmatched_baseline = (
            len(baseline_records) - len(processed_ml_data)
        )

        if unmatched_baseline:
            logger.warning(
                "%d baseline subjects did not have trajectory labels "
                "and were excluded.",
                unmatched_baseline,
            )

        logger.info(
            "Subject alignment completed: %d matched subjects.",
            len(processed_ml_data),
        )

        # --------------------------------------------------
        # Phase 5: Separate Features and Target
        # --------------------------------------------------

        X_train = processed_ml_data[feature_cols].copy()

        y_train = processed_ml_data[
            "Trajectory_Cluster"
        ].copy()

        if y_train.nunique() < 2:
            raise ValueError(
                "At least two trajectory classes are required "
                "for classification."
            )

        if X_train.isna().all().any():
            empty_features = X_train.columns[
                X_train.isna().all()
            ].tolist()

            raise ValueError(
                f"Features containing only missing values: "
                f"{empty_features}"
            )

        # --------------------------------------------------
        # Phase 6: Feature Type Detection
        # --------------------------------------------------

        numeric_features = X_train.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

        categorical_features = [
            col for col in feature_cols
            if col not in numeric_features
        ]

        if not numeric_features and not categorical_features:
            raise ValueError("No usable predictor features found.")

        logger.info(
            "Feature types identified: %d numeric, %d categorical.",
            len(numeric_features),
            len(categorical_features),
        )

        # --------------------------------------------------
        # Phase 7: Preprocessing Pipelines
        # --------------------------------------------------

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers = []

        if numeric_features:
            transformers.append(
                ("numeric", numeric_pipeline, numeric_features)
            )

        if categorical_features:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_features)
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        # --------------------------------------------------
        # Phase 8: Class-Balanced Random Forest
        # --------------------------------------------------

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            # Single-process execution is more portable on restricted
            # Windows/Jupyter environments where joblib worker creation can
            # fail with WinError 5.  The dataset is small enough that the
            # runtime impact is negligible.
            n_jobs=1,
        )

        forecasting_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", rf_model),
            ]
        )

        logger.info(
            "Training Random Forest: trees=%d, max_depth=%s, "
            "min_samples_leaf=%d.",
            n_estimators,
            max_depth,
            min_samples_leaf,
        )

        forecasting_pipeline.fit(X_train, y_train)

        # --------------------------------------------------
        # Phase 9: Training Summary
        # --------------------------------------------------

        class_distribution = (
            y_train.value_counts()
            .sort_index()
            .to_dict()
        )

        logger.info(
            "Baseline forecasting model trained successfully."
        )

        logger.info(
            "Training subjects: %d | Features: %d | "
            "Trajectory classes: %d.",
            len(X_train),
            len(feature_cols),
            y_train.nunique(),
        )

        logger.info(
            "Training class distribution: %s",
            class_distribution,
        )

        return forecasting_pipeline

    except (ValueError, TypeError):
        logger.exception(
            "Baseline forecasting failed input validation."
        )
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected error during baseline model training."
        )

        raise RuntimeError(
            "Baseline forecasting model training failed."
        ) from exc
