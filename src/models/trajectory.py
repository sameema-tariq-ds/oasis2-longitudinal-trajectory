
"""Longitudinal trajectory construction and clustering."""

import logging

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build_and_fit_trajectories(
    train_df: pd.DataFrame,
    tracking_metric: str = "nWBV",
    n_clusters: int = 2,
    random_state: int = 42,
):
    """
    Construct subject-level longitudinal trajectories and fit K-Means.

    Args:
        train_df: Training dataset containing subject visits.
        tracking_metric: Longitudinal variable used for clustering.
        n_clusters: Number of trajectory clusters.
        random_state: Random seed for reproducibility.

    Returns:
        tuple:
            trajectory_matrix: Subject-level trajectories with cluster labels.
            clustering_pipeline: Fitted scaler and K-Means pipeline.
    """

    try:
        # 1. Validate inputs.
        if not isinstance(train_df, pd.DataFrame):
            raise TypeError("train_df must be a pandas DataFrame.")

        if train_df.empty:
            raise ValueError("Training dataset is empty.")

        required_columns = {
            "Subject ID",
            "Visit",
            tracking_metric,
        }

        missing_columns = required_columns.difference(train_df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError("n_clusters must be an integer of at least 2.")

        if train_df["Subject ID"].isna().any():
            raise ValueError("Subject ID contains missing values.")

        if train_df["Visit"].isna().any():
            raise ValueError("Visit contains missing values.")

        if train_df.duplicated(["Subject ID", "Visit"]).any():
            raise ValueError(
                "Duplicate Subject ID and Visit combinations detected."
            )

        # 2. Convert tracking variable to numeric.
        working_df = train_df[
            ["Subject ID", "Visit", tracking_metric]
        ].copy()

        working_df[tracking_metric] = pd.to_numeric(
            working_df[tracking_metric],
            errors="coerce",
        )

        if working_df[tracking_metric].notna().sum() == 0:
            raise ValueError(
                f"No valid numeric values found for {tracking_metric}."
            )

        # 3. Construct subject-level trajectory matrix.
        trajectory_matrix = working_df.pivot(
            index="Subject ID",
            columns="Visit",
            values=tracking_metric,
        )

        # Ensure visits are in chronological order.
        visit_columns = sorted(
            trajectory_matrix.columns,
            key=lambda value: (
                int(str(value).lstrip("Vv"))
                if str(value).lstrip("Vv").isdigit()
                else str(value)
            ),
        )

        trajectory_matrix = trajectory_matrix.reindex(
            columns=visit_columns
        )

        # 4. Handle missing longitudinal measurements.
        trajectory_matrix = trajectory_matrix.ffill(axis=1).bfill(axis=1)

        if trajectory_matrix.isna().any().any():
            raise ValueError(
                "Missing trajectory values remain after imputation."
            )

        if not np.isfinite(
            trajectory_matrix.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Trajectory matrix contains non-finite values."
            )

        if len(trajectory_matrix) < n_clusters:
            raise ValueError(
                f"At least {n_clusters} subjects are required for "
                "the requested number of clusters."
            )

        if trajectory_matrix.nunique().max() < 1:
            raise ValueError("Trajectory matrix contains no usable variation.")

        logger.info(
            "Constructed trajectories for %d subjects across %d visits.",
            len(trajectory_matrix),
            len(trajectory_matrix.columns),
        )

        # 5. Fit scaler and K-Means using training subjects only.
        clustering_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    n_init=10,
                ),
            ),
        ])

        cluster_labels = clustering_pipeline.fit_predict(
            trajectory_matrix
        )

        # K-Means cluster numbers are arbitrary.  Give them a stable
        # clinical meaning: 0 = higher-volume/stable and 1 = lower-volume/
        # declining.  Store the mapping on the fitted pipeline so the same
        # convention can be applied to held-out subjects during evaluation.
        raw_centers = clustering_pipeline.named_steps["kmeans"].cluster_centers_
        centers = clustering_pipeline.named_steps["scaler"].inverse_transform(
            raw_centers
        )
        stable_raw_label = int(np.argmax(centers.mean(axis=1)))
        label_map = {
            stable_raw_label: 0,
            1 - stable_raw_label: 1,
        }
        clustering_pipeline.trajectory_label_map_ = label_map
        cluster_labels = np.asarray(
            [label_map[int(label)] for label in cluster_labels]
        )

        # 6. Attach cluster assignments.
        trajectory_matrix["Trajectory_Cluster"] = cluster_labels

        logger.info(
            "K-Means clustering completed successfully."
        )

        logger.info(
            "Cluster distribution:\n%s",
            trajectory_matrix["Trajectory_Cluster"].value_counts(),
        )

        return trajectory_matrix, clustering_pipeline

    except (ValueError, TypeError, KeyError):
        logger.exception("Trajectory construction or clustering failed.")
        raise

    except Exception:
        logger.exception("Unexpected trajectory clustering error.")
        raise
