
"""Input file and dataset validation."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Subject ID", "Group", "SES", "MMSE"]
SUPPORTED_EXTENSIONS = {".xlsx", ".xls"}


def validate_input_file(file_path: str) -> Path:
    """Validate input file path and format."""

    logger.info("Validating input file: %s", file_path)

    if not file_path or not str(file_path).strip():
        raise ValueError("Input file path cannot be empty.")

    file = Path(file_path)

    if not file.exists():
        raise FileNotFoundError(f"Input file not found: {file}")

    if not file.is_file():
        raise ValueError(f"Input path is not a file: {file}")

    if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {file.suffix}. "
            f"Expected: {SUPPORTED_EXTENSIONS}"
        )

    logger.info("Input file validation successful.")
    return file


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate dataset structure and required data fields."""

    logger.info("Starting dataset validation.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    if df["Subject ID"].isna().any():
        raise ValueError("Subject ID contains missing values.")

    if df["Group"].isna().any():
        raise ValueError("Group contains missing values.")

    for col in ["SES", "MMSE"]:
        converted = pd.to_numeric(df[col], errors="coerce")

        invalid_values = df[col].notna() & converted.isna()

        if invalid_values.any():
            raise ValueError(
                f"Column '{col}' contains non-numeric values."
            )

        df[col] = converted

    logger.info(
        "Dataset validation successful: %d rows, %d columns.",
        len(df),
        len(df.columns),
    )
