
"""Dataset ingestion."""

import logging
import pandas as pd

from .validation import validate_input_file, validate_dataset

logger = logging.getLogger(__name__)


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load and validate the input Excel dataset."""

    try:
        file = validate_input_file(file_path)

        logger.info("Loading dataset from: %s", file)

        df = pd.read_excel(file)

        validate_dataset(df)

        logger.info("Dataset loaded successfully.")

        return df

    except Exception:
        logger.exception("Dataset ingestion failed.")
        raise
