"""
data_loader.py – Telegram Message Loader for Amharic E-Commerce (B5W4)
------------------------------------------------------------------------------
Safely loads the cleaned Telegram message dataset for EthioMart NER labeling.
Performs robust file validation, CSV parsing, and structure diagnostics.

Core responsibilities:
  • Validates file path and CSV format
  • Loads structured messages with timestamps and metadata (if available)
  • Supports both full message metadata or single-column labeling prep
  • Prints diagnostic output for labeling readiness

Used in Task 2 NER labeling and all downstream NLP pipelines.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os  # File validation and access

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # CSV loading and DataFrame operations


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: TelegramMessageLoader
# ───────────────────────────────────────────────────────────────────────────────
class TelegramMessageLoader:
    """
    OOP wrapper for safely loading the EthioMart Telegram dataset.
    Validates structure before Named Entity Recognition (NER) workflows.
    """

    def __init__(self, filepath: str):
        """
        Initialize the loader with the cleaned Telegram message file path.

        Args:
            filepath (str): Full path to the cleaned .csv or .txt file.

        Raises:
            TypeError: If the filepath is not a string.
            FileNotFoundError: If the file does not exist at the given path.
        """
        if not isinstance(filepath, str):
            raise TypeError(f"filepath must be a string, got {type(filepath)}")

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Cannot find file at: {filepath}")

        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        """
        Loads the cleaned message dataset and ensures structural integrity.

        Returns:
            pd.DataFrame: DataFrame with validated message structure.

        Raises:
            ValueError: If the dataset is empty or missing critical columns.
        """
        try:
            # Initial load attempt
            df = pd.read_csv(self.filepath)

            # Fallback: handle headerless single-column files
            if df.shape[1] == 1 and "cleaned_message" not in df.columns:
                df = pd.read_csv(self.filepath, header=None)
                df.columns = ["cleaned_message"]

            # Check empty
            if df.empty:
                raise ValueError("Loaded Telegram message DataFrame is empty.")

            # Accept either full metadata or minimal message list
            acceptable_columns = {"cleaned_message", "message", "channel", "timestamp"}
            missing = [col for col in acceptable_columns if col not in df.columns]

            if len(missing) == len(acceptable_columns):
                raise ValueError(f"Missing expected columns: {missing}")

            print(
                f"✅ Telegram messages loaded: {df.shape[0]:,} rows × {df.shape[1]} columns"
            )
            return df

        except pd.errors.ParserError as e:
            raise ValueError(f"Could not parse CSV: {e}")

        except Exception as e:
            raise RuntimeError(f"Unexpected error during data load: {e}")
