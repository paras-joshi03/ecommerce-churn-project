# ------------------------------------------------------------------
# data_ingestion.py  -  Layer 1
#
# Loads any CSV or Excel file, standardises column names,
# and detects which of the three operating modes applies.
#
# Case 1 / Case 2  : Behavioural data, no Churn column.
#                    Pipeline loads saved model and predicts.
#                    Case 2 differs from Case 1 only in that some
#                    columns are missing and will be filled with
#                    training medians before prediction.
#
# Case 3           : Raw transaction rows (one row per purchase).
#                    Pipeline aggregates per customer, derives RFM,
#                    then predicts using the saved model.
#
# Training mode    : Labelled dataset with Churn column present.
#                    Pipeline trains, evaluates, saves artifacts.
#                    This is run once using your historical dataset.
# ------------------------------------------------------------------

import os
import pandas as pd
from config import COLUMN_ALIASES, BEHAVIOURAL_COLS, TRANSACTION_MIN_COLS

# Mode constants used throughout the pipeline.
MODE_TRAIN       = "train"        # Has Churn label - train the model
MODE_BEHAVIOURAL = "behavioural"  # Has behaviour cols, no Churn - predict
MODE_TRANSACTION = "transaction"  # Raw transaction rows - derive RFM then predict


class DataIngestion:

    def load(self, file) -> pd.DataFrame:
        """
        Load a CSV or Excel file into a DataFrame.
        Accepts a file path string or a Streamlit UploadedFile object.
        For Excel files, sheet index 1 is tried first because many
        e-commerce exports place data on the second sheet.
        """
        name = file.name if hasattr(file, "name") else str(file)
        ext  = os.path.splitext(name)[1].lower()

        if ext == ".csv":
            dataframe = pd.read_csv(file)

        elif ext == ".xlsx":
            try:
                dataframe = pd.read_excel(file, sheet_name=1)
            except Exception:
                dataframe = pd.read_excel(file, sheet_name=0)
        else:
            raise ValueError("Only .csv or .xlsx files are supported.")

        print("Loaded {} rows and {} columns from {}.".format(
            len(dataframe), len(dataframe.columns), name
        ))
        return dataframe

    # ------------------------------------------------------------------

    def standardise_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to the standard names defined in COLUMN_ALIASES.
        Matching is case-insensitive and strips whitespace.
        This lets the pipeline accept data from businesses that use
        different naming conventions for the same concepts.
        """
        rename_map = {}
        for col in dataframe.columns:
            normalised = col.lower().strip()
            if normalised in COLUMN_ALIASES:
                rename_map[col] = COLUMN_ALIASES[normalised]

        if rename_map:
            dataframe = dataframe.rename(columns=rename_map)
            print("Standardised column names: {}.".format(list(rename_map.values())))

        return dataframe

    # ------------------------------------------------------------------

    def detect_mode(self, dataframe: pd.DataFrame) -> str:
        """
        Determine which pipeline mode to apply based on available columns.

        Decision order:
          1. If Churn column present -> training mode.
          2. If at least 3 behavioural columns present -> behavioural mode.
             Missing columns will be filled with training medians.
          3. If transaction columns present -> transaction mode.
          Otherwise raise an error with clear instructions.
        """
        columns = [c.lower() for c in dataframe.columns]

        has_churn        = "churn" in columns
        has_customer_id  = "customerid" in columns

        behavioural_present = [
            col for col in BEHAVIOURAL_COLS
            if col.lower() in columns
        ]

        transaction_present = all(
            col.lower() in columns
            for col in TRANSACTION_MIN_COLS
        )

        if not has_customer_id:
            raise ValueError(
                "CustomerID column is required in all modes. "
                "Please check your data or rename the column."
            )

        if has_churn:
            print("Mode detected: TRAINING. Churn label found.")
            return MODE_TRAIN

        if len(behavioural_present) >= 3:
            print("Mode detected: BEHAVIOURAL PREDICTION. "
                  "{} of {} behaviour columns found.".format(
                      len(behavioural_present), len(BEHAVIOURAL_COLS)))
            return MODE_BEHAVIOURAL

        if transaction_present:
            print("Mode detected: TRANSACTION. "
                  "Raw transaction data will be aggregated to customer level.")
            return MODE_TRANSACTION

        raise ValueError(
            "Could not determine dataset mode. "
            "Please ensure your data contains either:\n"
            "  - Behavioural columns (Tenure, OrderCount, etc.) for prediction, or\n"
            "  - Transaction columns (CustomerID, InvoiceDate, Quantity, UnitPrice)."
        )

    # ------------------------------------------------------------------

    def validate(self, dataframe: pd.DataFrame, mode: str) -> dict:
        """
        Return a summary of the loaded dataset for display in the dashboard.
        Includes row count, column count, missing value counts,
        and churn distribution when the label is present.
        """
        summary = {
            "rows":        len(dataframe),
            "columns":     len(dataframe.columns),
            "column_list": dataframe.columns.tolist(),
            "missing":     dataframe.isnull().sum().to_dict(),
            "mode":        mode,
            "churn_dist":  {},
        }

        if "Churn" in dataframe.columns:
            counts = dataframe["Churn"].value_counts().to_dict()
            total  = len(dataframe)
            summary["churn_dist"] = {
                str(k): {"count": int(v), "pct": round(v / total * 100, 2)}
                for k, v in counts.items()
            }

        return summary
