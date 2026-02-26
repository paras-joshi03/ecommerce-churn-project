"""
Layer 1 – Data Ingestion Layer
Only handles data loading, validation, and basic cleaning.
No ML logic here.
"""

import pandas as pd


class DataIngestion:

    def __init__(self, config):
        self.config = config
        self.file_path = config["data"]["file_path"]
        self.required_columns = config["data"]["required_columns"]

    def load_data(self):
        print("Loading dataset...")

        df = pd.read_excel(self.file_path)

        # Convert date column
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

        self.validate_schema(df)
        df = self.basic_cleaning(df)

        print("Data loaded successfully.")
        print("Dataset shape:", df.shape)

        return df

    def validate_schema(self, df):
        missing = []

        for col in self.required_columns:
            if col not in df.columns:
                missing.append(col)

        if len(missing) > 0:
            raise ValueError(f"Missing required columns: {missing}")

        print("Schema validation passed.")

    def basic_cleaning(self, df):
        df = df[df["Quantity"] > 0]
        df = df[df["UnitPrice"] > 0]
        df = df.dropna(subset=["CustomerID"])

        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

        return df