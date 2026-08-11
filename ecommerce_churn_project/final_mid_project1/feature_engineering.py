# ------------------------------------------------------------------
# feature_engineering.py  -  Layer 2
#
# Handles three different input shapes:
#
#   Training mode    : Full behavioural data with Churn label.
#                      Fits encoders, scaler, and median store.
#                      Saves these so prediction modes can reuse them.
#
#   Behavioural mode : Full or partial behavioural data, no Churn.
#                      Loads fitted encoders and scaler.
#                      Fills missing columns with stored training medians.
#
#   Transaction mode : Raw transaction rows.
#                      Aggregates to customer level.
#                      Derives RFM and behavioural proxies.
#                      Then applies the same transform as behavioural mode.
#
# RFM derivation:
#   Recency   = days between last purchase and snapshot date.
#   Frequency = number of distinct invoices per customer.
#   Monetary  = total spend (Quantity * UnitPrice) per customer.
#   Tenure    = days between first and last purchase.
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import RANDOM_SEED, INACTIVITY_WINDOW_DAYS


class FeatureEngineer:

    def __init__(self):
        self.label_encoders  = {}
        self.scaler          = StandardScaler()
        self.training_medians = {}   # stored per column for filling missing cols
        self.feature_names   = []
        self.is_fitted       = False

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------

    def fill_missing(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing numeric values with column median.
        Fill missing categorical values with column mode.
        Median is used instead of mean because it is not affected by outliers,
        which are common in spend and order count columns.
        """
        dataframe = dataframe.copy()

        for col in dataframe.select_dtypes(include="number").columns:
            if dataframe[col].isnull().any():
                median = dataframe[col].median()
                dataframe[col] = dataframe[col].fillna(median)

        for col in dataframe.select_dtypes(include="object").columns:
            if dataframe[col].isnull().any():
                mode = dataframe[col].mode()
                if len(mode) > 0:
                    dataframe[col] = dataframe[col].fillna(mode[0])

        return dataframe

    # ------------------------------------------------------------------
    # RFM derivation from raw transaction data (Case 3)
    # ------------------------------------------------------------------

    def derive_rfm_from_transactions(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw transaction rows into one row per customer.
        Derives Recency, Frequency, Monetary, Tenure, and supporting features.

        Snapshot date is set to the maximum InvoiceDate in the dataset.
        This represents the point in time from which we look backwards.

        Churn is derived using the inactivity window from config:
        if a customer's last purchase is more than INACTIVITY_WINDOW_DAYS
        before the snapshot date, they are labelled as churned.
        """
        df = dataframe.copy()

        # Parse dates if not already datetime.
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df = df.dropna(subset=["InvoiceDate"])

        # Remove cancelled orders (InvoiceNo starting with C in UK retail data).
        if "InvoiceNo" in df.columns:
            df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

        # Remove rows with non-positive quantity or price.
        df = df[df["Quantity"] > 0]
        df = df[df["UnitPrice"] > 0]

        df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

        snapshot_date = df["InvoiceDate"].max()

        # Aggregate per customer.
        aggregated = df.groupby("CustomerID").agg(
            LastPurchaseDate  = ("InvoiceDate", "max"),
            FirstPurchaseDate = ("InvoiceDate", "min"),
            Frequency         = ("InvoiceDate", "nunique"),
            Monetary          = ("TotalSpend", "sum"),
            AvgOrderValue     = ("TotalSpend", "mean"),
            TotalQuantity     = ("Quantity", "sum"),
        ).reset_index()

        # Recency: days since last purchase.
        aggregated["Recency"] = (
            snapshot_date - aggregated["LastPurchaseDate"]
        ).dt.days

        # Tenure: days between first and last purchase.
        aggregated["Tenure"] = (
            aggregated["LastPurchaseDate"] - aggregated["FirstPurchaseDate"]
        ).dt.days

        # Orders per month (frequency normalised by tenure).
        aggregated["OrdersPerMonth"] = aggregated.apply(
            lambda row: row["Frequency"] / max(row["Tenure"] / 30, 1), axis=1
        )

        # Derive Churn label using the inactivity window.
        # This is used only when the aggregated data is being used for training.
        aggregated["Churn"] = (
            aggregated["Recency"] > INACTIVITY_WINDOW_DAYS
        ).astype(int)

        # Add country as a feature if present.
        if "Country" in df.columns:
            country_map = df.groupby("CustomerID")["Country"].first()
            aggregated   = aggregated.merge(country_map, on="CustomerID", how="left")

        # Drop intermediate date columns.
        aggregated = aggregated.drop(
            columns=["LastPurchaseDate", "FirstPurchaseDate"], errors="ignore"
        )

        print("Transaction aggregation complete. {} customers derived.".format(
            len(aggregated)
        ))
        return aggregated

    # ------------------------------------------------------------------
    # RFM proxy derivation from behavioural data (Case 1 / Case 2)
    # ------------------------------------------------------------------

    def derive_rfm_proxies(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        When raw transaction dates are not available, derive RFM-equivalent
        signals from existing behavioural columns.

        These are not true RFM but serve the same analytical purpose:
          Recency proxy   -> inverted DaySinceLastOrder
          Frequency proxy -> OrderCount
          Monetary proxy  -> CashbackAmount
        Interaction term between recency and monetary is a strong churn signal.
        """
        df = dataframe.copy()

        if "DaySinceLastOrder" in df.columns:
            max_days = df["DaySinceLastOrder"].max()
            df["Recency_Score"] = max_days - df["DaySinceLastOrder"]

        if "OrderCount" in df.columns:
            df["Frequency_Score"] = df["OrderCount"]

        if "CashbackAmount" in df.columns:
            df["Monetary_Score"] = df["CashbackAmount"]

        # Interaction feature: customers with high recency and low spend are
        # at elevated risk. This combined signal often outranks either alone.
        if "Recency_Score" in df.columns and "Monetary_Score" in df.columns:
            df["Recency_x_Monetary"] = df["Recency_Score"] * df["Monetary_Score"]

        # Engagement ratio: hours on app per order placed.
        if "HourSpendOnApp" in df.columns and "OrderCount" in df.columns:
            df["Engagement_Ratio"] = df["HourSpendOnApp"] / (df["OrderCount"] + 1)

        return df

    # ------------------------------------------------------------------
    # Fit and transform (training mode only)
    # ------------------------------------------------------------------

    def fit_transform(self, dataframe: pd.DataFrame, target_col: str = "Churn"):
        """
        Fit encoders and scaler on the training dataset.
        Returns X (scaled DataFrame), y (target Series), customer_ids (Series).

        CustomerID is extracted before any transformation so it is
        never lost or accidentally treated as a predictive feature.

        Training medians are stored so that missing columns in prediction
        mode can be filled with representative values from training data.
        """
        dataframe = self.fill_missing(dataframe)
        dataframe = self.derive_rfm_proxies(dataframe)

        customer_ids = dataframe["CustomerID"].reset_index(drop=True)
        y            = dataframe[target_col].reset_index(drop=True)
        X            = dataframe.drop(columns=[target_col, "CustomerID"], errors="ignore")

        # Encode categorical columns.
        # LabelEncoder is used here because tree-based models handle
        # ordinal-style integer encoding well without needing one-hot expansion.
        for col in X.select_dtypes(include="object").columns:
            encoder     = LabelEncoder()
            X[col]      = encoder.fit_transform(X[col].astype(str))
            self.label_encoders[col] = encoder

        # Store training medians for every feature.
        # These fill in missing columns when a new dataset has fewer features.
        for col in X.columns:
            self.training_medians[col] = float(X[col].median())

        # Scale all features to zero mean and unit variance.
        # This ensures that large-magnitude features like CashbackAmount
        # do not numerically dominate small-magnitude features like CityTier.
        self.feature_names = X.columns.tolist()
        X_scaled           = self.scaler.fit_transform(X)
        X_final            = pd.DataFrame(X_scaled, columns=self.feature_names)

        self.is_fitted = True

        print("Feature engineering complete. {} features prepared.".format(
            X_final.shape[1]
        ))
        return X_final, y, customer_ids

    # ------------------------------------------------------------------
    # Transform only (prediction modes)
    # ------------------------------------------------------------------

    def transform(self, dataframe: pd.DataFrame):
        """
        Transform a new dataset using the encoders and scaler fitted during training.

        Missing columns are filled with training medians rather than raising an error.
        This is the mechanism that allows Case 2 (partial data) to still produce
        predictions without crashing the pipeline.

        Returns X (scaled DataFrame) and customer_ids (Series).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FeatureEngineer has not been fitted. "
                "Run fit_transform on the training dataset first."
            )

        dataframe    = self.fill_missing(dataframe)
        dataframe    = self.derive_rfm_proxies(dataframe)
        customer_ids = dataframe["CustomerID"].reset_index(drop=True) \
                       if "CustomerID" in dataframe.columns else None

        X = dataframe.drop(columns=["Churn", "CustomerID"], errors="ignore")

        # Apply fitted label encoders to categorical columns.
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                X[col] = X[col].astype(str).apply(
                    lambda val, enc=encoder: (
                        enc.transform([val])[0] if val in enc.classes_ else 0
                    )
                )

        # Add any columns the model expects that are absent in this dataset.
        # Fill with training median so the model receives a valid input shape.
        missing_cols = [col for col in self.feature_names if col not in X.columns]
        if missing_cols:
            print("Filling {} missing columns with training medians: {}".format(
                len(missing_cols), missing_cols
            ))
        for col in missing_cols:
            X[col] = self.training_medians.get(col, 0.0)

        # Ensure column order matches exactly what the model was trained on.
        X        = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        X_final  = pd.DataFrame(X_scaled, columns=self.feature_names)

        return X_final, customer_ids
