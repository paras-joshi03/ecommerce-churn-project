"""
Layer 2  Adaptive Feature Engineering
All transformations wrapped inside sklearn Pipeline.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class FeatureEngineering:

    def __init__(self, config):
        self.config = config
        self.pipeline = None

    
    def create_customer_features(self, df):

        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "CustomerID": "count",
            "TotalAmount": "sum"
        })

        rfm.columns = ["Recency", "Frequency", "Monetary"]
        rfm.reset_index(inplace=True)

        return rfm

   
    def build_pipeline(self, df):

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.pipeline = ColumnTransformer([
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])

        return self.pipeline

    def fit_transform(self, df):
        self.build_pipeline(df)
        return self.pipeline.fit_transform(df)

    def transform(self, df):
        return self.pipeline.transform(df)