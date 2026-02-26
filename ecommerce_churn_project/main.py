"""
Main Pipeline Runner
"""

import yaml

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineering
from feature_selection import FeatureSelection
from modeling import ClassificationModel
from model_selection import ModelSelection


def main():

    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Layer 1
    ingestion = DataIngestion(config)
    df = ingestion.load_data()

    # Layer 2
    fe = FeatureEngineering(config)
    customer_df = fe.create_customer_features(df)

    X = customer_df.drop("CustomerID", axis=1)

    X_transformed = fe.fit_transform(X)

    # Fake target for now (placeholder)
    import numpy as np
    y = np.random.randint(0, 2, size=X_transformed.shape[0])

    # Layer 3
    fs = FeatureSelection(k=3)
    X_selected = fs.fit_transform(X_transformed, y)

    # Layer 4
    model = ClassificationModel(config)
    X_test, y_test = model.train(X_selected, y)

    y_pred = model.predict(X_test)

    # Layer 5
    selector = ModelSelection()
    selector.evaluate(y_test, y_pred)


if __name__ == "__main__":
    main()