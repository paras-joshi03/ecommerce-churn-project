"""
Layer 4 – Modeling Layer
Separate classes for classification and survival.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class ClassificationModel:

    def __init__(self, config):
        self.test_size = config["modeling"]["test_size"]
        self.random_state = config["modeling"]["random_state"]
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.model.fit(X_train, y_train)

        return X_test, y_test

    def predict(self, X):
        return self.model.predict(X)