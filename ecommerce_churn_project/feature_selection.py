"""
Layer 3 – Feature Selection
Ranks and selects top K features.
"""

from sklearn.feature_selection import SelectKBest, mutual_info_classif


class FeatureSelection:

    def __init__(self, k=5):
        self.k = k
        self.selector = SelectKBest(score_func=mutual_info_classif, k=k)

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        return self.selector.fit_transform(X, y)