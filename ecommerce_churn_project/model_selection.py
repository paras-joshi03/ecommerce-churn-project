"""
Layer 5 – Model Selection Engine
Evaluates and selects best model.
"""

from sklearn.metrics import accuracy_score, roc_auc_score


class ModelSelection:

    def evaluate(self, y_true, y_pred):

        accuracy = accuracy_score(y_true, y_pred)

        print("Accuracy:", accuracy)

        return accuracy