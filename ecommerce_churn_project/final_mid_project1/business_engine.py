# ------------------------------------------------------------------
# business_engine.py  -  Layer 6
#
# Converts raw model probabilities into business-readable outputs.
#
# Probability Calibration:
#   Raw XGBoost probabilities can be systematically over or under
#   confident. Platt scaling (sigmoid calibration) adjusts them so
#   that a predicted probability of 0.70 actually means roughly 70%
#   of those customers churn in reality.
#
# Risk Segmentation:
#   Customers are divided into three tiers based on calibrated probability.
#   Thresholds are defined in config.py and can be adjusted per business.
#
# CLV Estimation:
#   Customer Lifetime Value is estimated as annual expected spend.
#   When CashbackAmount is available it is used as a spend proxy.
#   Revenue at risk is Churn_Probability * Annual_CLV, representing
#   the expected loss if no retention action is taken.
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM, CLV_ANNUAL_MULTIPLIER

HIGH_RISK   = "HIGH RISK"
MEDIUM_RISK = "MEDIUM RISK"
LOW_RISK    = "LOW RISK"


class BusinessEngine:

    def __init__(self):
        self.high_threshold   = RISK_THRESHOLD_HIGH
        self.medium_threshold = RISK_THRESHOLD_MEDIUM

    # ------------------------------------------------------------------

    def calibrate(self, model, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Apply Platt scaling to the trained model.
        Uses cv='prefit' because the model is already trained and we
        only want to learn the sigmoid mapping, not retrain the model.
        Falls back to the original model if calibration fails,
        which can happen with pipeline-wrapped models.
        """
        try:
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calibrated.fit(X_train, y_train)
            print("Probability calibration applied (Platt scaling).")
            return calibrated
        except Exception as error:
            print("Calibration skipped ({}). Using raw probabilities.".format(str(error)))
            return model

    # ------------------------------------------------------------------

    def generate_risk_scores(
        self,
        model,
        X_test: pd.DataFrame,
        customer_ids: pd.Series = None
    ) -> pd.DataFrame:
        """
        Generate churn probability and risk segment for each customer.
        CustomerID is attached so results can be traced back to individuals.
        """
        probabilities = model.predict_proba(X_test)[:, 1]

        result = pd.DataFrame({"Churn_Probability": probabilities})

        if customer_ids is not None:
            result.insert(0, "CustomerID", customer_ids.values)

        result["Risk_Segment"] = np.where(
            result["Churn_Probability"] >= self.high_threshold,
            HIGH_RISK,
            np.where(
                result["Churn_Probability"] >= self.medium_threshold,
                MEDIUM_RISK,
                LOW_RISK
            )
        )

        counts = result["Risk_Segment"].value_counts()
        print("Risk scoring complete.")
        print("  High Risk:   {}".format(counts.get(HIGH_RISK, 0)))
        print("  Medium Risk: {}".format(counts.get(MEDIUM_RISK, 0)))
        print("  Low Risk:    {}".format(counts.get(LOW_RISK, 0)))

        return result

    # ------------------------------------------------------------------

    def calculate_clv(
        self,
        risk_df: pd.DataFrame,
        raw_df: pd.DataFrame = None
    ) -> tuple:
        """
        Estimate annual CLV and expected revenue loss per customer.

        When CashbackAmount is available in the original data it is used
        as a monthly spend proxy (multiplied by 12 for annual value).
        When it is not available a flat default of 500 is used.

        Revenue at risk is the sum of (Churn_Probability * Annual_CLV)
        for all high-risk customers.
        """
        df = risk_df.copy()

        if raw_df is not None and "CashbackAmount" in raw_df.columns:
            cashback       = raw_df["CashbackAmount"].reset_index(drop=True)
            df["Annual_CLV"] = cashback.values[:len(df)] * CLV_ANNUAL_MULTIPLIER
        else:
            df["Annual_CLV"] = 500.0

        df["Estimated_Value_Loss"] = df["Churn_Probability"] * df["Annual_CLV"]

        total_at_risk = df.loc[
            df["Risk_Segment"] == HIGH_RISK, "Estimated_Value_Loss"
        ].sum()

        print("Total revenue at risk (High Risk customers): {:.2f}".format(
            total_at_risk
        ))

        return df, total_at_risk
