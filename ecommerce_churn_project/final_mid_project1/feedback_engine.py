# ------------------------------------------------------------------
# feedback_engine.py  -  Layer 7  (Human-in-the-Loop)
#
# Allows analysts to manually override the model's risk assessment
# for specific customers. All overrides are stored in a JSON file
# so the audit trail persists across sessions.
#
# This layer serves two purposes:
#   1. Practical: account managers often have context the model lacks,
#      such as a customer who has already renewed their contract.
#   2. Systemic: accumulated overrides can be reviewed to identify
#      patterns where the model is consistently wrong, which informs
#      future retraining.
# ------------------------------------------------------------------

import os
import json
import pandas as pd
from datetime import datetime

FEEDBACK_FILE = "feedback_log.json"


class FeedbackEngine:

    def __init__(self):
        self.overrides = self._load()

    def _load(self) -> list:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(self.overrides, f, indent=2)

    def add_override(
        self,
        customer_id,
        original_segment: str,
        new_segment: str,
        reason: str,
        analyst: str = "Analyst"
    ) -> dict:
        """
        Record a manual risk override for a specific customer.
        Timestamp and analyst name are stored for audit purposes.
        """
        record = {
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "customer_id":      str(customer_id),
            "original_segment": original_segment,
            "new_segment":      new_segment,
            "reason":           reason,
            "analyst":          analyst,
        }
        self.overrides.append(record)
        self._save()
        return record

    def apply_overrides(self, risk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all stored overrides to the risk DataFrame.
        A column 'Override' is added so the dashboard can visually
        distinguish model predictions from analyst adjustments.
        """
        df = risk_df.copy()
        df["Override"] = False

        for record in self.overrides:
            mask = df["CustomerID"].astype(str) == str(record["customer_id"])
            if mask.any():
                df.loc[mask, "Risk_Segment"] = record["new_segment"]
                df.loc[mask, "Override"]     = True

        return df

    def get_log(self) -> list:
        return self.overrides

    def clear_log(self):
        self.overrides = []
        self._save()
