# ------------------------------------------------------------------
# config.py
#
# Central configuration for the entire churn prediction system.
# Every threshold, path, and constant lives here.
# Changing a value here propagates through the entire pipeline.
# ------------------------------------------------------------------


# Inactivity window used to derive the Churn label from raw
# transaction dates when no Churn column exists in the source data.
# 90 days is the standard definition in e-commerce literature.
INACTIVITY_WINDOW_DAYS = 90


# Risk segmentation thresholds applied to calibrated probabilities.
RISK_THRESHOLD_HIGH   = 0.70
RISK_THRESHOLD_MEDIUM = 0.40


# Number of top features retained after feature selection.
FEATURE_TOP_K = 12


# Train / test split configuration.
TEST_SIZE   = 0.20
RANDOM_SEED = 42


# Paths where trained artifacts are saved after Mode 1 training.
# The dashboard and prediction modes load from these paths.
import os
SAVE_DIR       = "saved_model"
MODEL_PATH     = os.path.join(SAVE_DIR, "best_model.joblib")
ENGINEER_PATH  = os.path.join(SAVE_DIR, "feature_engineer.joblib")
SELECTOR_PATH  = os.path.join(SAVE_DIR, "feature_selector.joblib")
DRIFT_PATH     = os.path.join(SAVE_DIR, "drift_monitor.joblib")
SURVIVAL_PATH  = os.path.join(SAVE_DIR, "survival_model.joblib")
META_PATH      = os.path.join(SAVE_DIR, "training_meta.joblib")


# Column name aliases. Maps common alternative names to the
# standard names the pipeline expects. Comparison is case-insensitive.
COLUMN_ALIASES = {
    "customer_id":   "CustomerID",
    "cust_id":       "CustomerID",
    "id":            "CustomerID",
    "client_id":     "CustomerID",
    "churn":         "Churn",
    "churned":       "Churn",
    "is_churn":      "Churn",
    "target":        "Churn",
    "invoicedate":   "InvoiceDate",
    "invoice_date":  "InvoiceDate",
    "order_date":    "InvoiceDate",
    "purchase_date": "InvoiceDate",
    "date":          "InvoiceDate",
    "unitprice":     "UnitPrice",
    "unit_price":    "UnitPrice",
    "price":         "UnitPrice",
    "invoiceno":     "InvoiceNo",
    "invoice_no":    "InvoiceNo",
    "qty":           "Quantity",
    "quantity":      "Quantity",
}


# All behavioural columns the trained model was built on.
# In Case 2, missing columns from this list are filled with
# the median values recorded during training.
BEHAVIOURAL_COLS = [
    "Tenure",
    "PreferredLoginDevice",
    "CityTier",
    "WarehouseToHome",
    "PreferredPaymentMode",
    "Gender",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "PreferedOrderCat",
    "SatisfactionScore",
    "MaritalStatus",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]


# Minimum columns required to operate in raw transaction mode (Case 3).
TRANSACTION_MIN_COLS = ["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]


# CLV estimation: monthly spend multiplied by this to get annual value.
CLV_ANNUAL_MULTIPLIER = 12


# KS test significance level for drift detection.
# p-value below this threshold indicates distribution shift.
DRIFT_SIGNIFICANCE = 0.05


# GridSearch parameter grid. Kept small to balance speed and coverage.
GRIDSEARCH_PARAMS = {
    "clf__n_estimators":  [50, 100],
    "clf__max_depth":     [3, 5],
    "clf__learning_rate": [0.05, 0.1],
}
GRIDSEARCH_CV = 3
