"""
utils.py — Shared helper functions for CMS Open Payments Analysis
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────
# DATA LOADING & COLUMN SELECTION
# ─────────────────────────────────────────────────────────────

SELECTED_COLUMNS = [
    "Physician_Primary_Type",
    "Physician_Specialty",
    "Recipient_State",
    "Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
    "Total_Amount_of_Payment_USDollars",
    "Number_of_Payments_Included_in_Total_Amount",
    "Nature_of_Payment_or_Transfer_of_Value",
    "Form_of_Payment_or_Transfer_of_Value",
    "Date_of_Payment",
    "Covered_Recipient_Type",
]

RENAME_MAP = {
    "Physician_Primary_Type": "physician_type",
    "Physician_Specialty": "specialty",
    "Recipient_State": "state",
    "Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name": "submitter",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name": "company",
    "Total_Amount_of_Payment_USDollars": "payment_amount",
    "Number_of_Payments_Included_in_Total_Amount": "num_payments",
    "Nature_of_Payment_or_Transfer_of_Value": "payment_nature",
    "Form_of_Payment_or_Transfer_of_Value": "payment_form",
    "Date_of_Payment": "date",
    "Covered_Recipient_Type": "recipient_type",
}


def load_data(filepath: str, nrows: int = 200_000) -> pd.DataFrame:
    """
    Load a subset of the CMS dataset, select relevant columns, and rename them.

    Args:
        filepath: Path to the raw CSV file.
        nrows: Number of rows to read (default 200k for performance).

    Returns:
        Clean DataFrame with renamed columns.
    """
    df = pd.read_csv(filepath, nrows=nrows, low_memory=False, usecols=SELECTED_COLUMNS)
    df.rename(columns=RENAME_MAP, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame:
    - Drop rows with missing payment amounts
    - Remove negative or zero payments
    - Parse dates
    - Strip whitespace from strings
    """
    df = df.dropna(subset=["payment_amount"])
    df = df[df["payment_amount"] > 0].copy()

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month.fillna(0).astype(int)
    df["quarter"] = df["date"].dt.quarter.fillna(0).astype(int)

    # Strip extra whitespace
    str_cols = ["physician_type", "specialty", "state", "company", "payment_nature", "payment_form", "recipient_type"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").str.strip()

    df["num_payments"] = df["num_payments"].fillna(1).astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features for modelling:
    - log_payment (log1p transform to handle skew)
    - payment_per_transaction
    - Encoded categorical columns
    """
    df = df.copy()
    df["log_payment"] = np.log1p(df["payment_amount"])
    df["payment_per_tx"] = df["payment_amount"] / df["num_payments"].clip(lower=1)

    le = LabelEncoder()
    for col in ["specialty", "payment_nature", "state", "physician_type", "payment_form"]:
        if col in df.columns:
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    return df


# ─────────────────────────────────────────────────────────────
# ANOMALY DETECTION (IQR-based)
# ─────────────────────────────────────────────────────────────

def detect_anomalies_iqr(df: pd.DataFrame, column: str = "payment_amount", factor: float = 3.0) -> pd.DataFrame:
    """
    Flag anomalies using the IQR method.

    Args:
        df: Input DataFrame.
        column: Column to detect anomalies on.
        factor: IQR multiplier (3.0 = extreme outlier threshold).

    Returns:
        DataFrame with additional boolean column `is_anomaly`.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    df = df.copy()
    df["is_anomaly"] = (df[column] < lower) | (df[column] > upper)
    return df


def detect_anomalies_zscore(df: pd.DataFrame, column: str = "payment_amount", threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag anomalies using the Z-score method.
    """
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    df["z_score"] = (df[column] - mean) / std
    df["is_anomaly_z"] = df["z_score"].abs() > threshold
    return df


# ─────────────────────────────────────────────────────────────
# BUSINESS INTERPRETATION HELPERS
# ─────────────────────────────────────────────────────────────

CLUSTER_LABELS = {
    0: "🟢 Low-Value Routine Payments — Food, beverage, and small promotional transfers",
    1: "🔵 Mid-Value Educational Payments — Conferences, CME, and travel reimbursements",
    2: "🟡 High-Value Consulting Arrangements — Speaker fees and research grants",
    3: "🔴 Top-Tier Strategic Payments — Royalties, equity, ownership, and IP transfers",
}

NATURE_INSIGHTS = {
    "Food and Beverage": "Often small-value routine payments; high in volume, lower policy risk.",
    "Consulting Fee": "Moderate-to-high value; raises potential conflict-of-interest concerns.",
    "Travel and Lodging": "Mid-range; common for conference-sponsored attendance.",
    "Education": "Low-risk; supports CME and professional development.",
    "Research": "Variable; can be high-value grants; closely scrutinized.",
    "Speaker Honoraria": "High policy risk; tied directly to product promotion.",
    "Royalty or License": "Very high value; indicates IP ownership/patent arrangements.",
}


def get_cluster_label(cluster_id: int) -> str:
    return CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")


def get_nature_insight(nature: str) -> str:
    for key in NATURE_INSIGHTS:
        if key.lower() in nature.lower():
            return NATURE_INSIGHTS[key]
    return "Payment type not categorized. Review manually for policy implications."
