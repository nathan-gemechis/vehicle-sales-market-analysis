"""
data_validation_01.py

Purpose:
    Validate and clean raw vehicle marketplace transaction data prior to
    feature engineering and modeling. This script enforces data integrity,
    standardizes categorical fields, and removes invalid or incomplete records.

Author:
    Nathan Gemechis
"""

import pandas as pd
import numpy as np

# -----------------------------
# Load raw data
# -----------------------------
RAW_DATA_PATH = "/Users/nathangemechis/Desktop/python_tech/vehicle_sales.csv"
OUTPUT_PATH = "cleaned.csv"

df = pd.read_csv(RAW_DATA_PATH)

print("Initial data snapshot:")
print(df.head(), "\n")

print("Initial schema:")
print(df.info(), "\n")

print("Initial descriptive statistics:")
print(df.describe(), "\n")

print("Missing values by column:")
print(df.isna().sum(), "\n")

print(f"Duplicate rows: {df.duplicated().sum()}")


print(f"Initial row count: {df.shape[0]}")

# -----------------------------
# Basic integrity checks
# -----------------------------
required_columns = {
    "year", "make", "model", "sellingprice", "mmr",
    "odometer", "condition", "seller", "saledate"
}

missing_required = required_columns - set(df.columns)
assert not missing_required, f"Missing required columns: {missing_required}"

# -----------------------------
# Remove incomplete records
# Rationale: pricing and demand analysis requires fully observed transactions
# -----------------------------
df = df[df.notna().all(axis=1)]
print(f"Row count after dropping missing values: {df.shape[0]}")

# -----------------------------
# Standardize categorical fields
# -----------------------------
df["make"] = df["make"].str.upper()
df["state"] = df["state"].str.upper()

# -----------------------------
# Convert salesdate to date type
# -----------------------------
df['saledate'] = pd.to_datetime(df['saledate'], errors="coerce")

# -----------------------------
# Remove non-analytical identifiers
# -----------------------------
if "vin" in df.columns:
    df = df.drop(columns=["vin"])

# -----------------------------
# Sanity checks on numeric fields
# -----------------------------
assert (df["sellingprice"] > 0).all(), "Non-positive selling prices detected"
assert (df["mmr"] > 0).all(), "Non-positive MMR values detected"
assert (df["odometer"] >= 0).all(), "Negative odometer values detected"

# -----------------------------
# Final dataset summary
# -----------------------------
print("\nFinal cleaned dataset summary:")
print(df.describe())
print(f"Final row count: {df.shape[0]}")

# -----------------------------
# Export cleaned data
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f"Cleaned data exported to {OUTPUT_PATH}")
