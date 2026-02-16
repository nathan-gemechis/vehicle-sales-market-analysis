"""
feature_engineering_02.py

Purpose:
    Create analytical features and aggregated summaries for pricing efficiency
    and demand signal analysis in a two-sided vehicle marketplace. This script
    derives core pricing metrics and produces Tableau-ready summary tables.

Author:
    Nathan Gemechis
"""

import pandas as pd
import numpy as np

# -----------------------------
# Load cleaned data
# -----------------------------
CLEANED_DATA_PATH = "PATH TO CLEANED DATA"
OUTPUT_DIR = "."

df = pd.read_csv(CLEANED_DATA_PATH)

print("Cleaned data snapshot:")
print(df.head(), "\n")

print(f"Row count: {df.shape[0]}")

# -----------------------------
# Feature engineering
# -----------------------------
CURRENT_YEAR = 2026

df["vehicle_age"] = CURRENT_YEAR - df["year"]

df["pricing_gap"] = df["sellingprice"] - df["mmr"]
df["pricing_gap_pct"] = df["pricing_gap"] / df["mmr"]

df["is_above_mmr"] = df["pricing_gap"] > 0


# -----------------------------
# Time-Series Pricing Trends
# -----------------------------
df["saledate"] = pd.to_datetime(df["saledate"], utc=True).dt.date

print(df["saledate"].dtype)   # should be date

df = df.dropna(subset=["saledate"])   # remove bad dates


# -----------------------------
# Sanity checks on engineered features
# -----------------------------
assert (df["vehicle_age"] >= 0).all(), "Negative vehicle age detected"
assert df["pricing_gap"].notna().all(), "Missing pricing gap values"
assert df["pricing_gap_pct"].notna().all(), "Missing pricing gap percentage values"

print("Feature engineering completed successfully.\n")

# -----------------------------
# Aggregate pricing metrics by make
# Rationale:
#   - Identify systematic over/underpricing by vehicle brand
#   - Serve as a high-level demand signal for Tableau dashboards
# -----------------------------
aggregate_make = (
    df
    .groupby("make")
    .agg(
        avg_selling_price=("sellingprice", "mean"),
        avg_mmr=("mmr", "mean"),
        avg_pricing_gap=("pricing_gap", "mean"),
        avg_pricing_gap_pct=("pricing_gap_pct", "mean"),
        pct_above_mmr=("is_above_mmr", "mean"),
        avg_vehicle_age=("vehicle_age", "mean"),
        transaction_count=("sellingprice", "count")
    )
    .reset_index()
)

print("Aggregated pricing metrics by make:")
print(aggregate_make.head(), "\n")

# -----------------------------
# Export feature tables
# -----------------------------
df.to_csv(f"{OUTPUT_DIR}/feature_engineered_data.csv", index=False)
aggregate_make.to_csv(f"{OUTPUT_DIR}/aggregate_make.csv", index=False)

print("Feature-engineered datasets exported:")
print("- feature_engineered_data.csv")
print("- aggregate_make.csv")
