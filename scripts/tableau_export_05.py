"""
05_tableau_export.py

Purpose:
    Prepare clean, analytics-ready CSV tables for Tableau dashboards.
    This script consolidates modeling and clustering outputs into
    denormalized, business-facing datasets optimized for BI consumption.

Author:
    Nathan Gemechis
"""

import pandas as pd
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
FEATURE_ENGINEERED_DATA_PATH = "/Users/nathangemechis/Desktop/python_tech/feature_engineered_data.csv"
SELLER_CLUSTER_PATH = "/Users/nathangemechis/Desktop/python_tech/seller_clusters.csv"

OUTPUT_DIR = "/Users/nathangemechis/Desktop/python_tech/outputs/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(FEATURE_ENGINEERED_DATA_PATH)
seller_stats = pd.read_csv(SELLER_CLUSTER_PATH)



# -----------------------------
# Vehicle Pricing Summary (Transaction-Level)
# -----------------------------
vehicle_pricing_summary = df[
    [
        "saledate",
        "make",
        "model",
        "state",
        "vehicle_age",
        "sellingprice",
        "mmr",
        "pricing_gap",
        "pricing_gap_pct",
        "is_above_mmr",
        "odometer",
        "condition",
        "seller"
    ]
]

vehicle_pricing_summary.to_csv(
    f"{OUTPUT_DIR}/vehicle_pricing_summary.csv",
    index=False
)

# -----------------------------
# Seller Segments (Seller-Level)
# -----------------------------
cluster_labels = {
    0: "Market-Aligned Sellers",
    1: "Opportunistic Sellers",
    2: "High-Variance Sellers"
}

seller_stats["cluster_label"] = seller_stats["cluster"].map(cluster_labels)

seller_stats.to_csv(
    f"{OUTPUT_DIR}/seller_segments.csv",
    index=False
)


pricing_trends = (
    df
    .groupby(pd.Grouper(key="saledate", freq="M"))
    .agg(
        mean_pricing_gap=("pricing_gap", "mean"),
        pct_above_mmr=("is_above_mmr", "mean"),
        transaction_volume=("pricing_gap", "count")
    )
    .reset_index()
)


pricing_trends.to_csv(
    f"{OUTPUT_DIR}/pricing_trends.csv",
    index=False
)

# -----------------------------
# Demand Signal Aggregates
# -----------------------------
df["vehicle_age_bucket"] = pd.cut(
    df["vehicle_age"],
    bins=[0, 2, 5, 10, 20, 50],
    labels=["0-2", "3-5", "6-10", "11-20", "20+"]
)

demand_signals = (
    df
    .groupby(["make", "vehicle_age_bucket"])
    .agg(
        mean_pricing_gap=("pricing_gap", "mean"),
        transaction_count=("pricing_gap", "count")
    )
    .reset_index()
)

demand_signals.to_csv(
    f"{OUTPUT_DIR}/demand_signals.csv",
    index=False
)

print("\nTableau export completed:")
print("- vehicle_pricing_summary.csv")
print("- seller_segments.csv")
print("- pricing_trends.csv")
print("- demand_signals.csv")
