"""
seller_clustering_04.py

Purpose:
    Segment marketplace sellers based on pricing behavior using unsupervised
    clustering. Sellers are grouped according to pricing consistency, tendency
    to overprice or discount, and frequency of extreme pricing events.

Methodology:
    - Aggregate transaction-level data to seller-level metrics
    - Filter sellers with insufficient transaction history
    - Apply log transformations to skewed features
    - Standardize features and apply KMeans clustering

Author:
    Nathan Gemechis
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Configuration
# -----------------------------
FEATURE_ENGINEERED_DATA_PATH = "FEATURE ENGINEERED DATA PATH"
OUTPUT_DIR = "OUTPUT DIRECTORY"

N_CLUSTERS = 3
MIN_TRANSACTIONS = 10
RANDOM_STATE = 42

# -----------------------------
# Load feature-engineered data
# -----------------------------
df = pd.read_csv(FEATURE_ENGINEERED_DATA_PATH)

print("Feature-engineered data snapshot:")
print(df.head(), "\n")

# -----------------------------
# Aggregate seller-level metrics
# -----------------------------
seller_stats = (
    df
    .groupby("seller")
    .agg(
        mean_pricing_gap=("pricing_gap", "mean"),
        pricing_gap_variance=("pricing_gap", "var"),
        extreme_pricing_count=("pricing_gap", lambda x: (x.abs() > 5000).sum()),
        transaction_count=("pricing_gap", "count")
    )
    .reset_index()
)

# Sellers with a single transaction have undefined variance
seller_stats["pricing_gap_variance"] = seller_stats["pricing_gap_variance"].fillna(0)

print(f"Total sellers before filtering: {seller_stats.shape[0]}")

# -----------------------------
# Filter low-activity sellers
# Rationale:
#   Sellers with very few transactions do not provide reliable
#   estimates of pricing behavior and can distort clustering.
# -----------------------------
seller_stats = seller_stats[
    seller_stats["transaction_count"] >= MIN_TRANSACTIONS
].reset_index(drop=True)

print(f"Total sellers after filtering (>= {MIN_TRANSACTIONS} transactions): {seller_stats.shape[0]}")

# -----------------------------
# Feature transformation
# Rationale:
#   Pricing variance and extreme pricing counts are highly skewed.
#   Log transformations reduce outlier influence and improve cluster stability.
# -----------------------------
seller_stats["log_pricing_gap_variance"] = np.log1p(seller_stats["pricing_gap_variance"])
seller_stats["log_extreme_pricing_count"] = np.log1p(seller_stats["extreme_pricing_count"])

cluster_features = [
    "mean_pricing_gap",
    "log_pricing_gap_variance",
    "log_extreme_pricing_count"
]

X = seller_stats[cluster_features]

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# KMeans clustering
# -----------------------------
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=20
)

seller_stats["cluster"] = kmeans.fit_predict(X_scaled)

# -----------------------------
# Cluster diagnostics
# -----------------------------
print("\n=== SELLER CLUSTERING RESULTS ===")
print(f"Number of clusters: {N_CLUSTERS}")

print("\nSellers per cluster:")
print(seller_stats["cluster"].value_counts().sort_index())

cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=cluster_features
)

print("\nCluster centers (transformed feature space):")
print(cluster_centers)

# -----------------------------
# Export results
# -----------------------------
seller_stats.to_csv(f"{OUTPUT_DIR}/seller_clusters.csv", index=False)
cluster_centers.to_csv(f"{OUTPUT_DIR}/seller_cluster_centers.csv", index=False)

print("\nClustering outputs exported:")
print("- seller_clusters.csv")
print("- seller_cluster_centers.csv")
