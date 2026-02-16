"""
modeling_pricing_gap_03.py

Purpose:
    Model pricing inefficiency in a two-sided vehicle marketplace by identifying
    vehicle attributes associated with systematic overpricing or underpricing.
    Uses linear regression to produce interpretable demand and pricing signals
    for business decision-making.

Author:
    Nathan Gemechis
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# -----------------------------
# Load feature-engineered data
# -----------------------------
FEATURE_ENGINEERED_DATA_PATH = "/Users/nathangemechis/Desktop/python_tech/feature_engineered_data.csv"
OUTPUT_DIR = "/Users/nathangemechis/Desktop/python_tech"

df = pd.read_csv(FEATURE_ENGINEERED_DATA_PATH)

print("Feature-engineered data snapshot:")
print(df.head(), "\n")

print(f"Row count: {df.shape[0]}")

# -----------------------------
# Validate modeling inputs
# -----------------------------
required_columns = {
    "pricing_gap", "vehicle_age", "odometer", "condition",
    "body", "transmission", "make"
}

missing_cols = required_columns - set(df.columns)
assert not missing_cols, f"Missing required modeling columns: {missing_cols}"

assert df["pricing_gap"].notna().all(), "Missing target variable values detected"
assert df["mmr"].notna().all(), "Missing MMR values detected"

# -----------------------------
# Model specification
# Rationale:
#   pricing_gap serves as a proxy for excess demand or discounting
#   C() used to properly encode categorical predictors
# -----------------------------
formula = (
    "pricing_gap ~ vehicle_age + odometer + condition "
    "+ C(body) + C(transmission) + C(make)"
)

model = smf.ols(formula=formula, data=df)
results = model.fit()

# -----------------------------
# Model output & interpretation
# -----------------------------
print("\nOLS Regression Results:")
print(results.summary())

print("\nKey interpretation notes:")
print("- Positive coefficients indicate systematic overpricing / excess demand")
print("- Negative coefficients indicate discounting or weak demand")
print("- Categorical variables are interpreted relative to a baseline category")

# -----------------------------
# Export model results
# -----------------------------
with open(f"{OUTPUT_DIR}/model_summary.txt", "w") as f:
    f.write(results.summary().as_text())

coefficients = (
    results.params
    .reset_index()
    .rename(columns={"index": "feature", 0: "coefficient"})
)

coefficients.to_csv(f"{OUTPUT_DIR}/model_coefficients.csv", index=False)

print("\nModel artifacts exported:")
print("- model_summary.txt")
print("- model_coefficients.csv")
