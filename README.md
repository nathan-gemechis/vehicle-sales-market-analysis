# Vehicle Marketplace Pricing & Seller Intelligence Pipeline

End-to-end analytical pipeline for identifying pricing inefficiencies, demand signals, and seller behavior segmentation in a two-sided vehicle marketplace.

## Overview

This project provides a modular, reproducible pipeline to validate raw vehicle transaction data, engineer pricing efficiency features, model systematic overpricing and underpricing behavior, segment sellers using unsupervised learning, and export analytics-ready datasets for business intelligence dashboards.

The workflow integrates data validation, feature construction, econometric modeling, clustering, and BI-ready exports to support pricing strategy, seller evaluation, and marketplace optimization.

## Key Highlights

- Robust data validation and integrity enforcement  
- Pricing efficiency feature engineering (pricing gaps and demand signals)  
- Linear regression modeling for interpretable pricing insights  
- Seller segmentation using KMeans clustering  
- Log-transformed and standardized feature preprocessing  
- Tableau-ready exports for transaction, seller, and demand analysis  

## Repository Structure

- Data  
- Outputs  
- Scripts  
- data_validation_01.py  
- feature_engineering_02.py  
- modeling_pricing_gap_03.py  
- seller_clustering_04.py  
- tableau_export_05.py  

## Execution Pipeline

The analysis is organized into five modular scripts designed to be run sequentially:

1. **data_validation_01.py** – Load raw transaction data, enforce schema requirements, remove incomplete records, standardize categorical fields, perform integrity checks, and export a cleaned dataset.

2. **feature_engineering_02.py** – Create analytical pricing features including vehicle age, pricing gaps, percentage deviations from MMR, and above-market indicators. Produces aggregated pricing metrics by make.

3. **modeling_pricing_gap_03.py** – Estimate an OLS regression model using `pricing_gap` as the dependent variable to identify vehicle characteristics associated with systematic overpricing or discounting. Exports model summaries and coefficient tables.

4. **seller_clustering_04.py** – Aggregate transaction data to seller-level metrics, apply log transformations to skewed variables, standardize features, and segment sellers using KMeans clustering.

5. **tableau_export_05.py** – Consolidate transaction-level, seller-level, and demand-level metrics into denormalized, BI-ready CSV files optimized for Tableau dashboards.

Running the scripts in sequence produces a complete pricing intelligence and seller segmentation workflow.

## Methodological Framework

### Pricing Inefficiency Modeling
- Target variable: `pricing_gap = sellingprice - mmr`
- Continuous predictors: vehicle age, odometer, condition  
- Categorical predictors: body type, transmission, make  
- Interpretation:
  - Positive coefficients → systematic overpricing / excess demand  
  - Negative coefficients → discounting / weaker demand  

### Seller Segmentation
- Seller-level aggregation metrics:
  - Mean pricing gap  
  - Pricing gap variance  
  - Frequency of extreme pricing deviations  
- Log transformations applied to skewed features  
- Standardization via `StandardScaler`  
- Clustering via KMeans (k = 3)

## Outputs

The pipeline produces:

- cleaned.csv – Fully validated transaction dataset  
- feature_engineered_data.csv – Enriched transaction-level dataset  
- aggregate_make.csv – Brand-level pricing metrics  
- model_summary.txt – OLS regression results  
- model_coefficients.csv – Structured coefficient output  
- seller_clusters.csv – Seller-level cluster assignments  
- seller_cluster_centers.csv – Cluster centroid diagnostics  
- vehicle_pricing_summary.csv – Tableau-ready transaction summary  
- seller_segments.csv – Seller segmentation with business labels  
- pricing_trends.csv – Monthly pricing dynamics  
- demand_signals.csv – Demand intensity by make and vehicle age bucket  

## Dependencies

The Python scripts use the following packages:

- **pandas** – data manipulation  
- **numpy** – numerical operations  
- **statsmodels** – regression modeling  
- **scikit-learn** – clustering and scaling  

## Installation

```bash
pip install pandas numpy statsmodels scikit-learn
```

## How to Run

- Place all scripts and raw data in the same project directory  
- Update file path constants in each script  
- Execute scripts sequentially:

```bash
python data_validation_01.py
python feature_engineering_02.py
python modeling_pricing_gap_03.py
python seller_clustering_04.py
python tableau_export_05.py
```

- This runs the full pipeline and produces all analytical and BI-ready outputs.
