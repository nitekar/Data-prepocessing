# requirements (install as needed):
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost librosa opencv-python face-recognition joblib

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss


# Paths - change to your actual CSV/Excel paths
social_profiles_path = "Customer_Dataset/customer_social_profiles.csv"   # from your sheet export
transactions_path = "Customer_Dataset/customer_transactions.csv"
id_mapping_path = "Customer_Dataset/customer_id_mapping.csv"  # optional; columns: customer_id_legacy, customer_id_new

# Load csvs
social = pd.read_csv(social_profiles_path)
trans = pd.read_csv(transactions_path)

# Quick type coercion & preview
print("social shape:", social.shape)
print("transactions shape:", trans.shape)

# If you have a mapping file:
if os.path.exists(id_mapping_path):
    mapping = pd.read_csv(id_mapping_path)
    trans = trans.merge(mapping, on='customer_id_legacy', how='left')
else:
    # Attempt naive join: sometimes legacy ids are transcribed as numbers vs codes:
    # If the social has numeric IDs embedded in new IDs, try to extract numeric part.
    # This is a heuristic — replace with mapping if available.
    try:
        social['legacy_guess'] = social['customer_id_new'].str.extract('(\d+)').astype(float)
        trans['legacy_num'] = pd.to_numeric(trans['customer_id_legacy'], errors='coerce')
        join_df = trans.merge(social, left_on='legacy_num', right_on='legacy_guess', how='left')
        # If many nulls, fall back to cross-join logic or ask for mapping file.
        trans = join_df.copy()
    except Exception as e:
        print("No mapping. Please provide mapping.csv. Falling back to left join on legacy id strings.")
        trans = trans.merge(social, left_on='customer_id_legacy', right_on='customer_id_new', how='left')

# Final merged dataset (join transactions to social profiles on the new ID)
# If mapping exists we have customer_id_new in trans; otherwise this may be null for many rows
merged = trans.copy()

# Ensure output folder exists
os.makedirs('merge-output', exist_ok=True)

# Basic merge validation: rows before/after and duplicates
pre_merge_rows = trans.shape[0]
pre_social_rows = social.shape[0]

# Drop obvious duplicate transaction rows
dupes_before = merged.duplicated().sum()
merged = merged.drop_duplicates()
dupes_after = merged.duplicated().sum()

# Basic cleaning & feature engineering
merged['purchase_date'] = pd.to_datetime(merged['purchase_date'], errors='coerce')
merged['purchase_amount'] = pd.to_numeric(merged['purchase_amount'], errors='coerce')

# Example engineered features:
merged['purchase_month'] = merged['purchase_date'].dt.month
merged['is_weekend'] = merged['purchase_date'].dt.weekday >= 5

# Convert categorical features
merged['product_category'] = merged['product_category'].astype('category')
merged['social_media_platform'] = merged['social_media_platform'].astype('category')

# Save merged
merged.to_csv('merge-output/merged_data.csv', index=False)
print("Saved merged_data.csv, shape:", merged.shape)

# Merge validation report
with open('merge-output/merge_validation.txt', 'w', encoding='utf-8') as fh:
    fh.write(f"pre_merge_transactions_rows: {pre_merge_rows}\n")
    fh.write(f"pre_social_profiles_rows: {pre_social_rows}\n")
    fh.write(f"rows_after_merge: {merged.shape[0]}\n")
    fh.write(f"duplicates_before_drop: {dupes_before}\n")
    fh.write(f"duplicates_after_drop: {dupes_after}\n")
    # Nulls summary
    fh.write("\nNull counts by column:\n")
    nulls = merged.isnull().sum()
    for col, n in nulls.items():
        fh.write(f"{col}: {n}\n")

    # How many transactions have no matched social profile (heuristic)
    if 'customer_id_new' in merged.columns:
        unmatched = merged['customer_id_new'].isna().sum()
        fh.write(f"\nunmatched_customer_id_new (na): {unmatched}\n")

    # Product category coverage
    if 'product_category' in merged.columns:
        fh.write(f"\nproduct_category value counts:\n")
        vc = merged['product_category'].value_counts(dropna=False)
        for k, v in vc.items():
            fh.write(f"{k}: {v}\n")

print('Wrote merge-output/merge_validation.txt')


# 1. Summary statistics & types
print(merged.describe(include='all'))
print(merged.dtypes)

# 2. Plot examples (≥3)
plt.figure(figsize=(6,4))
sns.histplot(merged['purchase_amount'].dropna(), kde=True)
plt.title('Distribution of Purchase Amount')
plt.xlabel('Purchase Amount')
plt.savefig('merge-output/plot_purchase_amount_dist.png')
plt.close()

plt.figure(figsize=(6,4))
sns.boxplot(x='product_category', y='purchase_amount', data=merged)
plt.title('Purchase amount by Product Category (outliers visible)')
plt.xticks(rotation=45)
plt.savefig('merge-output/plot_box_by_category.png')
plt.close()

# Correlation heatmap for numeric vars
numcols = merged.select_dtypes(include=[np.number]).columns.tolist()
if len(numcols) > 1:
    plt.figure(figsize=(6,5))
    sns.heatmap(merged[numcols].corr(), annot=True, fmt=".2f")
    plt.title('Numeric Correlations')
    plt.savefig('merge-output/plot_correlations.png')
    plt.close()

print('Generated EDA plots in merge-output/')
