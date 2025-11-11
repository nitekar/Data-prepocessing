import pandas as pd
import numpy as np

# Load the datasets
transactions = pd.read_csv('customer_transactions.csv')
social_profiles = pd.read_csv('customer_social_profiles.csv')

# Clean column names (remove any leading/trailing spaces)
transactions.columns = transactions.columns.str.strip()
social_profiles.columns = social_profiles.columns.str.strip()

print("Transaction columns:", transactions.columns.tolist())
print("Social profiles columns:", social_profiles.columns.tolist())

# Step 1: Create customer ID mapping
# Convert legacy IDs (e.g., 151) to new format (e.g., A151)
transactions['customer_id_new'] = 'A' + transactions['customer_id_legacy'].astype(str)

print(f"\nSample transactions after ID mapping:")
print(transactions[['customer_id_legacy', 'customer_id_new']].head())

# Step 2: Aggregate transaction data by customer
transaction_features = transactions.groupby('customer_id_new').agg({
    'transaction_id': 'count',  # Total number of purchases
    'purchase_amount': ['sum', 'mean'],  # Total and average spending
    'customer_rating': 'mean',  # Average rating
    'product_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else None  # Most frequent category
}).reset_index()

# Flatten column names
transaction_features.columns = [
    'customer_id_new', 
    'total_purchases', 
    'total_spent', 
    'avg_purchase_amount',
    'avg_customer_rating', 
    'favorite_category'
]

# Get last purchase information
last_purchase = transactions.sort_values('purchase_date').groupby('customer_id_new').last()[
    ['product_category', 'purchase_date']
].reset_index()
last_purchase.columns = ['customer_id_new', 'last_category_purchased', 'last_purchase_date']

# Merge last purchase info
transaction_features = transaction_features.merge(last_purchase, on='customer_id_new', how='left')

print(f"\nTransaction features shape: {transaction_features.shape}")

# Step 3: Aggregate social profile data by customer
social_features = social_profiles.groupby('customer_id_new').agg({
    'social_media_platform': ['count', lambda x: x.nunique()],  # Total profiles and unique platforms
    'engagement_score': 'mean',  # Average engagement
    'purchase_interest_score': 'mean',  # Average purchase interest
    'review_sentiment': lambda x: x.mode()[0] if len(x.mode()) > 0 else None  # Most common sentiment
}).reset_index()

# Flatten column names
social_features.columns = [
    'customer_id_new',
    'total_social_profiles',
    'platform_diversity',
    'avg_engagement_score',
    'avg_purchase_interest_score',
    'dominant_sentiment'
]

print(f"Social features shape: {social_features.shape}")

# Step 4: Calculate sentiment distribution for each customer
sentiment_pivot = social_profiles.groupby(['customer_id_new', 'review_sentiment']).size().unstack(fill_value=0)
sentiment_pivot.columns = [f'sentiment_{col.lower()}_count' for col in sentiment_pivot.columns]
sentiment_pivot = sentiment_pivot.reset_index()

# Merge sentiment distribution
social_features = social_features.merge(sentiment_pivot, on='customer_id_new', how='left')

# Step 5: Get platform list for each customer (optional)
platform_list = social_profiles.groupby('customer_id_new')['social_media_platform'].apply(
    lambda x: ', '.join(x.unique())
).reset_index()
platform_list.columns = ['customer_id_new', 'platforms_used']

social_features = social_features.merge(platform_list, on='customer_id_new', how='left')

# Step 6: Merge transaction and social features
merged_data = social_features.merge(transaction_features, on='customer_id_new', how='inner')

# Step 7: Save merged dataset
merged_data.to_csv('merged_customer_data.csv', index=False)

print(f"\n{'='*60}")
print(f"Merged dataset created successfully!")
print(f"{'='*60}")
print(f"Total customers in merged dataset: {len(merged_data)}")
print(f"\nMerged dataset columns:")
print(merged_data.columns.tolist())
print(f"\nFirst few rows:")
print(merged_data.head())
print(f"\nDataset shape: {merged_data.shape}")
print(f"\nMissing values:")
print(merged_data.isnull().sum())
print(f"\nDataset saved as 'merged_customer_data.csv'")