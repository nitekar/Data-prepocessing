from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os
import os
import pandas as pd


def load_merged_aggregated():
    # Prefer an aggregated merged dataset if present
    agg_path = os.path.join('merge-output', 'merged_data_aggregated.csv')
    fallback_path = os.path.join('merge-output', 'merged_data.csv')

    if os.path.exists(agg_path):
        return pd.read_csv(agg_path)

    # If aggregated file doesn't exist, try to build it on the fly
    trans_path = os.path.join('Customer_Dataset', 'customer_transactions.csv')
    social_path = os.path.join('Customer_Dataset', 'customer_social_profiles.csv')
    if os.path.exists(trans_path) and os.path.exists(social_path):
        trans = pd.read_csv(trans_path)
        social = pd.read_csv(social_path)
        trans = trans.copy()
        trans['customer_id_new'] = 'A' + trans['customer_id_legacy'].astype(str)

        # aggregate social profiles per customer
        agg_num = social.groupby('customer_id_new').agg({
            'engagement_score': 'mean',
            'purchase_interest_score': 'mean'
        }).rename(columns={'engagement_score':'avg_engagement_score','purchase_interest_score':'avg_purchase_interest_score'})
        mode_sentiment = social.groupby('customer_id_new')['review_sentiment'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        mode_platform = social.groupby('customer_id_new')['social_media_platform'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        counts = social.groupby('customer_id_new').size().rename('social_profile_count')
        agg_social = agg_num.join(mode_sentiment.rename('dominant_sentiment')).join(mode_platform.rename('dominant_platform')).join(counts).reset_index()

        merged_agg = trans.merge(agg_social, on='customer_id_new', how='left')
        # persist aggregated merged for reproducibility
        os.makedirs('merge-output', exist_ok=True)
        merged_agg.to_csv(agg_path, index=False)
        return merged_agg

    # Fallback: read the older merged CSV if available
    if os.path.exists(fallback_path):
        return pd.read_csv(fallback_path)

    raise FileNotFoundError('No merged dataset found. Run merge scripts to create merge-output/merged_data_aggregated.csv or provide source Customer_Dataset files.')


# Load merged aggregated dataframe
merged_df = load_merged_aggregated()
# Drop transaction_id and purchase_date for this model, as they are less likely to be direct predictors of future *category*
model_df = merged_df.drop(columns=[c for c in ['transaction_id', 'purchase_date', 'customer_id_new'] if c in merged_df.columns])

# Define features (X) and target (y)
categorical_features = ['social_media_platform', 'review_sentiment']
numerical_features = ['purchase_amount', 'customer_rating', 'engagement_score', 'purchase_interest_score']

# Ensure features exist in dataframe; otherwise adjust/raise informative error
available_num = [c for c in numerical_features if c in model_df.columns]
available_cat = [c for c in categorical_features if c in model_df.columns]
features_for_model = available_num + available_cat

if len(features_for_model) == 0:
    raise ValueError(f"No model features found in dataframe. Expected any of: {numerical_features + categorical_features}")

X = model_df[features_for_model].copy()
y = model_df['product_category'].copy()

# Encode target labels to integers and keep encoder for inverse mapping
le = LabelEncoder()
# Drop rows with missing target
notnull_mask = y.notnull()
X = X.loc[notnull_mask]
y = y.loc[notnull_mask]
y_encoded = le.fit_transform(y)


# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Use the available feature lists (detected above) for the ColumnTransformer so
# the pipeline is robust to different merged column names (e.g., avg_engagement_score)
num_cols_for_transform = available_num
cat_cols_for_transform = available_cat

if len(num_cols_for_transform) == 0 and len(cat_cols_for_transform) == 0:
    raise ValueError(f"No features available for preprocessing. Expected any of: {numerical_features + categorical_features}")

# Combine transformers into a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols_for_transform),
        ('cat', categorical_transformer, cat_cols_for_transform)
    ], remainder='drop')

# Create the full pipeline with a classifier, RandomForestClassifier
product_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', RandomForestClassifier(random_state=42))])

# Split data (use encoded labels for training with XGBoost if needed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Train RandomForest as a baseline
product_model_pipeline.fit(X_train, y_train)
y_pred = product_model_pipeline.predict(X_test)

print("\nRandomForest Product Recommendation Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save trained baseline model and label encoder
os.makedirs('trained-models/models', exist_ok=True)
joblib.dump(product_model_pipeline, 'trained-models/models/product_model_randomforest.joblib')
joblib.dump(le, 'trained-models/models/label_encoder.joblib')
print('Saved RandomForest model and label encoder to trained-models/models/')

# Example with XGBoost (uses same preprocessing). Use encoded integer labels.
try:
    product_model_pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                                               ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))])
    product_model_pipeline_xgb.fit(X_train, y_train)
    y_pred_xgb = product_model_pipeline_xgb.predict(X_test)
    print("\nXGBoost Product Recommendation Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"F1-Score (weighted): {f1_score(y_test, y_pred_xgb, average='weighted'):.4f}")
    print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))
    joblib.dump(product_model_pipeline_xgb, 'trained-models/models/product_model_xgb.joblib')
    print('Saved XGBoost model to trained-models/models/')
except Exception as e:
    print('XGBoost training skipped or failed:', str(e))