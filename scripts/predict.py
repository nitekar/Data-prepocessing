import os
import joblib
import pandas as pd

MODEL_DIR = os.path.join('trained-models', 'models')
XGB_PATH = os.path.join(MODEL_DIR, 'product_model_xgb.joblib')
RF_PATH = os.path.join(MODEL_DIR, 'product_model_randomforest.joblib')
LE_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
MERGED_CSV = os.path.join('merge-output', 'merged_data.csv')

# Choose model: prefer XGBoost if available
if os.path.exists(XGB_PATH):
    model_path = XGB_PATH
elif os.path.exists(RF_PATH):
    model_path = RF_PATH
else:
    raise FileNotFoundError('No trained model found in trained-models/models/. Run product_recommendation first.')

if not os.path.exists(LE_PATH):
    raise FileNotFoundError('Label encoder not found. Run product_recommendation to generate label encoder.')

model = joblib.load(model_path)
le = joblib.load(LE_PATH)
print(f'Loaded model from: {model_path}')

# Load merged data to build a realistic sample (use first non-null row)
if not os.path.exists(MERGED_CSV):
    raise FileNotFoundError('merge-output/merged_data.csv not found. Run merge_data_sets.py first.')

df = pd.read_csv(MERGED_CSV)

# Candidate feature names used by the training pipeline
candidate_features = ['purchase_amount', 'customer_rating', 'engagement_score', 'purchase_interest_score',
                      'social_media_platform', 'review_sentiment']

features = [c for c in candidate_features if c in df.columns]
if len(features) == 0:
    raise ValueError('No candidate features found in merged_data.csv')

# Take the first row that has any non-null values for these features
sample_row = df[features].dropna(how='all').iloc[[0]].copy()

# Fill numeric NaNs with mean and categorical with mode to match training preprocessing
for col in sample_row.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        mean_val = df[col].mean()
        sample_row[col] = sample_row[col].fillna(mean_val)
    else:
        mode_val = df[col].mode(dropna=True)
        sample_row[col] = sample_row[col].fillna(mode_val[0] if not mode_val.empty else 'missing')

print('Using sample input:')
print(sample_row.to_dict(orient='records')[0])

# Predict
pred_encoded = model.predict(sample_row)

# If model produced a single encoded prediction, ensure it's iterable
try:
    pred_labels = le.inverse_transform(pred_encoded)
except Exception:
    # maybe model returned string classes already
    pred_labels = pred_encoded

print('\nPredicted class (human-readable):')
print(pred_labels)
