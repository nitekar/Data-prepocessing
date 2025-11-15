# --------------------------------------------------------------
# train_face_model.py
# Facial Recognition Model Training Script
# --------------------------------------------------------------
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import numpy as np

# ---------- 1. Load the CSV that image_processing.py created ----------
CSV_PATH = Path("image_features.csv")
if not CSV_PATH.exists():
    raise FileNotFoundError("Run image_processing.py first → image_features.csv missing")

df = pd.read_csv(CSV_PATH)

print("="*60)
print("FACIAL RECOGNITION MODEL TRAINING")
print("="*60)
print(f"Loaded {len(df)} rows – {df['member_name'].nunique()} members")
print(f"\nMember distribution:")
print(df['member_name'].value_counts())
print(f"\nExpression distribution:")
print(df['expression'].value_counts())

# ---------- 2. Encode member names ----------
le = LabelEncoder()
df["label"] = le.fit_transform(df["member_name"])

# ---------- 3. Select numeric feature columns ----------
exclude = {"member_name", "expression", "augmentation", "image_path", "label"}
feature_cols = [c for c in df.columns if c not in exclude]

print(f"\nNumber of feature columns: {len(feature_cols)}")
print(f"First 10 features: {feature_cols[:10]}")

X = df[feature_cols].values.astype(np.float32)
y = df["label"].values

# Handle any NaN values
if np.isnan(X).any():
    print("Warning: NaN values found, filling with 0")
    X = np.nan_to_num(X)

# ---------- 4. Train / validation split ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

# ---------- 5. Train Random Forest Classifier ----------
print("\n" + "="*60)
print("Training Random Forest Classifier...")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf_model.fit(X_train, y_train)

# ---------- 6. Evaluate Random Forest ----------
rf_pred = rf_model.predict(X_val)
rf_acc = accuracy_score(y_val, rf_pred)
rf_f1 = f1_score(y_val, rf_pred, average='weighted')

print(f"\nRandom Forest Results:")
print(f"Validation Accuracy: {rf_acc:.1%}")
print(f"Validation F1-Score (weighted): {rf_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, rf_pred, target_names=le.classes_))

# ---------- 7. Train Logistic Regression (alternative model) ----------
print("\n" + "="*60)
print("Training Logistic Regression Classifier...")
print("="*60)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced",
    solver='lbfgs'
)
lr_model.fit(X_train, y_train)

# ---------- 8. Evaluate Logistic Regression ----------
lr_pred = lr_model.predict(X_val)
lr_acc = accuracy_score(y_val, lr_pred)
lr_f1 = f1_score(y_val, lr_pred, average='weighted')

print(f"\nLogistic Regression Results:")
print(f"Validation Accuracy: {lr_acc:.1%}")
print(f"Validation F1-Score (weighted): {lr_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, lr_pred, target_names=le.classes_))

# ---------- 9. Select best model and save ----------
if rf_acc >= lr_acc:
    print("\n" + "="*60)
    print("Selected Random Forest as the best model (higher accuracy)")
    print("="*60)
    best_model = rf_model
    model_name = "RandomForest"
else:
    print("\n" + "="*60)
    print("Selected Logistic Regression as the best model (higher accuracy)")
    print("="*60)
    best_model = lr_model
    model_name = "LogisticRegression"

# Save model, encoder, and feature columns
model_dir = Path("trained-models/models")
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_dir / "face_recognition_model.pkl")
joblib.dump(le, model_dir / "face_label_encoder.pkl")
joblib.dump(feature_cols, model_dir / "face_feature_columns.pkl")

print("\n" + "="*60)
print("Model saved successfully!")
print("="*60)
print(f"   -> {model_dir / 'face_recognition_model.pkl'}")
print(f"   -> {model_dir / 'face_label_encoder.pkl'}")
print(f"   -> {model_dir / 'face_feature_columns.pkl'}")
print(f"\nSelected model: {model_name}")
print(f"Final Accuracy: {max(rf_acc, lr_acc):.1%}")
print(f"Final F1-Score: {max(rf_f1, lr_f1):.4f}")
print("="*60)

