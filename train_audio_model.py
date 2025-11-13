# --------------------------------------------------------------
# train_and_save.py
# --------------------------------------------------------------
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# ---------- 1. Load the CSV that audio_processing.py created ----------
CSV_PATH = Path("audio_features.csv")
if not CSV_PATH.exists():
    raise FileNotFoundError("Run audio_processing.py first → audio_features.csv missing")

df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df)} rows – {df['member_name'].nunique()} speakers")
print(df['member_name'].value_counts())

# ---------- 2. Encode speaker names ----------
le = LabelEncoder()
df["label"] = le.fit_transform(df["member_name"])

# ---------- 3. Select numeric feature columns ----------
exclude = {"member_name", "phrase", "augmentation", "audio_path", "label"}
feature_cols = [c for c in df.columns if c not in exclude]

X = df[feature_cols].values.astype(np.float32)
y = df["label"].values

# ---------- 4. Train / validation split ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------- 5. Train a strong classifier ----------
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ---------- 6. Evaluate ----------
pred = model.predict(X_val)
acc = accuracy_score(y_val, pred)
print(f"\nValidation accuracy: {acc:.1%}")
print(classification_report(y_val, pred, target_names=le.classes_))

# ---------- 7. Save everything ----------
joblib.dump(model, "voice_verification_model.pkl")
joblib.dump(le,   "label_encoder.pkl")
joblib.dump(feature_cols, "feature_columns.pkl")   # needed for inference

print("\nModel, encoder and column list saved!")
print("   -> voice_verification_model.pkl")
print("   -> label_encoder.pkl")
print("   -> feature_columns.pkl")