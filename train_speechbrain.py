# train_speechbrain.py - FIXED padding issue
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import librosa
import numpy as np

import torchaudio

if hasattr(torchaudio, 'list_audio_backends'):
    backends = torchaudio.list_audio_backends()
else:
    backends = ['torchcodec']

import sys

sys.modules['torchaudio'].list_audio_backends = lambda: backends

# === FIX: Monkey-patch huggingface_hub ===
import huggingface_hub

original_hf_hub_download = huggingface_hub.hf_hub_download


def patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return original_hf_hub_download(*args, **kwargs)


huggingface_hub.hf_hub_download = patched_hf_hub_download

from speechbrain.inference import EncoderClassifier

# === 1. Load the pre-trained ECAPA-TDNN model ===
print("Loading speechbrain ECAPA-TDNN model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/"
)

print("Model loaded successfully!")

# === 2. Load your audio files ===
CSV_PATH = "audio_features.csv"
df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df)} samples")


# === 3. Extract embeddings (FIXED) ===
def get_embedding(audio_path):
    """Extract ECAPA-TDNN embedding from audio file."""
    # Load audio at 16kHz (required by ECAPA-TDNN)
    y, sr = librosa.load(audio_path, sr=16000)

    # Convert to torch tensor with shape [batch=1, time]
    # CRITICAL: Use 2D tensor, not 4D!
    waveform = torch.tensor(y).unsqueeze(0)  # Shape: [1, T]

    # Extract embedding
    with torch.no_grad():
        emb = classifier.encode_batch(waveform)

    return emb.squeeze().cpu().numpy()


embeddings = []
labels = []

print("Extracting embeddings...")
for idx, row in df.iterrows():
    if idx % 10 == 0:
        print(f"  {idx}/{len(df)}")
    try:
        emb = get_embedding(row['audio_path'])
        embeddings.append(emb)
        labels.append(row['member_name'])
    except Exception as e:
        print(f"  Skipping {row['audio_path']}: {e}")

print(f"\nSuccessfully extracted {len(embeddings)} embeddings from {len(df)} files")

if len(embeddings) == 0:
    print("ERROR: No embeddings were extracted! Check your audio files.")
    sys.exit(1)

X = np.array(embeddings)  # (N, 192)
y = np.array(labels)

# === 4. Train ===
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.3, stratify=y_enc, random_state=42
)

print(f"\nTraining with {len(X_train)} samples, validating with {len(X_val)} samples")

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# === 5. Evaluate ===
pred = model.predict(X_val)
acc = accuracy_score(y_val, pred)
print(f"\nValidation Accuracy: {acc:.1%}")
print(classification_report(y_val, pred, target_names=le.classes_))

# === 6. Save ===
joblib.dump(model, "speechbrain_classifier.pkl")
joblib.dump(le, "speechbrain_label_encoder.pkl")
print("\nSaved: speechbrain_classifier.pkl, speechbrain_label_encoder.pkl")