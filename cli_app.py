"""
Enhanced System Demonstration – Facial + Voice + Product Recommendation
------------------------------------------------------------------------

Features:
*   SpeechBrain ECAPA-TDNN voice recognition (state-of-the-art)
*   Live recording or file selection
*   Improved UI with progress indicators
*   Detailed confidence scores
*   Better error handling and user feedback
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional, Tuple, Dict
import torch
import librosa

# Monkey-patch for torchaudio
import torchaudio

if hasattr(torchaudio, 'list_audio_backends'):
    backends = torchaudio.list_audio_backends()
else:
    backends = ['torchcodec']
sys.modules['torchaudio'].list_audio_backends = lambda: backends

# Monkey-patch huggingface_hub
import huggingface_hub

original_hf_hub_download = huggingface_hub.hf_hub_download


def patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return original_hf_hub_download(*args, **kwargs)


huggingface_hub.hf_hub_download = patched_hf_hub_download


# ----------------------------------------------------------------------
# 1. FEATURE EXTRACTION HELPERS
# ----------------------------------------------------------------------
def extract_face_features(image_path: Path) -> np.ndarray:
    """Extract HOG features from face image."""
    import cv2
    from skimage import feature
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    img = cv2.resize(img, (128, 128))
    hog = feature.hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                      block_norm='L2-Hys')
    return hog.astype(np.float32)


def extract_audio_features_ecapa(audio_path: Path, classifier) -> np.ndarray:
    """Extract ECAPA-TDNN embeddings."""
    y, sr = librosa.load(str(audio_path), sr=16000)
    waveform = torch.tensor(y).unsqueeze(0)

    with torch.no_grad():
        emb = classifier.encode_batch(waveform)

    return emb.squeeze().cpu().numpy()


# ----------------------------------------------------------------------
# 2. MODEL LOADER CLASS
# ----------------------------------------------------------------------
class ModelLoader:
    """Centralized model loading with error handling."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.models = {}

    def load_all(self):
        """Load all models and encoders."""
        print("Loading models...")

        try:
            # Face recognition
            print("  ✓ Face recognition model")
            self.models['face_model'] = joblib.load(self.model_dir / "face_recognition_model.pkl")
            self.models['face_le'] = joblib.load(self.model_dir / "face_label_encoder.pkl")
            self.models['face_cols'] = joblib.load(self.model_dir / "face_feature_columns.pkl")

            # Voice recognition (SpeechBrain ECAPA-TDNN)
            print("  ✓ Voice recognition model (SpeechBrain ECAPA-TDNN)")
            from speechbrain.inference import EncoderClassifier
            self.models['ecapa_classifier'] = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/"
            )
            self.models['voice_model'] = joblib.load(self.model_dir / "speechbrain_classifier.pkl")
            self.models['voice_le'] = joblib.load(self.model_dir / "speechbrain_label_encoder.pkl")

            # Product recommendation
            print("  ✓ Product recommendation model")
            self.models['product_pipe'] = joblib.load(self.model_dir / "product_model_randomforest.joblib")
            self.models['product_le'] = joblib.load(self.model_dir / "label_encoder.joblib")

            print("\n✅ All models loaded successfully!\n")

        except FileNotFoundError as e:
            messagebox.showerror("Model Loading Error",
                                 f"Could not find model file:\n{e}\n\n"
                                 "Please ensure all models are in the correct directory.")
            sys.exit(1)


# ----------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ----------------------------------------------------------------------
def pad_to_cols(vec: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate vector to target length."""
    padded = np.zeros(target_len, dtype=np.float32)
    padded[:len(vec)] = vec[:target_len]
    return padded


def record_voice(duration_sec: int = 5, fs: int = 16000) -> Optional[Path]:
    """Record voice from microphone."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        messagebox.showerror("Missing dependency",
                             "sounddevice not installed.\n"
                             "Run: pip install sounddevice soundfile")
        return None

    print(f"\n🎤 Recording {duration_sec} seconds... Speak now!")
    rec = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    out_path = Path("live_voice_recording.wav")
    sf.write(out_path, rec, fs)
    print(f"✓ Recording saved → {out_path}\n")
    return out_path


# ----------------------------------------------------------------------
# 4. RECOGNITION PIPELINE
# ----------------------------------------------------------------------
class RecognitionPipeline:
    """Main recognition pipeline with all models."""

    def __init__(self, models: Dict):
        self.models = models

    def recognize_face(self, image_path: Path) -> Tuple[str, float]:
        """Recognize face and return name + confidence."""
        print("\n" + "=" * 60)
        print("STEP 1: FACIAL RECOGNITION")
        print("=" * 60)

        face_vec = extract_face_features(image_path)
        face_vec = pad_to_cols(face_vec, len(self.models['face_cols'])).reshape(1, -1)

        face_pred_id = self.models['face_model'].predict(face_vec)[0]
        face_name = self.models['face_le'].inverse_transform([face_pred_id])[0]
        face_conf = self.models['face_model'].predict_proba(face_vec).max()

        print(f"✓ Recognized Member: {face_name}")
        print(f"✓ Confidence: {face_conf:.1%}")

        return face_name, face_conf

    def recommend_product(self, member_name: str) -> str:
        """Recommend product based on member profile."""
        print("\n" + "=" * 60)
        print("STEP 2: PRODUCT RECOMMENDATION")
        print("=" * 60)

        # TODO: Replace with actual member profile lookup
        dummy_row = {
            "social_media_platform": "Instagram",
            "review_sentiment": "Positive",
            "purchase_amount": 75.0,
            "customer_rating": 4.5,
            "engagement_score": 0.82,
            "purchase_interest_score": 0.91
        }

        input_df = pd.DataFrame([dummy_row])
        prod_enc = self.models['product_pipe'].predict(input_df)[0]
        product_category = self.models['product_le'].inverse_transform([prod_enc])[0]

        print(f"✓ Recommended Category: {product_category}")
        print(f"  (Based on member profile for: {member_name})")

        return product_category

    def verify_voice(self, voice_path: Path, expected_name: str) -> Dict:
        """
        Verify voice using SpeechBrain ECAPA-TDNN model.

        Returns:
            dict with keys: 'match', 'confidence', 'predicted_name', 'all_scores'
        """
        print("\n" + "=" * 60)
        print("STEP 3: VOICE VERIFICATION (ECAPA-TDNN)")
        print("=" * 60)

        # Extract ECAPA-TDNN embeddings
        print("\n🎵 Extracting voice features...")
        voice_vec = extract_audio_features_ecapa(voice_path, self.models['ecapa_classifier'])

        # Predict speaker
        pred_id = self.models['voice_model'].predict([voice_vec])[0]
        pred_name = self.models['voice_le'].inverse_transform([pred_id])[0]

        # Get confidence scores
        proba = self.models['voice_model'].predict_proba([voice_vec])
        confidence = proba[0][pred_id]

        # Get all speaker scores
        all_scores = {name: prob for name, prob in
                      zip(self.models['voice_le'].classes_, proba[0])}

        # Check if matches expected speaker
        match = (pred_name.lower() == expected_name.lower())

        print(f"\n✓ Predicted Speaker: {pred_name}")
        print(f"✓ Confidence: {confidence:.1%}")
        print(f"✓ Expected Speaker: {expected_name}")
        print(f"✓ Match: {'✅ YES' if match else '❌ NO'}")

        print("\n📊 All Speaker Confidence Scores:")
        for speaker, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 30)
            indicator = " ← MATCH" if speaker.lower() == expected_name.lower() else ""
            print(f"  {speaker:15s}: {score:6.1%} {bar}{indicator}")

        # Verification threshold
        THRESHOLD = 0.70  # 70% confidence required

        verified = match and confidence >= THRESHOLD

        if verified:
            print(f"\n✅ VOICE VERIFIED (Confidence: {confidence:.1%} ≥ {THRESHOLD:.0%})")
        else:
            if not match:
                print(f"\n❌ VOICE MISMATCH (Expected: {expected_name}, Got: {pred_name})")
            else:
                print(f"\n❌ LOW CONFIDENCE (Got: {confidence:.1%} < {THRESHOLD:.0%})")

        return {
            'match': match,
            'verified': verified,
            'confidence': confidence,
            'predicted_name': pred_name,
            'expected_name': expected_name,
            'all_scores': all_scores,
            'threshold': THRESHOLD
        }


# ----------------------------------------------------------------------
# 5. MAIN INTERACTIVE FLOW
# ----------------------------------------------------------------------
def main():
    """Main demonstration flow."""
    print("\n" + "=" * 60)
    print("🔐 ENHANCED AUTHENTICATION SYSTEM")
    print("=" * 60)
    print("Features: Facial Recognition + Voice Verification (ECAPA-TDNN)")
    print("         + Product Recommendation")
    print("=" * 60 + "\n")

    # Initialize Tkinter
    root = tk.Tk()
    root.withdraw()

    # Load all models
    MODEL_DIR = Path(__file__).parent / "trained-models" / "models"
    loader = ModelLoader(MODEL_DIR)
    loader.load_all()

    # Initialize pipeline
    pipeline = RecognitionPipeline(loader.models)

    # ----- Get face image -----
    face_path = filedialog.askopenfilename(
        title="Select a face image (jpg/png)",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not face_path:
        messagebox.showinfo("Cancelled", "No face image selected – exiting.")
        return
    face_path = Path(face_path)

    # ----- Get voice clip -----
    choice = messagebox.askyesnocancel(
        "Voice Input",
        "How would you like to provide voice input?\n\n"
        "  • YES  – Record voice now (5 seconds)\n"
        "  • NO   – Select existing audio file\n"
        "  • CANCEL – Exit demo"
    )

    if choice is None:  # Cancel
        messagebox.showinfo("Cancelled", "Demo aborted.")
        return

    if choice:  # YES - record
        voice_path = record_voice(duration_sec=5)
        if not voice_path:
            return
    else:  # NO - browse
        voice_path = filedialog.askopenfilename(
            title="Select a voice clip (wav/mp3)",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg")]
        )
        if not voice_path:
            messagebox.showinfo("Cancelled", "No voice file selected – exiting.")
            return
        voice_path = Path(voice_path)

    # ========== RUN PIPELINE ==========
    try:
        # Step 1: Face recognition
        face_name, face_conf = pipeline.recognize_face(face_path)

        if face_conf < 0.50:
            print("\n⚠️  WARNING: Low face recognition confidence!")
            cont = messagebox.askyesno("Low Confidence",
                                       f"Face confidence is only {face_conf:.1%}\n"
                                       "Continue anyway?")
            if not cont:
                return

        # Step 2: Product recommendation
        product_category = pipeline.recommend_product(face_name)

        # Step 3: Voice verification
        voice_result = pipeline.verify_voice(voice_path, face_name)

        # ========== FINAL DECISION ==========
        print("\n" + "=" * 60)
        print("FINAL AUTHENTICATION RESULT")
        print("=" * 60)

        if voice_result['verified']:
            result_msg = (
                f"✅ AUTHENTICATION SUCCESSFUL\n\n"
                f"👤 Member: {face_name}\n"
                f"📷 Face Confidence: {face_conf:.1%}\n"
                f"🎤 Voice Verification: PASSED\n"
                f"   • Predicted: {voice_result['predicted_name']}\n"
                f"   • Confidence: {voice_result['confidence']:.1%}\n"
                f"   • Threshold: {voice_result['threshold']:.0%}\n\n"
                f"🛍️  Recommended Product: {product_category}\n\n"
                f"Access Granted!"
            )
            print(result_msg)
            messagebox.showinfo("✅ Access Granted", result_msg)
        else:
            result_msg = (
                f"❌ AUTHENTICATION FAILED\n\n"
                f"👤 Face Recognition: {face_name} ({face_conf:.1%})\n"
                f"🎤 Voice Verification: FAILED\n"
                f"   • Expected: {voice_result['expected_name']}\n"
                f"   • Predicted: {voice_result['predicted_name']}\n"
                f"   • Confidence: {voice_result['confidence']:.1%}\n"
                f"   • Threshold: {voice_result['threshold']:.0%}\n\n"
                f"❌ Access Denied"
            )
            print(result_msg)
            messagebox.showerror("❌ Access Denied", result_msg)

        print("=" * 60 + "\n")

    except Exception as e:
        error_msg = f"Error during processing:\n{str(e)}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", error_msg)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()