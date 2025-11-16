# live_voice_verification.py - Real-time voice verification
import torch
import joblib
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import sys
from pathlib import Path

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

from speechbrain.inference import EncoderClassifier

class VoiceVerifier:
    def __init__(self, model_path="speechbrain_classifier.pkl", 
                 encoder_path="speechbrain_label_encoder.pkl",
                 pretrained_dir="pretrained_models/"):
        """Initialize the voice verification system."""
        print("Loading models...")
        
        # Load SpeechBrain ECAPA-TDNN model
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=pretrained_dir
        )
        
        # Load trained classifier and label encoder
        self.model = joblib.load(model_path)
        self.le = joblib.load(encoder_path)
        
        print(f"✓ Models loaded. Known speakers: {list(self.le.classes_)}")
        
    def extract_embedding(self, audio_path):
        """Extract ECAPA-TDNN embedding from audio file."""
        y, sr = librosa.load(audio_path, sr=16000)
        waveform = torch.tensor(y).unsqueeze(0)
        
        with torch.no_grad():
            emb = self.classifier.encode_batch(waveform)
        
        return emb.squeeze().cpu().numpy()
    
    def predict_speaker(self, audio_path):
        """Predict speaker identity with confidence scores."""
        emb = self.extract_embedding(audio_path)
        
        # Get prediction
        pred_idx = self.model.predict([emb])[0]
        pred_name = self.le.inverse_transform([pred_idx])[0]
        
        # Get confidence scores (probabilities)
        proba = self.model.predict_proba([emb])[0]
        confidence = proba[pred_idx] * 100
        
        # Get all speaker scores
        all_scores = {name: prob * 100 for name, prob in zip(self.le.classes_, proba)}
        
        return pred_name, confidence, all_scores
    
    def record_audio(self, duration=3, sample_rate=16000):
        """Record audio from microphone."""
        print(f"\n🎤 Recording for {duration} seconds...")
        print("Speak now!")
        
        # Record audio
        recording = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        
        print("✓ Recording complete!")
        return recording, sample_rate
    
    def save_recording(self, recording, sample_rate, filename="temp_recording.wav"):
        """Save recorded audio to file."""
        sf.write(filename, recording, sample_rate)
        return filename
    
    def verify_live(self, expected_speaker=None, duration=3, threshold=70.0):
        """
        Record audio and verify speaker identity.
        
        Args:
            expected_speaker: Name of expected speaker (None for identification only)
            duration: Recording duration in seconds
            threshold: Confidence threshold for verification (0-100)
        
        Returns:
            dict: Verification results
        """
        # Record audio
        recording, sample_rate = self.record_audio(duration)
        
        # Save to temporary file
        temp_file = "temp_recording.wav"
        self.save_recording(recording, sample_rate, temp_file)
        
        # Predict speaker
        pred_name, confidence, all_scores = self.predict_speaker(temp_file)
        
        # Verification result
        result = {
            'predicted_speaker': pred_name,
            'confidence': confidence,
            'all_scores': all_scores,
            'recording_file': temp_file
        }
        
        if expected_speaker:
            verified = (pred_name.lower() == expected_speaker.lower() and 
                       confidence >= threshold)
            result['expected_speaker'] = expected_speaker
            result['verified'] = verified
            result['threshold'] = threshold
        
        return result


def print_results(result):
    """Pretty print verification results."""
    print("\n" + "="*50)
    print("VOICE VERIFICATION RESULTS")
    print("="*50)
    
    print(f"\n🎯 Predicted Speaker: {result['predicted_speaker']}")
    print(f"📊 Confidence: {result['confidence']:.1f}%")
    
    if 'expected_speaker' in result:
        print(f"\n👤 Expected Speaker: {result['expected_speaker']}")
        print(f"✓ Verified: {'YES ✅' if result['verified'] else 'NO ❌'}")
        print(f"🎚️  Threshold: {result['threshold']:.1f}%")
    
    print("\n📈 All Speaker Scores:")
    for speaker, score in sorted(result['all_scores'].items(), 
                                 key=lambda x: x[1], 
                                 reverse=True):
        bar = "█" * int(score / 5)  # Visual bar
        print(f"  {speaker:15s}: {score:5.1f}% {bar}")
    
    print(f"\n💾 Recording saved: {result['recording_file']}")
    print("="*50)


def main():
    """Main interactive verification system."""
    print("\n🔊 Voice Verification System")
    print("="*50)
    
    # Initialize verifier
    verifier = VoiceVerifier()
    
    while True:
        print("\n\nOptions:")
        print("1. Identify speaker (no verification)")
        print("2. Verify specific speaker")
        print("3. Quit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            # Speaker identification
            duration = input("Recording duration (seconds, default=3): ").strip()
            duration = int(duration) if duration else 3
            
            result = verifier.verify_live(duration=duration)
            print_results(result)
            
        elif choice == "2":
            # Speaker verification
            print(f"\nAvailable speakers: {list(verifier.le.classes_)}")
            expected = input("Enter expected speaker name: ").strip()
            
            if not expected:
                print("❌ Speaker name required!")
                continue
            
            duration = input("Recording duration (seconds, default=3): ").strip()
            duration = int(duration) if duration else 3
            
            threshold = input("Confidence threshold (%, default=70): ").strip()
            threshold = float(threshold) if threshold else 70.0
            
            result = verifier.verify_live(
                expected_speaker=expected,
                duration=duration,
                threshold=threshold
            )
            print_results(result)
            
        elif choice == "3":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()