# verify_live.py
import librosa
import pyaudio
import wave
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from audio_processing import AudioProcessor

# === Load model ===
model = joblib.load("voice_verification_model.pkl")
le = joblib.load("label_encoder.pkl")
feature_cols = joblib.load("feature_columns.pkl")
processor = AudioProcessor()

# === Fix: Use updated tempo function (NO MORE WARNING) ===
#def extract_tempo_fixed(y, sr):
    #onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    #tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
    #return tempo[0]

# Monkey-patch the method
#processor.extract_tempo = extract_tempo_fixed

# === Verification function ===
def verify_audio_file(wav_path):
    feature_dict = processor.extract_features_from_file(str(wav_path))
    df = pd.DataFrame([feature_dict])
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    X = df[feature_cols].values  # ← Fix: Use .values to avoid feature name warning

    proba = model.predict_proba(X)[0]
    idx = proba.argmax()
    speaker = le.classes_[idx]
    confidence = proba[idx]
    return speaker, confidence

# === Audio settings ===
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
MAX_RECORD_SECONDS = 6.0       # Max allowed
SILENCE_THRESHOLD = 500        # Amplitude to detect speech
MIN_SPEECH_SECONDS = 1.5       # Min speech duration
SILENCE_TO_STOP = 1.0          # Stop after 1 sec silence

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

print("Listening... Say your phrase (up to 6 sec). Silence to stop.\n")

frames = []
speech_detected = False
silence_chunks = 0
max_chunks = int(RATE / CHUNK * MAX_RECORD_SECONDS)
speech_start_chunk = None

for i in range(max_chunks):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)
    chunk_np = np.frombuffer(data, dtype=np.int16)
    amplitude = np.abs(chunk_np).mean()

    if amplitude > SILENCE_THRESHOLD:
        if not speech_detected:
            print("Speech detected...", end="", flush=True)
            speech_detected = True
            speech_start_chunk = i
        silence_chunks = 0
        print(".", end="", flush=True)
    else:
        if speech_detected:
            silence_chunks += 1
            # Stop after 1 sec of silence
            if silence_chunks > int(RATE / CHUNK * SILENCE_TO_STOP):
                print("\nSilence detected. Processing...")
                break
        else:
            print(".", end="", flush=True) if i % 10 == 0 else None

# Trim leading silence
if speech_start_chunk is not None:
    start_idx = max(0, speech_start_chunk - 5)  # keep 5 chunks before speech
    frames = frames[start_idx:]

# Save
temp_wav = "temp_live.wav"
wf = wave.open(temp_wav, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# Duration check
duration = len(frames) * CHUNK / RATE
if duration < MIN_SPEECH_SECONDS:
    print(f"Too short ({duration:.1f}s). Try again.")
else:
    speaker, conf = verify_audio_file(temp_wav)
    print(f"\nSPEAKER: {speaker} (Confidence: {conf:.1%})")

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
Path(temp_wav).unlink(missing_ok=True)