# 🔐 Multi-Modal Authentication System

A comprehensive authentication system combining **Facial Recognition**, **Voice Verification**, and **Product Recommendation** using state-of-the-art machine learning models.

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Training Models](#training-models)
- [Running the Demo](#running-the-demo)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **🎭 Facial Recognition**: HOG (Histogram of Oriented Gradients) features with machine learning classification
- **🎤 Voice Verification**: SpeechBrain ECAPA-TDNN (state-of-the-art speaker recognition)
- **🛍️ Product Recommendation**: Personalized product suggestions based on user profiles
- **🎙️ Live Voice Recording**: Record voice samples directly through the application
- **📊 Confidence Scores**: Detailed probability scores for all predictions
- **🖥️ Interactive GUI**: User-friendly file selection and result display

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  Face Image     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Face Recognition       │
│  (HOG + Classifier)     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Product Recommendation │
│  (Random Forest)        │
└─────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Voice Sample   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Voice Verification     │
│  (ECAPA-TDNN)          │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  ✅ Access Granted      │
│  ❌ Access Denied       │
└─────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Microphone (optional, for live recording)

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd Data-preprocessing
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- numpy, pandas, scikit-learn
- opencv-python, scikit-image
- librosa, sounddevice, soundfile
- torch, torchaudio
- speechbrain, huggingface-hub

---

## 🎓 Training Models

### 1. Prepare Your Data

Create the following directory structure:

```
Data-preprocessing/
├── Audios/
│   ├── person1_phrase1.wav
│   ├── person1_phrase2.wav
│   ├── person2_phrase1.wav
│   └── ...
├── Images/
│   ├── person1_photo1.jpg
│   ├── person1_photo2.jpg
│   ├── person2_photo1.jpg
│   └── ...
└── trained-models/
    └── models/
```

### 2. Train Face Recognition Model

```bash
python train_face_model.py
```

**Expected output:**
```
Processing 48 images...
Training face recognition model...
Validation Accuracy: 95.5%
✓ Saved: face_recognition_model.pkl
```

### 3. Train Voice Recognition Model

```bash
python train_speechbrain.py
```

**Expected output:**
```
Loading speechbrain ECAPA-TDNN model...
Extracting embeddings from 48 samples...
Training classifier...
Validation Accuracy: 100.0%
✓ Saved: speechbrain_classifier.pkl
```

### 4. Train Product Recommendation Model

```bash
python train_product_model.py
```

**Expected output:**
```
Training product recommendation model...
Model Accuracy: 92.3%
✓ Saved: product_model_randomforest.joblib
```

---

## 🎮 Running the Demo

### Start the Authentication System

```bash
python enhanced_system_demo.py
```

### Workflow

1. **Select Face Image**
   - Browse and select a face image (jpg/png)
   - System will recognize the person

2. **Provide Voice Sample**
   - Option A: Record 5 seconds of audio
   - Option B: Select existing audio file (.wav, .mp3)

3. **View Results**
   - Face recognition confidence
   - Voice verification status
   - Product recommendation
   - Final authentication decision

---

## 📁 Project Structure

```
Data-preprocessing/
├── README.md
├── requirements.txt
│
├── Audios/                          # Training audio files
│   ├── person1_phrase1.wav
│   └── ...
│
├── Images/                          # Training face images
│   ├── person1_photo1.jpg
│   └── ...
│
├── trained-models/                  # Face & Product models
│   └── models/
│       ├── face_recognition_model.pkl
│       ├── face_label_encoder.pkl
│       ├── face_feature_columns.pkl
│       ├── product_model_randomforest.joblib
│       └── label_encoder.joblib
│
├── speechbrain_classifier.pkl       # Voice model
├── speechbrain_label_encoder.pkl    # Voice encoder
├── pretrained_models/               # SpeechBrain cache
│
├── train_face_model.py              # Train face recognition
├── train_speechbrain.py             # Train voice verification
├── train_product_model.py           # Train product recommendation
├── enhanced_system_demo.py          # Main demo application
├── live_voice_verification.py       # Standalone voice verification
│
└── audio_features.csv               # Generated features
```

---

## 💡 Usage Examples

### Example 1: Complete Authentication

```bash
python enhanced_system_demo.py
```

**Output:**
```
============================================================
STEP 1: FACIAL RECOGNITION
============================================================
✓ Recognized Member: roxane
✓ Confidence: 95.3%

============================================================
STEP 2: PRODUCT RECOMMENDATION
============================================================
✓ Recommended Category: Electronics
  (Based on member profile for: roxane)

============================================================
STEP 3: VOICE VERIFICATION (ECAPA-TDNN)
============================================================
🎵 Extracting voice features...

✓ Predicted Speaker: roxane
✓ Confidence: 97.8%
✓ Expected Speaker: roxane
✓ Match: ✅ YES

📊 All Speaker Confidence Scores:
  roxane         :  97.8% ██████████████████████████████ ← MATCH
  gershom        :   1.5% █
  Oreste         :   0.5% 
  Ganza          :   0.2% 

✅ VOICE VERIFIED (Confidence: 97.8% ≥ 70%)

============================================================
FINAL AUTHENTICATION RESULT
============================================================
✅ AUTHENTICATION SUCCESSFUL

👤 Member: roxane
📷 Face Confidence: 95.3%
🎤 Voice Verification: PASSED
   • Predicted: roxane
   • Confidence: 97.8%
   • Threshold: 70%

🛍️  Recommended Product: Electronics

Access Granted!
```

### Example 2: Standalone Voice Verification

```bash
python live_voice_verification.py
```

Interactive menu for:
- Speaker identification
- Speaker verification
- Live recording

---

## 🎛️ Configuration

### Adjust Confidence Thresholds

Edit `enhanced_system_demo.py`:

```python
# Line ~240
THRESHOLD = 0.70  # Voice verification threshold (70%)

# For higher security
THRESHOLD = 0.85  # 85% confidence required

# For lower security
THRESHOLD = 0.60  # 60% confidence required
```

### Adjust Recording Duration

Edit `enhanced_system_demo.py`:

```python
# Line ~296
voice_path = record_voice(duration_sec=5)  # Change to 3, 7, etc.
```

---

## 🔧 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'skimage'`

**Solution:**
```bash
pip install scikit-image opencv-python
```

### Issue: `TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'`

**Solution:**
```bash
pip install --upgrade huggingface_hub
```

### Issue: Audio recording not working

**Solution:**
```bash
pip install sounddevice soundfile
# Test your microphone
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Issue: `Padding size 2 is not supported for 4D input tensor`

**Solution:** This was fixed in the latest version. Ensure you're using the correct tensor shape:
```python
waveform = torch.tensor(y).unsqueeze(0)  # Shape: [1, T]
```

### Issue: Low accuracy on voice verification

**Possible causes:**
1. **Audio quality**: Ensure 16kHz sample rate, minimal background noise
2. **Recording duration**: Use at least 3-5 seconds
3. **Training data**: Need multiple samples (6+ per person)
4. **Threshold too high**: Lower from 70% to 60%

---

## 📊 Model Performance

### Face Recognition
- **Algorithm**: HOG + Logistic Regression
- **Accuracy**: ~95%
- **Features**: 1764 HOG features per image

### Voice Verification
- **Algorithm**: SpeechBrain ECAPA-TDNN
- **Accuracy**: ~100% (on training data)
- **Features**: 192-dimensional embeddings
- **Pre-trained**: VoxCeleb dataset (7,000+ speakers)

### Product Recommendation
- **Algorithm**: Random Forest
- **Accuracy**: ~92%
- **Features**: User profile + engagement metrics

---

## 🔒 Security Considerations

1. **Threshold Settings**: Adjust based on security requirements
   - High-security: 80-90% threshold
   - Medium-security: 70-80% threshold
   - Low-security: 60-70% threshold

2. **Multi-Modal Verification**: Both face AND voice must match

3. **Logging**: Consider adding authentication attempt logging

4. **Data Privacy**: Store biometric data securely and comply with regulations (GDPR, etc.)

---

## 🛠️ Advanced Usage

### Batch Processing

Process multiple users at once:

```python
from enhanced_system_demo import RecognitionPipeline, ModelLoader

loader = ModelLoader(Path("trained-models/models"))
loader.load_all()
pipeline = RecognitionPipeline(loader.models)

users = [
    {"face": "user1.jpg", "voice": "user1.wav"},
    {"face": "user2.jpg", "voice": "user2.wav"},
]

for user in users:
    face_name, _ = pipeline.recognize_face(Path(user["face"]))
    result = pipeline.verify_voice(Path(user["voice"]), face_name)
    print(f"User: {face_name}, Verified: {result['verified']}")
```

### API Integration

Convert to REST API using Flask:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
pipeline = None  # Initialize in main

@app.route('/verify', methods=['POST'])
def verify():
    face_file = request.files['face']
    voice_file = request.files['voice']
    
    # Process and verify
    # Return JSON response
    
if __name__ == '__main__':
    app.run(debug=True)
```

---

## 📝 License

[Your License Here]

---

## 👥 Contributors

- [Your Name]

---

## 📧 Contact

For questions or issues, please contact: [your-email@example.com]

---

## 🙏 Acknowledgments

- **SpeechBrain**: For the ECAPA-TDNN model
- **scikit-learn**: For classical ML algorithms
- **OpenCV**: For image processing
- **librosa**: For audio feature extraction

---

## 🔄 Version History

### v1.0.0 (Current)
- ✅ Facial recognition with HOG features
- ✅ Voice verification with ECAPA-TDNN
- ✅ Product recommendation system
- ✅ Interactive GUI demo
- ✅ Live voice recording

### Future Enhancements
- [ ] Web-based interface
- [ ] Database integration
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Real-time video verification