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

### Issue: Low accuracy on voice verification

**Possible causes:**
1. **Audio quality**: Ensure 16kHz sample rate, minimal background noise
2. **Recording duration**: Use at least 3-5 seconds
3. **Training data**: Need multiple samples (6+ per person)
4. **Threshold too high**: Lower from 70% to 60%

---
