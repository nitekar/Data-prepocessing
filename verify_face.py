# --------------------------------------------------------------
# verify_face.py
# Facial Recognition Verification Script
# Used for system simulation and testing
# --------------------------------------------------------------
import cv2
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from image_processing import ImageProcessor

# ---------- Load model and dependencies ----------
model_dir = Path("trained-models/models")
model_path = model_dir / "face_recognition_model.pkl"
encoder_path = model_dir / "face_label_encoder.pkl"
feature_cols_path = model_dir / "face_feature_columns.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"Face recognition model not found: {model_path}\n"
                           "Please run train_face_model.py first")

model = joblib.load(model_path)
le = joblib.load(encoder_path)
feature_cols = joblib.load(feature_cols_path)

processor = ImageProcessor(base_dir='Images')

print("="*60)
print("FACE RECOGNITION VERIFICATION")
print("="*60)
print(f"Model loaded successfully")
print(f"Recognized members: {list(le.classes_)}")
print("="*60)

# ---------- Verification function ----------
def verify_face_image(image_path, threshold=0.6):
    """
    Verify if a face image matches any of the registered members.
    
    Parameters:
    -----------
    image_path : str or Path
        Path to the image file to verify
    threshold : float
        Minimum probability threshold for acceptance (default: 0.6)
    
    Returns:
    --------
    recognized_member : str or None
        Name of recognized member, or None if not recognized
    confidence : float
        Confidence score (probability) of the recognition
    is_authorized : bool
        True if recognized with confidence above threshold
    """
    # Load and process image
    try:
        img = processor.load_image(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, 0.0, False
    
    # Extract features (same as training)
    hist_features = processor.extract_histogram_features(img)
    hog_features = processor.extract_hog_features(img)
    lbp_features = processor.extract_lbp_features(img)
    color_moments = processor.extract_color_moments(img)
    
    # Create feature dictionary
    feature_dict = {
        'mean_intensity': np.mean(img),
        'std_intensity': np.std(img)
    }
    
    # Add histogram features (reduced for CSV - same as training)
    hist_reduced = hist_features[::10]
    for i, val in enumerate(hist_reduced):
        feature_dict[f'hist_{i}'] = val
    
    # Add HOG features (reduced)
    hog_reduced = hog_features[::50]
    for i, val in enumerate(hog_reduced):
        feature_dict[f'hog_{i}'] = val
    
    # Add LBP features (reduced)
    lbp_reduced = lbp_features[::10]
    for i, val in enumerate(lbp_reduced):
        feature_dict[f'lbp_{i}'] = val
    
    # Add color moments
    for i, val in enumerate(color_moments):
        feature_dict[f'color_moment_{i}'] = val
    
    # Convert to DataFrame
    df = pd.DataFrame([feature_dict])
    
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    X = df[feature_cols].values.astype(np.float32)
    
    # Handle NaN values
    if np.isnan(X).any():
        X = np.nan_to_num(X)
    
    # Predict
    proba = model.predict_proba(X)[0]
    idx = proba.argmax()
    recognized_member = le.classes_[idx]
    confidence = proba[idx]
    
    # Check if above threshold
    is_authorized = confidence >= threshold
    
    return recognized_member, confidence, is_authorized


# ---------- Main execution (if run as script) ----------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python verify_face.py <image_path> [threshold]")
        print("\nExample:")
        print("  python verify_face.py Images/Ganza_neutral.jpg")
        print("  python verify_face.py Images/oreste_smile.jpg 0.7")
        sys.exit(1)
    
    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    
    print(f"\nVerifying image: {image_path}")
    print(f"Threshold: {threshold:.2f}\n")
    
    member, confidence, authorized = verify_face_image(image_path, threshold)
    
    print("="*60)
    if authorized:
        print(f"✓ ACCESS GRANTED")
        print(f"Recognized as: {member}")
        print(f"Confidence: {confidence:.1%}")
    else:
        if member:
            print(f"✗ ACCESS DENIED")
            print(f"Best match: {member} (confidence: {confidence:.1%})")
            print(f"Confidence below threshold ({threshold:.1%})")
        else:
            print(f"✗ ACCESS DENIED")
            print(f"Face not recognized")
    print("="*60)

