import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os

class ImageProcessor:
    """
    Image processing pipeline for facial recognition
    Handles loading, augmentation, and feature extraction
    """
    
    def __init__(self, base_dir='Images'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def load_image(self, image_path):
        """Load image from file"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def display_images(self, images, titles, figsize=(15, 5)):
        """Display multiple images in a row"""
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
        
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images_display.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Display saved as 'sample_images_display.png'")
    
    def augment_rotation(self, image, angle=15):
        """Rotate image by specified angle"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (width, height))
        return rotated
    
    def augment_flip(self, image, flip_code=1):
        """Flip image (0=vertical, 1=horizontal, -1=both)"""
        return cv2.flip(image, flip_code)
    
    def augment_grayscale(self, image):
        """Convert to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    def augment_brightness(self, image, factor=1.5):
        """Adjust brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def augment_noise(self, image, noise_level=25):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    def extract_histogram_features(self, image):
        """Extract color histogram features"""
        if len(image.shape) == 2:
            # Grayscale
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            return hist.flatten()
        else:
            # RGB
            features = []
            for i in range(3):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                features.extend(hist.flatten())
            return np.array(features)
    
    def extract_hog_features(self, image):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize for consistency
        gray = cv2.resize(gray, (128, 128))
        
        # HOG parameters
        win_size = (128, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, 
                                cell_size, nbins)
        features = hog.compute(gray)
        return features.flatten()
    
    def extract_lbp_features(self, image, radius=3, n_points=24):
        """Extract Local Binary Pattern features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (128, 128))
        
        # Simple LBP implementation
        height, width = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = gray[i, j]
                code = 0
                code |= (gray[i-radius, j-radius] >= center) << 7
                code |= (gray[i-radius, j] >= center) << 6
                code |= (gray[i-radius, j+radius] >= center) << 5
                code |= (gray[i, j+radius] >= center) << 4
                code |= (gray[i+radius, j+radius] >= center) << 3
                code |= (gray[i+radius, j] >= center) << 2
                code |= (gray[i+radius, j-radius] >= center) << 1
                code |= (gray[i, j-radius] >= center) << 0
                lbp[i, j] = code
        
        # Get histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        return hist
    
    def extract_color_moments(self, image):
        """Extract color moments (mean, std, skewness)"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        features = []
        for i in range(3):
            channel = image[:, :, i]
            mean = np.mean(channel)
            std = np.std(channel)
            skewness = np.mean(((channel - mean) / (std + 1e-7)) ** 3)
            features.extend([mean, std, skewness])
        
        return np.array(features)
    
    def process_member_images(self, member_name, expressions=['neutral', 'smile', 'surprised']):
        """
        Process all images for a team member
        Returns list of feature dictionaries
        """
        features_list = []
        
        # Map of expression variations
        expression_map = {
            'smile': ['smile', 'smiling'],
            'surprised': ['surprised', 'suprised']
        }
        
        for expression in expressions:
            # Try standard naming first
            img_path = self.base_dir / f"{member_name}_{expression}.jpg"
            
            # Try variations if not found
            if not img_path.exists():
                if expression in expression_map:
                    for variant in expression_map[expression]:
                        # Try underscore
                        img_path = self.base_dir / f"{member_name}_{variant}.jpg"
                        if img_path.exists():
                            break
                        # Try hyphen
                        img_path = self.base_dir / f"{member_name}-{variant}.jpg"
                        if img_path.exists():
                            break
                    else:
                        # Try hyphen with original expression
                        img_path = self.base_dir / f"{member_name}-{expression}.jpg"
            
            if not img_path.exists():
                print(f"Warning: Image for {member_name} {expression} not found (tried variations), skipping...")
                continue
            
            # Load original image
            original = self.load_image(img_path)
            
            # Apply augmentations
            rotated = self.augment_rotation(original, angle=15)
            flipped = self.augment_flip(original)
            grayscale = self.augment_grayscale(original)
            bright = self.augment_brightness(original, factor=1.3)
            noisy = self.augment_noise(original, noise_level=20)
            
            # Save augmented images
            aug_dir = self.base_dir / 'augmented' / member_name
            aug_dir.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(aug_dir / f"{expression}_rotated.jpg"), 
                       cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(aug_dir / f"{expression}_flipped.jpg"), 
                       cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(aug_dir / f"{expression}_grayscale.jpg"), grayscale)
            cv2.imwrite(str(aug_dir / f"{expression}_bright.jpg"), 
                       cv2.cvtColor(bright, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(aug_dir / f"{expression}_noisy.jpg"), 
                       cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))
            
            # Process all versions (original + augmented)
            versions = [
                ('original', original),
                ('rotated', rotated),
                ('flipped', flipped),
                ('grayscale', cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB) if len(grayscale.shape) == 2 else grayscale),
                ('bright', bright),
                ('noisy', noisy)
            ]
            
            for aug_type, img in versions:
                # Extract features
                hist_features = self.extract_histogram_features(img)
                hog_features = self.extract_hog_features(img)
                lbp_features = self.extract_lbp_features(img)
                color_moments = self.extract_color_moments(img)
                
                # Create feature dictionary
                feature_dict = {
                    'member_name': member_name,
                    'expression': expression,
                    'augmentation': aug_type,
                    'image_path': str(img_path)
                }
                
                # Add histogram features (reduced for CSV)
                hist_reduced = hist_features[::10]  # Sample every 10th value
                for i, val in enumerate(hist_reduced):
                    feature_dict[f'hist_{i}'] = val
                
                # Add HOG features (reduced)
                hog_reduced = hog_features[::50]  # Sample every 50th value
                for i, val in enumerate(hog_reduced):
                    feature_dict[f'hog_{i}'] = val
                
                # Add LBP features (reduced)
                lbp_reduced = lbp_features[::10]
                for i, val in enumerate(lbp_reduced):
                    feature_dict[f'lbp_{i}'] = val
                
                # Add color moments
                for i, val in enumerate(color_moments):
                    feature_dict[f'color_moment_{i}'] = val
                
                # Add basic stats
                feature_dict['mean_intensity'] = np.mean(img)
                feature_dict['std_intensity'] = np.std(img)
                
                features_list.append(feature_dict)
            
            print(f"Processed {member_name} - {expression}")
        
        return features_list
    
    def create_sample_images(self, member_name='sample_member'):
        """
        Create sample placeholder images for testing
        (Replace this with actual photos)
        """
        expressions = ['neutral', 'smile', 'surprised']
        colors = [(200, 200, 200), (255, 200, 150), (150, 200, 255)]
        
        for expr, color in zip(expressions, colors):
            # Create 400x400 colored image with text
            img = np.ones((400, 400, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
            
            # Add text
            text = f"{member_name}\n{expr}"
            cv2.putText(img, member_name, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (0, 0, 0), 3)
            cv2.putText(img, expr, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (0, 0, 0), 3)
            
            # Save
            filename = self.base_dir / f"{member_name}_{expr}.jpg"
            cv2.imwrite(str(filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Created sample image: {filename}")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("IMAGE PROCESSING PIPELINE")
    print("="*60)
    
    processor = ImageProcessor(base_dir='Images')
    
    # Define your team members
    team_members = ['Ganza', 'Oreste', 'gershom', 'roxanne']
    
    # === ONLY UNCOMMENT THIS FOR TESTING WITHOUT REAL IMAGES ===
    # print("\nCreating sample images (for testing only)...")
    # for member in team_members:
    #     processor.create_sample_images(member)

    # === PROCESS REAL IMAGES ===
    all_features = []
    for member in team_members:
        print(f"\nProcessing images for {member}...")
        member_features = processor.process_member_images(member)
        if member_features:
            all_features.extend(member_features)
        else:
            print(f"No features extracted for {member}.")

    if all_features:
        df_features = pd.DataFrame(all_features)
        df_features.to_csv('image_features.csv', index=False)
        
        print(f"\n{'='*60}")
        print(f"Feature extraction complete!")
        print(f"Total samples processed: {len(all_features)}")
        print(f"Features saved to: image_features.csv")
        print(f"Shape: {df_features.shape}")
        print("="*60)
    else:
        print("No images were processed. Check your image directory and file names.")