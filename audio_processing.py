import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
from scipy import signal

class AudioProcessor:
    """
    Audio processing pipeline for voice verification
    Handles loading, augmentation, visualization, and feature extraction
    """
    
    def __init__(self, base_dir='Audios', sr=22050):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.sr = sr  # Sample rate
        
    def load_audio(self, audio_path):
        """Load audio file"""
        y, sr = librosa.load(str(audio_path), sr=self.sr)
        return y, sr
    
    def display_waveform(self, y, sr, title='Waveform'):
        """Display audio waveform"""
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        return plt.gcf()
    
    def display_spectrogram(self, y, sr, title='Spectrogram'):
        """Display audio spectrogram"""
        plt.figure(figsize=(12, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_audio(self, audio_path, save_prefix='audio_viz'):
        """Create and save waveform and spectrogram visualizations"""
        y, sr = self.load_audio(audio_path)
        
        # Waveform
        fig1 = self.display_waveform(y, sr, f'Waveform: {audio_path.name}')
        fig1.savefig(f'{save_prefix}_waveform.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Spectrogram
        fig2 = self.display_spectrogram(y, sr, f'Spectrogram: {audio_path.name}')
        fig2.savefig(f'{save_prefix}_spectrogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved: {save_prefix}_waveform.png and {save_prefix}_spectrogram.png")
    
    def augment_pitch_shift(self, y, sr, n_steps=2):
        """Shift pitch by n_steps semitones"""
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    def augment_time_stretch(self, y, rate=1.2):
        """Stretch or compress time by rate factor"""
        return librosa.effects.time_stretch(y, rate=rate)
    
    def augment_add_noise(self, y, noise_level=0.005):
        """Add white noise to audio"""
        noise = np.random.randn(len(y))
        augmented = y + noise_level * noise
        return augmented
    
    def augment_change_speed(self, y, sr, speed_factor=1.1):
        """Change playback speed"""
        return librosa.effects.time_stretch(y, rate=speed_factor)
    
    def augment_add_reverb(self, y, sr, room_size=0.5):
        """Add simple reverb effect"""
        # Simple reverb using convolution with exponential decay
        reverb_time = int(sr * room_size)
        reverb = np.exp(-np.arange(reverb_time) / (sr * 0.1))
        reverb = reverb / np.sum(reverb)
        
        augmented = signal.convolve(y, reverb, mode='same')
        return augmented
    
    def extract_mfcc(self, y, sr, n_mfcc=13):
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Return mean and std of each coefficient
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])
    
    def extract_spectral_features(self, y, sr):
        """Extract various spectral features"""
        features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def extract_energy_features(self, y):
        """Extract energy-related features"""
        features = {}
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Total energy
        features['total_energy'] = np.sum(y ** 2)
        
        # Energy entropy
        frame_energies = librosa.feature.rms(y=y)[0]
        frame_energies = frame_energies / (np.sum(frame_energies) + 1e-10)
        features['energy_entropy'] = -np.sum(frame_energies * np.log2(frame_energies + 1e-10))
        
        return features
    
    def extract_chroma_features(self, y, sr):
        """Extract chroma features"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        return np.concatenate([chroma_mean, chroma_std])
    
    def extract_tempo(self, y, sr):
        """Extract tempo (BPM) - Works with librosa <0.10 and >=0.10"""
        import librosa
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
        try:
            # Try new API (0.10+)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
            return tempo[0] if hasattr(tempo, '__len__') else tempo
        except (AttributeError, ValueError):
            # Fall back to old API (<0.10)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            return tempo[0] if hasattr(tempo, '__len__') else tempo
    
    def process_member_audio(self, member_name, phrases=['yes_approve', 'confirm_transaction']):
        """
        Process all audio files for a team member
        Returns list of feature dictionaries
        """
        features_list = []
        
        for phrase in phrases:
            # Construct file path
            audio_path = self.base_dir / f"{member_name}_{phrase}.wav"
            
            if not audio_path.exists():
                print(f"Warning: {audio_path} not found, skipping...")
                continue
            
            # Load original audio
            y, sr = self.load_audio(audio_path)
            
            # Visualize original
            self.visualize_audio(audio_path, save_prefix=f'{member_name}_{phrase}')
            
            # Apply augmentations
            pitch_shifted = self.augment_pitch_shift(y, sr, n_steps=2)
            time_stretched = self.augment_time_stretch(y, rate=1.2)
            noisy = self.augment_add_noise(y, noise_level=0.005)
            speed_changed = self.augment_change_speed(y, sr, speed_factor=1.1)
            reverb = self.augment_add_reverb(y, sr, room_size=0.3)
            
            # Save augmented audio
            aug_dir = self.base_dir / 'augmented' / member_name
            aug_dir.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(aug_dir / f"{phrase}_pitch_shifted.wav"), pitch_shifted, sr)
            sf.write(str(aug_dir / f"{phrase}_time_stretched.wav"), time_stretched, sr)
            sf.write(str(aug_dir / f"{phrase}_noisy.wav"), noisy, sr)
            sf.write(str(aug_dir / f"{phrase}_speed_changed.wav"), speed_changed, sr)
            sf.write(str(aug_dir / f"{phrase}_reverb.wav"), reverb, sr)
            
            # Process all versions (original + augmented)
            versions = [
                ('original', y),
                ('pitch_shifted', pitch_shifted),
                ('time_stretched', time_stretched),
                ('noisy', noisy),
                ('speed_changed', speed_changed),
                ('reverb', reverb)
            ]
            
            for aug_type, audio in versions:
                # Extract features
                mfcc_features = self.extract_mfcc(audio, sr)
                spectral_features = self.extract_spectral_features(audio, sr)
                energy_features = self.extract_energy_features(audio)
                chroma_features = self.extract_chroma_features(audio, sr)
                tempo = self.extract_tempo(audio, sr)
                
                # Create feature dictionary
                feature_dict = {
                    'member_name': member_name,
                    'phrase': phrase,
                    'augmentation': aug_type,
                    'audio_path': str(audio_path),
                    'duration': len(audio) / sr,
                    'tempo': tempo
                }
                
                # Add MFCC features
                for i, val in enumerate(mfcc_features):
                    feature_dict[f'mfcc_{i}'] = val
                
                # Add spectral features
                feature_dict.update(spectral_features)
                
                # Add energy features
                feature_dict.update(energy_features)
                
                # Add chroma features
                for i, val in enumerate(chroma_features):
                    feature_dict[f'chroma_{i}'] = val
                
                features_list.append(feature_dict)
            
            print(f"Processed {member_name} - {phrase}")
        
        return features_list
    
    
    def extract_features_from_file(self, audio_path, normalize=True):
        """
        Extract full feature set from a single WAV file.
        Used for real-time or batch verification.
        """
        import os
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = self.load_audio(audio_path)
        
        # Optional: normalize audio (highly recommended for consistency)
        if normalize:
            y = librosa.util.normalize(y)

        # Extract all features
        mfcc = self.extract_mfcc(y, sr, n_mfcc=13)
        spectral = self.extract_spectral_features(y, sr)
        energy = self.extract_energy_features(y)
        chroma = self.extract_chroma_features(y, sr)
        tempo = self.extract_tempo(y, sr)

        # Build feature dictionary
        features = {
            'duration': len(y) / sr,
            'tempo': tempo,
        }

        # MFCC: 13 mean + 13 std = 26 values
        for i, val in enumerate(mfcc):
            features[f'mfcc_{i}'] = val

        # Spectral, energy
        features.update(spectral)
        features.update(energy)

        # Chroma: 12 mean + 12 std = 24 values
        for i, val in enumerate(chroma):
            features[f'chroma_{i}'] = val

        return features
    
    
    def create_sample_audio(self, member_name='sample_member', duration=2.0):
        """
        Create sample audio files for testing
        (Replace this with actual recordings)
        """
        phrases = ['yes_approve', 'confirm_transaction']
        
        for phrase in phrases:
            # Generate a simple tone (replace with actual voice recording)
            t = np.linspace(0, duration, int(self.sr * duration))
            # Create a combination of frequencies to simulate voice
            frequency1 = 200 + np.random.randint(-50, 50)  # Base frequency
            frequency2 = 400 + np.random.randint(-50, 50)  # Harmonic
            
            audio = 0.3 * np.sin(2 * np.pi * frequency1 * t)
            audio += 0.2 * np.sin(2 * np.pi * frequency2 * t)
            
            # Add some noise to make it more realistic
            audio += 0.05 * np.random.randn(len(t))
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Save
            filename = self.base_dir / f"{member_name}_{phrase}.wav"
            sf.write(str(filename), audio, self.sr)
            print(f"Created sample audio: {filename}")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("AUDIO PROCESSING PIPELINE")
    print("="*60)
    
    processor = AudioProcessor(base_dir='Audios', sr=22050)
    
    # Define your team members
    team_members = ['Oreste', 'Ganza', 'gershom']  # Replace with actual names
    
    # Create sample audio files (REPLACE THIS WITH YOUR ACTUAL RECORDINGS)
    #print("\nCreating sample audio files (replace with your actual recordings)...")
    #for member in team_members:
    #    processor.create_sample_audio(member)
    
    # Process all audio and extract features
    all_features = []
    
    for member in team_members:
        print(f"\nProcessing audio for {member}...")
        member_features = processor.process_member_audio(member)
        all_features.extend(member_features)
    
    # Create DataFrame and save to CSV
    df_features = pd.DataFrame(all_features)
    df_features.to_csv('audio_features.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Audio feature extraction complete!")
    print(f"Total samples processed: {len(all_features)}")
    print(f"Features saved to: audio_features.csv")
    print(f"Shape: {df_features.shape}")
    print(f"\nFeature columns:")
    print(df_features.columns.tolist())
    print(f"\nSample data:")
    #print(df_features[['member_name', 'phrase', 'augmentation', 'duration', 'tempo']].head(10))
    print("="*60)