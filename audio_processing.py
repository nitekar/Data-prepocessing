# audio_processing.py
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
from scipy import signal
import os

# Required for formants, jitter, shimmer
try:
    import parselmouth
except ImportError:
    print("Warning: parselmouth not installed. Run: pip install praat-parselmouth")
    parselmouth = None


class AudioProcessor:
    """
    Enhanced Audio processing pipeline for voice verification
    Now with pitch, formants, jitter, shimmer, HNR, contrast
    """
    
    def __init__(self, base_dir='Audios', sr=22050):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.sr = sr  # Sample rate
        
    def load_audio(self, audio_path):
        """Load audio file with normalization"""
        y, sr = librosa.load(str(audio_path), sr=self.sr)
        y = librosa.util.normalize(y)  # ← CRITICAL: same volume
        return y, sr
    
    def display_waveform(self, y, sr, title='Waveform'):
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        return plt.gcf()
    
    def display_spectrogram(self, y, sr, title='Spectrogram'):
        plt.figure(figsize=(12, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_audio(self, audio_path, save_prefix='audio_viz'):
        y, sr = self.load_audio(audio_path)
        fig1 = self.display_waveform(y, sr, f'Waveform: {audio_path.name}')
        fig1.savefig(f'{save_prefix}_waveform.png', dpi=150, bbox_inches='tight')
        plt.close()
        fig2 = self.display_spectrogram(y, sr, f'Spectrogram: {audio_path.name}')
        fig2.savefig(f'{save_prefix}_spectrogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualizations saved: {save_prefix}_waveform.png and {save_prefix}_spectrogram.png")
    
    # === AUGMENTATIONS ===
    def augment_pitch_shift(self, y, sr, n_steps=2):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    def augment_time_stretch(self, y, rate=1.2):
        return librosa.effects.time_stretch(y, rate=rate)
    
    def augment_add_noise(self, y, noise_level=0.005):
        noise = np.random.randn(len(y))
        return y + noise_level * noise
    
    def augment_change_speed(self, y, sr, speed_factor=1.1):
        return librosa.effects.time_stretch(y, rate=speed_factor)
    
    def augment_add_reverb(self, y, sr, room_size=0.3):
        reverb_time = int(sr * room_size)
        reverb = np.exp(-np.arange(reverb_time) / (sr * 0.1))
        reverb = reverb / np.sum(reverb)
        return signal.convolve(y, reverb, mode='same')

    # === BASIC FEATURES ===
    def extract_mfcc(self, y, sr, n_mfcc=13):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.concatenate([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
    
    def extract_spectral_features(self, y, sr):
        features = {}
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['spectral_centroid_mean'] = np.mean(cent)
        features['spectral_centroid_std'] = np.std(cent)
        features['spectral_rolloff_mean'] = np.mean(roll)
        features['spectral_rolloff_std'] = np.std(roll)
        features['spectral_bandwidth_mean'] = np.mean(bw)
        features['spectral_bandwidth_std'] = np.std(bw)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        return features
    
    def extract_energy_features(self, y):
        rms = librosa.feature.rms(y=y)[0]
        energy_entropy = -np.sum((rms / (np.sum(rms) + 1e-10)) * np.log2(rms / (np.sum(rms) + 1e-10) + 1e-10))
        return {
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms),
            'total_energy': np.sum(y ** 2),
            'energy_entropy': energy_entropy
        }
    
    def extract_chroma_features(self, y, sr):
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        return np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])
    
    def extract_tempo(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        try:
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        except:
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return tempo[0] if hasattr(tempo, '__len__') else tempo

    # === NEW: ADVANCED FEATURES ===
    def extract_pitch_features(self, y, sr):
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        if not pitch_values:
            return {f'pitch_{k}': 0.0 for k in ['mean','std','min','max','range','voiced_ratio']}
        p = np.array(pitch_values)
        return {
            'pitch_mean': float(np.mean(p)),
            'pitch_std': float(np.std(p)),
            'pitch_min': float(np.min(p)),
            'pitch_max': float(np.max(p)),
            'pitch_range': float(np.ptp(p)),
            'voiced_ratio': len(pitch_values) / pitches.shape[1]
        }

    def extract_harmonics(self, y, sr):
        def hnr(signal):
            if len(signal) < 100: return 0.0
            corr = np.correlate(signal, signal, mode='full')
            corr = corr[len(corr)//2:]
            if len(corr) < 2: return 0.0
            zero_lag = corr[0]
            try:
                first_peak = max(corr[1:50])
            except:
                first_peak = 0
            return 10 * np.log10(first_peak / (zero_lag - first_peak + 1e-10)) if first_peak > 0 else 0.0
        return {'hnr': hnr(y)}

    def extract_spectral_contrast(self, y, sr):
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        return {
            'contrast_mean': float(np.mean(contrast)),
            'contrast_std': float(np.std(contrast))
        }

    def extract_formants(self, y, sr):
        if parselmouth is None:
            return {'formant_f1': 0.0, 'formant_f2': 0.0}
        try:
            snd = parselmouth.Sound(y, sampling_frequency=sr)
            formant = snd.to_formant_burg(max_number_of_formants=5)
            t = snd.duration / 2
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            return {
                'formant_f1': float(f1) if f1 else 0.0,
                'formant_f2': float(f2) if f2 else 0.0
            }
        except:
            return {'formant_f1': 0.0, 'formant_f2': 0.0}

    def extract_jitter_shimmer(self, y, sr):
        if parselmouth is None:
            return {'jitter': 0.0, 'shimmer': 0.0}
        try:
            snd = parselmouth.Sound(y, sampling_frequency=sr)
            pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)...", 75, 500)
            jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            return {'jitter': float(jitter), 'shimmer': float(shimmer)}
        except:
            return {'jitter': 0.0, 'shimmer': 0.0}

    # === MAIN PROCESSING ===
    def process_member_audio(self, member_name, phrases=['yes_approve', 'confirm_transaction']):
        features_list = []
        for phrase in phrases:
            audio_path = self.base_dir / f"{member_name}_{phrase}.wav"
            if not audio_path.exists():
                print(f"Warning: {audio_path} not found")
                continue

            y, sr = self.load_audio(audio_path)
            self.visualize_audio(audio_path, save_prefix=f'{member_name}_{phrase}')

            # Augmentations
            versions = [
                ('original', y),
                ('pitch_shifted', self.augment_pitch_shift(y, sr, 2)),
                ('time_stretched', self.augment_time_stretch(y, 1.2)),
                ('noisy', self.augment_add_noise(y, 0.005)),
                ('speed_changed', self.augment_change_speed(y, sr, 1.1)),
                ('reverb', self.augment_add_reverb(y, sr, 0.3))
            ]

            aug_dir = self.base_dir / 'augmented' / member_name
            aug_dir.mkdir(parents=True, exist_ok=True)
            for aug_type, audio in versions[1:]:  # Skip original
                sf.write(str(aug_dir / f"{phrase}_{aug_type}.wav"), audio, sr)

            # Extract ALL features
            for aug_type, audio in versions:
                mfcc = self.extract_mfcc(audio, sr)
                spectral = self.extract_spectral_features(audio, sr)
                energy = self.extract_energy_features(audio)
                chroma = self.extract_chroma_features(audio, sr)
                tempo = self.extract_tempo(audio, sr)
                pitch = self.extract_pitch_features(audio, sr)
                hnr = self.extract_harmonics(audio, sr)
                contrast = self.extract_spectral_contrast(audio, sr)
                formants = self.extract_formants(audio, sr)
                jitter_shimmer = self.extract_jitter_shimmer(audio, sr)

                feature_dict = {
                    'member_name': member_name,
                    'phrase': phrase,
                    'augmentation': aug_type,
                    'audio_path': str(audio_path),
                    'duration': len(audio) / sr,
                    'tempo': tempo
                }

                # Add all
                for i, v in enumerate(mfcc): feature_dict[f'mfcc_{i}'] = v
                for i, v in enumerate(chroma): feature_dict[f'chroma_{i}'] = v
                feature_dict.update(spectral)
                feature_dict.update(energy)
                feature_dict.update(pitch)
                feature_dict.update(hnr)
                feature_dict.update(contrast)
                feature_dict.update(formants)
                feature_dict.update(jitter_shimmer)

                features_list.append(feature_dict)

            print(f"Processed {member_name} - {phrase}")
        return features_list

    def extract_features_from_file(self, audio_path, normalize=True):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = self.load_audio(audio_path)
        if not normalize:
            y, _ = librosa.load(str(audio_path), sr=self.sr)

        # All features
        mfcc = self.extract_mfcc(y, sr)
        spectral = self.extract_spectral_features(y, sr)
        energy = self.extract_energy_features(y)
        chroma = self.extract_chroma_features(y, sr)
        tempo = self.extract_tempo(y, sr)
        pitch = self.extract_pitch_features(y, sr)
        hnr = self.extract_harmonics(y, sr)
        contrast = self.extract_spectral_contrast(y, sr)
        formants = self.extract_formants(y, sr)
        jitter_shimmer = self.extract_jitter_shimmer(y, sr)

        features = {
            'duration': len(y) / sr,
            'tempo': tempo,
        }

        for i, v in enumerate(mfcc): features[f'mfcc_{i}'] = v
        for i, v in enumerate(chroma): features[f'chroma_{i}'] = v
        features.update(spectral)
        features.update(energy)
        features.update(pitch)
        features.update(hnr)
        features.update(contrast)
        features.update(formants)
        features.update(jitter_shimmer)

        return features

    def create_sample_audio(self, member_name='sample_member', duration=2.0):
        phrases = ['yes_approve', 'confirm_transaction']
        for phrase in phrases:
            t = np.linspace(0, duration, int(self.sr * duration))
            f1 = 200 + np.random.randint(-50, 50)
            f2 = 400 + np.random.randint(-50, 50)
            audio = 0.3 * np.sin(2 * np.pi * f1 * t) + 0.2 * np.sin(2 * np.pi * f2 * t)
            audio += 0.05 * np.random.randn(len(t))
            audio = audio / np.max(np.abs(audio)) * 0.8
            filename = self.base_dir / f"{member_name}_{phrase}.wav"
            sf.write(str(filename), audio, self.sr)
            print(f"Created sample: {filename}")


# === MAIN ===
if __name__ == "__main__":
    print("="*60)
    print("AUDIO PROCESSING PIPELINE - ENHANCED")
    print("="*60)
    
    processor = AudioProcessor(base_dir='Audios', sr=22050)
    team_members = ['roxane', 'gershom', 'Oreste', 'Ganza']
    
    all_features = []
    for member in team_members:
        print(f"\nProcessing {member}...")
        all_features.extend(processor.process_member_audio(member))
    
    df = pd.DataFrame(all_features)
    df.to_csv('audio_features.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Total samples: {len(all_features)}")
    print(f"Features: {df.shape[1]}")
    print(f"Saved to: audio_features.csv")
    print("="*60)