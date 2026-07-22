"""
Feature extraction for cough/breath audio classification.

Two feature sets:
1. Statistical features (for classic ML) - MFCC stats, spectral features, ZCR
2. Mel-spectrograms (for DL) - 2D time-frequency representations for CNN input
"""

import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 2.0  # seconds - pad/trim all clips to this length (Virufy segmented clips are ~1.5-2s)
N_MFCC = 13
N_MELS = 128
HOP_LENGTH = 512


def load_audio(filepath, sr=SAMPLE_RATE, duration=DURATION):
    """Load and pad/trim audio to fixed duration."""
    y, _ = librosa.load(filepath, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_statistical_features(y, sr=SAMPLE_RATE):
    """
    Extract statistical features for classic ML (Random Forest / XGBoost).
    Returns a fixed-length feature vector.
    """
    features = {}

    # MFCCs - timbral/spectral shape
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
        features[f"mfcc_{i}_std"] = np.std(mfccs[i])

    # Zero crossing rate - correlates with "roughness" in cough/breath sounds
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    # Spectral centroid - "brightness" of sound
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spec_centroid_mean"] = np.mean(spec_cent)
    features["spec_centroid_std"] = np.std(spec_cent)

    # Spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spec_rolloff_mean"] = np.mean(spec_rolloff)

    # Spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spec_bandwidth_mean"] = np.mean(spec_bw)

    # RMS energy - loudness envelope, useful for detecting cough bursts
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)

    # Chroma - less relevant for cough but cheap to add, sometimes helps
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = np.mean(chroma)

    return features


def extract_mel_spectrogram(y, sr=SAMPLE_RATE):
    """
    Extract mel-spectrogram for DL (CNN input).
    Returns a 2D array (n_mels x time_steps), log-scaled.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize to [0, 1] for stable CNN training
    log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (
        log_mel_spec.max() - log_mel_spec.min() + 1e-8
    )
    return log_mel_spec


def process_file(filepath):
    """Convenience function: load a file and return both feature types."""
    y = load_audio(filepath)
    stat_features = extract_statistical_features(y)
    mel_spec = extract_mel_spectrogram(y)
    return stat_features, mel_spec
