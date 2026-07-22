"""
Waveform-level augmentation for cough audio.
Used ONLY on the training split — never on val/test — to avoid leakage
and inflated accuracy numbers.

Each augmentation simulates realistic real-world variation:
- noise: different microphone/room noise floors
- pitch_shift: natural voice/cough pitch variation between people
- time_shift: cough not starting at exactly the same moment in the clip
- time_stretch: coughs of slightly different duration/intensity
"""

import numpy as np
import librosa


def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def pitch_shift(y, sr, n_steps=None):
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def time_shift(y, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)


def time_stretch(y, rate=None):
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    # Pad or trim back to original length
    if len(y_stretched) < len(y):
        y_stretched = np.pad(y_stretched, (0, len(y) - len(y_stretched)))
    else:
        y_stretched = y_stretched[: len(y)]
    return y_stretched


def augment_clip(y, sr):
    """
    Returns a list of augmented versions of one clip.
    3 augmentations per original -> 4x effective training data.
    """
    augmented = [
        add_noise(y),
        pitch_shift(y, sr),
        time_shift(y),
    ]
    return augmented
