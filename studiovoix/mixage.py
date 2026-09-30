"""Mixage voix + instrumental en 44,1 kHz stéréo (numpy / soundfile)."""
import librosa
import numpy as np
import soundfile as sf

from .config import SR


def load_stereo(path, sr=SR):
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, canaux)
    y = y.T
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    return y[:2]


def mix(vocals_path, instru_path, out_path, gain_voix=1.0, gain_instru=1.0):
    v = load_stereo(vocals_path) * gain_voix
    i = load_stereo(instru_path) * gain_instru
    n = max(v.shape[1], i.shape[1])
    v = np.pad(v, ((0, 0), (0, n - v.shape[1])))
    i = np.pad(i, ((0, 0), (0, n - i.shape[1])))
    m = v + i
    peak = float(np.max(np.abs(m))) or 1.0
    if peak > 0.95:
        m = m / peak * 0.95
    sf.write(str(out_path), m.T, SR)
    return out_path
