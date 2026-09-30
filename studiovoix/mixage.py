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
    return mixer([(vocals_path, gain_voix), (instru_path, gain_instru)], out_path)


def mixer(pistes, out_path):
    """Additionne des pistes [(fichier, gain), …] en 44,1 kHz stéréo, avec limitation du pic à 0,95."""
    ys = [load_stereo(p) * g for p, g in pistes]
    n = max(y.shape[1] for y in ys)
    m = sum(np.pad(y, ((0, 0), (0, n - y.shape[1]))) for y in ys)
    peak = float(np.max(np.abs(m))) or 1.0
    if peak > 0.95:
        m = m / peak * 0.95
    sf.write(str(out_path), m.T, SR)
    return out_path
