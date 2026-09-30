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


def couper_jingle(src, out_path, duree, fondu=0.4, seuil_db=-45.0):
    """Jingle court à partir d'une génération plus longue (ACE-Step génère au moins 10 s) :
    retire le silence de début, coupe sur le creux d'énergie le plus proche de « duree » (± 1 s)
    puis ajoute un fondu de sortie. Renvoie la durée obtenue."""
    y = load_stereo(src)
    mono = np.abs(y).max(axis=0)
    seuil = 10 ** (seuil_db / 20)
    debut = int(np.argmax(mono > seuil)) if np.any(mono > seuil) else 0
    y = y[:, debut:]
    trame = int(0.02 * SR)  # énergie par trames de 20 ms
    n = y.shape[1] // trame
    if n == 0:
        raise ValueError("fichier trop court")
    energie = np.sqrt((y[:, : n * trame].reshape(2, n, trame) ** 2).mean(axis=(0, 2)))
    cible = int(duree * SR / trame)
    lo, hi = max(1, cible - int(1 / 0.02)), min(n, cible + int(1 / 0.02) + 1)
    fin = (lo + int(np.argmin(energie[lo:hi]))) * trame if lo < hi else min(y.shape[1], int(duree * SR))
    y = y[:, :fin]
    nf = min(int(fondu * SR), y.shape[1])
    y[:, y.shape[1] - nf:] *= np.linspace(1.0, 0.0, nf, dtype=np.float32)
    peak = float(np.max(np.abs(y))) or 1.0
    if peak > 0.95:
        y = y / peak * 0.95
    sf.write(str(out_path), y.T, SR)
    return y.shape[1] / SR
