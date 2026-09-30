"""Voix de référence de l'utilisateur, rangées dans data/voices/<nom>.wav."""
import gradio as gr
import librosa
import numpy as np
import soundfile as sf

from . import config as cfg


def list_voices():
    return sorted(p.stem for p in cfg.VOICES_DIR.glob("*.wav"))


def save_voice(audio_path, name):
    if not audio_path:
        raise gr.Error("Enregistre ou importe d'abord un échantillon de voix.")
    name = "".join(c for c in (name or "").strip() if c.isalnum() or c in "-_ ").strip()
    if not name:
        raise gr.Error("Donne un nom à cette voix.")
    y, _ = librosa.load(audio_path, sr=cfg.SR, mono=True)
    duree = len(y) / cfg.SR
    if duree < 5:
        raise gr.Error(f"Échantillon trop court ({duree:.1f} s). Vise 10 à 25 secondes.")
    y = y[: 30 * cfg.SR]  # Seed-VC exploite 1 à 30 s de référence
    peak = float(np.max(np.abs(y))) or 1.0
    y = y / peak * 0.95
    sf.write(cfg.VOICES_DIR / f"{name}.wav", y, cfg.SR)
    voices = list_voices()
    msg = f"Voix « {name} » enregistrée ({min(duree, 30):.0f} s utilisées)."
    return msg, gr.update(choices=voices, value=name)
