"""Lecture des fichiers audio fournis par l'utilisateur, quel que soit leur format.

librosa (soundfile) lit WAV, FLAC, OGG et MP3, mais pas le m4a / AAC des téléphones (« Format not recognised ») :
son repli audioread demande ffmpeg, absent chez l'utilisateur. PyAV (paquet `av`, roues Windows avec les
bibliothèques de FFmpeg incluses, rien à installer à part) décode alors m4a, AAC, Opus, WebM, vidéos… (demande de
l'utilisateur du 09/10 : « j'ai des voix en m4a »).
"""
from pathlib import Path

import numpy as np

# Formats proposés dans les champs d'import (tous lus par charger)
EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma", ".webm", ".mp4"]


def charger(chemin, sr=None, mono=True):
    """(signal float32, fréquence) comme librosa.load : 1-D si mono, sinon (canaux, échantillons).
    sr=None : fréquence d'origine. Lève une exception si le fichier est illisible."""
    import librosa

    chemin = str(chemin)
    try:
        import soundfile as sf

        sf.info(chemin)  # soundfile sait-il l'ouvrir ? (sinon librosa passerait par audioread, sans ffmpeg)
    except Exception:  # noqa: BLE001 - m4a, aac… : décodés par PyAV
        return _par_pyav(chemin, sr, mono)
    return librosa.load(chemin, sr=sr, mono=mono)


def _par_pyav(chemin, sr, mono):
    import av

    with av.open(chemin) as conteneur:
        if not conteneur.streams.audio:
            raise ValueError(f"aucune piste audio dans {Path(chemin).name}")
        flux = conteneur.streams.audio[0]
        canaux = 1 if mono or (flux.channels or 1) < 2 else 2
        frequence = int(sr or flux.rate or 44100)
        reechantillonneur = av.AudioResampler(format="fltp", layout="mono" if canaux == 1 else "stereo",
                                              rate=frequence)
        morceaux = []
        for trame in conteneur.decode(flux):
            morceaux += [t.to_ndarray() for t in reechantillonneur.resample(trame)]
        morceaux += [t.to_ndarray() for t in reechantillonneur.resample(None)]
    if not morceaux:
        raise ValueError(f"{Path(chemin).name} ne contient aucun son")
    y = np.concatenate(morceaux, axis=1).astype(np.float32)
    return (y[0] if canaux == 1 else y), frequence
