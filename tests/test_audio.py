"""Fichiers m4a / AAC (voix enregistrées au téléphone) : lus par PyAV, puisque librosa ne sait pas les lire sans
ffmpeg (« Format not recognised », constaté avec un m4a de l'utilisateur)."""
import numpy as np
import pytest
import soundfile as sf

from conftest import no_progress
from studiovoix import audio, nettoyage, rvc
from studiovoix import config as cfg
from studiovoix.voix import list_voices, save_voice


def ecrire_m4a(chemin, secondes=12.0, sr=48000, canaux=1, freq=220.0):
    """Note avec un peu d'aigus, encodée en AAC dans un conteneur m4a (comme un téléphone)."""
    import av

    t = np.arange(int(secondes * sr)) / sr
    y = (0.3 * np.sin(2 * np.pi * freq * t) + 0.03 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    donnees = np.tile(y, (canaux, 1))
    with av.open(str(chemin), "w", format="mp4") as sortie:
        flux = sortie.add_stream("aac", rate=sr, layout="mono" if canaux == 1 else "stereo")
        for i in range(0, donnees.shape[1], 1024):
            trame = av.AudioFrame.from_ndarray(np.ascontiguousarray(donnees[:, i:i + 1024]), format="fltp",
                                               layout="mono" if canaux == 1 else "stereo")
            trame.sample_rate = sr
            for paquet in flux.encode(trame):
                sortie.mux(paquet)
        for paquet in flux.encode(None):
            sortie.mux(paquet)
    return chemin


def test_lecture_m4a(tmp_path):
    m4a = ecrire_m4a(tmp_path / "Amélie.m4a", canaux=2)
    with pytest.raises(Exception):
        sf.info(str(m4a))  # soundfile ne connaît pas ce format : sans PyAV, fichier illisible
    y, sr = audio.charger(m4a)
    assert y.ndim == 1 and sr == 48000 and abs(len(y) / sr - 12.0) < 0.1 and 0.25 < np.abs(y).max() < 0.6  # AAC : léger dépassement au début
    y, sr = audio.charger(m4a, sr=44100, mono=False)
    assert y.shape[0] == 2 and sr == 44100 and abs(y.shape[1] / sr - 12.0) < 0.1
    wav = tmp_path / "a.wav"  # les autres formats passent toujours par librosa
    sf.write(str(wav), np.zeros(4800, "float32"), 48000)
    assert audio.charger(wav)[1] == 48000
    (tmp_path / "faux.m4a").write_bytes(b"pas du son")
    with pytest.raises(Exception):
        audio.charger(tmp_path / "faux.m4a")


def test_voix_m4a_dans_la_bibliotheque_le_nettoyage_et_rvc(env, monkeypatch):
    m4a = ecrire_m4a(env / "Amélie.m4a")
    save_voice(str(m4a), "Amélie")
    assert list_voices() == ["Amélie"] and sf.info(str(cfg.VOICES_DIR / "Amélie.wav")).samplerate == 44100
    # nettoyage : converti en WAV PCM avant le moteur (qui ne lit que ça sans ffmpeg)
    monkeypatch.setattr(nettoyage, "_verifier_installation", lambda: None)
    monkeypatch.setattr(nettoyage, "lancer_moteur", lambda *a, **k: None)
    monkeypatch.setattr(nettoyage.serveur_acestep, "liberer_gpu", lambda *a: None)
    nettoyage.nettoyer(str(m4a), list(nettoyage.NIVEAUX)[0], progress=no_progress)
    info = sf.info(str(next(cfg.CLEAN_DIR.iterdir()) / "original.wav"))
    assert (info.subtype, info.channels) == ("PCM_16", 1) and abs(info.duration - 12.0) < 0.1
    # entraînement RVC : m4a accepté, analysé et copié en WAV
    assert "12 s" in rvc.analyser_enregistrements([str(m4a)], [])
    dossier, total = rvc.preparer_jeu("essai", [str(m4a)], [])
    assert abs(total - 12.0) < 0.1 and [p.suffix for p in dossier.iterdir()] == [".wav"]
