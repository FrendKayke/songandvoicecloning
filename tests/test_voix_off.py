"""Vidéo avec voix off : voix lue (faux Chatterbox) ou enregistrement, son d'origine mixé, vidéo recopiée telle quelle
ou prolongée sur sa dernière image (vrai PyAV)."""
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import soundfile as sf

from conftest import no_progress
from studiovoix import chatterbox, galerie, voix_off
from studiovoix import config as cfg
from test_diffusion import _video_test


def _frequences(fichier, canal=0):
    """Fréquences dominantes (Hz) du son d'un MP4, lues par PyAV."""
    import av

    morceaux = []
    with av.open(str(fichier)) as v:
        flux = v.streams.audio[0]
        sr = flux.rate
        for trame in v.decode(flux):
            morceaux.append(trame.to_ndarray().astype(np.float32))
    son = np.concatenate(morceaux, axis=1)[canal]
    spectre = np.abs(np.fft.rfft(son))
    f = np.fft.rfftfreq(len(son), 1 / sr)
    return sorted(f[np.argsort(spectre)[-2:]].round()), len(son) / sr


def _images(fichier):
    import av

    with av.open(str(fichier)) as v:
        return sum(1 for _ in v.decode(video=0)), v.streams.video[0].codec_context.name


@pytest.fixture
def faux_tts(env, monkeypatch):
    """Chatterbox remplacé : 5 s de la à 440 Hz, 24 kHz, dans data/tts."""
    def synthese(voix, texte, langue, exag, cfg_w, temp, graine, progress=None):
        d = cfg.TTS_DIR / "lecture"
        d.mkdir(parents=True, exist_ok=True)
        t = np.arange(5 * 24000) / 24000
        sf.write(str(d / "parole.wav"), (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 24000)
        return str(d / "parole.wav"), f"lu : {texte}"

    monkeypatch.setattr(chatterbox, "synthese", synthese)
    return env


def test_voix_off_prolongee_et_son_garde(faux_tts):
    """Vidéo de 3 s (220 Hz) + voix de 5 s (440 Hz) : prolongée à 5 s sur la dernière image (réencodée), les deux
    sons présents ; création dans la galerie."""
    video = _video_test(faux_tts / "clip.mp4")  # 3 s, 12 images/s, la 220 Hz
    msg, sortie, fichiers, dossier = voix_off.generer(str(video), voix_off.TEXTE, "moi", "Français", "Bonjour",
                                                      None, True, 50, True, "présentation", progress=no_progress)
    assert "prolongée de 2.0 s" in msg and Path(sortie).name == "video_voix_off.mp4"
    freqs, duree = _frequences(sortie)
    assert freqs == [220.0, 440.0] and 4.9 < duree < 5.3
    n, codec = _images(sortie)
    assert n == 60 and codec == "h264"  # 36 images + 24 figées
    infos = galerie.lire(dossier)
    assert infos["type"] == "video" and infos["voix_off"]["source"] == "texte" and infos["voix_off"]["prolonge_s"] == 2.0
    assert (Path(dossier) / "voix.wav").exists() and (Path(dossier) / "original.mp4").exists()
    assert any(d == dossier for _, d in galerie.lister("Vidéos"))
    with pytest.raises(gr.Error, match="Vidéo avec voix off"):
        galerie.recreer(dossier, 1, progress=no_progress)


def test_voix_off_enregistrement_sans_son_d_origine(faux_tts):
    """Enregistrement de 2 s (330 Hz) sur une vidéo de 3 s sans garder le son : vidéo recopiée telle quelle
    (36 images), seule la voix dans le son ; sans prolongement, une voix trop longue est coupée."""
    video = _video_test(faux_tts / "clip.mp4")
    t = np.arange(2 * 22050) / 22050
    voix = faux_tts / "voix.wav"
    sf.write(str(voix), (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), 22050)
    msg, sortie, _, dossier = voix_off.generer(str(video), voix_off.FICHIER, None, "Français", "", str(voix), False,
                                               30, True, "voix", progress=no_progress)
    freqs, duree = _frequences(sortie)
    assert 330.0 in freqs and 220.0 not in freqs and 2.9 < duree < 3.2
    assert _images(sortie)[0] == 36 and galerie.lire(dossier)["voix_off"]["source"] == "fichier"
    # voix de 5 s, sans prolonger : coupée
    msg, sortie, _, _ = voix_off.generer(str(video), voix_off.TEXTE, "moi", "Français", "x", None, True, 0, False,
                                         "coupe", progress=no_progress)
    assert "coupée" in msg and 2.9 < _frequences(sortie)[1] < 3.2 and _images(sortie)[0] == 36
    # vidéo muette : on le dit ; erreurs claires
    muet = _video_test(faux_tts / "muet.mp4", avec_son=False)
    msg, *_ = voix_off.generer(str(muet), voix_off.TEXTE, "moi", "Français", "x", None, True, 30, False, "m",
                               progress=no_progress)
    assert "pas de son à garder" in msg
    with pytest.raises(gr.Error, match="Écris le texte"):
        voix_off.generer(str(video), voix_off.TEXTE, "moi", "Français", " ", None, True, 30, False, progress=no_progress)
    with pytest.raises(gr.Error, match="Ajoute l'enregistrement"):
        voix_off.generer(str(video), voix_off.FICHIER, None, "Français", "", None, True, 30, False, progress=no_progress)
    with pytest.raises(gr.Error, match="Ajoute la vidéo"):
        voix_off.generer(None, progress=no_progress)
