"""Diaporama : mouvement « Ken Burns » des photos (zoom, travelling), voix off posée dans le montage, photos
allongées pour que la voix tienne, musique baissée sous la voix (vrai PyAV)."""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from conftest import no_progress
from studiovoix import diaporama as dp
from studiovoix import montage as mt
from test_convertir import _frequence, _images, _son


def _degrade(w=400, h=300):
    x = np.linspace(0, 255, w).astype(np.uint8)
    a = np.zeros((h, w, 3), np.uint8)
    a[..., 0] = x
    a[..., 1] = np.linspace(0, 255, h).astype(np.uint8)[:, None]
    return a


def test_ken_burns_zoom_et_travelling():
    images = list(mt._images_ken_burns(_degrade(), 10, 160, 120, 0.25, 0))  # zoom avant
    assert len(images) == 10 and images[0].shape == (120, 160, 3)
    centre = [int(im[60, 80, 0]) for im in images]
    assert max(centre) - min(centre) <= 3  # le centre ne bouge pas
    assert images[0][60, 2, 0] < images[-1][60, 2, 0] - 10  # le bord gauche se rapproche du centre (zoom)
    arriere = list(mt._images_ken_burns(_degrade(), 10, 160, 120, 0.25, 1))
    assert arriere[0][60, 2, 0] > arriere[-1][60, 2, 0] + 10
    droite = list(mt._images_ken_burns(_degrade(), 10, 160, 120, 0.25, 2))
    assert droite[-1][60, 80, 0] > droite[0][60, 80, 0] + 10  # le cadre glisse vers la droite (plus rouge)
    assert abs(int(droite[-1][60, 80, 1]) - int(droite[0][60, 80, 1])) <= 3  # sans monter ni descendre


def test_diaporama_avec_enregistrement_et_musique(env, tmp_path):
    photos = []
    for i, c in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200)]):
        a = _degrade()
        a[100:200, 150:250] = c
        Image.fromarray(a).save(tmp_path / f"{i}.jpg")
        photos.append(str(tmp_path / f"{i}.jpg"))
    voix = tmp_path / "voix.wav"
    sf.write(voix, 0.4 * np.sin(2 * np.pi * 440 * np.arange(44100 * 6) / 44100), 44100)
    musique = tmp_path / "musique.wav"
    sf.write(musique, 0.4 * np.sin(2 * np.pi * 880 * np.arange(44100 * 3) / 44100), 44100)
    msg, video, fichiers, dossier = dp.creer(photos, 1.5, mt.ENCHAINE, 0.5, dp.MOUVEMENT_DEFAUT,
                                             "Paysage 1280×720 (plus léger)", str(musique), 50, dp.ENREGISTREMENT,
                                             enregistrement=str(voix), titre="Vacances", progress=no_progress)
    images, codec, fps = _images(video)
    assert fps == 30 and images[0].shape == (720, 1280, 3)
    assert len(images) / fps >= 1.0 + 6 + 1.0 - 0.05  # photos allongées : la voix de 6 s tient
    assert "Diaporama de 3" in msg and fichiers[-1].endswith("voix.wav")
    y, sr = _son(video)
    voix_zone = y[int(3 * sr):int(4 * sr)]
    spectre = np.abs(np.fft.rfft(voix_zone))
    f = np.fft.rfftfreq(len(voix_zone), 1 / sr)
    assert spectre[np.abs(f - 440).argmin()] > 4 * spectre[np.abs(f - 880).argmin()]  # musique baissée sous la voix
    fin = y[int(len(y) - 0.9 * sr):int(len(y) - 0.4 * sr)]
    assert _frequence(fin, sr) == pytest.approx(880, abs=5)  # après la voix : la musique seule
    infos = json.loads((Path(dossier) / "creation.json").read_text(encoding="utf-8"))
    assert infos["diaporama"]["photos"] == 3 and infos["montage"]["ken_burns"] == 0.15
    assert infos["montage"]["voix"] == "voix.wav"
    assert not any((Path(dossier).parent.parent / "_tmp").glob("*diaporama_voix*"))


def test_diaporama_texte_lu_et_erreurs(env, tmp_path, monkeypatch):
    from studiovoix import chatterbox

    Image.fromarray(_degrade()).save(tmp_path / "a.png")

    def synthese(voix, texte, langue, *reglages, progress=None):
        f = tmp_path / "lu.wav"
        sf.write(f, 0.3 * np.sin(2 * np.pi * 300 * np.arange(24000 * 2) / 24000), 24000)
        assert texte == "Bonjour" and voix == "moi"
        return str(f), "ok"

    monkeypatch.setattr(chatterbox, "synthese", synthese)
    msg, video, fichiers, dossier = dp.creer([str(tmp_path / "a.png")], 2, mt.COUPE, 0, "Aucun (photos immobiles)",
                                             "Carré 1080×1080 (Instagram)", None, 40, dp.TEXTE, "moi", "Français",
                                             "Bonjour", progress=no_progress)
    images, _, fps = _images(video)
    assert images[0].shape == (1080, 1080, 3) and len(images) / fps >= 4.0 - 0.05  # 1 + 2 + 1 s
    with pytest.raises(gr.Error, match="Ajoute des photos"):
        dp.creer([str(tmp_path / "lu.wav")], progress=no_progress)
