"""Montage vidéo avec le vrai PyAV : durée avec fondus enchaînés, fondu au noir, image fixe, vidéo en portrait dans
une sortie paysage, musique de fond répétée et baissée sous le son des vidéos, titre et logo posés."""
import json

import gradio as gr
import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from conftest import no_progress
from studiovoix import montage as mt
from test_convertir import _clip, _frequence, _images, _son


def _monter(fichiers, **reglages):
    return mt.monter(fichiers, progress=no_progress, **reglages)


@pytest.fixture
def morceaux(env, tmp_path):
    a = _clip(tmp_path / "a.mp4", 3, 12, (320, 240), 220)
    b = _clip(tmp_path / "b.mp4", 2, 25, (240, 320), 440, couleur=200)  # portrait, autre cadence
    Image.new("RGB", (640, 480), (255, 0, 0)).save(tmp_path / "image.png")
    t = np.arange(44100 * 2) / 44100
    sf.write(tmp_path / "musique.wav", 0.3 * np.sin(2 * np.pi * 880 * t), 44100)  # 2 s : répétée
    Image.new("RGBA", (100, 50), (0, 255, 0, 255)).save(tmp_path / "logo.png")
    return tmp_path, a, b


def test_fondus_enchaines_musique_titre_logo(morceaux):
    d, a, b = morceaux
    msg, video, fichiers, dossier = _monter(
        [str(a), str(d / "image.png"), str(b)], transition=mt.ENCHAINE, duree_transition=0.5, duree_image=2,
        musique=str(d / "musique.wav"), volume_musique=40, titre="Mon titre", duree_titre=2,
        logo=str(d / "logo.png"), nom="essai")
    assert "3 morceau" in msg
    images, codec, fps = _images(video)
    assert codec == "h264" and fps == 12 and len(images) == 72  # 3 + 2 + 2 s − 2 fondus de 0,5 s
    assert images[0].shape == (240, 320, 3)
    assert images[0].max() < 30  # part du noir
    assert images[36][120, 160, 0] > 240 and images[36][120, 160, 1] < 20  # l'image rouge, pleine
    assert 60 < images[33][120, 40, 0] < 250 and images[33][120, 40, 1] > 40  # mélange vidéo grise + rouge
    assert images[60][120, 20].max() < 20 and images[60][120, 160].max() > 150  # portrait : bandes noires
    assert images[40][20, 290, 1] > 200 and images[40][20, 290, 0] < 40  # logo vert en haut à droite
    titre = images[12][100:140, 100:220].astype(int)
    assert (titre.min(axis=2) > 200).sum() > 50  # texte blanc au centre pendant le titre
    assert (images[40][100:140, 100:220].min(axis=2) > 254).sum() == 0  # plus de titre ensuite

    y, sr = _son(video)
    assert abs(len(y) / sr - 6) < 0.1
    assert _frequence(y[int(0.5 * sr):int(2 * sr)], sr) == pytest.approx(220, abs=3)  # son de la vidéo a
    musique = y[int(3.2 * sr):int(3.8 * sr)]  # pendant l'image fixe : la musique seule, à pleine valeur
    assert _frequence(musique, sr) == pytest.approx(880, abs=3)
    fin = y[int(4.6 * sr):int(5.4 * sr)]  # vidéo b : 440 Hz devant la musique baissée
    spectre = np.abs(np.fft.rfft(fin))
    f = np.fft.rfftfreq(len(fin), 1 / sr)
    assert spectre[np.abs(f - 440).argmin()] > 5 * spectre[np.abs(f - 880).argmin()]

    infos = json.loads((mt.Path(dossier) / "creation.json").read_text(encoding="utf-8"))
    assert infos["type"] == "video" and infos["montage"]["morceaux"] == ["a.mp4", "image.png", "b.mp4"]
    assert fichiers == [video]


def test_fondu_au_noir_sans_chevauchement_et_portrait(morceaux):
    d, a, b = morceaux
    _, video, _, _ = _monter([str(a), str(b)], transition=mt.NOIR, duree_transition=1, fondu_debut_fin=False,
                             taille="Portrait 1080×1920 (TikTok, Shorts, Reels)", cadrage=mt.REMPLIR)
    images, _, fps = _images(video)
    assert len(images) == 60 and images[0].shape == (1920, 1080, 3)  # 3 + 2 s, sans chevauchement
    assert images[35].max() < 30 and images[36].max() < 30  # autour du raccord : noir
    assert images[10][960, 100].max() > 55  # pas de fondu au début : gris de l'image 10 de a (70)
    assert images[50][960, 20].max() > 100 and images[50][960, 1060].max() > 180  # b recadrée : pas de bande noire


def test_erreurs(env, tmp_path):
    with pytest.raises(gr.Error, match="Ajoute des vidéos"):
        _monter([])
    sf.write(tmp_path / "son.wav", np.zeros(44100), 44100)
    with pytest.raises(gr.Error, match="musique de fond"):
        _monter([str(tmp_path / "son.wav")])


def test_images_seules_sans_son(env, tmp_path):
    for i, c in enumerate([(0, 0, 255), (0, 255, 0)]):
        Image.new("RGB", (400, 300), c).save(tmp_path / f"{i}.jpg")
    _, video, _, _ = _monter([str(tmp_path / "0.jpg"), str(tmp_path / "1.jpg")], duree_image=1.5,
                             transition=mt.COUPE, fondu_debut_fin=False)
    images, _, fps = _images(video)
    assert fps == 30 and len(images) == 90 and images[0].shape == (300, 400, 3)
    assert images[10][150, 200, 2] > 240 and images[80][150, 200, 1] > 240
    assert _son(video)[0] is None
