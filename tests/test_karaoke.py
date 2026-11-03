"""Karaoké : alignement des paroles connues sur les mots entendus, lignes de Whisper, pages, LRC, et création
complète (faux Demucs, faux Whisper, vrai PyAV pour la vidéo)."""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import soundfile as sf

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import karaoke as ka
from test_diffusion import _journal, faux_diffusion  # noqa: F401 - fixture


def _mots(texte, debut=1.0, pas=0.5):
    return [{"debut": debut + pas * i, "fin": debut + pas * i + 0.4, "mot": m} for i, m in enumerate(texte.split())]


def test_paroles_et_alignement():
    lignes = ka.lignes_paroles("Couplet 1\nLe soleil se lève,\nSur la ville !\n\n[Chorus]\nLa la la\n")
    assert lignes == ["Le soleil se lève,", "Sur la ville !", "La la la"]
    assert ka.normal("L'Été,") == "lete" and ka.normal("Cœur") == "coeur"
    alignees, accord = ka.aligner(lignes, _mots("le solaire se lève sur la ville"), 10.0)
    assert [m["mot"] for m in alignees[0]] == ["Le", "soleil", "se", "lève,"]  # le texte des paroles est gardé
    assert alignees[0][0]["debut"] == 1.0 and alignees[0][2]["debut"] == 2.0
    # « soleil » pas reconnu : placé entre « Le » (fin 1,4) et « se » (début 2,0)
    assert 1.4 <= alignees[0][1]["debut"] < alignees[0][1]["fin"] <= 2.0
    # « La la la » jamais entendu : après le dernier mot, dans la chanson
    assert alignees[2][0]["debut"] >= alignees[1][-1]["fin"] and alignees[2][-1]["fin"] <= 10.0
    assert accord == pytest.approx(6 / 10)
    temps = [m["debut"] for l in alignees for m in l]
    assert temps == sorted(temps)


def test_lignes_sans_paroles_et_pages():
    mots = _mots("bonjour tout le monde.") + _mots("une autre phrase ici", debut=5.0)
    lignes = ka.lignes_entendues(mots)
    assert [" ".join(m["mot"] for m in l) for l in lignes] == ["bonjour tout le monde.", "une autre phrase ici"]
    p = ka.pages(lignes + lignes[:1], par_page=2)
    assert len(p) == 2 and p[0]["debut"] == 0.0 and p[0]["fin"] <= p[1]["debut"]
    assert ka._remplissage([(0, 10, 1.0, 2.0), (12, 20, 2.0, 3.0)], 1.5) == 5
    assert ka._remplissage([(0, 10, 1.0, 2.0), (12, 20, 2.0, 3.0)], 2.5) == 16
    assert ka._remplissage([(0, 10, 1.0, 2.0)], 0.5) == 0 and ka._remplissage([(0, 10, 1.0, 2.0)], 4) == 10
    texte = ka.lrc(lignes, "Ma chanson")
    assert texte.startswith("[ti:Ma chanson]\n[00:01.00]<00:01.00>bonjour <00:01.50>tout")
    assert "00:00:05,000 --> " in ka.srt(lignes)


def test_creer_un_karaoke(faux_diffusion, monkeypatch, tmp_path):  # noqa: F811
    """Chanson de la galerie (paroles reprises), voix isolée envoyée à Whisper en 16 kHz, musique sans voix dans la
    vidéo, mots allumés au bon moment."""
    from studiovoix import demucs

    t = np.arange(44100 * 6) / 44100
    chanson = tmp_path / "chanson.wav"
    sf.write(chanson, np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1), 44100)

    def separer(song, workdir):
        voix, instru = Path(workdir) / "voix.wav", Path(workdir) / "instru.wav"
        sf.write(voix, 0.3 * np.sin(2 * np.pi * 440 * t), 44100)
        sf.write(instru, 0.3 * np.sin(2 * np.pi * 110 * t), 44100)
        return voix, instru

    monkeypatch.setattr(demucs, "separate_vocals", separer)
    creation = cfg.SONGS_DIR / "20261103_120000"
    creation.mkdir(parents=True)
    (creation / "creation.json").write_text(json.dumps({
        "type": "chanson", "paroles": "Couplet\nLe soleil se lève\nSur la ville\nRefrain\nLa la la",
        "langue": "Français", "versions": [{"graine": 1, "dossier": ".", "fichier": str(chanson)}]}),
        encoding="utf-8")
    msg, video, fichiers, dossier = ka.creer(None, str(creation), "", None, titre="Le soleil", progress=no_progress)
    tache = _journal()[-1]["tache"]
    assert tache["mots"] is True and tache["langue"] == "fr"  # le faux moteur vérifie le WAV 16 kHz mono
    assert Path(dossier).name.endswith("_karaoke") and "karaoke_karaoke" not in dossier and "3 lignes" in msg and Path(video).name == "karaoke.mp4"
    assert [Path(f).name for f in fichiers] == ["karaoke.mp4", "paroles.lrc", "paroles.srt", "sans_voix.wav"]
    lrc = (Path(dossier) / "paroles.lrc").read_text(encoding="utf-8")
    assert "<00:01.00>Le" in lrc and "soleil" in lrc

    import av

    from test_convertir import _frequence, _son
    with av.open(video) as c:
        images = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    assert len(images) == 150 and images[0].shape == (720, 1280, 3)
    y, sr = _son(video)
    assert _frequence(y[sr:3 * sr], sr) == pytest.approx(110, abs=3)  # la musique sans la voix
    jaune = lambda im: int(((im[..., 0] > 200) & (im[..., 1] > 170) & (im[..., 2] < 120)).sum())  # noqa: E731
    assert jaune(images[int(0.9 * 25)]) == 0  # avant le premier mot : rien d'allumé
    assert 0 < jaune(images[int(1.6 * 25)]) < jaune(images[int(2.9 * 25)])  # les mots s'allument au fil du chant
    infos = json.loads((Path(dossier) / "creation.json").read_text(encoding="utf-8"))
    assert infos["type"] == "video" and infos["karaoke"]["paroles_connues"] is True
    with pytest.raises(gr.Error, match="Ajoute une chanson"):
        ka.creer(None, None, progress=no_progress)
