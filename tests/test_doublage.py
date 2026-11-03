"""Traduire et doubler : traduction par listes (faux Qwen), voix clonée depuis la vidéo ou voix de la bibliothèque
(faux Chatterbox en mode liste), phrases posées à leur moment, accélérées ou décalées, vidéo réécrite (vrai PyAV)."""
import json
import sys
import textwrap
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import soundfile as sf

from conftest import no_progress, write_tone
from studiovoix import config as cfg
from studiovoix import doublage as db
from test_diffusion import _journal, faux_diffusion  # noqa: F401 - fixture

FAUX_CHATTERBOX = textwrap.dedent('''
    import json, sys
    import numpy as np, soundfile as sf
    t = json.load(open(sys.argv[1], encoding="utf-8"))
    open("vu.json", "w", encoding="utf-8").write(json.dumps(t))
    for i, (texte, sortie) in enumerate(zip(t["textes"], t["sorties"]), 1):
        print(f"PROGRESSION {i}/{len(t['textes'])}", flush=True)
        n = int(24000 * 0.05 * len(texte))  # 50 ms par caractère
        sf.write(sortie, (0.3 * np.sin(2 * np.pi * 660 * np.arange(n) / 24000)).astype("float32"), 24000)
''')


@pytest.fixture
def faux_doublage(faux_diffusion, monkeypatch, tmp_path):  # noqa: F811
    from studiovoix import demucs

    (cfg.MOTEURS_DIR / "chatterbox_tts.py").write_text(FAUX_CHATTERBOX, encoding="utf-8")
    monkeypatch.setattr(cfg, "CHATTERBOX_PYTHON", sys.executable)
    monkeypatch.setattr(cfg, "CHATTERBOX_DIR", tmp_path / "StudioVoix" / "chatterbox")

    def separer(song, workdir):  # voix à 220 Hz sur 2,5 s puis silence, fond à 110 Hz
        y, sr = sf.read(song)
        t = np.arange(len(y)) / sr
        voix = np.where(t < 2.5, 0.4 * np.sin(2 * np.pi * 220 * t), 0)
        sf.write(Path(workdir) / "vocals.wav", voix, sr)
        sf.write(Path(workdir) / "no_vocals.wav", 0.2 * np.sin(2 * np.pi * 110 * t), sr)
        return Path(workdir) / "vocals.wav", Path(workdir) / "no_vocals.wav"

    monkeypatch.setattr(demucs, "separate_vocals", separer)
    return tmp_path


def test_placer_accelere_puis_decale():
    segs = [{"debut": 0.0, "fin": 1.0}, {"debut": 1.0, "fin": 2.0}, {"debut": 4.0, "fin": 5.0}]
    voix = [np.ones(int(db.SR * 1.2), np.float32), np.ones(int(db.SR * 4.5), np.float32),
            np.ones(int(db.SR * 0.5), np.float32)]
    piste, pos, accelerees, decalees = db.placer(segs, voix, 6.0)
    assert accelerees == 2 and decalees == 1
    assert pos[0] == (0.0, pytest.approx(0.92, abs=0.01))  # 1,2 s ramenées à la place libre (0,92 s)
    assert pos[1][1] == pytest.approx(1.0 + 4.5 / db.ACCELERATION_MAX, abs=0.01)  # accélérée au plus ×1,3…
    assert pos[2][0] == pytest.approx(pos[1][1] + db.ECART_MIN)  # … puis la suivante décalée
    assert len(piste) / db.SR >= pos[2][1]
    _, pos, accelerees, decalees = db.placer(segs, voix, 6.0, accelerer=False)
    assert accelerees == 0 and decalees == 2 and pos[1][0] == pytest.approx(1.2 + db.ECART_MIN)


def test_doubler_avec_sa_propre_voix(faux_doublage):
    from test_convertir import _clip, _frequence, _son

    video = _clip(faux_doublage / "interview.mp4", 3, 12, (320, 240), 220)
    msg, sortie, texte, fichiers, dossier = db.doubler(str(video), None, "Français", "Anglais", progress=no_progress)
    whisper = [j for j in _journal() if j["action"] == "transcrire"][-1]["tache"]
    assert whisper["langue"] == "fr"
    traduction = [j for j in _journal() if j["action"] == "decrire"][-1]["tache"]
    assert traduction["mode"] == "traduction" and traduction["langue_cible"] == "English" and traduction["nombre"] == 2
    vu = json.loads((Path(dossier) / "phrases" / "vu.json").read_text(encoding="utf-8"))
    assert vu["langue"] == "en" and vu["textes"][0] == "[English] Il était une fois"
    assert Path(vu["voix"]).parent == Path(dossier)  # la voix de la personne, prise dans la vidéo (Demucs)
    assert "Il était une fois\n→ [English] Il était une fois" in texte
    assert Path(sortie).name == "video_en.mp4" and "2 phrases" in msg and "figée" in msg
    srt = (Path(dossier) / "sous_titres_en.srt").read_text(encoding="utf-8-sig")
    assert "[English] Bonjour à tous" in srt
    import av

    with av.open(sortie) as c:
        duree = float(c.duration / av.time_base)
    assert duree > 3.5  # la voix doublée dépasse : dernière image figée
    y, sr = _son(sortie)
    assert _frequence(y[int(0.2 * sr):int(1.0 * sr)], sr) == pytest.approx(660, abs=5)  # voix doublée
    infos = json.loads((Path(dossier) / "creation.json").read_text(encoding="utf-8"))
    assert infos["type"] == "video" and infos["doublage"]["langue"] == "Anglais"


def test_doubler_un_son_avec_une_voix_de_la_bibliotheque(faux_doublage):
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=8)
    son = faux_doublage / "message.wav"
    sf.write(son, 0.3 * np.sin(2 * np.pi * 220 * np.arange(44100 * 3) / 44100), 44100)
    msg, sortie, texte, fichiers, dossier = db.doubler(str(son), None, db.LANGUE_AUTO, "Espagnol", "moi",
                                                       garder_fond=False, progress=no_progress)
    vu = json.loads((Path(dossier) / "phrases" / "vu.json").read_text(encoding="utf-8"))
    assert vu["voix"] == str(cfg.VOICES_DIR / "moi.wav") and vu["langue"] == "es"
    assert sortie is None and Path(fichiers[0]).name == "voix_es.wav"
    assert json.loads((Path(dossier) / "creation.json").read_text(encoding="utf-8"))["type"] == "tts"


def test_traduction_incomplete_refaite_phrase_par_phrase(faux_diffusion):  # noqa: F811
    traductions, gardees = db.traduire(["Bonjour", "MANQUE ici", "Au revoir"], "de")
    assert traductions == ["[German] Bonjour", "[German] MANQUE ici", "[German] Au revoir"] and gardees == 0


def test_erreurs(faux_doublage):
    with pytest.raises(gr.Error, match="Ajoute une vidéo"):
        db.doubler(None, None, progress=no_progress)
    muet = faux_doublage / "muet.mp4"
    from test_convertir import _clip
    _clip(muet, 1, 12, avec_son=False)
    with pytest.raises(gr.Error, match="pas de son"):
        db.doubler(str(muet), None, progress=no_progress)
