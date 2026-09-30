"""Synthèse vocale : découpage du texte (vrai script moteur) et client (faux moteur en sous-processus)."""
import importlib.util
import json
import sys
import textwrap

import gradio as gr
import pytest
import soundfile as sf

from conftest import no_progress, write_tone
from studiovoix import chatterbox
from studiovoix import config as cfg
from studiovoix.modeles import models_status_md

# Le script moteur est importable sans torch (imports lourds faits dans les fonctions)
_spec = importlib.util.spec_from_file_location("chatterbox_tts", cfg.MOTEURS_DIR / "chatterbox_tts.py")
moteur = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moteur)


# --- Découpage ----------------------------------------------------------------
def test_decoupage_regroupe_les_phrases():
    assert moteur.decouper("Bonjour.  Ça va ?\n Oui !") == ["Bonjour. Ça va ? Oui !"]
    assert moteur.decouper("  \n ") == []


def test_decoupage_respecte_la_limite_et_ne_perd_rien():
    texte = " ".join(f"Phrase numéro {i}, qui parle de choses et d'autres sans s'arrêter." for i in range(40))
    morceaux = moteur.decouper(texte, 120)
    assert all(len(m) <= 120 for m in morceaux)
    assert " ".join(morceaux) == texte
    assert all(m.endswith(".") for m in morceaux)  # coupe entre les phrases


def test_decoupage_phrase_trop_longue():
    longue = "un deux trois, " * 40 + "fin."
    morceaux = moteur.decouper(longue.strip(), 50)
    assert all(len(m) <= 50 for m in morceaux) and " ".join(morceaux) == longue.strip()
    mot = "a" * 130
    assert moteur.decouper(f"début {mot} fin", 50) == ["début", "a" * 50, "a" * 50, "a" * 30 + " fin"]


# --- Client --------------------------------------------------------------------------
FAUX_MOTEUR = textwrap.dedent('''
    import json, sys
    import numpy as np, soundfile as sf
    t = json.load(open(sys.argv[1], encoding="utf-8"))
    open("vu.json", "w", encoding="utf-8").write(json.dumps({"tache": t, "argv": sys.argv[1:]}))
    if "ERREUR" in t["texte"]:
        print("ERREUR : mémoire de la carte graphique insuffisante.", flush=True); sys.exit(3)
    if "PLANTE" in t["texte"]:
        print("Traceback: boum", flush=True); sys.exit(1)
    for i in (1, 2):
        print(f"PROGRESSION {i}/2", flush=True)
    sf.write(t["sortie"], np.zeros(24000, "float32"), 24000)
''')


@pytest.fixture
def faux_chatterbox(env, monkeypatch):
    moteurs = env / "moteurs"
    moteurs.mkdir()
    (moteurs / "chatterbox_tts.py").write_text(FAUX_MOTEUR, encoding="utf-8")
    monkeypatch.setattr(cfg, "MOTEURS_DIR", moteurs)
    monkeypatch.setattr(cfg, "CHATTERBOX_PYTHON", sys.executable)
    monkeypatch.setattr(cfg, "CHATTERBOX_DIR", env / "StudioVoix" / "chatterbox")
    monkeypatch.setattr(cfg, "TTS_DIR", env / "data" / "tts")
    monkeypatch.setenv("HF_HOME", str(env / "hf-home"))
    monkeypatch.delenv("PKUSEG_HOME", raising=False)
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=12)
    return env


def _dossier():
    return next(cfg.TTS_DIR.iterdir())


def test_synthese(faux_chatterbox):
    etapes = []
    sortie, msg = chatterbox.synthese("moi", " Bonjour à toi ! ", "Français", 0.7, 0.3, 0.8, 42,
                                      progress=lambda p, desc="": etapes.append(desc))
    d = _dossier()
    assert sortie == str(d / "parole.wav") and sf.info(sortie).samplerate == 24000 and "Terminé" in msg
    vu = json.loads((d / "vu.json").read_text(encoding="utf-8"))
    assert vu["tache"] == {"texte": "Bonjour à toi !", "langue": "fr", "voix": str(cfg.VOICES_DIR / "moi.wav"),
                           "sortie": sortie, "exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 0.8,
                           "graine": 42}
    assert (d / "texte.txt").read_text(encoding="utf-8") == "Bonjour à toi !\n"
    assert "Synthèse vocale : morceau 2/2…" in etapes


def test_environnement_du_sous_processus(faux_chatterbox, monkeypatch):
    env = chatterbox._env()
    assert env["PKUSEG_HOME"] == str(cfg.CHATTERBOX_DIR / "pkuseg")  # jamais ~/.pkuseg (C:)
    monkeypatch.setenv("PKUSEG_HOME", "X:/ailleurs")
    assert chatterbox._env()["PKUSEG_HOME"] == "X:/ailleurs"


def test_erreurs_du_moteur(faux_chatterbox):
    with pytest.raises(gr.Error, match="Chatterbox : mémoire de la carte graphique"):
        chatterbox.synthese("moi", "ERREUR", "Français", 0.5, 0.5, 0.8, 0, progress=no_progress)
    with pytest.raises(gr.Error, match="Traceback: boum"):
        chatterbox.synthese("moi", "PLANTE", "Français", 0.5, 0.5, 0.8, 0, progress=no_progress)


def test_saisies_refusees(faux_chatterbox, monkeypatch):
    with pytest.raises(gr.Error, match="introuvable"):
        chatterbox.synthese(None, "x", "Français", 0.5, 0.5, 0.8, 0, progress=no_progress)
    with pytest.raises(gr.Error, match="texte"):
        chatterbox.synthese("moi", "  ", "Français", 0.5, 0.5, 0.8, 0, progress=no_progress)
    with pytest.raises(gr.Error, match="trop long"):
        chatterbox.synthese("moi", "a" * 5001, "Français", 0.5, 0.5, 0.8, 0, progress=no_progress)
    monkeypatch.setattr(cfg, "CHATTERBOX_PYTHON", str(faux_chatterbox / "absent" / "python.exe"))
    with pytest.raises(gr.Error, match="INSTALLER.bat"):
        chatterbox.synthese("moi", "x", "Français", 0.5, 0.5, 0.8, 0, progress=no_progress)


def test_etat_des_modeles(faux_chatterbox):
    assert "| Chatterbox Multilingual V3 (synthèse vocale) | ❌ absent (modèle de texte V3" in models_status_md()
    snap = chatterbox.ckpt_dir() / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    for f in chatterbox.FICHIERS.values():
        (snap / f).write_bytes(b"0")
    assert chatterbox.missing_components() == []
    assert "| Chatterbox Multilingual V3 (synthèse vocale) | ✅ présent |" in models_status_md()
