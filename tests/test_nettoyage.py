"""Nettoyage de voix : client (faux moteur en sous-processus), choix de la version enregistrée, vrai script."""
import importlib.util
import json
import sys
import textwrap

import gradio as gr
import pytest
import soundfile as sf

from conftest import no_progress, write_tone
from studiovoix import config as cfg
from studiovoix import nettoyage
from studiovoix.modeles import models_status_md
from studiovoix.voix import GARDER_NETTOYEE, GARDER_ORIGINAL, list_voices, save_voice_choix

_spec = importlib.util.spec_from_file_location("nettoyage_voix", cfg.MOTEURS_DIR / "nettoyage_voix.py")
moteur = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moteur)  # importable sans torch : les imports lourds sont dans les fonctions

FAUX_MOTEUR = textwrap.dedent('''
    import json, os, sys
    import numpy as np, soundfile as sf
    t = json.load(open(sys.argv[1], encoding="utf-8"))
    open(os.path.join(os.path.dirname(sys.argv[1]), "vu.json"), "w").write(json.dumps({"tache": t, "cwd": os.getcwd()}))
    if "erreur" in t["entree"]:
        print("ERREUR : mémoire de la carte graphique insuffisante.", flush=True); sys.exit(3)
    y, sr = sf.read(t["entree"])
    for i in (1, 2):
        print(f"PROGRESSION {i}/2 etape", flush=True)
    sf.write(t["sortie"], y * 0.5, 48000)  # MossFormer2 sort en 48 kHz
''')


@pytest.fixture
def faux_nettoyage(env, monkeypatch):
    moteurs = env / "moteurs"
    moteurs.mkdir()
    (moteurs / "nettoyage_voix.py").write_text(FAUX_MOTEUR, encoding="utf-8")
    monkeypatch.setattr(cfg, "MOTEURS_DIR", moteurs)
    monkeypatch.setattr(cfg, "NETTOYAGE_PYTHON", sys.executable)
    monkeypatch.setattr(cfg, "NETTOYAGE_DIR", env / "StudioVoix" / "nettoyage")
    monkeypatch.setattr(cfg, "CLEAN_DIR", env / "data" / "nettoyage")
    return env


def test_niveaux_coherents_avec_le_moteur():
    assert set(nettoyage.NIVEAUX.values()) == set(moteur.NIVEAUX)
    assert moteur.NIVEAUX["maximal"] == ["mossformer2", "voicefixer"]


def test_nettoyage(faux_nettoyage):
    src = write_tone(faux_nettoyage / "ma voix.mp3.wav", seconds=12)
    etapes = []
    sortie, garder, msg = nettoyage.nettoyer(str(src), "Maximal — les deux à la suite",
                                             progress=lambda p, desc="": etapes.append(desc))
    d = next(cfg.CLEAN_DIR.iterdir())
    vu = json.loads((d / "vu.json").read_text())
    assert vu["tache"] == {"entree": str(d / "original.wav"), "sortie": str(d / "voix_nettoyee.wav"), "niveau": "maximal"}
    assert vu["cwd"] == str(cfg.NETTOYAGE_DIR)  # les modèles se rangent dans StudioVoix\\nettoyage
    assert sortie == str(d / "voix_nettoyee.wav") and garder["value"] == GARDER_NETTOYEE
    assert "Nettoyage de la voix : étape 2/2…" in etapes and "Léger" in msg


def test_erreurs(faux_nettoyage, monkeypatch):
    with pytest.raises(gr.Error, match="Importe ou enregistre"):
        nettoyage.nettoyer(None, list(nettoyage.NIVEAUX)[0], progress=no_progress)
    src = write_tone(faux_nettoyage / "erreur.wav", seconds=12)
    with pytest.raises(gr.Error, match="Nettoyage : mémoire"):
        nettoyage.nettoyer(str(src), list(nettoyage.NIVEAUX)[0], progress=no_progress)
    with pytest.raises(gr.Error, match="inconnu"):
        nettoyage.nettoyer(str(src), "Extrême", progress=no_progress)
    monkeypatch.setattr(cfg, "NETTOYAGE_PYTHON", str(faux_nettoyage / "absent" / "python.exe"))
    with pytest.raises(gr.Error, match="INSTALLER.bat"):
        nettoyage.nettoyer(str(src), list(nettoyage.NIVEAUX)[0], progress=no_progress)


def test_enregistrer_la_version_choisie(faux_nettoyage):
    original = write_tone(faux_nettoyage / "orig.wav", seconds=12, amp=0.3)
    nettoyee, _, _ = nettoyage.nettoyer(str(original), list(nettoyage.NIVEAUX)[0], progress=no_progress)
    with pytest.raises(gr.Error, match="Nettoyer"):
        save_voice_choix(str(original), None, GARDER_NETTOYEE, "x")
    save_voice_choix(str(original), nettoyee, GARDER_ORIGINAL, "brute")
    save_voice_choix(str(original), nettoyee, GARDER_NETTOYEE, "propre")
    assert list_voices() == ["brute", "propre"]
    assert sf.info(str(cfg.VOICES_DIR / "propre.wav")).samplerate == 44100  # 48 kHz ramené à 44,1 kHz


def test_etat_des_modeles(faux_nettoyage):
    assert "| Nettoyage de voix (MossFormer2, VoiceFixer) | ❌ absent (débruiteur MossFormer2" in models_status_md()
    for f in nettoyage.FICHIERS.values():
        p = cfg.NETTOYAGE_DIR / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"0")
    assert "| Nettoyage de voix (MossFormer2, VoiceFixer) | ✅ présent |" in models_status_md()


def test_reparation_telechargement_interrompu(tmp_path, monkeypatch):
    d = tmp_path / "checkpoints" / "MossFormer2_SE_48K"
    d.mkdir(parents=True)
    (d / "last_best_checkpoint").write_text("last_best_checkpoint.pt")
    monkeypatch.chdir(tmp_path)
    moteur._reparer_mossformer2()
    assert not (d / "last_best_checkpoint").exists()  # ClearerVoice retéléchargera le modèle
    (d / "last_best_checkpoint").write_text("x")
    (d / "last_best_checkpoint.pt").write_bytes(b"0")
    moteur._reparer_mossformer2()
    assert (d / "last_best_checkpoint").exists()


def test_dossier_personnel_redirige(tmp_path, monkeypatch):
    import os

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", os.environ.get("HOME", ""))  # restaurés après le test
    monkeypatch.setenv("USERPROFILE", os.environ.get("USERPROFILE", ""))
    moteur._rediriger_dossier_personnel()
    assert os.environ["USERPROFILE"] == os.environ["HOME"] == str(tmp_path / "voicefixer")
