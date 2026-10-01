"""RVC : jeu d'enregistrements, entraînement et conversion (faux moteur), intégration chanson et lecture."""
import importlib.util
import json
import sys
import textwrap

import gradio as gr
import numpy as np
import pytest
import soundfile as sf

from conftest import no_progress, write_tone
from studiovoix import config as cfg
from studiovoix import rvc
from studiovoix.pipeline import creer_chanson

_spec = importlib.util.spec_from_file_location("rvc_voix", cfg.MOTEURS_DIR / "rvc_voix.py")
moteur = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moteur)  # importable sans torch

FAUX_MOTEUR = textwrap.dedent("""
    import json, os, shutil, sys
    from pathlib import Path
    action, tache = sys.argv[1], json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    with open("appels.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"action": action, "tache": tache, "cwd": os.getcwd()}) + chr(10))
    if action == "entrainer":
        exp = Path("logs") / tache["nom"]; exp.mkdir(parents=True, exist_ok=True)
        n = tache["epoques"] + 3
        for i in range(1, n + 1):
            print(f"PROGRESSION {i}/{n} étape", flush=True)
        (exp / f"{tache['nom']}_{tache['epoques']}e_99s.pth").write_bytes(b"0")
        (exp / f"{tache['nom']}.index").write_bytes(b"0")
        print(f"TERMINE {exp}", flush=True)
    else:
        shutil.copy(tache["entree"], tache["sortie"])
""")


@pytest.fixture
def faux_rvc(env, monkeypatch):
    moteurs = env / "moteurs"
    moteurs.mkdir(exist_ok=True)
    (moteurs / "rvc_voix.py").write_text(FAUX_MOTEUR, encoding="utf-8")
    monkeypatch.setattr(cfg, "MOTEURS_DIR", moteurs)
    monkeypatch.setattr(cfg, "RVC_PYTHON", sys.executable)
    monkeypatch.setattr(cfg, "RVC_DIR", env / "StudioVoix" / "rvc")
    for f in rvc.MODELES_DE_BASE.values():
        (cfg.RVC_DIR / f).parent.mkdir(parents=True, exist_ok=True)
        (cfg.RVC_DIR / f).write_bytes(b"0")
    return env


def _appels():
    return [json.loads(l_) for l_ in (cfg.RVC_DIR / "appels.jsonl").read_text().splitlines()]


def test_meilleur_modele_du_vrai_moteur(tmp_path):
    exp = tmp_path / "ma voix"
    exp.mkdir()
    for f in ("ma voix_10e_100s.pth", "ma voix_300e_3000s.pth", "ma voix_30e_300s.pth", "G_2333333.pth"):
        (exp / f).write_bytes(b"0")
    assert moteur.meilleur_modele(exp, "ma voix").endswith("ma voix_300e_3000s.pth")
    assert moteur.meilleur_modele(tmp_path, "autre") is None


def test_liste_des_modeles(faux_rvc):
    exp = cfg.RVC_DIR / "logs" / "moi"
    exp.mkdir(parents=True)
    (exp / "moi_100e_10s.pth").write_bytes(b"0")
    assert rvc.modeles() == []  # pas d'index : modèle pas encore prêt
    (exp / "moi.index").write_bytes(b"0")
    (exp / "moi_300e_30s.pth").write_bytes(b"0")
    assert rvc.modeles() == [{"nom": "moi", "pth": str(exp / "moi_300e_30s.pth"), "index": str(exp / "moi.index"),
                              "epoques": 300}]
    assert rvc.choix_modeles() == [("moi (300 époques)", "moi")] and "| moi | 300 |" in rvc.tableau_modeles()


def test_controle_des_enregistrements(faux_rvc, tmp_path):
    long_ = write_tone(tmp_path / "long.wav", seconds=70)
    faible = write_tone(tmp_path / "faible.wav", seconds=20, amp=0.02)
    (tmp_path / "notes.txt").write_text("x")
    md = rvc.analyser_enregistrements([str(long_), str(faible), str(tmp_path / "notes.txt")], [])
    assert "Durée totale : 1.5 min" in md and "10 à 30 minutes" in md
    assert "| faible.wav | 20 s | ⚠️ Volume faible" in md and "format non pris en charge" in md


def test_entrainement(faux_rvc, tmp_path):
    a = tmp_path / "chant.flac"
    sf.write(str(a), (0.3 * np.sin(np.arange(40 * 44100) / 20)).astype("float32"), 44100, format="FLAC")
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=30)
    etapes = []
    msg = rvc.entrainer(" Ma voix! ", [str(a)], ["moi"], list(rvc.DUREES)[0], 6,
                        progress=lambda p, desc="": etapes.append(desc))
    (appel,) = _appels()
    t = appel["tache"]
    assert appel["action"] == "entrainer" and appel["cwd"] == str(cfg.RVC_DIR)
    assert t["nom"] == "Ma voix" and t["epoques"] == 100 and t["lot"] == 6 and t["frequence"] == 40000
    jeu = sorted(p.name for p in (cfg.RVC_DIR / "datasets" / "Ma voix").iterdir())
    assert jeu == ["001.wav", "002.wav"]  # FLAC converti en WAV + voix de la bibliothèque
    assert "Époque 1/100" in " ".join(etapes) and "Index" in etapes[-1]
    assert "Modèle « Ma voix » entraîné (100 époques)" in msg and "ajoute-en" in msg
    assert rvc.choix_modeles() == [("Ma voix (100 époques)", "Ma voix")]


def test_entrainement_refuse(faux_rvc, tmp_path, monkeypatch):
    court = write_tone(tmp_path / "court.wav", seconds=20)
    with pytest.raises(gr.Error, match="au moins 1 minute"):
        rvc.entrainer("x", [str(court)], [], list(rvc.DUREES)[0], 8, progress=no_progress)
    with pytest.raises(gr.Error, match="nom"):
        rvc.entrainer("///", [str(court)], [], list(rvc.DUREES)[0], 8, progress=no_progress)
    (cfg.RVC_DIR / rvc.MODELES_DE_BASE["détecteur de hauteur (RMVPE)"]).unlink()
    with pytest.raises(gr.Error, match="Modèles de base"):
        rvc.entrainer("x", [str(court)], [], list(rvc.DUREES)[0], 8, progress=no_progress)
    monkeypatch.setattr(cfg, "RVC_PYTHON", str(tmp_path / "absent" / "python.exe"))
    with pytest.raises(gr.Error, match="INSTALLER.bat"):
        rvc.entrainer("x", [str(court)], [], list(rvc.DUREES)[0], 8, progress=no_progress)


def _modele(nom="moi"):
    exp = cfg.RVC_DIR / "logs" / nom
    exp.mkdir(parents=True)
    (exp / f"{nom}_300e_30s.pth").write_bytes(b"0")
    (exp / f"{nom}.index").write_bytes(b"0")


def test_chanson_convertie_avec_rvc(fake_acestep, fake_engines, faux_rvc):
    fake_acestep()
    _modele()
    final, brute, conv, instru, msg, _ = creer_chanson(
        None, "", "", "", "", "", "Automatique", "[Verse]\nla", "Français", 60, 0, False, 5, 40, 1.0, 1.0,
        description="pop", moteur="rvc:moi", progress=no_progress)
    (appel,) = _appels()
    t = appel["tache"]
    assert appel["action"] == "convertir" and t["demi_tons"] == 5 and t["modele"].endswith("moi_300e_30s.pth")
    assert t["entree"].endswith("vocals.wav") and t["sortie"].endswith("voix_convertie_rvc.wav")
    assert not (cfg.SEEDVC_DIR / "appel.json").exists()  # Seed-VC pas utilisé
    workdir = cfg.SONGS_DIR / next(cfg.SONGS_DIR.iterdir()).name
    assert json.loads((workdir / "creation.json").read_text())["conversion"] == "rvc:moi"
    with pytest.raises(gr.Error, match="Modèle RVC introuvable"):
        creer_chanson(None, "", "", "", "", "", "Automatique", "[Verse]\nla", "Français", 60, 0, False, 0, 40,
                      1.0, 1.0, description="pop", moteur="rvc:absent", progress=no_progress)


def test_lecture_puis_rvc(faux_rvc, monkeypatch):
    from studiovoix import chatterbox
    from studiovoix.onglets import synthese as interface

    _modele()
    sortie_tts = cfg.TTS_DIR / "20260101_000000"
    sortie_tts.mkdir(parents=True)

    def fausse_synthese(voix, texte, *a, progress=None):
        f = write_tone(sortie_tts / "parole.wav", seconds=3)
        chatterbox.ecrire_creation(sortie_tts, {"type": "tts", "texte": texte,
                                                          "versions": [{"graine": 0, "fichier": str(f)}]})
        return str(f), "Terminé."

    monkeypatch.setattr(chatterbox, "synthese", fausse_synthese)
    fichier, msg = interface.synthese_puis_rvc("moi", "Bonjour", "Français", 0.5, 0.5, 0.8, 0, "moi", -2,
                                               progress=no_progress)
    assert fichier.endswith("parole_rvc.wav") and "modèle RVC « moi »" in msg
    infos = json.loads((sortie_tts / "creation.json").read_text())
    assert infos["rvc"] == "moi" and infos["versions"][0]["fichier"] == fichier
    assert _appels()[0]["tache"]["demi_tons"] == -2
    assert interface.synthese_puis_rvc("moi", "x", "Français", 0.5, 0.5, 0.8, 0, None, 0,
                                       progress=no_progress)[0].endswith("parole.wav")


def test_supprimer_modele(faux_rvc):
    _modele()
    assert rvc.supprimer_modele(None)[0] == "Suppression annulée."
    msg, tableau = rvc.supprimer_modele("moi")
    assert "supprimé" in msg and rvc.modeles() == [] and "Aucun modèle" in tableau


def test_config_applio_creee_comme_son_interface(tmp_path, monkeypatch):
    """Sans assets/config.json, Applio termine l'entraînement sans écrire le modèle final (vu en vrai)."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "config_template.json").write_text('{"precision": "fp16"}')
    monkeypatch.chdir(tmp_path)
    moteur._preparer_applio()
    assert (tmp_path / "assets" / "config.json").read_text() == '{"precision": "fp16"}'
    (tmp_path / "assets" / "config.json").write_text('{"precision": "bf16"}')
    moteur._preparer_applio()  # une configuration existante n'est jamais écrasée
    assert (tmp_path / "assets" / "config.json").read_text() == '{"precision": "bf16"}'
