"""Moteur de diffusion (faux moteur en sous-processus) : bruitages, descriptions, jeton, état des modèles."""
import importlib.util
import json
import sys
import textwrap

import gradio as gr
import pytest
import soundfile as sf

from conftest import no_progress
from studiovoix import bruitages, config as cfg, diffusion
from studiovoix.modeles import models_status_md

_spec = importlib.util.spec_from_file_location("diffusion_moteur", cfg.MOTEURS_DIR / "diffusion.py")
moteur = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moteur)  # importable sans torch

FAUX_MOTEUR = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    import numpy as np, soundfile as sf
    action = sys.argv[1]
    journal = Path(os.environ["FAUX_JOURNAL"])
    if action == "telecharger":
        journal.open("a").write(json.dumps({"action": action, "args": sys.argv[2:]}) + chr(10))
        print("Modèles prêts.", flush=True); sys.exit(0)
    t = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    journal.open("a").write(json.dumps({"action": action, "tache": t, "cwd": os.getcwd(),
                                        "u2net": os.environ.get("U2NET_HOME")}) + chr(10))
    if action == "decrire":
        print("PROGRESSION 1/2 x", flush=True); print("PROGRESSION 2/2 y", flush=True)
        if t.get("image"): texte = "metal clink on a table"
        else: texte = "wooden door creaking then slamming" if t["mode"] == "bruitage" else "a red potion bottle"
        print("RESULTAT " + json.dumps({"texte": texte}), flush=True)
    elif action == "bruitage":
        if "ERREUR" in t["prompt"]:
            print("ERREUR : mémoire de la carte graphique insuffisante.", flush=True); sys.exit(3)
        print("PROGRESSION 1/2 x", flush=True); print("PROGRESSION 2/2 y", flush=True)
        d = Path(t["dossier"]); d.mkdir(parents=True, exist_ok=True); fichiers = []
        for i in range(1, t["variantes"] + 1):
            f = d / f"variante_{i}.wav"; sf.write(str(f), np.zeros((int(44100 * t["duree"]), 2), "float32"), 44100)
            fichiers.append(str(f))
        graines = [t["graine"] or 111] + [222, 333][: t["variantes"] - 1]
        print("RESULTAT " + json.dumps({"fichiers": fichiers, "graines": graines, "frequence": 44100}), flush=True)
    print("TERMINE -", flush=True)
""")


@pytest.fixture
def faux_diffusion(env, monkeypatch):
    moteurs = env / "moteurs"
    moteurs.mkdir(exist_ok=True)
    (moteurs / "diffusion.py").write_text(FAUX_MOTEUR, encoding="utf-8")
    monkeypatch.setattr(cfg, "MOTEURS_DIR", moteurs)
    monkeypatch.setattr(cfg, "DIFFUSION_PYTHON", sys.executable)
    monkeypatch.setattr(cfg, "DIFFUSION_DIR", env / "StudioVoix" / "diffusion")
    cfg.DIFFUSION_DIR.mkdir(parents=True)
    monkeypatch.setattr(cfg, "SFX_DIR", env / "data" / "bruitages")
    monkeypatch.setenv("HF_HOME", str(env / "StudioVoix" / "hf-home"))
    monkeypatch.delenv("U2NET_HOME", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("FAUX_JOURNAL", str(env / "journal.jsonl"))
    for nom in diffusion.MODELES:  # modèles « présents » dans le cache
        for f in diffusion.MODELES[nom][2]:
            p = diffusion.ckpt_dir(nom) / "snapshots" / "abc" / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"0")
    return env


def _journal():
    return [json.loads(l_) for l_ in (cfg.DATA_DIR.parent / "journal.jsonl").read_text().splitlines()]


def test_consignes_du_vrai_moteur():
    assert set(moteur.CONSIGNES) == {"son", "objet", "bruitage", "image"}
    assert set(moteur.MODELES) == set(diffusion.MODELES)  # mêmes noms côté application et côté moteur


def test_preparer_le_prompt(faux_diffusion):
    assert bruitages.preparer("une porte qui grince", None, progress=no_progress) == "wooden door creaking then slamming"
    im = faux_diffusion / "scene.png"
    im.write_bytes(b"png")
    assert bruitages.preparer("", str(im), progress=no_progress) == "metal clink on a table"
    j = _journal()
    assert j[0]["tache"] == {"mode": "bruitage", "texte": "une porte qui grince"}
    assert j[1]["tache"] == {"mode": "son", "image": str(im)} and j[1]["cwd"] == str(cfg.DIFFUSION_DIR)
    assert j[1]["u2net"] == str(cfg.DIFFUSION_DIR / "u2net")
    with pytest.raises(gr.Error, match="Décris"):
        bruitages.preparer("  ", None, progress=no_progress)


def test_generer_bruitage(faux_diffusion):
    etapes = []
    msg, liste, premiere, dossier = bruitages.generer("door creak", "Porte!", 2.5, 3, 42, 50, None, "une porte",
                                                      progress=lambda p, desc="": etapes.append(desc))
    d = next(cfg.SFX_DIR.iterdir())
    assert dossier == str(d) and premiere == str(d / "variante_1.wav") and len(liste["choices"]) == 3
    assert sf.info(premiere).duration == pytest.approx(2.5) and "Graines : 42, 222, 333" in msg
    infos = json.loads((d / "creation.json").read_text(encoding="utf-8"))
    assert infos["type"] == "bruitage" and infos["nom"] == "Porte" and infos["description_fr"] == "une porte"
    assert [v["graine"] for v in infos["versions"]] == [42, 222, 333]
    t = _journal()[0]["tache"]
    assert t["negatif"] == bruitages.NEGATIF and t["etapes"] == 50 and t["duree"] == 2.5
    assert any("génération du bruitage" in e for e in etapes)


def test_erreurs(faux_diffusion, monkeypatch):
    with pytest.raises(gr.Error, match="prompt"):
        bruitages.generer("", "x", 3, 1, 0, 100, progress=no_progress)
    with pytest.raises(gr.Error, match="Stable Audio : mémoire"):
        bruitages.generer("ERREUR", "x", 3, 1, 0, 100, progress=no_progress)
    (diffusion.ckpt_dir("bruitages") / "snapshots" / "abc" / "vae" / "diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(gr.Error, match="jeton"):
        bruitages.generer("x", "x", 3, 1, 0, 100, progress=no_progress)
    assert "Stable Audio Open 1.0 (bruitages)" in diffusion.missing_components()
    monkeypatch.setattr(cfg, "DIFFUSION_PYTHON", str(faux_diffusion / "absent.exe"))
    with pytest.raises(gr.Error, match="INSTALLER.bat"):
        bruitages.preparer("x", None, progress=no_progress)


def test_jeton_et_telechargement(faux_diffusion):
    assert not diffusion.jeton_present()
    with pytest.raises(gr.Error, match="hf_"):
        diffusion.enregistrer_jeton("abc")
    assert "enregistré" in diffusion.enregistrer_jeton("  hf_abcdefghijklmnopqrstuvwxyz ")
    assert (diffusion.hf_home() / "token").read_text() == "hf_abcdefghijklmnopqrstuvwxyz" and diffusion.jeton_present()
    (diffusion.hf_home() / "token").unlink()
    assert "❌" in list(diffusion.download(["bruitages"]))[0]  # sans jeton, pas de téléchargement lancé
    dernier = list(diffusion.download(["qwen", "image"]))[-1]
    assert "Terminé" in dernier and _journal()[-1]["args"] == ["qwen", "image", "detourage"]


def test_etat_des_modeles(faux_diffusion):
    assert "| Diffusion : Qwen3-VL, Stable Audio Open, SDXL, Hunyuan3D-2 | ✅ présent |" in models_status_md()


def test_galerie_bruitage(faux_diffusion):
    from studiovoix import galerie

    bruitages.generer("door creak", "Porte", 2, 2, 42, 50, None, "une porte", progress=no_progress)
    (lib, dossier), = galerie.lister("Bruitages")
    assert "🔊 Bruitage · Porte — une porte" in lib
    md, fichier, versions, *_ = galerie.details(dossier, 2)
    assert "Demande : une porte (2.0 s)" in md and fichier.endswith("variante_2.wav") and versions["visible"]
    msg, nouveau = galerie.recreer(dossier, 2, progress=no_progress)
    assert "graine 222" in msg and _journal()[-1]["tache"]["graine"] == 222 and _journal()[-1]["tache"]["variantes"] == 1
    with pytest.raises(gr.Error, match="chansons et les pistes"):
        galerie.refaire_passage(dossier, 1, 0, 1, "", "", "", progress=no_progress)
