"""Rigging 3D (UniRig) : faux moteur à la place de moteurs/rigging.py (pas de carte graphique ici)."""
import json
import sys
import textwrap
from pathlib import Path

import gradio as gr
import pytest

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import rigging

FAUX = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    t = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    Path(os.environ["FAUX_JOURNAL"]).write_text(json.dumps({"argv": sys.argv[1:], "tache": t, "cwd": os.getcwd(),
                                                            "hors_ligne": os.environ.get("HF_HUB_OFFLINE")}))
    assert Path(t["entree"]).exists()
    for i in range(1, 6): print(f"PROGRESSION {i}/5 x", flush=True)
    Path(t["sortie"]).write_bytes(b"glTF"); Path(t["fbx"]).write_bytes(b"FBX")
    print("RESULTAT " + json.dumps({"sortie": t["sortie"], "fbx": t["fbx"], "os": 42, "graine": t["graine"]}), flush=True)
    print("TERMINE " + t["sortie"], flush=True)
""")


@pytest.fixture
def faux_rigging(env, monkeypatch):
    moteurs = env / "moteurs"
    moteurs.mkdir(exist_ok=True)
    (moteurs / "rigging.py").write_text(FAUX, encoding="utf-8")
    monkeypatch.setattr(cfg, "MOTEURS_DIR", moteurs)
    monkeypatch.setattr(cfg, "RIGGING_PYTHON", sys.executable)
    monkeypatch.setattr(cfg, "UNIRIG_DIR", env / "StudioVoix" / "rigging" / "UniRig")
    cfg.UNIRIG_DIR.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(env / "hf"))
    monkeypatch.setenv("FAUX_JOURNAL", str(env / "journal.json"))
    for f in [*rigging.FICHIERS.values()]:
        p = rigging.ckpt_dir() / "snapshots" / "abc" / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"0")
    opt = env / "hf" / "hub" / "models--facebook--opt-350m" / "snapshots" / "abc" / "config.json"
    opt.parent.mkdir(parents=True)
    opt.write_text("{}")
    return env


def _journal(env):
    return json.loads((env / "journal.json").read_text())


def test_squelette_du_dernier_modele_puis_d_un_fichier_importe(faux_rigging):
    env = faux_rigging
    creation = cfg.MODELS3D_DIR / "20261020_120000"
    creation.mkdir(parents=True)
    (creation / "forme.glb").write_bytes(b"g")
    (creation / "modele.glb").write_bytes(b"g")
    (creation / "creation.json").write_text(json.dumps({"type": "3d", "nom": "dragon"}), encoding="utf-8")
    msg, glb, fichiers = rigging.animer(str(creation), None, 0, progress=no_progress)
    j = _journal(env)
    assert j["argv"][0] == "animer" and Path(j["cwd"]) == cfg.UNIRIG_DIR and j["hors_ligne"] == "1"
    assert Path(j["tache"]["entree"]).name == "modele.glb" and j["tache"]["graine"] == 12345  # texturé d'abord
    assert glb == str(creation / "modele_anime.glb") and len(fichiers) == 2 and "42 os" in msg
    assert json.loads((creation / "creation.json").read_text(encoding="utf-8"))["anime"] == "modele_anime.glb"
    importe = env / "Mon perso.fbx"
    importe.write_bytes(b"FBX")
    msg, glb, _ = rigging.animer(None, str(importe), 7, progress=no_progress)
    d = Path(glb).parent
    assert d.parent == cfg.MODELS3D_DIR / "rigging" and (d / "original.fbx").exists()
    assert _journal(env)["tache"]["graine"] == 7 and not (d / "creation.json").exists()  # hors galerie
    with pytest.raises(gr.Error, match="Format"):
        rigging.animer(None, str(env / "image.png"), 0, progress=no_progress)
    with pytest.raises(gr.Error, match="Crée d'abord"):
        rigging.animer(None, None, 0, progress=no_progress)


def test_modeles_absents(faux_rigging):
    for f in (rigging.ckpt_dir() / "snapshots").rglob("model.ckpt"):
        f.unlink()
    assert rigging.missing_components() == ["squelette", "poids de peau"]
    creation = cfg.MODELS3D_DIR / "x"
    creation.mkdir(parents=True)
    (creation / "forme.glb").write_bytes(b"g")
    with pytest.raises(gr.Error, match="UniRig"):
        rigging.animer(str(creation), None, 0, progress=no_progress)
