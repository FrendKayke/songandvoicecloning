"""Modèle de feuilles de sprites : préparation des feuilles de marche, entraînement (faux kohya), feuille de marche
générée dans le kit de dresseur (faux SDXL)."""
import json
import sys
import textwrap
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import diffusion, dresseur, modele_sprites

GBA, DS = list(dresseur.STYLES)

FAUX_KOHYA = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    t = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    assert sys.argv[1] == "entrainer" and Path("sdxl_train_network.py").is_file()
    assert Path(t["sdxl"], "model_index.json").is_file() and list(Path(t["jeu"]).rglob("*.png"))
    print("PROGRESSION 1/2 chargement", flush=True)
    for i in (1, 2, 3):
        print(f"PROGRESSION {i}/3 entraînement", flush=True)
    s = Path(t["sortie"]) / (t["nom"] + ".safetensors")
    s.write_bytes(b"lora " + json.dumps({"epoques": t["epoques"], "dim": t["dim"], "graine": t["graine"],
                                          "taille": [t["largeur"], t["hauteur"]]}).encode())
    print("RESULTAT " + json.dumps({"lora": str(s), "duree_s": 3}), flush=True)
    print("TERMINE " + str(s), flush=True)
""")


def _feuille(k=2, fond=None, cols=4, rows=4, cw=16, ch=24, couleur=(200, 40, 40)):
    im = Image.new("RGBA", (cols * cw, rows * ch), fond or (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for r in range(rows):
        for c in range(cols):
            x, y = c * cw + (c % 3) - 1, r * ch
            d.rectangle((x + cw // 4, y + ch // 3, x + 3 * cw // 4 - 1, y + ch - 1), fill=couleur + (255,))
            d.ellipse((x + cw // 3, y + 1, x + 2 * cw // 3, y + ch // 3), fill=(240, 200, 160, 255))
    return im.resize((im.width * k, im.height * k), Image.Resampling.NEAREST)


@pytest.fixture
def faux_entrainement(env, monkeypatch, tmp_path):
    moteurs = env / "moteurs"
    moteurs.mkdir(exist_ok=True)
    (moteurs / "lora_sdxl.py").write_text(FAUX_KOHYA, encoding="utf-8")
    monkeypatch.setattr(cfg, "MOTEURS_DIR", moteurs)
    monkeypatch.setattr(cfg, "ENTRAINEMENT_PYTHON", sys.executable)
    sds = env / "StudioVoix" / "entrainement" / "sd-scripts"
    sds.mkdir(parents=True)
    (sds / "sdxl_train_network.py").write_text("# kohya", encoding="utf-8")
    monkeypatch.setattr(cfg, "SDSCRIPTS_DIR", sds)
    monkeypatch.setattr(cfg, "DIFFUSION_PYTHON", sys.executable)
    monkeypatch.setenv("HF_HOME", str(env / "StudioVoix" / "hf-home"))
    snap = diffusion.ckpt_dir("sprites_dresseur") / "snapshots" / "abc"
    snap.mkdir(parents=True)
    for f in diffusion.MODELES["sprites_dresseur"][2]:
        if isinstance(f, tuple):
            d = diffusion._dossier_depot(f[0]) / "snapshots" / "abc"
            d.mkdir(parents=True, exist_ok=True)
            (d / f[1]).write_bytes(b"0")
        else:
            (snap / f).parent.mkdir(parents=True, exist_ok=True)
            (snap / f).write_bytes(b"0")
    (snap / "model_index.json").write_text("{}")
    monkeypatch.setattr(diffusion, "present", lambda nom: False)
    feuilles = []
    for i in range(10):
        f = tmp_path / f"trchar{i:03d}.png"
        _feuille(2, fond=(120, 160, 200, 255) if i % 2 else None, couleur=(20 * i, 40, 200 - 10 * i)).save(f)
        feuilles.append(str(f))
    return feuilles


def test_preparation_des_feuilles(faux_entrainement, tmp_path, monkeypatch):
    feuilles = faux_entrainement
    with pytest.raises(gr.Error, match="au moins 8"):
        modele_sprites.preparer(feuilles[:3], "x", GBA, progress=no_progress)
    mauvaise = tmp_path / "mauvaise.png"
    _feuille(1, cols=3, rows=4, cw=32, ch=32).save(mauvaise)  # grille DS dans un modèle GBA
    legendes = []
    monkeypatch.setattr(diffusion, "present", lambda nom: nom == "qwen")
    monkeypatch.setattr(diffusion, "decrire", lambda mode, texte, image=None, progress=None:
                        legendes.append(mode) or "a boy with brown hair, red shirt.")
    msg, dossier = modele_sprites.preparer(feuilles + [str(mauvaise)], "Marche GBA !", GBA, progress=no_progress)
    d = Path(dossier)
    assert d == cfg.LORAS_DIR / "Marche_GBA" and "10 feuille(s)" in msg and "mauvaise.png" in msg
    jeu = d / "jeu" / f"{modele_sprites.REPETITIONS}_{modele_sprites.DECLENCHEUR}"
    images = sorted(jeu.glob("*.png"))
    assert len(images) == 10 and len(legendes) == 10 and legendes[0] == "feuille"
    im = Image.open(images[0])
    assert im.size == (64 * 8, 96 * 8) and im.mode == "RGB"
    a = np.array(im)
    assert tuple(a[0, 0]) == modele_sprites.FOND and (a[0, 0] == a[7, 7]).all()  # agrandi ×8 au plus proche
    assert images[0].with_suffix(".txt").read_text(encoding="utf-8") == (
        "pixelsheet, walking sprite sheet, 4 by 4 grid, a boy with brown hair, red shirt, on a magenta background")
    infos = json.loads((d / "modele.json").read_text(encoding="utf-8"))
    assert infos["grille"] == [4, 4] and infos["cellule"] == [16, 24] and len(infos["erreurs"]) == 1
    # sans Qwen : légende générique
    monkeypatch.setattr(diffusion, "present", lambda nom: False)
    msg, _ = modele_sprites.preparer(feuilles, "generique", GBA, progress=no_progress)
    assert "génériques" in msg and "a trainer" in msg


def test_entrainement_puis_feuille_generee(faux_entrainement, monkeypatch):
    feuilles = faux_entrainement
    with pytest.raises(gr.Error, match="Prépare d'abord"):
        modele_sprites.entrainer("inconnu", progress=no_progress)
    modele_sprites.preparer(feuilles, "marche_gba", GBA, legender=False, progress=no_progress)
    etapes = []
    msg, fichier = modele_sprites.entrainer("marche_gba", epoques=6, dim=16, graine=7,
                                            progress=lambda p, desc="": etapes.append(desc))
    assert Path(fichier) == cfg.LORAS_DIR / "marche_gba" / "marche_gba.safetensors" and "entraîné" in msg
    assert json.loads(Path(fichier).read_bytes()[5:]) == {"epoques": 6, "dim": 16, "graine": 7, "taille": [512, 768]}
    assert any("pas 2/3" in e for e in etapes)
    assert modele_sprites.loras() == {"marche_gba": Path(fichier)}
    infos = json.loads((cfg.LORAS_DIR / "marche_gba" / "modele.json").read_text(encoding="utf-8"))
    assert infos["entrainement"]["epoques"] == 6
    # le kit : feuille de marche générée avec ce modèle (faux SDXL : une feuille recolorée sur fond magenta)
    appels = []

    def sprite_dresseur(prompts, sorties, graines, largeur=1024, hauteur=1024, etapes=30, guidage=7.0, negatif=None,
                        lora_echelle=1.0, lora=None, progress=None):
        appels.append({"prompt": prompts, "taille": (largeur, hauteur), "lora": lora})
        a = np.array(_feuille(8, couleur=(60, 120, 220)).convert("RGBA"))
        rgb = np.where(a[..., 3:] > 0, a[..., :3], np.array(modele_sprites.FOND, np.uint8))
        Image.fromarray(rgb, "RGB").save(sorties[0])

    monkeypatch.setattr(diffusion, "sprite_dresseur", sprite_dresseur)
    monkeypatch.setattr(diffusion, "verifier", lambda *n: None)
    assert dresseur.liste_loras_marche()["value"] == "marche_gba"
    msg, apercus, fichiers, dossier, face, _, _ = dresseur.creer_kit(
        GBA, None, "un garçon aux cheveux bruns, chemise rouge", "", [dresseur.MARCHE], dresseur.LORA, None, None,
        None, 3, lora_marche="marche_gba", progress=no_progress)
    assert appels[0]["lora"] == str(fichier) and appels[0]["taille"] == (512, 768)
    assert appels[0]["prompt"].startswith("pixelsheet, walking sprite sheet, 4 by 4 grid, un garçon")
    d = Path(dossier)
    a = np.array(Image.open(d / "marche.png"))
    assert a.shape == (96, 64, 4) and set(np.unique(a[..., 3])) == {0, 255}
    assert a[0, 0, 3] == 0 and (a[..., 3] > 0).sum() > 500  # fond retiré, personnages gardés
    assert (d / "marche_bas_x2.gif").exists() and (d / "marche_brut.png").exists()
    infos = json.loads((d / "kit.json").read_text(encoding="utf-8"))
    assert infos["feuilles"]["marche"]["modele"] == "marche_gba"
    with pytest.raises(gr.Error, match="introuvable"):
        dresseur.creer_kit(GBA, None, "x", "", [dresseur.MARCHE], dresseur.LORA, None, None, None, 3,
                           lora_marche="autre", progress=no_progress)


def test_vers_feuille_palette_et_contour():
    a = np.array(_feuille(8, couleur=(60, 120, 220)).convert("RGBA"))
    rgb = np.where(a[..., 3:] > 0, a[..., :3], np.array(modele_sprites.FOND, np.uint8))
    palette = np.array([(60, 120, 220), (240, 200, 160), (16, 16, 24)], np.uint8)
    feuille = modele_sprites.vers_feuille(Image.fromarray(rgb, "RGB"), (4, 4), (16, 24), palette)
    op = feuille[..., 3] > 0
    assert feuille.shape == (96, 64, 4) and {tuple(p) for p in feuille[..., :3][op]} <= {tuple(p) for p in palette}
    assert (feuille[..., :3][(a[::8, ::8, 3] > 0) & ~op] == 0).sum() == 0  # rien de perdu sur la grille
