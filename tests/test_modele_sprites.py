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


def test_guide_et_verification_des_fichiers(env, tmp_path, monkeypatch):
    """Le guide donne le type de fichier attendu d'après la grille du style ; les fichiers déposés sont vérifiés un
    par un (archive .zip ouverte), avec la raison d'un refus."""
    import zipfile

    monkeypatch.setattr(diffusion, "present", lambda nom: False)
    g = modele_sprites.guide(GBA)
    assert "64×96 pixels" in g and "128×192 (×2)" in g and "PNG" in g and ".zip" in g and ".safetensors" in g
    assert "512×768 px" in g and "4 by 4 grid" in g
    d = modele_sprites.guide(DS)
    assert "96×128 pixels" in d and "3 by 4 grid" in d
    assert "METTRE_A_JOUR.bat" in modele_sprites.etat_prerequis()
    bonne, ds, texte = tmp_path / "bonne.png", tmp_path / "ds.png", tmp_path / "notes.txt"
    _feuille(2).save(bonne)
    _feuille(1, cols=3, rows=4, cw=32, ch=32).save(ds)
    texte.write_text("x")
    archive = tmp_path / "feuilles.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(bonne, "Characters/a/trchar001.png")
        z.write(bonne, "Characters/b/trchar001.png")  # même nom dans deux dossiers : les deux gardés
        z.writestr("../../evasion.png", bonne.read_bytes())  # chemin hors de l'archive : nom seul gardé
        z.writestr("Lisez-moi.txt", "x")
    msg, lignes = modele_sprites.verifier_feuilles([str(bonne), str(ds), str(texte), str(archive)], GBA)
    etats = {l[0]: l[3] for l in lignes}
    assert etats["bonne.png"] == "✅ bonne" and [l[2] for l in lignes if l[0] == "bonne.png"] == ["×2"]
    assert etats["ds.png"].startswith("❌") and "64×96" in etats["ds.png"]
    assert "type de fichier" in etats["notes.txt"]
    assert {"trchar001.png", "trchar001_2.png", "evasion.png"} <= set(etats) and "Lisez-moi.txt" not in etats
    assert not (tmp_path.parent / "evasion.png").exists()
    assert "4 feuille(s) bonne(s) sur 6" in msg and "au moins 8" in msg
    assert modele_sprites.verifier_feuilles([], GBA)[1] == []


def test_preparation_depuis_une_archive_et_legendes_modifiees(faux_entrainement, tmp_path):
    import zipfile

    archive = tmp_path / "Characters.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for f in faux_entrainement:
            z.write(f, f"Characters/{Path(f).name}")
    msg, _ = modele_sprites.preparer([str(archive)], "depuis_zip", GBA, legender=False, progress=no_progress)
    assert "10 feuille(s)" in msg
    lignes = modele_sprites.legendes("depuis_zip")
    assert len(lignes) == 10 and lignes[0][0] == "001.png" and "a trainer" in lignes[0][1]
    lignes[0][1] = "walking sprite sheet, 4 by 4 grid, a girl with pink hair, yellow dress"  # déclencheur effacé
    lignes[1][1] = ""  # vide : gardée telle quelle
    assert "1 légende(s)" in modele_sprites.enregistrer_legendes("depuis_zip", lignes[:2])
    jeu = cfg.LORAS_DIR / "depuis_zip" / "jeu" / f"{modele_sprites.REPETITIONS}_{modele_sprites.DECLENCHEUR}"
    assert (jeu / "001.txt").read_text(encoding="utf-8") == (
        "pixelsheet, walking sprite sheet, 4 by 4 grid, a girl with pink hair, yellow dress")
    assert "a trainer" in (jeu / "002.txt").read_text(encoding="utf-8")
    with pytest.raises(gr.Error, match="Prépare d'abord"):
        modele_sprites.enregistrer_legendes("inconnu", lignes)


def _sprite_face(k=3, couleur=(40, 90, 200), fond=None):
    """Sprite de face de 40×56 pixels de jeu (tête, yeux, corps), agrandi ×k."""
    im = Image.new("RGBA", (40, 56), fond or (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((10, 20, 29, 55), fill=couleur + (255,))
    d.ellipse((12, 2, 27, 19), fill=(240, 200, 160, 255))
    d.point([(16, 10), (23, 10)], fill=(16, 16, 24, 255))
    return im.resize((im.width * k, im.height * k), Image.Resampling.NEAREST)


def test_modele_de_sprites_de_face(faux_entrainement, tmp_path, monkeypatch):
    """Sprites de face (demande du 01/11) : vérifiés, posés en bas au centre de la toile, agrandis ×16 (1024² en GBA),
    légendés « pixelsprite… » ; modèle entraîné à 1024² puis choisi dans le kit, sprite relu sur sa grille exacte."""
    sprites = []
    for i in range(9):
        f = tmp_path / f"face{i:03d}.png"
        _sprite_face(1 + i % 3, (20 * i, 90, 200 - 10 * i), fond=(200, 220, 240, 255) if i % 2 else None).save(f)
        sprites.append(str(f))
    trop_grand = tmp_path / "geant.png"
    geant = Image.new("RGBA", (72, 72), (0, 0, 0, 0))
    ImageDraw.Draw(geant).rectangle((1, 1, 70, 70), fill=(200, 0, 0, 255))
    for x in range(1, 71):  # damier d'un pixel : pas d'agrandissement à retirer
        for y in range(1 + x % 2, 71, 2):
            geant.putpixel((x, y), (0, 0, 200, 255))
    geant.save(trop_grand)
    g = modele_sprites.guide(GBA, "Sprites de face")
    assert "64×64" in g and "×16" in g and "1024×1024" in g and "pixelsprite" in g
    msg, lignes = modele_sprites.verifier_feuilles(sprites + [str(trop_grand)], GBA, "Sprites de face")
    assert "9 sprite(s) bon(s) sur 10" in msg and "plus grand que la toile" in lignes[-1][3]
    modes = []
    monkeypatch.setattr(diffusion, "present", lambda nom: nom == "qwen")
    monkeypatch.setattr(diffusion, "decrire", lambda mode, texte, image=None, progress=None:
                        modes.append(mode) or "a girl with blue dress.")
    msg, dossier = modele_sprites.preparer(sprites, "face_gba", GBA, type_="Sprites de face", progress=no_progress)
    jeu = Path(dossier) / "jeu" / f"{modele_sprites.REPETITIONS}_{modele_sprites.DECLENCHEUR_FACE}"
    images = sorted(jeu.glob("*.png"))
    assert len(images) == 9 and set(modes) == {"sprite_face"}
    a = np.array(Image.open(images[0]))
    assert a.shape == (1024, 1024, 3)
    # personnage de 40×56 posé en bas au centre de la toile de 64 : pieds sur la dernière ligne
    assert tuple(a[-8, 512]) != modele_sprites.FOND and tuple(a[(64 - 57) * 16 + 8, 512]) == modele_sprites.FOND
    assert images[0].with_suffix(".txt").read_text(encoding="utf-8") == (
        "pixelsprite, trainer battle sprite, front view, full body, a girl with blue dress, on a magenta background")
    assert modele_sprites.legendes("face_gba")[0][1].startswith("pixelsprite")
    monkeypatch.setattr(diffusion, "present", lambda nom: False)
    _, fichier = modele_sprites.entrainer("face_gba", epoques=6, dim=16, progress=no_progress)
    assert json.loads(Path(fichier).read_bytes()[5:])["taille"] == [1024, 1024]
    assert modele_sprites.loras("face") == {"face_gba": Path(fichier)} and modele_sprites.loras("marche") == {}
    assert dresseur.liste_loras_face()["value"] == "face_gba"
    assert dresseur.liste_loras_marche()["value"] == dresseur.MARCHE_EXEMPLE

    appels = []

    def sprite_dresseur(prompts, sorties, graines, largeur=1024, hauteur=1024, etapes=30, guidage=7.0, negatif=None,
                        lora_echelle=1.0, lora=None, progress=None):
        appels.append({"prompt": prompts, "taille": (largeur, hauteur), "lora": lora, "graines": graines})
        for s in sorties:  # le modèle dessine à l'échelle exacte : un pixel du jeu = 16 px, fond magenta
            natif = np.array(modele_sprites.lire_sprite(sprites[0], 64))
            rgb = np.where(natif[..., 3:] > 0, natif[..., :3], np.array(modele_sprites.FOND, np.uint8))
            Image.fromarray(rgb, "RGB").resize((1024, 1024), Image.Resampling.NEAREST).save(s)

    monkeypatch.setattr(diffusion, "sprite_dresseur", sprite_dresseur)
    monkeypatch.setattr(diffusion, "verifier", lambda *n: None)
    msg, apercus, fichiers, dossier, face, _, description = dresseur.creer_kit(
        GBA, None, "x", "", [dresseur.FACE], dresseur.LORA, None, None, None, 5, variantes=2, lora_face="face_gba",
        description="a girl with a blue dress", progress=no_progress)
    assert appels[0]["lora"] == str(fichier) and appels[0]["taille"] == (1024, 1024) and appels[0]["graines"] == [5, 6]
    assert appels[0]["prompt"] == ("pixelsprite, trainer battle sprite, front view, full body, a girl with a blue "
                                   "dress, on a magenta background")
    a = np.array(Image.open(Path(dossier) / "face.png"))
    assert a.shape == (64, 64, 4) and a[0, 0, 3] == 0
    natif = modele_sprites.lire_sprite(sprites[0], 64)
    assert ((a[..., 3] > 0) == (natif[..., 3] > 0)).mean() > 0.97  # même silhouette, pixel pour pixel
    assert (a[63 - 45, 32 - 4, :3] < 60).all()  # œil gardé (point sombre à sa place)
    infos = json.loads((Path(dossier) / "kit.json").read_text(encoding="utf-8"))
    assert infos["feuilles"]["face"]["modele"] == "face_gba"
    with pytest.raises(gr.Error, match="entraîné pour le style"):
        dresseur.creer_kit(DS, None, "x", "", [dresseur.FACE], dresseur.LORA, None, None, None, 5,
                           lora_face="face_gba", description="x", progress=no_progress)
