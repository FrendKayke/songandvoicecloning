"""Kit de dresseur : sprites d'exemple recolorés aux couleurs de la personne (faux klein qui recolore)."""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import no_progress
from studiovoix import diffusion, dresseur

GBA, DS = list(dresseur.STYLES)
PEAU, CHEVEUX, CHEVEUX_OMBRE, HABIT, HABIT_OMBRE = (240, 200, 160), (120, 70, 30), (80, 45, 20), (200, 40, 40), \
    (140, 20, 20)


def _bonhomme(d, x, y, cw, ch):
    d.rectangle((x + cw // 4, y + ch // 3, x + 3 * cw // 4 - 1, y + ch - 1), fill=HABIT + (255,))
    d.rectangle((x + cw // 4, y + 2 * ch // 3, x + 3 * cw // 4 - 1, y + ch - 1), fill=HABIT_OMBRE + (255,))
    d.ellipse((x + cw // 3, y + 1, x + 2 * cw // 3, y + ch // 3), fill=PEAU + (255,))
    d.rectangle((x + cw // 3, y + 1, x + 2 * cw // 3, y + ch // 8), fill=CHEVEUX + (255,))
    d.rectangle((x + cw // 3, y + 1, x + cw // 2, y + ch // 16), fill=CHEVEUX_OMBRE + (255,))
    d.rectangle((x + cw // 4, y + ch - 1, x + 3 * cw // 4 - 1, y + ch - 1), fill=(0, 0, 0, 255))  # contour


def _feuille(cols, rows, cw, ch, k, fond=None):
    """Sprites d'exemple : un bonhomme par case (cheveux, peau, habit avec ombre, contour), agrandis ×k."""
    im = Image.new("RGBA", (cols * cw, rows * ch), fond or (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for r in range(rows):
        for c in range(cols):
            _bonhomme(d, c * cw + (c % 3) - 1, r * ch, cw, ch)  # poses un peu différentes d'une case à l'autre
    return im.resize((im.width * k, im.height * k), Image.Resampling.NEAREST)


NOUVELLES = {PEAU: (250, 220, 200), CHEVEUX: (230, 200, 60), CHEVEUX_OMBRE: (190, 160, 40), HABIT: (60, 120, 220),
             HABIT_OMBRE: (60, 120, 220)}  # deux nuances de l'habit peintes pareil : l'ombrage doit rester


@pytest.fixture
def faux_klein(env, monkeypatch, tmp_path):
    appels = []

    def personnage(prompt, references, sorties, graines, largeur, hauteur, progress=None):
        appels.append({"prompt": prompt, "references": [str(r) for r in references], "taille": (largeur, hauteur)})
        if "Recolor" not in prompt:  # sprite de face dessiné d'après la photo
            im = Image.new("RGB", (largeur, hauteur), (255, 255, 255))
            d = ImageDraw.Draw(im)
            d.rectangle((300, 300, 470, 1000), fill=(30, 60, 200))
            d.ellipse((320, 80, 450, 300), fill=(240, 190, 150))
        else:  # recoloriage : chaque couleur connue remplacée, fond et contours gardés (dessin un peu flou)
            a = np.array(Image.open(references[0]).convert("RGB"))
            b = a.copy()
            for ancienne, nouvelle in NOUVELLES.items():
                b[(a == ancienne).all(-1)] = nouvelle
            im = Image.fromarray(b, "RGB").resize((largeur, hauteur), Image.Resampling.NEAREST)
        for s in sorties:
            im.save(s)

    monkeypatch.setattr(diffusion, "personnage", personnage)
    monkeypatch.setattr(diffusion, "verifier", lambda *n: None)
    monkeypatch.setattr(diffusion, "present", lambda nom: False)
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (300, 400), (90, 120, 90)).save(photo)
    return appels, photo, tmp_path


def _bibliotheque(tmp, style=GBA):
    face, lancer, marche = tmp / "trainer108.png", tmp / "trback002.png", tmp / "trchar113.png"
    _feuille(1, 1, 40, 56, 2).save(face)
    _feuille(5, 1, 64, 64, 2).save(lancer)
    _feuille(4, 4, 16, 24, 2, fond=(120, 160, 200, 255)).save(marche)  # fond uni, pas de transparence
    for feuille, f in ((dresseur.FACE, face), (dresseur.LANCER, lancer), (dresseur.MARCHE, marche)):
        msg, *listes = dresseur.ajouter_gabarits(style, feuille, [str(f)])
        assert "✅" in msg
    return listes


def test_kit_recolore_les_exemples(faux_klein):
    appels, photo, tmp = faux_klein
    _bibliotheque(tmp)
    noms = {f: dresseur.liste_gabarits(GBA, f) for f in ("face", "lancer", "marche")}
    assert noms == {"face": ["trainer108"], "lancer": ["trback002"], "marche": ["trchar113"]}
    msg, apercus, fichiers, dossier, face, couleurs = dresseur.creer_kit(
        GBA, str(photo), "la femme", "hair: blond; top: blue", dresseur.FEUILLES, dresseur.RECOLORER,
        "trainer108", "trback002", "trchar113", 7, progress=no_progress)
    d = Path(dossier)
    assert len(appels) == 1 and "hair: blond" in appels[0]["prompt"]  # une seule image pour toutes les feuilles
    for f, nom in (("face", "trainer108"), ("lancer", "trback002"), ("marche", "trchar113")):
        g = dresseur.lire_gabarit(dresseur.chemin_gabarit(GBA, f, nom), GBA, f)
        a = np.array(Image.open(d / f"{f}.png"))
        assert a.shape == g.shape and (a[..., 3] == g[..., 3]).all()  # silhouette exacte de l'exemple
        op = g[..., 3] >= 128
        anciennes = {tuple(p) for p in g[..., :3][op]}
        nouvelles = {tuple(p) for p in a[..., :3][op]}
        assert len(nouvelles) == len(anciennes)  # ombrage gardé, même si klein peint deux nuances pareil
        assert (a[..., :3][(g[..., :3] == 0).all(-1) & op] == 0).all()  # contours noirs gardés
        assert (a[..., :3][(g[..., :3] == CHEVEUX).all(-1) & op] == NOUVELLES[CHEVEUX]).all()
    assert Image.open(d / "lancer.png").size == (320, 64) and Image.open(d / "marche.png").size == (64, 96)
    assert Image.open(face).size == (64, 64)
    noms_fichiers = {Path(p).name for p in fichiers}
    assert {"face_x2.png", "lancer_x2.gif", "marche_bas_x2.gif", "marche_haut_x2.gif"} <= noms_fichiers
    with Image.open(d / "lancer_x2.gif") as gif:
        assert gif.n_frames == 5
    infos = json.loads((d / "kit.json").read_text(encoding="utf-8"))
    assert infos["feuilles"]["marche"]["exemple"] == "trchar113" and couleurs == "hair: blond; top: blue"


def test_face_dessinee_et_couleurs_lues(faux_klein, monkeypatch):
    appels, photo, tmp = faux_klein
    _bibliotheque(tmp)
    with pytest.raises(gr.Error, match="couleurs"):
        dresseur.creer_kit(GBA, None, "", "", [dresseur.MARCHE], dresseur.RECOLORER, None, None, "trchar113", 1,
                           progress=no_progress)
    monkeypatch.setattr(diffusion, "present", lambda nom: nom == "qwen")
    lu = []
    monkeypatch.setattr(diffusion, "decrire", lambda mode, texte, image=None, progress=None: lu.append(mode) or
                        ("hair: black; skin: tan" if mode == "couleurs" else "a woman"))
    _, _, _, dossier, face, couleurs = dresseur.creer_kit(
        GBA, str(photo), "la femme", "", dresseur.FEUILLES, dresseur.DESSINER, None, "trback002", "trchar113", 3,
        progress=no_progress)
    assert couleurs == "hair: black; skin: tan" and lu == ["couleurs", "sprite"]
    # face dessinée d'après la photo, puis donnée au recoloriage comme référence des couleurs
    assert len(appels) == 2 and len(appels[1]["references"]) == 2 and "image 2" in appels[1]["prompt"]
    assert Image.open(face).size == (64, 64) and not (Path(dossier) / "face_source.png").name in str(
        __import__("zipfile").ZipFile(Path(dossier) / f"{Path(dossier).name}.zip").namelist())


def test_ds_lancer_du_gba_et_erreurs(faux_klein):
    appels, photo, tmp = faux_klein
    _bibliotheque(tmp)
    marche_ds, mauvaise = tmp / "marche_ds.png", tmp / "mauvaise.png"
    _feuille(3, 4, 32, 32, 1).save(marche_ds)
    _feuille(3, 3, 30, 30, 1).save(mauvaise)
    msg, *_ = dresseur.ajouter_gabarits(DS, dresseur.MARCHE, [str(mauvaise), str(marche_ds)])
    assert "3 colonnes et 4 lignes" in msg and "1 exemple(s)" in msg
    assert dresseur.liste_gabarits(DS, "lancer") == ["gba:trback002"]  # celui du GBA sert à la DS
    with pytest.raises(gr.Error, match="Pas de sprite d'exemple"):
        dresseur.creer_kit(DS, str(photo), "", "hair: red", dresseur.FEUILLES, dresseur.RECOLORER, None,
                           "gba:trback002", "marche_ds", 1, progress=no_progress)
    _, _, _, dossier, face, _ = dresseur.creer_kit(DS, str(photo), "", "hair: red", [dresseur.LANCER, dresseur.MARCHE],
                                                   dresseur.RECOLORER, None, "gba:trback002", "marche_ds", 1,
                                                   progress=no_progress)
    d = Path(dossier)
    assert face is None and Image.open(d / "lancer.png").size == (400, 80)  # cases agrandies à 80×80
    assert Image.open(d / "marche.png").size == (96, 128) and (d / "marche_droite_x2.gif").exists()
    assert dresseur.apercu_gabarit(DS, "lancer", "gba:trback002").size == (800, 160)


def test_nuances_gardees():
    proches = dresseur._garder_les_nuances({(200, 40, 40): (60, 120, 220), (140, 20, 20): (62, 121, 221),
                                            (0, 0, 0): (0, 0, 0)})
    clair, fonce = np.array(proches[(200, 40, 40)]), np.array(proches[(140, 20, 20)])
    assert dresseur._lum(fonce) < dresseur._lum(clair) - 0.1 and proches[(0, 0, 0)] == (0, 0, 0)
