"""Kit de dresseur : sprite de face, feuilles de lancer et de marche redessinées d'après des gabarits (faux klein)."""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import diffusion, dresseur

GBA, DS = list(dresseur.STYLES)


def _feuille(cols, rows, cw, ch, k, fond=None):
    """Feuille d'exemple : un bonhomme par case (tête et corps), agrandie ×k, fond transparent ou uni."""
    im = Image.new("RGBA", (cols * cw, rows * ch), fond or (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for r in range(rows):
        for c in range(cols):
            x, y = c * cw + (c % 3) - 1, r * ch  # poses un peu différentes d'une case à l'autre
            d.rectangle((x + cw // 4, y + ch // 3, x + 3 * cw // 4 - 1, y + ch - 1), fill=(200, 40, 40, 255))
            d.ellipse((x + cw // 3, y + 1, x + 2 * cw // 3, y + ch // 3), fill=(240, 200, 160, 255))
    return im.resize((im.width * k, im.height * k), Image.Resampling.NEAREST)


@pytest.fixture
def faux_klein(env, monkeypatch, tmp_path):
    appels = []

    def personnage(prompt, references, sorties, graines, largeur, hauteur, progress=None):
        appels.append({"prompt": prompt, "references": [str(r) for r in references], "taille": (largeur, hauteur)})
        if len(references) == 1:  # sprite de face : un personnage sur fond blanc
            im = Image.new("RGB", (largeur, hauteur), (255, 255, 255))
            d = ImageDraw.Draw(im)
            d.rectangle((300, 300, 470, 1000), fill=(30, 60, 200))
            d.ellipse((320, 80, 450, 300), fill=(240, 190, 150))
            d.rectangle((330, 60, 440, 130), fill=(220, 30, 30))
        else:  # feuille : le gabarit redessiné, personnage recoloré (le fond vif reste)
            g = np.array(Image.open(references[1]).convert("RGB")).astype(int)
            fond = g[0, 0]
            perso = np.abs(g - fond).sum(-1) > 30
            g[perso] = np.where(g[perso].sum(-1, keepdims=True) > 500, [240, 190, 150], [30, 60, 200])
            im = Image.fromarray(g.astype(np.uint8), "RGB").resize((largeur, hauteur), Image.Resampling.NEAREST)
        for s in sorties:
            im.save(s)

    monkeypatch.setattr(diffusion, "personnage", personnage)
    monkeypatch.setattr(diffusion, "verifier", lambda *n: None)
    monkeypatch.setattr(diffusion, "present", lambda nom: False)
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (300, 400), (90, 120, 90)).save(photo)
    return appels, photo, tmp_path


def test_kit_de_dresseur_gba(faux_klein):
    appels, photo, tmp = faux_klein
    lancer, marche = tmp / "trback.png", tmp / "trchar.png"
    _feuille(5, 1, 64, 64, 2).save(lancer)
    _feuille(4, 4, 16, 24, 2, fond=(120, 160, 200, 255)).save(marche)  # fond uni, pas de transparence
    msg, apercus, fichiers, dossier, face = dresseur.creer_kit(
        GBA, str(photo), "l'homme, en dresseur", dresseur.FEUILLES, str(lancer), str(marche), None, 7,
        progress=no_progress)
    d = Path(dossier)
    assert Image.open(face).size == (64, 64)
    feuille_lancer = np.array(Image.open(d / "lancer.png"))
    feuille_marche = np.array(Image.open(d / "marche.png"))
    assert feuille_lancer.shape[:2] == (64, 320) and feuille_marche.shape[:2] == (96, 64)
    # bande de 5 cases redessinée en 3 colonnes (image moins allongée), puis remise en bande
    assert dresseur.disposition(5, 64, 64) == 3 and appels[1]["taille"] == (1344, 896)
    # chaque case a un personnage à la place de celui du gabarit, transparence franche, couleurs du sprite de face
    gab, _, _ = dresseur.lire_gabarit(lancer, GBA, "lancer")
    for a, g in ((feuille_lancer, gab), (feuille_marche, dresseur.lire_gabarit(marche, GBA, "marche")[0])):
        assert set(np.unique(a[..., 3])) <= {0, 255}
        assert ((a[..., 3] > 0) & (g[..., 3] > 0)).sum() > 0.8 * (g[..., 3] > 0).sum()
        assert not ((a[..., 3] > 0) & ~(g[..., 3] > 0)).any()  # rien hors de la silhouette du gabarit
    face_a = np.array(Image.open(face))
    couleurs_face = {tuple(p) for p in face_a[..., :3][face_a[..., 3] > 0]}
    assert {tuple(p) for p in feuille_lancer[..., :3][feuille_lancer[..., 3] > 0]} <= couleurs_face
    noms = {p.split("/")[-1] for p in fichiers}
    assert {"face_x2.png", "lancer_x2.png", "lancer_x2.gif", "marche_bas_x2.gif", "marche_haut_x2.gif"} <= noms
    with Image.open(d / "lancer_x2.gif") as gif:
        assert gif.n_frames == 5
    assert "Redraw image 2 exactly" in appels[1]["prompt"] and "image 1" in appels[1]["prompt"]
    assert json.loads((d / "kit.json").read_text(encoding="utf-8"))["feuilles"]["marche"]["grille"] == [4, 4]
    # gabarits gardés : la fois suivante, sans feuille d'exemple ni nouvelle face (sprite déjà créé)
    assert dresseur.gabarit_enregistre(GBA, "lancer") and dresseur.gabarit_enregistre(GBA, "marche")
    _, _, fichiers2, _, _ = dresseur.creer_kit(GBA, None, "un dresseur", [dresseur.MARCHE], None, None, face, 7,
                                               progress=no_progress)
    assert any(f.endswith("marche_x2.png") for f in fichiers2)


def test_kit_de_dresseur_ds_et_erreurs(faux_klein):
    appels, photo, tmp = faux_klein
    lancer, marche, mauvaise = tmp / "lancer.png", tmp / "marche_ds.png", tmp / "mauvaise.png"
    _feuille(5, 1, 64, 64, 2).save(lancer)
    _feuille(3, 4, 32, 32, 1).save(marche)
    _feuille(3, 3, 30, 30, 1).save(mauvaise)
    with pytest.raises(gr.Error, match="Pas de feuille d'exemple"):
        dresseur.creer_kit(DS, str(photo), "x", [dresseur.FACE, dresseur.LANCER], None, None, None, 1,
                           progress=no_progress)
    dresseur.enregistrer_gabarit(GBA, "lancer", str(lancer))  # le lancer GBA sert à la DS
    with pytest.raises(gr.Error, match="3 colonnes et 4 lignes"):
        dresseur.creer_kit(DS, str(photo), "x", dresseur.FEUILLES, None, str(mauvaise), None, 1, progress=no_progress)
    _, _, _, dossier, face = dresseur.creer_kit(DS, str(photo), "x", dresseur.FEUILLES, None, str(marche), None, 1,
                                                progress=no_progress)
    d = Path(dossier)
    assert Image.open(face).size == (80, 80)
    assert Image.open(d / "lancer.png").size == (400, 80)  # cases du gabarit GBA agrandies à 80×80
    assert Image.open(d / "marche.png").size == (96, 128)
    assert (d / "marche_droite_x2.gif").exists() and "ℹ️" in dresseur.etat_gabarits(DS)
