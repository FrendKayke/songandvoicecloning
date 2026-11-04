"""Refaire les yeux d'un sprite : masque peint relu à l'échelle du sprite, fenêtre donnée à SDXL à l'échelle du
modèle, seuls les pixels peints changent (palette du sprite, fond gardé), modèle entraîné ou LoRA de sWizad."""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import diffusion, dresseur, modele_sprites
from studiovoix import yeux

PEAU, CHEVEUX, NOIR, BLANC = (240, 200, 160), (230, 220, 90), (16, 16, 24), (250, 250, 250)


def _sprite():
    """Personnage 80×80 : tête de peau, cheveux, yeux ratés (une tache de 3 pixels), corps."""
    a = np.zeros((80, 80, 4), np.uint8)
    a[10:20, 30:50] = CHEVEUX + (255,)
    a[20:34, 32:48] = PEAU + (255,)
    a[25:27, 36:39] = (120, 90, 140, 255)  # œil raté (violet baveux)
    a[34:76, 34:46] = (240, 110, 40, 255)
    a[[33], 32:48] = NOIR + (255,)
    a[78:80, 30:50] = 0
    return a


def _editeur(zone_affichee):
    """Valeur d'un gr.ImageEditor : un calque RGBA avec le pinceau."""
    calque = np.zeros(zone_affichee.shape + (4,), np.uint8)
    calque[zone_affichee] = (255, 0, 0, 200)
    return {"background": None, "layers": [calque], "composite": None}


def test_masque_peint_relu_a_l_echelle_du_sprite():
    affichage = np.zeros((80 * yeux.AFFICHAGE, 80 * yeux.AFFICHAGE), bool)
    affichage[25 * 8:27 * 8, 36 * 8:39 * 8 + 1] = True  # déborde d'1 px sur la colonne 39 (moins de 20 %)
    m = yeux.masque(_editeur(affichage), 80, 80)
    assert m.sum() == 6 and m[25:27, 36:39].all() and not m[:, 39].any()
    assert not yeux.masque(None, 80, 80).any()


def test_fenetre_preparee_et_pixels_remplaces():
    sprite = _sprite()
    zone = np.zeros((80, 80), bool)
    zone[25:27, 36:39] = True
    image, msk, (y0, x0), cote = yeux.preparer(sprite, zone, 16, modele_sprites.FOND)
    assert cote == 64 and image.size == (1024, 1024) and msk.size == (1024, 1024)
    assert 0 <= y0 <= 25 and y0 + 64 >= 27 and 0 <= x0 <= 36 and x0 + 64 >= 39
    a = np.asarray(image)
    assert tuple(a[(30 - y0) * 16 + 8, (40 - x0) * 16 + 8]) == PEAU  # sprite recopié au plus proche
    assert tuple(a[(2 - y0) * 16 + 8 if y0 <= 2 else 8, 8]) == modele_sprites.FOND  # fond du modèle
    m = np.asarray(msk) > 0
    assert m[(25 - y0) * 16 + 8, (36 - x0) * 16 + 8] and m[(24 - y0) * 16 + 8, (36 - x0) * 16 + 8]  # élargi
    assert not m[(30 - y0) * 16 + 8, (45 - x0) * 16 + 8]
    # « SDXL » : yeux noirs nets, un pixel blanc de reflet, et un pixel du masque laissé en fond magenta
    sortie = a.copy()
    for (y, x), c in {(25, 36): NOIR, (25, 37): NOIR, (26, 36): NOIR, (26, 37): (30, 20, 30), (25, 38): BLANC,
                      (26, 38): modele_sprites.FOND}.items():
        sortie[(y - y0) * 16:(y - y0 + 1) * 16, (x - x0) * 16:(x - x0 + 1) * 16] = c
    sortie[(30 - y0) * 16:(31 - y0) * 16, (40 - x0) * 16:(41 - x0) * 16] = NOIR  # hors masque : ignoré
    palette = np.vstack([yeux.palette_du_sprite(sprite), [BLANC]]).astype(np.uint8)
    nouveau, changes = yeux.appliquer(sprite, zone, Image.fromarray(sortie), (y0, x0), 16, modele_sprites.FOND,
                                      palette)
    assert changes == 5
    assert tuple(nouveau[26, 37, :3]) == NOIR  # presque noir → noir de la palette
    assert tuple(nouveau[25, 38, :3]) == BLANC and tuple(nouveau[26, 38]) == tuple(sprite[26, 38])  # fond : gardé
    autre = ~zone
    assert (nouveau[autre] == sprite[autre]).all()  # rien d'autre ne bouge


def test_retoucher_avec_le_lora_par_defaut_et_un_modele_entraine(env, monkeypatch):
    dossier = cfg.PIXEL_DIR / "20261103_190000_dresseur"
    dossier.mkdir(parents=True)
    Image.fromarray(_sprite(), "RGBA").save(dossier / "face.png")
    (dossier / "kit.json").write_text(json.dumps({"style": next(iter(dresseur.STYLES)), "description":
                                                  "a boy with blond hair and an orange jacket"}), encoding="utf-8")
    vu = {}

    def faux_sdxl(prompt, sorties, graines, largeur, hauteur, lora=None, image=None, masque=None, force=0.85,
                  progress=None, **k):
        vu.update(prompt=prompt, lora=lora, largeur=largeur, force=force, graines=graines)
        entree, msk = np.asarray(Image.open(image)).copy(), np.asarray(Image.open(masque)) > 0
        for s in sorties:  # tout le masque peint en noir
            sortie = entree.copy()
            sortie[msk] = NOIR
            Image.fromarray(sortie).save(s)
        return {"fichiers": [str(s) for s in sorties]}

    monkeypatch.setattr(diffusion, "sprite_dresseur", faux_sdxl)
    affichage = np.zeros((640, 640), bool)
    affichage[25 * 8:27 * 8, 36 * 8:39 * 8] = True
    msg, apercus, fichiers, d = yeux.retoucher(str(dossier / "face.png"), _editeur(affichage), "", None, 3, 0.7, 5,
                                               progress=no_progress)
    assert vu["lora"] is None and vu["largeur"] == 1024 and vu["force"] == 0.7 and vu["graines"] == [5, 6, 7]
    assert "a pixel image of a boy with blond hair" in vu["prompt"] and "gen3" in vu["prompt"]  # kit.json relu
    assert [Path(f).name for f in fichiers] == ["face_yeux_1.png", "face_yeux_2.png", "face_yeux_3.png"]
    assert Path(d) == dossier and len(apercus) == 3 and "6 pixels peints" in msg
    nouveau = np.array(Image.open(fichiers[0]))
    assert (nouveau[25:27, 36:39, :3] == NOIR).all() and (nouveau[24, 36] == _sprite()[24, 36]).all()

    lora = modele_sprites.dossier_lora("face_gba")
    lora.mkdir(parents=True)
    (lora / "face_gba.safetensors").write_bytes(b"x")
    (lora / "modele.json").write_text(json.dumps({"type": modele_sprites.FACE, "toile": 64, "facteur": 16}),
                                      encoding="utf-8")
    monkeypatch.setattr(modele_sprites, "loras", lambda type_=None: {"face_gba": lora / "face_gba.safetensors"})
    yeux.retoucher(str(dossier / "face.png"), _editeur(affichage), "", "face_gba", 1, progress=no_progress)
    assert vu["lora"].endswith("face_gba.safetensors") and vu["prompt"].startswith(modele_sprites.DECLENCHEUR_FACE)
    with pytest.raises(gr.Error, match="Peins"):
        yeux.retoucher(str(dossier / "face.png"), None, progress=no_progress)


def test_cote_multiple_de_8():
    """Modèle de face DS (facteur 12) : 85 × 12 = 1020 refusé par SDXL ; la fenêtre passe à 84 (1008 px)."""
    from studiovoix import yeux

    assert yeux.cote_fenetre(12) == 84
    for f in range(2, 33):
        c = yeux.cote_fenetre(f)
        assert (c * f) % 8 == 0 and c * f <= yeux.COTE_IMAGE and c * f > yeux.COTE_IMAGE - 8 * max(1, f)


def test_trait_ferme_autour_d_un_morceau_et_morceau_efface():
    """Essai de l'utilisateur du 04/11 : la robe en trop entourée d'un trait (pas peinte) ; elle doit être prise en
    entier, puis effacée là où le modèle met du fond. Un fond au milieu du visage (blanc d'un œil) reste."""
    sprite = _sprite()
    sprite[40:56, 56:70] = (120, 200, 210, 255)  # morceau de robe en trop, séparé du corps par du vide
    k = yeux.AFFICHAGE
    affichage = np.zeros((80 * k, 80 * k), bool)
    yy, xx = np.mgrid[:80 * k, :80 * k]
    rayon = np.hypot((yy - 48 * k) / (12 * k), (xx - 63 * k) / (10 * k))
    affichage[(rayon > 0.95) & (rayon < 1.05)] = True  # un anneau seulement
    zone = yeux.masque(_editeur(affichage), 80, 80) & (sprite[..., 3] > 0)
    assert zone[40:56, 56:70].all() and not zone[:, :50].any()
    image, _, coin, _ = yeux.preparer(sprite, zone, 12, modele_sprites.FOND)
    sortie = np.asarray(image).copy()
    y0, x0 = coin
    sortie[(40 - y0) * 12:(56 - y0) * 12, (56 - x0) * 12:(70 - x0) * 12] = modele_sprites.FOND  # SDXL : du fond
    nouveau, changes = yeux.appliquer(sprite, zone, Image.fromarray(sortie), coin, 12, modele_sprites.FOND,
                                      yeux.palette_du_sprite(sprite))
    assert (nouveau[40:56, 56:70, 3] == 0).all() and changes == 16 * 14
    assert (nouveau[:, :50] == sprite[:, :50]).all()


def test_consigne_en_francais_pas_envoyee_a_sdxl():
    d, remarque = yeux.description_anglaise("refait les yeux et la bouche et enlève la robe en trop")
    assert d == "" and "anglais" in remarque
    assert yeux.description_anglaise("a woman with curly black hair, blue mermaid dress") == (
        "a woman with curly black hair, blue mermaid dress", None)
