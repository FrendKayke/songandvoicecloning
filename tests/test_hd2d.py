"""Passer en HD-2D : type deviné, réduction du pixel art agrandi (décalage compris), normales, raccord des tuiles,
transparence des effets sur fond blanc, fichiers produits (cartes à la taille d'origine, GIF, GLB valide)."""
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageSequence

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import hd2d, pixel_art


def _sprite():
    im = Image.new("RGBA", (32, 48), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((8, 2, 24, 18), fill=(230, 180, 150, 255), outline=(0, 0, 0, 255))
    d.rectangle((6, 18, 26, 46), fill=(200, 40, 50, 255), outline=(0, 0, 0, 255))
    return im


def _briques(cote=16):
    t = Image.new("RGBA", (cote, cote), (120, 110, 90, 255))
    d = ImageDraw.Draw(t)
    for y in range(0, cote, 4):
        for x in range(-2 if (y // 4) % 2 else 0, cote, 4):
            d.rectangle((x + 1, y + 1, x + 3, y + 3), fill=(190, 175, 140, 255))
    return t


def _lire_glb(chemin):
    donnees = Path(chemin).read_bytes()
    magie, version, total = struct.unpack("<4sII", donnees[:12])
    longueur, genre = struct.unpack("<I4s", donnees[12:20])
    assert (magie, version, total, genre) == (b"glTF", 2, len(donnees), b"JSON")
    gltf = json.loads(donnees[20:20 + longueur])
    longueur_bin, genre_bin = struct.unpack("<I4s", donnees[20 + longueur:28 + longueur])
    assert genre_bin == b"BIN\0" and gltf["buffers"][0]["byteLength"] == longueur_bin
    for vue in gltf["bufferViews"]:
        assert vue["byteOffset"] + vue["byteLength"] <= longueur_bin
    return gltf


def test_types_devines():
    sprite = np.array(_sprite())
    assert hd2d.deviner_type(sprite) == hd2d.SPRITE
    assert hd2d.deviner_type(np.array(_briques())) == hd2d.TUILE  # joints reliés aux bords : quand même une tuile
    sur_blanc = Image.new("RGBA", (32, 48), "white")
    sur_blanc.alpha_composite(_sprite())
    assert hd2d.deviner_type(np.array(sur_blanc)) == hd2d.SPRITE
    effet = Image.new("RGB", (64, 64), "white")
    rond = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(rond).ellipse((16, 16, 48, 48), fill=(110, 60, 200, 220))
    effet.paste(rond.filter(ImageFilter.GaussianBlur(4)), (0, 0), rond.filter(ImageFilter.GaussianBlur(4)))
    assert hd2d.deviner_type(np.array(effet.convert("RGBA"))) == hd2d.EFFET


def test_reduction_du_pixel_art_avec_decalage():
    petit = np.array(_sprite())
    grand = np.repeat(np.repeat(petit, 2, 0), 2, 1)
    decale = np.roll(grand, 1, axis=1)  # comme un des sprites d'exemple de l'utilisateur
    r, k, d = pixel_art.reduire(decale)
    assert k == 2 and d == (0, 1) and (r == petit[:, :r.shape[1]]).all()
    assert (pixel_art.remettre(r, k, d, decale.shape[:2])[:, 1:-1] == decale[:, 1:-1]).all()


def test_normales_et_raccord_des_tuiles():
    plat = hd2d.normales(np.full((8, 8), 0.5))
    assert (plat == (128, 128, 255)).all()
    pente_x = hd2d.normales(np.tile(np.linspace(0, 1, 8), (8, 1)))
    assert (pente_x[2:-2, 2:-2, 0] < 128).all()  # monte vers la droite : la face regarde à gauche
    pente_y = np.tile(np.linspace(0, 1, 8)[:, None], (1, 8))
    assert (hd2d.normales(pente_y)[2:-2, 2:-2, 1] > 128).all()  # OpenGL : vert vers le haut
    assert (hd2d.normales(pente_y, convention=hd2d.DIRECTX)[2:-2, 2:-2, 1] < 128).all()
    tuile = np.array(_briques())
    h = hd2d.relief(tuile, hd2d.TUILE)
    n = hd2d.normales(h, boucle=True)
    n4 = hd2d.normales(np.tile(h, (2, 2)), boucle=True)
    assert (n4[:16, :16] == n).all() and (n4[16:, 16:] == n).all()  # la carte se raccorde à elle-même
    assert len(np.unique(n.reshape(-1, 3), axis=0)) > 4


def test_effet_sur_fond_blanc_devient_transparent():
    couleur, alpha = np.array([110, 60, 200]), 0.6
    pixel = np.round(alpha * couleur + (1 - alpha) * 255).astype(np.uint8)
    a = np.zeros((2, 2, 4), np.uint8)
    a[..., :3] = 255
    a[..., 3] = 255
    a[0, 0, :3] = pixel
    r = hd2d._blanc_vers_transparence(a)
    assert r[1, 1, 3] == 0 and r[0, 0, 3] > 0
    # sur fond blanc, seule l'opacité la plus faible est retrouvable ; recomposé sur du blanc, on retombe sur l'image
    alpha_ = r[0, 0, 3] / 255
    recompose = alpha_ * r[0, 0, :3].astype(float) + (1 - alpha_) * 255
    assert np.abs(recompose - pixel).max() <= 2


def test_convertir_sprite_planche_tuile(env):
    sprite = cfg.DATA_DIR.parent / "dresseur.png"
    pixel_art.agrandir(_sprite(), 2).save(sprite)
    planche = Image.new("RGBA", (128, 96), (0, 0, 0, 0))
    for i in range(4):
        for j in range(2):
            planche.alpha_composite(_sprite(), (i * 32, j * 48))
    planche.save(cfg.DATA_DIR.parent / "marche.png")
    pixel_art.agrandir(_briques(), 4).save(cfg.DATA_DIR.parent / "pierre.png")
    msg, apercus, fichiers, dossier = hd2d.convertir(
        [str(sprite), str(cfg.DATA_DIR.parent / "marche.png"), str(cfg.DATA_DIR.parent / "pierre.png")],
        hd2d.AUTO, 0, 0, 2, 4, hd2d.OPENGL, 32, "équipe", progress=no_progress)
    d = Path(dossier)
    assert d.parent == cfg.HD2D_DIR and len(apercus) == 3 and "tuile de décor" in msg
    assert Image.open(d / "dresseur_normales.png").size == (64, 96)  # taille d'origine (agrandi ×2)
    assert Image.open(d / "pierre_normales.png").size == (64, 64)
    gif = Image.open(apercus[0])
    assert sum(1 for _ in ImageSequence.Iterator(gif)) >= 2
    glb = _lire_glb(d / "dresseur_panneau.glb")
    assert glb["samplers"][0]["magFilter"] == 9728 and glb["materials"][0]["alphaMode"] == "MASK"
    assert "normalTexture" in glb["materials"][0] and len(glb["images"]) == 2
    position = glb["accessors"][glb["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert position["min"][1] == 0 and abs(position["max"][1] - 46 / 32) < 0.05  # pieds au sol, 32 px = 1 m
    bloc = _lire_glb(d / "pierre_bloc.glb")
    assert bloc["accessors"][0]["count"] == 24 and "alphaMode" not in bloc["materials"][0]
    # planche jouée image par image avec 4 colonnes et 2 lignes ; panneau = première image
    msg, apercus, fichiers, dossier = hd2d.convertir([str(cfg.DATA_DIR.parent / "marche.png")], hd2d.SPRITE, 4, 2,
                                                     progress=no_progress)
    assert Image.open(Path(dossier) / "marche_couleur.png").size == (128, 96)
    with pytest.raises(Exception, match="Ajoute"):
        hd2d.convertir([], progress=no_progress)


def test_effet_lumineux(env):
    effet = Image.new("RGB", (200, 100), "white")
    for i in range(2):
        rond = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        ImageDraw.Draw(rond).ellipse((20, 20, 80, 80), fill=(110, 60, 200, 220))
        rond = rond.filter(ImageFilter.GaussianBlur(4))
        effet.paste(rond, (i * 100, 0), rond)
    effet.save(cfg.DATA_DIR.parent / "magie.png")
    msg, apercus, fichiers, dossier = hd2d.convertir([str(cfg.DATA_DIR.parent / "magie.png")], hd2d.AUTO, 2, 1,
                                                     progress=no_progress)
    d = Path(dossier)
    assert "effet lumineux" in msg and (d / "magie_emission.png").exists()
    couleur = np.array(Image.open(d / "magie_couleur.png"))
    assert couleur[0, 0, 3] == 0 and couleur[50, 50, 3] > 150  # fond blanc retiré, cœur opaque
    glb = _lire_glb(d / "magie_panneau.glb")
    assert glb["materials"][0]["alphaMode"] == "BLEND" and "emissiveTexture" in glb["materials"][0]


def test_atlas_de_tuiles(env):
    """Atlas (cases de 32 px) : reconnu, cases marquées d'une croix rouge laissées plates, cases de sol raccordables
    calculées en boucle, cartes de la taille de l'atlas, pas d'objet 3D."""
    atlas = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    sol = _briques(32)
    for x in range(0, 128, 32):
        atlas.paste(sol, (x, 0))
    d = ImageDraw.Draw(atlas)
    d.ellipse((36, 40, 90, 92), fill=(60, 140, 60, 255), outline=(0, 0, 0, 255))  # un objet sur deux cases
    croix = Image.new("RGBA", (32, 32), "white")
    dc = ImageDraw.Draw(croix)
    dc.rectangle((0, 0, 31, 31), outline=(255, 0, 0, 255))
    dc.line((0, 0, 31, 31), fill=(255, 0, 0, 255))
    dc.line((0, 31, 31, 0), fill=(255, 0, 0, 255))
    atlas.paste(croix, (96, 96))
    a = np.array(atlas)
    assert hd2d.case_atlas(a.shape[:2]) == 32 and hd2d.deviner_type(a) == hd2d.ATLAS
    assert hd2d.deviner_type(np.array(_sprite())) == hd2d.SPRITE
    h, vides, raccords = hd2d.relief_atlas(a, 32)
    assert vides[96:, 96:].all() and not vides[:96].any() and len(raccords) == 4
    n = hd2d.normales_atlas(h, vides, raccords)
    assert (n[96:, 96:] == (128, 128, 255)).all() and (n[40:90, 36:90] != (128, 128, 255)).any()
    f = env / "atlas.png"
    atlas.save(f)
    msg, apercus, produits, dossier = hd2d.convertir([str(f)], progress=no_progress)
    assert "jeu de tuiles, 4×4 cases de 32 px (1 vides ignorées)" in msg
    assert not any(p.endswith(".glb") for p in produits)
    assert Image.open(Path(dossier) / "atlas_normales.png").size == (128, 128)
    assert (np.array(Image.open(Path(dossier) / "atlas_couleur.png")) == a).all()
    assert Image.open(Path(dossier) / "atlas_avant_apres.png").size == ((128 * 2 + 8) * 8, 128 * 8)
