"""Pixel art : style lu sur des sprites d'exemple, conversion, animation (calculée, et d'après une vidéo H.264)."""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import diffusion, pixel_art


def _sprite(toile=64):
    """Petit personnage : corps rouge, tête beige, jambes bleues, contour noir, fond transparent."""
    im = Image.new("RGBA", (toile, toile), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((24, 4, 40, 20), fill=(230, 180, 150, 255), outline=(0, 0, 0, 255))
    d.rectangle((20, 20, 44, 44), fill=(200, 40, 50, 255), outline=(0, 0, 0, 255))
    d.rectangle((24, 44, 30, 63), fill=(40, 60, 160, 255), outline=(0, 0, 0, 255))
    d.rectangle((34, 44, 40, 63), fill=(40, 60, 160, 255), outline=(0, 0, 0, 255))
    return im


def test_style_lu_sur_des_sprites_agrandis(env):
    sprite = _sprite()
    a = cfg.DATA_DIR.parent / "a.png"
    pixel_art.agrandir(sprite, 2).save(a)
    b = cfg.DATA_DIR.parent / "b.png"  # agrandi ×2 avec un décalage d'un pixel, comme un des exemples réels
    grand = np.array(pixel_art.agrandir(sprite, 2))
    Image.fromarray(np.roll(grand, 1, axis=1)).save(b)
    style = pixel_art.analyser_style([str(a), str(b)])
    assert (style["toile"], style["agrandissement"], style["hauteur"]) == (64, 2, 60)
    assert style["contour"] == (0, 0, 0) and 4 <= style["couleurs"] <= 6
    assert "64×64" in pixel_art.resume_style(style)
    h, n, t, k, msg = pixel_art.lire_style([str(a)])
    assert (h, t, k) == (60, 64, 2) and "contour #000000" in msg
    with pytest.raises(gr.Error):
        pixel_art.lire_style(None)


def test_conversion_d_une_image_floue(env):
    """Image agrandie ×8 et floutée sur fond blanc → sprite 64×64 : pieds en bas, hauteur demandée, palette limitée,
    transparence franche, contour au bord de la silhouette ; le blanc enfermé dans le sujet reste opaque."""
    sprite = _sprite()
    fond = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
    grand = pixel_art.agrandir(sprite, 8)
    fond.alpha_composite(grand)
    ImageDraw.Draw(fond).rectangle((240, 260, 270, 290), fill=(255, 255, 255, 255))  # blanc dans le corps
    flou = fond.convert("RGB").filter(ImageFilter.GaussianBlur(2))
    r, infos = pixel_art.pixeliser(flou, 50, 12, 64, (16, 16, 24))
    a = np.array(r)
    assert r.size == (64, 64) and set(np.unique(a[..., 3])) == {0, 255}
    lignes = np.where(a[..., 3].any(axis=1))[0]
    assert lignes.max() == 63 and 49 <= lignes.max() - lignes.min() + 1 <= 51
    assert len({tuple(p) for p in a[..., :3][a[..., 3] > 0]}) <= 13  # 12 couleurs + le contour
    opaque = a[..., 3] > 0
    p = np.pad(opaque, 1)
    bord = opaque & ~(p[2:, 1:-1] & p[:-2, 1:-1] & p[1:-1, 2:] & p[1:-1, :-2])
    assert (a[..., :3][bord] == (16, 16, 24)).all()
    assert ((a[..., :3] > 200).all(-1) & (a[..., 3] == 255)).sum() >= 6  # blanc enfermé dans le corps : gardé
    assert len(infos["palette"]) <= 12 and len(infos["cadre"]) == 3


def test_convertir_depuis_l_onglet(env, monkeypatch):
    monkeypatch.setattr(diffusion, "present", lambda nom: False)
    src = cfg.DATA_DIR.parent / "Mon héros.png"
    fond = Image.new("RGB", (300, 400), (240, 240, 240))
    fond.paste(pixel_art.agrandir(_sprite(), 4).convert("RGB"), (22, 100), pixel_art.agrandir(_sprite(), 4))
    fond.save(src)
    exemple = cfg.DATA_DIR.parent / "ex.png"
    pixel_art.agrandir(_sprite(), 2).save(exemple)
    msg, apercus, fichiers, dossier = pixel_art.convertir([str(src)], [str(exemple)], 0, 0, 0, 3, True, True,
                                                         "héros", progress=no_progress)
    d = Path(dossier)
    assert d.parent == cfg.PIXEL_DIR and Image.open(apercus[0]).size == (192, 192)  # agrandissement imposé ×3
    assert Image.open(d / "Mon_heros.png").size == (64, 64) and "1 sprite" in msg
    assert json.loads((d / "style.json").read_text(encoding="utf-8"))["agrandissement"] == 3
    with pytest.raises(gr.Error, match="Ajoute"):
        pixel_art.convertir([], None, 0, 0, 0, 0, True, False, progress=no_progress)


def test_respiration_et_fichiers_de_l_animation(env):
    images = pixel_art.respiration(_sprite())
    assert len(images) == 30 and any((np.array(i) != np.array(images[0])).any() for i in images)
    pieds = [np.array(i)[60:] for i in images]
    assert all((p == pieds[0]).all() for p in pieds)  # les pieds ne bougent pas
    fichiers = pixel_art.ecrire_animation(images, env / "anim", "heros", 2)
    assert [f.name for f in fichiers] == ["heros.gif", "heros_planche.png", "heros_x2.gif", "heros_x2_planche.png"]
    gif = Image.open(fichiers[2])
    total = 0
    for i in range(gif.n_frames):
        gif.seek(i)
        total += gif.info["duration"]
    assert total == 5000 and gif.size == (128, 128) and gif.info.get("loop") == 0
    assert Image.open(fichiers[3]).size == (128 * 30, 128) and len(list((env / "anim" / "images").glob("*.png"))) == 30
    boucle = pixel_art.aller_retour(list(range(61)))
    assert len(boucle) == 30 and boucle[0] == 0 and boucle[15] == 60 and boucle[-1] == boucle[1]


def test_animation_d_apres_une_video(env, monkeypatch):
    """Wan 2.2 simulé : vidéo H.264 du sprite agrandi sur fond vif qui monte et descend ; chaque image revient sur la
    grille et la palette du sprite (fond retiré, bruit de compression écarté), première image = le sprite."""
    import av

    appels = []

    def faux_wan(prompt, sortie, image, largeur, hauteur, images, etapes, graine, progress=None):
        appels.append((prompt, largeur, hauteur, images, etapes))
        depart = np.array(Image.open(image).convert("RGB"))
        with av.open(str(sortie), "w") as c:
            v = c.add_stream("libx264", rate=24)
            v.width, v.height, v.pix_fmt = largeur, hauteur, "yuv420p"
            for i in range(images):
                d = int(round(11 * np.sin(2 * np.pi * i / images)))  # 11 px de vidéo = 1 pixel du sprite
                for paquet in v.encode(av.VideoFrame.from_ndarray(np.roll(depart, d, axis=0) if i else depart,
                                                                  format="rgb24")):
                    c.mux(paquet)
            for paquet in v.encode(None):
                c.mux(paquet)
        return {"sortie": str(sortie), "graine": graine}

    monkeypatch.setattr(diffusion, "video", faux_wan)
    monkeypatch.setattr(diffusion, "present", lambda nom: False)
    source = env / "sprite_x2.png"
    sprite = _sprite()
    pixel_art.agrandir(sprite, 2).save(source)  # importé agrandi : ramené à 64×64
    msg, gif, fichiers, dossier = pixel_art.animer(str(source), pixel_art.IA, "", 5, 20, "heros",
                                                   progress=no_progress)
    assert appels[0][1:] == (704, 704, 61, 20) and "30 images" in msg and gif.endswith("_x2.gif")
    images = [np.array(Image.open(f)) for f in sorted(Path(dossier, "images").glob("*.png"))]
    ref = np.array(sprite)
    assert len(images) == 30 and ((images[0][..., 3] > 0) == (ref[..., 3] > 0)).all()
    assert (images[0][..., :3][ref[..., 3] > 0] == ref[..., :3][ref[..., 3] > 0]).all()
    decale = np.roll(ref, 1, axis=0)  # image 5 : sprite descendu d'un pixel, exactement
    assert ((images[4][..., 3] > 0) == (decale[..., 3] > 0)).all()
    assert (images[4][..., :3][decale[..., 3] > 0] == decale[..., :3][decale[..., 3] > 0]).all()
    couleurs = {tuple(p) for i in images for p in i[..., :3][i[..., 3] > 0]}
    assert couleurs <= {tuple(p) for p in ref[..., :3][ref[..., 3] > 0]}  # aucune couleur hors palette
    with pytest.raises(gr.Error, match="sprite"):
        pixel_art.animer(None, progress=no_progress)
