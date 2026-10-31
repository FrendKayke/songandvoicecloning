"""Pixel art (onglet « Image et vidéo → Pixel art et sprites ») : une image convertie en pixel art, une personne d'une
photo transformée en personnage de jeu (sprite), et un sprite animé (30 images sur 5 s, en boucle).

Le style vient de sprites d'exemple (`analyser_style`) : taille de la grille d'origine (un sprite de 128 px fait de
carrés de 2×2 est un sprite de 64 px agrandi ×2), hauteur du personnage, nombre de couleurs, couleur du contour.
La conversion (`pixeliser`) ne demande aucun modèle : réduction par « couleur dominante » de chaque bloc (pas de
moyenne, qui donne des couleurs boueuses), palette limitée, transparence franche, contour sombre d'un pixel.
Personnage : FLUX.2 klein dessine la personne de la photo (modifications comprises) en pied sur fond uni, BiRefNet
retire le fond, puis `pixeliser`. Animation : « respiration » calculée (instantanée) ou Wan 2.2 (image → vidéo) puis
chaque image ramenée sur la grille et la palette du sprite ; aller-retour pour une boucle parfaite.
Rangement : data/pixel_art/<horodatage>_<nom>/ (hors galerie).
"""
import json
import math
import shutil
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from . import config as cfg
from . import diffusion
from .outils import nouveau_dossier, chemin_fichier
from .serie import _slug

IMAGES_ANIMATION = 30  # comme les sprites animés des jeux de la DS : 30 images, 5 s, en boucle
DUREE_ANIMATION = 5.0
ECHELLE_TRAVAIL = 4  # sous-pixels par pixel pour choisir la couleur dominante d'un bloc
CONTOUR_DEFAUT = (16, 16, 24)
STYLE_DEFAUT = {"toile": 64, "hauteur": 60, "couleurs": 20, "agrandissement": 2, "contour": CONTOUR_DEFAUT}


# --- Lecture du style de sprites d'exemple -------------------------------------------------------------------
def _facteur(a, avec_decalage=False):
    """Agrandissement entier d'un sprite (pixels répétés en carrés k×k, avec un décalage éventuel) : 1 à 8.
    avec_decalage : renvoie (k, (décalage vertical, décalage horizontal)) — un des sprites d'exemple de
    l'utilisateur commence au milieu d'un carré."""
    for k in range(8, 1, -1):
        if a.shape[0] < 2 * k or a.shape[1] < 2 * k:
            continue
        decalages = []
        for axe in (0, 1):
            scores = []
            for decalage in range(k):
                b = np.moveaxis(a, axe, 0)[decalage:]
                n = b.shape[0] // k * k
                b = b[:n].reshape(n // k, k, *b.shape[1:])
                scores.append(float((b == b[:, :1]).all(axis=-1).all(axis=1).mean()))
            decalages.append(int(np.argmax(scores)) if max(scores) > 0.99 else None)
        if None not in decalages:
            return (k, tuple(decalages)) if avec_decalage else k
    return (1, (0, 0)) if avec_decalage else 1


def reduire(a):
    """Pixel art agrandi → ses vrais pixels (un par carré, décalage compris). Renvoie (tableau, k, décalage)."""
    k, (dy, dx) = _facteur(a, avec_decalage=True)
    if k == 1:
        return a, 1, (0, 0)
    h, w = (a.shape[0] - dy) // k, (a.shape[1] - dx) // k
    return a[dy:dy + h * k:k, dx:dx + w * k:k], k, (dy, dx)


def remettre(petit, k, decalage, forme):
    """Inverse de `reduire` : agrandi ×k au plus proche et replacé au même décalage, dans une image de la taille
    d'origine (les bandes du bord recopient le pixel voisin) : une carte se superpose exactement à l'original."""
    grand = np.repeat(np.repeat(petit, k, axis=0), k, axis=1)
    dy, dx = decalage
    pad = [(dy, max(0, forme[0] - dy - grand.shape[0])), (dx, max(0, forme[1] - dx - grand.shape[1]))]
    pad += [(0, 0)] * (grand.ndim - 2)
    return np.pad(grand, pad, mode="edge")[:forme[0], :forme[1]]


def analyser_style(fichiers):
    """Style commun de sprites d'exemple : {toile, hauteur, couleurs, agrandissement, contour} (médianes)."""
    mesures = []
    for f in fichiers or []:
        im = Image.open(chemin_fichier(f)).convert("RGBA")
        a = np.array(im)
        k = _facteur(a)
        opaque = a[..., 3] >= 128
        if not opaque.any():
            continue
        lignes, colonnes = np.where(opaque)
        couleurs = {tuple(p) for p in a[..., :3][opaque]}
        # contour : couleur la plus fréquente parmi les pixels opaques qui touchent le vide
        bord = opaque & ~(np.pad(opaque, 1)[2:, 1:-1] & np.pad(opaque, 1)[:-2, 1:-1]
                          & np.pad(opaque, 1)[1:-1, 2:] & np.pad(opaque, 1)[1:-1, :-2])
        valeurs, nombres = np.unique(a[..., :3][bord].reshape(-1, 3), axis=0, return_counts=True)
        mesures.append({"toile": max(im.size) // k, "hauteur": (lignes.max() - lignes.min() + 1) // k,
                        "couleurs": len(couleurs), "agrandissement": k,
                        "contour": tuple(int(x) for x in valeurs[nombres.argmax()])})
    if not mesures:
        return dict(STYLE_DEFAUT)
    style = {c: int(np.median([m[c] for m in mesures])) for c in ("toile", "hauteur", "couleurs", "agrandissement")}
    contours = [m["contour"] for m in mesures]
    style["contour"] = max(set(contours), key=contours.count)
    return style


def resume_style(style):
    r, g, b = style["contour"]
    return (f"Sprites de {style['toile']}×{style['toile']} px (affichés ×{style['agrandissement']}), personnage de "
            f"{style['hauteur']} px de haut, {style['couleurs']} couleurs, contour #{r:02x}{g:02x}{b:02x}.")


# --- Conversion en pixel art ---------------------------------------------------------------------------------
def _sans_fond_uni(im, tolerance=24):
    """Image sans transparence : le fond uni (couleur des coins) devient transparent, s'il est bien uni."""
    a = np.array(im.convert("RGBA")).astype(int)
    coins = np.array([a[0, 0, :3], a[0, -1, :3], a[-1, 0, :3], a[-1, -1, :3]])
    if np.abs(coins - coins[0]).max() > tolerance:
        return im.convert("RGBA")
    fond = np.abs(a[..., :3] - coins.mean(axis=0)).max(axis=-1) <= tolerance
    # seulement le fond relié aux bords (pas le blanc d'un œil ou d'un vêtement)
    from collections import deque

    vu = np.zeros_like(fond)
    file = deque((y, x) for y in (0, a.shape[0] - 1) for x in range(a.shape[1]) if fond[y, x])
    file.extend((y, x) for x in (0, a.shape[1] - 1) for y in range(a.shape[0]) if fond[y, x])
    for y, x in file:
        vu[y, x] = True
    while file:
        y, x = file.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < a.shape[0] and 0 <= nx < a.shape[1] and fond[ny, nx] and not vu[ny, nx]:
                vu[ny, nx] = True
                file.append((ny, nx))
    a[..., 3] = np.where(vu, 0, a[..., 3])
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def _palette(pixels, n):
    """n couleurs représentatives (médiane coupée + k-moyennes de Pillow) de pixels RGB (N×3)."""
    n = max(2, min(64, int(n)))
    cote = int(math.ceil(math.sqrt(len(pixels))))
    bloc = np.zeros((cote * cote, 3), np.uint8)
    bloc[:len(pixels)] = pixels
    bloc[len(pixels):] = pixels[0] if len(pixels) else 0
    q = Image.fromarray(bloc.reshape(cote, cote, 3), "RGB").quantize(
        n, method=Image.Quantize.MEDIANCUT, kmeans=3, dither=Image.Dither.NONE)
    pal = np.array(q.getpalette()[:3 * n], np.uint8).reshape(-1, 3)
    return pal[np.unique(np.array(q).ravel())]


def _plus_proche(pixels, palette):
    d = ((pixels[:, None, :].astype(int) - palette[None, :, :].astype(int)) ** 2).sum(-1)
    return d.argmin(axis=1)


def pixeliser(image, hauteur=60, couleurs=20, toile=64, contour=CONTOUR_DEFAUT, palette=None, cadre=None):
    """Image (PIL, transparente ou sur fond uni) → sprite RGBA de toile×toile pixels : le sujet fait `hauteur`
    pixels de haut, posé en bas au centre, `couleurs` couleurs au plus (ou la `palette` donnée), transparence
    franche, contour d'un pixel (None : sans). cadre=(gauche, haut, échelle) impose la position (images d'une
    animation : même cadrage que le sprite). Renvoie (sprite, infos {palette, cadre})."""
    im = image.convert("RGBA")
    if np.array(im)[..., 3].min() == 255:
        im = _sans_fond_uni(im)
    a = np.array(im)
    opaque = a[..., 3] >= 128
    if not opaque.any():
        raise gr.Error("L'image est vide (rien d'opaque).")
    if cadre is None:
        lignes, colonnes = np.where(opaque)
        haut, bas, gauche, droite = lignes.min(), lignes.max() + 1, colonnes.min(), colonnes.max() + 1
        echelle = max(bas - haut, (droite - gauche) * hauteur / toile) / hauteur  # px source par px du sprite
        largeur_sprite = (droite - gauche) / echelle
        x0 = gauche - (toile - largeur_sprite) / 2 * echelle
        y0 = bas - toile * echelle  # pieds en bas de la toile
        cadre = (float(x0), float(y0), float(echelle))
    x0, y0, echelle = cadre
    # rééchantillonnage à 4 sous-pixels par pixel (moyenne locale), puis couleur dominante de chaque bloc
    n = toile * ECHELLE_TRAVAIL
    x1, y1 = x0 + toile * echelle, y0 + toile * echelle
    # cadre parfois hors de l'image (sujet étroit, pieds en bas) : bord transparent ajouté avant la réduction
    marge = int(math.ceil(max(0, -x0, -y0, x1 - im.width, y1 - im.height))) + 1
    grand = Image.new("RGBA", (im.width + 2 * marge, im.height + 2 * marge), (0, 0, 0, 0))
    grand.paste(im, (marge, marge))
    fin = np.array(grand.resize((n, n), Image.Resampling.BOX, box=(x0 + marge, y0 + marge, x1 + marge, y1 + marge)))
    fin_opaque = fin[..., 3] >= 128
    if palette is None:
        palette = _palette(fin[..., :3][fin_opaque], couleurs)
    palette = np.asarray(palette, np.uint8)
    sprite = contourer(blocs_dominants(fin, palette, toile, toile), contour)
    return Image.fromarray(sprite, "RGBA"), {"palette": palette.tolist(), "cadre": list(cadre)}


def grille_approchee(a, periodes=range(6, 33), seuil=40):
    """Grille de « gros pixels » d'une image générée (SDXL + LoRA de sprites dessine en pixels de ~10 à 20 px de
    côté). Mesure où tombent les changements de couleur : pour chaque taille p, la part des changements qui tombent
    sur une grille de pas p (avec son décalage). La plus grande taille qui en recueille presque autant que la
    meilleure (une grille de pas 8 recueille aussi tous les bords d'une grille de pas 16). Renvoie (p, décalage x,
    décalage y, part) ou None si aucune grille nette (au moins la moitié des changements dessus)."""
    rgb = a[..., :3].astype(int)
    op = a[..., 3] >= 128 if a.shape[-1] == 4 else np.ones(a.shape[:2], bool)
    bx = ((np.abs(rgb[:, 1:] - rgb[:, :-1]).sum(-1) > seuil) & (op[:, 1:] | op[:, :-1])).sum(0)
    by = ((np.abs(rgb[1:] - rgb[:-1]).sum(-1) > seuil) & (op[1:] | op[:-1])).sum(1)
    total = bx.sum() + by.sum()
    if total < 50:
        return None
    ix, iy = np.arange(1, len(bx) + 1), np.arange(1, len(by) + 1)  # bord entre i-1 et i
    scores = {}
    for p_ in periodes:
        # un pixel de marge : un bord flouté (image générée) s'étale sur deux colonnes
        cx = np.array([bx[ix % p_ == d].sum() for d in range(p_)])
        cy = np.array([by[iy % p_ == d].sum() for d in range(p_)])
        sx = cx + np.roll(cx, 1) + np.roll(cx, -1)
        sy = cy + np.roll(cy, 1) + np.roll(cy, -1)
        dx, dy = int(np.argmax(sx)), int(np.argmax(sy))
        # décalage = colonne où le bord est le plus franc dans la fenêtre
        dx = int((dx + np.argmax([cx[(dx - 1) % p_], cx[dx], cx[(dx + 1) % p_]]) - 1) % p_)
        dy = int((dy + np.argmax([cy[(dy - 1) % p_], cy[dy], cy[(dy + 1) % p_]]) - 1) % p_)
        scores[p_] = (min(1.0, (sx.max() + sy.max()) / total), dx, dy)
    meilleur = max(s[0] for s in scores.values())
    if meilleur < 0.5:
        return None
    p_ = max(q for q, s in scores.items() if s[0] >= 0.85 * meilleur)
    return p_, scores[p_][1], scores[p_][2], round(float(scores[p_][0]), 3)


def nettoyer(sprite, ecart=130):
    """Pixels isolés retirés : un pixel entouré sur au moins 3 côtés d'une même couleur proche (écart < `ecart`)
    prend cette couleur (mouchetures grises dans des cheveux blonds). Un détail franc (œil noir sur la peau, reflet
    blanc) est gardé : il est loin de la couleur qui l'entoure."""
    s = sprite.copy()
    h, w = s.shape[:2]
    a = sprite.astype(int)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if sprite[y, x, 3] == 0:
                continue
            v = [a[y - 1, x], a[y + 1, x], a[y, x - 1], a[y, x + 1]]
            if any(c[3] == 0 for c in v):
                continue
            c0 = a[y, x, :3]
            for c in v:
                pareil = sum(1 for d in v if (d[:3] == c[:3]).all())
                if pareil >= 3 and not (c[:3] == c0).all() and np.abs(c[:3] - c0).sum() < ecart:
                    s[y, x, :3] = c[:3]
                    break
    return s


def pixeliser_grille(image, couleurs=15, toile=64, contour=CONTOUR_DEFAUT, hauteur_min=32):
    """Sprite d'une image déjà dessinée en gros pixels (SDXL + LoRA) : un pixel par case de sa grille
    (`grille_approchee`, couleur médiane du centre de la case), sans rééchantillonnage — le réduire à une hauteur
    fixe mélangeait deux cases en une et déformait les yeux (sprite de l'utilisateur du 31/10). Palette de
    `couleurs` couleurs, pixels isolés nettoyés, contour, posé en bas au centre. None si l'image n'a pas de grille
    nette ou si le personnage ne tient pas dans la toile (l'appelant revient à `pixeliser`)."""
    im = image.convert("RGBA")
    if np.array(im)[..., 3].min() == 255:
        im = _sans_fond_uni(im)
    a = np.array(im)
    g = grille_approchee(a)
    if g is None:
        return None
    p_, dx, dy, part = g
    h, w = (a.shape[0] - dy) // p_, (a.shape[1] - dx) // p_
    if h < 4 or w < 4:
        return None
    cases = a[dy:dy + h * p_, dx:dx + w * p_].reshape(h, p_, w, p_, 4).transpose(0, 2, 1, 3, 4)
    q = max(1, p_ // 4)
    centre = cases[:, :, q:p_ - q, q:p_ - q].reshape(h, w, -1, 4)
    opaque = (cases[..., 3] >= 128).mean(axis=(2, 3)) > 0.5
    natif = np.zeros((h, w, 4), np.uint8)
    natif[..., :3] = np.median(centre[..., :3], axis=2).astype(np.uint8)
    natif[..., 3] = np.where(opaque, 255, 0)
    lignes, colonnes = np.where(opaque)
    if not len(lignes):
        return None
    natif = natif[lignes.min():lignes.max() + 1, colonnes.min():colonnes.max() + 1]
    hn, wn = natif.shape[:2]
    if hn > toile or wn > toile or hn < hauteur_min:
        return None
    op = natif[..., 3] > 0
    palette = np.asarray(_palette(natif[..., :3][op], couleurs), np.uint8)
    # détails rares mais francs (reflet blanc d'un œil, bouton rouge) : la palette les fondait dans la peau
    pixels = natif[..., :3][op].astype(int)
    for _ in range(4):
        ecarts = np.abs(pixels[:, None, :] - palette[None, :, :].astype(int)).sum(-1).min(1)
        if ecarts.max() < 120:
            break
        palette = np.vstack([palette, pixels[int(ecarts.argmax())].astype(np.uint8)])
    natif[op, :3] = palette[_plus_proche(natif[..., :3][op], palette)]
    natif = contourer(nettoyer(natif), contour)
    sprite = np.zeros((toile, toile, 4), np.uint8)
    x0 = (toile - wn) // 2
    sprite[toile - hn:, x0:x0 + wn] = natif
    return Image.fromarray(sprite, "RGBA"), {"palette": palette.tolist(), "grille": p_, "part": part,
                                             "taille": [wn, hn]}


def blocs_dominants(fin, palette, h, w):
    """Image RGBA de h×w blocs de k×k sous-pixels → sprite h×w : couleur de la palette la plus présente dans chaque
    bloc (pas la moyenne, qui donne des couleurs boueuses), transparent si plus de la moitié du bloc l'est."""
    k = fin.shape[0] // h
    opaque = fin[..., 3] >= 128
    indices = _plus_proche(fin[..., :3].reshape(-1, 3), palette).reshape(fin.shape[:2])
    blocs = np.where(opaque, indices, len(palette)).reshape(h, k, w, k).transpose(0, 2, 1, 3)
    comptes = np.zeros((h, w, len(palette) + 1), int)
    for i in range(len(palette) + 1):
        comptes[..., i] = (blocs == i).sum(axis=(2, 3))
    vide = comptes[..., -1] * 2 > k * k
    sprite = np.zeros((h, w, 4), np.uint8)
    sprite[..., :3] = palette[comptes[..., :-1].argmax(axis=-1)]
    sprite[..., 3] = np.where(vide, 0, 255)
    return sprite


def contourer(sprite, contour):
    """Contour d'un pixel (couleur `contour`, None : aucun) sur le bord de la silhouette."""
    if contour is not None:
        o = sprite[..., 3] > 0
        p = np.pad(o, 1)
        bord = o & ~(p[2:, 1:-1] & p[:-2, 1:-1] & p[1:-1, 2:] & p[1:-1, :-2])
        sprite[bord, :3] = contour
    return sprite


def agrandir(sprite, facteur):
    return sprite.resize((sprite.width * facteur, sprite.height * facteur), Image.Resampling.NEAREST)


# --- Animation -------------------------------------------------------------------------------------------------
def respiration(sprite, n=IMAGES_ANIMATION):
    """Animation calculée : le haut du corps descend d'un pixel puis remonte (respiration), en boucle. La ligne de la
    taille est doublée ou retirée, les pieds ne bougent pas."""
    a = np.array(sprite.convert("RGBA"))
    lignes = np.where(a[..., 3].any(axis=1))[0]
    haut, bas = lignes.min(), lignes.max()
    taille = int(haut + 0.55 * (bas - haut))  # au-dessus : tête et buste
    images = []
    for i in range(n):
        d = 1 if math.sin(2 * math.pi * i / n) > 0.35 else 0  # un tiers du temps « inspiré »
        b = a.copy()
        if d:
            b[haut + 1:taille + 1] = a[haut:taille]
            b[haut] = 0
        images.append(Image.fromarray(b, "RGBA"))
    return images


def _lisser_temps(indices):
    """Pixels qui clignotent d'une image à l'autre (bruit de la vidéo) : chaque pixel prend la valeur majoritaire
    sur trois images consécutives."""
    t = indices.copy()
    for i in range(1, len(indices) - 1):
        prec, cour, suiv = indices[i - 1], indices[i], indices[i + 1]
        t[i] = np.where(prec == suiv, prec, cour)
    return t


def aller_retour(images, n=IMAGES_ANIMATION):
    """n images en boucle parfaite : n/2 + 1 images à l'aller, puis le retour sans répéter les extrémités."""
    moitie = n // 2 + 1
    choisies = [images[round(i * (len(images) - 1) / (moitie - 1))] for i in range(moitie)]
    return choisies + choisies[-2:0:-1]


def ecrire_animation(images, dossier, nom, agrandissement, durees=None):
    """GIF transparent en boucle (5 s, ou une durée par image en ms : `durees`), planche de sprites (toutes les
    images côte à côte) et images séparées, à la taille d'origine et agrandies. Renvoie la liste des fichiers."""
    dossier = Path(dossier)
    (dossier / "images").mkdir(parents=True, exist_ok=True)
    duree_ms = list(durees) if durees else int(round(DUREE_ANIMATION * 1000 / len(images)))
    fichiers = []
    for facteur, suffixe in [(1, "")] + ([(agrandissement, f"_x{agrandissement}")] if agrandissement > 1 else []):
        cadres = [agrandir(im, facteur) for im in images]
        gif = dossier / f"{nom}{suffixe}.gif"
        _gif(cadres, gif, duree_ms)
        planche = Image.new("RGBA", (cadres[0].width * len(cadres), cadres[0].height))
        for i, c in enumerate(cadres):
            planche.paste(c, (i * c.width, 0))
        planche.save(dossier / f"{nom}{suffixe}_planche.png")
        fichiers += [gif, dossier / f"{nom}{suffixe}_planche.png"]
    for i, im in enumerate(images, 1):
        im.save(dossier / "images" / f"{nom}_{i:02d}.png")
    return fichiers


def _gif(images, chemin, duree_ms):
    """GIF animé à transparence franche (une couleur de palette réservée), effacement entre les images."""
    cadres = []
    for im in images:
        a = np.array(im.convert("RGBA"))
        rgb = Image.fromarray(a[..., :3], "RGB").quantize(255, method=Image.Quantize.MEDIANCUT,
                                                           dither=Image.Dither.NONE)
        p = np.array(rgb)
        p[a[..., 3] < 128] = 255
        q = Image.fromarray(p.astype(np.uint8), "P")
        q.putpalette(rgb.getpalette()[:765] + [0, 0, 0])
        q.info["transparency"] = 255
        cadres.append(q)
    cadres[0].save(chemin, save_all=True, append_images=cadres[1:], duration=duree_ms, loop=0, disposal=2,
                   transparency=255, optimize=False)


# --- Onglet : convertir des images -----------------------------------------------------------------------------
def _style(exemples, hauteur, couleurs, toile, agrandissement, contour):
    """Réglages de l'onglet ; 0 = repris des sprites d'exemple (ou des valeurs par défaut)."""
    style = analyser_style(exemples) if exemples else dict(STYLE_DEFAUT)
    for cle, valeur in (("hauteur", hauteur), ("couleurs", couleurs), ("toile", toile),
                        ("agrandissement", agrandissement)):
        if valeur and int(valeur) > 0:
            style[cle] = int(valeur)
    style["hauteur"] = min(style["hauteur"], style["toile"])
    if not contour:
        style["contour"] = None
    return style


def lire_style(exemples):
    if not exemples:
        raise gr.Error("Ajoute un ou plusieurs sprites d'exemple.")
    style = analyser_style(exemples)
    return (style["hauteur"], style["couleurs"], style["toile"], style["agrandissement"], resume_style(style))


def _transparent(im):
    """L'image a déjà des pixels transparents (canal alpha ou couleur transparente d'une palette)."""
    if im.mode not in ("RGBA", "LA") and "transparency" not in im.info:
        return False
    return im.convert("RGBA").getchannel("A").getextrema()[0] < 128


def _detourer(source, dossier, progress):
    """Fond retiré par BiRefNet si le modèle est là, sinon fond uni retiré par `pixeliser`."""
    if not diffusion.present("photo_detourage"):
        return Image.open(source)
    sortie = Path(dossier) / f"{Path(source).stem}_detoure.png"
    diffusion.detourer(source, sortie, "general", progress=progress)
    return Image.open(sortie)


def convertir(images, exemples, hauteur, couleurs, toile, agrandissement, contour, detourer, nom="pixel_art",
              progress=gr.Progress()):
    """Chaque image → sprite (taille d'origine et agrandi). Renvoie (message, aperçus, fichiers, dossier)."""
    images = [chemin_fichier(i) for i in (images or [])]
    if not images:
        raise gr.Error("Ajoute une ou plusieurs images à convertir.")
    style = _style(exemples, hauteur, couleurs, toile, agrandissement, contour)
    dossier = nouveau_dossier(cfg.PIXEL_DIR, _slug(nom, "pixel_art"))
    apercus, fichiers, erreurs = [], [], []
    for n, source in enumerate(images, 1):
        progress((n - 1) / len(images), desc=f"Image {n}/{len(images)}…")
        base = _slug(Path(source).stem, f"image_{n}")
        try:
            with Image.open(source) as im:
                im = ImageOps.exif_transpose(im)
                im.load()
            if detourer and not _transparent(im):  # sprite déjà détouré : BiRefNet adoucirait ses bords
                copie = dossier / f"{base}_source.png"
                im.save(copie)
                im = _detourer(copie, dossier, progress)
            sprite, _ = pixeliser(im, style["hauteur"], style["couleurs"], style["toile"], style["contour"])
        except (OSError, ValueError) as e:
            erreurs.append(f"{Path(source).name} : {e}")
            continue
        sprite.save(dossier / f"{base}.png")
        grand = dossier / f"{base}_x{style['agrandissement']}.png"
        agrandir(sprite, style["agrandissement"]).save(grand)
        apercus.append(str(grand))
        fichiers += [str(dossier / f"{base}.png"), str(grand)]
    (dossier / "style.json").write_text(json.dumps({**style, "contour": style["contour"]}, ensure_ascii=False),
                                        encoding="utf-8")
    if not apercus:
        raise gr.Error("Aucune image convertie : " + " ; ".join(erreurs))
    msg = f"✅ {len(apercus)} sprite(s) : {resume_style({**style, 'contour': style['contour'] or (0, 0, 0)})} Dossier : {dossier}"
    if erreurs:
        msg += "\n\n⚠️ " + " ; ".join(erreurs)
    return msg, apercus, fichiers, str(dossier)


# --- Onglet : personnages d'après une photo ------------------------------------------------------------------
# Consignes ajoutées au prompt de FLUX.2 klein (jamais de nom de jeu ou d'œuvre)
STYLE_SPRITE = ("Full body game character sprite in crisp retro pixel art, as in a 2D handheld role-playing game "
                "battle screen: the whole body from head to feet, three-quarter view, confident dynamic pose, clean "
                "dark one-pixel outline, flat cel shading with two or three tones per color, limited color palette, "
                "no anti-aliasing, centered on a plain flat white background, no shadow, no ground, no text.")
MEME_PERSONNE = ("The character is the person described above, taken from image 1: keep their face, hair, beard, "
                 "glasses, skin tone and body shape recognizable.")


def lignes_personnages(texte):
    """Une ligne = un personnage (« qui, et ce qu'on change »)."""
    return [l_.strip(" -•\t") for l_ in (texte or "").splitlines() if l_.strip(" -•\t")]


def prompt_personnage(photo, demande, progress=None):
    """Prompt anglais d'un personnage : Qwen voit la photo et décrit la personne visée avec les modifications ;
    sans Qwen, la demande est gardée telle quelle."""
    if diffusion.present("qwen"):
        texte = diffusion.decrire("sprite", demande, image=photo, progress=progress)
    else:
        texte = demande
    return f"{texte} {MEME_PERSONNE} {STYLE_SPRITE}"


def creer_personnages(photo, texte, exemples, hauteur, couleurs, toile, agrandissement, contour, graine,
                      nom="personnages", progress=gr.Progress()):
    """Un sprite par ligne du texte. Renvoie (message, aperçus, fichiers, dossier, dernier sprite)."""
    photo = chemin_fichier(photo)
    if not photo or not Path(photo).is_file():
        raise gr.Error("Ajoute la photo des personnes.")
    demandes = lignes_personnages(texte)
    if not demandes:
        raise gr.Error("Écris une ligne par personnage : qui (« l'homme barbu à lunettes ») et ce qu'on change "
                       "(« en mage avec une cape violette »).")
    diffusion.verifier("personnages")
    style = _style(exemples, hauteur, couleurs, toile, agrandissement, contour)
    dossier = nouveau_dossier(cfg.PIXEL_DIR, _slug(nom, "personnages"))
    copie = dossier / f"photo{Path(photo).suffix.lower() or '.png'}"
    shutil.copy(photo, copie)
    graine = int(graine or 0) or diffusion.graines(1)[0]
    debut = time.monotonic()
    prompts = []
    for i, demande in enumerate(demandes, 1):
        progress(0.02, desc=f"Description du personnage {i}/{len(demandes)} (Qwen3-VL)…")
        prompts.append(prompt_personnage(copie, demande, progress=progress))
    sources = [dossier / f"personnage_{i}_source.png" for i in range(1, len(demandes) + 1)]
    diffusion.personnage(prompts, [copie], sources, [graine + i for i in range(len(demandes))], 768, 1024,
                         progress=progress)
    if diffusion.present("photo_detourage"):
        detoures = [s.with_name(s.stem.replace("_source", "_detoure") + ".png") for s in sources]
        diffusion.detourer_lot(sources, detoures, progress=progress)
        # une image en échec dans le lot : la source (fond uni retiré par pixeliser)
        detoures = [d if d.exists() else s for d, s in zip(detoures, sources)]
    else:
        detoures = sources
    apercus, fichiers, infos = [], [], []
    for i, (demande, prompt, detoure) in enumerate(zip(demandes, prompts, detoures), 1):
        sprite, _ = pixeliser(Image.open(detoure), style["hauteur"], style["couleurs"], style["toile"],
                              style["contour"])
        sprite.save(dossier / f"personnage_{i}.png")
        grand = dossier / f"personnage_{i}_x{style['agrandissement']}.png"
        agrandir(sprite, style["agrandissement"]).save(grand)
        apercus.append(str(grand))
        fichiers += [str(dossier / f"personnage_{i}.png"), str(grand)]
        infos.append({"demande": demande, "prompt": prompt, "graine": graine + i - 1})
    (dossier / "personnages.json").write_text(json.dumps({"style": style, "personnages": infos}, ensure_ascii=False,
                                                         indent=1), encoding="utf-8")
    msg = (f"✅ {len(apercus)} personnage(s) en {time.monotonic() - debut:.0f} s, graines {graine}…"
           f"{graine + len(apercus) - 1}. Dossier : {dossier}. Clique sur un sprite pour l'animer.")
    return msg, apercus, fichiers, str(dossier), str(dossier / "personnage_1.png")


# --- Onglet : animer un sprite ---------------------------------------------------------------------------------
SIMPLE = "Respiration (instantané)"
IA = "Mouvement libre avec Wan 2.2 (~10 min)"
MODES_ANIMATION = [SIMPLE, IA]
MOUVEMENT_DEFAUT = ("The pixel art character stays in place in an idle stance and breathes slowly, small natural "
                    "movements of the head, arms, hair and clothes, the feet do not move, the camera does not move, "
                    "plain flat background")
TAILLE_VIDEO = 704  # côté de la vidéo carrée de Wan (multiple de 32)
# Essai de l'utilisateur sur la RTX 4070 (24/10) : avec un geste, Wan reculait à la fin (personnage rapetissé dans la
# pose finale tenue) ; sans mouvement, il transformait le dresseur en un autre personnage dès le début (le prompt ne
# décrivait pas le personnage). Images coupées au premier recul ou changement de personnage.
RECUL_MAX = 0.12  # variation de la hauteur du personnage (zoom de la caméra) au-delà de laquelle la suite est coupée
RESSEMBLANCE_MIN = 0.6  # part des pixels du personnage restés de la même couleur que le sprite (sans mouvement)
IMAGES_UTILES_MIN = 8  # en dessous, la vidéo est inutilisable (repli sur la respiration pour une boucle)
# Action décrite (« il va chercher une peluche et la câline », essai de l'utilisateur du 23/10 : le personnage restait
# figé) : la consigne « reste sur place, pieds immobiles » n'est plus ajoutée, le personnage peut aller partout dans
# l'image, un objet qui apparaît garde ses couleurs, la vidéo dure 5 s et se joue une fois (pas d'aller-retour)
STYLE_ACTION = ("2D pixel art video game sprite animation: the same character with exactly the same colors, clothes "
                "and proportions as in the first frame, crisp pixels, the whole action fully visible, the camera "
                "does not move")
IMAGES_VIDEO = 61  # 2,5 s de vidéo : un geste de dresseur dure ~2 s
# Rythme des sprites animés de dresseurs (187 GIF d'exemple de l'utilisateur, 24/10) : médiane 23 images sur 2,1 s,
# 50 ms par image au milieu, première image tenue ~150 ms, dernière 600 ms (pose finale). Lu sur des GIF d'exemple
# s'il y en a (`lire_rythme`). L'ancien rythme (30 images sur 5 s, 6 images/s) était saccadé.
RYTHME_DEFAUT = {"image_ms": 50, "debut_ms": 150, "fin_ms": 600, "duree_ms": 2100}
ECART_FOND_NOUVELLE = 140  # une « nouvelle » couleur aussi proche du fond vif est un reste du fond (liseré vert)
COULEURS_EN_PLUS = 8  # couleurs ajoutées à la palette du sprite pour un objet qui apparaît
ECART_NOUVELLE = 60  # écart à la palette au-delà duquel une couleur est nouvelle
# Wan changeait la couleur du fond en cours de vidéo (gris, brun : constaté chez l'utilisateur le 23/10) : la couleur
# est nommée dans le prompt, le changement interdit par le négatif, et le fond est retiré image par image
NOMS_FONDS = {(0, 255, 0): "bright green", (255, 0, 255): "bright magenta", (0, 128, 255): "bright blue",
              (255, 255, 0): "bright yellow"}
FOND_FIXE = ("The plain flat solid {couleur} background stays exactly the same color in every frame: no floor, no "
             "shadow, no scenery, no lighting change")
NEGATIF_SPRITE = ("background changing color, gray background, floor, ground, shadow, scenery, room, landscape, "
                  "scene transition, camera movement, zoom")
MARGE_MOUVEMENT = 6  # pixels du sprite autour de sa silhouette où le personnage peut bouger ; au-delà : du fond
ECART_FOND = 60  # écart de couleur (0–441) en dessous duquel un pixel de la vidéo est pris pour du fond


def _sprite_de(source):
    """Sprite à la taille d'origine : une image agrandie (carrés k×k) est ramenée à 1 pixel par carré."""
    a, k, _ = reduire(np.array(Image.open(chemin_fichier(source)).convert("RGBA")))
    return Image.fromarray(np.ascontiguousarray(a), "RGBA"), k


def _fond_eloigne(palette):
    """Couleur de fond la plus éloignée des couleurs du sprite (vert, magenta ou bleu vif) : retirée sans erreur."""
    candidates = np.array([(0, 255, 0), (255, 0, 255), (0, 128, 255), (255, 255, 0)])
    d = ((candidates[:, None, :] - np.asarray(palette)[None, :, :].astype(int)) ** 2).sum(-1).min(axis=1)
    return tuple(int(x) for x in candidates[d.argmax()])


def lire_rythme(fichiers):
    """Rythme de GIF animés d'exemple : {image_ms, debut_ms, fin_ms, duree_ms} (médianes), ou None sans GIF animé."""
    mesures = []
    for f in fichiers or []:
        try:
            with Image.open(chemin_fichier(f)) as im:
                if getattr(im, "n_frames", 1) < 3:
                    continue
                d = []
                for i in range(im.n_frames):
                    im.seek(i)
                    d.append(int(im.info.get("duration") or 100))
        except OSError:
            continue
        mesures.append({"image_ms": float(np.median(d[1:-1])), "debut_ms": d[0], "fin_ms": d[-1], "duree_ms": sum(d)})
    if not mesures:
        return None
    return {c: int(np.median([m[c] for m in mesures])) for c in RYTHME_DEFAUT}


def durees_rythme(n, rythme):
    """Durée de chaque image (ms) : première et dernière tenues, les autres au pas du rythme."""
    if n == 1:
        return [rythme["fin_ms"]]
    return [rythme["debut_ms"]] + [rythme["image_ms"]] * (n - 2) + [rythme["fin_ms"]]


def animer(sprite_source, mode=SIMPLE, mouvement="", graine=0, etapes=25, nom="animation", exemples=None,
           progress=gr.Progress()):
    """Respiration : 30 images en boucle sur 5 s. Wan 2.2 : au rythme des GIF d'exemple (sinon celui des dresseurs
    animés : 50 ms par image). Renvoie (message, GIF agrandi, fichiers, dossier)."""
    if not sprite_source:
        raise gr.Error("Choisis le sprite à animer (créé ici ou importé).")
    sprite, k = _sprite_de(sprite_source)
    k = k if k > 1 else 2
    a = np.array(sprite)
    opaque = a[..., 3] > 0
    if not opaque.any():
        raise gr.Error("Le sprite est vide.")
    dossier = nouveau_dossier(cfg.PIXEL_DIR, _slug(nom, "animation"))
    sprite.save(dossier / "sprite.png")
    debut = time.monotonic()
    durees, rythme_lu = None, None
    remarque = ""
    if mode != IA:
        images = respiration(sprite)
    else:
        rythme_lu = lire_rythme(exemples)
        images, durees, remarque = _animer_ia(sprite, dossier, mouvement, graine, etapes, progress,
                                              rythme_lu or RYTHME_DEFAUT)
    fichiers = ecrire_animation(images, dossier, _slug(nom, "animation"), k, durees)
    gif = str([f for f in fichiers if f.suffix == ".gif"][-1])
    total = sum(durees) / 1000 if durees else DUREE_ANIMATION
    lecture = "jouée une fois puis reprise" if mode == IA and (mouvement or "").strip() else "en boucle"
    origine = (" (rythme lu sur les GIF d'exemple)" if rythme_lu else
               " (rythme des dresseurs animés : 50 ms par image)" if durees else "")
    msg = (f"✅ Animation de {len(images)} images sur {total:g} s{origine}, {lecture}, en "
           f"{time.monotonic() - debut:.0f} s : GIF, planche de sprites et images séparées dans {dossier}.")
    if remarque:
        msg += f"\n\n⚠️ {remarque}"
    return msg, gif, [str(f) for f in fichiers], str(dossier)


def _animer_ia(sprite, dossier, mouvement, graine, etapes, progress, rythme=RYTHME_DEFAUT):
    """Wan 2.2 anime le sprite agrandi sur un fond de couleur vive ; chaque image est ramenée sur la grille, la
    palette et le cadrage du sprite (le fond retombe sur sa couleur et redevient transparent)."""
    a = np.array(sprite)
    opaque = a[..., 3] > 0
    palette = np.unique(a[..., :3][opaque].reshape(-1, 3), axis=0)
    fond = _fond_eloigne(palette)
    toile = max(sprite.size)
    facteur = TAILLE_VIDEO // toile
    depart = Image.new("RGB", (TAILLE_VIDEO, TAILLE_VIDEO), fond)
    marge = (TAILLE_VIDEO - toile * facteur) // 2
    depart.paste(agrandir(sprite, facteur), (marge, marge), agrandir(sprite, facteur))
    depart.save(dossier / "depart.png")
    action = bool((mouvement or "").strip())
    if action:  # la consigne d'immobilité contredisait l'action demandée
        texte = f"{mouvement.strip()}. {STYLE_ACTION}"
        prompt = diffusion.decrire("video", texte, image=dossier / "depart.png", progress=progress) \
            if diffusion.present("qwen") else texte
    else:  # Qwen décrit aussi le personnage : sans description, Wan en inventait un autre (constaté)
        texte = f"{MOUVEMENT_DEFAUT}. {STYLE_ACTION}"
        prompt = diffusion.decrire("video", texte, image=dossier / "depart.png", progress=progress) \
            if diffusion.present("qwen") else texte
    prompt = f"{prompt.rstrip('. ')}. {FOND_FIXE.format(couleur=NOMS_FONDS.get(fond, 'bright'))}."
    graine = int(graine or 0) or diffusion.graines(1)[0]
    res = diffusion.video(prompt, dossier / "video.mp4", dossier / "depart.png", TAILLE_VIDEO, TAILLE_VIDEO,
                          IMAGES_VIDEO, etapes, graine, progress=progress, negatif_ajout=NEGATIF_SPRITE)
    trames = _lire_video(res["sortie"])
    cadre = (marge, marge, facteur)
    # une action peut emmener le personnage partout dans l'image ; une boucle reste près de la silhouette
    zone = (np.ones(trames[0].shape[:2], bool) if action
            else _zone_du_personnage(opaque, facteur, marge, trames[0].shape[:2]))
    masques = _masques_birefnet(trames, dossier, progress)
    alphas = [_alpha_personnage(t, zone, fond, None if masques is None else masques[n]) for n, t in enumerate(trames)]
    trames = _corriger_couleurs(trames, alphas)  # Wan assombrissait le personnage (pantalon gris devenu noir)
    if action:  # un objet qui apparaît (une peluche jaune) garde ses couleurs
        palette = _palette_avec_nouvelles(palette, trames, alphas, fond, facteur)
    palette_fond = np.vstack([palette, np.zeros((1, 3), np.uint8)])  # dernier indice = transparent
    indices = []
    for t, alpha in zip(trames, alphas):
        im = Image.fromarray(np.dstack([t, alpha]), "RGBA")
        # fond déjà transparent ; le reste prend la couleur du sprite la plus proche (bruit de compression écarté)
        if not np.array(im)[..., 3].any():  # image sans personnage : reprend la précédente
            indices.append(indices[-1] if indices else np.full((toile, toile), len(palette)))
            continue
        s_ = np.array(pixeliser(im, toile, 0, toile, None, palette=palette, cadre=cadre)[0])
        ind = _plus_proche(s_[..., :3].reshape(-1, 3), palette).reshape(toile, toile)
        indices.append(np.where(s_[..., 3] == 0, len(palette), ind))
    indices = _lisser_temps(np.array(indices))
    indices[0] = _plus_proche(a[..., :3].reshape(-1, 3), palette_fond).reshape(toile, toile)
    indices[0][~opaque] = len(palette)
    total_video = len(indices)
    fin = _fin_utilisable(indices, len(palette), action)
    indices = indices[:fin]
    if not action and fin < IMAGES_UTILES_MIN:
        (dossier / "animation.json").write_text(json.dumps({"prompt": prompt, "graine": res["graine"],
                                                            "repli": "respiration"}, ensure_ascii=False),
                                                encoding="utf-8")
        return respiration(sprite), None, ("Wan a changé le personnage dès le début de la vidéo : animation "
                                           "remplacée par la respiration (réessaie avec une autre graine).")
    remarque = (f"Vidéo coupée à {fin}/{total_video} images (recul de la caméra ou personnage changé)."
                if fin < total_video else "")
    images = []
    for ind in indices:
        b = np.zeros((toile, toile, 4), np.uint8)
        b[..., :3] = palette_fond[ind]
        b[..., 3] = np.where(ind == len(palette), 0, 255)
        images.append(Image.fromarray(b, "RGBA"))
    (dossier / "animation.json").write_text(json.dumps({"prompt": prompt, "graine": res["graine"], "fond": fond,
                                                        "action": action, "couleurs": len(palette),
                                                        "rythme": rythme, "images_gardees": fin,
                                                        "images_video": total_video},
                                                       ensure_ascii=False), encoding="utf-8")
    # autant d'images que le rythme en demande sur la durée du geste ; les images identiques à la suivante sont
    # fusionnées (leur durée s'additionne), comme les poses tenues des sprites d'exemple
    # au plus une image par image de la vidéo (des exemples plus longs, comme les Pokémon en boucle de 5,8 s,
    # donneraient sinon un ralenti fait d'images répétées)
    # (la première et la dernière image sont tenues plus longtemps : elles comptent pour leur durée ; avant, le GIF
    # durait 3,15 s au lieu de 2,35 s chez l'utilisateur)
    milieu = max(0, rythme["duree_ms"] - rythme["debut_ms"] - rythme["fin_ms"])
    if action:
        n = max(4, min(len(images), round(milieu / rythme["image_ms"]) + 2))
        choisies = une_fois(images, n)
        durees = durees_rythme(len(choisies), rythme)
    else:  # boucle : même durée que le geste, aller puis retour
        n = max(4, round(rythme["duree_ms"] / rythme["image_ms"]))
        choisies = aller_retour(images, n)
        durees = [rythme["image_ms"]] * len(choisies)
    return (*_fusionner_identiques(choisies, durees), remarque)


def _hauteur(ind, vide):
    lignes = np.where((ind != vide).any(axis=1))[0]
    return int(lignes[-1] - lignes[0] + 1) if len(lignes) else 0


def _fin_utilisable(indices, vide, action):
    """Nombre d'images à garder depuis le début : jusqu'au premier recul ou rapprochement de la caméra (hauteur du
    personnage qui change de plus de RECUL_MAX pendant 3 images de suite, ou jusqu'à la fin) et, sans mouvement,
    jusqu'au premier changement de personnage (moins de RESSEMBLANCE_MIN des pixels de la même couleur que le
    sprite)."""
    h0 = _hauteur(indices[0], vide) or 1
    ref = indices[0]
    mauvais = []
    for ind in indices:
        mal = abs(_hauteur(ind, vide) - h0) > RECUL_MAX * h0
        if not action:
            union = (ind != vide) | (ref != vide)
            mal = mal or bool(union.any() and (ind[union] == ref[union]).mean() < RESSEMBLANCE_MIN)
        mauvais.append(mal)
    for i in range(1, len(indices)):
        if all(mauvais[i:i + 3]):  # trois images de suite (ou toutes celles qui restent)
            return i
    return len(indices)


def _fusionner_identiques(images, durees):
    sorties, temps = [images[0]], [durees[0]]
    for im, d in zip(images[1:], durees[1:]):
        if np.array_equal(np.array(im), np.array(sorties[-1])):
            temps[-1] += d
        else:
            sorties.append(im)
            temps.append(d)
    return sorties, temps


def une_fois(images, n=IMAGES_ANIMATION):
    """n images réparties sur toute la vidéo, dans l'ordre : l'action est jouée une fois (puis le GIF recommence)."""
    return [images[round(i * (len(images) - 1) / (n - 1))] for i in range(n)]


def _corriger_couleurs(trames, alphas):
    """Couleurs du personnage ramenées à celles de la première image (le sprite) : moyenne et écart type de chaque
    canal sur les pixels du personnage, correction bornée (un objet qui apparaît change un peu ces mesures)."""
    ref = trames[0][alphas[0] > 0].astype(np.float32)
    if len(ref) < 16:
        return trames
    m0, s0 = ref.mean(axis=0), ref.std(axis=0) + 1
    sorties = [trames[0]]
    for t, alpha in zip(trames[1:], alphas[1:]):
        px = t[alpha > 0].astype(np.float32)
        if len(px) < 16:
            sorties.append(t)
            continue
        m, s_ = px.mean(axis=0), px.std(axis=0) + 1
        gain = np.clip(s0 / s_, 0.75, 1.33)
        decalage = np.clip(m0 - m * gain, -40, 40)
        sorties.append(np.clip(t.astype(np.float32) * gain + decalage, 0, 255).astype(np.uint8))
    return sorties


def _palette_avec_nouvelles(palette, trames, alphas, fond=None, facteur=1):
    """Palette du sprite + jusqu'à COULEURS_EN_PLUS couleurs nouvelles (pixels du personnage loin de toutes les
    couleurs du sprite sur au moins 0,5 % des pixels : un objet, pas du bruit de compression). Le bord de la
    silhouette (mélangé au fond par la vidéo) et les couleurs proches du fond vif sont écartés : sinon un liseré vert
    devenait une couleur et teintait le pantalon et la tête (constaté chez l'utilisateur le 24/10)."""
    from scipy import ndimage

    marge = max(1, facteur // 2)
    interieurs = [ndimage.binary_erosion(a > 0, iterations=marge) for a in alphas[::4]]
    pixels = np.concatenate([t[m] for t, m in zip(trames[::4], interieurs)])
    if fond is not None and len(pixels):
        pixels = pixels[np.sqrt(((pixels.astype(int) - np.array(fond)) ** 2).sum(-1)) > ECART_FOND_NOUVELLE]
    pixels = pixels[::max(1, len(pixels) // 200_000)]  # 200 000 pixels suffisent (sinon plusieurs Go de calcul)
    if not len(pixels):
        return palette
    d = np.sqrt(((pixels[:, None, :].astype(int) - palette[None, :, :].astype(int)) ** 2).sum(-1)).min(axis=1)
    loin = pixels[d > ECART_NOUVELLE]
    if len(loin) < max(50, 0.005 * len(pixels)):
        return palette
    return np.vstack([palette, _palette(loin, COULEURS_EN_PLUS)]).astype(np.uint8)


def _zone_du_personnage(opaque, facteur, marge, forme):
    """Où le personnage peut être dans la vidéo : sa silhouette élargie de MARGE_MOUVEMENT pixels du sprite (une
    idle ou un geste ne le déplace pas plus loin). Tout ce qui est au-delà est du fond, quelle que soit sa couleur."""
    from scipy import ndimage

    large = ndimage.binary_dilation(opaque, iterations=MARGE_MOUVEMENT)
    grand = np.repeat(np.repeat(large, facteur, axis=0), facteur, axis=1)
    zone = np.zeros(forme, bool)
    h, w = min(grand.shape[0], forme[0] - marge), min(grand.shape[1], forme[1] - marge)
    zone[marge:marge + h, marge:marge + w] = grand[:h, :w]
    return zone


def _masques_birefnet(trames, dossier, progress):
    """Silhouette de chaque image de la vidéo par BiRefNet s'il est installé (un fond devenu gris ou brun ne se
    distingue pas d'un pantalon gris par la couleur) ; None sinon (repli : écart à la couleur du fond)."""
    if not diffusion.present("photo_detourage"):
        return None
    d = Path(dossier) / "trames"
    d.mkdir(exist_ok=True)
    entrees, sorties = [], []
    for n, t in enumerate(trames):
        entrees.append(d / f"{n:03d}.png")
        sorties.append(d / f"{n:03d}_detouree.png")
        Image.fromarray(t, "RGB").save(entrees[-1])
    try:
        diffusion.detourer_lot(entrees, sorties, progress=progress)
        # image en échec dans le lot : None, son fond est retiré d'après sa couleur
        return [np.array(Image.open(s).convert("RGBA"))[..., 3] >= 128 if s.exists() else None for s in sorties]
    except (gr.Error, RuntimeError, OSError) as e:
        print(f"Détourage des images de l'animation impossible ({e}) : fond retiré d'après sa couleur.", flush=True)
        return None


def _alpha_personnage(trame, zone, fond, masque=None):
    """Canal alpha d'une image de la vidéo : personnage dans la zone permise ; fond = hors de la zone, ou proche de
    la couleur vive demandée, ou de la couleur que le fond a prise dans cette image (médiane hors de la zone)."""
    if masque is not None:
        garde = masque
    else:
        t = trame.astype(int)
        actuel = np.median(trame[~zone], axis=0) if (~zone).any() else np.array(fond)
        ecart = np.minimum(np.sqrt(((t - np.array(fond)) ** 2).sum(-1)), np.sqrt(((t - actuel) ** 2).sum(-1)))
        garde = ecart > ECART_FOND
    return np.where(garde & zone, 255, 0).astype(np.uint8)


def _lire_video(chemin):
    import av

    with av.open(str(chemin)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
