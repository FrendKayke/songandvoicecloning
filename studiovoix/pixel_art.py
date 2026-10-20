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
from .outils import nouveau_dossier
from .serie import _slug

IMAGES_ANIMATION = 30  # comme les sprites animés des jeux de la DS : 30 images, 5 s, en boucle
DUREE_ANIMATION = 5.0
ECHELLE_TRAVAIL = 4  # sous-pixels par pixel pour choisir la couleur dominante d'un bloc
CONTOUR_DEFAUT = (16, 16, 24)
STYLE_DEFAUT = {"toile": 64, "hauteur": 60, "couleurs": 20, "agrandissement": 2, "contour": CONTOUR_DEFAUT}


# --- Lecture du style de sprites d'exemple -------------------------------------------------------------------
def _facteur(a):
    """Agrandissement entier d'un sprite (pixels répétés en carrés k×k, avec un décalage éventuel) : 1 à 8."""
    for k in range(8, 1, -1):
        if a.shape[0] < 2 * k or a.shape[1] < 2 * k:
            continue
        ok = True
        for axe in (0, 1):
            meilleur = 0.0
            for decalage in range(k):
                b = np.moveaxis(a, axe, 0)[decalage:]
                n = b.shape[0] // k * k
                b = b[:n].reshape(n // k, k, *b.shape[1:])
                meilleur = max(meilleur, float((b == b[:, :1]).all(axis=-1).all(axis=1).mean()))
            ok = ok and meilleur > 0.97
        if ok:
            return k
    return 1


def analyser_style(fichiers):
    """Style commun de sprites d'exemple : {toile, hauteur, couleurs, agrandissement, contour} (médianes)."""
    mesures = []
    for f in fichiers or []:
        im = Image.open(getattr(f, "name", f)).convert("RGBA")
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
    indices = _plus_proche(fin[..., :3].reshape(-1, 3), palette).reshape(n, n)
    k = ECHELLE_TRAVAIL
    blocs = np.where(fin_opaque, indices, len(palette)).reshape(toile, k, toile, k).transpose(0, 2, 1, 3)
    comptes = np.zeros((toile, toile, len(palette) + 1), int)
    for i in range(len(palette) + 1):
        comptes[..., i] = (blocs == i).sum(axis=(2, 3))
    vide = comptes[..., -1] * 2 > k * k  # plus de la moitié du bloc transparente
    choix = comptes[..., :-1].argmax(axis=-1)
    sprite = np.zeros((toile, toile, 4), np.uint8)
    sprite[..., :3] = palette[choix]
    sprite[..., 3] = np.where(vide, 0, 255)
    if contour is not None:
        o = sprite[..., 3] > 0
        p = np.pad(o, 1)
        bord = o & ~(p[2:, 1:-1] & p[:-2, 1:-1] & p[1:-1, 2:] & p[1:-1, :-2])
        sprite[bord, :3] = contour
    return Image.fromarray(sprite, "RGBA"), {"palette": palette.tolist(), "cadre": list(cadre)}


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


def ecrire_animation(images, dossier, nom, agrandissement):
    """GIF transparent en boucle (5 s), planche de sprites (toutes les images côte à côte) et images séparées, à la
    taille d'origine et agrandies. Renvoie la liste des fichiers."""
    dossier = Path(dossier)
    (dossier / "images").mkdir(parents=True, exist_ok=True)
    duree_ms = int(round(DUREE_ANIMATION * 1000 / len(images)))
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
    images = [getattr(i, "name", i) for i in (images or [])]
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
            if detourer:
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
    photo = getattr(photo, "name", photo)
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


def _sprite_de(source):
    """Sprite à la taille d'origine : une image agrandie (carrés k×k) est ramenée à 1 pixel par carré."""
    im = Image.open(getattr(source, "name", source)).convert("RGBA")
    k = _facteur(np.array(im))
    if k > 1:
        im = im.resize((im.width // k, im.height // k), Image.Resampling.NEAREST)
    return im, k


def _fond_eloigne(palette):
    """Couleur de fond la plus éloignée des couleurs du sprite (vert, magenta ou bleu vif) : retirée sans erreur."""
    candidates = np.array([(0, 255, 0), (255, 0, 255), (0, 128, 255), (255, 255, 0)])
    d = ((candidates[:, None, :] - np.asarray(palette)[None, :, :].astype(int)) ** 2).sum(-1).min(axis=1)
    return tuple(int(x) for x in candidates[d.argmax()])


def animer(sprite_source, mode=SIMPLE, mouvement="", graine=0, etapes=25, nom="animation", progress=gr.Progress()):
    """30 images en boucle sur 5 s. Renvoie (message, GIF agrandi, fichiers, dossier)."""
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
    if mode != IA:
        images = respiration(sprite)
    else:
        images = _animer_ia(sprite, dossier, mouvement, graine, etapes, progress)
    fichiers = ecrire_animation(images, dossier, _slug(nom, "animation"), k)
    gif = str([f for f in fichiers if f.suffix == ".gif"][-1])
    msg = (f"✅ Animation de {len(images)} images sur {DUREE_ANIMATION:g} s ({len(images) / DUREE_ANIMATION:g} "
           f"images/s), en boucle, en {time.monotonic() - debut:.0f} s : GIF, planche de sprites et images séparées "
           f"dans {dossier}.")
    return msg, gif, [str(f) for f in fichiers], str(dossier)


def _animer_ia(sprite, dossier, mouvement, graine, etapes, progress):
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
    prompt = MOUVEMENT_DEFAUT
    if (mouvement or "").strip():
        prompt = diffusion.decrire("video", mouvement.strip() + ". " + MOUVEMENT_DEFAUT, image=dossier / "depart.png",
                                   progress=progress) if diffusion.present("qwen") else mouvement.strip()
    graine = int(graine or 0) or diffusion.graines(1)[0]
    # 2,5 s suffisent : l'aller-retour donne les 5 s de la boucle
    res = diffusion.video(prompt, dossier / "video.mp4", dossier / "depart.png", TAILLE_VIDEO, TAILLE_VIDEO, 61,
                          etapes, graine, progress=progress)
    trames = _lire_video(res["sortie"])
    palette_fond = np.vstack([palette, np.zeros((1, 3), np.uint8)])  # dernier indice = transparent
    cadre = (marge, marge, facteur)
    indices = []
    for t in trames:
        im = Image.fromarray(t, "RGB").convert("RGBA")
        # le fond uni (couleur vive) devient transparent dans pixeliser ; le reste prend la couleur du sprite la plus
        # proche (bruit de compression de la vidéo écarté)
        s_ = np.array(pixeliser(im, toile, 0, toile, None, palette=palette, cadre=cadre)[0])
        ind = _plus_proche(s_[..., :3].reshape(-1, 3), palette).reshape(toile, toile)
        indices.append(np.where(s_[..., 3] == 0, len(palette), ind))
    indices = _lisser_temps(np.array(indices))
    indices[0] = _plus_proche(a[..., :3].reshape(-1, 3), palette_fond).reshape(toile, toile)
    indices[0][~opaque] = len(palette)
    images = []
    for ind in indices:
        b = np.zeros((toile, toile, 4), np.uint8)
        b[..., :3] = palette_fond[ind]
        b[..., 3] = np.where(ind == len(palette), 0, 255)
        images.append(Image.fromarray(b, "RGBA"))
    (dossier / "animation.json").write_text(json.dumps({"prompt": prompt, "graine": res["graine"], "fond": fond},
                                                       ensure_ascii=False), encoding="utf-8")
    return aller_retour(images)


def _lire_video(chemin):
    import av

    with av.open(str(chemin)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
