"""Passer en HD-2D (onglet « Jeu → Passer en HD-2D ») : des sprites, planches d'animation, tuiles et effets en pixel art
deviennent des éléments éclairables en 3D, comme dans les jeux « HD-2D » (pixel art posé dans des décors 3D, éclairé
par des lumières dynamiques, effets lumineux, flou de profondeur).

Pour chaque image (ou planche : la disposition est gardée, pour que le moteur de jeu lise la carte avec la même
découpe que la couleur) :
  - couleur nettoyée (fond retiré ; effet sur fond blanc : le blanc devient transparence, lueur comprise) ;
  - carte de relief (`_relief`) : bombé depuis le bord de la silhouette pour un sprite, d'après la luminosité pour une
    tuile (raccord parfait : calcul en boucle sur chaque case) ;
  - carte de normales (OpenGL ou DirectX), calculée à la taille du pixel art puis agrandie sans flou ;
  - lumière émise pour un effet (magie, feu, particules) ;
  - aperçu : lumière qui tourne autour (GIF), lueur des effets ;
  - objet 3D (GLB, sans dépendance) : panneau vertical pour un sprite, bloc pour une tuile, panneau lumineux pour un
    effet ; couleur, normales et émission avec filtrage au plus proche (pixels nets).
Rien ne demande de modèle d'IA. Rangement : data/hd2d/<horodatage>_<nom>/ (hors galerie).
"""
import json
import math
import struct
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from . import config as cfg
from .outils import nouveau_dossier
from .pixel_art import _gif, _sans_fond_uni, agrandir, reduire, remettre
from .serie import _slug

AUTO = "Automatique"
SPRITE = "Personnage, créature ou objet (fond transparent)"
TUILE = "Tuile de décor (sol, mur, rocher…)"
EFFET = "Effet lumineux (magie, feu, particules)"
TYPES = [AUTO, SPRITE, TUILE, EFFET]
OPENGL = "OpenGL (Godot, Unity, Blender)"
DIRECTX = "DirectX (Unreal)"
CONVENTIONS = [OPENGL, DIRECTX]
IMAGES_APERCU = 30


# --- Lecture et nettoyage ----------------------------------------------------------------------------------------
def _blanc_vers_transparence(a):
    """Effet dessiné sur fond blanc (lueur comprise) : « couleur vers alpha » du blanc. Un pixel c = α·couleur +
    (1 − α)·blanc : α = plus grand écart au blanc, couleur retrouvée en retirant le blanc."""
    rgb = a[..., :3].astype(float) / 255
    alpha = (1 - rgb).max(axis=-1)
    alpha = np.where(alpha < 0.03, 0, alpha)
    couleur = np.where(alpha[..., None] > 0, (rgb - (1 - alpha[..., None])) / np.maximum(alpha[..., None], 1e-6), 0)
    sortie = np.zeros_like(a)
    sortie[..., :3] = np.clip(couleur * 255, 0, 255).round()
    sortie[..., 3] = np.clip(alpha * 255, 0, 255).round()
    return sortie


def deviner_type(a):
    """Sprite si la transparence est franche, effet si elle est douce. Image opaque : on regarde ce qu'un fond uni
    relié aux bords occuperait — presque rien = tuile (une tuile entourée de joints de la même couleur reste une
    tuile), ou motif qui touche les quatre bords = tuile (des briques : leurs joints forment un « fond » relié aux
    bords) ; fond blanc avec un bord clair qui s'y fond (lueur) = effet ; sinon sprite sur fond uni."""
    opaque = a[..., 3]
    if opaque.min() < 255:
        return SPRITE if set(np.unique(opaque)) <= {0, 255} else EFFET
    fond = np.array(_sans_fond_uni(Image.fromarray(a, "RGBA")))[..., 3] == 0
    reste = ~fond
    if fond.mean() < 0.15 or (reste[0].any() and reste[-1].any() and reste[:, 0].any() and reste[:, -1].any()):
        return TUILE  # le motif touche les quatre bords (briques et leurs joints) : une tuile, pas un sujet sur fond
    blanc = (a[..., :3][fond] > 235).all(axis=-1).mean() > 0.9
    # bord du sujet : clair et fondu dans le blanc pour un effet (lueur), sombre et net pour un sprite (contour)
    p = np.pad(fond, 1)
    bord = reste & (p[2:, 1:-1] | p[:-2, 1:-1] | p[1:-1, 2:] | p[1:-1, :-2])
    clair = bord.any() and float(a[..., :3][bord].mean()) > 150
    return EFFET if blanc and clair else SPRITE


def preparer(image, type_):
    """(RGBA nettoyé à la taille du pixel art, type, (agrandissement, décalage, taille d'origine)) : un pixel art
    agrandi est traité sur ses vrais pixels, et les cartes sont remises à la taille et au décalage d'origine."""
    origine = np.array(ImageOps.exif_transpose(image).convert("RGBA"))
    a, k, decalage = reduire(origine)
    a = np.ascontiguousarray(a)
    type_ = deviner_type(a) if type_ in (None, "", AUTO) else type_
    if type_ == TUILE and decalage != (0, 0):  # une tuile doit se raccorder à sa taille : pas de bande coupée
        a, k, decalage = origine, 1, (0, 0)
    if type_ == EFFET and a[..., 3].min() == 255:
        a = _blanc_vers_transparence(a)
    elif type_ == SPRITE and a[..., 3].min() == 255:
        a = np.array(_sans_fond_uni(Image.fromarray(a, "RGBA")))
    elif type_ == TUILE:
        a[..., 3] = 255
    return a, type_, (k, decalage, origine.shape[:2])


# --- Relief et normales ------------------------------------------------------------------------------------------
def _flou(x, rayon, boucle=False):
    """Flou gaussien séparable (numpy seul) ; boucle : bords raccordés (tuile)."""
    if rayon <= 0:
        return x
    r = int(math.ceil(rayon * 2.5))
    noyau = np.exp(-0.5 * (np.arange(-r, r + 1) / rayon) ** 2)
    noyau /= noyau.sum()
    for axe in (0, 1):
        if boucle:
            x = sum(w * np.roll(x, -d, axis=axe) for d, w in zip(range(-r, r + 1), noyau))
        else:
            p = np.pad(x, [(r, r) if i == axe else (0, 0) for i in range(x.ndim)], mode="edge")
            x = sum(w * np.take(p, np.arange(d + r, d + r + x.shape[axe]), axis=axe)
                    for d, w in zip(range(-r, r + 1), noyau))
    return x


def _distance_au_bord(masque, maxi):
    """Distance (en pixels, 4-voisins) de chaque pixel opaque au vide, plafonnée à maxi."""
    d = np.zeros(masque.shape, float)
    courant = masque.copy()
    for i in range(1, maxi + 1):
        p = np.pad(courant, 1)
        courant = courant & p[2:, 1:-1] & p[:-2, 1:-1] & p[1:-1, 2:] & p[1:-1, :-2]
        d += courant
        if not courant.any():
            break
    return np.where(masque, d + 1, 0)


def _cases(forme, grille):
    """Découpe (colonnes, lignes) en tranches de pixels ; (0, 0) = une seule case."""
    colonnes, lignes = (int(grille[0] or 0), int(grille[1] or 0)) if grille else (0, 0)
    colonnes, lignes = max(1, colonnes), max(1, lignes)
    h, w = forme
    return [(slice(round(j * h / lignes), round((j + 1) * h / lignes)),
             slice(round(i * w / colonnes), round((i + 1) * w / colonnes)))
            for j in range(lignes) for i in range(colonnes)]


def relief(a, type_, grille=None, arrondi=4):
    """Hauteur 0–1 par pixel. Sprite : bombé depuis le bord (arrondi en pixels) + un peu de luminosité (détails).
    Tuile : luminosité lissée, calculée en boucle sur chaque case (le relief se raccorde d'une tuile à l'autre)."""
    lum = (a[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])) / 255
    masque = a[..., 3] >= 128
    if type_ == TUILE:
        h = np.zeros(lum.shape)
        for lignes, colonnes in _cases(lum.shape, grille):
            case = lum[lignes, colonnes]
            # luminosité lissée, rapportée au contraste de la case (écart type) : étirée du plus sombre au plus clair,
            # chaque petite tache devenait une bosse (normales bruitées) ; telle quelle, une tuile terne restait plate
            case = _flou(case, 0.8, boucle=True)
            h[lignes, colonnes] = np.clip(0.5 + (case - case.mean()) / (4 * max(case.std(), 0.02)), 0, 1)
        return h
    d = _distance_au_bord(masque, max(1, int(arrondi)))
    bombe = np.sqrt(np.clip(1 - (1 - d / max(1, arrondi)) ** 2, 0, 1)) * masque
    detail = _flou(lum, 0.6) * masque
    if type_ == EFFET:  # une lueur : relief doux, le cœur lumineux en avant
        return np.clip(0.4 * _flou(a[..., 3] / 255, 1.0) + 0.6 * detail, 0, 1)
    return np.clip(0.75 * bombe + 0.25 * detail, 0, 1)


def normales(h, force=2.0, convention=OPENGL, boucle=False, grille=None):
    """Carte de normales RGB 0–255 (bleu = face à la caméra). OpenGL : vert vers le haut ; DirectX : inversé."""
    gx, gy = np.zeros_like(h), np.zeros_like(h)
    for lignes, colonnes in _cases(h.shape, grille if boucle else None):
        c = h[lignes, colonnes]
        if boucle:
            gx[lignes, colonnes] = (np.roll(c, -1, 1) - np.roll(c, 1, 1)) / 2
            gy[lignes, colonnes] = (np.roll(c, -1, 0) - np.roll(c, 1, 0)) / 2
        else:
            gy[lignes, colonnes], gx[lignes, colonnes] = np.gradient(c)
    n = np.stack([-gx * force, gy * force, np.ones_like(h)], axis=-1)  # image vers le bas : +y image = −haut
    if convention == DIRECTX:
        n[..., 1] *= -1
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.clip((n * 0.5 + 0.5) * 255, 0, 255).round().astype(np.uint8)


def emission(a, type_):
    """Lumière émise : l'effet lumineux entier (sa couleur, transparence comprise) ; rien pour le reste (deviner
    seul ce qui brille, comme un t-shirt rouge vif, donnait de faux néons)."""
    if type_ != EFFET:
        return np.zeros_like(a[..., :3])
    return np.clip(a[..., :3].astype(float) * (a[..., 3:4] / 255), 0, 255).astype(np.uint8)


# --- Aperçu éclairé ----------------------------------------------------------------------------------------------
def eclairer(a, n, angle, emis=None, type_=SPRITE):
    """Rendu d'une image avec une lumière qui vient de l'angle donné (radians), lumière ambiante, petit reflet et
    lueur de l'émission, sur un fond sombre (le fond d'une scène de nuit, pour voir la lumière)."""
    albedo = a[..., :3] / 255
    alpha = a[..., 3:4] / 255
    nn = n / 127.5 - 1
    lumiere = np.array([math.cos(angle), math.sin(angle), 0.45])  # lumière rasante : le relief se voit
    lumiere /= np.linalg.norm(lumiere)
    diffus = np.clip(nn @ lumiere, 0, 1)[..., None]
    demi = lumiere + np.array([0, 0, 1.0])
    demi /= np.linalg.norm(demi)
    reflet = np.clip(nn @ demi, 0, 1)[..., None] ** 24 * 0.25
    couleur = albedo * (0.25 + 1.1 * diffus) + reflet
    if type_ == EFFET:
        couleur = albedo
    fond = np.zeros_like(albedo)
    fond[..., :] = np.linspace([0.06, 0.07, 0.12], [0.12, 0.13, 0.2], a.shape[0])[:, None, :]
    image = couleur * alpha + fond * (1 - alpha)
    if emis is not None and emis.any():
        lueur = _flou(emis / 255, max(1.5, a.shape[0] / 40))
        image = image + lueur * 0.5
    return Image.fromarray((np.clip(image, 0, 1) * 255).round().astype(np.uint8), "RGB")


def apercu(a, n, emis, type_, k, chemin, grille=None):
    """GIF de IMAGES_APERCU images (agrandi ×k, pixels nets) : la lumière fait le tour ; une planche (grille de
    plusieurs cases) joue ses images à la suite, comme dans le jeu ; une tuile est répétée 3×3 (raccord visible)."""
    cases = _cases(a.shape[:2], grille)
    images = []
    for i in range(IMAGES_APERCU):
        lignes, colonnes = cases[i % len(cases)] if type_ != TUILE else cases[0]
        ca, cn = a[lignes, colonnes], n[lignes, colonnes]
        ce = emis[lignes, colonnes] if emis is not None else None
        if type_ == TUILE:
            ca, cn = np.tile(ca, (3, 3, 1)), np.tile(cn, (3, 3, 1))
        images.append(agrandir(eclairer(ca, cn, 2 * math.pi * i / IMAGES_APERCU, ce, type_).convert("RGBA"),
                               max(1, k)))
    _gif(images, chemin, int(5000 / IMAGES_APERCU))
    return chemin


# --- Objets 3D (GLB écrit à la main : pas de dépendance) ----------------------------------------------------------
def _png(a):
    import io

    tampon = io.BytesIO()
    Image.fromarray(a).save(tampon, format="PNG")
    return tampon.getvalue()


def _glb(positions, normales_, uvs, indices, textures, alpha_masque, emissif, chemin, nom):
    """GLB minimal (glTF 2.0) : un maillage, un matériau mat (métal 0, rugosité 1), textures couleur / normales /
    émission en PNG avec filtrage au plus proche (9728) : les pixels restent nets de près."""
    morceaux, vues, accesseurs = [], [], []

    def ajouter(octets, cible=None):
        decalage = sum(len(m) for m in morceaux)
        morceaux.append(octets + b"\0" * (-len(octets) % 4))
        vues.append({"buffer": 0, "byteOffset": decalage, "byteLength": len(octets),
                     **({"target": cible} if cible else {})})
        return len(vues) - 1

    def accesseur(tableau, type_, composant, cible, minmax=False):
        vue = ajouter(tableau.tobytes(), cible)
        acc = {"bufferView": vue, "componentType": composant, "count": len(tableau), "type": type_}
        if minmax:
            acc["min"], acc["max"] = tableau.min(axis=0).tolist(), tableau.max(axis=0).tolist()
        accesseurs.append(acc)
        return len(accesseurs) - 1

    pos = accesseur(np.asarray(positions, np.float32), "VEC3", 5126, 34962, True)
    nor = accesseur(np.asarray(normales_, np.float32), "VEC3", 5126, 34962)
    uv = accesseur(np.asarray(uvs, np.float32), "VEC2", 5126, 34962)
    ind = accesseur(np.asarray(indices, np.uint16), "SCALAR", 5123, 34963)
    images, textures_ = [], []
    for octets in textures.values():
        images.append({"bufferView": ajouter(octets), "mimeType": "image/png"})
        textures_.append({"sampler": 0, "source": len(images) - 1})
    cles = list(textures)
    materiau = {"name": nom, "pbrMetallicRoughness": {"baseColorTexture": {"index": cles.index("couleur")},
                                                      "metallicFactor": 0.0, "roughnessFactor": 1.0},
                "doubleSided": True}
    if "normales" in textures:
        materiau["normalTexture"] = {"index": cles.index("normales")}
    if "emission" in textures:
        materiau["emissiveTexture"] = {"index": cles.index("emission")}
        materiau["emissiveFactor"] = [1.0, 1.0, 1.0] if emissif else [0.6, 0.6, 0.6]
    if alpha_masque == "MASK":
        materiau.update(alphaMode="MASK", alphaCutoff=0.5)
    elif alpha_masque == "BLEND":
        materiau["alphaMode"] = "BLEND"
    binaire = b"".join(morceaux)
    gltf = {"asset": {"version": "2.0", "generator": "Studio Voix (HD-2D)"}, "scene": 0,
            "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": nom}],
            "meshes": [{"name": nom, "primitives": [{"attributes": {"POSITION": pos, "NORMAL": nor, "TEXCOORD_0": uv},
                                                     "indices": ind, "material": 0}]}],
            "materials": [materiau], "textures": textures_, "images": images,
            "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
            "accessors": accesseurs, "bufferViews": vues, "buffers": [{"byteLength": len(binaire)}]}
    texte = json.dumps(gltf, separators=(",", ":")).encode()
    texte += b" " * (-len(texte) % 4)
    total = 12 + 8 + len(texte) + 8 + len(binaire)
    with open(chemin, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(texte), b"JSON") + texte)
        f.write(struct.pack("<I4s", len(binaire), b"BIN\0") + binaire)
    return chemin


def panneau_glb(a, n, emis, type_, chemin, nom, pixels_par_metre=32):
    """Sprite ou effet sur un panneau vertical posé au sol (pieds à y = 0), à l'échelle du pixel art (32 pixels =
    1 mètre par défaut), comme les personnages des jeux HD-2D. Recadré sur ce qui est visible."""
    lignes, colonnes = np.where(a[..., 3] > 0)
    if not len(lignes):
        raise ValueError("image vide")
    haut, bas, gauche, droite = lignes.min(), lignes.max() + 1, colonnes.min(), colonnes.max() + 1
    a, n = a[haut:bas, gauche:droite], n[haut:bas, gauche:droite]
    emis = emis[haut:bas, gauche:droite] if emis is not None else None
    w, h = a.shape[1] / pixels_par_metre, a.shape[0] / pixels_par_metre
    positions = [[-w / 2, 0, 0], [w / 2, 0, 0], [w / 2, h, 0], [-w / 2, h, 0]]
    textures = {"couleur": _png(a), "normales": _png(n)}
    if emis is not None and emis.any():
        textures["emission"] = _png(emis)
    return _glb(positions, [[0, 0, 1]] * 4, [[0, 1], [1, 1], [1, 0], [0, 0]], [0, 1, 2, 0, 2, 3], textures,
                "BLEND" if type_ == EFFET else "MASK", type_ == EFFET, chemin, nom)


def bloc_glb(a, n, chemin, nom):
    """Tuile sur un cube de 1 m : la même texture sur chaque face (normales comprises), pour monter un décor HD-2D
    bloc par bloc."""
    faces = [((0, 1, 0), [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)]), ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
             ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]), ((0, 0, -1), [(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)]),
             ((1, 0, 0), [(1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)]), ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)])]
    positions, normales_, uvs, indices = [], [], [], []
    for normale, coins in faces:
        base = len(positions)
        positions += [[x - 0.5, y, z - 0.5] for x, y, z in coins]
        normales_ += [list(normale)] * 4
        uvs += [[0, 1], [1, 1], [1, 0], [0, 0]]
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return _glb(positions, normales_, uvs, indices, {"couleur": _png(a), "normales": _png(n)}, None, False, chemin,
                nom)


# --- Onglet --------------------------------------------------------------------------------------------------
def convertir(fichiers, type_label=AUTO, colonnes=0, lignes=0, force=2.0, arrondi=4, convention=OPENGL,
              pixels_par_metre=32, nom="hd2d", progress=gr.Progress()):
    """Chaque image → couleur, relief, normales (+ émission), aperçu éclairé et objet 3D. Renvoie (message, aperçus,
    fichiers, dossier)."""
    fichiers = [getattr(f, "name", f) for f in (fichiers or [])]
    if not fichiers:
        raise gr.Error("Ajoute un ou plusieurs fichiers : sprites, planches d'animation, tuiles ou effets.")
    dossier = nouveau_dossier(cfg.HD2D_DIR, _slug(nom, "hd2d"))
    apercus, produits, lignes_msg = [], [], []
    for i, source in enumerate(fichiers, 1):
        progress((i - 1) / len(fichiers), desc=f"Image {i}/{len(fichiers)}…")
        base = _slug(Path(source).stem, f"image_{i}")
        try:
            with Image.open(source) as im:
                im.load()
                a, type_, (k, decalage, forme) = preparer(im, type_label)
        except (OSError, ValueError) as e:
            lignes_msg.append(f"⚠️ {Path(source).name} : illisible ({e})")
            continue
        grille = (colonnes, lignes)
        h = relief(a, type_, grille, arrondi)
        n = normales(h, float(force), convention, boucle=type_ == TUILE, grille=grille)
        n[a[..., 3] == 0] = (128, 128, 255)
        e = emission(a, type_)
        sortie = {}
        rendu = max(1, k)
        for cle, img in (("couleur", a), ("normales", n),
                         ("relief", (h * 255).round().astype(np.uint8)), ("emission", e)):
            if cle == "emission" and not e.any():
                continue
            f = dossier / f"{base}_{cle}.png"
            Image.fromarray(remettre(img, rendu, decalage, forme) if rendu > 1 else img).save(f)
            sortie[cle] = f
        premiere = _cases(a.shape[:2], grille)[0]
        cote = max(a[premiere].shape[:2]) * (3 if type_ == TUILE else 1)
        gif = apercu(a, n, e if e.any() else None, type_, max(1, 256 // cote), dossier / f"{base}_apercu.gif", grille)
        try:
            if type_ == TUILE:
                premiere = _cases(a.shape[:2], grille)[0]
                glb = bloc_glb(a[premiere], n[premiere], dossier / f"{base}_bloc.glb", base)
            else:
                premiere = _cases(a.shape[:2], grille)[0]  # planche : la première image sur le panneau
                glb = panneau_glb(a[premiere], n[premiere], e[premiere] if e.any() else None, type_,
                                  dossier / f"{base}_panneau.glb", base, int(pixels_par_metre))
        except ValueError:
            glb = None
        apercus.append(str(gif))
        produits += [str(p) for p in sortie.values()] + [str(gif)] + ([str(glb)] if glb else [])
        lignes_msg.append(f"✅ {Path(source).name} : {type_.split(' (')[0].lower()}, {a.shape[1]}×{a.shape[0]} pixels"
                          + (f" (affiché ×{k})" if k > 1 else "") + ", " + ", ".join(sortie)
                          + (", objet 3D" if glb else ""))
    if not apercus:
        raise gr.Error("Aucune image convertie : " + " ; ".join(lignes_msg))
    (dossier / "hd2d.json").write_text(json.dumps({"convention": convention, "force": force, "arrondi": arrondi,
                                                   "grille": [colonnes, lignes], "pixels_par_metre": pixels_par_metre},
                                                  ensure_ascii=False), encoding="utf-8")
    msg = ("\n".join(f"- {l_}" for l_ in lignes_msg) + f"\n\nDossier : {dossier}. Dans le moteur de jeu : couleur en "
           "texture de base, normales en carte de normales (filtrage « au plus proche » pour garder les pixels nets), "
           "émission en lumière émise ; les planches gardent leur disposition.")
    return msg, apercus, produits, str(dossier)
