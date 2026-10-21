"""Tileset en objets 3D (onglet « Jeu → Tileset en objets 3D ») : un jeu de tuiles en pixel art (atlas de cases,
RPG Maker, Tiled…) → un modèle 3D « low poly » texturé par objet, pour monter le décor en vraie 3D (comme les
remakes 3D de jeux en pixel art).

1. Découpage (`decouper`, numpy + scipy, instantané) : cases vides (croix rouge de l'éditeur, noir uni) retirées,
   objets = zones reliées (un objet qui s'étend sur plusieurs cases reste entier), petits morceaux proches regroupés ;
   une zone est coupée entre deux cases quand l'image ne continue pas (objets rangés côte à côte, chacun son
   contour) ; les blocs d'au moins quatre cases pleines (sol, eau, mur) sont signalés et décochés par défaut : ce sont
   des textures, pas des objets.
2. Pour chaque objet coché : FLUX.2 klein le redessine en rendu 3D propre du même objet (mêmes formes, couleurs et
   détails, vue de trois quarts, fond blanc) — Hunyuan3D lit mal du pixel art de quelques dizaines de pixels — puis
   Hunyuan3D-2 en fait la forme et la texture (`modele3d.generer`), réduite au nombre de faces de la qualité
   « Low poly » (détails gardés dans la texture).
Rangement : data/3d/tilesets/<horodatage>_<nom>/ (pieces/, rendus/, <objet>.glb, lot.json) ; chaque modèle a aussi
sa création dans data/3d/<horodatage>/ (galerie, « Recréer »).
"""
import json
import shutil
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from . import config as cfg
from . import diffusion, hd2d, modele3d
from .outils import nouveau_dossier
from .serie import _slug

QUALITE = "Low poly (jeu : formes simples, détails dans la texture)"  # dans modele3d.QUALITES
AIRE_MIN = 24  # pixels : en dessous, une poussière (ou un morceau isolé d'un autre objet)
COTE_REFERENCE = 512  # objet agrandi (au plus proche) pour FLUX.2 klein
RENDU_3D = ("Recreate this pixel art game sprite as a clean 3D render of exactly the same object: a low poly 3D game "
            "asset with simple smooth geometry, same shape, same proportions, same colors, same details and markings, "
            "seen in three-quarter view from slightly above, the whole object visible and centered, single object, "
            "plain pure white background, soft studio lighting. No pixels, no pixel art, no text, no floor, no shadow")


def _vides(a, cote):
    """Pixels à ignorer : transparents, cases à croix rouge, cases unies noires ou blanches (vide ou fond)."""
    from scipy import ndimage

    vide = a[..., 3] == 0
    fonds = set()
    for lignes, colonnes in hd2d._cases(a.shape[:2], (a.shape[1] // cote, a.shape[0] // cote)):
        c = a[lignes, colonnes]
        uni = (c == c[0, 0]).all() and c[0, 0, 3] == 255 and not 60 <= int(c[0, 0, :3].sum()) <= 705
        if uni:
            fonds.add(tuple(c[0, 0].tolist()))
        if uni or hd2d._case_vide(c):
            vide[lignes, colonnes] = True
    # fond blanc ou noir autour des objets : la même couleur reliée à une case unie (le contour des objets l'arrête)
    for couleur in fonds:
        meme = (a == np.array(couleur, a.dtype)).all(axis=-1)
        zones, _ = ndimage.label(meme)
        vide |= np.isin(zones, np.unique(zones[meme & vide])) & meme
    return vide


def _continu(ga, gb, la, lb, ea, eb):
    """Deux bords de cases voisines (pixels face à face) : le même objet continue d'une case à l'autre si les pixels
    se ressemblent ; deux objets rangés côte à côte ont chacun leur contour sombre ou changent de couleur."""
    face = (ea > 0) & (ea == eb)
    if face.sum() < 4:
        return False
    if face.sum() < 0.6 * len(face):  # passage étroit (feuilles d'une plante, pied) : le même objet qui déborde
        return True
    proche = np.abs(ga.astype(int) - gb.astype(int)).sum(axis=-1) < 90
    sombres = (la < 0.3) & (lb < 0.3)
    return (proche & ~sombres & face).sum() >= 0.5 * face.sum()


def _separer(etiquettes, a, cote):
    """Zones reliées → objets : une zone est coupée entre deux cases quand l'image ne continue pas de l'une à
    l'autre (objets rangés côte à côte dans l'atlas, piliers voisins, meubles contre un sol)."""
    lum = (a[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])) / 255
    h, w = etiquettes.shape
    lignes, colonnes = h // cote, w // cote
    parent = {}

    def trouver(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in range(lignes):
        for c in range(colonnes):
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 >= lignes or c2 >= colonnes:
                    continue
                if dc:  # bord vertical : dernière colonne de la case de gauche, première de celle de droite
                    ys, xa, xb = slice(r * cote, (r + 1) * cote), c2 * cote - 1, c2 * cote
                    ea, eb = etiquettes[ys, xa], etiquettes[ys, xb]
                    ga, gb, la, lb = a[ys, xa, :3], a[ys, xb, :3], lum[ys, xa], lum[ys, xb]
                else:
                    xs, ya, yb = slice(c * cote, (c + 1) * cote), r2 * cote - 1, r2 * cote
                    ea, eb = etiquettes[ya, xs], etiquettes[yb, xs]
                    ga, gb, la, lb = a[ya, xs, :3], a[yb, xs, :3], lum[ya, xs], lum[yb, xs]
                for e in set(np.unique(ea[(ea > 0) & (ea == eb)]).tolist()):
                    if _continu(ga, gb, la, lb, np.where(ea == e, ea, 0), np.where(eb == e, eb, 0)):
                        parent[trouver((e, r, c))] = trouver((e, r2, c2))
    sortie = np.zeros_like(etiquettes)
    numeros = {}
    for r in range(lignes):
        for c in range(colonnes):
            bloc = etiquettes[r * cote:(r + 1) * cote, c * cote:(c + 1) * cote]
            for e in np.unique(bloc[bloc > 0]).tolist():
                n = numeros.setdefault(trouver((e, r, c)), len(numeros) + 1)
                sortie[r * cote:(r + 1) * cote, c * cote:(c + 1) * cote][bloc == e] = n
    return sortie


def decouper(chemin, nom="tileset", cote=0):
    """Atlas → objets. Renvoie (dossier, liste de pièces {nom, boite [x0, y0, x1, y1], cases, sol, piece,
    reference})."""
    from scipy import ndimage

    with Image.open(chemin) as im:
        a = np.array(im.convert("RGBA"))
    cote = int(cote or 0) or hd2d.case_atlas(a.shape[:2]) or 32
    plein = ~_vides(a, cote)
    # regroupe les morceaux à un pixel l'un de l'autre (feuilles d'une plante, pieds d'un meuble)
    etiquettes, n = ndimage.label(ndimage.binary_dilation(plein, np.ones((3, 3))), np.ones((3, 3)))
    etiquettes[~plein] = 0
    etiquettes = _separer(etiquettes, a, cote)
    dossier = nouveau_dossier(cfg.MODELS3D_DIR / "tilesets", _slug(nom, "tileset"))
    (dossier / "pieces").mkdir()
    shutil.copy(chemin, dossier / f"atlas{Path(chemin).suffix.lower() or '.png'}")
    pieces = []
    for i, boite in enumerate(ndimage.find_objects(etiquettes), 1):
        if boite is None:
            continue
        masque = etiquettes[boite] == i
        if masque.sum() < AIRE_MIN:
            continue
        y0, y1, x0, x1 = boite[0].start, boite[0].stop, boite[1].start, boite[1].stop
        cases = [(r, c) for r in range(y0 // cote, (y1 - 1) // cote + 1) for c in range(x0 // cote, (x1 - 1) // cote + 1)
                 if (etiquettes[r * cote:(r + 1) * cote, c * cote:(c + 1) * cote] == i).any()]
        # sol, eau, mur : un bloc d'au moins quatre cases entièrement pleines (une texture, pas un objet)
        pleines = [(r, c) for r, c in cases if (etiquettes[r * cote:(r + 1) * cote, c * cote:(c + 1) * cote] == i).all()]
        sol = len(pleines) == len(cases) >= 4
        morceau = a[boite].copy()
        morceau[~masque] = 0
        nom_piece = f"objet_{len(pieces) + 1:02d}"
        piece = dossier / "pieces" / f"{nom_piece}.png"
        Image.fromarray(morceau).save(piece)
        reference = dossier / "pieces" / f"{nom_piece}_reference.png"
        _reference(morceau).save(reference)
        pieces.append({"nom": nom_piece, "boite": [x0, y0, x1, y1], "cases": [x1 - x0, y1 - y0],
                       "case": [x0 // cote + 1, y0 // cote + 1], "sol": bool(sol), "piece": str(piece),
                       "reference": str(reference)})
    (dossier / "lot.json").write_text(json.dumps({"atlas": Path(chemin).name, "cote": cote, "pieces": pieces},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    return dossier, pieces


def _reference(morceau):
    """Objet agrandi au plus proche sur fond blanc, avec une marge : l'image que FLUX.2 klein redessine."""
    h, w = morceau.shape[:2]
    k = max(1, COTE_REFERENCE // max(h, w))
    grand = Image.fromarray(morceau).resize((w * k, h * k), Image.NEAREST)
    marge = max(grand.size) // 8
    fond = Image.new("RGBA", (grand.width + 2 * marge, grand.height + 2 * marge), "white")
    fond.alpha_composite(grand, (marge, marge))
    return fond.convert("RGB")


def libelle(p):
    largeur, hauteur = p["cases"]
    return (f"{p['nom']} — {largeur}×{hauteur} px, colonne {p['case'][0]}, ligne {p['case'][1]}"
            + (" (sol / surface)" if p["sol"] else ""))


def analyser(fichier, cote=0, nom="tileset"):
    """Onglet : (message, galerie [(image, légende)], choix possibles, choix cochés, dossier)."""
    fichier = getattr(fichier, "name", fichier)
    if not fichier:
        raise gr.Error("Dépose le tileset (PNG) à découper.")
    dossier, pieces = decouper(fichier, nom, cote)
    if not pieces:
        raise gr.Error("Aucun objet trouvé dans l'image (tout est vide ou transparent ?).")
    choix = [libelle(p) for p in pieces]
    sols = sum(p["sol"] for p in pieces)
    msg = (f"🧩 {len(pieces)} élément(s) trouvé(s)" + (f", dont {sols} sol(s) ou surface(s) décoché(s)" if sols else "")
           + ". Décoche ce que tu ne veux pas en 3D, puis « Créer les objets 3D » (une à deux minutes par objet).")
    return (msg, [(p["piece"], libelle(p)) for p in pieces], choix,
            [libelle(p) for p in pieces if not p["sol"]], str(dossier))


def generer(dossier, choisis, redessiner=True, qualite=QUALITE, texture=True, graine=0, projet=None,
            progress=gr.Progress()):
    """Objets cochés → modèles 3D. Tous les rendus FLUX.2 klein d'abord, puis tous les Hunyuan3D (le moteur ne
    change de modèle qu'une fois). Un objet en échec n'arrête pas les autres. Renvoie (message, dernier GLB,
    fichiers, dossier)."""
    if not dossier or not (Path(dossier) / "lot.json").exists():
        raise gr.Error("Découpe d'abord le tileset.")
    d = Path(dossier)
    lot = json.loads((d / "lot.json").read_text(encoding="utf-8"))
    pieces = [p for p in lot["pieces"] if libelle(p) in (choisis or [])]
    if not pieces:
        raise gr.Error("Coche au moins un objet.")
    graine = int(graine or 0) or diffusion.graines(1)[0]
    images = {p["nom"]: p["reference"] for p in pieces}
    if redessiner:
        (d / "rendus").mkdir(exist_ok=True)
        for i, p in enumerate(pieces):
            progress(i / len(pieces) * 0.3, desc=f"Rendu 3D de {p['nom']} ({i + 1}/{len(pieces)}, FLUX.2 klein)…")
            sortie = d / "rendus" / f"{p['nom']}.png"
            largeur, hauteur = _format(p["reference"])
            try:
                diffusion.personnage(RENDU_3D + ".", [p["reference"]], [sortie], [graine + i], largeur, hauteur)
                images[p["nom"]] = str(sortie)
            except gr.Error as e:
                p["erreur"] = f"rendu : {getattr(e, 'message', e)}"
    lignes, produits, dernier = [], [], None
    for i, p in enumerate(pieces):
        if p.get("erreur"):
            lignes.append(f"- ❌ {p['nom']} : {p['erreur']}")
            continue
        progress(0.3 + i / len(pieces) * 0.7, desc=f"Modèle 3D de {p['nom']} ({i + 1}/{len(pieces)}, Hunyuan3D)…")
        try:
            msg, glb, _, _, creation = modele3d.generer(images[p["nom"]], f"{lot['atlas'].rsplit('.', 1)[0]}_{p['nom']}",
                                                        qualite, texture, graine + i, ["glb"], projet=projet)
        except gr.Error as e:
            lignes.append(f"- ❌ {p['nom']} : {getattr(e, 'message', e)}")
            continue
        copie = d / f"{p['nom']}.glb"
        shutil.copy(glb, copie)
        p.update(glb=str(copie), creation=creation)
        produits.append(str(copie))
        dernier = str(copie)
        lignes.append(f"- ✅ {p['nom']} : {msg.split(' : ', 1)[-1].split('. Dossier')[0]}")
    for p in lot["pieces"]:  # notés dans lot.json : position dans l'atlas → modèle
        p.update(next((q for q in pieces if q["nom"] == p["nom"]), {}))
    lot.update(redessine=bool(redessiner), qualite=qualite, graine=graine)
    (d / "lot.json").write_text(json.dumps(lot, ensure_ascii=False, indent=1), encoding="utf-8")
    reussis = len(produits)
    return (f"**{reussis}/{len(pieces)} objet(s) en 3D** (GLB dans {d}, chacun aussi dans la Galerie) :\n\n"
            + "\n".join(lignes), dernier, produits, str(d))


def _format(image):
    """Format de klein d'après les proportions de l'objet (côtés multiples de 16, ~1 Mpx)."""
    from .images import AUTO, format_pour

    return format_pour(AUTO, image)
