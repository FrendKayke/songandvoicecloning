"""Tileset en objets 3D (onglet « Jeu → Tileset en objets 3D ») : un jeu de tuiles en pixel art (atlas de cases,
RPG Maker, Tiled…) → un modèle 3D « low poly » texturé par objet, pour monter le décor en vraie 3D (comme les
remakes 3D de jeux en pixel art).

1. Découpage (`decouper`, numpy + scipy, instantané) : cases vides (croix rouge de l'éditeur, noir uni) retirées,
   objets = zones reliées (un objet qui s'étend sur plusieurs cases reste entier), petits morceaux proches regroupés ;
   une zone est coupée entre deux cases quand l'image ne continue pas (objets rangés côte à côte, chacun son
   contour) ; les blocs d'au moins quatre cases pleines (sol, eau, mur) sont signalés et décochés par défaut : ce sont
   des textures, pas des objets.
2. Identification (`identifier`, Qwen3-VL mode « tuile3d ») : ce que représente chaque objet coché, décrit comme un
   objet 3D autonome (une tuile « trou avec échelle » = une échelle) ; texte modifiable dans l'onglet.
3. Dessin (`dessiner`) : une image 3D propre par objet, d'après la description, par Z-Image (FLUX.2 klein, avec le
   pixel art en référence, le recopiait : la première version donnait des pavés plats peints, constaté chez
   l'utilisateur) ; FLUX.2 klein reste au choix (« garder le dessin d'origine »).
4. 3D (`generer`) : Hunyuan3D-2 sur chaque image (`modele3d.generer`), qualité « Low poly » (8 000 faces, détails
   dans la texture).
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
STYLE_3D = "stylized low poly 3D game asset, clean simple shapes, smooth colors, 3D render"
# avec le dessin d'origine en référence (FLUX.2 klein) : il a tendance à recopier les pixels, d'où l'insistance
RENDU_3D = ("Use the reference image only for the design and colors of the object: show it as a smooth 3D rendered "
            "object with real volume and depth, not flat, no pixels, no pixel art, no text, no floor")


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
           + ". Décoche ce que tu ne veux pas en 3D, puis « 🔎 Identifier les objets ».")
    return (msg, [(p["piece"], libelle(p)) for p in pieces], choix,
            [libelle(p) for p in pieces if not p["sol"]], str(dossier))


def _lot(dossier):
    if not dossier or not (Path(dossier) / "lot.json").exists():
        raise gr.Error("Découpe d'abord le tileset.")
    return Path(dossier), json.loads((Path(dossier) / "lot.json").read_text(encoding="utf-8"))


def _ecrire_lot(d, lot):
    (d / "lot.json").write_text(json.dumps(lot, ensure_ascii=False, indent=1), encoding="utf-8")


def _lignes(table):
    """Tableau de l'onglet (liste de lignes [objet, description]) → [(nom, description)] non vides."""
    if hasattr(table, "values"):  # DataFrame
        table = table.values.tolist()
    return [(str(r[0]).split(" ")[0], str(r[1] or "").strip()) for r in (table or []) if r and str(r[0]).strip()]


def identifier(dossier, choisis, progress=gr.Progress()):
    """Qwen3-VL regarde chaque objet coché et dit ce qu'il représente, en objet 3D autonome (mode « tuile3d ») :
    une tuile vue de dessus n'est pas l'objet (trou + échelle → une échelle). Renvoie (tableau, message)."""
    d, lot = _lot(dossier)
    pieces = [p for p in lot["pieces"] if libelle(p) in (choisis or [])]
    if not pieces:
        raise gr.Error("Coche au moins un objet.")
    for i, p in enumerate(pieces):
        progress(i / len(pieces), desc=f"Que représente {p['nom']} ? ({i + 1}/{len(pieces)}, Qwen3-VL)")
        p["description"] = diffusion.decrire("tuile3d", image=p["reference"], dossier=d)
    _ecrire_lot(d, lot)
    return ([[p["nom"], p["description"]] for p in pieces],
            "🔎 Vérifie ce que chaque objet doit être et corrige le texte si besoin (« une échelle en bois », en "
            "français ou en anglais), puis « Dessiner les objets ».")


def _descriptions(d, lot, table, progress):
    """Objets du tableau avec leur description (Qwen pour celles qui manquent)."""
    par_nom = {p["nom"]: p for p in lot["pieces"]}
    pieces = []
    for nom, texte in _lignes(table):
        if nom not in par_nom:
            continue
        p = par_nom[nom]
        if not texte:
            progress(0, desc=f"Que représente {nom} ? (Qwen3-VL)")
            texte = diffusion.decrire("tuile3d", image=p["reference"], dossier=d)
        p["description"] = texte
        pieces.append(p)
    if not pieces:
        raise gr.Error("Identifie d'abord les objets cochés (bouton « 🔎 Identifier les objets »).")
    return pieces


def prompt_objet(description):
    return f"{description.rstrip('. ')}. {STYLE_3D}, {modele3d.OBJET_SEUL}"


def dessiner(dossier, table, garder_dessin=False, graine=0, progress=gr.Progress()):
    """Une image 3D propre par objet du tableau : Z-Image d'après la description (par défaut ; FLUX.2 klein, à qui
    on donnait le pixel art, le recopiait), ou FLUX.2 klein avec le dessin d'origine en référence. Renvoie
    (galerie, message, dossier)."""
    d, lot = _lot(dossier)
    pieces = _descriptions(d, lot, table, progress)
    graine = int(graine or 0) or diffusion.graines(1)[0]
    (d / "rendus").mkdir(exist_ok=True)
    sorties = [d / "rendus" / f"{p['nom']}.png" for p in pieces]
    prompts = [prompt_objet(p["description"]) for p in pieces]
    graines = [graine + i for i in range(len(pieces))]
    if garder_dessin:
        for i, p in enumerate(pieces):
            progress(i / len(pieces), desc=f"Rendu 3D de {p['nom']} ({i + 1}/{len(pieces)}, FLUX.2 klein)…")
            largeur, hauteur = _format(p["reference"])
            diffusion.personnage(f"{prompts[i]}. {RENDU_3D}.", [p["reference"]], [sorties[i]], [graines[i]], largeur,
                                 hauteur)
    else:
        progress(0, desc=f"Dessin de {len(pieces)} objet(s) (Z-Image)…")
        diffusion.image(prompts if len(prompts) > 1 else prompts[0], sorties, graines, 1024, 1024)
    for p, sortie, g, prompt in zip(pieces, sorties, graines, prompts):
        p.update(rendu=str(sortie), graine_rendu=g, prompt=prompt, description_rendu=p["description"],
                 moteur_rendu="klein" if garder_dessin else "zimage")
    lot["graine"] = graine
    _ecrire_lot(d, lot)
    return ([(str(s_), p["nom"]) for p, s_ in zip(pieces, sorties)],
            f"🎨 {len(pieces)} objet(s) dessiné(s) (graine {graine}). S'il y en a un raté, corrige sa description ou "
            "change la graine et redessine ; sinon « Créer les modèles 3D ».", str(d))


def generer(dossier, table, garder_dessin=False, qualite=QUALITE, texture=True, graine=0, projet=None,
            progress=gr.Progress()):
    """Objets du tableau → modèles 3D (Hunyuan3D-2) d'après leurs images dessinées (dessinées d'abord si besoin :
    tous les dessins, puis toute la 3D, le moteur ne change de modèle qu'une fois). Un objet en échec n'arrête pas
    les autres. Renvoie (message, dernier GLB, fichiers, dossier)."""
    d, lot = _lot(dossier)
    pieces = _descriptions(d, lot, table, progress)
    a_dessiner = [p for p in pieces if not p.get("rendu") or not Path(p["rendu"]).exists()
                  or p.get("description_rendu") != p["description"]]  # description corrigée depuis le dessin
    if a_dessiner:
        dessiner(dossier, [[p["nom"], p["description"]] for p in a_dessiner], garder_dessin, graine, progress=progress)
        d, lot = _lot(dossier)
        par_nom = {p["nom"]: p for p in lot["pieces"]}
        pieces = [par_nom[p["nom"]] for p in pieces]
    graine = int(graine or 0) or lot.get("graine") or diffusion.graines(1)[0]
    lignes, produits, dernier = [], [], None
    for i, p in enumerate(pieces):
        progress(i / len(pieces), desc=f"Modèle 3D de {p['nom']} ({i + 1}/{len(pieces)}, Hunyuan3D)…")
        try:
            msg, glb, _, _, creation = modele3d.generer(
                p["rendu"], f"{lot['atlas'].rsplit('.', 1)[0]}_{p['nom']}", qualite, texture, graine + i, ["glb"],
                description=p.get("prompt"), projet=projet)
        except gr.Error as e:
            lignes.append(f"- ❌ {p['nom']} : {getattr(e, 'message', e)}")
            continue
        copie = d / f"{p['nom']}.glb"
        shutil.copy(glb, copie)
        p.update(glb=str(copie), creation=creation)
        produits.append(str(copie))
        dernier = str(copie)
        lignes.append(f"- ✅ {p['nom']} ({p['description'][:60]}) : {msg.split(' : ', 1)[-1].split('. Dossier')[0]}")
    lot.update(qualite=qualite)
    _ecrire_lot(d, lot)
    return (f"**{len(produits)}/{len(pieces)} objet(s) en 3D** (GLB dans {d}, chacun aussi dans la Galerie) :\n\n"
            + "\n".join(lignes), dernier, produits, str(d))


def _format(image):
    """Format de klein d'après les proportions de l'objet (côtés multiples de 16, ~1 Mpx)."""
    from .images import AUTO, format_pour

    return format_pour(AUTO, image)
