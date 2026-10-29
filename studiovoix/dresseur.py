"""Kit de dresseur (onglet « Pixel art et sprites ») : à partir d'une photo, le sprite de combat de face, la feuille
de lancer vue de dos et la feuille de marche, dans le style des jeux de rôle portables GBA (64 px, 15 couleurs) ou
DS (80 px, 15 couleurs).

Aucun modèle nouveau. Sprite de face : comme les personnages de pixel_art (Qwen3-VL décrit la personne, FLUX.2 klein
la dessine, BiRefNet, `pixeliser`). Feuilles : un modèle de 4 B ne dessine pas seul une feuille de marche de 16 images
cohérentes ; FLUX.2 klein reçoit le sprite de face (image 1) et une **feuille d'exemple** de l'utilisateur (image 2,
« gabarit » : mêmes poses, même mise en page) et redessine la feuille avec ce personnage. FLUX.2 klein recopie
fidèlement ses références : défaut pour un style, qualité ici (poses, cadrage, nombre d'images). La feuille entière
part en une seule image, donc le personnage est le même partout. Puis chaque case est ramenée à la grille du gabarit
(couleur dominante par pixel, palette du sprite de face), le fond vif retiré, la silhouette limitée aux alentours de
celle du gabarit, contour d'un pixel.

Les gabarits viennent de l'utilisateur (jamais versionnés : sprites de jeux du commerce) et sont gardés dans
data/pixel_art/gabarits/<style>/<feuille>.png pour les fois suivantes. Rangement : data/pixel_art/<horodatage>_<nom>/.
"""
import json
import math
import shutil
import time
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from . import config as cfg
from . import diffusion
from . import pixel_art as px
from .outils import nouveau_dossier
from .serie import _slug

# Mesuré sur les exemples de l'utilisateur (28/10) : GBA = sprites de face 64×64 affichés ×2, 15 couleurs + la
# transparence (palette de 16) ; lancer de dos = 5 images de 64×64 ; marche = 4 lignes (bas, gauche, droite, haut) de
# 4 images (debout, pas, debout, autre pas) de 16×24. DS = sprites de face d'environ 73 px de haut sur 80, 15 couleurs ;
# marche = 4 lignes (haut, bas, gauche, droite) de 3 images (debout, pas, autre pas) de 32×32. Pas d'exemple de lancer
# DS : la feuille GBA sert de gabarit, ses cases agrandies à 80×80.
STYLES = {
    "Rouge Feu / Vert Feuille (GBA, 64 px)": {
        "cle": "gba", "toile": 64, "hauteur": 60, "couleurs": 15, "agrandissement": 2,
        "texte": ("crisp 16-color pixel art from the Game Boy Advance era of handheld role-playing games: chunky "
                  "pixels, thick dark outline, flat cel shading with two tones per color, no anti-aliasing"),
        "lancer": {"grille": (5, 1), "cellule": (64, 64)},
        "marche": {"grille": (4, 4), "cellule": (16, 24), "lignes": ["bas", "gauche", "droite", "haut"],
                   "cycle": [0, 1, 2, 3]},
    },
    "Noir et Blanc (DS, 80 px)": {
        "cle": "ds", "toile": 80, "hauteur": 74, "couleurs": 15, "agrandissement": 2,
        "texte": ("detailed pixel art from the Nintendo DS era of handheld role-playing games: slightly taller "
                  "proportions, thin dark outline, cel shading with three tones per color, no anti-aliasing"),
        "lancer": {"grille": (5, 1), "cellule": (80, 80)},
        "marche": {"grille": (3, 4), "cellule": (32, 32), "lignes": ["haut", "bas", "gauche", "droite"],
                   "cycle": [0, 1, 0, 2]},
    },
}
FACE = "Sprite de combat de face"
LANCER = "Feuille de lancer (de dos)"
MARCHE = "Feuille de marche"
FEUILLES = [FACE, LANCER, MARCHE]
NOMS_FEUILLES = {LANCER: "lancer", MARCHE: "marche"}

POSE_FACE = ("Full body trainer battle sprite: the whole body from head to feet, three-quarter view, confident "
             "dynamic pose holding a small red and white ball, centered on a plain flat white background, no shadow, "
             "no ground, no text")
CONSIGNE_FEUILLE = (
    "Image 2 is a pixel art sprite sheet. Redraw image 2 exactly: the same layout, the same number of figures at the "
    "same places, the same poses and viewing angles, the same size and framing of every figure, in {style}. Only the "
    "character changes: every figure becomes the character of image 1, with the same hair, face, skin tone, clothes, "
    "accessories and colors as image 1. The character: {description} Keep the plain flat solid {couleur} background "
    "everywhere, with nothing else on it: no shadow, no ground, no text.")
PRECISIONS = {"lancer": "The figures are seen from behind and throw a small red and white ball forward.",
              "marche": "The figures are small top-down role-playing game walking sprites, one direction per row."}
COTE_MAX = 2048  # côté le plus long donné à FLUX.2 klein
AIRE_MAX = 1_300_000
MARGE = {"lancer": 2, "marche": 1}  # pixels autour de la silhouette du gabarit où le nouveau personnage peut déborder
ECART_FOND_KIT = 80


# --- Gabarits ------------------------------------------------------------------------------------------------
def dossier_gabarits(style):
    return cfg.PIXEL_DIR / "gabarits" / STYLES[style]["cle"]


def gabarit_enregistre(style, feuille):
    f = dossier_gabarits(style) / f"{feuille}.png"
    return f if f.is_file() else None


def etat_gabarits(style):
    lignes = []
    for libelle, feuille in NOMS_FEUILLES.items():
        f = gabarit_enregistre(style, feuille)
        lignes.append(f"- {libelle} : " + (f"✅ gabarit enregistré (`{f}`)" if f else
                                            ("ℹ️ aucun : celui du style GBA servira" if feuille == "lancer"
                                             and gabarit_enregistre(_autre_style(style), "lancer") else
                                             "❌ aucun gabarit : dépose une feuille d'exemple")))
    return "\n".join(lignes)


def _autre_style(style):
    return next(s for s in STYLES if s != style)


def enregistrer_gabarit(style, feuille, fichier):
    """Copie la feuille d'exemple (vérifiée) dans data/pixel_art/gabarits/<style>/<feuille>.png."""
    chemin = getattr(fichier, "name", fichier)
    lire_gabarit(chemin, style, feuille)  # erreur claire si la grille ne tombe pas juste
    d = dossier_gabarits(style)
    d.mkdir(parents=True, exist_ok=True)
    with Image.open(chemin) as im:
        im.convert("RGBA").save(d / f"{feuille}.png")
    return d / f"{feuille}.png"


def lire_gabarit(chemin, style, feuille):
    """Feuille d'exemple → tableau RGBA à 1 pixel par pixel du jeu, fond transparent, cases à la taille du style.
    Renvoie (gabarit, (colonnes, lignes), (largeur, hauteur) d'une case)."""
    with Image.open(chemin) as im:
        im = im.convert("RGBA")
    if np.array(im)[..., 3].min() == 255:
        im = px._sans_fond_uni(im)
    a, _, _ = px.reduire(np.array(im))
    cols, rows = STYLES[style][feuille]["grille"]
    if a.shape[1] % cols or a.shape[0] % rows:
        raise gr.Error(f"La feuille d'exemple fait {a.shape[1]}×{a.shape[0]} pixels de jeu : elle doit se découper "
                       f"en {cols} colonnes et {rows} lignes de cases égales (une seule feuille d'un seul "
                       "personnage, sans marge).")
    cw, ch = STYLES[style][feuille]["cellule"]
    gw, gh = a.shape[1] // cols, a.shape[0] // rows
    if (gw, gh) != (cw, ch):  # gabarit d'un autre style (lancer GBA pour la DS) : cases agrandies au plus proche
        cases = [Image.fromarray(np.ascontiguousarray(a[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]), "RGBA")
                 .resize((cw, ch), Image.Resampling.NEAREST) for r in range(rows) for c in range(cols)]
        a = np.zeros((rows * ch, cols * cw, 4), np.uint8)
        for i, case in enumerate(cases):
            r, c = divmod(i, cols)
            a[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = np.array(case)
    return np.ascontiguousarray(a), (cols, rows), (cw, ch)


def _gabarit(style, feuille, fichier):
    """Gabarit donné (enregistré pour la suite), sinon celui enregistré, sinon (lancer) celui de l'autre style."""
    if fichier:
        return enregistrer_gabarit(style, feuille, fichier)
    f = gabarit_enregistre(style, feuille) or (gabarit_enregistre(_autre_style(style), feuille)
                                                if feuille == "lancer" else None)
    if not f:
        raise gr.Error(f"Pas de feuille d'exemple pour « {next(k for k, v in NOMS_FEUILLES.items() if v == feuille)} »"
                       " : dépose-en une dans « Feuilles d'exemple » (une seule feuille d'un personnage).")
    return f


# --- Mise en page pour FLUX.2 klein ----------------------------------------------------------------------------
def disposition(n, cw, ch):
    """Colonnes de la mise en page la plus proche d'un carré pour n cases (une bande de 5 cases donnait une image
    de 5:1, mal adaptée au générateur)."""
    return min(range(1, n + 1), key=lambda c: abs(math.log((c * cw) / (math.ceil(n / c) * ch))))


def _cases(a, grille, cellule):
    (cols, rows), (cw, ch) = grille, cellule
    return [a[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] for r in range(rows) for c in range(cols)]


def _assembler(cases, colonnes, cellule):
    cw, ch = cellule
    lignes = math.ceil(len(cases) / colonnes)
    a = np.zeros((lignes * ch, colonnes * cw, cases[0].shape[2]), cases[0].dtype)
    for i, case in enumerate(cases):
        r, c = divmod(i, colonnes)
        a[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = case
    return a


def agrandissement_pour(largeur, hauteur):
    """Facteur entier qui donne une image de côtés multiples de 16, ~1 Mpx au plus. Renvoie (k, largeur, hauteur)
    après complément éventuel (bande transparente à droite et en bas)."""
    for pas in (1, 2, 4, 8, 16):
        w, h = math.ceil(largeur / pas) * pas, math.ceil(hauteur / pas) * pas
        for k in range(16, 0, -1):
            if (w * k) % 16 or (h * k) % 16 or max(w, h) * k > COTE_MAX or w * h * k * k > AIRE_MAX:
                continue
            return k, w, h
    raise gr.Error(f"Feuille d'exemple trop grande ({largeur}×{hauteur} pixels de jeu).")


def _sur_fond(a, fond, k):
    """RGBA → RGB sur le fond vif, agrandi ×k au plus proche."""
    rgb = np.where(a[..., 3:] >= 128, a[..., :3], np.array(fond, np.uint8))
    return Image.fromarray(np.ascontiguousarray(rgb), "RGB").resize((a.shape[1] * k, a.shape[0] * k),
                                                                   Image.Resampling.NEAREST)


def vers_grille(image, k, palette, fond, zone, contour):
    """Image de FLUX.2 klein (k pixels par pixel de jeu) → feuille RGBA à la grille du gabarit : fond (couleur vive
    demandée, ou celle qu'il a prise hors de la zone) transparent, couleur dominante de la palette par pixel, rien
    hors de la zone (silhouette du gabarit élargie), contour d'un pixel."""
    h, w = zone.shape
    rgb = np.array(image.convert("RGB").resize((w * k, h * k), Image.Resampling.NEAREST))
    s = px.ECHELLE_TRAVAIL
    fin = np.array(Image.fromarray(rgb, "RGB").resize((w * s, h * s), Image.Resampling.BOX)).astype(int)
    grande_zone = np.repeat(np.repeat(zone, s, axis=0), s, axis=1)
    actuel = np.median(fin[~grande_zone], axis=0) if (~grande_zone).any() else np.array(fond)
    ecart = np.minimum(np.sqrt(((fin - np.array(fond)) ** 2).sum(-1)), np.sqrt(((fin - actuel) ** 2).sum(-1)))
    alpha = np.where((ecart > ECART_FOND_KIT) & grande_zone, 255, 0)
    rgba = np.dstack([fin, alpha]).astype(np.uint8)
    feuille = px.blocs_dominants(rgba, np.asarray(palette, np.uint8), h, w)
    return px.contourer(feuille, contour)


# --- Création du kit ---------------------------------------------------------------------------------------------
def _description(photo, demande, progress):
    """Description anglaise de la personne (Qwen3-VL voit la photo et applique les modifications demandées)."""
    if diffusion.present("qwen"):
        return diffusion.decrire("sprite", demande, image=photo, progress=progress).strip()
    return demande.strip()


def _face(style, photo, description, graine, dossier, progress):
    """Sprite de face : FLUX.2 klein d'après la photo, BiRefNet, puis la grille et la palette du style."""
    st = STYLES[style]
    source = dossier / "face_source.png"
    prompt = f"{description} {px.MEME_PERSONNE} {POSE_FACE}, in {st['texte']}."
    diffusion.personnage(prompt, [photo], [source], [graine], 768, 1024, progress=progress)
    image = source
    if diffusion.present("photo_detourage"):
        detoure = dossier / "face_detoure.png"
        try:
            diffusion.detourer_lot([source], [detoure], progress=progress)
            image = detoure if detoure.exists() else source
        except gr.Error:
            image = source  # fond uni retiré par pixeliser
    sprite, _ = px.pixeliser(Image.open(image), st["hauteur"], st["couleurs"], st["toile"], px.CONTOUR_DEFAUT)
    return sprite, prompt


def _feuille(style, feuille, gabarit_fichier, sprite, description, graine, dossier, progress):
    """Une feuille (lancer ou marche) redessinée par FLUX.2 klein d'après le gabarit, ramenée à la grille."""
    from scipy import ndimage

    st = STYLES[style]
    gabarit, grille, cellule = lire_gabarit(gabarit_fichier, style, feuille)
    cases = _cases(gabarit, grille, cellule)
    colonnes = disposition(len(cases), *cellule)
    page = _assembler(cases, colonnes, cellule)  # mise en page envoyée au générateur
    k, w, h = agrandissement_pour(page.shape[1], page.shape[0])
    page = np.pad(page, ((0, h - page.shape[0]), (0, w - page.shape[1]), (0, 0)))
    sprite_a = np.array(sprite.convert("RGBA"))
    palette = np.unique(sprite_a[..., :3][sprite_a[..., 3] >= 128].reshape(-1, 3), axis=0)
    fond = px._fond_eloigne(palette)
    ref_gabarit = dossier / f"{feuille}_gabarit.png"
    _sur_fond(page, fond, k).save(ref_gabarit)
    ref_sprite = dossier / f"{feuille}_personnage.png"
    ks = max(1, 512 // max(sprite_a.shape[:2]))
    _sur_fond(sprite_a, fond, ks).save(ref_sprite)
    prompt = CONSIGNE_FEUILLE.format(style=st["texte"], description=description,
                                     couleur=px.NOMS_FONDS.get(fond, "bright green")) + " " + PRECISIONS[feuille]
    sortie = dossier / f"{feuille}_dessin.png"
    diffusion.personnage(prompt, [ref_sprite, ref_gabarit], [sortie], [graine], w * k, h * k, progress=progress)
    zone = ndimage.binary_dilation(page[..., 3] >= 128, iterations=MARGE[feuille])
    page_finale = vers_grille(Image.open(sortie), k, palette, fond, zone, px.CONTOUR_DEFAUT)
    nouvelles = _cases(page_finale[:math.ceil(len(cases) / colonnes) * cellule[1], :colonnes * cellule[0]],
                       (colonnes, math.ceil(len(cases) / colonnes)), cellule)[:len(cases)]
    return _assembler(nouvelles, grille[0], cellule), grille, cellule, prompt


def _durees_lancer(n):
    return [250] + [100] * (n - 2) + [500] if n > 1 else [500]


def _ecrire_feuille(a, nom, dossier, agrandissement):
    im = Image.fromarray(a, "RGBA")
    im.save(dossier / f"{nom}.png")
    px.agrandir(im, agrandissement).save(dossier / f"{nom}_x{agrandissement}.png")
    return [dossier / f"{nom}.png", dossier / f"{nom}_x{agrandissement}.png"]


def creer_kit(style, photo, demande, feuilles, gabarit_lancer, gabarit_marche, face_existante, graine,
              nom="dresseur", progress=gr.Progress()):
    """Kit de dresseur d'après une photo. Renvoie (message, aperçus, fichiers, dossier, sprite de face)."""
    if style not in STYLES:
        raise gr.Error("Choisis un style.")
    feuilles = list(feuilles or [])
    if not feuilles:
        raise gr.Error("Coche au moins un élément à créer (sprite de face, lancer, marche).")
    photo = getattr(photo, "name", photo)
    face_existante = getattr(face_existante, "name", face_existante)
    if FACE not in feuilles and not face_existante:
        raise gr.Error("Sans « Sprite de combat de face », donne un sprite de face déjà créé : les feuilles en "
                       "reprennent le personnage et les couleurs.")
    if not photo and (FACE in feuilles or not (demande or "").strip()):
        raise gr.Error("Ajoute la photo de la personne.")
    st = STYLES[style]
    gabarits = {f: _gabarit(style, f, g) for f, g in (("lancer", gabarit_lancer), ("marche", gabarit_marche))
                if next(k for k, v in NOMS_FEUILLES.items() if v == f) in feuilles}
    diffusion.verifier("personnages")
    dossier = nouveau_dossier(cfg.PIXEL_DIR, _slug(nom, "dresseur"))
    graine = int(graine or 0) or diffusion.graines(1)[0]
    debut = time.monotonic()
    copie = None
    if photo:
        copie = dossier / f"photo{Path(photo).suffix.lower() or '.png'}"
        shutil.copy(photo, copie)
    progress(0.02, desc="Description de la personne (Qwen3-VL)…")
    description = _description(copie, (demande or "").strip() or "the person in the photo, as a trainer",
                               progress) if copie else demande.strip()
    infos = {"style": style, "demande": demande, "description": description, "graine": graine, "feuilles": {}}
    fichiers, apercus = [], []
    k_aff = st["agrandissement"]
    if FACE in feuilles:
        progress(0.1, desc="Sprite de face (FLUX.2 klein)…")
        sprite, prompt = _face(style, copie, description, graine, dossier, progress)
        infos["feuilles"]["face"] = {"prompt": prompt}
    else:
        sprite, _ = px._sprite_de(face_existante)
        if sprite.size != (st["toile"], st["toile"]):
            sprite, _ = px.pixeliser(sprite, st["hauteur"], st["couleurs"], st["toile"], px.CONTOUR_DEFAUT)
    fichiers += _ecrire_feuille(np.array(sprite.convert("RGBA")), "face", dossier, k_aff)
    apercus.append(str(dossier / f"face_x{k_aff}.png"))
    for n, (feuille, gabarit_fichier) in enumerate(gabarits.items(), 1):
        progress(0.3 + 0.3 * (n - 1), desc=f"Feuille de {feuille} (FLUX.2 klein)…")
        a, grille, cellule, prompt = _feuille(style, feuille, gabarit_fichier, sprite, description, graine, dossier,
                                              progress)
        fichiers += _ecrire_feuille(a, feuille, dossier, k_aff)
        apercus.append(str(dossier / f"{feuille}_x{k_aff}.png"))
        cases = [Image.fromarray(np.ascontiguousarray(c), "RGBA") for c in _cases(a, grille, cellule)]
        if feuille == "lancer":
            fichiers += px.ecrire_animation(cases, dossier, "lancer", k_aff, _durees_lancer(len(cases)))
            apercus.append(str(dossier / f"lancer_x{k_aff}.gif"))
        else:
            cols = grille[0]
            for r, direction in enumerate(st["marche"]["lignes"]):
                cycle = [cases[r * cols + i] for i in st["marche"]["cycle"]]
                gif = dossier / f"marche_{direction}_x{k_aff}.gif"
                px._gif([px.agrandir(c, k_aff) for c in cycle], gif, 150)
                fichiers.append(gif)
                apercus.append(str(gif))
        infos["feuilles"][feuille] = {"prompt": prompt, "gabarit": str(gabarit_fichier), "grille": list(grille),
                                      "cellule": list(cellule)}
    (dossier / "kit.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    archive = dossier / f"{dossier.name}.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for f in sorted(dossier.rglob("*")):
            if f.is_file() and f != archive and not f.name.endswith(("_dessin.png", "_source.png", "_detoure.png",
                                                                      "_gabarit.png", "_personnage.png")):
                z.write(f, f.relative_to(dossier))
    fichiers.append(archive)
    msg = (f"✅ Kit de dresseur en {time.monotonic() - debut:.0f} s (graine {graine}). Dossier : {dossier}. "
           "Le sprite de face est placé dans « Animer un sprite » pour sa feuille d'animation.")
    return msg, apercus, [str(f) for f in fichiers], str(dossier), str(dossier / "face.png")
