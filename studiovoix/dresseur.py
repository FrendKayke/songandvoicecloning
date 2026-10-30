"""Kit de dresseur (onglet « Pixel art et sprites ») : sprite de combat de face, feuille de lancer vue de dos et
feuille de marche aux couleurs d'une personne, dans le style exact des sprites d'exemple de l'utilisateur (jeux de rôle
portables GBA ou DS).

Premier essai (29/10, FLUX.2 klein qui redessinait les feuilles d'exemple avec le personnage) : style perdu (sprite
de face = image réaliste pixelisée, bruitée, proportions réalistes) et feuilles où klein recopiait le personnage de
l'exemple. Aucun générateur installé ne connaît ce style. D'où le **recoloriage** (« palette swap », comme dans les
jeux) : les sprites d'exemple (« gabarits ») sont gardés au pixel près — forme, ombrage, contour —, seules leurs
couleurs changent. Pour savoir quelle couleur va où (cheveux, peau, vêtements), FLUX.2 klein recolore une seule fois
toutes les feuilles assemblées d'après les couleurs de la personne (lues sur la photo par Qwen3-VL, modifiables) ; on
ne garde de son image que la couleur médiane peinte sur chaque couleur du gabarit. Rien de son dessin ne passe dans
le sprite : le style reste exactement celui de l'exemple, même si klein déforme. Option « dessiner d'après la photo »
pour le sprite de face (forme de la personne, style approché).

Gabarits : data/pixel_art/gabarits/<gba|ds>/<face|lancer|marche>/<nom>.png (jamais versionnés : sprites de jeux du
commerce). Rangement : data/pixel_art/<horodatage>_<nom>/.
"""
import colorsys
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
from . import modele_sprites
from . import pixel_art as px
from .outils import nouveau_dossier
from .serie import _slug

# Mesuré sur les exemples de l'utilisateur (28/10) : GBA = sprites de face 64×64 affichés ×2, 15 couleurs + la
# transparence ; lancer de dos = 5 images de 64×64 ; marche = 4 lignes (bas, gauche, droite, haut) de 4 images
# (debout, pas, debout, pas) de 16×24. DS = sprites de face d'environ 73 px de haut sur 80 ; marche = 4 lignes (haut,
# bas, gauche, droite) de 3 images (debout, pas, autre pas) de 32×32. Pas d'exemple de lancer DS : celui du GBA sert.
STYLES = {
    "Rouge Feu / Vert Feuille (GBA, 64 px)": {
        "cle": "gba", "toile": 64, "hauteur": 60, "couleurs": 15, "agrandissement": 2,
        "texte": ("crisp 16-color pixel art from the Game Boy Advance era of handheld role-playing games: chunky "
                  "pixels, thick dark outline, flat cel shading with two tones per color, no anti-aliasing"),
        "lancer": {"cellule": 64},
        "marche": {"grille": (4, 4), "cellule": (16, 24), "lignes": ["bas", "gauche", "droite", "haut"],
                   "cycle": [0, 1, 2, 3]},
    },
    "Noir et Blanc (DS, 80 px)": {
        "cle": "ds", "toile": 80, "hauteur": 74, "couleurs": 15, "agrandissement": 2,
        "texte": ("detailed pixel art from the Nintendo DS era of handheld role-playing games: slightly taller "
                  "proportions, thin dark outline, cel shading with three tones per color, no anti-aliasing"),
        "lancer": {"cellule": 80},
        "marche": {"grille": (3, 4), "cellule": (32, 32), "lignes": ["haut", "bas", "gauche", "droite"],
                   "cycle": [0, 1, 0, 2]},
    },
}
FACE = "Sprite de combat de face"
LANCER = "Feuille de lancer (de dos)"
MARCHE = "Feuille de marche"
FEUILLES = [FACE, LANCER, MARCHE]
CLES_FEUILLES = {FACE: "face", LANCER: "lancer", MARCHE: "marche"}
LIBELLES_FEUILLES = {v: k for k, v in CLES_FEUILLES.items()}
LORA = "Dessiner avec le modèle de sprites de dresseurs (SDXL + LoRA, style du jeu)"
RECOLORER = "Recolorer le sprite d'exemple (style exact, forme de l'exemple)"
MARCHE_EXEMPLE = "— recolorer un sprite d'exemple —"
DESSINER = "Dessiner avec FLUX.2 klein d'après la photo (style approché)"
MODES_FACE = [LORA, RECOLORER, DESSINER]
VARIANTES_MAX = 6
# Forme des légendes d'entraînement du LoRA (« a pixel image of a man with a pokeball », étiquette de génération :
# gen3 = Rouge Feu / Vert Feuille et Émeraude, 164 images ; gen5 = Noir et Blanc, 6 images seulement)
PROMPT_LORA = "a pixel image of {description}, {gen}, full body, standing, simple background"
GEN = {"gba": "gen3", "ds": "gen5"}

POSE_FACE = ("Full body trainer battle sprite: the whole body from head to feet, slightly stylized proportions with a "
             "large head, three-quarter view, confident dynamic pose holding a small red and white ball, flat colors "
             "without texture or gradient, centered on a plain flat white background, no shadow, no ground, no text")
CONSIGNE_RECOLORER = (
    "Image 1 is a sheet of pixel art game sprites on a plain flat {fond} background. Recolor every character of image 1 "
    "so that it has these colors: {couleurs}. Keep exactly the same drawing: every shape, pixel, outline, pose and "
    "position unchanged, the same light and dark shading of each area, black outlines stay black, and the {fond} "
    "background stays exactly the same. Only the colors of hair, skin, clothes and accessories change.")
AVEC_REFERENCE = " The character must look like image 2: use exactly the colors of image 2 for each part."
COTE_MAX = 2048  # côté le plus long donné à FLUX.2 klein
AIRE_MAX = 1_300_000
ESPACE = 4  # pixels de jeu entre deux feuilles dans l'image donnée à klein
COULEURS_DEFAUT = "hair: dark brown; skin: light peach; top: red; bottom: dark blue; shoes: black; accessories: white"


# --- Gabarits ------------------------------------------------------------------------------------------------
def dossier_gabarits(style, feuille):
    return cfg.PIXEL_DIR / "gabarits" / STYLES[style]["cle"] / feuille


def _migrer(style, feuille):
    """Gabarit de la première version (gabarits/<style>/<feuille>.png) rangé dans le dossier de la feuille."""
    ancien = cfg.PIXEL_DIR / "gabarits" / STYLES[style]["cle"] / f"{feuille}.png"
    if ancien.is_file():
        d = dossier_gabarits(style, feuille)
        d.mkdir(parents=True, exist_ok=True)
        ancien.replace(d / f"exemple_{int(ancien.stat().st_mtime)}.png")


def liste_gabarits(style, feuille, autres=True):
    """Noms des gabarits enregistrés, du plus récent au plus ancien (lancer DS : ceux du GBA s'il n'y en a pas)."""
    _migrer(style, feuille)
    d = dossier_gabarits(style, feuille)
    noms = [f.stem for f in sorted(d.glob("*.png"), key=lambda f: -f.stat().st_mtime)] if d.is_dir() else []
    if not noms and feuille == "lancer" and autres:
        autre = next(s for s in STYLES if s != style)
        return [f"{STYLES[autre]['cle']}:{n}" for n in liste_gabarits(autre, "lancer", autres=False)]
    return noms


def chemin_gabarit(style, feuille, nom):
    if not nom:
        return None
    if ":" in nom:  # gabarit d'un autre style (lancer GBA pour la DS)
        cle, nom = nom.split(":", 1)
        style = next(s for s in STYLES if STYLES[s]["cle"] == cle)
    f = dossier_gabarits(style, feuille) / f"{nom}.png"
    return f if f.is_file() else None


def apercu_gabarit(style, feuille, nom):
    f = chemin_gabarit(style, feuille, nom)
    if not f:
        return None
    a = lire_gabarit(f, style, feuille)
    return px.agrandir(Image.fromarray(a, "RGBA"), 2)


def ajouter_gabarits(style, feuille_libelle, fichiers):
    """Sprites d'exemple ajoutés à la bibliothèque (chacun vérifié). Renvoie (message, choix des trois listes)."""
    feuille = CLES_FEUILLES.get(feuille_libelle)
    fichiers = [getattr(f, "name", f) for f in (fichiers or [])]
    if not feuille or not fichiers:
        raise gr.Error("Choisis le type de feuille et dépose un ou plusieurs sprites d'exemple.")
    d = dossier_gabarits(style, feuille)
    d.mkdir(parents=True, exist_ok=True)
    ajoutes, erreurs = [], []
    for f in fichiers:
        try:
            lire_gabarit(f, style, feuille)
        except gr.Error as e:
            erreurs.append(f"{Path(f).name} : {e}")
            continue
        nom = _slug(Path(f).stem, "exemple")
        with Image.open(f) as im:
            im.convert("RGBA").save(d / f"{nom}.png")
        ajoutes.append(nom)
    msg = (f"✅ {len(ajoutes)} exemple(s) ajouté(s) : {', '.join(ajoutes)}." if ajoutes else "")
    if erreurs:
        msg += "\n\n⚠️ " + "\n".join(erreurs)
    return (msg,) + mises_a_jour_listes(style)


def mises_a_jour_listes(style):
    sorties = []
    for feuille in ("face", "lancer", "marche"):
        noms = liste_gabarits(style, feuille)
        sorties.append(gr.update(choices=noms, value=noms[0] if noms else None))
    return tuple(sorties)


def _cadre(a):
    """Recadré sur ce qui est opaque."""
    lignes, colonnes = np.where(a[..., 3] >= 128)
    if not len(lignes):
        raise gr.Error("Le sprite d'exemple est vide (rien d'opaque).")
    return a[lignes.min():lignes.max() + 1, colonnes.min():colonnes.max() + 1]


def lire_gabarit(chemin, style, feuille):
    """Sprite d'exemple → tableau RGBA à 1 pixel par pixel du jeu, fond transparent, à la taille du style.
    face : un personnage (recadré, posé en bas au centre de la toile) ; lancer : une rangée d'images carrées ;
    marche : la grille du style."""
    with Image.open(chemin) as im:
        im = im.convert("RGBA")
    if np.array(im)[..., 3].min() == 255:
        im = px._sans_fond_uni(im)
    a, _, _ = px.reduire(np.array(im))
    st = STYLES[style]
    if feuille == "face":
        a = _cadre(a)
        t = st["toile"]
        if a.shape[0] > t or a.shape[1] > t:
            raise gr.Error(f"Le sprite d'exemple fait {a.shape[1]}×{a.shape[0]} pixels de jeu : plus grand que la "
                           f"toile du style ({t}×{t}). Prends un sprite de ce jeu-là.")
        toile = np.zeros((t, t, 4), np.uint8)
        y, x = t - a.shape[0], (t - a.shape[1]) // 2
        toile[y:, x:x + a.shape[1]] = a
        return toile
    if feuille == "lancer":
        h, w = a.shape[:2]
        if w % h:
            raise gr.Error(f"La feuille de lancer fait {w}×{h} pixels de jeu : elle doit être une rangée d'images "
                           "carrées côte à côte (par exemple 5 images de 64×64), sans marge.")
        c = st["lancer"]["cellule"]
        if h != c:  # exemple d'un autre style (lancer GBA pour la DS) : agrandi au plus proche
            a = np.array(Image.fromarray(np.ascontiguousarray(a), "RGBA").resize((w // h * c, c),
                                                                                Image.Resampling.NEAREST))
        return np.ascontiguousarray(a)
    cols, rows = st["marche"]["grille"]
    cw, ch = st["marche"]["cellule"]
    if a.shape[1] % cols or a.shape[0] % rows:
        raise gr.Error(f"La feuille de marche fait {a.shape[1]}×{a.shape[0]} pixels de jeu : elle doit se découper "
                       f"en {cols} colonnes et {rows} lignes de cases égales (une feuille d'un seul personnage, sans "
                       "marge).")
    if (a.shape[1] // cols, a.shape[0] // rows) != (cw, ch):
        a = np.array(Image.fromarray(np.ascontiguousarray(a), "RGBA").resize((cols * cw, rows * ch),
                                                                            Image.Resampling.NEAREST))
    return np.ascontiguousarray(a)


# --- Mise en page pour FLUX.2 klein ----------------------------------------------------------------------------
def disposition(n, cw, ch):
    """Colonnes de la mise en page la plus proche d'un carré pour n cases."""
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
    après complément éventuel (bande vide à droite et en bas)."""
    for pas in (1, 2, 4, 8, 16):
        w, h = math.ceil(largeur / pas) * pas, math.ceil(hauteur / pas) * pas
        for k in range(16, 0, -1):
            if (w * k) % 16 or (h * k) % 16 or max(w, h) * k > COTE_MAX or w * h * k * k > AIRE_MAX:
                continue
            return k, w, h
    raise gr.Error(f"Sprites d'exemple trop grands ({largeur}×{hauteur} pixels de jeu).")


def planche(pages):
    """Feuilles côte à côte (bas alignés, ESPACE pixels entre elles). Une rangée d'images (lancer) est d'abord
    remise en page presque carrée (une bande de 5 images donnait une image de 5:1). Renvoie (planche, positions)
    avec positions = (y, x) de chaque page dans la planche."""
    hauteur = max(p.shape[0] for p in pages)
    largeur = sum(p.shape[1] for p in pages) + ESPACE * (len(pages) - 1)
    a = np.zeros((hauteur, largeur, 4), np.uint8)
    positions, x = [], 0
    for p in pages:
        y = hauteur - p.shape[0]
        a[y:, x:x + p.shape[1]] = p
        positions.append((y, x))
        x += p.shape[1] + ESPACE
    return a, positions


def _en_page(feuille, a, style):
    """Disposition de la feuille dans la planche (lancer : rangée remise en page carrée)."""
    if feuille != "lancer":
        return a
    c = a.shape[0]
    cases = _cases(a, (a.shape[1] // c, 1), (c, c))
    return _assembler(cases, disposition(len(cases), c, c), (c, c))


def _depuis_page(feuille, page, a):
    if feuille != "lancer":
        return page
    c = a.shape[0]
    n = a.shape[1] // c
    colonnes = disposition(n, c, c)
    cases = _cases(page, (colonnes, math.ceil(n / colonnes)), (c, c))[:n]
    return _assembler(cases, n, (c, c))


def _sur_fond(a, fond, k):
    """RGBA → RGB sur le fond uni, agrandi ×k au plus proche."""
    rgb = np.where(a[..., 3:] >= 128, a[..., :3], np.array(fond, np.uint8))
    return Image.fromarray(np.ascontiguousarray(rgb), "RGB").resize((a.shape[1] * k, a.shape[0] * k),
                                                                   Image.Resampling.NEAREST)


# --- Recoloriage -------------------------------------------------------------------------------------------------
def _contour(c):
    return max(c) < 40  # contours noirs ou presque : gardés tels quels


def correspondances(gabarit, image, k):
    """Couleur de chaque couleur du gabarit dans l'image recolorée (k pixels par pixel de jeu) : médiane des centres
    des pixels de cette couleur (le bord d'un carré mélange deux couleurs). Renvoie {couleur: nouvelle couleur}."""
    h, w = gabarit.shape[:2]
    rgb = np.array(image.convert("RGB").resize((w * k, h * k), Image.Resampling.NEAREST)).astype(int)
    centres = rgb[k // 2::k, k // 2::k][:h, :w]
    opaque = gabarit[..., 3] >= 128
    nouvelles = {}
    for c in np.unique(gabarit[..., :3][opaque].reshape(-1, 3), axis=0):
        c = tuple(int(x) for x in c)
        if _contour(c):
            nouvelles[c] = c
            continue
        m = opaque & (gabarit[..., :3] == c).all(-1)
        nouvelles[c] = tuple(int(x) for x in np.median(centres[m], axis=0))
    return _garder_les_nuances(nouvelles)


def _lum(c):
    return colorsys.rgb_to_hls(*(np.array(c) / 255))[1]


def _garder_les_nuances(nouvelles):
    """Deux nuances différentes du gabarit (clair / foncé d'une même zone) ne doivent pas devenir la même couleur :
    l'ombrage disparaîtrait. Une couleur trop proche d'une autre est éclaircie ou assombrie comme dans le gabarit."""
    res = dict(nouvelles)
    cles = [c for c in res if not _contour(c)]
    for i, a in enumerate(cles):
        for b in cles[i + 1:]:
            na, nb = np.array(res[a]), np.array(res[b])
            if np.abs(na - nb).sum() >= 24:
                continue
            ecart = _lum(a) - _lum(b)  # sens de l'ombrage dans le gabarit
            if abs(ecart) < 0.03:
                continue
            h, l, s = colorsys.rgb_to_hls(*(nb / 255))
            l = min(1.0, max(0.0, l - math.copysign(min(0.18, abs(ecart)), ecart)))
            res[b] = tuple(int(round(x * 255)) for x in colorsys.hls_to_rgb(h, l, s))
    return res


def recolorer(gabarit, nouvelles):
    a = gabarit.copy()
    opaque = a[..., 3] >= 128
    for ancienne, nouvelle in nouvelles.items():
        m = opaque & (gabarit[..., :3] == ancienne).all(-1)
        a[m, :3] = nouvelle
    return a


# --- Création du kit ---------------------------------------------------------------------------------------------
def preparer_description(photo, demande, progress=gr.Progress()):
    """Description anglaise du personnage (modifiable avant la création) : Qwen3-VL d'après la photo et les
    changements demandés, ou d'après la seule description (souvent en français) quand il n'y a pas de photo."""
    photo = getattr(photo, "name", photo)
    demande = (demande or "").strip()
    if not photo and not demande:
        raise gr.Error("Ajoute la photo de la personne, ou décris le personnage (« une dresseuse rousse aux cheveux "
                       "longs, veste rouge, short bleu, casquette blanche »).")
    if not diffusion.present("qwen"):
        raise gr.Error("Qwen3-VL n'est pas installé : écris la description toi-même, en anglais (« a young woman "
                       "with long red hair, red jacket, blue shorts and white cap »).")
    if photo:
        return diffusion.decrire("sprite", demande or "the person in the photo", image=photo,
                                 progress=progress).strip()
    return diffusion.decrire("sprite_texte", demande, progress=progress).strip()


def lire_couleurs(photo, demande, description=None, progress=gr.Progress()):
    """Couleurs de la personne, modifiables : la description du personnage si elle est écrite (elle fait foi), sinon
    la photo (Qwen3-VL la voit et applique les changements demandés), sinon la demande seule."""
    photo = getattr(photo, "name", photo)
    description, demande = (description or "").strip(), (demande or "").strip()
    if not photo and not description and not demande:
        raise gr.Error("Ajoute la photo de la personne, ou décris le personnage.")
    if not diffusion.present("qwen"):
        raise gr.Error("Qwen3-VL n'est pas installé : écris les couleurs à la main (cheveux, peau, haut, bas, "
                       "chaussures), en anglais ou en français.")
    if description or not photo:
        return diffusion.decrire("couleurs_texte", description or demande, progress=progress).strip()
    return diffusion.decrire("couleurs", demande or "the person in the photo", image=photo, progress=progress).strip()


def _decrire_personnage(photo, demande, description, progress):
    """Description anglaise du personnage : celle de l'utilisateur, sinon Qwen3-VL (photo ou texte), sinon la demande."""
    if description:
        return description
    if diffusion.present("qwen") and (photo or (demande or "").strip()):
        return preparer_description(photo, demande, progress=progress)
    return (demande or "").strip()


def _face_dessinee(style, photo, demande, couleurs, graine, dossier, progress, description=None):
    """Sprite de face d'après la photo : FLUX.2 klein, BiRefNet, puis la grille et la palette du style."""
    st = STYLES[style]
    source = dossier / "face_source.png"
    description = _decrire_personnage(photo, demande, description, progress)
    prompt = f"{description} Colors: {couleurs}. {px.MEME_PERSONNE} {POSE_FACE}, in {st['texte']}."
    diffusion.personnage(prompt, [photo], [source], [graine], 768, 1024, progress=progress)
    image = source
    if diffusion.present("photo_detourage"):
        detoure = dossier / "face_detoure.png"
        try:
            diffusion.detourer_lot([source], [detoure], progress=progress)
            image = detoure if detoure.exists() else source
        except gr.Error:
            image = source
    sprite, _ = px.pixeliser(Image.open(image), st["hauteur"], st["couleurs"], st["toile"], px.CONTOUR_DEFAUT)
    return np.array(sprite), prompt


def _face_lora(style, photo, demande, couleurs, graine, variantes, dossier, progress, description=None):
    """Sprites de face par SDXL + LoRA de sprites de dresseurs : `variantes` images 1024², fond uni retiré (BiRefNet
    si présent, sinon par pixeliser), puis la grille et la palette du style. Description : celle de l'utilisateur,
    sinon Qwen3-VL d'après la photo ou la seule demande (traduite : SDXL comprend mal le français), qui donne déjà
    les couleurs. Renvoie (liste de tableaux, prompt, description)."""
    st = STYLES[style]
    texte = _decrire_personnage(photo, demande, description, progress)
    if not texte or not (description or diffusion.present("qwen")):  # sans Qwen : demande telle quelle
        texte = texte or "a trainer"
        if couleurs:
            texte += f" ({couleurs})"
    description = texte.rstrip(".")
    prompt = PROMPT_LORA.format(description=description, gen=GEN[st["cle"]])
    sources = [dossier / f"face_{i}_source.png" for i in range(1, variantes + 1)]
    diffusion.sprite_dresseur(prompt, sources, [graine + i for i in range(variantes)], progress=progress)
    images = list(sources)
    if diffusion.present("photo_detourage"):
        detoures = [s.with_name(s.name.replace("_source", "_detoure")) for s in sources]
        try:
            diffusion.detourer_lot(sources, detoures, progress=progress)
            images = [d if d.exists() else s for d, s in zip(detoures, sources)]
        except gr.Error:
            pass
    sprites = []
    for im in images:
        sprite, _ = px.pixeliser(Image.open(im), st["hauteur"], st["couleurs"], st["toile"], px.CONTOUR_DEFAUT)
        sprites.append(np.array(sprite))
    return sprites, prompt, description


def _description_marche(photo, demande, couleurs, progress, description=None):
    """Description courte du personnage pour le modèle de feuilles (même forme que ses légendes d'entraînement) :
    d'après la description du personnage si elle existe (les modifications de l'utilisateur font foi), sinon la
    photo, sinon la demande."""
    texte = (demande or "").strip() or "the person in the photo"
    if diffusion.present("qwen"):
        if description:
            return diffusion.decrire("legende_texte", description, progress=progress).strip()
        if photo:
            return diffusion.decrire("legende", texte, image=photo, progress=progress).strip()
        if (demande or "").strip():
            return diffusion.decrire("legende_texte", demande.strip(), progress=progress).strip()
    texte = description or texte
    return f"{texte} ({couleurs})" if couleurs and not description else texte


def _ecrire_feuille(a, nom, dossier, agrandissement):
    im = Image.fromarray(a, "RGBA")
    im.save(dossier / f"{nom}.png")
    px.agrandir(im, agrandissement).save(dossier / f"{nom}_x{agrandissement}.png")
    return [dossier / f"{nom}.png", dossier / f"{nom}_x{agrandissement}.png"]


def _durees_lancer(n):
    return [250] + [100] * (n - 2) + [500] if n > 1 else [500]


def liste_loras_marche():
    """Choix de la feuille de marche : recoloriage d'un exemple, ou un modèle entraîné (onglet « Entraîner un modèle
    de sprites »)."""
    choix = [MARCHE_EXEMPLE] + list(modele_sprites.loras())
    return gr.update(choices=choix, value=choix[-1] if len(choix) > 1 else MARCHE_EXEMPLE)


def creer_kit(style, photo, demande, couleurs, feuilles, mode_face, g_face, g_lancer, g_marche, graine,
              nom="dresseur", variantes=2, lora_marche=None, description=None, progress=gr.Progress()):
    """Kit de dresseur, d'après une photo ou une simple description. lora_marche : nom d'un modèle entraîné pour la
    feuille de marche (sinon recoloriage d'un exemple). description : description anglaise du personnage (préparée
    par `preparer_description` puis modifiée), prioritaire sur la photo et la demande. Renvoie (message, aperçus,
    fichiers, dossier, sprite de face, couleurs utilisées, description utilisée)."""
    if style not in STYLES:
        raise gr.Error("Choisis un style.")
    feuilles = [CLES_FEUILLES[f] for f in (feuilles or []) if f in CLES_FEUILLES]
    if not feuilles:
        raise gr.Error("Coche au moins un élément à créer (sprite de face, lancer, marche).")
    photo = getattr(photo, "name", photo)
    description = (description or "").strip()
    dessiner = "face" in feuilles and mode_face == DESSINER
    lora = "face" in feuilles and mode_face == LORA
    if dessiner and not photo:
        raise gr.Error("Pour dessiner le sprite de face avec FLUX.2 klein, ajoute la photo (sans photo, choisis le "
                       "modèle de sprites de dresseurs : une description suffit).")
    if lora and not photo and not (demande or "").strip() and not description:
        raise gr.Error("Ajoute la photo de la personne, ou décris-la (« une femme aux cheveux bouclés, en veste "
                       "rouge »).")
    choix = {"face": g_face, "lancer": g_lancer, "marche": g_marche}
    lora_marche = lora_marche if lora_marche and lora_marche != MARCHE_EXEMPLE and "marche" in feuilles else None
    if lora_marche and lora_marche not in modele_sprites.loras():
        raise gr.Error(f"Modèle de feuilles « {lora_marche} » introuvable : entraîne-le dans « Entraîner un modèle de "
                       "sprites », ou choisis le recoloriage d'un exemple.")
    gabarits = {}
    for f in feuilles:
        if f == "face" and (dessiner or lora):
            continue
        if f == "marche" and lora_marche:
            continue
        chemin = chemin_gabarit(style, f, choix[f])
        if not chemin:
            raise gr.Error(f"Pas de sprite d'exemple pour « {LIBELLES_FEUILLES[f]} » : ajoute-en dans « Sprites "
                           "d'exemple » (des sprites du jeu dont tu veux le style).")
        gabarits[f] = lire_gabarit(chemin, style, f)  # exemple d'un autre style : mis à la taille de celui-ci
    if lora or lora_marche:
        diffusion.verifier("sprites_dresseur")
    if gabarits or dessiner:
        diffusion.verifier("personnages")
    dossier = nouveau_dossier(cfg.PIXEL_DIR, _slug(nom, "dresseur"))
    graine = int(graine or 0) or diffusion.graines(1)[0]
    debut = time.monotonic()
    st = STYLES[style]
    k_aff = st["agrandissement"]
    copie = None
    if photo:
        copie = dossier / f"photo{Path(photo).suffix.lower() or '.png'}"
        shutil.copy(photo, copie)
    couleurs = (couleurs or "").strip()
    if not couleurs and (gabarits or dessiner):  # le LoRA se contente de la description
        if diffusion.present("qwen") and (copie or description or (demande or "").strip()):
            progress(0.02, desc="Couleurs de la personne (Qwen3-VL)…")
            couleurs = lire_couleurs(copie, demande, description, progress=progress)
        else:
            raise gr.Error("Écris les couleurs de la personne (cheveux, peau, haut, bas, chaussures), ou ajoute la "
                           "photo ou une description pour qu'elles soient lues.")
    infos = {"style": style, "demande": demande, "couleurs": couleurs, "graine": graine, "feuilles": {}}
    if description:
        infos["description"] = description
    resultats = {}
    references = []
    variantes_face = []
    if lora:
        progress(0.1, desc="Sprites de face (SDXL + LoRA de sprites de dresseurs)…")
        variantes_face, prompt, description = _face_lora(style, copie, demande, couleurs, graine,
                                                         max(1, min(VARIANTES_MAX, int(variantes or 1))), dossier,
                                                         progress, description)
        resultats["face"] = variantes_face[0]
        infos["feuilles"]["face"] = {"mode": "lora", "prompt": prompt, "variantes": len(variantes_face)}
        references.append(resultats["face"])
    if dessiner:
        progress(0.1, desc="Sprite de face d'après la photo (FLUX.2 klein)…")
        resultats["face"], prompt = _face_dessinee(style, copie, demande, couleurs, graine, dossier, progress,
                                                   description)
        infos["feuilles"]["face"] = {"mode": "dessin", "prompt": prompt}
        references.append(resultats["face"])
    if gabarits:
        progress(0.4, desc="Recoloriage des sprites d'exemple (FLUX.2 klein)…")
        ordre = list(gabarits)
        pages = [_en_page(f, gabarits[f], style) for f in ordre]
        a, positions = planche(pages)
        k, w, h = agrandissement_pour(a.shape[1], a.shape[0])
        a = np.pad(a, ((0, h - a.shape[0]), (0, w - a.shape[1]), (0, 0)))
        fond = px._fond_eloigne(np.unique(a[..., :3][a[..., 3] >= 128].reshape(-1, 3), axis=0))
        entree = dossier / "exemples_assembles.png"
        _sur_fond(a, fond, k).save(entree)
        refs = [entree]
        for i, r in enumerate(references):  # sprite de face dessiné : ses couleurs font foi
            refs.append(dossier / f"reference_{i}.png")
            _sur_fond(r, fond, max(1, 512 // r.shape[0])).save(refs[-1])
        prompt = CONSIGNE_RECOLORER.format(fond=px.NOMS_FONDS.get(fond, "bright green"), couleurs=couleurs) + (
            AVEC_REFERENCE if references else "")
        sortie = dossier / "recoloriage.png"
        diffusion.personnage(prompt, refs, [sortie], [graine], w * k, h * k, progress=progress)
        image = Image.open(sortie)
        for f, page, (y, x) in zip(ordre, pages, positions):
            sous = image.crop((x * k, y * k, (x + page.shape[1]) * k, (y + page.shape[0]) * k))
            nouvelles = correspondances(page, sous, k)
            resultats[f] = recolorer(gabarits[f], nouvelles)
            infos["feuilles"][f] = {"mode": "recoloriage", "exemple": choix[f], "prompt": prompt,
                                    "couleurs": {"#%02x%02x%02x" % a_: "#%02x%02x%02x" % b_
                                                 for a_, b_ in nouvelles.items()}}
    if lora_marche:
        progress(0.7, desc=f"Feuille de marche (SDXL + modèle « {lora_marche} »)…")
        legende = _description_marche(copie, demande, couleurs, progress, description)
        palette = None
        if "face" in resultats:
            fa = resultats["face"]
            palette = np.unique(fa[..., :3][fa[..., 3] >= 128].reshape(-1, 3), axis=0)
        resultats["marche"], prompt = modele_sprites.generer_feuille(
            lora_marche, legende, dossier / "marche.png", graine, st["marche"]["grille"],
            st["marche"]["cellule"], palette, st["couleurs"], progress=progress)
        infos["feuilles"]["marche"] = {"mode": "lora", "modele": lora_marche, "prompt": prompt}
    fichiers, apercus = [], []
    for f in ("face", "lancer", "marche"):
        if f not in resultats:
            continue
        a = resultats[f]
        fichiers += _ecrire_feuille(a, f, dossier, k_aff)
        apercus.append(str(dossier / f"{f}_x{k_aff}.png"))
        for i, v in enumerate(variantes_face[1:], 2):  # autres variantes du sprite de face (cliquer pour animer)
            fichiers += _ecrire_feuille(v, f"face_{i}", dossier, k_aff)
            apercus.append(str(dossier / f"face_{i}_x{k_aff}.png"))
        if f == "lancer":
            c = a.shape[0]
            cases = [Image.fromarray(np.ascontiguousarray(x), "RGBA") for x in _cases(a, (a.shape[1] // c, 1), (c, c))]
            fichiers += px.ecrire_animation(cases, dossier, "lancer", k_aff, _durees_lancer(len(cases)))
            apercus.append(str(dossier / f"lancer_x{k_aff}.gif"))
        elif f == "marche":
            grille, cellule = st["marche"]["grille"], st["marche"]["cellule"]
            cases = [Image.fromarray(np.ascontiguousarray(x), "RGBA") for x in _cases(a, grille, cellule)]
            for r, direction in enumerate(st["marche"]["lignes"]):
                cycle = [cases[r * grille[0] + i] for i in st["marche"]["cycle"]]
                gif = dossier / f"marche_{direction}_x{k_aff}.gif"
                px._gif([px.agrandir(c, k_aff) for c in cycle], gif, 150)
                fichiers.append(gif)
                apercus.append(str(gif))
    (dossier / "kit.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    archive = dossier / f"{dossier.name}.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for f in sorted(dossier.rglob("*")):
            if f.is_file() and f != archive and f.suffix in (".png", ".gif", ".json") and not f.name.startswith(
                    ("photo", "face_source", "face_detoure", "exemples_assembles", "recoloriage", "reference_")
                    ) and not f.name.endswith(("_source.png", "_detoure.png")):
                z.write(f, f.relative_to(dossier))
    fichiers.append(archive)
    visage = dossier / "face.png"
    msg = (f"✅ Kit de dresseur en {time.monotonic() - debut:.0f} s (graine {graine}). Couleurs : {couleurs}. "
           f"Dossier : {dossier}."
           + (" Le sprite de face est placé dans « Animer un sprite » pour sa feuille d'animation." if visage.exists()
              else ""))
    return (msg, apercus, [str(f) for f in fichiers], str(dossier), str(visage) if visage.exists() else None,
            couleurs, description)
