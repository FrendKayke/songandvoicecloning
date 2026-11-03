"""Refaire les yeux d'un sprite (groupe Pixel art, onglet « 👁️ Refaire les yeux ») : tu peins au pinceau sur les
yeux ratés, SDXL redessine seulement cette zone avec le modèle de sprites, et seuls les pixels peints changent.

1. Le sprite est agrandi ×`AFFICHAGE` dans l'éditeur ; le masque peint est relu à l'échelle du sprite (un pixel du
   sprite est retenu s'il est peint sur au moins `COUVERTURE_MIN` de sa surface).
2. Une fenêtre de C×C pixels du sprite, centrée sur la zone peinte, est posée sur le fond du modèle (magenta pour un
   modèle entraîné ici, blanc pour le LoRA de sWizad) et agrandie au plus proche à l'échelle où le modèle dessine
   (un pixel du jeu = `facteur` pixels de l'image : celui du modèle entraîné, 8 pour sWizad), C = 1024 / facteur.
3. SDXL (inpainting, même LoRA) redessine la zone peinte, élargie d'un pixel pour raccorder.
4. Retour au sprite : pour chaque pixel peint, couleur dominante du centre de sa case dans l'image, ramenée à la
   palette du sprite ; une case devenue « fond » garde le pixel d'origine. Le reste du sprite est inchangé.
Rangement : à côté du sprite s'il vient de data/pixel_art, sinon data/pixel_art/<horodatage>_yeux/.
"""
import json
import re
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from . import config as cfg
from . import diffusion, dresseur, modele_sprites
from . import pixel_art as px
from .outils import chemin_fichier, nouveau_dossier

AFFICHAGE = 8  # agrandissement du sprite dans l'éditeur (pour peindre précisément)
COUVERTURE_MIN = 0.2
FACTEUR_SWIZAD, FOND_SWIZAD = 8, (255, 255, 255)
COTE_IMAGE = 1024  # côté de l'image donnée à SDXL (sa résolution native)
FORCE_DEFAUT = 0.85
VARIANTES_DEFAUT = 4
ECART_FOND = 120
COULEURS_MAX = 32
YEUX = "detailed expressive eyes, clean pixel eyes, symmetrical face"


def ouvrir_sprite(chemin):
    """Sprite RGBA (H×W×4 uint8), petit : un pixel = un pixel du jeu (réduit s'il a été agrandi)."""
    with Image.open(chemin) as im:
        a = np.array(im.convert("RGBA"))
    # aperçu agrandi (face_x2.png…) : ramené à ses vrais pixels. Pas un sprite à sa taille (≤ 160 px, toile DS
    # doublée) : un sprite aux détails tous de 2 pixels passerait pour un agrandi ×2.
    if re.search(r"_x\d+$", Path(chemin).stem) or max(a.shape[:2]) > 160:
        a = px.reduire(a)[0]
    return np.ascontiguousarray(a)


def pour_editeur(chemin):
    """Image affichée dans l'éditeur : le sprite agrandi au plus proche, sur un gris clair (transparence visible)."""
    if not chemin_fichier(chemin):
        return None
    a = ouvrir_sprite(chemin_fichier(chemin))
    fond = Image.new("RGBA", (a.shape[1], a.shape[0]), (220, 220, 220, 255))
    im = Image.alpha_composite(fond, Image.fromarray(a, "RGBA")).convert("RGB")
    return np.array(im.resize((im.width * AFFICHAGE, im.height * AFFICHAGE), Image.Resampling.NEAREST))


def masque(editeur, hauteur, largeur):
    """Pixels du sprite peints dans l'éditeur (booléens H×W). Le pinceau est lu dans les calques ; à défaut, dans
    l'écart entre l'image composée et le fond."""
    if not editeur:
        return np.zeros((hauteur, largeur), bool)
    peint = None
    for calque in editeur.get("layers") or []:
        if calque is None:
            continue
        c = np.asarray(calque)
        alpha = c[..., 3] > 0 if c.ndim == 3 and c.shape[2] == 4 else c.reshape(c.shape[:2] + (-1,)).max(-1) > 0
        peint = alpha if peint is None else (peint | alpha)
    if peint is None and editeur.get("composite") is not None and editeur.get("background") is not None:
        comp = np.asarray(editeur["composite"])[..., :3].astype(int)
        fond = np.asarray(editeur["background"])[..., :3].astype(int)
        if comp.shape == fond.shape:
            peint = np.abs(comp - fond).sum(-1) > 30
    if peint is None or not peint.any():
        return np.zeros((hauteur, largeur), bool)
    couverture = np.asarray(Image.fromarray((peint * 255).astype(np.uint8)).resize((largeur, hauteur),
                                                                                  Image.Resampling.BOX)) / 255
    return couverture >= COUVERTURE_MIN


def _modele(nom):
    """(fichier du LoRA ou None, facteur, fond, prompt sans description) du modèle choisi."""
    if nom and nom != dresseur.FACE_SWIZAD:
        fichier = modele_sprites.loras(modele_sprites.FACE).get(nom)
        if not fichier:
            raise gr.Error(f"Modèle de sprites de face « {nom} » introuvable.")
        infos = modele_sprites.infos_modele(nom)
        return str(fichier), int(infos["facteur"]), modele_sprites.FOND, None
    return None, FACTEUR_SWIZAD, FOND_SWIZAD, None


def _prompt(nom, description, gen):
    description = (description or "").strip().rstrip(".") or "a trainer"
    if nom and nom != dresseur.FACE_SWIZAD:
        return f"{modele_sprites._legende_face(description)}, {YEUX}"
    return f"{dresseur.PROMPT_LORA.format(description=description, gen=gen)}, {YEUX}"


def fenetre(zone, cote, hauteur, largeur):
    """Coin (y0, x0) d'une fenêtre cote×cote centrée sur la zone peinte ; peut dépasser le sprite (bord complété)."""
    ys, xs = np.nonzero(zone)
    cy, cx = (ys.min() + ys.max() + 1) // 2, (xs.min() + xs.max() + 1) // 2
    y0 = min(max(cy - cote // 2, min(0, hauteur - cote)), max(0, hauteur - cote))
    x0 = min(max(cx - cote // 2, min(0, largeur - cote)), max(0, largeur - cote))
    return int(y0), int(x0)


def _dilater(m):
    d = m.copy()
    d[1:] |= m[:-1]
    d[:-1] |= m[1:]
    d[:, 1:] |= m[:, :-1]
    d[:, :-1] |= m[:, 1:]
    return d


def preparer(sprite, zone, facteur, fond):
    """(image RGB pour SDXL, masque L, coin de la fenêtre, côté C) : fenêtre du sprite sur le fond du modèle,
    agrandie ×facteur ; masque = zone peinte élargie d'un pixel."""
    h, w = sprite.shape[:2]
    cote = max(8, COTE_IMAGE // facteur)
    y0, x0 = fenetre(zone, cote, h, w)
    toile = np.zeros((cote, cote, 4), np.uint8)
    m = np.zeros((cote, cote), bool)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(h, y0 + cote), min(w, x0 + cote)
    toile[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = sprite[sy0:sy1, sx0:sx1]
    m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = zone[sy0:sy1, sx0:sx1]
    base = Image.new("RGBA", (cote, cote), tuple(fond) + (255,))
    image = Image.alpha_composite(base, Image.fromarray(toile, "RGBA")).convert("RGB")
    image = image.resize((cote * facteur, cote * facteur), Image.Resampling.NEAREST)
    msk = Image.fromarray((_dilater(m) * 255).astype(np.uint8)).resize((cote * facteur, cote * facteur),
                                                                         Image.Resampling.NEAREST)
    return image, msk, (y0, x0), cote


def palette_du_sprite(sprite):
    opaques = sprite[sprite[..., 3] > 127][:, :3]
    couleurs = np.unique(opaques, axis=0)
    if len(couleurs) > COULEURS_MAX:
        couleurs = px._palette(opaques, COULEURS_MAX)
    return couleurs.astype(np.uint8)


def appliquer(sprite, zone, image, coin, facteur, fond, palette):
    """Le sprite avec ses pixels peints remplacés par ce que SDXL a dessiné (centre de chaque case, couleur
    dominante ramenée à la palette ; une case de fond garde le pixel d'origine). Renvoie (sprite, pixels changés)."""
    a = np.asarray(image.convert("RGB")).astype(int)
    resultat = sprite.copy()
    y0, x0 = coin
    q0, q1 = facteur // 4, facteur - facteur // 4
    changes = 0
    for y, x in zip(*np.nonzero(zone)):
        wy, wx = y - y0, x - x0
        bloc = a[wy * facteur + q0:wy * facteur + q1, wx * facteur + q0:wx * facteur + q1].reshape(-1, 3)
        if not len(bloc):
            continue
        d = ((bloc[:, None, :] - palette[None, :, :].astype(int)) ** 2).sum(-1)
        dominante = np.bincount(d.argmin(1), minlength=len(palette)).argmax()
        mediane = np.median(bloc, axis=0)
        if np.sqrt(((mediane - np.array(fond)) ** 2).sum()) < ECART_FOND:
            continue  # SDXL a mis du fond ici : le pixel d'origine reste
        nouveau = np.append(palette[dominante], 255).astype(np.uint8)
        if not np.array_equal(nouveau, resultat[y, x]):
            changes += 1
        resultat[y, x] = nouveau
    return resultat, changes


def _style_du_kit(dossier):
    kit = Path(dossier) / "kit.json"
    if kit.is_file():
        try:
            infos = json.loads(kit.read_text(encoding="utf-8"))
            return infos.get("description"), dresseur.GEN.get(dresseur.STYLES.get(infos.get("style"), {}).get("cle"))
        except ValueError:
            pass
    return None, None


def retoucher(sprite_fichier, editeur, description="", modele=None, variantes=VARIANTES_DEFAUT, force=FORCE_DEFAUT,
              graine=0, progress=gr.Progress()):
    """Renvoie (message, aperçus avant / après, fichiers, dossier)."""
    chemin = chemin_fichier(sprite_fichier)
    if not chemin or not Path(str(chemin)).is_file():
        raise gr.Error("Ajoute le sprite à corriger (ou crée un kit de dresseur : son sprite de face arrive ici).")
    chemin = Path(str(chemin))
    sprite = ouvrir_sprite(chemin)
    h, w = sprite.shape[:2]
    zone = masque(editeur, h, w) & (sprite[..., 3] > 0)
    if not zone.any():
        raise gr.Error("Peins au pinceau sur les yeux à refaire (dans la grande image), puis relance.")
    t0 = time.monotonic()
    description_kit, gen = _style_du_kit(chemin.parent)
    description = (description or "").strip() or description_kit or ""
    lora, facteur, fond, _ = _modele(modele)
    prompt = _prompt(modele, description, gen or "gen3")
    try:
        chemin.resolve().relative_to(cfg.PIXEL_DIR.resolve())
        dossier = chemin.parent
    except ValueError:
        dossier = nouveau_dossier(cfg.PIXEL_DIR, "yeux")
    travail = dossier / "yeux_travail"
    travail.mkdir(parents=True, exist_ok=True)
    image, msk, coin, cote = preparer(sprite, zone, facteur, fond)
    image.save(travail / "entree.png")
    msk.save(travail / "masque.png")
    n = max(1, min(8, int(variantes or VARIANTES_DEFAUT)))
    graine = int(graine or 0) or int(time.time()) % 100000
    sorties = [travail / f"sortie_{graine + i}.png" for i in range(n)]
    diffusion.sprite_dresseur(prompt, sorties, [graine + i for i in range(n)], cote * facteur, cote * facteur,
                              lora=lora, image=travail / "entree.png", masque=travail / "masque.png", force=force,
                              progress=progress)
    palette = palette_du_sprite(sprite)
    apercus, fichiers, lignes = [], [], []
    k = 4
    avant = px.agrandir(Image.fromarray(sprite, "RGBA"), k)
    for i, s in enumerate(sorties, 1):
        nouveau, changes = appliquer(sprite, zone, Image.open(s), coin, facteur, fond, palette)
        f = dossier / f"{chemin.stem}_yeux_{i}.png"
        Image.fromarray(nouveau, "RGBA").save(f)
        apres = px.agrandir(Image.fromarray(nouveau, "RGBA"), k)
        planche = Image.new("RGBA", (avant.width * 2 + 8, avant.height), (0, 0, 0, 0))
        planche.paste(avant, (0, 0))
        planche.paste(apres, (avant.width + 8, 0))
        p = travail / f"avant_apres_{i}.png"
        planche.save(p)
        apercus.append((str(p), f"Variante {i} : {changes} pixel(s) changé(s)"))
        fichiers.append(str(f))
        lignes.append(changes)
    (travail / "retouche.json").write_text(json.dumps({
        "sprite": chemin.name, "pixels_peints": int(zone.sum()), "modele": modele, "prompt": prompt,
        "force": float(force), "graines": [graine + i for i in range(n)], "facteur": facteur, "fenetre": cote,
        "coin": list(coin)}, ensure_ascii=False, indent=1), encoding="utf-8")
    msg = (f"✅ {n} variante(s) des yeux en {time.monotonic() - t0:.0f} s ({int(zone.sum())} pixels peints ; seuls "
           f"eux peuvent changer). À gauche l'original, à droite la variante. Fichiers `{chemin.stem}_yeux_<n>.png` "
           f"dans `{dossier}`.")
    if not any(lignes):
        msg += "\n\n⚠️ Aucun pixel n'a changé : augmente la force ou peins un peu plus large."
    return msg, apercus, fichiers, str(dossier)
