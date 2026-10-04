"""Onglet « Images » : une image à partir d'un prompt (Z-Image-Turbo) ou d'une photo (FLUX.2 klein 4B).

Avec une photo, trois usages : la modifier (mêmes personnes, même cadrage, avec les changements demandés), garder
son sujet dans une nouvelle scène, ou reprendre sa composition. Les deux modèles sont sous licence Apache 2.0.
Chaque création : data/images/<horodatage>/ avec photo.<ext> (si donnée), variante_<i>.png, creation.json type
« image » (galerie : affichage, « Recréer » avec la graine).
"""
import shutil
import time
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from . import cartes
from . import config as cfg
from . import diffusion
from .outils import ecrire_creation, nouveau_dossier
from .styles import texte
from .videos import MOTS_PROMPT_COURT, duree_lisible

AUTO = "Automatique (proportions de la photo, sinon carré)"
# Côtés multiples de 16, ~1 mégapixel (Z-Image va jusqu'à ~2 Mpx, klein est plus rapide vers 1 Mpx)
FORMATS = {
    AUTO: None,
    "Carré 1:1 (1024×1024)": (1024, 1024),
    "Portrait 3:4 (896×1184)": (896, 1184),
    "Portrait 2:3 (832×1248)": (832, 1248),
    "Paysage 4:3 (1184×896)": (1184, 896),
    "Paysage 16:9 (1344×768)": (1344, 768),
    "Vertical 9:16 (768×1344)": (768, 1344),
}
PIXELS_AUTO = 1024 * 1024
STYLES = [("Photo réaliste", "photorealistic photograph, natural light, sharp focus, high detail")] + [
    s for s in cartes.STYLES if "trading card" not in s[1]]

# Ce que l'image reprend de la photo (libellé → clé)
USAGES = {
    "Modifier la photo (mêmes personnes, même cadrage, avec les changements décrits)": "modifier",
    "Garder le sujet (même personne ou objet, dans une nouvelle scène)": "sujet",
    "Reprendre la composition (pose, cadrage)": "composition",
}
USAGE_DEFAUT = next(iter(USAGES))
GARDER_LE_RESTE = "Keep everything else exactly as in the reference image: same people, faces, pose and framing"
SUJET = ("The main subject is the person or object of the reference image: keep their face, features, hair, "
         "body shape and recognizable details")
COMPOSITION = "Follow the composition, pose and framing of the reference image"

EXEMPLES = [
    ("Paysage de montagne", "un lac de montagne au lever du soleil, brume sur l'eau, sapins enneigés"),
    ("Portrait", "portrait d'une vieille marin au visage buriné, ciré jaune, lumière douce de fin de journée"),
    ("Objet produit", "une montre ancienne en or posée sur du velours bleu, éclairage de studio"),
    ("Retouche : la nuit", "transforme la scène en nuit, avec la lune et des lumières chaudes aux fenêtres"),
    ("Retouche : chapeau", "ajoute un chapeau de pirate à la personne"),
    ("Retouche : peinture", "redessine la photo comme une peinture à l'huile"),
]


def usage_de(libelle):
    return USAGES.get(libelle, libelle if libelle in USAGES.values() else "modifier")


def format_pour(format_label, photo=None):
    """(largeur, hauteur) : le format choisi, ou en automatique les proportions de la photo à ~1 Mpx."""
    if FORMATS.get(format_label):
        return FORMATS[format_label]
    if photo:
        try:
            with Image.open(photo) as im:
                w, h = ImageOps.exif_transpose(im).size
        except OSError:
            return FORMATS["Carré 1:1 (1024×1024)"]
        echelle = (PIXELS_AUTO / (w * h)) ** 0.5
        return (max(512, min(1536, round(w * echelle / 16) * 16)), max(512, min(1536, round(h * echelle / 16) * 16)))
    return FORMATS["Carré 1:1 (1024×1024)"]


def prompt_final(prompt, styles=None, usage=None):
    """usage : None (sans photo), « modifier », « sujet » ou « composition »."""
    prompt = (prompt or "").strip().rstrip(".")
    ajout = {"modifier": GARDER_LE_RESTE, "sujet": SUJET, "composition": COMPOSITION}.get(usage)
    if ajout:
        prompt = f"{prompt}. {ajout}"
    style = texte(styles)
    return f"{prompt}. Style: {style}." if style else f"{prompt}."


def preparer(description, photo=None, usage_label=USAGE_DEFAUT, progress=gr.Progress()):
    """Description française → prompt anglais (Qwen3-VL), qui voit la photo s'il y en a une. « Modifier la photo » :
    une consigne de retouche courte ; sinon une description détaillée de l'image à créer."""
    description = (description or "").strip()
    if not description:
        raise gr.Error("Décris l'image à créer, ou la modification à faire sur la photo (en français ou en anglais).")
    mode = "retouche" if photo and usage_de(usage_label) == "modifier" else "scene"
    return diffusion.decrire(mode, description, image=photo or None, progress=progress)


def prompt_pret(description, prompt, photo=None, usage_label=USAGE_DEFAUT, progress=gr.Progress()):
    """Avant « Générer » : prompt vide → préparé depuis la description ; prompt court tapé à la main sans description
    → enrichi par Qwen (comme pour les vidéos : un texte trop court donne une image quelconque) ; sinon gardé."""
    prompt, description = (prompt or "").strip(), (description or "").strip()
    retouche = bool(photo) and usage_de(usage_label) == "modifier"
    if prompt and not description and not retouche and len(prompt.split()) < MOTS_PROMPT_COURT:
        return preparer(prompt, photo, usage_label, progress=progress)
    return prompt or preparer(description, photo, usage_label, progress=progress)


def generer(prompt, photo, usage_label, styles, format_label, variantes, graine, nom="", description_fr=None,
            progress=gr.Progress()):
    """Génère les variantes. Renvoie (message, galerie [(image, légende)], dossier)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Il manque le prompt : clique d'abord sur « Préparer le prompt », ou écris-le en anglais.")
    if photo and not Path(photo).is_file():
        raise gr.Error(f"Photo introuvable : {photo}")
    debut = time.monotonic()
    usage = usage_de(usage_label) if photo else None
    format_label = format_label if format_label in FORMATS else AUTO
    largeur, hauteur = format_pour(format_label, photo)
    n = max(1, min(4, int(variantes or 1)))
    dossier = nouveau_dossier(cfg.IMAGES_DIR)
    copie = None
    if photo:  # copiée dans la création : la recréer ne dépend pas du fichier d'origine
        copie = dossier / f"photo{Path(photo).suffix.lower() or '.png'}"
        shutil.copy(photo, copie)
    final = prompt_final(prompt, styles, usage)
    sorties = [dossier / f"variante_{i}.png" for i in range(1, n + 1)]
    if copie:
        res = diffusion.personnage(final, [copie], sorties, diffusion.graines(n, graine), largeur, hauteur,
                                   progress=progress)
    else:
        res = diffusion.image(final, sorties, diffusion.graines(n, graine), largeur, hauteur, progress=progress)
    temps = time.monotonic() - debut
    nom = "".join(c for c in (nom or "").strip() if c.isalnum() or c in "-_ ").strip() or "image"
    ecrire_creation(dossier, {
        "type": "image", "nom": nom, "description": prompt, "description_fr": (description_fr or "").strip() or None,
        "styles": styles, "format": format_label, "largeur": largeur, "hauteur": hauteur, "prompt": final,
        "photo": copie.name if copie else None, "usage": usage, "temps_s": round(temps), "choisie": 1,
        "moteur": "FLUX.2 klein 4B" if copie else "Z-Image-Turbo",
        "versions": [{"graine": g, "dossier": ".", "fichier": f} for f, g in zip(res["fichiers"], res["graines"])],
    })
    galerie = [(f, f"Variante {i} (graine {g})") for i, (f, g) in enumerate(zip(res["fichiers"], res["graines"]), 1)]
    origine = {"modifier": "photo modifiée", "sujet": "sujet de la photo", "composition": "composition de la photo",
               None: "à partir du texte"}[usage]
    msg = (f"✅ {len(galerie)} image(s) {largeur}×{hauteur} ({origine}) en {duree_lisible(temps)}, dans {dossier}. "
           f"Graines : {', '.join(str(g) for g in res['graines'])}. Clique sur une variante pour la retoucher ou "
           "l'animer.")
    return msg, galerie, str(dossier)


def choisir(dossier, evt: gr.SelectData):
    """Clic sur une variante : renvoie (message, chemin de l'image) ; notée « choisie » dans creation.json."""
    import json

    if not dossier:
        raise gr.Error("Génère d'abord une image.")
    chemin = Path(dossier) / "creation.json"
    infos = json.loads(chemin.read_text(encoding="utf-8"))
    versions = infos.get("versions") or []
    if not 0 <= evt.index < len(versions):
        raise gr.Error("Variante introuvable.")
    infos["choisie"] = evt.index + 1
    chemin.write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    return (f"⭐ Variante {evt.index + 1} choisie : « Retoucher cette image » la reprend comme photo, « Animer » "
            "l'envoie dans l'onglet Vidéos.", versions[evt.index]["fichier"])


def recreer(chemin, infos, graine, progress=gr.Progress()):
    """Galerie : même image, mêmes réglages, même graine."""
    photo = Path(chemin) / infos["photo"] if infos.get("photo") else None
    if photo is not None and not photo.exists():
        raise gr.Error(f"Photo introuvable : {photo}")
    _, _, dossier = generer(infos.get("description"), str(photo) if photo else None, infos.get("usage") or "modifier",
                            infos.get("styles"), infos.get("format") or AUTO, 1, graine, infos.get("nom"),
                            infos.get("description_fr"), progress=progress)
    return dossier
