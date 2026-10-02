"""Onglet « Illustrations de cartes » : images pour un jeu de cartes, avec Z-Image-Turbo (Apache 2.0 : les images
peuvent être vendues).

Cohérence d'un projet : le style (termes anglais choisis dans STYLES ou tapés librement, consignes, format) est gardé
dans data/cartes/<projet>/style.json et réappliqué à chaque carte du projet ; seule la scène change. La description
française de la scène est reformulée en anglais par Qwen3-VL (mode « carte »), modifiable avant génération.
Rangement : data/cartes/<projet>/<horodatage>/ : variante_<i>.png (+ .webp), creation.json (type « carte »).
Jamais de nom d'artiste ni d'œuvre dans les styles (comme pour la musique).
Personnage récurrent (personnages.py) : la scène est alors peinte par FLUX.2 klein 4B à partir des images de
référence du personnage, pour qu'il garde le même visage et la même tenue d'une carte à l'autre.
Photo modèle (facultative) : envoyée aussi à FLUX.2 klein, soit pour son sujet (la personne ou l'objet de la photo
devient celui de la carte), soit pour sa composition (pose, cadrage) ; copiée dans la création (photo_modele.<ext>).
"""
import shutil
import json
from pathlib import Path

import gradio as gr
from PIL import Image

from . import config as cfg
from . import diffusion, personnages
from .outils import ecrire_creation, nouveau_dossier
from .styles import texte

# (libellé français, termes anglais) : listes multiselect + saisie libre, comme l'onglet « Créer une chanson »
STYLES = [
    ("Peinture fantasy détaillée", "detailed digital fantasy painting, rich colors, dramatic lighting"),
    ("Illustration de carte à collectionner", "trading card game illustration, highly detailed, epic composition"),
    ("Peinture à l'huile épique", "epic oil painting, visible brush strokes, classical fantasy art"),
    ("Aquarelle", "watercolor illustration, soft edges, paper texture"),
    ("Anime / cel shading", "anime style, cel shading, clean line art, vibrant colors"),
    ("JRPG élégant (personnages fins)", "elegant Japanese RPG character art, delicate line work, ornate costumes"),
    ("Pixel art 16-bit", "16-bit pixel art, limited palette, crisp pixels"),
    ("Encre et lavis", "ink and wash illustration, monochrome with subtle color accents"),
    ("Art nouveau", "art nouveau illustration, ornamental curves, decorative frame elements"),
    ("Gravure ancienne", "old engraving style, cross-hatching, sepia tones"),
    ("Rendu 3D stylisé", "stylized 3D render, soft global illumination, game art"),
    ("Sombre / gothique", "dark gothic fantasy, moody shadows, desaturated palette"),
    ("Lumineux / féerique", "bright whimsical fairy-tale atmosphere, glowing light, pastel colors"),
    ("Cinématique", "cinematic composition, volumetric light, depth of field"),
]
STYLES_DEFAUT = ["detailed digital fantasy painting, rich colors, dramatic lighting",
                 "trading card game illustration, highly detailed, epic composition"]

# Z-Image-Turbo génère jusqu'à ~2 mégapixels ; côtés multiples de 16
FORMATS = {
    "Illustration de carte (paysage 4:3)": (1152, 864),
    "Carte entière (portrait 2:3)": (832, 1248),
    "Portrait 3:4": (896, 1184),
    "Carré 1:1": (1024, 1024),
    "Décor / fond d'écran (16:9)": (1344, 768),
}
FORMAT_DEFAUT = "Illustration de carte (paysage 4:3)"

EXEMPLES = [
    ("Chevalier de la lumière", "un chevalier en armure dorée qui brandit une épée lumineuse au sommet d'une tour"),
    ("Mage de glace", "une magicienne aux cheveux argentés qui invoque une tempête de cristaux de glace"),
    ("Dragon ancien", "un immense dragon rouge endormi sur un trésor dans une caverne éclairée par la lave"),
    ("Potion de soin", "une fiole de potion verte lumineuse posée sur une table d'alchimiste"),
    ("Cité céleste", "une cité flottante au-dessus des nuages au lever du soleil, dirigeables en approche"),
    ("Esprit de la forêt", "un cerf géant fait de feuillage et de lumière dans une forêt brumeuse"),
]


def nom_projet(projet):
    return "".join(c for c in (projet or "").strip() if c.isalnum() or c in "-_ ").strip() or "mon-jeu"


def projets():
    return sorted(d.name for d in cfg.CARDS_DIR.iterdir() if d.is_dir()) if cfg.CARDS_DIR.exists() else []


def _fichier_style(projet):
    return cfg.CARDS_DIR / nom_projet(projet) / "style.json"


def style_projet(projet):
    """Style enregistré du projet : (styles, consignes, format) pour remplir les champs ; défauts sinon."""
    try:
        s = json.loads(_fichier_style(projet).read_text(encoding="utf-8"))
        return s.get("styles") or STYLES_DEFAUT, s.get("consignes", ""), s.get("format", FORMAT_DEFAUT)
    except (OSError, ValueError):
        return STYLES_DEFAUT, "", FORMAT_DEFAUT


def charger_style(projet):
    styles, consignes, fmt = style_projet(projet)
    return gr.update(value=styles), consignes, fmt


# Ajouté à la scène quand un personnage est choisi : FLUX.2 klein reçoit ses images de référence
MEME_PERSONNAGE = ("The main character is exactly the same character as in the reference image(s): same face, "
                   "hair, body shape, outfit and colors, shown in a new pose for this scene")


# Photo modèle : ce qu'on en garde (libellé → consigne ajoutée à la scène)
PHOTO_SUJET = ("The main subject is the person or object shown in the reference photo: keep their face, features, "
               "hair, body shape and recognizable details, redrawn as an illustration (not a photograph) in the art "
               "style below, in a new pose for this scene")
PHOTO_COMPOSITION = ("Follow the composition, pose and framing of the reference photo, redrawn as an illustration "
                     "(not a photograph) in the art style below")
USAGES_PHOTO = {
    "Le sujet de la photo devient celui de la carte (même visage, mêmes traits)": "sujet",
    "Garder la composition de la photo (pose, cadrage)": "composition",
}
USAGE_PHOTO_DEFAUT = next(iter(USAGES_PHOTO))


def prompt_final(scene, styles, consignes, personnage=False, photo=None):
    """photo : None, « sujet » ou « composition ». Avec un personnage, la photo ne sert qu'à la composition."""
    scene = (scene or "").strip().rstrip(".")
    if personnage:
        scene = f"{scene}. {MEME_PERSONNAGE}"
    if photo:
        scene = f"{scene}. {PHOTO_COMPOSITION if personnage or photo == 'composition' else PHOTO_SUJET}"
    style = ", ".join(x for x in (texte(styles), (consignes or "").strip()) if x)
    return f"{scene}. Art style: {style}. No text, no letters, no frame." if style else f"{scene}. No text, no letters."


def preparer(description, progress=gr.Progress()):
    """Description française de la scène → prompt anglais (Qwen3-VL), modifiable."""
    description = (description or "").strip()
    if not description:
        raise gr.Error("Décris la scène de la carte (en français ou en anglais).")
    return diffusion.decrire("carte", description, progress=progress)


def prompt_pret(description, prompt, progress=gr.Progress()):
    """« Générer » sans avoir préparé le prompt (constaté : erreur « Il manque le prompt ») : le prompt anglais est
    préparé d'abord depuis la description, puis affiché ; un prompt déjà là (préparé ou retouché) est gardé."""
    return (prompt or "").strip() or preparer(description, progress=progress)


def generer(projet, nom, scene, styles, consignes, format_label, variantes, graine, webp=True, description_fr=None,
            personnage=None, photo=None, usage_photo=USAGE_PHOTO_DEFAUT, progress=gr.Progress()):
    """Génère les variantes (avec FLUX.2 klein si un personnage ou une photo modèle est donné, Z-Image-Turbo sinon).
    Renvoie (message, galerie [(image, légende)], dossier, liste des projets)."""
    scene = (scene or "").strip()
    if not scene:
        raise gr.Error("Il manque le prompt de la scène : clique d'abord sur « Préparer le prompt », ou écris-le en anglais.")
    projet = nom_projet(projet)
    personnage = None if personnage in (None, "", personnages.AUCUN) else personnage
    references = personnages.references(projet, personnage) if personnage else []
    if personnage and not references:
        raise gr.Error(f"Le personnage « {personnage} » n'a pas d'image de référence dans le projet {projet}.")
    if photo and not Path(photo).is_file():
        raise gr.Error(f"Photo modèle introuvable : {photo}")
    usage = USAGES_PHOTO.get(usage_photo, usage_photo if usage_photo in USAGES_PHOTO.values() else "sujet")
    format_label = format_label if format_label in FORMATS else FORMAT_DEFAUT
    largeur, hauteur = FORMATS[format_label]
    n = max(1, min(4, int(variantes or 1)))
    dossier = nouveau_dossier(cfg.CARDS_DIR / projet)
    photo_modele = None
    if photo:  # copiée dans la création : la recréer plus tard ne dépend pas du fichier d'origine
        photo_modele = dossier / f"photo_modele{Path(photo).suffix.lower() or '.png'}"
        shutil.copy(photo, photo_modele)
        references = list(references[:personnages.REFERENCES_MAX - 1]) + [photo_modele]
    klein = bool(references)
    # Le style du projet est mémorisé : la carte suivante du même projet le retrouve
    _fichier_style(projet).write_text(json.dumps({"styles": list(styles or []) if not isinstance(styles, str) else [styles],
                                                  "consignes": (consignes or "").strip(), "format": format_label},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    prompt = prompt_final(scene, styles, consignes, personnage=bool(personnage), photo=usage if photo else None)
    sorties = [dossier / f"variante_{i}.png" for i in range(1, n + 1)]
    if klein:
        res = diffusion.personnage(prompt, references, sorties, diffusion.graines(n, graine), largeur, hauteur,
                                   progress=progress)
    else:
        res = diffusion.image(prompt, sorties, diffusion.graines(n, graine), largeur, hauteur, progress=progress)
    if webp:  # léger pour un jeu web, sans perte visible
        for f in res["fichiers"]:
            with Image.open(f) as im:
                im.save(Path(f).with_suffix(".webp"), "WEBP", quality=90, method=6)
    nom = "".join(c for c in (nom or "").strip() if c.isalnum() or c in "-_ ").strip() or "carte"
    ecrire_creation(dossier, {
        "type": "carte", "projet": projet, "nom": nom, "description": scene,
        "description_fr": (description_fr or "").strip() or None, "styles": styles, "consignes": consignes,
        "format": format_label, "largeur": largeur, "hauteur": hauteur, "prompt": prompt, "webp": bool(webp), "choisie": 1,
        "personnage": personnage, "references": [str(r) for r in references] or None,
        "photo_modele": photo_modele.name if photo_modele else None, "usage_photo": usage if photo_modele else None,
        "moteur": "FLUX.2 klein 4B" if klein else "Z-Image-Turbo",
        "versions": [{"graine": g, "dossier": ".", "fichier": f} for f, g in zip(res["fichiers"], res["graines"])],
    })
    galerie = [(f, f"Variante {i} (graine {g})") for i, (f, g) in enumerate(zip(res["fichiers"], res["graines"]), 1)]
    avec = (f", avec le personnage « {personnage} »" if personnage else "") + (
        (", d'après la composition de la photo modèle" if personnage or usage == "composition"
         else ", d'après le sujet de la photo modèle") if photo_modele else "")
    msg = (f"✅ {len(galerie)} illustration(s) {largeur}×{hauteur} pour « {nom} » (projet {projet}{avec}) dans {dossier}. "
           f"Graines : {', '.join(str(g) for g in res['graines'])}.")
    return msg, galerie, str(dossier), gr.update(choices=projets(), value=projet)


def choisir(dossier, evt: gr.SelectData):
    """Clic sur une variante : elle devient celle du pack du jeu et l'illustration de la carte à composer.
    Renvoie (message, chemin de l'illustration choisie)."""
    if not dossier:
        raise gr.Error("Génère d'abord une illustration.")
    chemin = Path(dossier) / "creation.json"
    infos = json.loads(chemin.read_text(encoding="utf-8"))
    versions = infos.get("versions") or []
    if not 0 <= evt.index < len(versions):
        raise gr.Error("Variante introuvable.")
    infos["choisie"] = evt.index + 1
    chemin.write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    fichier = versions[evt.index]["fichier"]
    return (f"⭐ Variante {evt.index + 1} gardée pour « {infos.get('nom')} » (pack du projet {infos.get('projet')}) ; "
            "elle est prête à être composée en carte ci-dessous.", fichier)
