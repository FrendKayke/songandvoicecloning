"""Onglet « Vidéos » : texte → vidéo et image → vidéo avec Wan 2.2 TI2V-5B (Apache 2.0 : vidéos vendables).

La description française est reformulée en anglais par Qwen3-VL (mode « video » : sujet, action, décor, lumière,
mouvement de caméra), modifiable avant génération. Avec une image de départ (illustration, photo, dernière image
d'un clip précédent), la vidéo part de cette image. Wan 2.2 est entraîné en 720p à 24 images/s ; durée = 4k + 1
images (121 = 5 s). Pas de son : les bruitages et musiques se font dans leurs onglets.
Rangement : data/videos/<horodatage>/ : video.mp4, derniere_image.png, image_depart.<ext>, creation.json (type
« video »).
"""
import shutil
from pathlib import Path

import gradio as gr
from PIL import Image

from . import config as cfg
from . import diffusion
from .outils import ecrire_creation, nouveau_dossier

AUTO = "Automatique (d'après l'image de départ, sinon paysage)"
# côtés multiples de 32 (VAE ×16 et blocs ×2 du 5B) ; 720p = résolution d'entraînement
FORMATS = {
    AUTO: None,
    "Paysage 16:9, 720p (1280×704)": (1280, 704),
    "Portrait 9:16, 720p (704×1280)": (704, 1280),
    "Carré (960×960)": (960, 960),
    "Paysage léger (832×480, plus rapide)": (832, 480),
    "Portrait léger (480×832, plus rapide)": (480, 832),
}
DUREES = {"2 s": 49, "3 s": 73, "4 s": 97, "5 s": 121}
DUREE_DEFAUT = "5 s"
FPS = 24
EXEMPLES = [
    ("Dragon qui s'envole", "un dragon rouge déploie ses ailes et s'envole d'un sommet enneigé, la caméra le suit"),
    ("Carte qui s'anime", "le personnage de l'illustration tourne lentement la tête et sourit, ses cheveux bougent "
                          "dans le vent, léger zoom avant"),
    ("Potion magique", "une fiole de potion bouillonne et laisse échapper une fumée verte lumineuse, plan fixe"),
    ("Forêt enchantée", "travelling lent dans une forêt brumeuse au lever du soleil, des lucioles flottent"),
    ("Chevalier au combat", "un chevalier en armure dorée lève son épée lumineuse face à la caméra, éclairs au loin"),
]


def format_pour(format_label, image=None):
    """(largeur, hauteur) : le format choisi, ou en automatique celui qui ressemble le plus à l'image."""
    taille = FORMATS.get(format_label)
    if taille:
        return taille
    if image:
        with Image.open(image) as im:
            rapport = im.width / im.height
        if rapport > 1.15:
            return FORMATS["Paysage 16:9, 720p (1280×704)"]
        if rapport < 0.87:
            return FORMATS["Portrait 9:16, 720p (704×1280)"]
        return FORMATS["Carré (960×960)"]
    return FORMATS["Paysage 16:9, 720p (1280×704)"]


def preparer(description, progress=gr.Progress()):
    """Description française → prompt anglais pour la vidéo (Qwen3-VL), modifiable."""
    description = (description or "").strip()
    if not description:
        raise gr.Error("Décris la vidéo (en français ou en anglais).")
    return diffusion.decrire("video", description, progress=progress)


def generer(prompt, image, format_label, duree_label, etapes, graine, nom="", description_fr=None,
            progress=gr.Progress()):
    """Génère la vidéo. Renvoie (message, vidéo, dossier)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Il manque le prompt : clique d'abord sur « Préparer le prompt », ou écris-le en anglais.")
    largeur, hauteur = format_pour(format_label, image)
    images = DUREES.get(duree_label, DUREES[DUREE_DEFAUT])
    graine = int(graine or 0) or diffusion.graines(1)[0]
    dossier = nouveau_dossier(cfg.VIDEOS_DIR)
    depart = None
    if image:
        depart = dossier / f"image_depart{Path(image).suffix.lower() or '.png'}"
        shutil.copy(image, depart)
    res = diffusion.video(prompt, dossier / "video.mp4", depart, largeur, hauteur, images, etapes, graine,
                          progress=progress)
    nom = (nom or "").strip() or "video"
    ecrire_creation(dossier, {
        "type": "video", "nom": nom, "description": prompt, "description_fr": (description_fr or "").strip() or None,
        "image_depart": depart.name if depart else None, "format": format_label, "largeur": largeur,
        "hauteur": hauteur, "duree": duree_label, "images": images, "etapes": int(etapes), "fps": FPS,
        "versions": [{"graine": res["graine"], "dossier": ".", "fichier": res["sortie"]}],
    })
    mode = "à partir de l'image" if depart else "à partir du texte"
    msg = (f"✅ Vidéo {largeur}×{hauteur} de {res['duree']} s {mode}, graine {res['graine']}, dans {dossier}. "
           "Pas de son : ajoute un bruitage ou une musique depuis leurs onglets.")
    return msg, res["sortie"], str(dossier)


def continuer(dossier):
    """La dernière image de la vidéo devient l'image de départ du clip suivant."""
    derniere = Path(dossier or "") / "derniere_image.png"
    if not dossier or not derniere.exists():
        raise gr.Error("Génère d'abord une vidéo.")
    return str(derniere)


def recreer(chemin, infos, graine, progress=gr.Progress()):
    """Galerie : même vidéo, mêmes réglages, même graine (résultat proche)."""
    depart = Path(chemin) / infos["image_depart"] if infos.get("image_depart") else None
    if depart is not None and not depart.exists():
        raise gr.Error(f"Image de départ introuvable : {depart}")
    _, _, dossier = generer(infos.get("description"), str(depart) if depart else None, infos.get("format") or AUTO,
                            infos.get("duree") or DUREE_DEFAUT, infos.get("etapes", 30), graine, infos.get("nom"),
                            infos.get("description_fr"), progress=progress)
    return dossier
