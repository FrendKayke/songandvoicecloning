"""Onglet « Bruitages » : effets sonores pour un jeu, à partir d'une description (français ou anglais) ou d'une image.

Qwen3-VL transforme la description française en prompt anglais précis (ou décrit les sons d'une image) ;
Stable Audio Open génère 1 à 3 variantes ; l'export reprend le volume harmonisé de la bande-son.
Rangement : data/bruitages/<horodatage>/ (variante_<i>.wav, creation.json).
"""
import shutil
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import diffusion
from .outils import ecrire_creation, nouveau_dossier

DUREE_MAX = 30
NEGATIF = "music, melody, singing, speech, low quality, distortion"

EXEMPLES = [
    ("Carte jouée", "card sliding on a wooden table then a soft thump"),
    ("Épée dégainée", "metal sword drawn from a leather scabbard, sharp ring"),
    ("Sort de feu", "fireball whoosh and burst, crackling flames"),
    ("Sort de glace", "ice crystal forming and shattering, cold shimmer"),
    ("Soin magique", "gentle magical healing chime, warm sparkle"),
    ("Coup d'épée", "sword slash hitting armor, metallic clang"),
    ("Pièces d'or", "coins pouring into a leather pouch"),
    ("Clic de menu", "short soft UI click"),
    ("Erreur", "short negative UI buzz"),
    ("Porte de château", "heavy wooden castle door creaking open and slamming"),
    ("Pas sur la pierre", "footsteps on stone floor in a hall, echo"),
    ("Tonnerre", "distant thunder rumble"),
]


def preparer(texte, image_path, progress=gr.Progress()):
    """Description → prompt anglais (Qwen3-VL). Renvoie le prompt, modifiable avant génération."""
    texte = (texte or "").strip()
    if image_path:
        return diffusion.decrire("son", texte or None, image_path, progress=progress)
    if not texte:
        raise gr.Error("Décris le bruitage (en français ou en anglais), ou importe une image.")
    return diffusion.decrire("bruitage", texte, progress=progress)


def generer(prompt, nom, duree, variantes, graine, etapes, image_path=None, description=None,
            progress=gr.Progress()):
    """Génère les variantes. Renvoie (message, choix des variantes, première variante, dossier)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Il manque le prompt : clique d'abord sur « Préparer le prompt », ou écris-le en anglais.")
    duree = max(1.0, min(DUREE_MAX, float(duree or 3)))
    dossier = nouveau_dossier(cfg.SFX_DIR)
    if image_path:
        shutil.copy(image_path, dossier / ("image" + Path(image_path).suffix.lower()))
    res = diffusion.bruitage(prompt, dossier, duree, variantes, graine, etapes, NEGATIF, progress)
    nom = "".join(c for c in (nom or "").strip() if c.isalnum() or c in "-_ ").strip() or "bruitage"
    ecrire_creation(dossier, {
        "type": "bruitage", "nom": nom, "description": prompt, "description_fr": (description or "").strip() or None,
        "duree": duree, "etapes": int(etapes), "negatif": NEGATIF,
        "versions": [{"graine": g, "dossier": ".", "fichier": f} for f, g in zip(res["fichiers"], res["graines"])],
    })
    choix = [(f"Variante {i} (graine {g})", f) for i, (f, g) in enumerate(zip(res["fichiers"], res["graines"]), 1)]
    msg = (f"✅ {len(choix)} variante(s) de {duree:.0f} s dans {dossier}. Graines : "
           f"{', '.join(str(g) for g in res['graines'])}.")
    return msg, gr.update(choices=choix, value=choix[0][1]), choix[0][1], str(dossier)
