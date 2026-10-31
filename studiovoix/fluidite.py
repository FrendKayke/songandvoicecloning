"""Vidéos plus fluides et ralentis (RIFE 4.26, moteur de diffusion, action « fluidifier »).

RIFE intercale des images entre deux images voisines : « plus fluide » garde la durée et multiplie la cadence (une
vidéo Wan de 24 images/s passe à 48 ; le son est gardé), « ralenti » garde la cadence et allonge la durée (sans son).
Aux changements de plan, l'image est répétée au lieu d'un fondu inventé. Une vidéo de l'application (data/…) reçoit
sa version à côté d'elle (<nom>_fluide_x2.mp4, <nom>_ralenti_x4.mp4) ; une vidéo importée va dans
data/videos_fluides/<horodatage>_<nom>/.
"""
import time
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import diffusion
from .outils import nouveau_dossier, chemin_fichier
from .serie import _slug

FACTEURS = {"× 2 (24 → 48 images/s)": 2, "× 3": 3, "× 4": 4}
FACTEUR_DEFAUT = next(iter(FACTEURS))
FLUIDE = "Plus fluide (même durée)"
RALENTI = "Ralenti (même cadence, plus long, sans le son)"
MODES = [FLUIDE, RALENTI]


def _destination(source, facteur, ralenti):
    nom = f"{source.stem}_{'ralenti' if ralenti else 'fluide'}_x{facteur}.mp4"
    try:
        source.resolve().relative_to(cfg.DATA_DIR.resolve())
        return source.with_name(nom)
    except ValueError:  # vidéo importée : dossier à part, l'original n'est pas touché
        return nouveau_dossier(cfg.FLUIDE_DIR, _slug(source.stem, "video")) / nom


def fluidifier(video, facteur_label=FACTEUR_DEFAUT, mode=FLUIDE, progress=gr.Progress()):
    """Renvoie (message, vidéo obtenue, dossier)."""
    source = Path(str(chemin_fichier(video))) if video else None
    if not source or not source.is_file():
        raise gr.Error("Choisis d'abord une vidéo.")
    facteur = FACTEURS.get(facteur_label, 2)
    ralenti = mode == RALENTI
    sortie = _destination(source, facteur, ralenti)
    debut = time.monotonic()
    r = diffusion.fluidifier(source, sortie, facteur, ralenti, progress=progress)
    msg = (f"✅ {'Ralenti' if ralenti else 'Vidéo plus fluide'} × {facteur} : {r['images']} images à "
           f"{r['fps']:g} images/s ({r['duree']:.1f} s), en {time.monotonic() - debut:.0f} s. Fichier : {sortie}")
    if r.get("plans"):
        msg += f"\n\n{r['plans']} changement(s) de plan : image répétée à cet endroit, sans fondu inventé."
    return msg, str(sortie), str(sortie.parent)
