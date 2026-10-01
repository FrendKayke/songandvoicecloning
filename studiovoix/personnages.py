"""Personnages récurrents d'un projet de cartes : le même héros, le même monstre d'une carte à l'autre.

Un personnage = 1 à 4 images de référence (visage, en pied, tenue…), importées ou prises parmi les illustrations
générées. Quand une illustration est générée avec un personnage, FLUX.2 klein 4B (Apache 2.0) reçoit ces images
avec la scène (moteurs/diffusion.py, action « personnage ») au lieu de Z-Image-Turbo.
Rangement : data/cartes/<projet>/personnages/<nom>/ref_<i>.png (pas de creation.json : ni galerie, ni pack).
Comme Windows, les noms sont comparés sans tenir compte de la casse.
"""
import shutil
from pathlib import Path

import gradio as gr
from PIL import Image

from . import config as cfg

AUCUN = "(aucun : Z-Image-Turbo)"
REFERENCES_MAX = 4  # limite de FLUX.2 klein (API de Black Forest Labs) ; chaque image ajoute du calcul
COTE_MAX = 1024  # klein réduit de toute façon chaque référence à 1024² au plus


def _racine(projet):
    from .cartes import nom_projet

    return cfg.CARDS_DIR / nom_projet(projet) / "personnages"


def _nettoyer(nom):
    nom = "".join(c for c in (nom or "").strip() if c.isalnum() or c in "-_ ").strip()
    if not nom:
        raise gr.Error("Donne un nom au personnage (par exemple « Héroïne » ou « Roi gobelin »).")
    return nom


def liste(projet):
    racine = _racine(projet)
    if not racine.is_dir():
        return []
    return sorted((d.name for d in racine.iterdir() if d.is_dir() and any(d.glob("ref_*.png"))), key=str.casefold)


def _dossier(projet, nom, creer=False):
    """Dossier du personnage (nom retrouvé sans tenir compte de la casse) ; None s'il n'existe pas."""
    nom = _nettoyer(nom)
    racine = _racine(projet)
    if racine.is_dir():
        for d in racine.iterdir():
            if d.is_dir() and d.name.casefold() == nom.casefold():
                return d
    if creer:
        (racine / nom).mkdir(parents=True, exist_ok=True)
        return racine / nom
    return None


def references(projet, nom):
    if not nom or nom == AUCUN:
        return []
    d = _dossier(projet, nom)
    return sorted(d.glob("ref_*.png"), key=lambda p: int(p.stem.split("_")[1])) if d else []


def choix(projet, valeur=AUCUN):
    """Liste « Personnage de la scène » du projet."""
    noms = liste(projet)
    return gr.update(choices=[AUCUN] + noms, value=valeur if valeur in noms else AUCUN)


def _galerie(projet, nom):
    return [(str(p), f"Référence {i}") for i, p in enumerate(references(projet, nom), 1)]


def ajouter(projet, nom, images):
    """Ajoute des images de référence. Renvoie (message, galerie des références, liste des personnages)."""
    images = [i for i in (images or []) if i]
    if not images:
        raise gr.Error("Ajoute au moins une image du personnage (importée, ou la variante gardée ci-dessus).")
    deja = references(projet, nom)
    place = REFERENCES_MAX - len(deja)
    if place <= 0:
        raise gr.Error(f"Ce personnage a déjà {REFERENCES_MAX} images de référence (le maximum) : supprime-le pour "
                       "en choisir d'autres.")
    d = _dossier(projet, nom, creer=True)
    suivant = max((int(p.stem.split("_")[1]) for p in deja), default=0) + 1
    ajoutees = 0
    for i, chemin in enumerate(images[:place]):
        try:
            with Image.open(getattr(chemin, "name", chemin)) as im:
                im = im.convert("RGB")
                im.thumbnail((COTE_MAX, COTE_MAX))
                im.save(d / f"ref_{suivant + i}.png")
                ajoutees += 1
        except OSError as e:
            raise gr.Error(f"Image illisible ({Path(str(chemin)).name}) : {e}")
    reste = len(images) - ajoutees
    msg = (f"✅ {ajoutees} image(s) ajoutée(s) à « {d.name} » ({len(deja) + ajoutees}/{REFERENCES_MAX})."
           + (f" {reste} image(s) ignorée(s) : {REFERENCES_MAX} au plus." if reste else "")
           + " Choisis-le dans « Personnage de la scène » pour le faire apparaître sur tes cartes.")
    return msg, _galerie(projet, d.name), choix(projet, d.name)


def afficher(projet, nom):
    return _galerie(projet, nom) if nom and nom != AUCUN else []


def supprimer(projet, nom):
    """Supprime le personnage (ses images de référence ; les illustrations déjà faites restent)."""
    if not nom or nom == AUCUN:  # aussi quand la confirmation est refusée
        return "Suppression annulée (ou aucun personnage choisi).", gr.update(), gr.update()
    d = _dossier(projet, nom)
    if d is None or d.parent != _racine(projet):
        raise gr.Error(f"Personnage introuvable : {nom}")
    shutil.rmtree(d)
    return f"🗑️ Personnage « {d.name} » supprimé (les illustrations déjà faites sont gardées).", [], choix(projet)
