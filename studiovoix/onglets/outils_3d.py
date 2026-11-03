"""Éléments partagés par les onglets qui reprennent un modèle 3D déjà créé (corriger, squelette, alléger) : liste des
créations 3D (la plus récente d'abord) et lecture de leurs réglages."""
import json
from pathlib import Path

import gradio as gr

from .. import galerie


def modeles_3d():
    """Créations 3D de la galerie : [(libellé, dossier)], la plus récente en premier."""
    return galerie.lister("Modèles 3D")


def maj_liste(choisie=None):
    choix = modeles_3d()
    valeurs = [v for _, v in choix]
    return gr.update(choices=choix, value=choisie if choisie in valeurs else (valeurs[0] if valeurs else None))


def infos(dossier):
    """creation.json d'un modèle 3D (dict vide si absent)."""
    if not dossier:
        return {}
    f = Path(dossier) / "creation.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def glb(dossier):
    """GLB à afficher pour un modèle (le fichier final de sa version)."""
    i = infos(dossier)
    v = (i.get("versions") or [{}])[0]
    for cle in ("fichier", "forme"):
        if v.get(cle) and Path(v[cle]).is_file():
            return v[cle]
    for nom in ("modele.glb", "forme.glb"):
        if dossier and (Path(dossier) / nom).is_file():
            return str(Path(dossier) / nom)
    return None


def liste_choisie(dossier):
    """Dropdown mis sur ce dossier (après une création dans un autre onglet)."""
    return maj_liste(dossier)
