"""Éléments partagés par les onglets du groupe « Audio / vidéo » : entrée, sorties (message, lecteur du premier
résultat, fichiers, dossier)."""
from pathlib import Path

import gradio as gr

from .. import convertir
from ..outils import open_folder

TYPES = sorted(convertir.VIDEOS | convertir.AUDIOS)


def entree(label, plusieurs=True):
    return gr.File(file_count="multiple" if plusieurs else "single", file_types=TYPES, label=label, height=150)


def sorties():
    """(statut, vidéo, son, fichiers, dossier, bouton dossier)."""
    statut = gr.Markdown()
    dossier = gr.State()
    with gr.Row():
        video = gr.Video(label="Résultat", interactive=False, height=360, visible=False)
        son = gr.Audio(label="Résultat", type="filepath", interactive=False, visible=False)
        fichiers = gr.File(label="Fichiers produits", file_count="multiple", interactive=False)
    bouton = gr.Button("📂 Ouvrir le dossier")
    return statut, video, son, fichiers, dossier, bouton


def apercu(fichiers):
    """Le premier fichier produit dans le bon lecteur (vidéo ou son)."""
    premier = next(iter(fichiers or []), None)
    if not premier:
        return gr.update(value=None, visible=False), gr.update(value=None, visible=False)
    if Path(premier).suffix.lower() in convertir.VIDEOS:
        return gr.update(value=premier, visible=True), gr.update(value=None, visible=False)
    return gr.update(value=None, visible=False), gr.update(value=premier, visible=True)


def brancher_sorties(evenement, video, son, fichiers, dossier, bouton):
    evenement.then(apercu, fichiers, [video, son])
    bouton.click(lambda d: open_folder(d) if d else None, dossier)
