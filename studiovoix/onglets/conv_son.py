"""Onglet « Extraire le son » (groupe Audio / vidéo) : le son d'une vidéo en MP3, M4A, WAV…"""
import gradio as gr

from .. import convertir
from .commun import espace_de_noms
from .conv_commun import brancher_sorties, entree, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Récupère le son d'une vidéo** (musique, voix, message) en MP3, M4A, WAV, FLAC, OGG ou OPUS. Plusieurs vidéos "
        "d'un coup possibles.")
    cs_entree = entree("Vidéos")
    with gr.Row():
        cs_format = gr.Dropdown(list(convertir.FORMATS_AUDIO), value="MP3", label="Format du son")
        cs_nom = gr.Textbox(label="Nom du dossier", value="son")
    btn_cs = gr.Button("🎵 Extraire le son", variant="primary")
    cs_statut, cs_video, cs_son, cs_fichiers, cs_dossier, cs_btn_dossier = sorties()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_cs.click(convertir.extraire_son, [c.cs_entree, c.cs_format, c.cs_nom], [c.cs_statut, c.cs_fichiers, c.cs_dossier])
    brancher_sorties(evt, c.cs_video, c.cs_son, c.cs_fichiers, c.cs_dossier, c.cs_btn_dossier)
