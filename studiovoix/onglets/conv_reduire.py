"""Onglet « Réduire le poids » (groupe Audio / vidéo) : vidéos et sons plus légers."""
import gradio as gr

from .. import convertir
from .commun import espace_de_noms
from .conv_commun import brancher_sorties, entree, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Allège tes vidéos** pour les envoyer par message ou les mettre en ligne : image réduite (1080p, 720p, 480p, "
        "360p au plus) et réencodée en H.264 (MP4), son en AAC. Un fichier audio devient un MP3 plus léger. Les "
        "proportions et la cadence sont gardées.")
    cl_entree = entree("Vidéos ou sons")
    with gr.Row():
        cl_niveau = gr.Radio(list(convertir.REDUCTIONS), value=list(convertir.REDUCTIONS)[1], label="Niveau")
        cl_nom = gr.Textbox(label="Nom du dossier", value="allege")
    btn_cl = gr.Button("🪶 Réduire le poids", variant="primary")
    cl_statut, cl_video, cl_son, cl_fichiers, cl_dossier, cl_btn_dossier = sorties()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_cl.click(convertir.reduire, [c.cl_entree, c.cl_niveau, c.cl_nom], [c.cl_statut, c.cl_fichiers, c.cl_dossier])
    brancher_sorties(evt, c.cl_video, c.cl_son, c.cl_fichiers, c.cl_dossier, c.cl_btn_dossier)
