"""Onglet « Changer le format » (groupe Audio / vidéo) : un ou plusieurs fichiers dans un autre format."""
import gradio as gr

from .. import convertir
from .commun import espace_de_noms
from .conv_commun import brancher_sorties, entree, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Change le format** de tes fichiers audio ou vidéo : MP4, WebM, MKV, MOV, GIF animé, MP3, M4A, WAV, FLAC, OGG, "
        "OPUS. Une vidéo convertie dans un format audio ne garde que le son. Dépose plusieurs fichiers pour les "
        "convertir tous d'un coup.")
    cf_entree = entree("Fichiers audio ou vidéo")
    with gr.Row():
        cf_format = gr.Dropdown(list(convertir.FORMATS), value="MP4 (H.264, partout)", label="Format de sortie")
        cf_nom = gr.Textbox(label="Nom du dossier", value="conversion")
    btn_cf = gr.Button("🔁 Convertir", variant="primary")
    cf_statut, cf_video, cf_son, cf_fichiers, cf_dossier, cf_btn_dossier = sorties()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_cf.click(convertir.changer_format, [c.cf_entree, c.cf_format, c.cf_nom], [c.cf_statut, c.cf_fichiers, c.cf_dossier])
    brancher_sorties(evt, c.cf_video, c.cf_son, c.cf_fichiers, c.cf_dossier, c.cf_btn_dossier)
