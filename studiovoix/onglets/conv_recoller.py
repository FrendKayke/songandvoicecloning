"""Onglet « Recoller » (groupe Audio / vidéo) : plusieurs fichiers mis bout à bout."""
import gradio as gr

from .. import convertir
from .commun import espace_de_noms
from .conv_commun import brancher_sorties, entree, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Mets des vidéos (ou des sons) bout à bout**, dans l'ordre où tu les déposes. Les vidéos prennent la taille et la "
        "cadence de la première (bandes noires si une vidéo a d'autres proportions, une vidéo muette reçoit du "
        "silence). Pour des fondus, une musique de fond ou un titre : le futur « Montage vidéo ».")
    cr_entree = entree("Fichiers à recoller, dans l'ordre")
    with gr.Row():
        cr_format = gr.Dropdown(list(convertir.FORMATS), value="MP4 (H.264, partout)",
                                label="Format (des sons seuls sont recollés en audio)")
        cr_nom = gr.Textbox(label="Nom", value="montage")
    btn_cr = gr.Button("🔗 Recoller", variant="primary")
    cr_statut, cr_video, cr_son, cr_fichiers, cr_dossier, cr_btn_dossier = sorties()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_cr.click(convertir.recoller, [c.cr_entree, c.cr_format, c.cr_nom], [c.cr_statut, c.cr_fichiers, c.cr_dossier])
    brancher_sorties(evt, c.cr_video, c.cr_son, c.cr_fichiers, c.cr_dossier, c.cr_btn_dossier)
