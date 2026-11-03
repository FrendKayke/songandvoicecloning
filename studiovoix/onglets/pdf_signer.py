"""Onglet « Signer un PDF » (groupe PDF) : image de signature posée sur une page."""
import gradio as gr

from .. import pdf_outils
from .commun import espace_de_noms
from .pdf_commun import brancher_sorties, entree_pdf, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Pose ta signature** (une image : signature photographiée ou scannée, détourée en PNG transparent avec « Retoucher "
        "une photo → Détourer » pour un résultat propre) sur la dernière page, ou sur les pages indiquées. Position en "
        "% de la page depuis la gauche et depuis le bas, largeur en % de la page. Ce n'est pas une signature "
        "électronique certifiée : c'est l'image, comme une signature à la main.")
    si_fichier = entree_pdf("PDF")
    with gr.Row():
        si_image = gr.Image(type="filepath", label="Image de la signature", height=220)
        with gr.Column():
            si_pages = gr.Textbox(label="Pages (vide = la dernière)", placeholder="1, 3")
            with gr.Row():
                si_x = gr.Slider(0, 100, value=80, step=1, label="Position horizontale (% depuis la gauche)")
                si_y = gr.Slider(0, 100, value=12, step=1, label="Position verticale (% depuis le bas)")
                si_largeur = gr.Slider(5, 100, value=25, step=1, label="Largeur (% de la page)")
            si_nom = gr.Textbox(label="Nom (vide = nom du PDF)")
    btn_si = gr.Button("✍️ Signer", variant="primary")
    si_statut, si_fichiers, si_image, si_dossier, si_btn_dossier = sorties(True)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_si.click(pdf_outils.signer, [c.si_fichier, c.si_image, c.si_pages, c.si_x, c.si_y, c.si_largeur, c.si_nom], [c.si_statut, c.si_fichiers, c.si_dossier])
    brancher_sorties(evt, c.si_fichiers, c.si_image, c.si_dossier, c.si_btn_dossier)
