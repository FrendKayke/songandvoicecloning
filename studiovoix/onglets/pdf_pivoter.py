"""Onglet « Pivoter des pages » (groupe PDF) : rotation de 90, 180 ou 270° de toutes les pages ou de certaines."""
import gradio as gr

from .. import pdf_outils
from .commun import espace_de_noms
from .pdf_commun import brancher_sorties, entree_pdf, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Tourne les pages** d'un PDF scanné de travers ou en paysage : toutes, ou seulement celles indiquées.")
    pv_fichier = entree_pdf("PDF")
    with gr.Row():
        pv_angle = gr.Radio(list(pdf_outils.ANGLES), value=list(pdf_outils.ANGLES)[0], label="Rotation")
        pv_pages = gr.Textbox(label="Pages (vide = toutes)", placeholder="1-2, 5")
        pv_nom = gr.Textbox(label="Nom (vide = nom du PDF)")
    btn_pv = gr.Button("🔄 Pivoter", variant="primary")
    pv_statut, pv_fichiers, pv_image, pv_dossier, pv_btn_dossier = sorties(True)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_pv.click(pdf_outils.pivoter, [c.pv_fichier, c.pv_angle, c.pv_pages, c.pv_nom], [c.pv_statut, c.pv_fichiers, c.pv_dossier])
    brancher_sorties(evt, c.pv_fichiers, c.pv_image, c.pv_dossier, c.pv_btn_dossier)
