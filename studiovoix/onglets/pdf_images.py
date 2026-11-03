"""Onglet « PDF en images » (groupe PDF) : chaque page rendue en PNG ou JPEG."""
import gradio as gr

from .. import pdf_outils
from .commun import espace_de_noms
from .pdf_commun import brancher_sorties, entree_pdf, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Chaque page en image** (PNG ou JPEG, à la résolution choisie : 150 ppp pour l'écran, 300 pour l'impression), "
        "plus une archive .zip. Pour faire l'inverse, images → PDF : « Convertir en PDF ».")
    im_fichier = entree_pdf("PDF")
    with gr.Row():
        im_format = gr.Radio(list(pdf_outils.FORMATS_IMAGE), value="PNG", label="Format")
        im_ppp = gr.Dropdown(pdf_outils.PPP, value=150, label="Résolution (ppp)")
        im_pages = gr.Textbox(label="Pages (vide = toutes)", placeholder="1-3")
        im_nom = gr.Textbox(label="Nom (vide = nom du PDF)")
    btn_im = gr.Button("🖼️ Convertir en images", variant="primary")
    im_statut, im_fichiers, im_image, im_dossier, im_btn_dossier = sorties(False)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_im.click(pdf_outils.en_images, [c.im_fichier, c.im_format, c.im_ppp, c.im_pages, c.im_nom], [c.im_statut, c.im_fichiers, c.im_dossier])
    brancher_sorties(evt, c.im_fichiers, c.im_image, c.im_dossier, c.im_btn_dossier)
