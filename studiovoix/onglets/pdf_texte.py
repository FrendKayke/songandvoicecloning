"""Onglet « PDF en texte ou Word » (groupe PDF) : texte extrait (.txt) et document Word (.docx)."""
import gradio as gr

from .. import pdf_outils
from .commun import espace_de_noms
from .pdf_commun import brancher_sorties, entree_pdf, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Le texte d'un PDF** en fichier texte (.txt) et en document Word (.docx, un saut de page par page) : le texte, "
        "pas la mise en page. Un PDF scanné (images seules) n'a pas de texte à extraire : la reconnaissance de "
        "caractères n'est pas installée.")
    tx_fichier = entree_pdf("PDF")
    with gr.Row():
        tx_word = gr.Checkbox(value=True, label="Aussi en Word (.docx)")
        tx_nom = gr.Textbox(label="Nom (vide = nom du PDF)")
    btn_tx = gr.Button("📝 Extraire le texte", variant="primary")
    tx_statut, tx_fichiers, tx_image, tx_dossier, tx_btn_dossier = sorties(False)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_tx.click(pdf_outils.en_texte, [c.tx_fichier, c.tx_word, c.tx_nom], [c.tx_statut, c.tx_fichiers, c.tx_dossier])
    brancher_sorties(evt, c.tx_fichiers, c.tx_image, c.tx_dossier, c.tx_btn_dossier)
