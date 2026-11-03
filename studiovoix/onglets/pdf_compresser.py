"""Onglet « Compresser un PDF » (groupe PDF) : images réencodées en JPEG, flux compressés."""
import gradio as gr

from .. import pdf_outils
from .commun import espace_de_noms
from .pdf_commun import brancher_sorties, entree_pdf, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Réduis le poids d'un PDF** : ses images sont réencodées en JPEG (qualité et taille maximale selon le niveau), les "
        "flux de contenu sont compressés et les objets en double retirés. Le texte reste du texte (toujours lisible et "
        "sélectionnable). Un PDF déjà compact, sans image, ne maigrit presque pas.")
    cp_fichier = entree_pdf("PDF")
    with gr.Row():
        cp_qualite = gr.Radio(list(pdf_outils.QUALITES_COMPRESSION), value=list(pdf_outils.QUALITES_COMPRESSION)[1],
                              label="Niveau")
        cp_nom = gr.Textbox(label="Nom (vide = nom du PDF)")
    btn_cp = gr.Button("🗜️ Compresser", variant="primary")
    cp_statut, cp_fichiers, cp_image, cp_dossier, cp_btn_dossier = sorties(True)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_cp.click(pdf_outils.compresser, [c.cp_fichier, c.cp_qualite, c.cp_nom], [c.cp_statut, c.cp_fichiers, c.cp_dossier])
    brancher_sorties(evt, c.cp_fichiers, c.cp_image, c.cp_dossier, c.cp_btn_dossier)
