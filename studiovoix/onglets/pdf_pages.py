"""Onglet « Pages : extraire, supprimer, diviser » (groupe PDF) : extraire ou supprimer des pages, diviser en plusieurs fichiers, réordonner."""
import gradio as gr

from .. import pdf_outils
from .commun import espace_de_noms
from .pdf_commun import brancher_sorties, entree_pdf, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Les pages d'un PDF** : extrais celles qui t'intéressent, supprimes-en, divise le document en plusieurs fichiers "
        "(une page par fichier, toutes les N pages, ou des plages que tu écris), ou réordonne-les. Pages écrites comme "
        "« 1-3, 5, 8- » (8- = de la 8 à la fin). Plusieurs fichiers produits sont aussi réunis dans une archive .zip.")
    pg_fichier = entree_pdf("PDF")
    with gr.Row():
        pg_mode = gr.Radio(pdf_outils.MODES_PAGES, value=pdf_outils.MODE_EXTRAIRE, label="Que faire", scale=2)
        with gr.Column():
            pg_texte = gr.Textbox(label="Pages (« 1-3, 5, 8- »), N, ou une plage par ligne", lines=3,
                                  placeholder="1-3, 5, 8-")
            pg_nom = gr.Textbox(label="Nom des fichiers (vide = nom du PDF)")
    btn_pg = gr.Button("✂️ Appliquer aux pages", variant="primary")
    pg_statut, pg_fichiers, pg_image, pg_dossier, pg_btn_dossier = sorties(True)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_pg.click(pdf_outils.pages, [c.pg_fichier, c.pg_mode, c.pg_texte, c.pg_nom], [c.pg_statut, c.pg_fichiers, c.pg_dossier])
    brancher_sorties(evt, c.pg_fichiers, c.pg_image, c.pg_dossier, c.pg_btn_dossier)
