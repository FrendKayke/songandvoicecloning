"""Onglet « Filigrane » (groupe PDF) : texte en diagonale ou image en transparence sur chaque page."""
import gradio as gr

from .. import pdf_outils
from .commun import espace_de_noms
from .pdf_commun import brancher_sorties, entree_pdf, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Un filigrane sur chaque page** : un texte (« CONFIDENTIEL », « BROUILLON », ton nom) en gris transparent, en "
        "diagonale ou droit, et / ou une image (logo en PNG transparent) au centre de la page.")
    fi_fichier = entree_pdf("PDF")
    with gr.Row():
        with gr.Column():
            fi_texte = gr.Textbox(label="Texte du filigrane", placeholder="CONFIDENTIEL")
            with gr.Row():
                fi_taille = gr.Number(value=48, precision=0, label="Taille (points)")
                fi_angle = gr.Number(value=45, precision=0, label="Angle (°)")
                fi_opacite = gr.Slider(0.05, 1, value=0.25, step=0.05, label="Opacité")
            fi_nom = gr.Textbox(label="Nom (vide = nom du PDF)")
        fi_image = gr.Image(type="filepath", label="Ou une image (logo, PNG transparent)", height=220)
    btn_fi = gr.Button("💧 Poser le filigrane", variant="primary")
    fi_statut, fi_fichiers, fi_image, fi_dossier, fi_btn_dossier = sorties(True)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_fi.click(pdf_outils.filigrane, [c.fi_fichier, c.fi_texte, c.fi_image, c.fi_opacite, c.fi_taille, c.fi_angle, c.fi_nom], [c.fi_statut, c.fi_fichiers, c.fi_dossier])
    brancher_sorties(evt, c.fi_fichiers, c.fi_image, c.fi_dossier, c.fi_btn_dossier)
