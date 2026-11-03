"""Éléments partagés par les onglets du groupe PDF : entrée (un PDF), sorties (message, fichiers, aperçu de la
première page du résultat, dossier)."""
import gradio as gr

from .. import pdf_outils
from ..outils import open_folder


def entree_pdf(label="PDF", plusieurs=False):
    return gr.File(file_count="multiple" if plusieurs else "single", file_types=[".pdf"], label=label, height=140)


def sorties(apercu=True):
    """(statut, fichiers, aperçu, dossier, bouton dossier)."""
    statut = gr.Markdown()
    dossier = gr.State()
    with gr.Row():
        fichiers = gr.File(label="Résultat", file_count="multiple", interactive=False)
        image = gr.Image(label="Aperçu de la première page du résultat", interactive=False, height=420,
                         visible=apercu)
    bouton = gr.Button("📂 Ouvrir le dossier")
    return statut, fichiers, image, dossier, bouton


def brancher_sorties(evenement, fichiers, image, dossier, bouton):
    """Après un outil : aperçu du premier PDF produit, bouton du dossier."""
    evenement.then(lambda f: pdf_outils.apercu(next((x for x in (f or []) if str(x).lower().endswith(".pdf")), None)),
                   fichiers, image)
    bouton.click(lambda d: open_folder(d) if d else None, dossier)
