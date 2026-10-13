"""Onglet « Convertir en PDF » (groupe Outils) : n'importe quel document ou image en PDF, sur le PC."""
import gradio as gr

from .. import pdf
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Transforme tes fichiers en PDF, sans rien envoyer sur Internet** : documents Word, Excel, PowerPoint, "
        "LibreOffice, RTF, pages web, texte, Markdown, CSV, code, et images (JPG, PNG, WebP, HEIC d'iPhone, TIFF "
        "de plusieurs pages…). Dépose un ou plusieurs fichiers ; coche « Un seul PDF » pour les réunir dans l'ordre."
    )
    pdf_outils = gr.Markdown(pdf.outils_disponibles())
    pdf_fichiers = gr.File(file_count="multiple", label="Fichiers à convertir", height=160)
    with gr.Row():
        pdf_fusionner = gr.Checkbox(value=False, label="Un seul PDF (tous les fichiers à la suite, dans l'ordre)")
        pdf_nom = gr.Textbox(label="Nom", value="documents")
    btn_pdf = gr.Button("📄 Convertir en PDF", variant="primary")
    pdf_statut = gr.Markdown()
    pdf_sorties = gr.File(file_count="multiple", label="PDF obtenus", interactive=False)
    pdf_dossier = gr.State()
    btn_pdf_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_pdf.click(pdf.convertir, [c.pdf_fichiers, c.pdf_fusionner, c.pdf_nom],
                    [c.pdf_statut, c.pdf_sorties, c.pdf_dossier])
    c.btn_pdf_dossier.click(lambda d: open_folder(d) if d else None, c.pdf_dossier)
