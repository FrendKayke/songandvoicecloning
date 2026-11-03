"""Onglet « OCR (scans, photos) » (groupe PDF) : PDF scanné ou photos de documents → PDF cherchable, Word, texte."""
import gradio as gr

from .. import ocr
from .commun import espace_de_noms
from .pdf_commun import brancher_sorties, sorties

SORTIES = {"PDF cherchable (Ctrl+F, copier-coller)": "pdf", "Word (.docx)": "docx", "Texte (.txt)": "txt"}


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Reconnaissance de caractères** : un PDF scanné ou des photos de documents (factures, courriers, pages de "
        "livre) deviennent un **PDF cherchable** (les pages restent identiques, le texte reconnu est posé dessus, "
        "invisible : Ctrl+F et copier-coller marchent), un document **Word** et un fichier **texte**. Plusieurs "
        "photos = un seul document, une page par photo dans l'ordre. Tout se fait sur le processeur (1 à 3 s par "
        "page). Pour de bons résultats : document à plat, bien éclairé, photographié de face.")
    oc_fichiers = gr.File(file_count="multiple", file_types=[".pdf"] + list(ocr.EXTENSIONS_IMAGE),
                          label="PDF scanné(s) ou photos de document", height=140)
    with gr.Row():
        oc_sorties = gr.CheckboxGroup(list(SORTIES), value=list(SORTIES), label="Fichiers à produire")
        oc_nom = gr.Textbox(label="Nom (vide = nom du premier fichier)")
    btn_oc = gr.Button("🔍 Lire le texte", variant="primary")
    oc_statut, oc_resultats, oc_image, oc_dossier, oc_btn_dossier = sorties()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_oc.click(lambda f, s, n, progress=gr.Progress(): ocr.reconnaitre(f, [SORTIES[x] for x in s or []], n,
                                                                                progress=progress),
                         [c.oc_fichiers, c.oc_sorties, c.oc_nom], [c.oc_statut, c.oc_resultats, c.oc_dossier])
    brancher_sorties(evt, c.oc_resultats, c.oc_image, c.oc_dossier, c.oc_btn_dossier)
