"""Onglet « Fusionner des PDF » (groupe PDF) : plusieurs PDF réunis en un seul, dans l'ordre donné."""
import gradio as gr

from .. import pdf_outils
from .commun import espace_de_noms
from .pdf_commun import brancher_sorties, entree_pdf, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Réunis plusieurs PDF en un seul**, dans l'ordre où tu les déposes (pour y mettre aussi des Word, des images… : "
        "convertis-les d'abord dans « Convertir en PDF », qui sait aussi tout réunir d'un coup).")
    fu_fichiers_in = entree_pdf("PDF à réunir (dans l'ordre)", plusieurs=True)
    fu_nom = gr.Textbox(label="Nom du fichier", value="fusion")
    btn_fu = gr.Button("➕ Fusionner", variant="primary")
    fu_statut, fu_fichiers, fu_image, fu_dossier, fu_btn_dossier = sorties(True)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_fu.click(pdf_outils.fusionner, [c.fu_fichiers_in, c.fu_nom], [c.fu_statut, c.fu_fichiers, c.fu_dossier])
    brancher_sorties(evt, c.fu_fichiers, c.fu_image, c.fu_dossier, c.fu_btn_dossier)
