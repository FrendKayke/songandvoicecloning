"""Onglet « Numéroter les pages » (groupe PDF) : numéro de page dessiné sur chaque page."""
import gradio as gr

from .. import pdf_outils
from .commun import espace_de_noms
from .pdf_commun import brancher_sorties, entree_pdf, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Numérote les pages** : « 1 / 12 », « Page 3 », « - 7 - »… à la position et à la taille que tu choisis. Dans le format, "
        "{n} = numéro de la page, {total} = nombre de pages ; « Première page = » décale la numérotation (par exemple 3 "
        "si deux pages de garde précèdent).")
    nu_fichier = entree_pdf("PDF")
    with gr.Row():
        nu_position = gr.Dropdown(list(pdf_outils.POSITIONS), value=list(pdf_outils.POSITIONS)[0], label="Position")
        nu_format = gr.Textbox(label="Format", value=pdf_outils.FORMAT_NUMERO_DEFAUT)
        nu_premiere = gr.Number(value=1, precision=0, label="Première page =")
        nu_taille = gr.Number(value=11, precision=0, label="Taille (points)")
        nu_marge = gr.Number(value=28, precision=0, label="Marge (points)")
        nu_nom = gr.Textbox(label="Nom (vide = nom du PDF)")
    btn_nu = gr.Button("🔢 Numéroter", variant="primary")
    nu_statut, nu_fichiers, nu_image, nu_dossier, nu_btn_dossier = sorties(True)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_nu.click(pdf_outils.numeroter, [c.nu_fichier, c.nu_position, c.nu_format, c.nu_premiere, c.nu_taille, c.nu_marge, c.nu_nom], [c.nu_statut, c.nu_fichiers, c.nu_dossier])
    brancher_sorties(evt, c.nu_fichiers, c.nu_image, c.nu_dossier, c.nu_btn_dossier)
