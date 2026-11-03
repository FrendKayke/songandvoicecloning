"""Onglet « Protéger ou déverrouiller » (groupe PDF) : mot de passe d'ouverture (AES-256) ajouté ou retiré."""
import gradio as gr

from .. import pdf_outils
from .commun import espace_de_noms
from .pdf_commun import brancher_sorties, entree_pdf, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Protège un PDF par un mot de passe** (chiffrement AES-256 : il faut le mot de passe pour l'ouvrir ; le mot de "
        "passe propriétaire, facultatif, protège en plus contre la modification), ou **retire le mot de passe** d'un PDF "
        "dont tu connais le mot de passe. Rien n'est enregistré : garde ton mot de passe.")
    pr_fichier = entree_pdf("PDF")
    with gr.Row():
        pr_mode = gr.Radio(pdf_outils.MODES_PROTECTION, value=pdf_outils.PROTEGER, label="Que faire")
        pr_mdp = gr.Textbox(label="Mot de passe (d'ouverture)", type="password")
        pr_mdp2 = gr.Textbox(label="Mot de passe propriétaire (facultatif, protection seulement)", type="password")
        pr_nom = gr.Textbox(label="Nom (vide = nom du PDF)")
    btn_pr = gr.Button("🔒 Appliquer", variant="primary")
    pr_statut, pr_fichiers, pr_image, pr_dossier, pr_btn_dossier = sorties(False)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    evt = c.btn_pr.click(pdf_outils.proteger, [c.pr_fichier, c.pr_mode, c.pr_mdp, c.pr_mdp2, c.pr_nom], [c.pr_statut, c.pr_fichiers, c.pr_dossier])
    brancher_sorties(evt, c.pr_fichiers, c.pr_image, c.pr_dossier, c.pr_btn_dossier)
