"""Onglet « Couper » (groupe Audio / vidéo) : garder un passage entre deux instants."""
import gradio as gr

from .. import convertir
from .commun import espace_de_noms
from .conv_commun import brancher_sorties, entree, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Garde seulement un passage** d'un fichier audio ou vidéo. Écris le début et la fin en secondes (« 12,5 ») ou en "
        "minutes (« 1:30 », « 1 min 30 ») ; fin vide = jusqu'au bout. Le résultat garde le format d'origine.")
    cc_entree = entree("Fichier audio ou vidéo", plusieurs=False)
    cc_infos = gr.Markdown()
    with gr.Row():
        cc_debut = gr.Textbox(label="Début", value="0", placeholder="0:15")
        cc_fin = gr.Textbox(label="Fin (vide = jusqu'au bout)", placeholder="1:30")
        cc_nom = gr.Textbox(label="Nom (vide = nom du fichier)")
    btn_cc = gr.Button("✂️ Couper", variant="primary")
    cc_statut, cc_video, cc_son, cc_fichiers, cc_dossier, cc_btn_dossier = sorties()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.cc_entree.change(convertir.decrire, c.cc_entree, c.cc_infos)
    evt = c.btn_cc.click(convertir.couper, [c.cc_entree, c.cc_debut, c.cc_fin, c.cc_nom], [c.cc_statut, c.cc_fichiers, c.cc_dossier])
    brancher_sorties(evt, c.cc_video, c.cc_son, c.cc_fichiers, c.cc_dossier, c.cc_btn_dossier)
