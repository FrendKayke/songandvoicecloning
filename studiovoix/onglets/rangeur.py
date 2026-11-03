"""Onglet « Ranger mes créations » (groupe Outils) : copie les créations d'un projet, d'une période ou de quelques
types dans un dossier propre (un sous-dossier par type, noms lisibles, liste.csv, .zip)."""
import gradio as gr

from .. import rangeur
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Copie tes créations dans un dossier bien rangé : un dossier par type (Chansons, Images, Modèles 3D…), des "
        "noms lisibles avec la date (« 2026-11-03_Château dans la brume.png »), la liste de tout dans `liste.csv` "
        "(s'ouvre dans Excel) et une archive .zip à envoyer ou sauvegarder. Rien n'est déplacé ni supprimé : la "
        "galerie garde tout. Laisse les dates vides pour tout prendre."
    )
    with gr.Row():
        ra_projet = gr.Dropdown(rangeur.projets(), value=rangeur.TOUS, label="Projet de jeu",
                                info="bande-son, bruitages, illustrations et modèles 3D de ce projet")
        ra_types = gr.Dropdown(list(rangeur.TYPES), value=[], multiselect=True, label="Types (vide = tous)")
    with gr.Row():
        ra_du = gr.Textbox(label="Du (jj/mm/aaaa)", placeholder="01/11/2026")
        ra_au = gr.Textbox(label="Au (jj/mm/aaaa)", placeholder="30/11/2026")
        ra_nom = gr.Textbox(label="Nom du dossier", placeholder="mes_creations")
    with gr.Row():
        ra_toutes = gr.Checkbox(value=True, label="Toutes les versions et variantes",
                                info="décoché : seulement la variante choisie (ou la première)")
        ra_zip = gr.Checkbox(value=True, label="Faire aussi une archive .zip")
    with gr.Row():
        btn_ra_compter = gr.Button("🔎 Compter")
        btn_ra_ranger = gr.Button("📦 Ranger", variant="primary")
    ra_msg = gr.Markdown()
    ra_fichiers = gr.File(label="Archive et liste", file_count="multiple")
    ra_dossier = gr.State()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    filtres = [c.ra_projet, c.ra_types, c.ra_du, c.ra_au]
    c.btn_ra_compter.click(rangeur.apercu, filtres, c.ra_msg)
    c.btn_ra_ranger.click(rangeur.ranger, filtres + [c.ra_toutes, c.ra_zip, c.ra_nom],
                          [c.ra_msg, c.ra_fichiers, c.ra_dossier])
    c.onglet.select(lambda p: gr.update(choices=rangeur.projets(), value=p if p in rangeur.projets() else rangeur.TOUS),
                    c.ra_projet, c.ra_projet)
