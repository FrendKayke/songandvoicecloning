"""Onglet « Pack du jeu » (groupe Jeu) : tout le projet exporté pour le jeu (musiques, bruitages, illustrations,
cartes, modèles 3D) avec un manifest.json et une archive zip."""
import gradio as gr

from .. import export, projets
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Réunit tout le projet** dans `data/jeux/<projet>/export/` et une archive zip : la piste la plus récente "
        "de chaque situation et les bruitages gardés, au même volume (OGG pour les navigateurs récents, MP3 en "
        "secours, boucles exactes), les illustrations gardées, les cartes composées et les modèles 3D du projet "
        "(version web si elle existe), avec un seul `manifest.json`. Le même nom de projet dans les onglets du "
        "groupe Jeu réunit tout dans un seul pack."
    )
    with gr.Row():
        pk_projet = gr.Dropdown(projets.tous() or ["mon jeu de cartes"], value=(projets.tous() or ["mon jeu de cartes"])[0],
                                allow_custom_value=True, label="Projet de jeu")
        pk_cible = gr.Dropdown(list(export.CIBLES), value=list(export.CIBLES)[1], label="Volume cible")
        pk_formats = gr.CheckboxGroup([("OGG", "ogg"), ("MP3", "mp3")], value=["ogg", "mp3"], label="Formats audio")
    btn_pk = gr.Button("📦 Exporter le pack du jeu", variant="primary")
    pk_msg = gr.Markdown()
    pk_zip = gr.File(label="Archive à télécharger")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.onglet.select(lambda p: gr.update(choices=projets.tous() or [p], value=p if p in projets.tous() else (projets.tous() or [p])[0]),
                    c.pk_projet, c.pk_projet)
    c.btn_pk.click(export.exporter_pack, [c.pk_projet, c.pk_cible, c.pk_formats], [c.pk_zip, c.pk_msg])
