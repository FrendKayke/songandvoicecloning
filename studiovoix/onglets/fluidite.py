"""Onglet « Fluidifier ou ralentir une vidéo » (groupe Image et vidéo) : RIFE."""
import gradio as gr

from .. import fluidite
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Rends une vidéo plus fluide ou fais-en un ralenti** : RIFE invente les images intermédiaires entre deux "
        "images voisines. « Plus fluide » double (ou triple, quadruple) le nombre d'images par seconde sans changer "
        "la durée : les vidéos générées (24 images/s) deviennent bien plus douces. « Ralenti » garde la cadence et "
        "allonge la vidéo. Quelques secondes à une minute pour une vidéo de 5 s sur la RTX 4070. Les vidéos créées "
        "dans « Une vidéo » ont aussi leur bouton « Plus fluide »."
    )
    with gr.Row():
        fl_video = gr.Video(label="Vidéo", height=320)
        with gr.Column():
            fl_mode = gr.Radio(fluidite.MODES, value=fluidite.FLUIDE, label="Effet")
            fl_facteur = gr.Radio(list(fluidite.FACTEURS), value=fluidite.FACTEUR_DEFAUT, label="Facteur")
            btn_fl = gr.Button("⏩ Transformer", variant="primary")
    fl_statut = gr.Markdown()
    fl_resultat = gr.Video(label="Résultat", height=420, interactive=False)
    fl_dossier = gr.State()
    btn_fl_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_fl.click(fluidite.fluidifier, [c.fl_video, c.fl_facteur, c.fl_mode],
                   [c.fl_statut, c.fl_resultat, c.fl_dossier])
    c.btn_fl_dossier.click(lambda d: open_folder(d) if d else None, c.fl_dossier)
