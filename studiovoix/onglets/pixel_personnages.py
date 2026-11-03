"""Onglet « Personnages pixel art d'après une photo » : les personnes d'une photo deviennent des sprites de jeu
(Qwen3-VL décrit, FLUX.2 klein dessine, conversion au style des exemples)."""
import gradio as gr

from .. import pixel_art
from ..outils import open_folder
from .commun import espace_de_noms
from .pixel_commun import bloc_style, brancher_style, choisir, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Transforme les personnes d'une photo en personnages de jeu.** Dépose une photo (même de groupe) et écris "
        "une ligne par personnage : qui, et ce que tu changes. Qwen3-VL repère la personne et la décrit, FLUX.2 "
        "klein la dessine en pied dans un style de jeu rétro, puis elle est convertie au style de tes sprites "
        "d'exemple. Clique sur un sprite pour l'animer."
    )
    px_reglages, btn_px_style, px_style_msg = bloc_style()
    with gr.Row():
        px_photo = gr.Image(type="filepath", label="Photo des personnes", height=300)
        with gr.Column():
            px_personnes = gr.Textbox(
                label="Un personnage par ligne : qui, et ce que tu changes", lines=5,
                placeholder="l'homme barbu à lunettes, en dresseur avec un sweat noir et rouge, une balle à la main\n"
                            "la femme aux cheveux bouclés, en robe blanche avec un chapeau de paille et une valise")
            with gr.Row():
                px_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
                px_nom_perso = gr.Textbox(label="Nom du dossier", value="personnages")
            btn_px_perso = gr.Button("🧙 Créer les personnages", variant="primary")
    px_statut, px_galerie, px_fichiers, px_dossier, px_apercus, btn_px_dossier = sorties()
    return espace_de_noms(locals())


def _personnages(*args, progress=gr.Progress()):
    msg, apercus, fichiers, dossier, sprite = pixel_art.creer_personnages(*args, progress=progress)
    return msg, apercus, fichiers, dossier, apercus, sprite


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    brancher_style(c.px_reglages, c.btn_px_style, c.px_style_msg)
    c.btn_px_perso.click(_personnages, [c.px_photo, c.px_personnes, *c.px_reglages, c.px_graine, c.px_nom_perso],
                         [c.px_statut, c.px_galerie, c.px_fichiers, c.px_dossier, c.px_apercus,
                          o.pixel_animer.px_sprite])
    c.px_galerie.select(choisir, c.px_apercus, o.pixel_animer.px_sprite)
    c.btn_px_dossier.click(lambda d: open_folder(d) if d else None, c.px_dossier)
