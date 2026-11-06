"""Onglet « Convertir en pixel art » : n'importe quelle image → sprite au style de tes exemples."""
import gradio as gr

from .. import pixel_art
from ..outils import open_folder
from .commun import espace_de_noms
from .pixel_commun import bloc_style, brancher_style, choisir, sorties


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Convertis des images en pixel art.** Donne quelques sprites d'exemple : leur style (taille de la grille, "
        "hauteur du personnage, nombre de couleurs, contour) est lu et appliqué. Le fond est retiré (BiRefNet), "
        "les couleurs sont franches, la transparence nette. Tu obtiens chaque sprite à la taille du jeu et agrandi."
    )
    px_reglages, btn_px_style, px_style_msg = bloc_style()
    with gr.Row():
        px_images = gr.File(file_count="multiple", file_types=["image"], label="Images à convertir")
        with gr.Column():
            px_detourer = gr.Checkbox(value=True, label="Retirer le fond (BiRefNet) avant la conversion",
                                      info="Sinon, seul un fond uni est retiré.")
            px_deja_pixel = gr.Checkbox(
                value=False, label="Image déjà en pixel art, faite ailleurs (retrouver ses vrais pixels)",
                info="ChatGPT, Gemini, Retro Diffusion, sprite agrandi d'un facteur non entier ou flou : la grille de "
                     "ses « pixels » est retrouvée (Pixel Art Fixer), un vrai pixel par case, puis gardée telle quelle "
                     "si le personnage tient dans la taille du jeu.")
            px_nom = gr.Textbox(label="Nom du dossier", value="pixel_art")
            btn_px_convertir = gr.Button("👾 Convertir", variant="primary")
    px_statut, px_galerie, px_fichiers, px_dossier, px_apercus, btn_px_dossier = sorties()
    return espace_de_noms(locals())


def _convertir(*args, progress=gr.Progress()):
    msg, apercus, fichiers, dossier = pixel_art.convertir(*args, progress=progress)
    return msg, apercus, fichiers, dossier, apercus


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    brancher_style(c.px_reglages, c.btn_px_style, c.px_style_msg)
    c.btn_px_convertir.click(_convertir, [c.px_images, *c.px_reglages, c.px_detourer, c.px_nom, c.px_deja_pixel],
                             [c.px_statut, c.px_galerie, c.px_fichiers, c.px_dossier, c.px_apercus])
    c.px_galerie.select(choisir, c.px_apercus, o.pixel_animer.px_sprite)
    c.btn_px_dossier.click(lambda d: open_folder(d) if d else None, c.px_dossier)
