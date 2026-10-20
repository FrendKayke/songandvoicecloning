"""Onglet « Pixel art et sprites » : convertir des images en pixel art, faire des personnages de jeu d'après une
photo, animer un sprite (30 images sur 5 s, en boucle)."""
import gradio as gr

from .. import pixel_art
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Pixel art et sprites de jeu.** Donne quelques sprites d'exemple : leur style (taille de la grille, hauteur "
        "du personnage, nombre de couleurs, contour) est lu et appliqué à tout ce qui est créé ici. Puis : convertis "
        "n'importe quelle image, transforme les personnes d'une photo en personnages (avec tes modifications), et "
        "anime un sprite sur 30 images en 5 secondes, en boucle."
    )
    with gr.Accordion("🎨 Style : sprites d'exemple", open=True):
        with gr.Row():
            px_exemples = gr.File(file_count="multiple", file_types=["image"], label="Sprites d'exemple (PNG)")
            with gr.Column():
                btn_px_style = gr.Button("🔍 Lire le style des exemples")
                px_style_msg = gr.Markdown()
        with gr.Row():
            px_hauteur = gr.Number(value=0, precision=0, label="Hauteur du personnage (px, 0 = d'après les exemples)")
            px_couleurs = gr.Number(value=0, precision=0, label="Nombre de couleurs (0 = d'après les exemples)")
            px_toile = gr.Number(value=0, precision=0, label="Taille du sprite (px, 0 = d'après les exemples)")
            px_agrandi = gr.Number(value=0, precision=0, label="Agrandissement à l'export (0 = d'après les exemples)")
            px_contour = gr.Checkbox(value=True, label="Contour sombre d'un pixel")
    gr.Markdown("### 🖼️ Convertir des images en pixel art")
    with gr.Row():
        px_images = gr.File(file_count="multiple", file_types=["image"], label="Images à convertir")
        with gr.Column():
            px_detourer = gr.Checkbox(value=True, label="Retirer le fond (BiRefNet) avant la conversion",
                                      info="Sinon, seul un fond uni est retiré.")
            px_nom = gr.Textbox(label="Nom du dossier", value="pixel_art")
            btn_px_convertir = gr.Button("👾 Convertir", variant="primary")
    gr.Markdown("### 🧙 Personnages d'après une photo")
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
    px_statut = gr.Markdown()
    px_dossier = gr.State()
    px_apercus = gr.State([])
    px_galerie = gr.Gallery(label="Sprites (clique sur un sprite pour l'animer)", columns=6, height=300,
                            object_fit="contain")
    px_fichiers = gr.File(label="Fichiers produits", file_count="multiple", interactive=False)
    gr.Markdown("### 🎞️ Animer un sprite (30 images, 5 secondes, en boucle)")
    with gr.Row():
        px_sprite = gr.Image(type="filepath", image_mode="RGBA", label="Sprite à animer", height=260)
        with gr.Column():
            px_mode = gr.Radio(pixel_art.MODES_ANIMATION, value=pixel_art.SIMPLE, label="Animation")
            px_mouvement = gr.Textbox(label="Mouvement (facultatif, Wan 2.2)", lines=2,
                                      placeholder="il lance sa balle d'une main à l'autre, sa cape flotte")
            with gr.Row():
                px_anim_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
                px_anim_etapes = gr.Slider(10, 50, value=25, step=1, label="Étapes (Wan 2.2)")
                px_anim_nom = gr.Textbox(label="Nom", value="animation")
            btn_px_animer = gr.Button("🎞️ Animer", variant="primary")
    px_anim_statut = gr.Markdown()
    with gr.Row():
        px_anim = gr.Image(label="Animation (GIF)", height=260, interactive=False)
        px_anim_fichiers = gr.File(label="GIF, planche de sprites, images", file_count="multiple", interactive=False)
    btn_px_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def _choisir(apercus, evt: gr.SelectData):
    """Sprite cliqué dans la galerie → à animer."""
    return apercus[evt.index] if apercus and evt.index < len(apercus) else None


def _convertir(*args, progress=gr.Progress()):
    msg, apercus, fichiers, dossier = pixel_art.convertir(*args, progress=progress)
    return msg, apercus, fichiers, dossier, apercus


def _personnages(*args, progress=gr.Progress()):
    msg, apercus, fichiers, dossier, sprite = pixel_art.creer_personnages(*args, progress=progress)
    return msg, apercus, fichiers, dossier, apercus, sprite


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    reglages = [c.px_exemples, c.px_hauteur, c.px_couleurs, c.px_toile, c.px_agrandi, c.px_contour]
    c.btn_px_style.click(pixel_art.lire_style, c.px_exemples,
                         [c.px_hauteur, c.px_couleurs, c.px_toile, c.px_agrandi, c.px_style_msg])
    sorties = [c.px_statut, c.px_galerie, c.px_fichiers, c.px_dossier, c.px_apercus]
    c.btn_px_convertir.click(_convertir, [c.px_images, *reglages, c.px_detourer, c.px_nom], sorties)
    c.btn_px_perso.click(_personnages, [c.px_photo, c.px_personnes, *reglages, c.px_graine, c.px_nom_perso],
                         [*sorties, c.px_sprite])
    c.px_galerie.select(_choisir, c.px_apercus, c.px_sprite)
    c.btn_px_animer.click(pixel_art.animer, [c.px_sprite, c.px_mode, c.px_mouvement, c.px_anim_graine,
                                            c.px_anim_etapes, c.px_anim_nom],
                          [c.px_anim_statut, c.px_anim, c.px_anim_fichiers, c.px_dossier])
    c.btn_px_dossier.click(lambda d: open_folder(d) if d else None, c.px_dossier)
