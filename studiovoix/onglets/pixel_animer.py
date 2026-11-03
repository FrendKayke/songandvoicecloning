"""Onglet « Animer un sprite » : respiration instantanée ou mouvement libre (Wan 2.2), GIF transparent, planche."""
import gradio as gr

from .. import pixel_art
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Anime un sprite** : importe-le, ou clique sur un sprite dans « Convertir en pixel art », « Personnages "
        "d'après une photo » ou « Kit de dresseur ». **Respiration** : instantané, le haut du corps monte et descend "
        "d'un pixel. **Mouvement libre** : Wan 2.2 anime le sprite (une dizaine de minutes) ; chaque image est "
        "remise sur la grille et les couleurs du sprite. Avec une action décrite, le geste se joue une fois au "
        "rythme des sprites animés ; sans, petits mouvements sur place en boucle. Tu obtiens un GIF transparent, "
        "une planche de sprites et les images séparées."
    )
    with gr.Row():
        px_sprite = gr.Image(type="filepath", image_mode="RGBA", label="Sprite à animer", height=260)
        with gr.Column():
            px_mode = gr.Radio(pixel_art.MODES_ANIMATION, value=pixel_art.SIMPLE, label="Animation")
            px_mouvement = gr.Textbox(label="Mouvement (facultatif, Wan 2.2)", lines=2,
                                      placeholder="il lance sa balle d'une main à l'autre, sa cape flotte",
                                      info="Décrit : un geste joué une fois, au rythme des sprites animés "
                                           "(50 ms par image, pose finale tenue). Vide : petits mouvements sur "
                                           "place, en boucle. Une dizaine de minutes.")
            px_exemples = gr.File(file_count="multiple", file_types=["image"],
                                  label="Sprites animés d'exemple (GIF, facultatif) : leur rythme est repris")
            with gr.Row():
                px_anim_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
                px_anim_etapes = gr.Slider(10, 50, value=25, step=1, label="Étapes (Wan 2.2)")
                px_anim_nom = gr.Textbox(label="Nom", value="animation")
            btn_px_animer = gr.Button("🎞️ Animer", variant="primary")
    px_anim_statut = gr.Markdown()
    px_dossier = gr.State()
    with gr.Row():
        px_anim = gr.Image(label="Animation (GIF)", height=260, interactive=False)
        px_anim_fichiers = gr.File(label="GIF, planche de sprites, images", file_count="multiple", interactive=False)
    btn_px_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_px_animer.click(pixel_art.animer, [c.px_sprite, c.px_mode, c.px_mouvement, c.px_anim_graine,
                                            c.px_anim_etapes, c.px_anim_nom, c.px_exemples],
                          [c.px_anim_statut, c.px_anim, c.px_anim_fichiers, c.px_dossier])
    c.btn_px_dossier.click(lambda d: open_folder(d) if d else None, c.px_dossier)
