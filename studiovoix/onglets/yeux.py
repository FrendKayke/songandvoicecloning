"""Onglet « Refaire les yeux » (groupe Pixel art) : repasse de SDXL sur une zone peinte au pinceau d'un sprite."""
import gradio as gr

from .. import dresseur, yeux
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Refais les yeux (ou un autre petit détail) d'un sprite sans toucher au reste** : choisis le pinceau (🖌️, "
        "à gauche de la grande image), peins sur les yeux ratés, puis lance. Le modèle de sprites redessine seulement cette zone, à "
        "l'échelle de ses pixels, et **seuls les pixels peints peuvent changer**, avec les couleurs du sprite. "
        "Plusieurs variantes sont proposées : garde celle qui te plaît (fichiers `…_yeux_<n>.png`). Le sprite de "
        "face d'un kit de dresseur arrive ici tout seul."
    )
    with gr.Row():
        with gr.Column(scale=1):
            ye_sprite = gr.Image(type="filepath", image_mode="RGBA", label="Sprite à corriger (PNG)", height=220)
            ye_modele = gr.Dropdown(dresseur.choix_loras_face(), value=dresseur.choix_loras_face()[-1],
                                    label="Modèle du sprite de face (le même que pour le kit)")
            ye_description = gr.Textbox(label="Description du personnage (anglais, facultatif : reprise du kit)",
                                        lines=2)
            with gr.Row():
                ye_variantes = gr.Slider(1, 8, value=yeux.VARIANTES_DEFAUT, step=1, label="Variantes")
                ye_force = gr.Slider(0.5, 1.0, value=yeux.FORCE_DEFAUT, step=0.05,
                                     label="Force (plus = redessine plus librement)")
            ye_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
        with gr.Column(scale=2):
            ye_editeur = gr.ImageEditor(
                type="numpy", label="Peins sur les yeux à refaire (pinceau rouge ; gomme pour corriger)",
                sources=(), transforms=(), height=560,
                brush=gr.Brush(default_size=yeux.AFFICHAGE, colors=["#ff2020"], color_mode="fixed"))
    btn_ye = gr.Button("👁️ Refaire la zone peinte", variant="primary")
    ye_statut = gr.Markdown()
    ye_dossier = gr.State()
    ye_galerie = gr.Gallery(label="Avant / après", columns=4, height=320, object_fit="contain")
    ye_fichiers = gr.File(label="Sprites corrigés", file_count="multiple", interactive=False)
    ye_btn_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.ye_sprite.change(yeux.pour_editeur, c.ye_sprite, c.ye_editeur)
    c.onglet.select(dresseur.liste_loras_face, None, c.ye_modele)
    c.btn_ye.click(yeux.retoucher, [c.ye_sprite, c.ye_editeur, c.ye_description, c.ye_modele, c.ye_variantes,
                                    c.ye_force, c.ye_graine],
                   [c.ye_statut, c.ye_galerie, c.ye_fichiers, c.ye_dossier])
    c.ye_btn_dossier.click(lambda d: open_folder(d) if d else None, c.ye_dossier)
