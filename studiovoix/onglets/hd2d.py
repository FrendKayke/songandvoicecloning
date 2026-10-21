"""Onglet « Passer en HD-2D » : sprites, planches d'animation, tuiles et effets en pixel art → cartes de normales,
relief, émission, aperçu éclairé et objets 3D pour un décor HD-2D."""
import gradio as gr

from .. import hd2d
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Passe tes **sprites, planches d'animation, tuiles et effets** en pixel art au style **HD-2D** : le pixel art "
        "reste tel quel, mais il devient **éclairable en 3D** (lumières dynamiques, ombres, lueurs) dans Unity, Godot "
        "ou Unreal. Pour chaque fichier : carte de **normales** et de **relief**, **lumière émise** pour les effets, "
        "un **aperçu** avec une lumière qui tourne, et un **objet 3D** (panneau pour un personnage, bloc pour une "
        "tuile). Les planches gardent leur disposition. Aucun modèle d'IA : c'est instantané."
    )
    with gr.Row():
        hd_fichiers = gr.File(file_count="multiple", file_types=["image"], label="Sprites, planches, tuiles, effets")
        with gr.Column():
            hd_type = gr.Dropdown(hd2d.TYPES, value=hd2d.AUTO, label="Type")
            with gr.Row():
                hd_colonnes = gr.Number(value=0, precision=0, label="Planche : colonnes (0 = une seule image)")
                hd_lignes = gr.Number(value=0, precision=0, label="Planche : lignes")
            hd_nom = gr.Textbox(label="Nom du dossier", value="hd2d")
    with gr.Accordion("Réglages", open=False):
        with gr.Row():
            hd_force = gr.Slider(0.5, 6, value=2, step=0.5, label="Force du relief",
                                 info="Plus haut : la lumière marque davantage les formes.")
            hd_arrondi = gr.Slider(1, 12, value=4, step=1, label="Arrondi des bords (pixels)",
                                   info="Personnages : épaisseur du bombé depuis le contour.")
            hd_convention = gr.Radio(hd2d.CONVENTIONS, value=hd2d.OPENGL, label="Carte de normales")
            hd_echelle = gr.Number(value=32, precision=0, label="Pixels par mètre (objets 3D)",
                                   info="32 : un personnage de 64 px mesure 2 m.")
    btn_hd = gr.Button("🏔️ Passer en HD-2D", variant="primary")
    hd_statut = gr.Markdown()
    hd_dossier = gr.State()
    hd_apercus = gr.Gallery(label="Aperçus éclairés (lumière qui tourne)", columns=4, height=320, object_fit="contain")
    hd_produits = gr.File(label="Fichiers produits", file_count="multiple", interactive=False)
    btn_hd_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_hd.click(hd2d.convertir, [c.hd_fichiers, c.hd_type, c.hd_colonnes, c.hd_lignes, c.hd_force, c.hd_arrondi,
                                    c.hd_convention, c.hd_echelle, c.hd_nom],
                   [c.hd_statut, c.hd_apercus, c.hd_produits, c.hd_dossier])
    c.btn_hd_dossier.click(lambda d: open_folder(d) if d else None, c.hd_dossier)
