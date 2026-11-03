"""Onglet « Modèles 3D en lot » : plusieurs images à la suite, un modèle 3D par image, mêmes réglages."""
import gradio as gr

from .. import modele3d, projets
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Plusieurs modèles 3D à la suite** : chaque image donne son propre modèle (nom = préfixe + nom du "
        "fichier), rangé dans la Galerie. Une image qui échoue n'arrête pas les suivantes. Compte une à quelques "
        "minutes par modèle selon la qualité et la texture. Pour une liste d'objets décrits par du texte, va dans "
        "« Kit d'objets 3D »."
    )
    with gr.Row():
        lot_images = gr.File(file_count="multiple", file_types=["image"], label="Images des objets")
        with gr.Column():
            lot_prefixe = gr.Textbox(label="Préfixe des noms", value="carte")
            lot_projet = gr.Dropdown(projets.tous(), value=None, allow_custom_value=True,
                                     label="Projet de jeu (pour le pack, facultatif)")
            lot_qualite = gr.Radio(list(modele3d.QUALITES), value=modele3d.QUALITE_DEFAUT, label="Qualité")
            lot_texture = gr.Checkbox(value=True, label="Peindre la texture (plusieurs minutes de plus par modèle)")
            lot_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire pour chaque modèle)")
            lot_formats = gr.CheckboxGroup(modele3d.FORMATS, value=["glb"], label="Formats")
    btn_lot = gr.Button("🧊 Créer tous les modèles", variant="primary")
    lot_statut = gr.Markdown()
    lot_dossier = gr.State()
    lot_glb = gr.Model3D(label="Dernier modèle créé", clear_color=(0.92, 0.92, 0.92, 1.0))
    btn_lot_dossier = gr.Button("📂 Ouvrir le dossier du dernier modèle")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_lot.click(modele3d.generer_lot, [c.lot_images, c.lot_prefixe, c.lot_qualite, c.lot_texture, c.lot_graine,
                                           c.lot_formats, c.lot_projet],
                    [c.lot_statut, c.lot_glb, c.lot_dossier])
    c.btn_lot_dossier.click(lambda d: open_folder(d) if d else None, c.lot_dossier)
