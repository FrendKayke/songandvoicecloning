"""Onglet « Photos en lot » : le même traitement (améliorer, détourer, isoler une personne, garder la tête) sur
jusqu'à 200 photos d'un coup."""
import gradio as gr

from .. import photos
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Le même traitement sur plusieurs photos d'un coup** (jusqu'à 200) : améliorer la qualité, détourer, "
        "isoler une personne, garder seulement la tête sur fond transparent. Une photo en échec n'arrête pas les "
        "autres : elle est signalée dans le message. Résultats dans un dossier et une archive .zip. Pour une seule "
        "photo avec un avant / après, utilise « Retoucher une photo »."
    )
    with gr.Row():
        ph_lot = gr.File(file_count="multiple", file_types=["image"], label="Photos (jusqu'à 200)")
        with gr.Column():
            ph_action = gr.Radio(list(photos.ACTIONS), value=photos.ACTION_DEFAUT, label="Traitement")
            with gr.Group() as ph_reglages_qualite:
                ph_echelle = gr.Radio(list(photos.ECHELLES), value="×2", label="Agrandissement")
                ph_rapide = gr.Checkbox(value=False, label="Mode rapide (modèle léger, un peu moins fin)")
                ph_visages = gr.Checkbox(value=True, label="Restaurer les visages (GFPGAN)")
                ph_force = gr.Slider(0.3, 1.0, value=0.7, step=0.05, label="Force sur les visages")
            with gr.Group(visible=False) as ph_reglages_fond:
                ph_fond = gr.Radio(list(photos.FONDS), value=photos.FOND_DEFAUT, label="Fond")
                ph_couleur = gr.ColorPicker(value="#3a6ea5", label="Couleur du fond", visible=False)
            ph_lot_nom = gr.Textbox(label="Nom du lot", value="photos")
            btn_ph_lot = gr.Button("📦 Traiter toutes les photos", variant="primary")
    ph_lot_statut = gr.Markdown()
    ph_dossier = gr.State()
    ph_lot_galerie = gr.Gallery(label="Résultats", columns=6, height=320, object_fit="contain")
    ph_lot_fichiers = gr.File(label="Archive .zip et images", file_count="multiple", interactive=False)
    btn_ph_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.ph_action.change(photos.maj_action, c.ph_action, [c.ph_reglages_qualite, c.ph_reglages_fond])
    c.ph_fond.change(photos.maj_fond, c.ph_fond, c.ph_couleur)
    c.btn_ph_lot.click(photos.traiter_lot,
                       [c.ph_lot, c.ph_action, c.ph_echelle, c.ph_rapide, c.ph_visages, c.ph_force, c.ph_fond,
                        c.ph_couleur, c.ph_lot_nom],
                       [c.ph_lot_statut, c.ph_lot_galerie, c.ph_lot_fichiers, c.ph_dossier])
    c.btn_ph_dossier.click(lambda d: open_folder(d) if d else None, c.ph_dossier)
