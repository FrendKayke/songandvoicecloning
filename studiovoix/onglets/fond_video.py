"""Onglet « Retirer le fond d'une vidéo » (groupe Vidéo) : BiRefNet image par image, fond transparent ou nouveau."""
import gradio as gr

from .. import fond_video
from ..outils import open_folder
from .commun import espace_de_noms
from .voix_off import maj_liste


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Retire le fond d'une vidéo, sans fond vert au tournage** : chaque image est détourée (comme une photo), "
        "puis la vidéo est réécrite avec un **fond transparent** (pour un jeu, un site ou un logiciel de montage) ou "
        "un **nouveau fond** (vert d'incrustation, couleur, image, ou le fond d'origine flouté). Le son est gardé. "
        "Ça se fait sur la carte graphique et prend du temps : coupe d'abord le passage utile (Audio / vidéo → "
        "Couper)."
    )
    with gr.Row():
        with gr.Column():
            fv_video = gr.Video(label="Vidéo (importe un fichier…)", sources=["upload"], height=280)
            with gr.Row():
                fv_creation = gr.Dropdown([], label="… ou une vidéo créée ici", scale=4)
                fv_maj = gr.Button("🔄", scale=0)
        with gr.Column():
            fv_fond = gr.Radio(fond_video.FONDS, value=fond_video.TRANSPARENT_WEBM, label="Fond voulu")
            fv_couleur = gr.ColorPicker(value="#ffffff", label="Couleur du fond", visible=False)
            fv_image = gr.Image(type="filepath", label="Image de fond (recadrée à la taille de la vidéo)",
                                height=160, visible=False)
            fv_modele = gr.Radio(list(fond_video.MODELES), value=fond_video.MODELE_DEFAUT, label="Ce qu'il faut garder")
    with gr.Accordion("Réglages (avancé)", open=False):
        with gr.Row():
            fv_lissage = gr.Slider(0, 0.8, value=0.3, step=0.05,
                                   label="Lissage dans le temps (moins de scintillement aux bords ; trop = traînées)")
            fv_masque = gr.Checkbox(value=False, label="Enregistrer aussi la vidéo du masque (noir et blanc)")
    fv_nom = gr.Textbox(label="Nom (pour la galerie)", value="sans_fond")
    btn_fv = gr.Button("✂️ Retirer le fond", variant="primary")
    fv_statut = gr.Markdown()
    fv_dossier = gr.State()
    fv_resultat = gr.Video(label="Résultat", height=420, interactive=False)
    fv_fichiers = gr.File(label="Fichiers", file_count="multiple", interactive=False)
    fv_btn_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    for evt in (c.fv_maj.click, c.onglet.select):
        evt(maj_liste, c.fv_creation, c.fv_creation)
    c.fv_fond.change(lambda f: (gr.update(visible=f == fond_video.COULEUR), gr.update(visible=f == fond_video.IMAGE)),
                     c.fv_fond, [c.fv_couleur, c.fv_image])
    c.btn_fv.click(fond_video.traiter, [c.fv_video, c.fv_creation, c.fv_fond, c.fv_couleur, c.fv_image, c.fv_modele,
                                        c.fv_lissage, c.fv_masque, c.fv_nom],
                   [c.fv_statut, c.fv_resultat, c.fv_fichiers, c.fv_dossier])
    c.fv_btn_dossier.click(lambda d: open_folder(d) if d else None, c.fv_dossier)
