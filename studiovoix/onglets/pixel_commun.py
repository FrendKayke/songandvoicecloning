"""Éléments partagés par les onglets de pixel art : le bloc « style des sprites d'exemple » (grille, hauteur, couleurs,
contour lus sur des exemples) et les sorties (statut, galerie, fichiers)."""
import gradio as gr

from .. import pixel_art


def bloc_style(ouvert=True):
    """Volet « Style : sprites d'exemple » ; renvoie ses composants dans l'ordre attendu par pixel_art.convertir /
    creer_personnages : (exemples, hauteur, couleurs, toile, agrandissement, contour), plus (bouton, message)."""
    with gr.Accordion("🎨 Style : sprites d'exemple (taille, hauteur, couleurs, contour)", open=ouvert):
        with gr.Row():
            exemples = gr.File(file_count="multiple", file_types=["image"],
                               label="Sprites d'exemple (PNG, ou GIF animés : leur rythme sert aux animations)")
            with gr.Column():
                btn_style = gr.Button("🔍 Lire le style des exemples")
                style_msg = gr.Markdown()
        with gr.Row():
            hauteur = gr.Number(value=0, precision=0, label="Hauteur du personnage (px, 0 = d'après les exemples)")
            couleurs = gr.Number(value=0, precision=0, label="Nombre de couleurs (0 = d'après les exemples)")
            toile = gr.Number(value=0, precision=0, label="Taille du sprite (px, 0 = d'après les exemples)")
            agrandi = gr.Number(value=0, precision=0, label="Agrandissement à l'export (0 = d'après les exemples)")
            contour = gr.Checkbox(value=True, label="Contour sombre d'un pixel")
    return [exemples, hauteur, couleurs, toile, agrandi, contour], btn_style, style_msg


def brancher_style(reglages, btn_style, style_msg):
    exemples, hauteur, couleurs, toile, agrandi, _ = reglages
    btn_style.click(pixel_art.lire_style, exemples, [hauteur, couleurs, toile, agrandi, style_msg])


def sorties():
    """Statut, galerie (clic = sprite à animer), fichiers, dossier, aperçus (état)."""
    statut = gr.Markdown()
    dossier = gr.State()
    apercus = gr.State([])
    galerie = gr.Gallery(label="Sprites (clique sur un sprite : il est envoyé dans « Animer un sprite »)", columns=6,
                         height=300, object_fit="contain")
    fichiers = gr.File(label="Fichiers produits", file_count="multiple", interactive=False)
    btn_dossier = gr.Button("📂 Ouvrir le dossier")
    return statut, galerie, fichiers, dossier, apercus, btn_dossier


def choisir(apercus, evt: gr.SelectData):
    """Sprite cliqué dans la galerie → à animer (onglet « Animer un sprite »)."""
    return apercus[evt.index] if apercus and evt.index < len(apercus) else None
