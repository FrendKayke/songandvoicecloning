"""Onglet « Illustrations de cartes » : style mémorisé par projet, variantes Z-Image."""
import gradio as gr

from .. import cartes
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Des **illustrations** pour tes cartes, avec Z-Image-Turbo (licence Apache 2.0 : tu peux vendre les "
        "images). Le **style** est mémorisé par projet et réappliqué à chaque carte : seule la scène change, "
        "tes cartes restent cohérentes entre elles. Décris la scène en français, « Préparer le prompt » la "
        "traduit et la précise (Qwen3-VL) ; tu peux la retoucher. Chaque variante a sa graine : note celle "
        "que tu gardes pour la retrouver. ACE-Step est arrêté automatiquement pendant la génération."
    )
    with gr.Row():
        ill_projet = gr.Dropdown(cartes.projets() or ["mon-jeu"], value=(cartes.projets() or ["mon-jeu"])[0],
                                 allow_custom_value=True, label="Projet (style mémorisé)")
        ill_format = gr.Dropdown(list(cartes.FORMATS), value=cartes.FORMAT_DEFAUT, label="Format")
    with gr.Row():
        ill_styles = gr.Dropdown([(lib, val) for lib, val in cartes.STYLES], value=cartes.STYLES_DEFAUT,
                                 multiselect=True, allow_custom_value=True, label="Style (plusieurs choix, ou tape le tien en anglais)")
        ill_consignes = gr.Textbox(label="Consignes de style en plus (anglais)",
                                   placeholder="gold and deep blue palette, soft rim light")
    with gr.Row():
        with gr.Column():
            ill_texte = gr.Textbox(label="Scène de la carte (français ou anglais)", lines=3,
                                   placeholder="un chevalier en armure dorée qui brandit une épée lumineuse")
            ill_exemples = gr.Dropdown([(lib, txt) for lib, txt in cartes.EXEMPLES], label="Exemples", value=None,
                                       allow_custom_value=True)
        with gr.Column():
            btn_ill_prep = gr.Button("🧠 Préparer le prompt (traduction et précision)")
            ill_prompt = gr.Textbox(label="Scène envoyée à Z-Image (anglais, modifiable)", lines=4)
    with gr.Row():
        ill_nom = gr.Textbox(label="Nom de la carte", value="carte")
        ill_variantes = gr.Radio([1, 2, 3, 4], value=2, label="Variantes")
        ill_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
        ill_webp = gr.Checkbox(value=True, label="Aussi en WebP (léger pour le web)")
    btn_ill = gr.Button("🎨 Générer l'illustration", variant="primary")
    ill_statut = gr.Markdown()
    ill_dossier = gr.State()
    ill_galerie = gr.Gallery(label="Variantes", columns=4, height=460, object_fit="contain")
    btn_ill_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    # Illustrations de cartes
    c.ill_projet.change(cartes.charger_style, c.ill_projet, [c.ill_styles, c.ill_consignes, c.ill_format])
    c.ill_exemples.change(lambda v: v or "", c.ill_exemples, c.ill_texte)
    c.btn_ill_prep.click(cartes.preparer, c.ill_texte, c.ill_prompt)
    c.btn_ill.click(cartes.generer,
                  [c.ill_projet, c.ill_nom, c.ill_prompt, c.ill_styles, c.ill_consignes, c.ill_format, c.ill_variantes, c.ill_graine,
                   c.ill_webp, c.ill_texte],
                  [c.ill_statut, c.ill_galerie, c.ill_dossier, c.ill_projet])
    c.btn_ill_dossier.click(lambda d: open_folder(d) if d else None, c.ill_dossier)
