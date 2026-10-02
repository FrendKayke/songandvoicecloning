"""Onglet « Photos » : améliorer la qualité, détourer, isoler une personne."""
import gradio as gr

from .. import photos
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Améliore tes photos ou détoure-les, en local. **Améliorer** : Real-ESRGAN agrandit ×2 ou ×4 et nettoie "
        "le bruit, les artefacts JPEG et le flou (×1 nettoie sans agrandir) ; GFPGAN redonne des visages nets. "
        "**Détourer** : BiRefNet garde le sujet (même les cheveux fins) sur un fond transparent, une couleur ou "
        "son propre fond flouté. **Isoler une personne** : le même outil, entraîné sur des personnes. "
        "Licences libres (usage commercial permis). Pour enchaîner (améliorer puis détourer), clique sur "
        "« Continuer avec ce résultat ». ACE-Step est arrêté automatiquement pendant le traitement."
    )
    with gr.Row():
        with gr.Column():
            ph_image = gr.Image(type="filepath", label="Photo (PNG, JPG, WebP…)", height=420)
            ph_nom = gr.Textbox(label="Nom (facultatif, pour la galerie)")
        with gr.Column():
            ph_action = gr.Radio(list(photos.ACTIONS), value=photos.ACTION_DEFAUT, label="Traitement")
            with gr.Group() as ph_reglages_qualite:
                ph_echelle = gr.Radio(list(photos.ECHELLES), value="×2", label="Agrandissement")
                ph_rapide = gr.Checkbox(value=False, label="Mode rapide (modèle léger, un peu moins fin)")
                ph_visages = gr.Checkbox(value=True, label="Restaurer les visages (GFPGAN)")
                ph_force = gr.Slider(0.3, 1.0, value=0.7, step=0.05, label="Force sur les visages",
                                     info="Plus bas : garde davantage le grain d'origine, visage moins lissé.")
            with gr.Group(visible=False) as ph_reglages_fond:
                ph_fond = gr.Radio(list(photos.FONDS), value=photos.FOND_DEFAUT, label="Fond")
                ph_couleur = gr.ColorPicker(value="#3a6ea5", label="Couleur du fond", visible=False)
            btn_ph = gr.Button("🖼️ Traiter la photo", variant="primary")
    ph_statut = gr.Markdown()
    ph_dossier = gr.State()
    ph_comparaison = gr.ImageSlider(label="Avant / après (fais glisser le curseur)", type="filepath", height=560)
    with gr.Row():
        ph_fichiers = gr.File(label="Fichiers", file_count="multiple", interactive=False)
        with gr.Column():
            btn_ph_continuer = gr.Button("🔁 Continuer avec ce résultat")
            btn_ph_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.ph_action.change(photos.maj_action, c.ph_action, [c.ph_reglages_qualite, c.ph_reglages_fond])
    c.ph_fond.change(photos.maj_fond, c.ph_fond, c.ph_couleur)
    c.btn_ph.click(photos.traiter,
                   [c.ph_image, c.ph_action, c.ph_echelle, c.ph_rapide, c.ph_visages, c.ph_force, c.ph_fond,
                    c.ph_couleur, c.ph_nom],
                   [c.ph_statut, c.ph_comparaison, c.ph_fichiers, c.ph_dossier])
    c.btn_ph_continuer.click(photos.continuer, c.ph_dossier, c.ph_image)
    c.btn_ph_dossier.click(lambda d: open_folder(d) if d else None, c.ph_dossier)
