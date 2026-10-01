"""Onglet « Galerie » : toutes les créations, écoute ou affichage, recréation, retouche d'un passage, suppression."""
import gradio as gr

from .. import galerie
from ..outils import open_folder
from .commun import CONFIRMER_SUPPRESSION_CREATION, espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    with gr.Row():
        gal_filtre = gr.Radio(list(galerie.FILTRES), value="Tout", label="Afficher")
        gal_maj = gr.Button("🔄 Actualiser", scale=0)
    gal_liste = gr.Dropdown([], label="Création (la plus récente en premier)")
    gal_etat = gr.State()
    with gr.Row():
        with gr.Column(scale=3):
            gal_details = gr.Markdown()
        with gr.Column(scale=2):
            gal_version = gr.Radio([1], value=1, label="Version", visible=False)
            gal_audio = gr.Audio(type="filepath", label="Écouter")
            gal_modele = gr.Model3D(label="Modèle 3D", visible=False, clear_color=(0.92, 0.92, 0.92, 1.0))
            gal_image = gr.Image(label="Illustration", visible=False, interactive=False, type="filepath")
            with gr.Row():
                gal_recreer = gr.Button("🔁 Recréer (même graine)")
                gal_dossier = gr.Button("📂 Ouvrir le dossier")
                gal_suppr = gr.Button("🗑️ Supprimer", variant="stop")
    gal_msg = gr.Markdown()
    with gr.Accordion("✏️ Refaire un passage (chansons et pistes de jeu)", open=False):
        gr.Markdown(
            "Un refrain raté, une fin bizarre ? Indique le passage en secondes : seul ce passage est "
            "réinventé (tâche *repaint* d'ACE-Step), le reste est gardé, puis la suite du traitement est "
            "refaite (ta voix, retrait d'instruments, boucle…). Le résultat est une **nouvelle création**, "
            "l'originale reste dans la galerie. Tu peux modifier la description ou les paroles du passage."
        )
        with gr.Row():
            gal_debut = gr.Number(value=0, label="Début (s)")
            gal_fin = gr.Number(value=10, label="Fin (s)")
            gal_force = gr.Radio(list(galerie.FORCES), value=list(galerie.FORCES)[1], label="Retouche")
        gal_desc = gr.Textbox(label="Description", lines=2)
        gal_paroles = gr.Textbox(label="Paroles (chansons)", lines=6)
        gal_refaire = gr.Button("✏️ Refaire ce passage", variant="primary")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    sorties_details = [c.gal_details, c.gal_audio, c.gal_version, c.gal_desc, c.gal_paroles, c.gal_fin, c.gal_modele, c.gal_image]
    for evt in (c.gal_filtre.change, c.gal_maj.click, demo.load):
        evt(galerie.maj_liste, [c.gal_filtre, c.gal_liste], c.gal_liste)
    c.gal_liste.change(galerie.details, [c.gal_liste], sorties_details)
    # La visionneuse 3D (Babylon.js) ne s'initialise pas si sa valeur arrive pendant que l'onglet est caché
    # (chargement de la page) et ignore une valeur identique : on la vide puis on réaffiche la création.
    c.onglet.select(lambda: None, None, c.gal_modele).then(galerie.details, [c.gal_liste, c.gal_version], sorties_details)
    c.gal_version.input(galerie.details, [c.gal_liste, c.gal_version], sorties_details)
    c.gal_recreer.click(galerie.recreer, [c.gal_liste, c.gal_version], [c.gal_msg, c.gal_etat]).then(
        galerie.maj_liste, [c.gal_filtre, c.gal_etat], c.gal_liste)
    c.gal_refaire.click(galerie.refaire_passage,
                      [c.gal_liste, c.gal_version, c.gal_debut, c.gal_fin, c.gal_desc, c.gal_paroles, c.gal_force],
                      [c.gal_msg, c.gal_etat]).then(galerie.maj_liste, [c.gal_filtre, c.gal_etat], c.gal_liste)
    c.gal_suppr.click(galerie.supprimer, c.gal_liste, c.gal_msg, js=CONFIRMER_SUPPRESSION_CREATION).then(
        galerie.maj_liste, [c.gal_filtre], c.gal_liste)
    c.gal_dossier.click(lambda d: open_folder(d) if d else None, c.gal_liste)
