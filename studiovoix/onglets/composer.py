"""Onglet « Composer une carte » (groupe Jeu) : cadre, nom, coût, texte, attaque, défense autour d'une illustration
(Pillow, polices libres), PNG 300 ppp et WebP."""
import gradio as gr

from .. import compo_cartes, projets
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**La carte est dessinée autour de l'illustration**, avec des polices libres (usage commercial permis), en "
        "PNG 300 ppp prêt à imprimer et en WebP pour le web. Le texte rapetisse tout seul s'il est long. "
        "L'illustration : celle cliquée dans « Illustrations de cartes », ou une image importée. Les cartes "
        "composées rejoignent le pack du jeu du projet."
    )
    with gr.Row():
        with gr.Column():
            cc_projet = gr.Dropdown(projets.tous() or ["mon-jeu"], value=(projets.tous() or ["mon-jeu"])[0],
                                    allow_custom_value=True, label="Projet de jeu")
            cc_illustration = gr.Image(type="filepath",
                                       label="Illustration (variante cliquée dans « Illustrations de cartes », ou importée)")
        with gr.Column():
            with gr.Row():
                cc_nom = gr.Textbox(label="Nom de la carte")
                cc_cout = gr.Textbox(label="Coût", placeholder="3")
            cc_type = gr.Textbox(label="Type", placeholder="Créature — Dragon")
            cc_effet = gr.Textbox(label="Texte d'effet", lines=4,
                                  placeholder="Vol. Quand cette créature arrive en jeu, inflige 2 dégâts.")
            cc_ambiance = gr.Textbox(label="Texte d'ambiance (italique, facultatif)", lines=2)
            with gr.Row():
                cc_attaque = gr.Textbox(label="Attaque", placeholder="vide = aucune")
                cc_defense = gr.Textbox(label="Défense", placeholder="vide = aucune")
            with gr.Row():
                cc_faction = gr.Dropdown(list(compo_cartes.FACTIONS), value="Feu", label="Faction (couleur du cadre)")
                cc_rarete = gr.Dropdown(list(compo_cartes.RARETES), value="Commune", label="Rareté")
            cc_pied = gr.Textbox(label="Pied de carte (facultatif)", placeholder="© 2026 Mon jeu · 001/120")
            with gr.Row():
                cc_format = gr.Dropdown(list(compo_cartes.FORMATS), value=compo_cartes.FORMAT_DEFAUT, label="Format")
                cc_webp = gr.Checkbox(value=True, label="Aussi en WebP")
    btn_composer = gr.Button("🃏 Composer la carte", variant="primary")
    cc_msg = gr.Markdown()
    with gr.Row():
        cc_apercu = gr.Image(label="Carte composée", interactive=False, height=520)
        cc_fichiers = gr.File(label="Fichiers", file_count="multiple", interactive=False)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    o.illustrations.ill_projet.change(lambda p: gr.update(value=p), o.illustrations.ill_projet, c.cc_projet)
    c.btn_composer.click(compo_cartes.composer_et_enregistrer,
                         [c.cc_projet, c.cc_illustration, c.cc_nom, c.cc_cout, c.cc_type, c.cc_effet, c.cc_ambiance,
                          c.cc_attaque, c.cc_defense, c.cc_faction, c.cc_rarete, c.cc_pied, c.cc_format, c.cc_webp],
                         [c.cc_msg, c.cc_apercu, c.cc_fichiers])
