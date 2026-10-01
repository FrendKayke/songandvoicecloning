"""Onglet « Illustrations de cartes » : style mémorisé par projet, variantes Z-Image."""
import gradio as gr

from .. import cartes, compo_cartes, projets
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
        ill_projet = gr.Dropdown(projets.tous() or ["mon-jeu"], value=(projets.tous() or ["mon-jeu"])[0],
                                 allow_custom_value=True, label="Projet de jeu (style mémorisé, pack du jeu)")
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
    ill_choix_msg = gr.Markdown("Clique sur une variante pour la garder (pack du jeu) et la composer en carte.")
    btn_ill_dossier = gr.Button("📂 Ouvrir le dossier")
    with gr.Accordion("🃏 Composer la carte (cadre, nom, coût, texte, attaque, défense)", open=True):
        gr.Markdown("La carte est dessinée autour de l'illustration, avec des polices libres (usage commercial "
                    "permis), en PNG 300 ppp prêt à imprimer et en WebP pour le web. Le texte rapetisse tout seul "
                    "s'il est long. Les cartes composées rejoignent le pack du jeu du projet.")
        with gr.Row():
            with gr.Column():
                cc_illustration = gr.Image(type="filepath",
                                           label="Illustration (clique sur une variante ci-dessus, ou importe une image)")
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
    # Illustrations de cartes
    c.ill_projet.change(cartes.charger_style, c.ill_projet, [c.ill_styles, c.ill_consignes, c.ill_format])
    c.ill_exemples.change(lambda v: v or "", c.ill_exemples, c.ill_texte)
    c.btn_ill_prep.click(cartes.preparer, c.ill_texte, c.ill_prompt)
    c.btn_ill.click(cartes.generer,
                  [c.ill_projet, c.ill_nom, c.ill_prompt, c.ill_styles, c.ill_consignes, c.ill_format, c.ill_variantes, c.ill_graine,
                   c.ill_webp, c.ill_texte],
                  [c.ill_statut, c.ill_galerie, c.ill_dossier, c.ill_projet])
    c.btn_ill_dossier.click(lambda d: open_folder(d) if d else None, c.ill_dossier)
    c.ill_galerie.select(cartes.choisir, c.ill_dossier, [c.ill_choix_msg, c.cc_illustration]).then(
        lambda nom, actuel: actuel or nom, [c.ill_nom, c.cc_nom], c.cc_nom)
    c.btn_composer.click(compo_cartes.composer_et_enregistrer,
                         [c.ill_projet, c.cc_illustration, c.cc_nom, c.cc_cout, c.cc_type, c.cc_effet, c.cc_ambiance,
                          c.cc_attaque, c.cc_defense, c.cc_faction, c.cc_rarete, c.cc_pied, c.cc_format, c.cc_webp],
                         [c.cc_msg, c.cc_apercu, c.cc_fichiers])
