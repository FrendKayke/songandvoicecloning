"""Onglet « Illustrations de cartes » : style mémorisé par projet, variantes Z-Image (ou FLUX.2 klein avec une photo
modèle ou un personnage récurrent). Les personnages récurrents et la composition de la carte ont leurs onglets."""
import gradio as gr

from .. import cartes, personnages, projets
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
            ill_texte = gr.Textbox(label="Scène en français (facultatif : « Préparer le prompt » la traduit)", lines=3,
                                   placeholder="un chevalier en armure dorée qui brandit une épée lumineuse")
            ill_exemples = gr.Dropdown([(lib, txt) for lib, txt in cartes.EXEMPLES], label="Exemples", value=None,
                                       allow_custom_value=True)
        with gr.Column():
            btn_ill_prep = gr.Button("🧠 Préparer le prompt (traduction et précision)")
            ill_prompt = gr.Textbox(
                label="Contexte commun à toutes les images (anglais, envoyé tel quel) : personnages, décor, ambiance",
                lines=4, placeholder="A group of four men, all male: a tall bald man with a grey beard, a blond man "
                                     "with a braid... Inside an old stone tavern, warm candle light.")
    with gr.Accordion("📷 Photo modèle (facultative) : une personne, un objet ou une pose à reprendre", open=False):
        gr.Markdown("Envoie une photo : la personne ou l'objet qu'elle montre devient le sujet de la carte, redessiné "
                    "dans le style du projet (même visage, mêmes traits), ou bien la carte reprend sa pose et son "
                    "cadrage. Décris quand même la scène ci-dessus (décor, action). La carte est alors peinte par "
                    "FLUX.2 klein 4B. Avec un personnage choisi plus bas, la photo ne sert qu'à la pose. "
                    "N'utilise que des photos que tu as le droit d'utiliser (toi, tes objets, des modèles d'accord).")
        with gr.Row():
            ill_photo = gr.Image(type="filepath", label="Photo modèle", height=300)
            ill_usage_photo = gr.Radio(list(cartes.USAGES_PHOTO), value=cartes.USAGE_PHOTO_DEFAUT,
                                       label="Ce que la carte reprend de la photo")
    with gr.Row():
        ill_nom = gr.Textbox(label="Nom de la carte", value="carte")
        ill_variantes = gr.Radio([1, 2, 3, 4], value=2, label="Images (variantes)")
        ill_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
        ill_webp = gr.Checkbox(value=True, label="Aussi en WebP (léger pour le web)")
    with gr.Row():
        ill_personnage = gr.Dropdown([personnages.AUCUN], value=personnages.AUCUN, scale=2,
                                     label="Personnage de la scène (même visage et même tenue d'une carte à l'autre)")
        ill_perso_info = gr.Markdown("Avec un personnage, la scène est peinte par FLUX.2 klein 4B à partir de ses "
                                     "images de référence (onglet « 👤 Personnages récurrents »).")
    gr.Markdown("**Ce que montre chaque image** (anglais, facultatif) : ajouté au contexte ci-dessus. Vide = le "
                "contexte seul (variantes de la même scène). Donne le nombre et le genre des personnages "
                "(« four men »), le modèle n'invente alors personne ; évite les négations (« no women ») : "
                "elles font souvent apparaître ce qu'elles excluent.")
    with gr.Row():
        ill_image_1 = gr.Textbox(label="Image 1", lines=2, placeholder="they study an old map at the table")
        ill_image_2 = gr.Textbox(label="Image 2", lines=2, placeholder="they walk out of the tavern at dawn")
        ill_image_3 = gr.Textbox(label="Image 3", lines=2, visible=False)
        ill_image_4 = gr.Textbox(label="Image 4", lines=2, visible=False)
    btn_ill = gr.Button("🎨 Générer l'illustration", variant="primary")
    ill_statut = gr.Markdown()
    ill_dossier = gr.State()
    ill_galerie = gr.Gallery(label="Variantes", columns=4, height=460, object_fit="contain")
    ill_choix_msg = gr.Markdown("Clique sur une variante pour la garder (pack du jeu) : elle est envoyée dans "
                                "« 🃏 Composer une carte » et peut rejoindre un personnage récurrent.")
    ill_choisie = gr.State()
    btn_ill_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    # Illustrations de cartes
    c.ill_projet.change(cartes.charger_style, c.ill_projet, [c.ill_styles, c.ill_consignes, c.ill_format])
    c.ill_projet.change(personnages.choix, c.ill_projet, c.ill_personnage)
    demo.load(personnages.choix, c.ill_projet, c.ill_personnage)
    c.ill_exemples.change(lambda v: v or "", c.ill_exemples, c.ill_texte)
    c.btn_ill_prep.click(cartes.preparer, c.ill_texte, c.ill_prompt)
    images = [c.ill_image_1, c.ill_image_2, c.ill_image_3, c.ill_image_4]
    c.ill_variantes.change(cartes.maj_images, c.ill_variantes, images)
    c.btn_ill.click(cartes.prompt_pret, [c.ill_texte, c.ill_prompt, *images], c.ill_prompt).success(cartes.generer,
                  [c.ill_projet, c.ill_nom, c.ill_prompt, c.ill_styles, c.ill_consignes, c.ill_format, c.ill_variantes, c.ill_graine,
                   c.ill_webp, c.ill_texte, c.ill_personnage, c.ill_photo, c.ill_usage_photo, *images],
                  [c.ill_statut, c.ill_galerie, c.ill_dossier, c.ill_projet])
    c.btn_ill_dossier.click(lambda d: open_folder(d) if d else None, c.ill_dossier)
    c.ill_galerie.select(cartes.choisir, c.ill_dossier, [c.ill_choix_msg, c.ill_choisie]).then(
        lambda i: i, c.ill_choisie, o.composer.cc_illustration).then(
        lambda nom, actuel: actuel or nom, [c.ill_nom, o.composer.cc_nom], o.composer.cc_nom)
