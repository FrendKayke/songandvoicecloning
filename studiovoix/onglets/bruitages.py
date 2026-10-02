"""Onglet « Bruitages » : description ou image → prompt → variantes, export, choix pour le pack."""
import gradio as gr

from .. import bruitages, export
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Effets sonores pour ton jeu, à partir d'une **description** (en français ou en anglais) ou d'une "
        "**image**. « Préparer le prompt » traduit et précise ta description en anglais (Qwen3-VL) ; tu peux "
        "le retoucher avant de générer. Stable Audio Open produit 1 à 3 variantes en 44,1 kHz. "
        "ACE-Step est arrêté automatiquement pendant la génération (mémoire graphique), puis relancé à la "
        "chanson suivante."
    )
    with gr.Row():
        with gr.Column():
            sfx_texte = gr.Textbox(label="Description du bruitage", lines=3,
                                   placeholder="une porte de château en bois qui grince puis claque")
            sfx_exemples = gr.Dropdown([(lib, txt) for lib, txt in bruitages.EXEMPLES], label="Exemples (jeu de cartes)", value=None,
                                       allow_custom_value=True,
                                       info="Choisis un exemple : il remplit le prompt anglais directement.")
        with gr.Column():
            sfx_image = gr.Image(type="filepath", label="Ou une image (facultatif) : les sons de la scène")
    btn_sfx_prep = gr.Button("🧠 Préparer le prompt (traduction et précision)")
    sfx_prompt = gr.Textbox(label="Prompt envoyé à Stable Audio (anglais, modifiable)", lines=2)
    with gr.Row():
        sfx_nom = gr.Textbox(label="Nom", value="bruitage")
        sfx_projet = gr.Dropdown(bruitages.projets_de_jeu(), value=None, allow_custom_value=True,
                                 label="Projet de jeu (pour le pack)",
                                 info="Le même nom que dans « Bande-son de jeu » : le bruitage rejoint son pack.")
        sfx_duree = gr.Slider(1, bruitages.DUREE_MAX, value=3, step=0.5, label="Durée (s)")
        sfx_variantes = gr.Radio([1, 2, 3], value=2, label="Variantes")
        sfx_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
        sfx_etapes = gr.Slider(20, 200, value=100, step=10, label="Étapes (qualité / temps)")
    btn_sfx = gr.Button("🔊 Générer le bruitage", variant="primary")
    sfx_statut = gr.Markdown()
    sfx_dossier = gr.State()
    with gr.Row():
        sfx_liste = gr.Dropdown([], label="Écouter une variante")
        sfx_audio = gr.Audio(label="Bruitage", type="filepath")
    with gr.Row():
        sfx_cible = gr.Dropdown(list(export.CIBLES), value=list(export.CIBLES)[1], label="Volume de l'export")
        sfx_formats = gr.CheckboxGroup([("OGG", "ogg"), ("MP3", "mp3"), ("WAV", "wav")], value=["ogg", "mp3"],
                                       label="Formats")
        btn_sfx_export = gr.Button("📦 Exporter la variante écoutée")
        btn_sfx_choisir = gr.Button("⭐ Garder cette variante pour le pack du jeu")
    sfx_export_msg = gr.Markdown()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    # Bruitages
    c.sfx_exemples.change(lambda v: v or "", c.sfx_exemples, c.sfx_prompt)
    c.btn_sfx_prep.click(bruitages.preparer, [c.sfx_texte, c.sfx_image], c.sfx_prompt)
    c.btn_sfx.click(bruitages.prompt_pret, [c.sfx_texte, c.sfx_image, c.sfx_prompt], c.sfx_prompt).success(
        bruitages.generer,
                  [c.sfx_prompt, c.sfx_nom, c.sfx_duree, c.sfx_variantes, c.sfx_graine, c.sfx_etapes, c.sfx_image, c.sfx_texte,
                   c.sfx_projet],
                  [c.sfx_statut, c.sfx_liste, c.sfx_audio, c.sfx_dossier])
    c.btn_sfx_choisir.click(bruitages.choisir, [c.sfx_dossier, c.sfx_audio], c.sfx_export_msg)
    c.sfx_liste.change(lambda p: p, c.sfx_liste, c.sfx_audio)
    c.btn_sfx_export.click(export.exporter_fichier_formats, [c.sfx_audio, c.sfx_cible, c.sfx_formats],
                         [c.sfx_export_msg])
