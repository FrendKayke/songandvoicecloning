"""Onglet « Remixer une musique » : un morceau sans paroles rejoué dans un autre style (ACE-Step, tâche cover)."""
import gradio as gr

from .. import export, remix
from .commun import espace_de_noms, liste_style


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Une musique rejouée dans un autre style** : un thème de jeu vidéo en version orchestrale, une chanson pop "
        "en jazz, un générique en 8-bit… Envoie un ou plusieurs morceaux **sans paroles** (MP3, WAV, FLAC, OGG, M4A ; "
        "10 minutes au plus chacun), choisis le nouveau style, puis « Remixer ». ACE-Step garde la mélodie et la "
        "structure du morceau et change les instruments et le style ; plusieurs morceaux sont faits l'un après "
        "l'autre avec les mêmes réglages (pratique pour un quiz). Compte environ 30 s à 1 min par morceau sur la "
        "RTX 4070. Ne remixe que des musiques que tu as le droit d'utiliser (usage personnel)."
    )
    with gr.Row():
        with gr.Column():
            rmx_fichiers = gr.File(file_count="multiple", file_types=["audio"], height=140,
                                   label="Morceaux à remixer (un ou plusieurs)")
            rmx_styles = gr.Dropdown([(lib, val) for lib, val in remix.STYLES], value=[remix.STYLES[0][1]],
                                     multiselect=True, allow_custom_value=True,
                                     label="Nouveau style (plusieurs choix ou le tien en anglais)")
            rmx_instruments = liste_style("instruments", "Instruments (facultatif)")
            rmx_ambiance = liste_style("ambiance", "Ambiance (facultatif)")
            rmx_extra = liste_style("extra", "Autres consignes (facultatif)")
        with gr.Column():
            rmx_transformation = gr.Radio(list(remix.TRANSFORMATIONS), value=remix.TRANSFORMATION_DEFAUT,
                                          label="Transformation")
            rmx_structure = gr.Slider(0, 1, value=remix.TRANSFORMATIONS[remix.TRANSFORMATION_DEFAUT][0], step=0.05,
                                      label="Suivre la structure de l'original",
                                      info="Plus haut : rythme, accords et déroulé de l'original ; plus bas : "
                                           "réinterprétation plus libre (0,3 à 0,5 pour un grand changement de genre).")
            rmx_melodie = gr.Slider(0, 1, value=remix.TRANSFORMATIONS[remix.TRANSFORMATION_DEFAUT][1], step=0.05,
                                    label="Garder la mélodie",
                                    info="0,1 à 0,3 conseillé. Plus haut : la mélodie reste très reconnaissable mais le "
                                         "nouveau style s'impose moins.")
            rmx_fidele = gr.Checkbox(value=False, label="Mode très fidèle (garde aussi davantage le son d'origine)")
            with gr.Row():
                rmx_versions = gr.Radio([1, 2], value=1, label="Versions par morceau")
                rmx_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
    rmx_description = gr.Textbox(label="Description envoyée à ACE-Step (anglais, modifiable)", lines=2,
                                 value=remix.description([remix.STYLES[0][1]]),
                                 info="Recalculée quand tu changes une liste ; tu peux la retoucher (« full symphonic "
                                      "orchestra, heroic French horns… »).")
    btn_rmx = gr.Button("🎛️ Remixer", variant="primary")
    rmx_statut = gr.Markdown()
    rmx_choix = gr.Dropdown([], label="Résultats (choisis-en un pour l'écouter)")
    with gr.Row():
        rmx_original = gr.Audio(label="Original", type="filepath", interactive=False)
        rmx_resultat = gr.Audio(label="Remix", type="filepath", interactive=False)
    with gr.Row():
        rmx_cible = gr.Dropdown(list(export.CIBLES), value=list(export.CIBLES)[0], label="Volume de l'export")
        btn_rmx_export = gr.Button("💾 Exporter ce remix en MP3")
        rmx_mp3 = gr.File(label="MP3 à télécharger")
    rmx_export_msg = gr.Markdown()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.rmx_transformation.input(remix.reglages, c.rmx_transformation, [c.rmx_structure, c.rmx_melodie])
    listes = [c.rmx_styles, c.rmx_instruments, c.rmx_ambiance, c.rmx_extra]
    for liste in listes:
        liste.change(remix.description, listes, c.rmx_description)
    c.btn_rmx.click(remix.remixer,
                    [c.rmx_fichiers, *listes, c.rmx_description, c.rmx_structure, c.rmx_melodie, c.rmx_fidele,
                     c.rmx_versions, c.rmx_graine],
                    [c.rmx_statut, c.rmx_choix, c.rmx_resultat, c.rmx_original])
    c.rmx_choix.input(lambda f: (f, remix.original_de(f)), c.rmx_choix, [c.rmx_resultat, c.rmx_original])
    c.btn_rmx_export.click(export.exporter_fichier, [c.rmx_resultat, c.rmx_cible], [c.rmx_mp3, c.rmx_export_msg])
