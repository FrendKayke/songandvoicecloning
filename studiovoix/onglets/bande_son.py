"""Onglet « Bande-son de jeu » : situations, thème de référence, génération en lot, pack du jeu."""
import gradio as gr

from .. import export, jeu, projets
from .commun import espace_de_noms, liste_style


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Musiques **sans voix** pour ton jeu, par situation : écran titre, combat, victoire… "
        "Les musiques de fond sont faites pour tourner en boucle, les jingles sont courts. "
        "On décrit un style (JRPG, 16-bit…), jamais une œuvre existante : la musique produite est originale."
    )
    with gr.Row():
        jeu_projet = gr.Dropdown(projets.tous() or ["mon jeu de cartes"], value=(projets.tous() or ["mon jeu de cartes"])[0],
                                 allow_custom_value=True, label="Projet de jeu",
                                 info="Le même nom dans les onglets du groupe Jeu réunit tout dans un seul pack.")
        jeu_epoque = gr.Dropdown(jeu.EPOQUES, value=jeu.EPOQUES[0][1], allow_custom_value=True,
                                 label="Époque / style général",
                                 info="Choisis dans la liste ou tape ton style (en anglais de préférence).")
        jeu_univers = gr.Dropdown(jeu.UNIVERS, value=[], multiselect=True, allow_custom_value=True,
                                  label="Univers", info="Facultatif ; plusieurs choix possibles.")
    jeu_situations = gr.Dropdown(
        jeu.choix_situations(), value=["titre", "combat", "victoire"], multiselect=True,
        allow_custom_value=True, label="Situations à générer",
        info="Tu peux taper ta propre situation (en anglais, par ex. « fire faction theme, aggressive "
             "taiko drums ») : elle sera générée comme musique en boucle.",
    )
    jeu_extra = liste_style("instruments", "Instruments à ajouter à toutes les pistes (facultatif)")
    with gr.Accordion("🎼 Cohérence : partir d'un thème du projet", open=False):
        gr.Markdown(
            "Génère d'abord un **thème principal** (par exemple l'écran titre), puis choisis-le ici : "
            "« Même son » donne à toutes les pistes le même timbre et le même mixage ; « Variation du thème » "
            "réarrange sa mélodie selon chaque situation (version combat, version calme…), comme les "
            "leitmotivs des JRPG. En variation, les musiques de fond prennent la durée du thème et les "
            "jingles utilisent seulement le même son."
        )
        with gr.Row():
            jeu_ref = gr.Dropdown([], label="Thème de référence (pistes du projet)")
            jeu_usage = gr.Radio(jeu.REFERENCES, value=jeu.REF_AUCUNE, label="Utilisation")
            jeu_fidelite = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Fidélité au thème (variation)",
                                     info="Haut : très proche du thème. Bas : plus libre.")
    with gr.Row():
        jeu_duree = gr.Slider(30, 180, value=90, step=10, label="Durée des musiques en boucle (s)")
        jeu_thinking = gr.Checkbox(value=False, label="Mode réflexion (LM)",
                                   info="Désactivé par défaut : la description est suivie plus fidèlement.")
        jeu_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
    jeu_apercu = gr.Markdown()
    btn_jeu = gr.Button("🎮 Générer la bande-son", variant="primary")
    jeu_statut = gr.Markdown()
    with gr.Row():
        jeu_liste = gr.Dropdown([], label="Écouter une piste générée")
        jeu_audio = gr.Audio(label="Piste (en boucle dans le jeu)", type="filepath")
        jeu_jonction = gr.Audio(label="Jonction : 5 s de fin puis 5 s de début", type="filepath")
    with gr.Accordion("📦 Export pour le jeu (OGG / MP3, volume harmonisé, manifest.json)", open=False):
        gr.Markdown(
            "Réunit tout le projet dans `data/jeux/<projet>/export/` et une archive zip : la piste la plus récente "
            "de chaque situation et les bruitages gardés, au même volume (OGG pour les navigateurs récents, MP3 en "
            "secours, boucles exactes), les illustrations gardées, les cartes composées et les modèles 3D du projet "
            "(version web si elle existe), avec un seul `manifest.json`."
        )
        with gr.Row():
            jeu_cible = gr.Dropdown(list(export.CIBLES), value=list(export.CIBLES)[1], label="Volume cible")
            jeu_formats = gr.CheckboxGroup([("OGG", "ogg"), ("MP3", "mp3")], value=["ogg", "mp3"],
                                           label="Formats")
        btn_export_jeu = gr.Button("📦 Exporter le pack du jeu (musiques, bruitages, illustrations, cartes, 3D)")
        jeu_export_msg = gr.Markdown()
        jeu_zip = gr.File(label="Archive à télécharger")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_export_jeu.click(export.exporter_pack, [c.jeu_projet, c.jeu_cible, c.jeu_formats], [c.jeu_zip, c.jeu_export_msg])
    champs_jeu = [c.jeu_epoque, c.jeu_univers, c.jeu_situations, c.jeu_extra, c.jeu_duree]
    for champ in champs_jeu:
        champ.change(jeu.apercu, champs_jeu, c.jeu_apercu)
    demo.load(jeu.apercu, champs_jeu, c.jeu_apercu)
    c.btn_jeu.click(jeu.generer_bande_son,
                  [c.jeu_projet, c.jeu_epoque, c.jeu_univers, c.jeu_situations, c.jeu_extra, c.jeu_duree, c.jeu_thinking,
                   c.jeu_graine, c.jeu_ref, c.jeu_usage, c.jeu_fidelite],
                  [c.jeu_statut, c.jeu_liste, c.jeu_audio, c.jeu_jonction]).then(
        jeu.maj_references, [c.jeu_projet, c.jeu_ref], c.jeu_ref)
    c.jeu_projet.change(jeu.maj_references, [c.jeu_projet, c.jeu_ref], c.jeu_ref)
    demo.load(jeu.maj_references, [c.jeu_projet, c.jeu_ref], c.jeu_ref)
    c.jeu_liste.change(jeu.ecouter, c.jeu_liste, [c.jeu_audio, c.jeu_jonction])
