"""Onglet « Faire parler une image » : image + voix → vidéo aux lèvres synchronisées (StableAvatar)."""
import gradio as gr

from .. import config as cfg
from .. import fluidite, parlante
from ..outils import open_folder
from ..voix import list_voices
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Fais **parler ou chanter** une photo, une illustration ou un personnage : StableAvatar (licence MIT) anime "
        "le visage et synchronise les lèvres sur la voix, pour toute la durée du son (3 minutes au plus). La voix : "
        "un enregistrement, un message vocal, une chanson, ou un texte lu par une de tes voix. Une image **de face**, "
        "visage bien visible, donne les meilleurs résultats. **C'est long** : compte plusieurs minutes par tranche "
        "de 5 secondes sur une RTX 4070 (estimation, pas encore mesurée). Le son d'origine est gardé dans la vidéo."
    )
    with gr.Row():
        with gr.Column():
            par_image = gr.Image(type="filepath", label="Image à faire parler", height=320)
            par_format = gr.Dropdown(list(parlante.FORMATS), value=parlante.AUTO, label="Format",
                                     info="StableAvatar connaît trois formats ; l'image est recadrée au format choisi.")
        with gr.Column():
            par_source = gr.Radio(parlante.SOURCES, value=parlante.FICHIER, label="La voix")
            par_audio = gr.Audio(type="filepath", label="Son (wav, mp3, m4a…) ou enregistrement au micro",
                                 sources=["upload", "microphone"])
            par_isoler = gr.Checkbox(value=False, label="Isoler la voix (chanson avec musique)",
                                     info="La musique gêne la synchronisation des lèvres : Demucs la retire pour le "
                                          "calcul ; la vidéo garde le son complet.")
            par_voix = gr.Dropdown(choices=list_voices(), value=(list_voices() or [None])[0], label="Voix qui lit",
                                   visible=False)
            par_langue = gr.Dropdown(list(cfg.LANGUES), value="Français", label="Langue du texte", visible=False)
            par_texte = gr.Textbox(label="Texte à dire", lines=4, visible=False)
    with gr.Accordion("Description de la scène (facultatif)", open=False):
        par_souhaits = gr.Textbox(label="Souhaits (français) : expression, manière de parler…", lines=2,
                                  placeholder="elle sourit et parle avec enthousiasme")
        btn_par_prep = gr.Button("🧠 Préparer la description d'après l'image")
        par_prompt = gr.Textbox(label="Description envoyée à StableAvatar (anglais, modifiable ; vide = préparée "
                                      "automatiquement)", lines=3)
    with gr.Row():
        par_etapes = gr.Slider(20, 50, value=30, step=1, label="Étapes (qualité / temps)",
                               info="30 : bon compromis ; 50 : meilleure qualité, plus long.")
        par_guidage = gr.Slider(3, 7, value=5, step=0.5, label="Synchronisation des lèvres",
                                info="Plus haut : lèvres qui suivent mieux la voix (4 à 6 conseillé).")
        par_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
        par_nom = gr.Textbox(label="Nom (pour la galerie)", value="video_parlante")
    btn_par = gr.Button("🗣️ Faire parler l'image", variant="primary")
    par_statut = gr.Markdown()
    par_dossier = gr.State()
    par_video = gr.Video(label="Vidéo parlante", height=480, interactive=False)
    with gr.Row():
        btn_par_fluide = gr.Button("⏩ Plus fluide (50 images/s)")
        btn_par_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    def maj_source(source):
        texte = source == parlante.TEXTE
        return (gr.update(visible=not texte), gr.update(visible=not texte), gr.update(visible=texte),
                gr.update(visible=texte), gr.update(visible=texte))

    c.par_source.change(maj_source, c.par_source, [c.par_audio, c.par_isoler, c.par_voix, c.par_langue, c.par_texte])
    c.btn_par_prep.click(parlante.preparer, [c.par_image, c.par_souhaits], c.par_prompt)
    c.btn_par.click(parlante.prompt_pret, [c.par_image, c.par_souhaits, c.par_prompt], c.par_prompt).success(
        parlante.generer, [c.par_image, c.par_source, c.par_audio, c.par_voix, c.par_texte, c.par_langue,
                           c.par_isoler, c.par_prompt, c.par_format, c.par_etapes, c.par_graine, c.par_guidage,
                           c.par_nom, c.par_souhaits],
        [c.par_statut, c.par_video, c.par_dossier])
    c.btn_par_fluide.click(lambda v, progress=gr.Progress(): fluidite.fluidifier(v, progress=progress)[:2],
                           c.par_video, [c.par_statut, c.par_video])
    c.btn_par_dossier.click(lambda d: open_folder(d) if d else None, c.par_dossier)
