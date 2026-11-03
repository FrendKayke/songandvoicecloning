"""Onglet « Karaoké » (groupe Musique) : chanson → vidéo aux paroles qui s'allument mot à mot."""
import gradio as gr

from .. import galerie, karaoke
from ..montage import POLICES
from ..outils import open_folder
from ..sous_titres import LANGUE_AUTO, LANGUES
from .commun import espace_de_noms


def maj_chansons(choisie=None):
    choix = galerie.lister("Chansons")
    valeurs = [v for _, v in choix]
    return gr.update(choices=choix, value=choisie if choisie in valeurs else None)


def paroles_de(creation):
    """Paroles et langue d'une chanson de la galerie, pour les relire avant de lancer."""
    _, paroles, langue = karaoke.chanson_de(creation)
    return paroles, (langue if langue in LANGUES else LANGUE_AUTO)


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Fais un karaoké d'une chanson** : la voix est séparée de la musique, chaque mot chanté est repéré, et la "
        "vidéo affiche les paroles **qui s'allument mot à mot**, sur la musique **sans la voix** (pour chanter) ou "
        "avec (clip de paroles). Donne les paroles si tu les as : le texte affiché sera exact, seuls les temps "
        "viennent de l'écoute. Tu reçois aussi un fichier .lrc (lu par les lecteurs de karaoké) et la musique sans "
        "la voix."
    )
    with gr.Row():
        with gr.Column():
            ka_fichier = gr.Audio(type="filepath", label="Chanson (mp3, wav, m4a…)", sources=["upload"])
            with gr.Row():
                ka_creation = gr.Dropdown([], label="… ou une chanson créée ici", scale=4)
                ka_maj = gr.Button("🔄", scale=0)
            ka_langue = gr.Dropdown(list(LANGUES), value=LANGUE_AUTO, label="Langue des paroles")
        ka_paroles = gr.Textbox(label="Paroles (facultatif, une ligne par ligne chantée ; « Refrain », « [Verse] » "
                                      "sont ignorés)", lines=12)
    with gr.Row():
        ka_son = gr.Radio(karaoke.SONS, value=karaoke.SANS_VOIX, label="Son de la vidéo")
        ka_taille = gr.Dropdown(list(karaoke.TAILLES), value=karaoke.TAILLE_DEFAUT, label="Format")
        ka_titre = gr.Textbox(label="Titre affiché au début (facultatif)")
    with gr.Accordion("🎨 Apparence", open=False):
        with gr.Row():
            ka_fond = gr.ColorPicker(value=karaoke.FOND_DEFAUT, label="Fond")
            ka_texte = gr.ColorPicker(value=karaoke.TEXTE_DEFAUT, label="Paroles")
            ka_surligne = gr.ColorPicker(value=karaoke.SURLIGNE_DEFAUT, label="Mots chantés")
            ka_police = gr.Dropdown(list(POLICES), value=next(iter(POLICES)), label="Police")
        ka_image = gr.Image(type="filepath", label="Image de fond (facultatif, assombrie)", height=160)
    ka_nom = gr.Textbox(label="Nom (pour la galerie)", value="karaoke")
    btn_ka = gr.Button("🎤 Créer le karaoké", variant="primary")
    ka_statut = gr.Markdown()
    ka_dossier = gr.State()
    ka_video = gr.Video(label="Karaoké", height=420, interactive=False)
    ka_fichiers = gr.File(label="Fichiers (vidéo, .lrc, .srt, musique sans la voix)", file_count="multiple",
                          interactive=False)
    ka_btn_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    for evt in (c.ka_maj.click, c.onglet.select):
        evt(maj_chansons, c.ka_creation, c.ka_creation)
    c.ka_creation.input(paroles_de, c.ka_creation, [c.ka_paroles, c.ka_langue])
    c.btn_ka.click(karaoke.creer, [c.ka_fichier, c.ka_creation, c.ka_paroles, c.ka_langue, c.ka_son, c.ka_taille,
                                   c.ka_fond, c.ka_texte, c.ka_surligne, c.ka_image, c.ka_police, c.ka_titre,
                                   c.ka_nom],
                   [c.ka_statut, c.ka_video, c.ka_fichiers, c.ka_dossier])
    c.ka_btn_dossier.click(lambda d: open_folder(d) if d else None, c.ka_dossier)
