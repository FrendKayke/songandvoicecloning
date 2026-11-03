"""Onglet « Vidéo avec voix off » (groupe Vidéo) : une vidéo + une voix qui lit un texte (ou un enregistrement) → MP4."""
import gradio as gr

from .. import config as cfg
from .. import galerie, voix_off
from ..outils import open_folder
from ..voix import list_voices
from .commun import espace_de_noms


def maj_liste(choisie=None):
    """Vidéos créées dans Studio Voix (la plus récente en premier)."""
    choix = galerie.lister("Vidéos")
    valeurs = [v for _, v in choix]
    return gr.update(choices=choix, value=choisie if choisie in valeurs else (valeurs[0] if valeurs else None))


def _video_de(dossier):
    """Fichier vidéo d'une création (sa version 1)."""
    infos = galerie.lire(dossier) if dossier else None
    if not infos:
        return None
    return (infos.get("versions") or [{}])[0].get("fichier")


def _generer(fichier, creation, *args, progress=gr.Progress()):
    """La vidéo importée passe avant la création choisie."""
    video = fichier or _video_de(creation)
    return voix_off.generer(video, *args, progress=progress)


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Ajoute une voix off à une vidéo** : une vidéo créée ici (Une vidéo, Plusieurs plans, Mini-vidéos…) ou "
        "importée, et une voix qui **lit ton texte** (une voix de ta bibliothèque, synthèse vocale locale) ou un "
        "**enregistrement** (fichier, message vocal, micro). Le son d'origine de la vidéo peut être gardé en fond, "
        "plus bas. Si la voix est plus longue que la vidéo, la dernière image est figée jusqu'à la fin de la voix. "
        "Pour des lèvres qui bougent avec la voix, c'est l'onglet « Faire parler un visage »."
    )
    with gr.Row():
        with gr.Column():
            vo_fichier = gr.Video(label="Vidéo à sonoriser (importe un fichier…)", sources=["upload"], height=280)
            with gr.Row():
                vo_creation = gr.Dropdown([], label="… ou une vidéo créée ici (la plus récente en premier)", scale=4)
                vo_maj = gr.Button("🔄", scale=0)
        with gr.Column():
            vo_source = gr.Radio(voix_off.SOURCES, value=voix_off.TEXTE, label="La voix off")
            vo_voix = gr.Dropdown(choices=list_voices(), value=(list_voices() or [None])[0], label="Voix qui lit")
            vo_langue = gr.Dropdown(list(cfg.LANGUES), value="Français", label="Langue du texte")
            vo_texte = gr.Textbox(label="Texte à lire", lines=5,
                                  placeholder="Bienvenue dans le village de départ. Ici commence ton aventure…")
            vo_audio = gr.Audio(type="filepath", label="Enregistrement (wav, mp3, m4a…) ou micro",
                                sources=["upload", "microphone"], visible=False)
    with gr.Row():
        vo_garder = gr.Checkbox(value=True, label="Garder le son d'origine de la vidéo en fond")
        vo_volume = gr.Slider(0, 100, value=30, step=5, label="Volume du son d'origine (%)")
        vo_prolonger = gr.Checkbox(value=True, label="Figer la dernière image si la voix est plus longue")
        vo_nom = gr.Textbox(label="Nom (pour la galerie)", value="voix_off")
    with gr.Accordion("Réglages de la voix de synthèse (avancé)", open=False):
        vo_exag = gr.Slider(0.25, 2, value=0.5, step=0.05, label="Expressivité")
        vo_cfg = gr.Slider(0, 1, value=0.5, step=0.05, label="Guidage / rythme")
        vo_temp = gr.Slider(0.05, 5, value=0.8, step=0.05, label="Température (variété)")
        vo_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
    btn_vo = gr.Button("🎙️ Ajouter la voix off", variant="primary")
    vo_statut = gr.Markdown()
    vo_dossier = gr.State()
    vo_video = gr.Video(label="Vidéo avec voix off", height=480, interactive=False)
    vo_fichiers = gr.File(label="Fichiers (vidéo, voix)", file_count="multiple", interactive=False)
    btn_vo_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    for evt in (c.vo_maj.click, demo.load, c.onglet.select):
        evt(maj_liste, c.vo_creation, c.vo_creation)
    c.vo_source.change(lambda s: (gr.update(visible=s == voix_off.TEXTE),) * 3 + (gr.update(visible=s != voix_off.TEXTE),),
                       c.vo_source, [c.vo_voix, c.vo_langue, c.vo_texte, c.vo_audio])
    c.btn_vo.click(_generer, [c.vo_fichier, c.vo_creation, c.vo_source, c.vo_voix, c.vo_langue, c.vo_texte, c.vo_audio,
                              c.vo_garder, c.vo_volume, c.vo_prolonger, c.vo_nom, c.vo_exag, c.vo_cfg, c.vo_temp,
                              c.vo_graine],
                   [c.vo_statut, c.vo_video, c.vo_fichiers, c.vo_dossier]).then(maj_liste, c.vo_dossier, c.vo_creation)
    c.btn_vo_dossier.click(lambda d: open_folder(d) if d else None, c.vo_dossier)
