"""Onglet « Diaporama » (groupe Audio / vidéo) : des photos → une vidéo animée, avec musique et voix off."""
import gradio as gr

from .. import config as cfg
from .. import diaporama, montage
from ..outils import open_folder
from ..voix import list_voices
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Fais une vidéo avec tes photos** : chaque photo bouge doucement (zoom ou travelling, effet « Ken "
        "Burns »), les photos s'enchaînent en fondu, avec une musique de fond et, si tu veux, une **voix off** (un "
        "texte lu par une de tes voix, ou un enregistrement). La musique baisse quand la voix parle, et les photos "
        "restent assez longtemps pour toute la voix. Le résultat va dans la galerie."
    )
    with gr.Row():
        di_photos = gr.File(file_count="multiple", file_types=sorted(montage.IMAGES), height=200,
                            label="Photos, dans l'ordre")
        with gr.Column():
            di_mouvement = gr.Radio(list(diaporama.MOUVEMENTS), value=diaporama.MOUVEMENT_DEFAUT,
                                    label="Mouvement des photos")
            with gr.Row():
                di_duree = gr.Slider(2, 15, value=5, step=0.5, label="Durée d'une photo (s)")
                di_fondu = gr.Slider(0, 3, value=1, step=0.25, label="Fondu entre deux photos (s)")
            di_taille = gr.Dropdown([t for t in montage.TAILLES if montage.TAILLES[t]], value=diaporama.TAILLE_DEFAUT,
                                    label="Format")
    with gr.Accordion("🎙️ Voix off", open=True):
        di_source = gr.Radio(diaporama.SOURCES, value=diaporama.SANS_VOIX, label="Voix off")
        with gr.Row(visible=False) as di_bloc_texte:
            di_voix = gr.Dropdown(choices=list_voices(), value=(list_voices() or [None])[0], label="Voix qui lit")
            di_langue = gr.Dropdown(list(cfg.LANGUES), value="Français", label="Langue")
            di_texte = gr.Textbox(label="Texte à lire", lines=4, scale=3)
        di_enregistrement = gr.Audio(type="filepath", sources=["upload", "microphone"], visible=False,
                                     label="Enregistrement (wav, mp3, m4a…) ou micro")
        di_adapter = gr.Checkbox(value=True, label="Allonger les photos pour que toute la voix tienne")
    with gr.Accordion("🎵 Musique et titre", open=False):
        with gr.Row():
            di_musique = gr.Audio(type="filepath", sources=["upload"], label="Musique de fond (facultatif)")
            with gr.Column():
                di_volume = gr.Slider(0, 100, value=40, step=5, label="Volume de la musique (%)")
                di_titre = gr.Textbox(label="Titre au début (facultatif)")
                di_police = gr.Dropdown(list(montage.POLICES), value="Élégante (Cinzel)", label="Police du titre")
    di_nom = gr.Textbox(label="Nom (pour la galerie)", value="diaporama")
    btn_di = gr.Button("🖼️ Créer le diaporama", variant="primary")
    di_statut = gr.Markdown()
    di_dossier = gr.State()
    di_video = gr.Video(label="Diaporama", height=420, interactive=False)
    di_fichiers = gr.File(label="Fichiers", file_count="multiple", interactive=False)
    di_btn_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.di_source.change(lambda s: (gr.update(visible=s == diaporama.TEXTE),
                                  gr.update(visible=s == diaporama.ENREGISTREMENT)),
                       c.di_source, [c.di_bloc_texte, c.di_enregistrement])
    c.onglet.select(lambda v: gr.update(choices=list_voices(), value=v if v in list_voices()
                                        else (list_voices() or [None])[0]), c.di_voix, c.di_voix)
    c.btn_di.click(diaporama.creer, [c.di_photos, c.di_duree, gr.State(montage.ENCHAINE), c.di_fondu,
                                     c.di_mouvement, c.di_taille, c.di_musique, c.di_volume, c.di_source, c.di_voix,
                                     c.di_langue, c.di_texte, c.di_enregistrement, c.di_adapter, c.di_titre,
                                     c.di_police, c.di_nom],
                   [c.di_statut, c.di_video, c.di_fichiers, c.di_dossier])
    c.di_btn_dossier.click(lambda d: open_folder(d) if d else None, c.di_dossier)
