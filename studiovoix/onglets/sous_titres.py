"""Onglet « Transcrire et sous-titrer » (groupe Outils) : Whisper, texte, .srt / .vtt, vidéo sous-titrée."""
import gradio as gr

from .. import sous_titres
from ..audio import EXTENSIONS
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Transforme la parole en texte et en sous-titres**, sur ton PC (Whisper, 99 langues) : une vidéo, un "
        "message vocal, une chanson… Tu obtiens le texte, les sous-titres (.srt pour les lecteurs vidéo et le "
        "montage, .vtt pour le web) et, pour une vidéo, une copie avec les sous-titres incrustés dans l'image. "
        "Compte quelques secondes par minute de son sur la RTX 4070."
    )
    with gr.Row():
        st_fichier = gr.File(label="Fichier audio ou vidéo", height=140,
                             file_types=sorted(set(EXTENSIONS) | set(sous_titres.VIDEOS)))
        with gr.Column():
            st_langue = gr.Dropdown(list(sous_titres.LANGUES), value=sous_titres.LANGUE_AUTO, label="Langue parlée")
            st_traduire = gr.Checkbox(value=False, label="Traduire en anglais")
            st_incruster = gr.Checkbox(value=True, label="Vidéo : incruster les sous-titres dans l'image")
    btn_st = gr.Button("📝 Transcrire", variant="primary")
    st_statut = gr.Markdown()
    st_texte = gr.Textbox(label="Texte", lines=8, interactive=True)
    st_video = gr.Video(label="Vidéo sous-titrée", interactive=False, height=360)
    st_fichiers = gr.File(file_count="multiple", label="Fichiers (texte, .srt, .vtt, vidéo)", interactive=False)
    st_dossier = gr.State()
    btn_st_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_st.click(sous_titres.transcrire, [c.st_fichier, c.st_langue, c.st_traduire, c.st_incruster],
                   [c.st_statut, c.st_texte, c.st_fichiers, c.st_video, c.st_dossier])
    c.btn_st_dossier.click(lambda d: open_folder(d) if d else None, c.st_dossier)
