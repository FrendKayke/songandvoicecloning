"""Onglet « Mini-vidéos d'une histoire » (groupe Vidéo) : une mini-vidéo par image d'une histoire en images (Wan 2.2),
assemblées si demandé."""
import gradio as gr

from .. import galerie, videos
from ..outils import open_folder
from .commun import espace_de_noms


def histoires():
    """Histoires en images (créations de type image avec des scènes), les plus récentes d'abord : [(libellé, dossier)]."""
    return [(lib, d) for lib, d in galerie.lister("Images") if (galerie.lire(d) or {}).get("scenes")]


def maj_liste(choisie=None):
    choix = histoires()
    valeurs = [v for _, v in choix]
    return gr.update(choices=choix, value=choisie if choisie in valeurs else (valeurs[0] if valeurs else None))


def _apercu(dossier):
    """Images de l'histoire choisie."""
    infos = galerie.lire(dossier) if dossier else None
    if not infos or not infos.get("scenes"):
        return []
    from pathlib import Path

    return [str(Path(dossier) / s["fichier"]) for s in infos["scenes"] if (Path(dossier) / s.get("fichier", "")).is_file()]


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Une mini-vidéo par image d'une histoire** : Wan 2.2 anime chaque image (~4 à 7 min par mini-vidéo de 3 s "
        "sur la RTX 4070), puis les assemble en une seule vidéo si tu veux. Choisis une histoire créée dans "
        "« Histoire en images » (la dernière est proposée). Chaque clip part de son image, avec la scène comme "
        "description et le mouvement indiqué."
    )
    with gr.Row():
        mv_histoire = gr.Dropdown([], label="Histoire en images (la plus récente en premier)", scale=4)
        mv_maj = gr.Button("🔄 Actualiser", scale=0)
    mv_apercu = gr.Gallery(label="Images de l'histoire", columns=6, height=200, object_fit="contain")
    with gr.Row():
        mv_duree = gr.Radio(["2 s", "3 s"], value="3 s", label="Durée de chaque mini-vidéo")
        mv_etapes = gr.Slider(10, 40, value=20, step=1, label="Pas (plus = plus fin, plus long)")
        mv_assembler = gr.Checkbox(value=True, label="Assembler en une seule vidéo")
        mv_nom = gr.Textbox(label="Nom (pour la galerie)", value="histoire")
    mv_mouvement = gr.Textbox(label="Mouvement (facultatif, anglais ; vide = la scène s'anime doucement)",
                              placeholder=videos.MOUVEMENT_DEFAUT)
    btn_mv = gr.Button("🎬 Faire les mini-vidéos", variant="primary")
    mv_statut = gr.Markdown()
    mv_dossier = gr.State()
    mv_video = gr.Video(label="Mini-vidéos", height=420)
    btn_mv_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    for evt in (c.mv_maj.click, demo.load, c.onglet.select):
        evt(maj_liste, c.mv_histoire, c.mv_histoire)
    c.mv_histoire.change(_apercu, c.mv_histoire, c.mv_apercu)
    c.btn_mv.click(videos.generer_depuis_images,
                   [c.mv_histoire, c.mv_duree, c.mv_etapes, c.mv_mouvement, c.mv_assembler, c.mv_nom],
                   [c.mv_statut, c.mv_video, c.mv_dossier])
    c.btn_mv_dossier.click(lambda d: open_folder(d) if d else None, c.mv_dossier)
