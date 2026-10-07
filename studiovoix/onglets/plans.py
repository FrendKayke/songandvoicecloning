"""Onglet « Vidéo en plusieurs plans » : un plan par ligne, chacun part de la dernière image du précédent."""
import gradio as gr

from .. import videos
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Une **scène plus longue**, en plusieurs plans (Wan 2.2). Écris **un plan par ligne**, en français : chaque "
        "plan devient un clip de la durée choisie et part de la **dernière image du plan précédent** (le premier "
        "part de l'image de départ, s'il y en a une) ; tous les clips sont ensuite assemblés en une seule vidéo. "
        "Chaque plan est préparé avec toute l'histoire, pour garder les mêmes personnages et le même décor. "
        f"{videos.PLANS_MAX} plans au plus ; compte 15 à 25 minutes par plan de 5 s en 720p. Au fil des raccords "
        "l'image peut perdre un peu en netteté : préfère des plans courts. Si un plan échoue, les précédents sont "
        "gardés et assemblés."
    )
    with gr.Row():
        with gr.Column():
            pl_plans = gr.Textbox(label="Plans (un par ligne)", lines=8,
                                  placeholder="les aventuriers avancent sur le pont du navire, la caméra les suit\n"
                                              "un kraken géant surgit de l'eau devant eux\n"
                                              "le mage lance une boule de feu sur le kraken")
        with gr.Column():
            pl_image = gr.Image(type="filepath", label="Image de départ du premier plan (facultative)", height=300)
    with gr.Row():
        pl_format = gr.Dropdown(list(videos.FORMATS), value=videos.AUTO, label="Format")
        pl_duree = gr.Radio(list(videos.DUREES), value=videos.DUREE_DEFAUT, label="Durée de chaque plan")
        pl_etapes = gr.Slider(10, 50, value=30, step=1, label="Étapes (qualité / temps)")
        pl_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
        pl_nom = gr.Textbox(label="Nom (pour la galerie)", value="plans")
    btn_pl = gr.Button("🎞️ Générer les plans à la suite", variant="primary")
    pl_statut = gr.Markdown()
    pl_dossier = gr.State()
    pl_video = gr.Video(label="Vidéo assemblée", height=480, interactive=False)
    btn_pl_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_pl.click(videos.generer_suite,
                   [c.pl_plans, c.pl_image, c.pl_format, c.pl_duree, c.pl_etapes, c.pl_graine, c.pl_nom],
                   [c.pl_statut, c.pl_video, c.pl_dossier])
    c.btn_pl_dossier.click(lambda d: open_folder(d) if d else None, c.pl_dossier)
