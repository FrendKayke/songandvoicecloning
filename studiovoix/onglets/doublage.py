"""Onglet « Traduire et doubler » (groupe Voix) : une vidéo où la personne parle une autre langue."""
import gradio as gr

from .. import config as cfg
from .. import doublage
from ..outils import open_folder
from ..sous_titres import LANGUE_AUTO, LANGUES
from ..voix import list_voices
from .commun import espace_de_noms
from .voix_off import maj_liste


def choix_voix():
    return [doublage.SA_VOIX] + list_voices()


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Traduis et double une vidéo** : ce qui est dit est transcrit, traduit, puis relu dans l'autre langue "
        "**avec la voix de la personne** (clonée depuis la vidéo) ou une de tes voix ; la musique et l'ambiance "
        "sont gardées en dessous. Chaque phrase est placée au moment de l'originale. Tu reçois aussi les "
        "sous-titres traduits et le texte original à côté de sa traduction, pour relire."
    )
    with gr.Row():
        with gr.Column():
            db_video = gr.Video(label="Vidéo à doubler", sources=["upload"], height=280)
            with gr.Row():
                db_creation = gr.Dropdown([], label="… ou une vidéo créée ici", scale=4)
                db_maj = gr.Button("🔄", scale=0)
        with gr.Column():
            db_source = gr.Dropdown(list(LANGUES), value=LANGUE_AUTO, label="Langue parlée dans la vidéo")
            db_cible = gr.Dropdown(list(cfg.LANGUES), value="Anglais", label="Langue du doublage")
            db_voix = gr.Dropdown(choix_voix(), value=doublage.SA_VOIX, label="Voix du doublage")
            db_fond = gr.Checkbox(value=True, label="Garder la musique et l'ambiance (sans la voix d'origine)")
            db_volume = gr.Slider(0, 150, value=100, step=5, label="Volume de la musique et de l'ambiance (%)")
            db_accelerer = gr.Checkbox(value=True, label="Accélérer un peu une phrase trop longue pour sa place")
    with gr.Accordion("Réglages de la voix (avancé)", open=False):
        with gr.Row():
            db_exag = gr.Slider(0.25, 2, value=0.5, step=0.05, label="Expressivité")
            db_cfg = gr.Slider(0, 1, value=0.5, step=0.05, label="Guidage / rythme")
            db_temp = gr.Slider(0.05, 5, value=0.8, step=0.05, label="Température (variété)")
            db_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
    db_nom = gr.Textbox(label="Nom (pour la galerie)", value="doublage")
    btn_db = gr.Button("🌍 Traduire et doubler", variant="primary")
    db_statut = gr.Markdown()
    db_dossier = gr.State()
    with gr.Row():
        db_resultat = gr.Video(label="Vidéo doublée", height=400, interactive=False)
        db_texte = gr.Textbox(label="Original → traduction", lines=16, interactive=False)
    db_fichiers = gr.File(label="Fichiers (vidéo, son, sous-titres, texte)", file_count="multiple", interactive=False)
    db_btn_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    for evt in (c.db_maj.click, c.onglet.select):
        evt(maj_liste, c.db_creation, c.db_creation)
    c.onglet.select(lambda v: gr.update(choices=choix_voix(), value=v if v in choix_voix() else doublage.SA_VOIX),
                    c.db_voix, c.db_voix)
    c.btn_db.click(doublage.doubler, [c.db_video, c.db_creation, c.db_source, c.db_cible, c.db_voix, c.db_fond,
                                      c.db_volume, c.db_accelerer, c.db_exag, c.db_cfg, c.db_temp, c.db_graine,
                                      c.db_nom],
                   [c.db_statut, c.db_resultat, c.db_texte, c.db_fichiers, c.db_dossier])
    c.db_btn_dossier.click(lambda d: open_folder(d) if d else None, c.db_dossier)
