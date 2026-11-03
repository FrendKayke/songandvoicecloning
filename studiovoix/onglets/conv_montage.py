"""Onglet « Montage » (groupe Audio / vidéo) : vidéos et images bout à bout avec fondus, musique, titre et logo."""
import gradio as gr

from .. import galerie, montage
from .commun import espace_de_noms
from .conv_commun import TYPES
from ..outils import open_folder


def maj_creations(choisies=None):
    choix = galerie.lister("Vidéos")
    valeurs = {v for _, v in choix}
    return gr.update(choices=choix, value=[c for c in (choisies or []) if c in valeurs])


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Monte une vidéo** : dépose des vidéos et des images **dans l'ordre**, choisis les fondus, ajoute une "
        "musique de fond, un titre au début et ton logo dans un coin. Le résultat (MP4) va dans la galerie ; pour y "
        "ajouter une voix qui lit un texte, c'est l'onglet « Vidéo → Voix off »."
    )
    with gr.Row():
        mo_fichiers = gr.File(file_count="multiple", file_types=TYPES + sorted(montage.IMAGES),
                              label="Vidéos et images, dans l'ordre", height=170)
        with gr.Column():
            with gr.Row():
                mo_creations = gr.Dropdown([], multiselect=True, scale=4,
                                           label="… et/ou des vidéos créées ici (ajoutées après, dans l'ordre choisi)")
                mo_maj = gr.Button("🔄", scale=0)
            mo_taille = gr.Dropdown(list(montage.TAILLES), value=montage.D_APRES, label="Taille de la vidéo")
            mo_cadrage = gr.Radio(montage.CADRAGES, value=montage.BANDES,
                                  label="Vidéo ou image d'une autre forme que la sortie")
    with gr.Row():
        mo_transition = gr.Radio(montage.TRANSITIONS, value=montage.ENCHAINE, label="Passage d'un morceau au suivant")
        mo_duree_trans = gr.Slider(0.2, 3, value=0.8, step=0.1, label="Durée des fondus (s)")
        mo_duree_image = gr.Slider(1, 20, value=4, step=0.5, label="Durée d'une image (s)")
        mo_fondu = gr.Checkbox(value=True, label="Partir du noir et finir au noir")
    with gr.Accordion("🎵 Son et musique de fond", open=True):
        with gr.Row():
            mo_musique = gr.Audio(type="filepath", label="Musique de fond (facultatif, répétée si trop courte)",
                                  sources=["upload"])
            with gr.Column():
                mo_volume = gr.Slider(0, 100, value=40, step=5, label="Volume de la musique (%)")
                mo_son = gr.Slider(0, 100, value=100, step=5, label="Volume du son des vidéos (%) — 0 = muet")
                mo_baisser = gr.Checkbox(value=True, label="Baisser la musique quand les vidéos ont du son (voix)")
    with gr.Accordion("🔤 Titre et logo", open=False):
        with gr.Row():
            mo_titre = gr.Textbox(label="Titre (une ou deux lignes, vide = aucun)", lines=2)
            with gr.Column():
                mo_duree_titre = gr.Slider(0, 20, value=4, step=0.5, label="Titre affiché pendant (s) — 0 = toute la vidéo")
                mo_pos_titre = gr.Radio(montage.POSITIONS_TITRE, value="Au centre", label="Place du titre")
            with gr.Column():
                mo_police = gr.Dropdown(list(montage.POLICES), value="Élégante (Cinzel)", label="Police")
                mo_taille_titre = gr.Slider(3, 20, value=9, step=1, label="Taille du titre (% de l'image)")
        with gr.Row():
            mo_logo = gr.Image(type="filepath", label="Logo (PNG transparent conseillé)", height=150)
            with gr.Column():
                mo_pos_logo = gr.Radio(montage.POSITIONS_LOGO, value="En haut à droite", label="Coin du logo")
                mo_taille_logo = gr.Slider(4, 40, value=12, step=1, label="Largeur du logo (% de l'image)")
                mo_opacite = gr.Slider(10, 100, value=90, step=5, label="Opacité du logo (%)")
    mo_nom = gr.Textbox(label="Nom (pour la galerie)", value="montage")
    btn_mo = gr.Button("🎬 Monter la vidéo", variant="primary")
    mo_statut = gr.Markdown()
    mo_dossier = gr.State()
    mo_video = gr.Video(label="Montage", height=420, interactive=False)
    mo_resultat = gr.File(label="Fichier", file_count="multiple", interactive=False)
    mo_btn_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    for evt in (c.mo_maj.click, c.onglet.select):
        evt(maj_creations, c.mo_creations, c.mo_creations)
    c.btn_mo.click(montage.monter,
                   [c.mo_fichiers, c.mo_creations, c.mo_transition, c.mo_duree_trans, c.mo_duree_image, c.mo_fondu,
                    c.mo_taille, c.mo_cadrage, c.mo_son, c.mo_musique, c.mo_volume, c.mo_baisser, c.mo_titre,
                    c.mo_duree_titre, c.mo_pos_titre, c.mo_police, c.mo_taille_titre, c.mo_logo, c.mo_pos_logo,
                    c.mo_taille_logo, c.mo_opacite, c.mo_nom],
                   [c.mo_statut, c.mo_video, c.mo_resultat, c.mo_dossier])
    c.mo_btn_dossier.click(lambda d: open_folder(d) if d else None, c.mo_dossier)
