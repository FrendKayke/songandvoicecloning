"""Onglet « Alléger ou facetter un modèle 3D » : version web (texture 1024 px en JPEG) ou low poly facettée
(couleurs unies) d'un modèle déjà créé ; processeur seulement."""
import gradio as gr

from .. import modele3d
from ..outils import open_folder
from .commun import espace_de_noms
from .outils_3d import glb, maj_liste


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Deux versions légères d'un modèle 3D déjà créé.** « Alléger pour le web » : même forme, texture réduite à "
        "1024 px et réencodée en JPEG (12 Mo → moins d'1 Mo), pour un jeu web ou un site. « Version facettée » : "
        "le modèle est réduit à quelques milliers de faces planes, chacune d'une couleur unie lue dans la texture "
        "(look des jeux de construction de ville), sans texture : les couleurs sont dans les sommets. Les deux "
        "fichiers sont ajoutés au dossier du modèle ; le pack du jeu prend la version facettée, sinon la version "
        "web."
    )
    with gr.Row():
        al_modele = gr.Dropdown([], label="Modèle (le plus récent en premier)", scale=4)
        al_maj = gr.Button("🔄 Actualiser", scale=0)
    with gr.Row():
        btn_al_web = gr.Button("🪶 Alléger pour le web (texture 1024 px en JPEG)", variant="primary")
        al_faces = gr.Number(value=3000, precision=0, minimum=200, maximum=50000, label="Faces de la version facettée")
        btn_al_facettes = gr.Button("🔷 Version facettée (low poly, couleurs unies)", variant="primary")
    al_statut = gr.Markdown()
    with gr.Row():
        al_glb = gr.Model3D(label="Modèle", clear_color=(0.92, 0.92, 0.92, 1.0), scale=3)
        with gr.Column(scale=1):
            al_fichiers = gr.File(label="Fichiers du modèle", file_count="multiple", interactive=False)
            btn_al_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    for evt in (c.al_maj.click, demo.load, c.onglet.select):
        evt(maj_liste, c.al_modele, c.al_modele)
    c.al_modele.change(glb, c.al_modele, c.al_glb)
    c.btn_al_web.click(modele3d.alleger, c.al_modele, [c.al_statut, c.al_fichiers])
    c.btn_al_facettes.click(modele3d.facetter, [c.al_modele, c.al_faces], [c.al_statut, c.al_glb, c.al_fichiers])
    c.btn_al_dossier.click(lambda d: open_folder(d) if d else None, c.al_modele)
