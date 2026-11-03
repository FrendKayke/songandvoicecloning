"""Onglet « Squelette pour l'animation » : UniRig place un squelette dans un modèle 3D (créé ici ou importé)."""
import gradio as gr

from .. import rigging
from ..outils import open_folder
from .commun import espace_de_noms
from .outils_3d import glb, maj_liste


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Un squelette pour animer un modèle 3D.** UniRig (licence MIT) place un squelette dans le modèle et calcule "
        "l'influence de chaque os sur la surface : le modèle peut ensuite être animé dans Blender ou un moteur de jeu "
        "(Unity, Unreal, Godot). Choisis un modèle créé ici, ou importe un fichier. Personnages, animaux et "
        "créatures marchent le mieux ; compte une à quelques minutes. Résultat : `modele_anime.glb` et "
        "`modele_anime.fbx`."
    )
    with gr.Row():
        rig_modele = gr.Dropdown([], label="Modèle créé dans Studio Voix (le plus récent en premier)", scale=4)
        rig_maj = gr.Button("🔄 Actualiser", scale=0)
    with gr.Row():
        rig_fichier = gr.File(label="Ou un modèle à importer (GLB, FBX, OBJ, VRM) : il passe avant la liste",
                              file_types=[".glb", ".gltf", ".fbx", ".obj", ".vrm"])
        rig_graine = gr.Number(value=0, precision=0, label="Graine (0 = par défaut ; autre = autre squelette)")
    btn_rig = gr.Button("🦴 Ajouter un squelette", variant="primary")
    rig_statut = gr.Markdown()
    with gr.Row():
        rig_glb = gr.Model3D(label="Modèle (animable après le calcul)", clear_color=(0.92, 0.92, 0.92, 1.0), scale=3)
        with gr.Column(scale=1):
            rig_fichiers = gr.File(label="Fichiers produits", file_count="multiple", interactive=False)
            btn_rig_dossier = gr.Button("📂 Ouvrir le dossier du modèle")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    for evt in (c.rig_maj.click, demo.load, c.onglet.select):
        evt(maj_liste, c.rig_modele, c.rig_modele)
    c.rig_modele.change(glb, c.rig_modele, c.rig_glb)
    c.btn_rig.click(rigging.animer, [c.rig_modele, c.rig_fichier, c.rig_graine],
                    [c.rig_statut, c.rig_glb, c.rig_fichiers])
    c.btn_rig_dossier.click(lambda d: open_folder(d) if d else None, c.rig_modele)
