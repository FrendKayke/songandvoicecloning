"""Onglet « Tileset en objets 3D » : un jeu de tuiles en pixel art découpé en objets, chacun transformé en modèle
3D low poly texturé (FLUX.2 klein puis Hunyuan3D-2)."""
import gradio as gr

from .. import modele3d, projets, tileset3d
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Transforme les **objets d'un tileset** en pixel art (RPG Maker, Tiled…) en **modèles 3D low poly texturés**, "
        "pour monter ton décor en vraie 3D. 1) Dépose le tileset et **découpe-le** : les objets sont trouvés tout "
        "seuls (cases vides à croix rouge ignorées, objets sur plusieurs cases gardés entiers ; sols, eau et murs "
        "pleins décochés). 2) Décoche ce que tu ne veux pas, puis **crée les objets 3D** : chaque objet est redessiné "
        "en rendu 3D propre (FLUX.2 klein) puis mis en volume et peint (Hunyuan3D-2) — une à deux minutes par objet "
        "sur la carte graphique."
    )
    with gr.Row():
        t3_fichier = gr.File(file_types=["image"], label="Tileset (PNG)")
        with gr.Column():
            t3_case = gr.Number(value=0, precision=0, label="Taille d'une case en pixels (0 = automatique : 32, 48 ou 16)")
            t3_nom = gr.Textbox(label="Nom", value="tileset")
            btn_t3_decouper = gr.Button("🧩 Découper le tileset")
    t3_statut = gr.Markdown()
    t3_pieces = gr.Gallery(label="Objets trouvés", columns=8, height=300, object_fit="contain")
    t3_choix = gr.CheckboxGroup([], label="Objets à mettre en 3D")
    with gr.Row():
        t3_redessiner = gr.Checkbox(value=True, label="Redessiner chaque objet en rendu 3D avant (conseillé)",
                                    info="Hunyuan3D lit mal du pixel art de quelques dizaines de pixels.")
        t3_qualite = gr.Dropdown(list(modele3d.QUALITES), value=tileset3d.QUALITE, label="Qualité")
        t3_texture = gr.Checkbox(value=True, label="Peindre la texture (les détails du pixel art)")
        t3_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
        t3_projet = gr.Dropdown(projets.tous(), value=None, allow_custom_value=True,
                                label="Projet de jeu (pour le pack, facultatif)")
    btn_t3 = gr.Button("🧱 Créer les objets 3D", variant="primary")
    t3_message = gr.Markdown()
    t3_dossier = gr.State()
    t3_vue = gr.Model3D(label="Dernier objet créé (glisser pour tourner)")
    t3_produits = gr.File(label="Modèles GLB", file_count="multiple", interactive=False)
    btn_t3_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def _decouper(fichier, case, nom):
    msg, galerie, choix, coches, dossier = tileset3d.analyser(fichier, case, nom)
    return msg, galerie, gr.update(choices=choix, value=coches), dossier


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_t3_decouper.click(_decouper, [c.t3_fichier, c.t3_case, c.t3_nom],
                            [c.t3_statut, c.t3_pieces, c.t3_choix, c.t3_dossier])
    c.btn_t3.click(tileset3d.generer, [c.t3_dossier, c.t3_choix, c.t3_redessiner, c.t3_qualite, c.t3_texture,
                                       c.t3_graine, c.t3_projet],
                   [c.t3_message, c.t3_vue, c.t3_produits, c.t3_dossier])
    c.btn_t3_dossier.click(lambda d: open_folder(d) if d else None, c.t3_dossier)
