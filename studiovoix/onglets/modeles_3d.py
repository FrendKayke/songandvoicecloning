"""Onglet « Modèles 3D » : image ou texte → modèle 3D, version web, création par lot."""
import gradio as gr

from .. import modele3d, projets
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Un **modèle 3D** à partir d'une **image** d'objet (personnage, arme, objet de carte…) : Hunyuan3D-2 "
        "retire le fond, sculpte la forme puis peint la texture. Une image nette, un seul objet, fond simple : "
        "c'est ce qui marche le mieux. Résultat en GLB (visionneuse ci-dessous, utilisable tel quel dans un "
        "jeu web, Blender, Unity ou Godot) et en OBJ si demandé. "
        "ACE-Step est arrêté automatiquement pendant la génération : la texture demande beaucoup de mémoire "
        "graphique."
    )
    with gr.Accordion("✍️ Pas d'image ? Décris l'objet : une image est générée d'abord (Qwen3-VL + Z-Image-Turbo)",
                      open=False):
        m3_texte = gr.Textbox(label="Description de l'objet (français ou anglais)", lines=2,
                              placeholder="une potion de soin, fiole en verre rouge avec un bouchon de liège")
        btn_m3_prep = gr.Button("🧠 Préparer le prompt de l'image (traduction et précision)")
        m3_prompt = gr.Textbox(label="Prompt envoyé à Z-Image (anglais, modifiable)", lines=2)
        with gr.Row():
            m3_img_graine = gr.Number(value=0, precision=0, label="Graine de l'image (0 = aléatoire)")
            btn_m3_image = gr.Button("🖼️ Générer l'image de l'objet", variant="primary")
        m3_img_msg = gr.Markdown()
    m3_img_graine_ok = gr.State()
    with gr.Row():
        with gr.Column():
            m3_image = gr.Image(type="filepath", label="Image de l'objet (PNG ou JPG, ou l'image générée ci-dessus)")
        with gr.Column():
            m3_nom = gr.Textbox(label="Nom", value="modele")
            m3_projet = gr.Dropdown(projets.tous(), value=None, allow_custom_value=True,
                                    label="Projet de jeu (pour le pack, facultatif)")
            m3_qualite = gr.Radio(list(modele3d.QUALITES), value=modele3d.QUALITE_DEFAUT, label="Qualité")
            m3_texture = gr.Checkbox(value=True, label="Peindre la texture (plusieurs minutes de plus)")
            m3_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
            m3_formats = gr.CheckboxGroup(modele3d.FORMATS, value=["glb"], label="Formats")
    btn_m3 = gr.Button("🧊 Créer le modèle 3D", variant="primary")
    m3_statut = gr.Markdown()
    m3_dossier = gr.State()
    with gr.Row():
        m3_vue = gr.Model3D(label="Modèle 3D (glisser pour tourner, molette pour zoomer)",
                            clear_color=(0.92, 0.92, 0.92, 1.0), scale=3)
        with gr.Column(scale=1):
            m3_detouree = gr.Image(label="Image détourée", interactive=False)
            m3_fichiers = gr.File(label="Fichiers produits", file_count="multiple", interactive=False)
            btn_m3_web = gr.Button("🪶 Alléger pour le web (texture 1024 px en JPEG)")
            btn_m3_dossier = gr.Button("📂 Ouvrir le dossier")
    with gr.Accordion("📚 Plusieurs images à la suite (mêmes réglages)", open=False):
        gr.Markdown("Chaque image donne son propre modèle (nom = préfixe + nom du fichier), rangé dans la "
                    "Galerie. Une image qui échoue n'arrête pas les suivantes. Compte une à quelques minutes "
                    "par modèle selon la qualité et la texture.")
        m3_lot = gr.File(file_count="multiple", file_types=["image"], label="Images des objets")
        m3_lot_prefixe = gr.Textbox(label="Préfixe des noms", value="carte")
        btn_m3_lot = gr.Button("🧊 Créer tous les modèles", variant="primary")
        m3_lot_statut = gr.Markdown()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    # Modèles 3D
    c.btn_m3_web.click(modele3d.alleger, c.m3_dossier, [c.m3_statut, c.m3_fichiers])
    c.btn_m3_lot.click(modele3d.generer_lot, [c.m3_lot, c.m3_lot_prefixe, c.m3_qualite, c.m3_texture, c.m3_graine, c.m3_formats,
                                                     c.m3_projet],
                     [c.m3_lot_statut, c.m3_vue, c.m3_dossier])
    c.btn_m3.click(modele3d.generer,
                 [c.m3_image, c.m3_nom, c.m3_qualite, c.m3_texture, c.m3_graine, c.m3_formats, c.m3_prompt, c.m3_texte,
                  c.m3_img_graine_ok, c.m3_projet],
                 [c.m3_statut, c.m3_vue, c.m3_detouree, c.m3_fichiers, c.m3_dossier])
    c.btn_m3_prep.click(modele3d.preparer_prompt, c.m3_texte, c.m3_prompt)
    c.btn_m3_image.click(modele3d.generer_image, [c.m3_prompt, c.m3_img_graine], [c.m3_image, c.m3_img_graine_ok, c.m3_img_msg])
    c.m3_image.upload(lambda: None, None, c.m3_img_graine_ok)  # image importée : la o.chanson.graine de l'image générée ne vaut plus
    c.btn_m3_dossier.click(lambda d: open_folder(d) if d else None, c.m3_dossier)
