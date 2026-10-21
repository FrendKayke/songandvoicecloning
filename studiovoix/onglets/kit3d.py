"""Onglet « Kit d'objets 3D » : une liste d'objets (un par ligne) → un modèle 3D par objet, tous dans le même style
(low poly facetté par défaut)."""
import gradio as gr

from .. import kit3d, modele3d, projets
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Crée **tout un kit d'objets 3D dans le même style** : immeubles, voitures, arbres, mobilier urbain… "
        "1) Écris **un objet par ligne** (en français) et **prépare** : l'IA traduit chaque objet en description "
        "anglaise, modifiable. 2) **Dessine** les objets et vérifie les images. 3) **Crée les modèles 3D** : en "
        "« Low poly facetté », faces planes de couleur unie, comme dans les jeux de construction de ville "
        "(une à deux minutes par objet). Ne cite jamais le nom d'un jeu ou d'un pack existant."
    )
    with gr.Row():
        kit_texte = gr.Textbox(lines=8, label="Objets du kit (un par ligne)", placeholder=kit3d.EXEMPLE)
        with gr.Column():
            kit_style = gr.Textbox(value=kit3d.STYLE_DEFAUT, lines=3, label="Style commun (en anglais)",
                                   info="Ajouté à chaque objet : c'est lui qui donne l'unité du kit.")
            kit_nom = gr.Textbox(value="kit", label="Nom du kit")
            kit_traduire = gr.Checkbox(value=True, label="Traduire en anglais (Qwen3-VL)")
            btn_kit_preparer = gr.Button("📝 1. Préparer")
    kit_statut = gr.Markdown()
    kit_table = gr.Dataframe(headers=kit3d.ENTETES, datatype=["str", "str", "str"], type="array", interactive=True,
                             wrap=True, column_count=3, label="Objets (noms et descriptions modifiables)")
    with gr.Row():
        kit_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
        btn_kit_dessiner = gr.Button("🎨 2. Dessiner les objets")
    kit_rendus = gr.Gallery(label="Objets dessinés (la 3D sera faite d'après ces images)", columns=6, height=320,
                            object_fit="contain")
    with gr.Row():
        kit_qualite = gr.Dropdown(list(modele3d.QUALITES), value=kit3d.QUALITE, label="Qualité")
        kit_texture = gr.Checkbox(value=True, label="Peindre la texture (donne les couleurs des facettes)")
        kit_projet = gr.Dropdown(projets.tous(), value=None, allow_custom_value=True,
                                 label="Projet de jeu (pour le pack, facultatif)")
    btn_kit = gr.Button("🧱 3. Créer les modèles 3D", variant="primary")
    with gr.Accordion("🌙 Tout lancer pour la nuit", open=False):
        gr.Markdown("Prépare, dessine et crée tous les modèles **d'un seul clic**, sans vérification entre les étapes "
                    "(une à deux minutes par objet : 50 objets ≈ 1 h 30 à 2 h). Windows ne se met pas en veille "
                    "pendant le calcul (l'écran peut s'éteindre) ; **laisse cette page ouverte**. Si la nuit est "
                    "interrompue (coupure, redémarrage), « Reprendre » ne refait que ce qui manque.")
        with gr.Row():
            kit_eteindre = gr.Checkbox(value=False, label="Éteindre l'ordinateur à la fin (5 minutes après, annulable)")
            btn_kit_nuit = gr.Button("🌙 Tout faire d'un coup", variant="primary")
        with gr.Row():
            kit_reprise = gr.Textbox(label="Kit à reprendre (dossier ; vide = le dernier de cet onglet)")
            btn_kit_reprendre = gr.Button("🔁 Reprendre")
    kit_message = gr.Markdown()
    kit_dossier = gr.State()
    kit_vue = gr.Model3D(label="Dernier objet créé (glisser pour tourner)")
    kit_produits = gr.File(label="Modèles GLB et archive du kit", file_count="multiple", interactive=False)
    btn_kit_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def _dessiner(dossier, table, style, graine, progress=gr.Progress()):
    return kit3d.dessiner(dossier, table, style, graine, progress=progress)


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_kit_preparer.click(kit3d.preparer, [c.kit_texte, c.kit_traduire, c.kit_nom],
                             [c.kit_table, c.kit_statut, c.kit_dossier])
    c.btn_kit_dessiner.click(_dessiner, [c.kit_dossier, c.kit_table, c.kit_style, c.kit_graine],
                             [c.kit_rendus, c.kit_statut, c.kit_dossier])
    c.btn_kit.click(kit3d.generer, [c.kit_dossier, c.kit_table, c.kit_style, c.kit_qualite, c.kit_texture,
                                    c.kit_graine, c.kit_projet],
                    [c.kit_message, c.kit_vue, c.kit_produits, c.kit_dossier])
    c.btn_kit_nuit.click(kit3d.tout_faire, [c.kit_texte, c.kit_traduire, c.kit_nom, c.kit_style, c.kit_qualite,
                                            c.kit_texture, c.kit_graine, c.kit_projet, c.kit_eteindre],
                         [c.kit_table, c.kit_message, c.kit_vue, c.kit_produits, c.kit_dossier])
    c.btn_kit_reprendre.click(lambda chemin, dossier, q, t, p, progress=gr.Progress():
                              kit3d.reprendre(chemin.strip() or dossier, q, t, p, progress=progress),
                              [c.kit_reprise, c.kit_dossier, c.kit_qualite, c.kit_texture, c.kit_projet],
                              [c.kit_message, c.kit_vue, c.kit_produits, c.kit_dossier])
    c.btn_kit_dossier.click(lambda d: open_folder(d) if d else None, c.kit_dossier)
