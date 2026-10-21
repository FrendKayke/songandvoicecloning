"""Onglet « Tileset en objets 3D » : un jeu de tuiles en pixel art découpé en objets ; chaque objet est identifié
(Qwen3-VL), dessiné en 3D d'après sa description (Z-Image) puis mis en volume (Hunyuan3D-2)."""
import gradio as gr

from .. import modele3d, projets, tileset3d
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Transforme les **objets d'un tileset** en pixel art (RPG Maker, Tiled…) en **modèles 3D low poly texturés**. "
        "1) **Découpe** le tileset : les objets sont trouvés tout seuls (cases vides à croix rouge ignorées ; sols, "
        "eau et murs pleins décochés). 2) **Identifie** les objets cochés : l'IA dit ce que chacun représente "
        "(« une échelle en bois » pour un trou avec une échelle vu de dessus) ; **corrige le texte** si besoin. "
        "3) **Dessine**-les en 3D et regarde les images. 4) **Crée les modèles 3D** (une à deux minutes par objet)."
    )
    with gr.Row():
        t3_fichier = gr.File(file_types=["image"], label="Tileset (PNG)")
        with gr.Column():
            t3_case = gr.Number(value=0, precision=0, label="Taille d'une case en pixels (0 = automatique : 32, 48 ou 16)")
            t3_nom = gr.Textbox(label="Nom", value="tileset")
            btn_t3_decouper = gr.Button("🧩 1. Découper le tileset")
    t3_statut = gr.Markdown()
    t3_pieces = gr.Gallery(label="Objets trouvés", columns=8, height=300, object_fit="contain")
    t3_choix = gr.CheckboxGroup([], label="Objets à mettre en 3D")
    btn_t3_identifier = gr.Button("🔎 2. Identifier les objets cochés")
    t3_table = gr.Dataframe(headers=["Objet", "Ce que c'est (modifiable, français ou anglais)"],
                            datatype=["str", "str"], type="array", interactive=True, wrap=True,
                            column_count=2, label="Objets à créer")
    with gr.Row():
        t3_garder = gr.Checkbox(value=False, label="Garder le dessin d'origine (FLUX.2 klein)",
                                info="Décoché : dessin d'après le texte, en vraie 3D (conseillé). Coché : plus "
                                     "fidèle au dessin, mais il recopie souvent les pixels.")
        t3_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
    btn_t3_dessiner = gr.Button("🎨 3. Dessiner les objets")
    t3_rendus = gr.Gallery(label="Objets dessinés (la 3D sera faite d'après ces images)", columns=6, height=300,
                           object_fit="contain")
    with gr.Row():
        t3_qualite = gr.Dropdown(list(modele3d.QUALITES), value=tileset3d.QUALITE, label="Qualité")
        t3_texture = gr.Checkbox(value=True, label="Peindre la texture")
        t3_projet = gr.Dropdown(projets.tous(), value=None, allow_custom_value=True,
                                label="Projet de jeu (pour le pack, facultatif)")
    btn_t3 = gr.Button("🧱 4. Créer les modèles 3D", variant="primary")
    t3_message = gr.Markdown()
    t3_dossier = gr.State()
    t3_vue = gr.Model3D(label="Dernier objet créé (glisser pour tourner)")
    t3_produits = gr.File(label="Modèles GLB", file_count="multiple", interactive=False)
    btn_t3_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def _decouper(fichier, case, nom):
    msg, galerie, choix, coches, dossier = tileset3d.analyser(fichier, case, nom)
    return msg, galerie, gr.update(choices=choix, value=coches), dossier, [], []


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_t3_decouper.click(_decouper, [c.t3_fichier, c.t3_case, c.t3_nom],
                            [c.t3_statut, c.t3_pieces, c.t3_choix, c.t3_dossier, c.t3_table, c.t3_rendus])
    c.btn_t3_identifier.click(tileset3d.identifier, [c.t3_dossier, c.t3_choix], [c.t3_table, c.t3_statut])
    c.btn_t3_dessiner.click(tileset3d.dessiner, [c.t3_dossier, c.t3_table, c.t3_garder, c.t3_graine],
                            [c.t3_rendus, c.t3_statut, c.t3_dossier])
    c.btn_t3.click(tileset3d.generer, [c.t3_dossier, c.t3_table, c.t3_garder, c.t3_qualite, c.t3_texture,
                                       c.t3_graine, c.t3_projet],
                   [c.t3_message, c.t3_vue, c.t3_produits, c.t3_dossier])
    c.btn_t3_dossier.click(lambda d: open_folder(d) if d else None, c.t3_dossier)
