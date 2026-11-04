"""Onglet « Convertir des feuilles de marche » : anciennes feuilles (style GBA ou RPG Maker XP) vers un autre style
(Diamant / Perle / Platine), en masse, mises en page puis redessinées par un modèle de feuilles entraîné."""
import gradio as gr

from .. import conversion_marche as cm
from .. import dresseur
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    styles = list(dresseur.STYLES)
    gr.Markdown(
        "**Passer tes feuilles de marche d'un style à l'autre, toutes d'un coup.** Chaque direction et chaque pose est "
        "replacée dans la grille du style d'arrivée (par exemple 4×4 cases de 16×24 → 3×4 cases de 32×32, ordre haut, "
        "bas, gauche, droite), le personnage agrandi à la taille habituelle du style. Avec un **modèle de feuilles "
        "entraîné sur le style d'arrivée** (onglet 🧪 Entraîner un modèle de sprites), chaque feuille est ensuite "
        "redessinée dans ce style en gardant le personnage et ses couleurs. Sans modèle, tu obtiens la mise en page "
        "seule (personnage agrandi au plus proche, pas redessiné).")
    with gr.Row():
        mc_feuilles = gr.File(file_count="multiple", file_types=[".png", ".gif", ".webp", ".bmp", ".zip"],
                              label=f"Feuilles de marche à convertir : PNG ou archive .zip ({cm.CONVERSION_MAX} au plus)")
        with gr.Column():
            mc_depart = gr.Radio(styles, value=styles[0], label="Style de départ (tes anciennes feuilles)")
            mc_arrivee = gr.Radio(styles, value=styles[-1], label="Style d'arrivée")
            mc_modele = gr.Dropdown(cm.modeles_de_marche(), value=cm.SANS_IA, label="Modèle de feuilles du style "
                                    "d'arrivée")
    with gr.Accordion("Réglages", open=False):
        with gr.Row():
            mc_force = gr.Slider(0.3, 0.85, value=cm.FORCE_DEFAUT, step=0.05, label="Part redessinée",
                                 info="Bas = garde la forme d'origine ; haut = plus proche du style appris, mais le "
                                      "personnage peut changer.")
            mc_graine = gr.Number(value=0, precision=0, label="Graine (une de plus par feuille)")
        with gr.Row():
            mc_couleurs = gr.Checkbox(value=True, label="Garder les couleurs de la feuille d'origine")
            mc_miroir = gr.Checkbox(value=True, label="Droite = gauche en miroir (comme dans les jeux DS)")
            mc_legender = gr.Checkbox(value=True, label="Décrire chaque personnage avec Qwen3-VL (conseillé)")
        mc_nom = gr.Textbox(label="Nom du lot", value="anciennes_feuilles")
    btn_mc = gr.Button("🔁 Convertir", variant="primary")
    mc_msg = gr.Markdown()
    mc_galerie = gr.Gallery(label="Avant (à gauche) / après (à droite)", columns=3, height="auto",
                            object_fit="contain")
    with gr.Row():
        mc_zip = gr.File(label="Feuilles converties (.zip)", interactive=False)
        mc_dossier = gr.Textbox(label="Dossier", interactive=False)
    btn_mc_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_mc.click(cm.convertir, [c.mc_feuilles, c.mc_depart, c.mc_arrivee, c.mc_modele, c.mc_force, c.mc_couleurs,
                                  c.mc_miroir, c.mc_legender, c.mc_graine, c.mc_nom],
                   [c.mc_msg, c.mc_galerie, c.mc_zip, c.mc_dossier])
    demo.load(cm.liste_modeles, None, c.mc_modele)
    c.btn_mc_dossier.click(lambda d: open_folder(d) if d else None, c.mc_dossier)
