"""Onglet « Entraîner un modèle de sprites » : feuilles de marche d'exemple → légendes → LoRA SDXL (kohya), proposé
ensuite dans le kit de dresseur pour générer la feuille de marche d'un nouveau personnage."""
import gradio as gr

from .. import dresseur, modele_sprites
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Apprendre à SDXL les feuilles de marche de ton jeu.** Aucun générateur ne dessine une feuille de marche "
        "cohérente (16 poses de 16×24 px) : on l'apprend à SDXL à partir de tes feuilles (une soixantaine, d'un seul "
        "personnage chacune, 4×4 cases en GBA ou 3×4 en DS). 1) Les feuilles sont ramenées à leurs vrais pixels, "
        "agrandies sur un fond magenta et légendées par Qwen3-VL (« a man with brown hair, red cap… »). 2) kohya "
        "sd-scripts entraîne un LoRA (U-Net seul, 8 à 10 Go de mémoire graphique, environ une heure sur une RTX "
        "4070). 3) Dans le kit de dresseur, choisis ce modèle pour la feuille de marche : elle est générée d'après la "
        "description du personnage, puis remise sur la grille et les couleurs du jeu."
    )
    with gr.Row():
        lo_feuilles = gr.File(file_count="multiple", file_types=["image"],
                              label=f"Feuilles de marche d'exemple ({modele_sprites.FEUILLES_MIN} au moins)")
        with gr.Column():
            lo_style = gr.Radio(list(dresseur.STYLES), value=next(iter(dresseur.STYLES)), label="Style (grille)")
            lo_nom = gr.Textbox(label="Nom du modèle", value="marche_gba")
            lo_legender = gr.Checkbox(value=True, label="Légender chaque feuille avec Qwen3-VL (conseillé)")
            btn_lo_preparer = gr.Button("1️⃣ Préparer les feuilles", variant="primary")
    lo_prep_msg = gr.Markdown()
    with gr.Row():
        lo_epoques = gr.Slider(4, 30, value=modele_sprites.EPOQUES_DEFAUT, step=1, label="Époques",
                               info="Passages sur toutes les feuilles (×10 répétitions). 12 ≈ 1 h pour 60 feuilles.")
        lo_dim = gr.Slider(8, 64, value=modele_sprites.DIM_DEFAUT, step=8, label="Dimension du LoRA",
                           info="Plus grand = plus de détails appris, plus de mémoire.")
        lo_graine = gr.Number(value=0, precision=0, label="Graine (0 = 42)")
        btn_lo_entrainer = gr.Button("2️⃣ Entraîner (environ une heure)", variant="primary")
    lo_msg = gr.Markdown()
    lo_fichier = gr.File(label="Modèle entraîné (LoRA SDXL)", interactive=False)
    lo_liste = gr.Markdown(etat_modeles())
    btn_lo_dossier = gr.Button("📂 Ouvrir le dossier des modèles entraînés")
    return espace_de_noms(locals())


def etat_modeles():
    prets = modele_sprites.loras()
    if not prets:
        return "Aucun modèle entraîné pour l'instant."
    return "**Modèles entraînés** (proposés dans le kit de dresseur) : " + ", ".join(f"`{n}`" for n in prets)


def _preparer(feuilles, nom, style, legender, progress=gr.Progress()):
    msg, _ = modele_sprites.preparer(feuilles, nom, style, legender, progress=progress)
    return msg


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_lo_preparer.click(_preparer, [c.lo_feuilles, c.lo_nom, c.lo_style, c.lo_legender], c.lo_prep_msg)
    c.btn_lo_entrainer.click(modele_sprites.entrainer, [c.lo_nom, c.lo_epoques, c.lo_dim, c.lo_graine],
                             [c.lo_msg, c.lo_fichier]).then(etat_modeles, None, c.lo_liste).then(
        dresseur.liste_loras_marche, None, o.pixel_art.kit_lora_marche)
    c.btn_lo_dossier.click(lambda: open_folder(modele_sprites.cfg.LORAS_DIR, create=True))
