"""Onglet « Changer les vêtements » : des photos habillées avec la tenue d'une autre image."""
import gradio as gr

from .. import costumes
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Change **seulement les vêtements** sur tes photos, d'après l'image d'un costume (un personnage d'animé, "
        "une tenue vue en magasin…) : le visage, la pose, le fond et la lumière restent ceux de la photo. "
        "1) Dépose **les photos** (plusieurs à la fois) et **l'image du costume**. 2) **Décris la tenue** : l'IA "
        "regarde le costume et écrit sa description (modifiable). 3) **Habille les photos**. Une photo où la "
        "personne est bien visible, de face, donne le meilleur résultat. Les images obtenues font environ "
        "1 mégapixel, aux proportions de la photo."
    )
    with gr.Row():
        with gr.Column():
            cos_photos = gr.File(label="Photos à habiller (plusieurs possibles)", file_count="multiple",
                                 file_types=["image"])
            cos_costume = gr.Image(type="filepath", label="Image du costume", height=300)
        with gr.Column():
            cos_precisions = gr.Textbox(lines=2, label="Précisions (facultatif, en français)",
                                        placeholder="ex. sans l'épée, avec le bandeau du ninja, couleurs plus vives "
                                                    "— ou la tenue entière si tu n'as pas d'image")
            btn_cos_decrire = gr.Button("📝 1. Décrire la tenue")
            cos_tenue = gr.Textbox(lines=4, label="Tenue (en anglais, modifiable)",
                                   info="Vide : décrite automatiquement au lancement.")
            cos_rendu = gr.Radio(list(costumes.RENDUS), value=costumes.RENDU_DEFAUT, label="Rendu")
            cos_coiffure = gr.Checkbox(value=False, label="Prendre aussi la coiffure du personnage (perruque, "
                                                          "bonnet, oreilles…)")
            with gr.Row():
                cos_variantes = gr.Slider(1, 4, value=2, step=1, label="Variantes par photo")
                cos_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
            cos_nom = gr.Textbox(value="costume", label="Nom")
            btn_cos = gr.Button("👗 2. Habiller les photos", variant="primary")
    cos_statut = gr.Markdown()
    cos_dossier = gr.State()
    cos_galerie = gr.Gallery(label="Résultats", columns=4, height=520, object_fit="contain")
    with gr.Row():
        cos_fichiers = gr.File(label="Fichiers (archive .zip et images)", file_count="multiple", interactive=False)
        btn_cos_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_cos_decrire.click(costumes.decrire_tenue, [c.cos_costume, c.cos_precisions], c.cos_tenue)
    c.cos_costume.change(lambda _: "", c.cos_costume, c.cos_tenue)  # nouvelle image : nouvelle description
    c.btn_cos.click(costumes.habiller,
                    [c.cos_photos, c.cos_costume, c.cos_precisions, c.cos_tenue, c.cos_rendu, c.cos_coiffure,
                     c.cos_variantes, c.cos_graine, c.cos_nom],
                    [c.cos_statut, c.cos_galerie, c.cos_fichiers, c.cos_dossier, c.cos_tenue])
    c.btn_cos_dossier.click(lambda d: open_folder(d) if d else None, c.cos_dossier)
