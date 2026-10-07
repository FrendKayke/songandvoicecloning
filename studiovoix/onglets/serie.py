"""Onglet « Images en série » : une image par ligne d'un tableau Excel / CSV ou d'une liste (icônes, cartes…)."""
import gradio as gr

from .. import serie
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Beaucoup d'images d'un coup, dans le même style** : par exemple 250 icônes carrées pour une application. "
        "Une image par ligne de ton tableau.\n\n"
        "1. **Dépose ton fichier Excel** (.xlsx) ou CSV, puis choisis la **colonne des prompts** (en anglais de "
        "préférence) ; facultatif : une colonne de **noms** (noms des fichiers) et une colonne de **contexte** propre "
        "à chaque ligne. Pas de fichier ? Colle tes prompts, un par ligne.\n"
        "2. **Contexte commun** : écris-le, ou indique la case du fichier qui le contient (par exemple B1).\n"
        "3. **Style** : choisis-le dans la liste (styles d'icônes en tête) ; pour un rendu encore plus uniforme, "
        "ajoute 1 à 3 **images de style** (une icône que tu aimes) : chaque image en reprend le rendu, pas le sujet.\n"
        "4. « Voir les prompts » pour vérifier, puis « Générer ». Essaie d'abord sur 5 lignes (« Seulement les N "
        "premières ») pour régler le style.\n\n"
        "Compte environ 15 à 20 s par image 1024×1024 sur la RTX 4070 (250 images ≈ 1 h 15). Si le lot s'arrête, "
        "« Reprendre le lot » refait seulement les images manquantes."
    )
    with gr.Row():
        with gr.Column():
            ser_fichier = gr.File(label="Tableau (Excel .xlsx, CSV ou texte)", file_types=[".xlsx", ".xlsm", ".csv",
                                                                                         ".tsv", ".txt"], height=120)
            with gr.Row():
                ser_feuille = gr.Dropdown([], label="Feuille", visible=False)
                ser_entetes = gr.Checkbox(value=True, label="La première ligne contient les titres des colonnes")
            ser_col_prompt = gr.Dropdown([], label="Colonne des prompts")
            with gr.Row():
                ser_col_nom = gr.Dropdown([], label="Colonne des noms (facultatif)")
                ser_col_contexte = gr.Dropdown([], label="Colonne de contexte par ligne (facultatif)")
            ser_info = gr.Markdown("Dépose un fichier, ou colle tes prompts ci-dessous (un par ligne).")
            ser_liste = gr.Textbox(label="Ou colle tes prompts (un par ligne, sans fichier)", lines=5,
                                   placeholder="a red heart\na shopping cart\na gear wheel")
        with gr.Column():
            ser_contexte = gr.Textbox(label="Contexte commun (anglais, ajouté à chaque image)", lines=3,
                                      placeholder="icons for a cooking app, warm orange and cream palette, friendly")
            ser_case_contexte = gr.Textbox(label="Ou case du fichier qui contient le contexte (ex. B1, Feuil1!B1)")
            ser_styles = gr.Dropdown([(lib, val) for lib, val in serie.STYLES], value=[serie.STYLES[0][1]],
                                     multiselect=True, allow_custom_value=True,
                                     label="Style (plusieurs choix ou le tien en anglais)")
            ser_images_style = gr.File(file_count="multiple", file_types=["image"], height=110,
                                       label=f"Images de style (facultatif, {serie.STYLES_IMAGES_MAX} au plus) : "
                                             "leur rendu est repris, pas leur sujet")
            with gr.Row():
                ser_format = gr.Dropdown(list(serie.FORMATS), value=serie.FORMAT_DEFAUT, label="Format")
                ser_tailles = gr.Dropdown(serie.TAILLES, value=[], multiselect=True,
                                          label="Copies réduites (px, pour des icônes)")
            with gr.Row():
                ser_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
                ser_meme_graine = gr.Checkbox(value=True, label="Même graine pour toutes (rendu plus homogène)")
            with gr.Row():
                ser_nom = gr.Textbox(label="Nom du lot", value="icones")
                ser_limite = gr.Number(value=0, precision=0, label="Seulement les N premières (0 = toutes)")
    ser_apercu = gr.Dataframe(label="Aperçu du tableau", interactive=False, wrap=True, max_height=260)
    with gr.Row():
        btn_ser_prompts = gr.Button("👁️ Voir les prompts")
        btn_ser = gr.Button("🎨 Générer toutes les images", variant="primary")
    ser_statut = gr.Markdown()
    ser_dossier = gr.State()
    ser_galerie = gr.Gallery(label="Images", columns=8, height=560, object_fit="contain")
    with gr.Row():
        ser_zip = gr.File(label="Archive zip (images, copies réduites, lot.csv)", interactive=False)
        with gr.Column():
            ser_dossier_lot = gr.Textbox(label="Dossier du lot à reprendre (vide = le dernier lancé ici)",
                                         placeholder="data\\series\\20261007_101500_icones")
            btn_ser_reprendre = gr.Button("▶️ Reprendre le lot (images manquantes)")
            btn_ser_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    sorties_analyse = [c.ser_feuille, c.ser_col_prompt, c.ser_col_nom, c.ser_col_contexte, c.ser_apercu, c.ser_info]
    c.ser_fichier.change(serie.analyser, [c.ser_fichier, c.ser_feuille, c.ser_entetes], sorties_analyse)
    c.ser_feuille.input(serie.analyser, [c.ser_fichier, c.ser_feuille, c.ser_entetes], sorties_analyse)
    c.ser_entetes.input(serie.analyser, [c.ser_fichier, c.ser_feuille, c.ser_entetes], sorties_analyse)
    reglages = [c.ser_fichier, c.ser_feuille, c.ser_entetes, c.ser_col_prompt, c.ser_col_nom, c.ser_col_contexte,
                c.ser_case_contexte, c.ser_liste, c.ser_contexte, c.ser_styles, c.ser_images_style]
    c.btn_ser_prompts.click(serie.apercu_prompts, reglages + [c.ser_limite], [c.ser_apercu, c.ser_statut])
    c.btn_ser.click(serie.generer,
                    reglages + [c.ser_format, c.ser_tailles, c.ser_graine, c.ser_meme_graine, c.ser_nom, c.ser_limite],
                    [c.ser_statut, c.ser_galerie, c.ser_zip, c.ser_dossier])
    c.btn_ser_reprendre.click(lambda texte, dernier, progress=gr.Progress(): serie.reprendre(
        (texte or "").strip().strip('"') or dernier, progress=progress), [c.ser_dossier_lot, c.ser_dossier],
        [c.ser_statut, c.ser_galerie, c.ser_zip, c.ser_dossier])
    c.btn_ser_dossier.click(lambda d: open_folder(d) if d else None, c.ser_dossier)
