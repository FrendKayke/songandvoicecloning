"""Onglet « Musiques en série » : une musique par ligne d'un tableau Excel / CSV ou d'une liste, dans le même style."""
import gradio as gr

from .. import acestep, export, serie_musique
from .. import config as cfg
from ..outils import open_folder
from .commun import espace_de_noms, liste_style


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Beaucoup de musiques d'un coup, dans le même style** : les musiques d'un quiz, les ambiances d'un jeu… "
        "Une musique par ligne de ton tableau.\n\n"
        "1. **Dépose ton fichier Excel** (.xlsx) ou CSV, puis choisis la **colonne des descriptions** (ce que la "
        "musique doit évoquer, en français ou en anglais : « combat contre le boss final », « menu calme ») ; "
        "facultatif : une colonne de **noms de fichiers**, une de **paroles** (musiques chantées) et une de "
        "**durées** (« 90 », « 1:30 »). Pas de fichier ? Colle tes descriptions, une par ligne.\n"
        "2. **Style commun** : choisis-le dans les listes (comme pour une chanson), ou écris-le. Pour une série "
        "encore plus homogène, ajoute une **musique de référence** : toutes en reprennent le son (timbre, mixage).\n"
        "3. « Voir les descriptions » pour vérifier, puis « Générer ». Essaie d'abord sur 3 lignes.\n\n"
        "Compte environ 30 s à 1 min par musique d'une minute sur la RTX 4070. Si le lot s'arrête, « Reprendre le "
        "lot » refait seulement les musiques manquantes."
    )
    with gr.Row():
        with gr.Column():
            msr_fichier = gr.File(label="Tableau (Excel .xlsx, CSV ou texte)",
                                  file_types=[".xlsx", ".xlsm", ".csv", ".tsv", ".txt"], height=120)
            with gr.Row():
                msr_feuille = gr.Dropdown([], label="Feuille", visible=False)
                msr_entetes = gr.Checkbox(value=True, label="La première ligne contient les titres des colonnes")
            msr_col_desc = gr.Dropdown([], label="Colonne des descriptions")
            with gr.Row():
                msr_col_nom = gr.Dropdown([], label="Noms de fichiers (facultatif)",
                                          info="Chaque musique porte exactement ce nom ; le format vient des cases "
                                               "« Formats » ci-contre.")
                msr_col_paroles = gr.Dropdown([], label="Paroles (facultatif)")
                msr_col_duree = gr.Dropdown([], label="Durée (facultatif)")
            msr_numeroter = gr.Checkbox(value=False, label="Ajouter le numéro de ligne devant le nom (001_…)")
            msr_info = gr.Markdown("Dépose un fichier, ou colle tes descriptions ci-dessous (une par ligne).")
            msr_liste = gr.Textbox(label="Ou colle tes descriptions (une par ligne, sans fichier)", lines=5,
                                   placeholder="combat contre le boss final\nmenu principal calme\nvictoire, fanfare")
        with gr.Column():
            msr_genre = liste_style("genre", "Genre")
            msr_style = liste_style("style", "Style / époque / production")
            msr_instruments = liste_style("instruments", "Instruments")
            msr_ambiance = liste_style("ambiance", "Ambiance (commune à toutes)")
            msr_extra = liste_style("extra", "Autres consignes (facultatif)")
            msr_style_libre = gr.Textbox(label="Style commun en plus (texte libre, anglais de préférence)",
                                         placeholder="orchestral video game soundtrack, 16-bit synths")
            msr_reference = gr.Audio(type="filepath", label="Musique de référence (facultatif) : son commun à "
                                                             "toute la série")
    with gr.Row():
        msr_mode = gr.Radio(serie_musique.MODES, value=serie_musique.MODE_INSTRU, label="Musiques")
        msr_voix_base = gr.Dropdown(list(acestep.VOIX_CHANTEES), value="Automatique", label="Voix chantées",
                                    visible=False)
        msr_langue = gr.Dropdown(list(cfg.LANGUES), value="Français", label="Langue des paroles", visible=False)
    with gr.Row():
        msr_duree = gr.Slider(serie_musique.DUREE_MIN, 240, value=60, step=5,
                              label="Durée par défaut (s), si la colonne des durées est vide")
        msr_versions = gr.Radio([1, 2], value=1, label="Versions par ligne")
        msr_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
    with gr.Row():
        msr_traduire = gr.Checkbox(value=True, label="Traduire les descriptions en anglais (Qwen3-VL)",
                                   info="ACE-Step comprend mieux l'anglais ; une seule traduction pour tout le lot.")
        msr_thinking = gr.Checkbox(value=True, label="Mode réflexion (LM) — meilleure structure")
    with gr.Row():
        msr_formats = gr.CheckboxGroup(serie_musique.FORMATS, value=["wav", "mp3"], label="Formats")
        msr_cible = gr.Dropdown(list(export.CIBLES), value=list(export.CIBLES)[1], label="Volume (le même pour toutes)")
        msr_nom = gr.Textbox(label="Nom du lot", value="musiques")
        msr_limite = gr.Number(value=0, precision=0, label="Seulement les N premières (0 = toutes)")
    msr_apercu = gr.Dataframe(label="Aperçu", interactive=False, wrap=True, max_height=260)
    with gr.Row():
        btn_msr_apercu = gr.Button("👁️ Voir les descriptions")
        btn_msr = gr.Button("🎵 Générer toutes les musiques", variant="primary")
    msr_statut = gr.Markdown()
    msr_dossier = gr.State()
    msr_choix = gr.Dropdown([], label="Musiques créées (choisis-en une pour l'écouter)")
    msr_ecoute = gr.Audio(label="Écoute", type="filepath", interactive=False)
    with gr.Row():
        msr_zip = gr.File(label="Archive zip (musiques, lot.csv)", interactive=False)
        with gr.Column():
            msr_dossier_lot = gr.Textbox(label="Dossier du lot à reprendre (vide = le dernier lancé ici)",
                                         placeholder="data\\musiques_serie\\20261007_101500_musiques")
            btn_msr_reprendre = gr.Button("▶️ Reprendre le lot (musiques manquantes)")
            btn_msr_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    entrees_analyse = [c.msr_fichier, c.msr_feuille, c.msr_entetes]
    sorties_analyse = [c.msr_feuille, c.msr_col_desc, c.msr_col_nom, c.msr_col_paroles, c.msr_col_duree,
                       c.msr_apercu, c.msr_info]
    c.msr_fichier.change(serie_musique.analyser, entrees_analyse, sorties_analyse)
    c.msr_feuille.input(serie_musique.analyser, entrees_analyse, sorties_analyse)
    c.msr_entetes.input(serie_musique.analyser, entrees_analyse, sorties_analyse)
    c.msr_mode.change(lambda m: (gr.update(visible=m == serie_musique.MODE_CHANT),) * 2, c.msr_mode,
                      [c.msr_voix_base, c.msr_langue])
    tableau = [c.msr_fichier, c.msr_feuille, c.msr_entetes, c.msr_col_desc, c.msr_col_nom, c.msr_col_paroles,
               c.msr_col_duree, c.msr_liste]
    style = [c.msr_genre, c.msr_style, c.msr_instruments, c.msr_ambiance, c.msr_extra, c.msr_style_libre,
             c.msr_voix_base, c.msr_mode]
    c.btn_msr_apercu.click(serie_musique.apercu, tableau + style + [c.msr_duree, c.msr_limite, c.msr_numeroter],
                           [c.msr_apercu, c.msr_statut])
    sorties = [c.msr_statut, c.msr_choix, c.msr_ecoute, c.msr_zip, c.msr_dossier]
    c.btn_msr.click(serie_musique.generer,
                    tableau + style + [c.msr_langue, c.msr_duree, c.msr_reference, c.msr_traduire, c.msr_thinking,
                                       c.msr_versions, c.msr_graine, c.msr_formats, c.msr_cible, c.msr_nom,
                                       c.msr_limite, c.msr_numeroter], sorties)
    c.msr_choix.input(lambda f: f, c.msr_choix, c.msr_ecoute)
    c.btn_msr_reprendre.click(lambda texte, dernier, progress=gr.Progress(): serie_musique.reprendre(
        (texte or "").strip().strip('"') or dernier, progress=progress), [c.msr_dossier_lot, c.msr_dossier], sorties)
    c.btn_msr_dossier.click(lambda d: open_folder(d) if d else None, c.msr_dossier)
