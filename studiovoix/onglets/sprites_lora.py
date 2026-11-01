"""Onglet « Entraîner un modèle de sprites » : feuilles de marche d'exemple → légendes → LoRA SDXL (kohya), proposé
ensuite dans le kit de dresseur pour générer la feuille de marche d'un nouveau personnage. La marche à suivre et le
type de fichier attendu à chaque étape sont affichés dans l'onglet (`modele_sprites.guide`), d'après le style choisi."""
import inspect

import gradio as gr

from .. import dresseur, modele_sprites
from ..outils import open_folder
from .commun import espace_de_noms

# deux colonnes fixes : « column_count » depuis Gradio 6 (« col_count » y est obsolète et affiche un avertissement)
_COLONNES_FIXES = ({"column_count": (2, "fixed")} if "column_count" in inspect.signature(gr.Dataframe).parameters
                   else {"col_count": (2, "fixed")})


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Apprendre à SDXL les sprites de ton jeu.** Aucun générateur ne dessine seul une feuille de marche cohérente "
        "(16 poses de 16×24 px) : on la lui apprend à partir de tes feuilles. De même, un modèle entraîné sur tes "
        "**sprites de face** les dessine à l'échelle exacte du jeu (yeux, mains et détails au pixel près, mieux appris "
        "avec beaucoup d'exemples). Le modèle obtenu sert ensuite dans le kit de dresseur (onglet Pixel art).")
    _style0 = next(iter(dresseur.STYLES))
    lo_prerequis = gr.Markdown(modele_sprites.etat_prerequis())
    _type0 = next(iter(modele_sprites.TYPES))
    lo_type = gr.Radio(list(modele_sprites.TYPES), value=_type0, label="Ce que le modèle apprend")
    with gr.Accordion("📖 Marche à suivre et fichiers attendus", open=True):
        lo_guide = gr.Markdown(modele_sprites.guide(_style0))
    gr.Markdown("#### 1️⃣ Tes exemples")
    with gr.Row():
        lo_feuilles = gr.File(file_count="multiple",
                              file_types=list(modele_sprites.IMAGES) + list(modele_sprites.ARCHIVES),
                              label=f"Feuilles de marche ou sprites de face d'exemple : PNG ou archive .zip "
                                    f"({modele_sprites.FEUILLES_MIN} au moins ; 50 à 100 feuilles, 100 à 300 sprites "
                                    "de face conseillés)")
        with gr.Column():
            lo_style = gr.Radio(list(dresseur.STYLES), value=_style0, label="Style (grille et taille des sprites)")
            lo_nom = gr.Textbox(label="Nom du modèle", value="marche_gba")
            lo_legender = gr.Checkbox(value=True, label="Légender chaque feuille avec Qwen3-VL (conseillé)")
    lo_verif_msg = gr.Markdown()
    lo_verif = gr.Dataframe(headers=["Fichier", "Taille (px)", "Agrandissement", "Vérification"],
                            datatype=["str", "str", "str", "str"], interactive=False, wrap=True,
                            label="Vérification des fichiers déposés")
    gr.Markdown("#### 2️⃣ Préparer et vérifier les légendes")
    btn_lo_preparer = gr.Button("1️⃣ Préparer les exemples", variant="primary")
    lo_prep_msg = gr.Markdown()
    lo_legendes = gr.Dataframe(headers=["Image", "Légende (anglais, modifiable)"], datatype=["str", "str"],
                               interactive=True, wrap=True, **_COLONNES_FIXES,
                               label="Légendes : corrige-les si besoin, puis enregistre")
    with gr.Row():
        btn_lo_legendes = gr.Button("💾 Enregistrer les légendes")
        btn_lo_relire = gr.Button("🔄 Relire les légendes du modèle")
    lo_leg_msg = gr.Markdown()
    gr.Markdown("#### 3️⃣ Entraîner")
    with gr.Row():
        lo_epoques = gr.Slider(4, 30, value=modele_sprites.EPOQUES_DEFAUT, step=1, label="Époques",
                               info="Passages sur toutes les feuilles (×10 répétitions). 12 ≈ 1 h pour 60 feuilles.")
        lo_dim = gr.Slider(8, 64, value=modele_sprites.DIM_DEFAUT, step=8, label="Dimension du LoRA",
                           info="Plus grand = plus de détails appris, plus de mémoire.")
        lo_graine = gr.Number(value=0, precision=0, label="Graine (0 = 42)")
        btn_lo_entrainer = gr.Button("2️⃣ Entraîner (environ une heure)", variant="primary")
    lo_msg = gr.Markdown()
    lo_fichier = gr.File(label="Modèle entraîné (LoRA SDXL, .safetensors)", interactive=False)
    gr.Markdown("#### 4️⃣ S'en servir : onglet Pixel art → 🎒 Kit de dresseur → liste « Feuille de marche » ou "
                "« Modèle du sprite de face »")
    lo_liste = gr.Markdown(etat_modeles())
    btn_lo_dossier = gr.Button("📂 Ouvrir le dossier des modèles entraînés")
    return espace_de_noms(locals())


def etat_modeles():
    marche = modele_sprites.loras(modele_sprites.MARCHE)
    face = modele_sprites.loras(modele_sprites.FACE)
    if not marche and not face:
        return "Aucun modèle entraîné pour l'instant."
    lignes = []
    if marche:
        lignes.append("**Feuilles de marche** : " + ", ".join(f"`{n}`" for n in marche))
    if face:
        lignes.append("**Sprites de face** : " + ", ".join(f"`{n}`" for n in face))
    return "Modèles entraînés (proposés dans le kit de dresseur) — " + " · ".join(lignes)


def _preparer(feuilles, nom, style, legender, type_, progress=gr.Progress()):
    msg, _ = modele_sprites.preparer(feuilles, nom, style, legender, type_, progress=progress)
    return msg, modele_sprites.legendes(nom)


NOMS_DEFAUT = ("", "marche_gba", "marche_ds", "face_gba", "face_ds")


def _nom_par_defaut(style, type_, nom):
    """Nom proposé d'après le style et le type tant que l'utilisateur garde un nom par défaut."""
    if (nom or "").strip() in NOMS_DEFAUT:
        cle = "ds" if dresseur.STYLES[style]["cle"] == "ds" else "gba"
        return f"{modele_sprites.TYPES.get(type_, modele_sprites.MARCHE)}_{cle}"
    return nom


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    for evt in (c.lo_style.change, c.lo_type.change):
        evt(modele_sprites.guide, [c.lo_style, c.lo_type], c.lo_guide).then(
            _nom_par_defaut, [c.lo_style, c.lo_type, c.lo_nom], c.lo_nom)
    for evt in (c.lo_feuilles.change, c.lo_style.change, c.lo_type.change):
        evt(modele_sprites.verifier_feuilles, [c.lo_feuilles, c.lo_style, c.lo_type], [c.lo_verif_msg, c.lo_verif])
    c.btn_lo_preparer.click(_preparer, [c.lo_feuilles, c.lo_nom, c.lo_style, c.lo_legender, c.lo_type],
                            [c.lo_prep_msg, c.lo_legendes])
    c.btn_lo_legendes.click(modele_sprites.enregistrer_legendes, [c.lo_nom, c.lo_legendes], c.lo_leg_msg)
    c.btn_lo_relire.click(modele_sprites.legendes, c.lo_nom, c.lo_legendes)
    c.btn_lo_entrainer.click(modele_sprites.entrainer, [c.lo_nom, c.lo_epoques, c.lo_dim, c.lo_graine],
                             [c.lo_msg, c.lo_fichier]).then(etat_modeles, None, c.lo_liste).then(
        dresseur.liste_loras_marche, None, o.pixel_art.kit_lora_marche).then(
        dresseur.liste_loras_face, None, o.pixel_art.kit_lora_face)
    demo.load(modele_sprites.etat_prerequis, None, c.lo_prerequis)
    c.btn_lo_dossier.click(lambda: open_folder(modele_sprites.cfg.LORAS_DIR, create=True))
