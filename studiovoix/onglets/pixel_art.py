"""Onglet « Pixel art et sprites » : convertir des images en pixel art, faire des personnages de jeu d'après une
photo, kit de dresseur (sprite de face, feuilles de lancer et de marche), animer un sprite."""
import gradio as gr

from .. import dresseur, pixel_art
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Pixel art et sprites de jeu.** Donne quelques sprites d'exemple : leur style (taille de la grille, hauteur "
        "du personnage, nombre de couleurs, contour) est lu et appliqué à tout ce qui est créé ici. Puis : convertis "
        "n'importe quelle image, transforme les personnes d'une photo en personnages (avec tes modifications), et "
        "anime un sprite sur 30 images en 5 secondes, en boucle."
    )
    with gr.Accordion("🎨 Style : sprites d'exemple", open=True):
        with gr.Row():
            px_exemples = gr.File(file_count="multiple", file_types=["image"],
                                  label="Sprites d'exemple (PNG, ou GIF animés : leur rythme sert aux animations)")
            with gr.Column():
                btn_px_style = gr.Button("🔍 Lire le style des exemples")
                px_style_msg = gr.Markdown()
        with gr.Row():
            px_hauteur = gr.Number(value=0, precision=0, label="Hauteur du personnage (px, 0 = d'après les exemples)")
            px_couleurs = gr.Number(value=0, precision=0, label="Nombre de couleurs (0 = d'après les exemples)")
            px_toile = gr.Number(value=0, precision=0, label="Taille du sprite (px, 0 = d'après les exemples)")
            px_agrandi = gr.Number(value=0, precision=0, label="Agrandissement à l'export (0 = d'après les exemples)")
            px_contour = gr.Checkbox(value=True, label="Contour sombre d'un pixel")
    gr.Markdown("### 🖼️ Convertir des images en pixel art")
    with gr.Row():
        px_images = gr.File(file_count="multiple", file_types=["image"], label="Images à convertir")
        with gr.Column():
            px_detourer = gr.Checkbox(value=True, label="Retirer le fond (BiRefNet) avant la conversion",
                                      info="Sinon, seul un fond uni est retiré.")
            px_nom = gr.Textbox(label="Nom du dossier", value="pixel_art")
            btn_px_convertir = gr.Button("👾 Convertir", variant="primary")
    gr.Markdown("### 🧙 Personnages d'après une photo")
    with gr.Row():
        px_photo = gr.Image(type="filepath", label="Photo des personnes", height=300)
        with gr.Column():
            px_personnes = gr.Textbox(
                label="Un personnage par ligne : qui, et ce que tu changes", lines=5,
                placeholder="l'homme barbu à lunettes, en dresseur avec un sweat noir et rouge, une balle à la main\n"
                            "la femme aux cheveux bouclés, en robe blanche avec un chapeau de paille et une valise")
            with gr.Row():
                px_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
                px_nom_perso = gr.Textbox(label="Nom du dossier", value="personnages")
            btn_px_perso = gr.Button("🧙 Créer les personnages", variant="primary")
    gr.Markdown("### 🎒 Kit de dresseur : sprite de face, feuille de lancer, feuille de marche")
    gr.Markdown(
        "**Sprite de face** : un modèle entraîné sur des sprites de dresseurs (SDXL + LoRA, étiquette de "
        "génération « gen3 » = Rouge Feu / Vert Feuille) dessine la personne décrite par Qwen3-VL d'après la "
        "photo (cheveux, lunettes, barbe, tenue, avec tes changements), en plusieurs variantes ; chacune est "
        "ramenée à la grille et aux couleurs du jeu. **Feuilles de lancer et de marche** : les sprites d'exemple "
        "sont gardés au pixel près et recolorés aux couleurs du sprite de face. « Animer un sprite » donne ensuite "
        "sa feuille d'animation au sprite de face.")
    _style0 = next(iter(dresseur.STYLES))
    with gr.Row():
        kit_photo = gr.Image(type="filepath", label="Photo de la personne", height=300)
        with gr.Column():
            kit_style = gr.Radio(list(dresseur.STYLES), value=_style0, label="Style")
            kit_demande = gr.Textbox(
                label="Qui, et ce que tu changes", lines=2,
                placeholder="la femme aux cheveux bouclés, en dresseuse avec une veste rouge et un short bleu")
            with gr.Row():
                kit_couleurs = gr.Textbox(label="Couleurs de la personne (lues sur la photo, modifiables)", lines=2,
                                          placeholder=dresseur.COULEURS_DEFAUT, scale=3)
                btn_kit_couleurs = gr.Button("🎨 Lire les couleurs sur la photo", scale=1)
            kit_feuilles = gr.CheckboxGroup(dresseur.FEUILLES, value=dresseur.FEUILLES, label="À créer")
            kit_mode_face = gr.Radio(dresseur.MODES_FACE, value=dresseur.LORA, label="Sprite de face")
            _loras = [dresseur.MARCHE_EXEMPLE] + list(dresseur.modele_sprites.loras())
            kit_lora_marche = gr.Dropdown(_loras, value=_loras[-1] if len(_loras) > 1 else dresseur.MARCHE_EXEMPLE,
                                          label="Feuille de marche : modèle entraîné (onglet « Entraîner un modèle "
                                                "de sprites ») ou recoloriage d'un exemple")
            with gr.Row():
                kit_variantes = gr.Slider(1, dresseur.VARIANTES_MAX, value=2, step=1,
                                          label="Variantes du sprite de face (modèle de sprites)")
                kit_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
                kit_nom = gr.Textbox(label="Nom du dossier", value="dresseur")
            btn_kit = gr.Button("🎒 Créer le kit de dresseur", variant="primary")
    with gr.Accordion("📄 Sprites d'exemple (le style et la silhouette)", open=True):
        with gr.Row():
            listes = {}
            apercus_ex = {}
            for _f, _lib in (("face", "Sprite de face"), ("lancer", "Lancer (de dos)"), ("marche", "Marche")):
                with gr.Column():
                    _noms = dresseur.liste_gabarits(_style0, _f)
                    listes[_f] = gr.Dropdown(_noms, value=_noms[0] if _noms else None, label=f"Exemple : {_lib}")
                    apercus_ex[_f] = gr.Image(label="Aperçu", height=160, interactive=False,
                                             value=dresseur.apercu_gabarit(_style0, _f, _noms[0]) if _noms else None)
        kit_ex_face, kit_ex_lancer, kit_ex_marche = listes["face"], listes["lancer"], listes["marche"]
        kit_ap_face, kit_ap_lancer, kit_ap_marche = apercus_ex["face"], apercus_ex["lancer"], apercus_ex["marche"]
        with gr.Row():
            kit_ajout = gr.File(file_count="multiple", file_types=["image"],
                                label="Ajouter des sprites d'exemple (PNG ou GIF du jeu, agrandis ou non)")
            with gr.Column():
                kit_ajout_type = gr.Radio(dresseur.FEUILLES, value=dresseur.FACE, label="Ce sont des…",
                                          info="Lancer : une rangée d'images carrées. Marche : la feuille d'un "
                                               "seul personnage (GBA : 4×4 cases, DS : 3×4), sans marge.")
                btn_kit_ajout = gr.Button("➕ Ajouter à la bibliothèque")
                kit_ajout_msg = gr.Markdown()
    px_statut = gr.Markdown()
    px_dossier = gr.State()
    px_apercus = gr.State([])
    px_galerie = gr.Gallery(label="Sprites (clique sur un sprite pour l'animer)", columns=6, height=300,
                            object_fit="contain")
    px_fichiers = gr.File(label="Fichiers produits", file_count="multiple", interactive=False)
    gr.Markdown("### 🎞️ Animer un sprite")
    with gr.Row():
        px_sprite = gr.Image(type="filepath", image_mode="RGBA", label="Sprite à animer", height=260)
        with gr.Column():
            px_mode = gr.Radio(pixel_art.MODES_ANIMATION, value=pixel_art.SIMPLE, label="Animation")
            px_mouvement = gr.Textbox(label="Mouvement (facultatif, Wan 2.2)", lines=2,
                                      placeholder="il lance sa balle d'une main à l'autre, sa cape flotte",
                                      info="Décrit : un geste joué une fois, au rythme des sprites animés "
                                           "(50 ms par image, pose finale tenue). Vide : petits mouvements sur "
                                           "place, en boucle. Une dizaine de minutes.")
            with gr.Row():
                px_anim_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
                px_anim_etapes = gr.Slider(10, 50, value=25, step=1, label="Étapes (Wan 2.2)")
                px_anim_nom = gr.Textbox(label="Nom", value="animation")
            btn_px_animer = gr.Button("🎞️ Animer", variant="primary")
    px_anim_statut = gr.Markdown()
    with gr.Row():
        px_anim = gr.Image(label="Animation (GIF)", height=260, interactive=False)
        px_anim_fichiers = gr.File(label="GIF, planche de sprites, images", file_count="multiple", interactive=False)
    btn_px_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def _choisir(apercus, evt: gr.SelectData):
    """Sprite cliqué dans la galerie → à animer."""
    return apercus[evt.index] if apercus and evt.index < len(apercus) else None


def _convertir(*args, progress=gr.Progress()):
    msg, apercus, fichiers, dossier = pixel_art.convertir(*args, progress=progress)
    return msg, apercus, fichiers, dossier, apercus


def _personnages(*args, progress=gr.Progress()):
    msg, apercus, fichiers, dossier, sprite = pixel_art.creer_personnages(*args, progress=progress)
    return msg, apercus, fichiers, dossier, apercus, sprite


def _kit(*args, progress=gr.Progress()):
    msg, apercus, fichiers, dossier, sprite, couleurs = dresseur.creer_kit(*args, progress=progress)
    return msg, apercus, fichiers, dossier, apercus, sprite, couleurs


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    reglages = [c.px_exemples, c.px_hauteur, c.px_couleurs, c.px_toile, c.px_agrandi, c.px_contour]
    c.btn_px_style.click(pixel_art.lire_style, c.px_exemples,
                         [c.px_hauteur, c.px_couleurs, c.px_toile, c.px_agrandi, c.px_style_msg])
    sorties = [c.px_statut, c.px_galerie, c.px_fichiers, c.px_dossier, c.px_apercus]
    c.btn_px_convertir.click(_convertir, [c.px_images, *reglages, c.px_detourer, c.px_nom], sorties)
    c.btn_px_perso.click(_personnages, [c.px_photo, c.px_personnes, *reglages, c.px_graine, c.px_nom_perso],
                         [*sorties, c.px_sprite])
    listes = [c.kit_ex_face, c.kit_ex_lancer, c.kit_ex_marche]
    apercus = [c.kit_ap_face, c.kit_ap_lancer, c.kit_ap_marche]
    c.kit_style.change(dresseur.mises_a_jour_listes, c.kit_style, listes)
    for liste, apercu, feuille in zip(listes, apercus, ("face", "lancer", "marche")):
        liste.change(lambda style, nom, f=feuille: dresseur.apercu_gabarit(style, f, nom), [c.kit_style, liste],
                     apercu)
    c.btn_kit_ajout.click(dresseur.ajouter_gabarits, [c.kit_style, c.kit_ajout_type, c.kit_ajout],
                          [c.kit_ajout_msg, *listes])
    c.btn_kit_couleurs.click(dresseur.lire_couleurs, [c.kit_photo, c.kit_demande], c.kit_couleurs)
    c.btn_kit.click(_kit, [c.kit_style, c.kit_photo, c.kit_demande, c.kit_couleurs, c.kit_feuilles, c.kit_mode_face,
                           *listes, c.kit_graine, c.kit_nom, c.kit_variantes, c.kit_lora_marche],
                    [*sorties, c.px_sprite, c.kit_couleurs])
    c.px_galerie.select(_choisir, c.px_apercus, c.px_sprite)
    c.btn_px_animer.click(pixel_art.animer, [c.px_sprite, c.px_mode, c.px_mouvement, c.px_anim_graine,
                                            c.px_anim_etapes, c.px_anim_nom, c.px_exemples],
                          [c.px_anim_statut, c.px_anim, c.px_anim_fichiers, c.px_dossier])
    c.btn_px_dossier.click(lambda d: open_folder(d) if d else None, c.px_dossier)
