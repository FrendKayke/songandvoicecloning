"""Onglet « Kit de dresseur » : sprite de face, feuille de lancer et feuille de marche d'une personne (photo ou
description), dans le style d'un jeu de rôle portable ; bibliothèque des sprites d'exemple."""
import gradio as gr

from .. import dresseur
from ..outils import open_folder
from .commun import espace_de_noms
from .pixel_commun import choisir


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Kit de dresseur : sprite de face, feuille de lancer, feuille de marche.** "
        "**Sprite de face** : un modèle entraîné sur des sprites de dresseurs (SDXL + LoRA, étiquette de "
        "génération « gen3 » = Rouge Feu / Vert Feuille) dessine la personne décrite par Qwen3-VL d'après la "
        "photo (cheveux, lunettes, barbe, tenue, avec tes changements), en plusieurs variantes ; chacune est "
        "ramenée à la grille et aux couleurs du jeu. **Feuilles de lancer et de marche** : les sprites d'exemple "
        "sont gardés au pixel près et recolorés aux couleurs du sprite de face. Clique sur une variante pour "
        "l'envoyer dans « Animer un sprite ». **Sans photo** : décris simplement le personnage (en français), "
        "puis « 📝 Préparer la description » : Qwen3-VL l'écrit en anglais pour le modèle de sprites ; modifie-la à ta "
        "guise (couleur d'un vêtement, coiffure, accessoire) avant de créer le kit, elle fait foi pour toutes les "
        "feuilles.")
    gr.Markdown(
        "**Sprite de face** : un modèle entraîné sur des sprites de dresseurs (SDXL + LoRA, étiquette de "
        "génération « gen3 » = Rouge Feu / Vert Feuille) dessine la personne décrite par Qwen3-VL d'après la "
        "photo (cheveux, lunettes, barbe, tenue, avec tes changements), en plusieurs variantes ; chacune est "
        "ramenée à la grille et aux couleurs du jeu. **Feuilles de lancer et de marche** : les sprites d'exemple "
        "sont gardés au pixel près et recolorés aux couleurs du sprite de face. « Animer un sprite » donne ensuite "
        "sa feuille d'animation au sprite de face. **Sans photo** : décris simplement le personnage (en français), "
        "puis « 📝 Préparer la description » : Qwen3-VL l'écrit en anglais pour le modèle de sprites ; modifie-la à ta "
        "guise (couleur d'un vêtement, coiffure, accessoire) avant de créer le kit, elle fait foi pour toutes les "
        "feuilles.")
    _style0 = next(iter(dresseur.STYLES))
    with gr.Row():
        kit_photo = gr.Image(type="filepath", label="Photo de la personne (facultative)", height=300)
        with gr.Column():
            kit_style = gr.Radio(list(dresseur.STYLES), value=_style0, label="Style")
            kit_demande = gr.Textbox(
                label="Avec une photo : qui, et ce que tu changes. Sans photo : décris le personnage", lines=2,
                placeholder="la femme aux cheveux bouclés, en dresseuse avec une veste rouge et un short bleu ; ou, "
                            "sans photo : un jeune dresseur blond, casquette verte, t-shirt noir, sac à dos jaune")
            with gr.Row():
                kit_description = gr.Textbox(
                    label="Description du personnage (anglais, modifiable ; vide = préparée à la création ; vidée "
                          "quand tu changes de photo)", lines=3,
                    placeholder="a young man with short blond hair, green cap, black t-shirt and yellow backpack",
                    scale=3)
                btn_kit_description = gr.Button("📝 Préparer la description", scale=1)
            with gr.Row():
                kit_couleurs = gr.Textbox(label="Couleurs de la personne (lues sur la photo ou la description, "
                                                "modifiables)", lines=2, placeholder=dresseur.COULEURS_DEFAUT, scale=3)
                btn_kit_couleurs = gr.Button("🎨 Lire les couleurs", scale=1)
            kit_feuilles = gr.CheckboxGroup(dresseur.FEUILLES, value=dresseur.FEUILLES, label="À créer")
            kit_mode_face = gr.Radio(dresseur.MODES_FACE, value=dresseur.LORA, label="Sprite de face")
            _faces = dresseur.choix_loras_face()
            kit_lora_face = gr.Dropdown(_faces, value=_faces[1] if len(_faces) > 1 else dresseur.FACE_SWIZAD,
                                        label="Modèle du sprite de face : LoRA de sWizad ou modèle entraîné sur tes "
                                              "sprites de face (onglet « Entraîner un modèle de sprites »)")
            _loras = [dresseur.MARCHE_EXEMPLE] + list(dresseur.modele_sprites.loras(dresseur.modele_sprites.MARCHE))
            kit_lora_marche = gr.Dropdown(_loras, value=_loras[1] if len(_loras) > 1 else dresseur.MARCHE_EXEMPLE,
                                          label="Feuille de marche : modèle entraîné (onglet « Entraîner un modèle "
                                                "de sprites ») ou recoloriage d'un exemple")
            with gr.Row():
                kit_variantes = gr.Slider(1, dresseur.VARIANTES_MAX, value=2, step=1,
                                          label="Variantes du sprite de face (modèle de sprites)")
                kit_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
                kit_nom = gr.Textbox(label="Nom du dossier", value="dresseur")
            btn_kit = gr.Button("🎒 Créer le kit de dresseur", variant="primary")
    with gr.Accordion("📄 Sprites d'exemple (le style et la silhouette)", open=True):
        gr.Markdown(
            "Les exemples ne servent qu'aux feuilles **recolorées** : le sprite de face en mode « Recolorer », la "
            "feuille de lancer, et la feuille de marche quand la liste « Feuille de marche » est sur « recolorer un "
            "sprite d'exemple ». Le sprite de face dessiné par le modèle de sprites et la feuille de marche d'un "
            "modèle entraîné ne les utilisent pas.")
        with gr.Row():
            listes = {}
            apercus_ex = {}
            suppressions = {}
            for _f, _lib in (("face", "Sprite de face"), ("lancer", "Lancer (de dos)"), ("marche", "Marche")):
                with gr.Column():
                    _noms = dresseur.liste_gabarits(_style0, _f)
                    listes[_f] = gr.Dropdown(_noms, value=_noms[0] if _noms else None, label=f"Exemple : {_lib}")
                    apercus_ex[_f] = gr.Image(label="Aperçu", height=160, interactive=False,
                                             value=dresseur.apercu_gabarit(_style0, _f, _noms[0]) if _noms else None)
                    suppressions[_f] = gr.Button("🗑️ Supprimer cet exemple", size="sm")
        kit_ex_face, kit_ex_lancer, kit_ex_marche = listes["face"], listes["lancer"], listes["marche"]
        kit_ap_face, kit_ap_lancer, kit_ap_marche = apercus_ex["face"], apercus_ex["lancer"], apercus_ex["marche"]
        btn_kit_suppr_face, btn_kit_suppr_lancer, btn_kit_suppr_marche = (suppressions["face"], suppressions["lancer"],
                                                                          suppressions["marche"])
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
    px_galerie = gr.Gallery(label="Sprites et feuilles (clique sur un sprite : il est envoyé dans « Animer un sprite »)",
                            columns=6, height=300, object_fit="contain")
    px_fichiers = gr.File(label="Fichiers produits", file_count="multiple", interactive=False)
    btn_px_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def _kit(*args, progress=gr.Progress()):
    *args, description = args
    msg, apercus, fichiers, dossier, sprite, couleurs, description = dresseur.creer_kit(
        *args, description=description, progress=progress)
    return msg, apercus, fichiers, dossier, apercus, sprite, couleurs, description


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    listes = [c.kit_ex_face, c.kit_ex_lancer, c.kit_ex_marche]
    apercus = [c.kit_ap_face, c.kit_ap_lancer, c.kit_ap_marche]
    c.kit_style.change(dresseur.mises_a_jour_listes, c.kit_style, listes)
    for liste, apercu, feuille in zip(listes, apercus, ("face", "lancer", "marche")):
        liste.change(lambda style, nom, f=feuille: dresseur.apercu_gabarit(style, f, nom), [c.kit_style, liste],
                     apercu)
    confirmer = ("(s, n) => (n && confirm('Supprimer l\\'exemple « ' + n + ' » de la bibliothèque ?')) ? [s, n] "
                 ": [s, null]")
    for bouton, liste, feuille in zip((c.btn_kit_suppr_face, c.btn_kit_suppr_lancer, c.btn_kit_suppr_marche), listes,
                                      ("face", "lancer", "marche")):
        bouton.click(lambda style, nom, f=feuille: dresseur.supprimer_gabarit(style, nom, f), [c.kit_style, liste],
                     [c.kit_ajout_msg, liste], js=confirmer)
    c.btn_kit_ajout.click(dresseur.ajouter_gabarits, [c.kit_style, c.kit_ajout_type, c.kit_ajout],
                          [c.kit_ajout_msg, *listes])
    # nouvelle photo : la description et les couleurs de l'ancienne personne ne servent plus (elles sont
    # prioritaires sur la photo : le kit refaisait l'ancien personnage, constaté chez l'utilisateur le 01/11)
    c.kit_photo.change(lambda _: ("", ""), c.kit_photo, [c.kit_description, c.kit_couleurs])
    c.btn_kit_description.click(dresseur.preparer_description, [c.kit_photo, c.kit_demande], c.kit_description)
    c.btn_kit_couleurs.click(dresseur.lire_couleurs, [c.kit_photo, c.kit_demande, c.kit_description], c.kit_couleurs)
    c.btn_kit.click(_kit, [c.kit_style, c.kit_photo, c.kit_demande, c.kit_couleurs, c.kit_feuilles, c.kit_mode_face,
                           *listes, c.kit_graine, c.kit_nom, c.kit_variantes, c.kit_lora_marche, c.kit_lora_face,
                           c.kit_description],
                    [c.px_statut, c.px_galerie, c.px_fichiers, c.px_dossier, c.px_apercus, o.pixel_animer.px_sprite,
                     c.kit_couleurs, c.kit_description]
                    ).then(lambda sprite: sprite, o.pixel_animer.px_sprite, o.yeux.ye_sprite)  # « Refaire les yeux »
    c.px_galerie.select(choisir, c.px_apercus, o.pixel_animer.px_sprite)
    c.btn_px_dossier.click(lambda d: open_folder(d) if d else None, c.px_dossier)
