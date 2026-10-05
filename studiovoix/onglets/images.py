"""Onglet « Images » : à partir d'un prompt (Z-Image-Turbo) ou d'une photo (FLUX.2 klein)."""
import gradio as gr

from .. import images, videos
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Crée une **image à partir d'un texte**, ou **à partir d'une photo** : la modifier (« mets-lui un chapeau », "
        "« la même scène la nuit »), garder la personne ou l'objet dans une nouvelle scène, ou reprendre sa pose. "
        "Décris ce que tu veux en français ; « Préparer le prompt » le traduit et le précise (Qwen3-VL, qui voit "
        "aussi la photo). Texte seul : Z-Image-Turbo ; avec une photo : FLUX.2 klein 4B (licences Apache 2.0 : "
        "tu peux utiliser les images commercialement). Avec **plusieurs photos**, désigne-les par leur numéro : « la "
        "femme de la photo 1 dans le château de la photo 2 », « mets-lui la veste de la photo 2 ». N'utilise que des "
        "photos que tu as le droit d'utiliser."
    )
    with gr.Row():
        with gr.Column():
            img_texte = gr.Textbox(label="Ce que tu veux (français ou anglais)", lines=3,
                                   placeholder="un phare sur une falaise pendant une tempête, ou avec une photo : "
                                               "« ajoute-lui une cape rouge »")
            img_exemples = gr.Dropdown([(lib, txt) for lib, txt in images.EXEMPLES], label="Exemples", value=None,
                                       allow_custom_value=True)
            btn_img_prep = gr.Button("🧠 Préparer le prompt (traduction et précision)")
            img_prompt = gr.Textbox(label="Prompt envoyé au générateur (anglais, modifiable)", lines=4)
        with gr.Column():
            img_photo = gr.Image(type="filepath", label="Photo (facultative) — photo 1", height=300)
            img_autres = gr.File(file_count="multiple", file_types=["image"], height=120,
                                 label="Autres photos (facultatif, 3 au plus) — photo 2, 3, 4 : un autre personnage, "
                                       "un décor, un objet, un vêtement, un style")
            img_usage = gr.Radio(list(images.USAGES), value=images.USAGE_DEFAUT, label="Ce que l'image reprend de la photo")
    with gr.Row():
        img_styles = gr.Dropdown([(lib, val) for lib, val in images.STYLES], value=[], multiselect=True,
                                 allow_custom_value=True, label="Style (facultatif, plusieurs choix ou le tien en anglais)")
        img_format = gr.Dropdown(list(images.FORMATS), value=images.AUTO, label="Format")
    with gr.Row():
        img_variantes = gr.Radio([1, 2, 3, 4], value=2, label="Variantes")
        img_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
        img_nom = gr.Textbox(label="Nom (pour la galerie)", value="image")
    btn_img = gr.Button("🎨 Générer l'image", variant="primary")
    img_statut = gr.Markdown()
    img_dossier = gr.State()
    img_choisie = gr.State()
    img_galerie = gr.Gallery(label="Variantes", columns=4, height=520, object_fit="contain")
    img_choix_msg = gr.Markdown("Clique sur une variante pour la choisir.")
    with gr.Row():
        btn_img_retoucher = gr.Button("✏️ Retoucher cette image (elle devient la photo)")
        btn_img_animer = gr.Button("🎬 Animer dans l'onglet Vidéos")
        btn_img_dossier = gr.Button("📂 Ouvrir le dossier")
    with gr.Accordion("📖 Une histoire en plusieurs images (et en mini-vidéos)", open=False):
        gr.Markdown(
            "Colle un texte (une histoire, un résumé, une scène), choisis le **nombre d'images** : Qwen3-VL le découpe "
            "en scènes (modifiables, une par ligne), puis chaque scène devient une image. Avec « Mêmes personnages », "
            "les images 2, 3… reprennent les personnages et le style de la première (FLUX.2 klein). La photo, le style, "
            "le format, la graine et le nom ci-dessus sont repris. Tu peux aussi donner des **images de départ** (tes "
            "personnages, un lieu, un style) : Qwen les voit en découpant, et chaque image les reprend (sans elles, "
            "la photo de l'onglet sert d'image de départ). Ensuite, « Faire les mini-vidéos » anime chaque "
            "image (Wan 2.2, ~4 à 7 min par mini-vidéo de 3 s sur la RTX 4070)."
        )
        with gr.Row():
            with gr.Column():
                hist_texte = gr.Textbox(label="Ton texte", lines=6,
                                        placeholder="Trois aventuriers quittent leur village à l'aube…")
                hist_nombre = gr.Slider(2, images.NOMBRE_MAX, value=4, step=1, label="Nombre d'images")
                btn_hist_decouper = gr.Button("✂️ Découper en scènes")
            with gr.Column():
                hist_scenes = gr.Textbox(label="Scènes (une par ligne, anglais, modifiables)", lines=8)
                hist_memes = gr.Checkbox(value=True, label="Mêmes personnages d'une image à l'autre")
        hist_depart = gr.File(file_count="multiple", file_types=["image"], height=140,
                              label=f"Images de départ (facultatif, {images.REFERENCES_MAX} au plus) : tes personnages, "
                                    "une créature, un lieu ou un style à reprendre dans toutes les images")
        btn_hist = gr.Button("🎨 Générer les images de l'histoire", variant="primary")
        hist_statut = gr.Markdown()
        hist_dossier = gr.State()
        hist_galerie = gr.Gallery(label="Images de l'histoire", columns=4, height=520, object_fit="contain")
        with gr.Row():
            hist_duree = gr.Radio(["2 s", "3 s"], value="3 s", label="Durée de chaque mini-vidéo")
            hist_etapes = gr.Slider(10, 40, value=20, step=1, label="Pas (plus = plus fin, plus long)")
            hist_assembler = gr.Checkbox(value=True, label="Assembler en une seule vidéo")
        hist_mouvement = gr.Textbox(label="Mouvement (facultatif, anglais ; vide = la scène s'anime doucement)",
                                    placeholder=videos.MOUVEMENT_DEFAUT)
        btn_hist_videos = gr.Button("🎬 Faire les mini-vidéos")
        hist_vid_statut = gr.Markdown()
        hist_video = gr.Video(label="Mini-vidéos", height=420)
    return espace_de_noms(locals())


def _retoucher(choisie):
    if not choisie:
        raise gr.Error("Clique d'abord sur une variante.")
    return choisie, images.USAGE_DEFAUT, "", ""


def _animer(choisie):
    if not choisie:
        raise gr.Error("Clique d'abord sur une variante.")
    gr.Info("Image envoyée dans l'onglet « Vidéos » (Image et vidéo → Vidéos) comme image de départ.")
    return choisie



# photo principale + autres photos (numérotées 1, 2, 3, 4 dans l'ordre)
def _preparer(description, photo, autres, usage, progress=gr.Progress()):
    return images.preparer(description, images.photos_de(photo, autres), usage, progress=progress)


def _prompt_pret(description, prompt, photo, autres, usage, progress=gr.Progress()):
    return images.prompt_pret(description, prompt, images.photos_de(photo, autres), usage, progress=progress)


def _generer(prompt, photo, autres, usage, styles, format_label, variantes, graine, nom, description,
             progress=gr.Progress()):
    return images.generer(prompt, images.photos_de(photo, autres), usage, styles, format_label, variantes, graine,
                          nom, description, progress=progress)


# images de départ de l'histoire : celles du volet, sinon la photo de l'onglet
def _decouper(histoire, nombre, depart, photo, progress=gr.Progress()):
    return images.decouper(histoire, nombre, images.images_de_depart(depart, photo), progress=progress)


def _scenes_pretes(histoire, scenes, nombre, depart, photo, progress=gr.Progress()):
    return images.scenes_pretes(histoire, scenes, nombre, images.images_de_depart(depart, photo), progress=progress)


def _generer_histoire(scenes, depart, photo, styles, format_label, memes, graine, nom, histoire,
                      progress=gr.Progress()):
    return images.generer_histoire(scenes, images.images_de_depart(depart, photo), styles, format_label, memes,
                                   graine, nom, histoire, progress=progress)


def _mini_videos(dossier, duree, etapes, mouvement, assembler, nom, progress=gr.Progress()):
    msg, video, _ = videos.generer_depuis_images(dossier, duree, etapes, mouvement, assembler, nom, progress=progress)
    return msg, video


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.img_exemples.change(lambda v: v or "", c.img_exemples, c.img_texte)
    c.btn_img_prep.click(_preparer, [c.img_texte, c.img_photo, c.img_autres, c.img_usage], c.img_prompt)
    # prompt vide (ou court, tapé à la main) : préparé d'abord, puis l'image (seulement si la préparation a réussi)
    c.btn_img.click(_prompt_pret, [c.img_texte, c.img_prompt, c.img_photo, c.img_autres, c.img_usage],
                    c.img_prompt).success(
        _generer, [c.img_prompt, c.img_photo, c.img_autres, c.img_usage, c.img_styles, c.img_format, c.img_variantes,
                   c.img_graine, c.img_nom, c.img_texte], [c.img_statut, c.img_galerie, c.img_dossier])
    c.img_galerie.select(images.choisir, c.img_dossier, [c.img_choix_msg, c.img_choisie])
    c.btn_img_retoucher.click(_retoucher, c.img_choisie, [c.img_photo, c.img_usage, c.img_texte, c.img_prompt])
    c.btn_img_animer.click(_animer, c.img_choisie, o.videos.vid_image)
    c.btn_img_dossier.click(lambda d: open_folder(d) if d else None, c.img_dossier)
    c.btn_hist_decouper.click(_decouper, [c.hist_texte, c.hist_nombre, c.hist_depart, c.img_photo], c.hist_scenes)
    c.btn_hist.click(_scenes_pretes, [c.hist_texte, c.hist_scenes, c.hist_nombre, c.hist_depart, c.img_photo],
                     c.hist_scenes).success(
        _generer_histoire, [c.hist_scenes, c.hist_depart, c.img_photo, c.img_styles, c.img_format, c.hist_memes,
                            c.img_graine, c.img_nom, c.hist_texte], [c.hist_statut, c.hist_galerie, c.hist_dossier])
    c.btn_hist_videos.click(_mini_videos,
                            [c.hist_dossier, c.hist_duree, c.hist_etapes, c.hist_mouvement, c.hist_assembler, c.img_nom],
                            [c.hist_vid_statut, c.hist_video])
