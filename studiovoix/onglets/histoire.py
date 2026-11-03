"""Onglet « Histoire en images » : un texte ou des descriptions → plusieurs images (les mini-vidéos par image ont
leur onglet dans le groupe Vidéo)."""
import gradio as gr

from .. import images
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Plusieurs images qui racontent une histoire**, avec les mêmes personnages d'une image à l'autre (puis, "
        "si tu veux, une mini-vidéo par image : onglet « 🎞️ Mini-vidéos d'une histoire » du groupe Vidéo).\n\n"
        "1. **Écris tout toi-même, en anglais** : un **contexte commun** (envoyé tel quel avec chaque image : les "
        "personnages, leur nombre et leur genre — « four bearded men, all male » —, leur apparence, le lieu, "
        "l'époque), puis, pour chaque image, ce qu'elle montre. Une image laissée vide reprend le contexte seul.\n"
        "2. **Ou colle un texte** (français ou anglais) et « Découper en scènes » : Qwen3-VL remplit les champs "
        "« Image », que tu peux retoucher.\n\n"
        "Avec « Mêmes personnages », les images 2, 3… reprennent les personnages de la première (FLUX.2 klein). Des "
        "**images de départ** (tes personnages, un lieu, un style) sont reprises dans toutes les images. Évite les "
        "négations (« no women ») : le générateur a tendance à dessiner le mot nié."
    )
    with gr.Row():
        with gr.Column():
            hist_texte = gr.Textbox(label="Ton texte (facultatif si tu décris chaque image)", lines=6,
                                    placeholder="Trois aventuriers quittent leur village à l'aube…")
            hist_nombre = gr.Slider(2, images.NOMBRE_MAX, value=4, step=1, label="Nombre d'images")
            btn_hist_decouper = gr.Button("✂️ Découper en scènes (remplit les champs « Image »)")
            hist_depart = gr.File(file_count="multiple", file_types=["image"], height=140,
                                  label=f"Images de départ (facultatif, {images.REFERENCES_MAX} au plus) : tes "
                                        "personnages, une créature, un lieu ou un style à reprendre partout")
            hist_memes = gr.Checkbox(value=True, label="Mêmes personnages d'une image à l'autre")
            hist_styles = gr.Dropdown([(lib, val) for lib, val in images.STYLES], value=[], multiselect=True,
                                      allow_custom_value=True,
                                      label="Style (facultatif, plusieurs choix ou le tien en anglais)")
            with gr.Row():
                hist_format = gr.Dropdown(list(images.FORMATS), value=images.AUTO, label="Format")
                hist_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
                hist_nom = gr.Textbox(label="Nom (pour la galerie)", value="histoire")
        with gr.Column():
            hist_contexte = gr.Textbox(
                label="Contexte commun à toutes les images (anglais, envoyé tel quel)", lines=4,
                placeholder="Four bearded men, all male, in their forties, medieval adventurers in leather "
                            "armor: a tall blond warrior, a bald dwarf-like smith, a red-haired archer, an old "
                            "grey-bearded wizard. Misty northern forest, autumn.")
            champs_images = {}  # hist_image_1 … hist_image_12, ajoutés à l'espace de noms
            for i in range(1, images.NOMBRE_MAX + 1):
                champs_images[f"hist_image_{i}"] = gr.Textbox(
                    label=f"Image {i} (anglais)", lines=2, visible=i <= 4,
                    placeholder="they leave the village at dawn, seen from behind" if i == 1 else None)
    btn_hist = gr.Button("🎨 Générer les images de l'histoire", variant="primary")
    hist_statut = gr.Markdown()
    hist_dossier = gr.State()
    hist_galerie = gr.Gallery(label="Images de l'histoire", columns=4, height=520, object_fit="contain")
    with gr.Row():
        btn_hist_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms({**locals(), **champs_images})


# *champs = les NOMBRE_MAX champs « Image i » (seuls les `nombre` premiers comptent)
def _decouper(histoire, nombre, depart, contexte, progress=gr.Progress()):
    scenes = images.decouper(histoire, nombre, images.images_de_depart(depart), contexte, progress=progress)
    return images.champs_images(scenes, nombre)


def _scenes_pretes(histoire, nombre, depart, contexte, *champs, progress=gr.Progress()):
    pretes = images.scenes_pretes(histoire, list(champs), nombre, images.images_de_depart(depart), contexte,
                                  progress=progress)
    return images.champs_images(pretes, nombre)


def _generer_histoire(depart, styles, format_label, memes, graine, nom, histoire, contexte, nombre, *champs,
                      progress=gr.Progress()):
    n = max(2, min(images.NOMBRE_MAX, int(nombre or 4)))
    return images.generer_histoire(list(champs[:n]), images.images_de_depart(depart), styles, format_label,
                                   memes, graine, nom, histoire, contexte, progress=progress)


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    champs = [getattr(c, f"hist_image_{i}") for i in range(1, images.NOMBRE_MAX + 1)]
    c.hist_nombre.change(images.maj_champs_images, c.hist_nombre, champs)
    c.btn_hist_decouper.click(_decouper, [c.hist_texte, c.hist_nombre, c.hist_depart, c.hist_contexte], champs)
    # champs vides : découpés d'abord depuis le texte (affichés), puis les images (seulement si c'est prêt)
    c.btn_hist.click(_scenes_pretes, [c.hist_texte, c.hist_nombre, c.hist_depart, c.hist_contexte, *champs],
                     champs).success(
        _generer_histoire, [c.hist_depart, c.hist_styles, c.hist_format, c.hist_memes, c.hist_graine, c.hist_nom,
                            c.hist_texte, c.hist_contexte, c.hist_nombre, *champs],
        [c.hist_statut, c.hist_galerie, c.hist_dossier]).then(
        lambda d: gr.update(value=d), c.hist_dossier, o.mini_videos.mv_histoire)  # proposée dans l'onglet des mini-vidéos
    c.btn_hist_dossier.click(lambda d: open_folder(d) if d else None, c.hist_dossier)
