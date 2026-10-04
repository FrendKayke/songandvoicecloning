"""Onglet « Vidéos » : texte → vidéo et image → vidéo (Wan 2.2 TI2V-5B)."""
import gradio as gr

from .. import videos
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Des **vidéos** de 2 à 5 secondes avec Wan 2.2 (licence Apache 2.0 : tu peux les vendre). Décris la scène "
        "en français, « Préparer le prompt » la traduit et la précise (action, décor, mouvement de caméra) ; tu "
        "peux la retoucher. Ajoute une **image de départ** pour animer une illustration ou une photo. "
        "**C'est long** : compte une dizaine de minutes ou plus pour 5 s en 720p sur une RTX 4070 (davantage la "
        "première fois) ; le format « léger » et moins d'étapes vont plus vite. La vidéo est muette : ajoute "
        "bruitages et musique depuis leurs onglets. ACE-Step est arrêté automatiquement pendant la génération."
    )
    with gr.Row():
        with gr.Column():
            vid_texte = gr.Textbox(label="Description de la vidéo (français ou anglais)", lines=3,
                                   placeholder="un dragon rouge s'envole d'un sommet enneigé, la caméra le suit")
            vid_exemples = gr.Dropdown([(lib, txt) for lib, txt in videos.EXEMPLES], label="Exemples", value=None,
                                       allow_custom_value=True)
            btn_vid_prep = gr.Button("🧠 Préparer le prompt (traduction et précision)")
            vid_prompt = gr.Textbox(label="Prompt envoyé à Wan 2.2 (anglais, modifiable)", lines=4)
        with gr.Column():
            vid_image = gr.Image(type="filepath", label="Image de départ (facultative) : la vidéo part de cette image",
                                 height=320)
    with gr.Row():
        vid_format = gr.Dropdown(list(videos.FORMATS), value=videos.AUTO, label="Format")
        vid_duree = gr.Radio(list(videos.DUREES), value=videos.DUREE_DEFAUT, label="Durée")
        vid_etapes = gr.Slider(10, 50, value=30, step=1, label="Étapes (qualité / temps)",
                               info="30 : bon compromis ; 50 : réglage d'origine, plus long.")
        vid_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
        vid_nom = gr.Textbox(label="Nom (pour la galerie)", value="video")
    btn_vid = gr.Button("🎬 Générer la vidéo", variant="primary")
    with gr.Accordion("🎞️ Plusieurs plans à la suite (une scène plus longue)", open=False):
        gr.Markdown(
            "Écris **un plan par ligne**, en français. Chaque plan devient un clip de la durée choisie ci-dessus ; "
            "il part de la **dernière image du plan précédent** (le premier part de l'image de départ, s'il y en a "
            "une), puis tous les clips sont assemblés en une seule vidéo. Garde les mêmes personnages et le même "
            f"décor d'une ligne à l'autre. {videos.PLANS_MAX} plans au plus ; compte 15 à 25 minutes par plan de "
            "5 s en 720p. Au fil des raccords l'image peut perdre un peu en netteté : préfère des plans courts.")
        vid_plans = gr.Textbox(label="Plans (un par ligne)", lines=5,
                               placeholder="les aventuriers avancent sur le pont du navire, la caméra les suit\n"
                                           "un kraken géant surgit de l'eau devant eux\n"
                                           "le mage lance une boule de feu sur le kraken")
        btn_vid_suite = gr.Button("🎞️ Générer les plans à la suite", variant="primary")
    vid_statut = gr.Markdown()
    vid_dossier = gr.State()
    vid_video = gr.Video(label="Vidéo", height=480, interactive=False)
    with gr.Row():
        btn_vid_continuer = gr.Button("🔁 Continuer : la dernière image devient l'image de départ")
        btn_vid_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.vid_exemples.change(lambda v: v or "", c.vid_exemples, c.vid_texte)
    c.btn_vid_prep.click(videos.preparer, [c.vid_texte, c.vid_image], c.vid_prompt)
    # prompt vide : préparé d'abord depuis la description, puis la vidéo (seulement si la préparation a réussi)
    c.btn_vid.click(videos.prompt_pret, [c.vid_texte, c.vid_prompt, c.vid_image], c.vid_prompt).success(videos.generer,
                    [c.vid_prompt, c.vid_image, c.vid_format, c.vid_duree, c.vid_etapes, c.vid_graine, c.vid_nom,
                     c.vid_texte],
                    [c.vid_statut, c.vid_video, c.vid_dossier])
    c.btn_vid_suite.click(videos.generer_suite,
                          [c.vid_plans, c.vid_image, c.vid_format, c.vid_duree, c.vid_etapes, c.vid_graine, c.vid_nom],
                          [c.vid_statut, c.vid_video, c.vid_dossier])
    c.btn_vid_continuer.click(videos.continuer, c.vid_dossier, c.vid_image)
    c.btn_vid_dossier.click(lambda d: open_folder(d) if d else None, c.vid_dossier)
