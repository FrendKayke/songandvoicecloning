"""Onglet « Modèles » : état et téléchargement des modèles, import manuel, jeton Hugging Face, Qwen sans refus
(la carte graphique et le diagnostic ont leur onglet)."""
import gradio as gr

from .. import acestep, chatterbox, demucs, diffusion, nettoyage, rigging, rvc, seedvc
from .. import config as cfg
from ..modeles import models_status_md
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Les modèles sont volumineux (plusieurs Go au total) et ne sont téléchargés qu'une seule fois. "
        "Vérifie ici leur présence et leur emplacement, ou lance le téléchargement. La carte graphique, le serveur "
        "ACE-Step et le diagnostic sont dans l'onglet « 🩺 Diagnostic et carte graphique »."
    )
    status = gr.Markdown(models_status_md())
    btn_refresh = gr.Button("🔄 Actualiser l'état")
    with gr.Accordion("🔓 Assistant de prompts sans refus (facultatif)", open=False):
        gr.Markdown(
            "Qwen3-VL prépare les prompts (descriptions d'images, traductions en anglais). Il refuse parfois une "
            "demande ou l'adoucit (scène de bataille sanglante, horreur pour un jeu…). La version **sans refus** est "
            "le même modèle dont on a retiré cette réaction (« abliteration », licence Apache 2.0 comme "
            "l'original, 8,9 Go de plus sur le disque). Elle peut être un peu moins précise : garde l'original si "
            "tu n'as pas de refus. Elle ne change que l'aide à l'écriture des prompts : les modèles qui créent "
            "les images, vidéos, sons et modèles 3D restent les mêmes."
        )
        with gr.Row():
            qwen_libre = gr.Checkbox(value=diffusion.qwen_libre_actif(),
                                     label="Préparer les prompts avec Qwen3-VL sans refus")
            b_qwen_libre = gr.Button("⬇️ Télécharger Qwen3-VL sans refus (8,9 Go)")
        qwen_libre_msg = gr.Markdown()
    log = gr.Textbox(label="Journal de téléchargement", lines=14, max_lines=14, autoscroll=True, interactive=False)
    with gr.Row():
        b_ace = gr.Button("⬇️ Télécharger ACE-Step", variant="primary")
        b_sv = gr.Button("⬇️ Télécharger Seed-VC", variant="primary")
        b_dm = gr.Button("⬇️ Télécharger Demucs", variant="primary")
        b_cb = gr.Button("⬇️ Télécharger Chatterbox", variant="primary")
        b_nt = gr.Button("⬇️ Télécharger le nettoyage", variant="primary")
        b_rvc = gr.Button("⬇️ Télécharger RVC (modèles de base)", variant="primary")
    with gr.Row():
        b_dif = gr.Button("⬇️ Télécharger Qwen3-VL, Hunyuan3D, Z-Image, FLUX.2 klein et les outils photo", variant="primary")
        b_vid = gr.Button("⬇️ Télécharger Wan 2.2 (vidéo, ~20 Go)", variant="primary")
        b_sfx = gr.Button("⬇️ Télécharger Stable Audio Open (jeton requis)", variant="primary")
        b_par = gr.Button("⬇️ Télécharger StableAvatar (vidéo parlante, ~9 Go)", variant="primary")
        b_rig = gr.Button("⬇️ Télécharger UniRig (rigging 3D, ~6 Go)", variant="primary")
        b_spr = gr.Button("⬇️ Télécharger SDXL + LoRA de sprites de dresseurs (~7,5 Go)", variant="primary")
    with gr.Accordion("📥 Modèles téléchargés à la main (si le téléchargement échoue ici)", open=False):
        gr.Markdown(
            "Quand le téléchargement depuis l'application échoue (coupures, Hugging Face lent), télécharge le dépôt "
            "toi-même : ouvre son lien ci-dessous, prends **tous ses fichiers** (bouton « Files » puis le "
            "téléchargement de chaque fichier, ou un outil comme `hf download org/nom`), garde l'arborescence, et "
            f"dépose le dossier, nommé **`org--nom`** (par exemple `Qwen--Qwen3-VL-4B-Instruct`), ou son archive "
            f".zip, dans `{diffusion.dossier_a_installer()}`. Puis clique sur « Installer ». Pour un modèle dont les "
            "fichiers sont volumineux, seuls ceux de la colonne « Fichiers attendus » sont indispensables (les "
            "petits fichiers de configuration autour le sont aussi)."
        )
        with gr.Row():
            btn_manuel = gr.Button("📥 Installer les modèles déposés dans « a-installer »", variant="primary")
            btn_manuel_dossier = gr.Button("📂 Ouvrir le dossier « a-installer »")
        manuel_msg = gr.Markdown()
        manuel_liens = gr.Markdown(diffusion.liens_manuels())
    with gr.Accordion("🔑 Jeton Hugging Face (nécessaire pour Stable Audio Open)", open=not diffusion.jeton_present()):
        gr.Markdown(
            f"1. Connecte-toi sur Hugging Face et accepte la licence sur {diffusion.LICENCE_BRUITAGES} ; "
            f"2. crée un jeton (type « Read ») sur {diffusion.JETONS} ; 3. colle-le ici. "
            "Il est enregistré dans `StudioVoix\\hf-home\\token`, jamais ailleurs."
        )
        with gr.Row():
            hf_jeton = gr.Textbox(label="Jeton", type="password", placeholder="hf_…")
            btn_jeton = gr.Button("Enregistrer le jeton")
        hf_msg = gr.Markdown("✅ Un jeton est déjà enregistré." if diffusion.jeton_present() else "")
    with gr.Row():
        o_ace = gr.Button("📂 Ouvrir dossier ACE-Step")
        o_sv = gr.Button("📂 Ouvrir dossier Seed-VC")
        o_dm = gr.Button("📂 Ouvrir dossier Demucs")
        o_cb = gr.Button("📂 Ouvrir dossier Chatterbox")
        o_nt = gr.Button("📂 Ouvrir dossier nettoyage")
        o_rvc = gr.Button("📂 Ouvrir dossier RVC")
        o_dif = gr.Button("📂 Ouvrir dossier des modèles de diffusion")
        o_data = gr.Button("📂 Ouvrir mes chansons")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_refresh.click(models_status_md, None, c.status)
    c.qwen_libre.change(diffusion.regler_qwen_libre, c.qwen_libre, c.qwen_libre_msg).then(
        models_status_md, None, c.status)
    for b, fn in ((c.b_ace, acestep.download), (c.b_sv, seedvc.download), (c.b_dm, demucs.download),
                  (c.b_cb, chatterbox.download), (c.b_nt, nettoyage.download), (c.b_rvc, rvc.download),
                  (c.b_dif, lambda: diffusion.download(["qwen", "forme3d", "texture3d", "zimage", "personnages", "sprites_pixel", "photo_detourage", "photo_qualite"])),
                  (c.b_vid, lambda: diffusion.download(["video"])),
                  (c.b_qwen_libre, lambda: diffusion.download(["qwen_libre"])),
                  (c.b_spr, lambda: diffusion.download(["sprites_dresseur"])),
                  (c.b_sfx, lambda: diffusion.download(["bruitages"])),
                  (c.b_par, lambda: diffusion.download(["parlante"])), (c.b_rig, rigging.download)):
        b.click(fn, None, c.log).then(models_status_md, None, c.status)
    c.o_ace.click(lambda: open_folder(acestep.ckpt_dir()))
    c.o_sv.click(lambda: open_folder(seedvc.ckpt_dir()))
    c.o_dm.click(lambda: open_folder(demucs.ckpt_dir()))
    c.o_cb.click(lambda: open_folder(chatterbox.ckpt_dir()))
    c.o_nt.click(lambda: open_folder(nettoyage.ckpt_dir()))
    c.o_rvc.click(lambda: open_folder(cfg.RVC_DIR))
    c.o_dif.click(lambda: open_folder(diffusion.hf_home() / "hub", create=True))
    c.btn_jeton.click(diffusion.enregistrer_jeton, c.hf_jeton, c.hf_msg)
    c.btn_manuel.click(diffusion.importer_manuels, None, c.manuel_msg).then(
        models_status_md, None, c.status).then(diffusion.liens_manuels, None, c.manuel_liens)
    c.btn_manuel_dossier.click(lambda: open_folder(diffusion.dossier_a_installer(), create=True))

    c.o_data.click(lambda: open_folder(cfg.SONGS_DIR, create=True))
    demo.load(models_status_md, None, c.status)
