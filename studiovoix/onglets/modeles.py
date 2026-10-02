"""Onglet « Modèles » : état et téléchargement des modèles, carte graphique, diagnostic, jeton."""
import gradio as gr

from .. import acestep, chatterbox, demucs, diagnostic, diffusion, nettoyage, rvc, seedvc, serveur_acestep
from .. import config as cfg
from ..modeles import models_status_md
from ..outils import open_folder
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Les modèles sont volumineux (plusieurs Go au total) et ne sont téléchargés qu'une seule fois. "
        "Vérifie ici leur présence et leur emplacement, ou lance le téléchargement."
    )
    status = gr.Markdown(models_status_md())
    btn_refresh = gr.Button("🔄 Actualiser l'état")
    with gr.Accordion("🎛️ Carte graphique et serveur ACE-Step", open=False):
        gr.Markdown(
            "Le serveur ACE-Step (génération musicale) occupe ~8 Go de mémoire graphique tant qu'il tourne. "
            "Avec la libération automatique, il est arrêté avant les bruitages, la 3D, les illustrations, la "
            "synthèse vocale, le nettoyage et l'entraînement RVC, puis relancé à la chanson suivante "
            "(la première génération après une relance prend une minute de plus). "
            f"Journal du serveur : `{serveur_acestep.journal()}`."
        )
        with gr.Row():
            gpu_auto = gr.Checkbox(value=serveur_acestep.LIBERATION_AUTO,
                                   label="Libérer automatiquement la carte graphique")
            gpu_etat = gr.Markdown(serveur_acestep.etat())
        with gr.Row():
            btn_gpu_stop = gr.Button("⏹️ Arrêter ACE-Step maintenant")
            btn_gpu_maj = gr.Button("🔄 État du serveur")
        gpu_msg = gr.Markdown()
        with gr.Row():
            btn_gpu_mesure = gr.Button("📈 Mesurer la carte graphique pendant 1 minute")
            gr.Markdown("Lance une génération dans un autre onglet, puis clique ici : l'utilisation réelle de la "
                        "carte s'affiche en direct (mesures de nvidia-smi).")
        gpu_mesure = gr.Markdown()
    with gr.Accordion("🩺 Diagnostic (si quelque chose ne marche pas)", open=False):
        gr.Markdown(
            "**Diagnostic rapide** (moins d'une minute) : carte graphique, espace disque, chaque moteur "
            "(PyTorch, CUDA), modèles manquants. **Essai complet** (10 à 20 minutes) : une génération courte "
            "par moteur, avec sa durée et la mémoire graphique utilisée. Les fichiers d'essai vont dans "
            "`data\\diagnostic`, pas dans la galerie. Envoie le fichier `rapport.txt` pour un dépannage."
        )
        with gr.Row():
            btn_diag = gr.Button("🩺 Diagnostic rapide")
            btn_essai = gr.Button("🧪 Essai complet des moteurs")
        diag_rapport = gr.Markdown()
        diag_fichier = gr.File(label="Rapport à envoyer", interactive=False)
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
    c.gpu_auto.change(serveur_acestep.regler_liberation, c.gpu_auto, c.gpu_msg)
    c.btn_diag.click(diagnostic.rapide, None, [c.diag_rapport, c.diag_fichier])
    c.btn_essai.click(diagnostic.complet, None, [c.diag_rapport, c.diag_fichier])
    c.btn_gpu_stop.click(serveur_acestep.arreter_depuis_interface, None, c.gpu_etat)
    c.btn_gpu_maj.click(serveur_acestep.etat, None, c.gpu_etat)
    c.btn_gpu_mesure.click(diagnostic.surveiller_gpu, None, c.gpu_mesure)
    for b, fn in ((c.b_ace, acestep.download), (c.b_sv, seedvc.download), (c.b_dm, demucs.download),
                  (c.b_cb, chatterbox.download), (c.b_nt, nettoyage.download), (c.b_rvc, rvc.download),
                  (c.b_dif, lambda: diffusion.download(["qwen", "forme3d", "texture3d", "zimage", "personnages", "photo_detourage", "photo_qualite"])),
                  (c.b_vid, lambda: diffusion.download(["video"])),
                  (c.b_sfx, lambda: diffusion.download(["bruitages"]))):
        b.click(fn, None, c.log).then(models_status_md, None, c.status)
    c.o_ace.click(lambda: open_folder(acestep.ckpt_dir()))
    c.o_sv.click(lambda: open_folder(seedvc.ckpt_dir()))
    c.o_dm.click(lambda: open_folder(demucs.ckpt_dir()))
    c.o_cb.click(lambda: open_folder(chatterbox.ckpt_dir()))
    c.o_nt.click(lambda: open_folder(nettoyage.ckpt_dir()))
    c.o_rvc.click(lambda: open_folder(cfg.RVC_DIR))
    c.o_dif.click(lambda: open_folder(diffusion.hf_home() / "hub", create=True))
    c.btn_jeton.click(diffusion.enregistrer_jeton, c.hf_jeton, c.hf_msg)

    c.o_data.click(lambda: open_folder(cfg.SONGS_DIR, create=True))
    demo.load(models_status_md, None, c.status)
