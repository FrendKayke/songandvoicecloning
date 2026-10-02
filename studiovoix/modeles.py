"""Onglet « Modèles » : état, emplacement et téléchargement des modèles de chaque moteur."""
from . import acestep, chatterbox, demucs, diffusion, nettoyage, rvc, seedvc
from . import config as cfg


def _cell(ok, missing=None):
    if ok:
        return "✅ présent"
    return "❌ absent" + (f" ({', '.join(missing)})" if missing else "")


def models_status_md() -> str:
    ace_missing = acestep.missing_components()
    sv_missing = seedvc.missing_components()
    cb_missing = chatterbox.missing_components()
    nt_missing = nettoyage.missing_components()
    rvc_missing = rvc.missing_components()
    dif_missing = diffusion.missing_components()
    return (
        "| Composant | État | Dossier de stockage |\n|---|---|---|\n"
        f"| ACE-Step 1.5 (génération de la chanson) | {_cell(not ace_missing, ace_missing)} | `{acestep.ckpt_dir()}` |\n"
        f"| Seed-VC (conversion de voix chantée) | {_cell(not sv_missing, sv_missing)} | `{seedvc.ckpt_dir()}` |\n"
        f"| Demucs (séparation voix / musique) | {_cell(demucs.is_present())} | `{demucs.ckpt_dir()}` |\n"
        f"| Chatterbox Multilingual V3 (synthèse vocale) | {_cell(not cb_missing, cb_missing)} | `{chatterbox.ckpt_dir()}` |\n"
        f"| Nettoyage de voix (MossFormer2, VoiceFixer) | {_cell(not nt_missing, nt_missing)} | `{nettoyage.ckpt_dir()}` |\n"
        f"| RVC (Applio) — modèles de base | {_cell(not rvc_missing, rvc_missing)} | `{rvc.ckpt_dir()}` |\n"
        f"| Diffusion : Qwen3-VL, Stable Audio Open, Z-Image-Turbo, FLUX.2 klein, BiRefNet, Real-ESRGAN, GFPGAN, Hunyuan3D-2 | {_cell(not dif_missing, dif_missing)} | `{diffusion.hf_home() / 'hub'}` |\n\n"
        f"Tes modèles RVC entraînés : {len(rvc.modeles())} (`{cfg.RVC_DIR / 'logs'}`)  \n"
        f"Tes voix enregistrées : `{cfg.VOICES_DIR}`  \nTes chansons : `{cfg.SONGS_DIR}`  \n"
        f"Tes textes lus : `{cfg.TTS_DIR}`\n\n"
        "*Pour Demucs, l'état indique la présence d'au moins un fichier `.th` dans ce dossier.*"
    )
