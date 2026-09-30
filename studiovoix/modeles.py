"""Onglet « Modèles » : état, emplacement et téléchargement des modèles de chaque moteur."""
from . import acestep, demucs, seedvc
from . import config as cfg


def _cell(ok, missing=None):
    if ok:
        return "✅ présent"
    return "❌ absent" + (f" ({', '.join(missing)})" if missing else "")


def models_status_md() -> str:
    ace_missing = acestep.missing_components()
    sv_missing = seedvc.missing_components()
    return (
        "| Composant | État | Dossier de stockage |\n|---|---|---|\n"
        f"| ACE-Step 1.5 (génération de la chanson) | {_cell(not ace_missing, ace_missing)} | `{acestep.ckpt_dir()}` |\n"
        f"| Seed-VC (conversion de voix chantée) | {_cell(not sv_missing, sv_missing)} | `{seedvc.ckpt_dir()}` |\n"
        f"| Demucs (séparation voix / musique) | {_cell(demucs.is_present())} | `{demucs.ckpt_dir()}` |\n\n"
        f"Tes voix enregistrées : `{cfg.VOICES_DIR}`  \nTes chansons : `{cfg.SONGS_DIR}`\n\n"
        "*Pour Demucs, l'état indique la présence d'au moins un fichier `.th` dans ce dossier.*"
    )
