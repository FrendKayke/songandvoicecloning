"""Nettoyage de voix (micro bruyant, pièce qui résonne) avant l'enregistrement dans la bibliothèque.

Moteurs : MossFormer2_SE_48K (ClearerVoice-Studio, Apache-2.0 : débruitage fidèle) et VoiceFixer
(MIT, poids CC-BY 4.0 : restauration forte, retire aussi l'écho, mais peut adoucir l'articulation).
Environnement dédié (<lecteur>:\\StudioVoix\\nettoyage\\.venv), appelé en sous-processus : moteurs/nettoyage_voix.py.
Les modèles restent dans StudioVoix\\nettoyage (le script est lancé depuis ce dossier).
"""
import json
import shutil
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import serveur_acestep
from .outils import lancer_moteur, nouveau_dossier, stream_command

NIVEAUX = {
    "Léger — bruit de fond (MossFormer2, le plus fidèle)": "leger",
    "Fort — bruit et écho de la pièce (VoiceFixer)": "fort",
    "Maximal — les deux à la suite": "maximal",
}
# Fichiers de modèles, relatifs au dossier du moteur (vérifiés après un téléchargement réel)
FICHIERS = {
    "débruiteur MossFormer2": "checkpoints/MossFormer2_SE_48K/last_best_checkpoint.pt",
    "VoiceFixer (analyse)": "voicefixer/.cache/voicefixer/analysis_module/checkpoints/vf.ckpt",
    "VoiceFixer (synthèse)": "voicefixer/.cache/voicefixer/synthesis_module/44100/model.ckpt-1490000_trimed.pt",
}


def script() -> Path:
    return cfg.MOTEURS_DIR / "nettoyage_voix.py"


def ckpt_dir() -> Path:
    return cfg.NETTOYAGE_DIR


def missing_components():
    return [n for n, f in FICHIERS.items() if not (ckpt_dir() / f).is_file()]


def _verifier_installation():
    if not Path(cfg.NETTOYAGE_PYTHON).exists():
        raise gr.Error(
            f"Le nettoyage de voix n'est pas installé ({cfg.NETTOYAGE_PYTHON} introuvable). "
            "Relance INSTALLER.bat : seules les étapes manquantes seront faites."
        )


def download():
    if not Path(cfg.NETTOYAGE_PYTHON).exists():
        yield f"❌ Python du nettoyage introuvable : {cfg.NETTOYAGE_PYTHON}. Relance INSTALLER.bat."
        return
    cfg.NETTOYAGE_DIR.mkdir(parents=True, exist_ok=True)
    yield from stream_command(
        [cfg.NETTOYAGE_PYTHON, str(script()), "--telecharger"], cfg.NETTOYAGE_DIR,
        f"Téléchargement des modèles de nettoyage (~0,8 Go) vers {ckpt_dir()} …",
    )


def nettoyer(audio_path, niveau_label, progress=gr.Progress()):
    """Nettoie un échantillon (fichier importé ou enregistré). Renvoie (fichier nettoyé, choix « nettoyée », message)."""
    if not audio_path:
        raise gr.Error("Importe ou enregistre d'abord un échantillon de voix.")
    niveau = NIVEAUX.get(niveau_label)
    if not niveau:
        raise gr.Error(f"Niveau de nettoyage inconnu : {niveau_label}")
    _verifier_installation()

    workdir = nouveau_dossier(cfg.CLEAN_DIR)
    entree = workdir / f"original{Path(audio_path).suffix.lower() or '.wav'}"
    shutil.copy(audio_path, entree)  # le fichier temporaire de Gradio peut disparaître
    sortie = workdir / "voix_nettoyee.wav"
    tache = workdir / "tache.json"
    tache.write_text(json.dumps({"entree": str(entree), "sortie": str(sortie), "niveau": niveau},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    cfg.NETTOYAGE_DIR.mkdir(parents=True, exist_ok=True)

    serveur_acestep.liberer_gpu(progress)
    progress(0.05, desc="Chargement du nettoyage…")
    lancer_moteur(
        [cfg.NETTOYAGE_PYTHON, str(script()), str(tache)], cfg.NETTOYAGE_DIR, None, "Nettoyage",
        lambda i, n: progress(0.1 + 0.85 * (i - 1) / n, desc=f"Nettoyage de la voix : étape {i}/{n}…"),
        attendu=sortie,
    )
    msg = ("✅ Voix nettoyée. Écoute-la et compare avec l'original, puis choisis laquelle enregistrer."
           + ("" if niveau == "leger" else " Si l'articulation te semble moins nette, essaie le niveau « Léger »."))
    return str(sortie), gr.update(value="Version nettoyée"), msg
