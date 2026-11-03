"""Chatterbox Multilingual V3 : synthèse vocale (texte → parole) avec une voix de la bibliothèque.

Comme les autres moteurs, Chatterbox a son propre environnement Python (<lecteur>:\\StudioVoix\\chatterbox\\.venv)
et n'est jamais importé : l'application écrit un fichier de tâche JSON et lance moteurs/chatterbox_tts.py
en sous-processus, dont elle lit la progression (« PROGRESSION i/n ») et les erreurs (« ERREUR : … »).
"""
import json
import os
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import retraits
from . import serveur_acestep
from .outils import ecrire_creation, lancer_moteur, nouveau_dossier, stream_command
from .voix import chemin_voix

HF_REPO = "models--ResembleAI--chatterbox"
# Fichiers chargés par ChatterboxMultilingualTTS.from_pretrained(t3_model="v3") (mtl_tts.py)
FICHIERS = {
    "modèle de texte V3": "t3_mtl23ls_v3.safetensors",
    "décodeur audio (S3Gen)": "s3gen.pt",
    "encodeur de voix": "ve.pt",
    "vocabulaire": "grapheme_mtl_merged_expanded_v1.json",
}
TEXTE_MAX = 5000  # caractères, soit environ 5 minutes de parole


def script() -> Path:
    return cfg.MOTEURS_DIR / "chatterbox_tts.py"


def _env():
    """Variables pour le sous-processus : tout reste dans StudioVoix, sortie en UTF-8."""
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    # spacy-pkuseg (utilisé par Chatterbox) télécharge un modèle dans ~/.pkuseg sinon, donc sur C:
    env["PKUSEG_HOME"] = os.environ.get("PKUSEG_HOME") or str(cfg.CHATTERBOX_DIR / "pkuseg")
    return env


# --- Modèles -----------------------------------------------------------------
def ckpt_dir() -> Path:
    """Cache Hugging Face de Chatterbox (HF_HOME pointe dans StudioVoix, voir lancer.bat)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]) / HF_REPO
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub" / HF_REPO


def missing_components():
    snaps = ckpt_dir() / "snapshots"
    return [n for n, f in FICHIERS.items() if not any(snaps.glob(f"*/{f}"))]


def download():
    if not Path(cfg.CHATTERBOX_PYTHON).exists():
        yield f"❌ Python de Chatterbox introuvable : {cfg.CHATTERBOX_PYTHON}. {retraits.conseil('chatterbox')}"
        return
    yield from stream_command(
        [cfg.CHATTERBOX_PYTHON, str(script()), "--telecharger"], cfg.APP_DIR,
        f"Téléchargement des modèles Chatterbox (~3,2 Go) vers {ckpt_dir()} …", _env(),
    )


# --- Synthèse ------------------------------------------------------------------
def synthese(voix, texte, langue_label, exaggeration, cfg_weight, temperature, graine,
             progress=gr.Progress()):
    voice_ref = chemin_voix(voix)
    texte = (texte or "").strip()
    if not texte:
        raise gr.Error("Écris le texte à faire lire.")
    if len(texte) > TEXTE_MAX:
        raise gr.Error(f"Texte trop long ({len(texte)} caractères) : {TEXTE_MAX} au maximum. Découpe-le en plusieurs fois.")
    if not Path(cfg.CHATTERBOX_PYTHON).exists():
        raise gr.Error(
            f"Chatterbox n'est pas installé ({cfg.CHATTERBOX_PYTHON} introuvable). {retraits.conseil('chatterbox')}"
        )

    workdir = nouveau_dossier(cfg.TTS_DIR)
    sortie = workdir / "parole.wav"
    (workdir / "texte.txt").write_text(texte + "\n", encoding="utf-8")
    tache = workdir / "tache.json"
    tache.write_text(json.dumps({
        "texte": texte, "langue": cfg.LANGUES[langue_label], "voix": str(voice_ref), "sortie": str(sortie),
        "exaggeration": float(exaggeration), "cfg_weight": float(cfg_weight),
        "temperature": float(temperature), "graine": int(graine or 0),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    env = _env()
    if not missing_components():
        env["HF_HUB_OFFLINE"] = "1"  # modèles présents : pas de vérification en ligne à chaque lecture

    progress(0.02, desc="Chargement de Chatterbox…")
    ecrire_creation(workdir, {
        "type": "tts", "voix": voix, "texte": texte, "langue": langue_label,
        "reglages": {"exaggeration": float(exaggeration), "cfg_weight": float(cfg_weight),
                     "temperature": float(temperature)},
        "versions": [{"graine": int(graine or 0), "dossier": ".", "fichier": str(sortie)}],
    })
    serveur_acestep.liberer_gpu(progress)
    lancer_moteur(
        [cfg.CHATTERBOX_PYTHON, str(script()), str(tache)], workdir, env, "Chatterbox",
        lambda i, n: progress(0.1 + 0.85 * (i - 1) / n, desc=f"Synthèse vocale : morceau {i}/{n}…"),
        attendu=sortie, resident="Chatterbox",
    )
    return str(sortie), f"Terminé. Fichier : {sortie}"


def synthese_lot(voix_wav, textes, langue_code, dossier, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, graine=0,
                 progress=None):
    """Plusieurs textes lus par la même voix (un WAV 24 kHz par texte, voix analysée une seule fois) : doublage
    phrase par phrase. voix_wav : n'importe quel enregistrement (seules ses 10 premières secondes servent).
    Renvoie la liste des fichiers, dans l'ordre des textes."""
    textes = [(t or "").strip() for t in textes]
    if not any(textes):
        raise gr.Error("Aucun texte à faire lire.")
    if not Path(cfg.CHATTERBOX_PYTHON).exists():
        raise gr.Error(
            f"Chatterbox n'est pas installé ({cfg.CHATTERBOX_PYTHON} introuvable). {retraits.conseil('chatterbox')}"
        )
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    sorties = [dossier / f"phrase_{i:03d}.wav" for i in range(1, len(textes) + 1)]
    tache = dossier / "tache_voix.json"
    tache.write_text(json.dumps({
        "textes": [t or "…" for t in textes], "sorties": [str(s) for s in sorties], "langue": langue_code,
        "voix": str(voix_wav), "exaggeration": float(exaggeration), "cfg_weight": float(cfg_weight),
        "temperature": float(temperature), "graine": int(graine or 0),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    env = _env()
    if not missing_components():
        env["HF_HUB_OFFLINE"] = "1"
    serveur_acestep.liberer_gpu(progress)
    lancer_moteur(
        [cfg.CHATTERBOX_PYTHON, str(script()), str(tache)], dossier, env, "Chatterbox",
        (lambda i, n: progress(0.55 + 0.3 * (i - 1) / n, desc=f"Voix doublée : morceau {i}/{n}…")) if progress
        else None,
        attendu=sorties[-1], resident="Chatterbox",
    )
    return sorties

