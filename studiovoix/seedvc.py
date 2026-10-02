"""Seed-VC : conversion de voix chantée zero-shot (sous-processus, inference.py)."""
import subprocess
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from . import config as cfg
from .outils import stream_command

# Fichiers attendus dans checkpoints/ (téléchargés au premier inference.py)
SEEDVC_NEEDED = {
    "modèle de chant (DiT f0 44 kHz)": "DiT_seed_v2_uvit_whisper_base_f0_44k*.pth",
    "détecteur de hauteur (RMVPE)": "rmvpe.pt",
    "encodeur de voix (CAMPPlus)": "campplus_cn_common.bin",
}


REFERENCE_MAX = 15.0  # secondes de voix de référence données à Seed-VC


def reference_courte(voice_ref: Path, workdir: Path, duree=REFERENCE_MAX) -> Path:
    """Les `duree` secondes les plus sonores (donc les plus chantées) de la voix de référence.
    Seed-VC traite source et référence dans une fenêtre de 30 s (inference.py : max_context_window, référence coupée
    à 25 s) : avec 25 s de référence, il ne reste que ~5 s de chant par passe, soit ~25 passes pour une chanson de
    2 minutes, chacune avec un raccord ; avec 15 s, ~15 s par passe : environ 3 fois moins de calcul."""
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(voice_ref), always_2d=True)
    if len(y) <= int((duree + 1) * sr):
        return Path(voice_ref)
    mono = y.mean(axis=1)
    energie = np.concatenate([[0.0], np.cumsum(mono.astype(np.float64) ** 2)])
    n = int(duree * sr)
    pas = max(1, sr // 10)
    debut = max(range(0, len(mono) - n + 1, pas), key=lambda i: energie[i + n] - energie[i])
    sortie = Path(workdir) / "reference_voix.wav"
    sf.write(str(sortie), y[debut:debut + n], sr)
    return sortie


def convert_voice(vocals: Path, voice_ref: Path, semitones: int, steps: int, workdir: Path):
    if not cfg.SEEDVC_DIR.exists():
        raise gr.Error(f"Dossier Seed-VC introuvable : {cfg.SEEDVC_DIR} (variable SEEDVC_DIR).")
    if not Path(cfg.SEEDVC_PYTHON).exists():
        raise gr.Error(f"Python de Seed-VC introuvable : {cfg.SEEDVC_PYTHON} (variable SEEDVC_PYTHON).")
    out = workdir / "seedvc"
    out.mkdir(exist_ok=True)
    voice_ref = reference_courte(voice_ref, workdir)
    cmd = [
        cfg.SEEDVC_PYTHON, "inference.py",
        "--source", str(vocals),
        "--target", str(voice_ref),
        "--output", str(out),
        "--diffusion-steps", str(int(steps)),
        "--f0-condition", "True",          # obligatoire pour la voix chantée
        "--auto-f0-adjust", "False",
        "--semi-tone-shift", str(int(semitones)),
        "--fp16", "True",
    ]
    from . import residents

    residents.arreter_tous()  # la carte et la mémoire vive pour Seed-VC
    p = subprocess.run(cmd, cwd=cfg.SEEDVC_DIR, capture_output=True, text=True)
    if p.returncode != 0:
        raise gr.Error(f"Seed-VC a échoué :\n{p.stderr[-1500:]}")
    result = next(out.glob("*.wav"), None)
    if not result:
        raise gr.Error("Seed-VC n'a produit aucun fichier.")
    return result


# --- Modèles -----------------------------------------------------------------
def ckpt_dir() -> Path:
    return cfg.SEEDVC_DIR / "checkpoints"  # Seed-VC utilise ./checkpoints (relatif à son dossier)


def missing_components():
    sv = ckpt_dir()
    return [n for n, pat in SEEDVC_NEEDED.items() if not any(sv.rglob(pat))]


def download():
    """Seed-VC télécharge ses modèles au premier usage : on lance donc une mini-conversion de test."""
    if not cfg.SEEDVC_DIR.exists() or not Path(cfg.SEEDVC_PYTHON).exists():
        yield f"❌ Seed-VC introuvable ({cfg.SEEDVC_DIR}). Vérifie SEEDVC_DIR / SEEDVC_PYTHON."
        return
    sr = cfg.SR
    tmp = cfg.DATA_DIR / "_test_telechargement"
    tmp.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    src, tgt = tmp / "source.wav", tmp / "cible.wav"
    sf.write(src, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    sf.write(tgt, (0.3 * np.sin(2 * np.pi * 330 * t)).astype("float32"), sr)
    cmd = [cfg.SEEDVC_PYTHON, "inference.py", "--source", str(src), "--target", str(tgt),
           "--output", str(tmp), "--diffusion-steps", "4", "--f0-condition", "True"]
    yield from stream_command(
        cmd, cfg.SEEDVC_DIR,
        f"Téléchargement des modèles Seed-VC vers {ckpt_dir()} (via une mini-conversion de test) …",
    )
