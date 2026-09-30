"""Demucs : séparation voix / instrumental (sous-processus, environnement de Seed-VC)."""
import os
import subprocess
from pathlib import Path

import gradio as gr

from . import config as cfg
from .outils import stream_command


def separate_vocals(song: Path, workdir: Path):
    out = workdir / "demucs"
    cmd = [cfg.SEEDVC_PYTHON, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
           "-o", str(out), str(song)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise gr.Error(f"Demucs a échoué :\n{p.stderr[-1500:]}")
    vocals = next(out.rglob("vocals.wav"), None)
    instru = next(out.rglob("no_vocals.wav"), None)
    if not vocals or not instru:
        raise gr.Error("Demucs n'a pas produit les fichiers attendus.")
    return vocals, instru


# --- Modèle ------------------------------------------------------------------
def ckpt_dir() -> Path:
    home = os.environ.get("TORCH_HOME") or str(Path.home() / ".cache" / "torch")
    return Path(home) / "hub" / "checkpoints"


def is_present() -> bool:
    dm = ckpt_dir()
    return dm.is_dir() and any(dm.glob("*.th"))


def download():
    code = "from demucs.pretrained import get_model; get_model('htdemucs'); print('Modèle htdemucs prêt.')"
    if not Path(cfg.SEEDVC_PYTHON).exists():
        yield f"❌ Python de Seed-VC introuvable : {cfg.SEEDVC_PYTHON}. Lance INSTALLER.bat."
        return
    yield from stream_command(
        [cfg.SEEDVC_PYTHON, "-c", code], cfg.APP_DIR,
        f"Téléchargement du modèle Demucs vers {ckpt_dir()} …",
    )
