"""Demucs : séparation voix / instrumental (sous-processus, environnement de Seed-VC)."""
import os
import subprocess
from pathlib import Path

import gradio as gr

from . import config as cfg
from .outils import stream_command


# Pistes produites par htdemucs sans --two-stems (fichiers <sortie>/htdemucs/<morceau>/<piste>.wav)
PISTES = ("drums", "bass", "other", "vocals")


def _run(song: Path, out: Path, extra_args):
    if not Path(cfg.SEEDVC_PYTHON).exists():
        raise gr.Error(
            f"Python de Seed-VC introuvable : {cfg.SEEDVC_PYTHON} (il sert aussi à Demucs). "
            "Lance INSTALLER.bat."
        )
    cmd = [cfg.SEEDVC_PYTHON, "-m", "demucs", *extra_args, "-n", "htdemucs", "-o", str(out), str(song)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise gr.Error(f"Demucs a échoué :\n{p.stderr[-1500:]}")


def separate_vocals(song: Path, workdir: Path):
    """Voix / instrumental (2 pistes)."""
    out = workdir / "demucs"
    _run(song, out, ["--two-stems=vocals"])
    vocals = next(out.rglob("vocals.wav"), None)
    instru = next(out.rglob("no_vocals.wav"), None)
    if not vocals or not instru:
        raise gr.Error("Demucs n'a pas produit les fichiers attendus.")
    return vocals, instru


def separate_stems(song: Path, workdir: Path):
    """Batterie, basse, autres instruments et voix (4 pistes) : {nom de piste: fichier}."""
    out = workdir / "demucs4"
    _run(song, out, [])
    pistes = {p: next(out.rglob(f"{p}.wav"), None) for p in PISTES}
    if not all(pistes.values()):
        raise gr.Error("Demucs n'a pas produit les 4 pistes attendues.")
    return pistes


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
