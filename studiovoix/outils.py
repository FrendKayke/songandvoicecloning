"""Outils communs : journal de commande en direct, détection de poids, ouverture de dossier."""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".th", ".ckpt"}


def has_weights(folder: Path) -> bool:
    return folder.is_dir() and any(f.suffix in WEIGHT_SUFFIXES for f in folder.rglob("*") if f.is_file())


def nouveau_dossier(parent: Path) -> Path:
    """Crée parent/<horodatage> (suffixé _2, _3… si deux travaux démarrent dans la même seconde)."""
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(1, 1000):
        d = parent / (base if i == 1 else f"{base}_{i}")
        try:
            d.mkdir(parents=True)
            return d
        except FileExistsError:
            continue
    raise RuntimeError(f"Impossible de créer un dossier dans {parent}")


def lancer_moteur(cmd, cwd, extra_env, nom, suivi=None, attendu=None):
    """Lance un script de moteurs/ en sous-processus et suit son protocole :
    « PROGRESSION i/n … » → suivi(i, n) ; « ERREUR : message » → gr.Error(« <nom> : message »).
    Si « attendu » (fichier de sortie) n'existe pas à la fin, c'est aussi une erreur. Renvoie les dernières lignes."""
    env = os.environ.copy()
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", **(extra_env or {})})
    lignes = []
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for ligne in proc.stdout:
        ligne = ligne.rstrip()
        lignes = (lignes + [ligne])[-60:]
        if ligne.startswith("PROGRESSION ") and suivi:
            i, n = (int(x) for x in ligne.split()[1].split("/"))
            suivi(i, n)
    code = proc.wait()
    erreur = next((l_ for l_ in reversed(lignes) if l_.startswith("ERREUR : ")), None)
    if code != 0 or (attendu is not None and not Path(attendu).exists()):
        if erreur:
            raise gr.Error(f"{nom} : " + erreur[len("ERREUR : "):])
        raise gr.Error(f"{nom} a échoué :\n" + "\n".join(lignes)[-1500:])
    return lignes


def stream_command(cmd, cwd, header, extra_env=None):
    """Lance une commande et renvoie sa sortie au fur et à mesure (pour l'affichage)."""
    log = header + "\n"
    yield log
    env = os.environ.copy()
    env.update(extra_env or {})
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except OSError as e:
        yield log + f"\n❌ Impossible de lancer la commande : {e}"
        return
    for line in proc.stdout:
        log = (log + line.rstrip() + "\n")[-8000:]
        yield log
    code = proc.wait()
    yield log + ("\n✅ Terminé." if code == 0 else f"\n❌ Échec (code {code}). Regarde les dernières lignes ci-dessus.")


def open_folder(path, create=False):
    p = Path(path)
    if not p.exists():
        if not create:
            raise gr.Error(
                f"Ce dossier n'existe pas : {p}. Le composant n'est probablement pas encore installé "
                "(lance INSTALLER.bat)."
            )
        p.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception as e:
        raise gr.Error(f"Impossible d'ouvrir le dossier : {e}")
