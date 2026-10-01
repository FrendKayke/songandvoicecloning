"""Vérifie que les dépendances de chaque environnement se résolvent pour Windows (sans rien installer).

Rejoue, avec `uv pip compile --python-platform x86_64-pc-windows-msvc`, ce que fait installer.ps1 : PyTorch depuis son
index CUDA, puis la liste de installation/<env>.txt (ou le requirements.txt d'Applio au commit épinglé pour RVC).
Un conflit comme diffusers 0.40 / transformers 4.57 est ainsi repéré avant d'arriver chez l'utilisateur.

    python installation/verifier_dependances.py [environnement…]
Il faut uv dans le PATH. Code de sortie 1 si un environnement ne se résout pas.
"""
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
RACINE = DOSSIER.parent
TORCH = "https://download.pytorch.org/whl/"

# environnement → (version de Python, index PyTorch, fichier de dépendances)
ENVIRONNEMENTS = {
    "seed-vc": ("3.10", "cu124", DOSSIER / "seed-vc.txt"),
    "chatterbox": ("3.11", "cu124", DOSSIER / "chatterbox.txt"),
    "nettoyage": ("3.11", "cu124", DOSSIER / "nettoyage.txt"),
    "diffusion": ("3.12", "cu126", DOSSIER / "diffusion.txt"),
    "rvc": ("3.12", "cu128", None),  # requirements.txt d'Applio, au commit épinglé dans installer.ps1
    "application": ("3.12", None, RACINE / "requirements.txt"),
}


def commit_applio():
    texte = (RACINE / "installer.ps1").read_text(encoding="utf-8-sig")
    return re.search(r"\$RvcCommit = '([0-9a-f]{40})'", texte).group(1)


def liste(nom, dossier_tmp):
    fichier = ENVIRONNEMENTS[nom][2]
    if fichier:
        return fichier
    url = f"https://raw.githubusercontent.com/IAHispano/Applio/{commit_applio()}/requirements.txt"
    cible = Path(dossier_tmp) / "applio-requirements.txt"
    with urllib.request.urlopen(url, timeout=60) as r:
        cible.write_bytes(r.read())
    return cible


def verifier(nom, dossier_tmp):
    python, index, _ = ENVIRONNEMENTS[nom]
    cmd = ["uv", "pip", "compile", str(liste(nom, dossier_tmp)), "--python-version", python,
           "--python-platform", "x86_64-pc-windows-msvc", "--quiet", "--no-header"]
    if index:
        cmd += ["--extra-index-url", TORCH + index, "--index-strategy", "unsafe-best-match"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, r.stderr.strip()
    paquets = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+)", r.stdout, re.M))
    torch = paquets.get("torch")
    if index and not (torch and torch.endswith("+" + index)):
        # Windows recevrait la version de PyTorch sans carte graphique
        return False, f"PyTorch résolu en « {torch} » au lieu d'une version +{index}"
    return True, f"{len(paquets)} paquets" + (f", torch {torch}" if torch else "")


def main(noms):
    noms = noms or list(ENVIRONNEMENTS)
    inconnus = [n for n in noms if n not in ENVIRONNEMENTS]
    if inconnus:
        sys.exit(f"Environnement inconnu : {', '.join(inconnus)} (connus : {', '.join(ENVIRONNEMENTS)})")
    echecs = 0
    with tempfile.TemporaryDirectory() as tmp:
        for nom in noms:
            ok, detail = verifier(nom, tmp)
            echecs += not ok
            print(f"{'OK    ' if ok else 'ÉCHEC '} {nom:12} {detail}", flush=True)
    sys.exit(1 if echecs else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
