"""Outils communs : journal de commande en direct, détection de poids, ouverture de dossier."""
import json
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


def ecrire_creation(dossier, infos):
    """Décrit une création (réglages, versions) dans creation.json, en UTF-8 lisible : la galerie s'en sert
    pour réécouter, recréer avec la même graine ou refaire un passage."""
    infos = {"date": datetime.now().isoformat(timespec="seconds"), **infos}
    (Path(dossier) / "creation.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")


JOURNAL_MAX = 2_000_000  # octets par journal de moteur ; au-delà, l'ancien devient <nom>.1.log


def journal_moteur(nom):
    """data/journaux/<nom>.log : toute la sortie d'un moteur, horodatée (pour un dépannage à distance : la fenêtre
    du navigateur n'affiche que la progression)."""
    import unicodedata

    from . import config as cfg

    ascii_ = unicodedata.normalize("NFKD", str(nom or "moteur")).encode("ascii", "ignore").decode().lower()
    nom_fichier = "".join(c if c.isalnum() else "_" for c in ascii_).strip("_") or "moteur"
    d = cfg.DATA_DIR / "journaux"
    d.mkdir(parents=True, exist_ok=True)
    chemin = d / f"{nom_fichier}.log"
    try:
        if chemin.exists() and chemin.stat().st_size > JOURNAL_MAX:
            chemin.replace(d / f"{nom_fichier}.1.log")
    except OSError:
        pass
    return chemin


def _avertir(texte):
    """Avertissement d'un moteur : bandeau de l'interface (gr.Warning), avec les programmes qui occupent la carte
    graphique quand il s'agit de mémoire."""
    if "mémoire graphique" in texte:
        from .diagnostic import programmes_sur_la_carte

        occupants = programmes_sur_la_carte()
        if occupants:
            texte += f" Programmes sur la carte : {occupants}. Ferme-les, puis relance."
    try:
        gr.Warning(texte, duration=None)
    except Exception:  # noqa: BLE001 - hors d'un événement Gradio (essai, script)
        pass


def lancer_moteur(cmd, cwd, extra_env, nom, suivi=None, attendu=None, resident=None, annonce=None):
    """Lance un script de moteurs/ en sous-processus et suit son protocole :
    « PROGRESSION i/n … » → suivi(i, n) ; « ERREUR : message » → gr.Error(« <nom> : message »).
    Si « attendu » (fichier de sortie) n'existe pas à la fin, c'est aussi une erreur. Renvoie les dernières lignes.
    resident : nom du moteur résident (cmd = [python, script, *arguments]) qui garde ses modèles en mémoire entre
    deux tâches (residents.py) ; sinon, ou si le réglage est désactivé, processus neuf, et les moteurs résidents
    sont d'abord fermés (ils occuperaient la mémoire dont ce moteur a besoin).
    annonce(texte) : affiche ce qui se passe avant la première progression du moteur (attente, démarrage)."""
    from . import ollama, residents

    ollama.liberer()  # un modèle d'Ollama resté sur la carte ferait déborder la mémoire graphique
    lignes = []
    try:
        journal = open(journal_moteur(resident or nom), "a", encoding="utf-8", errors="replace")
        journal.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] === {nom} : {' '.join(map(str, cmd[1:]))}\n")
    except OSError:
        journal = None
    debut = datetime.now()

    def lire(flux):
        nonlocal lignes
        for ligne in flux:
            ligne = ligne.rstrip()
            lignes = (lignes + [ligne])[-60:]
            if journal:
                journal.write(f"[{(datetime.now() - debut).total_seconds():7.1f} s] {ligne}\n")
                journal.flush()
            if ligne.startswith("PROGRESSION ") and suivi:
                i, n = (int(x) for x in ligne.split()[1].split("/"))
                suivi(i, n)
            elif ligne.startswith("AVERTISSEMENT : "):
                _avertir(ligne[len("AVERTISSEMENT : "):])

    code = None
    try:
        if resident and residents.ACTIF:
            if annonce and residents.occupe():
                annonce("En attente : une autre génération utilise la carte graphique…")
            if annonce and not residents.ouvert(resident):
                annonce(f"Démarrage du moteur ({resident})…")
            flux = residents.lancer(resident, cmd[0], cmd[1], cmd[2:], cwd, extra_env)
            try:
                while True:
                    lire([next(flux)])
            except StopIteration as fin:  # le générateur renvoie le code de sortie de la tâche
                code = fin.value
        else:
            residents.arreter_tous()
            env = os.environ.copy()
            env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", **(extra_env or {})})
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            lire(proc.stdout)
            code = proc.wait()
    finally:
        if journal:
            journal.write(f"[{(datetime.now() - debut).total_seconds():7.1f} s] fin (code {code})\n")
            journal.close()
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
