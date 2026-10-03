"""Moteurs résidents : diffusion, Chatterbox et nettoyage restent ouverts entre deux générations.

Chaque génération lançait un processus neuf qui relisait ses modèles sur le disque (jusqu'à une minute pour
Z-Image ou Wan) : la carte graphique attendait. Désormais le moteur est lancé une fois (moteurs/resident.py) et
garde ses modèles en mémoire vive ; seule la génération est refaite. Règles :
  - un seul moteur résident à la fois, et jamais en même temps qu'ACE-Step (serveur_acestep.assurer les ferme) ni
    qu'un moteur lancé à part (Demucs, Seed-VC, RVC : outils.lancer_moteur les ferme d'abord) ;
  - une tâche à la fois (verrou) ;
  - après une erreur, le moteur est fermé (la tâche suivante repart d'un processus propre : après un manque de
    mémoire graphique, l'état de CUDA n'est pas fiable) ;
  - il se ferme de lui-même après GARDER_MIN minutes sans tâche, ou quand l'application s'arrête (son entrée
    standard se ferme ; sous Windows il est aussi rattaché au « job » qui tue les processus de l'application).
Réglage : STUDIOVOIX_GARDER_MODELES (minutes, 0 = désactivé) ou la case de l'onglet Modèles.
"""
import atexit
import json
import os
import subprocess
import threading
import time

import gradio as gr

FIN = "@@FIN@@"  # même marqueur que moteurs/resident.py


def _minutes():
    try:
        return max(0.0, float(os.environ.get("STUDIOVOIX_GARDER_MODELES", "15")))
    except ValueError:
        return 15.0


GARDER_MIN = _minutes()
ACTIF = GARDER_MIN > 0
# Moteurs légers en mémoire vive (quelques Go) qui savent sortir leurs modèles de la carte (« --ranger ») : avant
# ACE-Step ils sont rangés au lieu d'être fermés, et la chanson suivante ne les recharge pas.
RANGEABLES = {"Séparation"}
_verrou = threading.RLock()
_moteurs = {}  # nom → _Moteur


class _Moteur:
    def __init__(self, nom, python, script, cwd, env):
        self.nom, self.signature = nom, (str(python), str(script), str(cwd), tuple(sorted((env or {}).items())))
        environnement = os.environ.copy()
        environnement.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", **(env or {})})
        options = {"cwd": cwd, "env": environnement, "stdin": subprocess.PIPE, "stdout": subprocess.PIPE,
                   "stderr": subprocess.STDOUT, "text": True, "encoding": "utf-8", "errors": "replace", "bufsize": 1}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.proc = subprocess.Popen([str(python), str(script), "--resident", str(int(GARDER_MIN * 60))], **options)
        self.debut = self.derniere = time.time()
        self.taches = 0
        if os.name == "nt":
            from .serveur_acestep import _job_windows

            self.job = _job_windows(self.proc)
        # attend « prêt » (les imports de base sont faits)
        lignes = []
        for ligne in self.proc.stdout:
            ligne = ligne.rstrip()
            if ligne.startswith(FIN):
                return
            lignes.append(ligne)
        self.proc.wait()
        raise gr.Error(f"{nom} : le moteur n'a pas pu démarrer.\n" + "\n".join(lignes[-20:]))

    def vivant(self):
        return self.proc.poll() is None

    def executer(self, argv):
        """Envoie la tâche ; renvoie un itérateur des lignes de sortie, puis le code dans self.code."""
        self.code = None
        self.proc.stdin.write(json.dumps({"argv": [str(a) for a in argv]}) + "\n")  # ASCII : accents échappés
        self.proc.stdin.flush()
        for ligne in self.proc.stdout:
            ligne = ligne.rstrip()
            if ligne.startswith(FIN):
                self.code = int(ligne.split()[1])
                break
            yield ligne
        else:  # sortie fermée : le moteur s'est arrêté (plantage, mémoire vive épuisée…)
            self.code = self.proc.wait() or 1
        self.taches += 1
        self.derniere = time.time()

    def arreter(self):
        if self.vivant():
            try:
                self.proc.stdin.close()  # le moteur sort de sa boucle et rend la mémoire
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass


def lancer(nom, python, script, argv, cwd, env):
    """Exécute une tâche dans le moteur résident `nom` (démarré si besoin, après avoir fermé les autres).
    Générateur des lignes de sortie ; à la fin, renvoie le code de sortie (StopIteration.value)."""
    with _verrou:
        _arreter_autres(nom)
        m = _moteurs.get(nom)
        if m is not None and (not m.vivant() or m.signature != (str(python), str(script), str(cwd),
                                                                  tuple(sorted((env or {}).items())))):
            m.arreter()
            m = None
        if m is None:
            m = _moteurs[nom] = _Moteur(nom, python, script, cwd, env)
        try:
            yield from m.executer(argv)
        except BaseException:
            m.arreter()  # tâche interrompue (onglet fermé, erreur de lecture) : on ne sait plus où en est le moteur
            _moteurs.pop(nom, None)
            raise
        if m.code != 0 or not m.vivant():
            m.arreter()
            _moteurs.pop(nom, None)
        return m.code


def _arreter_autres(nom=None):
    for autre in [n for n in _moteurs if n != nom]:
        _moteurs.pop(autre).arreter()


def liberer_pour_ace():
    """Avant ACE-Step : les moteurs rangeables (Demucs + Seed-VC) passent leurs modèles en mémoire vive, les autres
    (diffusion : jusqu'à 20 Go de mémoire vive) sont fermés. True si la carte a été libérée."""
    with _verrou:
        libere = False
        for nom in list(_moteurs):
            m = _moteurs[nom]
            if not m.vivant():
                _moteurs.pop(nom)
                continue
            libere = True
            if nom in RANGEABLES:
                list(m.executer(["--ranger"]))
                if m.code == 0 and m.vivant():
                    continue
            _moteurs.pop(nom).arreter()
        return libere


def arreter_tous():
    """Ferme tous les moteurs résidents (avant ACE-Step ou un moteur lancé à part). True si au moins un tournait."""
    with _verrou:
        actifs = [m for m in _moteurs.values() if m.vivant()]
        _arreter_autres()
        return bool(actifs)


def regler(actif):
    global ACTIF
    ACTIF = bool(actif) and GARDER_MIN > 0
    if not ACTIF:
        arreter_tous()
    return etat()


def etat():
    if GARDER_MIN <= 0:
        return "⚪ Modèles gardés en mémoire : désactivé (STUDIOVOIX_GARDER_MODELES=0)."
    if not ACTIF:
        return "⚪ Modèles gardés en mémoire : désactivé (chaque génération recharge ses modèles)."
    vivants = [m for m in _moteurs.values() if m.vivant()]
    if not vivants:
        return (f"⚪ Aucun moteur ouvert. Le premier lancé gardera ses modèles en mémoire (fermé après "
                f"{GARDER_MIN:g} min sans génération).")
    m = vivants[0]
    reste = max(0, GARDER_MIN * 60 - (time.time() - m.derniere)) / 60
    return (f"🟢 Moteur **{m.nom}** ouvert, modèles en mémoire ({m.taches} génération(s) ; fermé dans "
            f"{reste:.0f} min sans génération).")


def fermer_depuis_interface():
    arreter_tous()
    return etat()


@atexit.register
def _a_la_sortie():
    for m in list(_moteurs.values()):
        if m.vivant():
            m.proc.kill()
