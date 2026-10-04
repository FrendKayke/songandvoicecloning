"""Mode résident des moteurs : le script reste ouvert entre deux tâches et garde ses modèles en mémoire.

Sans lui, chaque génération lance un nouveau processus qui relit ses modèles sur le disque (de quelques secondes
pour Real-ESRGAN à une minute pour Z-Image ou Wan) avant de calculer. En mode résident, l'application lance
le script une fois avec « --resident <secondes> » puis lui envoie ses tâches sur l'entrée standard, une par ligne :

    {"argv": ["image", "C:\\...\\tache_image.json"]}

Le script exécute la tâche comme s'il avait été lancé avec ces arguments (mêmes lignes PROGRESSION, RESULTAT,
ERREUR, TERMINE), puis écrit « @@FIN@@ <code> » (0 = réussite ; sys.exit(n) donne n). Il s'arrête de lui-même
quand l'entrée standard se ferme (application arrêtée) ou après <secondes> sans tâche (mémoire rendue au système).
Bibliothèque standard seulement : ce fichier est importé par des scripts de plusieurs environnements Python.
"""
import json
import os
import sys
import threading
import time
import traceback

FIN = "@@FIN@@"


def _surveiller(etat, inactivite):
    """Ferme le moteur après `inactivite` secondes sans tâche (jamais pendant une tâche)."""
    while True:
        time.sleep(min(30.0, inactivite / 4))
        if not etat["en_cours"] and time.time() - etat["derniere"] > inactivite:
            print("Moteur inactif depuis longtemps : fermeture (la mémoire est rendue).", flush=True)
            os._exit(0)


def servir(executer, inactivite):
    """Boucle du mode résident : executer(argv) pour chaque tâche reçue. inactivite : secondes sans tâche avant
    l'arrêt (0 ou moins : jamais).
    L'entrée standard n'est lue que par le fil principal, entre deux tâches : sous Windows, une lecture en attente
    sur un tube dans un autre fil bloque le chargement de certaines DLL (GetFileType sur les handles standard,
    appelé à l'initialisation du CRT d'OpenBLAS/numpy) : le moteur restait figé dans « import torch »."""
    etat = {"en_cours": False, "derniere": time.time()}
    if inactivite > 0:
        threading.Thread(target=_surveiller, args=(etat, inactivite), daemon=True).start()
    print(f"{FIN} 0", flush=True)  # prêt
    while True:
        ligne = sys.stdin.readline()
        if not ligne:
            return  # entrée fermée : l'application s'est arrêtée
        if not ligne.strip():
            continue
        etat["en_cours"] = True
        try:
            executer(json.loads(ligne)["argv"])
            code = 0
        except SystemExit as e:  # _erreur() des scripts : ERREUR déjà écrite
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except BaseException:  # noqa: BLE001 - erreur imprévue : on la montre et on la signale
            traceback.print_exc(file=sys.stdout)
            code = 1
        etat["en_cours"], etat["derniere"] = False, time.time()
        sys.stdout.flush()
        print(f"{FIN} {code}", flush=True)


def demande(argv):
    """argv = sys.argv[1:] ; renvoie le délai d'inactivité si le mode résident est demandé, sinon None."""
    if len(argv) == 2 and argv[0] == "--resident":
        try:
            return float(argv[1])
        except ValueError:
            return None
    return None
