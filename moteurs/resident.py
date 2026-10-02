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
import queue
import sys
import threading
import traceback

FIN = "@@FIN@@"


def _lecteur(file):
    for ligne in sys.stdin:
        file.put(ligne)
    file.put(None)  # entrée fermée : l'application s'est arrêtée


def servir(executer, inactivite):
    """Boucle du mode résident : executer(argv) pour chaque tâche reçue. inactivite : secondes sans tâche avant
    l'arrêt (0 ou moins : jamais)."""
    file = queue.Queue()
    threading.Thread(target=_lecteur, args=(file,), daemon=True).start()
    print(f"{FIN} 0", flush=True)  # prêt
    while True:
        try:
            ligne = file.get(timeout=inactivite if inactivite > 0 else None)
        except queue.Empty:
            print("Moteur inactif depuis longtemps : fermeture (la mémoire est rendue).", flush=True)
            return
        if ligne is None:
            return
        if not ligne.strip():
            continue
        try:
            executer(json.loads(ligne)["argv"])
            code = 0
        except SystemExit as e:  # _erreur() des scripts : ERREUR déjà écrite
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except BaseException:  # noqa: BLE001 - erreur imprévue : on la montre et on la signale
            traceback.print_exc(file=sys.stdout)
            code = 1
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
