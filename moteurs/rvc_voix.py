"""RVC (Applio) — script exécuté DANS l'environnement RVC, depuis le dossier d'Applio (dossier de travail).

    python rvc_voix.py entrainer <tache.json>   {nom, dataset, epoques, frequence, lot}
    python rvc_voix.py convertir <tache.json>   {modele, index, entree, sortie, demi_tons, index_rate, protect}
    python rvc_voix.py telecharger              modèles de base (pré-entraînés HiFi-GAN, RMVPE, ContentVec)

Sortie : « PROGRESSION i/n », « ERREUR : message » et « TERMINE <fichier> ».

Applio (IAHispano/Applio, MIT) au commit épinglé par installer.ps1. Sa commande core.py sort avec le code 0
même quand une étape échoue (elle affiche seulement un message) : on lance donc directement ses scripts, avec
les arguments positionnels que construit core.py à ce commit (run_preprocess_script, run_extract_script,
run_train_script, run_index_script), et on vérifie leur code de sortie. La conversion passe par
« core.py infer » (options click vérifiées) et on vérifie que le fichier de sortie existe.
"""
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

VOCODEUR = "HiFi-GAN"
EMBEDDER = "contentvec"
F0 = "rmvpe"
MODELES_DE_BASE = [
    "rvc/models/predictors/rmvpe.pt",
    "rvc/models/embedders/contentvec/pytorch_model.bin",
    "rvc/models/pretraineds/hifi-gan/f0G40k.pth",
    "rvc/models/pretraineds/hifi-gan/f0D40k.pth",
]


def _preparer_applio():
    """Applio lit assets/config.json (précision, extraction du modèle final…) mais ne le crée qu'au lancement de
    son interface (app.py : copie de config_template.json). Sans lui, l'entraînement se termine sans écrire le
    modèle final (erreur avalée dans extract_model.py). On fait donc la même copie."""
    config = Path("assets") / "config.json"
    if not config.exists():
        modele = Path("assets") / "config_template.json"
        if not modele.exists():
            _erreur(f"dossier d'Applio incomplet ({Path.cwd()}) : relance INSTALLER.bat.")
        config.write_bytes(modele.read_bytes())


def _erreur(msg, code=2):
    print(f"ERREUR : {msg}", flush=True)
    sys.exit(code)


def _gpu():
    import torch

    return torch.cuda.is_available()


def _lancer(cmd, etape, suivi=None):
    """Lance un script d'Applio, relaie sa sortie, appelle suivi(ligne) ; erreur claire si le code est non nul."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen([sys.executable, *map(str, cmd)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    dernieres = []
    for ligne in proc.stdout:
        ligne = ligne.rstrip()
        dernieres = (dernieres + [ligne])[-15:]
        print(ligne, flush=True)
        if suivi:
            suivi(ligne)
    if proc.wait() != 0:
        if any("CUDA out of memory" in l_ or "OutOfMemoryError" in l_ for l_ in dernieres):
            _erreur(f"{etape} : mémoire de la carte graphique insuffisante. Baisse la taille de lot. Ferme "
                     f"les autres programmes qui utilisent la carte (jeux, vidéos) ; si tu as désactivé la "
                     f"libération automatique, arrête ACE-Step dans l'onglet Modèles, puis relance.", 3)
        _erreur(f"{etape} a échoué (voir les lignes ci-dessus).")


def telecharger():
    print("Téléchargement des modèles de base de RVC (pré-entraînés, RMVPE, ContentVec)…", flush=True)
    _lancer(["core.py", "prerequisites", "--pretraineds-hifigan", "--models", "--no-exe"], "Téléchargement")
    manquants = [f for f in MODELES_DE_BASE if not Path(f).is_file()]
    if manquants:
        _erreur("modèles de base manquants après téléchargement : " + ", ".join(manquants))
    print("Modèles de base de RVC prêts.", flush=True)


def entrainer(chemin_tache):
    t = json.loads(Path(chemin_tache).read_text(encoding="utf-8"))
    nom, dataset = t["nom"], t["dataset"]
    epoques, sr, lot = int(t["epoques"]), int(t.get("frequence", 40000)), int(t.get("lot", 8))
    exp = Path.cwd() / "logs" / nom
    cpu = os.cpu_count() or 4
    gpu = _gpu()
    if not gpu:
        print("Attention : pas de carte graphique CUDA détectée, entraînement sur le processeur (très lent).",
              flush=True)
    total = epoques + 3
    print(f"PROGRESSION 1/{total} préparation", flush=True)
    # run_preprocess_script : exp, dataset, sr, cœurs, découpe, effets, débruitage, force, morceau, recouvrement, normalisation
    _lancer(["rvc/train/preprocess/preprocess.py", exp, dataset, sr, cpu, "Automatic", False, False, 0.7,
             3.0, 0.3, "none"], "La préparation des enregistrements")
    print(f"PROGRESSION 2/{total} extraction", flush=True)
    # run_extract_script : exp, méthode f0, cœurs, gpu (« - » = processeur), sr, embedder, embedder perso, silences
    _lancer(["rvc/train/extract/extract.py", exp, F0, cpu, "0" if gpu else "-", sr, EMBEDDER, None, 2],
            "L'extraction des caractéristiques")
    sys.path.insert(0, str(Path.cwd()))
    from rvc.lib.tools.pretrained_selector import pretrained_selector

    pg, pd = pretrained_selector(VOCODEUR, sr)
    if not pg:
        _erreur(f"modèles pré-entraînés {sr} Hz introuvables : lance le téléchargement des modèles de base.")
    epoque_re = re.compile(r"\| epoch=(\d+) \|")

    def suivi(ligne):
        m = epoque_re.search(ligne)
        if m:
            print(f"PROGRESSION {2 + int(m.group(1))}/{total} époque {m.group(1)}/{epoques}", flush=True)

    # run_train_script : nom, sauvegarde tous les N, époques, G, D, gpu, lot, sr, dernière seule, poids à chaque
    # sauvegarde, cache GPU, nettoyage, vocodeur, checkpointing. Sans nettoyage : un entraînement interrompu reprend.
    _lancer(["rvc/train/train.py", nom, max(1, min(10, epoques)), epoques, pg, pd, "0", lot, sr, True, False, False,
             False, VOCODEUR, False], "L'entraînement", suivi)
    print(f"PROGRESSION {total}/{total} index", flush=True)
    _lancer(["rvc/train/process/extract_index.py", exp, "Auto"], "La création de l'index")
    modele = meilleur_modele(exp, nom)
    if not modele:
        _erreur("l'entraînement n'a produit aucun modèle (.pth).")
    print(f"TERMINE {modele}", flush=True)


def meilleur_modele(exp, nom):
    """Modèle final : <nom>_<époque>e_<pas>s.pth avec la plus grande époque."""
    candidats = []
    for p in glob.glob(str(Path(exp) / f"{glob.escape(nom)}_*e_*s.pth")):
        m = re.search(r"_(\d+)e_(\d+)s\.pth$", p)
        if m:
            candidats.append((int(m.group(1)), int(m.group(2)), p))
    return max(candidats)[2] if candidats else None


def convertir(chemin_tache):
    t = json.loads(Path(chemin_tache).read_text(encoding="utf-8"))
    cmd = ["core.py", "infer", "--input-path", t["entree"], "--output-path", t["sortie"],
           "--pth-path", t["modele"], "--index-path", t.get("index") or "",
           "--pitch", int(t.get("demi_tons", 0)), "--index-rate", float(t.get("index_rate", 0.5)),
           "--protect", float(t.get("protect", 0.33)), "--f0-method", F0, "--export-format", "WAV",
           "--embedder-model", EMBEDDER, "--split-audio"]
    print("PROGRESSION 1/1 conversion", flush=True)
    _lancer(cmd, "La conversion RVC")
    if not Path(t["sortie"]).exists():
        _erreur("la conversion RVC n'a produit aucun fichier.")
    print(f"TERMINE {t['sortie']}", flush=True)


if __name__ == "__main__":
    actions = {"entrainer": entrainer, "convertir": convertir}
    _preparer_applio()
    if len(sys.argv) == 2 and sys.argv[1] == "telecharger":
        telecharger()
    elif len(sys.argv) == 3 and sys.argv[1] in actions:
        actions[sys.argv[1]](sys.argv[2])
    else:
        _erreur("usage : rvc_voix.py entrainer|convertir <tache.json> | telecharger")
