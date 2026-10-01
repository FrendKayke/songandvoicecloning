"""Nettoyage de voix — script exécuté DANS l'environnement « nettoyage » (jamais importé par l'application).

    python nettoyage_voix.py <tache.json>     nettoie le fichier décrit par la tâche
    python nettoyage_voix.py --telecharger    télécharge et charge les modèles (sur CPU) puis quitte

Fichier de tâche (UTF-8) : {"entree", "sortie", "niveau"} avec niveau = « leger » (MossFormer2 : bruit),
« fort » (VoiceFixer : bruit, écho, bande passante réduite, saturation) ou « maximal » (les deux).
Sortie : lignes « PROGRESSION i/n », « ERREUR : message » (erreur prévue) et « TERMINE <fichier> ».

Le dossier de travail doit être le dossier du moteur (<lecteur>:\\StudioVoix\\nettoyage) :
- ClearerVoice range ses modèles dans ./checkpoints/<modèle> (clearvoice/networks.py, snapshot_download) ;
- VoiceFixer les range dans ~/.cache/voicefixer, chemin calculé À L'IMPORT (vocoder/config.py) :
  on redirige donc ~ vers ./voicefixer avant de l'importer (sinon ils iraient sur C:).
API vérifiées : ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"]) puis
cv(input_path=…, online_write=False) et cv.write(sortie, output_path=…) (clearvoice/demo.py) ;
VoiceFixer().restore(input=…, output=…, cuda=…, mode=0) (voicefixer/base.py).
"""
import json
import os
import sys
from pathlib import Path

NIVEAUX = {"leger": ["mossformer2"], "fort": ["voicefixer"], "maximal": ["mossformer2", "voicefixer"]}


def _erreur(msg, code=2):
    print(f"ERREUR : {msg}", flush=True)
    sys.exit(code)


def _rediriger_dossier_personnel():
    home = Path.cwd() / "voicefixer"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["USERPROFILE"] = str(home)  # utilisé par os.path.expanduser sous Windows
    os.environ["HOME"] = str(home)


def _gpu():
    import torch

    return torch.cuda.is_available()


def _reparer_mossformer2():
    """ClearerVoice ne télécharge que si « last_best_checkpoint » manque : après un téléchargement interrompu,
    on retire ce petit fichier pour que le modèle (last_best_checkpoint.pt) soit bien récupéré."""
    d = Path.cwd() / "checkpoints" / "MossFormer2_SE_48K"
    if (d / "last_best_checkpoint").exists() and not (d / "last_best_checkpoint.pt").exists():
        (d / "last_best_checkpoint").unlink()


def mossformer2(entree, sortie):
    _reparer_mossformer2()
    from clearvoice import ClearVoice

    cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    cv.write(cv(input_path=str(entree), online_write=False), output_path=str(sortie))


def voicefixer(entree, sortie):
    _rediriger_dossier_personnel()
    from voicefixer import VoiceFixer

    VoiceFixer().restore(input=str(entree), output=str(sortie), cuda=_gpu(), mode=0)


def telecharger():
    print("Téléchargement / vérification des modèles de nettoyage (MossFormer2, VoiceFixer)…", flush=True)
    _reparer_mossformer2()
    from clearvoice import ClearVoice

    ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    _rediriger_dossier_personnel()
    from voicefixer import VoiceFixer

    VoiceFixer()
    print("Modèles de nettoyage prêts.", flush=True)


def nettoyer(chemin_tache):
    with open(chemin_tache, encoding="utf-8") as f:
        t = json.load(f)
    etapes = NIVEAUX.get(t.get("niveau"))
    if not etapes:
        _erreur(f"niveau de nettoyage inconnu : {t.get('niveau')}")
    if not _gpu():
        print("Attention : pas de carte graphique CUDA détectée, nettoyage sur le processeur (plus lent).", flush=True)
    import torch

    sortie = Path(t["sortie"])
    courant = Path(t["entree"])
    try:
        for i, nom in enumerate(etapes, 1):
            print(f"PROGRESSION {i}/{len(etapes)} {nom}", flush=True)
            cible = sortie if i == len(etapes) else sortie.with_name(f"etape{i}_{nom}.wav")
            {"mossformer2": mossformer2, "voicefixer": voicefixer}[nom](courant, cible)
            courant = cible
    except torch.cuda.OutOfMemoryError:
        _erreur("mémoire de la carte graphique insuffisante. Ferme les autres programmes qui utilisent la "
                "carte (jeux, vidéos) ; si tu as désactivé la libération automatique, arrête ACE-Step dans "
                "l'onglet Modèles, puis relance le nettoyage.", 3)
    if not sortie.exists():
        _erreur("le nettoyage n'a produit aucun fichier.")
    print(f"TERMINE {sortie}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        _erreur("usage : nettoyage_voix.py <tache.json> | --telecharger")
    if sys.argv[1] == "--telecharger":
        telecharger()
    else:
        nettoyer(sys.argv[1])
