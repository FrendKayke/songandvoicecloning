"""Demucs et Seed-VC dans un même processus — script exécuté DANS l'environnement de Seed-VC, depuis son dossier.

    python separation.py demucs <arguments de demucs.separate>   séparation (mêmes arguments que python -m demucs)
    python separation.py seedvc <arguments d'inference.py>        conversion de voix chantée (mêmes arguments)
    python separation.py --ranger                                  modèles gardés → mémoire vive (place pour ACE-Step)
    python separation.py --resident <secondes>                     reste ouvert (moteurs/resident.py)

Sans ce script, chaque chanson relançait inference.py, qui relit ses cinq modèles (DiT, Whisper, BigVGAN, CAMPPlus,
RMVPE) avant quelques secondes de calcul : 62 s pour convertir 10 s de voix dans l'essai complet de l'utilisateur.
Ici le code des deux projets est appelé tel quel (demucs.separate.main, inference.main : mêmes fichiers produits),
seul le chargement des modèles est gardé en mémoire (get_model_from_args, load_models).
Entre deux chansons, l'application demande « --ranger » avant ACE-Step : les modèles quittent la carte graphique
mais restent en mémoire vive (quelques Go).
"""
import argparse
import gc
import os
import sys

# Comme la première ligne d'inference.py (modèles de Seed-VC dans son dossier) ; à fixer avant tout import de
# huggingface_hub, qui lit HF_HUB_CACHE à son import (_hors_ligne l'importe avant inference.py).
os.environ["HF_HUB_CACHE"] = os.path.abspath("./checkpoints/hf_cache")

_DEMUCS = {}  # (nom, dépôt) → modèle
_SEEDVC = {}  # réglages de chargement → résultat de inference.load_models


def _erreur(msg, code=2):
    print(f"ERREUR : {msg}", flush=True)
    sys.exit(code)


def _modules(objet, vus=None, profondeur=0):
    """Tous les torch.nn.Module atteignables depuis objet (tuples, dictionnaires, attributs, variables capturées par
    les fonctions que renvoie load_models : semantic_fn garde Whisper, f0_fn le RMVPE, vocoder_fn BigVGAN…)."""
    try:
        import torch
    except ImportError:  # faux moteurs des tests
        return

    vus = set() if vus is None else vus
    if id(objet) in vus or profondeur > 5 or objet is None or isinstance(objet, (str, bytes, int, float, type)):
        return
    vus.add(id(objet))
    if isinstance(objet, torch.nn.Module):
        yield objet
        return
    if isinstance(objet, dict):
        enfants = list(objet.values())
    elif isinstance(objet, (list, tuple, set)):
        enfants = list(objet)
    elif callable(objet) and getattr(objet, "__closure__", None):
        enfants = [c.cell_contents for c in objet.__closure__ if c.cell_contents is not None]
    elif hasattr(objet, "__dict__") and not isinstance(objet, type(sys)):
        enfants = list(vars(objet).values())
    else:
        return
    for enfant in enfants:
        yield from _modules(enfant, vus, profondeur + 1)


def _deplacer(objet, appareil):
    for m in _modules(objet):
        m.to(appareil)


def ranger():
    """Les modèles de Seed-VC quittent la carte graphique (Demucs y passe déjà un sous-modèle à la fois)."""
    for charge in _SEEDVC.values():
        _deplacer(charge, "cpu")
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Modèles de séparation et de conversion en mémoire vive.", flush=True)


def _hors_ligne(actif):
    """huggingface_hub lit HF_HUB_OFFLINE à son import : on règle la valeur qu'il consulte (et la copie de
    transformers) tâche par tâche."""
    try:
        from huggingface_hub import constants
    except ImportError:
        return
    constants.HF_HUB_OFFLINE = actif
    hub = sys.modules.get("transformers.utils.hub")
    if hub is not None and hasattr(hub, "_is_offline_mode"):
        hub._is_offline_mode = actif


def demucs(argv):
    import demucs.separate as separation

    if not getattr(separation, "_studiovoix", False):
        charger = separation.get_model_from_args

        def en_memoire(args):
            cle = (args.name, str(args.repo))
            if cle not in _DEMUCS:
                _DEMUCS[cle] = charger(args)
            else:
                print(f"Demucs : {args.name} déjà en mémoire.", flush=True)
            return _DEMUCS[cle]

        separation.get_model_from_args = en_memoire
        separation._studiovoix = True
    separation.main(argv)


def _parseur_seedvc(str2bool):
    # mêmes options et mêmes valeurs par défaut que le bloc __main__ d'inference.py
    p = argparse.ArgumentParser(prog="inference.py")
    p.add_argument("--source", type=str, default="./examples/source/source_s1.wav")
    p.add_argument("--target", type=str, default="./examples/reference/s1p1.wav")
    p.add_argument("--output", type=str, default="./reconstructed")
    p.add_argument("--diffusion-steps", type=int, default=30)
    p.add_argument("--length-adjust", type=float, default=1.0)
    p.add_argument("--inference-cfg-rate", type=float, default=0.7)
    p.add_argument("--f0-condition", type=str2bool, default=False)
    p.add_argument("--auto-f0-adjust", type=str2bool, default=False)
    p.add_argument("--semi-tone-shift", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--fp16", type=str2bool, default=True)
    return p


def seedvc(argv):
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())  # inference.py, modules/, hf_utils.py (dossier de Seed-VC)
    import inference

    args = _parseur_seedvc(inference.str2bool).parse_args(argv)
    if not getattr(inference, "_studiovoix", False):
        charger = inference.load_models

        def en_memoire(a):
            cle = (bool(a.f0_condition), a.checkpoint, a.config, bool(a.fp16))
            if cle not in _SEEDVC:
                _SEEDVC[cle] = charger(a)
            else:
                print("Seed-VC : modèles déjà en mémoire.", flush=True)
                _deplacer(_SEEDVC[cle], inference.device)
                inference.fp16 = a.fp16  # réglé par load_models, lu par main
            return _SEEDVC[cle]

        inference.load_models = en_memoire
        inference._studiovoix = True
    inference.main(args)


ACTIONS = {"demucs": demucs, "seedvc": seedvc}


def principal(argv):
    if argv == ["--ranger"]:
        ranger()
        return
    if len(argv) < 1 or argv[0] not in ACTIONS:
        _erreur("usage : separation.py demucs|seedvc <arguments> | --ranger | --resident <secondes>")
    hors_ligne = os.environ.get("STUDIOVOIX_HORS_LIGNE") == "1"
    _hors_ligne(hors_ligne)
    try:
        ACTIONS[argv[0]](argv[1:])
    except (OSError, ValueError) as e:
        if not hors_ligne:
            raise
        # modèle présent mais incomplet (téléchargement interrompu) : une fois en ligne pour le compléter
        print(f"Fichier de modèle manquant hors ligne ({type(e).__name__}) : nouvel essai en ligne.", flush=True)
        _hors_ligne(False)
        ACTIONS[argv[0]](argv[1:])


if __name__ == "__main__":
    import resident

    delai = resident.demande(sys.argv[1:])
    if delai is None:
        principal(sys.argv[1:])
    else:
        resident.servir(principal, delai)
