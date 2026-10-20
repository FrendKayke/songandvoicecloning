"""Rigging 3D automatique : squelette et poids de peau d'un modèle 3D (UniRig, MIT), pour l'animer dans un jeu ou dans
Blender.

Environnement dédié (<lecteur>:\\StudioVoix\\rigging\\.venv, Python 3.11 : bpy 4.2), appelé en sous-processus :
moteurs/rigging.py, lancé depuis le code d'UniRig (StudioVoix\\rigging\\UniRig). Poids dans le cache Hugging Face.
Sortie : modele_anime.glb (le modèle d'origine, textures comprises, avec son squelette et ses poids) et
modele_anime.fbx (maillage simplifié + squelette, pour Blender, Unity, Unreal).
"""
import json
import os
import shutil
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import retraits
from . import serveur_acestep
from .outils import lancer_moteur, nouveau_dossier, stream_command
from .serie import _slug

DEPOT = "VAST-AI/UniRig"
FICHIERS = {"squelette": "skeleton/articulation-xl_quantization_256/model.ckpt",
            "poids de peau": "skin/articulation-xl/model.ckpt"}
FORMATS = (".glb", ".gltf", ".fbx", ".obj", ".vrm")


def script() -> Path:
    return cfg.MOTEURS_DIR / "rigging.py"


def _hub() -> Path:
    return Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")) / "hub"


def ckpt_dir() -> Path:
    return _hub() / ("models--" + DEPOT.replace("/", "--"))


def missing_components():
    return [n for n, f in FICHIERS.items() if not any((ckpt_dir() / "snapshots").glob(f"*/{f}"))]


def _env():
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    if not missing_components() and any((_hub() / "models--facebook--opt-350m" / "snapshots").glob("*/config.json")):
        env["HF_HUB_OFFLINE"] = "1"  # modèles présents : aucune requête réseau
    return env


def _installe():
    if not Path(cfg.RIGGING_PYTHON).exists() or not cfg.UNIRIG_DIR.is_dir():
        raise gr.Error(f"Le rigging 3D (UniRig) n'est pas installé ({cfg.RIGGING_PYTHON} introuvable). "
                       f"{retraits.conseil('rigging')}")


def download():
    if not Path(cfg.RIGGING_PYTHON).exists():
        yield f"❌ Python du rigging introuvable : {cfg.RIGGING_PYTHON}. {retraits.conseil('rigging')}"
        return
    yield from stream_command([cfg.RIGGING_PYTHON, str(script()), "telecharger"], cfg.UNIRIG_DIR,
                              f"Téléchargement des modèles d'UniRig (~5,8 Go) vers {ckpt_dir()} …", _env())


def source_du_modele(dossier):
    """GLB d'une création 3D à animer : le modèle texturé, sinon la forme."""
    d = Path(dossier)
    return next((d / n for n in ("modele.glb", "forme.glb") if (d / n).exists()), None)


def animer(dossier=None, fichier=None, graine=0, progress=gr.Progress()):
    """Ajoute un squelette au modèle importé (`fichier`) ou au dernier modèle créé (`dossier`).
    Renvoie (message, GLB animable, fichiers produits)."""
    fichier = getattr(fichier, "name", fichier)
    if fichier:
        source = Path(fichier)
        if source.suffix.lower() not in FORMATS:
            raise gr.Error(f"Format non pris en charge ({source.suffix}) : GLB, GLTF, FBX, OBJ ou VRM.")
        d = nouveau_dossier(cfg.MODELS3D_DIR / "rigging", _slug(source.stem, "modele"))
        source = Path(shutil.copy(source, d / f"original{source.suffix.lower()}"))
    elif dossier and source_du_modele(dossier):
        d, source = Path(dossier), source_du_modele(dossier)
    else:
        raise gr.Error("Crée d'abord un modèle 3D, ou importe un fichier GLB, FBX ou OBJ à animer.")
    _installe()
    if missing_components():
        raise gr.Error("Modèles d'UniRig à télécharger dans « Outils → Modèles et diagnostic » : "
                       + ", ".join(missing_components()) + ".")
    sortie, fbx = d / "modele_anime.glb", d / "modele_anime.fbx"
    tache = d / "tache_rigging.json"
    tache.write_text(json.dumps({"entree": str(source), "sortie": str(sortie), "fbx": str(fbx),
                                 "graine": int(graine or 0) or 12345, "faces": 50000}, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    serveur_acestep.liberer_gpu(progress)
    libelles = {1: "lecture du modèle", 2: "squelette", 3: "lecture du squelette", 4: "poids de peau",
                5: "report sur le modèle"}
    lignes = lancer_moteur([cfg.RIGGING_PYTHON, str(script()), "animer", str(tache)], cfg.UNIRIG_DIR, _env(), "UniRig",
                           lambda i, n: progress(0.05 + 0.9 * (i - 1) / n, desc=f"Rigging : {libelles.get(i, i)}…"),
                           attendu=sortie)
    res = next((json.loads(l_[9:]) for l_ in reversed(lignes) if l_.startswith("RESULTAT ")), {})
    creation = d / "creation.json"
    if creation.exists():  # modèle de la galerie : la version animée y est notée
        infos = json.loads(creation.read_text(encoding="utf-8"))
        infos["anime"] = sortie.name
        creation.write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    msg = (f"🦴 Squelette ajouté ({res.get('os', '?')} os) : `{sortie.name}` (modèle d'origine, textures comprises) et "
           f"`{fbx.name}` (pour Blender, Unity, Unreal), dans {d}. Vérifie-le dans Blender avant d'animer : un os "
           "manquant (queue, ailes) se corrige à la main.")
    return msg, str(sortie), [str(p) for p in (sortie, fbx) if p.exists()]
