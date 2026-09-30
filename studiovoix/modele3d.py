"""Onglet « Modèles 3D » : image d'un objet → modèle 3D (Hunyuan3D-2 : forme « turbo », puis texture peinte).

Rangement : data/3d/<horodatage>/ : image.<ext> (image de départ), image_detouree.png (fond retiré par rembg),
forme.glb (forme blanche), modele.glb (texturé) et, en option, modele.obj + material.mtl + texture PNG ;
creation.json décrit les réglages et la graine (galerie : réafficher, recréer, supprimer).
"""
import shutil
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import diffusion
from .outils import ecrire_creation, nouveau_dossier

# Le modèle de forme est la version « turbo » (distillée) : 5 pas suffisent (gradio_app.py d'Hunyuan3D-2 :
# `value=5 if 'turbo' in args.subfolder else 30`). La qualité se joue sur la résolution de l'octree (finesse de
# l'extraction de la surface, 256 par défaut, 380 dans les exemples du dépôt) et le nombre de faces gardées.
QUALITES = {
    "Aperçu (rapide)": {"etapes": 5, "octree": 192, "faces": 20000},
    "Normale": {"etapes": 5, "octree": 256, "faces": 40000},
    "Fine (plus lente, plus de mémoire)": {"etapes": 5, "octree": 384, "faces": 100000},
}
QUALITE_DEFAUT = "Normale"
FORMATS = [("GLB (web, Blender, Unity, Godot)", "glb"), ("OBJ (+ MTL et texture PNG)", "obj")]
EXTENSIONS_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def nettoyer_nom(nom, defaut="modele"):
    return "".join(c for c in (nom or "").strip() if c.isalnum() or c in "-_ ").strip() or defaut


def fichiers_produits(dossier, image=None):
    """Fichiers à télécharger : modèles, matériaux et textures (pas l'image de départ ni l'image détourée)."""
    exclus = {"image_detouree.png", image}
    return sorted(str(p) for p in Path(dossier).iterdir()
                  if p.suffix.lower() in (".glb", ".obj", ".mtl", ".png", ".jpg") and p.name not in exclus)


def generer(image_path, nom, qualite, texture, graine, formats, description=None, description_fr=None,
            image_graine=None, progress=gr.Progress()):
    """Image → modèle 3D. Renvoie (message, GLB à afficher, image détourée, fichiers à télécharger, dossier)."""
    if not image_path or not Path(image_path).is_file():
        raise gr.Error("Importe une image de l'objet (PNG ou JPG : un seul objet, bien visible, fond simple).")
    if Path(image_path).suffix.lower() not in EXTENSIONS_IMAGE:
        raise gr.Error("Format d'image non pris en charge : PNG, JPG, WEBP ou BMP.")
    q = QUALITES.get(qualite) or QUALITES[QUALITE_DEFAUT]
    formats = [f for f in (formats or []) if f in ("glb", "obj")]
    if "glb" not in formats:
        formats.insert(0, "glb")  # toujours produit : c'est lui qu'affiche la visionneuse
    dossier = nouveau_dossier(cfg.MODELS3D_DIR)
    image = dossier / ("image" + Path(image_path).suffix.lower())
    shutil.copy(image_path, image)
    res = diffusion.forme3d(image, dossier, q["etapes"], q["octree"], q["faces"], graine, texture, formats, progress)
    fichier = res.get("texture") or res["forme"]
    nom = nettoyer_nom(nom)
    ecrire_creation(dossier, {
        "type": "3d", "nom": nom, "image": image.name, "qualite": qualite if qualite in QUALITES else QUALITE_DEFAUT,
        **q, "texture": bool(texture), "formats": formats, "description": (description or "").strip() or None,
        "description_fr": (description_fr or "").strip() or None, "image_graine": image_graine,
        "faces_obtenues": res.get("faces"),
        "versions": [{"graine": res["graine"], "dossier": ".", "fichier": fichier, "forme": res["forme"],
                      "obj": res.get("obj")}],
    })
    note = ""
    if texture and not res.get("texture"):
        note = " ; texture non peinte (la peinture demande la carte graphique), forme blanche seulement"
    msg = f"✅ Modèle « {nom} » : {res.get('faces')} faces, graine {res['graine']}{note}. Dossier : {dossier}"
    detouree = dossier / "image_detouree.png"
    return (msg, fichier, str(detouree) if detouree.exists() else None, fichiers_produits(dossier, image.name),
            str(dossier))
