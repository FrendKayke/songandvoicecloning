"""Onglet « Modèles 3D » : image d'un objet → modèle 3D (Hunyuan3D-2 : forme « turbo », puis texture peinte).

Texte → 3D : la description française passe par Qwen3-VL (mode « objet » : prompt anglais d'un objet seul, vue de
trois quarts, fond blanc), Z-Image-Turbo en fait une image (data/3d/images/), que l'utilisateur vérifie
avant de lancer la 3D comme pour une image importée.

Révisions : l'image (face ou dos) est corrigée d'après un texte par FLUX.2 klein, puis le modèle est refait ; une vue
de dos (image_dos.<ext>) fait passer la forme à Hunyuan3D-2mv (face + dos).

Rangement : data/3d/<horodatage>/ : image.<ext> (image de départ), image_detouree.png (fond retiré par rembg),
forme.glb (forme blanche), modele.glb (texturé) et, en option, modele.obj + material.mtl + texture PNG ;
creation.json décrit les réglages et la graine (galerie : réafficher, recréer, supprimer).
"""
import shutil
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import diffusion, projets
from .outils import ecrire_creation, nouveau_dossier

# Le modèle de forme est la version « turbo » (distillée) : 5 pas suffisent (gradio_app.py d'Hunyuan3D-2 :
# `value=5 if 'turbo' in args.subfolder else 30`). La qualité se joue sur la résolution de l'octree (finesse de
# l'extraction de la surface, 256 par défaut, 380 dans les exemples du dépôt) et le nombre de faces gardées.
QUALITES = {
    "Aperçu (rapide)": {"etapes": 5, "octree": 192, "faces": 20000},
    "Normale": {"etapes": 5, "octree": 256, "faces": 40000},
    "Fine (plus lente, plus de mémoire)": {"etapes": 5, "octree": 384, "faces": 100000},
    # Jeu dans le navigateur : peu de faces, et modele_web.glb (texture 1024 en JPEG) à côté du modèle complet
    "Web léger (jeu dans le navigateur)": {"etapes": 5, "octree": 192, "faces": 10000, "web": True},
    # modèle de forme complet (non distillé) : 50 pas au lieu de 5, détails plus fins ; une à deux minutes de plus
    "Maximale (modèle complet, 50 étapes, la plus détaillée)": {"etapes": 50, "octree": 384, "faces": 150000,
                                                               "complet": True},
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


# Ajouté au prompt de l'objet : Z-Image-Turbo n'a pas de prompt négatif (guidage nul), on décrit donc ce qu'on veut
OBJET_SEUL = ("single object, centered, full object visible, three-quarter view, plain pure white background, "
              "soft studio lighting, no shadow on the ground, high detail")


def preparer_prompt(texte, progress=gr.Progress()):
    """Description française de l'objet → prompt anglais pour l'image (Qwen3-VL), modifiable avant génération."""
    texte = (texte or "").strip()
    if not texte:
        raise gr.Error("Décris l'objet (en français ou en anglais) : « une potion de soin, fiole rouge, bouchon de liège ».")
    return diffusion.decrire("objet", texte, progress=progress)


def prompt_pret(texte, prompt, progress=gr.Progress()):
    """« Générer » sans avoir préparé le prompt (constaté : erreur « Il manque le prompt ») : le prompt anglais est
    préparé d'abord depuis la description, puis affiché ; un prompt déjà là (préparé ou retouché) est gardé."""
    return (prompt or "").strip() or preparer_prompt(texte, progress=progress)


def generer_image(prompt, graine, progress=gr.Progress()):
    """Prompt anglais → image de l'objet (Z-Image-Turbo, 1024×1024, fond blanc) dans data/3d/images/.
    Renvoie (image, graine de l'image, message)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Il manque le prompt de l'image : clique d'abord sur « Préparer le prompt », ou écris-le en anglais.")
    dossier = cfg.MODELS3D_DIR / "images"
    dossier.mkdir(parents=True, exist_ok=True)
    sortie = nouveau_dossier(dossier)  # un dossier par image : pas de collision de nom
    res = diffusion.image(f"{prompt}, {OBJET_SEUL}", [sortie / "objet.png"], diffusion.graines(1, graine),
                          progress=progress)
    (sortie / "prompt.txt").write_text(prompt, encoding="utf-8")
    return (res["fichiers"][0], res["graines"][0],
            f"✅ Image générée (graine {res['graines'][0]}). Si elle te convient, lance « Créer le modèle 3D » ; "
            "sinon change la graine ou le prompt et regénère.")


# --- Révisions : Hunyuan3D ne modifie pas un modèle d'après un texte, il sculpte d'après des images. On corrige donc
# l'image (FLUX.2 klein, comme « Modifier la photo » de l'onglet Une image), puis on refait le modèle avec la même
# graine. Ce qui est dans le dos de l'objet (un arc, une cape) n'est presque pas visible de face : le modèle
# l'invente ; une vue de dos (dessinée par klein d'après la face, corrigeable de la même façon, ou importée) est
# donnée à Hunyuan3D-2mv avec la face.
VUES = {"La vue de face": "face", "La vue de dos": "dos"}
GARDER_OBJET = ("Keep everything else exactly as in the reference image: same object or character, same proportions, "
                "colors, materials, pose and view angle. Single object, centered, full object visible, plain pure "
                "white background, soft studio lighting")
VUE_DE_DOS = ("Show the very same object or character as in the reference image, seen from directly behind (back "
              "view, turned 180 degrees): same proportions, colors, materials, outfit and equipment, with everything "
              "worn or attached on the back clearly visible and complete. Single object, centered, full object "
              "visible, plain pure white background, soft studio lighting, orthographic view")


def _image_3d(image, nom, progress):
    """Vérifie l'image ; renvoie (chemin, largeur, hauteur) au format de klein (proportions de l'image, ~1 Mpx)."""
    from .images import AUTO, format_pour

    if not image or not Path(image).is_file():
        raise gr.Error(f"Il manque {nom} : importe-la, ou crée d'abord l'image de l'objet.")
    largeur, hauteur = format_pour(AUTO, image)
    return str(image), largeur, hauteur


def corriger_image(image, revision, progress=gr.Progress()):
    """Révision écrite en français (« refais l'arc dans le dos ») → image corrigée par FLUX.2 klein, le reste
    inchangé. Renvoie (image corrigée, message, révision notée)."""
    revision = (revision or "").strip()
    if not revision:
        raise gr.Error("Écris ce qu'il faut corriger, par exemple « refais l'arc : un long arc en bois, entier ».")
    image, largeur, hauteur = _image_3d(image, "l'image à corriger", progress)
    consigne = diffusion.decrire("retouche", revision, image=image, progress=progress)
    dossier = nouveau_dossier(cfg.MODELS3D_DIR / "images")
    res = diffusion.personnage(f"{consigne.rstrip('.')}. {GARDER_OBJET}.", [image], [dossier / "revision.png"],
                               diffusion.graines(1), largeur, hauteur, progress=progress)
    (dossier / "revision.txt").write_text(f"{revision}\n{consigne}\n", encoding="utf-8")
    return (res["fichiers"][0],
            f"✏️ Image corrigée (« {consigne} »). Regarde-la : si elle te convient, « Créer le modèle 3D » (même "
            "graine pour ne changer que le reste au minimum) ; sinon corrige encore ou reformule.", revision)


def creer_vue_de_dos(image, precision="", progress=gr.Progress()):
    """Vue de dos de l'objet (FLUX.2 klein d'après l'image de face). precision (français, facultatif) : ce qui doit
    se voir de dos (« l'arc et le carquois bien visibles »). Renvoie (image de dos, message)."""
    image, largeur, hauteur = _image_3d(image, "l'image de face", progress)
    prompt = VUE_DE_DOS
    if (precision or "").strip():
        prompt += ". " + diffusion.decrire("retouche", precision.strip(), image=image, progress=progress).rstrip(".")
    dossier = nouveau_dossier(cfg.MODELS3D_DIR / "images")
    res = diffusion.personnage(prompt + ".", [image], [dossier / "dos.png"], diffusion.graines(1), largeur, hauteur,
                               progress=progress)
    return (res["fichiers"][0],
            "🔙 Vue de dos créée. Vérifie-la (corrige-la si besoin avec « Que corriger : la vue de dos »), puis « Créer "
            "le modèle 3D » : la forme sera sculptée à partir des deux vues (Hunyuan3D-2mv).")


def reprendre_images(dossier):
    """Images (face, dos) et graine du dernier modèle créé, pour le réviser."""
    import json

    if not dossier or not (Path(dossier) / "creation.json").exists():
        raise gr.Error("Crée d'abord un modèle 3D.")
    infos = json.loads((Path(dossier) / "creation.json").read_text(encoding="utf-8"))
    dos = Path(dossier) / infos["image_dos"] if infos.get("image_dos") else None
    graine = (infos.get("versions") or [{}])[0].get("graine") or 0
    return str(Path(dossier) / infos["image"]), str(dos) if dos else None, graine


def generer(image_path, nom, qualite, texture, graine, formats, description=None, description_fr=None,
            image_graine=None, projet=None, image_dos=None, revision=None, revision_de=None, progress=gr.Progress()):
    """Image (et vue de dos facultative) → modèle 3D. Renvoie (message, GLB à afficher, image détourée, fichiers à
    télécharger, dossier)."""
    if not image_path or not Path(image_path).is_file():
        raise gr.Error("Importe une image de l'objet (PNG ou JPG : un seul objet, bien visible, fond simple).")
    if Path(image_path).suffix.lower() not in EXTENSIONS_IMAGE:
        raise gr.Error("Format d'image non pris en charge : PNG, JPG, WEBP ou BMP.")
    if image_dos and Path(image_dos).suffix.lower() not in EXTENSIONS_IMAGE:
        raise gr.Error("Format de la vue de dos non pris en charge : PNG, JPG, WEBP ou BMP.")
    q = QUALITES.get(qualite) or QUALITES[QUALITE_DEFAUT]
    formats = [f for f in (formats or []) if f in ("glb", "obj")]
    if "glb" not in formats:
        formats.insert(0, "glb")  # toujours produit : c'est lui qu'affiche la visionneuse
    dossier = nouveau_dossier(cfg.MODELS3D_DIR)
    image = dossier / ("image" + Path(image_path).suffix.lower())
    shutil.copy(image_path, image)
    dos = None
    if image_dos and Path(image_dos).is_file():
        dos = dossier / ("image_dos" + Path(image_dos).suffix.lower())
        shutil.copy(image_dos, dos)
    res = diffusion.forme3d(image, dossier, q["etapes"], q["octree"], q["faces"], graine, texture, formats, progress,
                            web=q.get("web", False), complet=q.get("complet", False), dos=dos)
    fichier = res.get("texture") or res["forme"]
    nom = nettoyer_nom(nom)
    ecrire_creation(dossier, {
        "type": "3d", "nom": nom, "projet": projets.nom(projet), "image": image.name, "qualite": qualite if qualite in QUALITES else QUALITE_DEFAUT,
        **{k: v for k, v in q.items() if k not in ("web", "complet")}, "web": bool(q.get("web")),
        "complet": bool(q.get("complet")), "texture": bool(texture), "formats": formats, "description": (description or "").strip() or None,
        "description_fr": (description_fr or "").strip() or None, "image_graine": image_graine,
        "faces_obtenues": res.get("faces"), "image_dos": dos.name if dos else None,
        "revision": (revision or "").strip() or None, "revision_de": revision_de or None,
        "versions": [{"graine": res["graine"], "dossier": ".", "fichier": fichier, "forme": res["forme"],
                      "obj": res.get("obj"), "web": res.get("web")}],
    })
    note = ""
    if texture and not res.get("texture"):
        note = " ; texture non peinte (la peinture demande la carte graphique), forme blanche seulement"
    if res.get("web"):
        note += f" ; version web légère : {Path(res['web']).name}"
    if dos:
        note += " ; forme sculptée d'après la face et le dos (Hunyuan3D-2mv)"
    msg = f"✅ Modèle « {nom} » : {res.get('faces')} faces, graine {res['graine']}{note}. Dossier : {dossier}"
    detouree = dossier / "image_detouree.png"
    return (msg, fichier, str(detouree) if detouree.exists() else None, fichiers_produits(dossier, image.name),
            str(dossier))


def _taille(octets):
    return f"{octets / 1e6:.1f} Mo" if octets >= 1e6 else f"{octets / 1e3:.0f} Ko"


def alleger(dossier, progress=gr.Progress()):
    """Version web du dernier modèle créé : modele_web.glb (texture 1024 px en JPEG). Renvoie (message, fichiers)."""
    if not dossier or not Path(dossier).is_dir():
        raise gr.Error("Crée d'abord un modèle 3D (ou ouvre-le depuis la galerie).")
    d = Path(dossier)
    source = d / "modele.glb" if (d / "modele.glb").exists() else d / "forme.glb"
    if not source.exists():
        raise gr.Error(f"Aucun modèle GLB dans {d}.")
    res = diffusion.alleger(source, d / (source.stem + "_web.glb"), progress=progress)
    if not res.get("textures"):
        msg = (f"🪶 {Path(res['sortie']).name} : {_taille(res['apres'])} (pas de texture à réduire : la forme seule est "
               "déjà légère ; pour moins de faces, choisis la qualité « Web léger »).")
    else:
        msg = f"🪶 {Path(res['sortie']).name} : {_taille(res['avant'])} → {_taille(res['apres'])} (texture 1024 px en JPEG)."
    return msg, fichiers_produits(d, next((p.name for p in d.glob("image.*")), None))


def generer_lot(images, prefixe, qualite, texture, graine, formats, projet=None, progress=gr.Progress()):
    """Plusieurs images à la suite, une création par image (même réglages ; graine 0 = aléatoire pour chacune).
    Une image en échec n'arrête pas les suivantes. Renvoie (rapport, GLB du dernier modèle réussi, dossier)."""
    images = [getattr(i, "name", i) for i in (images or [])]
    if not images:
        raise gr.Error("Ajoute des images (plusieurs fichiers à la fois).")
    prefixe = nettoyer_nom(prefixe, "modele")
    lignes, dernier, dossier = [], None, None
    for n, image in enumerate(images, 1):
        nom = f"{prefixe}_{Path(image).stem}"
        progress((n - 1) / len(images), desc=f"Modèle {n}/{len(images)} : {Path(image).name}…")
        try:
            msg, fichier, _, _, dossier = generer(image, nom, qualite, texture, graine, formats, projet=projet,
                                                  progress=progress)
            dernier = fichier
            lignes.append(f"- {msg}")
        except gr.Error as e:
            lignes.append(f"- ❌ {Path(image).name} : {e.message if hasattr(e, 'message') else e}")
    reussis = sum(1 for l_ in lignes if l_.startswith("- ✅"))
    return (f"**{reussis}/{len(images)} modèle(s) créé(s)** (un dossier par modèle, visibles dans la Galerie) :\n\n"
            + "\n".join(lignes), dernier, dossier)
