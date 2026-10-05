"""Client du moteur « diffusion » (moteurs/diffusion.py) : Qwen3-VL, Stable Audio Open, Z-Image-Turbo, FLUX.2 klein,
Hunyuan3D-2.

Environnement dédié (<lecteur>:\\StudioVoix\\diffusion\\.venv), appelé en sous-processus. Les modèles vont dans
le cache Hugging Face (HF_HOME, dans StudioVoix). Stable Audio Open exige l'acceptation de sa licence sur
Hugging Face : le jeton est enregistré dans <HF_HOME>\\token, là où huggingface_hub le lit.
"""
import json
import os
import random
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import retraits
from . import serveur_acestep
from .outils import lancer_moteur, stream_command

LOCAL_PHOTOS = "local:photos"  # pas un dépôt Hugging Face : fichiers dans <DIFFUSION_DIR>/photos
# nom → (dépôt Hugging Face, libellé, fichiers attendus dans le cache — un par composant essentiel ; un couple
# (autre dépôt, fichier) désigne un fichier d'un autre dépôt)
MODELES = {
    # 4B (8,9 Go, Apache 2.0) : prompts plus fidèles que le 2B ; tient seul sur 12 Go en bf16 (le 8B, 17 Go, non)
    "qwen": ("Qwen/Qwen3-VL-4B-Instruct", "Qwen3-VL-4B (descriptions, traduction)",
             ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]),
    "bruitages": ("stabilityai/stable-audio-open-1.0", "Stable Audio Open 1.0 (bruitages)",
                  ["transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
                   "text_encoder/model.safetensors"]),
    "zimage": ("Tongyi-MAI/Z-Image-Turbo", "Z-Image-Turbo (images : illustrations, texte → 3D)",
               ["text_encoder/model-00001-of-00003.safetensors", "vae/diffusion_pytorch_model.safetensors",
                ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q8_0.gguf")]),
    # encodeur de texte et tokeniseur repris de Z-Image (identiques) : « zimage » est aussi nécessaire
    "personnages": ("black-forest-labs/FLUX.2-klein-4B", "FLUX.2 klein 4B (personnages récurrents)",
                    ["vae/diffusion_pytorch_model.safetensors", "transformer/config.json",
                     ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf")]),
    # Vidéo : encodeur de texte (11,4 Go) et VAE du dépôt officiel, transformeur en GGUF 8 bits
    "video": ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "Wan 2.2 TI2V-5B (vidéo)",
              ["text_encoder/model-00003-of-00003.safetensors", "vae/diffusion_pytorch_model.safetensors",
               "tokenizer/tokenizer.json", ("QuantStack/Wan2.2-TI2V-5B-GGUF", "Wan2.2-TI2V-5B-Q8_0.gguf")]),
    # Photos : BiRefNet (deux dépôts) ; Real-ESRGAN, GFPGAN et YuNet, publiés sur GitHub, dans StudioVoix\diffusion\photos
    "photo_detourage": ("ZhengPeng7/BiRefNet_HR-matting", "BiRefNet (détourage des photos)",
                        ["model.safetensors", "birefnet.py", ("ZhengPeng7/BiRefNet-portrait", "model.safetensors")]),
    "photo_qualite": (LOCAL_PHOTOS, "Real-ESRGAN et GFPGAN (qualité des photos)",
                      ["RealESRGAN_x4plus.pth", "RealESRGAN_x2plus.pth", "realesr-general-x4v3.pth", "GFPGANv1.4.pth",
                       "face_detection_yunet_2023mar.onnx"]),
    # forme : modèle turbo (5 pas) et modèle complet (qualité « Maximale », 50 pas)
    "forme3d": ("tencent/Hunyuan3D-2", "Hunyuan3D-2 forme (image → 3D)",
                ["hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors", "hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors",
                 "hunyuan3d-dit-v2-0/model.fp16.safetensors"]),
    "texture3d": ("tencent/Hunyuan3D-2", "Hunyuan3D-2 texture (peinture)",
                  ["hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.safetensors",
                   "hunyuan3d-delight-v2-0/unet/diffusion_pytorch_model.safetensors"]),
}
LICENCE_BRUITAGES = "https://huggingface.co/stabilityai/stable-audio-open-1.0"
JETONS = "https://huggingface.co/settings/tokens"


def script() -> Path:
    return cfg.MOTEURS_DIR / "diffusion.py"


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))


def dossier_photos() -> Path:
    return cfg.DIFFUSION_DIR / "photos"


def _dossier_depot(depot) -> Path:
    if depot == LOCAL_PHOTOS:
        return dossier_photos()
    return hf_home() / "hub" / ("models--" + depot.replace("/", "--"))


def ckpt_dir(nom) -> Path:
    return _dossier_depot(MODELES[nom][0])


def _present(nom) -> bool:
    for f in MODELES[nom][2]:
        depot, chemin = f if isinstance(f, tuple) else (MODELES[nom][0], f)
        if depot == LOCAL_PHOTOS:
            if not (dossier_photos() / chemin).is_file():
                return False
        elif not any((_dossier_depot(depot) / "snapshots").glob(f"*/{chemin}")):
            return False
    return True


def missing_components():
    return [MODELES[n][1] for n in MODELES if not _present(n)]


def present(nom) -> bool:
    return _present(nom)


def _env():
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    # rembg range son modèle de détourage dans ~/.u2net sinon (donc sur C:)
    env["U2NET_HOME"] = os.environ.get("U2NET_HOME") or str(cfg.DIFFUSION_DIR / "u2net")
    env["STUDIOVOIX_PHOTOS"] = str(dossier_photos())
    return env


def _installe():
    if not Path(cfg.DIFFUSION_PYTHON).exists():
        raise gr.Error(f"Le moteur de diffusion n'est pas installé ({cfg.DIFFUSION_PYTHON} introuvable). "
                       f"{retraits.conseil('diffusion')}")


def _verifier(*noms):
    _installe()
    retires = [MODELES[n][1] for n in noms if not _present(n) and retraits.retire(f"diffusion:{n}")]
    if retires:
        raise gr.Error(", ".join(retires) + " : " + retraits.CONSEIL_REINSTALLER)
    manquants = [MODELES[n][1] for n in noms if not _present(n)]
    if manquants:
        raise gr.Error("Modèle(s) à télécharger dans l'onglet « Modèles » : " + ", ".join(manquants)
                       + (". Pour Stable Audio Open, il faut d'abord un jeton Hugging Face (voir l'onglet Modèles)."
                          if "bruitages" in noms and not _present("bruitages") else "."))


def lancer(action, tache: dict, dossier: Path, nom, progress=None, libelles=None, gpu=True):
    """Écrit tache.json dans dossier, lance l'action et renvoie le dictionnaire de la ligne RESULTAT.
    gpu : l'action utilise la carte graphique (ACE-Step est alors arrêté pour libérer sa mémoire)."""
    dossier.mkdir(parents=True, exist_ok=True)
    fichier = dossier / f"tache_{action}.json"
    fichier.write_text(json.dumps(tache, ensure_ascii=False, indent=1), encoding="utf-8")
    libelles = libelles or {}

    def suivi(i, n):
        if progress:
            progress(0.05 + 0.9 * (i - 1) / n, desc=f"{nom} : {libelles.get(i, f'étape {i}/{n}')}…")

    if progress:
        progress(0.01, desc=f"{nom} : préparation…")
    if gpu:
        serveur_acestep.liberer_gpu(progress)
    annonce = (lambda texte: progress(0.03, desc=f"{nom} : {texte}")) if progress else None
    lignes = lancer_moteur([cfg.DIFFUSION_PYTHON, str(script()), action, str(fichier)], cfg.DIFFUSION_DIR, _env(),
                           nom, suivi, resident="Diffusion", annonce=annonce)
    resultat = next((l_ for l_ in reversed(lignes) if l_.startswith("RESULTAT ")), None)
    if not resultat:
        raise gr.Error(f"{nom} : pas de résultat renvoyé par le moteur.\n" + "\n".join(lignes[-10:]))
    return json.loads(resultat[len("RESULTAT "):])


# --- Actions ----------------------------------------------------------------------------------------------
def decrire(mode, texte=None, image=None, dossier=None, progress=None, suite=None, nombre=None, photos=None):
    """Texte reformulé en anglais (mode « bruitage » ou « objet ») ou description d'une image (« son », « image »).
    suite : {plans, indice, precedent} pour un plan d'une vidéo en plusieurs plans (continuité)."""
    _verifier("qwen")
    dossier = dossier or (cfg.DATA_DIR / "_tmp")
    tache = {"mode": mode}
    if texte:
        tache["texte"] = texte
    if image:
        tache["image"] = str(image)
    if suite:
        tache["suite"] = suite
    if nombre:  # mode « histoire » : nombre d'images (longueur de la réponse de Qwen)
        tache["nombre"] = int(nombre)
    if photos and int(photos) > 1:  # image = planche de plusieurs photos numérotées de gauche à droite
        tache["photos"] = int(photos)
    return lancer("decrire", tache, dossier, "Qwen3-VL", progress,
                  {1: "chargement du modèle de description", 2: "description"})["texte"]


def bruitage(prompt, dossier, duree, variantes, graine, etapes=100, negatif=None, progress=None):
    _verifier("bruitages")
    return lancer("bruitage", {"prompt": prompt, "negatif": negatif or None, "duree": float(duree),
                               "variantes": int(variantes), "etapes": int(etapes), "graine": int(graine or 0),
                               "dossier": str(dossier)}, dossier, "Stable Audio", progress,
                  {1: "chargement de Stable Audio Open", 2: "génération du bruitage"})


def graines(n, graine=None):
    """Graine de chaque image : celle demandée pour la première (si > 0), puis des graines aléatoires."""
    tirage = [random.randint(1, 2**31 - 1) for _ in range(n)]
    if graine and int(graine) > 0:
        tirage[0] = int(graine)
    return tirage


def _prompts(prompt, sorties):
    """Un prompt pour toutes les images, ou une liste (un par image) : {"prompt"} ou {"prompt", "prompts"}."""
    if isinstance(prompt, (list, tuple)):
        if len(prompt) != len(sorties):
            raise ValueError(f"{len(prompt)} prompts pour {len(sorties)} images")
        return {"prompt": prompt[0], "prompts": list(prompt)}
    return {"prompt": prompt}


def image(prompt, sorties, graines_, largeur=1024, hauteur=1024, etapes=9, progress=None):
    """Z-Image-Turbo : une image par chemin de `sorties`, avec la graine correspondante ; prompt = un texte pour
    toutes, ou une liste d'un texte par image. RESULTAT {fichiers, graines}."""
    _verifier("zimage")
    sorties = [str(s) for s in sorties]
    return lancer("image", {**_prompts(prompt, sorties), "sorties": sorties, "graines": [int(g) for g in graines_],
                            "largeur": int(largeur), "hauteur": int(hauteur), "etapes": int(etapes)},
                  Path(sorties[0]).parent, "Z-Image", progress,
                  {1: "chargement de Z-Image-Turbo", 2: f"génération de {len(sorties)} image(s)"})


def personnage(prompt, references, sorties, graines_, largeur=1024, hauteur=1024, etapes=4, progress=None):
    """FLUX.2 klein 4B : images du personnage des `references` (1 à 4 images) dans la scène du prompt (un texte, ou
    une liste d'un texte par image)."""
    _verifier("zimage", "personnages")
    sorties = [str(s) for s in sorties]
    return lancer("personnage", {**_prompts(prompt, sorties), "references": [str(r) for r in references],
                                 "sorties": sorties,
                                 "graines": [int(g) for g in graines_], "largeur": int(largeur),
                                 "hauteur": int(hauteur), "etapes": int(etapes)},
                  Path(sorties[0]).parent, "FLUX.2 klein", progress,
                  {1: "chargement de FLUX.2 klein", 2: f"génération de {len(sorties)} image(s) du personnage"})


def video(prompt, sortie, image=None, largeur=1280, hauteur=704, images=121, etapes=30, graine=0, negatif=None,
          progress=None):
    """Wan 2.2 TI2V-5B : vidéo MP4 (24 images/s) à partir du prompt anglais, et de l'image de départ si donnée.
    RESULTAT {sortie, graine, images, duree, largeur, hauteur, derniere_image}."""
    _verifier("video")
    tache = {"prompt": prompt, "negatif": negatif or None, "image": str(image) if image else None,
             "sortie": str(sortie), "largeur": int(largeur), "hauteur": int(hauteur), "images": int(images),
             "etapes": int(etapes), "graine": int(graine)}
    libelles = {1: "lecture du texte (une à deux minutes)", 2: "chargement de Wan 2.2",
                int(etapes) + 3: "enregistrement de la vidéo"}
    libelles.update({k + 3: f"génération, étape {k + 1}/{int(etapes)}" for k in range(int(etapes))})
    return lancer("video", tache, Path(sortie).parent, "Vidéo", progress, libelles)


def assembler(clips, sortie, fps=24, progress=None, enchaines=True):
    """Clips MP4 bout à bout (première image des clips enchaînés retirée ; enchaines=False : clips indépendants,
    un par image d'une histoire), processeur seulement. RESULTAT {sortie, images, duree}."""
    _installe()
    return lancer("assembler", {"clips": [str(c) for c in clips], "sortie": str(sortie), "fps": int(fps),
                                "enchaines": bool(enchaines)},
                  Path(sortie).parent, "Assemblage", progress, gpu=False)


def detourer(entree, sortie, modele="general", fond=None, masque=None, progress=None):
    """BiRefNet : PNG transparent (fond None) ou sujet sur une couleur « #rrggbb » ou sur son fond flouté (« flou »).
    modele : « general » (tout sujet, cheveux fins) ou « personne ». RESULTAT {sortie, masque, couverture…}."""
    _verifier("photo_detourage")
    tache = {"entree": str(entree), "sortie": str(sortie), "modele": modele, "fond": fond,
             "masque": str(masque) if masque else None}
    return lancer("detourer", tache, Path(sortie).parent, "Détourage", progress,
                  {1: "chargement du modèle de détourage", 2: "détourage", 3: "finitions des bords"})


def ameliorer(entree, sortie, echelle=2, rapide=False, visages=True, force=0.7, progress=None):
    """Real-ESRGAN (agrandissement ×1, ×2 ou ×4 et restauration) puis GFPGAN sur les visages.
    RESULTAT {sortie, largeur, hauteur, visages, modele}."""
    _verifier("photo_qualite")
    tache = {"entree": str(entree), "sortie": str(sortie), "echelle": int(echelle), "rapide": bool(rapide),
             "visages": bool(visages), "force": float(force)}
    return lancer("ameliorer", tache, Path(sortie).parent, "Amélioration", progress,
                  {1: "chargement de Real-ESRGAN", 2: "agrandissement et restauration", 3: "restauration des visages"})


def forme3d(image_path, dossier, etapes, octree, faces, graine, texture, formats=("glb",), progress=None, web=False,
            complet=False):
    """Image → forme.glb (+ modele.glb texturé si `texture` et carte graphique ; + OBJ si demandé).
    complet : modèle de forme complet (hunyuan3d-dit-v2-0, non distillé : 30 à 50 pas) au lieu du turbo.
    RESULTAT {forme, faces, graine, texture: chemin ou None, obj: chemin ou None}."""
    _verifier("forme3d", *(["texture3d"] if texture else []))
    tache = {"image": str(image_path), "dossier": str(dossier), "etapes": int(etapes), "octree": int(octree),
             "faces": int(faces), "graine": int(graine or 0), "texture": bool(texture), "formats": list(formats),
             "web": bool(web)}
    if complet:
        tache["sous_dossier"] = "hunyuan3d-dit-v2-0"
    return lancer("forme3d", tache, dossier,
                  "Hunyuan3D", progress,
                  {1: "détourage de l'image", 2: "chargement du générateur de forme", 3: "génération de la forme",
                   4: "nettoyage et simplification", 5: "chargement du peintre de texture",
                   6: "peinture de la texture (plusieurs minutes)"})


def alleger(entree, sortie, texture_max=1024, progress=None):
    """GLB allégé pour le web (texture réduite, en JPEG). Sur le processeur : ACE-Step n'est pas arrêté."""
    _installe()
    return lancer("alleger", {"entree": str(entree), "sortie": str(sortie), "texture_max": int(texture_max)},
                  Path(sortie).parent, "Allègement", progress, {1: "allègement du modèle"}, gpu=False)


# --- Onglet Modèles ---------------------------------------------------------------------------------------
def jeton_present() -> bool:
    return (hf_home() / "token").is_file() or bool(os.environ.get("HF_TOKEN"))


def enregistrer_jeton(jeton):
    """Enregistre le jeton Hugging Face là où huggingface_hub le lit (<HF_HOME>/token)."""
    jeton = (jeton or "").strip()
    if not jeton.startswith("hf_") or len(jeton) < 20:
        raise gr.Error("Ce n'est pas un jeton Hugging Face (il commence par « hf_ »). Crée-le sur " + JETONS)
    hf_home().mkdir(parents=True, exist_ok=True)
    (hf_home() / "token").write_text(jeton, encoding="ascii")
    return "✅ Jeton enregistré. Tu peux maintenant télécharger Stable Audio Open."


def download(noms=None):
    noms = list(noms or MODELES)
    if not Path(cfg.DIFFUSION_PYTHON).exists():
        yield f"❌ Python du moteur de diffusion introuvable : {cfg.DIFFUSION_PYTHON}. {retraits.conseil('diffusion')}"
        return
    if "bruitages" in noms and not jeton_present():
        yield (f"❌ Stable Audio Open demande d'accepter sa licence sur {LICENCE_BRUITAGES} puis d'enregistrer "
               f"un jeton Hugging Face (créé sur {JETONS}) dans le champ ci-dessus.")
        return
    if "personnages" in noms and "zimage" not in noms and not _present("zimage"):
        noms.append("zimage")  # FLUX.2 klein reprend l'encodeur de texte de Z-Image
    for n in noms:  # télécharger un modèle retiré, c'est le réinstaller : l'installateur ne le saute plus
        retraits.enlever(f"diffusion:{n}")
    yield from stream_command([cfg.DIFFUSION_PYTHON, str(script()), "telecharger", *noms, "detourage"],
                              cfg.DIFFUSION_DIR, f"Téléchargement : {', '.join(MODELES[n][1] for n in noms)} …",
                              _env())
