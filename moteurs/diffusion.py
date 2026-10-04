"""Moteur « diffusion » — script exécuté DANS l'environnement diffusion (jamais importé par l'application).

Quatre modèles qui partagent la même pile (diffusers, transformers) :
  - Qwen3-VL-4B (Apache 2.0)      : décrit une image, reformule un texte français en prompt anglais ;
  - Stable Audio Open 1.0          : bruitages à partir d'une description (licence Stability Community,
                                     accès soumis à l'acceptation de la licence : jeton Hugging Face) ;
  - Hunyuan3D-2 (Tencent)          : image → forme 3D (turbo) puis texture (paint turbo + delight) ;
  - Z-Image-Turbo (Apache 2.0)     : texte → image (illustrations de cartes, objet du texte → 3D) ;
  - FLUX.2 klein 4B (Apache 2.0)   : image d'un personnage à partir de 1 à 4 images de référence (même personnage
                                     d'une carte à l'autre) ;
  - Wan 2.2 TI2V-5B (Apache 2.0)   : vidéo à partir d'un texte ou d'une image (720p, 24 images/s) ;
  - Photos : BiRefNet (MIT) détoure ou isole une personne ; Real-ESRGAN (BSD-3) agrandit et restaure, GFPGAN 1.4
    (Apache 2.0) restaure les visages trouvés par YuNet (OpenCV, MIT).

    python diffusion.py <action> <tache.json>      action : decrire | bruitage | image | personnage | video | assembler | detourer | ameliorer | forme3d
                                                   | alleger
    python diffusion.py telecharger [modele…]      qwen | bruitages | forme3d | texture3d | zimage | personnages
                                                   | video | photo_detourage | photo_qualite | detourage

    python diffusion.py --resident <secondes>      reste ouvert et reçoit ses tâches sur l'entrée standard
                                                   (moteurs/resident.py) : les modèles chargés sont gardés (_garder)

Sortie : « PROGRESSION i/n … », « RESULTAT <json> », « ERREUR : message » et « TERMINE <fichier> ».
API vérifiées dans les dépôts : diffusers 0.39 (StableAudioPipeline.__call__, ZImagePipeline, Flux2KleinPipeline),
Qwen3-VL (carte du modèle), Hunyuan3D-2 commit f8db630 (hy3dgen.shapegen.pipelines,
hy3dgen.texgen.pipelines, gradio_app.py pour l'ordre des étapes et le mode basse mémoire).
"""
import gc
import json
import os
import sys
from pathlib import Path

QWEN = "Qwen/Qwen3-VL-4B-Instruct"  # 8,9 Go en bf16 : seul sur la carte, prompts plus fidèles que le 2B
QWEN_COTE_MAX = 1024  # côté maximal des images données à Qwen (jetons visuels, mémoire graphique)
STABLE_AUDIO = "stabilityai/stable-audio-open-1.0"
# Z-Image-Turbo (Alibaba Tongyi-MAI, Apache 2.0) : encodeur de texte (Qwen3-4B), VAE et réglages depuis le dépôt
# officiel ; le transformeur (6 milliards de paramètres, 24,6 Go en fp32 dans le dépôt officiel) depuis sa version
# GGUF 8 bits (7,2 Go, qualité quasi identique), qui tient sur 12 Go avec le déchargement vers la mémoire vive.
ZIMAGE = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_GGUF = ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q8_0.gguf")
# FLUX.2 klein 4B (Black Forest Labs, Apache 2.0 ; les versions 9B sont non commerciales) : transformeur en GGUF 8 bits
# (4,3 Go au lieu de 7,75), VAE et réglages depuis le dépôt officiel. Son encodeur de texte est le même Qwen3-4B que
# celui de Z-Image (même configuration, mêmes poids sur les couches lues : klein lit les couches 9, 18 et 27) : on
# reprend celui de Z-Image, ce qui évite 8 Go de téléchargement. Le modèle est distillé : 4 pas, guidage ignoré.
KLEIN = "black-forest-labs/FLUX.2-klein-4B"
KLEIN_GGUF = ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf")
REFERENCES_MAX = 4  # limite de klein dans l'API de Black Forest Labs ; chaque référence ajoute jusqu'à 4096 jetons
# Wan 2.2 TI2V-5B (Alibaba, Apache 2.0) : un seul modèle pour texte → vidéo et image → vidéo, 720p à 24 images/s.
# Transformeur en GGUF 8 bits (5,4 Go ; 20 Go en bf16 dans le dépôt officiel), encodeur de texte umT5-xxl (11,4 Go)
# et VAE (2,8 Go, fp32 conseillé) depuis le dépôt officiel. Le VAE de Wan 2.2 en fichier unique ne se charge pas
# avec diffusers 0.39 (convertisseur prévu pour Wan 2.1) : il vient du dépôt.
WAN = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_GGUF = ("QuantStack/Wan2.2-TI2V-5B-GGUF", "Wan2.2-TI2V-5B-Q8_0.gguf")
# Prompt négatif de la documentation de Wan, sans « style, works, paintings, images » (on anime aussi des
# illustrations peintes)
NEGATIF_VIDEO = ("Bright tones, overexposed, static, blurred details, subtitles, overall gray, worst quality, low quality, "
                 "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
                 "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
                 "many people in the background, walking backwards, text, watermark")
# Photos. BiRefNet (MIT) : le code du modèle est dans son dépôt (trust_remote_code) : révision épinglée.
# HR-matting : alpha doux (cheveux), entrée 2048² ; portrait : entraîné sur des personnes, entrée 1024².
BIREFNET = {
    "general": ("ZhengPeng7/BiRefNet_HR-matting", "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f", 2048),
    "personne": ("ZhengPeng7/BiRefNet-portrait", "b6561965a70070d9143fd9e558f6ca3c481510db", 1024),
}
REVISIONS = {depot: rev for depot, rev, _ in BIREFNET.values()}
# Agrandissement et visages : poids officiels, publiés seulement sur GitHub (nom → adresse, SHA-256), rangés dans
# StudioVoix\diffusion\photos. spandrel (MIT) les charge ; jamais spandrel_extra_arches (architectures non
# commerciales). facexlib n'est pas utilisé : son modèle de segmentation du visage est non commercial.
FICHIERS_PHOTOS = {
    "RealESRGAN_x4plus.pth": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                              "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"),
    "RealESRGAN_x2plus.pth": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                              "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"),
    "realesr-general-x4v3.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"),
    "GFPGANv1.4.pth": ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
                       "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad"),
    "face_detection_yunet_2023mar.onnx": (
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
}
HUNYUAN = "tencent/Hunyuan3D-2"
HUNYUAN_FORME = "hunyuan3d-dit-v2-0-turbo"
HUNYUAN_TEXTURE = "hunyuan3d-paint-v2-0-turbo"
# nom → [(dépôt, motifs à télécharger ou None pour tout)]
MODELES = {
    "qwen": [(QWEN, None)],
    "bruitages": [(STABLE_AUDIO, None)],
    "zimage": [(ZIMAGE, ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "vae/*",
                         "transformer/config.json"]),
               (ZIMAGE_GGUF[0], [ZIMAGE_GGUF[1]])],
    # l'encodeur de texte et le tokeniseur viennent de Z-Image (« zimage » doit être présent aussi)
    "personnages": [(KLEIN, ["model_index.json", "scheduler/*", "vae/*", "transformer/config.json"]),
                    (KLEIN_GGUF[0], [KLEIN_GGUF[1]])],
    "video": [(WAN, ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "vae/*",
                     "transformer/config.json"]),
              (WAN_GGUF[0], [WAN_GGUF[1]])],
    "photo_detourage": [(depot, None) for depot, _, _ in BIREFNET.values()],
    "photo_qualite": [("local:photos", list(FICHIERS_PHOTOS))],
    # turbo (5 pas, par défaut) et modèle complet (qualité « Maximale ») ; flashvdm prend le VAE turbo pour les deux
    "forme3d": [(HUNYUAN, [f"{HUNYUAN_FORME}/*", "hunyuan3d-vae-v2-0-turbo/*", "hunyuan3d-dit-v2-0/config.yaml",
                           "hunyuan3d-dit-v2-0/model.fp16.safetensors"])],
    "texture3d": [(HUNYUAN, [f"{HUNYUAN_TEXTURE}/*", "hunyuan3d-delight-v2-0/*"])],
}


def _erreur(msg, code=2):
    print(f"ERREUR : {msg}", flush=True)
    sys.exit(code)


def _resultat(obj):
    print("RESULTAT " + json.dumps(obj, ensure_ascii=False), flush=True)


def _device():
    import torch

    if torch.cuda.is_available():
        # TF32 pour ce qui reste en fp32 (VAE de Wan, Hunyuan3D…) et choix automatique des algorithmes de
        # convolution : même résultat visuel, carte mieux employée
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return "cuda"
    print("Attention : pas de carte graphique CUDA détectée, calcul sur le processeur (très lent).", flush=True)
    return "cpu"


def _liberer(*objets):
    """Libère un modèle avant d'en charger un autre : sur 12 Go, on n'en garde jamais deux en mémoire."""
    import torch

    for o in objets:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _memoire(action):
    """Transforme une erreur de mémoire graphique en message clair."""
    import torch

    def enveloppe(*a, **k):
        try:
            return action(*a, **k)
        except torch.cuda.OutOfMemoryError:
            _erreur("mémoire de la carte graphique insuffisante. Ferme les autres programmes qui utilisent la "
                    "carte (jeux, vidéos) ; si tu as désactivé la libération automatique, arrête ACE-Step "
                    "dans l'onglet Modèles, puis relance.", 3)
    return enveloppe


# --- Modèles gardés en mémoire -----------------------------------------------------------------------------
# En mode résident (moteurs/resident.py), le script reste ouvert entre deux tâches : un modèle déjà chargé est
# repris tel quel au lieu d'être relu sur le disque. Un seul modèle à la fois occupe la carte graphique : avant
# d'en servir un, les autres repassent en mémoire vive (vers_cpu) ; au-delà du budget de mémoire vive, les plus
# anciens sont libérés. Les pipelines en enable_model_cpu_offload sont déjà en mémoire vive entre deux appels.
_CHARGES = {}  # clé → [objet, Go de mémoire vive, vers_cpu, vers_gpu] ; ordre = du moins au plus récent


def _budget_go():
    """Mémoire vive que les modèles gardés peuvent occuper : STUDIOVOIX_MEMOIRE_MODELES (Go), sinon la mémoire
    de la machine moins 12 Go (système, application, navigateur) ; 20 Go sur un PC de 32 Go."""
    try:
        return float(os.environ["STUDIOVOIX_MEMOIRE_MODELES"])
    except (KeyError, ValueError):
        pass
    try:
        import psutil

        return max(4.0, psutil.virtual_memory().total / 1e9 - 12)
    except Exception:  # noqa: BLE001
        return 16.0


def _vider_cache_cuda():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _ranger(sauf=None):
    """Les modèles gardés (sauf `sauf`) quittent la carte graphique."""
    for cle, (objet, _, vers_cpu, _) in list(_CHARGES.items()):
        if cle != sauf and vers_cpu:
            vers_cpu(objet)
    _vider_cache_cuda()


def _faire_place(go, sauf=None):
    """Libère les modèles gardés, du plus ancien au plus récent (`sauf` en dernier), jusqu'à ce que `go` Go de
    plus tiennent dans le budget."""
    ordre = [c for c in _CHARGES if c != sauf] + ([sauf] if sauf in _CHARGES else [])
    for cle in ordre:
        if sum(e[1] for e in _CHARGES.values()) + go <= _budget_go():
            break
        print(f"Mémoire vive : libération d'un modèle gardé ({cle[0]}).", flush=True)
        del _CHARGES[cle]
    _vider_cache_cuda()


def _garder(cle, fabrique, go, vers_cpu=None, vers_gpu=None):
    """L'objet de `cle` (tuple : nom du modèle puis ce qui le distingue), créé par fabrique() s'il n'est pas déjà
    en mémoire. go : mémoire vive qu'il occupe (estimation). vers_cpu(objet) / vers_gpu(objet) : le sortir de la
    carte graphique quand un autre modèle travaille, l'y remettre quand il resert (None : rien à faire)."""
    if cle in _CHARGES:
        entree = _CHARGES.pop(cle)
        _CHARGES[cle] = entree
        _ranger(sauf=cle)
        if entree[3]:
            entree[3](entree[0])
        print(f"{cle[0]} déjà en mémoire : pas de rechargement.", flush=True)
        return entree[0]
    _ranger()
    _faire_place(go)
    _verifier_memoire_carte()
    import time

    debut = time.time()
    objet = fabrique()
    print(f"{cle[0]} chargé en {time.time() - debut:.0f} s.", flush=True)
    _CHARGES[cle] = [objet, go, vers_cpu, vers_gpu]
    return objet


def _verifier_memoire_carte():
    """Nos autres modèles viennent de quitter la carte : elle devrait être presque vide. Sinon un autre programme
    l'occupe (Ollama, jeu, autre application d'IA) et le pilote NVIDIA déborde en mémoire vive : tout devient très
    lent (constaté : « Préparer le prompt » à plus de 4 minutes). La ligne AVERTISSEMENT est affichée à
    l'utilisateur par l'application, avec la liste des programmes sur la carte."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        libre, total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - mesure facultative
        return
    print(f"Mémoire graphique libre : {libre / 1e9:.1f} Go sur {total / 1e9:.1f} Go.", flush=True)
    if libre < 0.6 * total:
        print(f"AVERTISSEMENT : seulement {libre / 1e9:.1f} Go libres sur {total / 1e9:.1f} Go de mémoire graphique "
              "avant le chargement : un autre programme occupe la carte, la génération risque d'être très lente.",
              flush=True)


def _oublier(cle):
    """Retire un modèle gardé (état modifié, par exemple déchargement activé après un manque de mémoire)."""
    _CHARGES.pop(cle, None)
    _vider_cache_cuda()


def _lire(chemin):
    return json.loads(Path(chemin).read_text(encoding="utf-8"))


_HORS_LIGNE_IMPOSE = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on")
_EN_LIGNE_FORCE = False  # second essai d'une action après un fichier manquant hors ligne (principal)


def _hors_ligne_si_present(*repos):
    """Modèles déjà téléchargés (tous les dépôts nommés) : aucune requête réseau (ni la question du jeton), sinon
    chaque chargement interroge Hugging Face pour chaque fichier (et attend la fin du délai si la connexion est
    coupée). huggingface_hub lit HF_HUB_OFFLINE une seule fois, à son import (constants.py), et transformers en
    garde une copie (utils/hub.py) : changer la variable d'environnement après coup n'avait aucun effet. On règle
    donc leurs valeurs directement, tâche par tâche (en mode résident, la tâche suivante peut viser un modèle
    absent)."""
    from huggingface_hub import scan_cache_dir

    try:
        presents = {r.repo_id for r in scan_cache_dir().repos if r.revisions}
        hors_ligne = _HORS_LIGNE_IMPOSE or (set(repos) <= presents and not _EN_LIGNE_FORCE)
    except Exception:  # noqa: BLE001 - cache illisible : on garde le réglage actuel
        return
    _hors_ligne(hors_ligne)


def _precharger(*parties):
    """Lit d'un bloc, dans l'ordre, les fichiers de poids avant leur chargement. parties : (dépôt, sous-dossier ou
    nom de fichier ou None). Les chargeurs (safetensors, GGUF) projettent le fichier en mémoire et le lisent par
    morceaux, dans le désordre : sur un disque dur, les déplacements de la tête ramenaient la lecture à ~40 Mo/s
    (umT5 de Wan : 250 s), contre ~120 Mo/s en lecture suivie (mesuré sur la machine de l'utilisateur, modèles sur un
    disque dur). Une fois lus, les fichiers sont dans le cache de Windows : le chargement les y reprend. Sur un SSD
    ou si les fichiers sont déjà en mémoire, quelques secondes seulement."""
    import time

    from huggingface_hub import constants

    fichiers = []
    for depot, partie in parties:
        racine = Path(constants.HF_HUB_CACHE) / ("models--" + depot.replace("/", "--")) / "snapshots"
        for instantane in sorted(racine.glob("*")):
            cible = instantane / partie if partie else instantane
            if cible.is_file():
                fichiers.append(cible)
            elif cible.is_dir():
                fichiers += [f for f in sorted(cible.rglob("*"))
                             if f.suffix in (".safetensors", ".gguf", ".bin", ".pth", ".pt") and f.is_file()]
    total = sum(f.stat().st_size for f in fichiers)
    if total < 200e6:
        return
    debut = time.time()
    tampon = bytearray(16 << 20)
    for f in fichiers:
        try:
            with open(f, "rb", buffering=0) as h:
                while h.readinto(tampon):
                    pass
        except OSError:
            pass
    duree = time.time() - debut
    print(f"Fichiers lus : {total / 1e9:.1f} Go en {duree:.0f} s ({total / 1e6 / max(duree, 0.01):.0f} Mo/s).",
          flush=True)


def _en_memoire_vive(*modules):
    """Copie les poids restés sur le processeur dans la mémoire du moteur. transformers ne lit pas les safetensors :
    il les projette en mémoire (mmap) et les tenseurs pointent sur le fichier. Si la lecture d'un autre gros fichier
    (le GGUF du transformeur) chasse ces pages du cache de Windows, le passage sur la carte relit le disque dur
    au hasard : encodeur de Z-Image/klein (8 Go) envoyé sur la carte à ~40 Mo/s, 170 à 215 s d'attente (constaté).
    À appeler juste après le chargement, quand _precharger vient de mettre le fichier en cache : copie en mémoire."""
    import torch

    with torch.no_grad():
        for m in modules:
            if m is None:
                continue
            for t_ in list(m.parameters()) + list(m.buffers()):
                if t_.device.type == "cpu":
                    t_.data = t_.data.clone()


def _hors_ligne(actif):
    from huggingface_hub import constants

    constants.HF_HUB_OFFLINE = actif
    hub = sys.modules.get("transformers.utils.hub")
    if hub is not None and hasattr(hub, "_is_offline_mode"):
        hub._is_offline_mode = actif


# --- Qwen3-VL : description d'image, reformulation de texte ------------------------------------------
CONSIGNES = {
    "son": ("Describe, in English, the sound effect this image would make in a video game, as a prompt for a "
            "sound generator: sources of sound, material, intensity, environment. One or two sentences, "
            "no music, no introduction."),
    "objet": ("Rewrite the following description into an English prompt for an image generator that will "
              "produce ONE single object for a 3D model: describe the object, its materials and colors, seen "
              "from a three-quarter view, centered on a plain white background, no scene, no text. "
              "Answer with the prompt only:\n\n"),
    "bruitage": ("Rewrite the following description into a concise English prompt for a sound effect generator "
                 "(sources, material, intensity, environment; no music unless asked). Answer with the prompt "
                 "only:\n\n"),
    "image": ("Describe this image in English in two sentences, as a prompt for an image generator: the main "
              "object, its materials and colors. No introduction."),
    "video": ("Translate and rewrite the following description into an English prompt for a text-to-video "
              "generator. Keep EVERY element of the description: each character and how many there are, what each "
              "one does, the creatures, the setting and the camera movement, in the same order. Do not invent "
              "characters or change them. Add only visual details (lighting, colors, mood). Present tense, 2 to 4 "
              "sentences, no artist or existing work. Answer with the prompt only:\n\n"),
    "carte": ("Rewrite the following description into an English prompt for an image generator that will paint "
              "ONE illustration for a fantasy trading card: main subject, pose or action, setting, lighting, colors "
              "and mood, composition centered on the subject. Do not mention any art style, artist or existing "
              "work. No text, no letters, no card frame, no border, no user interface. Answer with the prompt "
              "only:\n\n"),
}


# Reformulation d'un texte accompagné de l'image de départ (vidéo) : la scène doit rester celle de l'image.
IMAGE_DE_DEPART = ("The attached image is the first frame of the video. Keep its characters, their number, clothes, "
                   "setting, colors and lighting exactly as they appear in the image. ")

# Plan d'une suite (plusieurs plans enchaînés) : chaque plan était reformulé seul, sans les autres ; « A kraken
# appears » devenait un océan sans les aventuriers, « They are fighting him » une ruelle à néons avec d'autres
# personnages (constaté). Qwen reçoit toute l'histoire et le prompt du plan précédent.
SUITE_VIDEO = ("This is shot {i} of {n} of ONE continuous video: each shot starts from the last frame of the previous "
               "shot. The whole story, shot by shot:\n{histoire}\n\n{precedent}"
               "Write the prompt for shot {i} only. Keep the SAME characters (same number, same faces, same clothes), "
               "the SAME place, the SAME lighting and colors as the previous shots, unless shot {i} explicitly says "
               "otherwise; a new creature or object appears in that same place. Replace pronouns (they, him, it) by "
               "who or what they refer to in the story. ")

# Modes qui reformulent un texte : leur consigne se termine par « \n\n » et le texte y est ajouté. Déduit des
# consignes : une liste écrite à la main avait oublié « video », et Qwen inventait une scène sans rapport (constaté).
MODES_TEXTE = {mode for mode, consigne in CONSIGNES.items() if consigne.endswith("\n\n")}


def decrire(chemin_tache):
    """Tâche : {mode: son|image (d'après une image) ou objet|bruitage|carte|video (texte reformulé), image?, texte?}.
    Renvoie RESULTAT {"texte": …}."""
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    t = _lire(chemin_tache)
    mode = t.get("mode")
    if mode not in CONSIGNES:
        _erreur(f"mode de description inconnu : {mode}")
    device = _device()
    _hors_ligne_si_present(QWEN)
    print("PROGRESSION 1/2 chargement de Qwen3-VL", flush=True)

    def charger():  # sans device_map : un modèle « dispatché » par accelerate ne se déplace plus avec .to()
        _precharger((QWEN, None))
        m = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
        return m, AutoProcessor.from_pretrained(QWEN)

    modele, processeur = _garder(("Qwen3-VL", device), charger, 9, vers_cpu=lambda o: o[0].to("cpu"),
                                 vers_gpu=lambda o: o[0].to(device))
    contenu = []
    if t.get("image"):
        # Qwen3-VL découpe l'image en carrés de 32 px à sa résolution d'origine : une illustration de 2900×4060
        # donnait ~11 000 jetons et un manque de mémoire graphique (constaté) ; 1024 px suffisent pour la décrire
        with Image.open(t["image"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((QWEN_COTE_MAX, QWEN_COTE_MAX), Image.LANCZOS)
        contenu.append({"type": "image", "image": im})
    consigne = CONSIGNES[mode]
    if t.get("image") and mode in MODES_TEXTE:  # vidéo à partir d'une image : Qwen la voit
        consigne = IMAGE_DE_DEPART + consigne
    suite = t.get("suite")  # {plans: [textes], indice (1…n), precedent: prompt du plan précédent ou None}
    if suite and mode in MODES_TEXTE:
        plans = suite["plans"]
        histoire = "\n".join(f"Shot {k}: {p}" for k, p in enumerate(plans, 1))
        precedent = f"Prompt of shot {suite['indice'] - 1}: {suite['precedent']}\n\n" if suite.get("precedent") else ""
        consigne = SUITE_VIDEO.format(i=suite["indice"], n=len(plans), histoire=histoire,
                                      precedent=precedent) + consigne
    if mode in MODES_TEXTE:
        texte_ = (t.get("texte") or "").strip()
        if not texte_:
            _erreur("description vide : écris ce que tu veux obtenir.")
        consigne += texte_
    contenu.append({"type": "text", "text": consigne})
    entrees = processeur.apply_chat_template([{"role": "user", "content": contenu}], tokenize=True,
                                             add_generation_prompt=True, return_dict=True, return_tensors="pt")
    entrees = entrees.to(modele.device)
    print("PROGRESSION 2/2 description", flush=True)
    with torch.inference_mode():
        sortie = _memoire(modele.generate)(**entrees, max_new_tokens=160, do_sample=False)
    texte = processeur.batch_decode(sortie[:, entrees["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    texte = " ".join(texte.strip().strip('"').split())
    _resultat({"texte": texte})
    print("TERMINE -", flush=True)


# --- Stable Audio Open : bruitages ----------------------------------------------------------------------
def bruitage(chemin_tache):
    """Tâche : {prompt, negatif?, duree (1–47 s), variantes (1–3), etapes, graine, dossier}.
    Écrit dossier/variante_<i>.wav (44,1 kHz stéréo) ; RESULTAT {"fichiers": […], "graines": […]}."""
    import soundfile as sf
    import torch
    from diffusers import StableAudioPipeline

    t = _lire(chemin_tache)
    device = _device()
    dossier = Path(t["dossier"])
    dossier.mkdir(parents=True, exist_ok=True)
    duree = min(47.0, max(1.0, float(t.get("duree", 5))))
    variantes = max(1, min(3, int(t.get("variantes", 1))))
    print("PROGRESSION 1/2 chargement de Stable Audio Open", flush=True)
    depot = t.get("depot", STABLE_AUDIO)
    _hors_ligne_si_present(depot)

    def charger():
        _precharger((depot, "transformer"), (depot, "vae"), (depot, "text_encoder"))
        try:
            return StableAudioPipeline.from_pretrained(
                depot, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        except Exception as e:  # accès refusé : licence non acceptée ou jeton absent
            if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower() or "Access" in str(e):
                _erreur("Stable Audio Open n'est pas téléchargé : accepte sa licence sur Hugging Face et enregistre "
                        "ton jeton (onglet Modèles → Télécharger les bruitages).", 4)
            raise

    pipe = _garder(("Stable Audio Open", depot, device), charger, 3.5, vers_cpu=lambda p_: p_.to("cpu"),
                   vers_gpu=lambda p_: p_.to(device))
    graines = [int(t.get("graine") or 0) or int(torch.randint(1, 2**31 - 1, (1,)))]
    graines += [int(x) for x in torch.randint(1, 2**31 - 1, (variantes - 1,))]
    generateurs = [torch.Generator(device).manual_seed(g) for g in graines]
    print("PROGRESSION 2/2 génération", flush=True)
    sortie = _memoire(pipe)(
        prompt=[t["prompt"]] * variantes, negative_prompt=[t.get("negatif") or "Low quality."] * variantes,
        num_inference_steps=int(t.get("etapes", 100)), audio_end_in_s=duree, num_waveforms_per_prompt=1,
        generator=generateurs,
    )
    fichiers = []
    for i, audio in enumerate(sortie.audios, 1):
        chemin = dossier / f"variante_{i}.wav"
        sf.write(str(chemin), audio.T.float().cpu().numpy(), pipe.vae.sampling_rate)
        fichiers.append(str(chemin))
    _resultat({"fichiers": fichiers, "graines": graines, "frequence": pipe.vae.sampling_rate})
    print(f"TERMINE {fichiers[0]}", flush=True)


# --- Z-Image-Turbo : images (illustrations de cartes, objet pour le texte → 3D) ------------------------------
def image(chemin_tache):
    """Tâche : {prompt, sorties: [chemins], graines: [entiers], etapes, largeur, hauteur}.
    Une image par sortie, chacune avec sa graine (modèle chargé une seule fois). Z-Image-Turbo est distillé :
    9 pas, sans guidage (guidance_scale=0) donc sans prompt négatif (carte du modèle). RESULTAT {fichiers, graines}.
    Clés de test : depot / gguf (autre modèle, fichier local), sans_encodeur (plongements aléatoires)."""
    import torch
    from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel
    from huggingface_hub import hf_hub_download

    t = _lire(chemin_tache)
    device = _device()
    depot = t.get("depot", ZIMAGE)
    _hors_ligne_si_present(depot, ZIMAGE_GGUF[0])
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print("PROGRESSION 1/2 chargement de Z-Image-Turbo", flush=True)
    sans_encodeur = bool(t.get("sans_encodeur"))

    def charger():
        gguf = t.get("gguf") or hf_hub_download(*ZIMAGE_GGUF)
        # chaque fichier lu juste avant son chargement : lus tous d'avance (15 Go), les premiers sortaient du cache
        _precharger(ZIMAGE_GGUF if not t.get("gguf") else (depot, "-"))
        transformeur = ZImageTransformer2DModel.from_single_file(
            gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), config=depot,
            subfolder="transformer", torch_dtype=dtype)
        _precharger((depot, "vae"), (depot, "text_encoder"))
        pipe = ZImagePipeline.from_pretrained(depot, transformer=transformeur, torch_dtype=dtype,
                                              **({"text_encoder": None, "tokenizer": None} if sans_encodeur else {}))
        _en_memoire_vive(pipe.text_encoder, pipe.vae)
        return pipe if device == "cuda" else pipe.to(device)  # sur la carte : placement à la main (_sur_carte)

    pipe = _garder(("Z-Image-Turbo", depot, t.get("gguf"), sans_encodeur, device), charger, 16,
                   vers_cpu=lambda p_: _sur_carte(p_))
    if sans_encodeur:  # test sans les 8 Go de l'encodeur : plongements de la bonne dimension
        options = {"prompt_embeds": [torch.randn(24, pipe.transformer.config.cap_feat_dim, dtype=dtype)]}
    else:
        if device == "cuda":
            _sur_carte(pipe, "vae", "text_encoder")
        options = {"prompt_embeds": _encoder_une_fois(pipe, t["prompt"], do_classifier_free_guidance=False)}
    if device == "cuda":
        _sur_carte(pipe, "vae", "transformer")
        options["prompt_embeds"] = [e.to(device) for e in options["prompt_embeds"]]
    _generer(pipe, t, dict(options, guidance_scale=0.0), etapes=9)


def _sur_carte(pipe, *noms):
    """Placement à la main sur 12 Go : les composants nommés (« text_encoder », « transformer », « vae ») sur la
    carte, les autres en mémoire vive. Avec enable_model_cpu_offload, diffusers renvoyait tout en mémoire vive à la
    fin de CHAQUE appel (maybe_free_model_hooks) : chaque variante rechargeait le transformeur (4 à 7 Go) et la
    carte attendait. Ici : encodeur seul le temps de lire le texte, puis transformeur et VAE pour toutes les
    variantes, et ils y restent pour la tâche suivante (moteur résident).
    pipe._execution_device (où la pipeline crée latents et plongements) est fixé à la carte : diffusers le déduit du
    premier module par ordre alphabétique (DiffusionPipeline.device, _get_signature_keys trié), donc de
    text_encoder, resté en mémoire vive pendant la génération → « cuda:0 and cpu » (constaté sur la RTX 4070)."""
    import torch

    if noms and not getattr(type(pipe), "_sur_carte_fixe", False):
        classe = type(pipe)
        pipe.__class__ = type(classe.__name__, (classe,), {
            "_sur_carte_fixe": True, "_execution_device": property(lambda self: torch.device("cuda"))})
    for nom in ("text_encoder", "transformer", "vae"):
        m = getattr(pipe, nom, None)
        if m is not None and nom not in noms and m.device.type != "cpu":
            m.to("cpu")
    _vider_cache_cuda()
    for nom in noms:
        m = getattr(pipe, nom, None)
        if m is not None and m.device.type != "cuda":
            m.to("cuda")
    if noms and torch.cuda.is_available():
        torch.cuda.synchronize()


def _encoder_une_fois(pipe, prompt, **options):
    """Le texte est encodé une seule fois pour toutes les variantes. Sinon, sur 12 Go (déchargement vers la mémoire
    vive), chaque variante recharge l'encodeur (8 Go) puis le transformeur sur la carte : la carte attend les
    transferts au lieu de calculer."""
    import torch

    with torch.no_grad():
        return pipe.encode_prompt(prompt=prompt, device=pipe._execution_device, **options)[0]


def _generer(pipe, t, options, etapes):
    """Une image par sortie de la tâche, chacune avec sa graine (générateur sur le processeur : reproductible)."""
    import torch

    largeur, hauteur = int(t.get("largeur", 1024)), int(t.get("hauteur", 1024))
    sorties, graines = t["sorties"], [int(g) for g in t["graines"]]
    print(f"PROGRESSION 2/2 génération de {len(sorties)} image(s)", flush=True)
    for i, (sortie, graine) in enumerate(zip(sorties, graines), 1):
        im = _memoire(pipe)(
            **options, height=hauteur, width=largeur, num_inference_steps=int(t.get("etapes", etapes)),
            generator=torch.Generator("cpu").manual_seed(graine),
        ).images[0]
        Path(sortie).parent.mkdir(parents=True, exist_ok=True)
        im.save(sortie)
        print(f"Image {i}/{len(sorties)} : {sortie} (graine {graine})", flush=True)
    _resultat({"fichiers": sorties, "graines": graines})
    print(f"TERMINE {sorties[0]}", flush=True)


# --- FLUX.2 klein 4B : le même personnage d'une image à l'autre -----------------------------------------------
def personnage(chemin_tache):
    """Tâche : {prompt, references: [images], sorties, graines, etapes, largeur, hauteur}.
    Les images de référence (1 à 4 : visage, en pied, tenue…) sont encodées par le VAE et données au transformeur
    avec le texte (argument `image` de Flux2KleinPipeline.__call__, images PIL obligatoirement) ; la taille de sortie
    est toujours donnée (sinon klein prend celle de la première référence). 4 pas, guidage 1 (carte du modèle).
    Clés de test : depot / gguf / depot_encodeur (autres modèles), sans_encodeur (plongements aléatoires)."""
    import torch
    from diffusers import (AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline,
                           Flux2Transformer2DModel, GGUFQuantizationConfig)
    from huggingface_hub import hf_hub_download
    from PIL import Image

    t = _lire(chemin_tache)
    references = [Image.open(r).convert("RGB") for r in t.get("references") or []][:REFERENCES_MAX]
    if not references:
        _erreur("il faut au moins une image de référence du personnage.")
    device = _device()
    depot, depot_encodeur = t.get("depot", KLEIN), t.get("depot_encodeur", ZIMAGE)
    _hors_ligne_si_present(depot, KLEIN_GGUF[0], depot_encodeur)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print("PROGRESSION 1/2 chargement de FLUX.2 klein", flush=True)
    sans_encodeur = bool(t.get("sans_encodeur"))

    def charger():
        gguf = t.get("gguf") or hf_hub_download(*KLEIN_GGUF)
        _precharger(KLEIN_GGUF if not t.get("gguf") else (depot, "-"))  # chaque fichier juste avant son chargement
        # config= obligatoire : sans lui, diffusers reconnaît « flux-2-dev » et lit la configuration de FLUX.2-dev
        # (dépôt soumis à licence non commerciale)
        transformeur = Flux2Transformer2DModel.from_single_file(
            gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), config=depot,
            subfolder="transformer", torch_dtype=dtype)
        encodeur = tokeniseur = None
        if not sans_encodeur:
            from transformers import AutoTokenizer, Qwen3ForCausalLM

            _precharger((depot_encodeur, "text_encoder"))
            encodeur = Qwen3ForCausalLM.from_pretrained(depot_encodeur, subfolder="text_encoder", torch_dtype=dtype)
            _en_memoire_vive(encodeur)
            tokeniseur = AutoTokenizer.from_pretrained(depot_encodeur, subfolder="tokenizer")
        # pipeline assemblée composant par composant : from_pretrained voudrait aussi les 16 Go de poids officiels
        pipe = Flux2KleinPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(depot, subfolder="scheduler"),
            vae=AutoencoderKLFlux2.from_pretrained(depot, subfolder="vae", torch_dtype=dtype),
            text_encoder=encodeur, tokenizer=tokeniseur, transformer=transformeur, is_distilled=True)
        return pipe if device == "cuda" else pipe.to(device)  # sur la carte : placement à la main (_sur_carte)

    pipe = _garder(("FLUX.2 klein", depot, depot_encodeur, t.get("gguf"), sans_encodeur, device), charger, 13,
                   vers_cpu=lambda p_: _sur_carte(p_))
    if sans_encodeur:  # 3 couches cachées de l'encodeur mises bout à bout
        options = {"prompt_embeds": torch.randn(1, 24, pipe.transformer.config.joint_attention_dim, dtype=dtype)}
    else:
        if device == "cuda":
            _sur_carte(pipe, "vae", "text_encoder")  # encodeur : 8 Go, seul sur la carte
        options = {"prompt_embeds": _encoder_une_fois(pipe, t["prompt"])}
    if device == "cuda":
        _sur_carte(pipe, "vae", "transformer")  # transformeur (4,3 Go) et VAE : sur la carte pour toutes les variantes
        options["prompt_embeds"] = options["prompt_embeds"].to(device)
    print(f"{len(references)} image(s) de référence du personnage.", flush=True)
    _generer(pipe, t, dict(options, image=references, guidance_scale=1.0), etapes=4)


# --- Wan 2.2 : vidéo à partir d'un texte ou d'une image ----------------------------------------------------
def _encoder_texte_video(t, device, dtype):
    """Phase 1 : umT5-xxl (11,4 Go en bf16, trop gros pour 12 Go d'un bloc) encode le prompt et le négatif.
    Sur la carte graphique, par groupes de 4 blocs chargés tour à tour (diffusers.hooks.apply_group_offloading) ;
    si cela échoue, sur le processeur (comme l'option --t5_cpu de Wan). Le modèle est libéré avant la phase 2."""
    import torch
    from diffusers import WanPipeline
    from transformers import AutoTokenizer, UMT5EncoderModel

    depot = t.get("depot_encodeur") or t.get("depot", WAN)
    if t.get("sans_encodeur"):  # tests : plongements aléatoires de la bonne forme (512 jetons × 4096)
        g = torch.Generator("cpu").manual_seed(0)
        return (torch.randn(1, 512, 4096, generator=g).to(dtype), torch.randn(1, 512, 4096, generator=g).to(dtype))
    tokeniseur = AutoTokenizer.from_pretrained(depot, subfolder="tokenizer")
    _precharger((depot, "text_encoder"))  # relu à chaque vidéo : depuis le cache de Windows s'il l'a gardé
    encodeur = UMT5EncoderModel.from_pretrained(depot, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    cible = "cpu"
    if device == "cuda":
        try:
            from diffusers.hooks import apply_group_offloading

            apply_group_offloading(encodeur, onload_device=torch.device("cuda"), offload_type="block_level",
                                   num_blocks_per_group=4)
            cible = "cuda"
        except Exception as e:  # noqa: BLE001 - repli sur le processeur, plus lent mais sûr
            print(f"Encodeur de texte sur le processeur ({type(e).__name__} : {e}).", flush=True)
    pipe = WanPipeline(tokenizer=tokeniseur, text_encoder=encodeur, vae=None, scheduler=None, transformer=None,
                       expand_timesteps=True)
    with torch.no_grad():
        try:
            prompt, negatif = pipe.encode_prompt(prompt=t["prompt"], negative_prompt=t.get("negatif") or NEGATIF_VIDEO,
                                                 do_classifier_free_guidance=True, max_sequence_length=512,
                                                 device=cible)
        except Exception as e:  # noqa: BLE001
            if cible == "cpu":
                raise
            print(f"Encodage sur la carte graphique impossible ({type(e).__name__}) : sur le processeur.", flush=True)
            _liberer(pipe, encodeur)
            encodeur = UMT5EncoderModel.from_pretrained(depot, subfolder="text_encoder", torch_dtype=torch.bfloat16)
            pipe = WanPipeline(tokenizer=tokeniseur, text_encoder=encodeur, vae=None, scheduler=None,
                               transformer=None, expand_timesteps=True)
            prompt, negatif = pipe.encode_prompt(prompt=t["prompt"],
                                                 negative_prompt=t.get("negatif") or NEGATIF_VIDEO,
                                                 do_classifier_free_guidance=True, max_sequence_length=512,
                                                 device="cpu")
    prompt, negatif = prompt.to("cpu", dtype), negatif.to("cpu", dtype)
    del pipe, encodeur
    _liberer()
    return prompt, negatif


def _par_blocs_wan(transformeur, vae):
    """Sur 12 Go, seuls les poids du bloc en cours sont sur la carte ; le suivant arrive pendant le calcul (flux CUDA).
    Avec enable_model_cpu_offload, tout le transformeur (5,4 Go) restait sur la carte pendant le débruitage : en 720p
    et 121 images (~109 000 jetons), les activations ne tenaient plus et le pilote débordait en mémoire vive
    (« Sysmem Fallback ») : 146 s par étape, carte à 60–100 W ; par blocs : 31 s (mesuré sur la RTX 4070). Le VAE
    (fp32, 2,8 Go) monte entier sur la carte quand il sert (crochet d'accelerate, déclenché par encode/decode) et en
    sort pendant le débruitage (crochet.offload()) ; module par module, le décodage en tuiles prenait 265 s.
    Renvoie le crochet du VAE."""
    import torch
    from accelerate import cpu_offload_with_hook
    from diffusers.hooks import apply_group_offloading

    carte = torch.device("cuda")
    apply_group_offloading(transformeur, onload_device=carte, offload_type="block_level", num_blocks_per_group=1,
                           use_stream=True)
    return cpu_offload_with_hook(vae, execution_device=carte)[1]


def video(chemin_tache):
    """Tâche : {prompt, negatif, image (facultative), sortie, largeur, hauteur, images, etapes, guidage, graine, fps}.
    Sans image : texte → vidéo (WanPipeline) ; avec image : la vidéo part de cette image (WanImageToVideoPipeline,
    sans encodeur d'image pour le 5B). images = 4k + 1 (121 = 5 s à 24 images/s) ; côtés multiples de 32.
    Le transformeur et le VAE (en tuiles) passent tour à tour sur la carte (enable_model_cpu_offload).
    Export H.264 (imageio-ffmpeg), lisible par les navigateurs, et dernière image en PNG.
    RESULTAT {sortie, graine, images, duree, largeur, hauteur, derniere_image}.
    Clés de test : depot / depot_encodeur / gguf (autres modèles), sans_encodeur (plongements aléatoires)."""
    import torch
    from diffusers import (AutoencoderKLWan, GGUFQuantizationConfig, UniPCMultistepScheduler, WanImageToVideoPipeline,
                           WanPipeline, WanTransformer3DModel)
    from diffusers.utils import export_to_video
    from huggingface_hub import hf_hub_download

    t = _lire(chemin_tache)
    device = _device()
    depot = t.get("depot", WAN)
    _hors_ligne_si_present(depot, WAN_GGUF[0])
    dtype = torch.bfloat16
    etapes = int(t.get("etapes", 30))
    total = etapes + 3
    cle = ("Wan 2.2", depot, t.get("gguf"), device)
    print(f"PROGRESSION 1/{total} lecture du texte (encodeur umT5)", flush=True)
    _ranger()
    _faire_place(11.5, sauf=cle)  # l'encodeur umT5 (11,4 Go) passe en mémoire vive le temps de la lecture
    prompt, negatif = _memoire(_encoder_texte_video)(t, device, dtype)
    print(f"PROGRESSION 2/{total} chargement de Wan 2.2", flush=True)

    def charger():
        gguf = t.get("gguf") or hf_hub_download(*WAN_GGUF)
        _precharger(WAN_GGUF if not t.get("gguf") else (depot, "-"))
        # config= obligatoire : sans lui, diffusers reconnaît « wan-i2v-14B » (dimensions différentes)
        transformeur = WanTransformer3DModel.from_single_file(
            gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), config=depot,
            subfolder="transformer", torch_dtype=dtype)
        _en_memoire_vive(transformeur)  # sinon relu sur le disque quand il passe sur la carte (96 s constatées)
        _precharger((depot, "vae"))
        vae = AutoencoderKLWan.from_pretrained(depot, subfolder="vae", torch_dtype=torch.float32)
        _en_memoire_vive(vae)
        vae.enable_tiling()
        crochet = _par_blocs_wan(transformeur, vae) if device == "cuda" else None
        return (transformeur, vae, UniPCMultistepScheduler.from_pretrained(depot, subfolder="scheduler").config,
                crochet)

    # transformeur et VAE gardés ; pipeline refaite à chaque vidéo (texte ou image), ordonnanceur neuf (il a un état)
    transformeur, vae, config_ordonnanceur, crochet_vae = _garder(cle, charger, 9)
    composants = dict(tokenizer=None, text_encoder=None, vae=vae,
                      scheduler=UniPCMultistepScheduler.from_config(config_ordonnanceur), transformer=transformeur,
                      expand_timesteps=True)
    largeur, hauteur = int(t.get("largeur", 1280)), int(t.get("hauteur", 704))
    options = {}
    if t.get("image"):
        from PIL import Image, ImageOps

        with Image.open(t["image"]) as im:  # recadrée au format de la vidéo (sinon déformée)
            options["image"] = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (largeur, hauteur),
                                            Image.LANCZOS)
        pipe = WanImageToVideoPipeline(**composants, image_encoder=None, image_processor=None)
    else:
        pipe = WanPipeline(**composants)
    if device != "cuda":  # sur la carte : poids amenés bloc par bloc (_par_blocs_wan, fait au chargement)
        pipe = pipe.to(device)
    # Les plongements de la phase 1 sont sur le processeur : WanPipeline ne les change que de type (pas d'appareil)
    # et le transformeur fait `.type_as(encoder_hidden_states)`, qui ramène aussi le calcul du pas de temps sur le
    # processeur → « Expected all tensors to be on the same device, cuda:0 and cpu » (constaté sur la RTX 4070).
    prompt, negatif = prompt.to(pipe._execution_device), negatif.to(pipe._execution_device)

    def suivi(_pipe, i, _t, kwargs):
        print(f"PROGRESSION {i + 3}/{total} étape {i + 1}/{etapes}", flush=True)
        return kwargs

    graine = int(t.get("graine") or 0)
    # l'image de départ encodée, le VAE laisse la carte au transformeur (sans effet s'il est déjà en mémoire vive)
    poignee = transformeur.register_forward_pre_hook(lambda *_: crochet_vae.offload()) if crochet_vae else None
    images = _memoire(pipe)(
        **options, prompt_embeds=prompt, negative_prompt_embeds=negatif, height=hauteur, width=largeur,
        num_frames=int(t.get("images", 121)), num_inference_steps=etapes, guidance_scale=float(t.get("guidage", 5.0)),
        generator=torch.Generator("cpu").manual_seed(graine), callback_on_step_end=suivi,
    ).frames[0]
    if crochet_vae:
        poignee.remove()
        crochet_vae.offload()
    print(f"PROGRESSION {total}/{total} enregistrement de la vidéo", flush=True)
    Path(t["sortie"]).parent.mkdir(parents=True, exist_ok=True)
    fps = int(t.get("fps", 24))
    export_to_video(list(images), t["sortie"], fps=fps, quality=8)
    # Dernière image en PNG : point de départ d'un clip suivant (enchaîner plusieurs vidéos)
    from PIL import Image

    derniere = str(Path(t["sortie"]).with_name("derniere_image.png"))
    Image.fromarray((images[-1] * 255).round().clip(0, 255).astype("uint8")).save(derniere)
    _resultat({"sortie": t["sortie"], "graine": graine, "images": len(images), "duree": round(len(images) / fps, 2),
               "largeur": largeur, "hauteur": hauteur, "derniere_image": derniere})
    print(f"TERMINE {t['sortie']}", flush=True)


def assembler(chemin_tache):
    """Tâche : {clips: [mp4…], sortie, fps}. Met les clips bout à bout en un seul MP4 H.264 (même écriture
    qu'export_to_video de diffusers : imageio + imageio-ffmpeg, libx264, yuv420p). La première image d'un clip
    enchaîné est la dernière du précédent (elle a servi d'image de départ) : elle est retirée, sinon l'image se
    fige un instant à chaque raccord. Processeur seulement. RESULTAT {sortie, images, duree}."""
    import imageio

    t = _lire(chemin_tache)
    clips, fps = t["clips"], int(t.get("fps", 24))
    if not clips:
        _erreur("aucun clip à assembler.")
    Path(t["sortie"]).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with imageio.get_writer(t["sortie"], fps=fps, quality=9, macro_block_size=16) as sortie:
        for i, clip in enumerate(clips, 1):
            print(f"PROGRESSION {i}/{len(clips)} clip {i}", flush=True)
            with imageio.get_reader(clip) as lecteur:
                for k, image in enumerate(lecteur):
                    if i > 1 and k == 0:
                        continue
                    sortie.append_data(image)
                    n += 1
    _resultat({"sortie": t["sortie"], "images": n, "duree": round(n / fps, 2)})
    print(f"TERMINE {t['sortie']}", flush=True)


# --- Photos : détourage (BiRefNet), agrandissement (Real-ESRGAN), visages (GFPGAN) -------------------------
def _dossier_photos():
    return Path(os.environ.get("STUDIOVOIX_PHOTOS") or Path.cwd() / "photos")


def _ouvrir_photo(chemin):
    """Image dans le bon sens (EXIF des téléphones) ; (RGB, alpha ou None)."""
    from PIL import Image, ImageOps

    with Image.open(chemin) as im:
        im = ImageOps.exif_transpose(im)
        alpha = im.getchannel("A") if im.mode in ("RGBA", "LA") or "transparency" in im.info else None
        if alpha is not None and alpha.getextrema() == (255, 255):
            alpha = None  # canal alpha entièrement opaque : inutile
        return im.convert("RGB"), alpha


def detourer(chemin_tache):
    """Tâche : {entree, sortie, modele: general | personne, fond: None | "#rrggbb" | "flou", masque: chemin}.
    BiRefNet donne un masque doux (0–255) ; les couleurs du fond mêlées aux bords (cheveux) sont retirées par
    l'estimation du premier plan de pymatting (rembg.bg.decontaminate_cutout). Sans fond : PNG transparent.
    fp16 sur la carte graphique (jamais bf16 : deform_conv2d de torchvision ne le gère pas). RESULTAT {sortie, ...}."""
    import numpy as np
    import torch
    from PIL import Image, ImageFilter
    from rembg.bg import decontaminate_cutout
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    t = _lire(chemin_tache)
    depot, revision, cote = BIREFNET[t.get("modele") or "general"]
    if t.get("cote"):  # tests : modèle plus petit
        cote = int(t["cote"])
    device = _device()
    _hors_ligne_si_present(depot)
    print("PROGRESSION 1/3 chargement de BiRefNet", flush=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.set_float32_matmul_precision("high")
    depot, revision = t.get("depot", depot), t.get("revision", revision)
    modele = _garder(
        ("BiRefNet", depot, revision, device),
        lambda: AutoModelForImageSegmentation.from_pretrained(depot, trust_remote_code=True, revision=revision)
        .to(device=device, dtype=dtype).eval(),
        1.0, vers_cpu=lambda m: m.to("cpu"), vers_gpu=lambda m: m.to(device))
    image, _ = _ouvrir_photo(t["entree"])
    print(f"PROGRESSION 2/3 détourage ({image.width}×{image.height})", flush=True)
    preparation = transforms.Compose([transforms.Resize((cote, cote)), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = preparation(image).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.inference_mode():
        prediction = _memoire(modele)(x)[-1].sigmoid().float().cpu()[0, 0].numpy()
    masque = Image.fromarray((prediction * 255).round().astype("uint8")).resize(image.size, Image.BILINEAR)
    print("PROGRESSION 3/3 finitions des bords", flush=True)
    decoupe = decontaminate_cutout(image, masque)  # RGBA, couleurs des bords sans le fond
    fond = t.get("fond")
    if fond == "flou":  # effet portrait : le sujet net devant son propre fond flouté
        base = image.filter(ImageFilter.GaussianBlur(max(image.size) / 80)).convert("RGBA")
        resultat = Image.alpha_composite(base, decoupe).convert("RGB")
    elif fond:
        base = Image.new("RGBA", image.size, fond)
        resultat = Image.alpha_composite(base, decoupe).convert("RGB")
    else:
        resultat = decoupe
    Path(t["sortie"]).parent.mkdir(parents=True, exist_ok=True)
    resultat.save(t["sortie"])
    if t.get("masque"):
        masque.save(t["masque"])
    couverture = float(np.asarray(masque, dtype=np.float32).mean() / 255)
    _resultat({"sortie": t["sortie"], "masque": t.get("masque"), "largeur": image.width, "hauteur": image.height,
               "couverture": round(couverture, 3)})
    print(f"TERMINE {t['sortie']}", flush=True)


def _par_tuiles(modele, image, tuile=512, marge=32):
    """Agrandit image (H×W×3, réels 0–1) par tuiles qui se chevauchent, fondues en rampe linéaire : une photo de
    plusieurs mégapixels ne tient pas d'un bloc dans 12 Go. spandrel ne découpe pas lui-même."""
    import numpy as np
    import torch

    h, w, _ = image.shape
    s = modele.scale
    param = next(modele.model.parameters())

    def passe(bloc):
        x = torch.from_numpy(np.ascontiguousarray(bloc)).permute(2, 0, 1)[None].to(param.device, param.dtype)
        with torch.inference_mode():
            return modele(x)[0].permute(1, 2, 0).float().cpu().numpy()

    if h <= tuile and w <= tuile:
        return passe(image)

    def positions(n):
        if n <= tuile:
            return [0]
        pas = tuile - marge
        p = list(range(0, n - tuile, pas))
        return p + [n - tuile]

    sortie = np.zeros((h * s, w * s, 3), np.float32)
    poids = np.zeros((h * s, w * s, 1), np.float32)
    lignes, colonnes = positions(h), positions(w)
    total, n = len(lignes) * len(colonnes), 0
    for y in lignes:
        for x in colonnes:
            bloc = passe(image[y:y + tuile, x:x + tuile])
            bh, bw = bloc.shape[:2]
            rampe = marge * s

            def fenetre(taille, debut, fin):
                f = np.ones(taille, np.float32)
                if debut:
                    f[:rampe] = np.linspace(0, 1, rampe + 2, dtype=np.float32)[1:-1]
                if fin:
                    f[taille - rampe:] = np.linspace(1, 0, rampe + 2, dtype=np.float32)[1:-1]
                return f

            masque = (fenetre(bh, y > 0, y + tuile < h)[:, None] * fenetre(bw, x > 0, x + tuile < w)[None, :])[..., None]
            sortie[y * s:y * s + bh, x * s:x * s + bw] += bloc * masque
            poids[y * s:y * s + bh, x * s:x * s + bw] += masque
            n += 1
            print(f"Tuile {n}/{total}", flush=True)
    return sortie / np.maximum(poids, 1e-6)


# Gabarit des 5 points du visage pour une image 512×512 (FFHQ, celui de GFPGAN)
GABARIT_VISAGE = [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043],
                  [313.08905, 371.15118]]


def _ovale_visage():
    """Masque 512×512 de l'ovale du visage (gabarit FFHQ : yeux vers y = 240, bouche vers y = 371), fondu."""
    import cv2
    import numpy as np

    m = np.zeros((512, 512), np.float32)
    cv2.ellipse(m, (256, 285), (165, 205), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 28)


def _restaurer_visages(bgr, dossier, device, force=0.7, cote_min=48):
    """GFPGAN 1.4 sur chaque visage trouvé par YuNet : alignement sur le gabarit, restauration 512×512, recollage
    fondu. On appelle le réseau directement (entrée et sortie en [-1, 1]) : le descripteur de spandrel travaille
    en [0, 1] et changeait l'identité (essai : yeux marron devenus bleus). Renvoie (image, nombre de visages)."""
    import cv2
    import numpy as np
    import torch
    from spandrel import ModelLoader

    h, w = bgr.shape[:2]
    reduction = min(1.0, 1600 / max(h, w))  # détection sur une copie réduite des grandes images
    petite = cv2.resize(bgr, (round(w * reduction), round(h * reduction)), interpolation=cv2.INTER_AREA)
    detecteur = cv2.FaceDetectorYN.create(str(dossier / "face_detection_yunet_2023mar.onnx"), "",
                                          (petite.shape[1], petite.shape[0]), 0.7, 0.3, 5000)
    _, visages = detecteur.detect(petite)
    if visages is None:
        return bgr, 0
    visages = [v for v in visages if v[2] / reduction >= cote_min]
    if not visages:
        return bgr, 0
    def charger():
        g = ModelLoader(device=device).load_from_file(str(dossier / "GFPGANv1.4.pth")).eval()
        return g.to(torch.bfloat16 if device == "cuda" and g.supports_bfloat16 else torch.float32)

    gfpgan = _garder(("GFPGAN", str(dossier), device), charger, 0.4, vers_cpu=lambda g: g.to(torch.device("cpu")),
                     vers_gpu=lambda g: g.to(torch.device(device)))
    dtype = next(gfpgan.model.parameters()).dtype
    gabarit = np.array(GABARIT_VISAGE, np.float32)
    ovale = _ovale_visage()
    for v in visages:
        points = v[4:14].reshape(5, 2) / reduction
        a, _ = cv2.estimateAffinePartial2D(points, gabarit, method=cv2.LMEDS)
        if a is None:
            continue
        recadre = cv2.warpAffine(bgr, a, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
        x = torch.from_numpy(cv2.cvtColor(recadre, cv2.COLOR_BGR2RGB) / 255.0).permute(2, 0, 1)[None]
        x = x.float().to(device=device, dtype=dtype)
        with torch.inference_mode():
            y = gfpgan.model(x * 2 - 1, randomize_noise=False)[0]
        y = ((y.float().clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
        y = cv2.cvtColor((y * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        restaure = cv2.addWeighted(y, force, recadre, 1 - force, 0)  # force < 1 : garde du grain d'origine
        inverse = cv2.invertAffineTransform(a)
        retour = cv2.warpAffine(restaure, inverse, (w, h))
        # Seul l'ovale du visage est recollé, bords très fondus : le reste du carré 512 (fond, cheveux du haut)
        # change de teinte avec GFPGAN et laissait un rectangle visible (constaté à l'essai).
        masque = cv2.warpAffine(ovale, inverse, (w, h))[..., None]
        bgr = (masque * retour + (1 - masque) * bgr).round().astype(np.uint8)
    return bgr, len(visages)


COTE_MAX = 8192  # côté le plus long d'une photo améliorée (fichier et affichage raisonnables)


def ameliorer(chemin_tache):
    """Tâche : {entree, sortie, echelle: 1 | 2 | 4, rapide: bool, visages: bool, force: 0–1}.
    Real-ESRGAN x4plus (x2plus pour ×2 ; general-x4v3, 5 Mo, en mode rapide) par tuiles, puis réduction à la
    taille voulue (×1 = restauration seule : bruit, artefacts JPEG, flou), puis GFPGAN sur les visages de l'image
    agrandie. La transparence éventuelle est agrandie à part. RESULTAT {sortie, largeur, hauteur, visages}."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from spandrel import ModelLoader

    t = _lire(chemin_tache)
    dossier = _dossier_photos()
    echelle = int(t.get("echelle", 2))
    device = _device()
    image, alpha = _ouvrir_photo(t["entree"])
    # Grande photo (24 Mpx d'un appareil récent : ×2 donnerait 6912×10368, constaté) : agrandissement réduit pour
    # rester sous COTE_MAX de côté au lieu d'un refus ; le message le signale (« echelle_obtenue »).
    facteur = min(float(echelle), COTE_MAX / max(image.width, image.height))
    largeur, hauteur = round(image.width * facteur), round(image.height * facteur)
    nom = ("realesr-general-x4v3.pth" if t.get("rapide")
           else "RealESRGAN_x2plus.pth" if echelle == 2 else "RealESRGAN_x4plus.pth")
    etapes = 3 if t.get("visages") else 2
    print(f"PROGRESSION 1/{etapes} chargement de Real-ESRGAN", flush=True)
    def charger():
        m = ModelLoader(device=device).load_from_file(str(dossier / nom)).eval()
        return m.half() if device == "cuda" and m.supports_half else m

    modele = _garder(("Real-ESRGAN", str(dossier / nom), device), charger, 0.2,
                     vers_cpu=lambda m: m.to(torch.device("cpu")), vers_gpu=lambda m: m.to(torch.device(device)))
    print(f"PROGRESSION 2/{etapes} agrandissement ({image.width}×{image.height} → {largeur}×{hauteur})", flush=True)
    # Tuiles de 1024 px sur la carte (512 → 4 fois plus de passes, carte sous-employée ; ~2 Go en fp16 pour
    # x4plus), 512 sur le processeur
    sortie = _memoire(_par_tuiles)(modele, np.asarray(image, np.float32) / 255.0, 1024 if device == "cuda" else 512)
    sortie = (np.clip(sortie, 0, 1) * 255).round().astype(np.uint8)
    if sortie.shape[1] != largeur or sortie.shape[0] != hauteur:
        sortie = cv2.resize(sortie, (largeur, hauteur), interpolation=cv2.INTER_AREA)
    nb_visages = 0
    if t.get("visages"):
        print(f"PROGRESSION 3/{etapes} restauration des visages", flush=True)
        bgr, nb_visages = _memoire(_restaurer_visages)(cv2.cvtColor(sortie, cv2.COLOR_RGB2BGR), dossier, device,
                                                       float(t.get("force", 0.7)))
        sortie = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resultat = Image.fromarray(sortie)
    if alpha is not None:
        resultat.putalpha(alpha.resize(resultat.size, Image.LANCZOS))
    Path(t["sortie"]).parent.mkdir(parents=True, exist_ok=True)
    resultat.save(t["sortie"])
    _resultat({"sortie": t["sortie"], "largeur": largeur, "hauteur": hauteur, "visages": nb_visages,
               "modele": nom, "echelle_obtenue": round(facteur, 2)})
    print(f"TERMINE {t['sortie']}", flush=True)


# --- Hunyuan3D-2 : image → forme → texture -------------------------------------------------------------
def forme3d(chemin_tache):
    """Tâche : {image, dossier, etapes, octree, faces, graine, texture: bool, formats: ["glb", "obj"]}.
    Écrit dossier/forme.glb (blanc) et, si texture, dossier/modele.glb (texturé) [+ modele_web.glb si « web »] ; avec « obj », le modèle final
    est aussi écrit en OBJ (+ material.mtl et texture PNG à côté, écrits par trimesh) ; RESULTAT {…}."""
    import torch
    from PIL import Image
    from rembg import new_session, remove
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.pipelines import export_to_trimesh

    t = _lire(chemin_tache)
    device = _device()
    dossier = Path(t["dossier"])
    dossier.mkdir(parents=True, exist_ok=True)
    texture = bool(t.get("texture", True)) and device == "cuda"  # le rasteriseur de la texture est CUDA
    total = 4 + (2 if texture else 0)
    graine = int(t.get("graine") or 0) or int(torch.randint(1, 2**31 - 1, (1,)))

    print(f"PROGRESSION 1/{total} détourage de l'image", flush=True)
    im = Image.open(t["image"]).convert("RGB")
    # Comme hy3dgen.rembg.BackgroundRemover, mais avec le modèle u2net (Apache 2.0) : le modèle par défaut
    # de rembg est désormais BRIA RMBG-2.0, à usage non commercial.
    im = remove(im, session=_garder(("u2net",), lambda: new_session("u2net"), 0.2), bgcolor=[255, 255, 255, 0])
    im.save(dossier / "image_detouree.png")

    print(f"PROGRESSION 2/{total} chargement du générateur de forme", flush=True)
    _hors_ligne_si_present(HUNYUAN)
    depot, sous_dossier = t.get("depot", HUNYUAN), t.get("sous_dossier", HUNYUAN_FORME)

    def charger():
        _precharger((depot, sous_dossier))
        p_ = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            depot, subfolder=sous_dossier, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
        if device == "cuda":
            p_.enable_flashvdm(mc_algo="mc")  # plus rapide (README : --enable_flashvdm), extraction sans diso
        return p_

    # Hunyuan3DDiTFlowMatchingPipeline.to(device) déplace VAE, DiT et encodeur d'image (shapegen/pipelines.py)
    pipe = _garder(("Hunyuan3D forme", depot, sous_dossier, device), charger, 4,
                   vers_cpu=lambda p_: p_.to("cpu"), vers_gpu=lambda p_: p_.to(device))
    print(f"PROGRESSION 3/{total} génération de la forme", flush=True)
    sorties = _memoire(pipe)(
        image=im, num_inference_steps=int(t.get("etapes", 30)), guidance_scale=float(t.get("guidage", 7.5)),
        generator=torch.Generator().manual_seed(graine), octree_resolution=int(t.get("octree", 256)),
        num_chunks=int(t.get("morceaux", 20000)), output_type="mesh",
    )
    mesh = export_to_trimesh(sorties)[0]

    print(f"PROGRESSION 4/{total} nettoyage et simplification", flush=True)
    mesh = _nettoyer(mesh, int(t.get("faces", 40000)))
    forme = dossier / "forme.glb"
    mesh.export(str(forme))
    formats = [f.lower() for f in t.get("formats") or ["glb"]]
    resultat = {"forme": str(forme), "faces": int(len(mesh.faces)), "graine": graine, "texture": None, "obj": None}
    if not texture:
        if "obj" in formats:
            resultat["obj"] = _exporter_obj(mesh, dossier / "forme.obj")
        _resultat(resultat)
        print(f"TERMINE {forme}", flush=True)
        return

    print(f"PROGRESSION 5/{total} chargement du peintre de texture", flush=True)
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    _permettre_code_peintre()

    cle_peintre = ("Hunyuan3D texture", HUNYUAN_TEXTURE)

    def pipelines(p_):  # délumination et vues multiples : deux pipelines diffusers (texgen/utils/*_utils.py)
        return [m.pipeline for m in p_.models.values()]

    # le générateur de forme passe en mémoire vive (_garder) avant que le peintre arrive sur la carte
    def charger_peintre():
        _precharger((HUNYUAN, HUNYUAN_TEXTURE), (HUNYUAN, "hunyuan3d-delight-v2-0"))
        return Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN, subfolder=HUNYUAN_TEXTURE)

    peintre = _garder(cle_peintre, charger_peintre, 6, vers_cpu=lambda p_: [x.to("cpu") for x in pipelines(p_)],
                      vers_gpu=lambda p_: [x.to("cuda") for x in pipelines(p_)])
    print(f"PROGRESSION 6/{total} peinture de la texture", flush=True)
    # Tout sur la carte d'abord : ses deux pipelines y sont déjà, en fp16, dès le chargement (~6 Go de poids, ~8 à
    # 10 Go en pointe estimés, ACE-Step étant arrêté) ; le mode basse mémoire de gradio_app.py (--low_vram_mode,
    # enable_model_cpu_offload) les fait aller et venir et laisse la carte attendre. En cas de manque de mémoire,
    # on y revient.
    try:
        texture_mesh = peintre(mesh.copy(), im)
    except torch.cuda.OutOfMemoryError:
        print("Mémoire graphique juste : peinture en mode basse mémoire.", flush=True)
        _oublier(cle_peintre)  # déchargement activé : on ne le déplace plus à la main (et il repartira neuf)
        peintre.enable_model_cpu_offload()
        texture_mesh = _memoire(peintre)(mesh, im)
    modele = dossier / "modele.glb"
    texture_mesh.export(str(modele), include_normals=True)
    resultat["texture"] = str(modele)
    if "obj" in formats:
        resultat["obj"] = _exporter_obj(texture_mesh, dossier / "modele.obj")
    if t.get("web"):
        resultat["web"] = _alleger_glb(modele, dossier / "modele_web.glb", int(t.get("texture_web", 1024)))["sortie"]
    _resultat(resultat)
    print(f"TERMINE {modele}", flush=True)


def _permettre_code_peintre():
    """Le peintre de Hunyuan3D-2 charge son pipeline de vues multiples avec un custom_pipeline local (dossier
    hy3dgen/texgen/hunyuanpaint du code épinglé) dont le UNet est lui aussi du code du modèle (unet/modules.py,
    UNet2p5DConditionModel, déclaré dans model_index.json). diffusers 0.39 refuse ce code sans trust_remote_code
    (dynamic_modules_utils.get_cached_module_file : « contains custom code in pipeline.py… », constaté sur la
    RTX 4070), et texgen/utils/multiview_utils.py ne le passe pas. On l'ajoute à ce seul appel."""
    import hy3dgen.texgen.utils.multiview_utils as vues
    from diffusers import DiffusionPipeline

    if getattr(vues.DiffusionPipeline, "_studiovoix", False):
        return

    class _AvecCodeDuModele:
        _studiovoix = True

        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs.setdefault("trust_remote_code", True)
            return DiffusionPipeline.from_pretrained(*args, **kwargs)

    vues.DiffusionPipeline = _AvecCodeDuModele


def _nettoyer(mesh, faces):
    """Post-traitement d'Hunyuan3D (pymeshlab) : retrait des morceaux flottants (< 0,5 % des faces), des faces
    dégénérées, puis simplification à `faces` faces. Si pymeshlab ne peut pas charger ses greffons (bibliothèque
    système absente, vu sous Linux sans libOpenGL), nettoyage de secours avec trimesh, sans simplification."""
    import trimesh
    from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

    try:
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        return FaceReducer()(mesh, max_facenum=faces)
    except Exception as e:  # noqa: BLE001 - pymeshlab lève une exception de son cru
        print(f"AVERTISSEMENT : nettoyage pymeshlab impossible ({e}) ; nettoyage simplifié avec trimesh.", flush=True)
    morceaux = mesh.split(only_watertight=False)
    if len(morceaux) > 1:
        seuil = 0.005 * len(mesh.faces)
        mesh = trimesh.util.concatenate([m for m in morceaux if len(m.faces) >= seuil])
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _alleger_glb(entree, sortie, texture_max=1024):
    """GLB plus léger pour un jeu web : textures réduites à texture_max pixels de côté et stockées en JPEG
    (au lieu de PNG 2048×2048), géométrie inchangée (la simplifier casserait les coordonnées de texture)."""
    import io

    import trimesh
    from PIL import Image

    scene = trimesh.load(str(entree), force="scene")
    textures = 0
    for geo in scene.geometry.values():
        mat = getattr(getattr(geo, "visual", None), "material", None)
        for attr in ("baseColorTexture", "image"):
            im = getattr(mat, attr, None) if mat is not None else None
            if im is None or not hasattr(im, "size"):
                continue
            im = im.convert("RGB")
            if max(im.size) > texture_max:
                im.thumbnail((texture_max, texture_max), Image.LANCZOS)
            tampon = io.BytesIO()
            im.save(tampon, format="JPEG", quality=85)
            tampon.seek(0)
            setattr(mat, attr, Image.open(tampon))  # format JPEG : trimesh l'écrit en JPEG dans le GLB
            textures += 1
    scene.export(str(sortie))
    return {"sortie": str(sortie), "textures": textures, "avant": Path(entree).stat().st_size,
            "apres": Path(sortie).stat().st_size}


def alleger(chemin_tache):
    """Tâche : {entree, sortie, texture_max}. RESULTAT {sortie, textures, avant, apres} (octets)."""
    t = _lire(chemin_tache)
    print("PROGRESSION 1/1 allègement du modèle", flush=True)
    res = _alleger_glb(t["entree"], t["sortie"], int(t.get("texture_max", 1024)))
    _resultat(res)
    print(f"TERMINE {res['sortie']}", flush=True)


def _exporter_obj(mesh, chemin):
    """OBJ à côté du GLB : trimesh écrit material.mtl et l'image de texture dans le même dossier
    (export_mesh crée un FilePathResolver quand on lui donne un chemin)."""
    mesh.export(str(chemin))
    return str(chemin)


# --- Téléchargement des modèles --------------------------------------------------------------------------
def _sha256(chemin):
    import hashlib

    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


REPRISES = 6  # essais par dépôt ou fichier en cas de coupure réseau


def _erreur_reseau(e):
    """Vrai pour une coupure ou un délai réseau, ou une erreur 5xx du serveur ; jamais pour un refus d'accès (401,
    dépôt soumis à licence) ni une erreur de nos fichiers."""
    import http.client
    import urllib.error

    import requests
    import urllib3

    while e is not None:
        reponse = getattr(e, "response", None)
        statut = getattr(reponse, "status_code", None) or getattr(e, "code", None)
        if isinstance(statut, int):
            return statut >= 500
        if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError,
                          requests.exceptions.Timeout, urllib3.exceptions.HTTPError, http.client.HTTPException,
                          urllib.error.URLError, ConnectionError, TimeoutError)):
            return True
        e = e.__cause__ or e.__context__
    return False


def _avec_reprises(action, quoi, essais=REPRISES, pause=None):
    """Relance action() après une coupure réseau : huggingface_hub reprend un fichier partiel (.incomplete) là où
    il s'était arrêté (constaté chez l'utilisateur : coupure au milieu du VAE de Wan, 2,8 Go)."""
    import time

    pause = pause or time.sleep
    for n in range(1, essais + 1):
        try:
            return action()
        except Exception as e:  # noqa: BLE001 - seules les erreurs réseau sont rattrapées, les autres repartent
            if n == essais or not _erreur_reseau(e):
                raise
            attente = min(60, 5 * 2 ** (n - 1))
            print(f"Connexion interrompue pendant le téléchargement de {quoi} ({type(e).__name__}) : reprise dans "
                  f"{attente} s, là où il s'était arrêté (essai {n + 1}/{essais}).", flush=True)
            pause(attente)


def _telecharger_fichiers(noms):
    """Poids des photos (GitHub) dans _dossier_photos(), via un fichier .part : un téléchargement interrompu ne
    laisse jamais un fichier tronqué sous le vrai nom ; empreinte SHA-256 vérifiée."""
    import shutil
    import urllib.request

    dossier = _dossier_photos()
    dossier.mkdir(parents=True, exist_ok=True)
    for nom in noms:
        url, empreinte = FICHIERS_PHOTOS[nom]
        cible = dossier / nom
        if cible.exists() and _sha256(cible) == empreinte:
            continue
        print(f"Téléchargement de {nom}…", flush=True)
        partiel = cible.with_name(nom + ".part")

        def recuperer():
            with urllib.request.urlopen(url, timeout=60) as r, open(partiel, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)

        _avec_reprises(recuperer, nom)
        if _sha256(partiel) != empreinte:
            partiel.unlink()
            _erreur(f"{nom} : fichier téléchargé corrompu (empreinte SHA-256 différente). Relance le téléchargement.")
        partiel.replace(cible)


def _tester_liens(depot):
    """Contourne une course de huggingface_hub (0.36) sous Windows sans mode développeur : are_symlinks_supported()
    note « pris en charge » AVANT de faire son essai ; les fils de snapshot_download (8 en parallèle) qui arrivent
    pendant l'essai tentent alors un vrai lien symbolique et échouent (WinError 1314, « le client ne dispose pas
    d'un privilège nécessaire »), au lieu de copier le fichier. On fait donc l'essai ici, avant, dans le fil principal,
    sur le dossier que la bibliothèque testera (le dossier du dépôt dans le cache, commun aux blobs et aux
    snapshots) : le résultat, mis en mémoire, est alors juste pour tous les fils."""
    from huggingface_hub import constants
    from huggingface_hub.file_download import are_symlinks_supported, repo_folder_name

    dossier = os.path.join(constants.HF_HUB_CACHE, repo_folder_name(repo_id=depot, repo_type="model"))
    return are_symlinks_supported(dossier)


def telecharger(noms):
    from huggingface_hub import snapshot_download

    noms = noms or list(MODELES)
    if "detourage" in noms:  # modèle u2net de rembg (dans U2NET_HOME)
        print("Téléchargement du modèle de détourage (rembg u2net)…", flush=True)
        from rembg import new_session

        new_session("u2net")
        noms = [n for n in noms if n != "detourage"]
    depots = [(nom, repo, motifs) for nom in noms for repo, motifs in MODELES[nom]]
    for n, (nom, repo, motifs) in enumerate(depots, 1):
        print(f"PROGRESSION {n}/{len(depots)} téléchargement : {repo}", flush=True)
        if repo.startswith("local:"):  # fichiers publiés hors de Hugging Face (GitHub), vérifiés par SHA-256
            _telecharger_fichiers(motifs)
            continue
        _tester_liens(repo)
        try:
            _avec_reprises(lambda: snapshot_download(repo, allow_patterns=motifs, revision=REVISIONS.get(repo)), repo)
        except Exception as e:
            if nom == "bruitages" and any(k in str(e) for k in ("401", "403", "gated", "Access")):
                _erreur("accès refusé à Stable Audio Open : accepte la licence sur "
                        "https://huggingface.co/stabilityai/stable-audio-open-1.0 puis enregistre un jeton "
                        "Hugging Face (voir README).", 4)
            raise
    print("Modèles prêts.", flush=True)


ACTIONS = {"decrire": decrire, "bruitage": bruitage, "image": image, "personnage": personnage, "video": video,
           "assembler": assembler, "detourer": detourer, "ameliorer": ameliorer, "forme3d": forme3d, "alleger": alleger}


def principal(argv):
    if len(argv) >= 1 and argv[0] == "telecharger":
        telecharger(argv[1:])
    elif len(argv) == 2 and argv[0] in ACTIONS:
        try:
            ACTIONS[argv[0]](argv[1])
        except (OSError, ValueError) as e:
            # Dépôt présent mais incomplet (téléchargement interrompu : constaté ici avec Qwen3-VL-4B sans son
            # preprocessor_config.json) : hors ligne, le fichier manquant ne peut pas être récupéré. Une seconde
            # fois en ligne, huggingface_hub complète le cache.
            from huggingface_hub import constants

            if not constants.HF_HUB_OFFLINE or _HORS_LIGNE_IMPOSE:
                raise
            print(f"Fichier de modèle manquant hors ligne ({type(e).__name__}) : nouvel essai en ligne.", flush=True)
            global _EN_LIGNE_FORCE
            _EN_LIGNE_FORCE = True
            try:
                ACTIONS[argv[0]](argv[1])
            finally:
                _EN_LIGNE_FORCE = False
    else:
        _erreur("usage : diffusion.py decrire|bruitage|image|personnage|video|assembler|detourer|ameliorer|forme3d|alleger "
                "<tache.json> | telecharger [modèle…] | --resident <secondes>")


if __name__ == "__main__":
    import resident

    delai = resident.demande(sys.argv[1:])
    if delai is None:
        principal(sys.argv[1:])
    else:  # l'application garde le moteur ouvert : les modèles chargés (_CHARGES) servent aux tâches suivantes
        resident.servir(principal, delai)
