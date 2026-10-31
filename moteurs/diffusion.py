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
# Facultatif : le même modèle sans refus (« abliteration » de la partie texte, Apache 2.0), choisi dans l'onglet
# Modèles ; l'application le passe dans la clé « depot » de la tâche
QWEN_LIBRE = "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"
QWEN_COTE_MAX = 1024  # côté maximal des images données à Qwen (jetons visuels, mémoire graphique)
STABLE_AUDIO = "stabilityai/stable-audio-open-1.0"
# Sprites de dresseurs (kit de dresseur) : SDXL (openrail++) + LoRA entraîné sur des sprites de dresseurs de jeux de rôle
# portables (sWizad, licence Civitai : images utilisables librement ; rang 64, 768 px, 2 940 images, étiquettes
# « gen1 »…« gen7 » par génération : gen3 = 164 images, gen5 = 6). Aucun modèle installé ne connaissait ce style
# (klein / Z-Image : illustration pixelisée, constaté chez l'utilisateur le 29/10).
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_DRESSEUR = ("sWizad/pokemon-trainer-sprite-pixelart", "pk_trainer_xl_v1.safetensors")
NEGATIF_SPRITE_SDXL = ("photo, realistic, 3d render, blurry, smooth shading, gradient, anti-aliasing, text, watermark, "
                       "signature, frame, multiple characters, cropped, deformed hands")
WHISPER = "openai/whisper-large-v3-turbo"  # MIT, 1,6 Go : transcription et sous-titres (99 langues)
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
    # RIFE 4.26 (MIT, hzwer) : vidéos plus fluides et ralentis ; même empreinte que l'archive officielle de hzwer
    "flownet_v4.26.pkl": ("https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.26.pkl",
                          "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f"),
    "face_detection_yunet_2023mar.onnx": (
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
}
# StableAvatar (MIT) : vidéo parlante ou chantée à partir d'une image et d'un son (Wan 2.1 1.3B + audio, durée
# illimitée par fenêtres qui se chevauchent). Son code (paquet « wan ») vient du dépôt GitHub au commit épinglé ;
# ses poids du dépôt Hugging Face. Le fichier StableAvatar-1.3B contient tout le transformeur (1143 tenseurs, vérifié) :
# les poids de base de Wan2.1-Fun (3,1 Go) ne sont pas téléchargés. L'encodeur de texte umT5-xxl est celui de
# Wan 2.2 (déjà là pour les vidéos) : mêmes poids que le .pth de StableAvatar (comparés tenseur par tenseur) et même
# calcul (UMT5EncoderModel de transformers = T5 de Wan, écart nul mesuré) : 11,4 Go de téléchargement en moins.
STABLEAVATAR = "FrancisRing/StableAvatar"
STABLEAVATAR_CODE = ("Francis-Rings/StableAvatar", "38625187dc10913ba4facc32341a841ce0a8f29b")
SA_BASE = "Wan2.1-Fun-V1.1-1.3B-InP"
SA_FICHIERS = ["StableAvatar-1.3B/transformer3d-square.pt", f"{SA_BASE}/config.json", f"{SA_BASE}/Wan2.1_VAE.pth",
               f"{SA_BASE}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"] + [
    f"wav2vec2-base-960h/{f}" for f in ("config.json", "feature_extractor_config.json", "preprocessor_config.json",
                                        "model.safetensors", "special_tokens_map.json", "tokenizer_config.json",
                                        "vocab.json")]
# Prompt négatif d'inference.py de StableAvatar (celui de Wan, en chinois : le modèle a été entraîné avec)
NEGATIF_PARLANTE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
                    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，"
                    "倒着走")
IMAGES_MIN_PARLANTE = 85  # au moins deux fenêtres de 81 images (voir parler)
FORMATS_PARLANTE = [(512, 512), (480, 832), (832, 480)]  # les trois formats d'entraînement (largeur, hauteur)
HUNYUAN = "tencent/Hunyuan3D-2"
HUNYUAN_FORME = "hunyuan3d-dit-v2-0-turbo"
HUNYUAN_TEXTURE = "hunyuan3d-paint-v2-0-turbo"
# Hunyuan3D-2mv : la forme à partir de plusieurs vues (face, gauche, dos, droite : MVImageProcessorV2 de
# hy3dgen/shapegen/preprocessors.py) ; même licence que Hunyuan3D-2 ; flashvdm reprend le VAE turbo de Hunyuan3D-2
HUNYUAN_MV = "tencent/Hunyuan3D-2mv"
HUNYUAN_MV_FORME = "hunyuan3d-dit-v2-mv-turbo"
# Seuls les .safetensors fp16 sont chargés (smart_load_model : use_safetensors=True, variant fp16) : les .ckpt du
# dépôt, mêmes poids, ne sont pas téléchargés (5,3 Go de moins pour la forme turbo et son VAE)
def _poids(*sous_dossiers):
    return [f"{d}/{f}" for d in sous_dossiers for f in ("config.yaml", "model.fp16.safetensors")]

# nom → [(dépôt, motifs à télécharger ou None pour tout)]
MODELES = {
    "qwen": [(QWEN, None)],
    "qwen_libre": [(QWEN_LIBRE, None)],
    "sprites_dresseur": [(SDXL, ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "*/config.json",
                                 "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors",
                                 "unet/diffusion_pytorch_model.fp16.safetensors",
                                 "vae/diffusion_pytorch_model.fp16.safetensors"]),
                         (LORA_DRESSEUR[0], [LORA_DRESSEUR[1]])],
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
    "forme3d": [(HUNYUAN, _poids(HUNYUAN_FORME, "hunyuan3d-vae-v2-0-turbo", "hunyuan3d-dit-v2-0"))],
    # face + dos (révision d'un modèle 3D : détails cachés dans le dos)
    "forme3d_vues": [(HUNYUAN_MV, _poids(HUNYUAN_MV_FORME))],
    "texture3d": [(HUNYUAN, [f"{HUNYUAN_TEXTURE}/*", "hunyuan3d-delight-v2-0/*"])],
    "transcription": [(WHISPER, None)],
    "fluidite": [("local:photos", ["flownet_v4.26.pkl"])],
    # vidéo parlante : poids de StableAvatar, encodeur de texte et tokeniseur de Wan 2.2, code au commit épinglé
    "parlante": [(STABLEAVATAR, SA_FICHIERS), (WAN, ["text_encoder/*", "tokenizer/*"]), ("code:stableavatar", None)],
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
                    "dans Outils → Modèles et diagnostic, puis relance.", 3)
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
    # images en série : style d'images d'exemple (aquarelle…) décrit en mots, ajouté à chaque prompt de Z-Image ;
    # données seules en référence à FLUX.2 klein, elles ne donnaient pas leur rendu (constaté chez l'utilisateur)
    "style": ("The attached image shows the art style to reproduce (several example images may be placed side by "
              "side). Describe ONLY this art style, in English, as style keywords for an image generator: medium "
              "and technique (for example watercolor painting, pencil drawing, flat vector, 3D render, pixel art), "
              "line work, shading, color palette, textures, background treatment and level of detail. Never mention "
              "the subjects, objects, characters or scene. Answer with one line of comma-separated keywords, at most "
              "40 words, nothing else."),
    "video": ("Translate and rewrite the following description into an English prompt for a text-to-video "
              "generator. Keep EVERY element of the description: each character and how many there are, what each "
              "one does, the creatures, the setting and the camera movement, in the same order. Do not invent "
              "characters or change them. Add only visual details (lighting, colors, mood). Present tense, 2 to 4 "
              "sentences, no artist or existing work. Answer with the prompt only:\n\n"),
    "carte": ("Rewrite the following description into an English prompt for an image generator that will paint "
              "ONE illustration for a fantasy trading card: main subject, pose or action, setting, lighting, colors "
              "and mood, composition centered on the subject. Keep exactly the characters of the description: the "
              "same number of people, their gender and age; never add a character, an animal or a creature that is "
              "not described. A single image, not a comic page: no panels. Do not mention any art style, artist or "
              "existing work. No text, no letters, no card frame, no border, no user interface. Answer with the "
              "prompt only:\n\n"),
    # onglet « Images » : image libre (texte seul, ou nouvelle scène d'après une photo)
    "scene": ("Rewrite the following description into a detailed English prompt for an image generator: main "
              "subject, action, setting, lighting, colors, mood and composition. If the description asks for a "
              "style (photo, painting, drawing, 3D render...), keep it; otherwise describe a realistic, high quality "
              "image. No artist names, no existing works. 2 to 4 sentences. Answer with the prompt only:\n\n"),
    # onglet « Images », histoire en plusieurs images : le texte reçu commence par « Number of images: N »
    "histoire": ("Split the following story into exactly the number of successive images given (a storyboard). For "
                 "each image, write one English prompt for an image generator: the moment of the story, every "
                 "character present with their FULL appearance (repeat the same description of each character, "
                 "word for word, in every prompt where they appear, so they look identical), the setting, lighting, "
                 "colors and mood, in 40 to 70 words. Spread the images over the WHOLE story, from its beginning "
                 "to its end. Keep the same characters, places and visual style from one image to the next. "
                 "No artist names, no text in the images. Answer with exactly that many lines, numbered 1., 2., "
                 "3. and so on, one prompt per line, nothing else:\n\n"),
    # onglet « Musiques en série » : une description de musique par ligne (souvent en français) → description
    # anglaise pour ACE-Step, qui comprend mieux l'anglais ; le texte reçu commence par « Number of lines: N »
    "musiques": ("Translate each numbered line below into a short English description for a music generator: what "
                 "the music sounds like and the feeling it gives (keep every instrument, tempo, mood, place and "
                 "situation mentioned). No song titles, no artist or existing work, no lyrics. Answer with exactly "
                 "the same number of lines, numbered 1., 2., 3. and so on, one description per line, nothing "
                 "else:\n\n"),
    # onglet « Kit d'objets 3D » : un objet par ligne (souvent en français) → description anglaise d'un objet seul
    "objets": ("Translate each numbered line below into a short English description of ONE object for a 3D game "
               "asset generator: what the object is, its shape, parts, materials and colors (keep every detail "
               "given: colors, number of floors, signs, wheels…). No brand, no game or existing work, no background. "
               "Answer with exactly the same number of lines, numbered 1., 2., 3. and so on, one description per "
               "line, nothing else:\n\n"),
    # onglet « Vidéo parlante » (StableAvatar) : forme conseillée par ses auteurs, « première image - ce que fait la
    # personne - décor »
    "parlante": ("The attached image is the first frame of a video in which this person talks or sings, with lips "
                 "synchronized to a voice. Write an English prompt for the video generator, in 2 or 3 sentences: first "
                 "describe the person as seen in the image (age, hair, clothes), then what they do (talking or singing "
                 "to the camera, natural expressions and head movements, mouth moving), then the background. No artist "
                 "or existing work. Answer with the prompt only."),
    # onglet « Pixel art » : une personne de la photo (souvent un groupe) devient un personnage de jeu ; la demande dit
    # qui (« l'homme barbu à lunettes ») et ce qui change (« en mage avec une cape violette »)
    "sprite": ("The attached photo shows one or several people. Find the person designated in the request below. "
               "Write an English description of that person as a video game character, for an image generator: "
               "gender, approximate age, face (beard, glasses, expression), hair (color, length, style), skin tone and "
               "build, exactly as in the photo; then their outfit, accessories and pose, applying every change asked "
               "in the request (keep their real clothes only where nothing else is asked). Two or three sentences, no "
               "name of any game, artist or existing work, nothing about the background. Answer with the description "
               "only:\n\n"),
    # kit de dresseur : les sprites d'exemple sont recolorés aux couleurs de la personne (palette swap) ; ces couleurs
    # sont données à FLUX.2 klein, qui montre où va chaque couleur
    "couleurs": ("The attached photo shows one or several people. Find the person designated in the request below, "
                 "apply every change it asks, and list the colors of their appearance for recoloring a pixel art game "
                 "sprite, in English, on one line exactly in this form: 'hair: ...; skin: ...; top: ...; bottom: ...; "
                 "shoes: ...; accessories: ...', with precise color names (for example 'dark chestnut brown', 'pale "
                 "peach', 'light sky blue'). For a dress, give its color for both top and bottom. No name of any game "
                 "or existing work. Answer with that line only:\n\n"),
    # entraînement d'un modèle de feuilles de sprites : légende de chaque feuille de marche
    "feuille": ("The attached image is a pixel art walking sprite sheet of one video game character (the same "
                "character seen from several sides, in a grid). Describe that character in one short English phrase "
                "for a training caption: gender or age (man, woman, boy, girl, old man…), hair color and style, hat or "
                "headwear, outfit and its colors, any held object. No name of any game or character. Answer with the "
                "phrase only, for example: a young man with brown hair, red cap, blue jacket and backpack."),
    # kit de dresseur avec un modèle de feuilles entraîné : description courte de la personne, dans la forme des
    # légendes d'entraînement (« a man with brown hair, red cap, blue jacket »)
    "legende": ("The attached photo shows one or several people. Find the person designated in the request below and "
                "apply every change it asks. Describe them in one short English phrase for a pixel art sprite "
                "caption: gender or age (man, woman, boy, girl, old man…), hair color and style, hat or headwear, "
                "outfit and its colors, any held object. No name of any game or existing work. Answer with the phrase "
                "only:\n\n"),
    # kit de dresseur sans photo : le personnage décrit en français (« une dresseuse rousse, veste rouge ») ; la
    # description sert au modèle de sprites (SDXL comprend mal le français) et reste modifiable dans l'onglet
    "sprite_texte": ("Rewrite the following character request (often in French) into an English description of a "
                     "video game character, for an image generator: gender, approximate age, face (beard, glasses, "
                     "expression), hair (color, length, style), skin tone and build, then the outfit with its colors, "
                     "accessories and any held object. Keep every detail of the request; only where it says nothing, "
                     "choose simple plausible colors. Two or three sentences, no name of any game, artist or existing "
                     "work (describe the look instead), nothing about the background. Answer with the description "
                     "only:\n\n"),
    "couleurs_texte": ("From the following character description (possibly in French), list the colors of their "
                       "appearance for recoloring a pixel art game sprite, in English, on one line exactly in this "
                       "form: 'hair: ...; skin: ...; top: ...; bottom: ...; shoes: ...; accessories: ...', with "
                       "precise color names (for example 'dark chestnut brown', 'pale peach', 'light sky blue'); "
                       "choose simple plausible colors for what is not said. For a dress, give its color for both top "
                       "and bottom. Answer with that line only:\n\n"),
    "legende_texte": ("Rewrite the following character description (possibly in French) as one short English phrase "
                      "for a pixel art sprite caption: gender or age (man, woman, boy, girl, old man…), hair color "
                      "and style, hat or headwear, outfit and its colors, any held object. No name of any game or "
                      "existing work. Answer with the phrase only, for example: a young man with brown hair, red cap, "
                      "blue jacket and backpack. Description:\n\n"),
    # onglet « Tileset en objets 3D » : ce que représente un morceau de tileset vu de dessus (une tuile « trou avec
    # échelle » redessinée telle quelle par FLUX.2 klein devenait un pavé plat chez l'utilisateur)
    "tuile3d": ("The attached image is one piece of a pixel art tileset from a top-down 2D video game: an object, a "
                "piece of furniture, a decoration, a wall part or a floor feature, drawn seen from above at an "
                "angle. Identify the real physical object it represents and describe it for a 3D modeler as a "
                "standalone object: its name first, then its shape, parts, proportions, materials and exact colors "
                "as in the image. If it is a floor feature (a hole with a ladder, a trapdoor, a carpet), describe "
                "only the physical object (for example: a wooden ladder with five rungs). One or two sentences in "
                "English, no mention of pixels, tiles, games or background. Answer with the description only."),
    # onglet « Changer les vêtements » : la tenue d'une image (souvent un personnage d'animé), décrite seule pour en
    # habiller une autre personne (FLUX.2 klein reçoit la photo et cette image)
    "costume": ("The attached image shows a costume or an outfit, often worn by an anime, cartoon or game character. "
                "Describe ONLY the outfit, for an image editing model that will dress another person in it: every "
                "garment with its cut, colors, patterns and materials, the accessories, the shoes, and any headwear, "
                "hood, mask or wig. Nothing about the face, the body, the pose, the background or the drawing style; "
                "no name of any character, anime, game or existing work. Two or three sentences in English. Answer "
                "with the description only."),
    # même onglet sans image du costume : la tenue décrite en français
    "tenue": ("Rewrite the following outfit request into an English description of the outfit only, for an image "
              "editing model that will dress a person in it: every garment with its cut, colors, patterns and "
              "materials, the accessories, the shoes, and any headwear or wig. No name of any character, anime, game "
              "or existing work (describe what they wear instead). Two or three sentences. Answer with the "
              "description only:\n\n"),
    # onglet « Images », « Modifier la photo » : consigne de retouche pour FLUX.2 klein (image de référence)
    "retouche": ("The attached image must be edited. Rewrite the following edit request into ONE short, precise "
                 "English instruction for an image editing model: say exactly what to change, and keep the people, "
                 "faces, pose, framing and everything else unchanged. Answer with the instruction only:\n\n"),
}


# Reformulation d'un texte accompagné de l'image de départ (vidéo) : la scène doit rester celle de l'image.
IMAGE_DE_DEPART = ("The attached image is the first frame of the video. Keep its characters, their number, clothes, "
                   "setting, colors and lighting exactly as they appear in the image. ")

# Image libre d'après une photo (sujet ou composition repris) : Qwen décrit précisément le sujet de la photo.
IMAGE_REFERENCE = ("The attached image is a reference photo for the picture to create: describe its main subject "
                   "precisely (appearance, clothes, colors) inside the new scene. ")
# onglet « Images » avec plusieurs photos : une planche, photos numérotées ; FLUX.2 klein reçoit les références
# dans le même ordre et comprend « image 1 », « image 2 »…
PHOTOS_NUMEROTEES = ("The attached picture shows {n} reference photos side by side, numbered 1 to {n} from left to "
                     "right. In your answer, refer to them as image 1, image 2... (for example: the woman from image "
                     "1, the castle from image 2) and say precisely what to take from each one. ")
IMAGE_PERSONNAGES = ("The attached image shows the main character(s), creatures or places of the story (several "
                     "reference pictures may be placed side by side): describe their appearance precisely, from "
                     "this image, in every prompt where they appear. ")
# Ajouté après l'histoire : un long texte français faisait oublier la consigne du début (prompts écrits en
# français, constaté avec une histoire de 27 000 caractères sur la RTX 4070).
RAPPEL_HISTOIRE = ("\n\nEnd of the story. Now write exactly {n} numbered prompts, one per line, IN ENGLISH (translate "
                   "everything except the characters' names), nothing else.")
RAPPEL_MUSIQUES = "\n\nEnd of the lines. Now write exactly {n} numbered lines, IN ENGLISH, nothing else."
# Modes qui répondent par une liste numérotée (une ligne par élément) : jetons par élément, rappel ajouté au texte
LISTES_NUMEROTEES = {"histoire": (180, RAPPEL_HISTOIRE), "musiques": (50, RAPPEL_MUSIQUES),
                     "objets": (50, RAPPEL_MUSIQUES)}

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


def _sans_gqa_sdpa():
    """Long texte pour Qwen3-VL (histoire de 27 000 caractères = 8 260 jetons) : transformers 4.57 passe
    `enable_gqa=True` à scaled_dot_product_attention quand il n'y a pas de masque, or les noyaux économes de
    PyTorch 2.6 n'acceptent pas des têtes de clés moins nombreuses que celles des requêtes : repli sur le calcul
    complet (32 × 8 260² par couche), 25,6 Go demandés, manque de mémoire (constaté sur la RTX 4070). Clés et
    valeurs répétées (`repeat_kv`) à la place : noyau économe, pic à 10,1 Go, même résultat."""
    from transformers.integrations import sdpa_attention

    sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False


def decrire(chemin_tache):
    """Tâche : {mode: son|image (d'après une image) ou objet|bruitage|carte|video (texte reformulé), image?, texte?}.
    Renvoie RESULTAT {"texte": …}."""
    t = _lire(chemin_tache)
    if t.get("image") and not Path(t["image"]).is_file():  # avant de charger Qwen (8,9 Go)
        _erreur(f"image introuvable : {t['image']}")
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    mode = t.get("mode")
    if mode not in CONSIGNES:
        _erreur(f"mode de description inconnu : {mode}")
    device = _device()
    depot = t.get("depot") or QWEN  # QWEN_LIBRE si la version sans refus est choisie
    _hors_ligne_si_present(depot)
    print("PROGRESSION 1/2 chargement de Qwen3-VL", flush=True)

    def charger():  # sans device_map : un modèle « dispatché » par accelerate ne se déplace plus avec .to()
        _precharger((depot, None))
        m = Qwen3VLForConditionalGeneration.from_pretrained(
            depot, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
        return m, AutoProcessor.from_pretrained(depot)

    modele, processeur = _garder(("Qwen3-VL", depot, device), charger, 9, vers_cpu=lambda o: o[0].to("cpu"),
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
    if t.get("image") and mode in MODES_TEXTE:  # texte + image : Qwen la voit (vidéo : première image ;
        consigne = {"video": IMAGE_DE_DEPART, "scene": IMAGE_REFERENCE,  # image : modèle)
                    "histoire": IMAGE_PERSONNAGES}.get(mode, "") + consigne
    if mode == "parlante" and (t.get("texte") or "").strip():  # souhaits de l'utilisateur (souvent en français)
        consigne += (" Include these wishes from the user, translated into English: "
                     + " ".join(t["texte"].split()))
    if mode == "costume" and (t.get("texte") or "").strip():  # précisions de l'utilisateur sur la tenue
        consigne += (" The user adds: " + " ".join(t["texte"].split())
                     + " (translate into English and follow it).")
    if mode == "tuile3d" and (t.get("texte") or "").strip():  # ce que l'utilisateur dit de l'objet
        consigne += (" The user says this object is: " + " ".join(t["texte"].split())
                     + " (translate into English and follow it).")
    if t.get("image") and int(t.get("photos") or 1) > 1:
        consigne = PHOTOS_NUMEROTEES.format(n=int(t["photos"])) + consigne
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
        if mode in LISTES_NUMEROTEES:
            consigne += LISTES_NUMEROTEES[mode][1].format(n=max(1, int(t.get("nombre", 4))))
    contenu.append({"type": "text", "text": consigne})
    entrees = processeur.apply_chat_template([{"role": "user", "content": contenu}], tokenize=True,
                                             add_generation_prompt=True, return_dict=True, return_tensors="pt")
    entrees = entrees.to(modele.device)
    print("PROGRESSION 2/2 description", flush=True)
    _sans_gqa_sdpa()
    with torch.inference_mode():
        # « histoire » : un prompt par image (jusqu'à 12, 40 à 70 mots chacun ; 120 jetons par image coupaient la
        # réponse à la 9ᵉ sur 12) ; un seul prompt : JETONS_QWEN (160 pouvait couper une description de vidéo ou de
        # scène détaillée en pleine phrase)
        jetons = (LISTES_NUMEROTEES[mode][0] * max(1, int(t.get("nombre", 4))) + 100 if mode in LISTES_NUMEROTEES
                  else JETONS_QWEN)
        sortie = _memoire(modele.generate)(**entrees, max_new_tokens=jetons, do_sample=False)
    produits = sortie[:, entrees["input_ids"].shape[1]:]
    texte = processeur.batch_decode(produits, skip_special_tokens=True)[0]
    if produits.shape[1] >= jetons:  # réponse arrêtée par la limite, pas par Qwen : coupée en pleine phrase
        texte = _phrases_completes(texte, mode)
        print(f"AVERTISSEMENT : la réponse de Qwen a atteint sa longueur maximale ({jetons} jetons) ; la dernière "
              "phrase inachevée a été retirée. Relis le prompt avant de générer.", flush=True)
    if mode in LISTES_NUMEROTEES:  # un élément par ligne : les retours à la ligne sont gardés
        texte = "\n".join(" ".join(ligne.split()) for ligne in texte.strip().splitlines() if ligne.strip())
    else:
        texte = " ".join(texte.strip().strip('"').split())
    _resultat({"texte": texte})
    print("TERMINE -", flush=True)


JETONS_QWEN = 320  # longueur maximale d'une réponse de Qwen (un prompt)


def _phrases_completes(texte, mode):
    """Retire la fin inachevée d'une réponse coupée : jusqu'au dernier point (une ligne entière pour « histoire »)."""
    if mode in LISTES_NUMEROTEES:
        lignes = texte.rstrip().splitlines()
        return "\n".join(lignes[:-1]) if len(lignes) > 1 else texte
    fin = max(texte.rfind(". "), texte.rfind("."), texte.rfind("! "), texte.rfind("? "))
    return texte[:fin + 1] if fin > len(texte) // 3 else texte


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
                        "ton jeton (Outils → Modèles et diagnostic → Télécharger les bruitages).", 4)
            raise

    pipe = _garder(("Stable Audio Open", depot, device), charger, 3.5, vers_cpu=lambda p_: p_.to("cpu"),
                   vers_gpu=lambda p_: p_.to(device))
    graines = [int(t.get("graine") or 0) or int(torch.randint(1, 2**31 - 1, (1,)))]
    graines += [int(x) for x in torch.randint(1, 2**31 - 1, (variantes - 1,))]
    generateurs = [torch.Generator(device).manual_seed(g) for g in graines]
    _avertir_si_coupe(pipe.tokenizer, t["prompt"], pipe.tokenizer.model_max_length, "Stable Audio")
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
    prompts = _prompts(t)
    if sans_encodeur:  # test sans les 8 Go de l'encodeur : plongements de la bonne dimension
        plongements = {p_: [torch.randn(24, pipe.transformer.config.cap_feat_dim, dtype=dtype)] for p_ in prompts}
    else:
        if device == "cuda":
            _sur_carte(pipe, "vae", "text_encoder")
        plongements = {p_: _encoder_une_fois(pipe, p_, do_classifier_free_guidance=False) for p_ in prompts}
    if device == "cuda":
        _sur_carte(pipe, "vae", "transformer")
        plongements = {p_: [e.to(device) for e in v] for p_, v in plongements.items()}
    _generer(pipe, t, {"guidance_scale": 0.0}, etapes=9, plongements=[plongements[p_] for p_ in _prompts(t, tous=True)])


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


# Longueur du texte lu par Z-Image et FLUX.2 klein : 512 jetons par défaut dans diffusers, et la suite est COUPÉE
# sans prévenir (truncation=True). Constaté : un contexte de quatre personnages décrits en détail dépassait 512
# jetons, la description de chaque image (à la fin) disparaissait et toutes les images montraient le même portrait
# de groupe. Au-delà de 512, la longueur est portée au multiple de 64 suffisant, jusqu'à TEXTE_MAX (Z-Image prévoit
# 1536 positions de texte, axes_lens[0] ; klein calcule ses positions à la volée). En dessous, rien ne change.
TEXTE_DEFAUT, TEXTE_MAX = 512, 1024


def _longueur_texte(pipe, prompt):
    """max_sequence_length pour ce prompt (gabarit de conversation compris) ; avertit s'il faut quand même couper."""
    jetons = len(pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                    add_generation_prompt=True))
    if jetons <= TEXTE_DEFAUT:
        return TEXTE_DEFAUT
    if jetons > TEXTE_MAX:
        print(f"AVERTISSEMENT : prompt trop long ({jetons} jetons) : le générateur n'en lit que {TEXTE_MAX}, la fin "
              f"est ignorée (« …{prompt[-80:]} »). Raccourcis le contexte ou les descriptions.", flush=True)
        return TEXTE_MAX
    return -(-jetons // 64) * 64


def _avertir_si_coupe(tokeniseur, texte, maximum, generateur):
    """Les encodeurs de texte coupent sans prévenir au-delà de leur longueur (Stable Audio : T5, 128 jetons ; Wan :
    umT5, 512 jetons) : ligne AVERTISSEMENT (bandeau de l'interface) si c'est le cas. Renvoie le nombre de jetons."""
    jetons = len(tokeniseur(texte).input_ids)
    if jetons > maximum:
        mots = texte.split()
        print(f"AVERTISSEMENT : prompt trop long pour {generateur} ({jetons} jetons, il n'en lit que {maximum}) : "
              f"la fin est ignorée (« …{' '.join(mots[-12:])} »). Raccourcis le prompt.", flush=True)
    return jetons


def _encoder_une_fois(pipe, prompt, **options):
    """Le texte est encodé une seule fois pour toutes les variantes. Sinon, sur 12 Go (déchargement vers la mémoire
    vive), chaque variante recharge l'encodeur (8 Go) puis le transformeur sur la carte : la carte attend les
    transferts au lieu de calculer. Longueur lue : _longueur_texte (sinon coupé à 512 jetons)."""
    import torch

    with torch.no_grad():
        return pipe.encode_prompt(prompt=prompt, device=pipe._execution_device,
                                  max_sequence_length=_longueur_texte(pipe, prompt), **options)[0]


def _prompts(t, tous=False):
    """Prompts de la tâche : « prompts » (un par sortie : contexte commun + description de chaque image) ou
    « prompt » (le même pour toutes). tous=False : chacun une seule fois, dans l'ordre (à encoder) ; tous=True :
    un par sortie."""
    sorties = t["sorties"]
    prompts = list(t.get("prompts") or [t["prompt"]] * len(sorties))
    if len(prompts) != len(sorties):
        _erreur(f"{len(prompts)} prompt(s) pour {len(sorties)} image(s).")
    return prompts if tous else list(dict.fromkeys(prompts))


def _generer(pipe, t, options, etapes, plongements):
    """Une image par sortie de la tâche, chacune avec sa graine (générateur sur le processeur : reproductible) et
    son prompt déjà encodé (plongements : un par sortie)."""
    import torch

    largeur, hauteur = int(t.get("largeur", 1024)), int(t.get("hauteur", 1024))
    sorties, graines = t["sorties"], [int(g) for g in t["graines"]]
    print(f"PROGRESSION 2/2 génération de {len(sorties)} image(s)", flush=True)
    for i, (sortie, graine, plongement) in enumerate(zip(sorties, graines, plongements), 1):
        im = _memoire(pipe)(
            **options, prompt_embeds=plongement, height=hauteur, width=largeur,
            num_inference_steps=int(t.get("etapes", etapes)),
            generator=torch.Generator("cpu").manual_seed(graine),
        ).images[0]
        Path(sortie).parent.mkdir(parents=True, exist_ok=True)
        im.save(sortie)
        print(f"Image {i}/{len(sorties)} : {sortie} (graine {graine})", flush=True)
    _resultat({"fichiers": sorties, "graines": graines})
    print(f"TERMINE {sorties[0]}", flush=True)


# --- SDXL + LoRA de sprites de dresseurs (kit de dresseur) ---------------------------------------------------
def _peft_requis():
    """diffusers ne charge un LoRA qu'avec peft (« PEFT backend is required for this method », constaté chez
    l'utilisateur le 31/10 : peft manquait dans l'environnement de diffusion)."""
    import importlib.util

    if importlib.util.find_spec("peft") is None:
        _erreur("le paquet peft manque dans l'environnement de diffusion (il sert à charger les modèles de "
                "sprites) : lance METTRE_A_JOUR.bat pour l'installer.")


def sprite_dresseur(chemin_tache):
    """Tâche : {prompt | prompts, negatif?, sorties, graines, largeur, hauteur, etapes, guidage, lora_echelle}.
    StableDiffusionXLPipeline (fp16, ~7 Go : tout sur la carte) + LoRA au format Kohya (lora_te1/te2/unet, converti
    par load_lora_weights). Clés de test : depot, lora (chemin local)."""
    import torch
    from diffusers import StableDiffusionXLPipeline
    from huggingface_hub import hf_hub_download

    t = _lire(chemin_tache)
    device = _device()
    depot = t.get("depot", SDXL)
    _hors_ligne_si_present(depot, LORA_DRESSEUR[0])
    dtype = torch.float16 if device == "cuda" else torch.float32
    print("PROGRESSION 1/2 chargement de SDXL et du LoRA de sprites", flush=True)

    def charger():
        _precharger((depot, "unet"))
        options = {"torch_dtype": dtype, "use_safetensors": True}
        if not t.get("depot"):
            options["variant"] = "fp16"
        pipe = StableDiffusionXLPipeline.from_pretrained(depot, **options)
        lora = t.get("lora") or hf_hub_download(*LORA_DRESSEUR)
        _peft_requis()
        pipe.load_lora_weights(lora, adapter_name="dresseur")
        return pipe.to(device)

    pipe = _garder(("SDXL sprites", depot, t.get("lora"), device), charger, 8, vers_cpu=lambda p_: p_.to("cpu"),
                   vers_gpu=lambda p_: p_.to(device))
    pipe.set_adapters(["dresseur"], [float(t.get("lora_echelle", 1.0))])
    largeur, hauteur = int(t.get("largeur", 1024)), int(t.get("hauteur", 1024))
    sorties, graines = t["sorties"], [int(g) for g in t["graines"]]
    prompts = _prompts(t, tous=True)
    print(f"PROGRESSION 2/2 génération de {len(sorties)} sprite(s)", flush=True)
    for i, (sortie, graine, prompt) in enumerate(zip(sorties, graines, prompts), 1):
        im = _memoire(pipe)(
            prompt=prompt, negative_prompt=t.get("negatif") or NEGATIF_SPRITE_SDXL, height=hauteur, width=largeur,
            num_inference_steps=int(t.get("etapes", 30)), guidance_scale=float(t.get("guidage", 7.0)),
            generator=torch.Generator("cpu").manual_seed(graine)).images[0]
        Path(sortie).parent.mkdir(parents=True, exist_ok=True)
        im.save(sortie)
        print(f"Sprite {i}/{len(sorties)} : {sortie} (graine {graine})", flush=True)
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
    from PIL import Image, ImageOps

    t = _lire(chemin_tache)
    # photos de téléphone : l'orientation est dans l'EXIF (sinon la référence arrive couchée)
    references = [ImageOps.exif_transpose(Image.open(r)).convert("RGB")
                  for r in t.get("references") or []][:REFERENCES_MAX]
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
    prompts = _prompts(t)
    if sans_encodeur:  # 3 couches cachées de l'encodeur mises bout à bout
        plongements = {p_: torch.randn(1, 24, pipe.transformer.config.joint_attention_dim, dtype=dtype)
                       for p_ in prompts}
    else:
        if device == "cuda":
            _sur_carte(pipe, "vae", "text_encoder")  # encodeur : 8 Go, seul sur la carte
        plongements = {p_: _encoder_une_fois(pipe, p_) for p_ in prompts}
    if device == "cuda":
        _sur_carte(pipe, "vae", "transformer")  # transformeur (4,3 Go) et VAE : sur la carte pour toutes les variantes
        plongements = {p_: v.to(device) for p_, v in plongements.items()}
    print(f"{len(references)} image(s) de référence du personnage.", flush=True)
    _generer(pipe, t, {"image": references, "guidance_scale": 1.0}, etapes=4,
             plongements=[plongements[p_] for p_ in _prompts(t, tous=True)])


# --- Wan 2.2 : vidéo à partir d'un texte ou d'une image ----------------------------------------------------
def _negatif_video(t):
    """Négatif de Wan : celui de la tâche ou celui par défaut, plus `negatif_ajout` (sprites : fond qui change)."""
    negatif = t.get("negatif") or NEGATIF_VIDEO
    return f"{negatif}, {t['negatif_ajout']}" if t.get("negatif_ajout") else negatif


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
    _avertir_si_coupe(tokeniseur, t["prompt"], 512, "Wan 2.2")  # max_sequence_length=512 (text_len de Wan)
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
            prompt, negatif = pipe.encode_prompt(prompt=t["prompt"], negative_prompt=_negatif_video(t),
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
                                                 negative_prompt=_negatif_video(t),
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
                    if i > 1 and k == 0 and t.get("enchaines", True):
                        continue  # clips indépendants (un par image d'une histoire) : rien n'est retiré
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
        if "transparency" in im.info and im.mode not in ("RGBA", "LA"):
            # PNG à palette ou GIF transparents (sprites de pixel art) : pas de canal « A » avant conversion
            # (« The image has no channel "A" », constaté chez l'utilisateur)
            im = im.convert("RGBA")
        alpha = im.getchannel("A") if im.mode in ("RGBA", "LA") else None
        if alpha is not None and alpha.getextrema() == (255, 255):
            alpha = None  # canal alpha entièrement opaque : inutile
        return im.convert("RGB"), alpha


def detourer(chemin_tache):
    """Tâche : {entree, sortie, modele: general | personne, fond: None | "#rrggbb" | "flou", masque: chemin}, ou
    {entrees, sorties, modele} pour un lot (images en série à fond transparent : le modèle n'est chargé qu'une fois).
    BiRefNet donne un masque doux (0–255) ; les couleurs du fond mêlées aux bords (cheveux) sont retirées par
    l'estimation du premier plan de pymatting (rembg.bg.decontaminate_cutout). Sans fond : PNG transparent.
    fp16 sur la carte graphique (jamais bf16 : deform_conv2d de torchvision ne le gère pas).
    RESULTAT {sortie, ...} ou, pour un lot, {sorties, couvertures}."""
    import torch
    from transformers import AutoModelForImageSegmentation

    t = _lire(chemin_tache)
    depot, revision, cote = BIREFNET[t.get("modele") or "general"]
    if t.get("cote"):  # tests : modèle plus petit
        cote = int(t["cote"])
    device = _device()
    _hors_ligne_si_present(depot)
    lot = "entrees" in t
    n = len(t["entrees"]) + 1 if lot else 3
    print(f"PROGRESSION 1/{n} chargement de BiRefNet", flush=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.set_float32_matmul_precision("high")
    depot, revision = t.get("depot", depot), t.get("revision", revision)
    modele = _garder(
        ("BiRefNet", depot, revision, device),
        lambda: AutoModelForImageSegmentation.from_pretrained(depot, trust_remote_code=True, revision=revision)
        .to(device=device, dtype=dtype).eval(),
        1.0, vers_cpu=lambda m: m.to("cpu"), vers_gpu=lambda m: m.to(device))
    tete = bool(t.get("tete"))  # « Garder seulement la tête » : détourage puis tête autour de chaque visage
    if lot:
        # une image en échec (aucun visage, fichier illisible…) est consignée et le lot continue : avant, un seul
        # échec faisait échouer tout le lot (constaté chez l'utilisateur le 24/10)
        couvertures, tetes, erreurs = [], [], []
        for i, (entree, sortie) in enumerate(zip(t["entrees"], t["sorties"]), 2):
            print(f"PROGRESSION {i}/{n} détourage de {Path(sortie).name}", flush=True)
            try:
                image, _, couverture = _detourer_image(modele, cote, device, dtype, entree, sortie)
                trouvees = _tetes(sortie, image, _dossier_photos()) if tete else None
            except torch.cuda.OutOfMemoryError:
                raise  # la carte n'est plus fiable : le moteur résident sera relancé
            except Exception as e:  # noqa: BLE001
                print(f"AVERTISSEMENT : {Path(entree).name} non traitée ({e}).", flush=True)
                erreurs.append({"index": i - 2, "fichier": Path(entree).name, "message": str(e)})
                Path(sortie).unlink(missing_ok=True)
                couverture, trouvees = None, []
            couvertures.append(couverture)
            if tete:
                tetes.append(trouvees)
        if len(erreurs) == len(t["entrees"]):
            _erreur("aucune image n'a pu être traitée : " + " ; ".join(f"{e['fichier']} : {e['message']}"
                                                                         for e in erreurs))
        _resultat({"sorties": t["sorties"], "couvertures": couvertures, "erreurs": erreurs,
                   **({"tetes": tetes} if tete else {})})
        print(f"TERMINE {t['sorties'][-1]}", flush=True)
        return
    image, masque, couverture = _detourer_image(modele, cote, device, dtype, t["entree"], t["sortie"],
                                                None if tete else t.get("fond"), etapes=True)
    if t.get("masque"):
        masque.save(t["masque"])
    try:
        tetes = _tetes(t["sortie"], image, _dossier_photos()) if tete else None
    except _EchecImage as e:
        _erreur(str(e))
    _resultat({"sortie": t["sortie"], "masque": t.get("masque"), "largeur": image.width, "hauteur": image.height,
               "couverture": couverture, **({"tetes": tetes} if tete else {})})
    print(f"TERMINE {t['sortie']}", flush=True)


def _detourer_image(modele, cote, device, dtype, entree, sortie, fond=None, etapes=False):
    """Une image détourée par BiRefNet, enregistrée dans sortie (RGBA sans fond). Renvoie (image, masque, part de
    l'image occupée par le sujet)."""
    import numpy as np
    import torch
    from PIL import Image, ImageFilter
    from rembg.bg import decontaminate_cutout
    from torchvision import transforms

    image, _ = _ouvrir_photo(entree)
    if etapes:
        print(f"PROGRESSION 2/3 détourage ({image.width}×{image.height})", flush=True)
    preparation = transforms.Compose([transforms.Resize((cote, cote)), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = preparation(image).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.inference_mode():
        prediction = _memoire(modele)(x)[-1].sigmoid().float().cpu()[0, 0].numpy()
    masque = Image.fromarray((prediction * 255).round().astype("uint8")).resize(image.size, Image.BILINEAR)
    if etapes:
        print("PROGRESSION 3/3 finitions des bords", flush=True)
    decoupe = decontaminate_cutout(image, masque)  # RGBA, couleurs des bords sans le fond
    if fond == "flou":  # effet portrait : le sujet net devant son propre fond flouté
        base = image.filter(ImageFilter.GaussianBlur(max(image.size) / 80)).convert("RGBA")
        resultat = Image.alpha_composite(base, decoupe).convert("RGB")
    elif fond:
        base = Image.new("RGBA", image.size, fond)
        resultat = Image.alpha_composite(base, decoupe).convert("RGB")
    else:
        resultat = decoupe
    Path(sortie).parent.mkdir(parents=True, exist_ok=True)
    resultat.save(sortie)
    couverture = float(np.asarray(masque, dtype=np.float32).mean() / 255)
    return image, masque, round(couverture, 3)


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


def _visages_yunet(bgr, dossier, cote_min=48):
    """Visages trouvés par YuNet (MIT, dans OpenCV), du plus grand au plus petit : tableaux de 15 valeurs (boîte
    x, y, l, h, 5 points du visage, score) dans les coordonnées de l'image."""
    import cv2

    h, w = bgr.shape[:2]
    reduction = min(1.0, 1600 / max(h, w))  # détection sur une copie réduite des grandes images
    petite = cv2.resize(bgr, (round(w * reduction), round(h * reduction)), interpolation=cv2.INTER_AREA)
    detecteur = cv2.FaceDetectorYN.create(str(dossier / "face_detection_yunet_2023mar.onnx"), "",
                                          (petite.shape[1], petite.shape[0]), 0.7, 0.3, 5000)
    _, visages = detecteur.detect(petite)
    if visages is None:
        return []
    visages = [v.copy() for v in visages]
    for v in visages:
        v[:14] /= reduction
    return sorted((v for v in visages if v[2] >= cote_min), key=lambda v: -v[2] * v[3])


# Boîte de la tête autour du visage trouvé par YuNet (largeur l, hauteur h du visage) : les cheveux au-dessus et
# sur les côtés ; en bas, coupe juste sous le menton (le bas de la boîte de YuNet), sans le cou (demande de
# l'utilisateur du 24/10 : « que la tête »), avec un fondu très court
TETE_HAUT, TETE_COTES, TETE_BAS, TETE_FONDU = 0.8, 0.5, 0.08, 0.05


def _decouper_tete(rgba, boite):
    """Image détourée (H×W×4) → la tête seule autour du visage `boite` (x, y, l, h) : transparence hors de la boîte
    de la tête, cou coupé en fondu, seulement la zone reliée au visage (pas l'épaule d'un voisin), recadrée au plus
    près. Renvoie un tableau RGBA ou None si rien ne reste."""
    import numpy as np
    from scipy import ndimage

    x, y, l, h = (float(v) for v in boite[:4])
    hauteur, largeur = rgba.shape[:2]
    x0, x1 = max(0, int(x - TETE_COTES * l)), min(largeur, int(np.ceil(x + l + TETE_COTES * l)))
    y0, y1 = max(0, int(y - TETE_HAUT * h)), min(hauteur, int(np.ceil(y + h + TETE_BAS * h)))
    alpha = np.zeros((hauteur, largeur), np.float32)
    alpha[y0:y1, x0:x1] = rgba[y0:y1, x0:x1, 3]
    lignes = np.arange(hauteur, dtype=np.float32)[:, None]
    debut_fondu = y + h + (TETE_BAS - TETE_FONDU) * h
    alpha *= np.clip(1 - (lignes - debut_fondu) / max(1.0, TETE_FONDU * h), 0, 1)
    zones, n = ndimage.label(alpha > 32)
    if n:
        cy, cx = min(hauteur - 1, int(y + h / 2)), min(largeur - 1, int(x + l / 2))
        garde = zones[cy, cx]
        if not garde:  # centre du visage pas opaque : la zone la plus présente dans la boîte du visage
            dans = zones[int(y):int(y + h), int(x):int(x + l)]
            valeurs, nombres = np.unique(dans[dans > 0], return_counts=True)
            garde = valeurs[nombres.argmax()] if len(valeurs) else 0
        if garde:
            proche = ndimage.binary_dilation(zones == garde, iterations=2)
            alpha *= proche
    if alpha.max() < 1:
        return None
    ys, xs = np.where(alpha > 0)
    m = 2
    a0, a1 = max(0, ys.min() - m), min(hauteur, ys.max() + 1 + m)
    b0, b1 = max(0, xs.min() - m), min(largeur, xs.max() + 1 + m)
    sortie = rgba[a0:a1, b0:b1].copy()
    sortie[..., 3] = np.round(alpha[a0:a1, b0:b1]).astype(np.uint8)
    return sortie


class _EchecImage(Exception):
    """Une image d'un lot n'a pas pu être traitée (aucun visage…) : consignée, le lot continue."""


def _tetes(sortie, image, dossier):
    """La photo détourée (`sortie`, PNG transparent) devient la tête du plus grand visage ; les autres visages vont
    dans <nom>_2.png, <nom>_3.png… Renvoie la liste des fichiers ; erreur si aucun visage."""
    import cv2
    import numpy as np
    from PIL import Image

    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    visages = _visages_yunet(bgr, dossier, cote_min=24)
    if not visages:
        raise _EchecImage(f"aucun visage trouvé sur {Path(sortie).name} : « Garder seulement la tête » a besoin "
                          "d'un visage bien visible.")
    rgba = np.asarray(Image.open(sortie).convert("RGBA"))
    fichiers = []
    for i, v in enumerate(visages, 1):
        tete = _decouper_tete(rgba, v)
        if tete is None:
            continue
        chemin = Path(sortie) if i == 1 else Path(sortie).with_name(f"{Path(sortie).stem}_{i}.png")
        Image.fromarray(tete, "RGBA").save(chemin)
        fichiers.append(str(chemin))
    if not fichiers:
        raise _EchecImage(f"la tête n'a pas pu être isolée sur {Path(sortie).name}.")
    return fichiers


def _restaurer_visages(bgr, dossier, device, force=0.7, cote_min=48):
    """GFPGAN 1.4 sur chaque visage trouvé par YuNet : alignement sur le gabarit, restauration 512×512, recollage
    fondu. On appelle le réseau directement (entrée et sortie en [-1, 1]) : le descripteur de spandrel travaille
    en [0, 1] et changeait l'identité (essai : yeux marron devenus bleus). Renvoie (image, nombre de visages)."""
    import cv2
    import numpy as np
    import torch
    from spandrel import ModelLoader

    h, w = bgr.shape[:2]
    visages = _visages_yunet(bgr, dossier, cote_min)
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
        points = v[4:14].reshape(5, 2)
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
    """Tâche : {image, dos?, dossier, etapes, octree, faces, graine, texture: bool, formats: ["glb", "obj"]}.
    dos : vue de dos de l'objet → forme par Hunyuan3D-2mv (face + dos) ; la texture est peinte depuis la face.
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
    # Comme hy3dgen.rembg.BackgroundRemover, mais avec le modèle u2net (Apache 2.0) : le modèle par défaut
    # de rembg est désormais BRIA RMBG-2.0, à usage non commercial.
    session = _garder(("u2net",), lambda: new_session("u2net"), 0.2)
    im = remove(Image.open(t["image"]).convert("RGB"), session=session, bgcolor=[255, 255, 255, 0])
    im.save(dossier / "image_detouree.png")
    vues = None
    if t.get("dos"):  # vue de dos : Hunyuan3D-2mv sculpte aussi ce qui est derrière (un arc porté dans le dos…)
        dos = remove(Image.open(t["dos"]).convert("RGB"), session=session, bgcolor=[255, 255, 255, 0])
        dos.save(dossier / "image_dos_detouree.png")
        vues = {"front": im, "back": dos}

    print(f"PROGRESSION 2/{total} chargement du générateur de forme", flush=True)
    _hors_ligne_si_present(HUNYUAN, *([HUNYUAN_MV] if vues else []))
    if vues:
        depot, sous_dossier = t.get("depot_vues", HUNYUAN_MV), t.get("sous_dossier_vues", HUNYUAN_MV_FORME)
    else:
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
        image=vues or im, num_inference_steps=int(t.get("etapes", 30)), guidance_scale=float(t.get("guidage", 7.5)),
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
        if t.get("facettes"):  # sans texture : facettes blanches
            resultat["facettes"] = _facetter_glb(forme, dossier / "modele_facettes.glb", int(t["facettes"]),
                                                 int(t.get("couleurs", 24)))["sortie"]
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
    if t.get("facettes"):
        resultat["facettes"] = _facetter_glb(modele, dossier / "modele_facettes.glb", int(t["facettes"]),
                                             int(t.get("couleurs", 24)))["sortie"]
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


def _simplifier(mesh, faces):
    """Maillage (sans texture) réduit à `faces` faces : pymeshlab (quadriques, formes planes gardées), sinon
    fast_simplification par trimesh s'il est installé, sinon tel quel."""
    import trimesh

    if len(mesh.faces) <= faces:
        return mesh
    try:
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(faces), preservenormal=True,
                                                    preservetopology=True, planarquadric=True, qualitythr=0.3)
        m = ms.current_mesh()
        return trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=True)
    except Exception as e:  # noqa: BLE001 - greffon pymeshlab absent (Linux sans libOpenGL)
        try:
            return mesh.simplify_quadric_decimation(face_count=int(faces))
        except Exception:  # noqa: BLE001
            print(f"AVERTISSEMENT : simplification impossible ({e}) ; faces gardées.", flush=True)
            return mesh


def _palette_kmeans(couleurs, poids, n, tours=12):
    """n couleurs représentatives (k-moyennes pondérées par l'aire des faces, départ sur les plus fréquentes)."""
    import numpy as np

    n = max(1, min(int(n), len(couleurs)))
    ordre = np.argsort(-poids)
    centres = [couleurs[ordre[0]]]
    for i in ordre[1:]:  # départ : couleurs fréquentes et éloignées les unes des autres
        if len(centres) >= n:
            break
        if np.min(np.abs(np.array(centres) - couleurs[i]).sum(axis=1)) > 24:
            centres.append(couleurs[i])
    centres = np.array(centres, float)
    for _ in range(tours):
        etiquettes = np.argmin(((couleurs[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        for k in range(len(centres)):
            m = etiquettes == k
            if m.any():
                centres[k] = np.average(couleurs[m], axis=0, weights=poids[m] + 1e-9)
    return centres, etiquettes


def _facetter_glb(entree, sortie, faces=3000, couleurs=24):
    """Version « low poly facettée » d'un modèle de Hunyuan3D (style des jeux de construction de ville) : forme
    réduite à `faces` faces, une couleur unie par face (lue dans la texture d'origine : plusieurs points de la face,
    couleur médiane des faces d'origine les plus proches), couleurs ramenées à une palette de `couleurs` teintes,
    arêtes nettes (sommets non partagés : normale de la face). Sans texture, faces blanches."""
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    origine = trimesh.load(str(entree), force="mesh")
    visuel = getattr(origine, "visual", None)
    image = getattr(getattr(visuel, "material", None), "baseColorTexture", None)
    if image is None:
        image = getattr(getattr(visuel, "material", None), "image", None)
    uv = getattr(visuel, "uv", None)
    # couleur de chaque face d'origine : texture lue en 4 points (centre et vers chaque sommet)
    if image is not None and uv is not None and len(uv) == len(origine.vertices):
        tex = np.asarray(image.convert("RGB"), dtype=float)
        h, w = tex.shape[:2]
        uv_faces = uv[origine.faces]  # (n, 3, 2)
        points = [uv_faces.mean(axis=1)] + [0.5 * uv_faces.mean(axis=1) + 0.5 * uv_faces[:, k] for k in range(3)]
        lectures = []
        for p in points:
            x = np.clip((p[:, 0] % 1.0) * (w - 1), 0, w - 1).round().astype(int)
            y = np.clip((1 - p[:, 1] % 1.0) * (h - 1), 0, h - 1).round().astype(int)
            lectures.append(tex[y, x])
        couleurs_origine = np.median(np.stack(lectures), axis=0)
    else:
        couleurs_origine = np.full((len(origine.faces), 3), 235.0)
    arbre = cKDTree(origine.triangles_center)
    forme = _simplifier(trimesh.Trimesh(origine.vertices, origine.faces, process=True), int(faces))
    tri = forme.triangles
    centres = tri.mean(axis=1)
    echantillons = np.concatenate([centres] + [0.6 * centres + 0.4 * tri[:, k] for k in range(3)])
    _, voisins = arbre.query(echantillons, k=3)
    lus = couleurs_origine[voisins].reshape(4, len(centres), 3, 3).transpose(1, 0, 2, 3).reshape(len(centres), 12, 3)
    teintes = np.median(lus, axis=1)
    if couleurs and int(couleurs) > 0:
        palette, etiquettes = _palette_kmeans(teintes, forme.area_faces, int(couleurs))
        teintes = palette[etiquettes]
    final = trimesh.Trimesh(forme.vertices, forme.faces, process=False)
    final.unmerge_vertices()  # arêtes nettes : chaque face a ses propres sommets (normale de la face)
    rgba = np.concatenate([np.clip(teintes, 0, 255), np.full((len(teintes), 1), 255.0)], axis=1).astype(np.uint8)
    final.visual = trimesh.visual.ColorVisuals(final, face_colors=rgba)
    final.export(str(sortie), include_normals=True)
    return {"sortie": str(sortie), "faces": int(len(final.faces)), "couleurs": int(len(np.unique(rgba, axis=0)))}


def facetter(chemin_tache):
    """Tâche : {entree, sortie, faces, couleurs}. RESULTAT {sortie, faces, couleurs}."""
    t = _lire(chemin_tache)
    print("PROGRESSION 1/1 version facettée", flush=True)
    res = _facetter_glb(t["entree"], t["sortie"], int(t.get("faces", 3000)), int(t.get("couleurs", 24)))
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


# --- RIFE : vidéos plus fluides, ralentis -----------------------------------------------------------------------
CHANGEMENT_DE_PLAN = 0.15  # écart moyen (0–1) entre deux images au-delà duquel on n'invente pas d'image intermédiaire


def fluidifier(chemin_tache):
    """Tâche : {entree, sortie, facteur: 2 à 4, ralenti: bool}. RIFE 4.26 intercale facteur − 1 images entre deux
    images voisines : sans ralenti, même durée et cadence × facteur (son d'origine gardé) ; en ralenti, même cadence
    et durée × facteur (sans son). Aux changements de plan (vidéo assemblée), l'image est répétée au lieu d'un
    fondu inventé. Lecture et écriture image par image (imageio-ffmpeg, H.264 yuv420p) : aucune limite de durée.
    RESULTAT {sortie, images, fps, duree}."""
    import imageio_ffmpeg
    import numpy as np
    import torch

    import rife

    t = _lire(chemin_tache)
    facteur = max(2, min(4, int(t.get("facteur", 2))))
    ralenti = bool(t.get("ralenti"))
    device = _device()
    print("PROGRESSION 1/2 chargement de RIFE", flush=True)
    poids = Path(t.get("poids") or (_dossier_photos() / "flownet_v4.26.pkl"))
    if not poids.exists():
        _erreur("modèle RIFE absent : télécharge-le dans Outils → Modèles et diagnostic.")
    net = _garder(("RIFE", str(poids), device), lambda: rife.charger(str(poids), device), 0.1,
                  vers_cpu=lambda m: m.to("cpu"), vers_gpu=lambda m: m.to(device))
    lecteur = imageio_ffmpeg.read_frames(t["entree"], pix_fmt="rgb24")
    meta = lecteur.__next__()
    largeur, hauteur = meta["size"]
    fps = float(meta.get("fps") or 24)
    total = max(1, int(round(float(meta.get("duration") or 0) * fps)))
    fps_sortie = fps if ralenti else fps * facteur
    son = None if ralenti or not meta.get("audio_codec") else t["entree"]
    ecrivain = imageio_ffmpeg.write_frames(t["sortie"], (largeur, hauteur), fps=fps_sortie, codec="libx264",
                                           pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=8,
                                           macro_block_size=1, audio_path=son, audio_codec="aac" if son else None)
    ecrivain.send(None)
    echelle = 0.5 if max(largeur, hauteur) > 2000 else 1.0  # 4K : flux calculé à mi-résolution (conseil de RIFE)

    def tenseur(octets):
        image = np.frombuffer(octets, np.uint8).reshape(hauteur, largeur, 3)
        return torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(device).float() / 255.0

    def octets(x):
        return (x[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy().tobytes()

    print(f"PROGRESSION 2/2 images intermédiaires ({total} images, × {facteur})", flush=True)
    precedent, ecrites, plans, n = None, 0, 0, 0
    for brut in lecteur:
        courant = tenseur(brut)
        if precedent is not None:
            ecart = float((precedent - courant).abs().mean())
            if ecart > CHANGEMENT_DE_PLAN:
                plans += 1
                intermediaires = [octets(precedent)] * (facteur - 1)
            else:
                with torch.inference_mode():
                    intermediaires = [octets(_memoire(rife.intercaler)(net, precedent, courant, k / facteur, echelle))
                                      for k in range(1, facteur)]
            for image in intermediaires:
                ecrivain.send(image)
                ecrites += 1
        ecrivain.send(bytes(brut))
        ecrites += 1
        precedent = courant
        n += 1
        if n % 10 == 0:
            print(f"Image {n}/{total}", flush=True)
    ecrivain.close()
    if n < 2:
        _erreur("la vidéo doit contenir au moins deux images.")
    print(f"{n} images lues, {ecrites} écrites, {plans} changement(s) de plan sans image inventée.", flush=True)
    _resultat({"sortie": t["sortie"], "images": ecrites, "fps": round(fps_sortie, 3),
               "duree": round(ecrites / fps_sortie, 2), "plans": plans})
    print(f"TERMINE {t['sortie']}", flush=True)


# --- StableAvatar : vidéo parlante (lèvres synchronisées sur une voix) -----------------------------------------
def _dossier_code():
    return Path(os.environ.get("STUDIOVOIX_CODE") or Path.cwd() / "code")


def _code_stableavatar():
    return _dossier_code() / f"StableAvatar-{STABLEAVATAR_CODE[1][:12]}"


def _telecharger_code_stableavatar():
    """Code de StableAvatar (paquet « wan » seulement) depuis l'archive GitHub du commit épinglé : le dossier n'est
    complet qu'une fois tout extrait (renommage final), un téléchargement interrompu repart de zéro."""
    import io
    import shutil
    import urllib.request
    import zipfile

    cible = _code_stableavatar()
    if (cible / "wan" / "pipeline" / "wan_inference_long_pipeline.py").is_file():
        return
    depot, commit = STABLEAVATAR_CODE
    print(f"Téléchargement du code de StableAvatar ({commit[:7]})…", flush=True)

    def recuperer():
        with urllib.request.urlopen(f"https://github.com/{depot}/archive/{commit}.zip", timeout=60) as r:
            return r.read()

    archive = zipfile.ZipFile(io.BytesIO(_avec_reprises(recuperer, "code de StableAvatar")))
    provisoire = cible.with_name(cible.name + ".part")
    shutil.rmtree(provisoire, ignore_errors=True)
    for membre in archive.namelist():
        relatif = membre.split("/", 1)[1] if "/" in membre else ""
        if relatif.startswith("wan/") and not membre.endswith("/"):
            chemin = provisoire / relatif
            chemin.parent.mkdir(parents=True, exist_ok=True)
            chemin.write_bytes(archive.read(membre))
    (provisoire / "LICENSE.txt").write_bytes(archive.read(f"StableAvatar-{commit}/LICENSE.txt"))
    shutil.rmtree(cible, ignore_errors=True)
    provisoire.rename(cible)


def format_parlante(largeur, hauteur):
    """Format d'entraînement le plus proche des proportions de l'image : carré, portrait ou paysage."""
    import math

    rapport = largeur / max(1, hauteur)
    return min(FORMATS_PARLANTE, key=lambda f: abs(math.log(rapport * f[1] / f[0])))


def _son_16k(chemin):
    """Son mono à 16 kHz (float32) pour wav2vec. L'application écrit déjà audio_16k.wav en 16 kHz mono ; librosa
    n'est pas dans l'environnement de diffusion (« No module named 'librosa' » chez l'utilisateur le 24/10) :
    soundfile, et un rééchantillonnage linéaire au cas où le fichier ne serait pas à 16 kHz."""
    import numpy as np
    import soundfile as sf

    voix, sr = sf.read(str(chemin), dtype="float32", always_2d=True)
    voix = voix.mean(axis=1)
    if sr != 16000 and len(voix):
        n = max(1, round(len(voix) * 16000 / sr))
        voix = np.interp(np.linspace(0, len(voix) - 1, n), np.arange(len(voix)), voix).astype(np.float32)
    return voix, 16000


def parler(chemin_tache):
    """Tâche : {image, audio (WAV 16 kHz mono, lu par le modèle : voix seule si elle a été isolée), son (son de la
    vidéo, complet ; par défaut audio), sortie (MP4), prompt, largeur/hauteur
    (un des FORMATS_PARLANTE, sinon d'après l'image), etapes (30 à 50 conseillés), graine, guidage_texte (3),
    guidage_audio (5 ; plus haut = lèvres mieux synchronisées), chevauchement (5 à 15 fenêtres latentes : plus =
    meilleurs raccords, plus lent)}.
    StableAvatar génère toute la durée du son par fenêtres de 81 images qui se chevauchent (25 images/s). Phase 1 :
    umT5 de Wan 2.2 lit le prompt (comme les vidéos), puis il est libéré ; phase 2 : transformeur, VAE et wav2vec
    sur la carte ; l'encodeur d'image CLIP n'y passe que le temps de lire l'image (sa partie texte, inutile, est
    retirée). Le son d'origine est ajouté à la vidéo (AAC). RESULTAT {sortie, graine, images, duree, largeur, hauteur}.
    Clés de test : code (dossier du code), poids (dossier des poids), depot_encodeur, sans_encodeur."""
    import contextlib

    import imageio_ffmpeg
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    t = _lire(chemin_tache)
    device = _device()
    code = Path(t.get("code") or _code_stableavatar())
    if not (code / "wan").is_dir():
        _erreur("code de StableAvatar absent : télécharge « StableAvatar » dans Outils → Modèles et diagnostic.")
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    _hors_ligne_si_present(STABLEAVATAR, WAN)
    etapes = int(t.get("etapes", 30))
    total = etapes + 3
    # bf16 sur la carte (réglage d'origine) ; fp32 sur le processeur, où le code mélangeait les types (il compte sur
    # l'autocast CUDA, qui remet CLIP en fp32 : « Input type BFloat16 and weight type Float », puis LayerNorm)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    with Image.open(t["image"]) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    largeur, hauteur = (int(t["largeur"]), int(t["hauteur"])) if t.get("largeur") else format_parlante(*image.size)
    # Le pipeline redimensionne l'image sans la recadrer (visage déformé) : recadrée ici au format, puis relue par lui
    reference = Path(t["sortie"]).with_name("reference.png")
    reference.parent.mkdir(parents=True, exist_ok=True)
    ImageOps.fit(image, (largeur, hauteur), Image.LANCZOS).save(reference)
    voix, sr = _son_16k(t["audio"])
    if len(voix) < sr:
        _erreur("le son doit durer au moins une seconde.")
    # Le pipeline avance par fenêtres de 81 images (21 latents, 3,24 s à 25 images/s) : avec moins de son, la boucle
    # ne tourne pas et la vidéo sort vide ; avec exactement une fenêtre (81 à 84 images), elle ne s'arrête jamais
    # (fin atteinte sans passer par arrive_last_frame, lu dans wan_inference_long_pipeline.py, constaté ici). Silence
    # ajouté jusqu'à 85 images (deux fenêtres), images en trop retirées à la fin.
    images_utiles = int(np.ceil(len(voix) / 640))  # 16 000 / 25 = 640 échantillons par image
    voix = np.pad(voix, (0, max(0, IMAGES_MIN_PARLANTE * 640 - len(voix))))

    print(f"PROGRESSION 1/{total} lecture du texte (encodeur umT5)", flush=True)
    cle = ("StableAvatar", str(t.get("poids")), device)
    _ranger()
    _faire_place(11.5, sauf=cle)
    tache_texte = {"prompt": t.get("prompt") or "A person is talking to the camera, natural expressions.",
                   "negatif": NEGATIF_PARLANTE, "depot": t.get("depot_encodeur") or WAN,
                   "sans_encodeur": t.get("sans_encodeur")}
    prompt, negatif = _memoire(_encoder_texte_video)(tache_texte, device, dtype)
    print(f"PROGRESSION 2/{total} chargement de StableAvatar", flush=True)

    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    from wan.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
    from wan.models.wan_image_encoder import CLIPModel
    from wan.models.wan_vae import AutoencoderKLWan
    from wan.pipeline.wan_inference_long_pipeline import WanI2VTalkingInferenceLongPipeline

    def dossier_poids():
        if t.get("poids"):
            return Path(t["poids"])
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(STABLEAVATAR, allow_patterns=SA_FICHIERS))

    def charger():
        racine = dossier_poids()
        base = racine / SA_BASE
        _precharger((STABLEAVATAR, "StableAvatar-1.3B/transformer3d-square.pt"), (STABLEAVATAR, SA_BASE))
        config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        # mêmes réglages que WanTransformer3DFantasyModel.from_pretrained (wan_civitai.yaml : dict_mapping), mais
        # sans lire les poids de base : le fichier de StableAvatar les contient tous
        options = {"in_channels": config["in_dim"], "hidden_size": config["dim"], "patch_size": (1, 2, 2),
                   "qk_norm": True, "window_size": (-1, -1), "cross_attn_norm": True}
        transformeur = WanTransformer3DFantasyModel.from_config(config, **options)
        etat = torch.load(racine / "StableAvatar-1.3B" / "transformer3d-square.pt", map_location="cpu",
                          weights_only=True)
        transformeur.load_state_dict(etat.get("state_dict", etat), strict=True)
        transformeur = transformeur.to(dtype).eval()
        del etat
        vae = AutoencoderKLWan.from_pretrained(str(base / "Wan2.1_VAE.pth"), additional_kwargs={}).eval()
        clip = CLIPModel.from_pretrained(str(base / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")).eval()
        clip.model.textual = None  # partie texte jamais utilisée (seule l'image est lue) : 2 Go de moins
        processeur = Wav2Vec2Processor.from_pretrained(str(racine / "wav2vec2-base-960h"))
        wav2vec = Wav2Vec2Model.from_pretrained(str(racine / "wav2vec2-base-960h")).eval()
        for m in (transformeur, vae, clip, wav2vec):
            _en_memoire_vive(m)
        return transformeur, vae, clip, processeur, wav2vec

    def sur(appareil):
        def deplacer(o):
            for m in (o[0], o[1], o[4]):
                m.to(appareil)
            o[2].to("cpu")  # CLIP : sur la carte seulement le temps de lire l'image (crochet ci-dessous)
        return deplacer

    transformeur, vae, clip, processeur, wav2vec = _garder(cle, charger, 9, vers_cpu=sur("cpu"),
                                                           vers_gpu=sur(device))
    if device == "cuda":
        sur(device)((transformeur, vae, clip, processeur, wav2vec))

    class _TexteLu(torch.nn.Module):
        """Tient la place de l'encodeur de texte : le pipeline n'en lit que le type (bf16)."""
        def __init__(self):
            super().__init__()
            self.temoin = torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)

        @property
        def dtype(self):
            return dtype

    class Pipeline(WanI2VTalkingInferenceLongPipeline):
        def encode_prompt(self, *a, **k):  # plongements de la phase 1 (listes : une entrée par vidéo)
            return [prompt[0].to(device)], [negatif[0].to(device)]

        def progress_bar(self, total=None, **k):  # une ligne PROGRESSION par étape ; CLIP quitte la carte
            clip.to("cpu")
            _vider_cache_cuda()
            moteur = self

            class Suivi(contextlib.AbstractContextManager):
                n = 0

                def update(self_):
                    self_.n += 1
                    print(f"PROGRESSION {self_.n + 2}/{moteur._total} étape {self_.n}/{total}", flush=True)

                def __exit__(self_, *exc):
                    return False
            return Suivi()

    ordonnanceur = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=5.0, use_dynamic_shifting=False, base_shift=0.5, max_shift=1.15,
        base_image_seq_len=256, max_image_seq_len=4096)  # scheduler_kwargs de wan_civitai.yaml
    pipe = Pipeline(tokenizer=None, text_encoder=_TexteLu().to(device), vae=vae, transformer=transformeur,
                    clip_image_encoder=clip, scheduler=ordonnanceur, wav2vec_processor=processeur, wav2vec=wav2vec)
    pipe._total = total
    clip.to(device)
    graine = int(t.get("graine") or 0)
    with torch.no_grad():
        video = _memoire(pipe)(
            prompt=tache_texte["prompt"],  # déjà lu en phase 1 : encode_prompt renvoie les plongements
            num_frames=81, height=hauteur, width=largeur, guidance_scale=6.0,
            generator=torch.Generator(device=device).manual_seed(graine), num_inference_steps=etapes,
            text_guide_scale=float(t.get("guidage_texte", 3.0)), audio_guide_scale=float(t.get("guidage_audio", 5.0)),
            vocal_input_values=voix, motion_frame=25, fps=25, sr=16000, cond_file_path=str(reference), seed=graine,
            overlap_window_length=max(2, int(t.get("chevauchement", 5))), overlapping_weight_scheme="uniform",
            clip_length=81).videos
    print(f"PROGRESSION {total}/{total} enregistrement de la vidéo", flush=True)
    images = (video[0].permute(1, 2, 3, 0).clamp(0, 1).float().numpy() * 255).round().astype(np.uint8)
    images = images[:images_utiles]
    duree = len(images) / 25
    ecrivain = imageio_ffmpeg.write_frames(t["sortie"], (largeur, hauteur), fps=25, codec="libx264",
                                           pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=8, macro_block_size=1,
                                           audio_path=t.get("son") or t["audio"], audio_codec="aac",
                                           output_params=["-t", f"{duree:.3f}"])
    ecrivain.send(None)
    for image_ in images:
        ecrivain.send(np.ascontiguousarray(image_).tobytes())
    ecrivain.close()
    _resultat({"sortie": t["sortie"], "graine": graine, "images": len(images), "duree": round(duree, 2),
               "largeur": largeur, "hauteur": hauteur})
    print(f"TERMINE {t['sortie']}", flush=True)


# --- Whisper : transcription (sous-titres) ------------------------------------------------------------------
def transcrire(chemin_tache):
    """Tâche : {entree: WAV 16 kHz mono (préparé par l'application), langue: code ISO (« fr ») ou None (détectée),
    traduire: bool (texte en anglais)}. Whisper large-v3-turbo par le pipeline de transformers ; au-delà de 30 s,
    transcription longue « séquentielle » (audio entier passé au pipeline, conseillée par la carte du modèle : plus
    juste que le découpage en morceaux). RESULTAT {segments: [{debut, fin, texte}], texte, duree}."""
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    t = _lire(chemin_tache)
    device = _device()
    depot = t.get("depot", WHISPER)
    _hors_ligne_si_present(depot)
    dtype = torch.float16 if device == "cuda" else torch.float32
    print("PROGRESSION 1/2 chargement de Whisper", flush=True)

    def charger():
        _precharger((depot, None))
        modele = AutoModelForSpeechSeq2Seq.from_pretrained(depot, dtype=dtype, low_cpu_mem_usage=True).to(device)
        processeur = AutoProcessor.from_pretrained(depot)
        return pipeline("automatic-speech-recognition", model=modele, tokenizer=processeur.tokenizer,
                        feature_extractor=processeur.feature_extractor, dtype=dtype, device=device)

    asr = _garder(("Whisper", depot, device), charger, 2, vers_cpu=lambda p_: p_.model.to("cpu"),
                  vers_gpu=lambda p_: p_.model.to(device))
    audio, sr = sf.read(t["entree"], dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        _erreur(f"audio à {sr} Hz : 16 000 Hz attendus.")
    duree = len(audio) / sr
    print(f"PROGRESSION 2/2 transcription ({duree:.0f} s d'audio)", flush=True)
    options = {"task": "translate" if t.get("traduire") else "transcribe"}
    if t.get("langue"):
        options["language"] = t["langue"]
    with torch.inference_mode():
        sortie = _memoire(asr)({"raw": np.ascontiguousarray(audio), "sampling_rate": sr}, return_timestamps=True,
                               generate_kwargs=options)
    segments = []
    for morceau in sortie.get("chunks") or []:
        debut, fin = morceau["timestamp"]
        texte = morceau["text"].strip()
        if texte:
            debut = float(debut or 0.0)
            segments.append({"debut": round(debut, 2), "fin": round(float(fin if fin is not None else duree), 2),
                             "texte": texte})
    _resultat({"segments": segments, "texte": sortie.get("text", "").strip(), "duree": round(duree, 2)})
    print("TERMINE -", flush=True)


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
        if repo == "code:stableavatar":
            _telecharger_code_stableavatar()
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
    if "forme3d" in noms or "forme3d_vues" in noms:
        _retirer_ckpt_inutiles()
    print("Modèles prêts.", flush=True)


def _retirer_ckpt_inutiles():
    """Les anciennes installations téléchargeaient aussi les .ckpt de Hunyuan3D (mêmes poids que les .safetensors,
    jamais chargés) : supprimés avec leur blob du cache, s'il y a bien le .safetensors à côté."""
    from huggingface_hub import constants

    libere = 0
    for depot in (HUNYUAN, HUNYUAN_MV):
        racine = Path(constants.HF_HUB_CACHE) / ("models--" + depot.replace("/", "--")) / "snapshots"
        for ckpt in sorted(racine.glob("*/hunyuan3d-*/*.ckpt")):
            if not any(ckpt.parent.glob("*.safetensors")) or "paint" in ckpt.parent.name:
                continue
            try:
                blob = ckpt.resolve() if ckpt.is_symlink() else None
                taille = (blob or ckpt).stat().st_size if (blob or ckpt).exists() else 0
                ckpt.unlink()
                if blob is not None and blob.exists():
                    blob.unlink()
                libere += taille
            except OSError as e:
                print(f"{ckpt.name} non supprimé ({e}).", flush=True)
    if libere:
        print(f"{libere / 1e9:.1f} Go libérés : fichiers .ckpt de Hunyuan3D en double des .safetensors.", flush=True)


ACTIONS = {"decrire": decrire, "bruitage": bruitage, "image": image, "personnage": personnage, "video": video,
           "sprite_dresseur": sprite_dresseur,
           "assembler": assembler, "detourer": detourer, "ameliorer": ameliorer, "forme3d": forme3d, "alleger": alleger,
           "facetter": facetter,
           "transcrire": transcrire, "fluidifier": fluidifier, "parler": parler}


def _present_localement(nom):
    """Le modèle est entièrement dans le cache (sans aucune requête réseau) : chaque motif de téléchargement a au
    moins un fichier dans un snapshot du dépôt ; fichiers GitHub des photos ; code de StableAvatar ; u2net."""
    import glob

    if nom == "detourage":
        dossier = Path(os.environ.get("U2NET_HOME") or Path.home() / ".u2net")
        return (dossier / "u2net.onnx").is_file()
    from huggingface_hub import constants

    hub = Path(constants.HF_HUB_CACHE)
    for repo, motifs in MODELES[nom]:
        if repo.startswith("local:"):
            if not all((_dossier_photos() / f).is_file() for f in motifs):
                return False
            continue
        if repo == "code:stableavatar":
            if not (_code_stableavatar() / "wan" / "pipeline" / "wan_inference_long_pipeline.py").is_file():
                return False
            continue
        racine = hub / ("models--" + repo.replace("/", "--"))
        snaps = [d for d in (racine / "snapshots").glob("*") if d.is_dir()]
        # fichier « .incomplete » : téléchargement interrompu (huggingface_hub le reprendra là où il s'était arrêté)
        if not snaps or any((racine / "blobs").glob("*.incomplete")):
            return False
        for motif in motifs or ["*"]:
            if not any(glob.glob(str(d / motif)) for d in snaps):
                return False
    return True


def manquants(noms):
    """Modèles absents ou incomplets parmi `noms`, sans réseau (installateur : refait le téléchargement de ce qui
    manque à chaque mise à jour, pas seulement quand la liste des modèles change). Ligne « MANQUANTS: a b c »."""
    absents = [n for n in noms if (n not in MODELES and n != "detourage") or not _present_localement(n)]
    print("MANQUANTS: " + " ".join(absents), flush=True)


def principal(argv):
    if len(argv) >= 1 and argv[0] == "telecharger":
        telecharger(argv[1:])
    elif len(argv) >= 1 and argv[0] == "manquants":
        manquants(argv[1:])
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
            nom_fichier = str(getattr(e, "filename", "") or "")
            if isinstance(e, FileNotFoundError) and nom_fichier and "models--" not in nom_fichier:
                raise  # fichier de l'utilisateur (image, son), pas du cache des modèles : réessayer en ligne n'y fait rien
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
