"""Moteur « diffusion » — script exécuté DANS l'environnement diffusion (jamais importé par l'application).

Quatre modèles qui partagent la même pile (diffusers, transformers) :
  - Qwen3-VL-2B (Apache 2.0)      : décrit une image, reformule un texte français en prompt anglais ;
  - Stable Audio Open 1.0          : bruitages à partir d'une description (licence Stability Community,
                                     accès soumis à l'acceptation de la licence : jeton Hugging Face) ;
  - Hunyuan3D-2 (Tencent)          : image → forme 3D (turbo) puis texture (paint turbo + delight) ;
  - Z-Image-Turbo (Apache 2.0)     : texte → image (illustrations de cartes, objet du texte → 3D) ;
  - FLUX.2 klein 4B (Apache 2.0)   : image d'un personnage à partir de 1 à 4 images de référence (même personnage
                                     d'une carte à l'autre).

    python diffusion.py <action> <tache.json>      action : decrire | bruitage | image | personnage | forme3d | alleger
    python diffusion.py telecharger [modele…]      qwen | bruitages | forme3d | texture3d | zimage | personnages | detourage

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

QWEN = "Qwen/Qwen3-VL-2B-Instruct"
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
    "forme3d": [(HUNYUAN, [f"{HUNYUAN_FORME}/*", "hunyuan3d-vae-v2-0-turbo/*"])],
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


def _lire(chemin):
    return json.loads(Path(chemin).read_text(encoding="utf-8"))


def _hors_ligne_si_present(*repos):
    """Modèles déjà téléchargés (tous les dépôts nommés) : on évite toute requête réseau (et la question du jeton)."""
    from huggingface_hub import scan_cache_dir

    try:
        presents = {r.repo_id for r in scan_cache_dir().repos if r.revisions}
        if set(repos) <= presents:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
    except Exception:
        pass


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
    "carte": ("Rewrite the following description into an English prompt for an image generator that will paint "
              "ONE illustration for a fantasy trading card: main subject, pose or action, setting, lighting, colors "
              "and mood, composition centered on the subject. Do not mention any art style, artist or existing "
              "work. No text, no letters, no card frame, no border, no user interface. Answer with the prompt "
              "only:\n\n"),
}


def decrire(chemin_tache):
    """Tâche : {mode: son|objet|bruitage|image|carte, image?: chemin, texte?: str}. Renvoie RESULTAT {"texte": …}."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    t = _lire(chemin_tache)
    mode = t.get("mode")
    if mode not in CONSIGNES:
        _erreur(f"mode de description inconnu : {mode}")
    device = _device()
    _hors_ligne_si_present(QWEN)
    print("PROGRESSION 1/2 chargement de Qwen3-VL", flush=True)
    modele = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map=device)
    processeur = AutoProcessor.from_pretrained(QWEN)
    contenu = []
    if t.get("image"):
        contenu.append({"type": "image", "image": Image.open(t["image"]).convert("RGB")})
    consigne = CONSIGNES[mode]
    if mode in ("objet", "bruitage", "carte"):
        consigne += (t.get("texte") or "").strip()
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
    _hors_ligne_si_present(STABLE_AUDIO)
    try:
        pipe = StableAudioPipeline.from_pretrained(
            t.get("depot", STABLE_AUDIO), torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    except Exception as e:  # accès refusé : licence non acceptée ou jeton absent
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower() or "Access" in str(e):
            _erreur("Stable Audio Open n'est pas téléchargé : accepte sa licence sur Hugging Face et enregistre "
                    "ton jeton (onglet Modèles → Télécharger les bruitages).", 4)
        raise
    pipe = pipe.to(device)
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
    gguf = t.get("gguf") or hf_hub_download(*ZIMAGE_GGUF)
    transformeur = ZImageTransformer2DModel.from_single_file(
        gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), config=depot, subfolder="transformer",
        torch_dtype=dtype)
    sans_encodeur = bool(t.get("sans_encodeur"))
    pipe = ZImagePipeline.from_pretrained(depot, transformer=transformeur, torch_dtype=dtype,
                                          **({"text_encoder": None, "tokenizer": None} if sans_encodeur else {}))
    if device == "cuda":
        pipe.enable_model_cpu_offload()  # encodeur de texte (8 Go) puis transformeur (7 Go) passent tour à tour
    else:
        pipe = pipe.to(device)
    options = {"prompt": t["prompt"]}
    if sans_encodeur:  # test sans les 8 Go de l'encodeur : plongements de la bonne dimension
        options = {"prompt_embeds": [torch.randn(24, pipe.transformer.config.cap_feat_dim, dtype=dtype)]}
    _generer(pipe, t, dict(options, guidance_scale=0.0), etapes=9)


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
    gguf = t.get("gguf") or hf_hub_download(*KLEIN_GGUF)
    # config= obligatoire : sans lui, diffusers reconnaît « flux-2-dev » et lit la configuration de FLUX.2-dev
    # (dépôt soumis à licence non commerciale)
    transformeur = Flux2Transformer2DModel.from_single_file(
        gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), config=depot, subfolder="transformer",
        torch_dtype=dtype)
    sans_encodeur = bool(t.get("sans_encodeur"))
    encodeur = tokeniseur = None
    if not sans_encodeur:
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        encodeur = Qwen3ForCausalLM.from_pretrained(depot_encodeur, subfolder="text_encoder", torch_dtype=dtype)
        tokeniseur = AutoTokenizer.from_pretrained(depot_encodeur, subfolder="tokenizer")
    # pipeline assemblée composant par composant : from_pretrained voudrait aussi les 16 Go de poids officiels
    pipe = Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(depot, subfolder="scheduler"),
        vae=AutoencoderKLFlux2.from_pretrained(depot, subfolder="vae", torch_dtype=dtype),
        text_encoder=encodeur, tokenizer=tokeniseur, transformer=transformeur, is_distilled=True)
    if device == "cuda":
        pipe.enable_model_cpu_offload()  # encodeur de texte (8 Go) puis transformeur (4 Go) passent tour à tour
    else:
        pipe = pipe.to(device)
    options = {"prompt": t["prompt"]}
    if sans_encodeur:  # 3 couches cachées de l'encodeur mises bout à bout
        options = {"prompt_embeds": torch.randn(1, 24, pipe.transformer.config.joint_attention_dim, dtype=dtype)}
    print(f"{len(references)} image(s) de référence du personnage.", flush=True)
    _generer(pipe, t, dict(options, image=references, guidance_scale=1.0), etapes=4)


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
    im = remove(im, session=new_session("u2net"), bgcolor=[255, 255, 255, 0])
    im.save(dossier / "image_detouree.png")

    print(f"PROGRESSION 2/{total} chargement du générateur de forme", flush=True)
    _hors_ligne_si_present(HUNYUAN)
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        t.get("depot", HUNYUAN), subfolder=t.get("sous_dossier", HUNYUAN_FORME), device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32)
    if device == "cuda":
        pipe.enable_flashvdm(mc_algo="mc")  # plus rapide (README : --enable_flashvdm), extraction sans diso
    print(f"PROGRESSION 3/{total} génération de la forme", flush=True)
    sorties = _memoire(pipe)(
        image=im, num_inference_steps=int(t.get("etapes", 30)), guidance_scale=float(t.get("guidage", 7.5)),
        generator=torch.Generator().manual_seed(graine), octree_resolution=int(t.get("octree", 256)),
        num_chunks=int(t.get("morceaux", 20000)), output_type="mesh",
    )
    mesh = export_to_trimesh(sorties)[0]
    _liberer(pipe)

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

    peintre = Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN, subfolder=HUNYUAN_TEXTURE)
    peintre.enable_model_cpu_offload()  # mode basse mémoire de gradio_app.py (--low_vram_mode)
    print(f"PROGRESSION 6/{total} peinture de la texture", flush=True)
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
        try:
            snapshot_download(repo, allow_patterns=motifs)
        except Exception as e:
            if nom == "bruitages" and any(k in str(e) for k in ("401", "403", "gated", "Access")):
                _erreur("accès refusé à Stable Audio Open : accepte la licence sur "
                        "https://huggingface.co/stabilityai/stable-audio-open-1.0 puis enregistre un jeton "
                        "Hugging Face (voir README).", 4)
            raise
    print("Modèles prêts.", flush=True)


if __name__ == "__main__":
    actions = {"decrire": decrire, "bruitage": bruitage, "image": image, "personnage": personnage, "forme3d": forme3d,
               "alleger": alleger}
    if len(sys.argv) >= 2 and sys.argv[1] == "telecharger":
        telecharger(sys.argv[2:])
    elif len(sys.argv) == 3 and sys.argv[1] in actions:
        actions[sys.argv[1]](sys.argv[2])
    else:
        _erreur("usage : diffusion.py decrire|bruitage|image|personnage|forme3d|alleger <tache.json> | telecharger [modèle…]")
