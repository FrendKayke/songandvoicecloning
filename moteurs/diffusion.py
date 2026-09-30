"""Moteur « diffusion » — script exécuté DANS l'environnement diffusion (jamais importé par l'application).

Quatre modèles qui partagent la même pile (diffusers, transformers) :
  - Qwen3-VL-2B (Apache 2.0)      : décrit une image, reformule un texte français en prompt anglais ;
  - Stable Audio Open 1.0          : bruitages à partir d'une description (licence Stability Community,
                                     accès soumis à l'acceptation de la licence : jeton Hugging Face) ;
  - Hunyuan3D-2 (Tencent)          : image → forme 3D (turbo) puis texture (paint turbo + delight) ;
  - Stable Diffusion XL            : texte → image d'objet, point de départ du texte → 3D.

    python diffusion.py <action> <tache.json>      action : decrire | bruitage | image | forme3d
    python diffusion.py telecharger [modele…]      qwen | bruitages | forme3d | texture3d | image | detourage

Sortie : « PROGRESSION i/n … », « RESULTAT <json> », « ERREUR : message » et « TERMINE <fichier> ».
API vérifiées dans les dépôts : diffusers 0.40 (StableAudioPipeline.__call__, StableDiffusionXLPipeline),
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
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
HUNYUAN = "tencent/Hunyuan3D-2"
HUNYUAN_FORME = "hunyuan3d-dit-v2-0-turbo"
HUNYUAN_TEXTURE = "hunyuan3d-paint-v2-0-turbo"
MODELES = {
    "qwen": (QWEN, None),
    "bruitages": (STABLE_AUDIO, None),
    "image": (SDXL, ["*.json", "*.txt", "text_encoder/*.fp16.safetensors", "text_encoder_2/*.fp16.safetensors",
                     "unet/*.fp16.safetensors", "vae/*.fp16.safetensors", "tokenizer*/*"]),
    "forme3d": (HUNYUAN, [f"{HUNYUAN_FORME}/*", "hunyuan3d-vae-v2-0-turbo/*"]),
    "texture3d": (HUNYUAN, [f"{HUNYUAN_TEXTURE}/*", "hunyuan3d-delight-v2-0/*"]),
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
            _erreur("mémoire de la carte graphique insuffisante. Ferme la fenêtre « ACE-Step - ne pas fermer » "
                    "et les autres programmes qui utilisent la carte, puis relance.", 3)
    return enveloppe


def _lire(chemin):
    return json.loads(Path(chemin).read_text(encoding="utf-8"))


def _hors_ligne_si_present(repo, motifs=None):
    """Modèle déjà téléchargé : on évite toute requête réseau (et la question du jeton)."""
    from huggingface_hub import scan_cache_dir

    try:
        for r in scan_cache_dir().repos:
            if r.repo_id == repo and r.revisions:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                return
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
}


def decrire(chemin_tache):
    """Tâche : {mode: son|objet|bruitage|image, image?: chemin, texte?: str}. Renvoie RESULTAT {"texte": …}."""
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
    if mode in ("objet", "bruitage"):
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


# --- Stable Diffusion XL : image d'un objet ------------------------------------------------------------
NEGATIF_IMAGE = "multiple objects, scene, background, text, watermark, blurry, cropped, low quality, deformed"


def image(chemin_tache):
    """Tâche : {prompt, negatif?, graine, etapes, sortie}. Image 1024×1024 d'un objet sur fond blanc."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    t = _lire(chemin_tache)
    device = _device()
    _hors_ligne_si_present(SDXL)
    print("PROGRESSION 1/2 chargement de Stable Diffusion XL", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL, torch_dtype=torch.float16 if device == "cuda" else torch.float32, variant="fp16",
        use_safetensors=True)
    if device == "cuda":
        pipe.enable_model_cpu_offload()  # 12 Go : l'UNet, les encodeurs et le VAE passent tour à tour sur la carte
    else:
        pipe = pipe.to(device)
    graine = int(t.get("graine") or 0) or int(torch.randint(1, 2**31 - 1, (1,)))
    taille = int(t.get("taille", 1024))
    print("PROGRESSION 2/2 génération de l'image", flush=True)
    im = _memoire(pipe)(
        prompt=t["prompt"] + ", single object, centered, plain white background, studio lighting, high detail",
        negative_prompt=t.get("negatif") or NEGATIF_IMAGE,
        num_inference_steps=int(t.get("etapes", 30)), guidance_scale=7.0, width=taille, height=taille,
        generator=torch.Generator(device).manual_seed(graine),
    ).images[0]
    im.save(t["sortie"])
    _resultat({"fichier": t["sortie"], "graine": graine})
    print(f"TERMINE {t['sortie']}", flush=True)


# --- Hunyuan3D-2 : image → forme → texture -------------------------------------------------------------
def forme3d(chemin_tache):
    """Tâche : {image, dossier, etapes, octree, faces, graine, texture: bool, formats: ["glb", "obj"]}.
    Écrit dossier/forme.glb (blanc) et, si texture, dossier/modele.glb (texturé) ; avec « obj », le modèle final
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
    for n, nom in enumerate(noms, 1):
        repo, motifs = MODELES[nom]
        print(f"PROGRESSION {n}/{len(noms)} téléchargement : {repo}", flush=True)
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
    actions = {"decrire": decrire, "bruitage": bruitage, "image": image, "forme3d": forme3d}
    if len(sys.argv) >= 2 and sys.argv[1] == "telecharger":
        telecharger(sys.argv[2:])
    elif len(sys.argv) == 3 and sys.argv[1] in actions:
        actions[sys.argv[1]](sys.argv[2])
    else:
        _erreur("usage : diffusion.py decrire|bruitage|image|forme3d <tache.json> | telecharger [modèle…]")
