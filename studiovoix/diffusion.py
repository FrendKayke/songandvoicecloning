"""Client du moteur « diffusion » (moteurs/diffusion.py) : Qwen3-VL, Stable Audio Open, SDXL, Hunyuan3D-2.

Environnement dédié (<lecteur>:\\StudioVoix\\diffusion\\.venv), appelé en sous-processus. Les modèles vont dans
le cache Hugging Face (HF_HOME, dans StudioVoix). Stable Audio Open exige l'acceptation de sa licence sur
Hugging Face : le jeton est enregistré dans <HF_HOME>\\token, là où huggingface_hub le lit.
"""
import json
import os
from pathlib import Path

import gradio as gr

from . import config as cfg
from .outils import lancer_moteur, stream_command

# nom → (dépôt Hugging Face, libellé, fichiers attendus dans le cache — un par composant essentiel)
MODELES = {
    "qwen": ("Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL-2B (descriptions, traduction)",
             ["model.safetensors"]),
    "bruitages": ("stabilityai/stable-audio-open-1.0", "Stable Audio Open 1.0 (bruitages)",
                  ["transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
                   "text_encoder/model.safetensors"]),
    "image": ("stabilityai/stable-diffusion-xl-base-1.0", "Stable Diffusion XL (texte → image)",
              ["unet/diffusion_pytorch_model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors"]),
    "forme3d": ("tencent/Hunyuan3D-2", "Hunyuan3D-2 forme (image → 3D)",
                ["hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors", "hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors"]),
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


def ckpt_dir(nom) -> Path:
    return hf_home() / "hub" / ("models--" + MODELES[nom][0].replace("/", "--"))


def _present(nom) -> bool:
    snaps = ckpt_dir(nom) / "snapshots"
    return all(any(snaps.glob(f"*/{f}")) for f in MODELES[nom][2])


def missing_components():
    return [MODELES[n][1] for n in MODELES if not _present(n)]


def present(nom) -> bool:
    return _present(nom)


def _env():
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    # rembg range son modèle de détourage dans ~/.u2net sinon (donc sur C:)
    env["U2NET_HOME"] = os.environ.get("U2NET_HOME") or str(cfg.DIFFUSION_DIR / "u2net")
    return env


def _installe():
    if not Path(cfg.DIFFUSION_PYTHON).exists():
        raise gr.Error(f"Le moteur de diffusion n'est pas installé ({cfg.DIFFUSION_PYTHON} introuvable). "
                       "Relance INSTALLER.bat : seules les étapes manquantes seront faites.")


def _verifier(*noms):
    _installe()
    manquants = [MODELES[n][1] for n in noms if not _present(n)]
    if manquants:
        raise gr.Error("Modèle(s) à télécharger dans l'onglet « Modèles » : " + ", ".join(manquants)
                       + (". Pour Stable Audio Open, il faut d'abord un jeton Hugging Face (voir l'onglet Modèles)."
                          if "bruitages" in noms and not _present("bruitages") else "."))


def lancer(action, tache: dict, dossier: Path, nom, progress=None, libelles=None):
    """Écrit tache.json dans dossier, lance l'action et renvoie le dictionnaire de la ligne RESULTAT."""
    dossier.mkdir(parents=True, exist_ok=True)
    fichier = dossier / f"tache_{action}.json"
    fichier.write_text(json.dumps(tache, ensure_ascii=False, indent=1), encoding="utf-8")
    libelles = libelles or {}

    def suivi(i, n):
        if progress:
            progress(0.05 + 0.9 * (i - 1) / n, desc=f"{nom} : {libelles.get(i, f'étape {i}/{n}')}…")

    lignes = lancer_moteur([cfg.DIFFUSION_PYTHON, str(script()), action, str(fichier)], cfg.DIFFUSION_DIR, _env(),
                           nom, suivi)
    resultat = next((l_ for l_ in reversed(lignes) if l_.startswith("RESULTAT ")), None)
    if not resultat:
        raise gr.Error(f"{nom} : pas de résultat renvoyé par le moteur.\n" + "\n".join(lignes[-10:]))
    return json.loads(resultat[len("RESULTAT "):])


# --- Actions ----------------------------------------------------------------------------------------------
def decrire(mode, texte=None, image=None, dossier=None, progress=None):
    """Texte reformulé en anglais (mode « bruitage » ou « objet ») ou description d'une image (« son », « image »)."""
    _verifier("qwen")
    dossier = dossier or (cfg.DATA_DIR / "_tmp")
    tache = {"mode": mode}
    if texte:
        tache["texte"] = texte
    if image:
        tache["image"] = str(image)
    return lancer("decrire", tache, dossier, "Qwen3-VL", progress,
                  {1: "chargement du modèle de description", 2: "description"})["texte"]


def bruitage(prompt, dossier, duree, variantes, graine, etapes=100, negatif=None, progress=None):
    _verifier("bruitages")
    return lancer("bruitage", {"prompt": prompt, "negatif": negatif or None, "duree": float(duree),
                               "variantes": int(variantes), "etapes": int(etapes), "graine": int(graine or 0),
                               "dossier": str(dossier)}, dossier, "Stable Audio", progress,
                  {1: "chargement de Stable Audio Open", 2: "génération du bruitage"})


def image(prompt, sortie, graine, etapes=30, progress=None):
    _verifier("image")
    return lancer("image", {"prompt": prompt, "graine": int(graine or 0), "etapes": int(etapes),
                            "sortie": str(sortie)}, Path(sortie).parent, "Stable Diffusion XL", progress,
                  {1: "chargement de Stable Diffusion XL", 2: "génération de l'image"})


def forme3d(image_path, dossier, etapes, octree, faces, graine, texture, progress=None):
    _verifier("forme3d", *(["texture3d"] if texture else []))
    return lancer("forme3d", {"image": str(image_path), "dossier": str(dossier), "etapes": int(etapes),
                              "octree": int(octree), "faces": int(faces), "graine": int(graine or 0),
                              "texture": bool(texture)}, dossier, "Hunyuan3D", progress,
                  {1: "détourage de l'image", 2: "chargement du générateur de forme", 3: "génération de la forme",
                   4: "nettoyage et simplification", 5: "chargement du peintre de texture",
                   6: "peinture de la texture (plusieurs minutes)"})


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
        yield f"❌ Python du moteur de diffusion introuvable : {cfg.DIFFUSION_PYTHON}. Relance INSTALLER.bat."
        return
    if "bruitages" in noms and not jeton_present():
        yield (f"❌ Stable Audio Open demande d'accepter sa licence sur {LICENCE_BRUITAGES} puis d'enregistrer "
               f"un jeton Hugging Face (créé sur {JETONS}) dans le champ ci-dessus.")
        return
    yield from stream_command([cfg.DIFFUSION_PYTHON, str(script()), "telecharger", *noms, "detourage"],
                              cfg.DIFFUSION_DIR, f"Téléchargement : {', '.join(MODELES[n][1] for n in noms)} …",
                              _env())
