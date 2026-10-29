"""Entraînement d'un modèle de feuilles de sprites (LoRA SDXL, kohya sd-scripts) à partir des feuilles de marche de
l'utilisateur, puis génération d'une feuille de marche d'un nouveau personnage avec ce modèle (kit de dresseur).

Aucun générateur installé ne dessine une feuille de marche cohérente (16 poses de 16×24 px) : on apprend donc à SDXL
« une feuille de marche de ce jeu » comme une seule image (la feuille entière, agrandie ×8 sur un fond magenta,
légendée par Qwen3-VL : « pixelsheet, walking sprite sheet, 4 by 4 grid, a man with brown hair… »). Une feuille
générée est ensuite ramenée à la grille du jeu (couleur dominante par pixel, palette du sprite de face, fond retiré).

Environnement dédié (<lecteur>:\\StudioVoix\\entrainement\\.venv, Python 3.10, PyTorch 2.6 CUDA 12.4, kohya sd-scripts au
commit épinglé dans installer.ps1), appelé en sous-processus : moteurs/lora_sdxl.py, lancé depuis le dossier de
sd-scripts. LoRA SDXL en 8 à 10 Go de mémoire graphique (U-Net seul, latents et encodeurs de texte en cache, gradient
checkpointing, AdamW 8 bits : doc de kohya). Rangement : data/loras/<nom>/ (jeu/, <nom>.safetensors, modele.json).
"""
import json
import shutil
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from . import config as cfg
from . import diffusion
from . import pixel_art as px
from . import retraits
from . import serveur_acestep
from .outils import lancer_moteur, nouveau_dossier
from .serie import _slug

FOND = (255, 0, 255)  # magenta : jamais dans un sprite, retiré sans erreur après la génération
DECLENCHEUR = "pixelsheet"
FACTEUR = 8  # pixels de l'image d'entraînement par pixel du jeu (64×96 → 512×768)
REPETITIONS = 10  # passages par feuille et par époque (dossier « 10_pixelsheet » de kohya)
FEUILLES_MIN = 8
LEGENDE = "{declencheur}, walking sprite sheet, 4 by 4 grid, {description}, on a magenta background"
EPOQUES_DEFAUT = 12
DIM_DEFAUT = 32


def script() -> Path:
    return cfg.MOTEURS_DIR / "lora_sdxl.py"


def dossier_lora(nom) -> Path:
    return cfg.LORAS_DIR / nom


def fichier_lora(nom) -> Path:
    return dossier_lora(nom) / f"{nom}.safetensors"


def loras():
    """Modèles entraînés prêts (nom → fichier), du plus récent au plus ancien."""
    if not cfg.LORAS_DIR.is_dir():
        return {}
    prets = [d for d in cfg.LORAS_DIR.iterdir() if d.is_dir() and fichier_lora(d.name).is_file()]
    return {d.name: fichier_lora(d.name) for d in sorted(prets, key=lambda d: -fichier_lora(d.name).stat().st_mtime)}


def _installe():
    if not Path(cfg.ENTRAINEMENT_PYTHON).exists() or not (cfg.SDSCRIPTS_DIR / "sdxl_train_network.py").is_file():
        raise gr.Error(f"L'environnement d'entraînement (kohya sd-scripts) n'est pas installé "
                       f"({cfg.ENTRAINEMENT_PYTHON} introuvable). {retraits.conseil('entrainement')}")


# --- Préparation des feuilles ----------------------------------------------------------------------------------
def lire_feuille(chemin, grille, cellule):
    """Feuille de marche → tableau RGBA à 1 pixel par pixel du jeu, fond transparent ; erreur si la grille ne
    correspond pas (cols × rows cases de cw × ch)."""
    with Image.open(chemin) as im:
        im = ImageOps.exif_transpose(im).convert("RGBA")
    if np.array(im)[..., 3].min() == 255:
        im = px._sans_fond_uni(im)
    a, _, _ = px.reduire(np.array(im))
    (cols, rows), (cw, ch) = grille, cellule
    if a.shape[1] != cols * cw or a.shape[0] != rows * ch:
        raise ValueError(f"{a.shape[1]}×{a.shape[0]} pixels de jeu au lieu de {cols * cw}×{rows * ch} "
                         f"({cols}×{rows} cases de {cw}×{ch})")
    return np.ascontiguousarray(a)


def image_entrainement(a, facteur=FACTEUR, fond=FOND):
    """Feuille (pixels du jeu) → image RGB sur le fond magenta, agrandie ×facteur au plus proche."""
    rgb = np.where(a[..., 3:] >= 128, a[..., :3], np.array(fond, np.uint8))
    im = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
    return im.resize((im.width * facteur, im.height * facteur), Image.Resampling.NEAREST)


def _legende(image, progress):
    """Description du personnage par Qwen3-VL (mode « feuille »), sinon une légende générique."""
    if diffusion.present("qwen"):
        try:
            texte = diffusion.decrire("feuille", "", image=image, progress=progress).strip().rstrip(".")
            if texte:
                return LEGENDE.format(declencheur=DECLENCHEUR, description=texte)
        except gr.Error as e:
            print(f"Légende impossible ({e}) : légende générique.", flush=True)
    return LEGENDE.format(declencheur=DECLENCHEUR, description="a trainer")


def preparer(feuilles, nom, style, legender=True, progress=gr.Progress()):
    """Feuilles de marche → data/loras/<nom>/jeu/<REPETITIONS>_pixelsheet/NNN.png + NNN.txt (légendes).
    Renvoie (message, dossier)."""
    from . import dresseur

    feuilles = [getattr(f, "name", f) for f in (feuilles or [])]
    if len(feuilles) < FEUILLES_MIN:
        raise gr.Error(f"Il faut au moins {FEUILLES_MIN} feuilles de marche (tu en as donné {len(feuilles)}) ; "
                       "une soixantaine donne un bon résultat.")
    if style not in dresseur.STYLES:
        raise gr.Error("Choisis un style.")
    nom = _slug(nom, "sprites")
    st = dresseur.STYLES[style]
    grille, cellule = st["marche"]["grille"], st["marche"]["cellule"]
    dossier = dossier_lora(nom)
    jeu = dossier / "jeu" / f"{REPETITIONS}_{DECLENCHEUR}"
    if jeu.exists():
        shutil.rmtree(jeu)
    jeu.mkdir(parents=True)
    gardees, erreurs = [], []
    for n, f in enumerate(feuilles, 1):
        progress((n - 1) / len(feuilles), desc=f"Feuille {n}/{len(feuilles)}…")
        try:
            a = lire_feuille(f, grille, cellule)
        except (OSError, ValueError) as e:
            erreurs.append(f"{Path(f).name} : {e}")
            continue
        image = image_entrainement(a)
        chemin = jeu / f"{len(gardees) + 1:03d}.png"
        image.save(chemin)
        legende = _legende(chemin, progress) if legender else LEGENDE.format(declencheur=DECLENCHEUR,
                                                                              description="a trainer")
        chemin.with_suffix(".txt").write_text(legende, encoding="utf-8")
        gardees.append({"source": Path(f).name, "image": chemin.name, "legende": legende})
    if len(gardees) < FEUILLES_MIN:
        raise gr.Error(f"Seulement {len(gardees)} feuille(s) utilisable(s) : " + " ; ".join(erreurs[:5]))
    infos = {"nom": nom, "style": style, "grille": list(grille), "cellule": list(cellule), "facteur": FACTEUR,
             "fond": list(FOND), "declencheur": DECLENCHEUR, "feuilles": gardees, "erreurs": erreurs}
    (dossier / "modele.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    msg = (f"✅ {len(gardees)} feuille(s) préparée(s) dans `{jeu}` ({image.width}×{image.height} px, légendes "
           f"{'par Qwen3-VL' if legender and diffusion.present('qwen') else 'génériques'}). Exemple : "
           f"« {gardees[0]['legende']} ».")
    if erreurs:
        msg += f"\n\n⚠️ {len(erreurs)} écartée(s) : " + " ; ".join(erreurs[:8])
    return msg, str(dossier)


# --- Entraînement ----------------------------------------------------------------------------------------------
def entrainer(nom, epoques=EPOQUES_DEFAUT, dim=DIM_DEFAUT, graine=0, progress=gr.Progress()):
    """Lance kohya sd-scripts sur data/loras/<nom>/jeu. Renvoie (message, fichier du LoRA)."""
    nom = _slug(nom, "sprites")
    dossier = dossier_lora(nom)
    jeu = dossier / "jeu"
    if not (dossier / "modele.json").is_file() or not any(jeu.rglob("*.png")):
        raise gr.Error(f"Prépare d'abord les feuilles du modèle « {nom} » (bouton « Préparer les feuilles »).")
    _installe()
    diffusion.verifier("sprites_dresseur")  # SDXL de base
    sdxl = diffusion.ckpt_dir("sprites_dresseur") / "snapshots"
    snapshot = next((d for d in sorted(sdxl.glob("*")) if (d / "model_index.json").is_file()), None)
    if snapshot is None:
        raise gr.Error("SDXL incomplet (model_index.json absent) : retélécharge « SDXL + LoRA de sprites ».")
    infos = json.loads((dossier / "modele.json").read_text(encoding="utf-8"))
    tache = {"sdxl": str(snapshot), "jeu": str(jeu), "sortie": str(dossier), "nom": nom, "epoques": int(epoques),
             "dim": int(dim), "graine": int(graine or 0) or 42, "largeur": infos["grille"][0] * infos["cellule"][0] * FACTEUR,
             "hauteur": infos["grille"][1] * infos["cellule"][1] * FACTEUR}
    fichier = dossier / "tache_entrainement.json"
    fichier.write_text(json.dumps(tache, ensure_ascii=False, indent=1), encoding="utf-8")
    debut = time.monotonic()
    progress(0.01, desc="Entraînement : préparation…")
    serveur_acestep.liberer_gpu(progress)

    def suivi(i, n):
        ecoule = time.monotonic() - debut
        reste = f", ~{(ecoule / i * (n - i)) / 60:.0f} min restantes" if i > 5 else ""
        progress(0.03 + 0.95 * i / n, desc=f"Entraînement : pas {i}/{n}{reste}")

    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"}
    lancer_moteur([cfg.ENTRAINEMENT_PYTHON, str(script()), "entrainer", str(fichier)], cfg.SDSCRIPTS_DIR, env,
                  "Entraînement", suivi, attendu=fichier_lora(nom),
                  annonce=lambda texte: progress(0.02, desc=f"Entraînement : {texte}"))
    infos["entrainement"] = {"epoques": int(epoques), "dim": int(dim), "graine": tache["graine"],
                             "duree_s": round(time.monotonic() - debut)}
    (dossier / "modele.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    return (f"✅ Modèle « {nom} » entraîné en {(time.monotonic() - debut) / 60:.0f} min : `{fichier_lora(nom)}`. "
            "Il est maintenant proposé dans le kit de dresseur (feuille de marche).", str(fichier_lora(nom)))


# --- Feuille de marche générée avec le modèle -----------------------------------------------------------------
def prompt_feuille(description):
    return LEGENDE.format(declencheur=DECLENCHEUR, description=description.strip().rstrip("."))


def vers_feuille(image, grille, cellule, palette, contour=px.CONTOUR_DEFAUT, fond=FOND, facteur=FACTEUR):
    """Image générée (cols·cw·facteur × rows·ch·facteur, fond magenta) → feuille RGBA à la grille du jeu : couleur
    dominante de la palette par pixel, fond (proche du magenta) transparent, contour d'un pixel."""
    (cols, rows), (cw, ch) = grille, cellule
    w, h = cols * cw, rows * ch
    s = px.ECHELLE_TRAVAIL
    fin = np.array(image.convert("RGB").resize((w * s, h * s), Image.Resampling.BOX)).astype(int)
    ecart = np.sqrt(((fin - np.array(fond)) ** 2).sum(-1))
    alpha = np.where(ecart > 120, 255, 0)
    rgba = np.dstack([fin, alpha]).astype(np.uint8)
    feuille = px.blocs_dominants(rgba, np.asarray(palette, np.uint8), h, w)
    return px.contourer(feuille, contour)


def generer_feuille(lora, description, sortie, graine, grille, cellule, palette=None, couleurs=15, etapes=30,
                    progress=None):
    """Feuille de marche d'un personnage avec un LoRA entraîné ici. Renvoie (feuille RGBA numpy, prompt)."""
    fichier = loras().get(lora)
    if not fichier:
        raise gr.Error(f"Modèle de feuilles « {lora} » introuvable : entraîne-le dans « Entraîner un modèle de sprites ».")
    (cols, rows), (cw, ch) = grille, cellule
    prompt = prompt_feuille(description)
    brut = Path(sortie).with_name(Path(sortie).stem + "_brut.png")
    diffusion.sprite_dresseur(prompt, [brut], [int(graine)], cols * cw * FACTEUR, rows * ch * FACTEUR, etapes=etapes,
                              lora=str(fichier), progress=progress)
    image = Image.open(brut)
    if palette is None:  # pas de sprite de face : palette lue sur l'image elle-même, hors fond
        a = np.array(image.convert("RGB")).astype(int)
        hors_fond = np.sqrt(((a - np.array(FOND)) ** 2).sum(-1)) > 120
        palette = px._palette(a[hors_fond].astype(np.uint8), couleurs)
    return vers_feuille(image, grille, cellule, palette), prompt
