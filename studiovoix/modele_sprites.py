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
LEGENDE = "{declencheur}, walking sprite sheet, {cols} by {rows} grid, {description}, on a magenta background"
IMAGES = (".png", ".gif", ".webp", ".bmp", ".jpg", ".jpeg")  # lues par Pillow ; JPEG déconseillé (artefacts)
ARCHIVES = (".zip",)
FEUILLES_MAX = 300


def _legende_texte(description, grille=(4, 4)):
    return LEGENDE.format(declencheur=DECLENCHEUR, cols=grille[0], rows=grille[1], description=description)
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


def _legende(image, progress, grille=(4, 4)):
    """Description du personnage par Qwen3-VL (mode « feuille »), sinon une légende générique."""
    if diffusion.present("qwen"):
        try:
            texte = diffusion.decrire("feuille", "", image=image, progress=progress).strip().rstrip(".")
            if texte:
                return _legende_texte(texte, grille)
        except gr.Error as e:
            print(f"Légende impossible ({e}) : légende générique.", flush=True)
    return _legende_texte("a trainer", grille)


# --- Guide et vérification des fichiers (affichés dans l'onglet) ----------------------------------------------
def fichiers_feuilles(fichiers):
    """Fichiers déposés → images : une archive .zip est ouverte (ses images, à n'importe quelle profondeur, sous des
    noms sans dossier) dans data/_tmp/feuilles_<horodatage>/. Les autres fichiers sont gardés tels quels (refusés
    ensuite avec un message)."""
    import zipfile

    sortie, dossier = [], None
    for f in fichiers or []:
        f = Path(getattr(f, "name", f))
        if f.suffix.lower() not in ARCHIVES:
            sortie.append(f)
            continue
        if dossier is None:
            dossier = nouveau_dossier(cfg.DATA_DIR / "_tmp", "feuilles")
        try:
            with zipfile.ZipFile(f) as z:
                for info in z.infolist():
                    nom = Path(info.filename).name  # jamais de chemin de l'archive : pas d'écriture hors du dossier
                    if info.is_dir() or not nom or nom.startswith(".") or Path(nom).suffix.lower() not in IMAGES:
                        continue
                    cible = dossier / nom
                    k = 2
                    while cible.exists():
                        cible = dossier / f"{Path(nom).stem}_{k}{Path(nom).suffix}"
                        k += 1
                    with z.open(info) as src, open(cible, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    sortie.append(cible)
        except zipfile.BadZipFile:
            sortie.append(f)  # signalé comme illisible
    return sortie


def attendu(style):
    """Ce qu'une feuille de marche doit être pour ce style : (colonnes, lignes, cw, ch, largeur, hauteur, sens)."""
    from . import dresseur

    st = dresseur.STYLES[style]["marche"]
    (cols, rows), (cw, ch) = st["grille"], st["cellule"]
    return cols, rows, cw, ch, cols * cw, rows * ch, st["lignes"]


def guide(style):
    """Marche à suivre de l'onglet, avec le type de fichier attendu à chaque étape (d'après la grille du style)."""
    from . import dresseur

    if style not in dresseur.STYLES:
        style = next(iter(dresseur.STYLES))
    cols, rows, cw, ch, w, h, sens = attendu(style)
    return f"""### Marche à suivre

**Étape 1 — Tes feuilles de marche** (champ « Feuilles de marche d'exemple »)
- **Type de fichier** : images **PNG** (GIF, WebP et BMP acceptés ; évite le JPEG, sa compression brouille les pixels), ou une **archive .zip** qui les contient (les sous-dossiers sont lus). Une archive .7z ou .rar : décompresse-la d'abord, puis dépose les images ou refais un .zip.
- **Une feuille = un seul personnage**, toutes ses poses de marche dans une grille, **sans marge, sans texte, sans autre personnage**.
- **Grille du style « {style} »** : {cols} colonnes × {rows} lignes de cases de {cw}×{ch} pixels, soit **{w}×{h} pixels** de jeu. La feuille peut être agrandie d'un nombre entier de fois : {w * 2}×{h * 2} (×2), {w * 3}×{h * 3} (×3)… Les lignes vont dans l'ordre : {", ".join(sens)}.
- **Fond** : transparent, ou une couleur unie (retirée automatiquement).
- **Combien** : {FEUILLES_MIN} au minimum, **50 à 100** pour un bon résultat (personnages variés : hommes, femmes, enfants, tenues et couleurs différentes).
- Dès que tu déposes les fichiers, le tableau « Vérification » dit pour chacun s'il est bon, et pourquoi sinon.

**Étape 2 — Préparer les feuilles** (bouton 1️⃣)
- Chaque feuille est ramenée à ses vrais pixels, agrandie ×{FACTEUR} sur un fond magenta (**{w * FACTEUR}×{h * FACTEUR} px**) et légendée par Qwen3-VL.
- **Résultat** : `data\\loras\\<nom>\\jeu\\{REPETITIONS}_{DECLENCHEUR}\\` avec, pour chaque feuille, `001.png` (l'image d'entraînement) et `001.txt` (sa légende, texte en anglais : « {_legende_texte("a boy with brown hair, red cap, blue jacket", (cols, rows))} »).
- **À vérifier** : le tableau « Légendes » sous le bouton. Corrige une légende fausse (couleur, casquette, sexe) directement dans le tableau, puis « 💾 Enregistrer les légendes ». Une bonne légende décrit le personnage, pas la feuille.

**Étape 3 — Entraîner** (bouton 2️⃣)
- Réglages : époques (passages sur toutes les feuilles, chacune vue {REPETITIONS} fois par époque ; {EPOQUES_DEFAUT} conseillées) et dimension du LoRA ({DIM_DEFAUT} conseillée). Compte environ une heure pour 60 feuilles sur une RTX 4070 ; ferme les jeux et logiciels qui utilisent la carte.
- **Résultat** : le fichier **`data\\loras\\<nom>\\<nom>.safetensors`** (le modèle LoRA SDXL), proposé au téléchargement ci-dessous. `modele.json` à côté garde les réglages.

**Étape 4 — S'en servir** (onglet Pixel art, section 🎒 Kit de dresseur)
- Choisis ce modèle dans la liste « Feuille de marche ». La feuille d'un nouveau personnage est générée d'après sa description (photo ou texte), puis remise sur la grille {cols}×{rows} de {cw}×{ch} px et aux couleurs du sprite de face.
- **Résultat** : `marche.png` ({w}×{h} px, fond transparent), sa version agrandie et un GIF par direction, dans le dossier du kit.
"""


def etat_prerequis():
    """Ce qu'il faut d'installé avant d'entraîner (affiché en tête de l'onglet)."""
    lignes = []
    env_ok = Path(cfg.ENTRAINEMENT_PYTHON).exists() and (cfg.SDSCRIPTS_DIR / "sdxl_train_network.py").is_file()
    lignes.append(("✅" if env_ok else "❌") + " Environnement d'entraînement (kohya sd-scripts)"
                  + ("" if env_ok else " : lance **METTRE_A_JOUR.bat** (étape 19 de l'installateur)."))
    sdxl = diffusion.present("sprites_dresseur")
    lignes.append(("✅" if sdxl else "❌") + " SDXL + LoRA de sprites (modèle de base)"
                  + ("" if sdxl else " : Outils → Modèles et diagnostic → Télécharger, ou METTRE_A_JOUR.bat."))
    qwen = diffusion.present("qwen")
    lignes.append(("✅" if qwen else "⚠️") + " Qwen3-VL pour les légendes"
                  + ("" if qwen else " (facultatif : sans lui, légendes génériques, moins bon résultat)."))
    return "**Avant de commencer** : " + " · ".join(lignes)


def verifier_feuilles(fichiers, style):
    """Contrôle immédiat des fichiers déposés. Renvoie (message, lignes [fichier, taille, agrandissement, état])."""
    from . import dresseur

    if style not in dresseur.STYLES:
        raise gr.Error("Choisis un style.")
    cols, rows, cw, ch, w, h, _ = attendu(style)
    images = fichiers_feuilles(fichiers)
    if not images:
        return "Dépose tes feuilles de marche (PNG ou archive .zip).", []
    lignes, bonnes = [], 0
    for f in images[:FEUILLES_MAX]:
        nom = f.name
        if f.suffix.lower() not in IMAGES:
            lignes.append([nom, "", "", f"❌ type de fichier non lu ({f.suffix or 'sans extension'}) : PNG, GIF, "
                                        "WebP, BMP ou archive .zip"])
            continue
        taille, k = "", 1
        try:
            with Image.open(f) as im:
                taille, k = f"{im.width}×{im.height}", max(1, im.width // w)
            a = lire_feuille(f, (cols, rows), (cw, ch))
        except (OSError, ValueError) as e:
            conseil = " (JPEG : sa compression brouille les pixels, exporte en PNG)" if f.suffix.lower() in (
                ".jpg", ".jpeg") else ""
            raison = str(e) if isinstance(e, ValueError) else "image illisible"
            lignes.append([nom, taille, "", f"❌ {raison}{conseil}"])
            continue
        if (a[..., 3] > 0).sum() < 0.05 * a.shape[0] * a.shape[1]:
            lignes.append([nom, taille, f"×{k}", "❌ presque vide (fond seul ?)"])
            continue
        bonnes += 1
        lignes.append([nom, taille, f"×{k}", "✅ bonne"])
    msg = f"**{bonnes} feuille(s) bonne(s) sur {len(images)}** (grille attendue : {cols}×{rows} cases de {cw}×{ch} px = {w}×{h} px, ou agrandie)."
    if len(images) > FEUILLES_MAX:
        msg += f" Seules les {FEUILLES_MAX} premières sont vérifiées."
    if bonnes < FEUILLES_MIN:
        msg += f" Il en faut au moins {FEUILLES_MIN}."
    elif bonnes < 40:
        msg += " C'est assez pour essayer ; 50 à 100 donnent un meilleur résultat."
    return msg, lignes


def legendes(nom):
    """Légendes du jeu préparé : lignes [image, légende]."""
    jeu = dossier_lora(_slug(nom, "sprites")) / "jeu" / f"{REPETITIONS}_{DECLENCHEUR}"
    return [[png.name, png.with_suffix(".txt").read_text(encoding="utf-8") if png.with_suffix(".txt").exists() else ""]
            for png in sorted(jeu.glob("*.png"))]


def enregistrer_legendes(nom, lignes):
    """Légendes corrigées dans le tableau → fichiers .txt du jeu (le déclencheur est remis s'il a été effacé)."""
    jeu = dossier_lora(_slug(nom, "sprites")) / "jeu" / f"{REPETITIONS}_{DECLENCHEUR}"
    if not jeu.is_dir():
        raise gr.Error("Prépare d'abord les feuilles (bouton 1️⃣).")
    if hasattr(lignes, "values"):  # pandas.DataFrame de gr.Dataframe
        lignes = lignes.values.tolist()
    n = 0
    for image, texte in (tuple(l[:2]) for l in lignes or [] if len(l) >= 2):
        png = jeu / Path(str(image)).name
        texte = " ".join(str(texte or "").split())
        if not png.is_file() or not texte:
            continue
        if not texte.startswith(DECLENCHEUR):
            texte = f"{DECLENCHEUR}, {texte}"
        png.with_suffix(".txt").write_text(texte, encoding="utf-8")
        n += 1
    return f"✅ {n} légende(s) enregistrée(s) dans `{jeu}`."


def preparer(feuilles, nom, style, legender=True, progress=gr.Progress()):
    """Feuilles de marche → data/loras/<nom>/jeu/<REPETITIONS>_pixelsheet/NNN.png + NNN.txt (légendes).
    Renvoie (message, dossier)."""
    from . import dresseur

    feuilles = [str(f) for f in fichiers_feuilles(feuilles)]
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
        legende = _legende(chemin, progress, grille) if legender else _legende_texte("a trainer", grille)
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
def prompt_feuille(description, grille=(4, 4)):
    return _legende_texte(description.strip().rstrip("."), grille)


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
    prompt = prompt_feuille(description, grille)
    brut = Path(sortie).with_name(Path(sortie).stem + "_brut.png")
    diffusion.sprite_dresseur(prompt, [brut], [int(graine)], cols * cw * FACTEUR, rows * ch * FACTEUR, etapes=etapes,
                              lora=str(fichier), progress=progress)
    image = Image.open(brut)
    if palette is None:  # pas de sprite de face : palette lue sur l'image elle-même, hors fond
        a = np.array(image.convert("RGB")).astype(int)
        hors_fond = np.sqrt(((a - np.array(FOND)) ** 2).sum(-1)) > 120
        palette = px._palette(a[hors_fond].astype(np.uint8), couleurs)
    return vers_feuille(image, grille, cellule, palette), prompt
