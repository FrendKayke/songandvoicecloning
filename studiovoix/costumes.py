"""Onglet « Changer les vêtements » : des photos (un bébé, un groupe…) habillées avec la tenue d'une autre image
(un costume de personnage d'animé, par exemple), le reste de la photo gardé.

Qwen3-VL décrit la tenue de l'image du costume (mode « costume », sans le personnage ni le style ; « tenue » si la
tenue est seulement décrite en français), puis FLUX.2 klein 4B (Apache 2.0) reçoit, pour chaque photo, la photo
(image 1) et l'image du costume (image 2) avec la consigne de ne changer que les vêtements. Résultats à ~1 Mpx
(proportions de la photo). data/costumes/<horodatage>_<nom>/ : costume.<ext>, originales/, resultats/, lot.json,
<nom>.zip ; hors galerie.
"""
import json
import shutil
import time
import zipfile
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import diffusion, images
from .outils import nouveau_dossier, chemin_fichier
from .serie import _slug
from .tileset3d import _partie
from .videos import duree_lisible

PHOTOS_MAX = 30
ECHECS_MAX = 3  # échecs à la suite avant d'arrêter (manque de mémoire, moteur absent…)
RENDUS = {
    "Vrai costume, la photo reste une photo (cosplay)": "photo",
    "Toute l'image dans le style de l'image du costume (dessin)": "dessin",
}
RENDU_DEFAUT = next(iter(RENDUS))
HABILLER = ("Dress the person in image 1 in the outfit of image 2: {tenue}. Adapt the outfit to the person's size, "
            "age and body shape")
HABILLER_TEXTE = ("Dress the person in the reference image in this outfit: {tenue}. Adapt the outfit to the person's "
                  "size, age and body shape")
GARDER = ("Keep the person of {qui} exactly the same: same face, facial features, expression, skin, age, body, "
          "pose and hands{cheveux}; same background, lighting, colors and framing. Only the clothes change")
COIFFURE = "Also give the person the hairstyle, headwear or wig of the outfit"
RENDU_PHOTO = "The outfit is a real fabric costume, photorealistic: the picture stays a real photograph"
RENDU_DESSIN = "Redraw the whole picture in the drawing style of image 2, the person staying recognizable"


def _chemins(fichiers):
    return [str(chemin_fichier(f)) for f in (fichiers or []) if f]


def decrire_tenue(costume, precisions="", progress=gr.Progress()):
    """Description anglaise de la tenue : d'après l'image du costume (et les précisions), ou d'après le texte seul."""
    precisions = (precisions or "").strip()
    if costume:
        return diffusion.decrire("costume", precisions or None, image=costume, progress=progress)
    if precisions:
        return diffusion.decrire("tenue", precisions, progress=progress)
    raise gr.Error("Donne l'image du costume, ou décris la tenue (en français).")


def prompt_costume(tenue, avec_image=True, rendu=RENDU_DEFAUT, coiffure=False):
    tenue = " ".join((tenue or "").split()).rstrip(". ")
    if not tenue:
        raise gr.Error("Il manque la description de la tenue : clique sur « Décrire la tenue », ou écris-la en anglais.")
    morceaux = [(HABILLER if avec_image else HABILLER_TEXTE).format(tenue=tenue),
                GARDER.format(qui="image 1" if avec_image else "the reference image",
                              cheveux="" if coiffure else ", same hair")]
    if coiffure:
        morceaux.append(COIFFURE)
    morceaux.append(RENDU_DESSIN if avec_image and RENDUS.get(rendu, rendu) == "dessin" else RENDU_PHOTO)
    return ". ".join(morceaux) + "."


def habiller(photos, costume, precisions="", tenue="", rendu=RENDU_DEFAUT, coiffure=False, variantes=1, graine=0,
             nom="", progress=gr.Progress()):
    """Chaque photo habillée de la tenue. Renvoie (message, galerie [(image, légende)], fichiers, dossier, tenue)."""
    photos = _chemins(photos)
    if not photos:
        raise gr.Error("Ajoute au moins une photo à habiller.")
    if len(photos) > PHOTOS_MAX:
        raise gr.Error(f"{len(photos)} photos : {PHOTOS_MAX} au plus à la fois.")
    for p in photos + ([costume] if costume else []):
        if not Path(p).is_file():
            raise gr.Error(f"Image introuvable : {p}")
    debut = time.monotonic()
    tenue = (tenue or "").strip() or decrire_tenue(costume, precisions, progress=progress)
    final = prompt_costume(tenue, bool(costume), rendu, coiffure)
    n = max(1, min(4, int(variantes or 1)))
    tirage = diffusion.graines(n, graine)  # mêmes graines pour toutes les photos : le costume varie moins
    nom = _slug(nom, "costume")
    dossier = nouveau_dossier(cfg.COSTUMES_DIR, nom)
    refs = []
    if costume:
        refs.append(dossier / f"costume{Path(costume).suffix.lower() or '.png'}")
        shutil.copy(costume, refs[0])
    (dossier / "originales").mkdir()
    (dossier / "resultats").mkdir()
    lot = {"tenue": tenue, "precisions": (precisions or "").strip() or None, "prompt": final,
           "costume": refs[0].name if refs else None, "rendu": RENDUS.get(rendu, rendu), "coiffure": bool(coiffure),
           "graines": tirage, "moteur": "FLUX.2 klein 4B", "photos": []}
    galerie, echecs, suite = [], [], 0
    for i, p in enumerate(photos, 1):
        base = f"{i:02d}_{_slug(Path(p).stem, 'photo')}"
        copie = dossier / "originales" / f"{base}{Path(p).suffix.lower() or '.png'}"
        shutil.copy(p, copie)
        largeur, hauteur = images.format_pour(images.AUTO, copie)
        sorties = [dossier / "resultats" / (f"{base}.png" if n == 1 else f"{base}_{v}.png") for v in range(1, n + 1)]
        entree = {"originale": f"originales/{copie.name}", "largeur": largeur, "hauteur": hauteur, "resultats": []}
        try:
            res = diffusion.personnage(final, [copie] + refs, sorties, tirage, largeur, hauteur,
                                       progress=_partie(progress, (i - 1) / len(photos), i / len(photos),
                                                        f"Photo {i}/{len(photos)}"))
        except (gr.Error, RuntimeError, OSError) as e:
            erreur = str(getattr(e, "message", e))
            echecs.append(f"photo {i} ({Path(p).name}) : {erreur}")
            entree["erreur"] = erreur
            lot["photos"].append(entree)
            suite += 1
            if suite >= ECHECS_MAX:
                echecs.append(f"arrêt après {ECHECS_MAX} échecs à la suite")
                break
            continue
        suite = 0
        for v, (f, g) in enumerate(zip(res["fichiers"], res["graines"]), 1):
            entree["resultats"].append(f"resultats/{Path(f).name}")
            galerie.append((f, f"Photo {i}" + (f", variante {v}" if n > 1 else "") + f" (graine {g})"))
        lot["photos"].append(entree)
    (dossier / "lot.json").write_text(json.dumps(lot, ensure_ascii=False, indent=1), encoding="utf-8")
    if not galerie:
        raise gr.Error("Aucune photo n'a pu être habillée : " + " ; ".join(echecs))
    archive = dossier / f"{nom}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:  # PNG déjà compressés
        for f, _ in galerie:
            z.write(f, Path(f).name)
    temps = time.monotonic() - debut
    reussies = sum(1 for e in lot["photos"] if e["resultats"])
    msg = (f"✅ {len(galerie)} image(s) pour {reussies}/{len(photos)} photo(s) en {duree_lisible(temps)}, dans {dossier}. Graines : "
           f"{', '.join(str(g) for g in tirage)} (garde-en une et relance si une variante te plaît).")
    if echecs:
        msg += "\n\n⚠️ " + "\n\n⚠️ ".join(echecs)
    return msg, galerie, [str(archive)] + [f for f, _ in galerie], str(dossier), tenue
