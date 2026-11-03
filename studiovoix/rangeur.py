"""Rangeur (groupe Outils, onglet « 📦 Ranger mes créations ») : copie les créations d'un projet de jeu, d'une période
ou de quelques types dans un dossier propre, un sous-dossier par type, des noms lisibles
(« 2026-11-03_Château dans la brume_2.png »), une liste `liste.csv` (Excel) et une archive .zip.

Les créations sont celles de la galerie (creation.json) : fichiers finaux de chaque version (ou de la variante
choisie seulement). Rien n'est déplacé ni supprimé. Rangement : data/rangements/<horodatage>_<nom>/.
"""
import csv
import re
import shutil
import zipfile
from datetime import date
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import galerie
from .outils import nouveau_dossier
from .serie import _slug, nom_de_fichier

TOUS = "Toutes mes créations"
DOSSIERS = {"chanson": "Chansons", "jeu": "Bande-son", "tts": "Lectures", "bruitage": "Bruitages",
            "3d": "Modèles 3D", "carte": "Illustrations", "photo": "Photos", "video": "Vidéos", "image": "Images",
            "remix": "Remix"}
TYPES = {lib: t for lib, t in galerie.FILTRES.items() if t}  # « Chansons » → « chanson »…
# fichiers gardés à côté du fichier principal (même dossier de version), s'ils existent
A_COTE = {"3d": ("modele.obj", "material.mtl", "material_0.png", "modele_anime.glb", "modele_anime.fbx",
                 "modele_web.glb", "modele_facettes.glb"),
          "video": ("sous_titres.srt", "paroles.lrc", "paroles.srt")}
MAX_CREATIONS = 5000


def projets():
    from .projets import tous

    return [TOUS] + tous()


def lire_date(texte, quoi):
    """« 03/11/2026 », « 3/11/26 », « 2026-11-03 » ou vide → date ou None."""
    t = (texte or "").strip()
    if not t:
        return None
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        a, mo, j = map(int, m.groups())
    else:
        m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2}|\d{4})", t)
        if not m:
            raise gr.Error(f"Date « {t} » mal lue ({quoi}) : écris-la comme 03/11/2026.")
        j, mo, a = map(int, m.groups())
        a += 2000 if a < 100 else 0
    try:
        return date(a, mo, j)
    except ValueError:
        raise gr.Error(f"Date « {t} » impossible ({quoi}).") from None


def _date_de(dossier):
    n = Path(dossier).name
    if len(n) >= 8 and n[:8].isdigit():
        try:
            return date(int(n[:4]), int(n[4:6]), int(n[6:8]))
        except ValueError:
            return None
    return None


def _nom(infos, dossier):
    """Nom lisible d'une création : son nom, sinon le début de sa description."""
    n = infos.get("nom") or infos.get("libelle") or ""
    if not n or n in ("bruitage", "objet", "image", "video", "doublage", "sans_fond"):
        texte = (infos.get("description_fr") or infos.get("description") or infos.get("texte") or n or "")
        n = " ".join(str(texte).split()[:6]).rstrip(" ,;:.-") or Path(dossier).name
    return Path(nom_de_fichier(str(n)[:60]) or "creation").stem


def _projet_de(infos, dossier):
    p = infos.get("projet")
    if p:
        return p
    d = Path(dossier)
    for racine in (cfg.GAMES_DIR, cfg.CARDS_DIR):  # bande-son : jeux/<projet>/… ; illustrations : cartes/<projet>/…
        try:
            return d.relative_to(racine).parts[0]
        except (ValueError, IndexError):
            continue
    return None


def selection(projet=TOUS, types=None, du="", au=""):
    """Créations retenues : [(dossier, infos, date)] des plus anciennes aux plus récentes."""
    voulus = {TYPES[t] for t in (types or []) if t in TYPES}
    debut, fin = lire_date(du, "début"), lire_date(au, "fin")
    if debut and fin and fin < debut:
        raise gr.Error("La date de fin est avant la date de début.")
    retenues = []
    for d in galerie._dossiers():
        infos = galerie.lire(d)
        if not infos or (voulus and infos.get("type") not in voulus):
            continue
        if projet and projet != TOUS and (_projet_de(infos, d) or "").lower() != projet.lower():
            continue
        jour = _date_de(d)
        if (debut and (not jour or jour < debut)) or (fin and (not jour or jour > fin)):
            continue
        retenues.append((d, infos, jour))
    retenues.sort(key=lambda x: (x[2] or date.min, Path(x[0]).name))
    return retenues


def apercu(projet=TOUS, types=None, du="", au=""):
    r = selection(projet, types, du, au)
    if not r:
        return "Aucune création ne correspond."
    compte = {}
    for _, infos, _ in r:
        compte[infos["type"]] = compte.get(infos["type"], 0) + 1
    return (f"**{len(r)} création(s)** : " + ", ".join(f"{DOSSIERS.get(t, t)} {n}" for t, n in sorted(compte.items()))
            + ".")


def _fichiers(infos, dossier, toutes):
    """Fichiers à copier : (chemin, numéro de version ou None)."""
    versions = [v for v in infos.get("versions") or [] if v.get("fichier")]
    choisie = infos.get("choisie")
    if not toutes and versions:
        if isinstance(choisie, int) and 0 <= choisie - 1 < len(versions):
            versions = [versions[choisie - 1]]
        elif isinstance(choisie, str):
            versions = [v for v in versions if Path(v["fichier"]).name == Path(choisie).name] or versions[:1]
        else:
            versions = versions[:1]
    sortie = []
    for k, v in enumerate(versions, 1):
        f = Path(v["fichier"])
        if not f.is_absolute():
            f = Path(dossier) / f
        if not f.is_file():
            continue
        sortie.append((f, k if len(versions) > 1 else None))
        for nom in A_COTE.get(infos.get("type"), ()):
            a = f.parent / nom
            if a.is_file() and a != f:
                sortie.append((a, k if len(versions) > 1 else None))
    return sortie


def ranger(projet=TOUS, types=None, du="", au="", toutes=True, archive=True, nom="", progress=gr.Progress()):
    """Renvoie (message, [fichiers à télécharger], dossier)."""
    r = selection(projet, types, du, au)
    if not r:
        raise gr.Error("Aucune création ne correspond (projet, types ou dates).")
    if len(r) > MAX_CREATIONS:
        raise gr.Error(f"{len(r)} créations : {MAX_CREATIONS} au plus d'un coup (choisis des dates ou des types).")
    nom = (nom or "").strip() or (projet if projet and projet != TOUS else "mes_creations")
    dossier = nouveau_dossier(cfg.RANGEMENTS_DIR, _slug(nom, "rangement"))
    pris, lignes, copies, taille = set(), [], 0, 0
    for i, (d, infos, jour) in enumerate(r):
        progress(i / len(r), desc=f"Copie {i + 1}/{len(r)}…")
        sous = DOSSIERS.get(infos.get("type"), "Autres")
        base = f"{jour.isoformat() if jour else 'sans_date'}_{_nom(infos, d)}"
        principaux = {}
        for f, k in _fichiers(infos, d, toutes):
            ext = f.suffix.lower()
            if k not in principaux:  # premier fichier de la version : nom de base ; les fichiers d'à côté gardent le leur
                stem = base + (f"_{k}" if k else "")
                n, x = stem, 2
                while f"{sous}/{n}{ext}".lower() in pris:
                    n, x = f"{stem}_{x}", x + 1
                principaux[k] = n
                cible = f"{sous}/{n}{ext}"
            else:
                cible = f"{sous}/{principaux[k]}_{f.name}"
            pris.add(cible.lower())
            (dossier / cible).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dossier / cible)
            copies += 1
            taille += f.stat().st_size
            lignes.append([jour.strftime("%d/%m/%Y") if jour else "", sous, _nom(infos, d),
                           str(infos.get("description_fr") or infos.get("description") or infos.get("texte") or "")
                           .replace("\n", " ")[:300], _projet_de(infos, d) or "", cible, str(d)])
    with open(dossier / "liste.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Date", "Type", "Nom", "Description", "Projet", "Fichier", "Création d'origine"])
        w.writerows(lignes)
    fichiers = [str(dossier / "liste.csv")]
    from .convertir import _taille

    msg = (f"✅ {len(r)} création(s), {copies} fichier(s) ({_taille(taille)}) rangés dans `{dossier}`, un dossier par "
           "type ; la liste est dans `liste.csv` (s'ouvre dans Excel).")
    if archive:
        progress(0.95, desc="Archive .zip…")
        z = dossier / f"{dossier.name}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(dossier.rglob("*")):
                if p.is_file() and p != z:
                    zf.write(p, p.relative_to(dossier).as_posix())
        fichiers.insert(0, str(z))
        msg += f" Archive : `{z.name}`."
    if copies == 0:
        msg += "\n\n⚠️ Aucun fichier trouvé : les fichiers de ces créations ont été déplacés ou supprimés."
    return msg, fichiers, str(dossier)
