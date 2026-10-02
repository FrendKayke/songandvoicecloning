"""Onglet « Bruitages » : effets sonores pour un jeu, à partir d'une description (français ou anglais) ou d'une image.

Qwen3-VL transforme la description française en prompt anglais précis (ou décrit les sons d'une image) ;
Stable Audio Open génère 1 à 3 variantes ; l'export reprend le volume harmonisé de la bande-son.
Rangement : data/bruitages/<horodatage>/ (variante_<i>.wav, creation.json). Un bruitage peut appartenir à un projet
de jeu (le même nom que dans « Bande-son de jeu ») : le pack du jeu reprend alors, pour chaque nom de bruitage, la
création la plus récente et sa variante choisie (« choisie » dans creation.json, 1 par défaut).
"""
import json
import shutil
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import diffusion, projets
from .outils import ecrire_creation, nouveau_dossier

DUREE_MAX = 30
NEGATIF = "music, melody, singing, speech, low quality, distortion"

EXEMPLES = [
    ("Carte jouée", "card sliding on a wooden table then a soft thump"),
    ("Épée dégainée", "metal sword drawn from a leather scabbard, sharp ring"),
    ("Sort de feu", "fireball whoosh and burst, crackling flames"),
    ("Sort de glace", "ice crystal forming and shattering, cold shimmer"),
    ("Soin magique", "gentle magical healing chime, warm sparkle"),
    ("Coup d'épée", "sword slash hitting armor, metallic clang"),
    ("Pièces d'or", "coins pouring into a leather pouch"),
    ("Clic de menu", "short soft UI click"),
    ("Erreur", "short negative UI buzz"),
    ("Porte de château", "heavy wooden castle door creaking open and slamming"),
    ("Pas sur la pierre", "footsteps on stone floor in a hall, echo"),
    ("Tonnerre", "distant thunder rumble"),
]


def preparer(texte, image_path, progress=gr.Progress()):
    """Description → prompt anglais (Qwen3-VL). Renvoie le prompt, modifiable avant génération."""
    texte = (texte or "").strip()
    if image_path:
        return diffusion.decrire("son", texte or None, image_path, progress=progress)
    if not texte:
        raise gr.Error("Décris le bruitage (en français ou en anglais), ou importe une image.")
    return diffusion.decrire("bruitage", texte, progress=progress)


def projets_de_jeu():
    """Projets connus, tous onglets confondus (projets.tous)."""
    return projets.tous()


def _nom_projet(projet):
    return projets.nom(projet)


def choisir(dossier, fichier):
    """Retient la variante écoutée pour le pack du jeu."""
    if not dossier or not fichier:
        raise gr.Error("Génère d'abord un bruitage, puis écoute la variante à garder.")
    chemin = Path(dossier) / "creation.json"
    infos = json.loads(chemin.read_text(encoding="utf-8"))
    fichiers = [Path(v["fichier"]).name for v in infos.get("versions", [])]
    if Path(fichier).name not in fichiers:
        raise gr.Error("Cette variante n'appartient pas au dernier bruitage généré.")
    infos["choisie"] = fichiers.index(Path(fichier).name) + 1
    chemin.write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    projet = infos.get("projet")
    return (f"⭐ Variante {infos['choisie']} gardée pour « {infos.get('nom')} »"
            + (f" (pack du projet {projet})." if projet else ". Donne un projet au bruitage pour l'ajouter à un pack."))


def prompt_pret(texte, image_path, prompt, progress=gr.Progress()):
    """« Générer » sans avoir préparé le prompt (constaté : erreur « Il manque le prompt ») : le prompt anglais est
    préparé d'abord depuis la description, puis affiché ; un prompt déjà là (préparé ou retouché) est gardé."""
    return (prompt or "").strip() or preparer(texte, image_path, progress=progress)


def generer(prompt, nom, duree, variantes, graine, etapes, image_path=None, description=None, projet=None,
            progress=gr.Progress()):
    """Génère les variantes. Renvoie (message, choix des variantes, première variante, dossier)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Il manque le prompt : clique d'abord sur « Préparer le prompt », ou écris-le en anglais.")
    duree = max(1.0, min(DUREE_MAX, float(duree or 3)))
    dossier = nouveau_dossier(cfg.SFX_DIR)
    if image_path:
        shutil.copy(image_path, dossier / ("image" + Path(image_path).suffix.lower()))
    res = diffusion.bruitage(prompt, dossier, duree, variantes, graine, etapes, NEGATIF, progress)
    nom = "".join(c for c in (nom or "").strip() if c.isalnum() or c in "-_ ").strip() or "bruitage"
    ecrire_creation(dossier, {
        "type": "bruitage", "nom": nom, "description": prompt, "description_fr": (description or "").strip() or None,
        "projet": _nom_projet(projet), "choisie": 1, "duree": duree, "etapes": int(etapes), "negatif": NEGATIF,
        "versions": [{"graine": g, "dossier": ".", "fichier": f} for f, g in zip(res["fichiers"], res["graines"])],
    })
    choix = [(f"Variante {i} (graine {g})", f) for i, (f, g) in enumerate(zip(res["fichiers"], res["graines"]), 1)]
    msg = (f"✅ {len(choix)} variante(s) de {duree:.0f} s dans {dossier}. Graines : "
           f"{', '.join(str(g) for g in res['graines'])}.")
    return msg, gr.update(choices=choix, value=choix[0][1]), choix[0][1], str(dossier)
