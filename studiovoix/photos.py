"""Onglet « Photos » : améliorer la qualité, détourer, isoler une personne.

Trois traitements du moteur de diffusion (moteurs/diffusion.py) :
  - amélioration : Real-ESRGAN (BSD-3) agrandit ×2 ou ×4 et restaure (bruit, artefacts JPEG, flou) ; ×1 restaure
    sans agrandir ; GFPGAN 1.4 (Apache 2.0) restaure les visages ;
  - détourage : BiRefNet HR-matting (MIT), masque doux pour les cheveux ;
  - personne : BiRefNet-portrait (MIT), entraîné sur des personnes.
Le fond peut rester transparent (PNG), devenir une couleur ou le fond d'origine flouté (effet portrait).
Rangement : data/photos/<horodatage>/ : originale.<ext>, resultat.png, masque.png (détourage), creation.json
(type « photo »). « Continuer avec ce résultat » enchaîne les traitements (améliorer puis détourer, par exemple).
"""
import shutil
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import diffusion
from .outils import ecrire_creation, nouveau_dossier

ACTIONS = {
    "✨ Améliorer la qualité (agrandir, nettoyer, visages)": "ameliorer",
    "✂️ Détourer (objet, animal, personne…)": "detourer",
    "🧍 Isoler une personne": "personne",
}
ACTION_DEFAUT = next(iter(ACTIONS))
ECHELLES = {"×1 (nettoyer sans agrandir)": 1, "×2": 2, "×4": 4}
FONDS = {
    "Transparent (PNG)": None,
    "Blanc": "#ffffff",
    "Noir": "#000000",
    "Vert d'incrustation": "#00b140",
    "Fond d'origine flouté (effet portrait)": "flou",
    "Couleur au choix": "couleur",
}
FOND_DEFAUT = "Transparent (PNG)"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _couleur(valeur):
    """Couleur du sélecteur (« #rrggbb », ou « rgba(r, g, b, a) » selon le navigateur) → « #rrggbb »."""
    v = (valeur or "").strip()
    if v.startswith("#") and len(v) in (4, 7):
        return v if len(v) == 7 else "#" + "".join(c * 2 for c in v[1:])
    if v.startswith("rgb"):
        r, g, b = (int(float(x)) for x in v[v.index("(") + 1:v.index(")")].split(",")[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    raise gr.Error(f"Couleur non reconnue : {valeur}")


def traiter(image, action_label, echelle_label, rapide, visages, force, fond_label, couleur, nom="",
            progress=gr.Progress()):
    """Applique le traitement choisi. Renvoie (message, (avant, après) pour la comparaison, fichiers, dossier)."""
    if not image:
        raise gr.Error("Ajoute d'abord une photo.")
    source = Path(image)
    if source.suffix.lower() not in EXTENSIONS:
        raise gr.Error(f"Format non pris en charge ({source.suffix}) : PNG, JPG, WebP, BMP ou TIFF.")
    action = ACTIONS.get(action_label, "ameliorer")
    dossier = nouveau_dossier(cfg.PHOTOS_DIR)
    originale = dossier / f"originale{source.suffix.lower()}"
    shutil.copy(source, originale)
    sortie = dossier / "resultat.png"
    nom = (nom or "").strip() or source.stem
    reglages = {"action": action}
    if action == "ameliorer":
        echelle = ECHELLES.get(echelle_label, 2)
        reglages.update(echelle=echelle, rapide=bool(rapide), visages=bool(visages), force=float(force))
        res = diffusion.ameliorer(originale, sortie, echelle, rapide, visages, force, progress=progress)
        visages_msg = (f", {res['visages']} visage(s) restauré(s)" if visages else "")
        msg = f"✅ Photo améliorée : {res['largeur']}×{res['hauteur']}{visages_msg}."
        fichiers = [str(sortie)]
    else:
        fond = FONDS.get(fond_label)
        if fond == "couleur":
            fond = _couleur(couleur)
        reglages.update(fond=fond, fond_label=fond_label)
        masque = dossier / "masque.png"
        res = diffusion.detourer(originale, sortie, "personne" if action == "personne" else "general", fond, masque,
                                 progress=progress)
        if res.get("couverture", 1) < 0.01:
            msg = ("⚠️ Presque rien n'a été gardé : le sujet n'a pas été reconnu. Essaie l'autre mode "
                   "(« Détourer » ou « Isoler une personne ») ou une photo où le sujet est plus net.")
        else:
            quoi = "Personne isolée" if action == "personne" else "Photo détourée"
            msg = f"✅ {quoi} ({'fond transparent' if not fond else fond_label.lower()})."
        fichiers = [str(sortie), str(masque)]
    ecrire_creation(dossier, {"type": "photo", "nom": nom, "action": action, "reglages": reglages,
                              "originale": originale.name,
                              "versions": [{"graine": None, "dossier": ".", "fichier": str(sortie)}]})
    return f"{msg} Dans {dossier}.", (str(originale), str(sortie)), fichiers, str(dossier)


def continuer(dossier):
    """Le résultat devient la photo de départ (pour enchaîner : améliorer puis détourer…)."""
    if not dossier or not (Path(dossier) / "resultat.png").exists():
        raise gr.Error("Traite d'abord une photo.")
    return str(Path(dossier) / "resultat.png")


def maj_action(action_label):
    """Réglages visibles selon le traitement : agrandissement pour l'amélioration, fond pour le détourage."""
    ameliorer = ACTIONS.get(action_label) == "ameliorer"
    return gr.update(visible=ameliorer), gr.update(visible=not ameliorer)


def maj_fond(fond_label):
    return gr.update(visible=FONDS.get(fond_label) == "couleur")


def recreer(chemin, infos, progress=gr.Progress()):
    """Galerie : refait le même traitement sur l'originale, dans une nouvelle création."""
    originale = Path(chemin) / (infos.get("originale") or "")
    if not originale.is_file():
        raise gr.Error(f"Photo d'origine introuvable : {originale}")
    r = infos.get("reglages") or {}
    action = next((lib for lib, a in ACTIONS.items() if a == r.get("action")), ACTION_DEFAUT)
    echelle = next((lib for lib, e in ECHELLES.items() if e == r.get("echelle")), "×2")
    fond_label = r.get("fond_label") or FOND_DEFAUT
    couleur = r.get("fond") if FONDS.get(fond_label) == "couleur" else None
    _, _, _, dossier = traiter(str(originale), action, echelle, r.get("rapide", False), r.get("visages", True),
                               r.get("force", 0.7), fond_label, couleur, infos.get("nom"), progress=progress)
    return dossier
