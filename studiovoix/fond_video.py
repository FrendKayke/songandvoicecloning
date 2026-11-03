"""Retirer ou remplacer le fond d'une vidéo (« fond vert sans fond vert ») : BiRefNet image par image dans le moteur de
diffusion (action « fond_video »), alpha lissé dans le temps pour que les bords ne scintillent pas.

Sorties : fond transparent en WebM VP9 (navigateurs, Unity, sites) ou en MOV ProRes 4444 (Premiere, DaVinci Resolve,
Final Cut, CapCut sur PC), ou MP4 avec un nouveau fond (vert d'incrustation, couleur, fond flouté, image). Le son
d'origine est gardé ; la vidéo du masque (noir et blanc) est facultative. Rangement : data/videos/<horodatage>_<nom>/
(creation.json type `video` avec `fond_video` : galerie, voix off, montage).
"""
import re
import shutil
import time
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import diffusion
from .convertir import _taille, duree_lisible, infos
from .outils import chemin_fichier, ecrire_creation, nouveau_dossier
from .serie import _slug

TRANSPARENT_WEBM = "Transparent — WebM (web, jeux, Unity)"
TRANSPARENT_MOV = "Transparent — MOV ProRes 4444 (Premiere, DaVinci, Final Cut, CapCut PC)"
VERT = "Fond vert (pour incruster ailleurs)"
COULEUR = "Couleur unie"
FLOU = "Son propre fond, flouté"
IMAGE = "Une image de fond"
FONDS = [TRANSPARENT_WEBM, TRANSPARENT_MOV, VERT, COULEUR, FLOU, IMAGE]
MODELES = {"Une ou des personnes (plus rapide)": "personne",
           "Tout sujet : animal, objet… (plus lent, bords plus fins)": "general"}
MODELE_DEFAUT = next(iter(MODELES))
DUREE_MAX = 10 * 60  # s


def _couleur(valeur):
    """« #rgb », « #rrggbb » ou « rgb(a)(r, g, b…) » (ColorPicker de Gradio) → « #rrggbb »."""
    v = str(valeur or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", v):
        return v.lower()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", v):
        return "#" + "".join(c * 2 for c in v[1:]).lower()
    m = re.match(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", v)
    if m:
        return "#" + "".join(f"{min(255, round(float(x))):02x}" for x in m.groups())
    raise gr.Error(f"Couleur mal lue : « {valeur} ».")


def _video_de(dossier):
    from .galerie import lire

    d = lire(dossier) if dossier else None
    return (d.get("versions") or [{}])[0].get("fichier") if d else None


def traiter(video, creation=None, fond_label=TRANSPARENT_WEBM, couleur="#ffffff", image_fond=None,
            modele_label=MODELE_DEFAUT, lissage=0.3, avec_masque=False, nom="sans_fond", progress=gr.Progress()):
    """Renvoie (message, vidéo, [fichiers], dossier)."""
    source = chemin_fichier(video) or _video_de(creation)
    if not source or not Path(str(source)).is_file():
        raise gr.Error("Ajoute une vidéo (ou choisis une vidéo créée ici).")
    source = Path(str(source))
    d = infos(source)
    if not d["video"]:
        raise gr.Error(f"« {source.name} » n'a pas d'image.")
    if d["duree"] > DUREE_MAX:
        raise gr.Error(f"Vidéo de {duree_lisible(d['duree'])} : {duree_lisible(DUREE_MAX)} au plus (coupe-la d'abord dans "
                       "« Audio / vidéo → Couper »).")
    if fond_label not in FONDS:
        raise gr.Error("Choisis le fond.")
    debut = time.monotonic()
    dossier = nouveau_dossier(cfg.VIDEOS_DIR, _slug(nom or source.stem, "sans_fond"))
    ext = {TRANSPARENT_WEBM: "webm", TRANSPARENT_MOV: "mov"}.get(fond_label, "mp4")
    fond = {TRANSPARENT_WEBM: "transparent", TRANSPARENT_MOV: "transparent", VERT: "#00ff00", FLOU: "flou"}.get(fond_label)
    if fond_label == COULEUR:
        fond = _couleur(couleur)
    elif fond_label == IMAGE:
        chemin = chemin_fichier(image_fond)
        if not chemin or not Path(str(chemin)).is_file():
            raise gr.Error("Ajoute l'image de fond (ou choisis un autre fond).")
        copie = dossier / f"fond{Path(str(chemin)).suffix.lower() or '.png'}"
        shutil.copy(chemin, copie)
        fond = str(copie)
    sortie = dossier / f"{_slug(nom or source.stem, 'video')}_{'transparent' if fond == 'transparent' else 'nouveau_fond'}.{ext}"
    masque = dossier / "masque.mp4" if avec_masque else None
    r = diffusion.fond_video(source, sortie, MODELES.get(modele_label, "personne"), fond, lissage, masque,
                             progress=progress)
    temps = time.monotonic() - debut
    ecrire_creation(dossier, {
        "type": "video", "nom": nom, "description": f"fond retiré : {source.name}",
        "fond_video": {"original": source.name, "fond": fond_label, "couleur": fond if fond_label == COULEUR else None,
                       "modele": modele_label, "lissage": lissage},
        "duree": r.get("duree"), "largeur": r.get("largeur"), "hauteur": r.get("hauteur"), "temps_s": round(temps),
        "versions": [{"graine": 0, "dossier": ".", "fichier": str(sortie)}],
    })
    fichiers = [str(sortie)] + ([str(masque)] if masque and masque.exists() else [])
    msg = (f"✅ Fond {'retiré' if fond == 'transparent' else 'remplacé'} sur {r['images']} images "
           f"({duree_lisible(r['duree'])}) en {duree_lisible(temps)} : `{sortie}` ({_taille(sortie.stat().st_size)}).")
    if r.get("couverture") is not None and r["couverture"] < 0.01:
        msg += ("\n\n⚠️ Presque rien n'a été gardé : le sujet n'a pas été reconnu. Si ce n'est pas une personne, "
                "choisis « Tout sujet ».")
    if fond == "transparent":
        msg += ("\n\nLe lecteur de la page montre parfois un fond noir à la place de la transparence : elle est bien "
                "dans le fichier (ouvre-le dans ton logiciel de montage ou ton moteur de jeu).")
    if ext == "mov":
        msg += " Les navigateurs ne lisent pas le ProRes : pas d'aperçu ici, le fichier est dans la liste."
    return msg, (None if ext == "mov" else str(sortie)), fichiers, str(dossier)
