"""Diaporama (groupe « 🔧 Audio / vidéo », onglet « 🖼️ Diaporama ») : des photos → une vidéo, chaque photo animée
d'un lent zoom ou travelling (effet « Ken Burns »), fondus enchaînés, musique de fond, voix off (texte lu par une
de mes voix ou enregistrement) et titre. Tout le rendu passe par `montage.monter` (options `ken_burns` et `voix`) :
la musique baisse sous la voix, les photos s'allongent pour que la voix tienne. Rangement : celui du montage
(data/videos/<horodatage>_<nom>/, creation.json type `video` avec `montage` et `diaporama`).
"""
import shutil
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import montage
from .outils import chemin_fichier, nouveau_dossier

SANS_VOIX = "Pas de voix off"
TEXTE = "Un texte lu par une de mes voix"
ENREGISTREMENT = "Un enregistrement (fichier ou micro)"
SOURCES = [SANS_VOIX, TEXTE, ENREGISTREMENT]
MOUVEMENTS = {"Aucun (photos immobiles)": 0.0, "Doux (zoom de 8 %)": 0.08, "Moyen (zoom de 15 %)": 0.15,
              "Marqué (zoom de 25 %)": 0.25}
MOUVEMENT_DEFAUT = "Moyen (zoom de 15 %)"
TAILLE_DEFAUT = "Paysage 1920×1080 (YouTube)"
PHOTOS_MAX = 200


def creer(photos, duree_photo=5.0, transition=montage.ENCHAINE, duree_transition=1.0, mouvement=MOUVEMENT_DEFAUT,
          taille=TAILLE_DEFAUT, musique=None, volume_musique=40, source=SANS_VOIX, voix=None, langue="Français",
          texte="", enregistrement=None, adapter=True, titre="", police=None, nom="diaporama", progress=gr.Progress()):
    """Renvoie (message, vidéo, [fichiers], dossier)."""
    from .voix_off import _voix

    images = [chemin_fichier(p) for p in (photos or []) if chemin_fichier(p)]
    images = [str(i) for i in images if montage._est_image(i)]
    if not images:
        raise gr.Error("Ajoute des photos (PNG, JPG, WebP…), dans l'ordre du diaporama.")
    if len(images) > PHOTOS_MAX:
        raise gr.Error(f"{len(images)} photos : {PHOTOS_MAX} au plus.")
    fichier_voix = None
    if source != SANS_VOIX:
        travail = nouveau_dossier(cfg.DATA_DIR / "_tmp", "diaporama_voix")
        progress(0.01, desc="Voix off…")
        _, desc = _voix(montage_source(source), voix, langue, texte, enregistrement, (0.5, 0.5, 0.8, 0), travail,
                        progress)
        fichier_voix = travail / "voix.wav"
    msg, video, fichiers, dossier = montage.monter(
        images, None, transition, duree_transition, duree_photo, True, taille, montage.REMPLIR, 100, musique,
        volume_musique, True, titre, 4.0, "Au centre", police, 9, nom=nom, ken_burns=MOUVEMENTS.get(mouvement, 0.15),
        voix=fichier_voix, adapter_a_la_voix=adapter,
        infos_en_plus={"diaporama": {"photos": len(images), "mouvement": mouvement, "voix_off": source}},
        progress=progress)
    if fichier_voix:
        shutil.move(str(fichier_voix), Path(dossier) / "voix.wav")
        shutil.rmtree(fichier_voix.parent, ignore_errors=True)
        fichiers = fichiers + [str(Path(dossier) / "voix.wav")]
    return msg.replace("Montage de", "Diaporama de"), video, fichiers, dossier


def montage_source(source):
    """Libellé de la source de voix attendu par `voix_off._voix`."""
    from . import voix_off

    return voix_off.TEXTE if source == TEXTE else voix_off.FICHIER
