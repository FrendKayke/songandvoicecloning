"""Vidéo parlante (onglet « Image et vidéo → Faire parler une image ») : StableAvatar, moteur de diffusion, action
« parler ».

Une image (photo, illustration, personnage) et une voix → une vidéo où la personne parle ou chante, lèvres
synchronisées. La voix : un fichier (tout format lu par audio.charger : message vocal m4a, chanson, wav…) ou un
texte lu par une voix de la bibliothèque (Chatterbox). Pour une chanson, « Isoler la voix » passe d'abord par
Demucs : la musique gênerait la synchronisation (conseil des auteurs de StableAvatar) ; la vidéo garde le son
d'origine complet. StableAvatar travaille à 25 images/s dans trois formats (512×512, 480×832, 832×480) ; l'image
est recadrée au format choisi (automatique : le plus proche de ses proportions).
Rangement : data/videos/<horodatage>_<nom>/ : video.mp4, image.<ext>, reference.png (image recadrée), voix.wav
(son de la vidéo), audio_16k.wav (son lu par le modèle), creation.json (type « video », parlante = true).
"""
import shutil
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from . import config as cfg
from . import diffusion
from .audio import charger
from .outils import ecrire_creation, nouveau_dossier
from .serie import _slug
from .videos import duree_lisible

AUTO = "Automatique (d'après l'image)"
FORMATS = {AUTO: None, "Carré (512×512)": (512, 512), "Portrait (480×832)": (480, 832),
           "Paysage (832×480)": (832, 480)}
FICHIER = "Un fichier audio (voix, message vocal, chanson)"
TEXTE = "Un texte lu par une de mes voix"
SOURCES = [FICHIER, TEXTE]
DUREE_MAX = 180  # secondes : ~10 min de calcul par tranche de 5 s sur une RTX 4070 (estimation) au-delà, des heures
FPS = 25
PROMPT_DEFAUT = ("A person is talking to the camera with natural facial expressions and head movements, the mouth "
                 "moving with the words, in front of a softly blurred background.")


def format_pour(format_label, image):
    """(largeur, hauteur) : choisi, ou le format d'entraînement le plus proche des proportions de l'image."""
    if FORMATS.get(format_label):
        return FORMATS[format_label]
    from PIL import Image, ImageOps

    with Image.open(image) as im:
        largeur, hauteur = ImageOps.exif_transpose(im).size
    rapport = largeur / max(1, hauteur)
    return min((f for f in FORMATS.values() if f), key=lambda f: abs(np.log(rapport * f[1] / f[0])))


def preparer(image, souhaits="", progress=gr.Progress()):
    """Prompt anglais d'après l'image (Qwen3-VL : la personne, ce qu'elle fait, le décor), avec les souhaits."""
    if not image:
        raise gr.Error("Ajoute d'abord l'image à faire parler.")
    return diffusion.decrire("parlante", (souhaits or "").strip() or None, image=image, progress=progress)


def prompt_pret(image, souhaits, prompt, progress=gr.Progress()):
    """Prompt déjà écrit : gardé ; sinon préparé par Qwen s'il est installé, sinon le prompt par défaut."""
    prompt = (prompt or "").strip()
    if prompt:
        return prompt
    if diffusion.present("qwen"):
        return preparer(image, souhaits, progress=progress)
    return PROMPT_DEFAUT


def _voix_lue(voix, texte, langue_label, graine, progress):
    from . import chatterbox

    if not voix:
        raise gr.Error("Choisis la voix qui lira le texte (onglet 🎙️ Mes voix pour en ajouter une).")
    fichier, _ = chatterbox.synthese(voix, texte, langue_label, 0.5, 0.5, 0.8, graine, progress=progress)
    return Path(fichier)


def generer(image, source, audio, voix, texte, langue_label, isoler, prompt, format_label, etapes, graine,
            guidage_audio, nom="", souhaits="", progress=gr.Progress()):
    """Renvoie (message, vidéo, dossier)."""
    if not image or not Path(image).is_file():
        raise gr.Error("Ajoute l'image à faire parler (photo, illustration, personnage de face de préférence).")
    if source == TEXTE and not (texte or "").strip():
        raise gr.Error("Écris le texte que la personne doit dire.")
    if source != TEXTE and (not audio or not Path(str(audio)).is_file()):
        raise gr.Error("Ajoute le son : enregistrement, message vocal ou chanson.")
    debut = time.monotonic()
    nom = (nom or "").strip() or "video_parlante"
    graine = int(graine or 0) or diffusion.graines(1)[0]
    dossier = nouveau_dossier(cfg.VIDEOS_DIR, _slug(nom, "parlante"))
    copie = dossier / f"image{Path(image).suffix.lower() or '.png'}"
    shutil.copy(image, copie)
    if source == TEXTE:
        progress(0.01, desc="Lecture du texte par la voix choisie…")
        son = _voix_lue(voix, texte, langue_label, graine, progress)
    else:
        son = Path(str(audio))
    try:
        y, sr = charger(son, sr=44100, mono=False)
    except Exception as e:  # noqa: BLE001 - format illisible
        raise gr.Error(f"Impossible de lire le son ({e}).") from e
    y = np.atleast_2d(y)
    duree = y.shape[-1] / sr
    if duree < 1:
        raise gr.Error("Le son doit durer au moins une seconde.")
    if duree > DUREE_MAX:
        raise gr.Error(f"Le son dure {duree:.0f} s : {DUREE_MAX} s au plus (le calcul prendrait des heures). "
                       "Coupe-le en plusieurs morceaux.")
    voix_video = dossier / "voix.wav"  # son complet de la vidéo (musique comprise pour une chanson)
    sf.write(str(voix_video), y.T, sr)
    lu = voix_video
    if isoler:
        from . import demucs

        progress(0.05, desc="Séparation de la voix et de la musique (Demucs)…")
        lu, _ = demucs.separate_vocals(voix_video, dossier)
    y16, _ = charger(lu, sr=16000, mono=True)
    audio_16k = dossier / "audio_16k.wav"
    sf.write(str(audio_16k), y16, 16000, subtype="PCM_16")
    prompt = (prompt or "").strip() or PROMPT_DEFAUT
    largeur, hauteur = format_pour(format_label, copie)
    res = diffusion.parler(copie, audio_16k, voix_video, dossier / "video.mp4", prompt, largeur, hauteur, etapes,
                           graine, guidage_audio, progress=progress)
    temps = time.monotonic() - debut
    ecrire_creation(dossier, {
        "type": "video", "parlante": True, "nom": nom, "description": prompt, "description_fr": souhaits or None,
        "image_depart": copie.name, "format": format_label, "largeur": largeur, "hauteur": hauteur,
        "duree": f"{res['duree']:.0f} s", "etapes": int(etapes), "fps": FPS, "voix_isolee": bool(isoler),
        "guidage_audio": float(guidage_audio), "source": source, "texte": texte if source == TEXTE else None,
        "voix": voix if source == TEXTE else None, "temps_s": round(temps),
        "versions": [{"graine": res["graine"], "dossier": ".", "fichier": res["sortie"]}],
    })
    msg = (f"✅ Vidéo parlante {largeur}×{hauteur} de {res['duree']:.1f} s ({res['images']} images à {FPS} images/s), "
           f"graine {res['graine']}, en {duree_lisible(temps)}, dans {dossier}.")
    return msg, res["sortie"], str(dossier)


def recreer(chemin, infos, graine, progress=gr.Progress()):
    """Galerie : même image, même son (voix.wav gardé dans la création), mêmes réglages, même graine."""
    d = Path(chemin)
    image, son = d / infos["image_depart"], d / "voix.wav"
    if not image.exists() or not son.exists():
        raise gr.Error(f"Image ou son de la création introuvable dans {d}.")
    _, _, dossier = generer(str(image), FICHIER, str(son), None, None, None, infos.get("voix_isolee", False),
                            infos.get("description"), infos.get("format") or AUTO, infos.get("etapes", 30), graine,
                            infos.get("guidage_audio", 5.0), infos.get("nom"), infos.get("description_fr") or "",
                            progress=progress)
    return dossier
