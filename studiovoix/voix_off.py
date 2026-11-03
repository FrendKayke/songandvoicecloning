"""Vidéo avec voix off (onglet « Vidéo → Vidéo avec voix off ») : une vidéo (générée ici ou importée) reçoit une voix
qui lit un texte (Chatterbox, une voix de la bibliothèque) ou un enregistrement (fichier, micro), mixée ou non avec
le son d'origine. Aucun modèle de plus : la voix vient de la synthèse vocale, l'assemblage est fait par PyAV.

- la voix plus courte que la vidéo : silence ensuite ; plus longue : coupée, ou, si « Figer la dernière image » est
  coché, la vidéo est prolongée sur sa dernière image jusqu'à la fin de la voix (réencodage H.264) ;
- sans prolongement, la vidéo n'est pas réencodée : ses paquets sont recopiés tels quels, seul le son est encodé
  (AAC stéréo, 44,1 kHz) ;
- `creation.json` type `video` avec `voix_off` (galerie : lecture ; « Recréer » renvoie à l'onglet).
Rangement : data/videos/<horodatage>_<nom>/ : video_voix_off.mp4, voix.wav, creation.json.
"""
import shutil
import time
from fractions import Fraction
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from . import chatterbox
from . import config as cfg
from .audio import charger
from .outils import chemin_fichier, ecrire_creation, nouveau_dossier
from .serie import _slug
from .sous_titres import VIDEOS

TEXTE = "Un texte lu par une de mes voix"
FICHIER = "Un enregistrement (fichier ou micro)"
SOURCES = [TEXTE, FICHIER]
SR = 44100
TAILLE_TRAME = 1024  # échantillons par trame AAC
DUREE_MAX = 600  # s


def _infos_video(chemin):
    """(durée en s, cadence, largeur, hauteur, présence d'un son) lues dans le conteneur."""
    import av

    with av.open(str(chemin)) as entree:
        v = entree.streams.video[0]
        cadence = v.average_rate or Fraction(25)
        duree = None
        if v.duration is not None and v.time_base:
            duree = float(v.duration * v.time_base)
        elif entree.duration:
            duree = entree.duration / av.time_base
        if not duree:
            duree = (v.frames or 0) / float(cadence)
        return duree, cadence, v.codec_context.width, v.codec_context.height, bool(entree.streams.audio)


def _voix(source, voix, langue, texte, audio, reglages, dossier, progress):
    """Signal mono 44,1 kHz de la voix off (fichier voix.wav écrit dans la création). Renvoie (signal, description)."""
    if source == TEXTE:
        fichier, msg = chatterbox.synthese(voix, texte, langue, *reglages, progress=progress)
        y, _ = charger(fichier, sr=SR, mono=True)
        desc = {"source": "texte", "voix": voix, "langue": langue, "texte": texte, "tts": str(fichier)}
    else:
        audio = chemin_fichier(audio)
        if not audio or not Path(str(audio)).is_file():
            raise gr.Error("Ajoute l'enregistrement de la voix (fichier ou micro), ou choisis « un texte lu ».")
        try:
            y, _ = charger(audio, sr=SR, mono=True)
        except Exception as e:  # noqa: BLE001 - fichier illisible
            raise gr.Error(f"Impossible de lire « {Path(str(audio)).name} » ({e}).") from e
        desc = {"source": "fichier", "fichier": Path(str(audio)).name}
    y = np.asarray(y, np.float32)
    if not len(y) or float(np.max(np.abs(y))) < 1e-4:
        raise gr.Error("La voix est vide (aucun son).")
    sf.write(str(dossier / "voix.wav"), y, SR, subtype="PCM_16")
    return y, desc


def mixer(voix, origine, duree, volume_origine=0.3):
    """Voix (mono, SR) + son d'origine atténué, sur `duree` secondes ; pic ramené sous 0,98."""
    n = int(round(duree * SR))
    mix = np.zeros(n, np.float32)
    mix[:min(n, len(voix))] += voix[:n]
    if origine is not None and volume_origine > 0:
        mix[:min(n, len(origine))] += np.asarray(origine[:n], np.float32) * float(volume_origine)
    pic = float(np.max(np.abs(mix))) if n else 0.0
    if pic > 0.98:
        mix *= 0.98 / pic
    return mix


def _ecrire(video, mix, sortie, prolonger_s=0.0, progress=None):
    """MP4 = vidéo (paquets recopiés, ou réencodée et prolongée de `prolonger_s` sur sa dernière image) + mix en AAC."""
    import av

    with av.open(str(video)) as entree, av.open(str(sortie), "w") as out:
        flux_v = entree.streams.video[0]
        cadence = flux_v.average_rate or Fraction(25)
        out_a = out.add_stream("aac", rate=SR, layout="stereo")
        if prolonger_s > 0:
            largeur, hauteur = flux_v.codec_context.width // 2 * 2, flux_v.codec_context.height // 2 * 2
            out_v = out.add_stream("libx264", rate=cadence)
            out_v.width, out_v.height, out_v.pix_fmt = largeur, hauteur, "yuv420p"
            out_v.options = {"crf": "18", "preset": "medium"}
            derniere, n = None, 0
            for trame in entree.decode(flux_v):
                image = trame.to_ndarray(format="rgb24")[:hauteur, :largeur]
                derniere = image
                for p in out_v.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    out.mux(p)
                n += 1
                if progress and n % 50 == 0:
                    progress(0.5, desc=f"Réencodage de la vidéo ({n} images)…")
            if derniere is not None:
                for _ in range(int(round(prolonger_s * float(cadence)))):
                    for p in out_v.encode(av.VideoFrame.from_ndarray(derniere, format="rgb24")):
                        out.mux(p)
            for p in out_v.encode(None):
                out.mux(p)
        else:
            out_v = out.add_stream_from_template(flux_v)
            for paquet in entree.demux(flux_v):
                if paquet.dts is None:
                    continue
                paquet.stream = out_v
                out.mux(paquet)
        stereo = np.ascontiguousarray(np.stack([mix, mix]))
        for i in range(0, stereo.shape[1], TAILLE_TRAME):
            trame = av.AudioFrame.from_ndarray(np.ascontiguousarray(stereo[:, i:i + TAILLE_TRAME]), format="fltp",
                                               layout="stereo")
            trame.sample_rate = SR
            trame.pts = i
            trame.time_base = Fraction(1, SR)
            for p in out_a.encode(trame):
                out.mux(p)
        for p in out_a.encode(None):
            out.mux(p)
    return sortie


def generer(video, source=TEXTE, voix=None, langue="Français", texte="", audio=None, garder_son=True,
            volume_origine=30, prolonger=True, nom="voix_off", exag=0.5, cfg_w=0.5, temp=0.8, graine=0,
            progress=gr.Progress()):
    """Vidéo + voix off → MP4. Renvoie (message, vidéo, fichiers, dossier)."""
    video = chemin_fichier(video)
    if not video or not Path(str(video)).is_file():
        raise gr.Error("Ajoute la vidéo (fichier MP4, MOV, WebM…), ou choisis une vidéo créée ici.")
    video = Path(str(video))
    if video.suffix.lower() not in VIDEOS:
        raise gr.Error(f"Format vidéo non pris en charge ({video.suffix}) : MP4, MOV, MKV, WebM, AVI.")
    if source == TEXTE and not (texte or "").strip():
        raise gr.Error("Écris le texte que la voix doit lire.")
    debut = time.monotonic()
    nom = _slug(nom, "voix_off")
    dossier = nouveau_dossier(cfg.VIDEOS_DIR, nom)
    progress(0.02, desc="Lecture de la vidéo…")
    try:
        duree, cadence, largeur, hauteur, avec_son = _infos_video(video)
    except Exception as e:  # noqa: BLE001 - fichier illisible
        raise gr.Error(f"Impossible de lire la vidéo « {video.name} » ({e}).") from e
    if duree > DUREE_MAX:
        raise gr.Error(f"Vidéo trop longue ({duree:.0f} s) : {DUREE_MAX} s au plus.")
    progress(0.1, desc="Voix off…")
    y, desc = _voix(source, voix, langue, texte, audio, (exag, cfg_w, temp, graine), dossier, progress)
    duree_voix = len(y) / SR
    origine = None
    if garder_son and avec_son:
        progress(0.4, desc="Son d'origine…")
        try:
            origine, _ = charger(video, sr=SR, mono=True)
        except Exception:  # noqa: BLE001 - piste audio illisible : vidéo traitée comme muette
            origine = None
    prolonger_s = max(0.0, duree_voix - duree) if prolonger else 0.0
    duree_finale = duree + prolonger_s
    mix = mixer(y, origine, duree_finale, (volume_origine or 0) / 100.0)
    progress(0.5, desc="Assemblage de la vidéo et de la voix…")
    sortie = _ecrire(video, mix, dossier / "video_voix_off.mp4", prolonger_s, progress)
    copie = dossier / f"original{video.suffix.lower()}"
    shutil.copy(video, copie)
    temps = time.monotonic() - debut
    ecrire_creation(dossier, {
        "type": "video", "nom": nom, "description": (texte or "").strip()[:200] or "voix off",
        "voix_off": {**desc, "garder_son": bool(garder_son and origine is not None), "volume_origine": volume_origine,
                     "prolonger": bool(prolonger), "prolonge_s": round(prolonger_s, 2), "original": copie.name},
        "duree": round(duree_finale, 2), "largeur": largeur, "hauteur": hauteur, "temps_s": round(temps),
        "versions": [{"graine": int(graine or 0), "dossier": ".", "fichier": str(sortie)}],
    })
    note = ""
    if duree_voix > duree and not prolonger:
        note = f" ; la voix ({duree_voix:.1f} s) dépasse la vidéo ({duree:.1f} s) : elle est coupée à la fin"
    elif prolonger_s > 0:
        note = f" ; vidéo prolongée de {prolonger_s:.1f} s sur sa dernière image pour finir la voix"
    if garder_son and not avec_son:
        note += " ; la vidéo n'avait pas de son à garder"
    msg = (f"✅ Vidéo avec voix off ({duree_finale:.1f} s) en {temps:.0f} s{note}. Dossier : {dossier}")
    fichiers = [str(sortie), str(dossier / "voix.wav")]
    return msg, str(sortie), fichiers, str(dossier)
