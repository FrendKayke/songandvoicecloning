"""Transcrire et sous-titrer (onglet « Outils → Transcrire et sous-titrer »).

Un fichier audio ou vidéo (tout format lu par audio.charger : wav, mp3, m4a, mp4, webm…) est converti en WAV 16 kHz
mono, transcrit par Whisper large-v3-turbo (moteur de diffusion, action « transcrire »), puis :
  - texte complet (.txt), sous-titres .srt (lecteurs vidéo, montage) et .vtt (navigateurs) ;
  - pour une vidéo, sur demande, une copie avec les sous-titres incrustés dans l'image (PyAV + Pillow, même son).
Les segments de Whisper (souvent 5 à 15 s) sont redécoupés en sous-titres lisibles : 2 lignes de 42 caractères au
plus, durée répartie selon le nombre de caractères. Sortie : data/transcriptions/<horodatage>_<nom>/.
"""
import time
from fractions import Fraction
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from . import config as cfg
from . import diffusion
from .audio import charger
from .outils import nouveau_dossier, chemin_fichier
from .serie import _slug

CARACTERES_LIGNE = 42
LIGNES = 2
LANGUE_AUTO = "Détection automatique"
LANGUES = {LANGUE_AUTO: None, **{lib: code for lib, code in cfg.LANGUES.items()}}
VIDEOS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


# --- Découpage et formats ---------------------------------------------------------------------------------------
def _lignes(mots):
    """Mots répartis en lignes de CARACTERES_LIGNE caractères au plus (un mot plus long reste seul sur sa ligne)."""
    lignes, courante = [], ""
    for mot in mots:
        if courante and len(courante) + 1 + len(mot) > CARACTERES_LIGNE:
            lignes.append(courante)
            courante = mot
        else:
            courante = f"{courante} {mot}" if courante else mot
    if courante:
        lignes.append(courante)
    return lignes


def decouper(segments):
    """Segments de Whisper → sous-titres [{debut, fin, texte}] de LIGNES lignes de CARACTERES_LIGNE caractères au
    plus ; la durée d'un segment est partagée entre ses sous-titres selon leur longueur."""
    sortie = []
    for seg in segments:
        lignes = _lignes(seg["texte"].split())
        if not lignes:
            continue
        paquets = ["\n".join(lignes[i:i + LIGNES]) for i in range(0, len(lignes), LIGNES)]
        total = sum(len(p) for p in paquets)
        debut, duree = float(seg["debut"]), max(0.3, float(seg["fin"]) - float(seg["debut"]))
        for p in paquets:
            fin = debut + duree * len(p) / total
            sortie.append({"debut": round(debut, 3), "fin": round(fin, 3), "texte": p})
            debut = fin
    return sortie


def _horodatage(secondes, separateur):
    ms = int(round(secondes * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{separateur}{ms:03d}"


def srt(sous_titres):
    return "\n".join(f"{i}\n{_horodatage(s['debut'], ',')} --> {_horodatage(s['fin'], ',')}\n{s['texte']}\n"
                     for i, s in enumerate(sous_titres, 1))


def vtt(sous_titres):
    return "WEBVTT\n\n" + "\n".join(f"{_horodatage(s['debut'], '.')} --> {_horodatage(s['fin'], '.')}\n{s['texte']}\n"
                                    for s in sous_titres)


# --- Incrustation dans une vidéo --------------------------------------------------------------------------------
def _police(taille):
    from PIL import ImageFont

    for nom in ("segoeuib.ttf", "arialbd.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans-Bold.ttf",
                "LiberationSans-Bold.ttf"):
        for dossier in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"),
                        Path("/usr/share/fonts/truetype/liberation")):
            if (dossier / nom).is_file():
                return ImageFont.truetype(str(dossier / nom), taille)
    return ImageFont.load_default(size=taille)


def _dessiner(image, texte, police):
    """Sous-titre centré en bas : texte blanc cerné de noir (lisible sur tout fond)."""
    from PIL import ImageDraw

    dessin = ImageDraw.Draw(image)
    marge, contour = image.height // 18, max(2, police.size // 12)
    boite = dessin.multiline_textbbox((0, 0), texte, font=police, align="center", stroke_width=contour, spacing=6)
    x = (image.width - (boite[2] - boite[0])) / 2
    y = image.height - marge - (boite[3] - boite[1])
    dessin.multiline_text((x, y), texte, font=police, fill="white", align="center", stroke_width=contour,
                          stroke_fill="black", spacing=6)
    return image


def incruster(video, sous_titres, sortie, progress=None):
    """Copie de la vidéo avec les sous-titres dessinés dans l'image (H.264, son d'origine recopié tel quel)."""
    import av

    with av.open(str(video)) as entree, av.open(str(sortie), "w") as out:
        flux_v = entree.streams.video[0]
        flux_a = entree.streams.audio[0] if entree.streams.audio else None
        cadence = flux_v.average_rate or Fraction(25)
        largeur, hauteur = flux_v.codec_context.width // 2 * 2, flux_v.codec_context.height // 2 * 2
        out_v = out.add_stream("libx264", rate=cadence)
        out_v.width, out_v.height, out_v.pix_fmt = largeur, hauteur, "yuv420p"
        out_v.options = {"crf": "18", "preset": "medium"}
        out_a = out.add_stream_from_template(flux_a) if flux_a else None
        police = _police(max(16, hauteur // 18))
        total = float(flux_v.duration * flux_v.time_base) if flux_v.duration else None
        flux = [flux_v] + ([flux_a] if flux_a else [])
        n = 0
        for paquet in entree.demux(*flux):
            if paquet.stream is flux_a:
                if paquet.dts is not None:
                    paquet.stream = out_a
                    out.mux(paquet)
                continue
            for trame in paquet.decode():
                t = float(trame.time or 0.0)
                image = trame.to_image()
                if image.size != (largeur, hauteur):
                    image = image.crop((0, 0, largeur, hauteur))
                actif = next((s for s in sous_titres if s["debut"] <= t < s["fin"]), None)
                if actif:
                    image = _dessiner(image, actif["texte"], police)
                nouvelle = av.VideoFrame.from_image(image)
                nouvelle.pts, nouvelle.time_base = trame.pts, trame.time_base
                for p in out_v.encode(nouvelle):
                    out.mux(p)
                n += 1
                if progress and total and n % 25 == 0:
                    progress(0.8 + 0.2 * min(1.0, t / total), desc=f"Incrustation des sous-titres ({t:.0f} s)…")
        for p in out_v.encode(None):
            out.mux(p)
    return sortie


# --- Onglet -----------------------------------------------------------------------------------------------------
def transcrire(fichier, langue_label=LANGUE_AUTO, traduire=False, incruster_video=True, progress=gr.Progress()):
    """Renvoie (message, texte, fichiers [txt, srt, vtt, vidéo sous-titrée], vidéo sous-titrée ou None, dossier)."""
    source = Path(str(chemin_fichier(fichier))) if fichier else None
    if not source or not source.is_file():
        raise gr.Error("Ajoute un fichier audio ou vidéo à transcrire.")
    debut = time.monotonic()
    dossier = nouveau_dossier(cfg.TRANSCRIPTIONS_DIR, _slug(source.stem, "transcription"))
    progress(0.02, desc="Lecture du son…")
    try:
        y, _ = charger(source, sr=16000, mono=True)
    except Exception as e:  # noqa: BLE001 - format illisible, vidéo sans son
        raise gr.Error(f"Impossible de lire le son de « {source.name} » ({e}).") from e
    if not len(y) or float(np.max(np.abs(y))) < 1e-4:
        raise gr.Error(f"« {source.name} » ne contient pas de son.")
    wav = dossier / "audio_16k.wav"
    sf.write(str(wav), y, 16000, subtype="PCM_16")
    resultat = diffusion.transcrire(wav, LANGUES.get(langue_label), traduire, progress=progress)
    sous_titres = decouper(resultat["segments"])
    if not sous_titres:
        raise gr.Error("Aucune parole reconnue dans ce fichier.")
    texte = resultat.get("texte") or " ".join(s["texte"] for s in resultat["segments"])
    nom = _slug(source.stem, "transcription")
    fichiers = [dossier / f"{nom}.txt", dossier / f"{nom}.srt", dossier / f"{nom}.vtt"]
    fichiers[0].write_text(texte + "\n", encoding="utf-8")
    fichiers[1].write_text(srt(sous_titres), encoding="utf-8-sig")  # BOM : accents bien lus par les lecteurs Windows
    fichiers[2].write_text(vtt(sous_titres), encoding="utf-8")
    video = None
    if incruster_video and source.suffix.lower() in VIDEOS:
        progress(0.8, desc="Incrustation des sous-titres…")
        video = incruster(source, sous_titres, dossier / f"{nom}_sous-titres.mp4", progress)
        fichiers.append(video)
    msg = (f"✅ {len(sous_titres)} sous-titres pour {resultat['duree']:.0f} s de son, en "
           f"{time.monotonic() - debut:.0f} s, dans {dossier}. Le fichier .srt s'ouvre avec la vidéo dans VLC ou un "
           "logiciel de montage ; le .vtt sert aux pages web.")
    return msg, texte, [str(f) for f in fichiers], str(video) if video else None, str(dossier)
