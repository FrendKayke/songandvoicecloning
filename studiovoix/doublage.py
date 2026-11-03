"""Traduire et doubler une vidéo (groupe Voix, onglet « 🌍 Traduire et doubler ») : la personne de la vidéo parle
dans une autre langue, avec sa propre voix (clonée) ou une voix de la bibliothèque.

1. Demucs sépare la voix du reste (musique, ambiance), qui est gardé sous le doublage.
2. Whisper transcrit la voix en phrases horodatées (`diffusion.transcrire`).
3. Qwen3-VL traduit les phrases, par listes numérotées (mode « traduction », longueur proche de l'original).
4. Chatterbox relit chaque phrase traduite (`chatterbox.synthese_lot`, la voix est analysée une fois) : voix de la
   personne = ses 10 secondes les plus sonores (`seedvc.reference_courte`), ou une voix de la bibliothèque.
5. Chaque phrase est posée au moment de l'originale ; trop longue pour sa place, elle est accélérée (jusqu'à
   ×`ACCELERATION_MAX`, sans changer la hauteur de la voix) puis, si besoin, décalée après la précédente.
6. Vidéo réécrite avec le nouveau son (`voix_off._ecrire` : image recopiée, figée à la fin si le doublage dépasse).
Rangement : data/videos/<horodatage>_<nom>/ (creation.json type `video` avec `doublage`).
"""
import json
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from . import chatterbox, config as cfg, demucs, diffusion
from .audio import charger
from .convertir import _taille, duree_lisible, infos
from .images import scenes_du_texte
from .outils import chemin_fichier, ecrire_creation, nouveau_dossier
from .serie import _slug
from .sous_titres import LANGUE_AUTO, LANGUES as LANGUES_SOURCE, _horodatage
from .voix import chemin_voix

SR = 44100
SA_VOIX = "Sa propre voix (clonée depuis la vidéo)"
LANGUES_EN = {"fr": "French", "en": "English", "es": "Spanish", "it": "Italian", "de": "German", "pt": "Portuguese",
              "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
TRADUCTION_PAQUET = 15
ACCELERATION_MAX = 1.3  # au-delà, la voix devient trop pressée : la phrase déborde plutôt sur le silence suivant
ECART_MIN = 0.08  # s entre deux phrases doublées
DUREE_MAX = 15 * 60


def traduire(phrases, langue_code, progress=None):
    """Phrases → phrases traduites (Qwen, listes numérotées). Un paquet rendu avec un mauvais nombre de lignes est
    retraduit phrase par phrase. Renvoie (traductions, nombre de phrases gardées telles quelles)."""
    langue = LANGUES_EN[langue_code]
    sortie, gardees = [], 0

    def une_liste(paquet):
        texte = f"Number of lines: {len(paquet)}\n\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(paquet, 1))
        return scenes_du_texte(diffusion.decrire("traduction", texte, nombre=len(paquet), langue_cible=langue))

    for p0 in range(0, len(phrases), TRADUCTION_PAQUET):
        paquet = phrases[p0:p0 + TRADUCTION_PAQUET]
        if progress:
            progress(0.3 + 0.2 * p0 / len(phrases), desc=f"Traduction des phrases {p0 + 1}–{p0 + len(paquet)} sur "
                                                         f"{len(phrases)}…")
        lignes = une_liste(paquet)
        if len(lignes) != len(paquet) or not all(lignes):
            lignes = []
            for p in paquet:
                une = une_liste([p])
                if une and une[0]:
                    lignes.append(une[0])
                else:
                    lignes.append(p)
                    gardees += 1
        sortie += lignes
    return sortie, gardees


def placer(segments, voix, duree_video, accelerer=True):
    """Chaque voix doublée (mono, SR) posée au moment de sa phrase d'origine. Renvoie (piste, positions
    [(debut, fin)], nombre de phrases accélérées, nombre de phrases décalées)."""
    import librosa

    positions, morceaux, accelerees, decalees = [], [], 0, 0
    fin_prec = 0.0
    for k, (seg, y) in enumerate(zip(segments, voix)):
        suivant = segments[k + 1]["debut"] if k + 1 < len(segments) else max(duree_video, seg["fin"])
        place = max(0.2, suivant - seg["debut"] - ECART_MIN)
        d = len(y) / SR
        if accelerer and d > place:
            taux = min(ACCELERATION_MAX, d / place)
            y = librosa.effects.time_stretch(np.asarray(y, np.float32), rate=taux)
            accelerees += 1
        debut = seg["debut"]
        if debut < fin_prec + ECART_MIN and k:
            if fin_prec + ECART_MIN - debut > 0.02:  # quelques ms de trop après l'accélération : pas un décalage
                decalees += 1
            debut = fin_prec + ECART_MIN
        positions.append((debut, debut + len(y) / SR))
        morceaux.append(y)
        fin_prec = debut + len(y) / SR
    total = max([duree_video] + [f for _, f in positions])
    piste = np.zeros(int(np.ceil(total * SR)) + 1, np.float32)
    for (debut, _), y in zip(positions, morceaux):
        i = int(round(debut * SR))
        piste[i:i + len(y)] += y[: len(piste) - i]
    return piste, positions, accelerees, decalees


def srt(textes, positions):
    return "\n".join(f"{i}\n{_horodatage(d, ',')} --> {_horodatage(f, ',')}\n{t}\n"
                     for i, (t, (d, f)) in enumerate(zip(textes, positions), 1))


def _video_de(dossier):
    from .galerie import lire

    d = lire(dossier) if dossier else None
    return (d.get("versions") or [{}])[0].get("fichier") if d else None


def doubler(fichier, creation=None, langue_source=LANGUE_AUTO, langue_cible="Anglais", voix=SA_VOIX,
            garder_fond=True, volume_fond=100, accelerer=True, exaggeration=0.5, cfg_weight=0.5, temperature=0.8,
            graine=0, nom="doublage", progress=gr.Progress()):
    """Renvoie (message, vidéo ou None, texte « original → traduction », [fichiers], dossier)."""
    source = chemin_fichier(fichier) or _video_de(creation)
    if not source or not Path(str(source)).is_file():
        raise gr.Error("Ajoute une vidéo (ou un enregistrement) à doubler.")
    source = Path(str(source))
    code = cfg.LANGUES.get(langue_cible)
    if code not in LANGUES_EN:
        raise gr.Error("Choisis la langue du doublage.")
    d = infos(source)
    if d["duree"] > DUREE_MAX:
        raise gr.Error(f"{duree_lisible(d['duree'])} : {duree_lisible(DUREE_MAX)} au plus (coupe d'abord le passage "
                       "utile dans « Audio / vidéo → Couper »).")
    if not d["son"]:
        raise gr.Error(f"« {source.name} » n'a pas de son : rien à traduire.")
    reference = None if voix == SA_VOIX else chemin_voix(voix)
    t0 = time.monotonic()
    dossier = nouveau_dossier(cfg.VIDEOS_DIR, _slug(nom or source.stem, "doublage"))
    progress(0.02, desc="Lecture du son…")
    y, _ = charger(str(source), sr=SR, mono=True)
    original = dossier / "son_original.wav"
    sf.write(str(original), y, SR)
    progress(0.05, desc="Séparation de la voix et du fond (Demucs)…")
    voix_isolee, fond = demucs.separate_vocals(original, dossier)
    v16, _ = charger(str(voix_isolee), sr=16000, mono=True)
    wav16 = dossier / "voix_16k.wav"
    sf.write(str(wav16), v16, 16000, subtype="PCM_16")
    progress(0.15, desc="Transcription (Whisper)…")
    r = diffusion.transcrire(wav16, LANGUES_SOURCE.get(langue_source), progress=progress)
    segments = [s for s in r.get("segments") or [] if s["texte"].strip()]
    if not segments:
        raise gr.Error("Aucune parole reconnue dans cette vidéo.")
    phrases = [s["texte"].strip() for s in segments]
    traductions, gardees = traduire(phrases, code, progress)
    if reference is None:  # la voix de la personne : ses 10 s les plus sonores (Chatterbox n'en lit que 10)
        from .seedvc import reference_courte

        reference = reference_courte(Path(voix_isolee), dossier, duree=10)
    progress(0.55, desc="Lecture des phrases traduites (Chatterbox)…")
    fichiers_voix = chatterbox.synthese_lot(reference, traductions, code, dossier / "phrases", exaggeration,
                                            cfg_weight, temperature, graine, progress=progress)
    progress(0.88, desc="Mise en place du doublage…")
    voix_doublees = [charger(str(f), sr=SR, mono=True)[0] for f in fichiers_voix]
    piste, positions, accelerees, decalees = placer(segments, voix_doublees, d["duree"], accelerer)
    duree_finale = len(piste) / SR
    if garder_fond and volume_fond:
        f, _ = charger(str(fond), sr=SR, mono=True)
        n = min(len(f), len(piste))
        piste[:n] += f[:n] * (float(volume_fond) / 100)
    pic = float(np.abs(piste).max()) if len(piste) else 0.0
    if pic > 0.98:
        piste *= 0.98 / pic
    voix_seule = dossier / f"voix_{code}.wav"
    sf.write(str(voix_seule), np.clip(piste, -1, 1), SR)
    (dossier / f"sous_titres_{code}.srt").write_text(srt(traductions, positions), encoding="utf-8-sig")
    texte = "\n\n".join(f"{p}\n→ {t}" for p, t in zip(phrases, traductions))
    (dossier / "traduction.txt").write_text(texte + "\n", encoding="utf-8")
    (dossier / "phrases.json").write_text(json.dumps(
        [{"debut": s["debut"], "fin": s["fin"], "original": p, "traduction": t, "place": list(pos)}
         for s, p, t, pos in zip(segments, phrases, traductions, positions)], ensure_ascii=False, indent=1),
        encoding="utf-8")
    video = None
    if d["video"]:
        from .voix_off import _ecrire

        progress(0.93, desc="Écriture de la vidéo doublée…")
        video = _ecrire(source, piste, dossier / f"video_{code}.mp4",
                        max(0.0, duree_finale - d.get("duree_video", d["duree"])), progress)
    for f in (original, wav16):
        f.unlink(missing_ok=True)
    temps = time.monotonic() - t0
    resultat = video or voix_seule
    ecrire_creation(dossier, {
        "type": "video" if video else "tts", "nom": nom, "description": f"doublage en {langue_cible} : {source.name}",
        "doublage": {"original": source.name, "langue": langue_cible, "voix": voix, "phrases": len(phrases),
                     "garder_fond": bool(garder_fond)},
        "texte": texte[:2000], "duree": round(duree_finale, 2), "temps_s": round(temps),
        "versions": [{"graine": int(graine or 0), "dossier": ".", "fichier": str(resultat)}],
    })
    msg = (f"✅ Doublé en {langue_cible} : {len(phrases)} phrases, {duree_lisible(duree_finale)}, en "
           f"{duree_lisible(temps)} : `{resultat}` ({_taille(Path(resultat).stat().st_size)}).")
    notes = []
    if accelerees:
        notes.append(f"{accelerees} phrase(s) un peu accélérée(s) pour tenir à leur place")
    if decalees:
        notes.append(f"{decalees} phrase(s) décalée(s) après la précédente")
    if duree_finale > d["duree"] + 0.05 and video:
        notes.append(f"dernière image figée {duree_finale - d['duree']:.1f} s pour finir la voix")
    if gardees:
        notes.append(f"⚠️ {gardees} phrase(s) non traduite(s) (réponse incomplète de Qwen3-VL), lue(s) telle(s) quelle(s)")
    if notes:
        msg += "\n\n" + " ; ".join(notes) + "."
    fichiers = [str(p) for p in ([video] if video else []) + [voix_seule, dossier / f"sous_titres_{code}.srt",
                                                              dossier / "traduction.txt"]]
    return msg, str(video) if video else None, texte, fichiers, str(dossier)
