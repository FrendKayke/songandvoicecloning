"""Remixer une musique dans un autre style (un thème de jeu vidéo en version orchestrale…), avec ACE-Step 1.5.

Tâche « cover » d'ACE-Step (vérifiée dans acestep/api/http/release_task_models.py et la documentation) : le morceau
envoyé (src_audio) guide la génération, la description donne le nouveau style. Deux réglages :
  - audio_cover_strength (0–1) : part des étapes qui suivent la structure du morceau (mélodie, rythme, accords) ;
    plus haut = plus proche de l'original ;
  - cover_noise_strength (0–1) : la génération part du morceau d'origine légèrement bruité au lieu d'un bruit pur ;
    plus haut = mélodie mieux gardée (le mode « Remix » de l'interface officielle le met à 0,2).
« cover-nofsq » envoie le morceau sans le quantifier : plus fidèle, garde aussi davantage le son d'origine.
La durée est celle du morceau (600 s au plus, DURATION_MAX d'ACE-Step). Le modèle de langage n'intervient pas.
Recherche du 07/10/2026 (usage personnel, licences non commerciales acceptées) : rien de mieux qu'ACE-Step pour
garder la mélodie sur 12 Go (MuLaCover : Linux seulement, pensé pour les reprises chantées ; MusicGen-Melody : 30 s,
qualité inférieure ; Stable Audio Open : 47 s, mélodie mal gardée).
Chaque morceau : data/remix/<horodatage>/ avec original.wav, remix_<v>.wav, creation.json type « remix ».
"""
import shutil
import time
from pathlib import Path

import gradio as gr
import numpy as np

from . import acestep
from . import config as cfg
from .outils import ecrire_creation, nouveau_dossier
from .styles import musique
from .videos import duree_lisible

DUREE_MAX = 600  # s, DURATION_MAX d'ACE-Step
INSTRUMENTAL = "[Instrumental]"
# Styles d'arrivée (libellé → description anglaise), multiselect + saisie libre
STYLES = [
    ("Orchestral symphonique (épique)", "epic symphonic orchestra arrangement, full string section, French horns, "
                                        "brass, timpani, cinematic, orchestral"),
    ("Orchestre de chambre / quatuor à cordes", "string quartet arrangement, chamber music, violins, viola, cello, "
                                                "intimate, classical"),
    ("Piano solo", "solo piano arrangement, expressive grand piano, classical piano"),
    ("Rock / métal", "heavy metal arrangement, distorted electric guitars, powerful drums, bass guitar, energetic"),
    ("Jazz", "jazz arrangement, upright bass, brushed drums, piano, saxophone, swing feel"),
    ("Big band / swing", "big band swing arrangement, brass section, saxophones, walking bass, swing drums"),
    ("Lo-fi hip-hop", "lo-fi hip hop arrangement, dusty drums, warm Rhodes piano, vinyl crackle, chill"),
    ("8-bit / chiptune", "8-bit chiptune arrangement, square wave synth, NES style, retro video game music"),
    ("Synthwave / années 80", "synthwave arrangement, analog synthesizers, gated reverb drums, retro 80s"),
    ("Électro / dance", "electronic dance arrangement, four on the floor kick, synth bass, energetic EDM"),
    ("Guitare acoustique", "acoustic guitar arrangement, fingerstyle guitar, warm, intimate"),
    ("Celtique / folk", "celtic folk arrangement, tin whistle, fiddle, bodhran, acoustic guitar"),
    ("Médiéval / taverne", "medieval tavern arrangement, lute, hurdy-gurdy, recorder, frame drum, early music"),
    ("Reggae", "reggae arrangement, offbeat guitar skank, deep bass, one drop drums"),
    ("Musique de film (trailer)", "epic trailer music arrangement, huge drums, choir pads, brass hits, cinematic"),
]
# Transformation : (audio_cover_strength, cover_noise_strength) — guide des musiciens d'ACE-Step : 0,3–0,5 pour un
# grand changement de genre, 0,5–0,7 pour un changement modéré ; mélodie : 0,1–0,25 conseillé
TRANSFORMATIONS = {
    "Légère (même morceau, autres instruments)": (0.7, 0.3),
    "Moyenne (conseillée)": (0.5, 0.2),
    "Forte (réinterprétation libre)": (0.3, 0.1),
}
TRANSFORMATION_DEFAUT = "Moyenne (conseillée)"


def reglages(transformation):
    """Curseurs « structure » et « mélodie » d'après la transformation choisie."""
    structure, melodie = TRANSFORMATIONS.get(transformation, TRANSFORMATIONS[TRANSFORMATION_DEFAUT])
    return gr.update(value=structure), gr.update(value=melodie)


def description(styles, instruments=None, ambiance=None, extra=None):
    """Description anglaise du nouveau style (« instrumental » ajouté : pas de voix inventée)."""
    termes = []
    for t in ", ".join(p for p in (musique(styles), musique(instruments), musique(ambiance), musique(extra), "instrumental")
                       if p and p.strip()).split(","):
        if t.strip() and t.strip().lower() not in (x.lower() for x in termes):
            termes.append(t.strip())
    return ", ".join(termes)


def _chemins(fichiers):
    chemins = [str(getattr(f, "name", f)) for f in (fichiers if isinstance(fichiers, (list, tuple)) else [fichiers])
               if f]
    if not chemins:
        raise gr.Error("Ajoute la musique à remixer (MP3, WAV, FLAC, OGG…).")
    for c in chemins:
        if not Path(c).is_file():
            raise gr.Error(f"Fichier introuvable : {c}")
    return chemins


def _preparer_original(source, dossier):
    """Copie le morceau en WAV 44,1 kHz stéréo (format sûr pour ACE-Step) ; renvoie (chemin, durée)."""
    import librosa
    import soundfile as sf

    try:
        audio, sr = librosa.load(source, sr=44100, mono=False)
    except Exception as e:  # noqa: BLE001 - format illisible
        raise gr.Error(f"Impossible de lire « {Path(source).name} » ({e}). Convertis-le en WAV ou MP3.") from e
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    duree = audio.shape[-1] / sr
    if duree < 5:
        raise gr.Error(f"« {Path(source).name} » est trop court ({duree:.1f} s).")
    if duree > DUREE_MAX:
        raise gr.Error(f"« {Path(source).name} » dure {duree_lisible(duree)} : {DUREE_MAX // 60} min au plus "
                       "(limite d'ACE-Step). Coupe-le d'abord.")
    original = dossier / "original.wav"
    sf.write(original, audio.T, sr)
    return original, duree


def remixer(fichiers, styles, instruments, ambiance, extra, desc, structure, melodie, tres_fidele, versions, graine,
            progress=gr.Progress()):
    """Remixe chaque morceau dans le nouveau style, l'un après l'autre. Renvoie (message, liste des résultats,
    premier résultat, son original)."""
    chemins = _chemins(fichiers)
    desc = (desc or "").strip() or description(styles, instruments, ambiance, extra)
    if desc == "instrumental":
        raise gr.Error("Choisis le nouveau style (par exemple « Orchestral symphonique »), ou écris la description.")
    n_versions = max(1, min(2, int(versions or 1)))
    debut, resultats, erreurs = time.monotonic(), [], []
    for k, source in enumerate(chemins, 1):
        nom = Path(source).stem
        etape = f"{k}/{len(chemins)} {nom}" if len(chemins) > 1 else nom
        try:
            dossier = nouveau_dossier(cfg.REMIX_DIR)
            original, duree = _preparer_original(source, dossier)
            sorties = _generer(dossier, original, duree, desc, structure, melodie, tres_fidele, n_versions, graine,
                               progress, etape)
        except gr.Error as e:  # un morceau en échec n'arrête pas les suivants
            erreurs.append(f"{nom} : {getattr(e, 'message', e)}")
            continue
        ecrire_creation(dossier, {
            "type": "remix", "nom": nom, "source": Path(source).name, "description": desc,
            "styles": styles, "structure": float(structure), "melodie": float(melodie),
            "tres_fidele": bool(tres_fidele), "duree": round(duree, 1),
            "versions": [{"graine": g, "dossier": ".", "fichier": str(f)} for f, g in sorties],
        })
        for v, (f, g) in enumerate(sorties, 1):
            resultats.append((f"{nom} — version {v} (graine {g})", str(f), str(original)))
    if not resultats:
        raise gr.Error("Aucun remix n'a pu être fait :\n" + "\n".join(erreurs))
    msg = (f"✅ {len(resultats)} remix en {duree_lisible(time.monotonic() - debut)}, dans {cfg.REMIX_DIR}. "
           "Choisis un résultat dans la liste pour l'écouter à côté de l'original.")
    if erreurs:
        msg += "\n\n⚠️ " + "\n\n⚠️ ".join(erreurs)
    choix = [(lib, f) for lib, f, _ in resultats]
    return msg, gr.update(choices=choix, value=choix[0][1]), choix[0][1], resultats[0][2]


def _generer(dossier, original, duree, desc, structure, melodie, tres_fidele, n_versions, graine, progress, etape):
    params = acestep.text2music_params(desc, INSTRUMENTAL, "en", round(duree, 1), 0, False)
    params.update(task_type="cover-nofsq" if tres_fidele else "cover",
                  audio_cover_strength=float(structure), cover_noise_strength=float(melodie))
    dests = [dossier / f"remix_{v}.wav" for v in range(1, n_versions + 1)]
    return acestep.generer(params, dests, progress, etape, {"src_audio": str(original)}, graine)


def original_de(resultat):
    """Original d'un remix (pour l'écouter à côté)."""
    if not resultat:
        return None
    o = Path(resultat).parent / "original.wav"
    return str(o) if o.exists() else None


def recreer(chemin, infos, graine, progress=gr.Progress()):
    """Galerie : même morceau, mêmes réglages, même graine."""
    original = Path(chemin) / "original.wav"
    if not original.exists():
        raise gr.Error(f"Morceau d'origine introuvable : {original}")
    dossier = nouveau_dossier(cfg.REMIX_DIR)
    shutil.copy(original, dossier / "original.wav")
    sorties = _generer(dossier, dossier / "original.wav", infos.get("duree") or 30, infos["description"],
                       infos.get("structure", 0.5), infos.get("melodie", 0.2), infos.get("tres_fidele"), 1, graine,
                       progress, infos.get("nom") or "remix")
    ecrire_creation(dossier, {**{k: v for k, v in infos.items() if k not in ("date", "versions")},
                              "versions": [{"graine": g, "dossier": ".", "fichier": str(f)} for f, g in sorties]})
    return dossier
