"""Bibliothèque de voix : data/voices/<nom>.wav (44,1 kHz mono, 30 s au plus, normalisé).

Import depuis un fichier (wav, mp3, flac, m4a…) ou le micro, avec un contrôle de qualité simple
(durée, volume trop faible, saturation), écoute, renommage et suppression.
"""
from dataclasses import dataclass, field

import gradio as gr
import librosa
import numpy as np
import soundfile as sf

from . import config as cfg
from .audio import charger

DUREE_MIN = 5       # s : en dessous, refus
DUREE_CONSEILLEE = 10  # s : en dessous, avertissement
DUREE_MAX = 30      # s : Seed-VC exploite 1 à 30 s de référence, le reste est coupé
NIVEAU_REFUS = -40  # dBFS : niveau de la voix en dessous duquel on refuse
NIVEAU_FAIBLE = -30  # dBFS : en dessous, avertissement
SEUIL_SATURATION = 0.99  # amplitude considérée comme saturée
PART_SATUREE_MAX = 0.001  # 0,1 % des échantillons
# Part de l'énergie au-dessus de 4 kHz en dessous de laquelle le son est jugé étouffé : un message vocal WhatsApp
# (compressé) n'en avait que 0,18 % (« s », « ch », souffle et timbre retirés) et la voix clonée sortait étouffée
# (rapport de l'utilisateur du 09/10) ; une voix enregistrée en direct au micro en a nettement plus
PART_AIGUS_MIN = 0.003


def list_voices():
    return sorted(p.stem for p in cfg.VOICES_DIR.glob("*.wav"))


def nettoyer_nom(name):
    return "".join(c for c in (name or "").strip() if c.isalnum() or c in "-_ ").strip()


def _existe_deja(name, sauf=None):
    """Windows ne distingue pas majuscules et minuscules dans les noms de fichiers."""
    return any(v.lower() == name.lower() for v in list_voices() if v != sauf)


def chemin_voix(name):
    """Chemin d'une voix existante de la bibliothèque (refuse tout nom qui n'y est pas)."""
    if not name or name not in list_voices():
        raise gr.Error(f"Voix introuvable : {name or '(aucune)'}. Choisis une voix dans la liste.")
    return cfg.VOICES_DIR / f"{name}.wav"


# --- Contrôle de qualité ------------------------------------------------------
@dataclass
class Analyse:
    duree: float
    niveau_db: float          # niveau de la voix (95e centile du RMS sur 50 ms), en dBFS
    part_saturee: float       # proportion d'échantillons à |x| >= 0,99
    erreurs: list = field(default_factory=list)
    avertissements: list = field(default_factory=list)
    infos: list = field(default_factory=list)


def analyser(y, sr) -> Analyse:
    """Contrôle de qualité d'un signal mono brut (avant normalisation)."""
    y = np.asarray(y, dtype="float32")
    duree = len(y) / sr if sr else 0.0
    trame = max(1, int(0.05 * sr))
    if len(y) >= trame:
        n = len(y) // trame
        rms = np.sqrt(np.mean(y[: n * trame].reshape(n, trame) ** 2, axis=1))
        niveau = float(np.percentile(rms, 95))
    else:
        niveau = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    niveau_db = 20 * np.log10(max(niveau, 1e-10))
    part_saturee = float(np.mean(np.abs(y) >= SEUIL_SATURATION)) if len(y) else 0.0
    a = Analyse(duree, float(niveau_db), part_saturee)

    if duree < DUREE_MIN:
        a.erreurs.append(
            f"Échantillon trop court ({duree:.1f} s) : il faut au moins {DUREE_MIN} s, idéalement 10 à 25 s."
        )
    elif duree < DUREE_CONSEILLEE:
        a.avertissements.append(
            f"Échantillon un peu court ({duree:.1f} s) : 10 à 25 s donnent une voix plus ressemblante."
        )
    elif duree > DUREE_MAX:
        a.infos.append(f"Échantillon de {duree:.0f} s : seules les {DUREE_MAX} premières secondes sont gardées.")

    if niveau_db < NIVEAU_REFUS:
        a.erreurs.append(
            f"Volume beaucoup trop faible ({niveau_db:.0f} dBFS) : on n'entend presque rien. "
            "Vérifie le micro choisi, rapproche-toi (15 à 30 cm) et recommence."
        )
    elif niveau_db < NIVEAU_FAIBLE:
        a.avertissements.append(
            f"Volume faible ({niveau_db:.0f} dBFS) : le souffle du micro risque d'être amplifié. "
            "Rapproche-toi du micro ou monte son niveau d'entrée."
        )

    aigus = _part_aigus(y, sr)
    if aigus is not None and aigus < PART_AIGUS_MIN and niveau_db >= NIVEAU_REFUS:
        part = f"{100 * aigus:.2f}".replace(".", ",")
        a.avertissements.append(
            f"Son étouffé (seulement {part} % de l'énergie au-dessus de 4 kHz) : on dirait un enregistrement "
            "compressé (message vocal WhatsApp, appel, vidéo) ou un micro de mauvaise qualité. La voix clonée "
            "reprendra ce son étouffé : enregistre-toi plutôt directement ici avec le micro du PC ou d'un casque, "
            "ou importe le fichier d'origine (WAV) d'un enregistreur."
        )

    if part_saturee > PART_SATUREE_MAX:
        a.avertissements.append(
            f"Son saturé ({part_saturee:.1%} des échantillons au maximum) : la voix sera déformée. "
            "Éloigne-toi un peu du micro ou baisse son niveau d'entrée, puis recommence."
        )
    return a


def _part_aigus(y, sr):
    """Part de l'énergie au-dessus de 4 kHz (None si le fichier ne peut pas en contenir : fréquence trop basse)."""
    if sr < 16000 or len(y) < sr:
        return None
    spectre = np.abs(np.fft.rfft(y.astype("float64"))) ** 2
    total = spectre.sum()
    if total <= 0:
        return None
    frequences = np.fft.rfftfreq(len(y), 1 / sr)
    return float(spectre[frequences >= 4000].sum() / total)


def _charger(audio_path):
    try:
        y, sr = charger(audio_path, sr=None, mono=True)  # fréquence d'origine ; m4a/AAC par PyAV
    except Exception:
        raise gr.Error("Fichier illisible. Utilise un fichier audio wav, mp3, flac, m4a ou ogg.")
    return y, sr


def rapport(a: Analyse) -> str:
    lignes = [f"⚠️ {m}" for m in a.avertissements] + [f"ℹ️ {m}" for m in a.infos]
    return "\n\n".join(lignes)


# --- Opérations de la bibliothèque -----------------------------------------------
def save_voice(audio_path, name):
    """Importe une voix (fichier ou micro) après contrôle de qualité. Renvoie (message, mise à jour de la liste)."""
    if not audio_path:
        raise gr.Error("Enregistre ou importe d'abord un échantillon de voix.")
    name = nettoyer_nom(name)
    if not name:
        raise gr.Error("Donne un nom à cette voix (lettres, chiffres, espaces, - et _).")
    if _existe_deja(name):
        raise gr.Error(f"Une voix s'appelle déjà « {name} ». Choisis un autre nom, ou supprime l'ancienne voix.")
    y, sr = _charger(audio_path)
    a = analyser(y, sr)
    if a.erreurs:
        raise gr.Error("Voix non enregistrée.\n" + "\n".join(a.erreurs))
    if sr != cfg.SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.SR)
    y = y[: DUREE_MAX * cfg.SR]
    peak = float(np.max(np.abs(y))) or 1.0
    y = y / peak * 0.95
    sf.write(cfg.VOICES_DIR / f"{name}.wav", y, cfg.SR)
    msg = f"✅ Voix « {name} » enregistrée ({min(a.duree, DUREE_MAX):.0f} s utilisées)."
    if rapport(a):
        msg += "\n\n" + rapport(a)
    return msg, gr.update(choices=list_voices(), value=name)


GARDER_ORIGINAL = "Original"
GARDER_NETTOYEE = "Version nettoyée"


def save_voice_choix(original, nettoyee, garder, name):
    """Enregistre l'original ou la version nettoyée, selon le choix de l'utilisateur."""
    if garder == GARDER_NETTOYEE:
        if not nettoyee:
            raise gr.Error("Pas encore de version nettoyée : clique d'abord sur « Nettoyer », ou garde l'original.")
        return save_voice(nettoyee, name)
    return save_voice(original, name)


def infos_voix(name):
    """Fichier à écouter et description de la voix choisie."""
    if not name or name not in list_voices():
        return None, ""
    p = cfg.VOICES_DIR / f"{name}.wav"
    info = sf.info(str(p))
    return str(p), f"**{name}** — {info.duration:.1f} s, {info.samplerate} Hz"


def rename_voice(old, new):
    src = chemin_voix(old)
    new = nettoyer_nom(new)
    if not new:
        raise gr.Error("Donne le nouveau nom (lettres, chiffres, espaces, - et _).")
    if new == old:
        return f"La voix s'appelle déjà « {new} ».", gr.update(choices=list_voices(), value=new)
    if _existe_deja(new, sauf=old):
        raise gr.Error(f"Une voix s'appelle déjà « {new} ».")
    try:
        src.rename(cfg.VOICES_DIR / f"{new}.wav")
    except OSError as e:
        raise gr.Error(f"Impossible de renommer la voix : le fichier est peut-être utilisé par un autre programme. ({e})")
    return f"✅ Voix « {old} » renommée en « {new} ».", gr.update(choices=list_voices(), value=new)


def delete_voice(name):
    if name is None:  # suppression annulée dans la fenêtre de confirmation
        return "Suppression annulée.", gr.update()
    p = chemin_voix(name)
    try:
        p.unlink()
    except OSError as e:
        raise gr.Error(f"Impossible de supprimer la voix : le fichier est peut-être utilisé par un autre programme. ({e})")
    voices = list_voices()
    return f"🗑️ Voix « {name} » supprimée.", gr.update(choices=voices, value=voices[0] if voices else None)
