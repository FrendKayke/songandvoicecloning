"""Boucles parfaites pour les musiques de jeu (ACE-Step ne génère pas de boucles).

1. On repère les temps (librosa.beat.beat_track) ; les candidats sont les couples (début a, fin b) séparés
   d'un nombre entier de mesures (4 temps).
2. Au bouclage, après b on entend a : on choisit le couple où ce qui SUIT b ressemble le plus à ce qui SUIT a
   (chroma = harmonie, MFCC = timbre), en préférant les boucles longues.
3. On ajuste b à l'échantillon près (corrélation des formes d'onde), puis on fond la suite naturelle de b
   (y[b:b+F]) dans le début de la boucle : le fichier y[a:b] boucle alors sans rupture, dans n'importe quel
   lecteur (Howler.js « loop: true », <audio loop>, Web Audio).
4. On mesure la jonction : au bouclage on doit entendre la suite naturelle de b. Même une répétition
   musicale parfaite n'est jamais identique à l'échantillon près (percussions, interprétation) : ≈ 1 est
   donc aussi fluide qu'une répétition naturelle ; une fin mal placée donne 3,5 ou plus (voir les tests).
"""
from dataclasses import asdict, dataclass

import librosa
import numpy as np
import soundfile as sf

from .config import SR
from .mixage import load_stereo

SR_ANALYSE = 22050
HOP = 512
TEMPS_PAR_MESURE = 4


@dataclass
class Boucle:
    debut: float        # s, dans le fichier d'origine
    fin: float          # s, dans le fichier d'origine
    duree: float        # s, durée du fichier bouclé
    bpm: float
    mesures: int
    fondu: float        # s
    similarite: float   # 0–1, ressemblance de ce qui suit le début et la fin
    jonction: float     # écart au bouclage avec la suite naturelle du morceau (≈ 1 : comme une répétition naturelle)

    def qualite(self):
        if self.jonction <= 1.5:
            return "excellente"
        if self.jonction <= 2.5:
            return "bonne"
        return "moyenne (écoute l'aperçu de la jonction)"


def _norme(m):
    return m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-9)


def _signature(feat, trame, largeur):
    """Moyenne normalisée des caractéristiques sur [trame, trame + largeur[."""
    bloc = feat[:, trame:trame + largeur]
    v = bloc.mean(axis=1) if bloc.size else np.zeros(feat.shape[0])
    return v / (np.linalg.norm(v) + 1e-9)


def _temps(mono, duree):
    tempo, beats = librosa.beat.beat_track(y=mono, sr=SR_ANALYSE, hop_length=HOP, units="frames")
    tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    if tempo <= 0 or len(beats) < 3 * TEMPS_PAR_MESURE:
        # Pas de pulsation nette (ambiance) : grille régulière d'une demi-seconde
        pas = int(0.5 * SR_ANALYSE / HOP)
        return 120.0, np.arange(0, int(duree * SR_ANALYSE / HOP), pas)
    return tempo, np.asarray(beats)


def trouver_boucle(y, duree_min_rel=0.5, marge_fin=1.0):
    """Cherche (a, b) en échantillons à SR dans y (stéréo, SR). Renvoie (a, b, bpm, mesures, similarité)."""
    duree = y.shape[1] / SR
    mono = librosa.resample(y.mean(axis=0), orig_sr=SR, target_sr=SR_ANALYSE)
    bpm, temps = _temps(mono, duree)
    chroma = _norme(librosa.feature.chroma_stft(y=mono, sr=SR_ANALYSE, hop_length=HOP))
    mfcc = librosa.feature.mfcc(y=mono, sr=SR_ANALYSE, hop_length=HOP, n_mfcc=20)[1:]  # sans l'énergie
    rms = librosa.feature.rms(y=mono, hop_length=HOP)[0]
    largeur = max(4, int(TEMPS_PAR_MESURE * 60 / bpm * SR_ANALYSE / HOP))  # une mesure
    fin_max = (duree - marge_fin) * SR_ANALYSE / HOP - largeur
    niveau = np.percentile(rms, 90) + 1e-9

    meilleur = None
    for i, ta in enumerate(temps):
        if ta > 0.35 * len(rms):
            break
        if rms[ta:ta + largeur].mean() < 0.1 * niveau:  # ne pas démarrer dans un silence
            continue
        ca, ma = _signature(chroma, ta, largeur), _signature(mfcc, ta, largeur)
        for j in range(i + TEMPS_PAR_MESURE, len(temps), TEMPS_PAR_MESURE):
            tb = temps[j]
            if tb > fin_max:
                break
            if (tb - ta) < duree_min_rel * len(rms):
                continue
            # rms comparable : évite de boucler sur une fin qui s'éteint
            ecart_niveau = abs(np.log((rms[tb:tb + largeur].mean() + 1e-9) / (rms[ta:ta + largeur].mean() + 1e-9)))
            sim = 0.6 * float(ca @ _signature(chroma, tb, largeur)) + 0.4 * float(ma @ _signature(mfcc, tb, largeur))
            score = sim - 0.3 * ecart_niveau + 0.05 * (tb - ta) / len(rms)
            if meilleur is None or score > meilleur[0]:
                meilleur = (score, ta, tb, (j - i) // TEMPS_PAR_MESURE, sim)
    if meilleur is None:
        raise ValueError("morceau trop court ou trop peu rythmé pour trouver une boucle")
    _, ta, tb, mesures, sim = meilleur
    a = int(ta * HOP * SR / SR_ANALYSE)
    b = int(tb * HOP * SR / SR_ANALYSE)
    return a, b, bpm, mesures, sim


def _caler_sur_attaque(y, a, b, bpm):
    """Décale a et b ensemble (la longueur de boucle ne change pas) pour démarrer sur l'attaque grave la plus
    proche : sinon la musique commencerait à contretemps au premier passage."""
    mono = librosa.resample(y.mean(axis=0), orig_sr=SR, target_sr=SR_ANALYSE)
    # Attaques graves (grosse caisse, basse) : elles marquent les temps forts, alors que les aigus
    # (charleston) tombent souvent à contretemps
    env = librosa.onset.onset_strength(y=mono, sr=SR_ANALYSE, hop_length=HOP, fmax=250, n_mels=32)
    ta, demi = int(a * SR_ANALYSE / SR / HOP), max(1, int(0.6 * 60 / bpm * SR_ANALYSE / HOP))  # ± 0,6 temps
    lo, hi = max(0, ta - demi), min(len(env), ta + demi + 1)
    if lo >= hi:
        return a, b
    pic = lo + int(np.argmax(env[lo:hi]))
    # l'enveloppe d'attaque culmine une trame après le début du son : on recule d'une trame
    decalage = int(max(0, pic - 1) * HOP * SR / SR_ANALYSE) - a
    if a + decalage < 0 or b + decalage + int(0.25 * SR) > y.shape[1]:
        return a, b
    return a + decalage, b + decalage


def _ajuster(y, a, b, fenetre=0.03, portee=0.02):
    """Décale b de ± portee pour que la forme d'onde après b colle à celle après a (corrélation)."""
    n, d = int(fenetre * SR), int(portee * SR)
    ref = y[:, a:a + n].mean(axis=0)
    meilleur, db = -np.inf, 0
    for k in range(-d, d + 1, 4):
        if b + k < 0 or b + k + n > y.shape[1]:
            continue
        seg = y[:, b + k:b + k + n].mean(axis=0)
        c = float(ref @ seg) / (np.linalg.norm(ref) * np.linalg.norm(seg) + 1e-9)
        if c > meilleur:
            meilleur, db = c, k
    return b + db


def boucler(y, a, b, fondu):
    """y[a:b] dont le début reçoit, en fondu, la suite naturelle de b : le fichier boucle sans rupture."""
    f = min(int(fondu * SR), y.shape[1] - b, b - a)
    s = y[:, a:b].copy()
    if f > 0:
        t = np.linspace(0, np.pi / 2, f, dtype=np.float32)
        s[:, :f] = y[:, a:a + f] * np.sin(t) + y[:, b:b + f] * np.cos(t)  # puissance constante
    return s


def mesure_jonction(y, s, b, secondes=1.0, trame=2048):
    """Au bouclage, on doit entendre la suite naturelle de b dans le morceau d'origine.
    Écart spectral moyen entre le début de la boucle s et y[b:b+1 s], rapporté à l'écart médian entre deux
    trames voisines du morceau (≈ 1 pour une répétition musicale naturelle)."""
    n = min(int(secondes * SR), s.shape[1], y.shape[1] - b)
    spec = lambda x: np.log(np.abs(librosa.stft(x.mean(axis=0), n_fft=trame, hop_length=trame // 2,  # noqa: E731
                                                  center=False)) + 1e-4)
    boucle, naturel, tout = spec(s[:, :n]), spec(y[:, b:b + n]), spec(y)
    echelle = np.median(np.linalg.norm(np.diff(tout, axis=1), axis=0)) + 1e-9
    return float(np.linalg.norm(boucle - naturel, axis=0).mean() / echelle)


def creer_boucle(src, out_path, apercu_path=None, fondu=None):
    """Transforme un morceau en boucle parfaite. Renvoie la description Boucle (points dans le fichier source)."""
    y = load_stereo(src)
    a, b, bpm, mesures, sim = trouver_boucle(y)
    a, b = _caler_sur_attaque(y, a, b, bpm)
    b = _ajuster(y, a, b)
    fondu = fondu if fondu is not None else min(0.2, 60 / bpm / 4)
    s = boucler(y, a, b, fondu)
    peak = float(np.max(np.abs(s))) or 1.0
    if peak > 0.95:
        s = s / peak * 0.95
    sf.write(str(out_path), s.T, SR)
    if apercu_path:  # 5 s avant et après la jonction, pour l'écouter
        n = min(5 * SR, s.shape[1] // 2)
        sf.write(str(apercu_path), np.concatenate([s[:, -n:], s[:, :n]], axis=1).T, SR)
    return Boucle(round(a / SR, 3), round(b / SR, 3), round(s.shape[1] / SR, 3), round(bpm, 1), int(mesures),
                  round(fondu, 3), round(sim, 3), round(mesure_jonction(y, s, b), 2))


def en_dict(boucle):
    return asdict(boucle)
