"""RVC (Applio) : entraîner un modèle de ta voix sur 10 à 30 minutes d'enregistrements, puis convertir avec.

Plus fidèle que Seed-VC (sans entraînement) une fois entraîné. Environnement dédié (<lecteur>:\\StudioVoix\\rvc,
Python 3.12), appelé en sous-processus : moteurs/rvc_voix.py, lancé depuis le dossier d'Applio.
Modèles entraînés : StudioVoix\\rvc\\logs\\<nom>\\<nom>_<époques>e_<pas>s.pth et <nom>.index.
"""
import json
import re
import shutil
from pathlib import Path

import gradio as gr
import librosa
import soundfile as sf

from . import config as cfg
from .outils import lancer_moteur, stream_command
from .voix import analyser, list_voices, nettoyer_nom

FREQUENCE = 40000
DUREES = {
    "Rapide — 100 époques (pour essayer)": 100,
    "Normal — 300 époques (conseillé)": 300,
    "Long — 500 époques": 500,
}
DUREE_CONSEILLEE = 10 * 60   # s
DUREE_MIN = 60               # s : en dessous, on refuse
FORMATS = {".wav", ".mp3", ".flac"}
MODELES_DE_BASE = {
    "détecteur de hauteur (RMVPE)": "rvc/models/predictors/rmvpe.pt",
    "encodeur (ContentVec)": "rvc/models/embedders/contentvec/pytorch_model.bin",
    "pré-entraîné 40 kHz (G)": "rvc/models/pretraineds/hifi-gan/f0G40k.pth",
    "pré-entraîné 40 kHz (D)": "rvc/models/pretraineds/hifi-gan/f0D40k.pth",
}


def script() -> Path:
    return cfg.MOTEURS_DIR / "rvc_voix.py"


def ckpt_dir() -> Path:
    return cfg.RVC_DIR / "rvc" / "models"


def missing_components():
    return [n for n, f in MODELES_DE_BASE.items() if not (cfg.RVC_DIR / f).is_file()]


def _installe():
    if not Path(cfg.RVC_PYTHON).exists():
        raise gr.Error(f"RVC n'est pas installé ({cfg.RVC_PYTHON} introuvable). Relance INSTALLER.bat : "
                       "seules les étapes manquantes seront faites.")


def download():
    if not Path(cfg.RVC_PYTHON).exists():
        yield f"❌ Python de RVC introuvable : {cfg.RVC_PYTHON}. Relance INSTALLER.bat."
        return
    yield from stream_command([cfg.RVC_PYTHON, str(script()), "telecharger"], cfg.RVC_DIR,
                              f"Téléchargement des modèles de base de RVC (~1,8 Go) vers {ckpt_dir()} …")


# --- Modèles entraînés ------------------------------------------------------------------
def modeles():
    """Modèles entraînés prêts (poids final + index) : [{nom, pth, index, epoques}]."""
    liste = []
    for exp in sorted((cfg.RVC_DIR / "logs").glob("*")):
        if not exp.is_dir():
            continue
        candidats = []
        for p in exp.glob("*_*e_*s.pth"):
            m = re.fullmatch(re.escape(exp.name) + r"_(\d+)e_(\d+)s\.pth", p.name)
            if m:
                candidats.append((int(m.group(1)), int(m.group(2)), p))
        index = exp / f"{exp.name}.index"
        if candidats and index.exists():
            ep, _, pth = max(candidats)
            liste.append({"nom": exp.name, "pth": str(pth), "index": str(index), "epoques": ep})
    return liste


def choix_modeles():
    return [(f"{m['nom']} ({m['epoques']} époques)", m["nom"]) for m in modeles()]


def modele(nom):
    m = next((m for m in modeles() if m["nom"] == nom), None)
    if not m:
        raise gr.Error(f"Modèle RVC introuvable : {nom}. Entraîne-le dans l'onglet « Entraîner ma voix ».")
    return m


def maj_modeles(actuel=None):
    choix = choix_modeles()
    valeurs = [v for _, v in choix]
    return gr.update(choices=choix, value=actuel if actuel in valeurs else (valeurs[0] if valeurs else None))


def tableau_modeles():
    ms = modeles()
    if not ms:
        return "*Aucun modèle entraîné pour l'instant.*"
    return "| Modèle | Époques | Dossier |\n|---|---|---|\n" + "\n".join(
        f"| {m['nom']} | {m['epoques']} | `{Path(m['pth']).parent}` |" for m in ms)


def supprimer_modele(nom):
    if nom is None:
        return "Suppression annulée.", tableau_modeles()
    m = modele(nom)
    try:
        shutil.rmtree(Path(m["pth"]).parent)
        shutil.rmtree(cfg.RVC_DIR / "datasets" / nom, ignore_errors=True)
    except OSError as e:
        raise gr.Error(f"Impossible de supprimer le modèle (fichier utilisé ?) : {e}")
    return f"🗑️ Modèle « {nom} » supprimé.", tableau_modeles()


# --- Jeu d'enregistrements ------------------------------------------------------------------------
def _sources(fichiers, voix_biblio):
    chemins = [Path(f if isinstance(f, str) else f.name) for f in (fichiers or [])]
    chemins += [cfg.VOICES_DIR / f"{v}.wav" for v in (voix_biblio or []) if v in list_voices()]
    return chemins


def analyser_enregistrements(fichiers, voix_biblio):
    """Tableau de contrôle du jeu d'enregistrements (durée totale, problèmes par fichier)."""
    sources = _sources(fichiers, voix_biblio)
    if not sources:
        return "*Ajoute tes enregistrements (ou des voix de ta bibliothèque).*"
    lignes, total = ["| Fichier | Durée | Remarques |", "|---|---|---|"], 0.0
    for p in sources:
        if p.suffix.lower() not in FORMATS:
            lignes.append(f"| {p.name} | — | ❌ format non pris en charge (wav, mp3, flac) |")
            continue
        try:
            y, sr = librosa.load(str(p), sr=None, mono=True)
        except Exception:
            lignes.append(f"| {p.name} | — | ❌ fichier illisible |")
            continue
        a = analyser(y, sr)
        total += a.duree
        remarques = [m.split(" : ")[0] for m in a.erreurs + a.avertissements if "court" not in m]
        lignes.append(f"| {p.name} | {a.duree:.0f} s | {'⚠️ ' + ' ; '.join(remarques) if remarques else '✅'} |")
    conseil = ("✅ Durée suffisante." if total >= DUREE_CONSEILLEE else
               f"⚠️ {total / 60:.1f} min : vise **10 à 30 minutes** pour un bon modèle (au moins 1 minute).")
    return f"**Durée totale : {total / 60:.1f} min.** {conseil}\n\n" + "\n".join(lignes)


def preparer_jeu(nom, fichiers, voix_biblio):
    """Copie les enregistrements lisibles en WAV dans StudioVoix\\rvc\\datasets\\<nom> (chemin court). Renvoie (dossier, durée)."""
    dossier = cfg.RVC_DIR / "datasets" / nom
    if dossier.exists():
        shutil.rmtree(dossier)
    dossier.mkdir(parents=True)
    total = 0.0
    for i, p in enumerate(_sources(fichiers, voix_biblio), 1):
        if p.suffix.lower() not in FORMATS:
            continue
        try:
            y, sr = sf.read(str(p), dtype="float32", always_2d=True)
        except Exception:
            continue
        total += len(y) / sr
        sf.write(str(dossier / f"{i:03d}.wav"), y, sr)
    return dossier, total


def entrainer(nom, fichiers, voix_biblio, duree_label, lot, progress=gr.Progress()):
    _installe()
    nom = nettoyer_nom(nom)
    if not nom:
        raise gr.Error("Donne un nom au modèle (lettres, chiffres, espaces, - et _).")
    if missing_components():
        raise gr.Error("Modèles de base de RVC absents : télécharge-les dans l'onglet « Modèles ».")
    epoques = DUREES.get(duree_label, 300)
    dossier, total = preparer_jeu(nom, fichiers, voix_biblio)
    if total < DUREE_MIN:
        raise gr.Error(f"Pas assez d'enregistrements ({total:.0f} s) : il faut au moins 1 minute, idéalement 10 à 30.")
    tache = dossier.parent / f"{nom}.json"
    tache.write_text(json.dumps({"nom": nom, "dataset": str(dossier), "epoques": epoques, "frequence": FREQUENCE,
                                 "lot": int(lot)}, ensure_ascii=False), encoding="utf-8")

    def suivi(i, n):
        etape = "Préparation" if i == 1 else "Extraction" if i == 2 else "Index" if i == n else f"Époque {i - 2}/{n - 3}"
        progress(i / n, desc=f"Entraînement RVC : {etape}…")

    lignes = lancer_moteur([cfg.RVC_PYTHON, str(script()), "entrainer", str(tache)], cfg.RVC_DIR, None, "RVC", suivi)
    termine = next((l_ for l_ in reversed(lignes) if l_.startswith("TERMINE ")), None)
    if not termine:
        raise gr.Error("RVC : l'entraînement s'est terminé sans modèle.")
    avert = "" if total >= DUREE_CONSEILLEE else f" (avec {total / 60:.1f} min d'enregistrements : ajoute-en pour plus de fidélité)"
    return (f"✅ Modèle « {nom} » entraîné ({epoques} époques){avert}. Choisis-le dans « Créer une chanson » "
            "(Réglages voix → Conversion) ou dans « Synthèse vocale ».")


# --- Conversion -------------------------------------------------------------------------------------
def convertir(entree, nom, demi_tons, sortie, index_rate=0.5, protect=0.33, progress=None):
    """Convertit une voix (chantée ou parlée) avec un modèle RVC entraîné. Renvoie le fichier de sortie."""
    _installe()
    m = modele(nom)
    sortie = Path(sortie)
    tache = sortie.with_suffix(".rvc.json")
    tache.write_text(json.dumps({"modele": m["pth"], "index": m["index"], "entree": str(entree), "sortie": str(sortie),
                                 "demi_tons": int(demi_tons), "index_rate": float(index_rate),
                                 "protect": float(protect)}, ensure_ascii=False), encoding="utf-8")
    if progress:
        progress(0.6, desc=f"Conversion avec le modèle RVC « {nom} »…")
    lancer_moteur([cfg.RVC_PYTHON, str(script()), "convertir", str(tache)], cfg.RVC_DIR, None, "RVC",
                  attendu=sortie)
    return sortie
