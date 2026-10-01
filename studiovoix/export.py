"""Export pour le web : volume harmonisé (LUFS), fichiers OGG et MP3, pack de bande-son avec manifest.json.

Mesure du volume : ITU-R BS.1770-4 (filtre K, blocs de 400 ms à 75 % de recouvrement, portes absolue −70 LUFS
et relative −10 LU), implémentée ici pour ne pas ajouter de dépendance à l'application ; les tests la
comparent à pyloudnorm. OGG Vorbis et MP3 (LAME, avec en-tête Xing de lecture sans blanc) sont écrits par
soundfile/libsndfile ; la longueur est conservée à l'échantillon près (vérifié), les boucles restent justes.
"""
import json
import shutil
import unicodedata
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from . import config as cfg
from .mixage import load_stereo

CIBLES = {
    "−14 LUFS (fort, comme les plateformes de streaming)": -14.0,
    "−16 LUFS (conseillé pour un jeu web)": -16.0,
    "−18 LUFS (musique de fond discrète)": -18.0,
    "−23 LUFS (norme radio/TV EBU R128)": -23.0,
}
PIC_MAX = 10 ** (-1 / 20)  # −1 dBFS : marge pour la compression OGG/MP3
FORMATS = {"ogg": ("OGG", "VORBIS"), "mp3": ("MP3", "MPEG_LAYER_III")}


def _biquad_k(sr):
    """Coefficients du filtre K (plateau haut +4 dB à 1,5 kHz puis passe-haut à 38 Hz), pour toute fréquence."""
    def plateau(g, q, fc):
        a = 10 ** (g / 40)
        w = 2 * np.pi * fc / sr
        al = np.sin(w) / (2 * q)
        c, r = np.cos(w), 2 * np.sqrt(a) * al
        b = [a * ((a + 1) + (a - 1) * c + r), -2 * a * ((a - 1) + (a + 1) * c), a * ((a + 1) + (a - 1) * c - r)]
        den = [(a + 1) - (a - 1) * c + r, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - r]
        return np.array(b) / den[0], np.array(den) / den[0]

    def passe_haut(q, fc):
        w = 2 * np.pi * fc / sr
        al, c = np.sin(w) / (2 * q), np.cos(w)
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        den = [1 + al, -2 * c, 1 - al]
        return np.array(b) / den[0], np.array(den) / den[0]

    return [plateau(4.0, 1 / np.sqrt(2), 1500.0), passe_haut(0.5, 38.0)]


def loudness(y, sr):
    """Volume intégré en LUFS d'un signal (canaux, échantillons) — ITU-R BS.1770-4."""
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    for b, a in _biquad_k(sr):
        y = lfilter(b, a, y, axis=1)
    bloc, pas = int(0.4 * sr), int(0.1 * sr)
    if y.shape[1] < bloc:
        z = np.mean(y ** 2, axis=1, keepdims=True)
    else:
        n = 1 + (y.shape[1] - bloc) // pas
        z = np.stack([np.mean(y[:, i * pas:i * pas + bloc] ** 2, axis=1) for i in range(n)], axis=1)
    l_bloc = -0.691 + 10 * np.log10(z.sum(axis=0) + 1e-20)
    garde = l_bloc > -70
    if not garde.any():
        return -np.inf
    relatif = -0.691 + 10 * np.log10(z[:, garde].mean(axis=1).sum()) - 10
    garde &= l_bloc > relatif
    return float(-0.691 + 10 * np.log10(z[:, garde].mean(axis=1).sum()))


def normaliser(y, sr, cible):
    """Ramène au volume cible, sans dépasser −1 dBFS de pic (on baisse alors le gain). Renvoie (y, LUFS final)."""
    mesure = loudness(y, sr)
    if not np.isfinite(mesure):
        return y, mesure
    y = y * 10 ** ((cible - mesure) / 20)
    pic = float(np.max(np.abs(y)))
    if pic > PIC_MAX:
        y = y * (PIC_MAX / pic)
    return y, loudness(y, sr)


BLOC = 16384  # trames écrites à la fois


def ecrire(y, sr, base: Path, formats):
    """Écrit base.ogg / base.mp3… ; renvoie {format: chemin}.

    Écriture par blocs : en un seul appel, l'encodeur Vorbis de libsndfile dépasse la pile de 1 Mo du fil principal
    sous Windows sur un morceau de quelques dizaines de secondes (« Windows fatal exception: stack overflow »,
    constaté par la vérification automatique). La longueur reste exacte à l'échantillon près."""
    donnees = np.ascontiguousarray(np.atleast_2d(y).T, dtype=np.float32)
    sortie = {}
    for f in formats:
        fmt, sub = FORMATS[f]
        chemin = base.with_suffix(f".{f}")
        with sf.SoundFile(str(chemin), "w", sr, donnees.shape[1], format=fmt, subtype=sub) as fichier:
            for i in range(0, len(donnees), BLOC):
                fichier.write(donnees[i:i + BLOC])
        sortie[f] = chemin
    return sortie


def _note_volume(lufs, cible):
    if not np.isfinite(lufs):
        return "(signal silencieux : volume inchangé)"
    if lufs >= cible - 0.2:
        return f"à {lufs:.1f} LUFS"
    return f"à {lufs:.1f} LUFS (limité par les crêtes pour ne pas saturer ; cible {cible:.0f})"


def exporter_fichier(src, cible_label, formats=("mp3",)):
    """Exporte un fichier (par exemple une chanson) à côté de l'original, au volume cible. Renvoie (fichier, message)."""
    if not src or not Path(src).exists():
        raise gr.Error("Rien à exporter : crée d'abord une chanson.")
    cible = CIBLES.get(cible_label, -14.0)
    y, lufs = normaliser(load_stereo(src), cfg.SR, cible)
    fichiers = ecrire(y, cfg.SR, Path(src).with_name(Path(src).stem + "_export"), formats)
    premier = next(iter(fichiers.values()))
    return str(premier), f"✅ Exporté {_note_volume(lufs, cible)} : {', '.join(str(p) for p in fichiers.values())}"


def exporter_fichier_formats(src, cible_label, formats):
    """Export d'un fichier (bruitage…) dans plusieurs formats, à côté de l'original. Renvoie un message."""
    formats = [f for f in (formats or []) if f in FORMATS or f == "wav"]
    if not formats:
        raise gr.Error("Choisis au moins un format.")
    if not src or not Path(src).exists():
        raise gr.Error("Rien à exporter : génère d'abord un fichier.")
    cible = CIBLES.get(cible_label, -16.0)
    y, lufs = normaliser(load_stereo(src), cfg.SR, cible)
    base = Path(src).with_name(Path(src).stem + "_export")
    fichiers = ecrire(y, cfg.SR, base, [f for f in formats if f != "wav"])
    if "wav" in formats:
        sf.write(str(base.with_suffix(".wav")), y.T, cfg.SR)
        fichiers["wav"] = base.with_suffix(".wav")
    return f"✅ Exporté {_note_volume(lufs, cible)} : {', '.join(str(p) for p in fichiers.values())}"


def _slug(nom, pris):
    """Identifiant de fichier sûr pour le web : minuscules ASCII sans accents, tirets, unique dans le pack."""
    ascii_ = unicodedata.normalize("NFKD", nom or "son").encode("ascii", "ignore").decode()
    base = "".join(c if c.isalnum() else "-" for c in ascii_.strip().lower()).strip("-") or "son"
    base = "-".join(x for x in base.split("-") if x)
    ident, i = base, 2
    while ident in pris:
        ident, i = f"{base}-{i}", i + 1
    pris.add(ident)
    return ident


def bruitages_du_projet(projet):
    """Pour chaque nom de bruitage du projet : (création la plus récente, variante choisie)."""
    derniers = {}
    for f in sorted(cfg.SFX_DIR.glob("*/creation.json")) if cfg.SFX_DIR.exists() else []:
        try:
            infos = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if infos.get("type") == "bruitage" and infos.get("projet") == projet:
            derniers[infos.get("nom") or "bruitage"] = (f.parent, infos)  # dossiers horodatés : le dernier gagne
    resultat = []
    for nom in sorted(derniers):
        dossier, infos = derniers[nom]
        versions = infos.get("versions") or []
        i = max(1, min(len(versions), int(infos.get("choisie") or 1))) - 1 if versions else 0
        fichier = Path(versions[i]["fichier"]) if versions else None
        if fichier is not None and not fichier.is_absolute():
            fichier = dossier / fichier
        if fichier is not None and fichier.exists():
            resultat.append((nom, dossier, infos, i + 1, fichier))
    return resultat


def exporter_pack(projet, cible_label, formats, progress=gr.Progress()):
    """Pack audio du jeu : la piste la plus récente de chaque situation et, pour chaque bruitage du projet, sa
    variante choisie ; tout au même volume, en OGG et/ou MP3, avec un seul manifest.json (musiques dans « pistes »,
    bruitages dans « bruitages ») et une archive zip."""
    from .jeu import nom_projet  # import tardif : jeu importe déjà ce module indirectement via l'interface

    projet = nom_projet(projet)
    formats = [f for f in (formats or []) if f in FORMATS]
    if not formats:
        raise gr.Error("Choisis au moins un format (OGG, MP3).")
    racine = cfg.GAMES_DIR / projet
    pistes = []
    for dossier_situation in sorted(p for p in racine.glob("*") if p.is_dir() and p.name != "export"):
        recentes = sorted((d for d in dossier_situation.iterdir() if (d / "piste.wav").exists()), reverse=True)
        if recentes:
            pistes.append(recentes[0])
    sons = bruitages_du_projet(projet)
    if not pistes and not sons:
        raise gr.Error(f"Aucune piste ni aucun bruitage dans le projet « {projet} » : génère d'abord la bande-son "
                       "ou des bruitages (en leur donnant ce nom de projet).")
    cible = CIBLES.get(cible_label, -16.0)
    racine.mkdir(parents=True, exist_ok=True)
    export = racine / "export"
    if export.exists():
        shutil.rmtree(export)
    export.mkdir(parents=True)
    total = len(pistes) + len(sons)
    manifest = {"projet": projet, "volume_cible_lufs": cible, "formats": formats, "pistes": [], "bruitages": []}
    for n, dossier in enumerate(pistes, 1):
        progress(n / (total + 1), desc=f"Export {n}/{total} : {dossier.parent.name}…")
        infos = json.loads((dossier / "creation.json").read_text(encoding="utf-8"))
        ident = dossier.parent.name
        y, lufs = normaliser(load_stereo(dossier / "piste.wav"), cfg.SR, cible)
        fichiers = ecrire(y, cfg.SR, export / ident, formats)
        points = infos.get("boucle_points")
        manifest["pistes"].append({
            "id": ident,
            "libelle": infos.get("libelle", ident),
            "fichiers": {f: p.name for f, p in fichiers.items()},
            "boucle": bool(points),  # le fichier entier boucle sans rupture : loop=true suffit
            "duree": round(y.shape[1] / cfg.SR, 3),
            "bpm": points.get("bpm") if points else None,
            "volume_lufs": round(lufs, 1),
            "graine": (infos.get("versions") or [{}])[0].get("graine"),
            "description": infos.get("description"),
        })
    if sons:
        (export / "bruitages").mkdir()
    pris = set()
    for n, (nom, dossier, infos, variante, fichier) in enumerate(sons, len(pistes) + 1):
        progress(n / (total + 1), desc=f"Export {n}/{total} : bruitage {nom}…")
        ident = _slug(nom, pris)
        y, lufs = normaliser(load_stereo(fichier), cfg.SR, cible)
        fichiers = ecrire(y, cfg.SR, export / "bruitages" / ident, formats)
        manifest["bruitages"].append({
            "id": ident,
            "nom": nom,
            "fichiers": {f: f"bruitages/{p.name}" for f, p in fichiers.items()},
            "duree": round(y.shape[1] / cfg.SR, 3),
            "volume_lufs": round(lufs, 1) if np.isfinite(lufs) else None,
            "variante": variante,
            "graine": (infos.get("versions") or [{}])[variante - 1].get("graine"),
            "description": infos.get("description"),
        })
    (export / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    archive = shutil.make_archive(str(racine / f"{projet}_bande-son"), "zip", export)
    msg = (f"✅ {len(pistes)} piste(s) et {len(sons)} bruitage(s) exportés à {cible:.0f} LUFS dans {export} "
           f"(formats : {', '.join(formats)}), archive : {archive}")
    return archive, msg
