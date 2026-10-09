"""Musiques en série : une musique par ligne d'un tableau (Excel, CSV) ou d'une liste collée, toutes dans le même
style — par exemple les musiques d'un quiz ou d'un jeu, d'un coup.

Le tableau (même lecture que les images en série : serie.lire_tableau) : colonne des descriptions (ce que la musique
doit évoquer : « combat contre le boss final », « menu calme »…), et facultatives : noms de fichiers (le fichier porte
exactement ce nom), paroles (musiques chantées par la voix d'ACE-Step), durée (« 90 », « 1:30 »). Le style commun
(listes de l'onglet « Créer une chanson » + texte libre) est ajouté à chaque description ; une musique de référence
facultative (reference_audio d'ACE-Step : timbre et mixage) rend la série encore plus homogène. Les descriptions sont
traduites en anglais par Qwen3-VL en une fois avant de lancer ACE-Step (il comprend mieux l'anglais).
Une tâche ACE-Step par ligne ; une ligne en échec n'arrête pas le lot, « Reprendre » refait seulement les musiques
manquantes (lot.json). Sortie : data/musiques_serie/<horodatage>_<nom>/ avec brut/<nom>.wav (sortie d'ACE-Step), les
fichiers finaux au même volume (WAV, MP3, OGG au choix), lot.csv (pour Excel) et une archive zip.
"""
import csv
import json
import re
import time
import zipfile
from pathlib import Path

import gradio as gr
import soundfile as sf

from . import acestep, diffusion, export, serie
from . import config as cfg
from .audio import charger
from .images import scenes_du_texte
from .mixage import load_stereo
from .outils import nouveau_dossier
from .styles import musique
from .videos import duree_lisible

SERIE_MAX = 300
ECHECS_MAX = 3  # musiques en échec à la suite : le lot s'arrête (serveur ACE-Step arrêté, carte saturée…)
AUCUNE = serie.AUCUNE
INSTRUMENTAL = "[Instrumental]"
MODE_INSTRU = "Instrumentales"
MODE_CHANT = "Chantées par la voix d'ACE-Step (colonne des paroles)"
MODES = [MODE_INSTRU, MODE_CHANT]
DUREE_MIN, DUREE_MAX = 10, 600  # limites d'ACE-Step (DURATION_MIN / DURATION_MAX)
FORMATS = ["wav", "mp3", "ogg"]
TRADUCTION_PAQUET = 15  # lignes traduites par Qwen à la fois
EXTENSIONS_AUDIO = (".wav", ".mp3", ".ogg", ".flac", ".m4a")


# --- Tableau ----------------------------------------------------------------------------------------------------
def analyser(fichier, feuille=None, entetes=True):
    """Après le dépôt du fichier : (feuille, colonne des descriptions, des noms, des paroles, des durées, aperçu,
    message)."""
    if not fichier:
        vide = gr.update(choices=[], value=None)
        return (gr.update(choices=[], value=None, visible=False), vide, vide, vide, vide, gr.update(value=None),
                "Dépose un fichier, ou colle tes descriptions ci-dessous (une par ligne).")
    feuilles = serie.lire_tableau(fichier)
    noms = list(feuilles)
    feuille = feuille if feuille in feuilles else next(
        (n for n in noms if any(any(c for c in l_) for l_ in feuilles[n])), noms[0])
    lignes = feuilles[feuille]
    choix = serie.colonnes(lignes, entetes)
    if not choix:
        raise gr.Error(f"La feuille « {feuille} » est vide.")
    desc = serie._devine(lignes, entetes, ["description", "desc", "prompt", "prompts", "musique", "music", "theme",
                                           "situation", "ambiance", "texte"], True)
    nom = serie._devine(lignes, entetes, ["nom", "name", "id", "fichier", "file", "titre", "title"])
    paroles = serie._devine(lignes, entetes, ["paroles", "lyrics", "chanson", "texte chante"])
    duree = serie._devine(lignes, entetes, ["duree", "duration", "temps", "secondes", "longueur"])
    pris = {desc}
    nom = nom if nom not in pris else None
    pris.add(nom)
    paroles = paroles if paroles not in pris else None
    pris.add(paroles)
    duree = duree if duree not in pris else None
    i = serie._indice(desc) if desc else 0
    n = sum(1 for l_ in lignes[1 if entetes else 0:] if desc and i < len(l_) and l_[i])
    msg = (f"Feuille « {feuille} » : {len(lignes) - (1 if entetes else 0)} ligne(s), {len(choix)} colonne(s). "
           f"Colonne des descriptions proposée : **{desc}** ({n} case(s) remplie(s)). Vérifie les choix ci-dessous.")
    avec_aucune = [(AUCUNE, AUCUNE)] + choix
    return (gr.update(choices=noms, value=feuille, visible=len(noms) > 1),
            gr.update(choices=choix, value=desc),
            gr.update(choices=avec_aucune, value=nom or AUCUNE),
            gr.update(choices=avec_aucune, value=paroles or AUCUNE),
            gr.update(choices=avec_aucune, value=duree or AUCUNE),
            gr.update(value=serie._apercu(lignes, entetes)), msg)


def duree_de(texte, defaut):
    """« 90 », « 90 s », « 1:30 », « 1 min 30 » → secondes, ramenées entre 10 et 600 ; vide ou illisible → defaut."""
    t = (texte or "").strip().lower().replace(",", ".")
    m = re.fullmatch(r"(\d+)\s*(?::|min|mn|m)\s*(\d{1,2})?\s*(?:s|sec)?", t)
    if m:
        valeur = int(m.group(1)) * 60 + int(m.group(2) or 0)
    else:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secondes?)?", t)
        valeur = float(m.group(1)) if m else defaut
    return int(max(DUREE_MIN, min(DUREE_MAX, round(float(valeur or defaut)))))


def nom_audio(nom):
    """Nom du fichier écrit dans le tableau, gardé tel quel (serie.nom_de_fichier), sans extension (le format est
    choisi dans l'onglet) ; une extension audio tapée (« boss.mp3 ») est retirée."""
    nom = (nom or "").strip()
    if nom.lower().endswith(EXTENSIONS_AUDIO):
        nom = nom[:-len(Path(nom).suffix)]
    return Path(serie.nom_de_fichier(nom)).stem if serie.nom_de_fichier(nom) else ""


def _colonne(col):
    return serie._indice(col) if col and col != AUCUNE else None


def entrees(fichier=None, feuille=None, entetes=True, col_desc=None, col_nom=None, col_paroles=None, col_duree=None,
            liste="", numeroter=False, duree_defaut=60):
    """Lignes à générer : [{numero, ligne, nom, description, paroles, duree, base, doublon}] depuis le tableau
    (lignes dont la description est remplie) ou la liste collée (une description par ligne). base = nom des fichiers
    sans extension : exactement celui du tableau, sinon 001_<début de la description> ; doublons (sans tenir compte
    de la casse, comme Windows) → « _2 »."""
    resultat = []
    if fichier:
        if not col_desc:
            raise gr.Error("Choisis la colonne qui contient les descriptions des musiques.")
        feuilles = serie.lire_tableau(fichier)
        lignes = feuilles.get(feuille) or next(iter(feuilles.values()))
        idesc = serie._indice(col_desc)
        inom, iparoles, iduree = _colonne(col_nom), _colonne(col_paroles), _colonne(col_duree)

        def case(l_, i):
            return l_[i] if i is not None and i < len(l_) else ""

        depart = 1 if entetes else 0
        for k, l_ in enumerate(lignes[depart:], depart + 1):
            d = case(l_, idesc)
            if d:
                resultat.append({"ligne": k, "nom": case(l_, inom), "description": d, "paroles": case(l_, iparoles),
                                 "duree": duree_de(case(l_, iduree), duree_defaut)})
    else:
        for k, d in enumerate((liste or "").splitlines(), 1):
            if d.strip():
                resultat.append({"ligne": k, "nom": "", "description": d.strip(), "paroles": "",
                                 "duree": duree_de("", duree_defaut)})
    if not resultat:
        raise gr.Error("Aucune description : la colonne choisie est vide, ou la liste est vide.")
    if len(resultat) > SERIE_MAX:
        raise gr.Error(f"{len(resultat)} musiques : {SERIE_MAX} au plus par lot (découpe le fichier).")
    pris = set()
    for i, e in enumerate(resultat, 1):
        e["numero"] = i
        voulu = nom_audio(e["nom"])
        base = (f"{i:03d}_{voulu}" if numeroter else voulu) if voulu else \
            f"{i:03d}_{serie._slug(e['description'][:30], f'musique_{i}')}"
        candidat, k = base, 1
        while candidat.lower() in pris:
            k += 1
            candidat = f"{base}_{k}"
        e["doublon"] = candidat != base
        pris.add(candidat.lower())
        e["base"] = candidat
    return resultat


# --- Descriptions -------------------------------------------------------------------------------------------------
def style_commun(genre, style, instruments, ambiance, extra, style_libre, voix_base="Automatique"):
    """Description anglaise du style commun (listes de l'onglet « Créer une chanson » + texte libre)."""
    return _sans_doublons(acestep.build_prompt(genre, style, instruments, ambiance, voix_base, extra),
                          musique(style_libre))


def _sans_doublons(*morceaux):
    termes = []
    for t in ", ".join(m.strip().rstrip(".") for m in morceaux if m and m.strip()).split(","):
        if t.strip() and t.strip().lower() not in (x.lower() for x in termes):
            termes.append(t.strip())
    return ", ".join(termes)


def description_finale(description, commun, instrumentale):
    """« description de la ligne, style commun[, instrumental] » : la ligne d'abord (ACE-Step coupe la description
    à 256 jetons : si c'est trop long, c'est la fin du style commun qui saute, pas ce qui distingue la musique)."""
    return _sans_doublons(musique(description), commun, "instrumental" if instrumentale else "")


def paroles_de(e, mode):
    """Paroles balisées de la ligne ; instrumental si le mode l'est ou si la case est vide."""
    if mode != MODE_CHANT or not (e.get("paroles") or "").strip():
        return INSTRUMENTAL
    return acestep.baliser_paroles(e["paroles"].strip()).strip() or INSTRUMENTAL


def traduire(descriptions, progress=None):
    """Descriptions (souvent en français) → descriptions anglaises pour ACE-Step, par Qwen3-VL (mode « musiques »),
    par paquets de TRADUCTION_PAQUET lignes numérotées. Une réponse incomplète garde le texte d'origine des lignes
    manquantes. Renvoie (descriptions, message ou "")."""
    sortie, manquees = [], 0
    for p0 in range(0, len(descriptions), TRADUCTION_PAQUET):
        paquet = descriptions[p0:p0 + TRADUCTION_PAQUET]
        texte = f"Number of lines: {len(paquet)}\n\n" + "\n".join(f"{i}. {d}" for i, d in enumerate(paquet, 1))
        if progress:
            progress(0.02, desc=f"Traduction des descriptions {p0 + 1}–{p0 + len(paquet)} sur {len(descriptions)}…")
        lignes = scenes_du_texte(diffusion.decrire("musiques", texte, nombre=len(paquet)))
        if len(lignes) != len(paquet):  # Qwen n'a pas rendu une ligne par description : on garde l'original
            manquees += len(paquet)
            sortie += paquet
        else:
            sortie += [t or d for t, d in zip(lignes, paquet)]
    msg = (f"⚠️ {manquees} description(s) envoyée(s) sans traduction (réponse incomplète de Qwen3-VL)."
           if manquees else "")
    return sortie, msg


def apercu(fichier, feuille, entetes, col_desc, col_nom, col_paroles, col_duree, liste, genre, style, instruments,
           ambiance, extra, style_libre, voix_base, mode, duree_defaut, limite=0, numeroter=False):
    """Tableau de ce qui sera envoyé (sans la traduction, faite au lancement)."""
    lot = _limiter(entrees(fichier, feuille, entetes, col_desc, col_nom, col_paroles, col_duree, liste, numeroter,
                           duree_defaut), limite)
    data = []
    for e in lot:
        paroles = paroles_de(e, mode)
        commun = style_commun(genre, style, instruments, ambiance, extra, style_libre,
                              voix_base if paroles != INSTRUMENTAL else "Automatique")
        data.append([e["numero"], e["base"], e["duree"], "chantée" if paroles != INSTRUMENTAL else "instrumentale",
                     description_finale(e["description"], commun, paroles == INSTRUMENTAL)])
    msg = f"{len(lot)} musique(s), {duree_lisible(sum(e['duree'] for e in lot))} en tout."
    if not style_commun(genre, style, instruments, ambiance, extra, style_libre):
        msg += "\n\nℹ️ Aucun style commun choisi : chaque musique prendra le style que sa description suggère."
    doublons = [e for e in lot if e["doublon"]]
    if doublons:
        msg += (f"\n\n⚠️ {len(doublons)} nom(s) de fichier en double : renommés « _2 », « _3 »… (lignes "
                + ", ".join(str(e["ligne"]) for e in doublons[:10]) + ("…" if len(doublons) > 10 else "") + ").")
    if mode == MODE_CHANT and not any(paroles_de(e, mode) != INSTRUMENTAL for e in lot):
        msg += "\n\n⚠️ Mode chanté mais aucune parole : choisis la colonne des paroles (sinon tout sera instrumental)."
    msg += "\n\nLes descriptions seront traduites en anglais au lancement si la case est cochée."
    return {"headers": ["N°", "Fichier", "Durée (s)", "Type", "Description envoyée"], "data": data}, msg


def _limiter(lot, limite):
    limite = int(limite or 0)
    return lot[:limite] if limite > 0 else lot


# --- Génération ---------------------------------------------------------------------------------------------------
def generer(fichier, feuille, entetes, col_desc, col_nom, col_paroles, col_duree, liste, genre, style, instruments,
            ambiance, extra, style_libre, voix_base, mode, langue_label, duree_defaut, reference, traduction, thinking,
            versions, graine, formats, cible_label, nom="", limite=0, numeroter=False, progress=gr.Progress()):
    """Prépare le lot (lot.json) puis génère les musiques une à une. Renvoie (message, liste des résultats, premier
    résultat, archive zip, dossier)."""
    mode = mode if mode in MODES else MODE_INSTRU
    lot = _limiter(entrees(fichier, feuille, entetes, col_desc, col_nom, col_paroles, col_duree, liste, numeroter,
                           duree_defaut), limite)
    formats = [f for f in (formats or []) if f in FORMATS] or ["wav"]
    reference = str(getattr(reference, "name", reference)) if reference else None
    if reference and not Path(reference).is_file():
        raise gr.Error(f"Musique de référence introuvable : {reference}")
    nom = serie._slug(nom, "musiques")
    note = ""
    descriptions = [e["description"] for e in lot]
    if traduction:
        if diffusion.present("qwen"):
            descriptions, note = traduire(descriptions, progress)
        else:
            note = ("ℹ️ Descriptions envoyées telles quelles : Qwen3-VL (traduction) n'est pas installé "
                    "(Outils → Modèles et diagnostic).")
    dossier = nouveau_dossier(cfg.MUSIQUES_SERIE_DIR, nom)
    if reference:  # copiée : la reprise ne dépend pas de l'original
        copie = dossier / "reference.wav"  # en WAV : le serveur ACE-Step ne lit pas forcément un m4a
        try:
            y, sr_ref = charger(reference, sr=None, mono=False)
        except Exception as e:  # noqa: BLE001 - format illisible
            raise gr.Error(f"Musique de référence illisible ({e}).") from e
        sf.write(str(copie), y.T if y.ndim > 1 else y, sr_ref)
        reference = copie.name
    graine = int(graine or 0) or acestep.graines(1)[0]
    for e, anglais in zip(lot, descriptions):
        paroles = paroles_de(e, mode)
        commun = style_commun(genre, style, instruments, ambiance, extra, style_libre,
                              voix_base if paroles != INSTRUMENTAL else "Automatique")
        e.update(description_envoyee=description_finale(anglais, commun, paroles == INSTRUMENTAL),
                 paroles_envoyees=paroles, graine=graine + e["numero"] - 1)
    infos = {"nom": nom, "mode": mode, "langue": cfg.LANGUES.get(langue_label, "fr"), "reference": reference,
             "thinking": bool(thinking), "versions": max(1, min(2, int(versions or 1))), "formats": formats,
             "cible": cible_label if cible_label in export.CIBLES else next(iter(export.CIBLES)),
             "style_commun": style_commun(genre, style, instruments, ambiance, extra, style_libre), "graine": graine,
             "source": Path(str(getattr(fichier, "name", fichier))).name if fichier else "liste", "musiques": lot}
    (dossier / "lot.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    return _executer(dossier, progress, note)


def reprendre(dossier, progress=gr.Progress()):
    """Refait seulement les musiques manquantes d'un lot (interrompu, ou musiques en échec)."""
    if dossier and not Path(dossier).is_absolute():  # « data\musiques_serie\… » tapé à la main
        dossier = cfg.APP_DIR / dossier
    if not dossier or not (Path(dossier) / "lot.json").exists():
        raise gr.Error("Aucun lot à reprendre : lance d'abord une série (ou indique le dossier du lot).")
    return _executer(Path(dossier), progress)


def _bases(e, versions):
    """Noms (sans extension) des versions d'une ligne : « nom », puis « nom_v2 »."""
    return [e["base"]] + [f"{e['base']}_v{v}" for v in range(2, versions + 1)]


def _fait(dossier, e, infos):
    return all((dossier / f"{b}.{f}").exists() for b in _bases(e, infos["versions"]) for f in infos["formats"])


def _executer(dossier, progress, note=""):
    infos = json.loads((dossier / "lot.json").read_text(encoding="utf-8"))
    lot = infos["musiques"]
    reference = dossier / infos["reference"] if infos.get("reference") else None
    a_faire = [e for e in lot if not _fait(dossier, e, infos)]
    debut, erreurs, echecs_suite = time.monotonic(), [], 0
    brut = dossier / "brut"
    brut.mkdir(exist_ok=True)
    for k, e in enumerate(a_faire):
        bruts = [brut / f"{b}.wav" for b in _bases(e, infos["versions"])]
        reste = ""
        if k:
            reste = f", environ {duree_lisible((time.monotonic() - debut) / k * (len(a_faire) - k))} restantes"
        etape = f"musique {k + 1}/{len(a_faire)} ({e['base']}){reste}"
        try:
            if not all(b.exists() for b in bruts):
                params = acestep.text2music_params(e["description_envoyee"], e["paroles_envoyees"], infos["langue"],
                                                   e["duree"], 0, infos["thinking"])
                acestep.generer(params, bruts, progress, etape,
                                {"reference_audio": str(reference)} if reference else None, e["graine"])
            for b in bruts:
                _finaliser(b, dossier, infos)
            echecs_suite = 0
        except gr.Error as err:  # la ligne suivante est tentée ; « Reprendre » refera celle-ci
            erreurs.append(f"musique {e['numero']} ({e['base']}) : {getattr(err, 'message', err)}")
            echecs_suite += 1
            if echecs_suite >= ECHECS_MAX:
                erreurs.append(f"{ECHECS_MAX} musiques en échec à la suite : lot arrêté.")
                break
    faites = [e for e in lot if _fait(dossier, e, infos)]
    _ecrire_csv(dossier, lot, infos)
    archive = _archive(dossier, infos, faites) if faites else None
    msg = (f"✅ {len(faites)}/{len(lot)} musique(s) en {duree_lisible(time.monotonic() - debut)}, dans {dossier} "
           f"({', '.join(f.upper() for f in infos['formats'])}, même volume pour toutes).")
    if note:
        msg += f"\n\n{note}"
    if erreurs:
        msg += ("\n\n⚠️ " + "\n\n⚠️ ".join(erreurs) +
                "\n\nClique sur « Reprendre le lot » pour refaire seulement les musiques manquantes.")
    ecoute = infos["formats"][0] if infos["formats"][0] != "ogg" else ("wav" if "wav" in infos["formats"] else "ogg")
    choix = [(f"{e['numero']} · {b}", str(dossier / f"{b}.{ecoute}")) for e in faites
             for b in _bases(e, infos["versions"])]
    return (msg, gr.update(choices=choix, value=choix[0][1] if choix else None), choix[0][1] if choix else None,
            str(archive) if archive else None, str(dossier))


def _finaliser(brut, dossier, infos):
    """Fichiers finaux d'une musique, tous au même volume (cible choisie) : WAV, MP3, OGG."""
    cible = export.CIBLES.get(infos["cible"], -16.0)
    y, _ = export.normaliser(load_stereo(brut), cfg.SR, cible)
    base = dossier / brut.stem
    export.ecrire(y, cfg.SR, base, [f for f in infos["formats"] if f != "wav"])
    if "wav" in infos["formats"]:
        sf.write(str(base.with_suffix(".wav")), y.T, cfg.SR)


def _ecrire_csv(dossier, lot, infos):
    """lot.csv lisible par Excel en français (point-virgule, UTF-8 avec BOM)."""
    with open(dossier / "lot.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["numero", "ligne", "nom", "description", "duree", "graine", "fichier", "fait",
                    "description_envoyee"])
        for e in lot:
            w.writerow([e["numero"], e["ligne"], e["nom"], e["description"], e["duree"], e["graine"], e["base"],
                        "oui" if _fait(dossier, e, infos) else "non", e["description_envoyee"]])


def _archive(dossier, infos, faites):
    chemin = dossier / f"{infos['nom']}.zip"
    with zipfile.ZipFile(chemin, "w", zipfile.ZIP_STORED) as z:  # MP3 et OGG déjà compressés
        for e in faites:
            for b in _bases(e, infos["versions"]):
                for f in infos["formats"]:
                    z.write(dossier / f"{b}.{f}", f"{b}.{f}")
        z.write(dossier / "lot.csv", "lot.csv")
    return chemin
