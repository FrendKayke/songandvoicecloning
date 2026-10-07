"""ACE-Step 1.5 : génération de la chanson via son API REST locale (port 8001)."""
import json
import os
import random
import shutil
import time
from pathlib import Path

import gradio as gr
import requests

from . import config as cfg
from . import serveur_acestep
from .outils import has_weights, stream_command
from .styles import musique

ACESTEP_COMPONENTS = ["acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"]


# Voix chantées (liste « Voix chantées » de l'onglet) → termes de la description. Vocabulaire de la documentation
# d'ACE-Step 1.5 (docs/en/Tutorial.md : « female vocal, male vocal… choir », balise [harmonies]) ; qui chante quelle
# partie se précise dans les paroles (« Couplet 1 (homme) », voir baliser_paroles).
VOIX_CHANTEES = {
    "Automatique": "",
    "Voix masculine": "male vocals",
    "Voix féminine": "female vocals",
    "Duo homme et femme": "male and female duet, male vocals, female vocals, vocal harmonies",
    "Duo de deux hommes": "male duet, two male vocalists, vocal harmonies",
    "Duo de deux femmes": "female duet, two female vocalists, vocal harmonies",
    "Voix principale et chœurs": "lead vocals with backing vocals, vocal harmonies",
    "Chœur (tout le monde chante ensemble)": "choir, group vocals, layered vocal harmonies",
}
PLUSIEURS_VOIX = [v for v in VOIX_CHANTEES if v not in ("Automatique", "Voix masculine", "Voix féminine")]


def build_prompt(genre, style, instruments, ambiance, voix_base, extra):
    """Description (« caption ») envoyée à ACE-Step. Chaque champ est un texte ou une sélection de liste."""
    parts = [musique(genre), musique(style), musique(instruments), musique(ambiance), VOIX_CHANTEES.get(voix_base, "")]
    parts.append(musique(extra))
    termes = []  # un même terme demandé dans plusieurs champs (genre et ambiance « médiéval »…) n'est écrit qu'une fois
    for t in ", ".join(p.strip() for p in parts if p and p.strip()).split(","):
        if t.strip() and t.strip().lower() not in (x.lower() for x in termes):
            termes.append(t.strip())
    return ", ".join(termes)


# Titres de section écrits en français (ou sans crochets) → balises d'ACE-Step : sans crochets, « Couplet 2 » était
# chanté comme une parole et « Refrain » ne disait pas de reprendre le refrain (constaté)
SECTIONS = [
    (r"(couplet|verse|strophe)", "Verse"),
    (r"(pre ?refrain|pre ?chorus)", "Pre-Chorus"),
    (r"(refrain|chorus)", "Chorus"),
    (r"(pont|bridge)", "Bridge"),
    (r"(intro|introduction)", "Intro"),
    (r"(outro|fin|final|conclusion)", "Outro"),
    (r"(solo|instrumental|interlude)", "Instrumental"),
]
# Qui chante une section, écrit après son titre : « Couplet 1 (homme) », « Refrain - ensemble », « Pont : chœur »
# → [Verse 1 - male vocal]… (forme « [Section - précision] » de la documentation d'ACE-Step, « [Chorus - anthemic] »)
CHANTEURS = [
    (r"(homme|masculin|masculine|garcon|lui|il|chanteur|male|man)", "male vocal"),
    (r"(femme|feminin|feminine|fille|elle|chanteuse|female|woman)", "female vocal"),
    (r"(ensemble|tous|toutes|duo|les deux|both|together|duet)", "duet, harmonies"),
    (r"(choeur|choeurs|chorale|choir|groupe)", "choir"),
    (r"(reponse|question reponse|call and response)", "call and response"),
]


def baliser_paroles(paroles):
    """« Couplet 2 », « Refrain : », « [Pont] »… seuls sur leur ligne → [Verse 2], [Chorus], [Bridge] ; avec qui
    chante (« Couplet 1 (homme) », « Refrain - ensemble ») → [Verse 1 - male vocal], [Chorus - duet, harmonies]. Une
    section laissée vide (« Refrain » sans texte) reprend le texte de la dernière section du même nom."""
    import re

    from .styles import _normal

    lignes, sections, courante, vues = [], {}, None, set()
    for brute in (paroles or "").splitlines():
        n = _normal(brute.strip().strip("[]()").rstrip(":"))
        balise = None
        for motif, nom in SECTIONS:
            m = re.fullmatch(motif + r"(?: (\d+))?(?: (.+))?", n)
            if not m or len(brute.strip()) > 40:
                continue
            numero, suite = m.group(2), m.group(3)
            chanteur = None
            if suite:  # seul un chanteur est accepté après le titre (sinon c'est une parole : « Refrain de ma vie »)
                chanteur = next((v for motif_c, v in CHANTEURS if re.fullmatch(motif_c, suite)), None)
                if not chanteur:
                    continue
            balise = nom + (f" {numero}" if numero else "") + (f" - {chanteur}" if chanteur else "")
            break
        if balise:
            _reprendre(lignes, sections, courante, vues)
            courante, vues = balise.split(" ")[0], set()
            lignes.append(f"[{balise}]")
            sections.setdefault(courante, None)
            continue
        lignes.append(brute)
        if courante and brute.strip():
            vues.add(len(lignes) - 1)
    _reprendre(lignes, sections, courante, vues)
    return "\n".join(lignes)


def _reprendre(lignes, sections, courante, vues):
    """Fin d'une section : retient son texte, ou recopie celui de la précédente du même nom si elle est vide."""
    if not courante:
        return
    if vues:
        sections[courante] = [lignes[i] for i in sorted(vues)]
    elif sections.get(courante):
        fin = len(lignes)
        while fin and not lignes[fin - 1].strip():  # le texte repris se place juste après la balise
            fin -= 1
        lignes[fin:fin] = sections[courante]


def _unwrap(resp_json):
    """L'API enveloppe ses réponses dans {"data": ..., "code": ...}."""
    if isinstance(resp_json, dict) and "data" in resp_json:
        return resp_json["data"]
    return resp_json


def wait_acestep(progress, timeout=15 * 60):
    """Serveur ACE-Step prêt : démarré par l'application s'il ne tourne pas (il a pu être arrêté pour libérer la
    carte graphique), puis attente de sa réponse (il charge ses modèles au démarrage)."""
    serveur_acestep.assurer(progress, timeout)


def _valeur_formulaire(v):
    """Les paramètres d'un envoi multipart arrivent en texte (le serveur relit « true »/« false »)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def graines(versions, graine=None):
    """Graine de chaque version : celle demandée pour la première (si > 0), aléatoires sinon.
    On les choisit nous-mêmes car /query_result ne renvoie pas la graine de chaque fichier."""
    tirage = [random.randint(1, 2**31 - 1) for _ in range(versions)]
    if graine and int(graine) > 0:
        tirage[0] = int(graine)
    return tirage


# Longueurs lues par ACE-Step (acestep/core/generation/handler/conditioning_text.py, tokeniseur Qwen3, coupure sans
# prévenir) : 256 jetons pour la consigne + la description + les métadonnées (55 jetons hors description, mesuré),
# 2048 jetons pour les paroles. Estimation sans tokeniseur (l'application n'en a pas) : ~4,4 caractères par jeton
# pour une description anglaise, ~3,1 pour des paroles françaises (mesurés avec le vrai tokeniseur) ; marge prise.
DESCRIPTION_MAX_CAR = 700
PAROLES_MAX_CAR = 5500


def longueurs_trop_grandes(params):
    """Avertissements (textes) si la description ou les paroles dépassent ce qu'ACE-Step lit."""
    avertissements = []
    description = params.get("prompt") or ""
    if len(description) > DESCRIPTION_MAX_CAR:
        avertissements.append(
            f"Description de {len(description)} caractères : ACE-Step n'en lit qu'environ {DESCRIPTION_MAX_CAR} "
            f"(256 jetons) ; la fin (« …{description[-60:]} ») et le tempo / la tonalité demandés sont ignorés. "
            "Raccourcis la description.")
    paroles = params.get("lyrics") or ""
    if len(paroles) > PAROLES_MAX_CAR:
        avertissements.append(
            f"Paroles de {len(paroles)} caractères : ACE-Step n'en lit qu'environ {PAROLES_MAX_CAR} (2048 jetons) ; "
            f"la fin (« …{paroles[-60:].strip()} ») ne sera pas chantée. Raccourcis les paroles ou fais deux chansons.")
    return avertissements


def generer(params, dests, progress, etape="1/1", fichiers=None, graine=None):
    """Lance une tâche ACE-Step (/release_task), attend le résultat et écrit une version par fichier de dests.

    params : paramètres de l'API (prompt, lyrics, task_type, repainting_start…), vérifiés dans
             acestep/api/http/release_task_request_builder.py ;
    fichiers : {"reference_audio": chemin, "src_audio": chemin} envoyés en multipart — le serveur refuse les
               chemins absolus hors de son dossier temporaire (release_task_audio_paths.validate_audio_path) ;
    Renvoie [(chemin, graine), …] dans l'ordre des versions (l'ordre des graines est celui des fichiers).
    """
    for avertissement in longueurs_trop_grandes(params):
        try:
            gr.Warning(avertissement, duration=None)
        except Exception:  # noqa: BLE001 - hors d'un événement Gradio
            pass
    wait_acestep(progress)
    seeds = graines(len(dests), graine)
    payload = {
        "audio_format": "wav",
        "batch_size": len(dests),
        "use_random_seed": False,
        "seed": ",".join(str(x) for x in seeds),
        # Sans cela, le modèle de langage d'ACE-Step réécrit la description et la langue avant de les
        # passer au générateur (inference.py : dit_input_caption = caption du LM), même sans « thinking » :
        # les styles peu courants (8-bit…) se diluent alors en pop générique.
        "use_cot_caption": False,
        "use_cot_language": False,
        **params,
    }
    fichiers = {k: v for k, v in (fichiers or {}).items() if v}
    try:
        if fichiers:
            ouverts = {k: (Path(v).name, open(v, "rb")) for k, v in fichiers.items()}
            try:
                r = requests.post(f"{cfg.ACESTEP_URL}/release_task", timeout=120, files=ouverts,
                                  data={k: _valeur_formulaire(v) for k, v in payload.items() if v is not None})
            finally:
                for _, f in ouverts.values():
                    f.close()
        else:
            r = requests.post(f"{cfg.ACESTEP_URL}/release_task", json=payload, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise gr.Error(
            f"Impossible de joindre le serveur ACE-Step sur {cfg.ACESTEP_URL}. "
            f"Fin du journal du serveur ({serveur_acestep.journal()}) :\n{serveur_acestep.fin_du_journal()}\n({e})"
        )
    task_id = _unwrap(r.json())["task_id"]

    t0 = time.time()
    while time.time() - t0 < 30 * 60:
        time.sleep(3)
        ecoule = int(time.time() - t0)
        # Pendant le chargement des modèles, le serveur peut mettre longtemps à répondre :
        # une réponse lente ou perdue n'est pas une erreur, on redemande simplement.
        try:
            q = requests.post(
                f"{cfg.ACESTEP_URL}/query_result", json={"task_id_list": [task_id]}, timeout=120
            )
            q.raise_for_status()
            items = _unwrap(q.json())
        except (requests.RequestException, ValueError):
            progress(0.1, desc=f"{etape} — Le serveur ACE-Step est occupé, génération en cours… ({ecoule} s)")
            continue
        progress(0.1, desc=f"{etape} — Génération de la musique (ACE-Step)… ({ecoule} s)")
        item = next((i for i in items if i.get("task_id") == task_id), None)
        if not item:
            continue
        status = item.get("status")
        if status == 1:
            result = item["result"]
            if isinstance(result, str):
                result = json.loads(result)
            urls = [x["file"] for x in result if x.get("file")]
            if len(urls) < len(dests):
                raise gr.Error(f"ACE-Step n'a renvoyé que {len(urls)} version(s) sur {len(dests)}.")
            for f, dest in zip(urls, dests):
                audio = requests.get(f if f.startswith("http") else f"{cfg.ACESTEP_URL}{f}", timeout=120)
                audio.raise_for_status()
                Path(dest).write_bytes(audio.content)
            return list(zip([Path(d) for d in dests], seeds))
        if status == 2:
            if params.get("thinking"):  # nouvel essai sans le LM
                progress(0.15, desc="Échec avec le mode réflexion, nouvel essai sans…")
                return generer({**params, "thinking": False}, dests, progress, etape, fichiers, seeds[0])
            raise gr.Error(f"ACE-Step a échoué : {item.get('result')}")
    raise gr.Error(f"Délai dépassé (30 min) pour la génération. Journal du serveur : {serveur_acestep.journal()}")


def text2music_params(prompt, lyrics, langue, duree, bpm, thinking, negatif=None):
    """Paramètres d'une génération à partir du texte (task_type par défaut : text2music)."""
    params = {
        "prompt": prompt,
        "lyrics": lyrics,
        "vocal_language": langue,
        "audio_duration": float(duree),
        "thinking": bool(thinking),
    }
    if bpm and int(bpm) > 0:
        params["bpm"] = int(bpm)
    # Ce qu'il faut éviter : seul le modèle de langage en tient compte, donc seulement en mode réflexion
    # (le générateur n'a pas de prompt négatif).
    if negatif:
        params["lm_negative_prompt"] = negatif
    return params


def acestep_generate(prompt, lyrics, langue, duree, bpm, thinking, dest: Path, progress, etape="1/4",
                     negatif=None):
    """Génère une version et l'écrit dans dest. « etape » sert seulement à l'affichage (« 1/4 »…)."""
    params = text2music_params(prompt, lyrics, langue, duree, bpm, thinking, negatif)
    return generer(params, [dest], progress, etape)[0][0]


# --- Modèles -----------------------------------------------------------------
def ckpt_dir() -> Path:
    env = os.environ.get("ACESTEP_CHECKPOINTS_DIR")  # même règle que ACE-Step
    return Path(env).expanduser().resolve() if env else cfg.ACESTEP_DIR / "checkpoints"


def missing_components():
    ck = ckpt_dir()
    return [c for c in ACESTEP_COMPONENTS if not has_weights(ck / c)]


def download():
    ck = ckpt_dir()
    if not cfg.ACESTEP_DIR.exists():
        yield f"❌ Dossier ACE-Step introuvable : {cfg.ACESTEP_DIR} (variable ACESTEP_DIR)."
        return
    base = ["-m", "acestep.model_downloader", "--dir", str(ck)]
    if Path(cfg.ACESTEP_PYTHON).exists():
        cmd = [cfg.ACESTEP_PYTHON] + base
    elif shutil.which("uv"):
        cmd = ["uv", "run", "python"] + base
    else:
        yield f"❌ Python d'ACE-Step introuvable : {cfg.ACESTEP_PYTHON} (variable ACESTEP_PYTHON), et `uv` n'est pas installé."
        return
    yield from stream_command(cmd, cfg.ACESTEP_DIR, f"Téléchargement des modèles ACE-Step vers {ck} …")
