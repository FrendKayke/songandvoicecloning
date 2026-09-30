"""ACE-Step 1.5 : génération de la chanson via son API REST locale (port 8001)."""
import json
import os
import shutil
import time
from pathlib import Path

import gradio as gr
import requests

from . import config as cfg
from .outils import has_weights, stream_command
from .styles import texte

ACESTEP_COMPONENTS = ["acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"]


def build_prompt(genre, style, instruments, ambiance, voix_base, extra):
    """Description (« caption ») envoyée à ACE-Step. Chaque champ est un texte ou une sélection de liste."""
    parts = [texte(genre), texte(style), texte(instruments), texte(ambiance)]
    if voix_base == "Voix masculine":
        parts.append("male vocals")
    elif voix_base == "Voix féminine":
        parts.append("female vocals")
    parts.append(texte(extra))
    return ", ".join(p.strip() for p in parts if p and p.strip())


def _unwrap(resp_json):
    """L'API enveloppe ses réponses dans {"data": ..., "code": ...}."""
    if isinstance(resp_json, dict) and "data" in resp_json:
        return resp_json["data"]
    return resp_json


def wait_acestep(progress, timeout=15 * 60):
    """Attend que le serveur ACE-Step réponde (il charge ses modèles au démarrage)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(f"{cfg.ACESTEP_URL}/health", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        progress(0.02, desc=f"Démarrage du serveur ACE-Step… ({int(time.time() - t0)} s)")
        time.sleep(3)
    raise gr.Error(
        "Le serveur ACE-Step ne répond pas. Regarde la fenêtre « ACE-Step - ne pas fermer » "
        "(elle s'ouvre avec lancer.bat) pour voir le message d'erreur."
    )


def acestep_generate(prompt, lyrics, langue, duree, bpm, thinking, dest: Path, progress, etape="1/4"):
    """Génère la chanson et l'écrit dans dest. « etape » sert seulement à l'affichage (« 1/4 »…)."""
    wait_acestep(progress)
    payload = {
        "prompt": prompt,
        "lyrics": lyrics,
        "vocal_language": langue,
        "audio_duration": float(duree),
        "audio_format": "wav",
        "batch_size": 1,
        "thinking": bool(thinking),
        "use_random_seed": True,
        # Sans cela, le modèle de langage d'ACE-Step réécrit la description et la langue avant de les
        # passer au générateur (inference.py : dit_input_caption = caption du LM), même sans « thinking » :
        # les styles peu courants (8-bit…) se diluent alors en pop générique.
        "use_cot_caption": False,
        "use_cot_language": False,
    }
    if bpm and int(bpm) > 0:
        payload["bpm"] = int(bpm)

    try:
        r = requests.post(f"{cfg.ACESTEP_URL}/release_task", json=payload, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise gr.Error(
            f"Impossible de joindre le serveur ACE-Step sur {cfg.ACESTEP_URL}. "
            f"Relance lancer.bat. ({e})"
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
        progress(0.1, desc=f"{etape} — Génération de la chanson (ACE-Step)… ({ecoule} s)")
        item = next((i for i in items if i.get("task_id") == task_id), None)
        if not item:
            continue
        status = item.get("status")
        if status == 1:
            result = item["result"]
            if isinstance(result, str):
                result = json.loads(result)
            f = result[0]["file"]
            audio = requests.get(
                f if f.startswith("http") else f"{cfg.ACESTEP_URL}{f}", timeout=120
            )
            audio.raise_for_status()
            dest.write_bytes(audio.content)
            return dest
        if status == 2:
            if thinking:  # nouvel essai sans le LM
                progress(0.15, desc="Échec avec le mode réflexion, nouvel essai sans…")
                return acestep_generate(prompt, lyrics, langue, duree, bpm, False, dest, progress, etape)
            raise gr.Error(f"ACE-Step a échoué : {item.get('result')}")
    raise gr.Error("Délai dépassé (30 min) pour la génération de la chanson. Regarde la fenêtre ACE-Step.")


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
