"""
Studio Voix — mini logiciel local :
  1. enregistre / importe un échantillon de TA voix
  2. génère une chanson avec ACE-Step 1.5 à partir d'un « preprompt »
     (genre, style, instruments, ambiance) + tes paroles
  3. sépare la voix de l'instrumental (Demucs)
  4. remplace la voix chantée par la tienne (Seed-VC, conversion de voix chantée)
  5. remixe le tout
Un onglet « Modèles » indique où sont stockés les modèles et permet de les télécharger.

Prérequis : voir README.md (serveur ACE-Step lancé + Seed-VC installé).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import requests
import soundfile as sf

# --------------------------------------------------------------------------
# Configuration (modifiable via variables d'environnement)
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
# Les moteurs sont installés par INSTALLER.bat dans un dossier au chemin court
# (évite la limite de 260 caractères de Windows) : <lecteur de l'appli>:\StudioVoix
ENG_DIR = Path(os.environ.get("STUDIOVOIX_MOTEURS", str(Path(APP_DIR.anchor) / "StudioVoix")))
ACESTEP_URL = os.environ.get("ACESTEP_URL", "http://127.0.0.1:8001").rstrip("/")
SEEDVC_DIR = Path(os.environ.get("SEEDVC_DIR", str(ENG_DIR / "seed-vc")))


def _venv_python(root: Path) -> str:
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return str(root / ".venv" / sub)


SEEDVC_PYTHON = os.environ.get("SEEDVC_PYTHON", _venv_python(SEEDVC_DIR))
ACESTEP_DIR = Path(os.environ.get("ACESTEP_DIR", str(ENG_DIR / "ace-step")))
ACESTEP_PYTHON = os.environ.get("ACESTEP_PYTHON", _venv_python(ACESTEP_DIR))

DATA_DIR = APP_DIR / "data"
VOICES_DIR = DATA_DIR / "voices"
SONGS_DIR = DATA_DIR / "songs"
for d in (VOICES_DIR, SONGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SR = 44100  # fréquence de sortie (le modèle chanté de Seed-VC produit du 44,1 kHz)

LANGUES = {
    "Français": "fr",
    "Anglais": "en",
    "Espagnol": "es",
    "Italien": "it",
    "Allemand": "de",
    "Portugais": "pt",
    "Japonais": "ja",
    "Coréen": "ko",
    "Chinois": "zh",
}


# --------------------------------------------------------------------------
# 1) Voix de référence
# --------------------------------------------------------------------------
def list_voices():
    return sorted(p.stem for p in VOICES_DIR.glob("*.wav"))


def save_voice(audio_path, name):
    if not audio_path:
        raise gr.Error("Enregistre ou importe d'abord un échantillon de voix.")
    name = "".join(c for c in (name or "").strip() if c.isalnum() or c in "-_ ").strip()
    if not name:
        raise gr.Error("Donne un nom à cette voix.")
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    duree = len(y) / SR
    if duree < 5:
        raise gr.Error(f"Échantillon trop court ({duree:.1f} s). Vise 10 à 25 secondes.")
    y = y[: 30 * SR]  # Seed-VC exploite 1 à 30 s de référence
    peak = float(np.max(np.abs(y))) or 1.0
    y = y / peak * 0.95
    sf.write(VOICES_DIR / f"{name}.wav", y, SR)
    voices = list_voices()
    msg = f"Voix « {name} » enregistrée ({min(duree, 30):.0f} s utilisées)."
    return msg, gr.update(choices=voices, value=name)


# --------------------------------------------------------------------------
# 2) Génération de la chanson (ACE-Step 1.5, API REST locale)
# --------------------------------------------------------------------------
def build_prompt(genre, style, instruments, ambiance, voix_base, extra):
    parts = [genre, style, instruments, ambiance]
    if voix_base == "Voix masculine":
        parts.append("male vocals")
    elif voix_base == "Voix féminine":
        parts.append("female vocals")
    parts.append(extra)
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
            if requests.get(f"{ACESTEP_URL}/health", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        progress(0.02, desc=f"Démarrage du serveur ACE-Step… ({int(time.time() - t0)} s)")
        time.sleep(3)
    raise gr.Error(
        "Le serveur ACE-Step ne répond pas. Regarde la fenêtre « ACE-Step - ne pas fermer » "
        "(elle s'ouvre avec lancer.bat) pour voir le message d'erreur."
    )


def acestep_generate(prompt, lyrics, langue, duree, bpm, thinking, dest: Path, progress):
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
    }
    if bpm and int(bpm) > 0:
        payload["bpm"] = int(bpm)

    try:
        r = requests.post(f"{ACESTEP_URL}/release_task", json=payload, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise gr.Error(
            f"Impossible de joindre le serveur ACE-Step sur {ACESTEP_URL}. "
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
                f"{ACESTEP_URL}/query_result", json={"task_id_list": [task_id]}, timeout=120
            )
            q.raise_for_status()
            items = _unwrap(q.json())
        except (requests.RequestException, ValueError):
            progress(0.1, desc=f"1/4 — Le serveur ACE-Step est occupé, génération en cours… ({ecoule} s)")
            continue
        progress(0.1, desc=f"1/4 — Génération de la chanson (ACE-Step)… ({ecoule} s)")
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
                f if f.startswith("http") else f"{ACESTEP_URL}{f}", timeout=120
            )
            audio.raise_for_status()
            dest.write_bytes(audio.content)
            return dest
        if status == 2:
            if thinking:  # nouvel essai sans le LM
                progress(0.15, desc="Échec avec le mode réflexion, nouvel essai sans…")
                return acestep_generate(prompt, lyrics, langue, duree, bpm, False, dest, progress)
            raise gr.Error(f"ACE-Step a échoué : {item.get('result')}")
    raise gr.Error("Délai dépassé (30 min) pour la génération de la chanson. Regarde la fenêtre ACE-Step.")


# --------------------------------------------------------------------------
# 3) Séparation voix / instrumental (Demucs)
# --------------------------------------------------------------------------
def separate_vocals(song: Path, workdir: Path):
    out = workdir / "demucs"
    cmd = [SEEDVC_PYTHON, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
           "-o", str(out), str(song)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise gr.Error(f"Demucs a échoué :\n{p.stderr[-1500:]}")
    vocals = next(out.rglob("vocals.wav"), None)
    instru = next(out.rglob("no_vocals.wav"), None)
    if not vocals or not instru:
        raise gr.Error("Demucs n'a pas produit les fichiers attendus.")
    return vocals, instru


# --------------------------------------------------------------------------
# 4) Conversion de voix chantée (Seed-VC)
# --------------------------------------------------------------------------
def convert_voice(vocals: Path, voice_ref: Path, semitones: int, steps: int, workdir: Path):
    if not SEEDVC_DIR.exists():
        raise gr.Error(f"Dossier Seed-VC introuvable : {SEEDVC_DIR} (variable SEEDVC_DIR).")
    if not Path(SEEDVC_PYTHON).exists():
        raise gr.Error(f"Python de Seed-VC introuvable : {SEEDVC_PYTHON} (variable SEEDVC_PYTHON).")
    out = workdir / "seedvc"
    out.mkdir(exist_ok=True)
    cmd = [
        SEEDVC_PYTHON, "inference.py",
        "--source", str(vocals),
        "--target", str(voice_ref),
        "--output", str(out),
        "--diffusion-steps", str(int(steps)),
        "--f0-condition", "True",          # obligatoire pour la voix chantée
        "--auto-f0-adjust", "False",
        "--semi-tone-shift", str(int(semitones)),
        "--fp16", "True",
    ]
    p = subprocess.run(cmd, cwd=SEEDVC_DIR, capture_output=True, text=True)
    if p.returncode != 0:
        raise gr.Error(f"Seed-VC a échoué :\n{p.stderr[-1500:]}")
    result = next(out.glob("*.wav"), None)
    if not result:
        raise gr.Error("Seed-VC n'a produit aucun fichier.")
    return result


# --------------------------------------------------------------------------
# 5) Mixage
# --------------------------------------------------------------------------
def _load_stereo(path, sr=SR):
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, canaux)
    y = y.T
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    return y[:2]


def mix(vocals_path, instru_path, out_path, gain_voix=1.0, gain_instru=1.0):
    v = _load_stereo(vocals_path) * gain_voix
    i = _load_stereo(instru_path) * gain_instru
    n = max(v.shape[1], i.shape[1])
    v = np.pad(v, ((0, 0), (0, n - v.shape[1])))
    i = np.pad(i, ((0, 0), (0, n - i.shape[1])))
    m = v + i
    peak = float(np.max(np.abs(m))) or 1.0
    if peak > 0.95:
        m = m / peak * 0.95
    sf.write(str(out_path), m.T, SR)
    return out_path


# --------------------------------------------------------------------------
# Pipeline complet
# --------------------------------------------------------------------------
def creer_chanson(
    voix, genre, style, instruments, ambiance, extra, voix_base,
    paroles, langue_label, duree, bpm, thinking,
    semitones, steps, gain_voix, gain_instru,
    progress=gr.Progress(),
):
    if not voix:
        raise gr.Error("Choisis (ou enregistre) d'abord une voix dans l'onglet « Ma voix ».")
    voice_ref = VOICES_DIR / f"{voix}.wav"
    if not voice_ref.exists():
        raise gr.Error(f"Voix introuvable : {voix}")
    if not (genre or style or instruments or ambiance or extra):
        raise gr.Error("Renseigne au moins un élément de style (genre, instruments…).")

    workdir = SONGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir.mkdir(parents=True)

    prompt = build_prompt(genre, style, instruments, ambiance, voix_base, extra)
    lyrics = (paroles or "").strip() or "[Instrumental]"
    (workdir / "prompt.txt").write_text(f"{prompt}\n\n{lyrics}\n", encoding="utf-8")

    progress(0.05, desc="1/4 — Génération de la chanson (ACE-Step)…")
    song = acestep_generate(
        prompt, lyrics, LANGUES[langue_label], duree, bpm, thinking, workdir / "chanson_brute.wav", progress
    )

    if lyrics == "[Instrumental]":
        return str(song), str(song), None, None, f"Instrumental généré. Dossier : {workdir}"

    progress(0.45, desc="2/4 — Séparation voix / instrumental (Demucs)…")
    vocals, instru = separate_vocals(song, workdir)

    progress(0.6, desc="3/4 — Remplacement par ta voix (Seed-VC)…")
    converted = convert_voice(vocals, voice_ref, semitones, steps, workdir)

    progress(0.92, desc="4/4 — Mixage…")
    final = mix(converted, instru, workdir / "chanson_finale.wav", gain_voix, gain_instru)

    shutil.copy(converted, workdir / "voix_convertie.wav")
    return (
        str(final), str(song), str(workdir / "voix_convertie.wav"), str(instru),
        f"Terminé. Tous les fichiers sont dans : {workdir}",
    )


# --------------------------------------------------------------------------
# Gestion des modèles : emplacement, état, téléchargement
# --------------------------------------------------------------------------
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".th", ".ckpt"}
ACESTEP_COMPONENTS = ["acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"]


def acestep_ckpt_dir() -> Path:
    env = os.environ.get("ACESTEP_CHECKPOINTS_DIR")  # même règle que ACE-Step
    return Path(env).expanduser().resolve() if env else ACESTEP_DIR / "checkpoints"


def seedvc_ckpt_dir() -> Path:
    return SEEDVC_DIR / "checkpoints"  # Seed-VC utilise ./checkpoints (relatif à son dossier)


def demucs_ckpt_dir() -> Path:
    home = os.environ.get("TORCH_HOME") or str(Path.home() / ".cache" / "torch")
    return Path(home) / "hub" / "checkpoints"


def _has_weights(folder: Path) -> bool:
    return folder.is_dir() and any(f.suffix in WEIGHT_SUFFIXES for f in folder.rglob("*") if f.is_file())


def models_status_md() -> str:
    ck = acestep_ckpt_dir()
    ace_missing = [c for c in ACESTEP_COMPONENTS if not _has_weights(ck / c)]
    sv = seedvc_ckpt_dir()
    sv_needed = {
        "modèle de chant (DiT f0 44 kHz)": "DiT_seed_v2_uvit_whisper_base_f0_44k*.pth",
        "détecteur de hauteur (RMVPE)": "rmvpe.pt",
        "encodeur de voix (CAMPPlus)": "campplus_cn_common.bin",
    }
    sv_missing = [n for n, pat in sv_needed.items() if not any(sv.rglob(pat))]
    dm = demucs_ckpt_dir()
    dm_ok = dm.is_dir() and any(dm.glob("*.th"))

    def cell(ok, missing=None):
        if ok:
            return "✅ présent"
        return "❌ absent" + (f" ({', '.join(missing)})" if missing else "")

    return (
        "| Composant | État | Dossier de stockage |\n|---|---|---|\n"
        f"| ACE-Step 1.5 (génération de la chanson) | {cell(not ace_missing, ace_missing)} | `{ck}` |\n"
        f"| Seed-VC (conversion de voix chantée) | {cell(not sv_missing, sv_missing)} | `{sv}` |\n"
        f"| Demucs (séparation voix / musique) | {cell(dm_ok)} | `{dm}` |\n\n"
        f"Tes voix enregistrées : `{VOICES_DIR}`  \nTes chansons : `{SONGS_DIR}`\n\n"
        "*Pour Demucs, l'état indique la présence d'au moins un fichier `.th` dans ce dossier.*"
    )


def _stream_command(cmd, cwd, header):
    """Lance une commande et renvoie sa sortie au fur et à mesure (pour l'affichage)."""
    log = header + "\n"
    yield log
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except OSError as e:
        yield log + f"\n❌ Impossible de lancer la commande : {e}"
        return
    for line in proc.stdout:
        log = (log + line.rstrip() + "\n")[-8000:]
        yield log
    code = proc.wait()
    yield log + ("\n✅ Terminé." if code == 0 else f"\n❌ Échec (code {code}). Regarde les dernières lignes ci-dessus.")


def download_acestep():
    ck = acestep_ckpt_dir()
    if not ACESTEP_DIR.exists():
        yield f"❌ Dossier ACE-Step introuvable : {ACESTEP_DIR} (variable ACESTEP_DIR)."
        return
    base = ["-m", "acestep.model_downloader", "--dir", str(ck)]
    if Path(ACESTEP_PYTHON).exists():
        cmd = [ACESTEP_PYTHON] + base
    elif shutil.which("uv"):
        cmd = ["uv", "run", "python"] + base
    else:
        yield f"❌ Python d'ACE-Step introuvable : {ACESTEP_PYTHON} (variable ACESTEP_PYTHON), et `uv` n'est pas installé."
        return
    yield from _stream_command(cmd, ACESTEP_DIR, f"Téléchargement des modèles ACE-Step vers {ck} …")


def download_seedvc():
    """Seed-VC télécharge ses modèles au premier usage : on lance donc une mini-conversion de test."""
    if not SEEDVC_DIR.exists() or not Path(SEEDVC_PYTHON).exists():
        yield f"❌ Seed-VC introuvable ({SEEDVC_DIR}). Vérifie SEEDVC_DIR / SEEDVC_PYTHON."
        return
    tmp = DATA_DIR / "_test_telechargement"
    tmp.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 3, 3 * SR, endpoint=False)
    src, tgt = tmp / "source.wav", tmp / "cible.wav"
    sf.write(src, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), SR)
    sf.write(tgt, (0.3 * np.sin(2 * np.pi * 330 * t)).astype("float32"), SR)
    cmd = [SEEDVC_PYTHON, "inference.py", "--source", str(src), "--target", str(tgt),
           "--output", str(tmp), "--diffusion-steps", "4", "--f0-condition", "True"]
    yield from _stream_command(
        cmd, SEEDVC_DIR,
        f"Téléchargement des modèles Seed-VC vers {seedvc_ckpt_dir()} (via une mini-conversion de test) …",
    )


def download_demucs():
    code = "from demucs.pretrained import get_model; get_model('htdemucs'); print('Modèle htdemucs prêt.')"
    if not Path(SEEDVC_PYTHON).exists():
        yield f"❌ Python de Seed-VC introuvable : {SEEDVC_PYTHON}. Lance INSTALLER.bat."
        return
    yield from _stream_command(
        [SEEDVC_PYTHON, "-c", code], APP_DIR,
        f"Téléchargement du modèle Demucs vers {demucs_ckpt_dir()} …",
    )


def open_folder(path, create=False):
    p = Path(path)
    if not p.exists():
        if not create:
            raise gr.Error(
                f"Ce dossier n'existe pas : {p}. Le composant n'est probablement pas encore installé "
                "(lance INSTALLER.bat)."
            )
        p.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception as e:
        raise gr.Error(f"Impossible d'ouvrir le dossier : {e}")


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Studio Voix") as demo:
        gr.Markdown("# 🎤 Studio Voix\nClone ta voix, écris tes paroles, choisis le style : la chanson est générée en local.")

        with gr.Tab("1. Ma voix"):
            gr.Markdown(
                "Enregistre **10 à 25 secondes** de toi, dans une pièce calme, sans musique ni écho "
                "(idéalement en chantant, sinon en parlant). Seule une voix propre donne un bon résultat."
            )
            with gr.Row():
                audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Échantillon de ta voix")
                with gr.Column():
                    nom = gr.Textbox(label="Nom de la voix", value="laurent")
                    btn_save = gr.Button("Enregistrer cette voix", variant="primary")
                    msg_voice = gr.Markdown()

        with gr.Tab("2. Créer une chanson"):
            with gr.Row():
                with gr.Column():
                    voix = gr.Dropdown(choices=list_voices(), value=(list_voices() or [None])[0], label="Voix à utiliser")
                    genre = gr.Textbox(label="Genre", placeholder="pop rock, chanson française, hip-hop…")
                    style = gr.Textbox(label="Style / références sonores", placeholder="énergique, années 80, lo-fi, épique…")
                    instruments = gr.Textbox(label="Instruments", placeholder="guitare acoustique, piano, batterie, synthé…")
                    ambiance = gr.Textbox(label="Ambiance / thème", placeholder="mélancolique, joyeux, nocturne…")
                    extra = gr.Textbox(label="Autres consignes (facultatif)", placeholder="refrain accrocheur, pont instrumental…")
                with gr.Column():
                    paroles = gr.Textbox(
                        label="Paroles (avec [Verse], [Chorus], [Bridge]…)", lines=16,
                        placeholder="[Verse 1]\nTes paroles…\n\n[Chorus]\nLe refrain…",
                    )
            with gr.Row():
                langue = gr.Dropdown(list(LANGUES), value="Français", label="Langue des paroles")
                duree = gr.Slider(30, 240, value=120, step=10, label="Durée (s)")
                bpm = gr.Number(value=0, precision=0, label="BPM (0 = auto)")
                thinking = gr.Checkbox(value=True, label="Mode réflexion (LM) — meilleure qualité")
            with gr.Accordion("Réglages voix (avancé)", open=False):
                voix_base = gr.Dropdown(
                    ["Automatique", "Voix masculine", "Voix féminine"], value="Automatique",
                    label="Voix chantée de base générée par ACE-Step",
                    info="Choisis le genre le plus proche de ta voix : moins de décalage à corriger.",
                )
                semitones = gr.Slider(-12, 12, value=0, step=1, label="Décalage de hauteur (demi-tons)",
                                      info="Voix de base féminine → voix masculine : essaie -12. L'inverse : +12.")
                steps = gr.Slider(25, 50, value=40, step=5, label="Étapes de diffusion Seed-VC (30–50 conseillé pour le chant)")
                gain_voix = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Volume voix")
                gain_instru = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Volume instrumental")

            btn = gr.Button("🎵 Créer la chanson", variant="primary")
            statut = gr.Markdown()
            final = gr.Audio(label="Chanson finale (avec ta voix)", type="filepath")
            with gr.Accordion("Étapes intermédiaires", open=False):
                brute = gr.Audio(label="Chanson brute ACE-Step", type="filepath")
                voix_conv = gr.Audio(label="Voix convertie", type="filepath")
                instru_out = gr.Audio(label="Instrumental", type="filepath")

        with gr.Tab("3. Modèles"):
            gr.Markdown(
                "Les modèles sont volumineux (plusieurs Go au total) et ne sont téléchargés qu'une seule fois. "
                "Vérifie ici leur présence et leur emplacement, ou lance le téléchargement."
            )
            status = gr.Markdown(models_status_md())
            btn_refresh = gr.Button("🔄 Actualiser l'état")
            log = gr.Textbox(label="Journal de téléchargement", lines=14, max_lines=14, autoscroll=True, interactive=False)
            with gr.Row():
                b_ace = gr.Button("⬇️ Télécharger ACE-Step", variant="primary")
                b_sv = gr.Button("⬇️ Télécharger Seed-VC", variant="primary")
                b_dm = gr.Button("⬇️ Télécharger Demucs", variant="primary")
            with gr.Row():
                o_ace = gr.Button("📂 Ouvrir dossier ACE-Step")
                o_sv = gr.Button("📂 Ouvrir dossier Seed-VC")
                o_dm = gr.Button("📂 Ouvrir dossier Demucs")
                o_data = gr.Button("📂 Ouvrir mes chansons")

        btn_save.click(save_voice, [audio_in, nom], [msg_voice, voix])
        btn_refresh.click(models_status_md, None, status)
        for b, fn in ((b_ace, download_acestep), (b_sv, download_seedvc), (b_dm, download_demucs)):
            b.click(fn, None, log).then(models_status_md, None, status)
        o_ace.click(lambda: open_folder(acestep_ckpt_dir()))
        o_sv.click(lambda: open_folder(seedvc_ckpt_dir()))
        o_dm.click(lambda: open_folder(demucs_ckpt_dir()))
        o_data.click(lambda: open_folder(SONGS_DIR, create=True))
        demo.load(models_status_md, None, status)
        btn.click(
            creer_chanson,
            [voix, genre, style, instruments, ambiance, extra, voix_base, paroles, langue, duree, bpm,
             thinking, semitones, steps, gain_voix, gain_instru],
            [final, brute, voix_conv, instru_out, statut],
        )
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(inbrowser=True)
