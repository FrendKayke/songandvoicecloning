"""Pipeline complet : ACE-Step → Demucs → Seed-VC → mixage, ou ACE-Step seul (musique seule)."""
import shutil
from datetime import datetime

import gradio as gr

from . import acestep, demucs, seedvc
from . import config as cfg
from .mixage import mix

MODE_MA_VOIX = "Chanson avec ma voix"
MODE_VOIX_ACE = "Chanson avec la voix d'ACE-Step"
MODE_INSTRU = "Instrumental"
MODES = [MODE_MA_VOIX, MODE_VOIX_ACE, MODE_INSTRU]
INSTRUMENTAL = "[Instrumental]"  # paroles reconnues par ACE-Step comme « sans voix » (server_utils.is_instrumental)


def creer_chanson(
    voix, genre, style, instruments, ambiance, extra, voix_base,
    paroles, langue_label, duree, bpm, thinking,
    semitones, steps, gain_voix, gain_instru,
    mode=MODE_MA_VOIX,
    progress=gr.Progress(),
):
    mode = mode or MODE_MA_VOIX
    if mode not in MODES:
        raise gr.Error(f"Mode inconnu : {mode}")
    voice_ref = None
    if mode == MODE_MA_VOIX:
        if not voix:
            raise gr.Error("Choisis (ou enregistre) d'abord une voix dans l'onglet « Bibliothèque de voix ».")
        voice_ref = cfg.VOICES_DIR / f"{voix}.wav"
        if not voice_ref.exists():
            raise gr.Error(f"Voix introuvable : {voix}")
    if not (genre or style or instruments or ambiance or extra):
        raise gr.Error("Renseigne au moins un élément de style (genre, instruments…).")

    if mode == MODE_INSTRU:
        lyrics = INSTRUMENTAL
        voix_base = "Automatique"  # pas de voix : aucune consigne de voix dans le prompt
    else:
        lyrics = (paroles or "").strip()
        if not lyrics:
            if mode == MODE_VOIX_ACE:
                raise gr.Error("Écris des paroles, ou choisis le mode « Instrumental ».")
            lyrics = INSTRUMENTAL

    workdir = cfg.SONGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir.mkdir(parents=True)

    prompt = acestep.build_prompt(genre, style, instruments, ambiance, voix_base, extra)
    (workdir / "prompt.txt").write_text(f"{prompt}\n\n{lyrics}\n", encoding="utf-8")

    etape = "1/4" if mode == MODE_MA_VOIX else "1/1"
    progress(0.05, desc=f"{etape} — Génération de la chanson (ACE-Step)…")
    song = acestep.acestep_generate(
        prompt, lyrics, cfg.LANGUES[langue_label], duree, bpm, thinking, workdir / "chanson_brute.wav",
        progress, etape,
    )

    if lyrics == INSTRUMENTAL:
        return str(song), str(song), None, None, f"Instrumental généré. Dossier : {workdir}"
    if mode == MODE_VOIX_ACE:
        return (
            str(song), str(song), None, None,
            f"Chanson générée avec la voix d'ACE-Step (sans conversion). Dossier : {workdir}",
        )

    progress(0.45, desc="2/4 — Séparation voix / instrumental (Demucs)…")
    vocals, instru = demucs.separate_vocals(song, workdir)

    progress(0.6, desc="3/4 — Remplacement par ta voix (Seed-VC)…")
    converted = seedvc.convert_voice(vocals, voice_ref, semitones, steps, workdir)

    progress(0.92, desc="4/4 — Mixage…")
    final = mix(converted, instru, workdir / "chanson_finale.wav", gain_voix, gain_instru)

    shutil.copy(converted, workdir / "voix_convertie.wav")
    return (
        str(final), str(song), str(workdir / "voix_convertie.wav"), str(instru),
        f"Terminé. Tous les fichiers sont dans : {workdir}",
    )
