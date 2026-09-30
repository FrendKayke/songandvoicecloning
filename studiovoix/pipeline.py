"""Pipeline complet : ACE-Step → Demucs → Seed-VC → mixage."""
import shutil
from datetime import datetime

import gradio as gr

from . import acestep, demucs, seedvc
from . import config as cfg
from .mixage import mix


def creer_chanson(
    voix, genre, style, instruments, ambiance, extra, voix_base,
    paroles, langue_label, duree, bpm, thinking,
    semitones, steps, gain_voix, gain_instru,
    progress=gr.Progress(),
):
    if not voix:
        raise gr.Error("Choisis (ou enregistre) d'abord une voix dans l'onglet « Ma voix ».")
    voice_ref = cfg.VOICES_DIR / f"{voix}.wav"
    if not voice_ref.exists():
        raise gr.Error(f"Voix introuvable : {voix}")
    if not (genre or style or instruments or ambiance or extra):
        raise gr.Error("Renseigne au moins un élément de style (genre, instruments…).")

    workdir = cfg.SONGS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir.mkdir(parents=True)

    prompt = acestep.build_prompt(genre, style, instruments, ambiance, voix_base, extra)
    lyrics = (paroles or "").strip() or "[Instrumental]"
    (workdir / "prompt.txt").write_text(f"{prompt}\n\n{lyrics}\n", encoding="utf-8")

    progress(0.05, desc="1/4 — Génération de la chanson (ACE-Step)…")
    song = acestep.acestep_generate(
        prompt, lyrics, cfg.LANGUES[langue_label], duree, bpm, thinking, workdir / "chanson_brute.wav", progress
    )

    if lyrics == "[Instrumental]":
        return str(song), str(song), None, None, f"Instrumental généré. Dossier : {workdir}"

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
