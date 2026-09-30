"""Pipeline : ACE-Step → Demucs → Seed-VC → mixage, ou ACE-Step seul (musique seule).

Si des instruments sont à retirer, Demucs sépare en 4 pistes et l'instrumental est remixé sans elles.
"""
import shutil

import gradio as gr

from . import acestep, demucs, seedvc
from . import config as cfg
from .mixage import mix, mixer
from .outils import nouveau_dossier

MODE_MA_VOIX = "Chanson avec ma voix"
MODE_VOIX_ACE = "Chanson avec la voix d'ACE-Step"
MODE_INSTRU = "Instrumental"
MODES = [MODE_MA_VOIX, MODE_VOIX_ACE, MODE_INSTRU]
# Instruments qu'on peut retirer (pistes de Demucs) : piste → libellé, et termes à éviter pour ACE-Step
RETRAITS = {"bass": "basse", "drums": "batterie"}
NEGATIFS = {"bass": "bass, bass guitar, sub-bass", "drums": "drums, drum kit, percussion"}
PISTES_INSTRU = ("drums", "bass", "other")
INSTRUMENTAL = "[Instrumental]"  # paroles reconnues par ACE-Step comme « sans voix » (server_utils.is_instrumental)


def creer_chanson(
    voix, genre, style, instruments, ambiance, extra, voix_base,
    paroles, langue_label, duree, bpm, thinking,
    semitones, steps, gain_voix, gain_instru,
    mode=MODE_MA_VOIX,
    description=None,
    retirer=None,
    progress=gr.Progress(),
):
    """« description » : description finale (modifiable dans l'interface) ; vide = construite depuis les listes.
    « retirer » : pistes à supprimer du mix (« bass », « drums »), garanti par une séparation Demucs en 4 pistes.
    """
    retirer = [p for p in (retirer or []) if p in RETRAITS]
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

    if mode == MODE_INSTRU:
        lyrics = INSTRUMENTAL
        voix_base = "Automatique"  # pas de voix : aucune consigne de voix dans le prompt
    else:
        lyrics = (paroles or "").strip()
        if not lyrics:
            if mode == MODE_VOIX_ACE:
                raise gr.Error("Écris des paroles, ou choisis le mode « Instrumental ».")
            lyrics = INSTRUMENTAL

    prompt = (description or "").strip() or acestep.build_prompt(
        genre, style, instruments, ambiance, voix_base, extra)
    if not prompt:
        raise gr.Error("Renseigne au moins un élément de style (genre, instruments…).")

    workdir = nouveau_dossier(cfg.SONGS_DIR)
    (workdir / "prompt.txt").write_text(f"{prompt}\n\n{lyrics}\n", encoding="utf-8")

    avec_voix = lyrics != INSTRUMENTAL
    conversion = avec_voix and mode == MODE_MA_VOIX
    separation = conversion or bool(retirer)
    total = 1 + separation + conversion + (avec_voix and separation)
    numeros = iter(range(1, total + 1))

    def etape():
        return f"{next(numeros)}/{total}"

    e = etape()
    progress(0.05, desc=f"{e} — Génération de la chanson (ACE-Step)…")
    negatif = ", ".join(NEGATIFS[p] for p in retirer) or None
    song = acestep.acestep_generate(
        prompt, lyrics, cfg.LANGUES[langue_label], duree, bpm, thinking, workdir / "chanson_brute.wav",
        progress, e, negatif,
    )

    if not separation:
        if not avec_voix:
            return str(song), str(song), None, None, f"Instrumental généré. Dossier : {workdir}"
        return (
            str(song), str(song), None, None,
            f"Chanson générée avec la voix d'ACE-Step (sans conversion). Dossier : {workdir}",
        )

    sans = " et ".join(RETRAITS[p] for p in retirer)
    if retirer:
        progress(0.45, desc=f"{etape()} — Séparation en pistes et retrait : {sans} (Demucs)…")
        pistes = demucs.separate_stems(song, workdir)
        vocals = pistes["vocals"]
        instru = mixer([(pistes[p], 1.0) for p in PISTES_INSTRU if p not in retirer], workdir / "instrumental.wav")
        sans = f" (sans {sans})"
    else:
        progress(0.45, desc=f"{etape()} — Séparation voix / instrumental (Demucs)…")
        vocals, instru = demucs.separate_vocals(song, workdir)

    if not avec_voix:  # la piste voix (résidus éventuels) est écartée
        return str(instru), str(song), None, None, f"Instrumental généré{sans}. Dossier : {workdir}"

    voix_finale = vocals
    if conversion:
        progress(0.6, desc=f"{etape()} — Remplacement par ta voix (Seed-VC)…")
        voix_finale = seedvc.convert_voice(vocals, voice_ref, semitones, steps, workdir)

    progress(0.92, desc=f"{etape()} — Mixage…")
    final = mix(voix_finale, instru, workdir / "chanson_finale.wav", gain_voix, gain_instru)

    if not conversion:
        return (
            str(final), str(song), None, str(instru),
            f"Chanson générée avec la voix d'ACE-Step{sans}. Dossier : {workdir}",
        )
    shutil.copy(voix_finale, workdir / "voix_convertie.wav")
    return (
        str(final), str(song), str(workdir / "voix_convertie.wav"), str(instru),
        f"Terminé{sans}. Tous les fichiers sont dans : {workdir}",
    )
