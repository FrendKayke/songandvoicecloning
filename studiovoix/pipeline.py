"""Pipeline : ACE-Step → Demucs → Seed-VC → mixage, ou ACE-Step seul (musique seule).

Si des instruments sont à retirer, Demucs sépare en 4 pistes et l'instrumental est remixé sans elles.
Chaque création est décrite dans creation.json (réglages, graine de chaque version) : la galerie et
« Refaire ce passage » s'en servent.
"""
import shutil

import gradio as gr

from . import acestep, demucs, rvc, seedvc
from . import config as cfg
from .mixage import mix, mixer
from .outils import ecrire_creation, nouveau_dossier

MODE_MA_VOIX = "Chanson avec ma voix"
MODE_VOIX_ACE = "Chanson avec la voix d'ACE-Step"
MODE_INSTRU = "Instrumental"
MODES = [MODE_MA_VOIX, MODE_VOIX_ACE, MODE_INSTRU]
# Instruments qu'on peut retirer (pistes de Demucs) : piste → libellé, et termes à éviter pour ACE-Step
RETRAITS = {"bass": "basse", "drums": "batterie"}
NEGATIFS = {"bass": "bass, bass guitar, sub-bass", "drums": "drums, drum kit, percussion"}
PISTES_INSTRU = ("drums", "bass", "other")
INSTRUMENTAL = "[Instrumental]"  # paroles reconnues par ACE-Step comme « sans voix » (server_utils.is_instrumental)
MAX_VERSIONS = 2  # au-delà, la mémoire graphique (12 Go) risque de manquer
SEEDVC = "seedvc"  # conversion sans entraînement ; « rvc:<nom> » = modèle RVC entraîné


def convertisseur(moteur, voix, semitones, steps):
    """(libellé, fonction(voix chantée, dossier) → voix convertie) selon le moteur de conversion choisi."""
    moteur = moteur or SEEDVC
    if moteur.startswith("rvc:"):
        nom = moteur[4:]
        rvc.modele(nom)  # erreur claire tout de suite si le modèle n'existe plus
        return (f"ton modèle RVC « {nom} »",
                lambda vocals, workdir: rvc.convertir(vocals, nom, semitones, workdir / "voix_convertie_rvc.wav"))
    if not voix:
        raise gr.Error("Choisis (ou enregistre) d'abord une voix dans l'onglet « Bibliothèque de voix ».")
    voice_ref = cfg.VOICES_DIR / f"{voix}.wav"
    if not voice_ref.exists():
        raise gr.Error(f"Voix introuvable : {voix} (supprimée ou renommée ?)")
    return "ta voix (Seed-VC)", lambda vocals, workdir: seedvc.convert_voice(vocals, voice_ref, semitones, steps, workdir)


def creer_chanson(
    voix, genre, style, instruments, ambiance, extra, voix_base,
    paroles, langue_label, duree, bpm, thinking,
    semitones, steps, gain_voix, gain_instru,
    mode=MODE_MA_VOIX,
    description=None,
    retirer=None,
    versions=1,
    graine=0,
    moteur=SEEDVC,
    progress=gr.Progress(),
):
    """« description » : description finale (modifiable dans l'interface) ; vide = construite depuis les listes.
    « retirer » : pistes à supprimer du mix (« bass », « drums »), garanti par une séparation Demucs en 4 pistes.
    « versions » : 1 ou 2 versions générées d'un coup ; « graine » : 0 = aléatoire, sinon graine de la version 1.
    « moteur » : conversion de voix, SEEDVC (échantillon « voix » de la bibliothèque) ou « rvc:<nom> ».
    Renvoie (finale, brute, voix convertie, instrumental, message, finale de la version 2 ou None).
    """
    retirer = [p for p in (retirer or []) if p in RETRAITS]
    versions = max(1, min(MAX_VERSIONS, int(versions or 1)))
    mode = mode or MODE_MA_VOIX
    if mode not in MODES:
        raise gr.Error(f"Mode inconnu : {mode}")
    conv = convertisseur(moteur, voix, semitones, steps) if mode == MODE_MA_VOIX else None

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
    # Une version : fichiers à la racine du dossier (comme avant) ; plusieurs : un sous-dossier par version
    dossiers = [workdir] if versions == 1 else [workdir / f"version_{i}" for i in range(1, versions + 1)]
    for d in dossiers:
        d.mkdir(exist_ok=True)

    avec_voix = lyrics != INSTRUMENTAL
    conversion = avec_voix and mode == MODE_MA_VOIX
    separation = conversion or bool(retirer)
    total = 1 + separation + conversion + (avec_voix and separation)

    progress(0.05, desc=f"1/{total} — Génération (ACE-Step)…")
    negatif = ", ".join(NEGATIFS[p] for p in retirer) or None
    params = acestep.text2music_params(prompt, lyrics, cfg.LANGUES[langue_label], duree, bpm, thinking, negatif)
    generes = acestep.generer(params, [d / "chanson_brute.wav" for d in dossiers], progress, f"1/{total}",
                              graine=graine)

    resultats = []
    for n, ((song, seed), dossier) in enumerate(zip(generes, dossiers), 1):
        prefixe = "" if versions == 1 else f"Version {n}/{versions} — "
        resultats.append(_finaliser(song, dossier, prefixe, total, avec_voix, conversion, separation, retirer,
                                    conv, gain_voix, gain_instru, progress))

    ecrire_creation(workdir, {
        "type": "chanson", "mode": mode, "description": prompt, "paroles": lyrics, "langue": langue_label,
        "duree": float(duree), "bpm": int(bpm or 0), "reflexion": bool(thinking), "retirer": retirer,
        "voix": voix if mode == MODE_MA_VOIX else None,
        "conversion": (moteur or SEEDVC) if mode == MODE_MA_VOIX else None,
        "seedvc": {"demi_tons": int(semitones), "etapes": int(steps)},
        "gains": {"voix": float(gain_voix), "instrumental": float(gain_instru)},
        "versions": [{"graine": seed, "dossier": d.name if d != workdir else ".", "fichier": str(r[0])}
                     for (_, seed), d, r in zip(generes, dossiers, resultats)],
    })
    final, brute, conv, instru, msg = resultats[0]
    if versions > 1:
        msg = f"{versions} versions générées (écoute-les ci-dessous). Dossier : {workdir}"
    msg_graines = ", ".join(f"{seed}" for _, seed in generes)
    msg = f"{msg} Graine{'s' if versions > 1 else ''} : {msg_graines}."
    return final, brute, conv, instru, msg, (resultats[1][0] if versions > 1 else None)


def finaliser_depuis_infos(song, workdir, infos, progress):
    """Refait séparation / conversion / mixage d'une création à partir de son creation.json
    (utilisé après « Refaire un passage » dans la galerie)."""
    retirer = [p for p in infos.get("retirer") or [] if p in RETRAITS]
    mode = infos.get("mode") or MODE_MA_VOIX
    avec_voix = (infos.get("paroles") or INSTRUMENTAL) != INSTRUMENTAL
    conversion = avec_voix and mode == MODE_MA_VOIX
    separation = conversion or bool(retirer)
    seedvc_infos, gains = infos.get("seedvc") or {}, infos.get("gains") or {}
    conv = None
    if conversion:
        conv = convertisseur(infos.get("conversion") or SEEDVC, infos.get("voix"),
                             seedvc_infos.get("demi_tons", 0), seedvc_infos.get("etapes", 40))
    total = 1 + separation + conversion + (avec_voix and separation)
    return _finaliser(song, workdir, "", total, avec_voix, conversion, separation, retirer, conv,
                      gains.get("voix", 1.0), gains.get("instrumental", 1.0), progress)


def _finaliser(song, workdir, prefixe, total, avec_voix, conversion, separation, retirer,
               conv, gain_voix, gain_instru, progress):
    """Tout ce qui suit la génération d'une version : séparation, conversion, mixage."""
    numeros = iter(range(2, total + 1))

    def etape():
        return f"{prefixe}{next(numeros)}/{total}"

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
        libelle, convertir = conv
        progress(0.6, desc=f"{etape()} — Remplacement par {libelle}…")
        voix_finale = convertir(vocals, workdir)

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
