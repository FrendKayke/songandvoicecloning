"""Pipeline complet avec faux moteurs : vérifie les appels et les fichiers produits."""
import json

import gradio as gr
import pytest
import soundfile as sf

from conftest import no_progress, write_tone
from studiovoix import config as cfg
from studiovoix.pipeline import MODE_INSTRU, MODE_VOIX_ACE, creer_chanson

ARGS = dict(genre="pop", style="", instruments="piano", ambiance="", extra="", voix_base="Automatique",
            langue_label="Français", duree=60, bpm=0, thinking=False, semitones=-12, steps=40,
            gain_voix=1.0, gain_instru=1.0)


def _call(voix, paroles, mode=None, **kw):
    # Paramètres nommés ajoutés après « mode » : transmis seulement s'ils sont donnés (sinon valeur par défaut)
    extra = {k: kw.pop(k) for k in ("description", "retirer") if k in kw}
    if mode is not None:
        extra["mode"] = mode
    a = dict(ARGS, **kw)
    return creer_chanson(voix, a["genre"], a["style"], a["instruments"], a["ambiance"], a["extra"],
                         a["voix_base"], paroles, a["langue_label"], a["duree"], a["bpm"], a["thinking"],
                         a["semitones"], a["steps"], a["gain_voix"], a["gain_instru"], **extra,
                         progress=no_progress)


def test_chanson_avec_ma_voix(fake_acestep, fake_engines):
    srv = fake_acestep()
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=10)
    final, brute, conv, instru, msg = _call("moi", "[Verse]\nBonjour")
    workdir = cfg.SONGS_DIR / next(cfg.SONGS_DIR.iterdir()).name
    assert final == str(workdir / "chanson_finale.wav") and sf.info(final).channels == 2
    assert brute == str(workdir / "chanson_brute.wav")
    assert conv == str(workdir / "voix_convertie.wav")
    assert instru.endswith("no_vocals.wav")
    assert "Terminé" in msg
    assert (workdir / "prompt.txt").read_text(encoding="utf-8") == "pop, piano\n\n[Verse]\nBonjour\n"
    appel = json.loads((cfg.SEEDVC_DIR / "appel.json").read_text())
    assert appel[appel.index("--semi-tone-shift") + 1] == "-12"
    assert appel[appel.index("--f0-condition") + 1] == "True"
    assert appel[appel.index("--target") + 1] == str(cfg.VOICES_DIR / "moi.wav")
    assert srv.payloads[0]["lyrics"] == "[Verse]\nBonjour"


def test_paroles_vides_donnent_un_instrumental(fake_acestep, fake_engines):
    srv = fake_acestep()
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=10)
    final, brute, conv, instru, msg = _call("moi", "  ")
    assert final == brute and conv is None and instru is None
    assert srv.payloads[0]["lyrics"] == "[Instrumental]"
    assert not (cfg.SEEDVC_DIR / "appel.json").exists()


def test_erreurs_de_saisie(env):
    with pytest.raises(gr.Error):
        _call(None, "x")
    with pytest.raises(gr.Error):
        _call("inconnue", "x")
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=10)
    with pytest.raises(gr.Error):
        _call("moi", "x", genre="", instruments="")


def test_seedvc_absent(fake_acestep, env):
    fake_acestep()
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=10)
    with pytest.raises(gr.Error, match="INSTALLER.bat"):
        _call("moi", "[Verse]\nx")


def test_mode_voix_ace_step_sans_conversion(fake_acestep, fake_engines):
    srv = fake_acestep()
    final, brute, conv, instru, msg = _call(None, "[Verse]\nBonjour", mode=MODE_VOIX_ACE,
                                            voix_base="Voix féminine")
    assert final == brute and brute.endswith("chanson_brute.wav") and conv is None and instru is None
    assert "voix d'ACE-Step" in msg
    assert srv.payloads[0]["lyrics"] == "[Verse]\nBonjour"
    assert srv.payloads[0]["prompt"] == "pop, piano, female vocals"
    assert not (cfg.SEEDVC_DIR / "appel.json").exists()  # ni Demucs ni Seed-VC
    assert not any(cfg.SONGS_DIR.rglob("demucs"))


def test_mode_voix_ace_step_exige_des_paroles(fake_acestep, env):
    fake_acestep()
    with pytest.raises(gr.Error, match="Instrumental"):
        _call(None, " ", mode=MODE_VOIX_ACE)


def test_mode_instrumental_ignore_paroles_et_voix(fake_acestep, fake_engines):
    srv = fake_acestep()
    final, brute, conv, instru, msg = _call(None, "[Verse]\nignoré", mode=MODE_INSTRU,
                                            voix_base="Voix masculine")
    assert final == brute and conv is None and instru is None and "Instrumental" in msg
    assert srv.payloads[0]["lyrics"] == "[Instrumental]"
    assert srv.payloads[0]["prompt"] == "pop, piano"  # pas de « male vocals »
    workdir = next(cfg.SONGS_DIR.iterdir())
    assert (workdir / "prompt.txt").read_text(encoding="utf-8") == "pop, piano\n\n[Instrumental]\n"


def test_mode_inconnu(env):
    with pytest.raises(gr.Error):
        _call(None, "x", mode="Karaoké")


def test_description_modifiee_prioritaire(fake_acestep, fake_engines):
    srv = fake_acestep()
    _call(None, "", mode=MODE_INSTRU, description="  8-bit chiptune, no drums fill  ")
    assert srv.payloads[0]["prompt"] == "8-bit chiptune, no drums fill"
    workdir = next(cfg.SONGS_DIR.iterdir())
    assert (workdir / "prompt.txt").read_text(encoding="utf-8").startswith("8-bit chiptune, no drums fill\n")


def test_listes_de_styles(fake_acestep, fake_engines):
    srv = fake_acestep()
    _call(None, "", mode=MODE_INSTRU, genre=["synthwave, retrowave"], instruments=["synth pads", "drum machine"],
          description="")
    assert srv.payloads[0]["prompt"] == "synthwave, retrowave, synth pads, drum machine"


# --- Retrait d'instruments (Demucs 4 pistes) -------------------------------------------
from conftest import FREQ_PISTES, energie  # noqa: E402


def _presentes(path):
    return {p for p, f in FREQ_PISTES.items() if energie(path, f) > 0.01}


def test_instrumental_sans_basse(fake_acestep, fake_engines):
    srv = fake_acestep()
    final, brute, conv, instru, msg = _call(None, "", mode=MODE_INSTRU, retirer=["bass"])
    assert final.endswith("instrumental.wav") and brute.endswith("chanson_brute.wav")
    assert _presentes(final) == {"drums", "other"}  # ni basse, ni résidus de voix
    assert "sans basse" in msg
    assert srv.payloads[0]["lm_negative_prompt"] == "bass, bass guitar, sub-bass"


def test_voix_ace_step_sans_batterie_ni_basse(fake_acestep, fake_engines):
    srv = fake_acestep()
    final, brute, conv, instru, msg = _call(None, "[Verse]\nla", mode=MODE_VOIX_ACE, retirer=["drums", "bass"])
    assert final.endswith("chanson_finale.wav") and conv is None
    assert _presentes(final) == {"other", "vocals"} and _presentes(instru) == {"other"}
    assert "sans batterie et basse" in msg
    assert srv.payloads[0]["lm_negative_prompt"] == "drums, drum kit, percussion, bass, bass guitar, sub-bass"


def test_ma_voix_sans_basse(fake_acestep, fake_engines):
    fake_acestep()
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=10)
    final, brute, conv, instru, msg = _call("moi", "[Verse]\nla", retirer=["bass"])
    appel = json.loads((cfg.SEEDVC_DIR / "appel.json").read_text())
    assert appel[appel.index("--source") + 1].endswith("vocals.wav") and "demucs4" in appel[appel.index("--source") + 1]
    assert _presentes(final) == {"drums", "other", "vocals"}  # la « voix convertie » du faux Seed-VC = piste voix
    assert _presentes(instru) == {"drums", "other"} and msg.startswith("Terminé (sans basse)")


def test_sans_retrait_pas_de_prompt_negatif(fake_acestep, fake_engines):
    srv = fake_acestep()
    _call(None, "", mode=MODE_INSTRU, retirer=["piano"])  # valeur inconnue ignorée
    assert "lm_negative_prompt" not in srv.payloads[0]
    assert not any(cfg.SONGS_DIR.rglob("demucs4"))
