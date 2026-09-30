"""Pipeline complet avec faux moteurs : vérifie les appels et les fichiers produits."""
import json

import gradio as gr
import pytest
import soundfile as sf

from conftest import no_progress, write_tone
from studiovoix import config as cfg
from studiovoix.pipeline import creer_chanson

ARGS = dict(genre="pop", style="", instruments="piano", ambiance="", extra="", voix_base="Automatique",
            langue_label="Français", duree=60, bpm=0, thinking=False, semitones=-12, steps=40,
            gain_voix=1.0, gain_instru=1.0)


def _call(voix, paroles, **kw):
    a = dict(ARGS, **kw)
    return creer_chanson(voix, a["genre"], a["style"], a["instruments"], a["ambiance"], a["extra"],
                         a["voix_base"], paroles, a["langue_label"], a["duree"], a["bpm"], a["thinking"],
                         a["semitones"], a["steps"], a["gain_voix"], a["gain_instru"], progress=no_progress)


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
