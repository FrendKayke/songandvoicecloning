"""Galerie : liste, détails, recréer avec la même graine, refaire un passage (repaint), supprimer."""
import json

import gradio as gr
import pytest
import soundfile as sf

from conftest import musique, no_progress, wav_octets, write_tone
from studiovoix import config as cfg
from studiovoix import galerie, jeu
from studiovoix.pipeline import MODE_INSTRU, creer_chanson

ARGS = ("", "", "", "", "", "Automatique")


@pytest.fixture
def gal(env, monkeypatch):
    monkeypatch.setattr(cfg, "GAMES_DIR", env / "data" / "jeux")
    monkeypatch.setattr(cfg, "TTS_DIR", env / "data" / "tts")
    return env


def _chanson(voix="moi", paroles="[Verse]\nla", mode=None, versions=1, graine=11):
    kw = {} if mode is None else {"mode": mode}
    return creer_chanson(voix, *ARGS, paroles, "Français", 60, 0, False, -3, 30, 1.0, 1.0, **kw,
                         description="pop, piano", versions=versions, graine=graine, progress=no_progress)


def test_liste_filtres_et_anciennes_creations(fake_acestep, fake_engines, gal):
    fake_acestep()
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=10)
    _chanson()
    jeu.generer_bande_son("p", jeu.EPOQUES[0][1], [], ["victoire"], [], 60, False, progress=no_progress)
    ancienne = cfg.SONGS_DIR / "20200101_120000"  # chanson d'avant creation.json
    ancienne.mkdir()
    (ancienne / "prompt.txt").write_text("rock, guitar\n\n[Verse]\nvieux\n", encoding="utf-8")
    write_tone(ancienne / "chanson_finale.wav", seconds=2)
    tts = cfg.TTS_DIR / "20200101_130000"
    tts.mkdir(parents=True)
    (tts / "tache.json").write_text(json.dumps({"texte": "Bonjour", "graine": 0}), encoding="utf-8")
    write_tone(tts / "parole.wav", seconds=2)

    tout = galerie.lister("Tout")
    assert len(tout) == 4 and tout[-1][1] == str(ancienne) and "🎵 Chanson · rock, guitar" in tout[-1][0]
    assert [c for _, c in galerie.lister("Bande-son de jeu")][0].startswith(str(cfg.GAMES_DIR))
    assert "🗣️ Lecture · Bonjour" in galerie.lister("Synthèse vocale")[0][0]
    md, fichier, versions = galerie.details(str(ancienne))
    assert "ancienne version" in md and fichier.endswith("chanson_finale.wav") and "rock, guitar" in md
    with pytest.raises(gr.Error, match="ancienne"):
        galerie.recreer(str(ancienne), progress=no_progress)


def test_details_deux_versions(fake_acestep, fake_engines, gal):
    fake_acestep()
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=10)
    _chanson(versions=2, graine=5)
    (_, dossier), = galerie.lister("Chansons")
    md, fichier, versions = galerie.details(dossier, 2)
    assert fichier.endswith("version_2/chanson_finale.wav") and versions["visible"] and versions["value"] == 2
    assert "Graine(s) : 5, " in md and "pop, piano" in md


def test_recreer_avec_la_meme_graine(fake_acestep, fake_engines, gal):
    srv = fake_acestep()
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=10)
    _chanson(versions=2, graine=5)
    (_, dossier), = galerie.lister("Chansons")
    graine_v2 = json.loads((galerie.Path(dossier) / "creation.json").read_text())["versions"][1]["graine"]
    msg, nouveau = galerie.recreer(dossier, 2, progress=no_progress)
    p = srv.payloads[-1]
    assert p["seed"] == str(graine_v2) and p["batch_size"] == 1 and p["prompt"] == "pop, piano"
    assert nouveau != dossier and galerie.lire(nouveau)["seedvc"] == {"demi_tons": -3, "etapes": 30}


def test_supprimer(fake_acestep, fake_engines, gal, tmp_path):
    fake_acestep()
    _chanson(mode=MODE_INSTRU)
    (_, dossier), = galerie.lister("Chansons")
    assert galerie.supprimer(None) == "Suppression annulée."
    assert "supprimée" in galerie.supprimer(dossier) and galerie.lister("Tout") == []
    etranger = tmp_path / "ailleurs"
    etranger.mkdir()
    (etranger / "creation.json").write_text('{"type": "chanson"}')
    with pytest.raises(gr.Error, match="pas une création"):
        galerie.supprimer(str(etranger))
    assert etranger.exists()
