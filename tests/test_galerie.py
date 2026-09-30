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
    md, fichier, versions, desc, paroles, fin = galerie.details(str(ancienne))
    assert "ancienne version" in md and fichier.endswith("chanson_finale.wav") and desc == "rock, guitar"
    with pytest.raises(gr.Error, match="ancienne"):
        galerie.recreer(str(ancienne), progress=no_progress)


def test_details_deux_versions(fake_acestep, fake_engines, gal):
    fake_acestep()
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=10)
    _chanson(versions=2, graine=5)
    (_, dossier), = galerie.lister("Chansons")
    md, fichier, versions, desc, paroles, fin = galerie.details(dossier, 2)
    assert fichier.endswith("version_2/chanson_finale.wav") and versions["visible"] and versions["value"] == 2
    assert "Graine(s) : 5, " in md and desc == "pop, piano" and paroles == "[Verse]\nla" and fin == 4.0


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


def test_refaire_un_passage_de_chanson(fake_acestep, fake_engines, gal):
    srv = fake_acestep()
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=10)
    _chanson()
    (_, dossier), = galerie.lister("Chansons")
    brute = (galerie.Path(dossier) / "chanson_brute.wav").read_bytes()
    msg, nouveau = galerie.refaire_passage(dossier, 1, 1.0, 3.0, "", "[Verse]\nnouveau", list(galerie.FORCES)[2],
                                           progress=no_progress)
    p, f = srv.payloads[-1], srv.fichiers[-1]
    assert p["task_type"] == "repaint" and p["repainting_start"] == "1.0" and p["repainting_end"] == "3.0"
    assert p["repaint_mode"] == "aggressive" and p["lyrics"] == "[Verse]\nnouveau" and f["src_audio"][1] == brute
    infos = galerie.lire(nouveau)
    assert infos["retouche_de"] == dossier and infos["passage"] == [1.0, 3.0] and infos["voix"] == "moi"
    assert (galerie.Path(nouveau) / "chanson_finale.wav").exists()  # voix convertie et remixée à nouveau
    appel = json.loads((cfg.SEEDVC_DIR / "appel.json").read_text())
    assert appel[appel.index("--target") + 1] == str(cfg.VOICES_DIR / "moi.wav")
    assert appel[appel.index("--semi-tone-shift") + 1] == "-3"  # réglages Seed-VC d'origine repris


def test_refaire_un_passage_de_piste_de_jeu(fake_acestep, gal):
    srv = fake_acestep()
    y, _, _ = musique(bpm=120)
    srv.version = lambda i: wav_octets(y)
    jeu.generer_bande_son("p", jeu.EPOQUES[0][1], [], ["combat"], [], 60, False, progress=no_progress)
    (_, dossier), = galerie.lister("Bande-son de jeu")
    msg, nouveau = galerie.refaire_passage(dossier, 1, 10, 14, "", "", list(galerie.FORCES)[1], progress=no_progress)
    assert srv.payloads[-1]["task_type"] == "repaint" and srv.payloads[-1]["lyrics"] == "[Instrumental]"
    infos = galerie.lire(nouveau)
    assert infos["boucle_points"] and infos["retouche_de"] == dossier and "boucle de" in msg


def test_passage_invalide_et_lecture(fake_acestep, fake_engines, gal):
    fake_acestep()
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=10)
    _chanson(mode=MODE_INSTRU)
    (_, dossier), = galerie.lister("Chansons")
    for debut, fin in ((3, 2), (0, 0.5), (0, 60)):
        with pytest.raises(gr.Error, match="passage"):
            galerie.refaire_passage(dossier, 1, debut, fin, "", "", "", progress=no_progress)
    tts = cfg.TTS_DIR / "20200101_130000"
    tts.mkdir(parents=True)
    galerie.ecrire_creation(tts, {"type": "tts", "texte": "x", "versions": [{"graine": 0, "fichier": "x"}]})
    with pytest.raises(gr.Error, match="chansons et les pistes de jeu"):
        galerie.refaire_passage(str(tts), 1, 0, 2, "", "", "", progress=no_progress)


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
