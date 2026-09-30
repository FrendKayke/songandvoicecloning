"""Bande-son de jeu : situations, génération en lot (faux serveur ACE-Step), jingles."""
import json

import gradio as gr
import numpy as np
import pytest
import soundfile as sf

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import jeu
from studiovoix.mixage import couper_jingle

SR = 44100
EPOQUE = jeu.EPOQUES[0][1]


@pytest.fixture
def jeux(env, monkeypatch):
    monkeypatch.setattr(cfg, "GAMES_DIR", env / "data" / "jeux")
    return env


def test_couper_jingle(tmp_path):
    """Silence au début, notes jusqu'à 6,8 s, creux, puis encore des notes : coupe dans le creux."""
    t = np.arange(10 * SR) / SR
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    y[(t > 6.8) & (t < 7.3)] = 0.0
    y = np.concatenate([np.zeros(SR // 2), y]).astype("float32")
    sf.write(str(tmp_path / "src.wav"), y, SR)
    duree = couper_jingle(tmp_path / "src.wav", tmp_path / "jingle.wav", 7)
    z, sr = sf.read(str(tmp_path / "jingle.wav"), always_2d=True)
    assert 6.8 <= duree <= 7.3 and abs(len(z) / sr - duree) < 0.01
    assert np.abs(z[:100]).max() > 0.01  # le silence de début est retiré
    assert np.abs(z[-50:]).max() < 0.05  # fondu de sortie


def test_situation_libre():
    ident, libelle, txt, duree, boucle = jeu.situation("Fire faction theme, taiko drums!")
    assert ident == "perso_fire_faction_theme_taiko_drums" and txt == "Fire faction theme, taiko drums!"
    assert boucle and duree == 90
    assert jeu.situation("victoire")[4] is False


def test_apercu():
    md = jeu.apercu(EPOQUE, ["crystals and magic, mystical"], ["combat", "booster"], [], 60)
    assert "| Combat | boucle | 60 s |" in md and "| Ouverture d'un booster | jingle | 3 s |" in md
    assert "instrumental, no vocals" in md and "crystals and magic" in md
    assert "Choisis" in jeu.apercu(EPOQUE, [], [], [], 60)


def test_generation_en_lot(fake_acestep, jeux):
    srv = fake_acestep()
    msg, liste, premiere = jeu.generer_bande_son(
        "Mon jeu!", EPOQUE, ["dark fantasy"], ["titre", "victoire", "fire faction theme"], ["electric guitar"],
        60, False, 0, progress=no_progress)
    racine = cfg.GAMES_DIR / "Mon jeu"
    assert sorted(p.name for p in racine.iterdir()) == ["perso_fire_faction_theme", "titre", "victoire"]
    assert [p["lyrics"] for p in srv.payloads] == ["[Instrumental]"] * 3
    assert [p["audio_duration"] for p in srv.payloads] == [60.0, 10.0, 60.0]  # jingle : 10 s minimum d'ACE-Step
    assert srv.payloads[0]["prompt"] == (f"{EPOQUE}, {jeu.SITUATIONS['titre'][1]}, dark fantasy, electric guitar, "
                                         "instrumental, no vocals")
    victoire = next((racine / "victoire").iterdir())
    creation = json.loads((victoire / "creation.json").read_text(encoding="utf-8"))
    assert creation["type"] == "jeu" and creation["boucle"] is False and creation["projet"] == "Mon jeu"
    assert sf.info(str(victoire / "piste.wav")).duration < 4.1  # le faux serveur rend 4 s : coupé ≤ 4 s
    assert len(liste["choices"]) == 3 and premiere.endswith("piste.wav") and "3 piste(s)" in msg


def test_erreurs(jeux):
    with pytest.raises(gr.Error, match="nom"):
        jeu.generer_bande_son("  ", EPOQUE, [], ["titre"], [], 60, False, progress=no_progress)
    with pytest.raises(gr.Error, match="situation"):
        jeu.generer_bande_son("x", EPOQUE, [], [], [], 60, False, progress=no_progress)
    with pytest.raises(gr.Error, match="époque"):
        jeu.generer_bande_son("x", "", [], ["titre"], [], 60, False, progress=no_progress)
