import gradio as gr
import pytest

from conftest import no_progress
from studiovoix import acestep


def test_build_prompt():
    assert acestep.build_prompt("pop", " 80s ", "", "triste", "Voix féminine", "") == "pop, 80s, triste, female vocals"
    assert acestep.build_prompt("rock", "", "", "", "Voix masculine", "x") == "rock, male vocals, x"
    assert acestep.build_prompt("rock", "", "", "", "Automatique", None) == "rock"


def test_generation_nominale(fake_acestep, env):
    srv = fake_acestep(pending_polls=2)
    dest = acestep.acestep_generate("pop", "[Verse]\nla", "fr", 60, 120, True, env / "s.wav", no_progress)
    assert dest.read_bytes() == srv.wav_bytes
    p = srv.payloads[0]
    assert p["prompt"] == "pop" and p["vocal_language"] == "fr" and p["bpm"] == 120
    assert p["audio_duration"] == 60.0 and p["thinking"] is True and p["audio_format"] == "wav"


def test_bpm_auto_non_envoye(fake_acestep, env):
    srv = fake_acestep()
    acestep.acestep_generate("pop", "x", "fr", 60, 0, False, env / "s.wav", no_progress)
    assert "bpm" not in srv.payloads[0]


def test_serveur_occupe_tolere(fake_acestep, env):
    """Des réponses en erreur pendant le chargement ne doivent pas faire échouer la génération."""
    srv = fake_acestep(broken_polls=3)
    acestep.acestep_generate("pop", "x", "fr", 60, 0, False, env / "s.wav", no_progress)
    assert srv.polls >= 4


def test_nouvel_essai_sans_reflexion(fake_acestep, env):
    srv = fake_acestep(fail_with_thinking=True)
    acestep.acestep_generate("pop", "x", "fr", 60, 0, True, env / "s.wav", no_progress)
    assert [p["thinking"] for p in srv.payloads] == [True, False]


def test_serveur_injoignable(env, monkeypatch):
    monkeypatch.setattr(acestep.cfg, "ACESTEP_URL", "http://127.0.0.1:9")
    with pytest.raises(gr.Error):
        acestep.wait_acestep(no_progress, timeout=0.5)
