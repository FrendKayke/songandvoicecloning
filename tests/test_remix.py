"""Remixer une musique dans un autre style : tâche « cover » d'ACE-Step (faux serveur)."""
import gradio as gr
import pytest
import soundfile as sf

from conftest import no_progress, write_tone
from studiovoix import config as cfg
from studiovoix import galerie, remix


def test_description_et_reglages():
    d = remix.description([remix.STYLES[0][1]], ["Violoncelle", "piano"], [], "")
    assert d.startswith("epic symphonic orchestra arrangement") and d.endswith("instrumental") and "piano" in d
    assert remix.description([]) == "instrumental"
    structure, melodie = remix.reglages("Forte (réinterprétation libre)")
    assert (structure["value"], melodie["value"]) == (0.3, 0.1)


def test_remixer_plusieurs_morceaux(env, fake_acestep):
    srv = fake_acestep()
    theme = write_tone(env / "theme.wav", seconds=8.0, channels=1)
    court = write_tone(env / "jingle.wav", seconds=2.0, channels=2)
    autre = write_tone(env / "boss.wav", seconds=6.0, channels=2)
    msg, choix, premier, original = remix.remixer(
        [str(theme), str(court), str(autre)], [remix.STYLES[0][1]], [], [], "", "", 0.5, 0.2, False, 2, 42,
        progress=no_progress)
    # deux morceaux remixés (2 versions chacun), le trop court signalé sans arrêter les autres
    assert len(srv.payloads) == 2 and "4 remix" in msg and "jingle : " in msg and "trop court" in msg
    p = srv.payloads[0]
    assert p["task_type"] == "cover" and p["lyrics"] == "[Instrumental]" and p["thinking"] == "false"
    assert float(p["audio_cover_strength"]) == 0.5 and float(p["cover_noise_strength"]) == 0.2
    assert float(p["audio_duration"]) == 8.0 and p["batch_size"] == "2" and p["seed"].startswith("42,")
    nom, octets = srv.fichiers[0]["src_audio"]
    assert nom == "original.wav" and len(octets) > 1000
    info = sf.info(original)
    assert (info.samplerate, info.channels) == (44100, 2)  # mono → stéréo 44,1 kHz avant l'envoi
    assert [lib for lib, _ in choix["choices"]][0].startswith("theme — version 1 (graine 42)")
    assert remix.original_de(premier) == original
    # galerie : listé, recréé avec la même graine
    (lib, chemin), *_ = galerie.lister("Remix")
    assert "🎛️ Remix" in lib and "boss" in lib
    msg, nouveau = galerie.recreer(chemin, 2, progress=no_progress)
    assert srv.payloads[-1]["seed"].split(",")[0] == str(galerie.lire(chemin)["versions"][1]["graine"])
    assert galerie.lire(nouveau)["type"] == "remix" and "Remix refait" in msg
    # mode très fidèle
    remix.remixer(str(theme), [], [], [], "", "jazz trio", 0.7, 0.3, True, 1, 0, progress=no_progress)
    assert srv.payloads[-1]["task_type"] == "cover-nofsq" and srv.payloads[-1]["prompt"] == "jazz trio"


def test_remixer_erreurs(env, fake_acestep):
    fake_acestep()
    with pytest.raises(gr.Error, match="Ajoute la musique"):
        remix.remixer(None, [remix.STYLES[0][1]], [], [], "", "", 0.5, 0.2, False, 1, 0, progress=no_progress)
    theme = write_tone(env / "theme.wav", seconds=8.0)
    with pytest.raises(gr.Error, match="nouveau style"):
        remix.remixer(str(theme), [], [], [], "", "", 0.5, 0.2, False, 1, 0, progress=no_progress)
    assert not any(cfg.REMIX_DIR.glob("*/creation.json"))
