import gradio as gr
import numpy as np
import pytest
import soundfile as sf

from conftest import write_tone
from studiovoix import config as cfg
from studiovoix.voix import analyser, delete_voice, infos_voix, list_voices, rename_voice, save_voice

SR = 44100


def _sine(seconds, amp, sr=SR, aigus=0.1):
    """Note de 220 Hz avec un peu d'aigus (6 kHz, comme les « s » d'une vraie voix) ; aigus=0 : son étouffé."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (amp * (np.sin(2 * np.pi * 220 * t) + aigus * np.sin(2 * np.pi * 6000 * t))).astype("float32")


# --- Contrôle de qualité ----------------------------------------------------------
def test_analyse_voix_correcte():
    a = analyser(_sine(15, 0.3), SR)
    assert not a.erreurs and not a.avertissements and not a.infos
    assert -15 < a.niveau_db < -12  # sinus d'amplitude 0,3 ≈ -13,5 dBFS


def test_analyse_son_etouffe():
    """Message vocal WhatsApp (0,18 % d'énergie au-dessus de 4 kHz chez l'utilisateur) : avertissement."""
    a = analyser(_sine(15, 0.3, aigus=0), SR)
    assert not a.erreurs and "Son étouffé" in a.avertissements[0] and "WhatsApp" in a.avertissements[0]
    assert not analyser(_sine(15, 0.3), SR).avertissements
    assert not analyser(_sine(15, 0.3, sr=8000, aigus=0), 8000).avertissements  # rien à mesurer à 8 kHz


def test_analyse_duree():
    assert "trop court" in analyser(_sine(3, 0.3), SR).erreurs[0]
    a = analyser(_sine(7, 0.3), SR)
    assert not a.erreurs and "un peu court" in a.avertissements[0]
    assert "30 premières secondes" in analyser(_sine(45, 0.3), SR).infos[0]


def test_analyse_volume():
    assert "beaucoup trop faible" in analyser(_sine(15, 0.005), SR).erreurs[0]   # ≈ -49 dBFS
    a = analyser(_sine(15, 0.03), SR)                                           # ≈ -33 dBFS
    assert not a.erreurs and "Volume faible" in a.avertissements[0]
    assert "beaucoup trop faible" in analyser(np.zeros(15 * SR, "float32"), SR).erreurs[0]


def test_analyse_niveau_insensible_aux_pauses():
    """Une voix correcte entrecoupée de silences ne doit pas être jugée trop faible."""
    y = np.concatenate([_sine(3, 0.3), np.zeros(9 * SR, "float32")])
    a = analyser(y, SR)
    assert not a.erreurs and not a.avertissements and a.niveau_db > -15


def test_analyse_saturation():
    y = np.clip(_sine(15, 2.0), -1, 1)
    a = analyser(y, SR)
    assert not a.erreurs and any("saturé" in m for m in a.avertissements)
    assert not any("saturé" in m for m in analyser(_sine(15, 0.95, aigus=0), SR).avertissements)


# --- Import --------------------------------------------------------------------
def test_import_normalise_et_tronque(env):
    src = write_tone(env / "long.wav", seconds=40, amp=0.2, sr=48000)
    msg, upd = save_voice(str(src), "  Laurent/2 ")
    assert list_voices() == ["Laurent2"]
    y, sr = sf.read(str(cfg.VOICES_DIR / "Laurent2.wav"))
    assert sr == 44100 and len(y) == 30 * 44100
    assert abs(np.max(np.abs(y)) - 0.95) < 1e-3
    assert "30 s utilisées" in msg and "30 premières secondes" in msg and upd["value"] == "Laurent2"


@pytest.mark.parametrize("fmt", ["FLAC", "MP3"])
def test_import_flac_et_mp3(env, fmt):
    src = env / f"voix.{fmt.lower()}"
    sf.write(str(src), _sine(12, 0.3), SR, format=fmt)
    msg, _ = save_voice(str(src), "moi")
    assert msg.startswith("✅") and sf.info(str(cfg.VOICES_DIR / "moi.wav")).samplerate == 44100


def test_import_avertissements_sans_blocage(env):
    src = env / "sature.wav"
    sf.write(str(src), np.clip(_sine(8, 2.0), -1, 1), SR)
    msg, _ = save_voice(str(src), "sature")
    assert "saturé" in msg and "un peu court" in msg and list_voices() == ["sature"]


def test_import_refus(env):
    with pytest.raises(gr.Error):
        save_voice(None, "x")
    with pytest.raises(gr.Error, match="trop court"):
        save_voice(str(write_tone(env / "court.wav", seconds=3)), "x")
    with pytest.raises(gr.Error, match="trop faible"):
        save_voice(str(write_tone(env / "faible.wav", seconds=12, amp=0.003)), "x")
    with pytest.raises(gr.Error, match="nom"):
        save_voice(str(write_tone(env / "ok.wav", seconds=12)), "///")
    (env / "texte.mp3").write_text("pas de l'audio")
    with pytest.raises(gr.Error, match="wav, mp3 ou flac"):
        save_voice(str(env / "texte.mp3"), "x")
    assert list_voices() == []


def test_import_nom_deja_pris(env):
    src = write_tone(env / "ok.wav", seconds=12)
    save_voice(str(src), "Moi")
    with pytest.raises(gr.Error, match="déjà"):
        save_voice(str(src), "moi")  # insensible à la casse, comme Windows


# --- Écoute, renommage, suppression -----------------------------------------------------
def test_ecoute(env):
    save_voice(str(write_tone(env / "ok.wav", seconds=12)), "moi")
    path, desc = infos_voix("moi")
    assert path == str(cfg.VOICES_DIR / "moi.wav") and "12.0 s" in desc
    assert infos_voix(None) == (None, "") and infos_voix("inconnue") == (None, "")


def test_renommer(env):
    src = write_tone(env / "ok.wav", seconds=12)
    save_voice(str(src), "a")
    save_voice(str(src), "b")
    msg, upd = rename_voice("a", " c ")
    assert list_voices() == ["b", "c"] and upd["value"] == "c"
    with pytest.raises(gr.Error, match="déjà"):
        rename_voice("c", "B")
    rename_voice("c", "C")  # changer seulement la casse est permis
    assert list_voices() == ["C", "b"]
    with pytest.raises(gr.Error):
        rename_voice("C", "  ")
    with pytest.raises(gr.Error, match="introuvable"):
        rename_voice("../../secret", "x")


def test_supprimer(env):
    src = write_tone(env / "ok.wav", seconds=12)
    save_voice(str(src), "a")
    save_voice(str(src), "b")
    msg, upd = delete_voice("a")
    assert list_voices() == ["b"] and upd["value"] == "b" and "supprimée" in msg
    msg, upd = delete_voice(None)  # confirmation refusée dans le navigateur
    assert "annulée" in msg and list_voices() == ["b"]
    with pytest.raises(gr.Error, match="introuvable"):
        delete_voice("../config")
    delete_voice("b")
    assert list_voices() == []
