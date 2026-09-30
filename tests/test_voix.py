import gradio as gr
import numpy as np
import pytest
import soundfile as sf

from conftest import write_tone
from studiovoix import config as cfg
from studiovoix.voix import list_voices, save_voice


def test_enregistrement_normalise_et_tronque(env):
    src = write_tone(env / "long.wav", seconds=40, amp=0.2, sr=48000)
    msg, upd = save_voice(str(src), "  Laurent/2 ")
    assert list_voices() == ["Laurent2"]
    y, sr = sf.read(str(cfg.VOICES_DIR / "Laurent2.wav"))
    assert sr == 44100 and len(y) == 30 * 44100
    assert abs(np.max(np.abs(y)) - 0.95) < 1e-3
    assert "30 s" in msg and upd["value"] == "Laurent2"


def test_refus(env):
    with pytest.raises(gr.Error):
        save_voice(None, "x")
    src = write_tone(env / "court.wav", seconds=3)
    with pytest.raises(gr.Error):
        save_voice(str(src), "x")
    src = write_tone(env / "ok.wav", seconds=8)
    with pytest.raises(gr.Error):
        save_voice(str(src), "///")
