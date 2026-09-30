import numpy as np
import soundfile as sf

from conftest import write_tone
from studiovoix.mixage import load_stereo, mix


def test_load_stereo_mono_et_reechantillonnage(tmp_path):
    p = write_tone(tmp_path / "a.wav", seconds=1.0, sr=22050)
    y = load_stereo(p)
    assert y.shape[0] == 2
    assert abs(y.shape[1] - 44100) <= 1


def test_mix_longueur_stereo_et_limitation(tmp_path):
    v = write_tone(tmp_path / "v.wav", seconds=1.0, amp=0.9)
    i = write_tone(tmp_path / "i.wav", seconds=2.0, amp=0.9, channels=2, sr=48000)
    out = mix(v, i, tmp_path / "m.wav", 1.0, 1.0)
    y, sr = sf.read(str(out), always_2d=True)
    assert sr == 44100 and y.shape[1] == 2
    assert abs(y.shape[0] - 2 * 44100) <= 2
    assert np.max(np.abs(y)) <= 0.95 + 1e-3


def test_mix_gains(tmp_path):
    v = write_tone(tmp_path / "v.wav", seconds=1.0, amp=0.2)
    i = write_tone(tmp_path / "i.wav", seconds=1.0, amp=0.2)
    y0, _ = sf.read(str(mix(v, i, tmp_path / "a.wav", 1.0, 1.0)))
    y1, _ = sf.read(str(mix(v, i, tmp_path / "b.wav", 0.5, 0.5)))
    assert np.allclose(y1, y0 * 0.5, atol=1e-3)


def test_nouveau_dossier_sans_collision(tmp_path):
    from studiovoix.outils import nouveau_dossier

    dossiers = [nouveau_dossier(tmp_path) for _ in range(3)]
    assert len(set(dossiers)) == 3 and all(d.is_dir() for d in dossiers)
