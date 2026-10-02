"""Seed-VC : la voix de référence est raccourcie aux 15 s les plus chantées (≈ 3 fois moins de passes)."""
import numpy as np
import soundfile as sf

from studiovoix import seedvc


def test_reference_courte(tmp_path):
    sr = 44100
    y = np.zeros(int(28 * sr), dtype="float32")
    y[int(10 * sr):int(25 * sr)] = 0.5 * np.sin(np.linspace(0, 15 * 2 * np.pi * 220, int(15 * sr)))  # chant de 10 à 25 s
    ref = tmp_path / "moi.wav"
    sf.write(ref, y, sr)
    courte = seedvc.reference_courte(ref, tmp_path)
    z, sr2 = sf.read(courte)
    assert sr2 == sr and len(z) == int(15 * sr) and courte.name == "reference_voix.wav"
    assert np.abs(z).mean() > 0.3  # la fenêtre gardée est celle où l'on chante, pas le silence du début
    # une voix déjà courte (≤ 16 s) est passée telle quelle
    sf.write(tmp_path / "court.wav", y[: int(12 * sr)], sr)
    assert seedvc.reference_courte(tmp_path / "court.wav", tmp_path) == tmp_path / "court.wav"
