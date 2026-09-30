"""Boucles parfaites, sur des musiques synthétiques (intro, grille de 4 accords, batterie, fin en fondu)."""
import numpy as np
import pytest
import soundfile as sf

from conftest import musique
from studiovoix import boucle
from studiovoix.config import SR


@pytest.mark.parametrize("bpm,graine", [(95, 1), (120, 0), (140, 2), (170, 1)])
def test_boucle_sur_le_temps_et_sur_la_grille(tmp_path, bpm, graine):
    y, intro, mesure = musique(bpm=bpm, seed=graine)
    sf.write(str(tmp_path / "m.wav"), y.T, SR)
    r = boucle.creer_boucle(tmp_path / "m.wav", tmp_path / "b.wav", tmp_path / "j.wav")
    temps = 60 / bpm
    ecart = (r.debut - intro) % temps
    assert min(ecart, temps - ecart) < 0.015                       # démarre sur un temps
    cycles = (r.fin - r.debut) / (4 * mesure)
    assert abs(cycles - round(cycles)) < 0.005 and cycles >= 1      # grille d'accords entière
    assert r.fin < intro + 16 * mesure                              # pas dans la fin en fondu
    assert r.jonction <= 2.5 and r.qualite() in ("excellente", "bonne")
    assert sf.info(str(tmp_path / "b.wav")).duration == pytest.approx(r.duree, abs=1e-3)


def test_mesure_distingue_une_mauvaise_boucle(tmp_path):
    y, intro, mesure = musique(bpm=120)
    a = int((intro + mesure) * SR)
    bonne = a + int(12 * mesure * SR)
    mauvaise = bonne - int(1.5 * mesure * SR)  # autre accord, à contretemps
    assert boucle.mesure_jonction(y, boucle.boucler(y, a, bonne, 0.1), bonne) < 1.6
    assert boucle.mesure_jonction(y, boucle.boucler(y, a, mauvaise, 0.1), mauvaise) > 3


def test_raccord_continu():
    """Au bouclage, on entend d'abord la suite naturelle de la fin : la forme d'onde ne saute pas."""
    y, intro, mesure = musique(bpm=120)
    a, b = int((intro + mesure) * SR), int((intro + 13 * mesure) * SR)
    s = boucle.boucler(y, a, b, 0.1)
    assert np.allclose(s[:, 0], y[:, b], atol=1e-6) and np.allclose(s[:, -1], y[:, b - 1])
    assert s.shape[1] == b - a


def test_morceau_sans_boucle_possible(tmp_path):
    t = np.arange(3 * SR) / SR
    sf.write(str(tmp_path / "court.wav"), (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), SR)
    with pytest.raises(ValueError):
        boucle.creer_boucle(tmp_path / "court.wav", tmp_path / "b.wav")
