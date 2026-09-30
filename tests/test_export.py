"""Export : mesure du volume (comparée à pyloudnorm), normalisation, OGG/MP3, pack de bande-son."""
import json
import zipfile

import gradio as gr
import numpy as np
import pytest
import soundfile as sf

from conftest import musique, no_progress, wav_octets
from studiovoix import config as cfg
from studiovoix import export, jeu

SR = 44100
CIBLE_16 = list(export.CIBLES)[1]


@pytest.mark.parametrize("sr", [22050, 44100, 48000])
def test_mesure_identique_a_pyloudnorm(sr):
    pyln = pytest.importorskip("pyloudnorm")  # référence de test seulement (pas installée dans l'application)
    rng = np.random.default_rng(sr)
    y = np.stack([0.1 * rng.standard_normal(5 * sr), 0.05 * rng.standard_normal(5 * sr)])
    assert export.loudness(y, sr) == pytest.approx(pyln.Meter(sr).integrated_loudness(y.T), abs=0.01)


def test_normalisation_et_pic():
    y = musique()[0].astype(float) * 0.2
    z, lufs = export.normaliser(y, SR, -16.0)
    assert lufs == pytest.approx(-16.0, abs=0.05) and np.max(np.abs(z)) <= export.PIC_MAX + 1e-9
    z, lufs = export.normaliser(y, SR, -6.0)  # trop fort : le pic plafonne à −1 dBFS, le volume reste en dessous
    assert np.max(np.abs(z)) == pytest.approx(export.PIC_MAX, abs=1e-6) and lufs < -6.0
    silence, l_silence = export.normaliser(np.zeros((2, SR)), SR, -16.0)
    assert not np.isfinite(l_silence) and not silence.any()


def test_export_d_une_chanson(tmp_path):
    src = tmp_path / "chanson_finale.wav"
    sf.write(str(src), musique()[0].T * 0.3, SR)
    fichier, msg = export.exporter_fichier(str(src), list(export.CIBLES)[0])
    assert fichier == str(tmp_path / "chanson_finale_export.mp3") and "LUFS" in msg
    y, sr = sf.read(fichier, always_2d=True)
    # −14 LUFS ferait dépasser −1 dBFS à cette musique : le gain est plafonné par les crêtes, et c'est dit
    assert -15.5 < export.loudness(y.T, sr) <= -13.5 and "limité par les crêtes" in msg
    assert np.max(np.abs(y)) <= export.PIC_MAX * 1.12  # marge de la compression MP3
    with pytest.raises(gr.Error):
        export.exporter_fichier(None, CIBLE_16)


def test_pack_de_bande_son(fake_acestep, env, monkeypatch):
    monkeypatch.setattr(cfg, "GAMES_DIR", env / "data" / "jeux")
    srv = fake_acestep()
    y, _, _ = musique(bpm=120)
    srv.version = lambda i: wav_octets(y)
    jeu.generer_bande_son("p", jeu.EPOQUES[0][1], [], ["combat", "victoire"], [], 60, False, progress=no_progress)
    jeu.generer_bande_son("p", jeu.EPOQUES[0][1], [], ["combat"], [], 60, False, progress=no_progress)  # plus récente
    archive, msg = export.exporter_pack("p", CIBLE_16, ["ogg", "mp3"], progress=no_progress)
    dossier = cfg.GAMES_DIR / "p" / "export"
    manifest = json.loads((dossier / "manifest.json").read_text(encoding="utf-8"))
    assert [p["id"] for p in manifest["pistes"]] == ["combat", "victoire"]
    combat, victoire = manifest["pistes"]
    assert combat["boucle"] is True and combat["bpm"] and victoire["boucle"] is False
    assert combat["fichiers"] == {"ogg": "combat.ogg", "mp3": "combat.mp3"}
    recente = sorted((cfg.GAMES_DIR / "p" / "combat").iterdir())[-1]
    assert combat["graine"] == json.loads((recente / "creation.json").read_text())["versions"][0]["graine"]
    for f in ("combat.ogg", "combat.mp3"):  # même longueur que la boucle : elle reste exacte
        assert sf.info(str(dossier / f)).frames == sf.info(str(recente / "piste.wav")).frames
    for p in manifest["pistes"]:
        assert p["volume_lufs"] == pytest.approx(-16.0, abs=0.6)
    assert sorted(zipfile.ZipFile(archive).namelist()) == sorted(
        ["manifest.json", "combat.ogg", "combat.mp3", "victoire.ogg", "victoire.mp3"])
    with pytest.raises(gr.Error, match="format"):
        export.exporter_pack("p", CIBLE_16, [], progress=no_progress)
    with pytest.raises(gr.Error, match="Aucune piste"):
        export.exporter_pack("vide", CIBLE_16, ["ogg"], progress=no_progress)
