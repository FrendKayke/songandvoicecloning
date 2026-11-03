"""Convertisseur audio / vidéo avec le vrai PyAV : formats, son extrait, coupe à la seconde, recollage de vidéos de
tailles et cadences différentes (son calé, silence pour une vidéo muette), allègement."""
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import soundfile as sf

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import convertir as cv


def _clip(chemin, secondes=3, fps=12, taille=(320, 240), freq=220, avec_son=True, couleur=0):
    """Vidéo H.264 + AAC : chaque image porte son numéro en gris (pour reconnaître les images gardées)."""
    import av

    w, h = taille
    with av.open(str(chemin), "w") as out:
        v = out.add_stream("libx264", rate=fps)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        a = out.add_stream("aac", rate=44100, layout="mono") if avec_son else None
        for i in range(secondes * fps):
            image = np.full((h, w, 3), couleur, np.uint8)
            image[:, : w // 2] = min(255, int(i / (secondes * fps) * 255))  # moitié gauche : avance dans le temps
            for p in v.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                out.mux(p)
        if a:
            t = np.arange(secondes * 44100) / 44100
            son = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None]
            for i in range(0, son.shape[1], 1024):
                tr = av.AudioFrame.from_ndarray(np.ascontiguousarray(son[:, i:i + 1024]), format="fltp", layout="mono")
                tr.sample_rate = 44100
                for p in a.encode(tr):
                    out.mux(p)
            for p in a.encode(None):
                out.mux(p)
        for p in v.encode(None):
            out.mux(p)
    return chemin


def _son(fichier):
    """(signal mono, fréquence) d'un fichier, lu par PyAV."""
    import av

    with av.open(str(fichier)) as c:
        if not c.streams.audio:
            return None, None
        s = c.streams.audio[0]
        r = av.AudioResampler(format="flt", layout="mono", rate=44100)
        morceaux = [x.to_ndarray().ravel() for tr in c.decode(s) for x in r.resample(tr)]
    return np.concatenate(morceaux), 44100


def _frequence(y, sr):
    spectre = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.fft.rfftfreq(len(y), 1 / sr)[spectre.argmax()])


def _images(fichier):
    import av

    with av.open(str(fichier)) as c:
        tr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        return tr, c.streams.video[0].codec_context.name, float(c.streams.video[0].average_rate or 0)


def test_temps_ecrits():
    assert cv.duree_lisible(1.5) == "1,5 s" and cv.duree_lisible(6) == "6 s" and cv.duree_lisible(90) == "1 min 30 s"
    assert cv.temps("90") == 90 and cv.temps("1:30") == 90 and cv.temps("1:02:03") == 3723
    assert cv.temps("1 min 30") == 90 and cv.temps("12,5") == 12.5 and cv.temps("", 7) == 7 and cv.temps("2h") == 7200
    with pytest.raises(gr.Error, match="mal écrit"):
        cv.temps("demain")


def test_formats_et_son_extrait(env):
    d = cfg.DATA_DIR.parent
    clip = _clip(d / "Mon clip.mp4", freq=330)
    assert "320×240 à 12 images/s" in cv.decrire(str(clip)) and "avec du son" in cv.decrire(str(clip))
    for label in cv.FORMATS:
        msg, fichiers, dossier = cv.changer_format([str(clip)], label, progress=no_progress)
        f = Path(fichiers[0])
        ext, codec_v, codec_a, _ = cv.FORMATS[label]
        assert f.suffix == f".{ext}" and f.stat().st_size > 0, label
        info = cv.infos(f)
        assert abs(info["duree"] - 3) < 0.3, (label, info)
        if codec_v:
            images, codec, fps = _images(f)
            assert len(images) in range(34, 38) if codec_v != "gif" else len(images) in range(34, 38), (label, len(images))
            assert codec == {"libx264": "h264", "libvpx-vp9": "vp9", "gif": "gif"}[codec_v]
        if codec_a:
            y, sr = _son(f)
            assert abs(_frequence(y, sr) - 330) < 5, label
    # son extrait d'une vidéo : MP3 à 330 Hz ; une vidéo muette est refusée
    msg, fichiers, _ = cv.extraire_son([str(clip)], "MP3", progress=no_progress)
    assert fichiers[0].endswith("Mon_clip.mp3") and not cv.infos(fichiers[0])["video"]
    muet = _clip(d / "muet.mp4", avec_son=False)
    with pytest.raises(gr.Error, match="pas de son"):
        cv.extraire_son([str(muet)], "MP3", progress=no_progress)
    # son seul vers un format vidéo : refusé clairement ; fichier illisible signalé sans arrêter les autres
    wav = d / "voix.wav"
    sf.write(str(wav), (0.2 * np.sin(np.arange(44100) / 5)).astype("float32"), 44100)
    with pytest.raises(gr.Error, match="pas d'image"):
        cv.changer_format([str(wav)], "MP4 (H.264, partout)", progress=no_progress)
    (d / "abime.mp4").write_bytes(b"pas une video")
    msg, fichiers, _ = cv.changer_format([str(d / "abime.mp4"), str(clip)], "MP3", progress=no_progress)
    assert len(fichiers) == 1 and "abime.mp4" in msg and "⚠️" in msg


def test_couper_au_bon_endroit(env):
    clip = _clip(cfg.DATA_DIR.parent / "clip.mp4", secondes=6, fps=10)
    msg, fichiers, _ = cv.couper(str(clip), "2", "4.5", progress=no_progress)
    images, codec, _ = _images(fichiers[0])
    assert len(images) == 25 and codec == "h264"  # 2,5 s à 10 images/s
    # la première image gardée est celle de 2 s (gris ≈ 20/60 × 255), la dernière celle de 4,4 s
    assert abs(int(images[0][10, 10, 0]) - round(20 / 60 * 255)) < 8
    assert abs(int(images[-1][10, 10, 0]) - round(44 / 60 * 255)) < 8
    y, sr = _son(fichiers[0])
    assert abs(len(y) / sr - 2.5) < 0.05
    with pytest.raises(gr.Error, match="après le début"):
        cv.couper(str(clip), "5", "1", progress=no_progress)
    # fin au-delà de la durée : coupée à la fin du fichier ; « 1:00 » de début refusé (plus long que le fichier)
    msg, fichiers, _ = cv.couper(str(clip), "5", "20", progress=no_progress)
    assert len(_images(fichiers[0])[0]) == 10
    with pytest.raises(gr.Error):
        cv.couper(str(clip), "1:00", "", progress=no_progress)


def test_recoller_tailles_cadences_et_muet(env):
    d = cfg.DATA_DIR.parent
    a = _clip(d / "a.mp4", secondes=2, fps=10, taille=(320, 240), freq=220, couleur=0)
    b = _clip(d / "b.mp4", secondes=2, fps=25, taille=(240, 320), freq=440, couleur=200)  # portrait, autre cadence
    c = _clip(d / "c.mp4", secondes=1, fps=10, taille=(640, 480), avec_son=False, couleur=100)  # muette, même forme
    msg, fichiers, _ = cv.recoller([str(a), str(b), str(c)], "MP4 (H.264, partout)", "montage", progress=no_progress)
    images, _, fps = _images(fichiers[0])
    assert fps == 10 and len(images) == 50 and images[0].shape == (240, 320, 3)  # 5 s à la cadence du premier
    assert images[25][120, 20].max() < 30 and images[25][120, 160].min() > 150  # b : bandes noires sur les côtés
    assert abs(int(images[45][120, 300, 0]) - 100) < 12  # c agrandi à 320×240, sa couleur gardée
    y, sr = _son(fichiers[0])
    assert abs(len(y) / sr - 5) < 0.1
    assert abs(_frequence(y[int(0.2 * sr):int(1.8 * sr)], sr) - 220) < 5
    assert abs(_frequence(y[int(2.2 * sr):int(3.8 * sr)], sr) - 440) < 5
    assert np.abs(y[int(4.1 * sr):int(4.9 * sr)]).max() < 0.01  # silence pendant la vidéo muette
    # sons seuls : MP3 de 4 s
    w1, w2 = d / "1.wav", d / "2.wav"
    sf.write(str(w1), np.zeros(2 * 44100, np.float32) + 0.1, 44100)
    sf.write(str(w2), np.zeros(2 * 22050, np.float32) + 0.1, 22050)
    msg, fichiers, _ = cv.recoller([str(w1), str(w2)], "MP4 (H.264, partout)", progress=no_progress)
    assert fichiers[0].endswith(".mp3") and abs(cv.infos(fichiers[0])["duree"] - 4) < 0.15
    with pytest.raises(gr.Error, match="pas d'image"):
        cv.recoller([str(a), str(w1)], "MP4 (H.264, partout)", progress=no_progress)
    with pytest.raises(gr.Error, match="au moins deux"):
        cv.recoller([str(a)], progress=no_progress)


def test_reduire_le_poids(env):
    d = cfg.DATA_DIR.parent
    import av

    grand = d / "grand.mp4"
    with av.open(str(grand), "w") as out:  # 1280×720 très détaillée (bruit), haute qualité : lourde
        v = out.add_stream("libx264", rate=24)
        v.width, v.height, v.pix_fmt, v.options = 1280, 720, "yuv420p", {"crf": "10"}
        rng = np.random.default_rng(0)
        for _ in range(48):
            for p in v.encode(av.VideoFrame.from_ndarray(rng.integers(0, 255, (720, 1280, 3), np.uint8), format="rgb24")):
                out.mux(p)
        for p in v.encode(None):
            out.mux(p)
    msg, fichiers, _ = cv.reduire([str(grand)], "Forte (480p, pour envoyer par message)", progress=no_progress)
    info = cv.infos(fichiers[0])
    assert (info["largeur"], info["hauteur"]) == (852, 480) and info["fps"] == 24
    assert Path(fichiers[0]).stat().st_size < grand.stat().st_size / 3 and "−" in msg
