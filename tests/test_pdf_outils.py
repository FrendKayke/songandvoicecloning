"""Outils PDF (vrais pypdf, pypdfium2, python-docx) : fusion, pages, rotation, compression, mot de passe, numéros,
filigrane, signature, images, texte / Word."""
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import pdf_outils as po


def _lire(chemin):
    from pypdf import PdfReader

    return PdfReader(str(chemin))


@pytest.fixture
def doc(env):
    """PDF de 5 pages : 4 pages de texte (Helvetica, accents) + 1 page image lourde (2400×1800 aléatoire)."""
    from pypdf import PdfWriter

    d = cfg.DATA_DIR.parent
    w = PdfWriter()
    for i in range(4):
        w.add_page(po._page_texte(595, 842, [(72, 760, 24, f"Page {i + 1} : Été à Noël, garçon", 0, 0)]))
    bruit = Image.fromarray(np.random.default_rng(0).integers(0, 255, (2400, 1800, 3)).astype(np.uint8))
    bruit.save(d / "img.pdf", "PDF", resolution=150)
    w.append(str(d / "img.pdf"))
    w.write(str(d / "doc.pdf"))
    sig = Image.new("RGBA", (400, 200), (0, 0, 0, 0))
    ImageDraw.Draw(sig).ellipse((20, 20, 380, 180), fill=(200, 0, 0, 255))
    sig.save(d / "sig.png")
    return d


def test_pages_voulues():
    assert po.pages_voulues("1-3, 5, 8-", 10) == [0, 1, 2, 4, 7, 8, 9]
    assert po.pages_voulues("-2", 5) == [0, 1] and po.pages_voulues("3", 3) == [2]
    for mauvais in ("", "0", "4-2", "6", "a"):
        with pytest.raises(gr.Error):
            po.pages_voulues(mauvais, 5)


def test_fusion_pages_rotation(doc):
    msg, f, dossier = po.fusionner([str(doc / "doc.pdf"), str(doc / "img.pdf")], "fusion", progress=no_progress)
    assert "2 PDF réunis (6 pages)" in msg and len(_lire(f[0]).pages) == 6 and Path(f[0]).name == "fusion.pdf"
    with pytest.raises(gr.Error, match="au moins deux"):
        po.fusionner([str(doc / "doc.pdf")], progress=no_progress)
    with pytest.raises(gr.Error, match="n'est pas un PDF"):
        po.fusionner([str(doc / "doc.pdf"), str(doc / "sig.png")], progress=no_progress)
    src = str(doc / "doc.pdf")
    msg, f, _ = po.pages(src, po.MODE_EXTRAIRE, "2-3, 5", progress=no_progress)
    assert "3 page(s) extraite(s)" in msg and "Page 2" in _lire(f[0]).pages[0].extract_text()
    msg, f, _ = po.pages(src, po.MODE_SUPPRIMER, "1", progress=no_progress)
    assert len(_lire(f[0]).pages) == 4 and "Page 2" in _lire(f[0]).pages[0].extract_text()
    msg, f, _ = po.pages(src, po.MODE_CHAQUE, "", progress=no_progress)
    assert len(f) == 6 and Path(f[0]).suffix == ".zip" and Path(f[1]).name == "doc_page_001.pdf"
    msg, f, _ = po.pages(src, po.MODE_PAQUETS, "2", progress=no_progress)
    assert "3 fichier(s)" in msg and len(_lire(f[-1]).pages) == 1
    msg, f, _ = po.pages(src, po.MODE_PLAGES, "1-2\n3-", progress=no_progress)
    assert len(f) == 3 and len(_lire(f[2]).pages) == 3
    msg, f, _ = po.pages(src, po.MODE_REORDONNER, "5,4,3,2,1", progress=no_progress)
    assert "Page 4" in _lire(f[0]).pages[1].extract_text()
    with pytest.raises(gr.Error, match="il manque"):
        po.pages(src, po.MODE_REORDONNER, "5,4,3", progress=no_progress)
    with pytest.raises(gr.Error, match="deux fois"):
        po.pages(src, po.MODE_REORDONNER, "1,1,2,3,4", progress=no_progress)
    with pytest.raises(gr.Error, match="toutes les pages"):
        po.pages(src, po.MODE_SUPPRIMER, "1-5", progress=no_progress)
    msg, f, _ = po.pivoter(src, "90° (quart de tour à droite)", "1-2", progress=no_progress)
    r = _lire(f[0])
    assert r.pages[0].rotation == 90 and r.pages[2].rotation == 0 and "2 page(s) tournée(s)" in msg


def test_compression_et_mot_de_passe(doc):
    src = doc / "doc.pdf"
    msg, f, _ = po.compresser(str(src), progress=no_progress)
    assert Path(f[0]).stat().st_size < src.stat().st_size / 5 and "1 image(s) réencodée(s)" in msg
    assert "Page 1" in _lire(f[0]).pages[0].extract_text()  # le texte reste du texte
    msg, f, _ = po.proteger(str(src), po.PROTEGER, "secret", progress=no_progress)
    r = _lire(f[0])
    assert r.is_encrypted and not r.decrypt("faux") and r.decrypt("secret")
    with pytest.raises(gr.Error, match="protégé par un mot de passe"):
        po.pages(f[0], po.MODE_CHAQUE, progress=no_progress)
    msg, f2, _ = po.proteger(f[0], po.DEVERROUILLER, "secret", progress=no_progress)
    assert not _lire(f2[0]).is_encrypted and "Page 3" in _lire(f2[0]).pages[2].extract_text()
    with pytest.raises(gr.Error, match="mot de passe"):
        po.proteger(f[0], po.DEVERROUILLER, "faux", progress=no_progress)
    with pytest.raises(gr.Error, match="Écris le mot de passe"):
        po.proteger(str(src), po.PROTEGER, "", progress=no_progress)


def _gris(image):
    return np.array(image.convert("L"))


def test_numeros_filigrane_signature(doc):
    src = str(doc / "doc.pdf")
    msg, f, _ = po.numeroter(src, "En bas à droite", "Page {n} / {total}", progress=no_progress)
    assert _lire(f[0]).pages[0].extract_text().endswith("Page 1 / 5")
    a = _gris(po.apercu(f[0]))
    h, w = a.shape
    assert (a[int(h * 0.95):, int(w * 0.75):] < 128).sum() > 50 and (a[:int(h * 0.05), :int(w * 0.25)] < 128).sum() == 0
    msg, f, _ = po.numeroter(src, "En haut au centre", "- {n} -", premiere=3, progress=no_progress)
    assert "- 3 -" in _lire(f[0]).pages[0].extract_text() and "- 7 -" in _lire(f[0]).pages[4].extract_text()
    with pytest.raises(gr.Error, match="Format du numéro"):
        po.numeroter(src, "En bas au centre", "{inconnu}", progress=no_progress)
    # filigrane : texte en diagonale (pâle) + image rouge transparente au centre
    msg, f, _ = po.filigrane(src, "CONFIDENTIEL", str(doc / "sig.png"), 0.3, 60, 45, progress=no_progress)
    assert "CONFIDENTIEL" in _lire(f[0]).pages[1].extract_text()
    im = np.array(po.apercu(f[0]).convert("RGB")).astype(int)
    centre = im[im.shape[0] // 2, im.shape[1] // 2]
    assert centre[0] > 200 and 150 < centre[1] < 220  # rouge pâle (opacité 0,3) sur blanc
    with pytest.raises(gr.Error, match="texte du filigrane"):
        po.filigrane(src, "", None, progress=no_progress)
    # signature : en bas à droite de la dernière page, transparence gardée (rien hors de l'ellipse)
    msg, f, _ = po.signer(src, str(doc / "sig.png"), "", 80, 12, 25, progress=no_progress)
    assert "page 5" in msg
    im = np.array(po.apercu(f[0], 5).convert("RGB")).astype(int)
    h, w, _ = im.shape
    zone = im[int(h * 0.84):int(h * 0.92), int(w * 0.70):int(w * 0.90)]
    assert ((zone[..., 0] > 150) & (zone[..., 1] < 80)).mean() > 0.6
    coin = im[int(h * 0.80):int(h * 0.83), int(w * 0.66):int(w * 0.69)]  # juste hors de l'ellipse : bruit gris, pas rouge
    assert ((coin[..., 0] > 150) & (coin[..., 1] < 80)).mean() < 0.05
    msg, f, _ = po.signer(src, str(doc / "sig.png"), "1, 3", progress=no_progress)
    assert "page 1, 3" in msg


def test_images_et_texte(doc):
    src = str(doc / "doc.pdf")
    msg, f, _ = po.en_images(src, "JPEG", 100, "1-2", progress=no_progress)
    assert len(f) == 3 and Path(f[0]).suffix == ".zip" and Image.open(f[1]).size == (827, 1170)
    assert _gris(Image.open(f[1]))[60:140, 80:450].min() < 100  # le titre dessiné
    msg, f, _ = po.en_texte(src, True, progress=no_progress)
    from docx import Document

    assert Path(f[0]).read_text(encoding="utf-8").startswith("Page 1 : Été à Noël, garçon")
    assert "1 page(s) sans texte" in msg and [p.text for p in Document(f[1]).paragraphs][0] == "Page 1 : Été à Noël, garçon"
    with pytest.raises(gr.Error, match="Aucun texte"):
        po.en_texte(str(doc / "img.pdf"), progress=no_progress)
    assert po.apercu(None) is None
