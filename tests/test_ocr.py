"""Reconnaissance de caractères : PDF cherchable (texte invisible posé sur les mots), Word, texte, repli de « PDF →
texte / Word ». Faux lecteur pour la mécanique ; vrai OnnxTR si ses modèles sont là (ONNXTR_CACHE_DIR)."""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import ocr
from studiovoix import pdf_outils as po

LIGNES = ["Facture n° 2026-118", "Le château médiéval accueille les élèves", "Montant à payer : 1 250,00 €"]


def _police(taille):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "C:/Windows/Fonts/times.ttf",
              "/System/Library/Fonts/Supplemental/Times New Roman.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, taille)
    return ImageFont.load_default(taille)


def _scan(chemin, lignes=LIGNES):
    """Page A4 à 150 ppp portant les lignes ; enregistrée en image ou en PDF (images seules : un scan)."""
    im = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(im)
    for i, ligne in enumerate(lignes):
        d.text((120, 150 + 90 * i), ligne, fill="black", font=_police(34))
    if Path(chemin).suffix == ".pdf":
        im.save(chemin, "PDF", resolution=150)
    else:
        im.save(chemin)
    return chemin


class FauxLecteur:
    """Comme le prédicteur d'OnnxTR : chaque ligne de LIGNES, mots en boîtes relatives (export())."""

    def __call__(self, pages):
        h, w = pages[0].shape[:2]
        sombre = (pages[0].mean(axis=2) < 128).mean()

        class R:
            def export(_):
                blocs = []
                if sombre > 0.001:
                    for i, ligne in enumerate(LIGNES):
                        mots, x = [], 120 / 1240
                        for m in ligne.split():
                            larg = len(m) * 18 / 1240
                            mots.append({"value": m, "confidence": 0.9,
                                         "geometry": ((x, (150 + 90 * i) / 1754), (x + larg, (195 + 90 * i) / 1754))})
                            x += larg + 10 / 1240
                        blocs.append({"lines": [{"words": mots}]})
                return {"pages": [{"blocks": blocs, "dimensions": (h, w)}]}
        return R()


@pytest.fixture
def faux(env, monkeypatch):
    monkeypatch.setattr(ocr, "_charger", lambda: FauxLecteur())
    return cfg.DATA_DIR.parent


def _texte_pdf(chemin):
    from pypdf import PdfReader

    return [p.extract_text() for p in PdfReader(str(chemin)).pages]


def test_pdf_scanne_rendu_cherchable(faux):
    src = _scan(faux / "scan.pdf")
    assert _texte_pdf(src) == [""]  # un scan : pas de texte
    msg, fichiers, dossier = ocr.reconnaitre([str(src)], ["pdf", "docx", "txt"], progress=no_progress)
    pdf, docx, txt = map(Path, fichiers)
    assert pdf.name == "scan_cherchable.pdf" and Path(dossier).parent == cfg.PDF_DIR
    texte = _texte_pdf(pdf)[0]
    for ligne in LIGNES:
        assert all(m in texte for m in ligne.split())
    # page d'origine gardée telle quelle (le texte est invisible) : même rendu que le scan
    avant, apres = np.asarray(po.apercu(str(src))), np.asarray(po.apercu(str(pdf)))
    assert avant.shape == apres.shape and np.abs(avant.astype(int) - apres.astype(int)).max() == 0
    from docx import Document

    assert [p.text for p in Document(str(docx)).paragraphs] == LIGNES
    assert txt.read_text(encoding="utf-8").startswith("Facture n° 2026-118\n\nLe château")
    assert "1 page(s), 16 mots, confiance moyenne 90 %" in msg


def test_photos_en_un_seul_document(faux):
    a, b = _scan(faux / "p1.png"), _scan(faux / "p2.jpg")
    vide = faux / "blanc.png"
    Image.new("RGB", (800, 1000), "white").save(vide)
    msg, fichiers, _ = ocr.reconnaitre([str(a), str(b), str(vide)], ["pdf"], nom="courrier", progress=no_progress)
    pages = _texte_pdf(fichiers[0])
    assert Path(fichiers[0]).name == "courrier_cherchable.pdf" and len(pages) == 3
    assert "château" in pages[1] and pages[2] == ""
    assert "3 page(s)" in msg and "1 page(s) sans texte" in msg


def test_formats_refuses(faux):
    with pytest.raises(Exception, match="Ajoute"):
        ocr.reconnaitre([], progress=no_progress)
    autre = faux / "note.docx"
    autre.write_text("x")
    with pytest.raises(Exception, match="non pris en charge"):
        ocr.reconnaitre([str(autre)], progress=no_progress)


def test_calque_sur_page_tournee(faux):
    """Page tournée de 90° : le calque suit l'orientation lue (pdf_outils._poser)."""
    from pypdf import PdfReader, PdfWriter

    src = _scan(faux / "scan.pdf")
    w = PdfWriter()
    w.append(str(src))
    w.pages[0].rotate(90)
    w.write(str(faux / "tourne.pdf"))
    _, fichiers, _ = ocr.reconnaitre([str(faux / "tourne.pdf")], ["pdf"], progress=no_progress)
    page = PdfReader(fichiers[0]).pages[0]
    assert page.rotation == 90 and "Montant" in page.extract_text()


def test_pdf_vers_texte_lit_les_pages_scannees(faux):
    from pypdf import PdfWriter

    _scan(faux / "scan.pdf")
    w = PdfWriter()
    w.add_page(po._page_texte(595, 842, [(72, 760, 24, "Page tapée : été", 0, 0)]))
    w.append(str(faux / "scan.pdf"))
    w.write(str(faux / "mixte.pdf"))
    msg, f, _ = po.en_texte(str(faux / "mixte.pdf"), True, progress=no_progress)
    texte = Path(f[0]).read_text(encoding="utf-8")
    assert "Page tapée : été" in texte and "Le château médiéval" in texte
    assert "1 page(s) scannée(s) lue(s)" in msg
    from docx import Document

    assert "Montant à payer : 1 250,00 €" in [p.text for p in Document(f[1]).paragraphs]


def test_paragraphes_recolles():
    page = {"blocs": [[[("Un", None, 1), ("pro-", None, 1)], [("blème", None, 1), ("ici.", None, 1)]],
                      [[("Fin", None, 1)]]]}
    assert ocr.paragraphes(page) == ["Un problème ici.", "Fin"]
    assert ocr.texte_de(page) == "Un pro-\nblème ici.\n\nFin"


@pytest.mark.skipif(not os.environ.get("ONNXTR_CACHE_DIR") or bool(ocr.manquants()),
                    reason="modèles d'OnnxTR absents (ONNXTR_CACHE_DIR)")
def test_vrai_onnxtr(env, monkeypatch):
    pytest.importorskip("onnxtr")
    monkeypatch.setattr(ocr, "_predicteur", None)
    src = _scan(cfg.DATA_DIR.parent / "vrai.pdf")
    msg, fichiers, _ = ocr.reconnaitre([str(src)], ["pdf", "txt"], progress=no_progress)
    texte = Path(fichiers[1]).read_text(encoding="utf-8")
    assert "château médiéval accueille les élèves" in texte and "1 250,00 €" in texte
    assert "élèves" in _texte_pdf(fichiers[0])[0]
