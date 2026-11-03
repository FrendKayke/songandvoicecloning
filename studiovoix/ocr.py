"""Reconnaissance de caractères (OCR) des PDF scannés et des photos de documents : OnnxTR (Apache 2.0, modèles de
docTR en ONNX, sur le processeur ; vocabulaire « french » : accents, €, guillemets).

Modèles (~105 Mo, `ONNXTR_CACHE_DIR` = <StudioVoix>/ocr) : détection `fast_base` (version reparamétrée) et lecture
`crnn_vgg16_bn`, téléchargés par l'installateur (`python -m studiovoix.ocr --telecharger`) ou au premier usage.

Sorties : PDF cherchable (page d'origine + texte invisible posé sur chaque mot : on peut chercher, sélectionner et
copier), Word (.docx, un paragraphe par bloc, un saut de page par page) et texte (.txt).
Rangement : data/pdf/<horodatage>_<nom>/ (comme les autres outils PDF).
"""
import io
import math
import os
import sys
import threading
from pathlib import Path

import gradio as gr
import numpy as np

from . import config as cfg
from .outils import chemin_fichier

# fichiers téléchargés par OnnxTR (nom de ses URL de publication, avec le début de leur SHA-256)
FICHIERS = ("rep_fast_base-1b89ebf9.onnx", "crnn_vgg16_bn-743599aa.onnx")
DETECTION, LECTURE = "fast_base", "crnn_vgg16_bn"
PPP = 200  # résolution de lecture des pages d'un PDF (docTR est entraîné vers 150–300 ppp)
EXTENSIONS_IMAGE = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif")
PAGES_MAX = 500
_verrou = threading.Lock()
_predicteur = None


def dossier_modeles():
    return Path(os.environ.get("ONNXTR_CACHE_DIR") or (cfg.ENG_DIR / "ocr"))


def manquants():
    return [f for f in FICHIERS if not (dossier_modeles() / "models" / f).is_file()]


def _charger():
    """Prédicteur OnnxTR gardé en mémoire (~0,5 Go). Télécharge les modèles s'ils manquent."""
    global _predicteur
    with _verrou:
        if _predicteur is None:
            os.environ["ONNXTR_CACHE_DIR"] = str(dossier_modeles())  # lu par OnnxTR à chaque téléchargement
            try:
                from onnxtr.models import ocr_predictor
            except ImportError:
                raise gr.Error("La reconnaissance de caractères n'est pas installée (paquet onnxtr) : relance "
                               "METTRE_A_JOUR.bat.") from None
            try:
                _predicteur = ocr_predictor(det_arch=DETECTION, reco_arch=LECTURE, assume_straight_pages=True,
                                            detect_language=False)
            except OSError as e:  # pas de réseau au premier usage
                raise gr.Error(f"Modèles de lecture absents et téléchargement impossible ({e}). Relance "
                               "METTRE_A_JOUR.bat avec Internet.") from None
        return _predicteur


# --- Pages à lire -------------------------------------------------------------------------------------------------

def _image_rgb(im):
    from PIL import ImageOps

    im = ImageOps.exif_transpose(im)  # photos de téléphone couchées
    if im.mode in ("RGBA", "LA", "P"):
        from PIL import Image

        im = im.convert("RGBA")
        fond = Image.new("RGBA", im.size, "white")
        im = Image.alpha_composite(fond, im)
    return im.convert("RGB")


def pages_pdf(chemin, numeros, ppp=PPP):
    """Pages (numéros à partir de 0) d'un PDF rendues en images."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(chemin))
    try:
        return [doc[i].render(scale=ppp / 72).to_pil().convert("RGB") for i in numeros]
    finally:
        doc.close()


def pages_de(chemin, ppp=PPP):
    """Fichier (PDF ou image, TIFF à plusieurs pages compris) → [image PIL RGB]."""
    from PIL import Image, ImageSequence

    chemin = Path(chemin)
    if chemin.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(chemin))
        try:
            if len(doc) > PAGES_MAX:
                raise gr.Error(f"{len(doc)} pages : {PAGES_MAX} au plus.")
            return [doc[i].render(scale=ppp / 72).to_pil().convert("RGB") for i in range(len(doc))]
        finally:
            doc.close()
    if chemin.suffix.lower() in (".heic", ".heif"):
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            pass
    with Image.open(chemin) as im:
        return [_image_rgb(p.copy()) for p in ImageSequence.Iterator(im)]


# --- Lecture ------------------------------------------------------------------------------------------------------

def lire_images(images, progress=None, debut=0.0, part=1.0):
    """[image PIL] → pages lues : [{"largeur", "hauteur", "blocs": [[ligne…]…]}], ligne = [(mot, (x0, y0, x1, y1),
    confiance)] en coordonnées relatives (0–1, origine en haut à gauche)."""
    predicteur = _charger()
    pages = []
    for i, im in enumerate(images):
        if progress:
            progress(debut + part * i / max(1, len(images)), desc=f"Lecture de la page {i + 1}/{len(images)}…")
        resultat = predicteur([np.asarray(im)]).export()["pages"][0]
        blocs = []
        for bloc in resultat["blocks"]:
            lignes = []
            for ligne in bloc["lines"]:
                mots = [(m["value"], (m["geometry"][0][0], m["geometry"][0][1], m["geometry"][1][0],
                                      m["geometry"][1][1]), float(m["confidence"])) for m in ligne["words"]]
                if mots:
                    lignes.append(mots)
            if lignes:
                blocs.append(lignes)
        pages.append({"largeur": im.width, "hauteur": im.height, "blocs": blocs})
    return pages


def texte_de(page):
    """Texte d'une page lue : une ligne par ligne, une ligne vide entre deux blocs."""
    return "\n\n".join("\n".join(" ".join(m[0] for m in ligne) for ligne in bloc) for bloc in page["blocs"])


def confiance(pages):
    c = [m[2] for p in pages for b in p["blocs"] for ligne in b for m in ligne]
    return float(np.mean(c)) if c else 0.0


# --- PDF cherchable -----------------------------------------------------------------------------------------------

def calque_texte(largeur, hauteur, page):
    """Page PDF (largeur × hauteur points) portant le texte lu, invisible (mode de rendu 3), chaque mot posé et
    étiré (Tz) sur sa boîte : la recherche et la sélection tombent sur les mots de l'image."""
    from pypdf import PageObject
    from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, FloatObject, NameObject, NumberObject

    from .pdf_outils import _largeur_texte, _texte_pdf

    calque = PageObject.create_blank_page(None, largeur, hauteur)
    police = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                               NameObject("/BaseFont"): NameObject("/Helvetica"),
                               NameObject("/Encoding"): NameObject("/WinAnsiEncoding")})
    calque[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): police}),
        NameObject("/ProcSet"): ArrayObject([NameObject("/PDF"), NameObject("/Text")])})
    morceaux = [b"BT 3 Tr"]
    for bloc in page["blocs"]:
        for ligne in bloc:
            for i, (mot, (x0, y0, x1, y1), _) in enumerate(ligne):
                texte = mot + (" " if i < len(ligne) - 1 else "")  # l'espace sépare les mots à la copie
                h = max(1.0, (y1 - y0) * hauteur)
                taille = h * 0.85
                naturelle = _largeur_texte(texte, taille) or 1.0
                cible = (x1 - x0) * largeur * (len(texte) / max(1, len(mot)))
                etirement = max(10.0, min(500.0, 100 * cible / naturelle))
                x, y = x0 * largeur, hauteur - y1 * hauteur + h * 0.2
                morceaux.append(f"/F1 {taille:.2f} Tf {etirement:.1f} Tz 1 0 0 1 {x:.2f} {y:.2f} Tm".encode()
                                + b" " + _texte_pdf(texte) + b" Tj")
    morceaux.append(b"ET")
    flux = DecodedStreamObject()
    flux.set_data(b"\n".join(morceaux))
    calque[NameObject("/Contents")] = flux
    calque[NameObject("/MediaBox")] = ArrayObject([NumberObject(0), NumberObject(0), FloatObject(largeur),
                                                  FloatObject(hauteur)])
    return calque


def pdf_cherchable(source, pages_lues, images, sortie, qualite=85):
    """PDF d'origine (ou images) + calque de texte invisible sur chaque page → `sortie`."""
    from pypdf import PdfReader, PdfWriter

    from .pdf_outils import _dimensions, _poser

    redacteur = PdfWriter()
    if Path(source).suffix.lower() == ".pdf":
        lecteur = PdfReader(str(source))
        for page, lue in zip(lecteur.pages, pages_lues):
            page = redacteur.add_page(page)  # calque posé sur la page du rédacteur (pypdf 6 : pas sur le lecteur)
            largeur, hauteur = _dimensions(page)
            boite = page.cropbox  # pdfium rend la zone visible : le calque la suit
            if lue["blocs"]:
                calque = calque_texte(largeur, hauteur, lue)
                if (page.rotation or 0) % 360 == 0 and (float(boite.left), float(boite.bottom)) != (0.0, 0.0):
                    from pypdf import Transformation

                    page.merge_transformed_page(calque, Transformation().translate(float(boite.left),
                                                                                   float(boite.bottom)))
                else:
                    _poser(page, calque)
    else:
        for im, lue in zip(images, pages_lues):
            tampon = io.BytesIO()
            im.save(tampon, "PDF", resolution=PPP, quality=qualite)  # image en JPEG dans le PDF, 200 ppp
            page = redacteur.add_page(PdfReader(io.BytesIO(tampon.getvalue())).pages[0])
            largeur, hauteur = float(page.mediabox.width), float(page.mediabox.height)
            if lue["blocs"]:
                page.merge_page(calque_texte(largeur, hauteur, lue))
    if hasattr(redacteur, "compress_identical_objects"):
        redacteur.compress_identical_objects()
    with open(sortie, "wb") as f:
        redacteur.write(f)
    return sortie


def ecrire_word(pages_textes, sortie):
    """[[paragraphe…] par page] → .docx, un saut de page entre deux pages."""
    from docx import Document
    from docx.enum.text import WD_BREAK

    document = Document()
    for i, paragraphes in enumerate(pages_textes):
        for p in paragraphes or ["(page sans texte)"]:
            document.add_paragraph(p)
        if i < len(pages_textes) - 1:
            document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    document.save(str(sortie))
    return sortie


def paragraphes(page):
    """Un paragraphe par bloc (les lignes d'un bloc recollées : un retour à la ligne du scan n'est pas une fin de
    paragraphe ; un mot coupé par un tiret en fin de ligne est recollé)."""
    sortie = []
    for bloc in page["blocs"]:
        texte = ""
        for ligne in bloc:
            l_ = " ".join(m[0] for m in ligne)
            if texte.endswith("-") and l_[:1].islower():
                texte = texte[:-1] + l_
            else:
                texte = f"{texte} {l_}".strip()
        sortie.append(texte)
    return sortie


def reconnaitre(fichiers, sorties=("pdf", "docx", "txt"), nom="", progress=gr.Progress()):
    """PDF scannés ou photos de documents → PDF cherchable, Word, texte. Plusieurs photos = un seul document (une
    page par photo, dans l'ordre) ; plusieurs PDF = un document par PDF. Renvoie (message, fichiers, dossier)."""
    from .pdf_outils import _dossier, _nom

    chemins = [Path(str(chemin_fichier(f))) for f in (fichiers or []) if chemin_fichier(f)]
    chemins = [c for c in chemins if c.is_file()]
    if not chemins:
        raise gr.Error("Ajoute un PDF scanné ou des photos de document.")
    autres = [c.name for c in chemins if c.suffix.lower() not in (".pdf",) + EXTENSIONS_IMAGE]
    if autres:
        raise gr.Error(f"Format non pris en charge : {', '.join(autres)} (PDF ou images).")
    sorties = [s for s in sorties or [] if s in ("pdf", "docx", "txt")] or ["pdf"]
    pdfs = [c for c in chemins if c.suffix.lower() == ".pdf"]
    photos = [c for c in chemins if c.suffix.lower() != ".pdf"]
    documents = [(c.stem, c, None) for c in pdfs] + ([(photos[0].stem if len(photos) == 1 else "photos", None, photos)]
                                                     if photos else [])
    dossier = _dossier(nom or chemins[0].stem, "ocr")
    fichiers_sortis, resume, n_doc = [], [], len(documents)
    for k, (base, pdf, images_src) in enumerate(documents):
        part = 1 / n_doc
        progress(k * part, desc=f"Ouverture de « {base} »…")
        images = pages_de(pdf) if pdf else [p for c in images_src for p in pages_de(c)]
        lues = lire_images(images, progress, k * part, part * 0.9)
        base = _nom(nom if n_doc == 1 else "", base)
        if "pdf" in sorties:
            progress(k * part + part * 0.92, desc="PDF cherchable…")
            fichiers_sortis.append(pdf_cherchable(pdf or images_src[0], lues, images,
                                                  dossier / f"{base}_cherchable.pdf"))
        if "docx" in sorties:
            fichiers_sortis.append(ecrire_word([paragraphes(p) for p in lues], dossier / f"{base}.docx"))
        if "txt" in sorties:
            t = dossier / f"{base}.txt"
            t.write_text("\n\n\f".join(texte_de(p) for p in lues) + "\n", encoding="utf-8")
            fichiers_sortis.append(t)
        mots = sum(len(ligne) for p in lues for b in p["blocs"] for ligne in b)
        vides = sum(1 for p in lues if not p["blocs"])
        resume.append(f"« {base} » : {len(lues)} page(s), {mots} mots, confiance moyenne "
                      f"{100 * confiance(lues):.0f} %" + (f", {vides} page(s) sans texte" if vides else ""))
    msg = ("✅ " + " ; ".join(resume) + f". Fichiers dans `{dossier}`.\n\nLe PDF cherchable garde les pages telles "
           "quelles, avec le texte reconnu par-dessus (invisible) : Ctrl+F et copier-coller marchent. Relis les "
           "chiffres importants : la lecture n'est jamais parfaite (photos floues, de travers, écriture à la main).")
    return msg, [str(f) for f in fichiers_sortis], str(dossier)


def telecharger():
    _charger()
    return dossier_modeles()


if __name__ == "__main__":
    if "--telecharger" in sys.argv:
        print(f"Modèles de lecture de texte dans {telecharger()}")
