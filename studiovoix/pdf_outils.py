"""Outils PDF (groupe « 📄 PDF », à la manière d'iLovePDF mais sur le PC, sans rien envoyer) : fusionner, pages
(extraire, supprimer, diviser, réordonner), pivoter, compresser, protéger / déverrouiller, numéroter, filigrane,
PDF → images, PDF → texte / Word, signer. Bibliothèques : pypdf (lecture, écriture, chiffrement AES avec
`cryptography`), pypdfium2 (rendu des pages en images, image posée sur une page : signature, filigrane image),
python-docx (Word), Pillow. Le texte dessiné sur les pages (numéros, filigrane) est écrit directement dans un flux de
contenu PDF avec la police standard Helvetica (encodage WinAnsi : accents français compris, pas d'incorporation).
Chaque outil renvoie (message, fichiers produits, dossier) ; rangement : data/pdf/<horodatage>_<nom>/.
"""
import io
import re
import time
import zipfile
from pathlib import Path

import gradio as gr

from . import config as cfg
from .outils import chemin_fichier, nouveau_dossier

# --- Pages ------------------------------------------------------------------------------------------------------
MODE_EXTRAIRE = "Extraire les pages indiquées (un seul PDF)"
MODE_SUPPRIMER = "Supprimer les pages indiquées"
MODE_CHAQUE = "Diviser : une page par fichier"
MODE_PAQUETS = "Diviser : toutes les N pages (N dans le champ)"
MODE_PLAGES = "Diviser : un fichier par plage (une plage par ligne)"
MODE_REORDONNER = "Réordonner : les pages dans le nouvel ordre"
MODES_PAGES = [MODE_EXTRAIRE, MODE_SUPPRIMER, MODE_CHAQUE, MODE_PAQUETS, MODE_PLAGES, MODE_REORDONNER]

ANGLES = {"90° (quart de tour à droite)": 90, "180° (tête en bas)": 180, "270° (quart de tour à gauche)": 270}

POSITIONS = {"En bas au centre": ("bas", "centre"), "En bas à droite": ("bas", "droite"),
             "En bas à gauche": ("bas", "gauche"), "En haut au centre": ("haut", "centre"),
             "En haut à droite": ("haut", "droite"), "En haut à gauche": ("haut", "gauche")}
FORMAT_NUMERO_DEFAUT = "{n} / {total}"

PROTEGER = "Protéger par un mot de passe"
DEVERROUILLER = "Retirer le mot de passe"
MODES_PROTECTION = [PROTEGER, DEVERROUILLER]

FORMATS_IMAGE = {"PNG": "png", "JPEG": "jpg"}
PPP = [72, 100, 150, 200, 300]

QUALITES_COMPRESSION = {"Forte (images en JPEG qualité 40, 1 000 px au plus)": (40, 1000),
                        "Moyenne (JPEG qualité 60, 1 600 px)": (60, 1600),
                        "Légère (JPEG qualité 80, taille des images gardée)": (80, 0)}
PAGES_MAX_IMAGES = 500


def _pdf(fichier):
    f = chemin_fichier(fichier)
    if not f or not Path(str(f)).is_file():
        raise gr.Error("Ajoute un fichier PDF.")
    p = Path(str(f))
    if p.suffix.lower() != ".pdf":
        raise gr.Error(f"« {p.name} » n'est pas un PDF (pour le convertir : onglet « Convertir en PDF »).")
    return p


def _pdfs(fichiers):
    chemins = [_pdf(f) for f in (fichiers or []) if f]
    if not chemins:
        raise gr.Error("Ajoute un ou plusieurs fichiers PDF.")
    return chemins


def _lire(chemin, mot_de_passe=None):
    """PdfReader du fichier, déchiffré si besoin (message clair sans le bon mot de passe)."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        lecteur = PdfReader(str(chemin))
    except PdfReadError as e:
        raise gr.Error(f"« {chemin.name} » n'est pas un PDF lisible ({e}).") from e
    if lecteur.is_encrypted:
        try:
            ok = lecteur.decrypt(mot_de_passe or "")
        except Exception as e:  # noqa: BLE001 - algorithme non pris en charge
            raise gr.Error(f"« {chemin.name} » est chiffré et n'a pas pu être ouvert ({e}).") from e
        if not ok:
            raise gr.Error(f"« {chemin.name} » est protégé par un mot de passe : donne-le dans « Protéger ou "
                           "déverrouiller » pour le retirer d'abord.")
    return lecteur


def _nom(nom, defaut):
    nom = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (nom or "").strip()).strip(". ")
    return nom or defaut


def _dossier(nom, defaut):
    return nouveau_dossier(cfg.PDF_DIR, _nom(nom, defaut))


def _ecrire(redacteur, chemin):
    with open(chemin, "wb") as f:
        redacteur.write(f)
    return chemin


def _taille(octets):
    return f"{octets / 1e6:.1f} Mo" if octets >= 1e6 else f"{octets / 1e3:.0f} Ko"


def pages_voulues(texte, total):
    """« 1-3, 5, 8- » → indices (0-based) dans l'ordre donné ; erreur claire si une page n'existe pas."""
    texte = (texte or "").strip().replace(";", ",")
    if not texte:
        raise gr.Error("Indique les pages, par exemple « 1-3, 5, 8- » (8- = de la page 8 à la fin).")
    indices = []
    for morceau in re.split(r"[,\s]+", texte):
        if not morceau:
            continue
        m = re.fullmatch(r"(\d*)\s*-\s*(\d*)", morceau)
        if m and (m.group(1) or m.group(2)):
            a = int(m.group(1)) if m.group(1) else 1
            b = int(m.group(2)) if m.group(2) else total
        elif morceau.isdigit():
            a = b = int(morceau)
        else:
            raise gr.Error(f"Pages mal écrites : « {morceau} ». Exemples : « 2 », « 1-3 », « 8- ».")
        if a < 1 or b > total or a > b:
            raise gr.Error(f"La page {a if a < 1 or a > total else b} n'existe pas : le PDF a {total} page(s).")
        indices.extend(range(a - 1, b))
    return indices


def _plages(texte, total):
    """Une plage par ligne (« 1-3 », « 4-6 », « 7- ») → [[indices…], …]."""
    lignes = [l_ for l_ in (texte or "").splitlines() if l_.strip()]
    if not lignes:
        raise gr.Error("Écris une plage par ligne, par exemple « 1-3 » puis « 4-6 » puis « 7- ».")
    return [pages_voulues(l_, total) for l_ in lignes]


# --- Fusionner ----------------------------------------------------------------------------------------------------
def fusionner(fichiers, nom="fusion", progress=gr.Progress()):
    """Plusieurs PDF → un seul, dans l'ordre donné. Renvoie (message, [fichier], dossier)."""
    from pypdf import PdfWriter

    chemins = _pdfs(fichiers)
    if len(chemins) < 2:
        raise gr.Error("Ajoute au moins deux PDF à réunir.")
    dossier = _dossier(nom, "fusion")
    redacteur = PdfWriter()
    total = 0
    for i, c in enumerate(chemins):
        progress(i / len(chemins), desc=f"{c.name}…")
        lecteur = _lire(c)
        redacteur.append(lecteur)
        total += len(lecteur.pages)
    sortie = _ecrire(redacteur, dossier / f"{_nom(nom, 'fusion')}.pdf")
    return (f"✅ {len(chemins)} PDF réunis ({total} pages) : `{sortie}`", [str(sortie)], str(dossier))


# --- Pages : extraire, supprimer, diviser, réordonner ------------------------------------------------------------
def pages(fichier, mode=MODE_EXTRAIRE, texte="", nom="", progress=gr.Progress()):
    """Extraire / supprimer / diviser / réordonner les pages d'un PDF. Renvoie (message, fichiers, dossier)."""
    from pypdf import PdfWriter

    chemin = _pdf(fichier)
    lecteur = _lire(chemin)
    total = len(lecteur.pages)
    dossier = _dossier(nom or chemin.stem, "pages")
    base = _nom(nom, chemin.stem)
    sorties = []

    def ecrire(indices, suffixe):
        redacteur = PdfWriter()
        for i in indices:
            redacteur.add_page(lecteur.pages[i])
        sorties.append(_ecrire(redacteur, dossier / f"{base}{suffixe}.pdf"))

    if mode == MODE_EXTRAIRE:
        indices = pages_voulues(texte, total)
        ecrire(indices, "_extrait")
        msg = f"✅ {len(indices)} page(s) extraite(s) sur {total}"
    elif mode == MODE_SUPPRIMER:
        retirees = set(pages_voulues(texte, total))
        gardees = [i for i in range(total) if i not in retirees]
        if not gardees:
            raise gr.Error("Tu supprimes toutes les pages : il ne resterait rien.")
        ecrire(gardees, "_sans_pages")
        msg = f"✅ {len(retirees)} page(s) supprimée(s), {len(gardees)} gardée(s)"
    elif mode == MODE_CHAQUE:
        if total > PAGES_MAX_IMAGES:
            raise gr.Error(f"{total} pages : {PAGES_MAX_IMAGES} au plus pour une page par fichier.")
        for i in range(total):
            progress(i / total, desc=f"Page {i + 1}/{total}…")
            ecrire([i], f"_page_{i + 1:03d}")
        msg = f"✅ {total} fichiers d'une page"
    elif mode == MODE_PAQUETS:
        n = int(texte) if (texte or "").strip().isdigit() and int(texte) > 0 else 0
        if not n:
            raise gr.Error("Indique N, le nombre de pages par fichier (par exemple 10).")
        for k, debut in enumerate(range(0, total, n), 1):
            ecrire(list(range(debut, min(total, debut + n))), f"_partie_{k:02d}")
        msg = f"✅ {len(sorties)} fichier(s) de {n} pages au plus"
    elif mode == MODE_PLAGES:
        for k, indices in enumerate(_plages(texte, total), 1):
            ecrire(indices, f"_partie_{k:02d}")
        msg = f"✅ {len(sorties)} fichier(s), un par plage"
    elif mode == MODE_REORDONNER:
        indices = pages_voulues(texte, total)
        if sorted(indices) != list(range(total)):
            manquantes = sorted(set(range(total)) - set(indices))
            if len(indices) != len(set(indices)):
                raise gr.Error("Une page est citée deux fois : donne chaque page une seule fois.")
            raise gr.Error("Pour réordonner, cite toutes les pages une fois ; il manque : "
                           + ", ".join(str(i + 1) for i in manquantes) + ".")
        ecrire(indices, "_reordonne")
        msg = f"✅ {total} pages réordonnées"
    else:
        raise gr.Error("Choisis ce qu'il faut faire des pages.")
    if len(sorties) > 1:
        archive = dossier / f"{base}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for s in sorties:
                z.write(s, s.name)
        sorties.insert(0, archive)
    return f"{msg}, dans `{dossier}`.", [str(s) for s in sorties], str(dossier)


# --- Pivoter --------------------------------------------------------------------------------------------------------
def pivoter(fichier, angle_label, texte_pages="", nom="", progress=gr.Progress()):
    """Tourne toutes les pages (ou celles indiquées) de 90, 180 ou 270°. Renvoie (message, [fichier], dossier)."""
    from pypdf import PdfWriter

    chemin = _pdf(fichier)
    angle = ANGLES.get(angle_label) or int(re.sub(r"\D", "", str(angle_label)) or 90)
    lecteur = _lire(chemin)
    total = len(lecteur.pages)
    voulues = set(pages_voulues(texte_pages, total)) if (texte_pages or "").strip() else set(range(total))
    redacteur = PdfWriter()
    redacteur.append(lecteur)
    for i in voulues:
        redacteur.pages[i].rotate(angle)
    dossier = _dossier(nom or chemin.stem, "pivote")
    sortie = _ecrire(redacteur, dossier / f"{_nom(nom, chemin.stem)}_pivote.pdf")
    return f"✅ {len(voulues)} page(s) tournée(s) de {angle}° : `{sortie}`", [str(sortie)], str(dossier)


# --- Compresser -------------------------------------------------------------------------------------------------
def compresser(fichier, qualite_label=None, nom="", progress=gr.Progress()):
    """Images réencodées en JPEG (qualité et taille maximale selon le niveau), flux de contenu compressés, objets en
    double retirés. Renvoie (message, [fichier], dossier)."""
    from PIL import Image
    from pypdf import PdfWriter

    chemin = _pdf(fichier)
    qualite, cote_max = QUALITES_COMPRESSION.get(qualite_label) or next(iter(QUALITES_COMPRESSION.values()))
    lecteur = _lire(chemin)
    redacteur = PdfWriter()
    redacteur.append(lecteur)
    n_images, total = 0, len(redacteur.pages)
    for i, page in enumerate(redacteur.pages):
        progress(i / total, desc=f"Page {i + 1}/{total}…")
        for img in page.images:
            try:
                im = img.image
                if im is None:
                    continue
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                if cote_max and max(im.size) > cote_max:
                    im.thumbnail((cote_max, cote_max), Image.Resampling.LANCZOS)
                img.replace(im, quality=qualite)
                n_images += 1
            except Exception:  # noqa: BLE001 - image dans un format que pypdf ne sait pas réécrire : gardée telle quelle
                continue
        page.compress_content_streams()
    try:
        redacteur.compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)
    except TypeError:  # pypdf < 6.2 : anciens noms des options
        redacteur.compress_identical_objects(remove_identicals=True, remove_orphans=True)
    dossier = _dossier(nom or chemin.stem, "compresse")
    sortie = _ecrire(redacteur, dossier / f"{_nom(nom, chemin.stem)}_compresse.pdf")
    avant, apres = chemin.stat().st_size, sortie.stat().st_size
    gain = (1 - apres / avant) * 100 if avant else 0
    note = "" if gain > 1 else " (déjà compact : peu ou pas d'images à réduire)"
    return (f"✅ {_taille(avant)} → {_taille(apres)} (−{gain:.0f} %), {n_images} image(s) réencodée(s){note} : "
            f"`{sortie}`", [str(sortie)], str(dossier))


# --- Protéger / déverrouiller ---------------------------------------------------------------------------------
def proteger(fichier, mode=PROTEGER, mot_de_passe="", mot_de_passe_proprietaire="", nom="", progress=gr.Progress()):
    """Chiffre un PDF (AES-256, mot de passe d'ouverture, mot de passe propriétaire facultatif) ou retire son mot
    de passe. Renvoie (message, [fichier], dossier)."""
    from pypdf import PdfWriter

    chemin = _pdf(fichier)
    if not (mot_de_passe or "").strip():
        raise gr.Error("Écris le mot de passe.")
    if mode == DEVERROUILLER:
        lecteur = _lire(chemin, mot_de_passe)
        if not lecteur.is_encrypted:
            gr.Warning("Ce PDF n'était pas protégé : il est recopié tel quel.")
        redacteur = PdfWriter()
        redacteur.append(lecteur)
        dossier = _dossier(nom or chemin.stem, "deverrouille")
        sortie = _ecrire(redacteur, dossier / f"{_nom(nom, chemin.stem)}_deverrouille.pdf")
        return f"✅ Mot de passe retiré : `{sortie}`", [str(sortie)], str(dossier)
    lecteur = _lire(chemin)
    redacteur = PdfWriter()
    redacteur.append(lecteur)
    redacteur.encrypt(mot_de_passe, (mot_de_passe_proprietaire or "").strip() or None, algorithm="AES-256")
    dossier = _dossier(nom or chemin.stem, "protege")
    sortie = _ecrire(redacteur, dossier / f"{_nom(nom, chemin.stem)}_protege.pdf")
    return (f"✅ PDF protégé (AES-256) : `{sortie}`. Le mot de passe n'est enregistré nulle part : garde-le.",
            [str(sortie)], str(dossier))


# --- Texte dessiné sur une page (numéros, filigrane) --------------------------------------------------------------
def _texte_pdf(texte):
    """Chaîne PDF entre parenthèses, en WinAnsi (accents français) ; caractères hors table remplacés par « ? »."""
    octets = texte.encode("cp1252", errors="replace")
    return b"(" + octets.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def _largeur_texte(texte, taille):
    """Largeur approchée d'un texte en Helvetica (largeurs moyennes : chiffres et lettres ≈ 0,55 em, espaces 0,28)."""
    etroits, larges = set(" .,:;'`|!il1"), set("mwMW")
    return sum(0.28 if ch in etroits else (0.85 if ch in larges else 0.55) for ch in texte) * taille


def _page_texte(largeur, hauteur, elements, opacite=1.0):
    """Page PDF (largeur × hauteur points) portant des textes : [(x, y, taille, texte, angle, gris)], gris 0 à 1."""
    from pypdf import PageObject
    from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject, FloatObject, NameObject,
                               NumberObject)
    import math

    page = PageObject.create_blank_page(None, largeur, hauteur)
    police = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                               NameObject("/BaseFont"): NameObject("/Helvetica"),
                               NameObject("/Encoding"): NameObject("/WinAnsiEncoding")})
    etat = DictionaryObject({NameObject("/Type"): NameObject("/ExtGState"), NameObject("/CA"): FloatObject(opacite),
                             NameObject("/ca"): FloatObject(opacite)})
    ressources = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): police}),
        NameObject("/ExtGState"): DictionaryObject({NameObject("/GS1"): etat}),
        NameObject("/ProcSet"): ArrayObject([NameObject("/PDF"), NameObject("/Text")]),
    })
    page[NameObject("/Resources")] = ressources
    morceaux = [b"q /GS1 gs"]
    for x, y, taille, texte, angle, gris in elements:
        a = math.radians(angle)
        c, s = math.cos(a), math.sin(a)
        morceaux.append(f"BT /F1 {taille:.2f} Tf {gris:.3f} g {c:.4f} {s:.4f} {-s:.4f} {c:.4f} {x:.2f} {y:.2f} Tm".encode()
                        + b" " + _texte_pdf(texte) + b" Tj ET")
    morceaux.append(b"Q")
    flux = DecodedStreamObject()
    flux.set_data(b"\n".join(morceaux))
    page[NameObject("/Contents")] = flux
    page[NameObject("/MediaBox")] = ArrayObject([NumberObject(0), NumberObject(0), FloatObject(largeur), FloatObject(hauteur)])
    return page


def _dimensions(page):
    """(largeur, hauteur) visibles d'une page, rotation comprise."""
    boite = page.mediabox
    largeur, hauteur = float(boite.width), float(boite.height)
    if (page.rotation or 0) % 180 == 90:
        largeur, hauteur = hauteur, largeur
    return largeur, hauteur


def _poser(page, calque):
    """Calque (page PDF) posé sur une page en tenant compte de sa rotation (le calque suit l'orientation lue)."""
    from pypdf import Transformation

    rotation = (page.rotation or 0) % 360
    largeur, hauteur = float(page.mediabox.width), float(page.mediabox.height)
    if rotation == 0:
        page.merge_page(calque)
    elif rotation == 90:
        page.merge_transformed_page(calque, Transformation().rotate(-90).translate(0, hauteur))
    elif rotation == 180:
        page.merge_transformed_page(calque, Transformation().rotate(180).translate(largeur, hauteur))
    else:
        page.merge_transformed_page(calque, Transformation().rotate(90).translate(largeur, 0))


def numeroter(fichier, position_label="En bas au centre", format_numero=FORMAT_NUMERO_DEFAUT, premiere=1, taille=11,
              marge=28, nom="", progress=gr.Progress()):
    """Numéro sur chaque page : « {n} / {total} » (ou le format donné), position et taille au choix, numérotation
    qui commence à `premiere`. Renvoie (message, [fichier], dossier)."""
    from pypdf import PdfWriter

    chemin = _pdf(fichier)
    lecteur = _lire(chemin)
    redacteur = PdfWriter()
    redacteur.append(lecteur)
    total = len(redacteur.pages)
    bas_haut, alignement = POSITIONS.get(position_label, ("bas", "centre"))
    gabarit = (format_numero or "").strip() or FORMAT_NUMERO_DEFAUT
    taille, marge, premiere = float(taille or 11), float(marge or 28), int(premiere or 1)
    for i, page in enumerate(redacteur.pages):
        progress(i / total, desc=f"Page {i + 1}/{total}…")
        try:
            texte = gabarit.format(n=i + premiere, total=total + premiere - 1, page=i + 1, pages=total)
        except (KeyError, IndexError, ValueError) as e:
            raise gr.Error(f"Format du numéro invalide ({e}) : utilise {{n}} et {{total}}, par exemple « Page {{n}} / {{total}} ».") from e
        largeur, hauteur = _dimensions(page)
        lt = _largeur_texte(texte, taille)
        x = {"centre": (largeur - lt) / 2, "droite": largeur - marge - lt, "gauche": marge}[alignement]
        y = marge if bas_haut == "bas" else hauteur - marge - taille
        _poser(page, _page_texte(largeur, hauteur, [(x, y, taille, texte, 0, 0.2)]))
    dossier = _dossier(nom or chemin.stem, "numerote")
    sortie = _ecrire(redacteur, dossier / f"{_nom(nom, chemin.stem)}_numerote.pdf")
    return f"✅ {total} page(s) numérotée(s) ({position_label.lower()}) : `{sortie}`", [str(sortie)], str(dossier)


def filigrane(fichier, texte="", image=None, opacite=0.25, taille=48, angle=45, nom="", progress=gr.Progress()):
    """Filigrane sur chaque page : un texte en diagonale (gris, transparent), ou une image (PNG avec transparence
    conseillé) au centre. Renvoie (message, [fichier], dossier)."""
    from pypdf import PdfWriter

    chemin = _pdf(fichier)
    texte = (texte or "").strip()
    image = chemin_fichier(image)
    if not texte and not image:
        raise gr.Error("Écris le texte du filigrane, ou ajoute une image.")
    lecteur = _lire(chemin)
    redacteur = PdfWriter()
    redacteur.append(lecteur)
    total = len(redacteur.pages)
    opacite = min(1.0, max(0.05, float(opacite or 0.25)))
    if texte:
        taille, angle = float(taille or 48), float(angle or 0)
        for i, page in enumerate(redacteur.pages):
            progress(i / total, desc=f"Page {i + 1}/{total}…")
            largeur, hauteur = _dimensions(page)
            lt = _largeur_texte(texte, taille)
            import math

            a = math.radians(angle)
            # texte centré : son milieu (lt/2, taille/3) tourné de `angle` placé au centre de la page
            x = largeur / 2 - (lt / 2) * math.cos(a) + (taille / 3) * math.sin(a)
            y = hauteur / 2 - (lt / 2) * math.sin(a) - (taille / 3) * math.cos(a)
            _poser(page, _page_texte(largeur, hauteur, [(x, y, taille, texte, angle, 0.5)], opacite))
        quoi = f"texte « {texte} »"
    dossier = _dossier(nom or chemin.stem, "filigrane")
    sortie = dossier / f"{_nom(nom, chemin.stem)}_filigrane.pdf"
    _ecrire(redacteur, sortie)
    if image:
        _poser_image(sortie, Path(str(image)), None, 0.5, 0.5, 0.5, opacite, progress)
        quoi = (quoi + " et image") if texte else "image"
    return f"✅ Filigrane ({quoi}) sur {total} page(s) : `{sortie}`", [str(sortie)], str(dossier)


# --- Image posée sur des pages (signature, filigrane image) : pypdfium2 ---------------------------------------------
def _poser_image(pdf, image, pages_voulues_, x_rel, y_rel, largeur_rel, opacite=1.0, progress=None):
    """Image (PNG transparent accepté) posée sur les pages (indices ; None = toutes), centre en (x_rel, y_rel) de la
    page (0 à 1, origine en bas à gauche), largeur = largeur_rel × largeur de la page. Modifie le fichier sur place."""
    import pypdfium2 as pdfium
    from PIL import Image

    with Image.open(image) as im:
        im = im.convert("RGBA")
        if opacite < 1.0:
            alpha = im.getchannel("A").point(lambda v: int(v * opacite))
            im.putalpha(alpha)
        bitmap = pdfium.PdfBitmap.from_pil(im)
        ratio = im.height / im.width
    doc = pdfium.PdfDocument(str(pdf))
    try:
        indices = range(len(doc)) if pages_voulues_ is None else pages_voulues_
        for k, i in enumerate(indices):
            if progress:
                progress(0.5 + 0.5 * k / max(1, len(indices)), desc=f"Image sur la page {i + 1}…")
            page = doc[i]
            largeur, hauteur = page.get_size()
            lg = largeur * largeur_rel
            ht = lg * ratio
            obj = pdfium.PdfImage.new(doc)
            obj.set_bitmap(bitmap)
            obj.set_matrix(pdfium.PdfMatrix().scale(lg, ht).translate(largeur * x_rel - lg / 2, hauteur * y_rel - ht / 2))
            page.insert_obj(obj)
            page.gen_content()
        doc.save(str(pdf))
    finally:
        doc.close()
    return pdf


def signer(fichier, image, texte_pages="", x=80, y=12, largeur=25, nom="", progress=gr.Progress()):
    """Signature (image, PNG transparent conseillé) posée sur la dernière page (ou les pages indiquées), position en
    % de la page depuis la gauche et le bas, largeur en % de la page. Renvoie (message, [fichier], dossier)."""
    import shutil

    chemin = _pdf(fichier)
    image = chemin_fichier(image)
    if not image or not Path(str(image)).is_file():
        raise gr.Error("Ajoute l'image de la signature (PNG avec fond transparent de préférence, ou JPG).")
    total = len(_lire(chemin).pages)
    indices = pages_voulues(texte_pages, total) if (texte_pages or "").strip() else [total - 1]
    dossier = _dossier(nom or chemin.stem, "signe")
    sortie = dossier / f"{_nom(nom, chemin.stem)}_signe.pdf"
    shutil.copy(chemin, sortie)
    _poser_image(sortie, Path(str(image)), indices, float(x or 0) / 100, float(y or 0) / 100,
                 float(largeur or 25) / 100, 1.0, progress)
    return (f"✅ Signature posée sur la page {', '.join(str(i + 1) for i in indices)} : `{sortie}`",
            [str(sortie)], str(dossier))


# --- PDF → images -------------------------------------------------------------------------------------------------
def en_images(fichier, format_label="PNG", ppp=150, texte_pages="", nom="", progress=gr.Progress()):
    """Chaque page rendue en image (pypdfium2) + archive zip. Renvoie (message, fichiers, dossier)."""
    import pypdfium2 as pdfium

    chemin = _pdf(fichier)
    ext = FORMATS_IMAGE.get(format_label, "png")
    ppp = int(ppp or 150)
    doc = pdfium.PdfDocument(str(chemin))
    try:
        total = len(doc)
        indices = pages_voulues(texte_pages, total) if (texte_pages or "").strip() else list(range(total))
        if len(indices) > PAGES_MAX_IMAGES:
            raise gr.Error(f"{len(indices)} pages : {PAGES_MAX_IMAGES} au plus d'un coup (indique les pages).")
        dossier = _dossier(nom or chemin.stem, "images")
        base = _nom(nom, chemin.stem)
        sorties = []
        for k, i in enumerate(indices):
            progress(k / len(indices), desc=f"Page {i + 1}/{total}…")
            im = doc[i].render(scale=ppp / 72).to_pil()
            s = dossier / f"{base}_page_{i + 1:03d}.{ext}"
            if ext == "jpg":
                im.convert("RGB").save(s, "JPEG", quality=90)
            else:
                im.save(s, "PNG")
            sorties.append(s)
    finally:
        doc.close()
    archive = dossier / f"{base}_images.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for s in sorties:
            z.write(s, s.name)
    return (f"✅ {len(sorties)} page(s) en {format_label} à {ppp} ppp, dans `{dossier}`.",
            [str(archive)] + [str(s) for s in sorties], str(dossier))


# --- PDF → texte / Word -------------------------------------------------------------------------------------------
def en_texte(fichier, word=True, nom="", progress=gr.Progress()):
    """Texte du PDF (.txt, une page après l'autre) et document Word (.docx, un saut de page par page). Sans
    reconnaissance de caractères : un PDF scanné (images seules) ne donne pas de texte. Renvoie (message, fichiers, dossier)."""
    chemin = _pdf(fichier)
    lecteur = _lire(chemin)
    total = len(lecteur.pages)
    textes = []
    for i, page in enumerate(lecteur.pages):
        progress(i / total, desc=f"Page {i + 1}/{total}…")
        textes.append((page.extract_text() or "").strip())
    if not any(textes):
        raise gr.Error("Aucun texte dans ce PDF : ce sont des images (document scanné). La reconnaissance de "
                       "caractères n'est pas installée ; passe par « PDF en images » puis un outil d'OCR.")
    dossier = _dossier(nom or chemin.stem, "texte")
    base = _nom(nom, chemin.stem)
    txt = dossier / f"{base}.txt"
    txt.write_text("\n\n".join(textes) + "\n", encoding="utf-8")
    sorties = [txt]
    if word:
        from docx import Document
        from docx.enum.text import WD_BREAK

        document = Document()
        for i, t in enumerate(textes):
            for ligne in (t or "(page sans texte)").splitlines():
                document.add_paragraph(ligne)
            if i < total - 1:
                document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        docx = dossier / f"{base}.docx"
        document.save(str(docx))
        sorties.append(docx)
    caracteres = sum(len(t) for t in textes)
    vides = sum(1 for t in textes if not t)
    note = f" ; {vides} page(s) sans texte (images ?)" if vides else ""
    return (f"✅ {caracteres} caractères sur {total} page(s){note} : "
            + ", ".join(f"`{s.name}`" for s in sorties) + f" dans `{dossier}`. Le Word reprend le texte, pas la mise "
            "en page.", [str(s) for s in sorties], str(dossier))


def apercu(fichier, page=1):
    """Première page (ou la page donnée) rendue en image, pour vérifier un résultat dans l'interface."""
    import pypdfium2 as pdfium

    f = chemin_fichier(fichier)
    if not f or not Path(str(f)).is_file():
        return None
    doc = pdfium.PdfDocument(str(f))
    try:
        i = max(0, min(len(doc) - 1, int(page or 1) - 1))
        return doc[i].render(scale=1.2).to_pil()
    except Exception:  # noqa: BLE001 - PDF chiffré ou abîmé : pas d'aperçu
        return None
    finally:
        doc.close()
