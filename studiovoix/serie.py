"""Images en série : une image par ligne d'un tableau (Excel, CSV) ou d'une liste collée, avec un contexte et un style
communs — par exemple 250 icônes carrées pour une application.

Le tableau : choix de la feuille, de la colonne des prompts, d'une colonne de noms (noms de fichiers) et du contexte
(écrit, dans une case du fichier, ou une colonne : un contexte par ligne). Chaque image reçoit « prompt. contexte de
la ligne. contexte commun » puis le style. Style imposé : la liste des styles, et/ou des images de style (FLUX.2 klein
les reçoit en référence et ne garde que leur rendu) ; sans image de style : Z-Image-Turbo.
Les images partent au moteur par paquets (PAQUET) : un paquet en échec n'arrête pas le lot, et « Reprendre » refait
seulement les images manquantes (lot.json). Sortie : data/series/<horodatage>_<nom>/ avec 001_<nom>.png…, les copies
réduites (<taille>px/), lot.csv (pour Excel), lot.json et une archive zip.
"""
import csv
import io
import json
import re
import shutil
import time
import unicodedata
import zipfile
from pathlib import Path

import gradio as gr
from PIL import Image

from . import config as cfg
from . import diffusion
from .images import FORMATS as FORMATS_IMAGES
from .images import STYLES as STYLES_IMAGES
from .outils import nouveau_dossier
from .styles import texte
from .videos import duree_lisible

SERIE_MAX = 1000
PAQUET = 8  # images par tâche du moteur : l'encodeur de texte ne passe sur la carte qu'une fois par paquet
ECHECS_MAX = 3  # paquets en échec à la suite : le lot s'arrête (moteur absent, carte saturée…)
AUCUNE = "(aucune)"
FORMATS = {k: v for k, v in FORMATS_IMAGES.items() if v}  # pas d'« automatique » : pas de photo
FORMAT_DEFAUT = "Carré 1:1 (1024×1024)"
TAILLES = [512, 256, 192, 128, 96, 64, 48, 32]  # copies réduites (côté le plus long), pour des icônes
# Styles pensés pour des icônes, en tête de liste ; puis ceux de l'onglet « Une image »
STYLES = [
    ("Icône d'application plate", "flat vector app icon, single centered subject, bold simple shapes, clean solid "
                                  "background, smooth gradients, no text, no letters"),
    ("Icône 3D douce", "3D rendered app icon, single centered subject, soft studio lighting, rounded glossy shapes, "
                       "clean solid background, no text, no letters"),
    ("Icône contour (line art)", "minimal line art icon, single centered subject, uniform stroke width, clean white "
                                 "background, no text, no letters"),
    ("Icône pixel art", "pixel art icon, 32x32 style, single centered subject, crisp pixels, limited palette, plain "
                        "background, no text"),
    ("Autocollant (sticker)", "die-cut sticker, single centered subject, thick white border, vibrant colors, plain "
                              "background, no text"),
] + list(STYLES_IMAGES)
STYLE_DES_IMAGES = ("Use only the art style of the reference image(s): same rendering technique, colors, line work, "
                    "shading, lighting and level of detail. Do not copy their subject, objects or composition")
STYLES_IMAGES_MAX = 3


# --- Lecture du tableau --------------------------------------------------------------------------------------------
def _lettre(i):
    """0 → A, 25 → Z, 26 → AA."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _indice(lettres):
    n = 0
    for c in lettres.upper():
        n = n * 26 + ord(c) - 64
    return n - 1


def _texte_cellule(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def _lire_csv(chemin):
    brut = Path(chemin).read_bytes()
    for codage in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            contenu = brut.decode(codage)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialecte = csv.Sniffer().sniff(contenu[:20000], delimiters=";,\t|")
    except csv.Error:  # une seule colonne, ou séparateur ambigu
        class dialecte(csv.excel):  # noqa: N801 - ne pas modifier csv.excel lui-même
            delimiter = ";" if contenu.count(";") > contenu.count(",") else ","
    return [[c.strip() for c in ligne] for ligne in csv.reader(io.StringIO(contenu), dialecte)]


def lire_tableau(chemin):
    """{feuille: lignes (listes de textes)}. .xlsx/.xlsm (openpyxl, valeurs calculées), .csv/.tsv, .txt (une ligne
    = une cellule)."""
    chemin = str(getattr(chemin, "name", chemin) or "")
    if not chemin or not Path(chemin).is_file():
        raise gr.Error("Fichier introuvable : dépose un fichier Excel (.xlsx), CSV ou texte.")
    ext = Path(chemin).suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        try:
            import openpyxl
        except ImportError as e:  # ajouté à requirements.txt : installé par l'étape 18 de l'installateur
            raise gr.Error("Lecture des fichiers Excel pas encore installée : relance INSTALLER.bat (ou enregistre "
                           "le tableau en CSV, lu sans rien installer).") from e
        try:
            classeur = openpyxl.load_workbook(chemin, read_only=True, data_only=True)
        except Exception as e:  # noqa: BLE001 - fichier abîmé, mot de passe, ancien format renommé…
            raise gr.Error(f"Impossible de lire ce fichier Excel ({e}). Enregistre-le en .xlsx ou en CSV.") from e
        feuilles = {}
        for feuille in classeur.worksheets:
            lignes = []
            for ligne in feuille.iter_rows(values_only=True):
                lignes.append([_texte_cellule(v) for v in ligne])
                if len(lignes) > SERIE_MAX * 5:
                    break
            feuilles[feuille.title] = lignes
        classeur.close()
        return feuilles
    if ext == ".xls":
        raise gr.Error("Ancien format Excel (.xls) : enregistre le fichier en .xlsx (Fichier → Enregistrer sous).")
    if ext == ".ods":
        raise gr.Error("Fichier LibreOffice (.ods) : enregistre-le en .xlsx ou en CSV.")
    if ext in (".csv", ".tsv"):
        return {"CSV": _lire_csv(chemin)}
    lignes = Path(chemin).read_text(encoding="utf-8-sig", errors="replace").splitlines()
    return {"Texte": [[l_.strip()] for l_ in lignes]}


def _largeur(lignes):
    return max((len(l_) for l_ in lignes), default=0)


def colonnes(lignes, entetes=True):
    """[(libellé « B — Prompt », lettre)] des colonnes non vides."""
    titres = lignes[0] if entetes and lignes else []
    choix = []
    for i in range(_largeur(lignes)):
        valeurs = [l_[i] for l_ in lignes[1 if entetes else 0:] if i < len(l_) and l_[i]]
        titre = titres[i] if i < len(titres) and titres[i] else ""
        if not valeurs and not titre:
            continue
        lettre = _lettre(i)
        choix.append((f"{lettre} — {titre[:40]}" if titre else f"{lettre} — « {valeurs[0][:40]} »", lettre))
    return choix


def _devine(lignes, entetes, mots, longue=False):
    """Lettre de la colonne dont le titre contient un des mots ; sinon (longue) celle aux textes les plus longs."""
    if entetes and lignes:
        for i, titre in enumerate(lignes[0]):
            t = _normal(titre)
            if any(m in t.split() or t.startswith(m) for m in mots):
                return _lettre(i)
    if longue:
        meilleure, score = None, 0
        for i in range(_largeur(lignes)):
            valeurs = [l_[i] for l_ in lignes[1 if entetes else 0:] if i < len(l_) and l_[i]]
            s = sum(len(v) for v in valeurs)
            if s > score:
                meilleure, score = _lettre(i), s
        return meilleure
    return None


def _normal(t):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def analyser(fichier, feuille=None, entetes=True):
    """Après le dépôt du fichier : feuilles, colonnes (avec une proposition pour les prompts et les noms), apercu_.
    Renvoie (feuille, colonne des prompts, colonne des noms, colonne de contexte, apercu_, message)."""
    if not fichier:
        vide = gr.update(choices=[], value=None)
        return (gr.update(choices=[], value=None, visible=False), vide, vide, vide, gr.update(value=None),
                "Dépose un fichier, ou colle tes prompts ci-dessous (un par ligne).")
    feuilles = lire_tableau(fichier)
    noms = list(feuilles)
    feuille = feuille if feuille in feuilles else next((n for n in noms if any(any(c for c in l_) for l_ in feuilles[n])),
                                                        noms[0])
    lignes = feuilles[feuille]
    choix = colonnes(lignes, entetes)
    if not choix:
        raise gr.Error(f"La feuille « {feuille} » est vide.")
    prompt = _devine(lignes, entetes, ["prompt", "prompts", "description", "desc", "texte", "text", "image"], True)
    nom = _devine(lignes, entetes, ["nom", "name", "id", "fichier", "file", "titre", "title", "icone", "icon"])
    nom = nom if nom != prompt else None
    contexte = _devine(lignes, entetes, ["contexte", "context"])
    contexte = contexte if contexte not in (prompt, nom) else None
    if contexte and prompt:  # une case « contexte global » seule n'est pas un contexte par ligne (constaté à l'essai)
        corps = lignes[1 if entetes else 0:]
        remplis = [l_ for l_ in corps if _indice(prompt) < len(l_) and l_[_indice(prompt)]]
        avec = [l_ for l_ in remplis if _indice(contexte) < len(l_) and l_[_indice(contexte)]]
        if len(avec) * 2 < len(remplis):
            contexte = None
    apercu_ = _apercu(lignes, entetes)
    n = sum(1 for l_ in lignes[1 if entetes else 0:] if prompt and _indice(prompt) < len(l_) and l_[_indice(prompt)])
    msg = (f"Feuille « {feuille} » : {len(lignes) - (1 if entetes else 0)} ligne(s), {len(choix)} colonne(s). "
           f"Colonne des prompts proposée : **{prompt}** ({n} case(s) remplie(s)). Vérifie les choix ci-dessous.")
    return (gr.update(choices=noms, value=feuille, visible=len(noms) > 1),
            gr.update(choices=choix, value=prompt),
            gr.update(choices=[(AUCUNE, AUCUNE)] + choix, value=nom or AUCUNE),
            gr.update(choices=[(AUCUNE, AUCUNE)] + choix, value=contexte or AUCUNE),
            gr.update(value=apercu_), msg)


def _apercu(lignes, entetes, n=8):
    largeur = min(_largeur(lignes), 12)
    if not largeur:
        return None
    titres = lignes[0] if entetes and lignes else []
    entete = [f"{_lettre(i)} {titres[i]}".strip() if i < len(titres) else _lettre(i) for i in range(largeur)]
    corps = [[(l_[i] if i < len(l_) else "")[:120] for i in range(largeur)] for l_ in lignes[1 if entetes else 0:][:n]]
    return {"headers": entete, "data": corps or [[""] * largeur]}


def valeur_case(fichier, feuille, reference):
    """Contenu d'une case (« B2 », ou « Feuille!B2 »)."""
    reference = (reference or "").strip().replace("$", "")
    if not reference:
        return ""
    feuilles = lire_tableau(fichier)
    if "!" in reference:
        feuille, reference = reference.rsplit("!", 1)
        feuille = feuille.strip("'\"")
        if feuille not in feuilles:
            raise gr.Error(f"Feuille « {feuille} » introuvable (feuilles : {', '.join(feuilles)}).")
    m = re.fullmatch(r"([A-Za-z]{1,3})(\d+)", reference.strip())
    if not m:
        raise gr.Error(f"Case « {reference} » : écris une case comme B2 (lettre de colonne puis numéro de ligne).")
    lignes = feuilles.get(feuille) or next(iter(feuilles.values()))
    i, j = int(m.group(2)) - 1, _indice(m.group(1))
    valeur = lignes[i][j] if i < len(lignes) and j < len(lignes[i]) else ""
    if not valeur:
        raise gr.Error(f"La case {reference.upper()} est vide.")
    return valeur


def _slug(t, defaut):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9_-]+", "_", t).strip("_")[:40]
    return t or defaut


def entrees(fichier=None, feuille=None, entetes=True, col_prompt=None, col_nom=None, col_contexte=None,
            liste=""):
    """Lignes à générer : [{numero, nom, prompt, contexte_ligne}] depuis le tableau (lignes dont la case du prompt
    est remplie) ou depuis la liste collée (un prompt par ligne)."""
    resultat = []
    if fichier:
        if not col_prompt:
            raise gr.Error("Choisis la colonne qui contient les prompts.")
        feuilles = lire_tableau(fichier)
        lignes = feuilles.get(feuille) or next(iter(feuilles.values()))
        ip = _indice(col_prompt)
        inom = _indice(col_nom) if col_nom and col_nom != AUCUNE else None
        ictx = _indice(col_contexte) if col_contexte and col_contexte != AUCUNE else None
        depart = 1 if entetes else 0
        for k, l_ in enumerate(lignes[depart:], depart + 1):
            p = l_[ip] if ip < len(l_) else ""
            if not p:
                continue
            nom = l_[inom] if inom is not None and inom < len(l_) else ""
            ctx = l_[ictx] if ictx is not None and ictx < len(l_) else ""
            resultat.append({"ligne": k, "nom": nom, "prompt": p, "contexte_ligne": ctx})
    else:
        for k, p in enumerate((liste or "").splitlines(), 1):
            if p.strip():
                resultat.append({"ligne": k, "nom": "", "prompt": p.strip(), "contexte_ligne": ""})
    if not resultat:
        raise gr.Error("Aucun prompt : la colonne choisie est vide, ou la liste est vide.")
    if len(resultat) > SERIE_MAX:
        raise gr.Error(f"{len(resultat)} prompts : {SERIE_MAX} au plus par lot (découpe le fichier).")
    vus = {}
    for i, e in enumerate(resultat, 1):
        base = _slug(e["nom"] or e["prompt"][:30], f"image_{i}")
        vus[base] = vus.get(base, 0) + 1
        e["numero"] = i
        e["fichier"] = f"{i:03d}_{base}" + (f"_{vus[base]}" if vus[base] > 1 else "") + ".png"
    return resultat


def prompt_image(entree, contexte, styles, avec_images_de_style=False):
    """« prompt. contexte de la ligne. contexte commun. [consigne des images de style.] Style: … » — le prompt
    d'abord : un texte trop long pour le générateur perd la fin du contexte, pas le sujet de l'image."""
    ligne = entree.get("contexte_ligne")
    if ligne and contexte and _normal(ligne) in _normal(contexte):  # même texte que le contexte commun : une fois
        ligne = None
    morceaux = [entree["prompt"], ligne, contexte, STYLE_DES_IMAGES if avec_images_de_style else None]
    corps = ". ".join(m.strip().rstrip(".") for m in morceaux if m and m.strip())
    style = texte(styles)
    return f"{corps}. Style: {style}." if style else f"{corps}."


def apercu_prompts(fichier, feuille, entetes, col_prompt, col_nom, col_contexte, case_contexte, liste, contexte,
                   styles, images_style, limite=0):
    """Tableau des prompts qui seront envoyés (pour vérifier avant de lancer 250 images)."""
    lot, contexte = _preparer(fichier, feuille, entetes, col_prompt, col_nom, col_contexte, case_contexte, liste,
                              contexte, limite)
    avec = bool(_images_de_style(images_style))
    data = [[e["numero"], e["fichier"], prompt_image(e, contexte, styles, avec)] for e in lot]
    return ({"headers": ["N°", "Fichier", "Prompt envoyé"], "data": data},
            f"{len(lot)} image(s) à générer" + (" (FLUX.2 klein, avec les images de style)." if avec
                                                 else " (Z-Image-Turbo)."))


def _preparer(fichier, feuille, entetes, col_prompt, col_nom, col_contexte, case_contexte, liste, contexte, limite):
    lot = entrees(fichier, feuille, entetes, col_prompt, col_nom, col_contexte, liste)
    limite = int(limite or 0)
    if limite > 0:
        lot = lot[:limite]
    morceaux = [(contexte or "").strip()]
    if fichier and (case_contexte or "").strip():
        morceaux.insert(0, valeur_case(fichier, feuille, case_contexte))
    return lot, ". ".join(m.rstrip(".") for m in morceaux if m)


def _images_de_style(fichiers):
    chemins = [str(getattr(f, "name", f)) for f in (fichiers or []) if f]
    if len(chemins) > STYLES_IMAGES_MAX:
        raise gr.Error(f"{len(chemins)} images de style : {STYLES_IMAGES_MAX} au plus.")
    for c in chemins:
        if not Path(c).is_file():
            raise gr.Error(f"Image introuvable : {c}")
    return chemins


# --- Génération ------------------------------------------------------------------------------------------------------
def generer(fichier, feuille, entetes, col_prompt, col_nom, col_contexte, case_contexte, liste, contexte, styles,
            images_style, format_label, tailles, graine, meme_graine, nom="", limite=0, progress=gr.Progress()):
    """Prépare le lot (lot.json) puis le génère. Renvoie (message, galerie, archive zip, dossier)."""
    lot, contexte = _preparer(fichier, feuille, entetes, col_prompt, col_nom, col_contexte, case_contexte, liste,
                              contexte, limite)
    format_label = format_label if format_label in FORMATS else FORMAT_DEFAUT
    largeur, hauteur = FORMATS[format_label]
    nom = _slug(nom, "serie")
    dossier = nouveau_dossier(cfg.SERIES_DIR)
    dossier = dossier.rename(dossier.with_name(f"{dossier.name}_{nom}"))
    references = []
    for k, c in enumerate(_images_de_style(images_style), 1):  # copiées : la reprise ne dépend pas des originaux
        references.append(dossier / f"style_{k}{Path(c).suffix.lower() or '.png'}")
        shutil.copy(c, references[-1])
    graine = int(graine or 0) or diffusion.graines(1)[0]
    for e in lot:
        e["graine"] = graine if meme_graine else graine + e["numero"] - 1
        e["prompt_final"] = prompt_image(e, contexte, styles, bool(references))
    infos = {"nom": nom, "contexte": contexte, "styles": styles, "format": format_label, "largeur": largeur,
             "hauteur": hauteur, "tailles": sorted({int(t) for t in tailles or []}, reverse=True),
             "references": [r.name for r in references], "graine": graine, "meme_graine": bool(meme_graine),
             "source": Path(str(getattr(fichier, "name", fichier))).name if fichier else "liste",
             "moteur": "FLUX.2 klein 4B" if references else "Z-Image-Turbo", "images": lot}
    (dossier / "lot.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    return _executer(dossier, progress)


def reprendre(dossier, progress=gr.Progress()):
    """Refait seulement les images manquantes d'un lot (interrompu, ou paquets en échec)."""
    if dossier and not Path(dossier).is_absolute():  # « data\series\… » tapé à la main : relatif à l'application
        dossier = cfg.APP_DIR / dossier
    if not dossier or not (Path(dossier) / "lot.json").exists():
        raise gr.Error("Aucun lot à reprendre : génère d'abord une série (ou indique le dossier du lot).")
    return _executer(Path(dossier), progress)


def _executer(dossier, progress):
    infos = json.loads((dossier / "lot.json").read_text(encoding="utf-8"))
    lot, references = infos["images"], [dossier / r for r in infos.get("references") or []]
    a_faire = [e for e in lot if not (dossier / e["fichier"]).exists()]
    total, debut, erreurs, echecs_suite = len(a_faire), time.monotonic(), [], 0
    for p0 in range(0, total, PAQUET):
        paquet = a_faire[p0:p0 + PAQUET]

        def suivi(valeur, desc="", p0=p0, n=len(paquet)):
            fait = p0 + valeur * n
            reste = ""
            if p0 and fait:
                reste = f", environ {duree_lisible((time.monotonic() - debut) / fait * (total - fait))} restantes"
            progress(fait / max(1, total), desc=f"Images {p0 + 1}–{p0 + n} sur {total}{reste} — {desc}")

        sorties = [dossier / e["fichier"] for e in paquet]
        prompts = [e["prompt_final"] for e in paquet]
        graines = [e["graine"] for e in paquet]
        try:
            if references:
                diffusion.personnage(prompts, references, sorties, graines, infos["largeur"], infos["hauteur"],
                                     progress=suivi)
            else:
                diffusion.image(prompts, sorties, graines, infos["largeur"], infos["hauteur"], progress=suivi)
            echecs_suite = 0
        except gr.Error as e:  # le paquet suivant repart d'un moteur neuf ; « Reprendre » refera celles-ci
            erreurs.append(f"images {paquet[0]['numero']}–{paquet[-1]['numero']} : {getattr(e, 'message', e)}")
            echecs_suite += 1
            if echecs_suite >= ECHECS_MAX:
                erreurs.append(f"{ECHECS_MAX} paquets en échec à la suite : lot arrêté.")
                break
        for s in sorties:
            if s.exists():
                _reduire(s, infos.get("tailles") or [])
    faites = [e for e in lot if (dossier / e["fichier"]).exists()]
    _ecrire_csv(dossier, lot)
    archive = _archive(dossier, infos, faites) if faites else None
    temps = time.monotonic() - debut
    msg = (f"✅ {len(faites)}/{len(lot)} image(s) {infos['largeur']}×{infos['hauteur']} ({infos['moteur']}) "
           f"en {duree_lisible(temps)}, dans {dossier}.")
    if infos.get("tailles"):
        msg += f" Copies réduites : {', '.join(f'{t} px' for t in infos['tailles'])}."
    if erreurs:
        msg += ("\n\n⚠️ " + "\n\n⚠️ ".join(erreurs) +
                "\n\nClique sur « Reprendre le lot » pour refaire seulement les images manquantes.")
    galerie = [(str(dossier / e["fichier"]), f"{e['numero']} · {e['nom'] or e['prompt'][:60]}") for e in faites]
    return msg, galerie, str(archive) if archive else None, str(dossier)


def _reduire(image, tailles):
    """Copies réduites (côté le plus long = taille), dans <taille>px/ à côté de l'image."""
    if not tailles:
        return
    with Image.open(image) as im:
        im.load()
        for t in tailles:
            d = image.parent / f"{int(t)}px"
            d.mkdir(exist_ok=True)
            copie = im.copy()
            copie.thumbnail((int(t), int(t)), Image.LANCZOS)
            copie.save(d / image.name, optimize=True)


def _ecrire_csv(dossier, lot):
    """lot.csv lisible par Excel en français (point-virgule, UTF-8 avec BOM)."""
    with open(dossier / "lot.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["numero", "ligne", "nom", "prompt", "graine", "fichier", "fait", "prompt_envoye"])
        for e in lot:
            w.writerow([e["numero"], e["ligne"], e["nom"], e["prompt"], e["graine"], e["fichier"],
                        "oui" if (dossier / e["fichier"]).exists() else "non", e["prompt_final"]])


def _archive(dossier, infos, faites):
    chemin = dossier / f"{infos['nom']}.zip"
    with zipfile.ZipFile(chemin, "w", zipfile.ZIP_STORED) as z:  # PNG déjà compressés
        for e in faites:
            z.write(dossier / e["fichier"], e["fichier"])
            for t in infos.get("tailles") or []:
                petite = dossier / f"{t}px" / e["fichier"]
                if petite.exists():
                    z.write(petite, f"{t}px/{e['fichier']}")
        z.write(dossier / "lot.csv", "lot.csv")
    return chemin
