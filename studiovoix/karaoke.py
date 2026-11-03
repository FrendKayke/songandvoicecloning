"""Karaoké (groupe Musique, onglet « 🎤 Karaoké ») : une chanson → une vidéo où les paroles s'allument mot à mot au
rythme du chant, avec la musique sans la voix (karaoké) ou avec elle (clip de paroles).

1. Demucs sépare la voix de la musique (la voix seule est bien mieux reconnue, la musique sans voix sert au karaoké).
2. Whisper donne le temps de chaque mot chanté (`diffusion.transcrire(..., mots=True)`).
3. Si les paroles sont connues (chanson de la galerie, ou collées), elles sont alignées sur les mots entendus
   (`aligner` : mots comparés sans accents ni ponctuation, les mots non reconnus prennent le temps laissé entre leurs
   voisins) : le texte affiché est le bon, même là où Whisper s'est trompé. Sinon, le texte de Whisper.
4. Vidéo (PyAV, `convertir._Ecrivain`) : deux lignes à la fois, chaque mot se remplit de couleur pendant qu'il est
   chanté. Aussi : paroles.lrc (temps des mots, lecteurs de karaoké), paroles.srt, la musique sans la voix.
Rangement : data/videos/<horodatage>_<nom>_karaoke/ (creation.json type `video` avec `karaoke`).
"""
import difflib
import json
import re
import time
import unicodedata
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from . import config as cfg
from . import demucs, diffusion
from .audio import charger
from .convertir import _Ecrivain, _taille, duree_lisible
from .outils import chemin_fichier, ecrire_creation, nouveau_dossier
from .serie import _slug

FREQ, FPS = 44100, 25
SANS_VOIX = "Sans la voix (karaoké, pour chanter)"
AVEC_VOIX = "Avec la voix (clip de paroles)"
SONS = [SANS_VOIX, AVEC_VOIX]
TAILLES = {"Paysage 1280×720": (1280, 720), "Paysage 1920×1080": (1920, 1080),
           "Portrait 1080×1920 (téléphone)": (1080, 1920)}
TAILLE_DEFAUT = next(iter(TAILLES))
FOND_DEFAUT, TEXTE_DEFAUT, SURLIGNE_DEFAUT = "#141428", "#ffffff", "#ffd84d"
APPARITION = 2.0  # s : une page de paroles apparaît au plus tôt 2 s avant son premier mot
TENUE = 1.0  # s : elle reste au plus 1 s après son dernier mot
CARACTERES_LIGNE = 38  # paroles venues de Whisper (sans texte connu) : longueur d'une ligne
PAUSE_LIGNE = 1.0  # s : une pause plus longue entre deux mots commence une nouvelle ligne
ACCORD_MIN = 0.3  # part des mots des paroles retrouvés dans le chant en dessous de laquelle on prévient
DUREE_MAX = 15 * 60


# --- Paroles et temps ------------------------------------------------------------------------------------------
def normal(mot):
    """Mot comparable : minuscules, sans accents ni ponctuation (« L'amour, » → « lamour »)."""
    m = unicodedata.normalize("NFKD", str(mot).lower().replace("œ", "oe").replace("æ", "ae"))
    return "".join(c for c in m if c.isalnum())


def lignes_paroles(paroles):
    """Lignes à chanter : titres de section (« Refrain », « [Verse 1] ») retirés, refrain laissé vide recopié."""
    from .acestep import baliser_paroles

    texte = baliser_paroles(paroles or "")
    lignes = []
    for ligne in texte.splitlines():
        ligne = ligne.strip()
        if not ligne or re.fullmatch(r"\[[^\]]*\]", ligne) or ligne.lower() == "[instrumental]":
            continue
        ligne = re.sub(r"\[[^\]]*\]", "", ligne).strip()  # balise collée au texte
        if ligne:
            lignes.append(ligne)
    return lignes


def aligner(lignes, mots_entendus, duree):
    """Paroles connues (une chaîne par ligne) + mots de Whisper [{debut, fin, mot}] → lignes de mots
    [[{mot, debut, fin}]] et part des mots des paroles retrouvés. Les mots non retrouvés prennent le temps laissé
    entre leurs voisins retrouvés, au prorata de leur longueur."""
    mots = [(i, m) for i, ligne in enumerate(lignes) for m in ligne.split()]
    cles = [normal(m) for _, m in mots]
    entendus = [normal(m["mot"]) for m in mots_entendus]
    temps = [None] * len(mots)
    trouves = 0
    for bloc in difflib.SequenceMatcher(None, cles, entendus, autojunk=False).get_matching_blocks():
        for k in range(bloc.size):
            if cles[bloc.a + k]:
                m = mots_entendus[bloc.b + k]
                temps[bloc.a + k] = (float(m["debut"]), max(float(m["fin"]), float(m["debut"]) + 0.05))
                trouves += 1
    # trous : temps partagé entre les voisins retrouvés
    i = 0
    while i < len(mots):
        if temps[i] is not None:
            i += 1
            continue
        j = i
        while j < len(mots) and temps[j] is None:
            j += 1
        n = j - i
        avant = temps[i - 1][1] if i else None
        apres = temps[j][0] if j < len(mots) else None
        if avant is None and apres is None:  # rien de reconnu : paroles réparties sur toute la chanson
            avant, apres = 0.0, duree
        elif avant is None:
            avant = max(0.0, apres - 0.5 * n)
        elif apres is None:
            apres = min(duree, avant + 0.5 * n) if duree > avant else avant + 0.5 * n
        longueurs = np.array([max(1, len(mots[k][1])) for k in range(i, j)], np.float64)
        bornes = avant + (apres - avant) * np.concatenate([[0], np.cumsum(longueurs) / longueurs.sum()])
        for k in range(n):
            temps[i + k] = (float(bornes[k]), float(max(bornes[k + 1], bornes[k] + 0.02)))
        i = j
    resultat = [[] for _ in lignes]
    for (num, mot), (debut, fin) in zip(mots, temps):
        resultat[num].append({"mot": mot, "debut": round(debut, 3), "fin": round(fin, 3)})
    total = sum(1 for c in cles if c)
    return [l for l in resultat if l], (trouves / total if total else 0.0)


def lignes_entendues(mots_entendus):
    """Sans paroles connues : les mots de Whisper regroupés en lignes (pause, ponctuation de fin, longueur)."""
    lignes, ligne = [], []
    for m in mots_entendus:
        mot = str(m["mot"]).strip()
        if not mot:
            continue
        if ligne and (float(m["debut"]) - ligne[-1]["fin"] > PAUSE_LIGNE
                      or len(" ".join(x["mot"] for x in ligne)) + 1 + len(mot) > CARACTERES_LIGNE):
            lignes.append(ligne)
            ligne = []
        ligne.append({"mot": mot, "debut": float(m["debut"]), "fin": max(float(m["fin"]), float(m["debut"]) + 0.05)})
        if mot[-1] in ".!?…":
            lignes.append(ligne)
            ligne = []
    if ligne:
        lignes.append(ligne)
    return lignes


def pages(lignes, par_page=2):
    """Lignes deux par deux, avec leur temps d'affichage [debut, fin[ sans chevauchement."""
    groupes = [lignes[i:i + par_page] for i in range(0, len(lignes), par_page)]
    resultat, fin_prec = [], 0.0
    for k, g in enumerate(groupes):
        premier, dernier = g[0][0]["debut"], g[-1][-1]["fin"]
        debut = max(fin_prec, premier - APPARITION)
        suivant = groupes[k + 1][0][0]["debut"] if k + 1 < len(groupes) else None
        fin = dernier + TENUE if suivant is None else min(dernier + TENUE, max(dernier + 0.1, suivant - 1.0))
        resultat.append({"lignes": g, "debut": debut, "fin": max(fin, debut + 0.1)})
        fin_prec = resultat[-1]["fin"]
    return resultat


def _horodatage(s, lrc=False):
    if lrc:
        return f"{int(s // 60):02d}:{s % 60:05.2f}"
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def lrc(lignes, titre=""):
    """LRC « amélioré » : temps de la ligne puis de chaque mot (<mm:ss.xx>), lu par les lecteurs de karaoké."""
    tete = [f"[ti:{titre}]"] if titre else []
    return "\n".join(tete + [f"[{_horodatage(l[0]['debut'], True)}]"
                             + " ".join(f"<{_horodatage(m['debut'], True)}>{m['mot']}" for m in l)
                             + f" <{_horodatage(l[-1]['fin'], True)}>" for l in lignes]) + "\n"


def srt(lignes):
    return "\n".join(f"{i}\n{_horodatage(l[0]['debut'])} --> {_horodatage(l[-1]['fin'])}\n"
                     + " ".join(m["mot"] for m in l) + "\n" for i, l in enumerate(lignes, 1))


# --- Images ----------------------------------------------------------------------------------------------------
def _police(nom, taille):
    from .montage import POLICES
    from .sous_titres import _police as police_systeme

    fichier = POLICES.get(nom)
    if fichier and (cfg.APP_DIR / "polices" / fichier).is_file():
        from PIL import ImageFont

        f = ImageFont.truetype(str(cfg.APP_DIR / "polices" / fichier), taille)
        try:
            f.set_variation_by_axes([700])
        except (OSError, AttributeError, ValueError):
            pass
        return f
    return police_systeme(taille)


def _couleur(valeur, defaut):
    from .fond_video import _couleur as lire

    try:
        return tuple(int(lire(valeur)[k:k + 2], 16) for k in (1, 3, 5))
    except gr.Error:
        return tuple(int(defaut[k:k + 2], 16) for k in (1, 3, 5))


def _image_ligne(ligne, largeur_max, taille, police_nom, couleur_texte, couleur_surligne):
    """(texte normal RGBA, texte allumé RGBA, [(x0, x1, debut, fin)] de chaque mot) d'une ligne, réduite si trop large."""
    from PIL import Image, ImageDraw

    texte = " ".join(m["mot"] for m in ligne)
    while True:
        f = _police(police_nom, taille)
        contour = max(2, taille // 12)
        if f.getlength(texte) + 2 * contour <= largeur_max or taille <= 14:
            break
        taille = int(taille * 0.92)
    bas = f.getbbox("Ày")[3]
    w, h = int(f.getlength(texte)) + 4 * contour, int(bas) + 4 * contour
    images = []
    for couleur in (couleur_texte, couleur_surligne):
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((2 * contour, contour), texte, font=f, fill=couleur + (255,), stroke_width=contour,
                                stroke_fill=(0, 0, 0, 255))
        images.append(np.asarray(im))
    spans, x = [], 2 * contour
    for k, m in enumerate(ligne):
        debut_x = x + f.getlength(" ".join(mm["mot"] for mm in ligne[:k]) + (" " if k else ""))
        spans.append((debut_x, debut_x + f.getlength(m["mot"]), m["debut"], m["fin"]))
    return images[0], images[1], spans


def _remplissage(spans, t):
    """Abscisse jusqu'où la ligne est allumée au temps t (le mot en cours se remplit au fil de sa durée)."""
    x = 0.0
    for x0, x1, debut, fin in spans:
        if t >= fin:
            x = x1
        elif t > debut:
            return x0 + (x1 - x0) * (t - debut) / max(fin - debut, 1e-3)
        else:
            return x
    return x


def _poser(fond, image, x, y):
    h, w = image.shape[:2]
    zone = fond[y:y + h, x:x + w].astype(np.float32)
    a = image[: zone.shape[0], : zone.shape[1], 3:4].astype(np.float32) / 255
    fond[y:y + h, x:x + w] = (zone * (1 - a) + image[: zone.shape[0], : zone.shape[1], :3] * a + 0.5).astype(np.uint8)


def _fond(largeur, hauteur, couleur, image_fond):
    from PIL import Image, ImageOps

    if image_fond:
        with Image.open(image_fond) as im:
            im = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (largeur, hauteur), Image.LANCZOS)
        return (np.asarray(im, np.float32) * 0.45).astype(np.uint8)  # assombrie : les paroles restent lisibles
    haut = np.array(couleur, np.float32)
    bas = haut * 0.35
    degrade = np.linspace(0, 1, hauteur, dtype=np.float32)[:, None, None]
    return np.broadcast_to(haut * (1 - degrade) + bas * degrade, (hauteur, largeur, 3)).astype(np.uint8).copy()


def video(lignes, son, sortie, largeur, hauteur, titre="", couleurs=(FOND_DEFAUT, TEXTE_DEFAUT, SURLIGNE_DEFAUT),
          image_fond=None, police_nom=None, progress=None):
    """Écrit la vidéo : son (2, n) à FREQ, paroles deux lignes à la fois, mots allumés au fil du chant."""
    fond_rgb, texte_rgb, surligne_rgb = (_couleur(c, d) for c, d in
                                         zip(couleurs, (FOND_DEFAUT, TEXTE_DEFAUT, SURLIGNE_DEFAUT)))
    fond = _fond(largeur, hauteur, fond_rgb, image_fond)
    taille = max(18, min(largeur, hauteur) // 13)
    rendus = [_image_ligne(l, int(largeur * 0.92), taille, police_nom, texte_rgb, surligne_rgb) for l in lignes]
    index = {id(l): i for i, l in enumerate(lignes)}
    lesPages = pages(lignes)
    titre_img = None
    if titre:
        from .montage import _calque_texte

        titre_img = _calque_texte(titre, largeur, hauteur, "Au centre", police_nom, 8)
    n_images = int(np.ceil(son.shape[1] / FREQ * FPS))
    ec = _Ecrivain(sortie, "libx264", "aac", FREQ, largeur, hauteur, FPS, crf=20)
    p = 0
    for k in range(n_images):
        t = k / FPS
        image = fond.copy()
        while p < len(lesPages) and t >= lesPages[p]["fin"]:
            p += 1
        page = lesPages[p] if p < len(lesPages) and t >= lesPages[p]["debut"] else None
        if page is None and titre_img is not None and (not lesPages or t < lesPages[0]["debut"]):
            x, y, rgba = titre_img
            _poser(image, rgba, x, y)
        if page is not None:
            n = len(page["lignes"])
            for r, ligne in enumerate(page["lignes"]):
                base, allume, spans = rendus[index[id(ligne)]]
                remplie = base.shape[1] if t >= spans[-1][3] else int(round(_remplissage(spans, t)))
                ligne_img = base.copy()
                if remplie > 0:
                    ligne_img[:, :remplie] = allume[:, :remplie]
                h, w = ligne_img.shape[:2]
                y_centre = hauteur * (0.5 + (r - (n - 1) / 2) * 0.16)
                _poser(image, ligne_img, (largeur - w) // 2, int(y_centre - h / 2))
        ec.image(image, t)
        a0, a1 = int(round(t * FREQ)), int(round((k + 1) / FPS * FREQ))
        ec.son(son[:, a0:a1], a0 / FREQ)
        if progress and k % 50 == 0:
            progress(0.7 + 0.3 * k / n_images, desc=f"Vidéo : {duree_lisible(t)} / {duree_lisible(n_images / FPS)}")
    ec.fermer()
    return sortie


# --- Création ---------------------------------------------------------------------------------------------------
def chanson_de(dossier):
    """(fichier, paroles, langue) d'une chanson de la galerie."""
    from .galerie import lire

    d = lire(dossier) if dossier else None
    if not d:
        return None, "", None
    fichier = (d.get("versions") or [{}])[0].get("fichier")
    paroles = d.get("paroles") or ""
    return fichier, "" if paroles.strip() == "[Instrumental]" else paroles, d.get("langue")


def creer(fichier, creation=None, paroles="", langue_label=None, son_label=SANS_VOIX, taille_label=TAILLE_DEFAUT,
          couleur_fond=FOND_DEFAUT, couleur_texte=TEXTE_DEFAUT, couleur_surligne=SURLIGNE_DEFAUT, image_fond=None,
          police_nom=None, titre="", nom="karaoke", progress=gr.Progress()):
    """Renvoie (message, vidéo, [fichiers], dossier)."""
    from .sous_titres import LANGUES

    source = chemin_fichier(fichier)
    if not source and creation:
        source, paroles_galerie, langue_galerie = chanson_de(creation)
        paroles = paroles or paroles_galerie
        langue_label = langue_label or langue_galerie
    if not source or not Path(str(source)).is_file():
        raise gr.Error("Ajoute une chanson (ou choisis-en une de la galerie).")
    source = Path(str(source))
    debut = time.monotonic()
    progress(0.02, desc="Lecture de la chanson…")
    try:
        y, _ = charger(str(source), sr=FREQ, mono=False)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"Impossible de lire « {source.name} » ({e}).") from e
    y = np.atleast_2d(np.asarray(y, np.float32))
    y = np.vstack([y, y]) if y.shape[0] == 1 else y[:2]
    duree = y.shape[1] / FREQ
    if duree > DUREE_MAX:
        raise gr.Error(f"Chanson de {duree_lisible(duree)} : {duree_lisible(DUREE_MAX)} au plus.")
    base = _slug(nom or source.stem, "karaoke")
    dossier = nouveau_dossier(cfg.VIDEOS_DIR, base if "karaoke" in base else f"{base}_karaoke")
    original = dossier / "chanson.wav"
    sf.write(str(original), y.T, FREQ)
    progress(0.05, desc="Séparation de la voix et de la musique (Demucs)…")
    voix, musique = demucs.separate_vocals(original, dossier)
    v, _ = charger(str(voix), sr=16000, mono=True)
    wav = dossier / "voix_16k.wav"
    sf.write(str(wav), v, 16000, subtype="PCM_16")
    progress(0.35, desc="Reconnaissance des mots chantés (Whisper)…")
    r = diffusion.transcrire(wav, LANGUES.get(langue_label), progress=progress, mots=True)
    entendus = r.get("mots") or []
    connues = lignes_paroles(paroles)
    accord = None
    if connues:
        lignes, accord = aligner(connues, entendus, duree)
    else:
        if not entendus:
            raise gr.Error("Aucune parole reconnue dans cette chanson (instrumental ?). Colle les paroles pour les "
                           "afficher quand même.")
        lignes = lignes_entendues(entendus)
    (dossier / "paroles_alignees.json").write_text(json.dumps(lignes, ensure_ascii=False, indent=1), encoding="utf-8")
    titre = (titre or "").strip()
    (dossier / "paroles.lrc").write_text(lrc(lignes, titre), encoding="utf-8")
    (dossier / "paroles.srt").write_text(srt(lignes), encoding="utf-8-sig")
    m, _ = charger(str(musique), sr=FREQ, mono=False)
    m = np.atleast_2d(np.asarray(m, np.float32))
    m = np.vstack([m, m]) if m.shape[0] == 1 else m[:2]
    sans_voix = dossier / "sans_voix.wav"
    sf.write(str(sans_voix), m.T, FREQ)
    son = m if son_label == SANS_VOIX else y
    largeur, hauteur = TAILLES.get(taille_label, TAILLES[TAILLE_DEFAUT])
    progress(0.7, desc="Vidéo des paroles…")
    sortie = video(lignes, son, dossier / "karaoke.mp4", largeur, hauteur, titre,
                   (couleur_fond, couleur_texte, couleur_surligne), chemin_fichier(image_fond), police_nom, progress)
    for f in (original, wav):
        f.unlink(missing_ok=True)
    temps = time.monotonic() - debut
    ecrire_creation(dossier, {
        "type": "video", "nom": nom, "description": f"karaoké : {titre or source.stem}",
        "karaoke": {"chanson": source.name, "son": son_label, "paroles_connues": bool(connues),
                    "accord": round(accord, 2) if accord is not None else None, "lignes": len(lignes)},
        "duree": round(duree, 2), "largeur": largeur, "hauteur": hauteur, "temps_s": round(temps),
        "versions": [{"graine": 0, "dossier": ".", "fichier": str(sortie)}],
    })
    msg = (f"✅ Karaoké de {duree_lisible(duree)} ({len(lignes)} lignes) en {duree_lisible(temps)} : `{sortie}` "
           f"({_taille(sortie.stat().st_size)}). Aussi : paroles.lrc (temps de chaque mot), paroles.srt et la musique "
           "sans la voix.")
    if accord is not None and accord < ACCORD_MIN:
        msg += (f"\n\n⚠️ Seulement {accord:.0%} des mots des paroles ont été entendus dans la chanson : vérifie que ce "
                "sont les bonnes paroles et la bonne langue ; les mots non entendus sont placés au jugé.")
    elif not connues:
        msg += "\n\nParoles reconnues par Whisper : colle les vraies paroles pour corriger les mots mal compris."
    fichiers = [str(sortie), str(dossier / "paroles.lrc"), str(dossier / "paroles.srt"), str(sans_voix)]
    return msg, str(sortie), fichiers, str(dossier)
