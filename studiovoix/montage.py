"""Montage vidéo simple (groupe « 🔧 Audio / vidéo », onglet « 🎬 Montage ») : des vidéos et des images mises bout à
bout avec des fondus, une musique de fond, un titre et un logo. Aucun modèle : PyAV (décodage, H.264 + AAC par
`convertir._Ecrivain`), numpy et Pillow.

Le temps est compté en images de la sortie : chaque morceau occupe une plage d'images [debut, debut + n[ ; un fondu
enchaîné fait se chevaucher deux morceaux de K images (mélange image par image, son croisé), un fondu au noir fait
descendre puis remonter la lumière sans chevauchement. Le son est préparé en entier dans un tableau (son des vidéos +
musique répétée jusqu'à la fin, baissée quand la vidéo a du son si on le demande), puis écrit au fil des images pour
que le fichier reste entrelacé. Titre et logo sont dessinés une fois (Pillow, RGBA) et posés dans leur seul rectangle.
Rangement : data/videos/<horodatage>_<nom>/ (montage.mp4, creation.json type `video` avec `montage` : galerie, voix off).
"""
import time
from pathlib import Path

import gradio as gr
import numpy as np

from . import config as cfg
from .convertir import _Ecrivain, _taille, _taille_sortie, duree_lisible, infos
from .outils import chemin_fichier, ecrire_creation, nouveau_dossier
from .serie import _slug

FREQ = 44100
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MORCEAUX_MAX = 60
DUREE_MAX = 30 * 60  # s : le son est préparé en mémoire (30 min stéréo flottant ≈ 640 Mo)

COUPE, ENCHAINE, NOIR = "Coupe franche", "Fondu enchaîné", "Fondu au noir"
TRANSITIONS = [ENCHAINE, NOIR, COUPE]
D_APRES = "D'après la première vidéo"
TAILLES = {
    D_APRES: None,
    "Paysage 1920×1080 (YouTube)": (1920, 1080),
    "Paysage 1280×720 (plus léger)": (1280, 720),
    "Portrait 1080×1920 (TikTok, Shorts, Reels)": (1080, 1920),
    "Carré 1080×1080 (Instagram)": (1080, 1080),
}
BANDES, REMPLIR = "Bandes noires (tout est visible)", "Recadrer (remplir l'écran)"
CADRAGES = [BANDES, REMPLIR]
POSITIONS_TITRE = ["Au centre", "En bas", "En haut"]
POSITIONS_LOGO = ["En haut à droite", "En haut à gauche", "En bas à droite", "En bas à gauche"]
POLICES = {"Simple (sans empattement)": None, "Élégante (Cinzel)": "Cinzel.ttf",
           "Livre (EB Garamond)": "EBGaramond.ttf"}
FONDU_TITRE = 0.5  # s
FONDU_MUSIQUE_DEBUT, FONDU_MUSIQUE_FIN = 1.0, 2.0  # s
MUSIQUE_SOUS_VOIX = 0.3  # gain de la musique pendant que la vidéo a du son (« Baisser la musique… »)
SEUIL_SON = 0.02  # RMS sur 0,1 s au-delà duquel la vidéo « a du son »


def _est_image(chemin):
    """Image fixe (un GIF animé est une vidéo)."""
    ext = Path(chemin).suffix.lower()
    return ext in IMAGES and not (ext == ".gif" and _gif_anime(chemin))


def _gif_anime(chemin):
    from PIL import Image

    try:
        with Image.open(chemin) as im:
            return getattr(im, "n_frames", 1) > 1
    except OSError:
        return False


def _ouvrir_image(chemin):
    from PIL import Image, ImageOps

    try:
        with Image.open(chemin) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                fond = Image.new("RGBA", im.size, (0, 0, 0, 255))
                im = Image.alpha_composite(fond, im)
            return np.asarray(im.convert("RGB"))
    except OSError as e:
        raise gr.Error(f"« {Path(chemin).name} » n'est pas une image lisible ({e}).") from e


def cadrer(rgb, largeur, hauteur, mode=BANDES):
    """Image à la taille de sortie : redimensionnée si elle a la même forme, sinon bandes noires ou recadrage."""
    h, w = rgb.shape[:2]
    if (w, h) == (largeur, hauteur):
        return rgb
    from PIL import Image, ImageOps

    im = Image.fromarray(rgb)
    if abs(w / h - largeur / hauteur) < 0.01:
        im = im.resize((largeur, hauteur), Image.Resampling.LANCZOS)
    elif mode == REMPLIR:
        im = ImageOps.fit(im, (largeur, hauteur), Image.Resampling.LANCZOS)
    else:
        im = ImageOps.pad(im, (largeur, hauteur), Image.Resampling.LANCZOS, color=(0, 0, 0))
    return np.asarray(im)


def _images_video(chemin, n, fps, largeur, hauteur, mode):
    """Exactement n images de la vidéo, sur la grille de la sortie (image répétée ou sautée selon les cadences ; la
    dernière répétée si la vidéo est plus courte)."""
    import av

    precedente, image = None, None  # dernière trame décodée et son image à la taille de sortie
    j = 0
    with av.open(str(chemin)) as c:
        flux = next((s for s in c.streams.video if s.codec_context.width), None)
        flux.thread_type = "AUTO"
        t0 = None
        for trame in c.decode(flux):
            if trame.time is None:
                continue
            if t0 is None:
                t0 = trame.time
            # images de la sortie qui tombent avant cette trame : la trame précédente (convertie une seule fois)
            while j < n and j / fps < trame.time - t0 - 1e-6:
                if image is None:
                    image = cadrer((precedente or trame).to_ndarray(format="rgb24"), largeur, hauteur, mode)
                yield image
                j += 1
            if j >= n:
                return
            precedente, image = trame, None
    if precedente is not None and image is None:
        image = cadrer(precedente.to_ndarray(format="rgb24"), largeur, hauteur, mode)
    if image is None:
        image = np.zeros((hauteur, largeur, 3), np.uint8)
    while j < n:
        yield image
        j += 1


def _images_ken_burns(rgb, n, largeur, hauteur, amplitude, numero):
    """Image fixe animée (effet « Ken Burns ») : zoom avant, zoom arrière, travelling vers la droite puis vers la
    gauche, à tour de rôle selon le numéro de l'image ; mouvement adouci au début et à la fin. L'image remplit
    l'écran (recadrée). Rendu sous-pixel (`resize` avec une boîte en réels) : pas de saccade."""
    from PIL import Image, ImageOps

    s = 1.0 + amplitude
    w, h = int(round(largeur * s)), int(round(hauteur * s))
    base = ImageOps.fit(Image.fromarray(rgb), (w, h), Image.Resampling.LANCZOS)
    mouvement = numero % 4
    for j in range(n):
        p = j / max(n - 1, 1)
        p = p * p * (3 - 2 * p)
        if mouvement == 0:
            z, decalage = 1.0 + amplitude * p, 0.0
        elif mouvement == 1:
            z, decalage = s - amplitude * p, 0.0
        else:
            z, decalage = s, (2 * p - 1) * (1 if mouvement == 2 else -1)
        cw, ch = w / z, h / z
        cx = w / 2 + decalage * (w - cw) / 2
        cy = h / 2
        boite = (max(0.0, cx - cw / 2), max(0.0, cy - ch / 2), min(w, cx + cw / 2), min(h, cy + ch / 2))
        yield np.asarray(base.resize((largeur, hauteur), Image.Resampling.BILINEAR, box=boite))


def _images_fixes(rgb, n):
    for _ in range(n):
        yield rgb


def _stereo(y):
    y = np.atleast_2d(np.asarray(y, np.float32))
    return np.vstack([y, y]) if y.shape[0] == 1 else y[:2]


def _rampe(n, montee=True):
    r = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32) + 1.0 / max(n, 1) if n else np.zeros(0, np.float32)
    return r if montee else r[::-1]


def _calque_texte(texte, largeur, hauteur, position, police_nom, taille_pct):
    """Titre en RGBA (texte blanc cerné de noir, ombre douce) : (x, y, rgba) du rectangle à poser."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    from .sous_titres import _police

    def police(taille):
        fichier = POLICES.get(police_nom)
        if fichier and (cfg.APP_DIR / "polices" / fichier).is_file():
            f = ImageFont.truetype(str(cfg.APP_DIR / "polices" / fichier), taille)
            try:
                f.set_variation_by_axes([700])  # polices variables : gras
            except (OSError, AttributeError, ValueError):
                pass
            return f
        return _police(taille)

    taille = max(14, int(min(largeur, hauteur) * (taille_pct or 9) / 100))
    while True:
        f = police(taille)
        contour = max(2, taille // 14)
        boite = ImageDraw.Draw(Image.new("L", (1, 1))).multiline_textbbox(
            (0, 0), texte, font=f, align="center", stroke_width=contour, spacing=taille // 4)
        if boite[2] - boite[0] <= largeur * 0.9 or taille <= 14:
            break
        taille = int(taille * 0.9)
    marge = contour * 4
    w, h = int(boite[2] - boite[0]) + 2 * marge, int(boite[3] - boite[1]) + 2 * marge
    calque = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ombre = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(ombre).multiline_text((marge - boite[0] + contour, marge - boite[1] + contour), texte, font=f,
                                         fill=(0, 0, 0, 160), align="center", stroke_width=contour,
                                         stroke_fill=(0, 0, 0, 160), spacing=taille // 4)
    calque = Image.alpha_composite(calque, ombre.filter(ImageFilter.GaussianBlur(contour * 1.5)))
    ImageDraw.Draw(calque).multiline_text((marge - boite[0], marge - boite[1]), texte, font=f, fill=(255, 255, 255, 255),
                                          align="center", stroke_width=contour, stroke_fill=(0, 0, 0, 255),
                                          spacing=taille // 4)
    w, h = min(w, largeur), min(h, hauteur)
    calque = calque.crop((0, 0, w, h)) if calque.size != (w, h) else calque
    x = (largeur - w) // 2
    y = {"En haut": hauteur // 12, "En bas": hauteur - h - hauteur // 12}.get(position, (hauteur - h) // 2)
    return x, max(0, y), np.asarray(calque)


def _calque_logo(chemin, largeur, hauteur, position, taille_pct, opacite_pct):
    from PIL import Image, ImageOps

    try:
        with Image.open(chemin) as im:
            im = ImageOps.exif_transpose(im).convert("RGBA")
    except OSError as e:
        raise gr.Error(f"Le logo « {Path(chemin).name} » n'est pas une image lisible ({e}).") from e
    w = max(8, int(largeur * (taille_pct or 12) / 100))
    h = max(8, round(im.height * w / im.width))
    if h > hauteur // 2:
        h = hauteur // 2
        w = max(8, round(im.width * h / im.height))
    im = im.resize((w, h), Image.Resampling.LANCZOS)
    a = np.asarray(im).copy()
    a[..., 3] = (a[..., 3].astype(np.float32) * (opacite_pct if opacite_pct is not None else 90) / 100).astype(np.uint8)
    marge = int(min(largeur, hauteur) * 0.03)
    x = marge if "gauche" in position else largeur - w - marge
    y = marge if "haut" in position else hauteur - h - marge
    return x, y, a


def _poser(image, calque, force=1.0):
    """Calque (x, y, rgba) posé sur l'image (copie si besoin), avec une force de 0 à 1 (fondu du titre)."""
    x, y, rgba = calque
    h, w = rgba.shape[:2]
    if force <= 0:
        return image
    image = image if image.flags.writeable else image.copy()
    zone = image[y:y + h, x:x + w].astype(np.float32)
    a = rgba[: zone.shape[0], : zone.shape[1], 3:4].astype(np.float32) / 255.0 * force
    image[y:y + h, x:x + w] = (zone * (1 - a) + rgba[: zone.shape[0], : zone.shape[1], :3] * a).astype(np.uint8)
    return image


def _musique(chemin, n, volume_pct):
    """Musique répétée (raccord fondu de 0,5 s) jusqu'à n échantillons, montée au début et descente à la fin."""
    from .audio import charger

    y, _ = charger(str(chemin), sr=FREQ, mono=False)
    y = _stereo(y)
    if y.shape[1] < FREQ // 2:
        raise gr.Error("La musique de fond est trop courte (moins d'une demi-seconde).")
    raccord = min(FREQ // 2, y.shape[1] // 4)
    piste = y.copy()
    while piste.shape[1] < n:
        r = _rampe(raccord)
        fin = piste[:, -raccord:] * r[::-1] + y[:, :raccord] * r
        piste = np.concatenate([piste[:, :-raccord], fin, y[:, raccord:]], axis=1)
    piste = piste[:, :n] * ((volume_pct if volume_pct is not None else 40) / 100.0)
    for duree, montee in ((FONDU_MUSIQUE_DEBUT, True), (FONDU_MUSIQUE_FIN, False)):
        k = min(int(duree * FREQ), n // 2)
        if k:
            if montee:
                piste[:, :k] *= _rampe(k)
            else:
                piste[:, -k:] *= _rampe(k, False)
    return piste


def _enveloppe_son(son):
    """Gain de la musique (1 ou MUSIQUE_SOUS_VOIX) d'après le son des vidéos, lissé sur 0,3 s."""
    pas = FREQ // 10
    n = son.shape[1]
    blocs = -(-n // pas)
    mono = np.pad(np.abs(son).mean(axis=0), (0, blocs * pas - n))
    rms = np.sqrt((mono.reshape(blocs, pas) ** 2).mean(axis=1))
    actif = (rms > SEUIL_SON).astype(np.float32)
    # garde la musique basse 0,3 s après la dernière parole (pas de pompage entre deux mots)
    actif = np.maximum.reduce([np.roll(actif, k) * (np.arange(blocs) >= k) for k in range(4)])
    gain_blocs = 1.0 - (1.0 - MUSIQUE_SOUS_VOIX) * actif
    noyau = np.ones(3, np.float32) / 3
    gain_blocs = np.convolve(np.pad(gain_blocs, 1, mode="edge"), noyau, mode="valid")
    return np.interp(np.arange(n), np.arange(blocs) * pas + pas / 2, gain_blocs).astype(np.float32)


def _sources(fichiers, creations):
    """Fichiers importés puis vidéos choisies dans la galerie, dans l'ordre."""
    from .galerie import lire

    chemins = [Path(str(chemin_fichier(f))) for f in (fichiers or []) if f and chemin_fichier(f)]
    for dossier in creations or []:
        d = lire(dossier) or {}
        fichier = (d.get("versions") or [{}])[0].get("fichier")
        if fichier and Path(fichier).is_file():
            chemins.append(Path(fichier))
    chemins = [c for c in chemins if c.is_file()]
    if not chemins:
        raise gr.Error("Ajoute des vidéos ou des images à monter (ou choisis des vidéos créées ici).")
    if len(chemins) > MORCEAUX_MAX:
        raise gr.Error(f"{len(chemins)} morceaux : {MORCEAUX_MAX} au plus.")
    return chemins


def monter(fichiers, creations=None, transition=ENCHAINE, duree_transition=0.8, duree_image=4.0,
           fondu_debut_fin=True, taille=D_APRES, cadrage=BANDES, son_videos=100, musique=None, volume_musique=40,
           baisser_musique=True, titre="", duree_titre=4.0, position_titre="Au centre", police_titre=None,
           taille_titre=9, logo=None, position_logo="En haut à droite", taille_logo=12, opacite_logo=90,
           nom="montage", ken_burns=0.0, voix=None, debut_voix=1.0, adapter_a_la_voix=True, infos_en_plus=None,
           progress=gr.Progress()):
    """Le montage en MP4 (H.264 + AAC). ken_burns : amplitude du mouvement des images fixes (0 = immobiles,
    0,15 = zoom de 15 %). voix : fichier d'une voix off posée à `debut_voix` s ; les images s'allongent pour qu'elle
    tienne si `adapter_a_la_voix` (images seules). Renvoie (message, vidéo, [fichiers], dossier)."""
    from .audio import charger

    t_debut = time.monotonic()
    chemins = _sources(fichiers, creations)
    progress(0, desc="Lecture des morceaux…")
    morceaux = []
    for c in chemins:
        if _est_image(c):
            morceaux.append({"chemin": c, "image": _ouvrir_image(c), "duree": float(duree_image or 4)})
        else:
            d = infos(c)
            if not d["video"]:
                raise gr.Error(f"« {c.name} » n'a pas d'image : mets-le plutôt en musique de fond.")
            morceaux.append({"chemin": c, "infos": d, "duree": d.get("duree_video") or d["duree"]})
    videos = [m for m in morceaux if "infos" in m]
    voix_y = None
    chemin_voix = chemin_fichier(voix)
    if chemin_voix:
        voix_y, _ = charger(str(chemin_voix), sr=FREQ, mono=False)
        voix_y = _stereo(voix_y)
        besoin = float(debut_voix or 0) + voix_y.shape[1] / FREQ + 1.0
        fixes = [m for m in morceaux if "image" in m]
        if adapter_a_la_voix and fixes:
            chevauchement = float(duree_transition or 0) * (len(morceaux) - 1) if transition == ENCHAINE else 0.0
            reste = besoin + chevauchement - sum(m["duree"] for m in morceaux if "infos" in m)
            par_image = reste / len(fixes)
            for m in fixes:
                m["duree"] = max(m["duree"], par_image)

    # taille et cadence de sortie
    if TAILLES.get(taille):
        largeur, hauteur = TAILLES[taille]
    else:
        ref = videos[0]["infos"] if videos else None
        lw, lh = (ref["largeur"], ref["hauteur"]) if ref else morceaux[0]["image"].shape[1::-1]
        largeur, hauteur = _taille_sortie(lw, lh, 1080 if lh <= lw else 1920, 1920 if lh <= lw else 1080)
    fps = min(60.0, round(videos[0]["infos"]["fps"], 3)) if videos else 30.0

    # plages d'images : début et nombre d'images de chaque morceau
    for m in morceaux:
        m["n"] = max(1, int(round(m["duree"] * fps)))
    plus_court = min(m["n"] for m in morceaux)
    if transition == COUPE or len(morceaux) == 1:
        k_trans = 0
    else:
        k_trans = int(round(max(0.0, float(duree_transition or 0)) * fps))
        k_trans = min(k_trans, plus_court // 2 if transition == ENCHAINE else plus_court)
    position = 0
    for i, m in enumerate(morceaux):
        if transition == ENCHAINE and i:
            position -= k_trans
        m["debut"] = position
        position += m["n"]
    n_total = position
    duree = n_total / fps
    if duree > DUREE_MAX:
        raise gr.Error(f"Montage de {duree_lisible(duree)} : {duree_lisible(DUREE_MAX)} au plus.")
    k_global = int(round(min(1.0, duree / 4) * fps)) if fondu_debut_fin else 0
    demi = (k_trans + 1) // 2 if transition == NOIR else 0  # fondu au noir : moitié sortie, moitié entrée

    def gain_morceau(i, j):
        """Lumière (et volume) du morceau i à son image j : fondus au noir et fondus du début et de la fin."""
        m, g = morceaux[i], 1.0
        if demi:
            if i and j < demi:
                g = min(g, j / demi)
            if i < len(morceaux) - 1 and j >= m["n"] - demi:
                g = min(g, (m["n"] - 1 - j) / demi)
        k = m["debut"] + j
        return min(g, fondu_global(k))

    def fondu_global(k):
        """Départ du noir et fin au noir (première et dernière images noires)."""
        return min(1.0, k / k_global, (n_total - 1 - k) / k_global) if k_global else 1.0

    # son : vidéos (avec leurs fondus) + musique
    progress(0.02, desc="Préparation du son…")
    n_son = int(round(duree * FREQ))
    piste_videos = np.zeros((2, n_son), np.float32)
    avec_son_video = False
    for i, m in enumerate(morceaux):
        if "infos" not in m or not m["infos"]["son"] or not son_videos:
            continue
        y, _ = charger(str(m["chemin"]), sr=FREQ, mono=False)
        a0 = int(round(m["debut"] / fps * FREQ))
        longueur = min(int(round(m["n"] / fps * FREQ)), n_son - a0)
        y = _stereo(y)[:, :longueur]
        if y.shape[1] < longueur:
            y = np.pad(y, ((0, 0), (0, longueur - y.shape[1])))
        gains = np.ones(longueur, np.float32)
        if transition == ENCHAINE and k_trans:
            kf = min(longueur // 2, int(round(k_trans / fps * FREQ)))
            if i:
                gains[:kf] *= _rampe(kf)
            if i < len(morceaux) - 1:
                gains[-kf:] *= _rampe(kf, False)
        images_gain = np.array([gain_morceau(i, j) for j in range(m["n"])], np.float32)
        gains *= np.interp(np.arange(longueur), (np.arange(m["n"]) + 0.5) / fps * FREQ, images_gain).astype(np.float32)
        piste_videos[:, a0:a0 + longueur] += y * gains * (son_videos / 100.0)
        avec_son_video = True
    if voix_y is not None:
        a0 = int(round(float(debut_voix or 0) * FREQ))
        longueur = max(0, min(voix_y.shape[1], n_son - a0))
        piste_videos[:, a0:a0 + longueur] += voix_y[:, :longueur]
        avec_son_video = True
    piste = piste_videos
    chemin_musique = chemin_fichier(musique)
    if chemin_musique:
        progress(0.04, desc="Musique de fond…")
        fond = _musique(chemin_musique, n_son, volume_musique)
        if baisser_musique and avec_son_video:
            fond = fond * _enveloppe_son(piste_videos)
        piste = piste_videos + fond
    avec_son = avec_son_video or bool(chemin_musique)
    pic = float(np.abs(piste).max()) if piste.size else 0.0
    if pic > 0.98:
        piste *= 0.98 / pic

    # calques
    calque_titre = (_calque_texte(titre.strip(), largeur, hauteur, position_titre, police_titre, taille_titre)
                    if (titre or "").strip() else None)
    n_titre = int(round(float(duree_titre or 0) * fps)) if calque_titre else 0
    if calque_titre and n_titre <= 0:
        n_titre = n_total  # 0 = titre pendant toute la vidéo
    k_fondu_titre = int(round(FONDU_TITRE * fps))
    chemin_logo = chemin_fichier(logo)
    calque_logo = _calque_logo(chemin_logo, largeur, hauteur, position_logo, taille_logo, opacite_logo) if chemin_logo else None

    # écriture
    dossier = nouveau_dossier(cfg.VIDEOS_DIR, _slug(nom, "montage"))
    sortie = dossier / "montage.mp4"
    ec = _Ecrivain(sortie, "libx264", "aac" if avec_son else None, FREQ, largeur, hauteur, fps, crf=20)
    generateurs = {}

    def image_de(i, k):
        m = morceaux[i]
        if i not in generateurs:
            generateurs[i] = (_images_ken_burns(m["image"], m["n"], largeur, hauteur, float(ken_burns), i)
                              if "image" in m and ken_burns
                              else _images_fixes(cadrer(m["image"], largeur, hauteur, cadrage), m["n"]) if "image" in m
                              else _images_video(m["chemin"], m["n"], fps, largeur, hauteur, cadrage))
        img = next(generateurs[i])
        if k - m["debut"] == m["n"] - 1:
            generateurs.pop(i).close()
        g = gain_morceau(i, k - m["debut"])
        return img.astype(np.float32) * g if g < 1 else img  # pleine lumière : l'image telle quelle (uint8)

    try:
        for k in range(n_total):
            actifs = [i for i, m in enumerate(morceaux) if m["debut"] <= k < m["debut"] + m["n"]]
            if len(actifs) == 2:  # fondu enchaîné : le suivant monte pendant que le précédent descend
                a, b = actifs
                w = (k - morceaux[b]["debut"] + 1) / (k_trans + 1)
                image = image_de(a, k) * (1 - w) + image_de(b, k) * w
            else:
                image = image_de(actifs[0], k)
            if image.dtype != np.uint8:
                image = np.clip(image + 0.5, 0, 255).astype(np.uint8)
            if calque_titre and k < n_titre:
                force = min(1.0, (k + 1) / (k_fondu_titre + 1), (n_titre - k) / (k_fondu_titre + 1)) \
                    if n_titre < n_total else 1.0
                image = _poser(image, calque_titre, min(force, fondu_global(k)))
            if calque_logo:  # le logo suit le départ du noir et la fin au noir
                image = _poser(image, calque_logo, fondu_global(k))
            ec.image(image, k / fps)
            if avec_son:
                a0, a1 = int(round(k / fps * FREQ)), int(round((k + 1) / fps * FREQ))
                ec.son(piste[:, a0:a1], a0 / FREQ)
            if k % 30 == 0:
                progress(0.05 + 0.95 * k / n_total, desc=f"Montage : {duree_lisible(k / fps)} / {duree_lisible(duree)}")
        if avec_son:
            ec.silence_jusqua(duree)
        ec.fermer()
    finally:
        for g in generateurs.values():
            g.close()

    temps = time.monotonic() - t_debut
    ecrire_creation(dossier, {
        "type": "video", "nom": nom, "description": (titre or "").strip()[:200] or f"montage de {len(morceaux)} morceaux",
        "montage": {"morceaux": [m["chemin"].name for m in morceaux], "transition": transition,
                    "duree_transition": round(k_trans / fps, 2), "musique": Path(str(chemin_musique)).name
                    if chemin_musique else None, "titre": (titre or "").strip(), "logo": bool(chemin_logo),
                    "ken_burns": float(ken_burns or 0), "voix": Path(str(chemin_voix)).name if chemin_voix else None},
        "duree": round(duree, 2), "largeur": largeur, "hauteur": hauteur, "temps_s": round(temps),
        "versions": [{"graine": 0, "dossier": ".", "fichier": str(sortie)}],
        **(infos_en_plus or {}),
    })
    note = ""
    if transition != COUPE and len(morceaux) > 1 and k_trans < int(round(float(duree_transition or 0) * fps)):
        note = f" ; fondus raccourcis à {k_trans / fps:.1f} s (un morceau est très court)"
    msg = (f"✅ Montage de {len(morceaux)} morceau(x), {duree_lisible(duree)} en {largeur}×{hauteur} à {fps:g} images/s, "
           f"fait en {temps:.0f} s{note} : `{sortie}` ({_taille(sortie.stat().st_size)}). Il est aussi dans la galerie "
           "et dans la liste de l'onglet « Voix off ».")
    return msg, str(sortie), [str(sortie)], str(dossier)
