"""Convertisseur universel audio / vidéo (groupe « 🔧 Audio / vidéo ») : changer le format, extraire le son, couper,
recoller, réduire le poids. Tout passe par PyAV (déjà installé : ses roues Windows contiennent FFmpeg avec libx264,
libvpx-vp9, AAC, LAME, Opus, FLAC), sauf l'OGG Vorbis (soundfile : l'encodeur Vorbis de FFmpeg est expérimental).

Un seul transcodeur (`_Ecrivain`) sert à tout : les images sont replacées sur une grille de temps fixe à la cadence de
sortie (une image répétée ou sautée au besoin : cadences différentes recollées sans accélérer), redimensionnées avec
bandes noires si leur taille diffère (`ImageOps.pad`), le son rééchantillonné en flottant stéréo puis converti au
format du codec et redécoupé à sa taille de trame (`AudioFifo`) ; un morceau sans son reçoit du silence pour que son et
image restent calés. Rangement : data/conversions/<horodatage>_<nom>/.
"""
import time
from fractions import Fraction
from pathlib import Path

import gradio as gr
import numpy as np

from . import config as cfg
from .outils import chemin_fichier, nouveau_dossier
from .serie import _slug

# format → (extension, codec vidéo ou None, codec audio, fréquence)
FORMATS_AUDIO = {
    "MP3": ("mp3", None, "libmp3lame", 44100),
    "M4A (AAC, téléphones)": ("m4a", None, "aac", 44100),
    "WAV (sans perte)": ("wav", None, "pcm_s16le", 44100),
    "FLAC (sans perte, compressé)": ("flac", None, "flac", 44100),
    "OGG (Vorbis, jeux)": ("ogg", None, "vorbis", 44100),
    "OPUS (léger)": ("opus", None, "libopus", 48000),
}
FORMATS_VIDEO = {
    "MP4 (H.264, partout)": ("mp4", "libx264", "aac", 44100),
    "WebM (VP9, web)": ("webm", "libvpx-vp9", "libopus", 48000),
    "MKV (H.264)": ("mkv", "libx264", "aac", 44100),
    "MOV (H.264)": ("mov", "libx264", "aac", 44100),
    "GIF animé (sans son)": ("gif", "gif", None, None),
}
FORMATS = {**FORMATS_VIDEO, **FORMATS_AUDIO}
DEBITS_AUDIO = {"libmp3lame": 192_000, "aac": 192_000, "libopus": 128_000}
VIDEOS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp", ".ts"}
AUDIOS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma", ".aiff", ".aif"}
FICHIERS_MAX = 50
GIF_FPS, GIF_LARGEUR = 12, 480  # au-delà, un GIF devient énorme

# réduire le poids : hauteur maximale, qualité H.264 (crf : 18 = très bonne, 28 = moyenne), débit audio
REDUCTIONS = {
    "Légère (1080p au plus, très bonne qualité)": (1080, 23, 160_000),
    "Moyenne (720p, bonne qualité, ~4 fois plus léger)": (720, 26, 128_000),
    "Forte (480p, pour envoyer par message)": (480, 30, 96_000),
    "Très forte (360p, le plus léger)": (360, 33, 64_000),
}


def _entree(fichier):
    f = chemin_fichier(fichier)
    if not f or not Path(str(f)).is_file():
        raise gr.Error("Ajoute un fichier audio ou vidéo.")
    return Path(str(f))


def _entrees(fichiers):
    chemins = [_entree(f) for f in (fichiers or []) if f]
    if not chemins:
        raise gr.Error("Ajoute un ou plusieurs fichiers audio ou vidéo.")
    if len(chemins) > FICHIERS_MAX:
        raise gr.Error(f"{len(chemins)} fichiers : {FICHIERS_MAX} au plus d'un coup.")
    return chemins


def _nom_libre(dossier, base, ext):
    s, k = dossier / f"{base}.{ext}", 1
    while s.exists():
        k += 1
        s = dossier / f"{base}_{k}.{ext}"
    return s


def _taille(octets):
    return f"{octets / 1e6:.1f} Mo" if octets >= 1e6 else f"{octets / 1e3:.0f} Ko"


def duree_lisible(s):
    if s < 60:
        return f"{s:.1f} s".replace(".0 s", " s").replace(".", ",")
    s = int(round(s))
    return f"{s // 3600} h {s % 3600 // 60:02d} min {s % 60:02d} s" if s >= 3600 else f"{s // 60} min {s % 60:02d} s"


def infos(chemin):
    """{duree, video: bool, son: bool, largeur, hauteur, fps} lus dans le conteneur."""
    import av

    try:
        with av.open(str(chemin)) as c:
            v = next((s for s in c.streams.video if s.codec_context.width), None)
            a = c.streams.audio[0] if c.streams.audio else None
            duree = c.duration / av.time_base if c.duration else 0.0
            if not duree:
                for s in (v, a):
                    if s is not None and s.duration and s.time_base:
                        duree = max(duree, float(s.duration * s.time_base))
            d = {"duree": duree, "video": v is not None, "son": a is not None}
            if v is not None:
                fps = float(v.average_rate or v.guessed_rate or 25)
                duree_v = float(v.duration * v.time_base) if v.duration and v.time_base else (
                    v.frames / fps if v.frames else duree)
                d.update(largeur=v.codec_context.width, hauteur=v.codec_context.height, fps=fps,
                         duree_video=duree_v or duree)
            return d
    except av.error.FFmpegError as e:
        raise gr.Error(f"« {Path(chemin).name} » n'est pas un fichier audio ou vidéo lisible ({e}).") from e


def temps(texte, defaut=None):
    """« 90 », « 1:30 », « 1:02:03 », « 1 min 30 », « 12,5 » → secondes ; vide → defaut."""
    import re

    t = str(texte if texte is not None else "").strip().lower().replace(",", ".")
    if not t:
        return defaut
    if ":" in t:
        parties = [float(p or 0) for p in t.split(":")]
        s = 0.0
        for p in parties:
            s = s * 60 + p
        return s
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s*h)?\s*(?:(\d+(?:\.\d+)?)\s*(?:min|m))?\s*(?:(\d+(?:\.\d+)?)\s*s?)?", t)
    if m and any(m.groups()):
        h, mi, s = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + s
    raise gr.Error(f"Temps mal écrit : « {texte} ». Exemples : « 90 », « 1:30 », « 1 min 30 ».")


class _Ecrivain:
    """Fichier de sortie : flux vidéo (ou rien) et flux audio (ou rien), alimentés morceau par morceau."""

    def __init__(self, sortie, codec_v, codec_a, frequence, largeur=None, hauteur=None, fps=None, crf=23,
                 debit_audio=None):
        import av

        self.av = av
        self.sortie = Path(sortie)
        self.out = av.open(str(sortie), "w")
        self.v = self.a = None
        self.largeur, self.hauteur = largeur, hauteur
        self.fps = Fraction(fps).limit_denominator(1001) if fps else None
        self.image_suivante = 0  # indice de la prochaine image sur la grille de temps
        self.echantillons = 0  # échantillons audio déjà écrits (à `frequence`)
        if codec_v:
            self.v = self.out.add_stream(codec_v, rate=self.fps)
            self.v.width, self.v.height = largeur, hauteur
            if codec_v == "gif":
                self.v.pix_fmt = "rgb8"
            else:
                self.v.pix_fmt = "yuv420p"
                self.v.options = ({"crf": str(crf), "preset": "medium"} if codec_v == "libx264"
                                  else {"crf": str(crf + 8), "b": "0", "row-mt": "1", "deadline": "good"})
        if codec_a:
            self.frequence = frequence
            self.a = self.out.add_stream(codec_a, rate=frequence, layout="stereo")
            if codec_a in DEBITS_AUDIO or debit_audio:
                self.a.bit_rate = debit_audio or DEBITS_AUDIO[codec_a]
            if codec_a == "vorbis":
                self.a.codec_context.options = {"strict": "experimental"}
            self.vers_codec = av.AudioResampler(format=self.a.format.name, layout="stereo", rate=frequence)
            self.fifo = av.AudioFifo()

    # --- image ---
    def image(self, image_rgb, t):
        """Image (tableau RGB) au temps t (s) de la sortie : écrite autant de fois que la grille le demande."""
        if self.v is None:
            return
        cible = int(round(t * float(self.fps)))
        if cible < self.image_suivante:
            return  # cadence d'entrée plus haute : image sautée
        trame = self.av.VideoFrame.from_ndarray(self._ajuster(image_rgb), format="rgb24")
        while self.image_suivante <= cible:
            trame.pts = self.image_suivante
            trame.time_base = 1 / self.fps
            for p in self.v.encode(trame):
                self.out.mux(p)
            self.image_suivante += 1

    def _ajuster(self, rgb):
        h, w = rgb.shape[:2]
        if (w, h) == (self.largeur, self.hauteur):
            return rgb
        from PIL import Image, ImageOps

        im = Image.fromarray(rgb)
        meme_forme = abs(w / h - self.largeur / self.hauteur) < 0.01
        im = (im.resize((self.largeur, self.hauteur), Image.Resampling.LANCZOS) if meme_forme
              else ImageOps.pad(im, (self.largeur, self.hauteur), Image.Resampling.LANCZOS, color=(0, 0, 0)))
        return np.asarray(im)

    def completer_images(self, t_fin):
        """Dernière image répétée jusqu'à t_fin (morceau dont le son dure plus que l'image)."""
        if self.v is None or not hasattr(self, "_derniere"):
            return
        self.image(self._derniere, t_fin - 1 / float(self.fps))

    # --- son ---
    def son(self, pcm, t):
        """Son flottant stéréo (2, n) à `frequence`, commençant au temps t (s) de la sortie : un trou est comblé de
        silence, un chevauchement retiré."""
        if self.a is None or pcm.shape[1] == 0:
            return
        debut = int(round(t * self.frequence))
        if debut > self.echantillons:
            self._pousser(np.zeros((2, debut - self.echantillons), np.float32))
        elif debut < self.echantillons:
            pcm = pcm[:, self.echantillons - debut:]
        self._pousser(pcm)

    def silence_jusqua(self, t):
        if self.a is not None and int(round(t * self.frequence)) > self.echantillons:
            self._pousser(np.zeros((2, int(round(t * self.frequence)) - self.echantillons), np.float32))

    def _pousser(self, pcm):
        if pcm.shape[1] == 0:
            return
        trame = self.av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm, np.float32), format="fltp", layout="stereo")
        trame.sample_rate = self.frequence
        for t in self.vers_codec.resample(trame):
            self.fifo.write(t)
        self.echantillons += pcm.shape[1]
        self._vider(False)

    def _vider(self, fin):
        taille = self.a.codec_context.frame_size or 4096
        while self.fifo.samples >= taille or (fin and self.fifo.samples):
            trame = self.fifo.read(min(taille, self.fifo.samples) if fin else taille)
            for p in self.a.encode(trame):
                self.out.mux(p)

    def fermer(self):
        if self.v is not None:
            for p in self.v.encode(None):
                self.out.mux(p)
        if self.a is not None:
            self._vider(True)
            for p in self.a.encode(None):
                self.out.mux(p)
        self.out.close()
        return self.sortie


def _alimenter(ec, source, decalage=0.0, debut=0.0, fin=None, video=True, son=True, progress=None, part=(0, 1)):
    """Décode `source` entre debut et fin (s) et l'écrit dans `ec` à partir de `decalage` (s) de la sortie.
    Renvoie la durée écrite (s)."""
    import av

    d = infos(source)
    fin = min(fin, d["duree"]) if fin is not None and d["duree"] else fin
    if video and ec.v is not None and d.get("video") and d.get("duree_video"):
        # l'image fait foi : le son d'un AAC dépasse de quelques ms (remplissage de l'encodeur), il est coupé là
        fin = min(fin, d["duree_video"]) if fin is not None else d["duree_video"]
    with av.open(str(source)) as c:
        fv = next((s for s in c.streams.video if s.codec_context.width), None) if (video and ec.v) else None
        fa = c.streams.audio[0] if (son and ec.a and c.streams.audio) else None
        flux = [s for s in (fv, fa) if s is not None]
        if not flux:
            return 0.0
        for s in flux:
            s.thread_type = "AUTO"
        if debut > 1:
            c.seek(int(max(0.0, debut - 1) * av.time_base), backward=True, any_frame=False)
        vers_pcm = av.AudioResampler(format="fltp", layout="stereo", rate=ec.frequence) if fa else None
        fin_v = fin_a = 0.0
        pos_a = None
        total = (fin or d["duree"] or 1) - debut
        for paquet in c.demux(*flux):
            for trame in paquet.decode():
                t = trame.time
                if t is None:
                    continue
                if paquet.stream is fv:
                    if t < debut - 1e-6:
                        continue
                    if fin is not None and t >= fin:
                        continue
                    rgb = trame.to_ndarray(format="rgb24")
                    ec._derniere = ec._ajuster(rgb)
                    ec.image(rgb, decalage + t - debut)
                    fin_v = max(fin_v, t - debut + 1 / float(ec.fps))
                    if progress and ec.image_suivante % 50 == 0:
                        progress(part[0] + part[1] * min(1.0, (t - debut) / max(total, 1e-3)),
                                 desc=f"{Path(source).name} : {duree_lisible(t - debut)} / {duree_lisible(total)}")
                else:
                    for r in vers_pcm.resample(trame):
                        pcm = r.to_ndarray()
                        if pos_a is None:  # temps du son compté en échantillons depuis la première trame
                            pos_a = t
                        t0, n = pos_a, pcm.shape[1]
                        pos_a += n / ec.frequence
                        i0 = max(0, int(round((debut - t0) * ec.frequence)))
                        i1 = n if fin is None else min(n, int(round((fin - t0) * ec.frequence)))
                        if i1 <= i0:
                            continue
                        ec.son(pcm[:, i0:i1], decalage + t0 + i0 / ec.frequence - debut)
                        fin_a = max(fin_a, t0 + i1 / ec.frequence - debut)
        duree = max(fin_v, fin_a)
        if fin is not None:
            duree = min(duree, fin - debut) if duree else fin - debut
        ec.completer_images(decalage + duree)
        ec.silence_jusqua(decalage + duree)
        return duree


def _taille_sortie(largeur, hauteur, hauteur_max=None, largeur_max=None, pair=True):
    """Taille de sortie qui garde les proportions, sous hauteur_max / largeur_max, côtés pairs (H.264, yuv420p)."""
    echelle = 1.0
    if hauteur_max and hauteur > hauteur_max:
        echelle = min(echelle, hauteur_max / hauteur)
    if largeur_max and largeur > largeur_max:
        echelle = min(echelle, largeur_max / largeur)
    w, h = max(2, round(largeur * echelle)), max(2, round(hauteur * echelle))
    if pair:
        w, h = w // 2 * 2, h // 2 * 2
    return w, h


def _ecrivain_pour(sortie, format_label, d, hauteur_max=None, crf=23, debit_audio=None):
    ext, cv, ca, freq = FORMATS[format_label]
    if cv and not d.get("video"):
        raise gr.Error(f"Ce fichier n'a pas d'image : choisis un format audio (MP3, M4A, WAV…) au lieu de {format_label}.")
    largeur = hauteur = fps = None
    if cv:
        gif = cv == "gif"
        largeur, hauteur = _taille_sortie(d["largeur"], d["hauteur"], hauteur_max, GIF_LARGEUR if gif else None,
                                          pair=not gif)
        fps = min(d.get("fps") or 25, GIF_FPS if gif else 60)
    if ca and not d.get("son") and not cv:
        raise gr.Error("Ce fichier n'a pas de son.")
    return _Ecrivain(sortie, cv, ca if (ca and d.get("son")) else None, freq, largeur, hauteur, fps, crf, debit_audio)


def _resume(sources, sorties, debut, dossier, quoi):
    lignes = [f"- {Path(s).name} ({_taille(Path(s).stat().st_size)}) → **{Path(o).name}** ({_taille(Path(o).stat().st_size)})"
              for s, o in zip(sources, sorties)]
    return (f"✅ {quoi} : {len(sorties)} fichier(s) en {time.monotonic() - debut:.0f} s, dans `{dossier}`.\n\n"
            + "\n".join(lignes))


# --- Les cinq outils --------------------------------------------------------------------------------------------
def changer_format(fichiers, format_label, nom="conversion", progress=gr.Progress()):
    """Chaque fichier dans le format choisi (une vidéo vers un format audio = son seul). Renvoie (message, fichiers,
    dossier)."""
    if format_label not in FORMATS:
        raise gr.Error("Choisis le format de sortie.")
    chemins = _entrees(fichiers)
    debut = time.monotonic()
    dossier = nouveau_dossier(cfg.CONVERSIONS_DIR, _slug(nom, "conversion"))
    ext = FORMATS[format_label][0]
    sorties, faits, erreurs = [], [], []
    for i, source in enumerate(chemins):
        progress(i / len(chemins), desc=f"{source.name}…")
        try:
            d = infos(source)
            sortie = _nom_libre(dossier, _slug(source.stem, "fichier"), ext)
            if ext == "ogg":
                _ogg_vorbis(source, sortie)
            else:
                ec = _ecrivain_pour(sortie, format_label, d)
                _alimenter(ec, source, progress=progress, part=(i / len(chemins), 1 / len(chemins)))
                ec.fermer()
            sorties.append(sortie)
            faits.append(source)
        except gr.Error as e:
            erreurs.append(f"{source.name} : {e.message if hasattr(e, 'message') else e}")
    if not sorties:
        raise gr.Error("Aucun fichier converti :\n" + "\n".join(erreurs))
    msg = _resume(faits, sorties, debut, dossier,
                  f"Conversion en {format_label}")
    if erreurs:
        msg += "\n\n⚠️ " + "\n\n⚠️ ".join(erreurs)
    return msg, [str(s) for s in sorties], str(dossier)


def _ogg_vorbis(source, sortie):
    """OGG Vorbis par soundfile (le son décodé par PyAV / librosa : audio.charger)."""
    import soundfile as sf

    from .audio import charger

    y, sr = charger(source, sr=44100, mono=False)
    y = np.atleast_2d(y)
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    sf.write(str(sortie), y.T, sr, format="OGG", subtype="VORBIS")
    return sortie


def extraire_son(fichiers, format_label="MP3", nom="son", progress=gr.Progress()):
    """Le son de chaque vidéo, dans un format audio. Renvoie (message, fichiers, dossier)."""
    if format_label not in FORMATS_AUDIO:
        raise gr.Error("Choisis un format audio.")
    for f in _entrees(fichiers):
        if not infos(f)["son"]:
            raise gr.Error(f"« {f.name} » n'a pas de son à extraire.")
    return changer_format(fichiers, format_label, nom, progress=progress)


def couper(fichier, debut_txt="0", fin_txt="", nom="", progress=gr.Progress()):
    """Garde le passage entre début et fin (même format que l'original ; MP4 pour une vidéo d'un format peu courant).
    Renvoie (message, [fichier], dossier)."""
    source = _entree(fichier)
    d = infos(source)
    debut = max(0.0, temps(debut_txt, 0.0))
    fin = temps(fin_txt, d["duree"])
    if d["duree"] and fin > d["duree"] + 0.05:
        fin = d["duree"]
    if fin <= debut:
        raise gr.Error(f"La fin ({duree_lisible(fin)}) doit venir après le début ({duree_lisible(debut)}) ; le fichier dure "
                       f"{duree_lisible(d['duree'])}.")
    format_label = _format_d_origine(source, d)
    ext = FORMATS[format_label][0]
    t0 = time.monotonic()
    dossier = nouveau_dossier(cfg.CONVERSIONS_DIR, _slug(nom or source.stem, "coupe"))
    sortie = _nom_libre(dossier, f"{_slug(nom or source.stem, 'extrait')}_extrait", ext)
    ec = _ecrivain_pour(sortie, format_label, d)
    _alimenter(ec, source, 0.0, debut, fin, progress=progress)
    ec.fermer()
    return (f"✅ Passage de {duree_lisible(debut)} à {duree_lisible(fin)} ({duree_lisible(fin - debut)}) gardé en "
            f"{time.monotonic() - t0:.0f} s : `{sortie}`", [str(sortie)], str(dossier))


def _format_d_origine(source, d):
    ext = source.suffix.lower().lstrip(".")
    for label, (e, cv, _, _) in FORMATS.items():
        if e == ext and (cv is not None) == bool(d.get("video")):
            return label
    return "MP4 (H.264, partout)" if d.get("video") else "M4A (AAC, téléphones)" if ext in ("aac", "m4a") else "MP3"


def recoller(fichiers, format_label="MP4 (H.264, partout)", nom="montage", progress=gr.Progress()):
    """Les fichiers mis bout à bout dans l'ordre (taille et cadence de la première vidéo ; bandes noires si une vidéo
    a d'autres proportions ; silence pour une vidéo muette). Renvoie (message, [fichier], dossier)."""
    chemins = _entrees(fichiers)
    if len(chemins) < 2:
        raise gr.Error("Ajoute au moins deux fichiers à recoller, dans l'ordre.")
    toutes = [infos(c) for c in chemins]
    videos = [d for d in toutes if d["video"]]
    if videos and format_label not in FORMATS_VIDEO:
        format_label = "MP4 (H.264, partout)"
    if not videos and format_label in FORMATS_VIDEO:
        format_label = "MP3"
    ext, cv, ca, freq = FORMATS[format_label]
    t0 = time.monotonic()
    dossier = nouveau_dossier(cfg.CONVERSIONS_DIR, _slug(nom, "montage"))
    sortie = _nom_libre(dossier, _slug(nom, "montage"), ext)
    if cv:
        ref = videos[0]
        gif = cv == "gif"
        largeur, hauteur = _taille_sortie(ref["largeur"], ref["hauteur"], None, GIF_LARGEUR if gif else None,
                                          pair=not gif)
        fps = min(ref["fps"], GIF_FPS if gif else 60)
    else:
        largeur = hauteur = fps = None
    if ext == "ogg":
        raise gr.Error("Pour recoller en OGG, recolle d'abord en WAV ou FLAC, puis change le format.")
    avec_son = ca is not None and any(d["son"] for d in toutes)
    ec = _Ecrivain(sortie, cv, ca if avec_son else None, freq, largeur, hauteur, fps)
    position = 0.0
    for i, (c, d) in enumerate(zip(chemins, toutes)):
        progress(i / len(chemins), desc=f"{i + 1}/{len(chemins)} : {c.name}…")
        if cv and not d["video"]:
            raise gr.Error(f"« {c.name} » n'a pas d'image : on ne peut pas le recoller dans une vidéo "
                           "(ajoute-le plutôt en musique de fond, une fois cette fonction disponible).")
        position += _alimenter(ec, c, position, progress=progress, part=(i / len(chemins), 1 / len(chemins)))
    ec.fermer()
    return (f"✅ {len(chemins)} fichiers recollés ({duree_lisible(position)}) en {time.monotonic() - t0:.0f} s : "
            f"`{sortie}` ({_taille(sortie.stat().st_size)})", [str(sortie)], str(dossier))


def reduire(fichiers, niveau_label=None, nom="allege", progress=gr.Progress()):
    """Vidéos réencodées en H.264 plus léger (hauteur maximale, qualité) ; un fichier audio est réencodé en MP3 à
    plus bas débit. Renvoie (message, fichiers, dossier)."""
    hauteur_max, crf, debit = REDUCTIONS.get(niveau_label) or list(REDUCTIONS.values())[1]
    chemins = _entrees(fichiers)
    t0 = time.monotonic()
    dossier = nouveau_dossier(cfg.CONVERSIONS_DIR, _slug(nom, "allege"))
    sorties, lignes = [], []
    for i, source in enumerate(chemins):
        progress(i / len(chemins), desc=f"{source.name}…")
        d = infos(source)
        label = "MP4 (H.264, partout)" if d["video"] else "MP3"
        sortie = _nom_libre(dossier, f"{_slug(source.stem, 'fichier')}_leger", FORMATS[label][0])
        ec = _ecrivain_pour(sortie, label, d, hauteur_max, crf, debit)
        _alimenter(ec, source, progress=progress, part=(i / len(chemins), 1 / len(chemins)))
        ec.fermer()
        avant, apres = source.stat().st_size, sortie.stat().st_size
        sorties.append(sortie)
        gain = (1 - apres / avant) * 100 if avant else 0
        lignes.append(f"- {source.name} : {_taille(avant)} → **{_taille(apres)}** "
                      + (f"(−{gain:.0f} %)" if gain > 0 else "(pas plus léger : le fichier était déjà compact)"))
    return (f"✅ {len(sorties)} fichier(s) allégé(s) en {time.monotonic() - t0:.0f} s, dans `{dossier}`.\n\n"
            + "\n".join(lignes), [str(s) for s in sorties], str(dossier))


def decrire(fichier):
    """Texte court affiché quand un fichier est déposé : durée, image, son."""
    f = chemin_fichier(fichier)
    if not f:
        return ""
    try:
        d = infos(f)
    except gr.Error as e:
        return f"⚠️ {e}"
    morceaux = [f"durée {duree_lisible(d['duree'])}"]
    if d["video"]:
        morceaux.append(f"vidéo {d['largeur']}×{d['hauteur']} à {d['fps']:.0f} images/s")
    morceaux.append("avec du son" if d["son"] else "sans son")
    return f"**{Path(str(f)).name}** : " + ", ".join(morceaux) + "."
