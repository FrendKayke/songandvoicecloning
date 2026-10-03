"""Diagnostic (onglet Modèles) : état de la machine et essai réel de chaque moteur, avec un rapport texte à envoyer
pour dépannage.

- Diagnostic rapide (moins d'une minute) : système, carte graphique (nvidia-smi), espace disque, chaque environnement
  Python (PyTorch importable, CUDA disponible), modèles manquants, serveur ACE-Step.
- Essai complet (10 à 20 minutes) : une vraie génération courte par moteur (ACE-Step, Demucs, Seed-VC, Chatterbox,
  nettoyage, Qwen3-VL, Z-Image, FLUX.2 klein, photos, Wan 2.2, Stable Audio, Hunyuan3D forme + texture), dans l'ordre qui évite de recharger
  ACE-Step, avec la durée et la mémoire graphique utilisée après chaque étape. Les fichiers produits vont dans
  data/diagnostic/<horodatage>/ (les dossiers de la galerie sont redirigés pendant l'essai : rien ne s'y ajoute).
Une étape en échec n'arrête pas les suivantes ; le message d'erreur est gardé dans le rapport.
"""
import contextlib
import os
import platform
import shutil
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from . import acestep, chatterbox, demucs, diffusion, nettoyage, retraits, rvc, seedvc, serveur_acestep
from . import config as cfg
from .voix import list_voices

# (nom, chemin du Python lu au moment de l'appel — les tests redirigent cfg —, clé de retrait ou None)
ENVIRONNEMENTS = [
    ("ACE-Step", lambda: cfg.ACESTEP_PYTHON, None),
    ("Seed-VC + Demucs", lambda: cfg.SEEDVC_PYTHON, None),
    ("Chatterbox", lambda: cfg.CHATTERBOX_PYTHON, "chatterbox"),
    ("Nettoyage", lambda: cfg.NETTOYAGE_PYTHON, "nettoyage"),
    ("RVC (Applio)", lambda: cfg.RVC_PYTHON, "rvc"),
    ("Diffusion", lambda: cfg.DIFFUSION_PYTHON, "diffusion"),
]
RETIRE = "retiré pour gagner de la place (Outils → Espace disque)"
SONDE = ("import sys, torch; ok = torch.cuda.is_available(); "
         "print('SONDE', sys.version.split()[0], torch.__version__, ok, torch.cuda.get_device_name(0) if ok else '-')")
NIVEAU_LEGER = next(iter(nettoyage.NIVEAUX))


def _rien(*a, **k):
    pass


def carte_graphique():
    """« nom, pilote, mémoire utilisée / totale » d'après nvidia-smi, ou None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        nom, pilote, utilisee, totale = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
        return f"{nom}, pilote {pilote}, mémoire {int(utilisee) / 1024:.1f} / {int(totale) / 1024:.1f} Go"
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _mesure_gpu():
    """(utilisation %, mémoire utilisée en Go, puissance en W ou None) d'après nvidia-smi, ou None."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        util, memoire, puissance = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
        try:
            watts = float(puissance)
        except ValueError:  # « [N/A] » sur certaines cartes
            watts = None
        return float(util), float(memoire) / 1024, watts
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


class SuiviGPU:
    """Pendant une étape, relève chaque seconde l'utilisation de la carte graphique (nvidia-smi) : le Gestionnaire
    des tâches de Windows affiche par défaut le graphe « 3D », qui reste presque à zéro pendant un calcul CUDA ;
    ces mesures disent si la carte travaille vraiment (et où l'attente se fait : chargement, processeur…)."""

    def __init__(self, periode=1.0):
        self.periode, self.mesures = periode, []
        self._fin = threading.Event()
        self._fil = None

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            self._fil = threading.Thread(target=self._boucle, daemon=True)
            self._fil.start()
        return self

    def _boucle(self):
        while not self._fin.wait(self.periode):
            m = _mesure_gpu()
            if m:
                self.mesures.append(m)

    def __exit__(self, *exc):
        self._fin.set()
        if self._fil:
            self._fil.join(timeout=15)
        return False

    def resume(self):
        """« GPU 85 % en moyenne (pointe 100 %, occupé > 50 % : 90 % du temps), mémoire jusqu'à 10.2 Go, 190 W
        au plus », ou "" sans mesure."""
        if not self.mesures:
            return ""
        utils = [u for u, _, _ in self.mesures]
        occupe = sum(u > 50 for u in utils) / len(utils)
        watts = [w for _, _, w in self.mesures if w is not None]
        return (f"GPU {sum(utils) / len(utils):.0f} % en moyenne (pointe {max(utils):.0f} %, occupé à plus de 50 % "
                f"{occupe:.0%} du temps), mémoire jusqu'à {max(m for _, m, _ in self.mesures):.1f} Go"
                + (f", {max(watts):.0f} W au plus" if watts else ""))


def surveiller_gpu(duree=60, periode=2.0, attendre=time.sleep):
    """Générateur (bouton de l'onglet Modèles) : utilisation de la carte graphique en direct pendant `duree` s,
    pendant qu'une génération tourne dans un autre onglet."""
    if not shutil.which("nvidia-smi"):
        yield "❌ nvidia-smi introuvable : le pilote NVIDIA n'est pas installé ou pas dans le PATH."
        return
    suivi = SuiviGPU()
    debut = time.monotonic()
    while time.monotonic() - debut < duree:
        m = _mesure_gpu()
        if m:
            suivi.mesures.append(m)
            util, memoire, watts = m
            barre = "█" * round(util / 5) + "░" * (20 - round(util / 5))
            yield (f"**Maintenant** : `{barre}` {util:.0f} % · mémoire {memoire:.1f} Go"
                   + (f" · {watts:.0f} W" if watts is not None else "")
                   + f"\n\n**Depuis le début** ({len(suivi.mesures)} mesures) : {suivi.resume()}"
                   + f"\n\n⏳ Encore {duree - (time.monotonic() - debut):.0f} s…")
        attendre(periode)
    yield (f"**Bilan sur {duree} s** : {suivi.resume() or 'aucune mesure'}\n\n" + CONSEIL_GPU)


CONSEIL_GPU = (
    "*À savoir :* le Gestionnaire des tâches de Windows affiche par défaut le graphe « 3D », qui reste presque à "
    "zéro pendant un calcul d'intelligence artificielle (CUDA) : dans Performances → GPU, clique sur le titre d'un "
    "graphe et choisis « Cuda » (ou « Compute_0 »). Une utilisation qui retombe régulièrement est normale pendant "
    "les chargements et les transferts : avec 12 Go, les gros modèles (encodeurs de texte de 8 à 11 Go) passent "
    "tour à tour sur la carte. Pendant la génération proprement dite, elle doit dépasser 90 %.")


def sonde(python):
    """(ok, détail) : Python de l'environnement présent, PyTorch importable, CUDA disponible."""
    if not Path(python).exists():
        return False, f"Python introuvable ({python}) : relance INSTALLER.bat"
    try:
        r = subprocess.run([python, "-c", SONDE], capture_output=True, text=True, timeout=300,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:
        return False, "pas de réponse en 5 minutes"
    ligne = next((l_ for l_ in r.stdout.splitlines() if l_.startswith("SONDE ")), None)
    if not ligne:
        erreur = (r.stderr.strip().splitlines() or ["erreur inconnue"])[-1]
        return False, f"PyTorch ne se charge pas : {erreur[:300]}"
    _, py, torch, cuda, gpu = ligne.split(" ", 4)
    if cuda != "True":
        return False, f"Python {py}, PyTorch {torch}, **CUDA indisponible** (calcul sur le processeur, très lent)"
    return True, f"Python {py}, PyTorch {torch}, carte : {gpu}"


class Rapport:
    def __init__(self, titre):
        self.lignes = [f"# {titre}", f"{datetime.now():%d/%m/%Y %H:%M}", ""]

    def section(self, titre):
        self.lignes += ["", f"## {titre}"]

    def ligne(self, ok, nom, detail=""):
        icone = {True: "✅", False: "❌", None: "➖"}[ok]
        self.lignes.append(f"- {icone} **{nom}**" + (f" : {detail}" if detail else ""))

    def texte(self):
        return "\n".join(self.lignes)

    def enregistrer(self, dossier):
        dossier.mkdir(parents=True, exist_ok=True)
        f = dossier / "rapport.txt"
        f.write_text(self.texte(), encoding="utf-8")
        return str(f)


def _systeme(r):
    r.section("Machine")
    r.ligne(None, "Système", f"{platform.system()} {platform.release()}, Python de l'application "
                             f"{platform.python_version()}")
    gpu = carte_graphique()
    r.ligne(bool(gpu), "Carte graphique", gpu or "nvidia-smi introuvable : pilote NVIDIA absent ?")
    try:
        libre = shutil.disk_usage(cfg.ENG_DIR if cfg.ENG_DIR.exists() else cfg.APP_DIR).free / 1e9
        r.ligne(libre > 20, "Espace libre (disque des moteurs)", f"{libre:.0f} Go")
    except OSError as e:
        r.ligne(False, "Espace libre", str(e))


def _modeles(r):
    r.section("Modèles")
    for nom, manquants in (("ACE-Step", acestep.missing_components()), ("Seed-VC", seedvc.missing_components()),
                           ("Demucs", [] if demucs.is_present() else [demucs.MODELE]),
                           ("Chatterbox", chatterbox.missing_components()),
                           ("Nettoyage", nettoyage.missing_components()), ("RVC", rvc.missing_components()),
                           ("Diffusion", diffusion.missing_components())):
        r.ligne(not manquants, nom, "présents" if not manquants else "manquants : " + ", ".join(map(str, manquants)))
    r.ligne(diffusion.jeton_present(), "Jeton Hugging Face (Stable Audio)",
            "enregistré" if diffusion.jeton_present() else "absent : bruitages indisponibles")


def moteur_lm_ace():
    """(ok, détail) : moteur du modèle de langage d'ACE-Step. Sous Windows, le moteur rapide (nano-vLLM, avec
    triton-windows et flash-attention) n'existe qu'en Python 3.11 ; sinon ACE-Step retombe sur PyTorch et l'écrit
    dans son journal (« vLLM backend is unavailable… », acestep/llm_backend_compat.py)."""
    import re

    version = None
    try:
        cfg_venv = (cfg.ACESTEP_DIR / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^version(?:_info)?\s*=\s*([\d.]+)", cfg_venv, re.M)
        version = m.group(1) if m else None
    except OSError:
        pass
    try:
        journal = serveur_acestep.journal().read_text(encoding="utf-8", errors="replace")
    except OSError:
        journal = ""
    py = f"Python {version}" if version else "Python inconnu"
    charge = re.findall(r"LLM model loaded: (\S+)", journal)  # startup_llm_init.py d'ACE-Step
    if charge:
        py += f", modèle de langage {charge[-1]}"
    if "vLLM backend is unavailable" in journal:
        return False, (f"{py} : moteur lent (PyTorch), Triton absent. Lance METTRE_A_JOUR.bat : ACE-Step est "
                       "réinstallé en Python 3.11, avec le moteur rapide.")
    if version and version.startswith("3.11"):
        return True, f"{py} : moteur rapide (nano-vLLM) disponible"
    if version:
        return False, f"{py} : moteur lent (PyTorch). Lance METTRE_A_JOUR.bat (passage en Python 3.11)."
    return None, "environnement ACE-Step introuvable"


def rapide():
    """Générateur : (rapport en Markdown, fichier du rapport ou None)."""
    r = Rapport("Diagnostic rapide de Studio Voix")
    _systeme(r)
    yield r.texte(), None
    r.section("Environnements Python")
    for nom, python, cle in ENVIRONNEMENTS:
        if cle and retraits.retire(cle):
            r.ligne(None, nom, RETIRE)
            continue
        ok, detail = sonde(python())
        r.ligne(ok, nom, detail)
        yield r.texte(), None
    _modeles(r)
    r.section("Serveur ACE-Step")
    r.ligne(None, "État", serveur_acestep.etat())
    ok, detail = moteur_lm_ace()
    r.ligne(ok, "Modèle de langage (réflexion)", detail)
    fichier = r.enregistrer(cfg.DATA_DIR / "diagnostic" / f"{datetime.now():%Y%m%d_%H%M%S}_rapide")
    yield r.texte() + f"\n\nRapport enregistré : `{fichier}`", fichier


@contextlib.contextmanager
def _galerie_redirigee(racine):
    """Pendant l'essai, les dossiers de la galerie pointent dans le dossier du diagnostic."""
    noms = ("SONGS_DIR", "TTS_DIR", "CLEAN_DIR", "SFX_DIR", "MODELS3D_DIR", "CARDS_DIR")
    anciens = {n: getattr(cfg, n) for n in noms}
    try:
        for n in noms:
            setattr(cfg, n, racine / n.lower().replace("_dir", ""))
            getattr(cfg, n).mkdir(parents=True, exist_ok=True)
        yield
    finally:
        for n, v in anciens.items():
            setattr(cfg, n, v)


def _ton(chemin, secondes=4.0, freq=220.0):
    t = np.linspace(0, secondes, int(secondes * cfg.SR), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * freq * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    sf.write(str(chemin), y.astype("float32"), cfg.SR)
    return chemin


def _etapes(d, voix):
    """(nom, fonction) de l'essai complet, dans l'ordre : ACE-Step d'abord, puis les moteurs qui l'arrêtent."""
    etat = {}

    def ace():
        params = acestep.text2music_params("8-bit chiptune, retro video game music, short loop", "[Instrumental]",
                                           "en", 10, 0, False)
        ((f, g),) = acestep.generer(params, [d / "acestep.wav"], _rien, "diagnostic")
        etat["musique"] = Path(f)
        return f"10 s de musique, graine {g}"

    def ace_chaud():  # serveur déjà démarré, modèles chargés : la carte seule
        params = acestep.text2music_params("8-bit chiptune, retro video game music, short loop", "[Instrumental]",
                                           "en", 10, 0, False)
        ((_, g),) = acestep.generer(params, [d / "acestep_2.wav"], _rien, "diagnostic")
        return f"10 s de musique, graine {g}, serveur déjà prêt"

    def dem():
        source = etat.get("musique") or _ton(d / "ton.wav")
        voix_sep, _ = demucs.separate_vocals(source, d)
        etat["voix_sep"] = voix_sep
        return "voix et instrumental séparés"

    def svc():
        ref = Path(voix) if voix else _ton(d / "reference.wav", 6, 180)
        sortie = seedvc.convert_voice(etat.get("voix_sep") or _ton(d / "source.wav"), ref, 0, 10, d)
        return f"conversion écrite ({Path(sortie).name})"

    def tts():
        if not voix:
            raise gr.Error("aucune voix dans la bibliothèque : enregistre-en une pour tester la synthèse vocale")
        fichier, _ = chatterbox.synthese(Path(voix).stem, "Bonjour, ceci est un essai de Studio Voix.", "Français",
                                         0.5, 0.5, 0.8, 42, progress=_rien)
        return f"lecture écrite ({Path(fichier).name})"

    def net():
        fichier, *_ = nettoyage.nettoyer(str(voix) if voix else str(_ton(d / "voix_test.wav")), NIVEAU_LEGER,
                                         progress=_rien)
        return f"nettoyage léger écrit ({Path(fichier).name})"

    def qwen():
        return "« " + diffusion.decrire("bruitage", "une porte de château qui grince", dossier=d) + " »"

    def zimage():
        res = diffusion.image("a red potion bottle with a cork stopper, single object, centered, plain white "
                              "background", [d / "zimage.png"], [42], 1024, 1024)
        etat["image"] = Path(res["fichiers"][0])
        return "image 1024×1024"

    def zimage_chaud():  # moteur resté ouvert : plus de chargement, la carte calcule
        diffusion.image("a blue mana crystal on a stone pedestal, single object, centered, plain white background",
                        [d / "zimage_2.png"], [43], 1024, 1024)
        return "2ᵉ image 1024×1024, modèles déjà en mémoire"

    def klein():
        image = etat.get("image")
        if not image:
            raise gr.Error("pas d'image de référence (l'étape Z-Image a échoué)")
        diffusion.personnage("The same red potion bottle from the reference image, standing on a wooden table in a "
                             "tavern, warm candle light", [image], [d / "personnage.png"], [42], 768, 768)
        return "image 768×768 à partir d'une référence"

    def photo():
        image = etat.get("image")
        if not image:
            raise gr.Error("pas d'image de départ (l'étape Z-Image a échoué)")
        res = diffusion.ameliorer(image, d / "photo_amelioree.png", 2, False, True)
        diffusion.detourer(d / "photo_amelioree.png", d / "photo_detouree.png", "general")
        return f"agrandie en {res['largeur']}×{res['hauteur']}, puis détourée"

    def film():
        image = etat.get("image")
        if not image:
            raise gr.Error("pas d'image de départ (l'étape Z-Image a échoué)")
        res = diffusion.video("The red potion bottle slowly rotates on a wooden table, warm candle light, static "
                              "camera", d / "video" / "video.mp4", image, 832, 480, 17, 6, 42)
        return f"vidéo {res['largeur']}×{res['hauteur']} de {res['images']} images (6 étapes)"

    def audio():
        if "bruitages" in [k for k in diffusion.MODELES if not diffusion.present(k)]:
            raise gr.Error("Stable Audio Open non téléchargé (jeton Hugging Face requis)")
        res = diffusion.bruitage("wooden door creaking", d / "bruitage", 3, 1, 42, 50)
        return f"bruitage de 3 s ({Path(res['fichiers'][0]).name})"

    def forme():
        image = etat.get("image")
        if not image:
            raise gr.Error("pas d'image de départ (l'étape Z-Image a échoué)")
        res = diffusion.forme3d(image, d / "3d", 5, 256, 40000, 42, True, ["glb"], web=True)
        if not res.get("texture"):
            raise gr.Error(f"forme créée ({res.get('faces')} faces) mais texture non peinte")
        return f"{res.get('faces')} faces, texture peinte, version web écrite"

    return [("ACE-Step (génération musicale)", ace, None),
            ("ACE-Step, 2ᵉ chanson (à chaud)", ace_chaud, None), ("Demucs (séparation)", dem, None),
            ("Seed-VC (conversion)", svc, None), ("Chatterbox (synthèse vocale)", tts, "chatterbox"),
            ("Nettoyage de voix (léger)", net, "nettoyage"), ("Qwen3-VL (description)", qwen, "diffusion:qwen"),
            ("Z-Image-Turbo (image)", zimage, "diffusion:zimage"),
            ("Z-Image-Turbo, 2ᵉ image (à chaud)", zimage_chaud, "diffusion:zimage"),
            ("FLUX.2 klein 4B (personnage d'après une référence)", klein, "diffusion:personnages"),
            ("Photos (Real-ESRGAN, GFPGAN, BiRefNet)", photo, "diffusion:photo_qualite"),
            ("Wan 2.2 (vidéo courte d'après une image)", film, "diffusion:video"),
            ("Stable Audio Open (bruitage)", audio, "diffusion:bruitages"),
            ("Hunyuan3D-2 (forme + texture)", forme, "diffusion:texture3d")]


def complet():
    """Générateur : essai réel de chaque moteur. (rapport en Markdown, fichier du rapport ou None)."""
    r = Rapport("Essai complet des moteurs de Studio Voix")
    _systeme(r)
    d = cfg.DATA_DIR / "diagnostic" / f"{datetime.now():%Y%m%d_%H%M%S}_complet"
    d.mkdir(parents=True, exist_ok=True)
    voix = next((str(cfg.VOICES_DIR / f"{v}.wav") for v in list_voices()), None)
    r.section("Essais (une génération courte par moteur ; « à chaud » = modèles déjà chargés)")
    yield r.texte() + "\n\n⏳ Essais en cours : compte 10 à 20 minutes…", None
    with _galerie_redirigee(d):
        for nom, fonction, cle in _etapes(d, voix):
            if cle and (retraits.retire(cle) or (cle == "diffusion:texture3d" and retraits.retire("diffusion:forme3d"))):
                r.ligne(None, nom, RETIRE)
                yield r.texte() + "\n\n⏳ Essais en cours…", None
                continue
            t0 = time.time()
            suivi = SuiviGPU()
            try:
                with suivi:
                    detail = fonction()
                ok = True
            except gr.Error as e:
                ok, detail = False, str(getattr(e, "message", e))[:800]
            except Exception as e:  # noqa: BLE001 - le rapport doit tout consigner
                ok = False
                detail = f"{type(e).__name__} : {e}\n```\n{''.join(traceback.format_exc().splitlines(True)[-6:])}```"
            gpu = carte_graphique()
            r.ligne(ok, nom, f"{detail} — {time.time() - t0:.0f} s" + (f" — pendant : {suivi.resume()}"
                                                                         if suivi.resume() else "")
                    + (f" — après : {gpu.split('mémoire ')[-1]}" if gpu else ""))
            yield r.texte() + "\n\n⏳ Essais en cours…", None
    fichier = r.enregistrer(d)
    yield (r.texte() + f"\n\nRapport et fichiers produits dans `{d}` : envoie `rapport.txt` pour un dépannage.",
           fichier)
