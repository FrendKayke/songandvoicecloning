"""Serveur ACE-Step géré par l'application : démarré à la demande, arrêté pour libérer la carte graphique.

ACE-Step garde ~8 Go de mémoire graphique tant que son serveur tourne. Avant un moteur gourmand (bruitages, 3D,
illustrations, synthèse vocale, nettoyage, entraînement RVC), `liberer_gpu()` l'arrête ; la génération musicale
suivante le relance (`assurer()`), le premier appel rechargeant ses modèles.

Commande lancée : celle de start_api_server.bat d'ACE-Step (`uv run --no-sync acestep-api --host … --port …`,
depuis son dossier), sans fenêtre, journal dans <ACE-Step>/serveur.log. Sous Windows, le serveur est rattaché à un
« job » qui le termine avec l'application ; on arrête toujours tout l'arbre de processus (uv → lanceur → python) :
arrêter seulement python laisserait start_api_server.bat relancer le serveur.
Un serveur démarré autrement (ancien lancer.bat, à la main) est repris : on retrouve le processus qui écoute le port.
"""
import atexit
import os
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

import gradio as gr
import requests

from . import config as cfg

_processus = None  # serveur lancé par l'application
LIBERATION_AUTO = os.environ.get("STUDIOVOIX_LIBERER_GPU", "1") != "0"

# Modèle de langage d'ACE-Step (la « réflexion » : structure, durée, tonalité, codes audio guidant le générateur).
# ACE-Step choisit d'après la mémoire de la carte (acestep/gpu_config.py) : jusqu'à 12 Go inclus (`<= 12`, catégorie
# « tier4 »), seul le plus petit modèle (0.6B) est permis et le générateur quitte la carte pendant la réflexion ;
# au-delà (« tier5 », 12–16 Go), modèle 1.7B (celui de son téléchargement par défaut, conseillé par sa
# documentation : « turbo + 1.7B ») et générateur gardé sur la carte. Une RTX 4070 annonce 11,99 Go : elle tombait
# dans la première catégorie. MAX_CUDA_VRAM (variable de test d'ACE-Step) annonce 12,5 Go ; supérieure à la mémoire
# réelle, elle ne pose aucune limite (set_per_process_memory_fraction n'est appliqué qu'en dessous) et la place
# réservée au modèle de langage reste calculée sur la mémoire réellement libre (get_lm_gpu_memory_ratio).
MODELES_LM = {"1.7B : meilleure musique (conseillé sur 12 Go)": "1.7B", "0.6B : plus léger": "0.6B"}
MODELE_LM = os.environ.get("STUDIOVOIX_ACE_LM", "1.7B")


def _hote_port():
    u = urlparse(cfg.ACESTEP_URL)
    return u.hostname or "127.0.0.1", u.port or 8001


def local():
    """Serveur sur cette machine (sinon on ne le démarre ni ne l'arrête)."""
    return _hote_port()[0] in ("127.0.0.1", "localhost", "::1")


def repond(timeout=3):
    try:
        return requests.get(f"{cfg.ACESTEP_URL}/health", timeout=timeout).ok
    except requests.RequestException:
        return False


def journal():
    return cfg.ACESTEP_DIR / "serveur.log"


def fin_du_journal(n=15):
    try:
        lignes = journal().read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lignes[-n:])


def _uv():
    exe = cfg.ENG_DIR / "uv" / ("uv.exe" if os.name == "nt" else "uv")
    return str(exe) if exe.exists() else (shutil.which("uv") or "uv")


def commande():
    hote, port = _hote_port()
    return [_uv(), "run", "--no-sync", "acestep-api", "--host", hote, "--port", str(port)]


def memoire_gpu_go():
    """Mémoire totale de la carte graphique en Gio (nvidia-smi), ou None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        return float(r.stdout.split()[0]) / 1024 if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def env_serveur():
    """Variables du serveur : modèle de langage choisi (voir MODELES_LM)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    if "ACESTEP_LM_MODEL_PATH" not in os.environ:
        env["ACESTEP_LM_MODEL_PATH"] = f"acestep-5Hz-lm-{MODELE_LM}"
    total = memoire_gpu_go()
    if MODELE_LM == "1.7B" and "MAX_CUDA_VRAM" not in os.environ and total and 11.0 <= total <= 12.05:
        env["MAX_CUDA_VRAM"] = "12.5"
    return env


def regler_modele_lm(libelle):
    """Change le modèle de langage : le serveur est arrêté, la prochaine chanson le relance avec ce modèle."""
    global MODELE_LM
    MODELE_LM = MODELES_LM.get(libelle, MODELE_LM)
    arrete = arreter() if (_vivant() or (local() and repond(1))) else False
    return (f"✅ Modèle de langage d'ACE-Step : {MODELE_LM}."
            + (" Serveur arrêté : il redémarrera avec ce modèle à la prochaine chanson." if arrete else "")
            + (" Le premier lancement télécharge ce modèle s'il manque." if MODELE_LM == "0.6B" else ""))


def libelle_modele_lm():
    return next((lib for lib, v in MODELES_LM.items() if v == MODELE_LM), next(iter(MODELES_LM)))


def _vivant():
    return _processus is not None and _processus.poll() is None


def _job_windows(processus):
    """Rattache le serveur à un job Windows « tué à la fermeture » : si l'application s'arrête (fenêtre fermée),
    le serveur s'arrête aussi au lieu de garder la mémoire graphique en arrière-plan. Sans effet en cas d'échec."""
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE

        class Base(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Io(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

        class Etendue(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Base), ("IoInfo", Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        info = Etendue()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        # 9 = JobObjectExtendedLimitInformation
        if job and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            if k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(processus._handle))):
                return job  # garder la référence : le job vit tant que l'application vit
    except Exception:  # noqa: BLE001 - confort seulement : le serveur marche sans job
        pass
    return None


def demarrer():
    """Lance le serveur s'il ne répond pas et n'est pas déjà en train de démarrer. Renvoie True s'il a été lancé."""
    global _processus
    if not local() or repond(1) or _vivant():
        return False
    if not cfg.ACESTEP_DIR.exists():
        raise gr.Error(f"ACE-Step n'est pas installé ({cfg.ACESTEP_DIR} introuvable). Relance INSTALLER.bat.")
    log = open(journal(), "w", encoding="utf-8", errors="replace")
    options = {"cwd": cfg.ACESTEP_DIR, "stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
               "env": env_serveur()}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    try:
        _processus = subprocess.Popen(commande(), **options)
    finally:
        log.close()  # le processus garde sa propre copie du descripteur
    if os.name == "nt":
        _processus.job = _job_windows(_processus)
    return True


def assurer(progress=None, timeout=15 * 60):
    """Serveur prêt à répondre (démarré si besoin ; le premier démarrage charge ses modèles).
    Les moteurs résidents (diffusion, Chatterbox…) sont fermés d'abord : ACE-Step (~8 Go) puis Demucs et Seed-VC
    ont besoin de la carte et de la mémoire vive qu'ils gardent."""
    from . import residents

    if residents.arreter_tous() and progress:
        progress(0.01, desc="Libération de la carte graphique : fermeture du moteur ouvert…")
    if repond():
        return
    demarrer()
    t0 = time.time()
    while time.time() - t0 < timeout:
        if repond():
            return
        if _processus is not None and _processus.poll() is not None and not repond(1):
            raise gr.Error("Le serveur ACE-Step s'est arrêté au démarrage. Fin de son journal "
                           f"({journal()}) :\n{fin_du_journal()}")
        if progress:
            progress(0.02, desc=f"Démarrage du serveur ACE-Step… ({int(time.time() - t0)} s)")
        time.sleep(3)
    raise gr.Error(f"Le serveur ACE-Step ne répond pas. Fin de son journal ({journal()}) :\n{fin_du_journal()}")


def _tuer_arbre(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


# Processus qui écoute le port, puis on remonte ses parents tant qu'ils font partie du lancement d'ACE-Step
# (lanceur acestep-api.exe, uv.exe, cmd.exe de start_api_server.bat) sans jamais atteindre l'application.
_REPRISE_WINDOWS = (
    "$p = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
    "Select-Object -First 1 -ExpandProperty OwningProcess; if (-not $p) {{ exit 3 }}; $racine = $p; "
    "for ($i = 0; $i -lt 6; $i++) {{ "
    "$proc = Get-CimInstance Win32_Process -Filter \"ProcessId=$racine\"; if (-not $proc) {{ break }}; "
    "$parent = Get-CimInstance Win32_Process -Filter \"ProcessId=$($proc.ParentProcessId)\"; "
    "if (-not $parent -or $parent.ProcessId -eq {moi} -or "
    "@('uv.exe', 'acestep-api.exe', 'cmd.exe') -notcontains $parent.Name) {{ break }}; "
    "if ($parent.Name -eq 'cmd.exe' -and $parent.CommandLine -notmatch 'start_api_server|acestep') {{ break }}; "
    "$racine = $parent.ProcessId }}; taskkill /PID $racine /T /F | Out-Null; exit 0"
)


def _arreter_serveur_externe():
    if os.name != "nt":
        return False
    script = _REPRISE_WINDOWS.format(port=_hote_port()[1], moi=os.getpid())
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                       capture_output=True, timeout=60)
    return r.returncode == 0


def arreter(attente=30):
    """Arrête le serveur (le nôtre ou un serveur repris) et attend que le port soit libéré. True si arrêté."""
    global _processus
    if not local():
        return False
    if _vivant():
        _tuer_arbre(_processus.pid)
        try:
            _processus.wait(timeout=attente)
        except subprocess.TimeoutExpired:
            pass
    elif repond(1):
        if not _arreter_serveur_externe():
            return False
    else:
        return False
    _processus = None
    t0 = time.time()
    while repond(1) and time.time() - t0 < attente:
        time.sleep(0.5)
    return not repond(1)


def liberer_gpu(progress=None):
    """À appeler avant un moteur gourmand : arrête ACE-Step s'il tourne (réglage « libération automatique »)."""
    if not (LIBERATION_AUTO and local()) or not (_vivant() or repond(1)):
        return False
    if progress:
        progress(0.01, desc="Libération de la carte graphique : arrêt du serveur ACE-Step…")
    return arreter()


def regler_liberation(active):
    global LIBERATION_AUTO
    LIBERATION_AUTO = bool(active)
    return ("✅ ACE-Step sera arrêté automatiquement avant les autres moteurs, puis relancé à la chanson suivante."
            if LIBERATION_AUTO else
            "⚠️ Libération automatique désactivée : ferme toi-même ACE-Step (bouton ci-dessous) si la mémoire manque.")


def etat():
    if not local():
        return f"Serveur ACE-Step distant ({cfg.ACESTEP_URL}) : non géré par l'application."
    if repond(1):
        return "🟢 Serveur ACE-Step en marche (il occupe ~8 Go de mémoire graphique)."
    if _vivant():
        return "🟡 Serveur ACE-Step en cours de démarrage."
    return "⚪ Serveur ACE-Step arrêté : il démarrera à la prochaine génération musicale."


def arreter_depuis_interface():
    arreter()
    return etat()


@atexit.register
def _a_la_sortie():
    if _vivant():
        _tuer_arbre(_processus.pid)


if __name__ == "__main__":  # python -m studiovoix.serveur_acestep arreter|demarrer
    {"arreter": arreter, "demarrer": demarrer}[sys.argv[1]]()
