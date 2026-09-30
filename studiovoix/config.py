"""Configuration : chemins, variables d'environnement et constantes partagées."""
import os
from pathlib import Path

# Dossier de l'application (celui qui contient studio_voix.py)
APP_DIR = Path(__file__).resolve().parent.parent
# Les moteurs sont installés par INSTALLER.bat dans un dossier au chemin court
# (évite la limite de 260 caractères de Windows) : <lecteur de l'appli>:\StudioVoix
ENG_DIR = Path(os.environ.get("STUDIOVOIX_MOTEURS", str(Path(APP_DIR.anchor) / "StudioVoix")))


def venv_python(root: Path) -> str:
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return str(root / ".venv" / sub)


ACESTEP_URL = os.environ.get("ACESTEP_URL", "http://127.0.0.1:8001").rstrip("/")
ACESTEP_DIR = Path(os.environ.get("ACESTEP_DIR", str(ENG_DIR / "ace-step")))
ACESTEP_PYTHON = os.environ.get("ACESTEP_PYTHON", venv_python(ACESTEP_DIR))
SEEDVC_DIR = Path(os.environ.get("SEEDVC_DIR", str(ENG_DIR / "seed-vc")))
SEEDVC_PYTHON = os.environ.get("SEEDVC_PYTHON", venv_python(SEEDVC_DIR))  # sert aussi à Demucs
CHATTERBOX_DIR = Path(os.environ.get("CHATTERBOX_DIR", str(ENG_DIR / "chatterbox")))
CHATTERBOX_PYTHON = os.environ.get("CHATTERBOX_PYTHON", venv_python(CHATTERBOX_DIR))
NETTOYAGE_DIR = Path(os.environ.get("NETTOYAGE_DIR", str(ENG_DIR / "nettoyage")))
NETTOYAGE_PYTHON = os.environ.get("NETTOYAGE_PYTHON", venv_python(NETTOYAGE_DIR))
RVC_DIR = Path(os.environ.get("RVC_DIR", str(ENG_DIR / "rvc")))  # Applio : code, modèles de base, logs/<modèle>
RVC_PYTHON = os.environ.get("RVC_PYTHON", venv_python(RVC_DIR))
DIFFUSION_DIR = Path(os.environ.get("DIFFUSION_DIR", str(ENG_DIR / "diffusion")))  # Hunyuan3D, SDXL, Stable Audio, Qwen
DIFFUSION_PYTHON = os.environ.get("DIFFUSION_PYTHON", venv_python(DIFFUSION_DIR))
# Scripts exécutés dans l'environnement d'un moteur (jamais importés par l'application)
MOTEURS_DIR = APP_DIR / "moteurs"

DATA_DIR = APP_DIR / "data"
VOICES_DIR = DATA_DIR / "voices"
SONGS_DIR = DATA_DIR / "songs"
TTS_DIR = DATA_DIR / "tts"  # textes lus par la synthèse vocale
CLEAN_DIR = DATA_DIR / "nettoyage"  # voix nettoyées (avant enregistrement dans la bibliothèque)
GAMES_DIR = DATA_DIR / "jeux"  # bandes-son de jeu : jeux/<projet>/<situation>/<horodatage>/
SFX_DIR = DATA_DIR / "bruitages"
MODELS3D_DIR = DATA_DIR / "3d"
for d in (VOICES_DIR, SONGS_DIR, TTS_DIR, CLEAN_DIR, GAMES_DIR, SFX_DIR, MODELS3D_DIR):
    d.mkdir(parents=True, exist_ok=True)

SR = 44100  # fréquence de sortie (le modèle chanté de Seed-VC produit du 44,1 kHz)

LANGUES = {
    "Français": "fr",
    "Anglais": "en",
    "Espagnol": "es",
    "Italien": "it",
    "Allemand": "de",
    "Portugais": "pt",
    "Japonais": "ja",
    "Coréen": "ko",
    "Chinois": "zh",
}
