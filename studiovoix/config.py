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

DATA_DIR = APP_DIR / "data"
VOICES_DIR = DATA_DIR / "voices"
SONGS_DIR = DATA_DIR / "songs"
for d in (VOICES_DIR, SONGS_DIR):
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
