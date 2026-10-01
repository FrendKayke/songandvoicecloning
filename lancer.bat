@echo off
cd /d "%~dp0"
set "ENG=%~d0\StudioVoix"
if not exist "%ENG%\installation.ok" goto pas_installe
if not exist ".venv\Scripts\python.exe" goto pas_installe

set "PATH=%ENG%\uv;%PATH%"
set "UV_CACHE_DIR=%ENG%\uv-cache"
set "UV_PYTHON_INSTALL_DIR=%ENG%\python"
set "TORCH_HOME=%ENG%\torch-cache"
set "HF_HOME=%ENG%\hf-home"
set "PKUSEG_HOME=%ENG%\chatterbox\pkuseg"
set "U2NET_HOME=%ENG%\diffusion\u2net"
set "PYTHONIOENCODING=utf-8"
set "STUDIOVOIX_MOTEURS=%ENG%"
set "ACESTEP_URL=http://127.0.0.1:8001"
set "SEEDVC_DIR="
set "SEEDVC_PYTHON="
set "ACESTEP_DIR="
set "ACESTEP_PYTHON="
set "CHATTERBOX_DIR="
set "CHATTERBOX_PYTHON="
set "NETTOYAGE_DIR="
set "NETTOYAGE_PYTHON="
set "RVC_DIR="
set "RVC_PYTHON="
set "DIFFUSION_DIR="
set "DIFFUSION_PYTHON="

rem Le serveur ACE-Step est demarre (et arrete pour liberer la carte graphique) par l'application elle-meme,
rem sans fenetre separee : son journal est dans %ENG%\ace-step\serveur.log
echo Studio Voix demarre. Ferme cette fenetre pour tout arreter (ACE-Step compris).
".venv\Scripts\python.exe" studio_voix.py
pause
exit /b

:pas_installe
echo L'installation n'est pas terminee. Lance d'abord INSTALLER.bat.
pause
