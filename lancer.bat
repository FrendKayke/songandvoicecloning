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

curl -s -f -o nul -m 2 %ACESTEP_URL%/health
if not errorlevel 1 goto serveur_ok
echo Demarrage du serveur ACE-Step dans une fenetre separee - ne la ferme pas.
start "ACE-Step - ne pas fermer" /min /D "%ENG%\ace-step" cmd /k start_api_server.bat

:serveur_ok
".venv\Scripts\python.exe" studio_voix.py
pause
exit /b

:pas_installe
echo L'installation n'est pas terminee. Lance d'abord INSTALLER.bat.
pause
