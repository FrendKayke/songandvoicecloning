@echo off
rem File de taches pour la nuit : glisse un fichier de taches (.txt) sur ce fichier.
rem Sans fichier : taches_nuit.txt (cree avec un exemple s'il n'existe pas, puis ouvert dans le Bloc-notes).
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
set "ONNXTR_CACHE_DIR=%ENG%\ocr"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
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
chcp 65001 >nul

set "TACHES=%~1"
if "%TACHES%"=="" set "TACHES=%~dp0taches_nuit.txt"
if exist "%TACHES%" goto verifier
".venv\Scripts\python.exe" -m studiovoix.nuit --exemple "%TACHES%"
echo.
echo Un exemple de file de taches a ete cree : %TACHES%
echo Modifie-le dans le Bloc-notes, enregistre, puis relance NUIT.bat.
start "" notepad "%TACHES%"
pause
exit /b

:verifier
".venv\Scripts\python.exe" -m studiovoix.nuit --verifier "%TACHES%"
if errorlevel 1 goto erreur
echo.
echo Ferme cette fenetre pour tout arreter. Lancement dans 10 secondes...
timeout /t 10
".venv\Scripts\python.exe" -m studiovoix.nuit "%TACHES%"
pause
exit /b

:erreur
echo.
echo Corrige le fichier de taches puis relance NUIT.bat.
pause
exit /b

:pas_installe
echo L'installation n'est pas terminee. Lance d'abord INSTALLER.bat.
pause
