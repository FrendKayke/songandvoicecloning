@echo off
rem Studio Voix - mise a jour : recupere la derniere version sur GitHub, puis relance l'installateur
rem (seules les etapes nouvelles ou modifiees sont refaites). Tes creations (dossier data) ne sont pas touchees.
rem cmd.exe lit un .bat au fur et a mesure : comme git pull peut remplacer ce fichier, on l'execute depuis une copie.
if not "%~1"=="--copie" (
    copy /y "%~f0" "%TEMP%\studiovoix_maj.bat" >nul
    "%TEMP%\studiovoix_maj.bat" --copie "%~dp0"
)
set "APP=%~2"
cd /d "%APP%"
echo === Mise a jour de Studio Voix ===
echo Ferme d'abord Studio Voix (la fenetre de lancer.bat).
pause

where git >nul 2>nul
if errorlevel 1 goto pas_de_git
if not exist ".git" goto pas_de_depot

git pull --ff-only origin main
if not errorlevel 1 goto installer

echo.
echo La mise a jour automatique a echoue : des fichiers du code ont ete modifies sur ce PC,
echo ou la connexion a GitHub n'a pas abouti (dans ce cas, reessaie plus tard).
echo Tes creations (voix, chansons, bruitages, modeles 3D dans data) et les moteurs ne seront pas touches.
choice /c ON /m "Remplacer le code modifie par la version de GitHub"
if errorlevel 2 goto fin
git fetch origin main
if errorlevel 1 goto echec
git reset --hard origin/main
if errorlevel 1 goto echec

:installer
echo.
echo Code a jour. Lancement de l'installateur (etapes nouvelles ou modifiees seulement).
powershell -NoProfile -ExecutionPolicy Bypass -File "%APP%installer.ps1"
goto fin

:pas_de_git
echo Git est introuvable. Installe-le depuis https://git-scm.com/download/win puis relance ce fichier.
goto fin

:pas_de_depot
echo Ce dossier n'est pas relie a GitHub (pas de dossier .git). Dans PowerShell, ici :
echo   git init
echo   git remote add origin https://github.com/FrendKayke/songandvoicecloning.git
echo   git fetch origin main
echo   git checkout -f -B main origin/main
goto fin

:echec
echo La connexion a GitHub a echoue. Verifie ta connexion internet et reessaie.

:fin
pause
