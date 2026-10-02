@echo off
rem Studio Voix - actualise le code depuis GitHub, sans relancer tout l'installateur (rapide).
rem Tes creations (dossier data) et les moteurs (dossier StudioVoix) ne sont pas touches.
rem Avec Git : git pull. Sans Git : l'archive du code est telechargee et copiee par-dessus.
rem Si la nouvelle version ajoute des dependances ou des modeles, il propose de lancer l'installateur.
rem cmd.exe lit un .bat au fur et a mesure : comme la mise a jour peut remplacer ce fichier, on l'execute
rem depuis une copie.
if not "%~1"=="--copie" (
    copy /y "%~f0" "%TEMP%\studiovoix_actualiser.bat" >nul
    "%TEMP%\studiovoix_actualiser.bat" --copie "%~dp0"
)
set "APP=%~2"
cd /d "%APP%"
echo === Actualisation de Studio Voix depuis GitHub ===
echo Ferme d'abord Studio Voix (la fenetre de lancer.bat).
pause

where git >nul 2>nul
if errorlevel 1 goto archive
if not exist ".git" goto archive

for /f %%i in ('git rev-parse HEAD') do set "AVANT=%%i"
git pull --ff-only origin main
if not errorlevel 1 goto verifier

echo.
echo L'actualisation a echoue : des fichiers du code ont ete modifies sur ce PC,
echo ou la connexion a GitHub n'a pas abouti (dans ce cas, reessaie plus tard).
echo Tes creations (dossier data) et les moteurs ne seront pas touches.
choice /c ON /m "Remplacer le code modifie par la version de GitHub"
if errorlevel 2 goto fin
git fetch origin main
if errorlevel 1 goto echec
git reset --hard origin/main
if errorlevel 1 goto echec

:verifier
git diff --quiet %AVANT% HEAD -- requirements.txt installer.ps1 installation/*.txt
if errorlevel 1 goto dependances
goto a_jour

:archive
echo Git n'est pas installe ou ce dossier n'est pas relie a GitHub : telechargement de l'archive du code.
powershell -NoProfile -ExecutionPolicy Bypass -File "%APP%installation\actualiser.ps1" -App "%APP%."
if errorlevel 3 goto dependances
if errorlevel 1 goto echec
goto a_jour

:dependances
echo.
echo Cette version ajoute ou change des dependances ou des modeles.
choice /c ON /m "Lancer l'installateur maintenant (seules les etapes modifiees sont refaites)"
if errorlevel 2 goto rappel
powershell -NoProfile -ExecutionPolicy Bypass -File "%APP%installer.ps1"
goto fin

:rappel
echo Pense a lancer METTRE_A_JOUR.bat avant d'utiliser les nouveautes.
goto fin

:a_jour
echo.
echo Code a jour. Relance Studio Voix avec lancer.bat.
goto fin

:echec
echo La connexion a GitHub a echoue. Verifie ta connexion internet et reessaie.

:fin
pause
