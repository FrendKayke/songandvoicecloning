@echo off
rem Ferme les applications gourmandes en memoire vive, puis lance Studio Voix.
rem Les modeles (jusqu'a 20 Go) ont besoin de la memoire vive : avec trop de logiciels ouverts,
rem Windows compresse ou deplace la memoire et les generations ralentissent beaucoup.
rem Pour changer la liste, modifie la ligne "set APPLIS=" ci-dessous (noms des .exe).
cd /d "%~dp0"
set "APPLIS=chrome.exe Discord.exe steam.exe steamwebhelper.exe StreamDeck.exe RzSynapse.exe SteelSeriesGG.exe AdobeCollabSync.exe vlc.exe wallpaper32.exe"

echo Fermeture des applications gourmandes...
for %%A in (%APPLIS%) do (
    tasklist /FI "IMAGENAME eq %%A" 2>nul | find /I "%%A" >nul && (
        echo   - %%A
        taskkill /IM %%A >nul 2>&1
    )
)
rem Laisse 3 secondes aux applications pour se fermer proprement, puis force celles qui restent
timeout /t 3 /nobreak >nul
for %%A in (%APPLIS%) do taskkill /F /IM %%A >nul 2>&1

echo.
call "%~dp0lancer.bat"
