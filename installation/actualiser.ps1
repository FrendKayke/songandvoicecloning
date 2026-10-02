# Studio Voix - actualisation du code sans Git (appelé par ACTUALISER.bat) : l'archive du dépôt GitHub est
# téléchargée puis copiée par-dessus le dossier de l'application, sans rien supprimer, et sans toucher à data\,
# .venv\ ni .git\. Code de sortie : 0 = à jour, 3 = à jour et les dépendances ou les modèles ont changé
# (l'installateur est à relancer), 1 = échec.
param([Parameter(Mandatory = $true)][string]$App)
$ErrorActionPreference = 'Stop'
$App = (Resolve-Path $App).Path.TrimEnd('\')
$Url = 'https://github.com/FrendKayke/songandvoicecloning/archive/refs/heads/main.zip'
$Temp = [IO.Path]::GetTempPath()
$Zip = Join-Path $Temp 'studiovoix_main.zip'
$Dest = Join-Path $Temp 'studiovoix_main'

# Empreinte de ce qui demande l'installateur (dépendances, installateur, listes de modèles), calculée sur les
# fichiers de la version téléchargée : un fichier local retiré depuis sur GitHub ne compte pas (rien n'est supprimé)
function Get-Empreinte([string]$Racine, [string[]]$Relatifs) {
    ($Relatifs | ForEach-Object {
        $f = Join-Path $Racine $_
        $_ + ':' + $(if (Test-Path $f) { (Get-FileHash $f -Algorithm SHA256).Hash } else { 'absent' })
    }) -join ';'
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Write-Host 'Téléchargement de la dernière version…'
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Zip
    if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
    Expand-Archive -Path $Zip -DestinationPath $Dest -Force
} catch {
    Write-Host "Téléchargement impossible : $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
$Source = (Get-ChildItem $Dest -Directory | Select-Object -First 1).FullName
$relatifs = @('requirements.txt', 'installer.ps1') | Where-Object { Test-Path (Join-Path $Source $_) }
if (Test-Path (Join-Path $Source 'installation')) {
    $relatifs += @(Get-ChildItem (Join-Path $Source 'installation') -Filter '*.txt' -File | Sort-Object FullName |
            ForEach-Object { $_.FullName.Substring($Source.Length + 1) })
}
$avant = Get-Empreinte $App $relatifs
$apres = Get-Empreinte $Source $relatifs
# /E : sous-dossiers ; jamais /MIR (il supprimerait des fichiers) ; data, .venv et .git exclus
robocopy $Source $App /E /XD data .venv .git __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
$code = $LASTEXITCODE
Remove-Item $Zip, $Dest -Recurse -Force -ErrorAction SilentlyContinue
if ($code -ge 8) {
    Write-Host "Copie incomplète (robocopy, code $code) : un fichier est peut-être ouvert. Ferme Studio Voix et relance." -ForegroundColor Red
    exit 1
}
if ($avant -ne $apres) { exit 3 }
exit 0
