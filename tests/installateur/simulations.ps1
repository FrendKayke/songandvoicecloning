# Remplace les commandes qui touchent au réseau, au disque ou à l'utilisateur, pour exécuter installer.ps1 sous
# Linux (pwsh) avec de faux uv.exe / python.exe (faux_outil.sh). Les appels sont journalisés dans $env:JOURNAL.
function global:Invoke-WebRequest { param($Uri, $OutFile, [switch]$UseBasicParsing)
    Add-Content $env:JOURNAL "TELECHARGEMENT $Uri"
    if ($Uri -like '*.whl' -or $Uri -like '*.pyd' -or $Uri -like '*.zip' -and $Uri -like '*uv-x86_64*') {
        Set-Content -Path $OutFile -Value 'binaire'; return
    }
    # Archive de dépôt GitHub : un dossier racine contenant un fichier qui porte l'adresse (pour suivre les versions)
    $t = Join-Path ([IO.Path]::GetTempPath()) ('z' + [guid]::NewGuid())
    New-Item -ItemType Directory "$t/depot-main" | Out-Null
    Set-Content "$t/depot-main/version.txt" $Uri
    Compress-Archive -Path "$t/depot-main" -DestinationPath $OutFile -Force
    Remove-Item $t -Recurse -Force
}
function global:Get-PSDrive { param($Name) [pscustomobject]@{ Name = 'D'; Free = 100GB } }
function global:Read-Host { param($p) Add-Content $env:JOURNAL "READ-HOST $p"; return $env:REPONSE }
