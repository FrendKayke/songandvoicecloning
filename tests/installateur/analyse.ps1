# Analyse des scripts PowerShell livrés (installer.ps1, installation/actualiser.ps1) avec PowerShell 7 : syntaxe,
# puis compatibilité avec Windows PowerShell 5.1 (PSScriptAnalyzer : syntaxe, commandes et types disponibles dans le
# profil Windows 10 / PowerShell 5.1).
#   pwsh -NoProfile -File tests/installateur/analyse.ps1
$ErrorActionPreference = 'Stop'
$scripts = @((Join-Path $PSScriptRoot '..\..\installer.ps1'), (Join-Path $PSScriptRoot '..\..\installation\actualiser.ps1'))
if (-not (Get-Module -ListAvailable PSScriptAnalyzer)) {
    Install-Module PSScriptAnalyzer -Scope CurrentUser -Force -AcceptLicense
}
$profil = 'win-48_x64_10.0.17763.0_5.1.17763.316_x64_4.0.30319.42000_framework'
$regles = @{ Rules = @{
        PSUseCompatibleSyntax   = @{ Enable = $true; TargetVersions = @('5.1') }
        PSUseCompatibleCommands = @{ Enable = $true; TargetProfiles = @($profil) }
        PSUseCompatibleTypes    = @{ Enable = $true; TargetProfiles = @($profil) }
    }
}
$total = 0
foreach ($script in $scripts) {
    $nom = Split-Path $script -Leaf
    $erreurs = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($script, [ref]$null, [ref]$erreurs)
    if ($erreurs.Count) { $erreurs | ForEach-Object { Write-Host "$nom, syntaxe : ligne $($_.Extent.StartLineNumber) : $($_.Message)" }; exit 1 }
    $problemes = @(Invoke-ScriptAnalyzer -Path $script -Settings $regles `
            -IncludeRule PSUseCompatibleSyntax, PSUseCompatibleCommands, PSUseCompatibleTypes)
    $problemes | ForEach-Object { Write-Host "$nom, ligne $($_.Line) : $($_.Message)" }
    Write-Host "$nom : syntaxe correcte, $($problemes.Count) problème(s) de compatibilité avec PowerShell 5.1"
    $total += $problemes.Count
}
exit ([int]($total -gt 0))
