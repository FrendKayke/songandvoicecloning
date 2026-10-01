# Analyse de installer.ps1 avec PowerShell 7 : syntaxe, puis compatibilité avec Windows PowerShell 5.1
# (PSScriptAnalyzer : syntaxe, commandes et types disponibles dans le profil Windows 10 / PowerShell 5.1).
#   pwsh -NoProfile -File tests/installateur/analyse.ps1
$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot '..\..\installer.ps1'
$erreurs = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($script, [ref]$null, [ref]$erreurs)
if ($erreurs.Count) { $erreurs | ForEach-Object { Write-Host "Syntaxe : ligne $($_.Extent.StartLineNumber) : $($_.Message)" }; exit 1 }
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
$problemes = @(Invoke-ScriptAnalyzer -Path $script -Settings $regles `
        -IncludeRule PSUseCompatibleSyntax, PSUseCompatibleCommands, PSUseCompatibleTypes)
$problemes | ForEach-Object { Write-Host "Ligne $($_.Line) : $($_.Message)" }
Write-Host "installer.ps1 : syntaxe correcte, $($problemes.Count) problème(s) de compatibilité avec PowerShell 5.1"
exit ([int]($problemes.Count -gt 0))
