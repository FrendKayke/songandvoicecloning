# Studio Voix — installation complète, sans droits administrateur.
# Installe : uv (gestionnaire Python), Python 3.10 / 3.11 / 3.12, ACE-Step 1.5, Seed-VC, Demucs,
# Chatterbox (synthèse vocale), le nettoyage de voix (MossFormer2, VoiceFixer), RVC (Applio : entraînement
# d'un modèle de ta voix), le moteur de diffusion (Qwen3-VL, Stable Audio Open, Hunyuan3D-2, Z-Image, FLUX.2 klein, Wan 2.2 : bruitages,
# modèles 3D), tous leurs modèles, et l'environnement de l'application.
# Relançable : chaque étape terminée est sautée.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # sinon Invoke-WebRequest est très lent
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$App = $PSScriptRoot
# Chemin court sur le même disque que l'appli : évite la limite de 260 caractères de Windows
# (STUDIOVOIX_MOTEURS, la variable que lit aussi l'application, sert aux tests de l'installateur)
$Eng = if ($env:STUDIOVOIX_MOTEURS) { $env:STUDIOVOIX_MOTEURS } else { Join-Path ([IO.Path]::GetPathRoot($App)) 'StudioVoix' }
$UvDir = Join-Path $Eng 'uv'
$Uv = Join-Path $UvDir 'uv.exe'
$Ace = Join-Path $Eng 'ace-step'
$Sv = Join-Path $Eng 'seed-vc'
$SvPy = Join-Path $Sv '.venv\Scripts\python.exe'
$Cb = Join-Path $Eng 'chatterbox'
$CbSrc = Join-Path $Cb 'src'
$CbPy = Join-Path $Cb '.venv\Scripts\python.exe'
# Version de Chatterbox épinglée (Multilingual V3) : le paquet PyPI 0.1.7 ne contient pas encore V3
$CbCommit = '5de7a54aa4e5e2baadb0182dde554908b48b85c2'
$Nt = Join-Path $Eng 'nettoyage'
$NtPy = Join-Path $Nt '.venv\Scripts\python.exe'
$Rvc = Join-Path $Eng 'rvc'
$RvcPy = Join-Path $Rvc '.venv\Scripts\python.exe'
# Version d'Applio (RVC) épinglée : moteurs\rvc_voix.py reprend les arguments de ses scripts à ce commit
$RvcCommit = 'c7665ac9a305b3683570ed914f1577d53d4b75c4'
$Dif = Join-Path $Eng 'diffusion'
$DifPy = Join-Path $Dif '.venv\Scripts\python.exe'
$HySrc = Join-Path $Dif 'hunyuan3d'
# Hunyuan3D-2 épinglé, et binaires Windows précompilés de son rasteriseur de texture (dépôt de kijai,
# ComfyUI-Hunyuan3DWrapper, commit épinglé) : sans eux il faudrait Visual Studio et le CUDA Toolkit.
$HyCommit = 'f8db63096c8282cb27354314d896feba5ba6ff8a'
$KijaiCommit = '2609efa38f6a98292476f714839b7c1e5f9b699a'
$KijaiRaw = "https://raw.githubusercontent.com/kijai/ComfyUI-Hunyuan3DWrapper/$KijaiCommit"
$AppPy = Join-Path $App '.venv\Scripts\python.exe'
# Listes de dépendances de chaque environnement (lues aussi par la vérification automatique du dépôt)
$Listes = Join-Path $App 'installation'

# Tout reste sur ce disque (caches compris), rien d'important sur C:
$env:UV_CACHE_DIR = Join-Path $Eng 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Eng 'python'
$env:UV_PYTHON_PREFERENCE = 'only-managed'   # ignore le Python 3.14 du système
$env:TORCH_HOME = Join-Path $Eng 'torch-cache'
$env:HF_HOME = Join-Path $Eng 'hf-home'
$env:PKUSEG_HOME = Join-Path $Cb 'pkuseg'   # sinon spacy-pkuseg (Chatterbox) écrit dans ~\.pkuseg, sur C:
$env:U2NET_HOME = Join-Path $Dif 'u2net'    # sinon rembg (détourage) écrit dans ~\.u2net, sur C:
$env:PYTHONIOENCODING = 'utf-8'
$env:PATH = "$UvDir;$env:PATH"

function Step($n, $txt) { Write-Host ''; Write-Host "=== [$n/18] $txt ===" -ForegroundColor Cyan }
# Marqueurs d'étape : le fichier contient la « signature » de l'étape (empreinte de sa liste de dépendances, des
# versions épinglées, des modèles demandés). Si une mise à jour change la signature, l'étape est refaite.
# Un marqueur vide (installation d'avant les signatures) est adopté tel quel.
function Get-Signature([string[]]$parts) {
    $texte = ($parts | ForEach-Object {
            if ($_ -and (Test-Path -LiteralPath $_ -PathType Leaf)) { [IO.File]::ReadAllText($_) -replace "`r`n", "`n" } else { $_ }
        }) -join "`n--`n"
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $octets = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($texte)) } finally { $sha.Dispose() }
    return ([BitConverter]::ToString($octets) -replace '-', '').ToLowerInvariant()
}
function Test-Done([string]$marker, [string]$signature = '') {
    if (-not (Test-Path -LiteralPath $marker)) { return $false }
    $actuelle = ([IO.File]::ReadAllText($marker)).Trim()
    if (-not $signature) { return $true }
    if (-not $actuelle) {
        if ($signature) { [IO.File]::WriteAllText($marker, $signature) }
        return $true
    }
    if ($actuelle -eq $signature) { return $true }
    Write-Host 'Mise à jour : cette étape a changé depuis la dernière installation, elle est refaite.' -ForegroundColor Yellow
    return $false
}
# Moteurs et modèles retirés pour gagner de la place (Outils → Modèles → Espace disque) : leurs étapes sont sautées,
# sinon l'installateur les réinstallerait aussitôt. Une clé par ligne : chatterbox, nettoyage, rvc, diffusion,
# diffusion:<modèle> (qwen, bruitages, zimage, personnages, photo_detourage, photo_qualite, video,
# forme3d, texture3d).
function Get-Retires {
    $f = Join-Path $Eng 'moteurs-retires.txt'
    if (-not (Test-Path $f)) { return @() }
    return @(Get-Content $f -Encoding UTF8 | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
function Test-Retire([string]$cle) {
    $r = Get-Retires
    return ($r -contains $cle) -or ($cle.StartsWith('diffusion:') -and ($r -contains 'diffusion'))
}
function Write-Retire {
    Write-Host 'Retiré pour gagner de la place (Outils → Modèles → Espace disque pour le réinstaller).' -ForegroundColor DarkGray
}

function Done([string]$marker, [string]$signature = '') {
    New-Item -ItemType Directory -Force -Path (Split-Path $marker) | Out-Null
    [IO.File]::WriteAllText($marker, $signature)
}

function Run([string]$exe, [string[]]$argList, [string]$cwd = $null) {
    if ($cwd) { Push-Location $cwd }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'   # les outils écrivent leur progression sur stderr : ce n'est pas une erreur
    try {
        & $exe @argList
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
        if ($cwd) { Pop-Location }
    }
    if ($code -ne 0) { throw "Échec (code $code) de : $exe $($argList -join ' ')" }
}

# Bug connu de PyTorch 2.4.0 sous Windows : fbgemm.dll réclame libomp140.x86_64.dll (fourni avec
# Visual Studio). Même correctif que ComfyUI : on copie la DLL OpenMP livrée avec PyTorch sous ce nom.
function Repair-TorchOmp([string]$py, [string]$venv) {
    $torchLib = Join-Path $venv 'Lib\site-packages\torch\lib'
    $ErrorActionPreference = 'Continue'   # sous PowerShell 5.1, rediriger stderr avec 'Stop' lèverait une exception
    & $py -c 'import torch' 2>$null
    $importOk = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = 'Stop'
    if (-not $importOk) {
        $omp = Join-Path $torchLib 'libomp140.x86_64.dll'
        $iomp = Join-Path $torchLib 'libiomp5md.dll'
        if ((Test-Path $iomp) -and -not (Test-Path $omp)) {
            Write-Host 'Correctif PyTorch : ajout de libomp140.x86_64.dll' -ForegroundColor Yellow
            Copy-Item $iomp $omp
        }
    }
}

# Supprime un ancien environnement ; message clair s'il est verrouillé (application ouverte)
function Remove-Venv([string]$venv) {
    if (-not (Test-Path $venv)) { return }
    try { Remove-Item $venv -Recurse -Force }
    catch {
        throw ("Impossible de supprimer l'ancien environnement ($venv) : un fichier est utilisé. " +
            "Ferme la fenêtre de lancer.bat (et Studio Voix dans le navigateur), puis relance INSTALLER.bat.")
    }
}

function Get-Repo([string]$url, [string]$dest) {
    $ok = Join-Path $dest '.complet'
    if (Test-Done $ok $url) { Write-Host "Déjà présent : $dest"; return }
    # Dossier déjà là mais d'une autre version (mise à jour) : le nouveau code est copié par-dessus, sans rien
    # supprimer (le dossier d'Applio contient tes modèles RVC entraînés, celui d'ACE-Step ses modèles).
    $miseAJour = Test-Path (Join-Path $dest '*')
    $zip = Join-Path $Eng '_depot.zip'
    $tmp = Join-Path $Eng '_depot'
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    Write-Host "Téléchargement de $url"
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
    if ($miseAJour) {
        Write-Host "Mise à jour du code dans $dest (tes fichiers sont gardés)"
        Copy-Item -Path (Join-Path $inner.FullName '*') -Destination $dest -Recurse -Force
    } else {
        if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
        Move-Item $inner.FullName $dest
    }
    Remove-Item $tmp, $zip -Recurse -Force
    Done $ok $url
}

try {
    Write-Host 'Studio Voix — installation complète' -ForegroundColor Green
    Write-Host "Moteurs et modèles  : $Eng"
    Write-Host "Application         : $App"
    Write-Host 'Environ 55 à 60 Go à télécharger : compte une bonne heure selon ta connexion.'
    Write-Host 'Tu peux fermer et relancer INSTALLER.bat : les étapes finies seront sautées.'

    New-Item -ItemType Directory -Force -Path $Eng | Out-Null

    # --- Vérifications ---
    $drive = Get-PSDrive -Name $Eng.Substring(0, 1)
    $freeGo = [math]::Round($drive.Free / 1GB)
    Write-Host "Espace libre sur $($drive.Name): : $freeGo Go"
    if ($freeGo -lt 80) {
        Write-Host 'Attention : moins de 80 Go libres, l''installation complète risque de manquer de place.' -ForegroundColor Yellow
        if ((Read-Host 'Continuer quand même ? (o/n)') -ne 'o') { exit 1 }
    }
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1)
        Write-Host "Carte graphique : $gpu"
        $drv = 0.0
        [double]::TryParse((($gpu -split ',')[-1].Trim()), [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture, [ref]$drv) | Out-Null
        if ($drv -gt 0 -and $drv -lt 570.65) {
            Write-Host "Ton pilote NVIDIA ($drv) est trop ancien pour ACE-Step (CUDA 12.8)." -ForegroundColor Yellow
            Write-Host 'Mets-le à jour via GeForce Experience / l''application NVIDIA, puis relance.' -ForegroundColor Yellow
            if ((Read-Host 'Continuer quand même ? (o/n)') -ne 'o') { exit 1 }
        }
    } else {
        Write-Host 'nvidia-smi introuvable : vérifie que le pilote NVIDIA est installé.' -ForegroundColor Yellow
    }

    # --- 1. uv ---
    Step 1 'Gestionnaire Python (uv)'
    if (-not (Test-Path $Uv)) {
        $zip = Join-Path $Eng '_uv.zip'
        Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip -UseBasicParsing
        Expand-Archive -Path $zip -DestinationPath $UvDir -Force
        Remove-Item $zip -Force
    }
    Run $Uv @('--version')

    # --- 2. Code des moteurs ---
    Step 2 'Code d''ACE-Step, de Seed-VC, de Chatterbox, de RVC (Applio) et de Hunyuan3D-2'
    Get-Repo 'https://github.com/ace-step/ACE-Step-1.5/archive/refs/heads/main.zip' $Ace
    Get-Repo 'https://github.com/Plachtaa/seed-vc/archive/refs/heads/main.zip' $Sv
    if (-not (Test-Retire 'chatterbox')) {
        New-Item -ItemType Directory -Force -Path $Cb | Out-Null
        Get-Repo "https://github.com/resemble-ai/chatterbox/archive/$CbCommit.zip" $CbSrc
    }
    Get-Repo "https://github.com/IAHispano/Applio/archive/$RvcCommit.zip" $Rvc
    if (-not (Test-Retire 'diffusion')) {
        New-Item -ItemType Directory -Force -Path $Dif | Out-Null
        Get-Repo "https://github.com/Tencent-Hunyuan/Hunyuan3D-2/archive/$HyCommit.zip" $HySrc
    }

    # --- 3. Environnement ACE-Step (PyTorch CUDA 12.8, via sa propre config) ---
    # Python 3.11 : sous Windows, le pyproject d'ACE-Step n'installe le moteur rapide de son modèle de langage
    # (nano-vLLM : triton-windows et flash-attention, roues cp311) qu'en 3.11 ; en 3.12, choisi auparavant par uv,
    # le modèle de langage tourne sur la boucle PyTorch, bien plus lente, et la carte graphique attend.
    # Si l'installation en 3.11 échoue, repli en 3.12 (la musique marche, plus lentement).
    # Nouveau marqueur (.env-py311-ok) : les installations existantes, en 3.12, sont reconstruites une fois.
    Step 3 'Environnement ACE-Step (le plus long : PyTorch ~3 Go)'
    $m = Join-Path $Ace '.env-py311-ok'
    $sig = Get-Signature @('python 3.11, repli 3.12 ; uv sync')
    if (-not (Test-Done $m $sig)) {
        $versionPy = Join-Path $Ace '.python-version'   # lu aussi par « uv run » au lancement du serveur
        Remove-Venv (Join-Path $Ace '.venv')
        try {
            [IO.File]::WriteAllText($versionPy, "3.11`n")
            Run $Uv @('sync', '--python', '3.11') $Ace
        } catch {
            Write-Host "ACE-Step en Python 3.11 impossible ($($_.Exception.Message)) : installation en 3.12, le modèle de langage sera plus lent." -ForegroundColor Yellow
            Remove-Venv (Join-Path $Ace '.venv')
            [IO.File]::WriteAllText($versionPy, "3.12`n")
            Run $Uv @('sync', '--python', '3.12') $Ace
        }
        Done $m $sig
    } else { Write-Host 'Déjà fait.' }

    # --- 4. Modèles ACE-Step (~10 Go) ---
    Step 4 'Modèles ACE-Step (~10 Go)'
    $m = Join-Path $Ace '.modeles-ok'
    if (-not (Test-Path $m)) {
        Run $Uv @('run', '--no-sync', 'acestep-download', '--dir', (Join-Path $Ace 'checkpoints')) $Ace
        Done $m
    } else { Write-Host 'Déjà fait.' }

    # --- 5. Environnement Seed-VC + Demucs (Python 3.10, PyTorch 2.4 CUDA 12.4) ---
    Step 5 'Environnement Seed-VC et Demucs (Python 3.10)'
    $m = Join-Path $Sv '.env-ok'
    $sig = Get-Signature @('torch==2.4.0 torchaudio==2.4.0 cu124', (Join-Path $Listes 'seed-vc.txt'))
    if (-not (Test-Done $m $sig)) {
        $venv = Join-Path $Sv '.venv'
        Remove-Venv $venv
        Run $Uv @('venv', '--python', '3.10', $venv)
        Run $Uv @('pip', 'install', '--python', $SvPy, 'torch==2.4.0', 'torchaudio==2.4.0',
            '--index-url', 'https://download.pytorch.org/whl/cu124')
        Run $Uv @('pip', 'install', '--python', $SvPy, '-r', (Join-Path $Listes 'seed-vc.txt'))
        Repair-TorchOmp $SvPy $venv
        Run $SvPy @('-c', 'import torch; assert torch.cuda.is_available(), ''CUDA indisponible''; print(''GPU :'', torch.cuda.get_device_name(0))')
        Done $m $sig
    } else { Write-Host 'Déjà fait.' }

    # --- 6. Modèles Seed-VC (téléchargés par une mini-conversion de test) ---
    Step 6 'Modèles Seed-VC (conversion de test)'
    $m = Join-Path $Sv '.modeles-ok'
    if (-not (Test-Path $m)) {
        $t = Join-Path $Eng '_test'
        New-Item -ItemType Directory -Force -Path $t | Out-Null
        $src = Join-Path $t 'source.wav'; $tgt = Join-Path $t 'cible.wav'
        Run $SvPy @('-c', 'import sys, numpy as np, soundfile as sf; t = np.linspace(0, 3, 3 * 44100, endpoint=False); sf.write(sys.argv[1], (0.3 * np.sin(2 * np.pi * 220 * t)).astype(''float32''), 44100); sf.write(sys.argv[2], (0.3 * np.sin(2 * np.pi * 330 * t)).astype(''float32''), 44100)', $src, $tgt)
        Run $SvPy @('inference.py', '--source', $src, '--target', $tgt, '--output', $t,
            '--diffusion-steps', '4', '--f0-condition', 'True') $Sv
        Remove-Item $t -Recurse -Force
        Done $m
    } else { Write-Host 'Déjà fait.' }

    # --- 7. Modèle Demucs ---
    Step 7 'Modèle Demucs (séparation voix / musique)'
    $m = Join-Path $Sv '.demucs-ok'
    if (-not (Test-Path $m)) {
        Run $SvPy @('-c', 'from demucs.pretrained import get_model; get_model(''htdemucs''); print(''htdemucs prêt'')')
        Done $m
    } else { Write-Host 'Déjà fait.' }

    # --- 8. Environnement Chatterbox (Python 3.11, PyTorch 2.6 CUDA 12.4) ---
    Step 8 'Environnement Chatterbox, synthèse vocale (Python 3.11, PyTorch ~2,5 Go)'
    $m = Join-Path $Cb '.env-ok'
    $sig = Get-Signature @('torch==2.6.0 torchaudio==2.6.0 cu124', $CbCommit, (Join-Path $Listes 'chatterbox.txt'))
    if (Test-Retire 'chatterbox') { Write-Retire } elseif (-not (Test-Done $m $sig)) {
        $venv = Join-Path $Cb '.venv'
        Remove-Venv $venv
        Run $Uv @('venv', '--python', '3.11', $venv)
        # PyTorch CUDA d'abord : depuis PyPI, Windows recevrait la version sans carte graphique
        Run $Uv @('pip', 'install', '--python', $CbPy, 'torch==2.6.0', 'torchaudio==2.6.0',
            '--index-url', 'https://download.pytorch.org/whl/cu124')
        # Dépendances de Chatterbox (versions testées, voir installation\chatterbox.txt)
        Run $Uv @('pip', 'install', '--python', $CbPy, '-r', (Join-Path $Listes 'chatterbox.txt'))
        Run $Uv @('pip', 'install', '--python', $CbPy, '--no-deps', $CbSrc)
        Repair-TorchOmp $CbPy $venv
        Run $CbPy @('-c', 'import torch, perth; from chatterbox.mtl_tts import ChatterboxMultilingualTTS; assert perth.PerthImplicitWatermarker is not None, ''filigrane Perth indisponible''; assert torch.cuda.is_available(), ''CUDA indisponible''; print(''Chatterbox prêt, GPU :'', torch.cuda.get_device_name(0))')
        Done $m $sig
    } else { Write-Host 'Déjà fait.' }

    # --- 9. Modèles Chatterbox (~3,2 Go, chargés une fois sur le processeur pour vérification) ---
    Step 9 'Modèles Chatterbox Multilingual V3 (~3,2 Go)'
    $m = Join-Path $Cb '.modeles-ok'
    if (Test-Retire 'chatterbox') { Write-Retire } elseif (-not (Test-Path $m)) {
        Run $CbPy @((Join-Path $App 'moteurs\chatterbox_tts.py'), '--telecharger')
        Done $m
    } else { Write-Host 'Déjà fait.' }

    # --- 10. Environnement du nettoyage de voix (Python 3.11, même PyTorch 2.6 que Chatterbox) ---
    Step 10 'Environnement du nettoyage de voix (Python 3.11)'
    $m = Join-Path $Nt '.env-ok'
    $sig = Get-Signature @('torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 cu124 voicefixer==0.1.3', (Join-Path $Listes 'nettoyage.txt'))
    if (Test-Retire 'nettoyage') { Write-Retire } elseif (-not (Test-Done $m $sig)) {
        New-Item -ItemType Directory -Force -Path $Nt | Out-Null
        $venv = Join-Path $Nt '.venv'
        Remove-Venv $venv
        Run $Uv @('venv', '--python', '3.11', $venv)
        # Même PyTorch que Chatterbox : uv le reprend de son cache, sans nouveau téléchargement
        Run $Uv @('pip', 'install', '--python', $NtPy, 'torch==2.6.0', 'torchaudio==2.6.0', 'torchvision==0.21.0',
            '--index-url', 'https://download.pytorch.org/whl/cu124')
        # ClearerVoice (MossFormer2) avec ses dépendances ; matplotlib et torchlibrosa pour VoiceFixer
        Run $Uv @('pip', 'install', '--python', $NtPy, '-r', (Join-Path $Listes 'nettoyage.txt'))
        # VoiceFixer sans ses dépendances inutiles ici (streamlit, GitPython)
        Run $Uv @('pip', 'install', '--python', $NtPy, '--no-deps', 'voicefixer==0.1.3')
        Repair-TorchOmp $NtPy $venv
        # Surtout pas « import voicefixer » ici : il téléchargerait ses modèles dans le dossier personnel (C:)
        Run $NtPy @('-c', 'import importlib.util, torch, clearvoice; assert importlib.util.find_spec(''voicefixer''), ''VoiceFixer absent''; assert torch.cuda.is_available(), ''CUDA indisponible''; print(''Nettoyage prêt, GPU :'', torch.cuda.get_device_name(0))')
        Done $m $sig
    } else { Write-Host 'Déjà fait.' }

    # --- 11. Modèles du nettoyage (~0,8 Go), rangés dans StudioVoix\nettoyage ---
    Step 11 'Modèles du nettoyage de voix (~0,8 Go)'
    $m = Join-Path $Nt '.modeles-ok'
    if (Test-Retire 'nettoyage') { Write-Retire } elseif (-not (Test-Path $m)) {
        # Lancé depuis $Nt : ClearerVoice y range ses modèles (checkpoints\), VoiceFixer aussi (voicefixer\)
        Run $NtPy @((Join-Path $App 'moteurs\nettoyage_voix.py'), '--telecharger') $Nt
        Done $m
    } else { Write-Host 'Déjà fait.' }

    # --- 12. Environnement RVC (Applio, Python 3.12, PyTorch 2.11 CUDA 12.8) ---
    Step 12 'Environnement RVC, entraînement de ta voix (Python 3.12, PyTorch ~2,8 Go)'
    $m = Join-Path $Rvc '.env-ok'
    $sig = Get-Signature @($RvcCommit, 'cu128 unsafe-best-match')
    if (Test-Retire 'rvc') { Write-Retire } elseif (-not (Test-Done $m $sig)) {
        $venv = Join-Path $Rvc '.venv'
        Remove-Venv $venv
        Run $Uv @('venv', '--python', '3.12', $venv)
        # Même commande que l'installateur officiel d'Applio (run-install.bat), avec uv
        Run $Uv @('pip', 'install', '--python', $RvcPy, '-r', (Join-Path $Rvc 'requirements.txt'),
            '--extra-index-url', 'https://download.pytorch.org/whl/cu128', '--index-strategy', 'unsafe-best-match')
        Repair-TorchOmp $RvcPy $venv
        Run $RvcPy @('-c', 'import torch, faiss, librosa; assert torch.cuda.is_available(), ''CUDA indisponible''; print(''RVC prêt, GPU :'', torch.cuda.get_device_name(0))')
        Done $m $sig
    } else { Write-Host 'Déjà fait.' }

    # --- 13. Modèles de base de RVC (~1,8 Go : pré-entraînés, RMVPE, ContentVec) ---
    Step 13 'Modèles de base de RVC (~1,8 Go)'
    $m = Join-Path $Rvc '.modeles-ok'
    if (Test-Retire 'rvc') { Write-Retire } elseif (-not (Test-Path $m)) {
        # Lancé depuis le dossier d'Applio : ses modèles vont dans rvc\models, ses entraînements dans logs
        Run $RvcPy @((Join-Path $App 'moteurs\rvc_voix.py'), 'telecharger') $Rvc
        Done $m
    } else { Write-Host 'Déjà fait.' }

    # --- 15. Environnement de diffusion (Python 3.12, PyTorch 2.6 CUDA 12.6 : version des binaires de kijai) ---
    Step 15 'Environnement de diffusion : Qwen3-VL, Stable Audio, Hunyuan3D-2, Z-Image (Python 3.12)'
    $m = Join-Path $Dif '.env-ok'
    $sig = Get-Signature @('torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 cu126', $HyCommit, $KijaiCommit,
        (Join-Path $Listes 'diffusion.txt'))
    if (Test-Retire 'diffusion') { Write-Retire } elseif (-not (Test-Done $m $sig)) {
        $venv = Join-Path $Dif '.venv'
        Remove-Venv $venv
        Run $Uv @('venv', '--python', '3.12', $venv)
        Run $Uv @('pip', 'install', '--python', $DifPy, 'torch==2.6.0', 'torchvision==0.21.0', 'torchaudio==2.6.0',
            '--index-url', 'https://download.pytorch.org/whl/cu126')
        # Pile commune (versions testées et raisons des versions dans installation\diffusion.txt)
        Run $Uv @('pip', 'install', '--python', $DifPy, '-r', (Join-Path $Listes 'diffusion.txt'))
        # Hunyuan3D-2 sans ses dépendances (déjà listées ci-dessus, sans gradio ni outils d'entraînement), en mode
        # « editable » : le code reste dans $HySrc, où l'on dépose le module compilé mesh_processor
        Run $Uv @('pip', 'install', '--python', $DifPy, '--no-deps', '-e', $HySrc)
        # Rasteriseur de texture (CUDA) précompilé pour Python 3.12 + torch 2.6 + CUDA 12.6, et module d'inpainting
        $roue = Join-Path $Dif 'custom_rasterizer-0.1.0+torch260.cuda126-cp312-cp312-win_amd64.whl'
        Invoke-WebRequest -Uri "$KijaiRaw/wheels/custom_rasterizer-0.1.0+torch260.cuda126-cp312-cp312-win_amd64.whl" -OutFile $roue -UseBasicParsing
        Run $Uv @('pip', 'install', '--python', $DifPy, $roue)
        $pyd = Join-Path $HySrc 'hy3dgen\texgen\differentiable_renderer\mesh_processor.cp312-win_amd64.pyd'
        New-Item -ItemType Directory -Force -Path (Split-Path $pyd) | Out-Null
        Invoke-WebRequest -Uri "$KijaiRaw/hy3dgen/texgen/differentiable_renderer/mesh_processor.cp312-win_amd64.pyd" -OutFile $pyd -UseBasicParsing
        Repair-TorchOmp $DifPy $venv
        Run $DifPy @('-c', 'import torch, diffusers, transformers, custom_rasterizer; from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; from hy3dgen.texgen.differentiable_renderer.mesh_processor import meshVerticeInpaint; assert torch.cuda.is_available(), ''CUDA indisponible''; print(''Diffusion prête, GPU :'', torch.cuda.get_device_name(0))')
        Done $m $sig
    } else { Write-Host 'Déjà fait.' }

    # --- 16. Jeton Hugging Face (facultatif : seulement pour Stable Audio Open, sous licence à accepter) ---
    Step 16 'Jeton Hugging Face pour Stable Audio Open (bruitages) — facultatif'
    $jeton = Join-Path $env:HF_HOME 'token'
    if (Test-Retire 'diffusion:bruitages') {
        Write-Retire
    } elseif (Test-Path $jeton) {
        Write-Host 'Déjà fait (jeton présent).'
    } else {
        Write-Host 'Stable Audio Open (bruitages) demande d''accepter sa licence sur Hugging Face :' -ForegroundColor Yellow
        Write-Host '  1. connecte-toi sur https://huggingface.co/stabilityai/stable-audio-open-1.0 et accepte la licence ;'
        Write-Host '  2. crée un jeton (type Read) sur https://huggingface.co/settings/tokens ;'
        Write-Host '  3. colle-le ci-dessous. Laisse vide pour passer (tu pourras le faire plus tard dans l''onglet Modèles).'
        $saisie = Read-Host 'Jeton Hugging Face (hf_...)'
        if ($saisie -and $saisie.Trim().StartsWith('hf_')) {
            New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
            [IO.File]::WriteAllText($jeton, $saisie.Trim())   # sans BOM ni retour à la ligne
            Write-Host 'Jeton enregistré.'
        } else { Write-Host 'Pas de jeton : les bruitages seront disponibles après l''avoir enregistré dans l''onglet Modèles.' -ForegroundColor Yellow }
    }

    # --- 17. Modèles de diffusion (~72 Go : Qwen3-VL 4 Go, Z-Image-Turbo 15 Go, FLUX.2 klein 4,5 Go, photos 2 Go, vidéo Wan 2.2 20 Go, Hunyuan3D forme 5 Go et texture 16 Go, Stable Audio 5 Go) ---
    Step 17 'Modèles de diffusion (~72 Go)'
    $m = Join-Path $Dif '.modeles-ok'
    # Modèles téléchargés d'office (Stable Audio à part : il demande un jeton). Changer cette liste refait l'étape :
    # le téléchargement reprend seulement ce qui manque.
    $ModelesDif = @(@('qwen', 'forme3d', 'texture3d', 'zimage', 'personnages', 'photo_detourage', 'photo_qualite', 'video', 'detourage') | Where-Object { -not (Test-Retire "diffusion:$_") })
    $sig = Get-Signature @('modeles : ' + ($ModelesDif -join ' '))
    if (Test-Retire 'diffusion') { Write-Retire } elseif (-not (Test-Done $m $sig)) {
        Run $DifPy (@((Join-Path $App 'moteurs\diffusion.py'), 'telecharger') + $ModelesDif) $Dif
        # Stable Diffusion XL a été remplacé par Z-Image-Turbo (bien meilleur) : on libère ses 7 Go s'il est là
        $sdxl = Join-Path $env:HF_HOME 'hub\models--stabilityai--stable-diffusion-xl-base-1.0'
        if (Test-Path $sdxl) {
            Write-Host 'Suppression de Stable Diffusion XL, remplacé par Z-Image-Turbo (7 Go libérés).'
            Remove-Item $sdxl -Recurse -Force -ErrorAction SilentlyContinue
        }
        if ((Test-Path $jeton) -and -not (Test-Retire 'diffusion:bruitages')) {
            Run $DifPy @((Join-Path $App 'moteurs\diffusion.py'), 'telecharger', 'bruitages') $Dif
        } else { Write-Host 'Stable Audio Open non téléchargé (pas de jeton) : onglet Modèles → Télécharger, une fois le jeton enregistré.' -ForegroundColor Yellow }
        Done $m $sig
    } else { Write-Host 'Déjà fait.' }

    # --- 18. Environnement de l'application ---
    Step 18 'Environnement de Studio Voix (Python 3.12)'
    $m = Join-Path $App '.venv\installe.ok'
    $sig = Get-Signature @((Join-Path $App 'requirements.txt'))
    if (-not (Test-Done $m $sig)) {
        $venv = Join-Path $App '.venv'
        Remove-Venv $venv
        Run $Uv @('venv', '--python', '3.12', $venv)
        Run $Uv @('pip', 'install', '--python', $AppPy, '-r', (Join-Path $App 'requirements.txt'))
        Done $m $sig
    } else { Write-Host 'Déjà fait.' }

    Done (Join-Path $Eng 'installation.ok')
    Write-Host ''
    Write-Host '=== Installation terminée ===' -ForegroundColor Green
    Write-Host 'Lance maintenant lancer.bat.'
    exit 0
}
catch {
    Write-Host ''
    Write-Host "ERREUR : $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Fais une capture de cette fenêtre (les lignes au-dessus aussi), puis relance INSTALLER.bat :' -ForegroundColor Red
    Write-Host 'les étapes déjà réussies seront sautées.' -ForegroundColor Red
    exit 1
}
