# Studio Voix — installation complète, sans droits administrateur.
# Installe : uv (gestionnaire Python), Python 3.10 / 3.11 / 3.12, ACE-Step 1.5, Seed-VC, Demucs,
# Chatterbox (synthèse vocale), tous leurs modèles, et l'environnement de l'application.
# Relançable : chaque étape terminée est sautée.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # sinon Invoke-WebRequest est très lent
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$App = $PSScriptRoot
# Chemin court sur le même disque que l'appli : évite la limite de 260 caractères de Windows
$Eng = Join-Path ([IO.Path]::GetPathRoot($App)) 'StudioVoix'
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
$AppPy = Join-Path $App '.venv\Scripts\python.exe'

# Tout reste sur ce disque (caches compris), rien d'important sur C:
$env:UV_CACHE_DIR = Join-Path $Eng 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Eng 'python'
$env:UV_PYTHON_PREFERENCE = 'only-managed'   # ignore le Python 3.14 du système
$env:TORCH_HOME = Join-Path $Eng 'torch-cache'
$env:HF_HOME = Join-Path $Eng 'hf-home'
$env:PKUSEG_HOME = Join-Path $Cb 'pkuseg'   # sinon spacy-pkuseg (Chatterbox) écrit dans ~\.pkuseg, sur C:
$env:PYTHONIOENCODING = 'utf-8'
$env:PATH = "$UvDir;$env:PATH"

function Step($n, $txt) { Write-Host ''; Write-Host "=== [$n/10] $txt ===" -ForegroundColor Cyan }
function Done($marker) { New-Item -ItemType File -Force -Path $marker | Out-Null }

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
    if (Test-Path $ok) { Write-Host "Déjà présent : $dest"; return }
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    $zip = Join-Path $Eng '_depot.zip'
    $tmp = Join-Path $Eng '_depot'
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    Write-Host "Téléchargement de $url"
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
    Move-Item $inner.FullName $dest
    Remove-Item $tmp, $zip -Recurse -Force
    Done $ok
}

try {
    Write-Host 'Studio Voix — installation complète' -ForegroundColor Green
    Write-Host "Moteurs et modèles  : $Eng"
    Write-Host "Application         : $App"
    Write-Host 'Environ 26 à 31 Go à télécharger : compte une bonne heure selon ta connexion.'
    Write-Host 'Tu peux fermer et relancer INSTALLER.bat : les étapes finies seront sautées.'

    New-Item -ItemType Directory -Force -Path $Eng | Out-Null

    # --- Vérifications ---
    $drive = Get-PSDrive -Name $Eng.Substring(0, 1)
    $freeGo = [math]::Round($drive.Free / 1GB)
    Write-Host "Espace libre sur $($drive.Name): : $freeGo Go"
    if ($freeGo -lt 40) {
        Write-Host 'Attention : moins de 40 Go libres, l''installation complète risque de manquer de place.' -ForegroundColor Yellow
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
    Step 2 'Code d''ACE-Step, de Seed-VC et de Chatterbox'
    Get-Repo 'https://github.com/ace-step/ACE-Step-1.5/archive/refs/heads/main.zip' $Ace
    Get-Repo 'https://github.com/Plachtaa/seed-vc/archive/refs/heads/main.zip' $Sv
    New-Item -ItemType Directory -Force -Path $Cb | Out-Null
    Get-Repo "https://github.com/resemble-ai/chatterbox/archive/$CbCommit.zip" $CbSrc

    # --- 3. Environnement ACE-Step (Python 3.12 + PyTorch CUDA 12.8, via sa propre config) ---
    Step 3 'Environnement ACE-Step (le plus long : PyTorch ~3 Go)'
    $m = Join-Path $Ace '.env-ok'
    if (-not (Test-Path $m)) { Run $Uv @('sync') $Ace; Done $m } else { Write-Host 'Déjà fait.' }

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
    if (-not (Test-Path $m)) {
        $venv = Join-Path $Sv '.venv'
        if (Test-Path $venv) { Remove-Item $venv -Recurse -Force }
        Run $Uv @('venv', '--python', '3.10', $venv)
        Run $Uv @('pip', 'install', '--python', $SvPy, 'torch==2.4.0', 'torchaudio==2.4.0',
            '--index-url', 'https://download.pytorch.org/whl/cu124')
        # Dépendances utiles à la conversion (on laisse de côté l'interface et les outils d'évaluation)
        Run $Uv @('pip', 'install', '--python', $SvPy,
            'torch==2.4.0', 'torchaudio==2.4.0', 'accelerate', 'scipy==1.13.1', 'librosa==0.10.2',
            'huggingface-hub>=0.28.1', 'munch==4.0.0', 'einops==0.8.0', 'descript-audio-codec==1.0.0',
            'pydub==0.25.1', 'transformers==4.46.3', 'soundfile==0.12.1', 'numpy==1.26.4',
            'hydra-core==1.3.2', 'pyyaml', 'python-dotenv', 'matplotlib', 'demucs==4.0.1')
        Repair-TorchOmp $SvPy $venv
        Run $SvPy @('-c', 'import torch; assert torch.cuda.is_available(), ''CUDA indisponible''; print(''GPU :'', torch.cuda.get_device_name(0))')
        Done $m
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
    if (-not (Test-Path $m)) {
        $venv = Join-Path $Cb '.venv'
        Remove-Venv $venv
        Run $Uv @('venv', '--python', '3.11', $venv)
        # PyTorch CUDA d'abord : depuis PyPI, Windows recevrait la version sans carte graphique
        Run $Uv @('pip', 'install', '--python', $CbPy, 'torch==2.6.0', 'torchaudio==2.6.0',
            '--index-url', 'https://download.pytorch.org/whl/cu124')
        # Dépendances du pyproject de Chatterbox (versions testées), sans gradio (inutile ici) et avec
        # resemble-perth pris sur PyPI (le dépôt le demande via git, qui n'est pas forcément installé).
        # setuptools < 81 : le filigrane Perth importe pkg_resources, retiré des versions récentes.
        Run $Uv @('pip', 'install', '--python', $CbPy,
            'torch==2.6.0', 'torchaudio==2.6.0', 'numpy==1.26.4', 'librosa==0.11.0', 's3tokenizer==0.3.0',
            'transformers==5.2.0', 'diffusers==0.29.0', 'resemble-perth==1.0.1', 'conformer==0.3.2',
            'safetensors==0.5.3', 'spacy-pkuseg==1.0.1', 'pykakasi==2.3.0', 'pyloudnorm==0.2.0',
            'omegaconf==2.3.1', 'soundfile', 'setuptools<81')
        Run $Uv @('pip', 'install', '--python', $CbPy, '--no-deps', $CbSrc)
        Repair-TorchOmp $CbPy $venv
        Run $CbPy @('-c', 'import torch, perth; from chatterbox.mtl_tts import ChatterboxMultilingualTTS; assert perth.PerthImplicitWatermarker is not None, ''filigrane Perth indisponible''; assert torch.cuda.is_available(), ''CUDA indisponible''; print(''Chatterbox prêt, GPU :'', torch.cuda.get_device_name(0))')
        Done $m
    } else { Write-Host 'Déjà fait.' }

    # --- 9. Modèles Chatterbox (~3,2 Go, chargés une fois sur le processeur pour vérification) ---
    Step 9 'Modèles Chatterbox Multilingual V3 (~3,2 Go)'
    $m = Join-Path $Cb '.modeles-ok'
    if (-not (Test-Path $m)) {
        Run $CbPy @((Join-Path $App 'moteurs\chatterbox_tts.py'), '--telecharger')
        Done $m
    } else { Write-Host 'Déjà fait.' }

    # --- 10. Environnement de l'application ---
    Step 10 'Environnement de Studio Voix (Python 3.12)'
    $m = Join-Path $App '.venv\installe.ok'
    if (-not (Test-Path $m)) {
        $venv = Join-Path $App '.venv'
        Remove-Venv $venv
        Run $Uv @('venv', '--python', '3.12', $venv)
        Run $Uv @('pip', 'install', '--python', $AppPy, 'gradio>=5.0', 'requests', 'numpy', 'librosa', 'soundfile')
        Done $m
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
