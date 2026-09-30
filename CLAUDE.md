# CLAUDE.md — contexte du projet Studio Voix

## En bref
Application locale Windows (Gradio) pour créer des chansons chantées avec la voix de l'utilisateur, de la musique seule, et lire un texte avec sa voix (synthèse vocale).
Machine cible : PC Windows 11, RTX 4070 (12 Go de VRAM), utilisateur francophone.
Toute l'interface, les messages et la documentation sont **en français**.

## Architecture actuelle
- `studio_voix.py` : point d'entrée mince (lancé par `lancer.bat`), qui appelle `studiovoix.interface.build_ui`.
- `studiovoix/` : le code de l'application, un module par rôle :
  - `config.py` (chemins, variables d'environnement, `SR`, `LANGUES`) — les autres modules lisent `cfg.X` au moment de l'appel, ce qui permet aux tests de rediriger les dossiers ;
  - un module par moteur, chacun avec son action, `ckpt_dir()`, son état et `download()` : `acestep.py` (HTTP), `demucs.py`, `seedvc.py` et `chatterbox.py` (sous-processus) ;
  - `voix.py` (bibliothèque de voix : `data/voices/<nom>.wav`, 44,1 kHz mono, 30 s au plus, normalisée à 0,95 ; import wav/mp3/flac ou micro avec contrôle de qualité `analyser()` — durée, niveau de la voix au 95e centile du RMS sur 50 ms, saturation —, écoute, renommage, suppression ; les noms sont comparés sans tenir compte de la casse, comme Windows, et toute opération vérifie que le nom est dans la liste), `mixage.py`, `pipeline.py` (enchaînement), `modeles.py` (tableau de l'onglet « Modèles »), `outils.py` (journal de commande, ouverture de dossier), `interface.py` (Gradio : onglets « Bibliothèque de voix », « Créer une chanson », « Synthèse vocale », « Modèles » ; après chaque opération sur la bibliothèque, `synchro_voix` met à jour les listes de voix des autres onglets ; la suppression passe par un `confirm()` du navigateur).
- `moteurs/chatterbox_tts.py` : script exécuté **dans l'environnement de Chatterbox** (jamais importé par l'application, qui n'a pas torch ; ses imports lourds sont dans les fonctions pour que `decouper()` soit testable). Protocole : `python chatterbox_tts.py <tache.json>` (JSON UTF-8 : texte, langue, voix, sortie, exaggeration, cfg_weight, temperature, graine) ; il écrit `PROGRESSION i/n`, `ERREUR : …` (erreur prévue, par exemple manque de mémoire graphique) et `TERMINE <fichier>`. `--telecharger` charge le modèle sur CPU (téléchargement + vérification). Le fichier de tâche évite de passer du texte accentué par la ligne de commande Windows.
- `tests/` : pytest sans GPU. `conftest.py` fournit un faux serveur ACE-Step (HTTP), un faux Demucs (`python -m demucs`) et un faux Seed-VC (`inference.py`). Lancer : `uv run --python 3.12 --with-requirements requirements.txt --with pytest pytest tests`.
- `installer.ps1` (lancé par `INSTALLER.bat`) : installation complète en un clic, sans droits admin.
- `lancer.bat` : définit les variables d'environnement, démarre le serveur ACE-Step s'il ne tourne pas, lance l'application.

Pipeline de création d'une chanson :
1. **ACE-Step 1.5** génère la chanson complète via son API REST locale (`/release_task`, `/query_result`, `/v1/audio`, `/health`, port 8001).
2. **Demucs** (`htdemucs`, `--two-stems=vocals`) sépare voix et instrumental.
3. **Seed-VC** (`inference.py`, `--f0-condition True`) convertit la voix chantée vers l'échantillon de l'utilisateur (zero-shot, 1 à 30 s de référence).
4. Mixage numpy/soundfile en 44,1 kHz stéréo, sortie dans `data/songs/<horodatage>/`.

Modes (`pipeline.MODES`, sélecteur en haut de l'onglet) : « Chanson avec ma voix » (pipeline complet ci-dessus), « Chanson avec la voix d'ACE-Step » et « Instrumental » (étape 1 seule, résultat = `chanson_brute.wav`). L'instrumental s'obtient avec `lyrics="[Instrumental]"` : c'est le signal reconnu par ACE-Step (`acestep/api/server_utils.py`, `is_instrumental`), `/release_task` n'a pas de paramètre dédié. `interface.maj_mode` masque les réglages inutiles au mode choisi.

Synthèse vocale : `chatterbox.synthese` → `ChatterboxMultilingualTTS.from_pretrained(device, t3_model="v3")` puis `generate(text, language_id, audio_prompt_path, exaggeration, cfg_weight, temperature)` (API vérifiée dans `src/chatterbox/mtl_tts.py`), sortie 24 kHz dans `data/tts/<horodatage>/parole.wav`. `generate()` plafonne à 1000 jetons (≈ 40 s) : le texte est découpé en morceaux de 300 caractères au plus (limite de l'interface officielle), recollés avec 0,25 s de silence. Seules les 10 premières secondes de la référence sont utilisées. Chaque sortie porte le filigrane Perth.

## Quatre environnements Python séparés (volontairement)
Ils sont gérés par **uv** (Pythons « managed », jamais le Python système, qui est en 3.14) :

| Environnement | Python | Contenu | Emplacement |
|---|---|---|---|
| ACE-Step | 3.11/3.12 (choisi par son `pyproject`) | `uv sync` officiel, torch cu128 | `<lecteur>:\StudioVoix\ace-step\.venv` |
| Seed-VC + Demucs | 3.10 | torch 2.4.0 cu124 + dépendances minimales | `<lecteur>:\StudioVoix\seed-vc\.venv` |
| Chatterbox | 3.11 | torch 2.6.0 cu124 + dépendances épinglées + Chatterbox (`--no-deps`, source au commit `$CbCommit`) | `<lecteur>:\StudioVoix\chatterbox\.venv` |
| Application | 3.12 | gradio, requests, numpy, librosa, soundfile (**pas de torch**) | `<appli>\.venv` |

L'application n'importe jamais les moteurs : elle les appelle en **sous-processus** (Seed-VC, Demucs, Chatterbox) ou en **HTTP** (ACE-Step). Garde ce découplage pour tout nouveau moteur (par exemple un moteur de synthèse vocale) : un environnement dédié, installé par `installer.ps1`, appelé en sous-processus ou en HTTP.

## Contraintes apprises en déboguant (à respecter)
- **Chemins courts** : les moteurs vont dans `<lecteur de l'appli>:\StudioVoix`, pas dans le dossier de l'appli (`D:\Projets 3d\Voix et chanson` contient des espaces et le cache Hugging Face produit des chemins proches de la limite de 260 caractères).
- **Tout reste sur le même disque** : `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `TORCH_HOME` et `HF_HOME` pointent dans `StudioVoix`. `lancer.bat` et `installer.ps1` doivent rester cohérents entre eux.
- **PyTorch 2.4.0 sous Windows** : `fbgemm.dll` réclame `libomp140.x86_64.dll`. Le correctif (repris de ComfyUI) copie `libiomp5md.dll` sous ce nom dans `torch\lib` si `import torch` échoue.
- **Windows PowerShell 5.1** (pas PowerShell 7) : rediriger le stderr d'un programme natif alors que `$ErrorActionPreference = 'Stop'` lève une exception. La fonction `Run` passe temporairement en `Continue` et vérifie `$LASTEXITCODE`.
- **Encodages** : `installer.ps1` en UTF-8 **avec BOM** et CRLF (sinon PowerShell 5.1 corrompt les accents) ; fichiers `.bat` en ASCII pur et CRLF (voir `.gitattributes`).
- **Installateur idempotent** : chaque étape pose un fichier marqueur (`.env-ok`, `.modeles-ok`, etc.) et est sautée si elle a déjà réussi. Toute nouvelle étape doit suivre ce modèle.
- **Fichiers verrouillés** : un `.venv` ne peut pas être supprimé tant que l'application tourne ; afficher un message clair plutôt qu'une erreur brute.
- **Serveur ACE-Step lent au premier appel** : il charge ses modèles à la première requête et peut ne pas répondre pendant plus de 30 s. Le polling doit tolérer les délais dépassés et réessayer (c'est déjà le cas, ne pas régresser).
- **Chatterbox** (étapes 8 et 9 de l'installateur, marqueurs `chatterbox\.env-ok` et `chatterbox\.modeles-ok`) :
  - le paquet PyPI `chatterbox-tts` 0.1.7 **ne contient pas** le modèle Multilingual V3 : on installe le code source d'un commit épinglé (`$CbCommit` dans `installer.ps1`, zip GitHub via `Get-Repo` vers `chatterbox\src`) avec `--no-deps` ;
  - son `pyproject` réclame `resemble-perth` via `git+https` (Git n'est pas requis sur la machine) : les dépendances sont donc listées explicitement dans l'installateur, versions testées, `resemble-perth` venant de PyPI, sans `gradio` (non importé par la bibliothèque) ;
  - torch 2.6.0 est installé d'abord depuis l'index cu124 (depuis PyPI, Windows aurait la version CPU) ; la seconde commande répète `torch==2.6.0` pour qu'il ne soit pas remplacé ;
  - **`setuptools<81` obligatoire** : Perth importe `pkg_resources`, retiré des setuptools récents ; sans lui, `perth.PerthImplicitWatermarker` vaut `None` et le chargement plante (`'NoneType' object is not callable`). L'étape 8 le vérifie ;
  - **`PKUSEG_HOME`** (dans `installer.ps1`, `lancer.bat` et par défaut dans `chatterbox._env()`) : sinon `spacy-pkuseg` télécharge un modèle de 35 Mo dans `~\.pkuseg`, donc sur C: ;
  - les modèles (~3,2 Go : `t3_mtl23ls_v3.safetensors`, `s3gen.pt`, `ve.pt`, vocabulaire) vont dans le cache Hugging Face (`HF_HOME`) ; quand ils sont présents, l'application lance le moteur avec `HF_HUB_OFFLINE=1` (chargement vérifié hors ligne) ;
  - VRAM : ACE-Step (serveur resté ouvert) + Chatterbox peuvent dépasser 12 Go ; le script intercepte `torch.cuda.OutOfMemoryError` et demande de fermer la fenêtre ACE-Step.
- **Seed-VC** : le dépôt n'a pas de commande de téléchargement ; les modèles se téléchargent au premier `inference.py` (dans `checkpoints/hf_cache`, relatif à son dossier). Le fichier de sortie s'appelle `vc_<source>_<cible>_<...>.wav`.

## Règles de travail
- Vérifie les API réelles (code source, documentation des dépôts) avant d'appeler un moteur ; ne suppose pas un nom de paramètre.
- Aucun test ne peut tourner sur GPU dans ton environnement si tu n'es pas sur la machine Windows : dis clairement ce qui a été testé et ce qui ne l'a pas été.
- Pour l'installateur : analyse syntaxique et compatibilité 5.1 avec PowerShell 7 + PSScriptAnalyzer (`PSUseCompatibleSyntax`/`Commands`/`Types`, profil Windows PowerShell 5.1) ; on peut aussi l'exécuter sous Linux avec de faux `uv.exe`/`python.exe` et des fonctions qui remplacent `Invoke-WebRequest` et `Get-PSDrive`, pour vérifier l'ordre des étapes, les marqueurs et la reprise.
- Ne jamais versionner `data/` (voix et chansons personnelles) ni les `.venv`.
- Garder l'installation « un clic » : toute nouvelle dépendance ou tout nouveau modèle doit être ajouté à `installer.ps1`, à l'onglet « Modèles » et au README.
