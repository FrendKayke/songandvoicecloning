# CLAUDE.md — contexte du projet Studio Voix

## En bref
Application locale Windows (Gradio) pour créer des chansons chantées avec la voix de l'utilisateur.
Machine cible : PC Windows 11, RTX 4070 (12 Go de VRAM), utilisateur francophone.
Toute l'interface, les messages et la documentation sont **en français**.

## Architecture actuelle
- `studio_voix.py` : point d'entrée mince (lancé par `lancer.bat`), qui appelle `studiovoix.interface.build_ui`.
- `studiovoix/` : le code de l'application, un module par rôle :
  - `config.py` (chemins, variables d'environnement, `SR`, `LANGUES`) — les autres modules lisent `cfg.X` au moment de l'appel, ce qui permet aux tests de rediriger les dossiers ;
  - un module par moteur, chacun avec son action, `ckpt_dir()`, son état et `download()` : `acestep.py` (HTTP), `demucs.py` et `seedvc.py` (sous-processus) ;
  - `voix.py` (voix de référence), `mixage.py`, `pipeline.py` (enchaînement), `modeles.py` (tableau de l'onglet « Modèles »), `outils.py` (journal de commande, ouverture de dossier), `interface.py` (Gradio : onglets « Ma voix », « Créer une chanson », « Modèles »).
- `tests/` : pytest sans GPU. `conftest.py` fournit un faux serveur ACE-Step (HTTP), un faux Demucs (`python -m demucs`) et un faux Seed-VC (`inference.py`). Lancer : `uv run --python 3.12 --with-requirements requirements.txt --with pytest pytest tests`.
- `installer.ps1` (lancé par `INSTALLER.bat`) : installation complète en un clic, sans droits admin.
- `lancer.bat` : définit les variables d'environnement, démarre le serveur ACE-Step s'il ne tourne pas, lance l'application.

Pipeline de création d'une chanson :
1. **ACE-Step 1.5** génère la chanson complète via son API REST locale (`/release_task`, `/query_result`, `/v1/audio`, `/health`, port 8001).
2. **Demucs** (`htdemucs`, `--two-stems=vocals`) sépare voix et instrumental.
3. **Seed-VC** (`inference.py`, `--f0-condition True`) convertit la voix chantée vers l'échantillon de l'utilisateur (zero-shot, 1 à 30 s de référence).
4. Mixage numpy/soundfile en 44,1 kHz stéréo, sortie dans `data/songs/<horodatage>/`.

Modes (`pipeline.MODES`, sélecteur en haut de l'onglet) : « Chanson avec ma voix » (pipeline complet ci-dessus), « Chanson avec la voix d'ACE-Step » et « Instrumental » (étape 1 seule, résultat = `chanson_brute.wav`). L'instrumental s'obtient avec `lyrics="[Instrumental]"` : c'est le signal reconnu par ACE-Step (`acestep/api/server_utils.py`, `is_instrumental`), `/release_task` n'a pas de paramètre dédié. `interface.maj_mode` masque les réglages inutiles au mode choisi.

## Trois environnements Python séparés (volontairement)
Ils sont gérés par **uv** (Pythons « managed », jamais le Python système, qui est en 3.14) :

| Environnement | Python | Contenu | Emplacement |
|---|---|---|---|
| ACE-Step | 3.11/3.12 (choisi par son `pyproject`) | `uv sync` officiel, torch cu128 | `<lecteur>:\StudioVoix\ace-step\.venv` |
| Seed-VC + Demucs | 3.10 | torch 2.4.0 cu124 + dépendances minimales | `<lecteur>:\StudioVoix\seed-vc\.venv` |
| Application | 3.12 | gradio, requests, numpy, librosa, soundfile (**pas de torch**) | `<appli>\.venv` |

L'application n'importe jamais les moteurs : elle les appelle en **sous-processus** (Seed-VC, Demucs) ou en **HTTP** (ACE-Step). Garde ce découplage pour tout nouveau moteur (par exemple un moteur de synthèse vocale) : un environnement dédié, installé par `installer.ps1`, appelé en sous-processus ou en HTTP.

## Contraintes apprises en déboguant (à respecter)
- **Chemins courts** : les moteurs vont dans `<lecteur de l'appli>:\StudioVoix`, pas dans le dossier de l'appli (`D:\Projets 3d\Voix et chanson` contient des espaces et le cache Hugging Face produit des chemins proches de la limite de 260 caractères).
- **Tout reste sur le même disque** : `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `TORCH_HOME` et `HF_HOME` pointent dans `StudioVoix`. `lancer.bat` et `installer.ps1` doivent rester cohérents entre eux.
- **PyTorch 2.4.0 sous Windows** : `fbgemm.dll` réclame `libomp140.x86_64.dll`. Le correctif (repris de ComfyUI) copie `libiomp5md.dll` sous ce nom dans `torch\lib` si `import torch` échoue.
- **Windows PowerShell 5.1** (pas PowerShell 7) : rediriger le stderr d'un programme natif alors que `$ErrorActionPreference = 'Stop'` lève une exception. La fonction `Run` passe temporairement en `Continue` et vérifie `$LASTEXITCODE`.
- **Encodages** : `installer.ps1` en UTF-8 **avec BOM** et CRLF (sinon PowerShell 5.1 corrompt les accents) ; fichiers `.bat` en ASCII pur et CRLF (voir `.gitattributes`).
- **Installateur idempotent** : chaque étape pose un fichier marqueur (`.env-ok`, `.modeles-ok`, etc.) et est sautée si elle a déjà réussi. Toute nouvelle étape doit suivre ce modèle.
- **Fichiers verrouillés** : un `.venv` ne peut pas être supprimé tant que l'application tourne ; afficher un message clair plutôt qu'une erreur brute.
- **Serveur ACE-Step lent au premier appel** : il charge ses modèles à la première requête et peut ne pas répondre pendant plus de 30 s. Le polling doit tolérer les délais dépassés et réessayer (c'est déjà le cas, ne pas régresser).
- **Seed-VC** : le dépôt n'a pas de commande de téléchargement ; les modèles se téléchargent au premier `inference.py` (dans `checkpoints/hf_cache`, relatif à son dossier). Le fichier de sortie s'appelle `vc_<source>_<cible>_<...>.wav`.

## Règles de travail
- Vérifie les API réelles (code source, documentation des dépôts) avant d'appeler un moteur ; ne suppose pas un nom de paramètre.
- Aucun test ne peut tourner sur GPU dans ton environnement si tu n'es pas sur la machine Windows : dis clairement ce qui a été testé et ce qui ne l'a pas été.
- Ne jamais versionner `data/` (voix et chansons personnelles) ni les `.venv`.
- Garder l'installation « un clic » : toute nouvelle dépendance ou tout nouveau modèle doit être ajouté à `installer.ps1`, à l'onglet « Modèles » et au README.
