# Studio Voix

Logiciel local pour Windows : tu enregistres ta voix, tu écris un prompt (genre, style, instruments, ambiance) et tes paroles, et il te rend une chanson chantée avec **ta** voix. Il sait aussi composer de la musique seule, et lire un texte avec ta voix (synthèse vocale). Tout tourne sur ta machine, sans service en ligne.

Moteurs utilisés : [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (génération musicale), [Demucs](https://github.com/facebookresearch/demucs) (séparation voix / musique), [Seed-VC](https://github.com/Plachtaa/seed-vc) (conversion de voix chantée) et [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (synthèse vocale, licence MIT). Ils sont téléchargés par l'installateur, ils ne sont pas inclus dans ce dépôt.

Configuration testée : Windows 11, NVIDIA RTX 4070 (12 Go).

## Structure du dépôt

| Fichier | Rôle |
|---|---|
| `studio_voix.py` | point d'entrée de l'application (lancé par `lancer.bat`) |
| `studiovoix/` | le code de l'application, un module par rôle (voir ci-dessous) |
| `moteurs/` | scripts exécutés dans l'environnement d'un moteur (`chatterbox_tts.py`) |
| `tests/` | tests automatiques sans carte graphique (moteurs simulés) |
| `installer.ps1` / `INSTALLER.bat` | installation complète en un clic |
| `lancer.bat` | démarrage (serveur ACE-Step + application) |
| `CLAUDE.md` | contexte technique pour Claude Code |

Modules de `studiovoix/` :

| Module | Rôle |
|---|---|
| `config.py` | chemins, variables d'environnement, constantes |
| `acestep.py` | génération de la chanson (API REST d'ACE-Step) |
| `demucs.py` | séparation voix / instrumental |
| `seedvc.py` | conversion de voix chantée |
| `chatterbox.py` | synthèse vocale (lance `moteurs/chatterbox_tts.py`) |
| `voix.py` | bibliothèque de voix (import, contrôle de qualité, renommage, suppression) |
| `mixage.py` | mixage voix + instrumental |
| `pipeline.py` | enchaînement complet des étapes |
| `modeles.py` | état des modèles (onglet « Modèles ») |
| `outils.py` | journal de commande en direct, ouverture de dossier |
| `interface.py` | interface Gradio |

## Installation (une seule fois)

1. Clone le dépôt (ou télécharge-le en zip) dans un dossier, par exemple `D:\Projets 3d\Voix et chanson`.
2. Double-clique sur **INSTALLER.bat**.

C'est tout. Aucun droit administrateur n'est nécessaire, et ni Python ni Git n'ont besoin d'être installés au préalable. L'installateur récupère et installe :
- **uv**, qui télécharge lui-même les bonnes versions de Python (3.12 pour ACE-Step et pour l'application, 3.10 pour Seed-VC, 3.11 pour Chatterbox), sans toucher au Python 3.14 de ton système ;
- **ACE-Step 1.5** et ses modèles (~10 Go) ;
- **Seed-VC** et ses modèles ;
- **Demucs** et son modèle ;
- **Chatterbox Multilingual V3** et ses modèles (~3,2 Go, plus ~2,5 Go pour PyTorch) ;
- l'environnement de l'application.

Compte 26 à 31 Go à télécharger, soit une bonne heure selon ta connexion, et au moins 40 Go libres. Si l'installation s'interrompt, relance INSTALLER.bat : les étapes terminées sont sautées.

**Tu avais déjà installé Studio Voix avant l'arrivée de la synthèse vocale ?** Relance simplement INSTALLER.bat : seules les deux étapes de Chatterbox s'exécutent (environ 6 Go à télécharger, 10 à 25 minutes). Sans cela, tout le reste fonctionne et l'onglet « Synthèse vocale » t'indique qu'il manque Chatterbox.

### Où tout est installé

Les moteurs et les modèles vont dans **`<lecteur>:\StudioVoix`** (par exemple `D:\StudioVoix` si l'application est sur D:). Un chemin court évite la limite de 260 caractères de Windows, qui casse certaines installations.

| Dossier | Contenu |
|---|---|
| `StudioVoix\ace-step\checkpoints` | modèles ACE-Step |
| `StudioVoix\seed-vc\checkpoints` | modèles Seed-VC |
| `StudioVoix\torch-cache` | modèle Demucs |
| `StudioVoix\chatterbox` | Chatterbox (code, environnement, modèle de découpage `pkuseg`) |
| `StudioVoix\hf-home` | modèles Chatterbox (cache Hugging Face) |
| `StudioVoix\python`, `uv`, `uv-cache` | Python et outils d'installation |
| `<dossier de l'application>\data` | tes voix, tes chansons et tes textes lus |

Pour tout désinstaller : supprime `StudioVoix` et le dossier `.venv` de l'application.

## Utilisation

1. Double-clique sur **lancer.bat**. Il ouvre, réduite, une fenêtre « ACE-Step - ne pas fermer » (le serveur de génération), puis l'application dans ton navigateur. La première génération attend que ce serveur ait fini de charger ses modèles.
2. Onglet **Bibliothèque de voix** : importe un fichier (wav, mp3 ou flac) ou enregistre-toi au micro, 10 à 25 s, dans une pièce calme, sans musique ni écho. Nomme la voix et clique sur « Vérifier et enregistrer ». Dans « Mes voix », tu peux écouter, renommer ou supprimer chaque voix.
3. Onglet **Créer une chanson** : choisis le mode, remplis genre, style, instruments, ambiance et paroles (avec `[Verse]`, `[Chorus]`…), puis « Créer la chanson ». Trois modes :
   - **Chanson avec ma voix** (par défaut) : la chanson est chantée avec ta voix ;
   - **Chanson avec la voix d'ACE-Step** : musique seule, la voix générée par ACE-Step est gardée telle quelle (plus rapide : ni séparation ni conversion, pas besoin de voix enregistrée) ;
   - **Instrumental** : musique sans voix, les paroles sont ignorées.
4. Onglet **Synthèse vocale** : choisis une voix de ta bibliothèque, la langue, écris le texte et clique sur « Lire le texte avec cette voix ». Les textes longs (jusqu'à 5 000 caractères) sont découpés en phrases. Chaque lecture est rangée dans `data\tts\<date>\` (texte et `parole.wav`).
5. Quand tu as fini, ferme aussi la fenêtre ACE-Step pour libérer la carte graphique.

Chaque chanson est rangée dans `data\songs\<date>\` : version brute, voix convertie, instrumental, mix final et prompt.

## Comment ça marche

1. **ACE-Step 1.5** génère la chanson complète, avec une voix chantée générique.
2. **Demucs** sépare la voix de l'instrumental.
3. **Seed-VC** (conversion de voix chantée, sans entraînement) remplace cette voix par la tienne.
4. L'application remixe voix et instrumental.

En mode « voix d'ACE-Step » ou « Instrumental », seule l'étape 1 a lieu.

### Contrôle de qualité à l'import

| Vérification | Refus | Avertissement (la voix est quand même enregistrée) |
|---|---|---|
| Durée | moins de 5 s | moins de 10 s ; au-delà de 30 s, seules les 30 premières secondes sont gardées |
| Volume (niveau de la voix, pauses ignorées) | sous −40 dBFS | sous −30 dBFS |
| Saturation | — | plus de 0,1 % des échantillons au maximum |

Chaque message explique quoi changer (se rapprocher du micro, baisser le niveau d'entrée…).

## Réglages utiles

- **Voix chantée de base** : choisis masculine ou féminine selon ta voix. Sinon, joue sur le décalage de hauteur (−12 / +12 demi-tons).
- **Étapes Seed-VC** : 40 par défaut. Monte à 50 pour plus de qualité, au prix du temps.
- **Synthèse vocale** : « Expressivité » à 0,5 = neutre ; pour un ton plus dramatique, monte vers 0,7 et baisse « Guidage / rythme » vers 0,3. Si ton échantillon n'est pas dans la langue du texte, mets « Guidage / rythme » à 0 pour ne pas garder l'accent. La graine (≠ 0) rend un résultat reproductible.

## Limites à connaître

- **Installation** : en cas d'erreur, l'installateur s'arrête avec un message en rouge. Relance-le après correction : les étapes réussies sont sautées.
- **Pilote NVIDIA** : ACE-Step utilise CUDA 12.8, qui demande un pilote récent (570.65 ou plus). L'installateur le vérifie.
- **Fidélité de la voix** : la conversion sans entraînement donne une ressemblance correcte mais pas parfaite, surtout sur les notes aiguës. Pour mieux faire, il faudrait entraîner un modèle sur 10 à 30 minutes de tes enregistrements (par exemple RVC).
- **Artefacts** : la séparation sur de la musique générée laisse parfois de légers résidus.
- **Mémoire graphique et synthèse vocale** : Chatterbox a besoin de 3 à 4 Go de mémoire graphique. Si la fenêtre ACE-Step est ouverte et a déjà chargé ses modèles, la carte peut manquer de mémoire : l'application te demande alors de fermer cette fenêtre, puis de relancer la lecture.
- **Filigrane** : chaque fichier produit par la synthèse vocale porte un filigrane inaudible ([Perth](https://github.com/resemble-ai/perth)) qui permet de reconnaître une voix de synthèse. Il est ajouté par Chatterbox lui-même.
- **Référence de voix pour la synthèse** : Chatterbox n'utilise que les 10 premières secondes de l'échantillon.
- **Consentement** : clone uniquement ta propre voix, ou celle de personnes d'accord.

## Tests (pour le développement)

Les tests n'ont pas besoin de carte graphique : ACE-Step est remplacé par un faux serveur HTTP, Demucs et Seed-VC par de faux scripts. Depuis le dossier de l'application :

```
uv run --python 3.12 --with-requirements requirements.txt --with pytest pytest tests
```

## Licences

Ce dépôt ne contient que le code de l'application et de l'installation. Chaque moteur garde sa propre licence (voir leurs dépôts respectifs), à vérifier avant tout usage commercial des sons produits.
