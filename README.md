# Studio Voix

Logiciel local pour Windows : tu enregistres ta voix, tu écris un prompt (genre, style, instruments, ambiance) et tes paroles, et il te rend une chanson chantée avec **ta** voix. Il sait aussi composer de la musique seule, lire un texte avec ta voix (synthèse vocale) et nettoyer un enregistrement fait avec un micro médiocre. Tout tourne sur ta machine, sans service en ligne.

Moteurs utilisés : [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (génération musicale), [Demucs](https://github.com/facebookresearch/demucs) (séparation voix / musique), [Seed-VC](https://github.com/Plachtaa/seed-vc) (conversion de voix chantée) et [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (synthèse vocale, licence MIT), [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) (MossFormer2, débruitage, Apache-2.0) et [VoiceFixer](https://github.com/haoheliu/voicefixer) (restauration de voix, code MIT, poids CC-BY 4.0). Ils sont téléchargés par l'installateur, ils ne sont pas inclus dans ce dépôt.

Configuration testée : Windows 11, NVIDIA RTX 4070 (12 Go).

## Structure du dépôt

| Fichier | Rôle |
|---|---|
| `studio_voix.py` | point d'entrée de l'application (lancé par `lancer.bat`) |
| `studiovoix/` | le code de l'application, un module par rôle (voir ci-dessous) |
| `moteurs/` | scripts exécutés dans l'environnement d'un moteur (`chatterbox_tts.py`, `nettoyage_voix.py`) |
| `tests/` | tests automatiques sans carte graphique (moteurs simulés) |
| `installer.ps1` / `INSTALLER.bat` | installation complète en un clic |
| `lancer.bat` | démarrage (serveur ACE-Step + application) |
| `CLAUDE.md` | contexte technique pour Claude Code |

Modules de `studiovoix/` :

| Module | Rôle |
|---|---|
| `config.py` | chemins, variables d'environnement, constantes |
| `acestep.py` | génération de la chanson (API REST d'ACE-Step) |
| `styles.py` | listes déroulantes des styles (libellés français, termes anglais) |
| `demucs.py` | séparation voix / instrumental |
| `seedvc.py` | conversion de voix chantée |
| `chatterbox.py` | synthèse vocale (lance `moteurs/chatterbox_tts.py`) |
| `nettoyage.py` | nettoyage de voix (lance `moteurs/nettoyage_voix.py`) |
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
- le **nettoyage de voix** (MossFormer2 et VoiceFixer, ~0,8 Go de modèles ; il réutilise le PyTorch de Chatterbox) ;
- l'environnement de l'application.

Compte 27 à 32 Go à télécharger, soit une bonne heure selon ta connexion, et au moins 40 Go libres. Si l'installation s'interrompt, relance INSTALLER.bat : les étapes terminées sont sautées.

**Tu avais déjà installé Studio Voix avant l'arrivée de la synthèse vocale ou du nettoyage ?** Relance simplement INSTALLER.bat : seules les étapes manquantes s'exécutent (Chatterbox : environ 6 Go, 10 à 25 minutes ; nettoyage : environ 1 Go de plus, 5 minutes). Sans cela, tout le reste fonctionne et l'application t'indique ce qui manque.

### Où tout est installé

Les moteurs et les modèles vont dans **`<lecteur>:\StudioVoix`** (par exemple `D:\StudioVoix` si l'application est sur D:). Un chemin court évite la limite de 260 caractères de Windows, qui casse certaines installations.

| Dossier | Contenu |
|---|---|
| `StudioVoix\ace-step\checkpoints` | modèles ACE-Step |
| `StudioVoix\seed-vc\checkpoints` | modèles Seed-VC |
| `StudioVoix\torch-cache` | modèle Demucs |
| `StudioVoix\chatterbox` | Chatterbox (code, environnement, modèle de découpage `pkuseg`) |
| `StudioVoix\hf-home` | modèles Chatterbox (cache Hugging Face) |
| `StudioVoix\nettoyage` | nettoyage de voix (environnement, modèles dans `checkpoints\` et `voicefixer\`) |
| `StudioVoix\python`, `uv`, `uv-cache` | Python et outils d'installation |
| `<dossier de l'application>\data` | tes voix, tes chansons, tes textes lus et tes voix nettoyées |

Pour tout désinstaller : supprime `StudioVoix` et le dossier `.venv` de l'application.

## Utilisation

1. Double-clique sur **lancer.bat**. Il ouvre, réduite, une fenêtre « ACE-Step - ne pas fermer » (le serveur de génération), puis l'application dans ton navigateur. La première génération attend que ce serveur ait fini de charger ses modèles.
2. Onglet **Bibliothèque de voix** : importe un fichier (wav, mp3 ou flac) ou enregistre-toi au micro, 10 à 25 s, dans une pièce calme, sans musique ni écho. Nomme la voix et clique sur « Vérifier et enregistrer ». Dans « Mes voix », tu peux écouter, renommer ou supprimer chaque voix.
   - **Micro médiocre ?** Avant d'enregistrer, ouvre « 🧽 Nettoyer la voix », choisis un niveau et clique sur « Nettoyer l'échantillon ». Écoute la version nettoyée, compare avec l'original, puis choisis dans « Version à enregistrer » celle que tu gardes. Pour une voix déjà enregistrée : « Reprendre cette voix pour la nettoyer », puis enregistre-la sous un nouveau nom.
3. Onglet **Créer une chanson** : choisis le mode, puis genre, style, instruments, ambiance et consignes dans les listes déroulantes (plusieurs choix possibles ; tu peux aussi taper ton propre terme, en anglais de préférence, puis Entrée). La **description envoyée à ACE-Step** s'affiche en dessous, en anglais, et tu peux la retoucher. Écris les paroles (avec `[Verse]`, `[Chorus]`…), puis « Créer la chanson ». Trois modes :
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

En mode « voix d'ACE-Step » ou « Instrumental », seule l'étape 1 a lieu, sauf si tu retires un instrument : Demucs sépare alors la chanson en 4 pistes (voix, batterie, basse, autres) et l'instrumental est remixé sans l'instrument retiré.

### Nettoyage de voix

| Niveau | Moteur | Retire | À savoir |
|---|---|---|---|
| Léger | MossFormer2 | bruit de fond (souffle, ventilateur, ronflement) | le plus fidèle, garde l'articulation |
| Fort | VoiceFixer | bruit, **écho de la pièce**, son « téléphone », saturation | régénère la voix : peut adoucir la diction |
| Maximal | les deux | tout ce qui précède | le plus proche de ton timbre dans nos tests, diction la moins nette |

Mesures sur une voix dégradée comme par un micro médiocre dans une pièce qui résonne (ressemblance du timbre, 1 = identique) : brut 0,78 → Léger 0,82 → Fort 0,87 → Maximal 0,89. La diction se dégrade en revanche avec Fort et Maximal. Fie-toi à ton oreille : c'est pour ça que tu écoutes avant de choisir.

### Contrôle de qualité à l'import

| Vérification | Refus | Avertissement (la voix est quand même enregistrée) |
|---|---|---|
| Durée | moins de 5 s | moins de 10 s ; au-delà de 30 s, seules les 30 premières secondes sont gardées |
| Volume (niveau de la voix, pauses ignorées) | sous −40 dBFS | sous −30 dBFS |
| Saturation | — | plus de 0,1 % des échantillons au maximum |

Chaque message explique quoi changer (se rapprocher du micro, baisser le niveau d'entrée…).

## Réglages utiles

- **Plusieurs versions et graine** : « Versions : 2 » génère deux versions d'un coup (plus long) pour garder la meilleure. Après chaque création, la **graine** s'affiche : remets-la dans « Graine » avec les mêmes réglages pour obtenir un résultat proche (0 = aléatoire). Tout est noté dans `creation.json`, dans le dossier de la création.
- **Style pas respecté ?** La description est transmise telle quelle à ACE-Step. Si le résultat reste trop « pop », décoche le **mode réflexion** : le générateur suit alors la description seule. Pour un style peu courant, répète les mots importants (« 8-bit chiptune, chiptune, retro video game music »).
- **Exclure un instrument** : ne l'écris pas dans la description, même précédé de « sans » ou « no » : le mot suffit à l'ajouter. Utilise plutôt **« Retirer de la musique »** (basse, batterie) : la chanson est séparée en pistes par Demucs et l'instrument est supprimé du mix, c'est garanti (environ 1 minute de plus). Les pistes séparées restent dans le dossier de la chanson (`demucs4\`).

- **Voix chantée de base** : choisis masculine ou féminine selon ta voix. Sinon, joue sur le décalage de hauteur (−12 / +12 demi-tons).
- **Étapes Seed-VC** : 40 par défaut. Monte à 50 pour plus de qualité, au prix du temps.
- **Synthèse vocale** : « Expressivité » à 0,5 = neutre ; pour un ton plus dramatique, monte vers 0,7 et baisse « Guidage / rythme » vers 0,3. Si ton échantillon n'est pas dans la langue du texte, mets « Guidage / rythme » à 0 pour ne pas garder l'accent. La graine (≠ 0) rend un résultat reproductible.

## Limites à connaître

- **Installation** : en cas d'erreur, l'installateur s'arrête avec un message en rouge. Relance-le après correction : les étapes réussies sont sautées.
- **Pilote NVIDIA** : ACE-Step utilise CUDA 12.8, qui demande un pilote récent (570.65 ou plus). L'installateur le vérifie.
- **Fidélité de la voix** : la conversion sans entraînement donne une ressemblance correcte mais pas parfaite, surtout sur les notes aiguës. Pour mieux faire, il faudrait entraîner un modèle sur 10 à 30 minutes de tes enregistrements (par exemple RVC).
- **Artefacts** : la séparation sur de la musique générée laisse parfois de légers résidus.
- **Mémoire graphique et synthèse vocale** : Chatterbox a besoin de 3 à 4 Go de mémoire graphique. Si la fenêtre ACE-Step est ouverte et a déjà chargé ses modèles, la carte peut manquer de mémoire : l'application te demande alors de fermer cette fenêtre, puis de relancer la lecture.
- **Nettoyage** : il ne fait pas de miracle sur une voix très saturée ou noyée dans la musique. Enregistre-toi au calme, à 15–30 cm du micro, c'est toujours le plus efficace.
- **Filigrane** : chaque fichier produit par la synthèse vocale porte un filigrane inaudible ([Perth](https://github.com/resemble-ai/perth)) qui permet de reconnaître une voix de synthèse. Il est ajouté par Chatterbox lui-même.
- **Référence de voix pour la synthèse** : Chatterbox n'utilise que les 10 premières secondes de l'échantillon.
- **Consentement** : clone uniquement ta propre voix, ou celle de personnes d'accord.

## Tests (pour le développement)

Les tests n'ont pas besoin de carte graphique : ACE-Step est remplacé par un faux serveur HTTP, Demucs et Seed-VC par de faux scripts. Depuis le dossier de l'application :

```
uv run --python 3.12 --with-requirements requirements.txt --with pytest pytest tests
```

## Licences

Ce dépôt ne contient que le code de l'application et de l'installation. Les poids de VoiceFixer sont sous licence CC-BY 4.0 (auteurs : Haohe Liu et al., « VoiceFixer: Toward General Speech Restoration with Neural Vocoder », 2021). Chaque moteur garde sa propre licence (voir leurs dépôts respectifs), à vérifier avant tout usage commercial des sons produits.
