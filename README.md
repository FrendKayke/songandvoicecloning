# Studio Voix

Logiciel local pour Windows : tu enregistres ta voix, tu écris un prompt (genre, style, instruments, ambiance) et tes paroles, et il te rend une chanson chantée avec **ta** voix. Il sait aussi composer de la musique seule, lire un texte avec ta voix (synthèse vocale), nettoyer un enregistrement fait avec un micro médiocre, créer des bruitages pour un jeu et des modèles 3D à partir d'une image ou d'un texte. Tout tourne sur ta machine, sans service en ligne.

Moteurs utilisés : [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (génération musicale), [Demucs](https://github.com/facebookresearch/demucs) (séparation voix / musique), [Seed-VC](https://github.com/Plachtaa/seed-vc) (conversion de voix chantée) et [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (synthèse vocale, licence MIT), [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) (MossFormer2, débruitage, Apache-2.0) et [VoiceFixer](https://github.com/haoheliu/voicefixer) (restauration de voix, code MIT, poids CC-BY 4.0) [Applio](https://github.com/IAHispano/Applio) (RVC : entraînement d'un modèle de ta voix, MIT), [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) (bruitages, licence Stability AI Community), [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (descriptions et traductions, Apache 2.0), [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) (modèles 3D, licence Tencent Hunyuan Community) [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (images : illustrations, texte → 3D ; Alibaba, Apache 2.0) [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (personnages récurrents d'après des images de référence ; Black Forest Labs, Apache 2.0), [Wan 2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (vidéos, Alibaba, Apache 2.0), [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (détourage des photos, MIT), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (agrandissement, BSD-3) et [GFPGAN](https://github.com/TencentARC/GFPGAN) (visages, Apache 2.0). Ils sont téléchargés par l'installateur, ils ne sont pas inclus dans ce dépôt.

Configuration testée : Windows 11, NVIDIA RTX 4070 (12 Go).

## Structure du dépôt

| Fichier | Rôle |
|---|---|
| `studio_voix.py` | point d'entrée de l'application (lancé par `lancer.bat`) |
| `studiovoix/` | le code de l'application, un module par rôle (voir ci-dessous) |
| `moteurs/` | scripts exécutés dans l'environnement d'un moteur (`chatterbox_tts.py`, `nettoyage_voix.py`, `rvc_voix.py`, `diffusion.py`, `separation.py` : Demucs et Seed-VC) ; `resident.py` : leur mode « résident » |
| `tests/` | tests automatiques sans carte graphique (moteurs simulés) |
| `installer.ps1` / `INSTALLER.bat` | installation complète en un clic |
| `METTRE_A_JOUR.bat` | mise à jour en un clic (dernière version sur GitHub, puis étapes nouvelles ou modifiées) |
| `installation/` | listes des dépendances de chaque environnement (lues par l'installateur et par la vérification automatique) |
| `lancer.bat` | démarrage (serveur ACE-Step + application) |
| `LANCER_SANS_APPLIS.bat` | ferme d'abord les applications gourmandes en mémoire (Chrome, Discord, Steam, Stream Deck, Synapse, SteelSeries GG, Wallpaper Engine…), puis lance `lancer.bat` ; liste modifiable en tête du fichier |
| `CLAUDE.md` | contexte technique pour Claude Code |

Modules de `studiovoix/` :

| Module | Rôle |
|---|---|
| `config.py` | chemins, variables d'environnement, constantes |
| `acestep.py` | génération de la chanson (API REST d'ACE-Step) |
| `styles.py` | listes déroulantes des styles (libellés français, termes anglais) |
| `jeu.py` | bande-son de jeu : situations, génération en lot, jingles |
| `boucle.py` | boucles parfaites pour les musiques de fond |
| `galerie.py` | galerie des créations : écouter ou afficher (modèles 3D), recréer avec la même graine, refaire un passage, supprimer |
| `export.py` | export OGG/MP3 au volume harmonisé (LUFS), pack complet du jeu (musiques, bruitages, illustrations, cartes composées, modèles 3D) avec `manifest.json` |
| `projets.py` | liste des projets de jeu (tous onglets du groupe Jeu confondus) |
| `demucs.py` | séparation voix / instrumental |
| `seedvc.py` | conversion de voix chantée |
| `chatterbox.py` | synthèse vocale (lance `moteurs/chatterbox_tts.py`) |
| `nettoyage.py` | nettoyage de voix (lance `moteurs/nettoyage_voix.py`) |
| `rvc.py` | RVC : entraîner un modèle de ta voix et convertir avec (lance `moteurs/rvc_voix.py`) |
| `diffusion.py` | client du moteur de diffusion (Qwen3-VL, Stable Audio Open, Z-Image-Turbo, FLUX.2 klein, Hunyuan3D-2 ; lance `moteurs/diffusion.py`), jeton Hugging Face |
| `bruitages.py` | bruitages : description ou image → prompt anglais → variantes |
| `cartes.py` | illustrations de cartes : style mémorisé par projet, formats de carte, variantes, WebP, variante gardée |
| `videos.py` | vidéos : texte → vidéo et image → vidéo (Wan 2.2 TI2V-5B) |
| `photos.py` | photos : améliorer (Real-ESRGAN, GFPGAN), détourer et isoler une personne (BiRefNet) |
| `personnages.py` | personnages récurrents : images de référence par personnage et par projet |
| `compo_cartes.py` | composition des cartes : cadre, nom, coût, type, texte, attaque, défense, rareté (Pillow, polices libres de `polices/`) |
| `espace.py` / `retraits.py` | espace disque : mesure, retrait et réinstallation des moteurs et modèles |
| `diagnostic.py` | diagnostic rapide et essai réel de chaque moteur, rapport à envoyer |
| `serveur_acestep.py` | serveur ACE-Step géré par l'application (démarrage, arrêt pour libérer la carte graphique) |
| `modele3d.py` | modèles 3D : image d'objet → forme puis texture (Hunyuan3D-2), GLB et OBJ |
| `voix.py` | bibliothèque de voix (import, contrôle de qualité, renommage, suppression) |
| `mixage.py` | mixage voix + instrumental |
| `pipeline.py` | enchaînement complet des étapes |
| `modeles.py` | état des modèles (onglet « Modèles ») |
| `outils.py` | journal de commande en direct, ouverture de dossier, lancement des moteurs |
| `residents.py` | moteurs gardés ouverts entre deux générations (modèles en mémoire) |
| `interface.py` | interface Gradio : assemble les onglets |
| `onglets/` | un fichier par onglet (composants et événements) |

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
- **RVC** (Applio), pour entraîner un modèle de ta voix (~2,8 Go de PyTorch et ~1,8 Go de modèles de base) ;
- le **moteur de diffusion** : Qwen3-VL-4B (9 Go), Z-Image-Turbo (15 Go), FLUX.2 klein 4B (4,5 Go), outils photo (2 Go), Wan 2.2 pour la vidéo (20 Go), Hunyuan3D-2 (forme 10 Go : modèle rapide et modèle complet, texture 16 Go), Stable Audio Open (5 Go, **jeton Hugging Face requis**, voir ci-dessous) et ~3 Go de PyTorch ;
- l'environnement de l'application.

Compte 82 à 87 Go à télécharger, soit deux heures ou plus selon ta connexion, et au moins 100 Go libres (tu peux retirer ensuite ce qui ne te sert pas : Outils → Espace disque). Si l'installation s'interrompt, relance INSTALLER.bat : les étapes terminées sont sautées.

### Mettre à jour

Ferme Studio Voix (la fenêtre de lancer.bat), puis double-clique sur **METTRE_A_JOUR.bat**. Il récupère la dernière version sur GitHub, puis relance l'installateur : seules les étapes nouvelles ou modifiées sont refaites (chaque étape garde une empreinte de ses dépendances et refait le travail si elles ont changé). Tes voix, chansons, bruitages et modèles 3D (dossier `data`) ne sont jamais touchés, ni tes modèles RVC entraînés. Il faut [Git](https://git-scm.com/download/win) pour cela ; si tu as modifié des fichiers du code, il te propose de les remplacer par la version de GitHub.

**Juste le code, en vitesse : ACTUALISER.bat.** Il récupère la dernière version du code sans relancer tout l'installateur, en quelques secondes. Il marche aussi **sans Git** : il télécharge alors l'archive du code sur GitHub et la copie par-dessus, sans rien supprimer et sans toucher à `data`. Si la nouvelle version ajoute des dépendances ou des modèles, il te propose de lancer l'installateur. Relance ensuite Studio Voix.

**Tu avais déjà installé Studio Voix avant l'arrivée de la synthèse vocale ou du nettoyage ?** Relance simplement INSTALLER.bat : seules les étapes manquantes s'exécutent (Chatterbox : environ 6 Go, 10 à 25 minutes ; nettoyage : environ 1 Go, 5 minutes ; RVC : environ 5 Go, 10 à 20 minutes ; diffusion : environ 40 Go, une heure). Sans cela, tout le reste fonctionne et l'application t'indique ce qui manque.

### Jeton Hugging Face (pour les bruitages seulement)

Stable Audio Open est distribué sous la licence Stability AI Community (gratuite, usage commercial permis sous 1 M$ de revenus annuels) et Hugging Face demande de l'accepter avant le téléchargement :
1. crée un compte sur https://huggingface.co et accepte la licence sur https://huggingface.co/stabilityai/stable-audio-open-1.0 ;
2. crée un jeton de type « Read » sur https://huggingface.co/settings/tokens ;
3. colle-le quand INSTALLER.bat te le demande (étape 16), ou plus tard dans l'onglet « Modèles » → « Jeton Hugging Face », puis « Télécharger Stable Audio Open ».

Le jeton est enregistré dans `StudioVoix\hf-home\token` et n'est utilisé que pour ce téléchargement. Sans lui, tout le reste fonctionne.

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
| `StudioVoix\diffusion` | moteur de diffusion (environnement, code de Hunyuan3D-2, modèle de détourage dans `u2net\`) ; ses modèles sont dans `StudioVoix\hf-home` |
| `StudioVoix\rvc` | RVC (Applio) : modèles de base dans `rvc\models\`, **tes modèles entraînés dans `logs\<nom>\`**, tes enregistrements d'entraînement dans `datasets\` |
| `StudioVoix\python`, `uv`, `uv-cache` | Python et outils d'installation |
| `<dossier de l'application>\data` | tes voix, tes chansons, tes textes lus, tes voix nettoyées, tes bruitages et tes modèles 3D |

Pour tout désinstaller : supprime `StudioVoix` et le dossier `.venv` de l'application.

## Utilisation

Les onglets sont rangés en cinq groupes, et dans chaque groupe **un onglet = une chose à faire** :

| Groupe | Onglets |
|---|---|
| 🎤 Voix | 🎙️ Mes voix (enregistrer, nettoyer) · 🗣️ Lire un texte avec une voix · 🧠 Entraîner un modèle de ma voix |
| 🎵 Musique | 🎵 Créer une chanson ou une musique · 🎛️ Remixer une musique (autre style) |
| 🖼️ Image et vidéo | 🖼️ Une image (texte ou photo) · 🗂️ Images en série (Excel, liste) · 📖 Histoire en images · 🎬 Une vidéo · 🎞️ Vidéo en plusieurs plans · 📷 Retoucher une photo |
| 🎮 Jeu | 🎼 Musiques de jeu · 🔊 Bruitages · 🃏 Illustrations de cartes · 🧊 Modèles 3D |
| 🧰 Outils | 🗃️ Galerie (toutes mes créations) · ⚙️ Modèles et diagnostic · 💽 Espace disque |

Les paragraphes ci-dessous gardent les anciens noms courts (« Bibliothèque de voix » = 🎙️ Mes voix, « Synthèse vocale » = 🗣️ Lire un texte, « Modèles » = ⚙️ Modèles et diagnostic…).

1. Double-clique sur **lancer.bat**. Il ouvre l'application dans ton navigateur et démarre en arrière-plan le serveur de génération musicale (ACE-Step), sans fenêtre à part. La première génération attend que ce serveur ait fini de charger ses modèles. Pour tout arrêter, ferme la fenêtre de lancer.bat. Si ton PC a peu de mémoire vive libre (beaucoup de logiciels ouverts), lance plutôt **LANCER_SANS_APPLIS.bat** : il ferme d'abord les applications gourmandes ; les modèles se chargent et calculent alors bien plus vite (Windows n'a plus à compresser la mémoire).
2. Onglet **Bibliothèque de voix** : importe un fichier (wav, mp3 ou flac) ou enregistre-toi au micro, 10 à 25 s, dans une pièce calme, sans musique ni écho. Nomme la voix et clique sur « Vérifier et enregistrer ». Dans « Mes voix », tu peux écouter, renommer ou supprimer chaque voix.
   - **Micro médiocre ?** Avant d'enregistrer, ouvre « 🧽 Nettoyer la voix », choisis un niveau et clique sur « Nettoyer l'échantillon ». Écoute la version nettoyée, compare avec l'original, puis choisis dans « Version à enregistrer » celle que tu gardes. Pour une voix déjà enregistrée : « Reprendre cette voix pour la nettoyer », puis enregistre-la sous un nouveau nom.
3. Onglet **Créer une chanson** : choisis le mode, puis genre, style, instruments, ambiance et consignes dans les listes déroulantes (plusieurs choix possibles ; tu peux aussi taper ton propre terme, en anglais de préférence, puis Entrée). La **description envoyée à ACE-Step** s'affiche en dessous, en anglais, et tu peux la retoucher. Écris les paroles : tu peux titrer les parties en français (« Couplet 2 », « Refrain », « Pont », « Intro », « Fin », seuls sur leur ligne), elles sont converties en balises `[Verse 2]`, `[Chorus]`… pour ACE-Step ; un « Refrain » laissé vide reprend le texte du refrain précédent. **Plusieurs voix** : dans « Voix chantées », choisis un duo (homme et femme, deux hommes, deux femmes), une voix principale avec des chœurs ou un chœur ; puis, dans les paroles, écris qui chante chaque partie après son titre : « Couplet 1 (homme) », « Couplet 2 (femme) », « Refrain (ensemble) », « Pont (chœur) » (converti en `[Verse 1 - male vocal]`, `[Chorus - duet, harmonies]`…). Les mots entre parenthèses dans une ligne, « (oh oh) », sont chantés en chœur derrière la voix principale. En mode « Chanson avec ma voix », toutes les voix deviennent la tienne (harmonies avec toi-même) : pour un vrai duo, prends « Chanson avec la voix d'ACE-Step ». Puis « Créer la chanson ». Les termes français courants tapés dans les listes (« médiéval », « taverne », « celtique », ou un libellé de la liste) sont traduits en anglais dans la description. Pour une musique d'époque, prends le genre « Médiéval / musique ancienne » et des instruments d'époque (luth, vielle à roue, cornemuse, flûte à bec…) ; si une basse moderne s'invite quand même, coche « Basse » (et « Batterie ») dans « Retirer de la musique ». Trois modes :
   - **Chanson avec ma voix** (par défaut) : la chanson est chantée avec ta voix ;
   - **Chanson avec la voix d'ACE-Step** : musique seule, la voix générée par ACE-Step est gardée telle quelle (plus rapide : ni séparation ni conversion, pas besoin de voix enregistrée) ;
   - **Instrumental** : musique sans voix, les paroles sont ignorées.
**Remixer une musique** (onglet 🎛️ du groupe Musique) : une musique sans paroles rejouée dans un autre style, par exemple un thème de jeu vidéo en version orchestrale pour un quiz. Dépose un ou plusieurs morceaux (MP3, WAV, FLAC, OGG ; 10 min au plus chacun), choisis le **nouveau style** (orchestral symphonique, quatuor à cordes, piano solo, rock/métal, jazz, big band, lo-fi, 8-bit, synthwave, électro, guitare acoustique, celtique, médiéval, reggae, musique de film… ou le tien en anglais), et la **transformation** : légère (même morceau, autres instruments), moyenne (conseillée) ou forte (réinterprétation libre). Deux curseurs affinent : « Suivre la structure de l'original » et « Garder la mélodie » (0,1 à 0,3 conseillé ; plus haut, la mélodie reste très reconnaissable mais le nouveau style s'impose moins). « Mode très fidèle » garde aussi davantage le son d'origine. Plusieurs morceaux sont remixés l'un après l'autre avec les mêmes réglages ; écoute chaque résultat à côté de l'original et exporte-le en MP3. Tout est dans `data\remix\<date>\` (`original.wav`, `remix_1.wav`) et dans la Galerie (« Recréer » = même graine). Moteur : ACE-Step 1.5 (tâche « cover »). Une recherche du 07/10/2026, en acceptant les licences non commerciales puisque c'est pour un usage personnel, n'a rien trouvé de mieux pour garder la mélodie sur une carte de 12 Go (MuLaCover ne tourne que sous Linux et vise les reprises chantées ; MusicGen-Melody se limite à 30 s avec une qualité inférieure ; Stable Audio Open à 47 s, mélodie mal gardée). Ne remixe que des musiques que tu as le droit d'utiliser.

4. Onglet **Bande-son de jeu** : musiques sans voix pour un jeu. Donne un nom de projet, choisis l'époque (16-bit par défaut, 8-bit, orchestral…), l'univers et les situations (écran titre, menu, plateau, tension, combat, boss, boutique, et les jingles victoire, défaite, booster, carte rare). Tu peux taper ta propre situation en anglais (« fire faction theme, taiko drums »). Le tableau montre la description envoyée pour chaque piste. « Générer la bande-son » les crée une par une dans `data\jeux\<projet>\<situation>\<date>\` (`piste.wav`). Les jingles sont générés sur 10 s (le minimum d'ACE-Step) puis coupés proprement à leur durée. Les musiques de fond deviennent des **boucles parfaites** : le fichier `piste.wav` tourne en boucle sans coupure (`loop: true` dans Howler.js, `<audio loop>`…) ; l'aperçu « Jonction » te fait écouter le passage fin → début, et le message donne la qualité de la jonction. Pendant que la carte graphique compose une piste, le processeur cherche déjà la boucle de la précédente.
   - **Une bande-son cohérente** (« 🎼 Cohérence ») : génère d'abord un thème principal (l'écran titre, par exemple), puis choisis-le comme *thème de référence*. « Même son » donne à toutes les pistes son timbre et son mixage ; « Variation du thème » réarrange sa mélodie selon chaque situation (version combat, version calme…), comme les leitmotivs des JRPG. Le curseur « Fidélité » règle la ressemblance. En variation, les musiques de fond prennent la durée du thème.
   - **Pack complet du jeu** (« 📦 ») : toutes les pistes du projet (la plus récente de chaque situation) et ses bruitages (variante gardée) au même volume (−16 LUFS conseillé), en OGG et MP3, plus ses illustrations (variante gardée, WebP), ses cartes composées et ses modèles 3D (version web quand elle existe), avec un seul `manifest.json`, dans `data\jeux\<projet>\export\` et dans l'archive `<projet>_pack.zip`. Le **projet** est une liste commune à tous les onglets du groupe Jeu : donne le même nom à tes musiques, bruitages, illustrations et modèles 3D.
   - Dans l'onglet « Créer une chanson », « 💾 Exporter la chanson en MP3 » fait de même pour une chanson (−14 LUFS par défaut).
5. Onglet **Synthèse vocale** : choisis une voix de ta bibliothèque, la langue, écris le texte et clique sur « Lire le texte avec cette voix ». Les textes longs (jusqu'à 5 000 caractères) sont découpés en phrases. Chaque lecture est rangée dans `data\tts\<date>\` (texte et `parole.wav`).
6. Onglet **Galerie** : toutes tes créations (chansons, pistes de jeu, lectures, bruitages, modèles 3D), la plus récente en premier, avec un filtre. Pour chacune : description, paroles, graine, écoute (et choix de la version s'il y en a deux), « 🔁 Recréer (même graine) » pour obtenir un résultat proche, « 📂 Ouvrir le dossier » et « 🗑️ Supprimer » (avec confirmation). Les créations faites avant cette version apparaissent aussi (écoute et suppression seulement).
   - **« ✏️ Refaire un passage »** : un refrain raté, une fin bizarre ? Indique le début et la fin en secondes : seul ce passage est réinventé, le reste est gardé, puis le reste du traitement est refait (ta voix, retrait d'instruments, boucle…). Tu peux changer la description ou les paroles du passage et choisir la force de la retouche (légère, équilibrée, complète). Le résultat est une nouvelle création ; l'originale est conservée.
7. Onglet **Entraîner ma voix (RVC)** : pour une ressemblance nettement meilleure, surtout au chant. Ajoute 10 à 30 minutes d'enregistrements de toi seul (plusieurs fichiers wav/mp3/flac, et/ou des voix de ta bibliothèque) ; le tableau vérifie la durée totale et signale les fichiers trop faibles ou saturés. Donne un nom, choisis la durée (300 époques conseillées, 1 à 2 heures sur une RTX 4070) et lance. Ensuite, dans « Créer une chanson » → Réglages voix → **Conversion de ta voix**, choisis « RVC — ton modèle ». Dans « Synthèse vocale », tu peux aussi faire passer la lecture dans ton modèle RVC. ACE-Step est arrêté automatiquement pendant l'entraînement, qui a besoin de toute la mémoire graphique.
8. Onglet **Bruitages** : décris l'effet en français (ou choisis un exemple, ou importe une image de la scène), clique sur « Préparer le prompt » : Qwen3-VL le traduit en un prompt anglais précis, que tu peux retoucher. Règle la durée (1 à 30 s), le nombre de variantes (1 à 3), la graine, puis « Générer ». Écoute les variantes et exporte celle que tu gardes (OGG, MP3, WAV, au volume harmonisé). Tout est rangé dans `data\bruitages\<date>\` et visible dans la Galerie. ACE-Step est arrêté automatiquement avant : ces modèles ont besoin de la carte graphique.
9. Onglet **Modèles 3D** : importe une image de l'objet (PNG ou JPG : un seul objet, net, sur un fond simple ; une carte, un personnage, une arme…), donne un nom, choisis la qualité (« Normale » convient ; « Fine » demande plus de mémoire et de temps ; « Maximale » utilise le modèle de forme complet de Hunyuan3D-2, 50 étapes au lieu de 5 : les détails les plus fins, une à deux minutes de plus), garde « Peindre la texture » coché pour un modèle coloré, et clique sur « Créer le modèle 3D ». Hunyuan3D-2 retire le fond (l'image détourée s'affiche), sculpte la forme (quelques dizaines de secondes) puis peint la texture (plusieurs minutes). Le modèle s'affiche dans la visionneuse (glisser pour tourner, molette pour zoomer). Fichiers dans `data\3d\<date>\` : `modele.glb` (texturé, prêt pour un jeu web avec three.js ou Babylon.js, Blender, Unity, Godot), `forme.glb` (forme blanche) et, si tu coches OBJ, `modele.obj` + `material.mtl` + la texture PNG. La graine permet de retrouver le même modèle ; la Galerie les affiche aussi. Le champ « Projet de jeu » range le modèle dans le pack du jeu (le plus récent de chaque nom, en version web s'il y en a une). **Pour un jeu web** : la qualité « Web léger » garde 10 000 faces et ajoute `modele_web.glb` (texture 1024 px en JPEG, souvent dix fois plus petit) ; le bouton « 🪶 Alléger pour le web » fait de même pour le dernier modèle créé. **Plusieurs objets** : le volet « 📚 Plusieurs images à la suite » crée un modèle par image avec les mêmes réglages ; une image qui échoue n'arrête pas les suivantes. **Pas d'image ?** Ouvre « Décris l'objet » : écris-le en français, « Préparer le prompt » le traduit en un prompt d'image précis (objet seul, vue de trois quarts, fond blanc), « Générer l'image de l'objet » la crée avec Z-Image-Turbo (quelques secondes) et la place dans le champ image ; regarde-la, change la graine ou le prompt si elle ne convient pas, puis lance la 3D comme avec une image importée. ACE-Step est arrêté automatiquement avant : la texture a besoin de toute la carte graphique.
10. Onglet **Illustrations de cartes** : choisis un **projet** (le nom de ton jeu) ; son style est mémorisé et réappliqué à chaque carte, pour que toutes se ressemblent. Choisis le style dans la liste (peinture fantasy, carte à collectionner, aquarelle, anime, pixel art 16-bit…, plusieurs à la fois, ou tape le tien en anglais) et le format (illustration paysage 4:3, carte entière 2:3, carré, décor 16:9). Décris la scène en français, « Préparer le prompt » la traduit et la précise, tu peux la retoucher, puis « Générer l'illustration » : 1 à 4 images, chacune avec sa graine.
   - **Contexte commun + une description par image** : le champ « Contexte commun » (en anglais, envoyé tel quel ; ou écris en français dans « Scène en français » puis « Préparer le prompt ») décrit ce qui vaut pour toutes les images : les personnages, le décor, l'ambiance. Puis, selon le nombre d'images choisi, un champ « Image 1 », « Image 2 »… décrit ce que montre chacune (en anglais) ; laissé vide, l'image reprend seulement le contexte (même scène, autre graine). Chaque image reçoit « description. contexte » suivi du style (la description d'abord : si le texte est trop long pour le générateur, c'est la fin du contexte qui est ignorée, avec un avertissement). Pour un groupe, **donne le nombre et le genre de chacun** (« four men, all male, bearded… ») et décris chaque personnage : le modèle a tendance à ajouter des femmes ou des personnages quand le groupe n'est pas précisé. Évite les négations (« no women ») : ce modèle n'a pas de prompt négatif et un mot nié a tendance à apparaître. Si une image ressemble à une planche de BD (plusieurs cases), retire les styles « comic » / « manga » et précise « single illustration ». Dans la Galerie, « Recréer » reprend le contexte et la description de l'image choisie. Les images sont dans `data\cartes\<projet>\<date>\` en PNG et en WebP (léger pour un jeu web), et dans la Galerie. Z-Image-Turbo est sous licence Apache 2.0 : tu peux utiliser et vendre ces images. Il n'a pas de « prompt négatif » : décris ce que tu veux voir plutôt que ce que tu ne veux pas. **Clique sur la variante que tu gardes** : c'est elle qui part dans le pack du jeu, et elle est placée dans le volet de composition.
   - **Personnages récurrents** (volet « 👤 ») : pour retrouver le même héros, le même monstre sur plusieurs cartes. Un personnage, c'est 1 à 4 images de référence : génère d'abord une illustration nette du personnage (en pied, fond simple), clique sur la variante réussie, tape son nom et « ⭐ Ajouter la variante gardée à ce personnage » ; ajoute si tu veux un gros plan du visage ou importe tes propres images. Ensuite, choisis-le dans « Personnage de la scène » : la scène est alors peinte par **FLUX.2 klein 4B** (Apache 2.0, images vendables) à partir de ses références, avec le même visage, la même coiffure et la même tenue. Dans la scène, décris la pose, l'action et le décor, pas son apparence. Les références sont dans `data\cartes\<projet>\personnages\<nom>\` ; supprimer un personnage garde les illustrations déjà faites.
   - **Photo modèle** (volet « 📷 », facultatif) : envoie une photo (toi, un ami d'accord, un objet). Au choix, la personne ou l'objet de la photo devient le sujet de la carte, redessiné dans le style du projet avec le même visage et les mêmes traits, ou la carte reprend seulement sa pose et son cadrage. Décris quand même la scène (décor, action). La carte est alors peinte par FLUX.2 klein ; la photo est copiée avec la création (la Galerie peut la recréer).
   - **Composer la carte** (volet « 🃏 ») : nom, coût, type, texte d'effet, texte d'ambiance (italique), attaque, défense (vides = pas de pastille), faction (couleur du cadre : Feu, Eau, Nature, Lumière, Ténèbres, Neutre), rareté (couleur du symbole) et pied de carte. La carte est dessinée autour de l'illustration au format des cartes à collectionner (63×88 mm : 750×1050 px à 300 ppp, ou 1500×2100 en haute définition) ; un texte long rapetisse tout seul pour tenir. Résultat dans `data\cartes\<projet>\composees\` : PNG prêt à imprimer, WebP pour le web et un `.json` avec les champs (le pack du jeu les reprend). Polices Cinzel et EB Garamond (licence SIL OFL, usage commercial permis), fournies dans `polices\`.
11. Onglet **🖼️ Une image** (groupe 🖼️ Image et vidéo) : décris l'image voulue en français, « Préparer le prompt » la traduit et la précise, puis « Générer l'image » (1 à 4 variantes, en moins d'une minute). Sans photo, Z-Image-Turbo la crée à partir du texte. Avec une **photo**, FLUX.2 klein au choix la **modifie** (« mets-lui un chapeau », « la même scène la nuit » : mêmes personnes, même cadrage), garde son **sujet** dans une nouvelle scène, ou reprend sa **composition** (pose, cadrage). Tu peux donner jusqu'à **4 photos** (la photo principale + 3 autres) et les désigner par leur numéro : « la femme de la photo 1 dans le château de la photo 2 », « mets-lui la veste de la photo 2 ». Style facultatif (photo réaliste, peinture, aquarelle…) ; format automatique d'après la photo. Clique sur une variante, puis « Retoucher cette image » pour enchaîner les modifications, ou « Animer » pour l'envoyer dans l'onglet Vidéos. Le temps de génération est affiché.
    **Histoire en images** (onglet 📖 du groupe Image et vidéo ; il a ses propres style, format, graine et nom) : choisis le nombre d'images (2 à 12) : autant de champs « Image 1 », « Image 2 »… apparaissent. Écris un **contexte commun** en anglais (envoyé tel quel devant chaque image : les personnages avec leur nombre et leur genre, par exemple « four bearded men, all male », leur apparence, le lieu, l'époque), puis décris chaque image dans son champ (en anglais) ; un champ laissé vide reprend le contexte seul. Chaque image reçoit sa description, puis le contexte. Le générateur lit environ 750 mots au plus par image (1 024 jetons) : au-delà, la fin du contexte est ignorée et un avertissement s'affiche ; décris chaque personnage en une ou deux phrases plutôt qu'un paragraphe. Ou colle un texte (français ou anglais) et « Découper en scènes » remplit les champs (Qwen tient compte du contexte), modifiables avant « Générer les images de l'histoire » ; avec « Mêmes personnages », les images suivantes reprennent les personnages et le style de la première. Tu peux aussi déposer jusqu'à 3 **images de départ** (tes personnages, une créature, un lieu, un style) : le découpage en tient compte et chaque image les reprend, dans une nouvelle pose. « Faire les mini-vidéos » anime ensuite chaque image (2 ou 3 s chacune, environ 4 à 7 minutes par image sur une RTX 4070) et peut les assembler en une seule vidéo.
11. Onglet **🗂️ Images en série** (groupe 🖼️ Image et vidéo) : beaucoup d'images d'un coup dans le même style, par exemple 250 icônes carrées pour une application, **une image par ligne d'un tableau**.
    - Dépose ton fichier **Excel (.xlsx) ou CSV** : la feuille non vide et les colonnes sont proposées toutes seules (titre « Prompt », « Nom », « Contexte »), vérifie-les ; un aperçu du tableau s'affiche. Pas de fichier ? Colle tes prompts, un par ligne. Un ancien `.xls` ou un `.ods` LibreOffice : enregistre-le d'abord en `.xlsx` ou CSV.
    - **Contexte commun** : écris-le, et/ou indique la **case** du fichier qui le contient (« B1 », ou « Feuil1!B1 ») ; une **colonne de contexte** donne en plus un contexte propre à chaque ligne. Chaque image reçoit « prompt. contexte de la ligne. contexte commun » puis le style.
    - **Style** : liste avec des styles d'icônes en tête (icône d'application plate, icône 3D douce, contour, pixel art, autocollant) et ceux de l'onglet Une image, ou le tien en anglais. Pour un rendu encore plus uniforme, ajoute 1 à 3 **images de style** (une icône que tu aimes) : FLUX.2 klein en reprend le rendu (couleurs, trait, ombrage), pas le sujet. « Même graine pour toutes » rend aussi la série plus homogène.
    - « 👁️ Voir les prompts » affiche exactement ce qui sera envoyé pour chaque image. Essaie d'abord sur 5 lignes (« Seulement les N premières »), puis lance tout. Compte 15 à 20 s par image 1024×1024 sur une RTX 4070 (250 images ≈ 1 h 15).
    - Résultat dans `data\series\<date>_<nom>\` : `001_<nom>.png`, `002_…`, les **copies réduites** choisies (`512px\`, `128px\`, `64px\`…, pour les icônes d'une application), `lot.csv` (s'ouvre dans Excel : numéro, nom, prompt, graine, fichier, fait ou non) et une archive zip de tout. Si le lot s'arrête (erreur, PC éteint), « ▶️ Reprendre le lot » refait seulement les images manquantes (indique le dossier du lot si l'application a été relancée).
    - La lecture des fichiers Excel demande une petite bibliothèque (openpyxl) installée par INSTALLER.bat ; les CSV se lisent sans rien installer.
11. Onglet **Photos** (groupe 🖼️ Image et vidéo) : importe une photo et choisis le traitement.
   - **Améliorer la qualité** : Real-ESRGAN agrandit ×2 ou ×4 et retire le bruit, les artefacts JPEG et le flou ; ×1 nettoie sans agrandir. « Restaurer les visages » passe chaque visage dans GFPGAN ; baisse la force si le visage paraît trop lisse. Le mode rapide utilise un modèle de 5 Mo, un peu moins fin.
   - **Détourer** : BiRefNet garde le sujet (objet, animal, personne), même les cheveux fins, sur un fond transparent (PNG), blanc, noir, vert d'incrustation, une couleur au choix ou son propre fond flouté (effet portrait). Le masque est enregistré à côté.
   - **Isoler une personne** : le même outil, entraîné sur des personnes : il écarte ce qui n'est pas la personne.
   - Le curseur « Avant / après » compare les deux. « Continuer avec ce résultat » enchaîne les traitements (améliorer une vieille photo, puis isoler la personne). Tout est dans `data\photos\<date>\` et dans la Galerie, où « Recréer » refait le même traitement. Licences libres : tu peux vendre les résultats.
12. Onglet **Vidéos** (groupe 🖼️ Image et vidéo) : décris la vidéo en français (ou choisis un exemple), « Préparer le prompt » la traduit et la précise (action, décor, lumière, mouvement de caméra), retouche-la si besoin. Ajoute une **image de départ** pour animer une illustration de carte ou une photo : la vidéo part de cette image (le format automatique suit sa forme : paysage, portrait ou carré). Choisis la durée (2 à 5 s, 24 images/s), le nombre d'étapes (30 conseillé ; moins = plus rapide) et clique sur « Générer la vidéo ». **C'est long** : une dizaine de minutes ou plus pour 5 s en 720p sur une RTX 4070 (estimation), davantage la première fois ; les formats « légers » (832×480) vont nettement plus vite. « Continuer » prend la dernière image comme départ du clip suivant, pour enchaîner plusieurs plans à la main. **Plusieurs plans d'un coup** : onglet « 🎞️ Vidéo en plusieurs plans », un plan par ligne (12 au plus) ; chaque plan part de la dernière image du précédent, et tous sont assemblés en une seule vidéo (`video.mp4`, les clips restent dans `plan_1\`, `plan_2\`…). Si un plan échoue, les précédents sont gardés et assemblés. Au fil des raccords l'image peut perdre un peu en netteté : préfère des plans courts et des descriptions qui gardent les mêmes personnages et le même décor. La vidéo (MP4 H.264, lisible partout) est muette : ajoute bruitages et musique depuis leurs onglets. Tout est dans `data\videos\<date>\` et dans la Galerie (« Recréer » = même graine). Wan 2.2 est sous licence Apache 2.0 : tu peux vendre tes vidéos.
Dans les onglets Bruitages, Illustrations, Vidéos et Modèles 3D, « Générer » prépare tout seul le prompt anglais si tu ne l'as pas fait (il s'affiche ensuite, modifiable) ; un prompt déjà préparé ou retouché est gardé tel quel.

13. **Carte graphique** : tu n'as rien à fermer, l'application fait une génération à la fois et donne toute la carte à celle qui tourne. Réglages dans Outils → Modèles → « 🎛️ Carte graphique : performances et serveur ACE-Step » :
    - **Modèles gardés en mémoire** (activé) : la diffusion (images, illustrations, 3D, vidéos, photos, bruitages), la synthèse vocale, le nettoyage, Demucs et Seed-VC restent ouverts entre deux générations (Demucs et Seed-VC gardent même leurs modèles en mémoire vive pendant qu'ACE-Step compose la chanson suivante) ; la suivante démarre sans relire ses modèles sur le disque (de quelques secondes à une minute gagnées à chaque fois). Un seul moteur ouvert à la fois, fermé avant une chanson et après 15 minutes sans génération (`STUDIOVOIX_GARDER_MODELES` dans `lancer.bat` pour changer ce délai, 0 pour désactiver ; `STUDIOVOIX_MEMOIRE_MODELES` pour la mémoire vive qu'ils peuvent occuper, par défaut la mémoire du PC moins 12 Go). Le bouton « ⏏️ Fermer le moteur ouvert » rend la mémoire tout de suite (avant de lancer un jeu, par exemple).
    - **Modèle de langage d'ACE-Step** : 1.7B par défaut. ACE-Step classe les cartes de 12 Go (une RTX 4070 annonce 11,99 Go) avec les cartes de 8 Go et leur donnait son plus petit modèle (0.6B) ; le 1.7B, celui que sa documentation conseille, donne des chansons mieux construites. Si la mémoire manque (autre programme sur la carte), repasse en 0.6B.
    - La mise à jour réinstalle ACE-Step en Python 3.11 (une fois, quelques minutes) : c'est la seule version où son modèle de langage a son moteur rapide sous Windows ; le diagnostic rapide indique le moteur et le modèle utilisés.
    - Pour voir si la carte travaille vraiment : « 📈 Mesurer la carte graphique pendant 1 minute », pendant qu'une génération tourne (le Gestionnaire des tâches de Windows montre par défaut le graphe « 3D », presque à zéro pendant un calcul d'IA : choisis plutôt le graphe « Cuda »). L'**essai complet** (volet « 🩺 Diagnostic et essai complet des moteurs », juste en dessous) note aussi l'utilisation de la carte pendant chaque moteur.
    - **Ollama** : s'il tourne sur le PC, il garde le dernier modèle utilisé sur la carte graphique (5 minutes par défaut). L'application lui demande de le décharger avant chaque génération (Ollama reste lancé et le rechargera à son prochain usage) ; `STUDIOVOIX_LIBERER_OLLAMA=0` dans `lancer.bat` pour ne jamais y toucher. Le diagnostic rapide liste les modèles d'Ollama chargés et les programmes qui utilisent la carte.
    - Conseil NVIDIA : Panneau de configuration NVIDIA → Gérer les paramètres 3D → « CUDA - Sysmem Fallback Policy » → « Prefer No Sysmem Fallback » : sinon, quand les 12 Go sont pleins, le pilote déborde en silence dans la mémoire vive et tout devient très lent.
    - ACE-Step occupe environ 8 Go de mémoire graphique ; l'application l'arrête d'elle-même avant les autres moteurs, puis le relance à la chanson suivante (une minute de chargement en plus à ce moment-là).

Chaque chanson est rangée dans `data\songs\<date>\` : version brute, voix convertie, instrumental, mix final et prompt.

## Comment ça marche

1. **ACE-Step 1.5** génère la chanson complète, avec une voix chantée générique.
2. **Demucs** (`htdemucs_ft`, sa meilleure séparation : quatre modèles affinés, un par piste, avec deux passes moyennées) sépare la voix de l'instrumental.
3. **Seed-VC** (conversion de voix chantée, sans entraînement) remplace cette voix par la tienne.
4. L'application remixe voix et instrumental.

En mode « voix d'ACE-Step » ou « Instrumental », seule l'étape 1 a lieu, sauf si tu retires un instrument : Demucs sépare alors la chanson en 4 pistes (voix, batterie, basse, autres) et l'instrumental est remixé sans l'instrument retiré.

### Utiliser le pack dans ton jeu web

Chaque piste en boucle tourne sans coupure sur le fichier entier : pas besoin de points de boucle. Exemple avec [Howler.js](https://howlerjs.com/) :

```js
const manifest = await (await fetch("audio/manifest.json")).json();
const sons = {};
for (const p of manifest.pistes) {
  sons[p.id] = new Howl({
    src: [`audio/${p.fichiers.ogg}`, `audio/${p.fichiers.mp3}`], // OGG d'abord, MP3 en secours
    loop: p.boucle,
    html5: p.boucle, // lecture en flux pour les longues musiques
  });
}
for (const b of manifest.bruitages) {  // bruitages du projet (dans audio/bruitages/)
  sons[b.id] = new Howl({ src: [`audio/${b.fichiers.ogg}`, `audio/${b.fichiers.mp3}`] });
}
sons.combat.play();          // musique de fond en boucle
sons.victoire.play();        // jingle
sons["epee"].play();         // bruitage « Épée » (identifiant sans accents)
```

Le volume est déjà harmonisé entre les pistes et les bruitages : règle seulement le volume général dans ton jeu. Pour mettre un bruitage dans le pack, donne-lui le nom du projet (champ « Projet de jeu » de l'onglet Bruitages) et clique sur « ⭐ Garder cette variante pour le pack du jeu » ; pour chaque nom de bruitage, le pack prend le plus récent.

Les images et les modèles 3D sont dans le même pack, décrits par le même `manifest.json` :

```js
for (const c of manifest.cartes) {        // cartes composées : c.fichier = "cartes/dragon-ancien.webp"
  ajouterCarte({ image: `audio/${c.fichier}`, nom: c.nom, cout: c.cout, attaque: c.attaque, defense: c.defense });
}
// manifest.illustrations : illustrations seules (fond, menu…) ; manifest.modeles3d : GLB pour three.js
new GLTFLoader().load(`audio/${manifest.modeles3d[0].fichier}`, (gltf) => scene.add(gltf.scene));
```

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

## Bien enregistrer sa voix (le plus important pour la ressemblance)

Un casque-micro, même bon, capte une voix étroite et « nasale » : il est fait pour être compris, pas pour être fidèle. Pour cloner ta voix, **un vrai micro de studio change tout**, bien plus que n'importe quel réglage.

### Quel micro ?

Dans une pièce non traitée (le cas de la plupart des chambres et bureaux), préfère un **micro dynamique** : il capte ta voix de près et beaucoup moins l'écho et les bruits de la pièce. Un **micro statique** (« à condensateur ») est plus détaillé mais capte aussi toute la pièce : réserve-le à une pièce calme et amortie.

| Budget | Micro | Type | Branchement | Pour qui |
|---|---|---|---|---|
| ~70 € | **Samson Q2U** (ou Fifine K688) | dynamique | USB **et** XLR | le meilleur rapport qualité-prix, pièce bruyante |
| ~150 € | **Audio-Technica AT2020USB-X** | statique | USB | pièce calme, voix très naturelle |
| ~170 € | **Rode NT-USB+** | statique | USB | pièce calme, excellent pour le chant |
| ~250 € | **Shure MV7+** (ou MV7X en XLR) | dynamique | USB et XLR | le choix sûr en pièce non traitée, voix et chant |
| ~250 € | **Shure SM58** + interface **Focusrite Scarlett Solo** | dynamique | XLR | le classique du chant, évolutif (tu pourras changer de micro plus tard) |

Les prix sont indicatifs (vérifie les prix actuels et les versions récentes de ces modèles). Ajoute un **filtre anti-pop** (~10 €) et un pied ou un bras articulé : le micro ne doit pas être tenu à la main.

### Comment enregistrer

- **Distance** : 5 à 10 cm pour un dynamique, 15 à 20 cm pour un statique, légèrement de biais pour éviter les « p » et les souffles.
- **La pièce** : coupe ventilateurs et climatisation, ferme la fenêtre ; enregistre face à un mur couvert (rideaux, armoire ouverte pleine de vêtements, couette tendue derrière toi) plutôt que dans une pièce vide et carrelée.
- **Le niveau** : règle le gain pour que tes passages les plus forts montent vers −12 à −6 dB, jamais au maximum (le contrôle de qualité signale la saturation).
- **Windows** : dans Paramètres → Son → ton micro, désactive les « améliorations audio » et toute réduction de bruit ; elles abîment le timbre.
- **Le navigateur** applique souvent sa propre suppression de bruit et d'écho au micro. Pour les enregistrements importants (surtout ceux pour RVC), enregistre plutôt avec **Audacity** (gratuit) en WAV 48 kHz, 24 bits, mono, puis importe le fichier.
- **Pour RVC** : 10 à 30 minutes au total, en plusieurs fichiers, ta voix seule : chante des mélodies variées (graves, aiguës, douces, fortes), parle, lis un texte à voix haute. Pas de musique ni de chœurs, pas d'effets (réverbération, autotune).

## Réglages utiles

- **Plusieurs chansons d'un coup et graine** : « Nombre de chansons » (1 à 8) crée plusieurs chansons avec les mêmes réglages et des graines différentes, pour garder la meilleure ; ACE-Step en compose 2 à la fois (limite des 12 Go), les suivantes à la suite. La liste « Écouter les autres chansons » les fait entendre une à une. Après chaque création, la **graine** s'affiche : remets-la dans « Graine » avec les mêmes réglages pour obtenir un résultat proche (0 = aléatoire). Tout est noté dans `creation.json`, dans le dossier de la création.
- **Style pas respecté ?** La description est transmise telle quelle à ACE-Step. Si le résultat reste trop « pop », décoche le **mode réflexion** : le générateur suit alors la description seule. Pour un style peu courant, répète les mots importants (« 8-bit chiptune, chiptune, retro video game music »).
- **Exclure un instrument** : ne l'écris pas dans la description, même précédé de « sans » ou « no » : le mot suffit à l'ajouter. Utilise plutôt **« Retirer de la musique »** (basse, batterie) : la chanson est séparée en pistes par Demucs et l'instrument est supprimé du mix, c'est garanti (environ 1 minute de plus). Les pistes séparées restent dans le dossier de la chanson (`demucs4\`).

- **Voix chantées** : choisis masculine ou féminine selon ta voix. Sinon, joue sur le décalage de hauteur (−12 / +12 demi-tons).
- **Étapes Seed-VC** : 40 par défaut. Monte à 50 pour plus de qualité, au prix du temps.
- **Synthèse vocale** : « Expressivité » à 0,5 = neutre ; pour un ton plus dramatique, monte vers 0,7 et baisse « Guidage / rythme » vers 0,3. Si ton échantillon n'est pas dans la langue du texte, mets « Guidage / rythme » à 0 pour ne pas garder l'accent. La graine (≠ 0) rend un résultat reproductible.

## Limites à connaître

- **Installation** : en cas d'erreur, l'installateur s'arrête avec un message en rouge. Relance-le après correction : les étapes réussies sont sautées.
- **Place sur le disque** : les moteurs et modèles occupent environ 100 Go. Outils → **Espace disque** mesure la place de chacun et retire ceux que tu n'utilises pas (par exemple la texture 3D, 16 Go). Un élément retiré n'est plus réinstallé par les mises à jour ; « Réinstaller » puis METTRE_A_JOUR.bat le remet. ACE-Step, Seed-VC, tes créations et tes modèles RVC entraînés ne sont jamais supprimés.
- **Quelque chose ne marche pas ?** Outils → Modèles → « 🩺 Diagnostic et essai complet des moteurs » (volet ouvert) : le diagnostic rapide (moins d'une minute) vérifie la carte graphique, l'espace disque, chaque moteur et les modèles ; l'essai complet (10 à 20 minutes) fait une génération courte avec chaque moteur et note sa durée et la mémoire graphique utilisée. Envoie le fichier `rapport.txt` proposé : il dit précisément ce qui ne va pas. Toute la sortie de chaque moteur (temps de chargement des modèles, erreurs complètes) est aussi enregistrée dans `data\journaux\` (`diffusion.log`, `separation.log`…) : envoie celui du moteur concerné si une génération est anormalement lente.
- **Pilote NVIDIA** : ACE-Step utilise CUDA 12.8, qui demande un pilote récent (570.65 ou plus). L'installateur le vérifie.
- **Fidélité de la voix** : la conversion sans entraînement (Seed-VC) donne une ressemblance correcte mais pas parfaite, surtout sur les notes aiguës. Pour mieux faire, entraîne un modèle RVC sur 10 à 30 minutes de tes enregistrements (onglet « Entraîner ma voix »). La qualité de ton micro et de ta pièce compte encore plus que la durée.
- **Artefacts** : la séparation sur de la musique générée laisse parfois de légers résidus.
- **Mémoire graphique et synthèse vocale** : Chatterbox a besoin de 3 à 4 Go de mémoire graphique. ACE-Step est arrêté automatiquement avant la lecture ; si la carte manque quand même de mémoire (un jeu ou une vidéo ouverts), l'application te le dit.
- **Nettoyage** : il ne fait pas de miracle sur une voix très saturée ou noyée dans la musique. Enregistre-toi au calme, à 15–30 cm du micro, c'est toujours le plus efficace.
- **Mémoire graphique et diffusion** : Hunyuan3D, Z-Image, FLUX.2 klein, Qwen3-VL et Stable Audio passent tour à tour sur la carte (un seul à la fois, les autres attendent en mémoire vive) et occupent jusqu'à 11 Go. ACE-Step est arrêté automatiquement avant ; l'application te prévient si la mémoire manque quand même.
- **Modèles 3D** : Hunyuan3D-2 travaille à partir d'une seule image : le dos de l'objet est inventé, les parties fines (lames, cheveux, anses) peuvent être épaissies ou coupées, et la texture est parfois floue au dos. Une image bien cadrée sur fond uni change tout. Si la carte manque de mémoire pendant la texture, décoche « Peindre la texture » : la forme seule tient dans 6 Go. Les modèles sortent normalisés (environ 1 unité de côté) : remets-les à l'échelle dans ton moteur de jeu.
- **Filigrane** : chaque fichier produit par la synthèse vocale porte un filigrane inaudible ([Perth](https://github.com/resemble-ai/perth)) qui permet de reconnaître une voix de synthèse. Il est ajouté par Chatterbox lui-même.
- **Référence de voix pour la synthèse** : Chatterbox n'utilise que les 10 premières secondes de l'échantillon.
- **Consentement** : clone uniquement ta propre voix, ou celle de personnes d'accord.

## Tests (pour le développement)

À chaque envoi sur GitHub, une vérification automatique (`.github/workflows/verification.yml`, onglet « Actions » du dépôt) lance les tests sous Linux et Windows, analyse l'installateur (compatibilité PowerShell 5.1) et vérifie que les dépendances de chaque environnement se résolvent pour Windows (`installation/verifier_dependances.py`, utilisable aussi en local). Les tests n'ont pas besoin de carte graphique : ACE-Step est remplacé par un faux serveur HTTP, Demucs et Seed-VC par de faux scripts. L'installateur est simulé sous Linux avec PowerShell 7 et de faux `uv.exe` / `python.exe` (`tests/test_installateur.py`, ignoré sans `pwsh`). Depuis le dossier de l'application :

```
uv run --python 3.12 --with-requirements requirements.txt --with pytest --with pyloudnorm pytest tests
```

## Licences

Ce dépôt ne contient que le code de l'application et de l'installation. Hunyuan3D-2 (Tencent) est sous licence Tencent Hunyuan Community : **elle exclut l'Union européenne, le Royaume-Uni et la Corée du Sud** (lieu d'utilisation) ; vérifie qu'elle s'applique à toi. Stable Audio Open et ses sorties : licence Stability AI Community (usage commercial permis sous 1 M$ de revenus annuels). Applio (RVC) est sous licence MIT ; ses conditions d'utilisation demandent de n'utiliser que des voix dont tu as le droit (la tienne, ou avec l'accord de la personne). Les poids de VoiceFixer sont sous licence CC-BY 4.0 (auteurs : Haohe Liu et al., « VoiceFixer: Toward General Speech Restoration with Neural Vocoder », 2021). Chaque moteur garde sa propre licence (voir leurs dépôts respectifs), à vérifier avant tout usage commercial des sons produits.
