# Studio Voix

Logiciel local pour Windows : tu enregistres ta voix, tu écris un prompt (genre, style, instruments, ambiance) et tes paroles, et il te rend une chanson chantée avec **ta** voix. Il sait aussi composer de la musique seule, lire un texte avec ta voix (synthèse vocale), nettoyer un enregistrement fait avec un micro médiocre, créer des bruitages pour un jeu et des modèles 3D à partir d'une image ou d'un texte. Tout tourne sur ta machine, sans service en ligne.

Moteurs utilisés : [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (génération musicale), [Demucs](https://github.com/facebookresearch/demucs) (séparation voix / musique), [Seed-VC](https://github.com/Plachtaa/seed-vc) (conversion de voix chantée) et [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (synthèse vocale, licence MIT), [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) (MossFormer2, débruitage, Apache-2.0) et [VoiceFixer](https://github.com/haoheliu/voicefixer) (restauration de voix, code MIT, poids CC-BY 4.0) [Applio](https://github.com/IAHispano/Applio) (RVC : entraînement d'un modèle de ta voix, MIT), [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) (bruitages, licence Stability AI Community), [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (descriptions et traductions, Apache 2.0), [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) (modèles 3D, licence Tencent Hunyuan Community) et [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (texte → image, OpenRAIL++). Ils sont téléchargés par l'installateur, ils ne sont pas inclus dans ce dépôt.

Configuration testée : Windows 11, NVIDIA RTX 4070 (12 Go).

## Structure du dépôt

| Fichier | Rôle |
|---|---|
| `studio_voix.py` | point d'entrée de l'application (lancé par `lancer.bat`) |
| `studiovoix/` | le code de l'application, un module par rôle (voir ci-dessous) |
| `moteurs/` | scripts exécutés dans l'environnement d'un moteur (`chatterbox_tts.py`, `nettoyage_voix.py`, `rvc_voix.py`, `diffusion.py`) |
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
| `jeu.py` | bande-son de jeu : situations, génération en lot, jingles |
| `boucle.py` | boucles parfaites pour les musiques de fond |
| `galerie.py` | galerie des créations : écouter ou afficher (modèles 3D), recréer avec la même graine, refaire un passage, supprimer |
| `export.py` | export OGG/MP3 au volume harmonisé (LUFS), pack de bande-son avec `manifest.json` |
| `demucs.py` | séparation voix / instrumental |
| `seedvc.py` | conversion de voix chantée |
| `chatterbox.py` | synthèse vocale (lance `moteurs/chatterbox_tts.py`) |
| `nettoyage.py` | nettoyage de voix (lance `moteurs/nettoyage_voix.py`) |
| `rvc.py` | RVC : entraîner un modèle de ta voix et convertir avec (lance `moteurs/rvc_voix.py`) |
| `diffusion.py` | client du moteur de diffusion (Qwen3-VL, Stable Audio Open, SDXL, Hunyuan3D-2 ; lance `moteurs/diffusion.py`), jeton Hugging Face |
| `bruitages.py` | bruitages : description ou image → prompt anglais → variantes |
| `modele3d.py` | modèles 3D : image d'objet → forme puis texture (Hunyuan3D-2), GLB et OBJ |
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
- **RVC** (Applio), pour entraîner un modèle de ta voix (~2,8 Go de PyTorch et ~1,8 Go de modèles de base) ;
- le **moteur de diffusion** : Qwen3-VL (4 Go), Stable Diffusion XL (7 Go), Hunyuan3D-2 (forme 5 Go, texture 16 Go), Stable Audio Open (5 Go, **jeton Hugging Face requis**, voir ci-dessous) et ~3 Go de PyTorch ;
- l'environnement de l'application.

Compte 55 à 60 Go à télécharger, soit deux heures ou plus selon ta connexion, et au moins 80 Go libres. Si l'installation s'interrompt, relance INSTALLER.bat : les étapes terminées sont sautées.

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

1. Double-clique sur **lancer.bat**. Il ouvre, réduite, une fenêtre « ACE-Step - ne pas fermer » (le serveur de génération), puis l'application dans ton navigateur. La première génération attend que ce serveur ait fini de charger ses modèles.
2. Onglet **Bibliothèque de voix** : importe un fichier (wav, mp3 ou flac) ou enregistre-toi au micro, 10 à 25 s, dans une pièce calme, sans musique ni écho. Nomme la voix et clique sur « Vérifier et enregistrer ». Dans « Mes voix », tu peux écouter, renommer ou supprimer chaque voix.
   - **Micro médiocre ?** Avant d'enregistrer, ouvre « 🧽 Nettoyer la voix », choisis un niveau et clique sur « Nettoyer l'échantillon ». Écoute la version nettoyée, compare avec l'original, puis choisis dans « Version à enregistrer » celle que tu gardes. Pour une voix déjà enregistrée : « Reprendre cette voix pour la nettoyer », puis enregistre-la sous un nouveau nom.
3. Onglet **Créer une chanson** : choisis le mode, puis genre, style, instruments, ambiance et consignes dans les listes déroulantes (plusieurs choix possibles ; tu peux aussi taper ton propre terme, en anglais de préférence, puis Entrée). La **description envoyée à ACE-Step** s'affiche en dessous, en anglais, et tu peux la retoucher. Écris les paroles (avec `[Verse]`, `[Chorus]`…), puis « Créer la chanson ». Trois modes :
   - **Chanson avec ma voix** (par défaut) : la chanson est chantée avec ta voix ;
   - **Chanson avec la voix d'ACE-Step** : musique seule, la voix générée par ACE-Step est gardée telle quelle (plus rapide : ni séparation ni conversion, pas besoin de voix enregistrée) ;
   - **Instrumental** : musique sans voix, les paroles sont ignorées.
4. Onglet **Bande-son de jeu** : musiques sans voix pour un jeu. Donne un nom de projet, choisis l'époque (16-bit par défaut, 8-bit, orchestral…), l'univers et les situations (écran titre, menu, plateau, tension, combat, boss, boutique, et les jingles victoire, défaite, booster, carte rare). Tu peux taper ta propre situation en anglais (« fire faction theme, taiko drums »). Le tableau montre la description envoyée pour chaque piste. « Générer la bande-son » les crée une par une dans `data\jeux\<projet>\<situation>\<date>\` (`piste.wav`). Les jingles sont générés sur 10 s (le minimum d'ACE-Step) puis coupés proprement à leur durée. Les musiques de fond deviennent des **boucles parfaites** : le fichier `piste.wav` tourne en boucle sans coupure (`loop: true` dans Howler.js, `<audio loop>`…) ; l'aperçu « Jonction » te fait écouter le passage fin → début, et le message donne la qualité de la jonction.
   - **Une bande-son cohérente** (« 🎼 Cohérence ») : génère d'abord un thème principal (l'écran titre, par exemple), puis choisis-le comme *thème de référence*. « Même son » donne à toutes les pistes son timbre et son mixage ; « Variation du thème » réarrange sa mélodie selon chaque situation (version combat, version calme…), comme les leitmotivs des JRPG. Le curseur « Fidélité » règle la ressemblance. En variation, les musiques de fond prennent la durée du thème.
   - **Export pour le jeu** (« 📦 ») : toutes les pistes du projet (la plus récente de chaque situation) au même volume (−16 LUFS conseillé), en OGG et MP3, avec un `manifest.json`, dans `data\jeux\<projet>\export\` et en archive zip à télécharger.
   - Dans l'onglet « Créer une chanson », « 💾 Exporter la chanson en MP3 » fait de même pour une chanson (−14 LUFS par défaut).
5. Onglet **Synthèse vocale** : choisis une voix de ta bibliothèque, la langue, écris le texte et clique sur « Lire le texte avec cette voix ». Les textes longs (jusqu'à 5 000 caractères) sont découpés en phrases. Chaque lecture est rangée dans `data\tts\<date>\` (texte et `parole.wav`).
6. Onglet **Galerie** : toutes tes créations (chansons, pistes de jeu, lectures, bruitages, modèles 3D), la plus récente en premier, avec un filtre. Pour chacune : description, paroles, graine, écoute (et choix de la version s'il y en a deux), « 🔁 Recréer (même graine) » pour obtenir un résultat proche, « 📂 Ouvrir le dossier » et « 🗑️ Supprimer » (avec confirmation). Les créations faites avant cette version apparaissent aussi (écoute et suppression seulement).
   - **« ✏️ Refaire un passage »** : un refrain raté, une fin bizarre ? Indique le début et la fin en secondes : seul ce passage est réinventé, le reste est gardé, puis le reste du traitement est refait (ta voix, retrait d'instruments, boucle…). Tu peux changer la description ou les paroles du passage et choisir la force de la retouche (légère, équilibrée, complète). Le résultat est une nouvelle création ; l'originale est conservée.
7. Onglet **Entraîner ma voix (RVC)** : pour une ressemblance nettement meilleure, surtout au chant. Ajoute 10 à 30 minutes d'enregistrements de toi seul (plusieurs fichiers wav/mp3/flac, et/ou des voix de ta bibliothèque) ; le tableau vérifie la durée totale et signale les fichiers trop faibles ou saturés. Donne un nom, choisis la durée (300 époques conseillées, 1 à 2 heures sur une RTX 4070) et lance. Ensuite, dans « Créer une chanson » → Réglages voix → **Conversion de ta voix**, choisis « RVC — ton modèle ». Dans « Synthèse vocale », tu peux aussi faire passer la lecture dans ton modèle RVC. **Ferme la fenêtre ACE-Step pendant l'entraînement** : il a besoin de toute la mémoire graphique.
8. Onglet **Bruitages** : décris l'effet en français (ou choisis un exemple, ou importe une image de la scène), clique sur « Préparer le prompt » : Qwen3-VL le traduit en un prompt anglais précis, que tu peux retoucher. Règle la durée (1 à 30 s), le nombre de variantes (1 à 3), la graine, puis « Générer ». Écoute les variantes et exporte celle que tu gardes (OGG, MP3, WAV, au volume harmonisé). Tout est rangé dans `data\bruitages\<date>\` et visible dans la Galerie. **Ferme la fenêtre ACE-Step avant** : ces modèles ont besoin de la carte graphique.
9. Onglet **Modèles 3D** : importe une image de l'objet (PNG ou JPG : un seul objet, net, sur un fond simple ; une carte, un personnage, une arme…), donne un nom, choisis la qualité (« Normale » convient ; « Fine » demande plus de mémoire et de temps), garde « Peindre la texture » coché pour un modèle coloré, et clique sur « Créer le modèle 3D ». Hunyuan3D-2 retire le fond (l'image détourée s'affiche), sculpte la forme (quelques dizaines de secondes) puis peint la texture (plusieurs minutes). Le modèle s'affiche dans la visionneuse (glisser pour tourner, molette pour zoomer). Fichiers dans `data\3d\<date>\` : `modele.glb` (texturé, prêt pour un jeu web avec three.js ou Babylon.js, Blender, Unity, Godot), `forme.glb` (forme blanche) et, si tu coches OBJ, `modele.obj` + `material.mtl` + la texture PNG. La graine permet de retrouver le même modèle ; la Galerie les affiche aussi. **Ferme la fenêtre ACE-Step avant** : la texture a besoin de toute la carte graphique.
10. Quand tu as fini, ferme aussi la fenêtre ACE-Step pour libérer la carte graphique.

Chaque chanson est rangée dans `data\songs\<date>\` : version brute, voix convertie, instrumental, mix final et prompt.

## Comment ça marche

1. **ACE-Step 1.5** génère la chanson complète, avec une voix chantée générique.
2. **Demucs** sépare la voix de l'instrumental.
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
sons.combat.play();          // musique de fond en boucle
sons.victoire.play();        // jingle
```

Le volume est déjà harmonisé entre les pistes : règle seulement le volume général de la musique dans ton jeu.

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

- **Plusieurs versions et graine** : « Versions : 2 » génère deux versions d'un coup (plus long) pour garder la meilleure. Après chaque création, la **graine** s'affiche : remets-la dans « Graine » avec les mêmes réglages pour obtenir un résultat proche (0 = aléatoire). Tout est noté dans `creation.json`, dans le dossier de la création.
- **Style pas respecté ?** La description est transmise telle quelle à ACE-Step. Si le résultat reste trop « pop », décoche le **mode réflexion** : le générateur suit alors la description seule. Pour un style peu courant, répète les mots importants (« 8-bit chiptune, chiptune, retro video game music »).
- **Exclure un instrument** : ne l'écris pas dans la description, même précédé de « sans » ou « no » : le mot suffit à l'ajouter. Utilise plutôt **« Retirer de la musique »** (basse, batterie) : la chanson est séparée en pistes par Demucs et l'instrument est supprimé du mix, c'est garanti (environ 1 minute de plus). Les pistes séparées restent dans le dossier de la chanson (`demucs4\`).

- **Voix chantée de base** : choisis masculine ou féminine selon ta voix. Sinon, joue sur le décalage de hauteur (−12 / +12 demi-tons).
- **Étapes Seed-VC** : 40 par défaut. Monte à 50 pour plus de qualité, au prix du temps.
- **Synthèse vocale** : « Expressivité » à 0,5 = neutre ; pour un ton plus dramatique, monte vers 0,7 et baisse « Guidage / rythme » vers 0,3. Si ton échantillon n'est pas dans la langue du texte, mets « Guidage / rythme » à 0 pour ne pas garder l'accent. La graine (≠ 0) rend un résultat reproductible.

## Limites à connaître

- **Installation** : en cas d'erreur, l'installateur s'arrête avec un message en rouge. Relance-le après correction : les étapes réussies sont sautées.
- **Pilote NVIDIA** : ACE-Step utilise CUDA 12.8, qui demande un pilote récent (570.65 ou plus). L'installateur le vérifie.
- **Fidélité de la voix** : la conversion sans entraînement (Seed-VC) donne une ressemblance correcte mais pas parfaite, surtout sur les notes aiguës. Pour mieux faire, entraîne un modèle RVC sur 10 à 30 minutes de tes enregistrements (onglet « Entraîner ma voix »). La qualité de ton micro et de ta pièce compte encore plus que la durée.
- **Artefacts** : la séparation sur de la musique générée laisse parfois de légers résidus.
- **Mémoire graphique et synthèse vocale** : Chatterbox a besoin de 3 à 4 Go de mémoire graphique. Si la fenêtre ACE-Step est ouverte et a déjà chargé ses modèles, la carte peut manquer de mémoire : l'application te demande alors de fermer cette fenêtre, puis de relancer la lecture.
- **Nettoyage** : il ne fait pas de miracle sur une voix très saturée ou noyée dans la musique. Enregistre-toi au calme, à 15–30 cm du micro, c'est toujours le plus efficace.
- **Mémoire graphique et diffusion** : Hunyuan3D, SDXL et Stable Audio se chargent tour à tour et occupent jusqu'à 10 Go. Ferme la fenêtre ACE-Step avant de les utiliser ; l'application te le rappelle si la mémoire manque.
- **Modèles 3D** : Hunyuan3D-2 travaille à partir d'une seule image : le dos de l'objet est inventé, les parties fines (lames, cheveux, anses) peuvent être épaissies ou coupées, et la texture est parfois floue au dos. Une image bien cadrée sur fond uni change tout. Si la carte manque de mémoire pendant la texture, décoche « Peindre la texture » : la forme seule tient dans 6 Go. Les modèles sortent normalisés (environ 1 unité de côté) : remets-les à l'échelle dans ton moteur de jeu.
- **Filigrane** : chaque fichier produit par la synthèse vocale porte un filigrane inaudible ([Perth](https://github.com/resemble-ai/perth)) qui permet de reconnaître une voix de synthèse. Il est ajouté par Chatterbox lui-même.
- **Référence de voix pour la synthèse** : Chatterbox n'utilise que les 10 premières secondes de l'échantillon.
- **Consentement** : clone uniquement ta propre voix, ou celle de personnes d'accord.

## Tests (pour le développement)

Les tests n'ont pas besoin de carte graphique : ACE-Step est remplacé par un faux serveur HTTP, Demucs et Seed-VC par de faux scripts. Depuis le dossier de l'application :

```
uv run --python 3.12 --with-requirements requirements.txt --with pytest --with pyloudnorm pytest tests
```

## Licences

Ce dépôt ne contient que le code de l'application et de l'installation. Hunyuan3D-2 (Tencent) est sous licence Tencent Hunyuan Community : **elle exclut l'Union européenne, le Royaume-Uni et la Corée du Sud** (lieu d'utilisation) ; vérifie qu'elle s'applique à toi. Stable Audio Open et ses sorties : licence Stability AI Community (usage commercial permis sous 1 M$ de revenus annuels). Applio (RVC) est sous licence MIT ; ses conditions d'utilisation demandent de n'utiliser que des voix dont tu as le droit (la tienne, ou avec l'accord de la personne). Les poids de VoiceFixer sont sous licence CC-BY 4.0 (auteurs : Haohe Liu et al., « VoiceFixer: Toward General Speech Restoration with Neural Vocoder », 2021). Chaque moteur garde sa propre licence (voir leurs dépôts respectifs), à vérifier avant tout usage commercial des sons produits.
