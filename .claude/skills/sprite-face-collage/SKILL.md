---
name: sprite-face-collage
description: Crée un sprite de face (sprite de combat de dresseur, 80×80, style Pokémon Diamant / Perle / Platine) d'une personne réelle ou décrite, par collage d'un vrai sprite du jeu recolorié, retouché et complété pixel par pixel — sans IA générative. À utiliser dès que l'utilisateur veut un sprite de face, un sprite de combat, un « trainer sprite », « me / la mettre en dresseur Pokémon », « reproduire cette photo en sprite », même s'il ne dit pas « collage » ; aussi quand le kit de dresseur ou la génération (SDXL, klein, LoRA) donne des yeux ou des mains ratés et qu'il faut un résultat propre, dans le style exact. Pour les sprites de marche (overworld), voir le skill sprite-marche-collage.
---

# Sprite de face par collage

Un vrai sprite de combat du jeu a déjà la pose, l'anatomie, les yeux et l'ombrage justes ; les générateurs ratent
justement ces détails à 80 px. On choisit le sprite dont la **forme** ressemble à la personne, on **recolore** chaque
matière, on **efface** ce qui ne va pas (micro, objet tenu) et on **dessine** quelques pixels (bijou, manche, robe plus
longue). Quelques allers-retours suffisent.

Données : `D:\RPG Maker\Assets\sprites_dresseurs_dataset\datasets\` — `3_face_ds_diamant_perle\images\` (273 sprites
officiels `gen4__…Spr_DP_*` + fanmade, 80×80, ≤ 16 couleurs, alpha net : **à préférer**, même style que les bandes de
marche Platine), `2_face_ds_noir_blanc` (Noir et Blanc / Noir 2 Blanc 2, 80×80 : 207 officiels `gen5__…Spr_BW_*` /
`Spr_B2W2_*` + 872 fanmade ; plus de poses calmes, mains devant soi), `1_face_gba_rfvf` (GBA 64×64). Lancer les scripts avec
le Python de l'application (`.venv/Scripts/python.exe`, numpy + Pillow).

## Démarche

1. **Lire la personne** (photo ou description) : genre et âge apparent, coupe et couleur des cheveux, vêtement
   principal (robe longue, pantalon, veste…), couleurs, 1 à 3 détails lisibles à 80 px (bijou doré, lunettes, sac,
   chapeau). Les traits du visage ne se reproduisent pas : seule l'expression (sourire, bouche fermée) compte.
2. **Choisir la base** : regarder une planche de candidats et prendre la silhouette la plus proche ; la couleur ne
   compte pas. Effacer un objet tenu ou allonger une jupe est facile ; enlever un grand chapeau ou des cheveux très
   longs ne l'est pas (il faudrait redessiner le crâne).
   ```
   python scripts/inspecter.py planche <dossier> --filtre Spr_DP_ --noms Lady,Reporter,Beauty --sortie <scratchpad>/planche.png
   ```
   Sans `--noms`, toute la liste (273 sprites, 35 lignes) : la lire par morceaux ou filtrer par classe ; `--noms` compare la classe sans le jeu
   (`Lady` prend `DP_Lady`, `BW_Lady` et `B2W2_Lady`), `--filtre gen5__` garde les officiels de Noir et Blanc. Classes (`Lady`,
   `Lass`, `Beauty`, `Ace_Trainer`, `Hiker`, `Gentleman`, `School_Kid`, `Maid`, `Nurse`, `Socialite`…).
3. **Lire sa palette** :
   ```
   python scripts/inspecter.py palette <sprite.png> --sortie <scratchpad>/palette/
   ```
   `zones.png` montre où sert chaque couleur (en rouge) : le plus utile pour voir qu'une couleur sert à deux matières
   (cheveux et chaussures, micro et boucle d'oreille). `grille.png` donne le numéro de chaque pixel avec des règles
   tous les 5 px. Pour un relevé exact (rectangles, ajouts), imprimer le sprite en texte (un caractère hexadécimal par
   couleur, `.` = transparent) avec quelques lignes de numpy : plus sûr que de lire les chiffres sur l'image.
4. **Écrire la recette** JSON (format : `references/recette.md` ; exemple complet : `exemples/femme_robe_bleue.json`,
   reporter → femme enceinte en robe bleu pâle ;
   `exemples/femme_robe_bleue_noir_blanc.json` : la même d'après la Maid de Noir 2 / Blanc 2, coiffe changée en chignon,
   Poké Ball tenue changée en mains jointes) : zones par lignes (tête, haut, bas, jambes, pieds), rectangles pour
   les cas particuliers (bouche, boucle d'oreille, objet à effacer, manches), rampes de couleurs (du plus clair au
   plus foncé, écarts de clarté proches de ceux de la base pour garder l'ombrage), correspondances, ajouts.
5. **Lancer** :
   ```
   python scripts/collage.py <recette.json>
   ```
   Sortie `data/pixel_art/<horodatage>_face_<nom>/` (ou `sortie`) : `face.png` (taille de la base), `face_x2.png`,
   `apercu.png` (photo, base, résultat ×4). Pour juger les détails, agrandir le résultat ×8–9 sur fond blanc et le
   regarder (Read de l'image).
6. **Corriger et relancer** jusqu'à ce que ce soit propre. Défauts habituels :
   - pixel resté à la couleur de la base (aucune règle ne le couvre, ou couleur partagée) → relever ses coordonnées
     et sa couleur avec numpy ;
   - trou ou bord ouvert après un effacement → `"contour_auto": true` referme la silhouette en noir ;
   - objet tenu effacé → recolorer en peau les pixels de la main qui le tenaient (rectangle « poing ») ;
   - bouche ouverte de la base (personnage qui parle) → rectangle « bouche » : rouge vif → peau, rouge foncé → lèvres ;
   - ajout posé sur le bord de la silhouette : `contour_auto` le repeint en noir → le décaler d'un pixel vers l'intérieur ;
   - motif généré par script : un index négatif en Python pose le pixel à l'autre bout de la ligne.
7. Envoyer l'aperçu à l'utilisateur (SendUserFile) et dire honnêtement les limites : pose et coiffure de la base,
   ventre ou silhouette particulière seulement suggérés.

## Ajouts dessinés

Motif de caractères posé à (`x`, `y`) = coin haut gauche : `.` ou espace = rien, `0` = contour noir, autre caractère =
couleur de la `legende` (`"rampe:i"`, `"#rrggbb"`, ou `"transparent"` pour gommer). Un motif régulier (bas de robe en
trapèze, ceinture) se génère avec quelques lignes de Python qui écrivent les chaînes dans la recette. Tailles
réalistes à 80 px : pendentif = 1 pixel et chaîne de 3, manche = 4×5 pixels recoloriés, robe allongée = 8 lignes.

## Utiliser dans l'application

`face.png` (80×80, fond transparent) est au format du style DS de l'application : il se dépose dans l'onglet
« 👁️ Refaire les yeux », sert de sprite de face d'un kit, ou d'exemple pour « 🧪 Entraîner un modèle de sprites »
(type « Sprites de face »).
