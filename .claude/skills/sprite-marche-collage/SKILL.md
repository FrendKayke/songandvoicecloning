---
name: sprite-marche-collage
description: Crée un sprite de marche (overworld) style Pokémon Diamant / Perle / Platine d'un personnage réel ou décrit, par collage d'un vrai sprite du jeu recolorié et retouché pixel par pixel — sans IA générative. À utiliser dès que l'utilisateur veut un sprite de marche, un « overworld », une feuille de marche DS / Platine / 4 directions, ou « reproduire cette photo en sprite qui marche », même s'il ne dit pas « collage » ; aussi quand la génération (SDXL, klein, LoRA) donne des feuilles de marche ratées et qu'il faut un résultat propre et fidèle au style.
---

# Sprite de marche par collage

Les générateurs d'images ne savent pas tenir un sprite de 32×32 cohérent sur 16 images. Un vrai sprite du jeu, lui,
a déjà la silhouette, l'animation et l'ombrage justes : on le **recolore** et on **ajoute** quelques pixels (chignon,
bijou, ventre…) pour qu'il ressemble à la personne. Résultat dans le style exact du jeu, en quelques minutes.

Données : `D:\RPG Maker\Assets\sprites_dresseurs_dataset\datasets\6_marche_ds_bandes_dppt\images\` = 125 bandes
Platine `Pt_npc_<nom>.png` de 32×512 (16 images de 32×32 : haut, bas, gauche, droite × debout, pas, debout, autre
pas). Lancer les scripts avec le Python de l'application (`.venv/Scripts/python.exe`, numpy + Pillow).

## Démarche

1. **Lire la personne** (photo ou description) : silhouette (corpulence, robe longue / pantalon / jupe), coiffure et
   couleur des cheveux, couleurs principales des vêtements, un ou deux signes distinctifs lisibles à 32 px
   (chignon, lunettes, casquette, sac, ventre rond). Les traits du visage ne se voient pas à cette taille : inutile.
2. **Choisir la base** : le sprite dont la *forme* est la plus proche (la couleur ne compte pas, elle sera changée).
   ```
   python scripts/inspecter.py planche <dossier des bandes> --sortie <scratchpad>/planche.png [--noms mom,lady,woman1]
   ```
   Regarder la planche (Read de l'image) : de face, de dos, de profil. Privilégier la silhouette du bas (robe,
   pantalon) et la coiffure ; un chignon ou des lunettes s'ajoutent facilement, un manteau long ne s'enlève pas.
3. **Lire sa palette** :
   ```
   python scripts/inspecter.py palette <bande> --sortie <scratchpad>/palette/
   ```
   `palette.png` donne le numéro et le hex de chaque couleur, `images.png` montre le numéro dans chaque pixel des
   vues de face, de dos et de profil, avec les lignes repérées tous les 4 pixels. Noter :
   - à quelle ligne commence le corps (cou / col), les jambes, les pieds → `zones` ;
   - quelles couleurs servent à quoi **dans chaque zone** : une même couleur sert souvent à deux matières (les
     cheveux bleus de « mom » sont aussi ses bas) ; c'est pour ça que la correspondance se fait par zone ;
   - la couleur de l'iris (souvent celle des cheveux), entourée du blanc de l'œil.
4. **Écrire la recette** JSON (format dans `references/recette.md`, exemple complet dans
   `exemples/femme_robe_bleue.json`) : rampes de couleurs de la personne du plus clair au plus foncé (3 à 5 tons,
   garder les écarts de clarté de la base pour que l'ombrage reste lisible), correspondances par zone, iris, ajouts.
5. **Lancer** :
   ```
   python scripts/collage.py <recette.json>
   ```
   Sortie dans `data/pixel_art/<horodatage>_collage_<nom>/` (ou `sortie` de la recette) : bande Platine, feuille DS
   96×128 de l'application (+ ×2), un GIF par direction, `apercu.png` (photo + 4 directions debout et en marche).
6. **Regarder `apercu.png` et les GIF**, corriger la recette, relancer. Défauts habituels : cheveux trop clairs (la
   rampe doit être très foncée pour des cheveux bruns ou noirs : le contour noir les avale sinon), ajout qui flotte
   ou fait un trou (contour `0` sur le pourtour libre, reflet `4`/`5` en haut à gauche), pixel de couleur oublié
   (une couleur de la base sans règle garde sa couleur d'origine : chercher les restes bleus / rouges), iris non
   remplacé de profil.
7. Montrer l'aperçu à l'utilisateur (SendUserFile) et dire honnêtement les limites : la silhouette reste celle de
   la base (volume des cheveux, longueur d'une robe, ventre seulement esquissé).

## Ajouts dessinés

Un ajout est un petit motif de caractères : `.` = rien, `0` = contour noir, tout autre caractère = une couleur de la
`legende` (`"rampe:i"`, `"#rrggbb"` ou `"transparent"`). Position : `x` fixe (avec `miroir_droite` pour reprendre
le motif retourné sur la vue de droite) ou `"arriere"` = collé à l'arrière de la tête des profils (le motif est
dessiné pour la vue de gauche, retourné automatiquement à droite). Penser à chaque direction concernée : un chignon
bas se voit de dos et de profil, pas de face ; un pendentif se voit de face seulement.

Faire petit : à 32 px, un pendentif est **un** pixel, un chignon 4×4 à 6×6 avec son contour.

## Importer dans l'application

La bande se dépose telle quelle dans l'onglet « 🧪 Entraîner un modèle de sprites » (style DS, bandes Platine
acceptées) ou dans la conversion de feuilles de marche ; `feuille_ds_96x128.png` est la grille DS de l'application
(3 colonnes debout / pas / autre pas, lignes haut / bas / gauche / droite).
