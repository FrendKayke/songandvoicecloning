# Format de la recette (collage.py, sprites de face)

```json
{
 "nom": "court_sans_espace",
 "base": "chemin/sprite.png",
 "photo": "chemin/photo.jpg",
 "sortie": "dossier",
 "zones": {"tete": 0, "haut": 30, "jupe": 46, "jambes": 63, "pieds": 74},
 "rectangles": [ {"nom": "boucle", "x": [35, 38], "y": [24, 27]} ],
 "contour_auto": true,
 "rampes": {"robe": ["#deeef8", "#b8d0e4", "#84a0c0", "#5f7a9c"]},
 "couleurs": {
   "*": {"#ffd5bd": "peau:0"},
   "tete": {"#394a4a": "cheveux:0"},
   "boucle": {"#ff6239": "or:0"}
 },
 "ajouts": [ {"x": 39, "y": 34, "motif": ["c..", ".c.", "..o"], "legende": {"c": "or:1", "o": "or:0"}} ]
}
```

- `base` : sprite du jeu (80×80 pour DP). `photo` (facultatif) : affichée dans `apercu.png`. `sortie` (facultatif) :
  dossier de sortie, sinon `data/pixel_art/<horodatage>_face_<nom>/`.
- `zones` : première ligne y de chaque zone. `rectangles` : bornes incluses ; plusieurs rectangles peuvent porter le
  même nom.
- Ordre de recherche d'une règle pour un pixel : rectangles qui le contiennent (le dernier déclaré d'abord), puis sa
  zone de lignes, puis `"*"`. Un rectangle ne précise donc que ce qui change ; le reste vient de la zone.
- Cible : `"rampe:i"`, `"#rrggbb"`, `"transparent"` (efface) ou `"garder"`. Couleur sans règle = inchangée (le contour
  noir n'en a jamais besoin).
- Les clés `_note` sont ignorées : y décrire la personne et la palette de la base (numéro, hex, matière).
- Les ajouts sont posés dans l'ordre, après la recoloration ; `contour_auto` (bord de silhouette non noir → noir)
  passe en dernier.
- Pixels semi-transparents de la base : seuil à 128, puis alpha 0 ou 255.
