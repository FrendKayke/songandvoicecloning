# Format de la recette (collage.py)

```json
{
 "nom": "court_sans_espace",            // nom du dossier de sortie
 "base": "chemin/Pt_npc_xxx.png",       // bande Platine 32×512
 "photo": "chemin/photo.jpg",           // facultatif : affichée à gauche dans apercu.png
 "sortie": "dossier",                    // facultatif (défaut data/pixel_art/<horodatage>_collage_<nom>/)
 "zones": {"corps": 21, "jambes": 26, "pieds": 28},   // première ligne (y, 0-31) de chaque zone ; avant = "tete"
 "rampes": {"cheveux": ["#…", …], "robe": [...]},       // couleurs nommées, référencées "robe:1"
 "iris": {"base": "#hex", "blanc": ["#hex", …], "y_min": 16, "cible": "divers:0"},
 "couleurs": {                           // couleur de la base → cible, par zone
   "tete": {"#292931": "cheveux:0", …},
   "corps": {…}, "jambes": {…}, "pieds": {…},
   "*": {…}                              // règles valables dans toutes les zones (la zone l'emporte)
 },
 "ajouts": [ {"sens": ["haut"], "y": 16, "x": 13, "motif": ["…"], "legende": {"2": "cheveux:1"}} ],
 "ventre": {"lignes": [24, 25], "couleur": "robe:1"}    // facultatif : un pixel de plus vers l'avant de profil
}
```

- Cible : `"rampe:i"`, `"#rrggbb"`, `"transparent"` (efface le pixel) ou `"garder"`. Une couleur sans règle garde sa
  couleur d'origine (le contour noir n'a jamais besoin de règle).
- Les clés `_note` sont ignorées : s'en servir pour décrire la personne et la palette de la base.
- Iris : un pixel de la couleur `base`, dans la tête, à partir de `y_min`, voisin à gauche ou à droite d'une couleur
  `blanc`, prend la couleur `cible` (sauf vue de dos).
- Ajout : `x` entier (+ `"miroir_droite": true` pour qu'une vue de droite reçoive le motif retourné et symétrique)
  ou `"arriere"` avec `lignes_tete` [y0, y1[ (lignes où mesurer la tête) et `dx` (décalage, négatif = chevauche).
  Les motifs sont posés dans l'ordre de la liste, après la recoloration.
