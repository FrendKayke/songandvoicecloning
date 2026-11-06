"""Collage d'un sprite de marche (bande Diamant / Perle / Platine 32×512) d'après une recette JSON.

  python collage.py recette.json

La recette (voir ../references/recette.md) donne : la bande de base, les lignes qui séparent les zones (tête, corps,
jambes, pieds), les rampes de couleurs de la personne, la correspondance « couleur de la base → couleur cible » par
zone, l'iris des yeux, des ajouts dessinés (chignon, pendentif, lunettes…) et le ventre rond éventuel.

Sorties (dossier `sortie`, par défaut data/pixel_art/<horodatage>_collage_<nom>/ du dépôt Studio Voix) :
  bande_platine_32x512.png   même format que la base (importable comme bande Platine)
  feuille_ds_96x128.png      grille DS de l'application (3 colonnes : debout, pas, autre pas ; lignes haut, bas,
                             gauche, droite) + feuille_ds_x2.png
  marche_<sens>.gif          une animation par direction (agrandie ×4)
  apercu.png                 photo (si donnée) + les 4 directions debout et en marche, agrandies ×7
  recette.json               copie de la recette
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SENS = ["haut", "bas", "gauche", "droite"]
DEPOT = Path(__file__).resolve().parents[4]  # .claude/skills/<skill>/scripts/collage.py → racine du dépôt


def hexa(c):
    return "#%02x%02x%02x" % tuple(int(v) for v in c[:3])


def rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


class Recette:
    def __init__(self, r):
        self.r = r
        self.rampes = {k: [rgb(h) for h in v] for k, v in r.get("rampes", {}).items()}
        z = r.get("zones", {})
        self.limites = [("corps", z.get("corps", 21)), ("jambes", z.get("jambes", 26)), ("pieds", z.get("pieds", 28))]

    def cible(self, valeur):
        """« rampe:i », « #rrggbb », « garder » ou « transparent » → (rgb ou None, alpha ou None)."""
        if valeur in (None, "garder"):
            return None, None
        if valeur == "transparent":
            return None, 0
        if valeur.startswith("#"):
            return rgb(valeur), 255
        nom, i = valeur.split(":")
        return self.rampes[nom][int(i)], 255

    def zone(self, y):
        nom = "tete"
        for z, limite in self.limites:
            if y >= limite:
                nom = z
        return nom

    def regle(self, zone, h):
        couleurs = self.r.get("couleurs", {})
        for cle in (zone, "*"):
            if h in couleurs.get(cle, {}):
                return couleurs[cle][h]
        return None


def recolorer(rec, F, O, sens):
    iris = rec.r.get("iris")
    for y in range(32):
        zone = rec.zone(y)
        for x in range(32):
            if F[y, x, 3] == 0:
                continue
            h = hexa(F[y, x])
            if iris and sens != "haut" and h == iris["base"].lower() and zone == "tete" and y >= iris.get("y_min", 0):
                voisins = {hexa(F[y, xx]) for xx in (x - 1, x + 1) if 0 <= xx < 32 and F[y, xx, 3]}
                if voisins & {b.lower() for b in iris["blanc"]}:
                    O[y, x, :3] = rec.cible(iris["cible"])[0]
                    continue
            c, a = rec.cible(rec.regle(zone, h))
            if c is not None:
                O[y, x, :3] = c
            if a is not None:
                O[y, x, 3] = a


def ajouter(rec, F, O, sens, ajout):
    if sens not in ajout["sens"]:
        return
    motif = [list(l) for l in ajout["motif"]]
    legende = ajout.get("legende", {})
    y0 = int(ajout["y"])
    x = ajout["x"]
    if x == "arriere":  # arrière de la tête d'un profil : côté droit si le personnage regarde à gauche
        lignes = ajout.get("lignes_tete", [y0 - 2, y0 + 3])
        cols = np.where(F[lignes[0]:lignes[1], :, 3].any(axis=0))[0]
        dx = int(ajout.get("dx", 0))
        if sens == "gauche":
            x0 = cols.max() + dx
        else:
            motif = [l[::-1] for l in motif]
            x0 = cols.min() - (len(motif[0]) - 1) - dx
    else:
        x0 = int(x)
        if sens == "droite" and ajout.get("miroir_droite"):
            motif = [l[::-1] for l in motif]
            x0 = 31 - x0 - (len(motif[0]) - 1)
    for dy, ligne in enumerate(motif):
        for dx_, ch in enumerate(ligne):
            if ch in (".", " "):
                continue
            yy, xx = y0 + dy, x0 + dx_
            if not (0 <= yy < 32 and 0 <= xx < 32):
                continue
            valeur = legende.get(ch, "#000000" if ch == "0" else None)
            if valeur is None:
                sys.exit(f"Ajout : caractère « {ch} » sans couleur dans la légende.")
            c, a = rec.cible(valeur)
            if c is not None:
                O[yy, xx, :3] = c
            O[yy, xx, 3] = 255 if a is None else a


def ventre(rec, O, sens, v):
    if sens not in ("gauche", "droite"):
        return
    couleur = rec.cible(v["couleur"])[0]
    for y in v["lignes"]:
        xs = np.where(O[y, :, 3] > 0)[0]
        if not len(xs):
            continue
        x = xs.min() if sens == "gauche" else xs.max()
        n = x - 1 if sens == "gauche" else x + 1
        if 0 <= n < 32:
            O[y, n] = (0, 0, 0, 255)
            O[y, x, :3] = couleur


def feuille_ds(bande):
    """Bande 32×512 → grille DS de l'application 96×128 (colonnes debout, pas, autre pas)."""
    a = np.array(bande)
    f = np.zeros((128, 96, 4), np.uint8)
    for r in range(4):
        for c, i in enumerate((0, 1, 3)):
            f[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = a[(r * 4 + i) * 32:(r * 4 + i + 1) * 32]
    return Image.fromarray(f, "RGBA")


def exporter(bande, rec, dossier, source_recette):
    dossier.mkdir(parents=True, exist_ok=True)
    bande.save(dossier / "bande_platine_32x512.png")
    ds = feuille_ds(bande)
    ds.save(dossier / "feuille_ds_96x128.png")
    ds.resize((192, 256), Image.Resampling.NEAREST).save(dossier / "feuille_ds_x2.png")
    k = 4
    for d, nom in enumerate(SENS):
        images = []
        for i in range(4):
            fond = Image.new("RGBA", (32 * k, 32 * k), (255, 255, 255, 255))
            fond.alpha_composite(bande.crop((0, (d * 4 + i) * 32, 32, (d * 4 + i + 1) * 32))
                                 .resize((32 * k, 32 * k), Image.Resampling.NEAREST))
            images.append(fond.convert("P", palette=Image.ADAPTIVE))
        images[0].save(dossier / f"marche_{nom}.gif", save_all=True, append_images=images[1:], duration=160, loop=0)
    k = 7
    photo = rec.r.get("photo")
    gauche = 0
    hauteur = 2 * 32 * k
    if photo and os.path.exists(photo):
        im = ImageOps.contain(ImageOps.exif_transpose(Image.open(photo)).convert("RGBA"), (300, hauteur))
        gauche = im.width + 20
    pl = Image.new("RGBA", (gauche + 4 * 32 * k, hauteur), (255, 255, 255, 255))
    if gauche:
        pl.alpha_composite(im, (0, 0))
    for j, i in enumerate((4, 0, 8, 12, 5, 1, 9, 13)):
        pl.alpha_composite(bande.crop((0, i * 32, 32, i * 32 + 32)).resize((32 * k, 32 * k), Image.Resampling.NEAREST),
                           (gauche + (j % 4) * 32 * k, (j // 4) * 32 * k))
    pl.save(dossier / "apercu.png")
    shutil.copy(source_recette, dossier / "recette.json")


def main():
    chemin = sys.argv[1]
    with open(chemin, encoding="utf-8") as f:
        rec = Recette(json.load(f))
    base = Image.open(rec.r["base"]).convert("RGBA")
    if base.size != (32, 512):
        sys.exit(f"Base {base.size[0]}×{base.size[1]} : une bande Platine 32×512 est attendue.")
    a = np.array(base)
    out = a.copy()
    for i in range(16):
        sens = SENS[i // 4]
        F, O = a[i * 32:(i + 1) * 32], out[i * 32:(i + 1) * 32]
        recolorer(rec, F, O, sens)
        for ajout in rec.r.get("ajouts", []):
            ajouter(rec, F, O, sens, ajout)
        if rec.r.get("ventre"):
            ventre(rec, O, sens, rec.r["ventre"])
    dossier = Path(rec.r.get("sortie") or DEPOT / "data" / "pixel_art" /
                   f"{time.strftime('%Y%m%d_%H%M%S')}_collage_{rec.r.get('nom', 'personnage')}")
    exporter(Image.fromarray(out, "RGBA"), rec, dossier, chemin)
    print(dossier)


if __name__ == "__main__":
    main()
