"""Collage d'un sprite de face (combat, 80×80 Diamant / Perle / Platine) d'après une recette JSON.

  python collage.py recette.json

Voir ../references/recette.md. Sorties (dossier `sortie`, par défaut data/pixel_art/<horodatage>_face_<nom>/) :
  face.png (taille de la base), face_x2.png, apercu.png (photo si donnée, base, résultat ×4), recette.json
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DEPOT = Path(__file__).resolve().parents[4]


def hexa(c):
    return "#%02x%02x%02x" % tuple(int(v) for v in c[:3])


def rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


class Recette:
    def __init__(self, r):
        self.r = r
        self.rampes = {k: [rgb(h) for h in v] for k, v in r.get("rampes", {}).items()}
        self.zones = sorted(r.get("zones", {}).items(), key=lambda z: z[1])  # {"tete": 0, "haut": 30, …}
        self.rects = r.get("rectangles", [])  # [{"nom", "x": [x0, x1], "y": [y0, y1]}] bornes incluses

    def cible(self, valeur):
        if valeur in (None, "garder"):
            return None, None
        if valeur == "transparent":
            return None, 0
        if valeur.startswith("#"):
            return rgb(valeur), 255
        nom, i = valeur.split(":")
        return self.rampes[nom][int(i)], 255

    def zone(self, x, y):
        """Zones du pixel, de la plus précise à la plus générale : rectangles (le dernier déclaré d'abord), zone de
        lignes, « * »."""
        noms = [z["nom"] for z in reversed(self.rects)
                if z["x"][0] <= x <= z["x"][1] and z["y"][0] <= y <= z["y"][1]]
        ligne = None
        for z, debut in self.zones:
            if y >= debut:
                ligne = z
        return noms + ([ligne] if ligne else []) + ["*"]

    def regle(self, zones, h):
        couleurs = self.r.get("couleurs", {})
        for cle in zones:
            if h in couleurs.get(cle, {}):
                return couleurs[cle][h]
        return None


def appliquer(rec, a):
    o = a.copy()
    h, w = a.shape[:2]
    for y in range(h):
        for x in range(w):
            if a[y, x, 3] == 0:
                continue
            c, al = rec.cible(rec.regle(rec.zone(x, y), hexa(a[y, x])))
            if c is not None:
                o[y, x, :3] = c
            if al is not None:
                o[y, x, 3] = al
    for ajout in rec.r.get("ajouts", []):
        legende = ajout.get("legende", {})
        for dy, ligne in enumerate(ajout["motif"]):
            for dx, ch in enumerate(ligne):
                if ch in ". ":
                    continue
                yy, xx = int(ajout["y"]) + dy, int(ajout["x"]) + dx
                if not (0 <= yy < h and 0 <= xx < w):
                    continue
                valeur = legende.get(ch, "#000000" if ch == "0" else None)
                if valeur is None:
                    sys.exit(f"Ajout : caractère « {ch} » sans couleur dans la légende.")
                c, al = rec.cible(valeur)
                if c is not None:
                    o[yy, xx, :3] = c
                o[yy, xx, 3] = 255 if al is None else al
    if rec.r.get("contour_auto"):
        o = contour(o)
    return o


def contour(o):
    """Pixel opaque non noir au bord de la silhouette (voisin transparent en 4-connexité) → noir : referme le contour
    après un effacement (micro retiré, cheveux raccourcis)."""
    op = o[..., 3] > 0
    bord = np.zeros_like(op)
    p = np.pad(op, 1)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        bord |= ~p[dy:dy + op.shape[0], dx:dx + op.shape[1]]
    m = op & bord & (o[..., :3].sum(-1) > 0)
    o = o.copy()
    o[m, :3] = 0
    return o


def main():
    chemin = sys.argv[1]
    with open(chemin, encoding="utf-8") as f:
        rec = Recette(json.load(f))
    base = Image.open(rec.r["base"]).convert("RGBA")
    a = np.array(base)
    a[a[..., 3] < 128] = 0
    a[..., 3] = np.where(a[..., 3] > 0, 255, 0)
    res = Image.fromarray(appliquer(rec, a), "RGBA")
    dossier = Path(rec.r.get("sortie") or DEPOT / "data" / "pixel_art" /
                   f"{time.strftime('%Y%m%d_%H%M%S')}_face_{rec.r.get('nom', 'personnage')}")
    dossier.mkdir(parents=True, exist_ok=True)
    res.save(dossier / "face.png")
    res.resize((res.width * 2, res.height * 2), Image.Resampling.NEAREST).save(dossier / "face_x2.png")
    k = 4
    W, H = base.width * k, base.height * k
    photo = rec.r.get("photo")
    im = None
    if photo and os.path.exists(photo):
        im = ImageOps.contain(ImageOps.exif_transpose(Image.open(photo)).convert("RGBA"), (W, H))
    x = 0
    pl = Image.new("RGBA", ((W + 20) * (3 if im else 2), H), (255, 255, 255, 255))
    for vue in ([im] if im else []) + [base, res]:
        if vue is not base and vue is not res:
            pl.alpha_composite(vue, (x, 0))
        else:
            pl.alpha_composite(vue.resize((W, H), Image.Resampling.NEAREST), (x, 0))
        x += W + 20
    pl.save(dossier / "apercu.png")
    shutil.copy(chemin, dossier / "recette.json")
    print(dossier)


if __name__ == "__main__":
    main()
