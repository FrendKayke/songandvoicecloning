"""Inspecter des sprites de face (combat, 80×80 de Diamant / Perle / Platine, ou toute taille) pour choisir une base de
collage et lire sa palette.

  planche : python inspecter.py planche <dossier_ou_fichiers…> --sortie planche.png [--noms Lady,Beauty] [--filtre DP]
            → les sprites côte à côte (×3) avec leur nom
  palette : python inspecter.py palette <sprite.png> --sortie dossier/
            → palette.png (numéro, nombre de pixels, hex), grille.png (sprite ×10, numéro de couleur dans chaque pixel,
              règles tous les 5 pixels), palette.json (numéro → hex), zones.png (chaque couleur seule, pour voir où
              elle sert)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw


def court(chemin):
    nom = os.path.splitext(os.path.basename(chemin))[0]
    return re.sub(r"^.*?Spr_(DP_|Pt_|HGSS_)?", "", nom)


def planche(chemins, sortie, k=3):
    if not chemins:
        sys.exit("Aucun PNG.")
    tailles = [Image.open(c).size for c in chemins]
    cw = max(t[0] for t in tailles) * k + 8
    ch = max(t[1] for t in tailles) * k + 16
    par_ligne = 8
    lignes = -(-len(chemins) // par_ligne)
    o = Image.new("RGBA", (par_ligne * cw, lignes * ch), (205, 210, 205, 255))
    d = ImageDraw.Draw(o)
    for n, c in enumerate(chemins):
        im = Image.open(c).convert("RGBA")
        x, y = (n % par_ligne) * cw, (n // par_ligne) * ch
        o.alpha_composite(im.resize((im.width * k, im.height * k), Image.Resampling.NEAREST), (x + 4, y + 14))
        d.text((x + 3, y + 2), court(c)[:30], fill=(0, 0, 0, 255))
    o.save(sortie)
    print(sortie)


def palette(chemin, dossier, k=10):
    os.makedirs(dossier, exist_ok=True)
    a = np.array(Image.open(chemin).convert("RGBA"))
    h, w = a.shape[:2]
    cols, nb = np.unique(a[a[..., 3] > 0][:, :3], axis=0, return_counts=True)
    hexa = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in cols]
    idx = {hx: i for i, hx in enumerate(hexa)}
    sw = Image.new("RGB", (len(cols) * 76, 92), (255, 255, 255))
    d = ImageDraw.Draw(sw)
    for i, (c, n) in enumerate(zip(cols, nb)):
        d.rectangle((i * 76, 0, i * 76 + 66, 50), fill=tuple(int(v) for v in c))
        d.text((i * 76, 55), f"{i} : {n} px", fill=(0, 0, 0))
        d.text((i * 76, 72), hexa[i], fill=(0, 0, 0))
    sw.save(os.path.join(dossier, "palette.png"))
    g = Image.new("RGB", (w * k + 20, h * k + 20), (230, 230, 230))
    d = ImageDraw.Draw(g)
    for y in range(h):
        for x in range(w):
            if a[y, x, 3]:
                c = tuple(int(v) for v in a[y, x, :3])
                d.rectangle((20 + x * k, 20 + y * k, 20 + x * k + k - 1, 20 + y * k + k - 1), fill=c)
                d.text((20 + x * k + 1, 20 + y * k), str(idx["#%02x%02x%02x" % c]),
                       fill=(255, 255, 255) if sum(c) < 300 else (0, 0, 0))
    for v in range(0, max(w, h), 5):
        if v < h:
            d.text((1, 20 + v * k), str(v), fill=(150, 0, 0))
            d.line((20, 20 + v * k, 20 + w * k, 20 + v * k), fill=(255, 120, 120))
        if v < w:
            d.text((20 + v * k, 4), str(v), fill=(150, 0, 0))
            d.line((20 + v * k, 20, 20 + v * k, 20 + h * k), fill=(255, 120, 120))
    g.save(os.path.join(dossier, "grille.png"))
    kz = 3
    z = Image.new("RGB", (len(cols) * (w * kz + 6), h * kz + 16), (255, 255, 255))
    d = ImageDraw.Draw(z)
    for i, c in enumerate(cols):
        m = (a[..., 3] > 0) & (a[..., :3] == c).all(-1)
        im = np.full((h, w, 3), 225, np.uint8)
        im[a[..., 3] > 0] = 180
        im[m] = (220, 0, 0)
        z.paste(Image.fromarray(im).resize((w * kz, h * kz), Image.Resampling.NEAREST), (i * (w * kz + 6), 16))
        d.text((i * (w * kz + 6) + 2, 2), f"{i} {hexa[i]}", fill=(0, 0, 0))
    z.save(os.path.join(dossier, "zones.png"))
    with open(os.path.join(dossier, "palette.json"), "w", encoding="utf-8") as f:
        json.dump({str(i): hx for i, hx in enumerate(hexa)}, f, indent=1)
    for n in ("palette.png", "grille.png", "zones.png"):
        print(os.path.join(dossier, n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["planche", "palette"])
    p.add_argument("chemins", nargs="+")
    p.add_argument("--sortie", required=True)
    p.add_argument("--noms", help="planche : noms courts à garder (Lady, Aroma_Lady…), séparés par des virgules")
    p.add_argument("--filtre", help="planche : texte que le nom de fichier doit contenir (ex. Spr_DP_)")
    a = p.parse_args()
    if a.action == "planche":
        fichiers = []
        for c in a.chemins:
            fichiers += sorted(glob.glob(os.path.join(c, "*.png"))) if os.path.isdir(c) else [c]
        if a.filtre:
            fichiers = [f for f in fichiers if a.filtre in os.path.basename(f)]
        if a.noms:
            garder = {n.strip().lower() for n in a.noms.split(",")}
            fichiers = [f for f in fichiers if court(f).lower() in garder]
        planche(fichiers, a.sortie)
    else:
        palette(a.chemins[0], a.sortie)


if __name__ == "__main__":
    main()
