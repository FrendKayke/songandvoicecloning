"""Inspecter des sprites de marche (bandes Diamant / Perle / Platine 32×512 = 16 images de 32×32 : haut, bas, gauche,
droite × debout, pas, debout, autre pas) pour choisir une base de collage et lire sa palette.

  planche  : python inspecter.py planche <dossier_ou_fichiers…> --sortie planche.png [--noms lady,mom,…]
             → une ligne par sprite : de face, de dos, de profil (choix de la base)
  palette  : python inspecter.py palette <bande.png> --sortie dossier/
             → palette.png (numéro, nombre de pixels, couleur hex), images.png (images 4, 0, 8, 12 et 5, 1, 9, 13
               agrandies avec le numéro de couleur dans chaque pixel et les lignes repérées tous les 4 pixels),
               palette.json (numéro → hex)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

SENS = ["haut", "bas", "gauche", "droite"]


def image(bande, i):
    return bande.crop((0, i * 32, 32, i * 32 + 32))


def planche(chemins, sortie, k=4):
    chemins = [c for c in chemins if c.lower().endswith(".png")]
    if not chemins:
        sys.exit("Aucun PNG.")
    par_ligne = 6
    lw, lh = 3 * 32 * k + 16, 32 * k + 22
    lignes = -(-len(chemins) // par_ligne)
    o = Image.new("RGBA", (par_ligne * lw, lignes * lh), (205, 210, 205, 255))
    d = ImageDraw.Draw(o)
    for n, c in enumerate(chemins):
        b = Image.open(c).convert("RGBA")
        if b.size != (32, 512):
            continue
        x, y = (n % par_ligne) * lw, (n // par_ligne) * lh
        for j, i in enumerate((4, 0, 8)):
            o.alpha_composite(image(b, i).resize((32 * k, 32 * k), Image.Resampling.NEAREST), (x + j * 32 * k, y + 18))
        nom = os.path.splitext(os.path.basename(c))[0].replace("Pt_npc_", "")
        d.text((x + 4, y + 3), nom, fill=(0, 0, 0, 255))
    o.save(sortie)
    print(sortie)


def palette(chemin, dossier, k=14):
    os.makedirs(dossier, exist_ok=True)
    a = np.array(Image.open(chemin).convert("RGBA"))
    if a.shape[:2] != (512, 32):
        sys.exit(f"{chemin} : {a.shape[1]}×{a.shape[0]}, une bande 32×512 est attendue.")
    cols, nb = np.unique(a[a[..., 3] > 0][:, :3], axis=0, return_counts=True)
    hexa = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in cols]
    idx = {h: i for i, h in enumerate(hexa)}
    sw = Image.new("RGB", (len(cols) * 76, 92), (255, 255, 255))
    d = ImageDraw.Draw(sw)
    for i, (c, n) in enumerate(zip(cols, nb)):
        d.rectangle((i * 76, 0, i * 76 + 66, 50), fill=tuple(int(v) for v in c))
        d.text((i * 76, 55), f"{i} : {n} px", fill=(0, 0, 0))
        d.text((i * 76, 72), hexa[i], fill=(0, 0, 0))
    sw.save(os.path.join(dossier, "palette.png"))
    g = Image.new("RGB", (4 * 32 * k, 2 * 32 * k), (230, 230, 230))
    d = ImageDraw.Draw(g)
    for n, i in enumerate((4, 0, 8, 12, 5, 1, 9, 13)):
        f = a[i * 32:(i + 1) * 32]
        ox, oy = (n % 4) * 32 * k, (n // 4) * 32 * k
        for y in range(32):
            for x in range(32):
                if f[y, x, 3]:
                    c = tuple(int(v) for v in f[y, x, :3])
                    d.rectangle((ox + x * k, oy + y * k, ox + x * k + k - 1, oy + y * k + k - 1), fill=c)
                    d.text((ox + x * k + 2, oy + y * k + 1), str(idx["#%02x%02x%02x" % c]),
                           fill=(255, 255, 255) if sum(c) < 300 else (0, 0, 0))
        for y in range(0, 32, 4):
            d.text((ox + 1, oy + y * k), str(y), fill=(150, 0, 0))
        d.text((ox + 2, oy + 2), f"image {i} ({SENS[i // 4]})", fill=(200, 0, 0))
    g.save(os.path.join(dossier, "images.png"))
    with open(os.path.join(dossier, "palette.json"), "w", encoding="utf-8") as f:
        json.dump({str(i): h for i, h in enumerate(hexa)}, f, indent=1)
    print(os.path.join(dossier, "palette.png"))
    print(os.path.join(dossier, "images.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["planche", "palette"])
    p.add_argument("chemins", nargs="+")
    p.add_argument("--sortie", required=True)
    p.add_argument("--noms", help="planche : noms à garder (après Pt_npc_), séparés par des virgules")
    a = p.parse_args()
    if a.action == "planche":
        fichiers = []
        for c in a.chemins:
            fichiers += sorted(glob.glob(os.path.join(c, "*.png"))) if os.path.isdir(c) else [c]
        if a.noms:
            garder = {n.strip() for n in a.noms.split(",")}
            fichiers = [f for f in fichiers if os.path.splitext(os.path.basename(f))[0].replace("Pt_npc_", "") in garder]
        planche(fichiers, a.sortie)
    else:
        palette(a.chemins[0], a.sortie)


if __name__ == "__main__":
    main()
