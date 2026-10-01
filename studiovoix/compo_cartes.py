"""Composition d'une carte à jouer : illustration + cadre + nom, coût, type, texte d'effet, texte d'ambiance,
attaque, défense, rareté et pied de carte, en PNG (et WebP) prêt pour le jeu ou l'impression.

Tout est dessiné avec Pillow (déjà dans l'application) : cadre en dégradé aux couleurs de la faction, zones de texte
dont la taille de police diminue jusqu'à ce que tout tienne. Polices libres (SIL Open Font License, usage
commercial permis) fournies dans polices/ : Cinzel (titres) et EB Garamond (texte).
Format de base 750×1050 px = 63×88 mm à 300 ppp (format des cartes à collectionner) ; tout est proportionnel.
Rangement : data/cartes/<projet>/composees/<id>.png (+ .webp) et <id>.json (les champs, pour recomposer).
"""
import json
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import config as cfg

POLICES = cfg.APP_DIR / "polices"
BASE = (750, 1050)
FORMATS = {
    "Standard 63×88 mm à 300 ppp (750×1050)": 1,
    "Haute définition (1500×2100)": 2,
}
FORMAT_DEFAUT = next(iter(FORMATS))
# faction → (couleur sombre, couleur claire) du cadre
FACTIONS = {
    "Feu": ((120, 22, 12), (236, 120, 46)),
    "Eau": ((14, 52, 120), (86, 164, 236)),
    "Nature": ((28, 84, 32), (128, 190, 84)),
    "Lumière": ((132, 98, 22), (246, 214, 120)),
    "Ténèbres": ((40, 14, 62), (140, 82, 190)),
    "Neutre": ((64, 64, 70), (180, 180, 188)),
}
RARETES = {
    "Commune": (30, 30, 30),
    "Peu commune": (176, 186, 196),
    "Rare": (226, 176, 40),
    "Épique": (150, 60, 204),
    "Légendaire": (244, 112, 20),
}
PARCHEMIN = (242, 230, 204)
ENCRE = (34, 26, 18)


def _police(nom, taille, gras=False):
    chemin = POLICES / nom
    try:
        f = ImageFont.truetype(str(chemin), int(taille))
        if gras:
            f.set_variation_by_axes([700])
        return f
    except OSError:  # polices absentes : police intégrée de Pillow
        return ImageFont.load_default(int(taille))


def _ajuster_ligne(texte, nom_police, taille, largeur, gras=False, minimum=10):
    """Police la plus grande (≤ taille) pour que le texte tienne sur une ligne."""
    while taille > minimum:
        f = _police(nom_police, taille, gras)
        if f.getlength(texte) <= largeur:
            return f
        taille -= 1
    return _police(nom_police, minimum, gras)


def _couper(texte, police, largeur):
    """Lignes du texte coupées aux espaces pour tenir dans la largeur (les retours à la ligne sont gardés)."""
    lignes = []
    for paragraphe in (texte or "").split("\n"):
        mots, ligne = paragraphe.split(), ""
        if not mots:
            lignes.append("")
            continue
        for mot in mots:
            essai = f"{ligne} {mot}".strip()
            if police.getlength(essai) <= largeur or not ligne:
                ligne = essai
            else:
                lignes.append(ligne)
                ligne = mot
        lignes.append(ligne)
    return lignes


def _bloc_texte(effet, ambiance, largeur, hauteur, taille_max):
    """Taille de police la plus grande pour que l'effet (normal) et l'ambiance (italique) tiennent dans la boîte.
    Renvoie (taille, lignes d'effet, lignes d'ambiance, interligne)."""
    for taille in range(int(taille_max), 9, -1):
        normal = _police("EBGaramond.ttf", taille)
        italique = _police("EBGaramond-Italic.ttf", taille)
        l_effet = _couper(effet, normal, largeur) if effet else []
        l_amb = _couper(ambiance, italique, largeur) if ambiance else []
        pas = taille * 1.22
        total = pas * (len(l_effet) + len(l_amb)) + (taille * 0.8 if l_effet and l_amb else 0)
        if total <= hauteur:
            return taille, l_effet, l_amb, pas
    return 10, l_effet, l_amb, 12.2


def _degrade(taille, haut, bas):
    """Dégradé vertical."""
    l_, h = taille
    im = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(1, h - 1)
        im.putpixel((0, y), tuple(int(a + (b - a) * t) for a, b in zip(haut, bas)))
    return im.resize((l_, h))


def composer(illustration, nom, cout="", type_="", effet="", ambiance="", attaque="", defense="", faction="Neutre",
             rarete="Commune", pied="", format_label=FORMAT_DEFAUT):
    """Carte composée (image PIL RGB) et informations de mise en page (taille de police du texte…)."""
    k = FORMATS.get(format_label, 1)
    L, H = BASE[0] * k, BASE[1] * k

    def s(*v):  # coordonnées de la maquette 750×1050 → format choisi
        return tuple(int(x * k) for x in v)

    sombre, clair = FACTIONS.get(faction, FACTIONS["Neutre"])
    carte = Image.new("RGBA", (L, H), (0, 0, 0, 0))
    masque = Image.new("L", (L, H), 0)
    ImageDraw.Draw(masque).rounded_rectangle((0, 0, L - 1, H - 1), radius=36 * k, fill=255)
    fond = _degrade((L, H), clair, sombre)
    carte.paste(fond, (0, 0), masque)
    d = ImageDraw.Draw(carte)
    d.rounded_rectangle((0, 0, L - 1, H - 1), radius=36 * k, outline=(12, 10, 8), width=10 * k)
    d.rounded_rectangle(s(22, 22, 728, 1028), radius=24 * k, outline=(*clair, 255), width=3 * k)

    # Illustration (recadrée pour remplir la fenêtre, sans déformation)
    x0, y0, x1, y1 = s(48, 124, 702, 604)
    if illustration and Path(illustration).exists():
        with Image.open(illustration) as im:
            vue = ImageOps.fit(im.convert("RGB"), (x1 - x0, y1 - y0), Image.LANCZOS, centering=(0.5, 0.4))
        carte.paste(vue, (x0, y0))
    else:
        d.rectangle((x0, y0, x1, y1), fill=(30, 30, 34))
    d.rectangle((x0 - 3 * k, y0 - 3 * k, x1 + 3 * k, y1 + 3 * k), outline=(12, 10, 8), width=5 * k)

    # Bandeau du nom et coût
    d.rounded_rectangle(s(38, 38, 712, 110), radius=14 * k, fill=(*PARCHEMIN, 236), outline=(12, 10, 8), width=3 * k)
    marge_cout = 96 if str(cout).strip() else 20
    police_nom = _ajuster_ligne(nom or "", "Cinzel.ttf", 38 * k, (712 - 58 - marge_cout) * k, gras=True)
    d.text(s(58, 74), nom or "", font=police_nom, fill=ENCRE, anchor="lm")
    if str(cout).strip():
        cx, cy, r = s(668, 74, 40)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*clair, 255), outline=(12, 10, 8), width=4 * k)
        d.ellipse((cx - r + 6 * k, cy - r + 6 * k, cx + r - 6 * k, cy + r - 6 * k), outline=(*sombre, 255),
                  width=2 * k)
        police_cout = _ajuster_ligne(str(cout).strip(), "Cinzel.ttf", 40 * k, 2 * r - 22 * k, gras=True)
        d.text((cx, cy), str(cout).strip(), font=police_cout, fill=ENCRE, anchor="mm")

    # Ligne de type et gemme de rareté
    d.rounded_rectangle(s(38, 612, 712, 664), radius=12 * k, fill=(*PARCHEMIN, 236), outline=(12, 10, 8), width=3 * k)
    police_type = _ajuster_ligne(type_ or "", "Cinzel.ttf", 26 * k, (712 - 58 - 70) * k)
    d.text(s(58, 638), type_ or "", font=police_type, fill=ENCRE, anchor="lm")
    gx, gy, gr_ = s(678, 638, 15)
    d.polygon([(gx, gy - gr_), (gx + gr_, gy), (gx, gy + gr_), (gx - gr_, gy)], fill=RARETES.get(rarete, RARETES["Commune"]),
              outline=(12, 10, 8))

    # Texte d'effet puis texte d'ambiance (italique), taille réduite jusqu'à ce que tout tienne
    tx0, ty0, tx1, ty1 = s(48, 676, 702, 940)
    d.rounded_rectangle((tx0, ty0, tx1, ty1), radius=14 * k, fill=(*PARCHEMIN, 244), outline=(12, 10, 8), width=3 * k)
    marge = 20 * k
    taille, l_effet, l_amb, pas = _bloc_texte(effet, ambiance, tx1 - tx0 - 2 * marge, ty1 - ty0 - 2 * marge, 32 * k)
    y = ty0 + marge
    normal, italique = _police("EBGaramond.ttf", taille), _police("EBGaramond-Italic.ttf", taille)
    for ligne in l_effet:
        d.text((tx0 + marge, y), ligne, font=normal, fill=ENCRE)
        y += pas
    if l_effet and l_amb:
        y += taille * 0.3
        d.line((tx0 + 3 * marge, y, tx1 - 3 * marge, y), fill=(150, 130, 100), width=max(1, k))
        y += taille * 0.5
    for ligne in l_amb:
        d.text((tx0 + marge, y), ligne, font=italique, fill=(70, 56, 40))
        y += pas

    # Attaque et défense (facultatives), pied de carte
    for valeur, (bx, etiquette) in ((attaque, (52, "ATQ")), (defense, (602, "DEF"))):
        if str(valeur).strip():
            b0, b1, b2, b3 = s(bx, 952, bx + 96, 1012)
            d.rounded_rectangle((b0, b1, b2, b3), radius=12 * k, fill=(*clair, 255), outline=(12, 10, 8), width=3 * k)
            d.text(((b0 + b2) // 2, b1 + 14 * k), etiquette, font=_police("Cinzel.ttf", 13 * k, True), fill=ENCRE,
                   anchor="mm")
            police_stat = _ajuster_ligne(str(valeur).strip(), "Cinzel.ttf", 32 * k, 80 * k, gras=True)
            d.text(((b0 + b2) // 2, b1 + 38 * k), str(valeur).strip(), font=police_stat, fill=ENCRE, anchor="mm")
    if pied:
        police_pied = _ajuster_ligne(pied, "EBGaramond.ttf", 18 * k, 420 * k)
        d.text(s(375, 990), pied, font=police_pied, fill=(250, 246, 236), anchor="mm")

    fond_final = Image.new("RGB", (L, H), (255, 255, 255))
    fond_final.paste(carte, (0, 0), carte)
    return fond_final, {"taille_texte": taille, "lignes_effet": len(l_effet), "lignes_ambiance": len(l_amb)}


def identifiant(nom):
    from .export import _slug

    return _slug(nom or "carte", set())


def composer_et_enregistrer(projet, illustration, nom, cout, type_, effet, ambiance, attaque, defense, faction, rarete,
                            pied, format_label, webp=True):
    """Compose la carte, l'enregistre dans data/cartes/<projet>/composees/. Renvoie (message, image, fichiers)."""
    from .cartes import nom_projet

    if not illustration or not Path(illustration).exists():
        raise gr.Error("Choisis d'abord une illustration (clique sur une variante générée, ou importe une image).")
    if not (nom or "").strip():
        raise gr.Error("Donne un nom à la carte.")
    image, infos = composer(illustration, nom.strip(), cout, type_, effet, ambiance, attaque, defense, faction,
                            rarete, pied, format_label)
    dossier = cfg.CARDS_DIR / nom_projet(projet) / "composees"
    dossier.mkdir(parents=True, exist_ok=True)
    ident = identifiant(nom)
    png = dossier / f"{ident}.png"
    image.save(png, dpi=(300, 300))
    fichiers = [str(png)]
    if webp:
        image.save(png.with_suffix(".webp"), "WEBP", quality=90, method=6)
        fichiers.append(str(png.with_suffix(".webp")))
    champs = {"nom": nom.strip(), "cout": cout, "type": type_, "effet": effet, "ambiance": ambiance, "attaque": attaque,
              "defense": defense, "faction": faction, "rarete": rarete, "pied": pied, "format": format_label,
              "illustration": str(illustration)}
    png.with_suffix(".json").write_text(json.dumps(champs, ensure_ascii=False, indent=1), encoding="utf-8")
    note = " (texte réduit pour tenir dans la boîte)" if infos["taille_texte"] < 26 * FORMATS.get(format_label, 1) else ""
    return f"🃏 Carte « {nom.strip()} » composée{note} : {png}", str(png), fichiers
