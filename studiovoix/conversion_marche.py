"""Conversion en masse de feuilles de marche d'un style vers un autre (onglet « Convertir des feuilles de marche »).

Demande de l'utilisateur du 04/11 : ses anciennes feuilles de marche (style GBA, 4×4 cases de 16×24, ou RPG Maker XP
en 32×48) vers le style Diamant / Perle / Platine (3×4 cases de 32×32, personnage de ~19×26), en masse.

1. **Mise en page** (sans modèle) : chaque direction et chaque pose de la feuille d'origine est replacée dans la grille
   du style d'arrivée (ordre des lignes et des colonnes du style, `dresseur.STYLES[...]["marche"]`), le personnage
   agrandi au plus proche à la hauteur habituelle du style, posé en bas au centre, le pas de marche gardé.
2. **Redessin** (facultatif) : la mise en page, agrandie ×8 sur le fond magenta comme les images d'entraînement, est
   repassée par SDXL avec le modèle de feuilles de marche entraîné sur le style d'arrivée (image + masque blanc entier,
   `force` = part redessinée), puis ramenée à la grille (`modele_sprites.vers_feuille`) avec les couleurs de la
   feuille d'origine. La ligne « droite » peut être refaite en miroir de la ligne « gauche » (règle des jeux DS).

Rangement : data/pixel_art/<horodatage>_conversion_<nom>/ (mise_en_page/, travail/, resultats/, avant_apres/,
lot.json, <nom>.zip), hors galerie.
"""
import json
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from . import config as cfg
from . import diffusion
from . import modele_sprites as ms
from . import pixel_art as px
from .outils import nouveau_dossier
from .serie import _slug

SANS_IA = "Mise en page seulement (sans redessiner)"
FORCE_DEFAUT = 0.55
CONVERSION_MAX = 300
ECHECS_MAX = 3  # échecs du redessin à la suite avant d'arrêter
COULEURS_MAX = 32  # au-delà, la palette d'origine est résumée


def _marche(style):
    from . import dresseur

    if style not in dresseur.STYLES:
        raise gr.Error("Choisis un style.")
    return dresseur.STYLES[style]["marche"]


def poses(marche):
    """Colonnes de la feuille qui portent (debout, pas, autre pas) : 4 colonnes = debout, pas, debout, autre pas ;
    3 colonnes = debout, pas, autre pas."""
    return (0, 1, 3) if marche["grille"][0] == 4 else (0, 1, 2)


def _colonnes_arrivee(marche):
    """Pour chaque colonne du style d'arrivée, la pose (0 debout, 1 pas, 2 autre pas)."""
    return [0, 1, 0, 2] if marche["grille"][0] == 4 else [0, 1, 2]


def mettre_en_page(feuille, depart, arrivee):
    """Feuille RGBA (pixels du jeu, grille du style de départ) → feuille RGBA à la grille du style d'arrivée."""
    md, ma = _marche(depart), _marche(arrivee)
    (cd, rd), (cwd, chd) = md["grille"], md["cellule"]
    (ca, ra), (cwa, cha) = ma["grille"], ma["cellule"]
    cases = feuille.reshape(rd, chd, cd, cwd, 4).transpose(0, 2, 1, 3, 4)
    y, x = np.where((cases[..., 3] > 0).any(axis=(0, 1)))
    if not len(y):
        raise ValueError("feuille vide (fond seul)")
    y0, y1, x0, x1 = y.min(), y.max() + 1, x.min(), x.max() + 1
    h, w = y1 - y0, x1 - x0
    echelle = min(ma.get("hauteur", cha) / h, (cha - 1) / h, cwa / w)
    nh, nw = max(1, round(h * echelle)), max(1, round(w * echelle))
    sortie = np.zeros((ra * cha, ca * cwa, 4), np.uint8)
    source_poses, roles = poses(md), _colonnes_arrivee(ma)
    for r, sens in enumerate(ma["lignes"]):
        if sens not in md["lignes"]:
            raise ValueError(f"direction « {sens} » absente du style de départ")
        rs = md["lignes"].index(sens)
        for c, role in enumerate(roles):
            morceau = Image.fromarray(np.ascontiguousarray(cases[rs, source_poses[role], y0:y1, x0:x1]), "RGBA")
            morceau = np.array(morceau.resize((nw, nh), Image.Resampling.NEAREST))
            oy, ox = r * cha + cha - 1 - nh, c * cwa + (cwa - nw) // 2  # pieds un pixel au-dessus du bas
            sortie[oy:oy + nh, ox:ox + nw] = morceau
    return sortie


def miroir_gauche(feuille, style):
    """Ligne « droite » = ligne « gauche » retournée case par case (comme dans les jeux DS)."""
    m = _marche(style)
    (cols, _), (cw, ch) = m["grille"], m["cellule"]
    if "gauche" not in m["lignes"] or "droite" not in m["lignes"]:
        return feuille
    g, d = m["lignes"].index("gauche"), m["lignes"].index("droite")
    sortie = feuille.copy()
    for c in range(cols):
        case = feuille[g * ch:(g + 1) * ch, c * cw:(c + 1) * cw]
        sortie[d * ch:(d + 1) * ch, c * cw:(c + 1) * cw] = case[:, ::-1]
    return sortie


def palette_de(feuille):
    """Couleurs opaques de la feuille d'origine (résumées au-delà de COULEURS_MAX)."""
    pixels = feuille[feuille[..., 3] >= 128][:, :3]
    if not len(pixels):
        raise ValueError("feuille vide (fond seul)")
    uniques = np.unique(pixels, axis=0)
    return uniques if len(uniques) <= COULEURS_MAX else px._palette(pixels, COULEURS_MAX)


def modeles_de_marche():
    """Choix de la liste « Modèle » : la mise en page seule, puis les modèles de feuilles entraînés (avec leur style)."""
    choix = [SANS_IA]
    for nom in ms.loras(ms.MARCHE):
        style = ms.infos_modele(nom).get("style", "")
        choix.append(f"{nom} ({style})" if style else nom)
    return choix


def liste_modeles():
    choix = modeles_de_marche()
    return gr.update(choices=choix, value=choix[1] if len(choix) > 1 else choix[0])


def _nom_modele(choix):
    return (choix or "").split(" (")[0]


def _planche(avant, apres, k=4):
    """Avant (à gauche) et après (à droite), agrandis ×k sur gris clair, pour la galerie."""
    a = Image.fromarray(avant, "RGBA").resize((avant.shape[1] * k, avant.shape[0] * k), Image.Resampling.NEAREST)
    b = Image.fromarray(apres, "RGBA").resize((apres.shape[1] * k, apres.shape[0] * k), Image.Resampling.NEAREST)
    m = 4 * k
    planche = Image.new("RGBA", (a.width + b.width + 3 * m, max(a.height, b.height) + 2 * m), (215, 215, 220, 255))
    planche.alpha_composite(a, (m, m))
    planche.alpha_composite(b, (a.width + 2 * m, m))
    return planche


def convertir(fichiers, depart, arrivee, modele=SANS_IA, force=FORCE_DEFAUT, garder_couleurs=True, miroir=True,
              legender=True, graine=0, nom="", progress=gr.Progress()):
    """Feuilles de marche (images ou .zip) → feuilles au style d'arrivée. Renvoie (message, planches avant / après,
    archive .zip, dossier)."""
    sources = [Path(f) for f in ms.fichiers_feuilles(fichiers)]
    sources = [f for f in sources if f.suffix.lower() in ms.IMAGES]
    if not sources:
        raise gr.Error("Dépose tes feuilles de marche (PNG ou archive .zip).")
    if len(sources) > CONVERSION_MAX:
        raise gr.Error(f"{len(sources)} feuilles : {CONVERSION_MAX} au plus par conversion.")
    md, ma = _marche(depart), _marche(arrivee)
    lora = None
    if modele and modele != SANS_IA:
        lora = ms.loras(ms.MARCHE).get(_nom_modele(modele))
        if not lora:
            raise gr.Error(f"Modèle « {modele} » introuvable : entraîne-le dans « Entraîner un modèle de sprites ».")
        style_modele = ms.infos_modele(_nom_modele(modele)).get("style")
        if style_modele and style_modele != arrivee:
            raise gr.Error(f"Le modèle « {_nom_modele(modele)} » a appris le style « {style_modele} » : choisis ce "
                           "style d'arrivée ou un autre modèle.")
        diffusion.verifier("sprites_dresseur")
    nom = _slug(nom or "feuilles", "feuilles")
    dossier = nouveau_dossier(cfg.PIXEL_DIR, f"conversion_{nom}")
    for sous in ("mise_en_page", "resultats", "avant_apres") + (("travail",) if lora else ()):
        (dossier / sous).mkdir()
    (cols, rows), (cw, ch) = ma["grille"], ma["cellule"]
    lot, erreurs, noms = [], [], set()

    # 1. lecture et mise en page de toutes les feuilles (rapide, sans moteur)
    for i, f in enumerate(sources, 1):
        progress(0.05 * i / len(sources), desc=f"Mise en page {i}/{len(sources)}…")
        base = _slug(f.stem, f"feuille_{i:03d}")
        while base.lower() in noms:
            base += "_2"
        noms.add(base.lower())
        try:
            avant = ms.lire_feuille(f, md["grille"], md["cellule"])
            page = mettre_en_page(avant, depart, arrivee)
        except (OSError, ValueError) as e:
            erreurs.append(f"{f.name} : {e}")
            continue
        Image.fromarray(page, "RGBA").save(dossier / "mise_en_page" / f"{base}.png")
        lot.append({"source": f.name, "nom": base, "avant": avant, "page": page})
    if not lot:
        raise gr.Error("Aucune feuille lisible : " + " ; ".join(erreurs[:5]))

    # 2. légendes (Qwen) d'abord, puis tous les redessins (SDXL) : le moteur ne passe d'un modèle à l'autre qu'une fois
    if lora:
        qwen = legender and diffusion.present("qwen")
        for i, e in enumerate(lot, 1):
            entree = dossier / "travail" / f"{e['nom']}_entree.png"
            ms.image_entrainement(e["page"]).save(entree)
            e["entree"] = entree
            e["description"] = "a trainer"
            if qwen:
                progress(0.05 + 0.15 * (i - 1) / len(lot), desc=f"Description {i}/{len(lot)} (Qwen3-VL)…")
                try:
                    e["description"] = diffusion.decrire("feuille", "", image=entree, progress=None).strip().rstrip(".")
                except gr.Error as err:
                    print(f"Description impossible ({err}) : description générique.", flush=True)
        masque = dossier / "travail" / "masque.png"
        Image.new("L", Image.open(lot[0]["entree"]).size, 255).save(masque)
        a_la_suite = 0
        for i, e in enumerate(lot, 1):
            debut, part = 0.2 + 0.78 * (i - 1) / len(lot), 0.78 / len(lot)

            def suivi(p, desc="", debut=debut, part=part, i=i):
                progress(debut + part * min(1.0, max(0.0, float(p or 0))), desc=f"Feuille {i}/{len(lot)} : {desc}")

            brut = dossier / "travail" / f"{e['nom']}_sdxl.png"
            e["prompt"] = ms.prompt_feuille(e["description"], (cols, rows))
            try:
                diffusion.sprite_dresseur(e["prompt"], [brut], [int(graine or 0) + i - 1], cols * cw * ms.FACTEUR,
                                          rows * ch * ms.FACTEUR, lora=str(lora), image=e["entree"], masque=masque,
                                          force=float(force), progress=suivi)
                image = Image.open(brut)
                palette = palette_de(e["avant"]) if garder_couleurs else ms._palette_hors_fond(image, 15)
                e["resultat"] = ms.vers_feuille(image, (cols, rows), (cw, ch), palette)
                a_la_suite = 0
            except (gr.Error, OSError, ValueError) as err:
                erreurs.append(f"{e['source']} : {err}")
                a_la_suite += 1
                if a_la_suite >= ECHECS_MAX:
                    erreurs.append(f"arrêt après {ECHECS_MAX} échecs à la suite")
                    break
    else:
        for e in lot:
            e["resultat"] = e["page"]

    # 3. résultats
    faits = [e for e in lot if "resultat" in e]
    if not faits:
        raise gr.Error("Aucune feuille convertie : " + " ; ".join(erreurs[:5]))
    planches = []
    for e in faits:
        if miroir:
            e["resultat"] = miroir_gauche(e["resultat"], arrivee)
        im = Image.fromarray(e["resultat"], "RGBA")
        im.save(dossier / "resultats" / f"{e['nom']}.png")
        im.resize((im.width * 2, im.height * 2), Image.Resampling.NEAREST).save(
            dossier / "resultats" / f"{e['nom']}_x2.png")
        chemin = dossier / "avant_apres" / f"{e['nom']}.png"
        _planche(e["avant"], e["resultat"]).save(chemin)
        planches.append((str(chemin), e["source"]))
    archive = dossier / f"{nom}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for p in sorted((dossier / "resultats").glob("*.png")):
            z.write(p, p.name)
    (dossier / "lot.json").write_text(json.dumps({
        "depart": depart, "arrivee": arrivee, "modele": _nom_modele(modele) if lora else None,
        "force": float(force) if lora else None, "garder_couleurs": bool(garder_couleurs), "miroir": bool(miroir),
        "graine": int(graine or 0), "erreurs": erreurs,
        "feuilles": [{k: e.get(k) for k in ("source", "nom", "description", "prompt")} | {"fait": "resultat" in e}
                     for e in lot]}, ensure_ascii=False, indent=1), encoding="utf-8")
    msg = (f"✅ {len(faits)} feuille(s) convertie(s) sur {len(sources)} ({cols}×{rows} cases de {cw}×{ch} px"
           + (f", redessinées avec « {_nom_modele(modele)} »" if lora else ", mise en page seulement") + ").")
    if erreurs:
        msg += "\n\n⚠️ " + "\n\n⚠️ ".join(erreurs[:10]) + (f"\n\n… et {len(erreurs) - 10} autre(s)." if len(erreurs) > 10 else "")
    progress(1.0, desc="Terminé")
    return msg, planches, str(archive), str(dossier)
