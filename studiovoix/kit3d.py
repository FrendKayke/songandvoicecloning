"""Kit d'objets 3D (onglet « Jeu → Kit d'objets 3D ») : une liste d'objets (un par ligne, en français) → un modèle 3D
par objet, tous dans le même style (par exemple les immeubles, voitures et arbres d'un jeu de construction de ville
en low poly).

1. Préparer (`preparer`) : une ligne = un objet ; descriptions traduites en anglais par Qwen3-VL (mode « objets »,
   liste numérotée, par paquets) ; tableau [nom, objet, description] modifiable.
2. Dessiner (`dessiner`) : Z-Image d'après « description, style commun, objet seul » (par paquets de 8, progression
   affichée) ; images à vérifier avant la 3D.
3. Créer les modèles 3D (`generer`) : Hunyuan3D-2 sur chaque image (`modele3d.generer`), par défaut en « Low poly
   facetté » (faces planes de couleur unie, ~3 000 faces) ; `<nom>.glb` de chaque objet et `<kit>.zip`.
Rangement : data/3d/kits/<horodatage>_<nom>/ (rendus/, <nom>.glb, kit.json, <nom>.zip) ; chaque modèle a aussi sa
création dans data/3d/<horodatage>/ (galerie, pack du projet).
"""
import json
import shutil
import zipfile
from pathlib import Path

import gradio as gr

from . import config as cfg
from . import diffusion, modele3d
from .images import scenes_du_texte
from .outils import nouveau_dossier
from .serie import _slug
from .tileset3d import PAQUET, _partie

STYLE_DEFAUT = ("low poly 3D game asset, flat shading, simple geometric shapes, chunky proportions, bright saturated "
                "solid colors, no texture detail, city builder game style")
QUALITE = "Low poly facetté (couleurs unies, ~3 000 faces)"  # dans modele3d.QUALITES
KIT_MAX = 100
TRADUCTION_PAQUET = 15
EXEMPLE = ("un petit immeuble en briques rouges de quatre étages, toit plat avec un réservoir d'eau\n"
           "une station-service avec un auvent rouge et blanc et deux pompes\n"
           "un taxi jaune\n"
           "un camion de pompiers rouge avec une échelle\n"
           "un arbre rond vert au tronc marron\n"
           "une montgolfière rayée rouge et blanche")
ENTETES = ["Nom (fichier)", "Objet", "Description anglaise (modifiable)"]


def objets_du_texte(texte):
    """Une ligne non vide = un objet (puces et numéros retirés)."""
    return [l_ for l_ in scenes_du_texte(texte) if l_]


def _traduire(objets, progress):
    """Descriptions anglaises (Qwen3-VL, mode « objets ») ; une réponse incomplète garde le texte d'origine du
    paquet. Renvoie (descriptions, nombre de lignes non traduites)."""
    sortie, manquees = [], 0
    for p0 in range(0, len(objets), TRADUCTION_PAQUET):
        paquet = objets[p0:p0 + TRADUCTION_PAQUET]
        progress(p0 / len(objets), desc=f"Traduction des objets {p0 + 1} à {p0 + len(paquet)} sur {len(objets)}…")
        texte = f"Number of lines: {len(paquet)}\n\n" + "\n".join(f"{i}. {o}" for i, o in enumerate(paquet, 1))
        lignes = scenes_du_texte(diffusion.decrire("objets", texte, nombre=len(paquet)))
        if len(lignes) != len(paquet):
            manquees += len(paquet)
            sortie += paquet
        else:
            sortie += [t or o for t, o in zip(lignes, paquet)]
    return sortie, manquees


def _noms(objets):
    """Nom de fichier ASCII de chaque objet, numéroté (deux objets semblables ne s'écrasent pas)."""
    return [f"{i:02d}_{_slug(' '.join(o.split()[:5]), 'objet')}" for i, o in enumerate(objets, 1)]


def preparer(texte, traduire=True, nom="kit", progress=gr.Progress()):
    """Liste → tableau [nom, objet, description anglaise]. Renvoie (tableau, message, dossier)."""
    objets = objets_du_texte(texte)
    if not objets:
        raise gr.Error("Écris les objets du kit, un par ligne (« un taxi jaune », « un arbre rond »…).")
    if len(objets) > KIT_MAX:
        raise gr.Error(f"{len(objets)} objets : {KIT_MAX} au plus par kit.")
    msg = ""
    if traduire and diffusion.present("qwen"):
        descriptions, manquees = _traduire(objets, progress)
        if manquees:
            msg = f" ⚠️ {manquees} ligne(s) gardée(s) telles quelles (réponse incomplète de Qwen3-VL)."
    else:
        descriptions = list(objets)
        if traduire:
            msg = " Qwen3-VL n'est pas installé : descriptions gardées telles quelles (l'anglais marche mieux)."
    dossier = nouveau_dossier(cfg.MODELS3D_DIR / "kits", _slug(nom, "kit"))
    table = [[n, o, d] for n, o, d in zip(_noms(objets), objets, descriptions)]
    _ecrire(dossier, {"nom": nom or "kit", "objets": [dict(nom=n, objet=o, description=d) for n, o, d in table]})
    return table, (f"📝 {len(table)} objet(s). Vérifie et corrige les descriptions si besoin, puis « Dessiner les "
                   "objets »." + msg), str(dossier)


def _lire(dossier):
    if not dossier or not (Path(dossier) / "kit.json").exists():
        raise gr.Error("Prépare d'abord la liste des objets (bouton « 📝 Préparer »).")
    return Path(dossier), json.loads((Path(dossier) / "kit.json").read_text(encoding="utf-8"))


def _ecrire(d, kit):
    (Path(d) / "kit.json").write_text(json.dumps(kit, ensure_ascii=False, indent=1), encoding="utf-8")


def _objets(kit, table):
    """Lignes du tableau (modifiées par l'utilisateur) → objets du kit mis à jour (nom, description)."""
    if hasattr(table, "values"):  # DataFrame
        table = table.values.tolist()
    anciens = {o["nom"]: o for o in kit["objets"]}
    objets = []
    for ligne in table or []:
        if not ligne or not str(ligne[2] if len(ligne) > 2 else "").strip():
            continue
        nom = _slug(str(ligne[0] or ""), f"objet_{len(objets) + 1:02d}")
        o = dict(anciens.get(nom, {}), nom=nom, objet=str(ligne[1] or ""), description=str(ligne[2]).strip())
        objets.append(o)
    if not objets:
        raise gr.Error("Le tableau des objets est vide : prépare d'abord la liste.")
    return objets


def prompt_objet(description, style):
    return f"{description.rstrip('. ')}, {(style or STYLE_DEFAUT).strip().rstrip('.')}, {modele3d.OBJET_SEUL}"


def dessiner(dossier, table, style=STYLE_DEFAUT, graine=0, seulement_manquants=False, progress=gr.Progress()):
    """Une image par objet (Z-Image, par paquets). Renvoie (galerie, message, dossier)."""
    d, kit = _lire(dossier)
    objets = _objets(kit, table)
    graine = int(graine or 0) or kit.get("graine") or diffusion.graines(1)[0]
    (d / "rendus").mkdir(exist_ok=True)
    for i, o in enumerate(objets):
        o.update(prompt=prompt_objet(o["description"], style), graine=graine + i,
                 rendu=str(d / "rendus" / f"{o['nom']}.png"))
    a_faire = [o for o in objets if not seulement_manquants or not Path(o["rendu"]).exists()
               or o.get("prompt_rendu") != o["prompt"]]
    for i in range(0, len(a_faire), PAQUET):
        paquet = a_faire[i:i + PAQUET]
        prompts = [o["prompt"] for o in paquet]
        diffusion.image(prompts if len(prompts) > 1 else prompts[0], [o["rendu"] for o in paquet],
                        [o["graine"] for o in paquet], 1024, 1024,
                        progress=_partie(progress, i / len(a_faire), (i + len(paquet)) / len(a_faire),
                                         f"objets {i + 1} à {i + len(paquet)} sur {len(a_faire)}"))
        for o in paquet:
            o["prompt_rendu"] = o["prompt"]
    kit.update(objets=objets, style=style, graine=graine)
    _ecrire(d, kit)
    return ([(o["rendu"], o["nom"]) for o in objets],
            f"🎨 {len(a_faire)} objet(s) dessiné(s) (graine {graine}). Un objet raté : corrige sa description ou "
            "change la graine et redessine ; sinon « Créer les modèles 3D ».", str(d))


def generer(dossier, table, style=STYLE_DEFAUT, qualite=QUALITE, texture=True, graine=0, projet=None,
            progress=gr.Progress()):
    """Images manquantes (ou dont la description a changé) dessinées d'abord, puis un modèle 3D par objet. Un objet
    en échec n'arrête pas les autres. Renvoie (message, dernier GLB, fichiers, dossier)."""
    dessiner(dossier, table, style, graine, seulement_manquants=True, progress=progress)
    d, kit = _lire(dossier)
    objets = kit["objets"]
    lignes, produits, dernier = [], [], None
    for i, o in enumerate(objets):
        try:
            msg, glb, _, _, creation = modele3d.generer(
                o["rendu"], o["nom"], qualite, texture, o["graine"], ["glb"], description=o["prompt"],
                description_fr=o.get("objet") or None, projet=projet,
                progress=_partie(progress, i / len(objets), (i + 1) / len(objets),
                                 f"modèle 3D de {o['nom']} ({i + 1}/{len(objets)})"))
        except gr.Error as e:
            lignes.append(f"- ❌ {o['nom']} : {getattr(e, 'message', e)}")
            continue
        copie = d / f"{o['nom']}.glb"
        shutil.copy(glb, copie)
        o.update(glb=str(copie), creation=creation)
        produits.append(str(copie))
        dernier = str(copie)
        lignes.append(f"- ✅ {o['nom']} : {msg.split(' : ', 1)[-1].split('. Dossier')[0]}")
    if produits:
        archive = d / f"{_slug(kit.get('nom'), 'kit')}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in produits:
                z.write(p, Path(p).name)
        produits.insert(0, str(archive))
    kit.update(qualite=qualite)
    _ecrire(d, kit)
    reussis = len([l_ for l_ in lignes if l_.startswith("- ✅")])
    return (f"**{reussis}/{len(objets)} objet(s) en 3D** (dans {d}, archive .zip comprise ; chacun aussi dans la "
            "Galerie) :\n\n" + "\n".join(lignes), dernier, produits, str(d))
