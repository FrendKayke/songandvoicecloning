"""Onglet « Images » : une image à partir d'un prompt (Z-Image-Turbo) ou d'une photo (FLUX.2 klein 4B).

Avec une photo, trois usages : la modifier (mêmes personnes, même cadrage, avec les changements demandés), garder
son sujet dans une nouvelle scène, ou reprendre sa composition. Les deux modèles sont sous licence Apache 2.0.
Chaque création : data/images/<horodatage>/ avec photo.<ext> (si donnée), variante_<i>.png, creation.json type
« image » (galerie : affichage, « Recréer » avec la graine).
"""
import shutil
import time
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from . import cartes
from . import config as cfg
from . import diffusion
from .outils import ecrire_creation, nouveau_dossier
from .styles import image as style_image
from .videos import MOTS_PROMPT_COURT, duree_lisible

AUTO = "Automatique (proportions de la photo, sinon carré)"
# Côtés multiples de 16, ~1 mégapixel (Z-Image va jusqu'à ~2 Mpx, klein est plus rapide vers 1 Mpx)
FORMATS = {
    AUTO: None,
    "Carré 1:1 (1024×1024)": (1024, 1024),
    "Portrait 3:4 (896×1184)": (896, 1184),
    "Portrait 2:3 (832×1248)": (832, 1248),
    "Paysage 4:3 (1184×896)": (1184, 896),
    "Paysage 16:9 (1344×768)": (1344, 768),
    "Vertical 9:16 (768×1344)": (768, 1344),
}
PIXELS_AUTO = 1024 * 1024
STYLES = [("Photo réaliste", "photorealistic photograph, natural light, sharp focus, high detail")] + [
    s for s in cartes.STYLES if "trading card" not in s[1]]

# Ce que l'image reprend de la photo (libellé → clé)
USAGES = {
    "Modifier la photo (mêmes personnes, même cadrage, avec les changements décrits)": "modifier",
    "Garder le sujet (même personne ou objet, dans une nouvelle scène)": "sujet",
    "Reprendre la composition (pose, cadrage)": "composition",
}
USAGE_DEFAUT = next(iter(USAGES))
GARDER_LE_RESTE = "Keep everything else exactly as in the reference image: same people, faces, pose and framing"
SUJET = ("The main subject is the person or object of the reference image: keep their face, features, hair, "
         "body shape and recognizable details")
COMPOSITION = "Follow the composition, pose and framing of the reference image"
# plusieurs photos (1 à 4, références de FLUX.2 klein dans l'ordre) : le texte parle de « photo 1 », « photo 2 »…
PHOTOS_MAX = 4
GARDER_LE_RESTE_N = "Keep everything else exactly as in image 1: same people, faces, pose and framing"
SUJET_N = ("Keep the faces, features, hair, outfits and recognizable details of the people, creatures and objects "
           "taken from the reference images")
COMPOSITION_N = "Follow the composition, pose and framing of image 1"


def photos_de(photo, autres=None):
    """La photo principale (chemin ou None) puis les autres photos (chemins ou fichiers Gradio), 4 au plus ; accepte
    aussi une liste déjà faite. Vérifie qu'elles existent."""
    liste = list(photo) if isinstance(photo, (list, tuple)) else ([photo] if photo else [])
    liste += [getattr(f, "name", f) for f in (autres or []) if f]
    liste = [str(c) for c in liste]
    if len(liste) > PHOTOS_MAX:
        raise gr.Error(f"{len(liste)} photos : {PHOTOS_MAX} au plus (la photo principale et 3 autres).")
    for c in liste:
        if not Path(c).is_file():
            raise gr.Error(f"Photo introuvable : {c}")
    return liste

EXEMPLES = [
    ("Paysage de montagne", "un lac de montagne au lever du soleil, brume sur l'eau, sapins enneigés"),
    ("Portrait", "portrait d'une vieille marin au visage buriné, ciré jaune, lumière douce de fin de journée"),
    ("Objet produit", "une montre ancienne en or posée sur du velours bleu, éclairage de studio"),
    ("Retouche : la nuit", "transforme la scène en nuit, avec la lune et des lumières chaudes aux fenêtres"),
    ("Retouche : chapeau", "ajoute un chapeau de pirate à la personne"),
    ("Retouche : peinture", "redessine la photo comme une peinture à l'huile"),
]


def usage_de(libelle):
    return USAGES.get(libelle, libelle if libelle in USAGES.values() else "modifier")


def format_pour(format_label, photo=None):
    """(largeur, hauteur) : le format choisi, ou en automatique les proportions de la photo à ~1 Mpx."""
    if FORMATS.get(format_label):
        return FORMATS[format_label]
    if photo:
        try:
            with Image.open(photo) as im:
                w, h = ImageOps.exif_transpose(im).size
        except OSError:
            return FORMATS["Carré 1:1 (1024×1024)"]
        echelle = (PIXELS_AUTO / (w * h)) ** 0.5
        return (max(512, min(1536, round(w * echelle / 16) * 16)), max(512, min(1536, round(h * echelle / 16) * 16)))
    return FORMATS["Carré 1:1 (1024×1024)"]


def prompt_final(prompt, styles=None, usage=None, nombre_photos=1):
    """usage : None (sans photo), « modifier », « sujet » ou « composition » ; plusieurs photos : consignes qui
    désignent « image 1 »."""
    prompt = (prompt or "").strip().rstrip(".")
    if nombre_photos > 1:
        ajout = {"modifier": GARDER_LE_RESTE_N, "sujet": SUJET_N, "composition": COMPOSITION_N}.get(usage)
    else:
        ajout = {"modifier": GARDER_LE_RESTE, "sujet": SUJET, "composition": COMPOSITION}.get(usage)
    if ajout:
        prompt = f"{prompt}. {ajout}"
    style = style_image(styles, STYLES)  # « dessin », « aquarelle » tapés en français : termes anglais
    return f"{prompt}. Style: {style}." if style else f"{prompt}."


def preparer(description, photo=None, usage_label=USAGE_DEFAUT, progress=gr.Progress()):
    """Description française → prompt anglais (Qwen3-VL), qui voit la photo s'il y en a une. « Modifier la photo » :
    une consigne de retouche courte ; sinon une description détaillée de l'image à créer."""
    description = (description or "").strip()
    if not description:
        raise gr.Error("Décris l'image à créer, ou la modification à faire sur la photo (en français ou en anglais).")
    photos = photos_de(photo)
    mode = "retouche" if photos and usage_de(usage_label) == "modifier" else "scene"
    image = _planche(photos, numeros=True) if len(photos) > 1 else (photos[0] if photos else None)
    return diffusion.decrire(mode, description, image=image, progress=progress, photos=len(photos))


def prompt_pret(description, prompt, photo=None, usage_label=USAGE_DEFAUT, progress=gr.Progress()):
    """Avant « Générer » : prompt vide → préparé depuis la description ; prompt court tapé à la main sans description
    → enrichi par Qwen (comme pour les vidéos : un texte trop court donne une image quelconque) ; sinon gardé."""
    prompt, description = (prompt or "").strip(), (description or "").strip()
    retouche = bool(photo) and usage_de(usage_label) == "modifier"
    if prompt and not description and not retouche and len(prompt.split()) < MOTS_PROMPT_COURT:
        return preparer(prompt, photo, usage_label, progress=progress)
    return prompt or preparer(description, photo, usage_label, progress=progress)


def generer(prompt, photo, usage_label, styles, format_label, variantes, graine, nom="", description_fr=None,
            progress=gr.Progress()):
    """Génère les variantes. Renvoie (message, galerie [(image, légende)], dossier)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Il manque le prompt : clique d'abord sur « Préparer le prompt », ou écris-le en anglais.")
    photos = photos_de(photo)
    debut = time.monotonic()
    usage = usage_de(usage_label) if photos else None
    format_label = format_label if format_label in FORMATS else AUTO
    largeur, hauteur = format_pour(format_label, photos[0] if photos else None)
    n = max(1, min(4, int(variantes or 1)))
    dossier = nouveau_dossier(cfg.IMAGES_DIR)
    copies = []  # copiées dans la création : la recréer ne dépend pas des fichiers d'origine
    for k, ph in enumerate(photos, 1):
        copies.append(dossier / (f"photo{Path(ph).suffix.lower() or '.png'}" if k == 1
                                 else f"photo_{k}{Path(ph).suffix.lower() or '.png'}"))
        shutil.copy(ph, copies[-1])
    final = prompt_final(prompt, styles, usage, len(copies))
    sorties = [dossier / f"variante_{i}.png" for i in range(1, n + 1)]
    if copies:
        res = diffusion.personnage(final, copies, sorties, diffusion.graines(n, graine), largeur, hauteur,
                                   progress=progress)
    else:
        res = diffusion.image(final, sorties, diffusion.graines(n, graine), largeur, hauteur, progress=progress)
    temps = time.monotonic() - debut
    nom = "".join(c for c in (nom or "").strip() if c.isalnum() or c in "-_ ").strip() or "image"
    ecrire_creation(dossier, {
        "type": "image", "nom": nom, "description": prompt, "description_fr": (description_fr or "").strip() or None,
        "styles": styles, "format": format_label, "largeur": largeur, "hauteur": hauteur, "prompt": final,
        "photo": copies[0].name if copies else None, "autres_photos": [c.name for c in copies[1:]],
        "usage": usage, "temps_s": round(temps), "choisie": 1,
        "moteur": "FLUX.2 klein 4B" if copies else "Z-Image-Turbo",
        "versions": [{"graine": g, "dossier": ".", "fichier": f} for f, g in zip(res["fichiers"], res["graines"])],
    })
    galerie = [(f, f"Variante {i} (graine {g})") for i, (f, g) in enumerate(zip(res["fichiers"], res["graines"]), 1)]
    origine = {"modifier": "photo modifiée", "sujet": "sujet de la photo", "composition": "composition de la photo",
               None: "à partir du texte"}[usage]
    if len(copies) > 1:
        origine = f"{len(copies)} photos combinées"
    msg = (f"✅ {len(galerie)} image(s) {largeur}×{hauteur} ({origine}) en {duree_lisible(temps)}, dans {dossier}. "
           f"Graines : {', '.join(str(g) for g in res['graines'])}. Clique sur une variante pour la retoucher ou "
           "l'animer.")
    return msg, galerie, str(dossier)


def choisir(dossier, evt: gr.SelectData):
    """Clic sur une variante : renvoie (message, chemin de l'image) ; notée « choisie » dans creation.json."""
    import json

    if not dossier:
        raise gr.Error("Génère d'abord une image.")
    chemin = Path(dossier) / "creation.json"
    infos = json.loads(chemin.read_text(encoding="utf-8"))
    versions = infos.get("versions") or []
    if not 0 <= evt.index < len(versions):
        raise gr.Error("Variante introuvable.")
    infos["choisie"] = evt.index + 1
    chemin.write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    return (f"⭐ Variante {evt.index + 1} choisie : « Retoucher cette image » la reprend comme photo, « Animer » "
            "l'envoie dans l'onglet Vidéos.", versions[evt.index]["fichier"])


def recreer(chemin, infos, graine, progress=gr.Progress()):
    """Galerie : même image, mêmes réglages, même graine."""
    photo = Path(chemin) / infos["photo"] if infos.get("photo") else None
    if photo is not None and not photo.exists():
        raise gr.Error(f"Photo introuvable : {photo}")
    if infos.get("scenes"):  # histoire : mêmes scènes, même graine de départ (celle de la première image)
        depart = [str(Path(chemin) / f) for f in infos.get("depart") or []] or ([str(photo)] if photo else [])
        # avec un contexte : mêmes descriptions par image ; sinon (créations d'avant) : les scènes complètes
        scenes = ([s.get("image") or "" for s in infos["scenes"]] if infos.get("contexte")
                  else [s["scene"] for s in infos["scenes"]])
        _, _, dossier = generer_histoire(scenes, depart, infos.get("styles"), infos.get("format") or AUTO,
                                         infos.get("memes_personnages", True), infos["scenes"][0]["graine"],
                                         infos.get("nom"), infos.get("description_fr"),
                                         contexte=infos.get("contexte") or "", progress=progress)
        return dossier
    photos = ([str(photo)] if photo else []) + [str(Path(chemin) / f) for f in infos.get("autres_photos") or []]
    _, _, dossier = generer(infos.get("description"), photos, infos.get("usage") or "modifier",
                            infos.get("styles"), infos.get("format") or AUTO, 1, graine, infos.get("nom"),
                            infos.get("description_fr"), progress=progress)
    return dossier


# --- Une histoire en plusieurs images ---------------------------------------------------------------------------
NOMBRE_MAX = 12
# Images 2…N : la première sert de référence (FLUX.2 klein), sinon chaque image inventait d'autres visages
# l'image 1 en référence : klein reprend parfois aussi ses poses ou son décor ; préciser « nouvelles poses, autre
# décor » ne changeait presque rien et ajoutait parfois un personnage (essais sur la RTX 4070) : ça dépend de la graine
MEMES_PERSONNAGES = ("Same characters as in the reference image(s): same faces, hair, bodies, outfits and colors, "
                     "same art style, shown in this new moment of the story")
# contexte commun de l'histoire (personnages, lieu) donné à Qwen : il est ajouté devant chaque prompt ensuite
CONTEXTE_POUR_QWEN = ("Common context, added automatically to every prompt (do not repeat it, never contradict "
                      "it: same characters, same number of people, same gender): {contexte}\n\nFor each image, "
                      "describe only what happens in this moment: action, poses, place, lighting.\n\n")
# images de départ données par l'utilisateur (personnages, créature, lieu, style) : FLUX.2 klein accepte 4
# références, la 4ᵉ est l'image 1 de l'histoire (« mêmes personnages »)
REFERENCES_MAX = 3
DEPART = ("Use the characters, creatures, places and art style shown in the reference image(s): keep their faces, "
          "features, hair, outfits and colors exactly, but NOT their pose or the framing of the reference: show them "
          "in a new scene, new poses and new camera angle matching this moment of the story")


def images_de_depart(fichiers, photo=None):
    """Images de départ de l'histoire : celles du volet (3 au plus), sinon la photo de l'onglet. Accepte des
    chemins ou des fichiers Gradio."""
    chemins = [str(getattr(f, "name", f)) for f in (fichiers or []) if f]
    if not chemins and photo:
        chemins = [str(photo)]
    if len(chemins) > REFERENCES_MAX:
        raise gr.Error(f"{len(chemins)} images de départ : {REFERENCES_MAX} au plus (le générateur en accepte 4, "
                       "la 4ᵉ place sert à l'image 1 de l'histoire).")
    for c in chemins:
        if not Path(c).is_file():
            raise gr.Error(f"Image introuvable : {c}")
    return chemins


def _planche(chemins, numeros=False):
    """Qwen ne reçoit qu'une image : plusieurs images sont posées côte à côte (hauteur 768 px) ; numeros : « 1 »,
    « 2 »… écrits en haut à gauche de chacune (photos de l'onglet, désignées par leur numéro dans le texte)."""
    if len(chemins) == 1:
        return chemins[0]
    ims = []
    for c in chemins:
        with Image.open(c) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        ims.append(im.resize((max(1, round(im.width * 768 / im.height)), 768)))
    planche = Image.new("RGB", (sum(i.width for i in ims) + 16 * (len(ims) - 1), 768), "white")
    from PIL import ImageDraw, ImageFont

    dessin, police = ImageDraw.Draw(planche), ImageFont.load_default(size=72)
    x = 0
    for k, im in enumerate(ims, 1):
        planche.paste(im, (x, 0))
        if numeros:
            dessin.rectangle((x, 0, x + 90, 100), fill="black")
            dessin.text((x + 45, 50), str(k), fill="white", font=police, anchor="mm")
        x += im.width + 16
    sortie = cfg.DATA_DIR / "_tmp" / ("planche_photos.jpg" if numeros else "planche_histoire.jpg")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    planche.save(sortie, quality=90)
    return str(sortie)


def scenes_du_texte(texte):
    """Une ligne non vide = une scène ; la numérotation « 1. » / « 2) » ou les puces de Qwen sont retirées. Scènes
    numérotées à la suite sur une même ligne (« … 2. … 3. … », constaté avec le vrai Qwen) : coupées aux numéros."""
    import re

    texte = texte or ""
    if len(re.findall(r"(?:^|\s)\d+[.)]\s", texte)) > len([l_ for l_ in texte.splitlines() if l_.strip()]):
        texte = re.sub(r"\s+(?=\d+[.)]\s)", "\n", texte)
    return [re.sub(r"^\s*(?:[-–*•]\s*)?(?:(?:image|shot|scène|scene)?\s*\d+\s*[.):\-–]\s*)?", "", ligne, flags=re.I).strip()
            for ligne in texte.splitlines() if ligne.strip()]


def decouper(histoire, nombre, photo=None, contexte="", progress=gr.Progress()):
    """Le texte → `nombre` prompts anglais (Qwen3-VL, mode « histoire »), un par ligne, modifiables. Qwen voit les
    images de départ s'il y en a (photo : un chemin ou une liste ; plusieurs = une planche) et le contexte commun
    (ajouté ensuite à chaque prompt : Qwen ne décrit que le moment de chaque image, sans le contredire)."""
    histoire = (histoire or "").strip()
    if not histoire:
        raise gr.Error("Écris d'abord l'histoire (en français ou en anglais).")
    n = max(2, min(NOMBRE_MAX, int(nombre or 4)))
    photos = [photo] if isinstance(photo, str) else list(photo or [])
    contexte = (contexte or "").strip()
    entete = f"Number of images: {n}\n\n"
    if contexte:
        entete += CONTEXTE_POUR_QWEN.format(contexte=contexte)
    texte_ = diffusion.decrire("histoire", f"{entete}Story: {histoire}",
                               image=_planche(photos) if photos else None, progress=progress, nombre=n)
    scenes = [s for s in scenes_du_texte(texte_) if len(s.split()) > 3]
    if not scenes:
        raise gr.Error("Le découpage n'a rien donné : reformule l'histoire, ou écris les scènes toi-même (une par ligne).")
    while len(scenes) < n:  # réponse trop courte : la dernière scène est reprise (modifiable avant de générer)
        scenes.append(scenes[-1])
    return "\n".join(scenes[:n])


def scenes_pretes(histoire, scenes, nombre, photo=None, contexte="", progress=gr.Progress()):
    """Avant « Générer les images » : scènes vides → découpées d'abord depuis l'histoire ; sinon gardées.
    scenes : un texte (une scène par ligne) ou la liste des champs « Image i » (alors une liste de `nombre` textes ;
    vides sans histoire mais avec un contexte : chaque image reprend le contexte seul)."""
    if isinstance(scenes, str) or scenes is None:
        return (scenes or "").strip() or decouper(histoire, nombre, photo, contexte, progress=progress)
    n = max(2, min(NOMBRE_MAX, int(nombre or 4)))
    champs = [(d or "").strip() for d in (list(scenes) + [""] * n)[:n]]
    if any(champs) or (not (histoire or "").strip() and (contexte or "").strip()):
        return champs
    return (scenes_du_texte(decouper(histoire, n, photo, contexte, progress=progress)) + [""] * n)[:n]


def champs_images(scenes, nombre=None):
    """Valeurs des NOMBRE_MAX champs « Image i » (texte d'une scène par ligne, ou liste) ; seuls les `nombre`
    premiers sont visibles."""
    liste = scenes_du_texte(scenes) if isinstance(scenes, str) else [(d or "").strip() for d in scenes or []]
    n = max(2, min(NOMBRE_MAX, int(nombre or len(liste) or 4)))
    return [gr.update(value=liste[i] if i < len(liste) else "", visible=i < n) for i in range(NOMBRE_MAX)]


def maj_champs_images(nombre):
    """Slider « Nombre d'images » : autant de champs « Image i » visibles."""
    n = max(2, min(NOMBRE_MAX, int(nombre or 4)))
    return [gr.update(visible=i < n) for i in range(NOMBRE_MAX)]


def scenes_avec_contexte(contexte, descriptions, ajout=None):
    """Prompt de chaque image : d'abord ce qu'elle montre (sa description, puis `ajout`, la consigne des images de
    référence), ensuite le contexte commun (personnages, lieu, ambiance) ; une description vide = le contexte seul.
    Dans cet ordre, un texte trop long pour le générateur perd la fin du contexte, jamais l'action de l'image
    (constaté : contexte de quatre personnages devant, description coupée, même portrait de groupe partout)."""
    contexte = (contexte or "").strip().rstrip(".")
    return [". ".join(x for x in ((d or "").strip().rstrip("."), (ajout or "").rstrip("."), contexte) if x)
            for d in descriptions]


def generer_histoire(scenes_texte, photo, styles, format_label, memes_personnages=True, graine=0, nom="",
                     histoire=None, contexte="", progress=gr.Progress()):
    """Une image par scène. scenes_texte : une scène par ligne, ou la liste des descriptions (une par image, vides
    permises avec un contexte). contexte : texte anglais ajouté à chaque scène, après sa description. photo : une image de départ, une
    liste (3 au plus : personnages, lieu, style) ou None. Sans image de départ, l'image 1 vient de Z-Image ; avec,
    toutes viennent de FLUX.2 klein et les reprennent. Images suivantes : l'image 1 en plus comme référence si
    « mêmes personnages ». Renvoie (message, galerie, dossier)."""
    contexte = (contexte or "").strip()
    if isinstance(scenes_texte, str) or scenes_texte is None:
        descriptions = scenes_du_texte(scenes_texte)
    else:
        descriptions = [(d or "").strip() for d in scenes_texte]
        if not contexte:
            vides = [str(i) for i, d in enumerate(descriptions, 1) if not d]
            if vides and len(vides) < len(descriptions):
                raise gr.Error(f"Image(s) {', '.join(vides)} sans description : décris-les, ou écris un contexte "
                               "commun (il suffit alors pour elles).")
            if vides:
                descriptions = []
    if not descriptions:
        raise gr.Error("Il n'y a aucune scène : écris un contexte et décris chaque image, ou clique sur « Découper "
                       "en scènes ».")
    if len(descriptions) > NOMBRE_MAX:
        raise gr.Error(f"{len(descriptions)} scènes : {NOMBRE_MAX} au plus.")
    scenes = scenes_avec_contexte(contexte, descriptions)
    photos = images_de_depart([photo] if isinstance(photo, str) else photo)
    debut = time.monotonic()
    format_label = format_label if format_label in FORMATS else AUTO
    largeur, hauteur = format_pour(format_label, photos[0] if photos else None)
    graine = int(graine or 0) or diffusion.graines(1)[0]
    dossier = nouveau_dossier(cfg.IMAGES_DIR)
    copies = []
    for k, ph in enumerate(photos, 1):  # copiées : la recréation ne dépend pas des fichiers d'origine
        copies.append(dossier / f"depart_{k}{Path(ph).suffix.lower() or '.png'}")
        shutil.copy(ph, copies[-1])
    n, details = len(scenes), []
    for i, scene in enumerate(scenes, 1):
        def suivi(valeur, desc="", i=i):
            progress((i - 1 + valeur) / n, desc=f"Image {i}/{n} — {desc}")

        sortie = dossier / f"scene_{i}.png"
        # l'image 1 ne sert de référence que sans images de départ : avec, les personnages viennent déjà d'elles, et
        # deux références de même pose faisaient recopier cette pose (« ils repartent à cheval » restait la pose de
        # l'illustration, constaté sur la RTX 4070)
        suite = [dossier / "scene_1.png"] if i > 1 and memes_personnages and not copies else []
        references = copies + suite
        ajout = MEMES_PERSONNAGES if suite else (DEPART if copies else None)
        final = prompt_final(scenes_avec_contexte(contexte, [descriptions[i - 1]], ajout)[0], styles)
        if references:
            res = diffusion.personnage(final, references, [sortie], [graine + i - 1], largeur, hauteur, progress=suivi)
        else:
            res = diffusion.image(final, [sortie], [graine + i - 1], largeur, hauteur, progress=suivi)
        details.append({"scene": scene, "image": descriptions[i - 1], "prompt": final, "graine": res["graines"][0],
                        "fichier": res["fichiers"][0]})
    temps = time.monotonic() - debut
    nom = "".join(c for c in (nom or "").strip() if c.isalnum() or c in "-_ ").strip() or "histoire"
    ecrire_creation(dossier, {
        "type": "image", "nom": nom, "description": " / ".join(scenes),
        "description_fr": (histoire or "").strip() or None, "styles": styles, "format": format_label,
        "largeur": largeur, "hauteur": hauteur, "photo": None, "depart": [c.name for c in copies], "usage": None,
        "memes_personnages": bool(memes_personnages), "contexte": contexte or None, "scenes": details, "temps_s": round(temps), "choisie": 1,
        "moteur": "FLUX.2 klein 4B" if copies or (memes_personnages and n > 1) else "Z-Image-Turbo",
        "versions": [{"graine": d["graine"], "dossier": ".", "fichier": d["fichier"]} for d in details],
    })
    galerie = [(d["fichier"], f"Scène {i} : {d['scene'][:80]}") for i, d in enumerate(details, 1)]
    msg = (f"✅ {n} images {largeur}×{hauteur} en {duree_lisible(temps)}, dans {dossier}. Pour les animer, règle les "
           "mini-vidéos ci-dessous et clique sur « Faire les mini-vidéos ».")
    return msg, galerie, str(dossier)
