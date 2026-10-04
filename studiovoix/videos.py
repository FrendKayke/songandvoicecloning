"""Onglet « Vidéos » : texte → vidéo et image → vidéo avec Wan 2.2 TI2V-5B (Apache 2.0 : vidéos vendables).

La description française est reformulée en anglais par Qwen3-VL (mode « video » : sujet, action, décor, lumière,
mouvement de caméra), modifiable avant génération. Avec une image de départ (illustration, photo, dernière image
d'un clip précédent), la vidéo part de cette image. Wan 2.2 est entraîné en 720p à 24 images/s ; durée = 4k + 1
images (121 = 5 s). Pas de son : les bruitages et musiques se font dans leurs onglets.
Rangement : data/videos/<horodatage>/ : video.mp4, derniere_image.png, image_depart.<ext>, creation.json (type
« video »).
"""
import shutil
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from . import config as cfg
from . import diffusion
from .outils import ecrire_creation, nouveau_dossier

AUTO = "Automatique (d'après l'image de départ, sinon paysage)"
# côtés multiples de 32 (VAE ×16 et blocs ×2 du 5B) ; 720p = résolution d'entraînement
FORMATS = {
    AUTO: None,
    "Paysage 16:9, 720p (1280×704)": (1280, 704),
    "Portrait 9:16, 720p (704×1280)": (704, 1280),
    "Carré (960×960)": (960, 960),
    "Paysage léger (832×480, plus rapide)": (832, 480),
    "Portrait léger (480×832, plus rapide)": (480, 832),
}
DUREES = {"2 s": 49, "3 s": 73, "4 s": 97, "5 s": 121}
DUREE_DEFAUT = "5 s"
FPS = 24
EXEMPLES = [
    ("Dragon qui s'envole", "un dragon rouge déploie ses ailes et s'envole d'un sommet enneigé, la caméra le suit"),
    ("Carte qui s'anime", "le personnage de l'illustration tourne lentement la tête et sourit, ses cheveux bougent "
                          "dans le vent, léger zoom avant"),
    ("Potion magique", "une fiole de potion bouillonne et laisse échapper une fumée verte lumineuse, plan fixe"),
    ("Forêt enchantée", "travelling lent dans une forêt brumeuse au lever du soleil, des lucioles flottent"),
    ("Chevalier au combat", "un chevalier en armure dorée lève son épée lumineuse face à la caméra, éclairs au loin"),
]


def format_pour(format_label, image=None):
    """(largeur, hauteur) : le format choisi, ou en automatique celui qui ressemble le plus à l'image."""
    taille = FORMATS.get(format_label)
    if taille:
        return taille
    if image:
        with Image.open(image) as im:
            rapport = im.width / im.height
        if rapport > 1.15:
            return FORMATS["Paysage 16:9, 720p (1280×704)"]
        if rapport < 0.87:
            return FORMATS["Portrait 9:16, 720p (704×1280)"]
        return FORMATS["Carré (960×960)"]
    return FORMATS["Paysage 16:9, 720p (1280×704)"]


MOTS_PROMPT_COURT = 25  # un prompt préparé par Qwen fait 50 à 120 mots ; en dessous, il est d'abord enrichi
MOTS_LIGNE_PLAN = 15  # plusieurs lignes aussi courtes dans la case du prompt : une liste de plans


def duree_lisible(secondes):
    """« 18 min 32 s », « 1 h 05 min », « 42 s »."""
    s = int(round(secondes))
    if s >= 3600:
        return f"{s // 3600} h {s % 3600 // 60:02d} min"
    return f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"


def preparer(description, image=None, progress=gr.Progress(), suite=None):
    """Description française → prompt anglais pour la vidéo (Qwen3-VL), modifiable. Avec une image de départ,
    Qwen la voit : sans elle, il inventait couleurs et lumière (« tons cramoisis » sur une scène bleue, constaté)."""
    description = (description or "").strip()
    if not description:
        raise gr.Error("Décris la vidéo (en français ou en anglais).")
    return diffusion.decrire("video", description, image=image or None, progress=progress, suite=suite)


def prompt_pret(description, prompt, image=None, progress=gr.Progress()):
    """« Générer » sans avoir préparé le prompt (constaté : erreur « Il manque le prompt ») : le prompt anglais est
    préparé d'abord depuis la description, puis affiché ; un prompt déjà là (préparé ou retouché) est gardé."""
    prompt, description = (prompt or "").strip(), (description or "").strip()
    lignes = [l_ for l_ in prompt.splitlines() if l_.strip()]
    if len(lignes) > 1 and all(len(l_.split()) <= MOTS_LIGNE_PLAN for l_ in lignes):
        # constaté : les plans tapés dans la case du prompt, envoyés tels quels à Wan → vidéo sans rapport
        raise gr.Error(f"Ton prompt contient {len(lignes)} lignes courtes : on dirait une liste de plans. Pour les "
                       "enchaîner, ouvre « 🎞️ Plusieurs plans à la suite » plus bas et écris-y une ligne par plan. "
                       "Pour une seule vidéo, écris une seule description.")
    if prompt and not description and len(prompt.split()) < MOTS_PROMPT_COURT:
        # prompt court tapé à la main (« aventurers are fighting ») : Wan a besoin d'une description détaillée en
        # bon anglais ; Qwen la rédige (et corrige les fautes), la case affiche ce qui est vraiment envoyé
        return preparer(prompt, image, progress=progress)
    return prompt or preparer(description, image, progress=progress)


def generer(prompt, image, format_label, duree_label, etapes, graine, nom="", description_fr=None,
            progress=gr.Progress()):
    """Génère la vidéo. Renvoie (message, vidéo, dossier)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Il manque le prompt : clique d'abord sur « Préparer le prompt », ou écris-le en anglais.")
    debut = time.monotonic()
    largeur, hauteur = format_pour(format_label, image)
    images = DUREES.get(duree_label, DUREES[DUREE_DEFAUT])
    graine = int(graine or 0) or diffusion.graines(1)[0]
    dossier = nouveau_dossier(cfg.VIDEOS_DIR)
    depart = None
    if image:
        depart = dossier / f"image_depart{Path(image).suffix.lower() or '.png'}"
        shutil.copy(image, depart)
    res = diffusion.video(prompt, dossier / "video.mp4", depart, largeur, hauteur, images, etapes, graine,
                          progress=progress)
    nom = (nom or "").strip() or "video"
    temps = time.monotonic() - debut
    ecrire_creation(dossier, {
        "type": "video", "nom": nom, "description": prompt, "temps_s": round(temps), "description_fr": (description_fr or "").strip() or None,
        "image_depart": depart.name if depart else None, "format": format_label, "largeur": largeur,
        "hauteur": hauteur, "duree": duree_label, "images": images, "etapes": int(etapes), "fps": FPS,
        "versions": [{"graine": res["graine"], "dossier": ".", "fichier": res["sortie"]}],
    })
    mode = "à partir de l'image" if depart else "à partir du texte"
    msg = (f"✅ Vidéo {largeur}×{hauteur} de {res['duree']} s {mode}, graine {res['graine']}, en "
           f"{duree_lisible(temps)}, dans {dossier}. Pas de son : ajoute un bruitage ou une musique depuis leurs onglets.")
    return msg, res["sortie"], str(dossier)


PLANS_MAX = 12  # ~15 à 25 minutes par plan de 5 s en 720p sur une RTX 4070 : déjà plusieurs heures


def plans_du_texte(texte):
    """Une ligne non vide = un plan."""
    return [ligne.strip() for ligne in (texte or "").splitlines() if ligne.strip()]


def generer_suite(plans_texte, image, format_label, duree_label, etapes, graine, nom="", prompts=None,
                  progress=gr.Progress()):
    """Plusieurs plans enchaînés : la dernière image de chaque clip est l'image de départ du suivant (comme le bouton
    « Continuer », mais d'un coup), puis les clips sont assemblés en une seule vidéo. Les prompts sont tous
    préparés d'abord (Qwen3-VL), puis les clips générés (Wan 2.2) : un seul changement de modèle.
    prompts : prompts anglais déjà prêts (recréation depuis la galerie). Renvoie (message, vidéo, dossier)."""
    plans = plans_du_texte(plans_texte) if prompts is None else [None] * len(prompts)
    if not plans:
        raise gr.Error("Écris au moins un plan : une ligne par plan (par exemple « le dragon décolle », puis "
                       "« il survole la forêt »).")
    if len(plans) > PLANS_MAX:
        raise gr.Error(f"{len(plans)} plans : {PLANS_MAX} au plus (chaque plan de 5 s prend 15 à 25 minutes).")
    n = len(plans)
    debut = time.monotonic()
    if prompts is None:
        prompts = []
        for i, plan in enumerate(plans, 1):
            progress(0.05 * (i - 1) / n, desc=f"Plan {i}/{n} : préparation du prompt…")
            # le premier plan part de l'image de départ : Qwen la voit (les suivants partent du clip précédent) ;
            # chaque plan est écrit avec toute l'histoire et le prompt du précédent (mêmes personnages, même lieu)
            suite = {"plans": plans, "indice": i, "precedent": prompts[-1] if prompts else None} if n > 1 else None
            prompts.append(preparer(plan, image if i == 1 else None, progress=lambda *a, **k: None, suite=suite))
    largeur, hauteur = format_pour(format_label, image)
    images = DUREES.get(duree_label, DUREES[DUREE_DEFAUT])
    graine = int(graine or 0) or diffusion.graines(1)[0]
    dossier = nouveau_dossier(cfg.VIDEOS_DIR)
    depart = None
    if image:
        depart = dossier / f"image_depart{Path(image).suffix.lower() or '.png'}"
        shutil.copy(image, depart)
    image_depart = depart.name if depart else None
    clips, details, erreur = [], [], None
    for i, prompt in enumerate(prompts, 1):
        sous = dossier / f"plan_{i}"
        sous.mkdir()

        def suivi(valeur, desc="", i=i):  # progression du plan i dans celle de toute la suite
            progress(0.05 + 0.9 * ((i - 1) + valeur) / n, desc=f"Plan {i}/{n} — {desc}")

        try:
            res = diffusion.video(prompt, sous / "video.mp4", depart, largeur, hauteur, images, etapes, graine + i - 1,
                                  progress=suivi)
        except gr.Error as e:  # on garde les plans réussis
            erreur = f"plan {i} : {getattr(e, 'message', e)}"
            break
        clips.append(res["sortie"])
        details.append({"plan": plans[i - 1], "prompt": prompt, "graine": res["graine"], "fichier": res["sortie"]})
        depart = Path(res["derniere_image"])
    if not clips:
        raise gr.Error(f"Aucun plan n'a pu être généré ({erreur}).")
    progress(0.97, desc="Assemblage des plans…")
    sortie = dossier / "video.mp4"
    assemblee = diffusion.assembler(clips, sortie, FPS)
    shutil.copy(depart, dossier / "derniere_image.png")  # « Continuer » repart de la fin de la suite
    nom = (nom or "").strip() or "video"
    temps = time.monotonic() - debut
    ecrire_creation(dossier, {
        "type": "video", "nom": nom, "description": " / ".join(prompts[:len(clips)]), "temps_s": round(temps),
        "description_fr": " / ".join(p for p in plans[:len(clips)] if p) or None,
        "image_depart": image_depart,
        "format": format_label, "largeur": largeur, "hauteur": hauteur, "duree": duree_label, "images": images,
        "etapes": int(etapes), "fps": FPS, "plans": details,
        "versions": [{"graine": graine, "dossier": ".", "fichier": str(sortie)}],
    })
    msg = (f"✅ {len(clips)} plan(s) enchaînés : vidéo {largeur}×{hauteur} de {assemblee['duree']} s, en "
           f"{duree_lisible(temps)}, dans {dossier}. "
           "Chaque plan part de la dernière image du précédent.")
    if erreur:
        msg += f"\n\n⚠️ Arrêt au {erreur} ; les plans réussis sont assemblés."
    return msg, str(sortie), str(dossier)


def continuer(dossier):
    """La dernière image de la vidéo devient l'image de départ du clip suivant."""
    derniere = Path(dossier or "") / "derniere_image.png"
    if not dossier or not derniere.exists():
        raise gr.Error("Génère d'abord une vidéo.")
    return str(derniere)


def recreer(chemin, infos, graine, progress=gr.Progress()):
    """Galerie : même vidéo, mêmes réglages, même graine (résultat proche) ; une suite de plans est refaite avec
    les mêmes prompts."""
    depart = Path(chemin) / infos["image_depart"] if infos.get("image_depart") else None
    if depart is not None and not depart.exists():
        raise gr.Error(f"Image de départ introuvable : {depart}")
    if infos.get("plans"):
        _, _, dossier = generer_suite(None, str(depart) if depart else None, infos.get("format") or AUTO,
                                      infos.get("duree") or DUREE_DEFAUT, infos.get("etapes", 30), graine,
                                      infos.get("nom"), prompts=[p["prompt"] for p in infos["plans"]],
                                      progress=progress)
        return dossier
    _, _, dossier = generer(infos.get("description"), str(depart) if depart else None, infos.get("format") or AUTO,
                            infos.get("duree") or DUREE_DEFAUT, infos.get("etapes", 30), graine, infos.get("nom"),
                            infos.get("description_fr"), progress=progress)
    return dossier
