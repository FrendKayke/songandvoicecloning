"""Galerie : toutes les créations (chansons, pistes de jeu, lectures, bruitages, modèles 3D, illustrations, photos,
vidéos) à réécouter, recréer avec la même
graine, retoucher (« Refaire un passage », tâche repaint d'ACE-Step) ou supprimer.

Chaque création est un dossier décrit par creation.json (outils.ecrire_creation). Les chansons et lectures
plus anciennes, sans creation.json, sont listées d'après prompt.txt / tache.json (écoute et suppression seulement).
"""
import json
import shutil
from pathlib import Path

import gradio as gr
import soundfile as sf

from . import acestep, chatterbox, jeu
from . import config as cfg
from .outils import ecrire_creation, nouveau_dossier
from .pipeline import INSTRUMENTAL, finaliser_depuis_infos

TYPES = {"chanson": "🎵 Chanson", "jeu": "🎮 Bande-son", "tts": "🗣️ Lecture", "bruitage": "🔊 Bruitage",
         "3d": "🧊 Modèle 3D", "carte": "🃏 Illustration", "photo": "🖼️ Photo",
         "video": "🎬 Vidéo", "image": "🎨 Image"}
FILTRES = {"Tout": None, "Chansons": "chanson", "Bande-son de jeu": "jeu", "Synthèse vocale": "tts",
           "Bruitages": "bruitage", "Modèles 3D": "3d", "Illustrations": "carte", "Photos": "photo",
           "Vidéos": "video", "Images": "image"}
# repaint_mode d'ACE-Step (release_task_models.py : conservative / balanced / aggressive)
FORCES = {
    "Légère (garde au maximum l'original)": "conservative",
    "Équilibrée": "balanced",
    "Complète (réinvente le passage)": "aggressive",
}


def _racines():
    return {"chanson": cfg.SONGS_DIR, "jeu": cfg.GAMES_DIR, "tts": cfg.TTS_DIR, "bruitage": cfg.SFX_DIR,
            "3d": cfg.MODELS3D_DIR, "carte": cfg.CARDS_DIR, "photo": cfg.PHOTOS_DIR,
            "video": cfg.VIDEOS_DIR, "image": cfg.IMAGES_DIR}


def lire(dossier):
    """Description d'une création (creation.json, ou reconstruite pour une création ancienne), sinon None."""
    d = Path(dossier)
    f = d / "creation.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            return None
    if (d / "prompt.txt").exists():  # chanson d'avant creation.json
        prompt, _, paroles = (d / "prompt.txt").read_text(encoding="utf-8").partition("\n\n")
        fichier = next((d / n for n in ("chanson_finale.wav", "chanson_brute.wav") if (d / n).exists()), None)
        return {"type": "chanson", "ancienne": True, "description": prompt.strip(), "paroles": paroles.strip(),
                "versions": [{"graine": None, "dossier": ".", "fichier": str(fichier) if fichier else None}]}
    if (d / "tache.json").exists() and (d / "parole.wav").exists():  # lecture d'avant creation.json
        t = json.loads((d / "tache.json").read_text(encoding="utf-8"))
        return {"type": "tts", "ancienne": True, "texte": t.get("texte", ""),
                "versions": [{"graine": t.get("graine"), "dossier": ".", "fichier": str(d / "parole.wav")}]}
    return None


def _dossiers():
    r = _racines()
    yield from (d for d in r["chanson"].glob("*") if d.is_dir())
    yield from (d for d in r["jeu"].glob("*/*/*") if d.is_dir() and "export" not in d.parts[-3:])
    yield from (d for d in r["tts"].glob("*") if d.is_dir())
    yield from (d for d in r["bruitage"].glob("*") if d.is_dir())
    yield from (d for d in r["3d"].glob("*") if d.is_dir())
    yield from (d for d in r["carte"].glob("*/*") if d.is_dir())
    yield from (d for d in r["photo"].glob("*") if d.is_dir())
    yield from (d for d in r["video"].glob("*") if d.is_dir())
    yield from (d for d in r["image"].glob("*") if d.is_dir())


_ACTIONS_PHOTO = {"ameliorer": "améliorée", "detourer": "détourée", "personne": "personne isolée"}


def _resume(infos):
    if infos["type"] == "jeu":
        return f"{infos.get('projet')} — {infos.get('libelle')}"
    if infos["type"] == "carte":
        return f"{infos.get('projet')} — {infos.get('nom')} — {infos.get('description_fr') or infos.get('description')}"
    if infos["type"] == "photo":
        return f"{infos.get('nom')} — {_ACTIONS_PHOTO.get(infos.get('action'), '')}"
    if infos["type"] in ("bruitage", "3d", "video", "image"):
        return f"{infos.get('nom')} — {infos.get('description_fr') or infos.get('description') or 'depuis une image'}"
    txt = infos.get("texte") if infos["type"] == "tts" else infos.get("description")
    txt = (txt or "").replace("\n", " ")
    return txt[:60] + ("…" if len(txt) > 60 else "")


def _date(dossier):
    n = Path(dossier).name
    return f"{n[6:8]}/{n[4:6]}/{n[:4]} {n[9:11]}h{n[11:13]}" if len(n) >= 15 and n[:8].isdigit() else n


def lister(filtre="Tout"):
    """Créations les plus récentes d'abord : [(libellé, dossier)]."""
    type_voulu = FILTRES.get(filtre)
    elements = []
    for d in _dossiers():
        infos = lire(d)
        if not infos or (type_voulu and infos.get("type") != type_voulu):
            continue
        elements.append((d.name, f"{_date(d)} · {TYPES.get(infos['type'], '?')} · {_resume(infos)}", str(d)))
    return [(lib, chemin) for _, lib, chemin in sorted(elements, reverse=True)]


def maj_liste(filtre, choisie=None):
    choix = lister(filtre)
    valeurs = [v for _, v in choix]
    return gr.update(choices=choix, value=choisie if choisie in valeurs else (valeurs[0] if valeurs else None))


def _infos(chemin):
    if not chemin or not Path(chemin).is_dir():
        raise gr.Error("Choisis une création dans la liste.")
    infos = lire(chemin)
    if not infos:
        raise gr.Error("Cette création n'a plus de description (creation.json).")
    return infos


def _version(infos, version):
    versions = infos.get("versions") or [{}]
    i = max(1, min(len(versions), int(version or 1))) - 1
    return i, versions[i]


def details(chemin, version=1):
    """(description en Markdown, fichier audio de la version, choix des versions, description, paroles,
    fin conseillée, modèle 3D à afficher, image à afficher, vidéo à afficher)."""
    pas_de_3d = gr.update(value=None, visible=False)
    pas_d_image = gr.update(value=None, visible=False)
    pas_de_video = gr.update(value=None, visible=False)
    if not chemin:
        return ("*Aucune création pour l'instant.*", None, gr.update(choices=[1], value=1, visible=False), "", "", 10,
                pas_de_3d, pas_d_image, pas_de_video)
    infos = _infos(chemin)
    i, v = _version(infos, version)
    fichier = v.get("fichier")
    fichier = fichier if fichier and Path(fichier).exists() else None
    lignes = [f"**{TYPES.get(infos['type'], '?')}** — {_date(chemin)}", f"Dossier : `{chemin}`"]
    if infos["type"] == "jeu":
        lignes.append(f"Projet **{infos.get('projet')}**, situation **{infos.get('libelle')}** "
                      f"({'boucle' if infos.get('boucle') else 'jingle'})")
    if infos.get("description"):
        lignes.append(f"Description : {infos['description']}")
    if infos["type"] == "tts":
        lignes.append(f"Texte : {infos.get('texte', '')[:500]}")
    if infos["type"] == "bruitage" and infos.get("description_fr"):
        lignes.append(f"Demande : {infos['description_fr']} ({infos.get('duree')} s)")
    if infos["type"] == "carte":
        lignes.append(f"Projet **{infos.get('projet')}**, carte **{infos.get('nom')}**, {infos.get('largeur')}×"
                      f"{infos.get('hauteur')}" + (f", personnage **{infos['personnage']}**" if infos.get("personnage") else "")
                      + (", d'après une photo modèle" if infos.get("photo_modele") else "")
                      + (f", demande : {infos['description_fr']}" if infos.get("description_fr") else ""))
    if infos["type"] == "photo":
        r = infos.get("reglages") or {}
        detail = (f"agrandissement ×{r.get('echelle')}" + (", visages restaurés" if r.get("visages") else "")
                  if infos.get("action") == "ameliorer" else f"fond : {r.get('fond_label') or 'transparent'}")
        lignes.append(f"Photo **{infos.get('nom')}** {_ACTIONS_PHOTO.get(infos.get('action'), '')} ({detail})")
    if infos["type"] == "image":
        origine = {"modifier": "photo modifiée", "sujet": "sujet d'une photo", "composition": "composition d'une photo"}
        lignes.append(f"Image **{infos.get('nom')}** : {infos.get('largeur')}×{infos.get('hauteur')}, "
                      f"{origine.get(infos.get('usage'), 'à partir du texte')}, {infos.get('moteur')}"
                      + (f", demande : {infos['description_fr']}" if infos.get("description_fr") else ""))
    if infos["type"] == "video":
        lignes.append(f"Vidéo **{infos.get('nom')}** : {infos.get('largeur')}×{infos.get('hauteur')}, "
                      f"{infos.get('duree')}, {infos.get('etapes')} étapes, "
                      + ("à partir d'une image" if infos.get("image_depart") else "à partir du texte")
                      + (f", demande : {infos['description_fr']}" if infos.get("description_fr") else ""))
    if infos["type"] == "3d":
        texture = "texturé" if v.get("fichier") and v.get("fichier") != v.get("forme") else "forme seule"
        lignes.append(f"Modèle **{infos.get('nom')}** : qualité {infos.get('qualite')}, {infos.get('faces_obtenues')} "
                      f"faces, {texture}" + (f", demande : {infos['description_fr']}" if infos.get("description_fr") else ""))
    elif infos.get("paroles") and infos["paroles"] != INSTRUMENTAL:
        lignes.append("Paroles :\n\n```\n" + infos["paroles"][:1500] + "\n```")
    graines = ", ".join(str(x.get("graine")) for x in infos.get("versions") or [] if x.get("graine") is not None)
    if graines:
        lignes.append(f"Graine(s) : {graines}")
    if infos.get("retouche_de"):
        lignes.append(f"Retouche de `{infos['retouche_de']}` (passage {infos['passage'][0]}–{infos['passage'][1]} s)")
    if infos.get("ancienne"):
        lignes.append("*Création d'une ancienne version de Studio Voix : écoute et suppression seulement.*")
    n = len(infos.get("versions") or [])
    versions = gr.update(choices=list(range(1, n + 1)), value=i + 1, visible=n > 1)
    if infos["type"] == "3d":
        return ("\n\n".join(lignes), None, versions, infos.get("description") or "", "", 10,
                gr.update(value=fichier, visible=True), pas_d_image, pas_de_video)
    if infos["type"] in ("carte", "photo", "image"):
        return ("\n\n".join(lignes), None, versions, infos.get("description") or "", "", 10, pas_de_3d,
                gr.update(value=fichier, visible=True), pas_de_video)
    if infos["type"] == "video":
        return ("\n\n".join(lignes), None, versions, infos.get("description") or "", "", 10, pas_de_3d,
                pas_d_image, gr.update(value=fichier, visible=True))
    duree = round(sf.info(fichier).duration, 1) if fichier else 10
    return ("\n\n".join(lignes), fichier, versions, infos.get("description") or "", infos.get("paroles") or "", duree,
            pas_de_3d, pas_d_image, pas_de_video)


def supprimer(chemin):
    """Supprime une création (après confirmation dans le navigateur ; None = annulé)."""
    if chemin is None:
        return "Suppression annulée."
    d = Path(chemin).resolve()
    if not any(r.resolve() in d.parents for r in _racines().values()) or not lire(d):
        raise gr.Error("Ce dossier n'est pas une création de Studio Voix.")
    try:
        shutil.rmtree(d)
    except OSError as e:
        raise gr.Error(f"Impossible de supprimer : un fichier est peut-être en cours de lecture. ({e})")
    return f"🗑️ Création supprimée : {d.name}"


def _dossier_de(fichier):
    """Dossier de création contenant un fichier produit (remonte jusqu'au creation.json)."""
    for p in Path(fichier).parents:
        if (p / "creation.json").exists():
            return str(p)
    return str(Path(fichier).parent)


def _fichiers_jeu(infos):
    ref = infos.get("reference")
    if not ref or not Path(ref).exists():
        return None, {}
    if infos.get("usage_reference") == "variation":
        return {"src_audio": ref}, {"task_type": "cover", "audio_cover_strength": infos.get("fidelite") or 0.5,
                                    "thinking": False}
    return {"reference_audio": ref}, {}


def recreer(chemin, version=1, progress=gr.Progress()):
    """Même création, même graine (résultat proche). Renvoie (message, dossier de la nouvelle création)."""
    infos = _infos(chemin)
    if infos.get("ancienne"):
        raise gr.Error("Création d'une ancienne version : pas assez d'informations pour la recréer.")
    _, v = _version(infos, version)
    graine = v.get("graine") or 0
    if infos["type"] == "chanson":
        from .pipeline import creer_chanson

        seedvc_infos, gains = infos.get("seedvc") or {}, infos.get("gains") or {}
        res = creer_chanson(
            infos.get("voix"), "", "", "", "", "", "Automatique", infos.get("paroles"), infos.get("langue"),
            infos.get("duree"), infos.get("bpm"), infos.get("reflexion"), seedvc_infos.get("demi_tons", 0),
            seedvc_infos.get("etapes", 40), gains.get("voix", 1.0), gains.get("instrumental", 1.0),
            infos.get("mode"), infos.get("description"), infos.get("retirer"), 1, graine,
            infos.get("conversion") or "seedvc", progress=progress)
        return f"✅ Recréée avec la graine {graine}.", _dossier_de(res[0])
    if infos["type"] == "jeu":
        duree = infos.get("duree") if infos.get("boucle") else max(jeu.DUREE_MIN_ACESTEP, infos.get("duree_cible") or 0)
        params = acestep.text2music_params(infos["description"], INSTRUMENTAL, "en", duree, 0, infos.get("reflexion"))
        fichiers, extra = _fichiers_jeu(infos)
        params.update(extra)
        garde = {k: infos.get(k) for k in ("projet", "situation", "libelle", "boucle", "duree_cible", "reference",
                                           "usage_reference", "fidelite")}
        piste, note = jeu.generer_piste(params, garde, progress, "Recréation", fichiers, graine)
        return f"✅ Recréée avec la graine {graine} : {note}.", _dossier_de(piste)
    if infos["type"] == "bruitage":
        from . import bruitages

        msg, _, fichier, dossier = bruitages.generer(infos["description"], infos.get("nom"), infos.get("duree", 3),
                                                     1, graine, infos.get("etapes", 100), None,
                                                     infos.get("description_fr"), infos.get("projet"),
                                                     progress=progress)
        return f"✅ Bruitage recréé avec la graine {graine}.", dossier
    if infos["type"] == "3d":
        from . import modele3d

        image = Path(chemin) / (infos.get("image") or "image.png")
        if not image.exists():
            raise gr.Error(f"Image de départ introuvable : {image}")
        *_, dossier = modele3d.generer(str(image), infos.get("nom"), infos.get("qualite"), infos.get("texture", True),
                                       graine, infos.get("formats"), infos.get("description"),
                                       infos.get("description_fr"), infos.get("image_graine"), infos.get("projet"),
                                       progress=progress)
        return f"✅ Modèle 3D recréé avec la graine {graine}.", dossier
    if infos["type"] == "carte":
        from . import cartes

        _, _, dossier, _ = cartes.generer(infos.get("projet"), infos.get("nom"), infos.get("description"),
                                          infos.get("styles"), infos.get("consignes"), infos.get("format"), 1, graine,
                                          infos.get("webp", True), infos.get("description_fr"),
                                          personnage=infos.get("personnage"),
                                          photo=str(Path(chemin) / infos["photo_modele"]) if infos.get("photo_modele") else None,
                                          usage_photo=infos.get("usage_photo") or "sujet",
                                          image_1=cartes.description_de_version(infos, graine), progress=progress)
        return f"✅ Illustration recréée avec la graine {graine}.", dossier
    if infos["type"] == "image":
        from . import images

        return f"✅ Image recréée avec la graine {graine}.", images.recreer(chemin, infos, graine, progress=progress)
    if infos["type"] == "video":
        from . import videos

        return f"✅ Vidéo recréée avec la graine {graine}.", videos.recreer(chemin, infos, graine, progress=progress)
    if infos["type"] == "photo":
        from . import photos

        return "✅ Photo traitée à nouveau.", photos.recreer(chemin, infos, progress=progress)
    reg = infos.get("reglages") or {}
    fichier, _ = chatterbox.synthese(infos.get("voix"), infos.get("texte"), infos.get("langue"),
                                     reg.get("exaggeration", 0.5), reg.get("cfg_weight", 0.5),
                                     reg.get("temperature", 0.8), graine, progress=progress)
    note = "" if graine else " (graine aléatoire à l'origine : le résultat sera différent)"
    return f"✅ Lecture recréée{note}.", _dossier_de(fichier)


def refaire_passage(chemin, version, debut, fin, description, paroles, force_label, progress=gr.Progress()):
    """Redessine seulement [debut, fin] (secondes) d'une chanson ou d'une piste de jeu, puis refait le reste du
    traitement (séparation, conversion, mixage, ou boucle / jingle). Nouvelle création, l'originale est gardée."""
    infos = _infos(chemin)
    if infos["type"] not in ("chanson", "jeu") or infos.get("ancienne"):
        raise gr.Error("« Refaire un passage » marche sur les chansons et les pistes de jeu créées avec cette version.")
    i, v = _version(infos, version)
    d_version = Path(chemin) / (v.get("dossier") or ".")
    brute = d_version / ("brute.wav" if infos["type"] == "jeu" else "chanson_brute.wav")
    if not brute.exists():
        raise gr.Error(f"Fichier d'origine introuvable : {brute}")
    duree = sf.info(str(brute)).duration
    debut, fin = float(debut or 0), float(fin or 0)
    if not (0 <= debut < fin <= duree + 0.01) or fin - debut < 1:
        raise gr.Error(f"Choisis un passage d'au moins 1 s entre 0 et {duree:.1f} s (début < fin).")
    paroles = INSTRUMENTAL if infos["type"] == "jeu" else (acestep.baliser_paroles((paroles or "").strip()).strip()
                                                           or infos.get("paroles") or INSTRUMENTAL)
    params = {
        "task_type": "repaint", "prompt": (description or "").strip() or infos.get("description", ""),
        "lyrics": paroles, "vocal_language": cfg.LANGUES.get(infos.get("langue"), "en"),
        "repainting_start": debut, "repainting_end": min(fin, duree), "repaint_mode": FORCES.get(force_label, "balanced"),
        "thinking": False,
    }
    retouche = {"retouche_de": str(chemin), "passage": [debut, min(fin, duree)]}
    if infos["type"] == "jeu":
        garde = {k: infos.get(k) for k in ("projet", "situation", "libelle", "boucle", "duree_cible", "reference",
                                           "usage_reference", "fidelite")}
        piste, note = jeu.generer_piste(params, {**garde, **retouche}, progress, "Retouche", {"src_audio": str(brute)})
        return f"✅ Passage refait : {note}.", _dossier_de(piste)
    workdir = nouveau_dossier(cfg.SONGS_DIR)
    ((song, seed),) = acestep.generer(params, [workdir / "chanson_brute.wav"], progress, "Retouche",
                                      fichiers={"src_audio": str(brute)})
    final, *_ = finaliser_depuis_infos(song, workdir, {**infos, "paroles": paroles}, progress)
    ecrire_creation(workdir, {**{k: val for k, val in infos.items() if k not in ("date", "versions")},
                              "description": params["prompt"], "paroles": paroles, **retouche,
                              "versions": [{"graine": seed, "dossier": ".", "fichier": str(final)}]})
    return f"✅ Passage {debut:.1f}–{fin:.1f} s refait.", str(workdir)
