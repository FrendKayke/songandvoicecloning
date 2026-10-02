"""Espace disque (Outils → Modèles → « Espace disque ») : place prise par chaque moteur, modèle et dossier de
créations, et retrait des moteurs ou modèles inutilisés pour gagner de la place.

Retirer = supprimer les fichiers ET noter la clé dans moteurs-retires.txt (retraits.py), sinon l'installateur les
réinstallerait à la prochaine mise à jour. « Réinstaller » enlève la clé ; METTRE_A_JOUR.bat (ou INSTALLER.bat)
refait alors les étapes, puisque leurs marqueurs ont disparu avec les fichiers (ou que la liste de modèles de
l'étape 17 a changé de signature).
Jamais supprimés ici : ACE-Step, Seed-VC/Demucs, l'application, tes créations, tes modèles RVC entraînés (logs).
"""
import os
import shutil
import stat
from pathlib import Path

import gradio as gr

from . import chatterbox, diffusion, retraits
from . import config as cfg

HUNYUAN_PARTIES = {"forme3d": ["hunyuan3d-dit-v2-0-turbo", "hunyuan3d-vae-v2-0-turbo", "hunyuan3d-dit-v2-0"],
                   "texture3d": ["hunyuan3d-paint-v2-0-turbo", "hunyuan3d-delight-v2-0"]}
LIBELLES = {
    "chatterbox": "Synthèse vocale (Chatterbox) : environnement et modèles",
    "nettoyage": "Nettoyage de voix : environnement et modèles",
    "rvc": "RVC : environnement et modèles de base (tes modèles entraînés sont gardés)",
    "diffusion": "Moteur de diffusion entier (bruitages, illustrations, personnages, photos, vidéos, 3D) : environnement et tous ses modèles",
    "diffusion:qwen": "Qwen3-VL (préparation des prompts)",
    "diffusion:bruitages": "Stable Audio Open (bruitages)",
    "diffusion:zimage": "Z-Image-Turbo (illustrations, texte → 3D ; son encodeur de texte sert aussi à FLUX.2 klein)",
    "diffusion:personnages": "FLUX.2 klein 4B (personnages récurrents)",
    "diffusion:video": "Wan 2.2 TI2V-5B (vidéo)",
    "diffusion:photo_detourage": "BiRefNet (détourage des photos)",
    "diffusion:photo_qualite": "Real-ESRGAN et GFPGAN (qualité des photos)",
    "diffusion:forme3d": "Hunyuan3D-2 : forme",
    "diffusion:texture3d": "Hunyuan3D-2 : texture (le plus gros)",
}


def _depot(depot):
    return diffusion._dossier_depot(depot)


def chemins(cle):
    """Fichiers et dossiers supprimés quand on retire la clé."""
    if cle == "chatterbox":
        return [cfg.CHATTERBOX_DIR, chatterbox.ckpt_dir()]
    if cle == "nettoyage":
        return [cfg.NETTOYAGE_DIR]
    if cle == "rvc":
        m = cfg.RVC_DIR / "rvc" / "models"
        return ([cfg.RVC_DIR / ".venv", cfg.RVC_DIR / ".env-ok", cfg.RVC_DIR / ".modeles-ok",
                 m / "pretraineds" / "hifi-gan", m / "embedders" / "contentvec"]
                + sorted((m / "predictors").glob("*.pt")))
    if cle == "diffusion":
        return [cfg.DIFFUSION_DIR] + [p for n in diffusion.MODELES for p in chemins(f"diffusion:{n}")]
    if cle.startswith("diffusion:"):
        nom = cle.split(":", 1)[1]
        if nom in HUNYUAN_PARTIES:  # même dépôt pour la forme et la texture : on ne retire que ses sous-dossiers
            snaps = _depot(diffusion.MODELES[nom][0]) / "snapshots"
            return [d for s in sorted(snaps.glob("*")) for d in (s / p for p in HUNYUAN_PARTIES[nom])]
        depots = [diffusion.MODELES[nom][0]] + [f[0] for f in diffusion.MODELES[nom][2] if isinstance(f, tuple)]
        return [_depot(d) for d in dict.fromkeys(depots)]
    raise gr.Error(f"Élément inconnu : {cle}")


def taille(chemin):
    """Octets occupés (les liens du cache Hugging Face comptent pour la taille du fichier visé, une seule fois)."""
    p = Path(chemin)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total, vus = 0, set()
    for racine, _, fichiers in os.walk(p):
        for f in fichiers:
            chemin_f = Path(racine) / f
            try:
                cible = chemin_f.resolve() if chemin_f.is_symlink() else chemin_f
                if cible in vus:
                    continue
                vus.add(cible)
                total += cible.stat().st_size
            except OSError:
                pass
    return total


def _go(octets):
    return f"{octets / 1e9:.1f} Go" if octets >= 1e8 else f"{octets / 1e6:.0f} Mo"


def _supprimer(chemin):
    """Supprime un fichier ou un dossier ; pour les liens du cache Hugging Face, supprime aussi le fichier visé."""
    p = Path(chemin)
    if p.is_symlink():
        cible = p.resolve()
        p.unlink()
        if cible.exists() and cible.is_file():
            cible.unlink()
        return
    if p.is_file():
        p.unlink()
        return
    if p.is_dir():
        for racine, _, fichiers in os.walk(p):
            for f in fichiers:
                lien = Path(racine) / f
                if lien.is_symlink() and lien.resolve().is_file():
                    lien.resolve().unlink()

        def lecture_seule(fonction, chemin_, _):  # Windows : fichiers en lecture seule (git, caches)
            os.chmod(chemin_, stat.S_IWRITE)
            fonction(chemin_)
        shutil.rmtree(p, onerror=lecture_seule)


def inventaire():
    """Tableau Markdown : ce qui occupe le disque, retirable ou non."""
    lignes = ["| Élément | Taille | État |", "|---|---|---|"]
    total = 0

    def ligne(libelle, octets, etat):
        nonlocal total
        total += octets
        lignes.append(f"| {libelle} | {_go(octets)} | {etat} |")

    ligne("ACE-Step (génération musicale)", taille(cfg.ACESTEP_DIR), "indispensable")
    ligne("Seed-VC et Demucs (ta voix dans les chansons)", taille(cfg.SEEDVC_DIR) + taille(
        Path(os.environ.get("TORCH_HOME") or cfg.ENG_DIR / "torch-cache")), "indispensable")
    for cle in ("chatterbox", "nettoyage", "rvc", "diffusion"):
        if cle == "diffusion":
            ligne("Moteur de diffusion : environnement", taille(cfg.DIFFUSION_DIR),
                  "retiré" if retraits.retire("diffusion") else "retirable")
            for nom in diffusion.MODELES:
                sous = f"diffusion:{nom}"
                ligne(f"&nbsp;&nbsp;· {LIBELLES[sous]}", sum(taille(c) for c in chemins(sous)),
                      "retiré" if retraits.retire(sous) else "retirable")
        else:
            ligne(LIBELLES[cle], sum(taille(c) for c in chemins(cle)), "retiré" if retraits.retire(cle) else "retirable")
    ligne("Tes modèles RVC entraînés", taille(cfg.RVC_DIR / "logs"), "gardés")
    ligne("Cache de téléchargement (uv)", taille(cfg.ENG_DIR / "uv-cache"), "à vider sans risque")
    ligne("Fichiers d'essai du diagnostic", taille(cfg.DATA_DIR / "diagnostic"), "à vider sans risque")
    ligne("Tes créations (voix, chansons, jeux, bruitages, 3D, illustrations, photos, vidéos)",
          sum(taille(d) for d in (cfg.VOICES_DIR, cfg.SONGS_DIR, cfg.TTS_DIR, cfg.CLEAN_DIR, cfg.GAMES_DIR,
                                  cfg.SFX_DIR, cfg.MODELS3D_DIR, cfg.CARDS_DIR, cfg.PHOTOS_DIR,
                                  cfg.VIDEOS_DIR)), "gardées")
    try:
        libre = shutil.disk_usage(cfg.ENG_DIR if cfg.ENG_DIR.exists() else cfg.APP_DIR).free
        pied = f"\n\n**Total : {_go(total)}** · espace libre sur le disque : {_go(libre)}"
    except OSError:
        pied = f"\n\n**Total : {_go(total)}**"
    return "\n".join(lignes) + pied


# Choix proposés dans l'interface : (libellé, clé)
CHOIX_RETRAIT = [(LIBELLES[c], c) for c in ("diffusion:video", "diffusion:texture3d", "diffusion:zimage", "diffusion:personnages",
                                             "diffusion:bruitages", "diffusion:photo_detourage",
                                             "diffusion:photo_qualite",
                                             "diffusion:qwen", "diffusion:forme3d", "diffusion", "rvc", "chatterbox",
                                             "nettoyage")]
CHOIX_VIDER = [("Cache de téléchargement (uv) — retéléchargé si besoin", "uv-cache"),
               ("Fichiers d'essai du diagnostic", "diagnostic")]


def retirer(cle):
    """Supprime les fichiers de la clé et la note comme retirée. Renvoie (message, inventaire)."""
    if cle is None:
        return "Retrait annulé.", gr.update()
    if cle not in LIBELLES:
        raise gr.Error("Choisis un élément dans la liste.")
    avant = sum(taille(c) for c in chemins(cle))
    retraits.ajouter(cle)  # d'abord : si la suppression échoue à moitié, l'installateur ne réinstalle pas en douce
    try:
        for c in chemins(cle):
            _supprimer(c)
    except OSError as e:
        raise gr.Error(f"Suppression incomplète : un fichier est utilisé ({e}). Attends la fin de la génération en "
                       "cours ou relance Studio Voix, puis recommence.")
    return (f"🗑️ {LIBELLES[cle]} : {_go(avant)} libérés. Pour le réinstaller plus tard : « Réinstaller » puis "
            "METTRE_A_JOUR.bat.", inventaire())


def vider(cle):
    if cle is None:
        return "Annulé.", gr.update()
    dossier = {"uv-cache": cfg.ENG_DIR / "uv-cache", "diagnostic": cfg.DATA_DIR / "diagnostic"}.get(cle)
    if dossier is None:
        raise gr.Error("Choisis un élément dans la liste.")
    avant = taille(dossier)
    try:
        _supprimer(dossier)
    except OSError as e:
        raise gr.Error(f"Suppression incomplète : un fichier est utilisé ({e}).")
    return f"🧹 {_go(avant)} libérés.", inventaire()


def reinstaller(cle):
    if cle not in LIBELLES:
        raise gr.Error("Choisis un élément dans la liste.")
    if not retraits.retire(cle):
        return f"{LIBELLES[cle]} n'est pas retiré.", inventaire()
    if cle.startswith("diffusion:") and "diffusion" in retraits.liste():
        raise gr.Error("Tout le moteur de diffusion est retiré : réinstalle d'abord « " + LIBELLES["diffusion"] + " ».")
    retraits.enlever(cle)
    return (f"✅ {LIBELLES[cle]} sera réinstallé : ferme Studio Voix et lance METTRE_A_JOUR.bat (ou INSTALLER.bat).",
            inventaire())
