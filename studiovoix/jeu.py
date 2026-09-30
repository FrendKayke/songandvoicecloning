"""Bande-son de jeu : musiques sans voix par situation (écran titre, combat, victoire…), générées en lot.

Chaque situation a une description en anglais (la langue que comprend le mieux ACE-Step), une durée et
un type : « boucle » (musique de fond) ou « jingle » (effet court). On décrit un style (JRPG, 16-bit…),
jamais une œuvre ou un nom (« Final Fantasy ») : le modèle ne les connaît pas de façon fiable et cela
pousserait à l'imitation. Rangement : data/jeux/<projet>/<situation>/<horodatage>/.
"""
import json
import re
import shutil
from pathlib import Path

import gradio as gr

from . import acestep, boucle as boucles
from . import config as cfg
from .mixage import couper_jingle
from .outils import ecrire_creation, nouveau_dossier
from .pipeline import INSTRUMENTAL
from .styles import texte

EPOQUES = [
    ("16-bit (SNES)", "16-bit SNES era JRPG soundtrack, sampled orchestral instruments, retro game music"),
    ("8-bit (NES)", "8-bit NES chiptune, square wave lead, triangle bass, noise percussion, retro game music"),
    ("Orchestral moderne", "modern orchestral JRPG soundtrack, full symphony orchestra, cinematic game music"),
    ("Rock JRPG", "JRPG rock soundtrack, electric guitars, organ, orchestral hits, game music"),
    ("Piano et cordes", "emotional JRPG piano and strings, intimate game music"),
]

UNIVERS = [
    ("Fantasy héroïque", "heroic high fantasy"),
    ("Cristaux et magie", "crystals and magic, mystical"),
    ("Médiéval", "medieval"),
    ("Steampunk", "steampunk, industrial"),
    ("Sombre", "dark fantasy"),
    ("Lumineux / féerique", "whimsical fairy tale"),
    ("Oriental", "East Asian instruments, koto and shakuhachi"),
    ("Celtique", "Celtic, tin whistle and fiddle"),
]

# id → (libellé, description, durée en s, boucle ?)
SITUATIONS = {
    "titre": ("Écran titre", "majestic main theme, memorable heroic melody, strings and brass, slow build", 90, True),
    "menu": ("Menu / construction du deck", "calm and thoughtful, harp, woodwinds, gentle strings", 90, True),
    "plateau": ("Plateau (partie calme)", "relaxed strategic mood, pizzicato strings, flute melody, soft percussion", 120, True),
    "tension": ("Dernier tour / tension", "tense and suspenseful, ticking percussion, low strings ostinato", 60, True),
    "combat": ("Combat", "fast battle theme, driving strings ostinato, brass stabs, timpani, energetic", 90, True),
    "boss": ("Combat de boss", "epic boss battle, dramatic choir, pipe organ, heavy drums, intense", 120, True),
    "boutique": ("Boutique", "cheerful shop music, playful and bouncy, light percussion", 60, True),
    "victoire": ("Victoire (fanfare)", "short triumphant victory fanfare, brass, final major chord", 7, False),
    "defaite": ("Défaite", "short sad defeat jingle, slow descending melody, soft strings", 6, False),
    "booster": ("Ouverture d'un booster", "short magical sparkle jingle, rising arpeggio, bells and chimes", 3, False),
    "rare": ("Carte rare obtenue", "short exciting reveal sting, shimmering chimes, triumphant chord", 4, False),
}
DUREE_MIN_ACESTEP = 10  # acestep/constants.py : DURATION_MIN

# Cohérence de la bande-son à partir d'un thème (piste déjà générée du projet)
REF_AUCUNE = "Aucune"
REF_TIMBRE = "Même son (référence de timbre et de mixage)"
REF_VARIATION = "Variation du thème (même mélodie réarrangée)"
REFERENCES = [REF_AUCUNE, REF_TIMBRE, REF_VARIATION]


def pistes_projet(projet):
    """Pistes déjà générées d'un projet, les plus récentes d'abord : [(libellé, chemin)]."""
    try:
        racine = cfg.GAMES_DIR / nom_projet(projet)
    except gr.Error:
        return []
    choix = []
    for piste in sorted(racine.glob("*/*/piste.wav"), key=lambda p: p.parent.name, reverse=True):
        try:
            infos = json.loads((piste.parent / "creation.json").read_text(encoding="utf-8"))
            libelle = infos.get("libelle", piste.parent.parent.name)
        except (OSError, ValueError):
            libelle = piste.parent.parent.name
        date = piste.parent.name
        choix.append((f"{libelle} — {date[6:8]}/{date[4:6]} {date[9:11]}h{date[11:13]}", str(piste)))
    return choix


def maj_references(projet, actuelle=None):
    choix = pistes_projet(projet)
    valeurs = [v for _, v in choix]
    return gr.update(choices=choix, value=actuelle if actuelle in valeurs else None)


def choix_situations():
    return [(v[0], k) for k, v in SITUATIONS.items()]


def nom_projet(projet):
    nom = "".join(c for c in (projet or "").strip() if c.isalnum() or c in "-_ ").strip()
    if not nom:
        raise gr.Error("Donne un nom au projet (lettres, chiffres, espaces, - et _).")
    return nom


def situation(cle):
    """Situation prédéfinie, ou saisie libre (description en anglais, musique en boucle de 90 s)."""
    if cle in SITUATIONS:
        return cle, *SITUATIONS[cle]
    libre = cle.strip()
    ident = re.sub(r"[^a-z0-9]+", "_", libre.lower()).strip("_")[:40] or "perso"
    return f"perso_{ident}", libre, libre, 90, True


def description(epoque, univers, texte_situation, extra):
    return ", ".join(p for p in [texte(epoque), texte_situation, texte(univers), texte(extra),
                                 "instrumental, no vocals"] if p)


def apercu(epoque, univers, situations, extra, duree_boucles):
    """Tableau des descriptions qui seront envoyées, situation par situation."""
    if not situations:
        return "*Choisis au moins une situation.*"
    lignes = ["| Situation | Type | Durée | Description envoyée |", "|---|---|---|---|"]
    for cle in situations:
        _, libelle, txt, duree, boucle = situation(cle)
        duree = int(duree_boucles) if boucle else duree
        lignes.append(f"| {libelle} | {'boucle' if boucle else 'jingle'} | {duree} s | "
                      f"{description(epoque, univers, txt, extra)} |")
    return "\n".join(lignes)


def generer_bande_son(projet, epoque, univers, situations, extra, duree_boucles, thinking, graine=0,
                      reference=None, usage=REF_AUCUNE, fidelite=0.5, progress=gr.Progress()):
    """Génère, une par une, les musiques des situations choisies.

    « reference » : piste du projet servant de thème ; « usage » : REF_TIMBRE (reference_audio : timbre et
    mixage communs) ou REF_VARIATION (task_type « cover » : src_audio réarrangé selon la nouvelle description,
    audio_cover_strength = fidélité). En cover, ACE-Step fixe la durée sur celle du thème : les jingles
    n'utilisent donc que la référence de timbre.
    Renvoie (message, liste des pistes, piste 1, aperçu de sa jonction).
    """
    projet = nom_projet(projet)
    usage = usage or REF_AUCUNE
    if usage != REF_AUCUNE and not (reference and Path(reference).exists()):
        raise gr.Error("Choisis le thème de référence (une piste déjà générée du projet), ou « Aucune ».")
    if not situations:
        raise gr.Error("Choisis au moins une situation (ou tape la tienne).")
    if not texte(epoque):
        raise gr.Error("Choisis une époque (style général de la bande-son).")
    pistes = []
    for n, cle in enumerate(situations, 1):
        ident, libelle, txt, duree, boucle = situation(cle)
        duree_gen = int(duree_boucles) if boucle else max(DUREE_MIN_ACESTEP, duree)
        params = acestep.text2music_params(description(epoque, univers, txt, extra), INSTRUMENTAL, "en",
                                           duree_gen, 0, thinking)
        fichiers = None
        if usage == REF_VARIATION and boucle:
            params.update(task_type="cover", audio_cover_strength=float(fidelite), thinking=False)
            fichiers = {"src_audio": reference}
        elif usage != REF_AUCUNE:
            fichiers = {"reference_audio": reference}
        infos = {"projet": projet, "situation": ident, "libelle": libelle, "boucle": boucle,
                 "duree_cible": None if boucle else duree,
                 "reference": str(reference) if fichiers else None,
                 "usage_reference": ("variation" if "src_audio" in fichiers else "timbre") if fichiers else None,
                 "fidelite": float(fidelite) if fichiers and "src_audio" in fichiers else None}
        piste, note = generer_piste(params, infos, progress, f"{n}/{len(situations)} {libelle}", fichiers, graine)
        pistes.append((libelle, str(piste), note))
    choix = [(lib, p) for lib, p, _ in pistes]
    msg = (f"✅ {len(pistes)} piste(s) générée(s) dans {cfg.GAMES_DIR / projet} :\n\n"
           + "\n".join(f"- **{lib}** : {note}" for lib, _, note in pistes))
    return msg, gr.update(choices=choix, value=choix[0][1]), choix[0][1], jonction(choix[0][1])


def generer_piste(params, infos, progress, etape, fichiers=None, graine=0, dossier=None):
    """Une piste de jeu : génération ACE-Step (params prêts : text2music, cover ou repaint), puis boucle
    parfaite ou jingle selon infos["boucle"], et creation.json. Sert à la bande-son, à « Recréer » et à
    « Refaire un passage » (galerie). Renvoie (piste, note lisible)."""
    dossier = dossier or nouveau_dossier(cfg.GAMES_DIR / infos["projet"] / infos["situation"])
    ((brute, seed),) = acestep.generer(params, [dossier / "brute.wav"], progress, etape, fichiers=fichiers,
                                       graine=graine)
    piste = dossier / "piste.wav"
    infos_boucle = None
    if infos["boucle"]:
        try:
            b = boucles.creer_boucle(brute, piste, dossier / "apercu_jonction.wav")
            infos_boucle = boucles.en_dict(b)
            note = f"boucle de {b.duree:.0f} s ({b.mesures} mesures), jonction {b.qualite()}"
        except ValueError:
            shutil.copy(brute, piste)
            note = "pas de boucle trouvée (morceau trop peu rythmé) : piste gardée telle quelle"
        duree = params.get("audio_duration")
    else:
        duree = couper_jingle(brute, piste, infos["duree_cible"])
        note = f"jingle de {duree:.1f} s"
    ecrire_creation(dossier, {
        "type": "jeu", **infos, "description": params["prompt"],
        "duree": round(float(duree), 2) if duree else None, "reflexion": bool(params.get("thinking")),
        "boucle_points": infos_boucle, "versions": [{"graine": seed, "dossier": ".", "fichier": str(piste)}],
    })
    return piste, note


def jonction(piste):
    """Aperçu de la jonction (5 s de fin puis 5 s de début) d'une piste en boucle, sinon None."""
    if not piste:
        return None
    p = Path(piste).with_name("apercu_jonction.wav")
    return str(p) if p.exists() else None


def ecouter(piste):
    return piste, jonction(piste)
