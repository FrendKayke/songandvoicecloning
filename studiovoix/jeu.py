"""Bande-son de jeu : musiques sans voix par situation (écran titre, combat, victoire…), générées en lot.

Chaque situation a une description en anglais (la langue que comprend le mieux ACE-Step), une durée et
un type : « boucle » (musique de fond) ou « jingle » (effet court). On décrit un style (JRPG, 16-bit…),
jamais une œuvre ou un nom (« Final Fantasy ») : le modèle ne les connaît pas de façon fiable et cela
pousserait à l'imitation. Rangement : data/jeux/<projet>/<situation>/<horodatage>/.
"""
import re
import shutil
from pathlib import Path

import gradio as gr

from . import acestep, boucle as boucles
from . import config as cfg
from .mixage import couper_jingle
from .outils import nouveau_dossier
from .pipeline import INSTRUMENTAL, ecrire_creation
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
                      progress=gr.Progress()):
    """Génère, une par une, les musiques des situations choisies. Renvoie (message, liste des pistes, piste 1)."""
    projet = nom_projet(projet)
    if not situations:
        raise gr.Error("Choisis au moins une situation (ou tape la tienne).")
    if not texte(epoque):
        raise gr.Error("Choisis une époque (style général de la bande-son).")
    pistes = []
    for n, cle in enumerate(situations, 1):
        ident, libelle, txt, duree, boucle = situation(cle)
        etape = f"{n}/{len(situations)} {libelle}"
        duree_gen = int(duree_boucles) if boucle else max(DUREE_MIN_ACESTEP, duree)
        prompt = description(epoque, univers, txt, extra)
        dossier = nouveau_dossier(cfg.GAMES_DIR / projet / ident)
        params = acestep.text2music_params(prompt, INSTRUMENTAL, "en", duree_gen, 0, thinking)
        ((brute, seed),) = acestep.generer(params, [dossier / "brute.wav"], progress, etape, graine=graine)
        piste = dossier / "piste.wav"
        infos_boucle, note = None, ""
        if boucle:
            try:
                b = boucles.creer_boucle(brute, piste, dossier / "apercu_jonction.wav")
                infos_boucle = boucles.en_dict(b)
                note = f"boucle de {b.duree:.0f} s ({b.mesures} mesures), jonction {b.qualite()}"
            except ValueError:
                shutil.copy(brute, piste)
                note = "pas de boucle trouvée (morceau trop peu rythmé) : piste gardée telle quelle"
        else:
            duree = couper_jingle(brute, piste, duree)
            note = f"jingle de {duree:.1f} s"
        ecrire_creation(dossier, {
            "type": "jeu", "projet": projet, "situation": ident, "libelle": libelle, "boucle": boucle,
            "description": prompt, "duree": round(float(duree_gen if boucle else duree), 2),
            "reflexion": bool(thinking), "boucle_points": infos_boucle,
            "versions": [{"graine": seed, "dossier": ".", "fichier": str(piste)}],
        })
        pistes.append((libelle, str(piste), note))
    choix = [(lib, p) for lib, p, _ in pistes]
    msg = (f"✅ {len(pistes)} piste(s) générée(s) dans {cfg.GAMES_DIR / projet} :\n\n"
           + "\n".join(f"- **{lib}** : {note}" for lib, _, note in pistes))
    return msg, gr.update(choices=choix, value=choix[0][1]), choix[0][1], jonction(choix[0][1])


def jonction(piste):
    """Aperçu de la jonction (5 s de fin puis 5 s de début) d'une piste en boucle, sinon None."""
    if not piste:
        return None
    p = Path(piste).with_name("apercu_jonction.wav")
    return str(p) if p.exists() else None


def ecouter(piste):
    return piste, jonction(piste)
