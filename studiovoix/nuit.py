"""File de tâches pour la nuit : un fichier texte décrit des chansons, musiques, images, modèles 3D, lectures,
bruitages et vidéos, lancés l'un après l'autre sans interface (`NUIT.bat`, `python -m studiovoix.nuit`).

Format du fichier (UTF-8, ou cp1252 du Bloc-notes) :

    # commentaire
    eteindre: oui                      ← réglages de la file, avant la première tâche

    == chanson : Ballade du soir       ← « == type : nom » commence une tâche
    style: ballade pop, piano          ← « clé: valeur » sur une ligne
    voix: Laurent
    paroles:                           ← clé sans valeur : tout ce qui suit, jusqu'à la tâche suivante
    Couplet 1
    ...

Une tâche en échec est consignée et la suivante est lancée. Journal et rapport : data/nuit/<horodatage>_<nom>/
(journal.txt, rapport.json). Le fichier est vérifié en entier avant la première génération (`verifier`), pour ne pas
découvrir une faute de frappe à 3 h du matin.
"""
import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

from . import config as cfg
from .styles import _normal

# type (forme normalisée tapée) → type
TYPES = {"chanson": "chanson", "musique": "musique", "instrumental": "musique", "image": "image", "3d": "3d",
         "objet 3d": "3d", "modele 3d": "3d", "lecture": "lecture", "synthese vocale": "lecture",
         "texte lu": "lecture", "bruitage": "bruitage", "son": "bruitage", "video": "video"}
# clés acceptées pour chaque type (formes normalisées ; synonymes → clé)
CLES = {
    "chanson": {"style": "style", "description": "style", "voix": "voix", "chanteurs": "chanteurs",
                "paroles": "paroles", "langue": "langue", "duree": "duree", "bpm": "bpm", "versions": "versions",
                "graine": "graine", "reflexion": "reflexion"},
    "musique": {"style": "style", "description": "style", "duree": "duree", "bpm": "bpm", "versions": "versions",
                "graine": "graine", "reflexion": "reflexion"},
    "image": {"description": "description", "prompt": "prompt", "style": "style", "format": "format",
              "variantes": "variantes", "graine": "graine"},
    "3d": {"description": "description", "prompt": "prompt", "image": "image", "qualite": "qualite",
           "texture": "texture", "obj": "obj", "projet": "projet", "graine": "graine"},
    "lecture": {"voix": "voix", "texte": "texte", "langue": "langue", "expressivite": "expressivite",
                "graine": "graine"},
    "bruitage": {"description": "description", "prompt": "prompt", "duree": "duree", "variantes": "variantes",
                 "etapes": "etapes", "projet": "projet", "graine": "graine"},
    "video": {"description": "description", "prompt": "prompt", "image": "image", "format": "format",
              "duree": "duree", "etapes": "etapes", "graine": "graine"},
}
REGLAGES = {"eteindre": "eteindre", "arreter apres": "arret_echecs", "arret apres echecs": "arret_echecs"}
TACHES_MAX = 500
OUI = {"oui", "o", "yes", "y", "1", "vrai", "true"}
NON = {"non", "n", "no", "0", "faux", "false"}

_ENTETE = re.compile(r"^==+\s*([^:=]+?)\s*(?::\s*(.*?))?\s*=*\s*$")
_CLE = re.compile(r"^([A-Za-zÀ-ÿ0-9 '_-]{1,30}?)\s*:\s*(.*)$")


class ErreurFichier(ValueError):
    """Faute dans le fichier de tâches : message en français avec le numéro de ligne."""


def lire_texte(chemin):
    donnees = Path(chemin).read_bytes()
    for codage in ("utf-8-sig", "cp1252"):
        try:
            return donnees.decode(codage)
        except UnicodeDecodeError:
            continue
    raise ErreurFichier("Fichier illisible (ni UTF-8 ni Windows-1252).")


def lire(texte):
    """Texte du fichier → (réglages {clé: valeur}, tâches [{type, nom, ligne, champs}]). Lève ErreurFichier."""
    reglages, taches = {}, []
    courante, multi = None, None  # tâche en cours ; clé multi-ligne en cours
    for num, brute in enumerate(texte.splitlines(), 1):
        ligne = brute.rstrip()
        m = _ENTETE.match(ligne.strip()) if ligne.lstrip().startswith("==") else None
        if m:
            t = TYPES.get(_normal(m.group(1)))
            if not t:
                raise ErreurFichier(f"Ligne {num} : type de tâche inconnu « {m.group(1)} » (possibles : "
                                    f"{', '.join(sorted(set(TYPES.values())))}).")
            courante = {"type": t, "nom": (m.group(2) or "").strip(), "ligne": num, "champs": {}}
            taches.append(courante)
            multi = None
            continue
        if multi is not None:  # une clé sans valeur prend toutes les lignes jusqu'à la tâche suivante
            courante["champs"][multi].append(brute.rstrip())
            continue
        if not ligne.strip() or ligne.lstrip().startswith("#"):
            continue
        m = _CLE.match(ligne.strip())
        if not m:
            raise ErreurFichier(f"Ligne {num} : « {ligne.strip()[:60]} » n'est pas une ligne « clé: valeur » (pour un "
                                "texte sur plusieurs lignes, écris la clé seule, « paroles: », puis le texte dessous, "
                                "à la fin de la tâche).")
        cle, valeur = _normal(m.group(1)), m.group(2).strip()
        if courante is None:
            if cle not in REGLAGES:
                raise ErreurFichier(f"Ligne {num} : réglage inconnu « {m.group(1)} » avant la première tâche "
                                    f"(possibles : eteindre, arreter apres). Une tâche commence par « == type : nom ».")
            reglages[REGLAGES[cle]] = valeur
            continue
        connues = CLES[courante["type"]]
        if cle not in connues:
            raise ErreurFichier(f"Ligne {num} : « {m.group(1)} » n'est pas une clé de {courante['type']} (possibles : "
                                f"{', '.join(sorted(set(connues.values())))}).")
        if valeur:
            courante["champs"][connues[cle]] = valeur
        else:
            multi = connues[cle]
            courante["champs"][multi] = []
    for t in taches:
        for k, v in list(t["champs"].items()):
            if isinstance(v, list):
                t["champs"][k] = "\n".join(v).strip()
    if len(taches) > TACHES_MAX:
        raise ErreurFichier(f"{len(taches)} tâches : {TACHES_MAX} au plus.")
    return reglages, taches


def oui_non(valeur, defaut=False):
    v = _normal(valeur)
    if not v:
        return defaut
    if v in OUI:
        return True
    if v in NON:
        return False
    raise ErreurFichier(f"« {valeur} » : écris oui ou non.")


def nombre(valeur, defaut, mini=None, maxi=None, entier=True):
    if valeur in (None, ""):
        return defaut
    try:
        x = float(str(valeur).replace(",", ".").strip())
    except ValueError:
        raise ErreurFichier(f"« {valeur} » n'est pas un nombre.") from None
    if mini is not None:
        x = max(mini, x)
    if maxi is not None:
        x = min(maxi, x)
    return int(round(x)) if entier else x


def choix(valeur, options, defaut, quoi):
    """Libellé d'une liste de l'interface retrouvé d'après le début tapé, sans accents ni majuscules
    (« low poly facette » → « Low poly facetté (couleurs unies, ~3 000 faces) »)."""
    if not (valeur or "").strip():
        return defaut
    v = _normal(valeur)
    for o in options:
        if _normal(o) == v:
            return o
    debut = [o for o in options if _normal(o).startswith(v)] or [o for o in options if v in _normal(o)]
    if len(debut) >= 1:
        return debut[0]
    raise ErreurFichier(f"{quoi} inconnu(e) « {valeur} » (possibles : {' ; '.join(options)}).")


def langue(valeur):
    if not (valeur or "").strip():
        return "Français"
    v = _normal(valeur)
    for lib, code in cfg.LANGUES.items():
        if v in (_normal(lib), code):
            return lib
    raise ErreurFichier(f"Langue inconnue « {valeur} » (possibles : {', '.join(cfg.LANGUES)}).")


def voix_de_la_bibliotheque(valeur):
    from .voix import list_voices

    voix = list_voices()
    for v in voix:
        if v.lower() == (valeur or "").strip().lower():
            return v
    raise ErreurFichier(f"Voix « {valeur} » absente de la bibliothèque (voix : {', '.join(voix) or 'aucune'}).")


def _fichier(valeur, base):
    p = Path(str(valeur).strip().strip('"'))
    if not p.is_absolute():
        p = Path(base) / p
    if not p.is_file():
        raise ErreurFichier(f"Fichier introuvable : {p}")
    return str(p)


def _virgules(texte):
    return [x.strip() for x in (texte or "").split(",") if x.strip()]


# --- Préparation : champs → appel prêt à lancer (toutes les erreurs de fichier sortent ici, avant la nuit) ---

def preparer(tache, base="."):
    """Tâche lue → fonction sans argument (progress) qui la lance et renvoie (message, dossier). Lève ErreurFichier."""
    t, c, nom = tache["type"], tache["champs"], tache["nom"]
    graine = nombre(c.get("graine"), 0, 0)
    if t in ("chanson", "musique"):
        from . import acestep, pipeline
        from .serie_musique import duree_de

        style = c.get("style", "")
        if not style:
            raise ErreurFichier("il manque « style: » (genre, instruments, ambiance…).")
        mode, voix = pipeline.MODE_INSTRU, None
        if t == "chanson":
            v = _normal(c.get("voix"))
            if v in ("instrumental", "aucune", "sans voix"):
                mode = pipeline.MODE_INSTRU
            elif v in ("", "ace", "ace step", "acestep"):
                mode = pipeline.MODE_VOIX_ACE
            else:
                mode, voix = pipeline.MODE_MA_VOIX, voix_de_la_bibliotheque(c["voix"])
            if mode != pipeline.MODE_INSTRU and not c.get("paroles"):
                raise ErreurFichier("il manque les paroles (« paroles: » puis le texte dessous), ou écris "
                                    "« voix: instrumental ».")
        chanteurs = choix(c.get("chanteurs"), list(acestep.VOIX_CHANTEES), "Automatique", "Voix chantées")
        description = acestep.build_prompt(None, _virgules(style), None, None,
                                           chanteurs if mode != pipeline.MODE_INSTRU else "Automatique", None)
        lang = langue(c.get("langue"))
        duree = duree_de(c.get("duree"), 120 if t == "chanson" else 90)
        bpm, versions = nombre(c.get("bpm"), 0, 0, 300), nombre(c.get("versions"), 1, 1, pipeline.MAX_CHANSONS)
        reflexion = oui_non(c.get("reflexion"), True)

        def lancer(progress):
            r = pipeline.creer_chanson(voix, None, None, None, None, None, chanteurs, c.get("paroles", ""), lang,
                                       duree, bpm, reflexion, 0, 40, 1.0, 1.0, mode=mode, description=description,
                                       versions=versions, graine=graine, progress=progress)
            return r[4], str(Path(r[0]).parent)
        return lancer
    if t == "image":
        from . import images

        if not (c.get("description") or c.get("prompt")):
            raise ErreurFichier("il manque « description: » (en français) ou « prompt: » (en anglais).")
        fmt = choix(c.get("format"), list(images.FORMATS), images.AUTO, "Format")
        n = nombre(c.get("variantes"), 2, 1, 4)

        def lancer(progress):
            prompt = images.prompt_pret(c.get("description"), c.get("prompt"), progress=progress)
            msg, _, dossier = images.generer(prompt, None, None, _virgules(c.get("style")), fmt, n, graine,
                                             nom=nom, description_fr=c.get("description"), progress=progress)
            return msg, dossier
        return lancer
    if t == "3d":
        from . import modele3d

        image = _fichier(c["image"], base) if c.get("image") else None
        if not (image or c.get("description") or c.get("prompt")):
            raise ErreurFichier("il manque « description: », « prompt: » ou « image: » (chemin d'une image).")
        qualite = choix(c.get("qualite"), list(modele3d.QUALITES), modele3d.QUALITE_DEFAUT, "Qualité")
        texture = oui_non(c.get("texture"), True)
        formats = ["glb"] + (["obj"] if oui_non(c.get("obj"), False) else [])

        def lancer(progress):
            img, img_graine, prompt = image, None, None
            if not img:
                prompt = modele3d.prompt_pret(c.get("description"), c.get("prompt"), progress=progress)
                img, img_graine, _ = modele3d.generer_image(prompt, graine, progress=progress)
            r = modele3d.generer(img, nom or "objet", qualite, texture, graine, formats, description=prompt,
                                 description_fr=c.get("description"), image_graine=img_graine,
                                 projet=c.get("projet") or None, progress=progress)
            return r[0], r[4]
        return lancer
    if t == "lecture":
        from . import chatterbox

        if not c.get("texte"):
            raise ErreurFichier("il manque le texte (« texte: » puis le texte dessous).")
        if len(c["texte"]) > chatterbox.TEXTE_MAX:
            raise ErreurFichier(f"texte trop long ({len(c['texte'])} caractères, {chatterbox.TEXTE_MAX} au plus).")
        voix = voix_de_la_bibliotheque(c.get("voix"))
        lang = langue(c.get("langue"))
        expr = nombre(c.get("expressivite"), 0.5, 0.25, 2, entier=False)

        def lancer(progress):
            fichier, msg = chatterbox.synthese(voix, c["texte"], lang, expr, 0.5, 0.8, graine, progress=progress)
            return msg, str(Path(fichier).parent)
        return lancer
    if t == "bruitage":
        from . import bruitages

        if not (c.get("description") or c.get("prompt")):
            raise ErreurFichier("il manque « description: » ou « prompt: ».")
        duree = nombre(c.get("duree"), 3, 1, bruitages.DUREE_MAX, entier=False)
        n, etapes = nombre(c.get("variantes"), 3, 1, 3), nombre(c.get("etapes"), 100, 10, 200)

        def lancer(progress):
            prompt = bruitages.prompt_pret(c.get("description"), None, c.get("prompt"), progress=progress)
            r = bruitages.generer(prompt, nom, duree, n, graine, etapes, description=c.get("description"),
                                  projet=c.get("projet") or None, progress=progress)
            return r[0], r[3]
        return lancer
    if t == "video":
        from . import videos

        image = _fichier(c["image"], base) if c.get("image") else None
        if not (c.get("description") or c.get("prompt")):
            raise ErreurFichier("il manque « description: » ou « prompt: ».")
        fmt = choix(c.get("format"), list(videos.FORMATS), videos.AUTO, "Format")
        duree = choix(c.get("duree"), list(videos.DUREES), videos.DUREE_DEFAUT, "Durée")
        etapes = nombre(c.get("etapes"), 30, 10, 50)

        def lancer(progress):
            prompt = videos.prompt_pret(c.get("description"), c.get("prompt"), image, progress=progress)
            msg, _, dossier = videos.generer(prompt, image, fmt, duree, etapes, graine, nom=nom,
                                             description_fr=c.get("description"), progress=progress)
            return msg, dossier
        return lancer
    raise ErreurFichier(f"type inconnu : {t}")


def verifier(chemin):
    """Lit et prépare tout le fichier. Renvoie (réglages, [(tâche, lancer)]). Lève ErreurFichier (toutes les fautes,
    une par ligne)."""
    reglages, taches = lire(lire_texte(chemin))
    if not taches:
        raise ErreurFichier("Aucune tâche : chaque tâche commence par une ligne « == type : nom » (par exemple "
                            "« == image : Château »).")
    oui_non(reglages.get("eteindre"))
    pretes, fautes = [], []
    for t in taches:
        try:
            pretes.append((t, preparer(t, Path(chemin).parent)))
        except ErreurFichier as e:
            fautes.append(f"Tâche « {t['nom'] or t['type']} » (ligne {t['ligne']}) : {e}")
    if fautes:
        raise ErreurFichier("\n".join(fautes))
    return reglages, pretes


def libelle(t):
    return f"{t['type']}{' « ' + t['nom'] + ' »' if t['nom'] else ''}"


class Suivi:
    """Remplace gr.Progress hors de l'interface : une ligne au plus toutes les 5 s (et à chaque nouveau message)."""

    def __init__(self, ecrire):
        self.ecrire, self.dernier, self.temps = ecrire, None, 0.0

    def __call__(self, fraction=None, desc=None, **_):
        maintenant = time.monotonic()
        if desc != self.dernier or maintenant - self.temps > 5:
            pct = f"{100 * float(fraction):3.0f} % " if isinstance(fraction, (int, float)) else ""
            self.ecrire(f"    {pct}{desc or ''}".rstrip())
            self.dernier, self.temps = desc, maintenant

    def tqdm(self, iterable, *_, **__):
        return iterable


def executer(chemin, a_partir_de=1, eteindre=None, sortie=print):
    """Lance toute la file. Renvoie le rapport (dict) ; il est aussi écrit dans data/nuit/<horodatage>_<nom>/."""
    from .outils import eteindre_plus_tard, nouveau_dossier, sans_veille
    from .serie import _slug

    reglages, pretes = verifier(chemin)
    dossier = nouveau_dossier(cfg.NUIT_DIR, _slug(Path(chemin).stem, "file"))
    journal = dossier / "journal.txt"

    def ecrire(texte):
        ligne = f"{time.strftime('%H:%M:%S')} {texte}"
        sortie(ligne)
        with open(journal, "a", encoding="utf-8") as f:
            f.write(ligne + "\n")

    eteindre = oui_non(reglages.get("eteindre")) if eteindre is None else eteindre
    arret = nombre(reglages.get("arret_echecs"), 0, 0)
    rapport = {"fichier": str(Path(chemin).resolve()), "debut": time.strftime("%Y-%m-%d %H:%M:%S"), "taches": []}
    (dossier / Path(chemin).name).write_bytes(Path(chemin).read_bytes())
    ecrire(f"{len(pretes)} tâche(s) dans « {Path(chemin).name} », journal : {journal}")
    debut, echecs_suite = time.monotonic(), 0
    with sans_veille():
        for i, (t, lancer) in enumerate(pretes, 1):
            if i < a_partir_de:
                continue
            ecrire(f"[{i}/{len(pretes)}] {libelle(t)}…")
            t0, entree = time.monotonic(), {"numero": i, "type": t["type"], "nom": t["nom"], "ligne": t["ligne"]}
            try:
                msg, resultat = lancer(Suivi(ecrire))
                entree.update(etat="reussie", dossier=str(resultat or ""), message=str(msg or ""))
                ecrire(f"    ✅ {str(msg or '').splitlines()[0][:300] if msg else 'terminé'}")
                echecs_suite = 0
            except KeyboardInterrupt:
                entree.update(etat="interrompue")
                rapport["taches"].append(entree)
                ecrire("Interrompu (Ctrl+C).")
                break
            except Exception as e:  # noqa: BLE001 - une tâche en échec n'arrête pas la nuit
                entree.update(etat="echec", message=str(e) or type(e).__name__)
                ecrire(f"    ❌ {str(e) or type(e).__name__}")
                with open(dossier / "erreurs.txt", "a", encoding="utf-8") as f:
                    f.write(f"--- [{i}] {libelle(t)}\n{traceback.format_exc()}\n")
                echecs_suite += 1
            entree["temps_s"] = round(time.monotonic() - t0)
            rapport["taches"].append(entree)
            (dossier / "rapport.json").write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
            if arret and echecs_suite >= arret:
                ecrire(f"Arrêt : {echecs_suite} échecs à la suite.")
                break
    reussies = sum(e["etat"] == "reussie" for e in rapport["taches"])
    rapport.update(fin=time.strftime("%Y-%m-%d %H:%M:%S"), reussies=reussies,
                   echecs=sum(e["etat"] == "echec" for e in rapport["taches"]), dossier=str(dossier))
    (dossier / "rapport.json").write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    from .convertir import duree_lisible

    ecrire(f"Fini en {duree_lisible(time.monotonic() - debut)} : {reussies} réussie(s), {rapport['echecs']} en "
           f"échec. Rapport : {dossier / 'rapport.json'}")
    if eteindre:
        eteindre_plus_tard(300)
        ecrire("L'ordinateur s'éteindra dans 5 minutes (tape « shutdown /a » pour annuler).")
    return rapport


EXEMPLE = """# File de tâches de Studio Voix : lancée par NUIT.bat, une tâche après l'autre.
# Les lignes qui commencent par # sont ignorées. Une tâche commence par « == type : nom ».
# Types : chanson, musique, image, 3d, lecture, bruitage, video.
# Un texte sur plusieurs lignes (paroles, texte lu) : la clé seule (« paroles: »), le texte dessous, en fin de tâche.

# Éteindre l'ordinateur à la fin (oui / non) :
eteindre: non

== musique : Thème de la taverne
style: médiéval, taverne, luth, flûte, joyeux
duree: 1:30

== image : Château dans la brume
description: un château fort sur une falaise, au lever du soleil, dans la brume
style: aquarelle
variantes: 2

== 3d : Potion de soin
description: une fiole de potion rouge avec un bouchon de liège
qualite: low poly facetté

== bruitage : Porte qui grince
description: une vieille porte en bois qui grince en s'ouvrant
duree: 3

== chanson : Ballade du soir
style: ballade pop, piano, nostalgique
# voix : une voix de ta bibliothèque, « ace » (voix d'ACE-Step) ou « instrumental »
voix: ace
duree: 2:00
paroles:
Couplet 1
Le soir tombe sur la ville
Et je pense encore à toi

Refrain
Reste encore un peu
Juste le temps d'un feu
"""


def principal(argv=None):
    p = argparse.ArgumentParser(prog="NUIT.bat", description="File de tâches de Studio Voix.")
    p.add_argument("fichier", nargs="?", help="fichier de tâches (.txt)")
    p.add_argument("--verifier", action="store_true", help="vérifier le fichier sans rien lancer")
    p.add_argument("--a-partir-de", type=int, default=1, help="commencer à la tâche n (reprise)")
    p.add_argument("--exemple", metavar="FICHIER", help="écrire un fichier d'exemple")
    a = p.parse_args(argv)
    if a.exemple:
        Path(a.exemple).write_text(EXEMPLE, encoding="utf-8-sig")
        print(f"Exemple écrit : {a.exemple}")
        return 0
    if not a.fichier or not Path(a.fichier).is_file():
        print("Donne le fichier de tâches (glisse-le sur NUIT.bat).")
        return 2
    try:
        if a.verifier:
            _, pretes = verifier(a.fichier)
            print(f"✅ Fichier correct : {len(pretes)} tâche(s).")
            for i, (t, _) in enumerate(pretes, 1):
                print(f"  {i}. {libelle(t)} (ligne {t['ligne']})")
            return 0
        rapport = executer(a.fichier, a.a_partir_de)
    except ErreurFichier as e:
        print(f"❌ Le fichier de tâches contient des erreurs :\n{e}")
        return 3
    return 0 if not rapport["echecs"] else 1


if __name__ == "__main__":
    sys.exit(principal())
