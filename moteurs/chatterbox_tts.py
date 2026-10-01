"""Synthèse vocale Chatterbox Multilingual — script exécuté DANS l'environnement de Chatterbox.

L'application (qui n'a pas torch) le lance en sous-processus :
    python chatterbox_tts.py <tache.json>      synthèse décrite par le fichier de tâche
    python chatterbox_tts.py --telecharger     télécharge les modèles (chargement sur CPU) puis quitte

Fichier de tâche (UTF-8) : {"texte", "langue", "voix", "sortie", "exaggeration", "cfg_weight",
"temperature", "graine"}. Le script écrit sur sa sortie des lignes « PROGRESSION i/n » et, en cas
d'erreur prévue, une ligne « ERREUR : message » avant de quitter avec un code non nul.

API vérifiée dans resemble-ai/chatterbox (src/chatterbox/mtl_tts.py) :
ChatterboxMultilingualTTS.from_pretrained(device, t3_model="v3") puis
generate(text, language_id, audio_prompt_path, exaggeration, cfg_weight, temperature) → tenseur (1, n) à model.sr.
generate() plafonne à 1000 jetons de parole (≈ 40 s) : le texte est découpé en morceaux de 300 caractères
au plus, comme dans l'interface officielle (multilingual_app.py).
"""
import json
import re
import sys

T3_MODEL = "v3"
MAX_CARACTERES = 300
PAUSE_S = 0.25  # silence entre deux morceaux


def decouper(texte, max_car=MAX_CARACTERES):
    """Découpe un texte en morceaux d'au plus max_car caractères, de préférence entre les phrases."""
    texte = re.sub(r"\s+", " ", texte or "").strip()
    if not texte:
        return []
    phrases = re.split(r"(?<=[.!?…])\s+", texte)
    morceaux = []

    def ajouter(bout):
        if len(bout) <= max_car:
            morceaux.append(bout)
            return
        # Phrase trop longue : on coupe après une ponctuation faible, sinon entre deux mots
        parties = re.split(r"(?<=[,;:])\s+", bout)
        if len(parties) == 1:
            parties = bout.split(" ")
        courant = ""
        for p in parties:
            while len(p) > max_car:  # mot démesuré : coupe brute
                if courant:
                    morceaux.append(courant)
                    courant = ""
                morceaux.append(p[:max_car])
                p = p[max_car:]
            if courant and len(courant) + 1 + len(p) > max_car:
                ajouter(courant)
                courant = p
            else:
                courant = f"{courant} {p}".strip()
        if courant:
            ajouter(courant)

    courant = ""
    for ph in phrases:
        if courant and len(courant) + 1 + len(ph) <= max_car:
            courant = f"{courant} {ph}"
            continue
        if courant:
            ajouter(courant)
        courant = ph
    if courant:
        ajouter(courant)
    return morceaux


def _erreur(msg, code=2):
    print(f"ERREUR : {msg}", flush=True)
    sys.exit(code)


def _charger_modele(device):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=T3_MODEL)


def telecharger():
    print("Téléchargement / vérification des modèles Chatterbox Multilingual V3…", flush=True)
    modele = _charger_modele("cpu")  # le CPU suffit pour télécharger et vérifier le chargement
    print(f"Modèles Chatterbox prêts (sortie {modele.sr} Hz).", flush=True)


def synthese(chemin_tache):
    import numpy as np
    import soundfile as sf
    import torch

    with open(chemin_tache, encoding="utf-8") as f:
        t = json.load(f)
    morceaux = decouper(t["texte"])
    if not morceaux:
        _erreur("Le texte est vide.")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print("Attention : pas de carte graphique CUDA détectée, synthèse sur le processeur (lente).", flush=True)
    graine = int(t.get("graine") or 0)
    if graine:  # mêmes réglages que set_seed() de l'interface officielle
        import random

        random.seed(graine)
        torch.manual_seed(graine)
        np.random.seed(graine)
        if device == "cuda":
            torch.cuda.manual_seed_all(graine)

    print(f"Chargement du modèle ({device})…", flush=True)
    try:
        modele = _charger_modele(device)
        segments = []
        for i, bout in enumerate(morceaux, 1):
            print(f"PROGRESSION {i}/{len(morceaux)}", flush=True)
            wav = modele.generate(
                bout,
                language_id=t["langue"],
                audio_prompt_path=t["voix"],
                exaggeration=float(t.get("exaggeration", 0.5)),
                cfg_weight=float(t.get("cfg_weight", 0.5)),
                temperature=float(t.get("temperature", 0.8)),
            )
            segments.append(wav.squeeze(0).detach().cpu().numpy().astype("float32"))
            if i < len(morceaux):
                segments.append(np.zeros(int(PAUSE_S * modele.sr), dtype="float32"))
    except torch.cuda.OutOfMemoryError:
        _erreur(
            "mémoire de la carte graphique insuffisante. Ferme les autres programmes qui utilisent la carte "
            "(jeux, vidéos) ; si tu as désactivé la libération automatique, arrête ACE-Step dans l'onglet "
            "Modèles, puis relance la synthèse.", 3
        )
    sf.write(t["sortie"], np.concatenate(segments), modele.sr)
    print(f"TERMINE {t['sortie']}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        _erreur("usage : chatterbox_tts.py <tache.json> | --telecharger")
    if sys.argv[1] == "--telecharger":
        telecharger()
    else:
        synthese(sys.argv[1])
