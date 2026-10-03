"""Ollama (s'il tourne sur le PC) : ses modèles libèrent la carte graphique avant chaque génération.

Le service Ollama garde le dernier modèle utilisé chargé sur la carte (5 minutes par défaut, « keep_alive ») :
plusieurs Go de mémoire graphique en moins pour nos moteurs, et sur 12 Go le pilote NVIDIA déborde alors en
silence dans la mémoire vive (« CUDA Sysmem Fallback ») : tout devient très lent (constaté chez l'utilisateur :
« Préparer le prompt » à plus de 4 minutes au lieu de quelques secondes).
API d'Ollama (docs/api.md) : GET /api/ps liste les modèles chargés (name, size_vram) ; POST /api/generate avec
{"model": nom, "keep_alive": 0} et sans prompt décharge le modèle. Ollama reste lancé et le rechargera à son
prochain usage. Adresse : OLLAMA_HOST (comme Ollama lui-même), sinon 127.0.0.1:11434.
Réglage : STUDIOVOIX_LIBERER_OLLAMA=0 pour ne jamais y toucher.
"""
import os

import requests

ACTIF = os.environ.get("STUDIOVOIX_LIBERER_OLLAMA", "1") != "0"


def adresse():
    hote = (os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434").strip()
    if hote.startswith("0.0.0.0"):  # adresse d'écoute (« toutes les interfaces ») : on s'adresse à la machine locale
        hote = "127.0.0.1" + hote[len("0.0.0.0"):]
    if not hote.startswith(("http://", "https://")):
        hote = "http://" + hote
    return hote.rstrip("/")


def modeles_charges(timeout=1.0):
    """[(nom, Go sur la carte graphique)] des modèles qu'Ollama garde chargés ; [] s'il ne tourne pas."""
    try:
        r = requests.get(f"{adresse()}/api/ps", timeout=timeout)
        r.raise_for_status()
        return [(m.get("name") or m.get("model"), (m.get("size_vram") or 0) / 1e9) for m in r.json().get("models", [])]
    except (requests.RequestException, ValueError):
        return []


def liberer(progress=None):
    """Décharge les modèles d'Ollama qui occupent la carte graphique. Renvoie les noms déchargés."""
    if not ACTIF:
        return []
    decharges = []
    for nom, go in modeles_charges():
        if not nom or go <= 0:
            continue  # modèle sur le processeur seulement : il ne gêne pas la carte
        if progress:
            progress(0.01, desc=f"Libération de la carte graphique : Ollama décharge {nom}…")
        try:
            requests.post(f"{adresse()}/api/generate", json={"model": nom, "keep_alive": 0}, timeout=30)
            decharges.append(nom)
        except requests.RequestException:
            pass
    return decharges


def etat():
    """Ligne du diagnostic : (ok, détail)."""
    charges = modeles_charges()
    if not charges:
        return None, "pas de modèle Ollama chargé (ou Ollama absent)"
    detail = ", ".join(f"{nom} ({go:.1f} Go sur la carte)" for nom, go in charges)
    return (False if any(go > 0 for _, go in charges) and not ACTIF else None), (
        detail + (" : déchargé(s) automatiquement avant chaque génération" if ACTIF else
                  " : occupe la carte (STUDIOVOIX_LIBERER_OLLAMA=0)"))
